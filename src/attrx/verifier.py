"""Верификатор кандидатов — открытый аналог JEV: «какова вероятность, что значение верно?».

Признаки описывают не текст, а то, как значение было найдено:
- какие источники согласились и было ли противоречие;
- поле (название или описание) и тип характеристики;
- насколько значение типично для категории (частота в справочнике train);
- насколько характеристика в целом надёжно извлекается (target encoding по фолдам).

Модель: градиентный бустинг + изотоническая калибровка. Обучение только на val с GroupKFold по продавцу,
поэтому test не участвует ни в обучении, ни в выборе порога (docs/DATA_CONTRACT.md).
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import polars as pl

from attrx.stack import SOURCES

KINDS = ("bool", "num", "enum", "text")
MAX_PROBA = 0.99  # изотоническая регрессия даёт 1.0 в верхней корзине; полной уверенности не заявляем
PRIOR_WEIGHT = 20.0  # сглаживание target encoding
MIN_SUPPORT = 30  # порог авто-публикации выбирается, только если над ним не меньше стольких кандидатов val
JUDGE_FEATURES = ("judge_confirm", "judge_contra")  # вероятности судьи (attrx.judge), если он подключён


def _source_flags(frame: pl.DataFrame) -> list[pl.Expr]:
    return [pl.col("source").str.contains(rf"(^|\+){s}(\+|$)").cast(pl.Int8).alias(f"src_{s}") for s in SOURCES]


@dataclass
class Verifier:
    value_freq: dict[tuple[str, str, str], int] = field(default_factory=dict)
    attr_fill: dict[tuple[str, str], float] = field(default_factory=dict)
    attr_te: dict[str, float] = field(default_factory=dict)
    kind_te: dict[str, float] = field(default_factory=dict)
    model: object | None = None
    iso: object | None = None
    report: dict = field(default_factory=dict)
    judge_threshold: float | None = None  # отдельный порог accept для значений, подтверждённых только судьёй
    judge_id: str | None = None  # с каким судьёй обучен (признаки judge_*): другой судья — переобучить

    # ---------------------------------------------------------------- признаки
    @classmethod
    def from_reference(cls, schema: pl.DataFrame, reference: pl.DataFrame) -> Verifier:
        freq = {(c, a, v): n for c, a, v, n in reference.select("category_id", "attribute", "value", "count").iter_rows()}
        fill = {(c, a): f for c, a, f in schema.select("category_id", "attribute", "fill").iter_rows()}
        return cls(value_freq=freq, attr_fill=fill)

    def features(self, frame: pl.DataFrame, attr_te: dict[str, float] | None = None,
                 kind_te: dict[str, float] | None = None) -> np.ndarray:
        attr_te = self.attr_te if attr_te is None else attr_te
        kind_te = self.kind_te if kind_te is None else kind_te
        f = frame.with_columns(*_source_flags(frame))
        for col in JUDGE_FEATURES:  # судья не подключён или значение не проверял: -1
            f = f.with_columns(pl.col(col).fill_null(-1.0) if col in f.columns else pl.lit(-1.0).alias(col))
        rows = f.select("category_id", "attribute", "kind", "value", "raw_confidence", "n_sources", "n_values",
                        "conflict", "evidence_field", *JUDGE_FEATURES, *[f"src_{s}" for s in SOURCES]).iter_rows()
        out = []
        for cat, attr, kind, value, conf, n_src, n_val, conflict, fld, j_yes, j_no, *src in rows:
            value = value or ""
            te = attr_te.get(attr, kind_te.get(kind, 0.5))
            out.append([
                conf, n_src, n_val, float(conflict), float(fld == "title"), float(KINDS.index(kind) if kind in KINDS else 3),
                len(value), float(value.replace(".", "").replace(",", "").isdigit()),
                math.log1p(self.value_freq.get((cat, attr, value), 0)), self.attr_fill.get((cat, attr), 0.0), te,
                j_yes, j_no, *src,
            ])
        return np.asarray(out, dtype=np.float64)

    @staticmethod
    def _encodings(frame: pl.DataFrame, y: np.ndarray) -> tuple[dict[str, float], dict[str, float]]:
        d = frame.select("attribute", "kind").with_columns(pl.Series("y", y))
        kind = {k: v for k, v in d.group_by("kind").agg(pl.col("y").mean()).iter_rows()}
        prior = float(y.mean()) if len(y) else 0.5
        attr = {}
        for a, k, s, n in d.group_by("attribute").agg(pl.col("kind").first(), pl.col("y").sum(), pl.len()).iter_rows():
            base = kind.get(k, prior)
            attr[a] = (s + PRIOR_WEIGHT * base) / (n + PRIOR_WEIGHT)
        return attr, kind

    # ---------------------------------------------------------------- обучение
    def fit(self, frame: pl.DataFrame, y: np.ndarray, groups: np.ndarray, folds: int = 5, seed: int = 13) -> Verifier:
        """frame — предсказания Stack на val с эталоном; y — 1, если значение верно; groups — продавец."""
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.isotonic import IsotonicRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold

        def make():
            return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                  l2_regularization=1.0, categorical_features=[5], random_state=seed)

        def make_iso():
            return IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

        splits = list(GroupKFold(n_splits=folds).split(np.zeros(len(y)), y, groups))
        oof = np.zeros(len(y))
        for tr, te in splits:
            attr, kind = self._encodings(frame[tr], y[tr])
            m = make().fit(self.features(frame[tr], attr, kind), y[tr])
            oof[te] = m.predict_proba(self.features(frame[te], attr, kind))[:, 1]
        # калибровка для выбора порога — тоже вне фолда: изотоника, обученная на тех же точках, оптимистична вверху
        calibrated = np.zeros(len(y))
        for tr, te in splits:
            calibrated[te] = make_iso().fit(oof[tr], y[tr]).predict(oof[te])
        self.iso = make_iso().fit(oof, y)  # для предсказаний — на всех OOF-точках
        self.attr_te, self.kind_te = self._encodings(frame, y)
        self.model = make().fit(self.features(frame), y)
        calibrated = np.minimum(calibrated, MAX_PROBA)
        self.report = {"n": int(len(y)), "positive_rate": round(float(y.mean()), 4),
                       "oof_auc": round(float(roc_auc_score(y, oof)), 4),
                       "oof_auc_raw_confidence": round(float(roc_auc_score(y, frame["raw_confidence"].to_numpy())), 4)}
        self.oof_ = calibrated
        return self

    def predict_proba(self, frame: pl.DataFrame) -> np.ndarray:
        raw = self.model.predict_proba(self.features(frame))[:, 1]
        return np.round(np.minimum(self.iso.predict(raw), MAX_PROBA), 4)

    def threshold_for(self, y: np.ndarray, target_precision: float = 0.95, eligible: np.ndarray | None = None,
                      min_support: int = MIN_SUPPORT) -> float:
        """Минимальный порог t, при котором точность среди принятых на val (OOF-вероятность ≥ t) не ниже цели.

        Перебираются различные значения вероятности: изотоническая калибровка даёт ступени, и все кандидаты одной
        ступени принимаются вместе, поэтому порог внутри ступени дал бы завышенную оценку точности.
        eligible — кандидаты, которые политика статусов вообще может принять (есть подтверждение в тексте).
        Цель недостижима — возвращается 1.0: авто-публикация выключена, всё идёт модератору."""
        p, yy = self.oof_, np.asarray(y)
        if eligible is not None:
            p, yy = p[eligible], yy[eligible]
        for t in np.unique(p):  # по возрастанию: первый подходящий порог — минимальный
            take = p >= t
            if take.sum() >= min_support and yy[take].mean() >= target_precision:
                return float(t)
        warnings.warn(f"точность {target_precision} на val недостижима: авто-публикация выключена", stacklevel=2)
        return 1.0
