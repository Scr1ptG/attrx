"""Обучение компонентов каскада на реальном каталоге и выбор порогов (только train и val).

  train  -> справочники (Lexicon), выученные шаблоны (TuplePatterns), индекс аналогов (AnalogIndex, SimilarItems)
  val    -> верификатор и порог авто-публикации (precision ≥ 0.95 на OOF-предсказаниях val)
  test   -> только оценка (notebooks/03_hybrids.ipynb)

Артефакты — наши собственные объекты, сохраняются pickle в ATTRX_DATA_DIR/work/artifacts.
Чужие pickle-файлы этим кодом не загружаются.
"""

from __future__ import annotations

import dataclasses
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl

from attrx import metrics as M
from attrx.analogs import AnalogIndex, SimilarItems
from attrx.catalog import BRAND, DataPaths, load
from attrx.configs import CONFIGS, MAIN, REVIEW_THRESHOLD, TARGET_PRECISION
from attrx.lexicon import Lexicon
from attrx.patterns import TuplePatterns
from attrx.stack import TEXT_SOURCES, Stack
from attrx.verifier import Verifier

# Источники, которые дают право на авто-публикацию: подтверждение в тексте карточки.
# Аналоги и похожие товары подтверждены каталогом, а не текстом. LLM одна значение не публикует никогда:
# её ответ без согласия дешёвых источников идёт модератору.
ACCEPT_SOURCES = TEXT_SOURCES - {"llm"}
ASKABLE_MIN_RATE, ASKABLE_MIN_N = 0.15, 5  # характеристику спрашиваем у LLM, если на val она в тексте ≥ 15%


@dataclass
class Components:
    lexicon: Lexicon
    patterns: TuplePatterns | None = None
    analogs: AnalogIndex | None = None
    similar: SimilarItems | None = None
    timings: dict[str, float] = field(default_factory=dict)

    def stack(self, patterns: bool = True, analogs: bool = True, similar: bool = True, extra: list | None = None,
              verifier=None, judge=None) -> Stack:
        sources = ([self.patterns] if patterns and self.patterns else []) + ([self.analogs] if analogs and self.analogs else [])
        sources += [self.similar] if similar and analogs and self.similar else []
        return Stack(self.lexicon, sources + list(extra or []), verifier, judge=judge)


# Где лежат открытые модели-основы судьи (скачиваются один раз в ATTRX_DATA_DIR/models/hf).
JUDGE_BACKBONES = {"laya": "models--convaiinnovations--laya-multilingual", "frida": "models--ai-forever--FRIDA"}


def backbone_path(paths: DataPaths, name: str) -> Path:
    root = Path(os.environ.get("ATTRX_MODELS_DIR", paths.root / "models" / "hf" / "hub")) / JUDGE_BACKBONES[name] / "snapshots"
    snapshots = sorted(root.glob("*")) if root.exists() else []
    if not snapshots:
        raise SystemExit(f"Нет весов {name} в {root}: скачайте модель (README, раздел «Судья»)")
    return snapshots[-1]


def load_judge(paths: DataPaths, name: str):
    """Обученный судья из work/artifacts/judge_<name>.pt или None, если его ещё не обучали."""
    path = paths.work / "artifacts" / f"judge_{name}.pt"
    if not path.exists():
        return None
    import torch

    from attrx.judge import Judge

    return Judge.load(path, backbone_path(paths, name), device="cuda" if torch.cuda.is_available() else "cpu")


def build_stack(comp: Components, name: str, verifiers: dict, paths: DataPaths, verifier: bool = True) -> Stack:
    """Каскад конфигурации name (configs.CONFIGS) с её верификатором и, если нужно, судьёй."""
    kw = dict(CONFIGS[name])
    judge_name = kw.pop("judge", None)
    judge = load_judge(paths, judge_name) if judge_name else None
    if judge_name and judge is None:
        raise SystemExit(f"Конфигурация {name} требует судью {judge_name}: обучите его (notebooks/05_judge.ipynb)")
    ver = verifiers[name][0] if verifier else None
    if ver is not None and judge is not None and getattr(ver, "judge_id", None) != judge.report.get("id"):
        raise SystemExit(f"Верификатор {name} обучен с другим судьёй: python -m attrx train --reuse-components")
    return comp.stack(**kw, judge=judge, verifier=ver)


def fit_components(paths: DataPaths, with_analogs: bool = True) -> Components:
    cards, gold, schema, ref = (load(paths, n) for n in ("cards", "gold", "schema", "reference"))
    train = cards.filter(pl.col("split") == "train")
    titles = train.select("sku_id", "title").join(
        gold.filter(pl.col("attribute") == BRAND).select("sku_id", pl.col("value").alias("brand")), on="sku_id")
    timings = {}
    t = time.time(); lex = Lexicon.fit(schema, ref, titles=titles); timings["lexicon_s"] = round(time.time() - t, 1)
    t = time.time(); pat = TuplePatterns.fit(train, gold, schema, ref); timings["patterns_s"] = round(time.time() - t, 1)
    idx = sim = None
    if with_analogs:
        t = time.time()
        offers = load(paths, "analog_offers")
        idx = AnalogIndex.build(offers, load(paths, "analog_facts"), schema)
        sim = SimilarItems.build(idx, offers)
        timings["analogs_s"] = round(time.time() - t, 1)
    return Components(lex, pat, idx, sim, timings)


def fit_verifier(stack: Stack, paths: DataPaths, per_group: int = 300) -> tuple[Verifier, float]:
    """Верификатор на val (GroupKFold по продавцу) и порог авто-публикации по OOF-предсказаниям."""
    cards, gold, schema, ref, cats = (load(paths, n) for n in ("cards", "gold", "schema", "reference", "categories"))
    val = M.eval_sample(cards, cats, "val", top_k=None, per_category=None, per_group=per_group)
    frame = stack.predict_frame(val)
    pairs = M.score_pairs(frame, gold, val, schema)
    labeled = pairs.filter(pl.col("pred").is_not_null() & pl.col("gold").is_not_null()).select("sku_id", "attribute", "correct")
    data = frame.join(labeled, on=["sku_id", "attribute"], how="inner").join(val.select("sku_id", "group"), on="sku_id")
    y = data["correct"].cast(pl.Int8).to_numpy()
    ver = Verifier.from_reference(schema, ref).fit(data, y, data["group"].to_numpy())
    flags = data.select(_accept_eligible().alias("e"), _judge_only().alias("j"))
    eligible, judge_only = flags["e"].to_numpy(), flags["j"].to_numpy()
    # порог считаем по тем, кого можно принять; без значений «только судья»: точные текстовые строки не должны
    # прятать менее точные подтверждённые судьёй
    thr = ver.threshold_for(y, TARGET_PRECISION, eligible=eligible & ~judge_only)
    if stack.judge is not None:
        ver.judge_id = stack.judge.report.get("id")
        # свой порог для «только судья»: точность ≥ 0.95 на val именно у них; мало примеров — авто-публикации нет
        ver.judge_threshold = max(thr, ver.threshold_for(y, TARGET_PRECISION, eligible=judge_only)) if judge_only.any() else 1.0
        ver.report["judge_only_rows"] = int(judge_only.sum())
    return ver, thr


def fit_askable(paths: DataPaths) -> dict[str, dict[str, float]]:
    """Какие характеристики имеет смысл спрашивать у LLM: категория -> {характеристика: доля подтверждаемости на val}.

    Доля подтверждаемости — как часто эталонное значение встречается в тексте карточек категории. Страну или
    гарантию в тексте почти не пишут: такие поля LLM не спрашиваем, иначе она будет их выдумывать."""
    cards, gold, schema, cats = (load(paths, n) for n in ("cards", "gold", "schema", "categories"))
    val = M.eval_sample(cards, cats, "val", top_k=None, per_category=None, per_group=300)
    rates = (M.supported_flags(val, gold, schema).join(val.select("sku_id", "category_id"), on="sku_id")
             .group_by("category_id", "attribute").agg(pl.col("supported").mean().alias("rate"), pl.len().alias("n")))
    askable: dict[str, dict[str, float]] = {}
    keep = rates.filter((pl.col("rate") >= ASKABLE_MIN_RATE) & (pl.col("n") >= ASKABLE_MIN_N))
    for c, a, r in keep.select("category_id", "attribute", "rate").sort("category_id", "attribute").iter_rows():
        askable.setdefault(c, {})[a] = round(float(r), 4)
    return askable


def _has_source(sources) -> pl.Expr:
    return pl.col("source").str.contains("|".join(rf"(^|\+){s}(\+|$)" for s in sorted(sources)))


def _accept_eligible() -> pl.Expr:
    return _has_source(ACCEPT_SOURCES)


def _judge_only() -> pl.Expr:
    """Текстовое подтверждение есть только от судьи (значение аналога или похожего товара)."""
    return _has_source({"judge"}) & ~_has_source(ACCEPT_SOURCES - {"judge"})


def assign_status(frame: pl.DataFrame, accept: float, review: float = REVIEW_THRESHOLD,
                  text_required: bool = True, judge_accept: float | None = None) -> pl.DataFrame:
    """accept — в каталог без проверки, review — модератору, reject — не показываем. Единая политика для CLI,
    демо и ноутбуков.

    text_required: значение без подтверждения в тексте карточки (аналог, похожие товары, одна LLM)
    автоматически не публикуется.
    judge_accept: порог для значений, подтверждённых только судьёй (Verifier.judge_threshold); по умолчанию accept."""
    can_accept = _accept_eligible() if text_required else pl.lit(True)
    need = accept if judge_accept is None else pl.when(_judge_only()).then(pl.lit(judge_accept)).otherwise(pl.lit(accept))
    return frame.with_columns(
        pl.when((pl.col("confidence") >= need) & can_accept).then(pl.lit("accept"))
        .when(pl.col("confidence") >= review).then(pl.lit("review"))
        .otherwise(pl.lit("reject")).alias("status"))


def save(obj, paths: DataPaths, name: str) -> Path:
    """Атомарно: сначала во временный файл, потом переименование — прерванный train не оставит битый артефакт."""
    out = paths.work / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.pkl"
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return path


def load_artifact(paths: DataPaths, name: str):
    """Артефакт из work/artifacts или None, если его нет. Устаревший артефакт (без полей текущего кода) — ошибка."""
    path = paths.work / "artifacts" / f"{name}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as f:
        obj = pickle.load(f)  # nosec B301 — только собственные артефакты, созданные save()
    if dataclasses.is_dataclass(obj):
        missing = [f.name for f in dataclasses.fields(obj) if f.name not in vars(obj)]
        if missing:
            raise SystemExit(f"Артефакт {path.name} устарел (нет полей {missing}): переобучите python -m attrx train")
    return obj


def status_summary(pairs: pl.DataFrame) -> pl.DataFrame:
    """Нагрузка на модератора: сколько полей авто-заполнено, предложено на проверку и сколько из них верны."""
    g = pl.col("gold").is_not_null()
    total = pairs.filter(g).height
    return (pairs.filter(g).with_columns(pl.col("status").fill_null("empty"))
            .group_by("status").agg(pl.len().alias("pairs"), pl.col("correct").mean().alias("precision"))
            .with_columns((pl.col("pairs") / total).alias("share_of_gold")).sort("status"))


__all__ = ["Components", "fit_components", "fit_verifier", "fit_askable", "assign_status", "save", "load_artifact",
           "status_summary", "ACCEPT_SOURCES", "CONFIGS", "MAIN", "REVIEW_THRESHOLD", "TARGET_PRECISION"]
