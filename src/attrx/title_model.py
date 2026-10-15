"""Обучаемый извлекатель «Модель» и «Серия» из названия товара. ЭКСПЕРИМЕНТ, В КАСКАД НЕ ВКЛЮЧЁН.

Итог (EXPERIMENTS.md, итерация 2): AUC ранжирования 0.99 на train-продавцах, но на val (новые продавцы) точность 0.30.
Продавцы по-разному понимают «модель» и «серию» («Elica DAISY WH/A/60» и «DAISY WH/A/60»), поэтому соглашение
эталона из текста не выучивается. Модуль оставлен для воспроизводимости отрицательного результата.

Модель — второй по частоте тип поисковых запросов (8% кликов). Справочник train её не знает, если модель новая,
но в названии она почти всегда есть: «Фен Braun Satin Hair 1 HD130 черный» → серия «Satin Hair 1».

Задача ранжирования: кандидаты — фрагменты названия длиной 1–6 слов, начинающиеся в пределах 3 слов после бренда.
Признаки: смещение от бренда, длина, форма токенов (цифры, латиница, регистр), что идёт следом (характеристика,
цвет, скобка, конец), априорное распределение (смещение, длина) для категории и характеристики, наличие в справочнике.
Модель: градиентный бустинг на train, по одной на характеристику. Выбирается лучший фрагмент; ниже порога — пропуск.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from attrx.extractors.base import Candidate, Evidence
from attrx.lexicon import BRAND, Lexicon
from attrx.values import card_text, key, tokens

TARGETS = ("Модель", "Серия")
WORD = re.compile(r"[^\s,;]+")
SPEC = re.compile(r"^\d+([.,]\d+)?(/\d+([.,]\d+)?)*(gb|гб|tb|тб|mb|см|мм|вт|w|л|кг|г|мач|mah|\"|”|″)?$|^\d+\s*[xх×*]\s*\d+", re.I)
MAX_OFFSET, MAX_LEN = 3, 6
MIN_PROBA = 0.5


def _words(title: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in WORD.finditer(title)]


def _brand_end(lex: Lexicon, title: str) -> int | None:
    """Индекс первого слова после бренда в названии (бренд ищется справочником Lexicon)."""
    cands = lex._brand({"title": title, "description": ""}, {"title": tokens(title), "description": []})
    if not cands or cands[0].evidence.field != "title":
        return None
    end = cands[0].evidence.end
    for i, (_, s, _) in enumerate(_words(title)):
        if s >= end:
            return i
    return None


@dataclass
class TitleModel:
    lexicon: Lexicon
    models: dict[str, object] = field(default_factory=dict)
    priors: dict[tuple[str, str, int, int], float] = field(default_factory=dict)
    colors: dict[str, set[str]] = field(default_factory=dict)
    report: dict[str, dict] = field(default_factory=dict)

    # ---------------------------------------------------------------- признаки
    def _spans(self, title: str, category: str, attribute: str):
        words = _words(title)
        a = _brand_end(self.lexicon, title)
        if a is None or a >= len(words):
            return words, []
        colors = self.colors.get(category, set())
        vocab = self.lexicon.values.get(category, {})
        out = []
        for off in range(MAX_OFFSET + 1):
            s = a + off
            for length in range(1, MAX_LEN + 1):
                e = s + length
                if e > len(words):
                    break
                span = [w for w, _, _ in words[s:e]]
                nxt = words[e][0] if e < len(words) else ""
                joined = " ".join(span)
                letters = [ch for ch in joined if ch.isalpha()]
                in_vocab = any(a_ == attribute for a_, _, _ in vocab.get(tuple(key(joined).split()), ()))
                feats = [
                    off, length,
                    sum(any(ch.isdigit() for ch in w) for w in span) / length,
                    float(any("a" <= ch.lower() <= "z" for ch in joined)),
                    float(any("а" <= ch.lower() <= "я" for ch in joined)),
                    (sum(ch.isupper() for ch in letters) / len(letters)) if letters else 0.0,
                    float(bool(SPEC.match(nxt))), float(key(nxt) in colors), float(nxt.startswith("(")), float(nxt == ""),
                    float(any(SPEC.match(w) for w in span)), float(any(key(w) in colors for w in span)),
                    float(span[-1].isdigit()), float(in_vocab),
                    self.priors.get((category, attribute, off, length), 0.0),
                    self.priors.get(("*", attribute, off, length), 0.0),
                ]
                out.append((s, e, feats))
        return words, out

    # ---------------------------------------------------------------- обучение
    @classmethod
    def fit(cls, lexicon: Lexicon, cards, gold, per_attribute: int = 40000, seed: int = 13) -> TitleModel:
        """cards — train-карточки; gold — эталон. Метки: фрагмент совпадает с эталоном после нормализации."""
        import polars as pl
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.metrics import roc_auc_score

        tm = cls(lexicon)
        for cat, slots in lexicon.slots.items():
            if "Цвет" in slots:
                tm.colors[cat] = {" ".join(k) for k, entries in lexicon.values.get(cat, {}).items()
                                  if any(a == "Цвет" for a, _, _ in entries)}
        for attribute in TARGETS:
            g = gold.filter(pl.col("attribute") == attribute).select("sku_id", pl.col("value").alias("gold"))
            d = cards.join(g, on="sku_id", how="inner").select("sku_id", "category_id", "title", "gold").sort("sku_id")
            d = d.sample(n=min(per_attribute, d.height), seed=seed)
            # проход 1: априорное распределение (смещение от бренда, длина) верного фрагмента
            counts: dict[tuple, int] = defaultdict(int)
            totals: dict[str, int] = defaultdict(int)
            prepared = []
            for _, cat, title, value in d.iter_rows():
                title, target = card_text(title), key(value)
                words, a = _words(title), _brand_end(lexicon, title)
                totals[cat] += 1
                totals["*"] += 1
                if a is not None:
                    hit = next(((off, n) for off in range(MAX_OFFSET + 1) for n in range(1, MAX_LEN + 1)
                                if a + off + n <= len(words)
                                and key(" ".join(w for w, _, _ in words[a + off:a + off + n])) == target), None)
                    if hit:
                        counts[(cat, attribute, *hit)] += 1
                        counts[("*", attribute, *hit)] += 1
                prepared.append((cat, title, target))
            tm.priors.update({k: v / totals[k[0]] for k, v in counts.items()})
            # проход 2: признаки фрагментов и метки
            X, y, groups = [], [], []
            for gi, (cat, title, target) in enumerate(prepared):
                words, spans = tm._spans(title, cat, attribute)
                for s, e, feats in spans:
                    X.append(feats)
                    y.append(int(key(" ".join(w for w, _, _ in words[s:e])) == target))
                    groups.append(gi)
            X, y = np.asarray(X), np.asarray(y)
            if len(set(y.tolist())) < 2:
                continue
            hold = np.asarray(groups) % 5 == 0  # каждая пятая карточка — для отчёта о качестве ранжирования
            model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed).fit(X[~hold], y[~hold])
            auc = roc_auc_score(y[hold], model.predict_proba(X[hold])[:, 1])
            tm.models[attribute] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, random_state=seed).fit(X, y)
            tm.report[attribute] = {"cards": len(prepared), "spans": int(len(y)), "hit_rate": round(sum(counts[k] for k in counts
                                    if k[0] == "*") / max(totals["*"], 1), 4), "holdout_auc": round(float(auc), 4)}
        return tm

    # ---------------------------------------------------------------- извлечение
    def candidates(self, title: str, description: str, category: str | None) -> list[Candidate]:
        slots = self.lexicon.slots.get(category or "", {})
        text = card_text(title)
        out = []
        for attribute, model in self.models.items():
            if attribute not in slots:
                continue
            words, spans = self._spans(text, category, attribute)
            if not spans:
                continue
            p = model.predict_proba(np.asarray([f for _, _, f in spans]))[:, 1]
            i = int(p.argmax())
            if p[i] < MIN_PROBA:
                continue
            s, e, _ = spans[i]
            start, end = words[s][1], words[e - 1][2]
            value = text[start:end]
            out.append(Candidate(attribute, value, key(value), round(float(p[i]), 4), "model",
                                 Evidence("title", start, end, value)))
        return out


__all__ = ["TitleModel", "TARGETS", "BRAND"]
