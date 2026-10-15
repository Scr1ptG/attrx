"""Аналоги из каталога: тот же товар (или вариант той же модели) у другого продавца.

84% эталонных характеристик в тексте карточки не упоминаются: страна, гарантия, паспортные параметры.
Извлечь их из текста нельзя, но они есть у уже заведённого в каталог аналога. Алгоритм:
  1. кандидаты-аналоги — train-товары той же категории, у которых совпадает хотя бы один «код модели»
     (токен с цифрами: a57, 32h750nb, 17);
  2. ранжирование по сходству названий (Jaccard по словам), отбор с порогом;
  3. значение характеристики предлагается, только если аналоги согласны между собой.

Значение от аналога не подтверждено текстом карточки. Поэтому у него источник `analog`, а evidence ссылается
на карточку-аналог. В продакшене такие значения идут модератору со статусом «на проверку», а не публикуются молча.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from attrx.extractors.base import Candidate, Evidence
from attrx.values import UNITS, canon, number

WORD = re.compile(r"[0-9a-zа-яё]+(?:[-.][0-9a-zа-яё]+)*", re.IGNORECASE)
MAX_POSTING = 3000  # слишком частые «коды» (2024, 1) не используются для поиска


# Кириллица, похожая на латиницу, в кодах моделей: «КТ-2239» и «KT-2239» — один код.
HOMOGLYPHS = str.maketrans("авекмнорстух", "abekmhopctyx")


def _norm_word(w: str) -> str:
    w = w.lower().replace("ё", "е")
    if any(ch.isdigit() for ch in w):  # код модели: без разделителей и с латиницей
        w = w.translate(HOMOGLYPHS).replace("-", "").replace(".", "")
    return w


def title_words(title: str | None) -> frozenset[str]:
    return frozenset(_norm_word(w) for w in WORD.findall(title or ""))


def is_code(word: str) -> bool:
    return any(ch.isdigit() for ch in word) and len(word) >= 2


@dataclass
class AnalogIndex:
    ids: list[str]
    words: list[frozenset[str]]
    names: list[str]
    postings: dict[str, dict[str, list[int]]]  # категория -> код -> индексы товаров
    facts: object  # polars DataFrame (offer_id, attribute, value, unit), отсортирован по offer_id
    ranges: dict[str, tuple[int, int]]
    k: int = 5
    min_sim: float = 0.4  # подобрано на val (notebooks/03_hybrids)
    min_agreement: float = 0.6
    slot_units: dict[tuple[str, str], str] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)

    @classmethod
    def build(cls, offers, facts, schema=None, k: int = 5, min_sim: float = 0.4, min_agreement: float = 0.6) -> AnalogIndex:
        """offers: (offer_id, category_id, name); facts: (offer_id, attribute, value, unit) — только train.
        schema — для перевода чисел аналога в единицы слота категории (550 г -> 0.55 кг)."""
        import polars as pl

        ids, words, names = [], [], []
        postings: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for i, (oid, cat, name) in enumerate(offers.select("offer_id", "category_id", "name").iter_rows()):
            w = title_words(name)
            ids.append(oid); words.append(w); names.append(name or "")
            for t in w:
                if is_code(t):
                    postings[cat][t].append(i)
        facts = facts.sort("offer_id")
        bounds = (facts.with_row_index("row").group_by("offer_id")
                  .agg(pl.col("row").min().alias("lo"), pl.col("row").max().alias("hi")))
        ranges = {o: (lo, hi + 1) for o, lo, hi in bounds.iter_rows()}
        units = {}
        if schema is not None:
            units = {(c, a): u for c, a, k_, u in schema.select("category_id", "attribute", "kind", "unit").iter_rows()
                     if k_ == "num" and u}
        idx = cls(ids, words, names, {c: dict(v) for c, v in postings.items()}, facts, ranges, k, min_sim, min_agreement,
                  units)
        idx.stats = {"offers": len(ids), "categories": len(idx.postings), "facts": facts.height}
        return idx

    def neighbors(self, title: str, category: str | None) -> list[tuple[int, float]]:
        q = title_words(title)
        cat = self.postings.get(category or "")
        if not cat or not q:
            return []
        pool: set[int] = set()
        for t in q:
            if is_code(t):
                posting = cat.get(t, ())
                if len(posting) <= MAX_POSTING:
                    pool.update(posting)
        scored = []
        for i in pool:
            w = self.words[i]
            sim = len(q & w) / len(q | w)
            if sim >= self.min_sim:
                scored.append((i, sim))
        scored.sort(key=lambda x: -x[1])
        return scored[:self.k]

    def _facts(self, offer_id: str):
        r = self.ranges.get(offer_id)
        return self.facts.slice(r[0], r[1] - r[0]).select("attribute", "value", "unit").iter_rows() if r else ()

    def candidates(self, title: str, description: str, category: str | None) -> list[Candidate]:
        near = self.neighbors(title, category)
        if not near:
            return []
        votes: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(lambda: [0.0, None, None, None]))
        weight: dict[str, float] = defaultdict(float)
        for i, sim in near:
            for attr, value, unit in self._facts(self.ids[i]):
                key = tuple(sorted(canon(value, unit)))
                v = votes[attr][key]
                v[0] += sim
                if v[1] is None:
                    v[1], v[2], v[3] = value, i, unit
                weight[attr] += sim
        top_sim = near[0][1]
        out = []
        for attr, by_value in votes.items():
            key, (score, value, i, _) = max(by_value.items(), key=lambda kv: kv[1][0])
            agreement = score / weight[attr]
            if agreement < self.min_agreement:
                continue
            value = self._to_slot_unit(category, attr, value, by_value[key][3])
            ev = Evidence(f"analog:{self.ids[i]}", 0, 0, self.names[i][:120])
            conf = round(top_sim * agreement * 0.75, 4)  # текст карточки при прочих равных важнее аналога
            out.append(Candidate(attr, value, "|".join(sorted(canon(value))), conf, "analog", ev, known=True))
        return out

    def _to_slot_unit(self, category: str | None, attribute: str, value: str, unit: str | None) -> str:
        target = self.slot_units.get((category or "", attribute))
        amount = number(value)
        if not target or not unit or amount is None or unit == target:
            return value
        src, dst = UNITS.get(unit.strip().lower()), UNITS.get(target.strip().lower())
        if not src or not dst or src[0] != dst[0]:
            return value
        return f"{round(amount * src[1] / dst[1], 4):g}"


@dataclass
class SimilarItems:
    """«Похожие товары»: если точного аналога (общий код модели) нет, берём товары той же категории
    с близким названием (обычно та же линейка бренда) и предлагаем только то, в чём они единодушны.

    Это более слабое подтверждение, чем аналог: источник `similar`, статус не выше `review`."""

    index: AnalogIndex
    k: int = 10
    min_sim: float = 0.3
    min_agreement: float = 0.8
    scale: float = 0.6
    word_postings: dict[str, dict[str, list[int]]] = field(default_factory=dict)

    @classmethod
    def build(cls, index: AnalogIndex, offers, **kw) -> SimilarItems:
        postings: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for i, cat in enumerate(offers["category_id"].to_list()):
            for w in index.words[i]:
                if not is_code(w) and len(w) >= 3:
                    postings[cat][w].append(i)
        return cls(index, word_postings={c: dict(v) for c, v in postings.items()}, **kw)

    def neighbors(self, title: str, category: str | None) -> list[tuple[int, float]]:
        if self.index.neighbors(title, category):  # есть точный аналог — работает AnalogIndex
            return []
        q = title_words(title)
        cat = self.word_postings.get(category or "", {})
        words = sorted((w for w in q if w in cat and len(cat[w]) <= MAX_POSTING), key=lambda w: len(cat[w]))[:2]
        pool: set[int] = set()
        for w in words:
            pool.update(cat[w])
        scored = [(i, len(q & self.index.words[i]) / len(q | self.index.words[i])) for i in pool]
        scored = [x for x in scored if x[1] >= self.min_sim]
        scored.sort(key=lambda x: -x[1])
        return scored[:self.k]

    def candidates(self, title: str, description: str, category: str | None) -> list[Candidate]:
        near = self.neighbors(title, category)
        if len(near) < 3:  # единодушие нескольких похожих товаров, а не одного
            return []
        votes: dict[str, dict[tuple, list]] = defaultdict(lambda: defaultdict(lambda: [0.0, None, None, None]))
        weight: dict[str, float] = defaultdict(float)
        for i, sim in near:
            for attr, value, unit in self.index._facts(self.index.ids[i]):
                v = votes[attr][tuple(sorted(canon(value, unit)))]
                v[0] += sim
                if v[1] is None:
                    v[1], v[2], v[3] = value, i, unit
                weight[attr] += sim
        out = []
        for attr, by_value in votes.items():
            key, (score, value, i, unit) = max(by_value.items(), key=lambda kv: kv[1][0])
            agreement = score / weight[attr]
            if agreement < self.min_agreement:
                continue
            value = self.index._to_slot_unit(category, attr, value, unit)
            ev = Evidence(f"similar:{self.index.ids[i]}", 0, 0, self.index.names[i][:120])
            out.append(Candidate(attr, value, "|".join(sorted(canon(value))), round(near[0][1] * agreement * self.scale, 4),
                                 "similar", ev, known=True))
        return out
