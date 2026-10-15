"""Правила и словари: извлечение из заголовка и описания с подтверждающим фрагментом текста."""

from __future__ import annotations

import re
from functools import lru_cache

from attrx.extractors.base import Candidate, Card, Evidence
from attrx.normalize import Quantity, find_in_dictionary, find_patterns, find_quantities
from attrx.schema import AttributeSpec, Schema
from attrx.text import fold

FIELDS = ("title", "description")
BASE_CONFIDENCE = {"title": 0.8, "description": 0.7}
CONTEXT_BONUS = 0.1
AMBIGUITY_PENALTY = 0.25  # в тексте несколько разных значений с одинаковой поддержкой
CONTEXT_WINDOW = 40
# в записи «8/256 ГБ» первое число — оперативная память, второе — встроенная
PAIR_HINTS = {"ram": "first", "storage": "second"}


def extract_rules(card: Card, schema: Schema, attributes: tuple[str, ...]) -> list[Candidate]:
    candidates: list[Candidate] = []
    used_spans: set[tuple[str, int]] = set()
    specs = [schema.attributes[a] for a in attributes]
    # quantity-атрибуты одного семейства (ram/storage) делят упоминания: сначала самые уверенные
    quantity_scored = []
    for spec in specs:
        if spec.type == "quantity":
            quantity_scored.extend(_score_quantities(card, schema, spec))
    for score, field, q, spec, ambiguous in sorted(quantity_scored, key=lambda x: -x[0]):
        if any(c.attribute == spec.name for c in candidates) or (field, q.start) in used_spans:
            continue
        used_spans.add((field, q.start))
        conf = BASE_CONFIDENCE[field] + (CONTEXT_BONUS if score > 0 else 0) - (AMBIGUITY_PENALTY if ambiguous else 0)
        text = card.field_text(field)
        candidates.append(Candidate(spec.name, q.display, q.key, conf, "rules",
                                    Evidence(field, q.start, q.end, text[q.start:q.end])))
    for spec in specs:
        if spec.type == "enum":
            candidates.extend(_dictionary_candidates(card, schema, spec))
        elif spec.type == "pattern":
            candidates.extend(_pattern_candidates(card, spec))
        elif spec.type == "text" and spec.name == "model":
            candidates.extend(_model_heuristic(card, schema))
    return candidates


@lru_cache(maxsize=None)
def _keyword_regex(words: tuple[str, ...]) -> re.Pattern[str] | None:
    if not words:
        return None
    alts = "|".join(re.escape(w.lower().replace("ё", "е")) for w in sorted(words, key=len, reverse=True))
    return re.compile(rf"(?<![a-zа-я])(?:{alts})")


def _nearest(text: str, q: Quantity, words: tuple[str, ...]) -> int:
    """Расстояние в символах от упоминания до ближайшего ключевого слова (большое число, если нет)."""
    regex = _keyword_regex(words)
    if regex is None:
        return 10**6
    lo, hi = max(0, q.start - CONTEXT_WINDOW), q.end + CONTEXT_WINDOW
    lowered = text.lower().replace("ё", "е")
    best = 10**6
    for m in regex.finditer(lowered, lo, hi):
        best = min(best, q.start - m.end() if m.end() <= q.start else max(0, m.start() - q.end))
    return best


def _context_score(text: str, q: Quantity, spec: AttributeSpec) -> int:
    """+1 если ближайшее ключевое слово — «своё» (оперативн для ram), −1 если чужое, 0 если слов нет.
    В «8 ГБ ОЗУ, SSD 512 ГБ» число 8 ближе к «ОЗУ», а 512 — к «SSD»."""
    own, other = _nearest(text, q, spec.context), _nearest(text, q, spec.anti_context)
    score = 0 if own == other else (1 if own < other else -1)
    hint = PAIR_HINTS.get(spec.name)
    if q.hint and hint:
        score += 2 if q.hint == hint else -2
    return score


def _score_quantities(card: Card, schema: Schema, spec: AttributeSpec):
    """(оценка, поле, упоминание, атрибут, неоднозначность) для лучшего упоминания в каждом поле."""
    value_range = schema.value_range(card.category, spec.name)
    out = []
    for field in FIELDS:
        text = card.field_text(field)
        scored = []
        for q in find_quantities(text, spec.unit_family or ""):
            if value_range and not value_range[0] <= q.value <= value_range[1]:
                continue
            score = _context_score(text, q, spec)
            if spec.anti_context and score < 0:
                continue
            scored.append((score, q))
        if not scored:
            continue
        best = max(s for s, _ in scored)
        top = [q for s, q in scored if s == best]
        ambiguous = len({q.key for q in top}) > 1
        out.append((best + (0.5 if field == "title" else 0), field, top[0], spec, ambiguous))
    return out


def _dictionary_candidates(card: Card, schema: Schema, spec: AttributeSpec) -> list[Candidate]:
    dictionary = schema.dictionaries.get(spec.dictionary or "", {})
    for field in FIELDS:  # заголовок надёжнее описания: берём первое поле, где нашлось значение
        text = card.field_text(field)
        matches = find_in_dictionary(text, dictionary)
        if not matches:
            continue
        distinct = {m.canonical for m in matches}
        first = matches[0]
        conf = BASE_CONFIDENCE[field] - (AMBIGUITY_PENALTY if len(distinct) > 1 else 0)
        return [Candidate(spec.name, first.canonical, fold(first.canonical), conf, "rules",
                          Evidence(field, first.start, first.end, text[first.start:first.end]))]
    return []


def _pattern_candidates(card: Card, spec: AttributeSpec) -> list[Candidate]:
    for field in FIELDS:
        text = card.field_text(field)
        matches = sorted(find_patterns(text, spec.name), key=lambda m: (not m.specific, m.start))
        if matches:
            m = matches[0]
            conf = BASE_CONFIDENCE[field] - (0 if m.specific else 0.2)
            return [Candidate(spec.name, m.canonical, fold(m.canonical), conf, "rules",
                              Evidence(field, m.start, m.end, text[m.start:m.end]))]
    return []


_MODEL_STOP = re.compile(r"\d+(?:[.,]\d+)?\s*(?:/|\+|гб|gb|тб|tb|\"|дюйм|мач|mah)|[,(;|]", re.IGNORECASE)


def _model_heuristic(card: Card, schema: Schema) -> list[Candidate]:
    """Модель = текст заголовка после бренда до первой характеристики («12/256 ГБ», «6.7"», запятая, цвет)."""
    title = card.title
    brands = find_in_dictionary(title, schema.dictionaries.get("brands", {}))
    if not brands:
        return []
    start = brands[0].end
    rest = title[start:]
    stop = _MODEL_STOP.search(rest)
    end_rel = stop.start() if stop else len(rest)
    colors = find_in_dictionary(rest[:end_rel], schema.dictionaries.get("colors", {}))
    if colors:
        end_rel = colors[0].start
    value = rest[:end_rel].strip(" -")
    if not value or len(value) > 60:
        return []
    begin = start + rest.index(value)
    return [Candidate("model", value, fold(value).replace(" ", ""), 0.55, "heuristic",
                      Evidence("title", begin, begin + len(value), value), known=False)]
