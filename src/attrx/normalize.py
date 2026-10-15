"""Приведение значений к каноническому виду: числа с единицами, справочники, процессоры."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from functools import lru_cache

from attrx.schema import AttributeSpec, Schema
from attrx.text import clean, fold, parse_number, ru_plural

# ---------------------------------------------------------------- величины с единицами

# семейство -> {написание единицы (lower): (каноническая единица, множитель к базовой)}
UNIT_FAMILIES: dict[str, dict[str, tuple[str, float]]] = {
    "data": {
        "тб": ("ТБ", 1024.0), "tb": ("ТБ", 1024.0), "терабайт": ("ТБ", 1024.0),
        "гб": ("ГБ", 1.0), "gb": ("ГБ", 1.0), "гигабайт": ("ГБ", 1.0),
        "мб": ("МБ", 1 / 1024), "mb": ("МБ", 1 / 1024), "мегабайт": ("МБ", 1 / 1024),
    },
    "length_in": {'"': ("дюйм", 1.0), "дюйм": ("дюйм", 1.0), "inch": ("дюйм", 1.0), "in.": ("дюйм", 1.0)},
    "charge": {"мач": ("мАч", 1.0), "mah": ("мАч", 1.0), "ма·ч": ("мАч", 1.0), "ма*ч": ("мАч", 1.0)},
}
_NUM = r"(\d+(?:[.,]\d+)?)"
_NO_LETTER = r"(?![a-zа-яё])"


@dataclass(frozen=True)
class Quantity:
    value: float  # в базовых единицах семейства (ГБ, дюймы, мАч)
    family: str
    start: int = -1
    end: int = -1
    hint: str | None = None  # подсказка позиции: "first"/"second" в записи вида 8/256 ГБ

    @property
    def display(self) -> str:
        v = self.value
        if self.family == "data":
            if v >= 1024 and v % 1024 == 0:
                return f"{v / 1024:g} ТБ"
            return f"{v:g} ГБ" if v >= 1 else f"{v * 1024:g} МБ"
        if self.family == "length_in":
            return f"{v:g} {ru_plural(v, 'дюйм', 'дюйма', 'дюймов')}"
        if self.family == "charge":
            return f"{v:g} мАч"
        return f"{v:g}"

    @property
    def key(self) -> str:
        return f"{self.family}:{round(self.value, 3):g}"


def _unit_alternation(family: str) -> str:
    parts = []
    for unit in sorted(UNIT_FAMILIES[family], key=len, reverse=True):
        esc = re.escape(unit)
        # длинные словесные единицы склоняются: дюйма, дюймов, гигабайт(а)
        parts.append(esc + r"[a-zа-яё]*" if len(unit) >= 4 and unit.isalpha() else esc + (_NO_LETTER if unit[-1].isalpha() else ""))
    return "(" + "|".join(parts) + ")"


@lru_cache(maxsize=None)
def _quantity_regexes(family: str) -> tuple[re.Pattern[str], re.Pattern[str]]:
    units = _unit_alternation(family)
    single = re.compile(rf"(?<![\w.,]){_NUM}\s*{units}", re.IGNORECASE)
    paired = re.compile(rf"(?<![\w.,]){_NUM}\s*(?:{units})?\s*[/+]\s*{_NUM}\s*{units}", re.IGNORECASE)
    return single, paired


def _unit_factor(family: str, raw_unit: str) -> float:
    raw = raw_unit.lower()
    for unit in sorted(UNIT_FAMILIES[family], key=len, reverse=True):
        if raw.startswith(unit):
            return UNIT_FAMILIES[family][unit][1]
    raise ValueError(f"Неизвестная единица {raw_unit!r} для {family}")


def find_quantities(text: str, family: str) -> list[Quantity]:
    """Все упоминания величины семейства в тексте, включая пары «8/256 ГБ»."""
    single, paired = _quantity_regexes(family)
    found: list[Quantity] = []
    taken: set[tuple[int, int]] = set()
    for m in paired.finditer(text):
        n1, u1, n2, u2 = m.groups()
        f2 = _unit_factor(family, u2)
        f1 = _unit_factor(family, u1) if u1 else f2
        found.append(Quantity(parse_number(n1) * f1, family, m.start(1), m.end(1), hint="first"))
        found.append(Quantity(parse_number(n2) * f2, family, m.start(3), m.end(), hint="second"))
        taken.add((m.start(), m.end()))
    for m in single.finditer(text):
        if any(s <= m.start() < e for s, e in taken):
            continue
        found.append(Quantity(parse_number(m.group(1)) * _unit_factor(family, m.group(2)), family, m.start(), m.end()))
    return sorted(found, key=lambda q: q.start)


def parse_quantity(raw: str, family: str) -> Quantity | None:
    """Значение из поля продавца или эталона: «256 ГБ», «256GB», «256» (единица по умолчанию — базовая)."""
    text = clean(raw)
    found = find_quantities(text, family)
    if found:
        return found[0]
    if re.fullmatch(_NUM, text):
        return Quantity(parse_number(text), family)
    return None


# ---------------------------------------------------------------- справочники (цвета, бренды)

_ADJ_ENDINGS = "ый|ий|ой|ая|яя|ое|ее|ые|ие|ого|его|ую|юю|ым|им|ом|ем|ых|их|ыми|ими"


def _synonym_regex(syn: str) -> str:
    if syn.endswith("*"):
        return re.escape(syn[:-1]) + f"(?:{_ADJ_ENDINGS})"
    return r"\s+".join(re.escape(part) for part in syn.split())


@lru_cache(maxsize=None)
def _dictionary_regex(entries: tuple[tuple[str, tuple[str, ...]], ...]) -> tuple[re.Pattern[str], dict[str, str]]:
    groups, names = [], {}
    for i, (canon, syns) in enumerate(entries):
        alts = sorted({_synonym_regex(s) for s in syns}, key=len, reverse=True)
        groups.append(f"(?P<g{i}>{'|'.join(alts)})")
        names[f"g{i}"] = canon
    pattern = re.compile(r"(?<![\w-])(?:" + "|".join(groups) + r")(?![\w])", re.IGNORECASE)
    return pattern, names


@dataclass(frozen=True)
class DictMatch:
    canonical: str
    start: int
    end: int


def find_in_dictionary(text: str, dictionary: dict[str, tuple[str, ...]]) -> list[DictMatch]:
    pattern, names = _dictionary_regex(tuple(dictionary.items()))
    return [DictMatch(names[m.lastgroup], m.start(), m.end()) for m in pattern.finditer(text) if m.lastgroup]


# ---------------------------------------------------------------- процессоры

def _cap(s: str | None) -> str:
    return (s or "").upper()


PATTERNS: dict[str, list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]]] = {
    "cpu": [
        (re.compile(r"(?:intel\s+)?core\s+ultra\s+([3579])\s*-?\s*(\d{3}[a-z]{0,2})\b", re.I),
         lambda m: f"Intel Core Ultra {m[1]} {_cap(m[2])}"),
        (re.compile(r"(?:intel\s+)?core\s+i([3579])(?:[\s-]+(\d{4,5}[a-z]{0,2}))?\b", re.I),
         lambda m: f"Intel Core i{m[1]}" + (f"-{_cap(m[2])}" if m[2] else "")),
        (re.compile(r"(?<![\w-])i([3579])-(\d{4,5}[a-z]{0,2})\b", re.I),
         lambda m: f"Intel Core i{m[1]}-{_cap(m[2])}"),
        (re.compile(r"(?:amd\s+)?ryzen\s+([3579])(?:\s+(?:pro\s+)?(\d{4}[a-z]{0,2}))?\b", re.I),
         lambda m: f"AMD Ryzen {m[1]}" + (f" {_cap(m[2])}" if m[2] else "")),
        (re.compile(r"(?:intel\s+)?(celeron|pentium(?:\s+silver|\s+gold)?)\s+([a-z]?\d{4,5}[a-z]?)\b", re.I),
         lambda m: f"Intel {m[1].title()} {_cap(m[2])}"),
        (re.compile(r"intel\s+(?:processor\s+)?n(\d{2,4})\b", re.I),
         lambda m: f"Intel N{m[1]}"),
        (re.compile(r"(?:apple|чип|chip)\s+m([1-5])(?:\s+(pro|max|ultra))?\b", re.I),
         lambda m: f"Apple M{m[1]}" + (f" {m[2].title()}" if m[2] else "")),
        (re.compile(r"snapdragon\s+x\s+(elite|plus)\b", re.I),
         lambda m: f"Qualcomm Snapdragon X {m[1].title()}"),
    ],
}


@dataclass(frozen=True)
class PatternMatch:
    canonical: str
    start: int
    end: int
    specific: bool  # найдена конкретная модель (i5-1235U), а не только серия (Core i5)


def find_patterns(text: str, attribute: str) -> Iterator[PatternMatch]:
    for regex, fmt in PATTERNS.get(attribute, []):
        for m in regex.finditer(text):
            canonical = fmt(m)
            specific = bool(re.search(r"\d{3,}", canonical)) or canonical.startswith(("Apple ", "Qualcomm "))
            yield PatternMatch(canonical, m.start(), m.end(), specific)


# ---------------------------------------------------------------- канонизация значения атрибута

@dataclass(frozen=True)
class Canonical:
    display: str  # как показывать в каталоге
    key: str  # ключ сравнения (эквивалентные записи совпадают)
    known: bool  # значение найдено в справочнике/распознано по формату


def canonicalize(spec: AttributeSpec, raw: str, schema: Schema) -> Canonical | None:
    """Канонический вид сырого значения атрибута (от продавца, LLM или из эталона)."""
    text = clean(raw)
    if not text:
        return None
    if spec.type == "quantity":
        q = parse_quantity(text, spec.unit_family or "")
        return Canonical(q.display, q.key, True) if q else None
    if spec.type == "enum":
        dictionary = schema.dictionaries.get(spec.dictionary or "", {})
        for canon in dictionary:
            if fold(canon) == fold(text):
                return Canonical(canon, fold(canon), True)
        matches = find_in_dictionary(text, dictionary)
        if matches:
            return Canonical(matches[0].canonical, fold(matches[0].canonical), True)
        return Canonical(text, fold(text), False)
    if spec.type == "pattern":
        for match in find_patterns(text, spec.name):
            return Canonical(match.canonical, fold(match.canonical), True)
        return Canonical(text, fold(text), False)
    return Canonical(text, _text_key(text, schema), False)


def _text_key(text: str, schema: Schema) -> str:
    """Ключ свободного текста (модель): без регистра и без ведущего бренда — «Samsung Galaxy S24» == «Galaxy S24»."""
    key = fold(text)
    for match in find_in_dictionary(text, schema.dictionaries.get("brands", {})):
        if match.start == 0:
            key = fold(text[match.end:])
            break
    return key.replace(" ", "")
