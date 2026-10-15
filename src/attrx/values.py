"""Токенизация и сравнение значений характеристик из реального каталога.

Значения в каталоге и в тексте записаны по-разному: «256 ГБ» и «256Gb», «Wi-Fi» и «wifi»,
«1 ТБ» и «1024 ГБ», «металл/ стекло» и «стекло, металл». Здесь одна точка правды о том,
когда два значения считаются одинаковыми (для оценки) и как искать значение в тексте (для извлечения).
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

from attrx.text import clean

# Единица -> (измерение, множитель к базовой единице измерения).
UNITS: dict[str, tuple[str, float]] = {
    "мм": ("length", 1e-3), "mm": ("length", 1e-3), "см": ("length", 1e-2), "cm": ("length", 1e-2),
    "м": ("length", 1.0), "дюйм": ("inch", 1.0), "дюйма": ("inch", 1.0), "дюймов": ("inch", 1.0),
    '"': ("inch", 1.0), "inch": ("inch", 1.0),
    "г": ("mass", 1e-3), "гр": ("mass", 1e-3), "кг": ("mass", 1.0), "kg": ("mass", 1.0),
    "мл": ("volume", 1e-3), "ml": ("volume", 1e-3), "л": ("volume", 1.0),
    "вт": ("power", 1.0), "w": ("power", 1.0), "квт": ("power", 1e3), "kw": ("power", 1e3),
    "мач": ("charge", 1.0), "mah": ("charge", 1.0),
    "мб": ("data", 1 / 1024), "mb": ("data", 1 / 1024), "гб": ("data", 1.0), "gb": ("data", 1.0),
    "тб": ("data", 1024.0), "tb": ("data", 1024.0),
    "гц": ("freq", 1.0), "hz": ("freq", 1.0), "кгц": ("freq", 1e3), "мгц": ("freq", 1e6), "ггц": ("freq", 1e9),
    "ghz": ("freq", 1e9),
    "ч": ("time", 3600.0), "час": ("time", 3600.0), "часов": ("time", 3600.0), "мин": ("time", 60.0),
    "мпикс": ("mpix", 1.0), "мп": ("mpix", 1.0), "mp": ("mpix", 1.0),
    "шт": ("pcs", 1.0), "дб": ("db", 1.0), "db": ("db", 1.0), "пикс": ("px", 1.0),
    "в": ("volt", 1.0), "v": ("volt", 1.0), "а": ("amp", 1.0),
}
# Латинские и составные написания -> русская форма (для поиска n-грамм).
TOKEN_ALIASES = {"gb": "гб", "tb": "тб", "mb": "мб", "mah": "мач", "hz": "гц", "ghz": "ггц", "kg": "кг",
                 "mm": "мм", "cm": "см", "ml": "мл", "w": "вт", "kw": "квт", "mp": "мп", "ё": "е"}

_TAGS = re.compile(r"<[^>]+>")
_BREAKS = re.compile(r"(?i)<\s*(br|/p|p|li|/li|/div|tr|h\d)[^>]*>|[\r\n]+")
_TOKEN = re.compile(r"\d+(?:[.,]\d+)?|[^\W\d_]+|\"")
_NUMBER = re.compile(r"^-?\d+(?:[.,]\d+)?$")
_QUANTITY = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*([^\d\s].*)?$")
_SPLIT = re.compile(r"\s*(?:/|;|,(?!\d)|\+)\s*")
LINE_BREAK = " ¶ "  # видимый разделитель строк описания: нужен для пар «ключ: значение»


def card_text(raw: str | None) -> str:
    """Текст карточки без HTML: переносы строк сохраняются как ¶, остальное — как в text.clean."""
    if not raw:
        return ""
    text = _BREAKS.sub(LINE_BREAK, str(raw))
    text = html.unescape(_TAGS.sub(" ", text))
    return clean(text)


@dataclass(frozen=True)
class Token:
    norm: str
    start: int
    end: int


def tokens(text: str) -> list[Token]:
    """Слова и числа в нижнем регистре; '256Gb' -> ['256', 'гб'], '6,7' -> '6.7'."""
    out = []
    for m in _TOKEN.finditer(text):
        t = m.group().lower().replace("ё", "е").replace(",", ".")
        out.append(Token(TOKEN_ALIASES.get(t, t), m.start(), m.end()))
    return out


def key(text: str | None) -> str:
    """Ключ значения для словаря и сравнения: нормализованные токены через пробел."""
    return " ".join(t.norm for t in tokens(text or ""))


def number(text: str) -> float | None:
    t = text.strip().replace(",", ".")
    return float(t) if _NUMBER.match(t) else None


def quantity(value: str, unit: str | None = None) -> tuple[str, float] | None:
    """'256 ГБ' -> ('data', 256.0); ('6.7', '"') -> ('inch', 6.7). Без единицы измерение = ''."""
    m = _QUANTITY.match(value or "")
    if not m:
        return None
    raw_unit = (m.group(2) or unit or "").strip().lower().replace("ё", "е").rstrip(".")
    amount = float(m.group(1).replace(",", "."))
    if not raw_unit:
        return "", amount
    dim = UNITS.get(raw_unit) or UNITS.get(TOKEN_ALIASES.get(raw_unit, raw_unit))
    if dim is None:
        return None if m.group(2) else ("", amount)
    return dim[0], amount * dim[1]


def _part_key(part: str, unit: str | None) -> str:
    q = quantity(part, unit)
    if q is not None:
        return f"{q[0]}:{round(q[1], 6):g}"
    return key(part)


def canon(value: str | None, unit: str | None = None) -> frozenset[str]:
    """Каноническое множество частей значения: порядок и разделители (/ , ; +) не важны."""
    parts = [p for p in _SPLIT.split(clean(value)) if p]
    return frozenset(k for k in (_part_key(p, unit) for p in parts) if k)


def same(pred: str | None, pred_unit: str | None, gold: str | None, gold_unit: str | None) -> bool:
    """Совпадение значения с эталоном после нормализации. Число без единицы сравнивается в единице эталона."""
    if not pred or not gold:
        return False
    gq, pq = quantity(gold, gold_unit), quantity(pred, pred_unit or gold_unit)
    if gq is not None and pq is not None:
        if gq[0] and pq[0] and gq[0] != pq[0]:
            return False
        return abs(gq[1] - pq[1]) <= 1e-6 + 1e-3 * abs(gq[1])
    return canon(pred, pred_unit) == canon(gold, gold_unit)
