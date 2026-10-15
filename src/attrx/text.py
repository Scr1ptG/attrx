"""Нормализация текста карточки перед извлечением."""

from __future__ import annotations

import re
import unicodedata

_SPACES = re.compile(r"\s+")
_QUOTES = str.maketrans(
    {"”": '"', "“": '"', "″": '"', "«": '"', "»": '"', "’": "'", "‘": "'", "′": "'", "–": "-", "—": "-", "‑": "-"}
)


def clean(text: str | None) -> str:
    """Unicode NFKC, единые кавычки и тире, схлопнутые пробелы. Регистр и длина смысла сохраняются."""
    if not text:
        return ""
    # NFKC раскладывает ″ на два штриха, поэтому замены делаем до и после нормализации.
    text = unicodedata.normalize("NFKC", str(text).translate(_QUOTES)).translate(_QUOTES)
    return _SPACES.sub(" ", text.replace("''", '"')).strip()


def fold(text: str | None) -> str:
    """Ключ для сравнения: нижний регистр, ё -> е, только буквы и цифры через пробел."""
    text = clean(text).lower().replace("ё", "е")
    return _SPACES.sub(" ", re.sub(r"[^\w.]+", " ", text)).strip()


def parse_number(raw: str) -> float:
    """'6,7' -> 6.7"""
    return float(raw.replace(",", "."))


def ru_plural(n: float, one: str, few: str, many: str) -> str:
    """Согласование существительного с числом: 1 дюйм, 2 дюйма, 5 дюймов, 6.7 дюйма."""
    if n != int(n):
        return few
    n = int(abs(n))
    if n % 10 == 1 and n % 100 != 11:
        return one
    if 2 <= n % 10 <= 4 and not 12 <= n % 100 <= 14:
        return few
    return many
