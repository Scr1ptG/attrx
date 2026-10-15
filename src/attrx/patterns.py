"""Выученные шаблоны составных чисел: «8/128Gb», «42.5x72.2x6.3 см», «1366x768», «120х190».

Вручную под каждую категорию шаблоны не пишем. На train для каждой категории считаем, какая позиция
кортежа чисел (или весь кортеж целиком) совпадает с какой характеристикой эталона.
Сохраняем отображения с достаточной точностью и поддержкой, точность становится уверенностью кандидата.

Пример выученного правила: категория «Смартфоны», кортеж «A/B ГБ»:
  позиция 0 -> «Оперативная память (RAM)», позиция 1 -> «Встроенная память (ROM)»,
  целиком -> «Оперативная и встроенная память».
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from attrx.extractors.base import Candidate, Evidence
from attrx.values import UNITS, canon, card_text, same

_NUM = r"\d+(?:[.,]\d+)?"
_UNIT = r"(?:\s*(?:мм|см|мл|мб|mb|гб|gb|тб|tb|пикс|px|дюйм\w*|кг|вт|w|м|л|г)(?![a-zа-я])|\")?"
TUPLE = re.compile(rf"(?<![\d.,])({_NUM})({_UNIT})\s*([x×х*/+])\s*({_NUM})({_UNIT})(?:\s*([x×х*/+])\s*({_NUM})({_UNIT}))?(?![\d])",
                   re.IGNORECASE)
SEP_CLASS = {"x": "x", "×": "x", "х": "x", "*": "x", "/": "/", "+": "/"}
UNIT_DISPLAY = {"гб": "ГБ", "gb": "ГБ", "тб": "ТБ", "tb": "ТБ", "мб": "МБ", "mb": "МБ", "мм": "мм", "см": "см", "м": "м",
                '"': '"', "кг": "кг", "г": "г", "л": "л", "мл": "мл", "вт": "Вт", "w": "Вт", "пикс": "Пикс", "px": "Пикс"}
WHOLE = -1  # позиция «весь кортеж целиком»


@dataclass(frozen=True)
class Match:
    field: str
    start: int
    end: int
    key: tuple[str, int]  # (класс разделителя, число элементов)
    numbers: tuple[float, ...]
    units: tuple[str | None, ...]


def find_tuples(text: str, fld: str) -> list[Match]:
    out = []
    for m in TUPLE.finditer(text):
        nums_raw = [m.group(1), m.group(4)] + ([m.group(7)] if m.group(7) else [])
        units_raw = [m.group(2), m.group(5)] + ([m.group(8)] if m.group(7) else [])
        if m.group(6) and SEP_CLASS[m.group(6).lower()] != SEP_CLASS[m.group(3).lower()]:
            continue
        units = [(u or "").strip().lower() or None for u in units_raw]
        units = [_unit_norm(u) for u in units]
        tail = units[-1]
        units = tuple(u or tail for u in units)  # «8/128Gb»: единица последнего числа относится ко всем
        nums = tuple(float(x.replace(",", ".")) for x in nums_raw)
        out.append(Match(fld, m.start(), m.end(), (SEP_CLASS[m.group(3).lower()], len(nums)), nums, units))
    return out


def _unit_norm(u: str | None) -> str | None:
    if not u:
        return None
    if u.startswith("дюйм"):
        return '"'
    return u if u in UNITS or u in UNIT_DISPLAY else None


def _fmt(x: float) -> str:
    return f"{round(x, 4):g}"


@dataclass(frozen=True)
class Target:
    attribute: str
    kind: str
    unit: str | None  # единица слота (для num)
    suffix: str | None  # типичная единица в значениях enum/text («ГБ»)


def render(target: Target, match: Match, position: int) -> str | None:
    """Значение характеристики из кортежа в формате каталога."""
    if position == WHOLE:
        unit = match.units[-1]
        disp = UNIT_DISPLAY.get(unit or "", target.suffix or "")
        if match.key[0] == "/":
            return "/".join(f"{_fmt(n)} {UNIT_DISPLAY.get(u or '', disp)}".strip() for n, u in zip(match.numbers, match.units))
        return ("*".join(_fmt(n) for n in match.numbers) + (f" {disp}" if disp else "")).strip()
    n, u = match.numbers[position], match.units[position]
    if target.kind == "num":
        if u and target.unit:
            src, dst = UNITS.get(u), UNITS.get(target.unit.strip().lower())
            if src and dst:
                if src[0] != dst[0]:
                    return None
                return _fmt(n * src[1] / dst[1])
        return _fmt(n)
    disp = UNIT_DISPLAY.get(u or "", target.suffix or "")
    return f"{_fmt(n)} {disp}".strip()


@dataclass
class TuplePatterns:
    """(категория, класс кортежа) -> [(позиция, характеристика, точность)]."""

    rules: dict[str, dict[tuple[str, int], list[tuple[int, Target, float]]]] = field(default_factory=dict)
    stats: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, cards, gold, schema, reference, per_category: int = 400, min_support: int = 5,
            min_precision: float = 0.6, seed: int = 13) -> TuplePatterns:
        """cards — train-карточки (sku_id, category_id, title, description); gold — эталон; schema/reference — train."""
        import polars as pl

        sample = (cards.sort("sku_id").with_columns(pl.int_range(pl.len()).shuffle(seed=seed).over("category_id").alias("_r"))
                  .filter(pl.col("_r") < per_category))
        targets = _targets(schema, reference)
        g = gold.join(sample.select("sku_id"), on="sku_id", how="semi")
        by_card: dict[str, dict[str, tuple[str, str | None]]] = defaultdict(dict)
        for sku, attr, value, unit in g.select("sku_id", "attribute", "value", "unit").iter_rows():
            by_card[sku][attr] = (value, unit)
        seen: dict[tuple, int] = defaultdict(int)
        hits: dict[tuple, int] = defaultdict(int)
        for sku, cat, title, desc in sample.select("sku_id", "category_id", "title", "description").iter_rows():
            cat_targets = targets.get(cat)
            if not cat_targets:
                continue
            truth = by_card.get(sku, {})
            counted = set()
            for fld, text in (("title", title), ("description", desc)):
                for m in find_tuples(card_text(text), fld):
                    for pos in (*range(m.key[1]), WHOLE):
                        for t in cat_targets:
                            k = (cat, m.key, pos, t.attribute)
                            if k in counted or t.attribute not in truth:
                                continue
                            counted.add(k)
                            seen[k] += 1
                            value = render(t, m, pos)
                            if value is not None and same(value, t.unit if t.kind == "num" else None, *truth[t.attribute]):
                                hits[k] += 1
        rules: dict[str, dict[tuple[str, int], list[tuple[int, Target, float]]]] = defaultdict(lambda: defaultdict(list))
        index = {(cat, t.attribute): t for cat, ts in targets.items() for t in ts}
        for k, n in seen.items():
            p = hits[k] / n
            if hits[k] >= min_support and p >= min_precision:
                cat, key, pos, attr = k
                rules[cat][key].append((pos, index[(cat, attr)], round(p, 4)))
        out = cls({c: dict(v) for c, v in rules.items()})
        out.stats = {"categories": len(out.rules), "rules": sum(len(r) for v in out.rules.values() for r in v.values())}
        return out

    def candidates(self, title: str, description: str, category: str | None) -> list[Candidate]:
        cat_rules = self.rules.get(category or "")
        if not cat_rules:
            return []
        out, used = [], set()
        texts = {"title": card_text(title), "description": card_text(description)}
        for fld in ("title", "description"):
            for m in find_tuples(texts[fld], fld):
                for pos, target, precision in cat_rules.get(m.key, ()):
                    if target.attribute in used:
                        continue
                    value = render(target, m, pos)
                    if value is None:
                        continue
                    used.add(target.attribute)
                    ev = Evidence(fld, m.start, m.end, texts[fld][m.start:m.end])
                    conf = precision if fld == "title" else precision * 0.95
                    out.append(Candidate(target.attribute, value, "|".join(sorted(canon(value))), round(conf, 4), "pattern", ev))
        return out


def _targets(schema, reference) -> dict[str, list[Target]]:
    """Числовые слоты и слоты, чьи значения — число с единицей («8 ГБ», «42.5*72.2*6.3 см»)."""
    import polars as pl

    suffix = (reference.filter(pl.col("value").str.contains(r"^\d"))
              .with_columns(pl.col("value").str.extract(r"([^\d\s.,*/x×]+)\s*$").alias("suffix"))
              .group_by("category_id", "attribute")
              .agg(pl.col("suffix").drop_nulls().mode().first(), (pl.col("count").sum()).alias("numeric_count")))
    total = reference.group_by("category_id", "attribute").agg(pl.col("count").sum().alias("total"))
    s = (schema.join(suffix, on=["category_id", "attribute"], how="left").join(total, on=["category_id", "attribute"], how="left")
         .with_columns((pl.col("numeric_count").fill_null(0) / pl.col("total").fill_null(1)).alias("numeric_share")))
    out: dict[str, list[Target]] = defaultdict(list)
    for r in s.filter((pl.col("kind") == "num") | (pl.col("numeric_share") >= 0.8)).iter_rows(named=True):
        out[r["category_id"]].append(Target(r["attribute"], r["kind"], r["unit"], r["suffix"]))
    return dict(out)

