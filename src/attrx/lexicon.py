"""Baseline без моделей: словари и правила, выученные на train-каталоге (docs/DATA_CONTRACT.md).

Источники кандидатов:
  kv      — пара «Ключ: значение» в описании, ключ совпадает с названием характеристики;
  lexicon — значение из справочника категории найдено в тексте (поиск n-грамм токенов);
  number  — число рядом с названием характеристики или с единицей, уникальной для категории;
  flag    — упоминание булевой характеристики («эффект памяти») -> «Да»;
  brand   — бренд из справочника брендов.

Каждый кандидат несёт evidence — фрагмент текста карточки. Значений без подтверждения не бывает.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from attrx.extractors.base import Candidate, Evidence
from attrx.merge import Prediction, merge
from attrx.values import LINE_BREAK, UNITS, Token, card_text, canon, key, tokens

BRAND = "Бренд"
MAX_NGRAM = 6
NEAR_WINDOW = 8
# Общие слова в названиях булевых характеристик: «Поддержка 3G» ищем как «3G».
FLAG_STOP = frozenset({"поддержка", "наличие", "функция", "есть", "режим", "технология", "встроенный", "встроенная"})
NEGATIONS = frozenset({"без", "нет", "не"})
YES = frozenset({"да", "есть", "имеется", "+"})
NO = frozenset({"нет", "отсутствует", "-"})
_KV = re.compile(r"(?:^|¶|•|;)\s*([^:¶•;]{2,60}?)\s*:\s*([^¶•;]{1,120})")

BASE = {"title": 0.8, "description": 0.65}


@dataclass(frozen=True)
class Slot:
    attribute: str
    kind: str  # bool | num | enum | text
    unit: str | None
    name: tuple[str, ...]  # токены названия
    stems: tuple[str, ...]  # основы значимых слов названия (для «рядом с названием»)


def _stems(name: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(t[:5] for t in name if len(t) >= 4 and t not in FLAG_STOP) or tuple(t for t in name if t)


def make_slot(attribute: str, kind: str, unit: str | None) -> Slot:
    name = tuple(key(attribute).split())
    return Slot(attribute, kind or "text", unit, name, _stems(name))


@dataclass
class Lexicon:
    slots: dict[str, dict[str, Slot]]  # категория -> характеристика -> слот
    values: dict[str, dict[tuple[str, ...], list[tuple[str, str, float]]]]  # категория -> n-грамма -> [(хар-ка, значение, prior)]
    brands: dict[tuple[str, ...], str]
    max_ngram: int = MAX_NGRAM
    stats: dict[str, int] = field(default_factory=dict)

    @classmethod
    def fit(cls, schema, reference, titles=None, min_count: int = 2, max_values: int = 3000,
            brand_precision: float = 0.5) -> Lexicon:
        """schema, reference — polars DataFrame из catalog.build (только train-группы).

        titles — train-карточки (title, brand): бренды, которые в названиях чаще означают не бренд
        («СМ» в «Стиральная машина … СМ»), отбрасываются по точности на train."""
        import polars as pl

        slots: dict[str, dict[str, Slot]] = defaultdict(dict)
        for row in schema.iter_rows(named=True):
            slots[row["category_id"]][row["attribute"]] = make_slot(row["attribute"], row["kind"], row["unit"])

        kinds = schema.select("category_id", "attribute", "kind")
        ref = (reference.join(kinds, on=["category_id", "attribute"], how="inner")
               .filter(pl.col("kind").is_in(["enum", "text"]) & (pl.col("attribute") != BRAND)
                       & (pl.col("count") >= min_count))
               .sort("count", descending=True)
               .group_by("category_id", "attribute", maintain_order=True).head(max_values))
        counts: dict[str, dict[tuple[str, ...], dict[str, list]]] = defaultdict(lambda: defaultdict(dict))
        for row in ref.iter_rows(named=True):
            k = tuple(key(row["value"]).split())
            if not k or (len(k) == 1 and (len(k[0]) < 2 or k[0].replace(".", "").isdigit())):
                continue
            entry = counts[row["category_id"]][k].setdefault(row["attribute"], [row["value"], 0])
            entry[1] += row["count"]
        values = {}
        for cat, by_key in counts.items():
            values[cat] = {}
            for k, attrs in by_key.items():
                total = sum(c for _, c in attrs.values())
                values[cat][k] = [(a, display, c / total) for a, (display, c) in attrs.items()]

        brand_counts: dict[tuple[str, ...], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for row in reference.filter(pl.col("attribute") == BRAND).iter_rows(named=True):
            k = tuple(key(row["value"]).split())
            if k and len("".join(k)) >= 2 and not "".join(k).isdigit():
                brand_counts[k][row["value"]] += row["count"]
        brands = {k: max(v.items(), key=lambda x: x[1])[0] for k, v in brand_counts.items()}
        dropped = 0
        if titles is not None:
            kept = _reliable_brands(brands, titles, brand_precision)
            dropped = len(brands) - len(kept)
            brands = kept
        lex = cls(dict(slots), values, brands)
        lex.stats = {"categories": len(slots), "slots": sum(len(s) for s in slots.values()),
                     "value_keys": sum(len(v) for v in values.values()), "brands": len(lex.brands), "brands_dropped": dropped}
        return lex

    # ---------------------------------------------------------------- извлечение
    def candidates(self, title: str, description: str, category: str | None) -> list[Candidate]:
        texts = {"title": card_text(title), "description": card_text(description)}
        toks = {f: tokens(t) for f, t in texts.items()}
        out = self._brand(texts, toks)
        slots = self.slots.get(category or "", {})
        if slots:
            out += self._kv(texts, slots, category)
            out += self._lexicon(texts, toks, slots, category)
            out += self._numbers(texts, toks, slots)
            out += self._flags(texts, toks, slots)
        return out

    def predict(self, title: str, description: str, category: str | None) -> dict[str, Prediction]:
        slots = self.slots.get(category or "", {})
        preds = merge(self.candidates(title, description, category))
        return {a: p for a, p in preds.items() if a in slots or a == BRAND}

    def predict_frame(self, cards):
        """cards: polars DataFrame (sku_id, category_id, title, description) -> таблица предсказаний."""
        import polars as pl

        rows = []
        for r in cards.select("sku_id", "category_id", "title", "description").iter_rows(named=True):
            slots = self.slots.get(r["category_id"], {})
            for attribute, p in self.predict(r["title"], r["description"], r["category_id"]).items():
                slot = slots.get(attribute)
                rows.append((r["sku_id"], attribute, p.value, slot.unit if slot and slot.kind == "num" else None,
                             p.confidence, "+".join(p.sources), p.evidence.field if p.evidence else None,
                             p.evidence.text if p.evidence else None))
        cols = ["sku_id", "attribute", "value", "unit", "confidence", "source", "evidence_field", "evidence_text"]
        types = [pl.String, pl.String, pl.String, pl.String, pl.Float64, pl.String, pl.String, pl.String]
        return pl.DataFrame(rows, schema=dict(zip(cols, types)), orient="row")

    def _brand(self, texts, toks) -> list[Candidate]:
        for fld, conf in (("title", 0.9), ("description", 0.6)):
            for i, n, k in _ngrams(toks[fld], 4, longest_first=True):
                if k in self.brands:
                    value = self.brands[k]
                    return [_cand(BRAND, value, conf, "brand", texts, fld, toks[fld], i, n)]
        return []

    def _kv(self, texts, slots, category) -> list[Candidate]:
        text = texts["description"]
        if ":" not in text:
            return []
        by_name = {" ".join(s.name): s for s in slots.values()}
        vocab = self.values.get(category, {})
        out = []
        for m in _KV.finditer(text):
            slot = by_name.get(key(m.group(1)))
            if slot is None:
                continue
            raw = m.group(2).strip().rstrip(".")
            start = m.start(2)
            ev = Evidence("description", start, start + len(raw), raw)
            value = _kv_value(slot, raw, vocab)
            if value:
                out.append(Candidate(slot.attribute, value, "|".join(sorted(canon(value))), 0.85, "kv", ev))
        return out

    def _lexicon(self, texts, toks, slots, category) -> list[Candidate]:
        vocab = self.values.get(category)
        if not vocab:
            return []
        best: dict[str, tuple[tuple, Candidate]] = {}
        for fld in ("title", "description"):
            tk = toks[fld]
            for i, n, k in _ngrams(tk, self.max_ngram):
                for attribute, display, prior in vocab.get(k, ()):
                    slot = slots.get(attribute)
                    if slot is None:
                        continue
                    near = _near(tk, i, slot.stems)
                    score = BASE[fld] * (0.5 + 0.5 * prior) + (0.15 if near else 0.0)
                    rank = (round(score, 6), n, -i)
                    if attribute not in best or rank > best[attribute][0]:
                        best[attribute] = (rank, _cand(attribute, display, min(score, 0.95), "lexicon", texts, fld, tk, i, n))
        return [c for _, c in best.values()]

    def _numbers(self, texts, toks, slots) -> list[Candidate]:
        num_slots = [s for s in slots.values() if s.kind == "num"]
        dims = defaultdict(list)
        for s in num_slots:
            if s.unit and _dim(s.unit):
                dims[_dim(s.unit)[0]].append(s)
        out = []
        for slot in num_slots:
            found = None
            for fld in ("title", "description"):
                found = _number_after_name(toks[fld], slot)
                if found:
                    i, n, value = found
                    out.append(_cand(slot.attribute, value, BASE[fld] + 0.05, "number", texts, fld, toks[fld], i, n))
                    break
            if found or not slot.unit or not _dim(slot.unit) or len(dims[_dim(slot.unit)[0]]) != 1:
                continue
            for fld in ("title", "description"):  # единица измерения уникальна для категории («мАч»)
                hits = _numbers_with_unit(toks[fld], slot)
                if hits:
                    i, n, value = hits[0]
                    conf = BASE[fld] - 0.1 - (0.15 if len({h[2] for h in hits}) > 1 else 0.0)
                    out.append(_cand(slot.attribute, value, conf, "number", texts, fld, toks[fld], i, n))
                    break
        return out

    def _flags(self, texts, toks, slots) -> list[Candidate]:
        out = []
        for slot in slots.values():
            if slot.kind != "bool":
                continue
            needle = tuple(t for t in slot.name if t not in FLAG_STOP) or slot.name
            for fld in ("title", "description"):
                i = _find(toks[fld], needle)
                if i is not None and not (i > 0 and toks[fld][i - 1].norm in NEGATIONS):
                    out.append(_cand(slot.attribute, "Да", BASE[fld] - 0.15, "flag", texts, fld, toks[fld], i, len(needle)))
                    break
        return out


def _reliable_brands(brands: dict, titles, min_precision: float, min_hits: int = 3) -> dict:
    """Точность «n-грамма в названии -> бренд карточки» на train; редкие n-граммы сохраняем."""
    hits: dict[tuple[str, ...], int] = defaultdict(int)
    good: dict[tuple[str, ...], int] = defaultdict(int)
    for title, brand in titles.select("title", "brand").iter_rows():
        own = tuple(key(brand).split()) if brand else ()
        seen = set()
        for _, _, k in _ngrams(tokens(card_text(title)), 4):
            if k in brands and k not in seen:
                seen.add(k)
                hits[k] += 1
                good[k] += k == own
    return {k: v for k, v in brands.items() if hits[k] < min_hits or good[k] / hits[k] >= min_precision}


# -------------------------------------------------------------------- помощники
def _ngrams(tk: list[Token], n_max: int, longest_first: bool = False):
    sizes = range(n_max, 0, -1) if longest_first else range(1, n_max + 1)
    for i in range(len(tk)):
        for n in sizes:
            if i + n <= len(tk):
                yield i, n, tuple(t.norm for t in tk[i:i + n])


def _near(tk: list[Token], i: int, stems: tuple[str, ...]) -> bool:
    lo = max(0, i - NEAR_WINDOW)
    return any(t.norm.startswith(s) for t in tk[lo:i] for s in stems)


def _find(tk: list[Token], needle: tuple[str, ...]) -> int | None:
    n = len(needle)
    if not n:
        return None
    for i in range(len(tk) - n + 1):
        if tk[i].norm == needle[0] and tuple(t.norm for t in tk[i:i + n]) == needle:
            return i
    return None


def _dim(unit: str | None) -> tuple[str, float] | None:
    if not unit:
        return None
    u = unit.strip().lower().rstrip(".")
    return UNITS.get(u)


def _fmt(x: float) -> str:
    return f"{round(x, 4):g}"


def _convert(amount: float, found_unit: str | None, slot: Slot) -> str | None:
    """Число в единицах слота. Несовместимая единица -> None."""
    target = _dim(slot.unit)
    if found_unit is None:
        return _fmt(amount)
    src = UNITS.get(found_unit)
    if src is None:
        return _fmt(amount)
    if target is None or src[0] != target[0]:
        return None
    return _fmt(amount * src[1] / target[1])


def _number_after_name(tk: list[Token], slot: Slot):
    """«Ёмкость аккумулятора — 5000 мАч»: число в пределах 6 токенов после названия."""
    stems = slot.stems
    if not stems:
        return None
    for j, t in enumerate(tk):
        if not t.norm.startswith(stems[0]):
            continue
        end = j + 1
        if len(stems) > 1:  # остальные значимые слова названия идут следом
            window = [x.norm for x in tk[j + 1:j + 1 + len(stems) + 1]]
            if not all(any(w.startswith(s) for w in window) for s in stems[1:]):
                continue
            end = j + len(stems)
        for k in range(end, min(end + 6, len(tk))):
            amount = _num(tk[k].norm)
            if amount is None:
                continue
            unit = tk[k + 1].norm if k + 1 < len(tk) and tk[k + 1].norm in UNITS else None
            value = _convert(amount, unit, slot)
            if value is not None:
                return k, 2 if unit else 1, value
            break
    return None


def _numbers_with_unit(tk: list[Token], slot: Slot):
    dim = _dim(slot.unit)
    hits = []
    for k in range(len(tk) - 1):
        amount = _num(tk[k].norm)
        src = UNITS.get(tk[k + 1].norm)
        if amount is not None and src is not None and src[0] == dim[0]:
            hits.append((k, 2, _fmt(amount * src[1] / dim[1])))
    return hits


def _num(t: str) -> float | None:
    try:
        return float(t) if t[0].isdigit() else None
    except ValueError:
        return None


def _kv_value(slot: Slot, raw: str, vocab) -> str | None:
    norm = key(raw)
    if slot.kind == "bool":
        first = norm.split(" ")[0] if norm else ""
        return "Да" if first in YES else "Нет" if first in NO else None
    if slot.kind == "num":
        tk = tokens(raw)
        for k, t in enumerate(tk):
            amount = _num(t.norm)
            if amount is not None:
                unit = tk[k + 1].norm if k + 1 < len(tk) and tk[k + 1].norm in UNITS else None
                return _convert(amount, unit, slot)
        return None
    for attribute, display, _ in vocab.get(tuple(norm.split()), ()):
        if attribute == slot.attribute:
            return display
    return raw if len(raw) <= 80 else None


def _cand(attribute, value, conf, source, texts, fld, tk, i, n) -> Candidate:
    start, end = tk[i].start, tk[i + n - 1].end
    ev = Evidence(fld, start, end, texts[fld][start:end])
    return Candidate(attribute, value, "|".join(sorted(canon(value))), round(conf, 4), source, ev)


__all__ = ["Lexicon", "Slot", "make_slot", "BRAND", "LINE_BREAK"]
