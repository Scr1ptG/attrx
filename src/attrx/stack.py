"""Каскад источников кандидатов на реальном каталоге.

  словари (Lexicon) + выученные шаблоны (TuplePatterns) + [аналоги, модель, LLM]
      -> merge (noisy-or, конфликты) -> верификатор (откалиброванная вероятность «значение верно»)
      -> статус: accept (в каталог) / review (модератору) / reject

Каждый источник реализует candidates(title, description, category) -> list[Candidate].
Источники, которые подключаются позже (модели, LLM), не меняют остальной код.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol

from attrx.extractors.base import Candidate
from attrx.lexicon import BRAND, Lexicon
from attrx.merge import Prediction, merge

SOURCES = ("brand", "lexicon", "number", "flag", "kv", "pattern", "analog", "similar", "model", "llm", "judge")
COLUMNS = ["sku_id", "category_id", "attribute", "kind", "value", "unit", "confidence", "raw_confidence", "source",
           "n_sources", "n_values", "conflict", "evidence_field", "evidence_text"]


class Source(Protocol):
    def candidates(self, title: str, description: str, category: str | None) -> list[Candidate]: ...


# judge — значение аналога или похожего товара, которое судья нашёл подтверждённым в тексте карточки
TEXT_SOURCES = frozenset({"brand", "lexicon", "number", "flag", "kv", "pattern", "model", "llm", "judge"})


@dataclass
class Stack:
    lexicon: Lexicon
    sources: list[Source] = field(default_factory=list)
    verifier: object | None = None  # attrx.verifier.Verifier
    gap_filler: object | None = None  # attrx.llm_fill.LLMFiller — только для слотов без уверенного текстового кандидата
    gap_threshold: float = 0.6
    uncalibrated: tuple[str, ...] = ("llm",)  # источники, которых верификатор не видел: их уверенность не трогаем
    use_lexicon: bool = True  # False — только для замера «LLM без каскада»
    judge: object | None = None  # attrx.judge.Judge — проверка значений без текстового подтверждения

    def card(self, title: str, description: str, category: str | None) -> tuple[dict[str, Prediction], dict[str, int]]:
        slots = self.lexicon.slots.get(category or "", {})
        cands: list[Candidate] = self.lexicon.candidates(title, description, category) if self.use_lexicon else []
        for source in self.sources:
            cands += source.candidates(title, description, category)
        cands = [c for c in cands if c.attribute in slots or c.attribute == BRAND]
        if self.gap_filler is not None:
            covered = {c.attribute for c in cands if c.source in TEXT_SOURCES and c.confidence >= self.gap_threshold}
            cands += self.gap_filler.candidates_for(title, description, category, covered)
        values: dict[str, set[str]] = defaultdict(set)
        for c in cands:
            values[c.attribute].add(c.key)
        return merge(cands), {a: len(v) for a, v in values.items()}

    def llm_requests(self, cards) -> list[tuple[str, list[dict]]]:
        """Промпты, которые каскад отправит LLM (для пакетной генерации на своей GPU и оценки стоимости)."""
        out = []
        if self.gap_filler is None:
            return out
        for r in cards.select("sku_id", "category_id", "title", "description").iter_rows(named=True):
            slots = self.lexicon.slots.get(r["category_id"], {})
            cands = self.lexicon.candidates(r["title"], r["description"], r["category_id"]) if self.use_lexicon else []
            for source in self.sources:
                cands += source.candidates(r["title"], r["description"], r["category_id"])
            covered = {c.attribute for c in cands
                       if c.attribute in slots and c.source in TEXT_SOURCES and c.confidence >= self.gap_threshold}
            ask = self.gap_filler.slots_to_ask(r["category_id"], covered)
            if ask and self.gap_filler.expected_fields(r["category_id"], ask) >= self.gap_filler.min_expected:
                out.append((r["sku_id"], self.gap_filler.prompt(r["title"], r["description"], r["category_id"], ask)))
        return out

    def predict_frame(self, cards):
        """cards: polars DataFrame (sku_id, category_id, title, description) -> предсказания с признаками."""
        import polars as pl

        rows = []
        for r in cards.select("sku_id", "category_id", "title", "description").iter_rows(named=True):
            slots = self.lexicon.slots.get(r["category_id"], {})
            preds, n_values = self.card(r["title"], r["description"], r["category_id"])
            for attribute, p in preds.items():
                slot = slots.get(attribute)
                kind = slot.kind if slot else "text"
                rows.append((r["sku_id"], r["category_id"], attribute, kind, p.value, slot.unit if slot and kind == "num" else None,
                             p.confidence, p.confidence, "+".join(p.sources), len(p.sources), n_values.get(attribute, 1),
                             p.conflict is not None, p.evidence.field if p.evidence else None, p.evidence.text if p.evidence else None))
        types = [pl.String] * 6 + [pl.Float64, pl.Float64, pl.String, pl.Int32, pl.Int32, pl.Boolean, pl.String, pl.String]
        frame = pl.DataFrame(rows, schema=dict(zip(COLUMNS, types)), orient="row")
        if self.judge is not None and frame.height:
            frame = self.judge.annotate(frame, cards, self.lexicon)
        if self.verifier is not None and frame.height:
            # Верификатор не видел источников из uncalibrated (LLM): оцениваем строку так, будто их нет, —
            # согласие с LLM не повышает вероятность. Строки только от таких источников сохраняют свою уверенность.
            unc = list(self.uncalibrated)
            parts = pl.col("source").str.split("+")
            seen = frame.with_columns(
                parts.list.eval(pl.element().filter(~pl.element().is_in(unc))).list.join("+").alias("source"),
                (pl.col("n_sources") - parts.list.eval(pl.element().is_in(unc)).list.sum()).cast(pl.Int32).alias("n_sources"))
            calibrated = pl.Series("confidence", self.verifier.predict_proba(seen))
            keep = parts.list.eval(pl.element().is_in(unc)).list.all()
            frame = frame.with_columns(pl.when(keep).then(pl.col("raw_confidence")).otherwise(calibrated).alias("confidence"))
        if self.judge is not None and "judge_contra" in frame.columns:  # LLM и судья не согласны — не показываем
            # порог вето подобран на val по ответам LLM (05_judge): доля ложных вето среди верных значений ограничена
            veto = pl.col("source").str.contains(r"(^|\+)llm(\+|$)") & (pl.col("judge_contra") >= self.judge.veto)
            frame = frame.with_columns(pl.when(veto.fill_null(False)).then(0.0).otherwise(pl.col("confidence")).alias("confidence"))
        return frame
