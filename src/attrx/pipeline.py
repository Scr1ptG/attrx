"""Каскад извлечения: продавец + правила (+ доп. экстракторы) -> LLM для пропусков -> арбитраж.

Источники включаются флагами, чтобы на одном коде мерить разные гибриды (см. bench.py)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from attrx.extractors.base import Candidate, Card
from attrx.extractors.rules import extract_rules
from attrx.extractors.seller import extract_seller
from attrx.llm import LLMExtractor
from attrx.merge import REVIEW_THRESHOLD, Prediction, merge
from attrx.schema import Schema

# Доп. экстрактор (ML/NER, эмбеддинги и т.п.): (карточка, схема, атрибуты) -> кандидаты
Extractor = Callable[[Card, Schema, tuple[str, ...]], list[Candidate]]


@dataclass(frozen=True)
class CardResult:
    card_id: str
    category: str | None
    required: tuple[str, ...]
    predictions: dict[str, Prediction]
    llm_used: bool

    @property
    def missing(self) -> list[str]:
        return [a for a in self.required if a not in self.predictions or self.predictions[a].status == "reject"]


@dataclass
class Pipeline:
    schema: Schema
    llm: LLMExtractor | None = None
    use_seller: bool = True
    use_rules: bool = True
    extractors: list[Extractor] = field(default_factory=list)
    llm_for_all: bool = False  # True — LLM по всем обязательным атрибутам (верхняя оценка качества/стоимости)

    def run(self, card: Card) -> CardResult:
        category = card.category if card.category in self.schema.categories else self.schema.detect_category(card.title)
        required = self.schema.required(category)
        if category != card.category:
            card = Card(card.id, card.title, card.description, card.seller_attrs, category)
        candidates: list[Candidate] = []
        if self.use_seller:
            candidates += extract_seller(card, self.schema, required)
        if self.use_rules:
            candidates += extract_rules(card, self.schema, required)
        for extractor in self.extractors:
            candidates += extractor(card, self.schema, required)
        predictions = merge(candidates)
        if self.llm_for_all:
            ask = list(required)
        else:
            ask = [a for a in required if a not in predictions or predictions[a].confidence < REVIEW_THRESHOLD]
        llm_used = False
        if self.llm is not None and ask:
            candidates += self.llm.extract(card, ask)
            predictions = merge(candidates)
            llm_used = True
        return CardResult(card.id, category, required, predictions, llm_used)
