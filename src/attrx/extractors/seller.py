"""Кандидаты из атрибутов продавца: сопоставление ключей по синонимам и нормализация значений."""

from __future__ import annotations

from attrx.extractors.base import Candidate, Card, Evidence
from attrx.normalize import canonicalize
from attrx.schema import Schema
from attrx.text import fold

SELLER_CONFIDENCE = 0.9
UNKNOWN_VALUE_PENALTY = 0.15  # значение вне справочника (например, маркетинговый цвет)


def map_seller_key(key: str, schema: Schema) -> str | None:
    """Атрибут схемы по ключу продавца: синоним должен входить в ключ целыми словами
    («Цвет корпуса» -> color, но «Тип аккумулятора» не -> battery_capacity)."""
    padded = f" {fold(key)} "
    for synonym, attribute in schema.seller_key_index:
        if f" {synonym} " in padded:
            return attribute
    return None


def extract_seller(card: Card, schema: Schema, attributes: tuple[str, ...]) -> list[Candidate]:
    candidates = []
    for key, raw in card.seller_attrs.items():
        attribute = map_seller_key(key, schema)
        if attribute not in attributes:
            continue
        canon = canonicalize(schema.attributes[attribute], raw, schema)
        if canon is None:
            continue
        confidence = SELLER_CONFIDENCE - (0 if canon.known else UNKNOWN_VALUE_PENALTY)
        candidates.append(Candidate(
            attribute, canon.display, canon.key, confidence, "seller",
            Evidence(f"seller:{key}", 0, len(raw), raw), canon.known,
        ))
    return candidates
