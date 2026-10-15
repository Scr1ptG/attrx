"""Чтение карточек и эталона, запись предсказаний. Форматы: JSONL, JSON (список), CSV."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from attrx.extractors.base import Card
from attrx.pipeline import CardResult

# Возможные имена колонок во входных данных; уточнить по датасету организаторов.
FIELD_ALIASES = {
    "id": ("id", "card_id", "sku", "product_id", "item_id"),
    "title": ("title", "name", "наименование", "название"),
    "description": ("description", "desc", "описание"),
    "seller_attrs": ("seller_attrs", "attributes", "attrs", "характеристики"),
    "category": ("category", "category_name", "категория"),
}
PREDICTION_COLUMNS = ["id", "category", "attribute", "value", "confidence", "status", "sources",
                      "evidence_field", "evidence_text", "conflict"]


def read_records(path: Path | str) -> Iterator[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)
    elif path.suffix == ".json":
        yield from json.loads(path.read_text(encoding="utf-8"))
    elif path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as f:
            yield from csv.DictReader(f)
    else:
        raise ValueError(f"Неподдерживаемый формат: {path}")


def _pick(record: dict, field: str):
    for alias in FIELD_ALIASES[field]:
        if alias in record and record[alias] not in (None, ""):
            return record[alias]
    return None


def _as_dict(value) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip().startswith(("{", "[")):
        parsed = json.loads(value)
        if isinstance(parsed, list):  # [{"name": ..., "value": ...}]
            return {str(i.get("name") or i.get("key")): i.get("value") for i in parsed if isinstance(i, dict)}
        return parsed
    return {}


def load_cards(path: Path | str) -> list[Card]:
    cards = []
    for i, record in enumerate(read_records(path)):
        cards.append(Card.build(
            id=_pick(record, "id") or str(i),
            title=_pick(record, "title") or "",
            description=_pick(record, "description") or "",
            seller_attrs=_as_dict(_pick(record, "seller_attrs")),
            category=_pick(record, "category"),
        ))
    return cards


def prediction_rows(results: Iterable[CardResult]) -> Iterator[dict]:
    for r in results:
        for p in r.predictions.values():
            yield {
                "id": r.card_id, "category": r.category or "", "attribute": p.attribute, "value": p.value,
                "confidence": p.confidence, "status": p.status, "sources": "+".join(p.sources),
                "evidence_field": p.evidence.field if p.evidence else "",
                "evidence_text": p.evidence.text if p.evidence else "", "conflict": p.conflict or "",
            }


def write_predictions(results: list[CardResult], path: Path | str) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(prediction_rows(results))
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=PREDICTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def load_predictions(path: Path | str) -> tuple[dict[tuple[str, str], tuple[str, float]], dict[str, str]]:
    """{(id, атрибут): (значение, уверенность)}, {id: категория}"""
    preds, categories = {}, {}
    for row in read_records(path):
        preds[(str(row["id"]), row["attribute"])] = (row["value"], float(row["confidence"]))
        if row.get("category"):
            categories[str(row["id"])] = row["category"]
    return preds, categories


def load_gold(path: Path | str) -> tuple[dict[str, dict[str, str]], dict[str, str]]:
    """Эталон в широком формате {"id", "category", "attributes": {...}} или длинном CSV (id, attribute, value)."""
    gold: dict[str, dict[str, str]] = {}
    categories: dict[str, str] = {}
    for row in read_records(path):
        card_id = str(_pick(row, "id"))
        if "attribute" in row:
            gold.setdefault(card_id, {})
            if row.get("value"):
                gold[card_id][row["attribute"]] = row["value"]
        else:
            gold[card_id] = {k: v for k, v in _as_dict(row.get("attributes")).items() if v not in (None, "")}
        if _pick(row, "category"):
            categories[card_id] = _pick(row, "category")
    return gold, categories
