"""Арбитраж кандидатов из разных источников и итоговая уверенность."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from attrx.extractors.base import Candidate, Evidence

ACCEPT_THRESHOLD = 0.8  # сразу в каталог
REVIEW_THRESHOLD = 0.5  # в очередь модерации; ниже — не публикуем
CONFLICT_FACTOR = 0.7  # источники противоречат друг другу
MAX_CONFIDENCE = 0.99


@dataclass(frozen=True)
class Prediction:
    attribute: str
    value: str
    key: str
    confidence: float
    sources: tuple[str, ...]
    evidence: Evidence | None
    conflict: str | None = None  # альтернативное значение из другого источника

    @property
    def status(self) -> str:
        if self.confidence >= ACCEPT_THRESHOLD:
            return "accept"
        return "review" if self.confidence >= REVIEW_THRESHOLD else "reject"


def merge(candidates: list[Candidate]) -> dict[str, Prediction]:
    """Для каждого атрибута выбирает значение с наибольшей совокупной поддержкой.

    Согласные источники усиливают друг друга (noisy-or), противоречие снижает уверенность
    и сохраняется в поле conflict — это сигнал о качестве карточки продавца."""
    by_attribute: dict[str, dict[str, list[Candidate]]] = defaultdict(lambda: defaultdict(list))
    for c in candidates:
        by_attribute[c.attribute][c.key].append(c)
    result = {}
    for attribute, groups in by_attribute.items():
        scored = sorted(((_noisy_or(g), g) for g in groups.values()), key=lambda x: (-x[0], -_priority(x[1])))
        best_score, best = scored[0]
        rivals = [g for s, g in scored[1:] if s >= REVIEW_THRESHOLD]
        confidence = best_score * (CONFLICT_FACTOR if rivals else 1.0)
        lead = max(best, key=lambda c: (c.evidence is not None and not c.evidence.field.startswith("seller"), c.confidence))
        result[attribute] = Prediction(
            attribute=attribute,
            value=max(best, key=lambda c: c.confidence).value,
            key=lead.key,
            confidence=round(min(confidence, MAX_CONFIDENCE), 4),
            sources=tuple(sorted({c.source for c in best})),
            evidence=lead.evidence,
            conflict=rivals[0][0].value if rivals else None,
        )
    return result


def _noisy_or(group: list[Candidate]) -> float:
    miss = 1.0
    for source in {c.source for c in group}:  # один источник не усиливает сам себя
        miss *= 1 - max(c.confidence for c in group if c.source == source)
    return 1 - miss


def _priority(group: list[Candidate]) -> int:
    """При равной уверенности предпочитаем продавца и значения из справочника."""
    return int(any(c.source == "seller" for c in group)) + int(any(c.known for c in group))
