"""Честная оценка: P/R/F1 по каждому атрибуту, покрытие, взвешивание по трафику, срез по категориям.

Правила подсчёта (на уровне пары карточка × атрибут, после нормализации обеих сторон):
  TP — предсказание совпало с эталоном;
  FP — предсказание неверно ИЛИ заполнено при пустом эталоне (выдумано);
  FN — эталон заполнен, а предсказания нет или оно неверно.
Неверное значение считается и FP, и FN — стандарт для slot filling.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from attrx.merge import REVIEW_THRESHOLD
from attrx.normalize import canonicalize
from attrx.schema import Schema

SWEEP = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    unsupported: int = 0  # FP при пустом эталоне: значение, которого нет в разметке
    cards: int = 0  # карточек, где атрибут обязателен
    filled: int = 0  # заполнено нами
    gold_filled: int = 0  # заполнено в эталоне

    def add(self, other: Counts) -> None:
        for k in self.__dict__:
            setattr(self, k, getattr(self, k) + getattr(other, k))

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def coverage(self) -> float:
        return self.filled / self.cards if self.cards else 0.0


@dataclass
class Report:
    threshold: float
    per_attribute: dict[str, Counts]
    per_category: dict[str, dict[str, Counts]]
    weights: dict[str, float]
    sweep: list[tuple[float, Counts]] = field(default_factory=list)

    @property
    def micro(self) -> Counts:
        total = Counts()
        for c in self.per_attribute.values():
            total.add(c)
        return total

    @property
    def macro_f1(self) -> float:
        values = [c.f1 for c in self.per_attribute.values()]
        return sum(values) / len(values) if values else 0.0

    @property
    def weighted_f1(self) -> float:
        total = sum(self.weights.get(a, 1.0) for a in self.per_attribute)
        return sum(c.f1 * self.weights.get(a, 1.0) for a, c in self.per_attribute.items()) / total if total else 0.0

    def as_dict(self) -> dict:
        m = self.micro
        return {
            "threshold": self.threshold,
            "micro": {"precision": m.precision, "recall": m.recall, "f1": m.f1, "coverage": m.coverage},
            "macro_f1": self.macro_f1,
            "traffic_weighted_f1": self.weighted_f1,
            "per_attribute": {a: {"precision": c.precision, "recall": c.recall, "f1": c.f1, "coverage": c.coverage,
                                  "support": c.gold_filled, "unsupported": c.unsupported}
                              for a, c in self.per_attribute.items()},
            "per_category_macro_f1": {cat: _macro(attrs) for cat, attrs in self.per_category.items()},
        }

    def to_markdown(self, schema: Schema) -> str:
        m = self.micro
        lines = [
            f"Порог публикации: уверенность ≥ {self.threshold}",
            "",
            "| Метрика | Значение |", "|---|---|",
            f"| Micro precision | {m.precision:.3f} |", f"| Micro recall | {m.recall:.3f} |",
            f"| Micro F1 | {m.f1:.3f} |", f"| Macro F1 (по атрибутам) | {self.macro_f1:.3f} |",
            f"| F1 с весами трафика | {self.weighted_f1:.3f} |",
            f"| Покрытие (наше / эталон) | {m.coverage:.3f} / {m.gold_filled / m.cards if m.cards else 0:.3f} |",
            f"| Значений без подтверждения в эталоне | {m.unsupported} |",
            "", "### По атрибутам", "",
            "| Атрибут | Вес | Precision | Recall | F1 | Покрытие | В эталоне | Без подтверждения |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for a, c in sorted(self.per_attribute.items(), key=lambda x: x[1].f1):
            title = schema.attributes[a].title if a in schema.attributes else a
            lines.append(f"| {title} | {self.weights.get(a, 1.0):g} | {c.precision:.3f} | {c.recall:.3f} | {c.f1:.3f} "
                         f"| {c.coverage:.3f} | {c.gold_filled} | {c.unsupported} |")
        lines += ["", "### По категориям (macro F1, худшие сверху)", "", "| Категория | Карточек | Macro F1 |", "|---|---|---|"]
        for cat, attrs in sorted(self.per_category.items(), key=lambda x: _macro(x[1])):
            n = max((c.cards for c in attrs.values()), default=0)
            lines.append(f"| {cat} | {n} | {_macro(attrs):.3f} |")
        if self.sweep:
            lines += ["", "### Точность ↔ покрытие при разных порогах", "",
                      "| Порог | Precision | Recall | F1 | Покрытие |", "|---|---|---|---|---|"]
            for t, c in self.sweep:
                lines.append(f"| {t:g} | {c.precision:.3f} | {c.recall:.3f} | {c.f1:.3f} | {c.coverage:.3f} |")
        return "\n".join(lines) + "\n"


def _macro(attrs: dict[str, Counts]) -> float:
    return sum(c.f1 for c in attrs.values()) / len(attrs) if attrs else 0.0


def evaluate(
    predictions: dict[tuple[str, str], tuple[str, float]],
    gold: dict[str, dict[str, str]],
    schema: Schema,
    categories: dict[str, str] | None = None,
    threshold: float = REVIEW_THRESHOLD,
    sweep: bool = True,
) -> Report:
    categories = categories or {}
    report = _evaluate_at(predictions, gold, schema, categories, threshold)
    if sweep:
        report.sweep = [(t, _evaluate_at(predictions, gold, schema, categories, t).micro) for t in SWEEP]
    return report


def calibration_error(predictions, gold, schema: Schema, bins: int = 10) -> tuple[float, list[tuple[float, float, int]]]:
    """ECE: насколько уверенность совпадает с реальной долей верных значений.

    Возвращает (ECE, [(средняя уверенность, точность, n) по корзинам]). ECE = 0 — «0.9» действительно значит 90%."""
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for (card_id, attribute), (value, conf) in predictions.items():
        if card_id not in gold:
            continue
        correct = _key(schema, attribute, value) == _key(schema, attribute, gold[card_id].get(attribute))
        buckets[min(int(conf * bins), bins - 1)].append((conf, correct))
    total = sum(len(b) for b in buckets.values())
    rows, ece = [], 0.0
    for i in sorted(buckets):
        items = buckets[i]
        mean_conf = sum(c for c, _ in items) / len(items)
        accuracy = sum(ok for _, ok in items) / len(items)
        ece += len(items) / total * abs(mean_conf - accuracy)
        rows.append((round(mean_conf, 3), round(accuracy, 3), len(items)))
    return round(ece, 4), rows


def _key(schema: Schema, attribute: str, value: str | None) -> str | None:
    if not value or attribute not in schema.attributes:
        return None
    canon = canonicalize(schema.attributes[attribute], value, schema)
    return canon.key if canon else None


def _evaluate_at(predictions, gold, schema: Schema, categories, threshold: float) -> Report:
    per_attribute: dict[str, Counts] = defaultdict(Counts)
    per_category: dict[str, dict[str, Counts]] = defaultdict(lambda: defaultdict(Counts))
    for card_id, gold_attrs in gold.items():
        category = categories.get(card_id)
        attributes = set(schema.required(category)) | {a for a in gold_attrs if a in schema.attributes}
        for a in attributes:
            pred_value, conf = predictions.get((card_id, a), (None, 0.0))
            p = _key(schema, a, pred_value) if conf >= threshold else None
            g = _key(schema, a, gold_attrs.get(a))
            c = Counts(cards=1, filled=int(p is not None), gold_filled=int(g is not None))
            if p is not None and p == g:
                c.tp = 1
            else:
                c.fp = int(p is not None)
                c.fn = int(g is not None)
                c.unsupported = int(p is not None and g is None)
            per_attribute[a].add(c)
            per_category[category or "unknown"][a].add(c)
    weights = {a: s.traffic_weight for a, s in schema.attributes.items()}
    return Report(threshold, dict(per_attribute), {k: dict(v) for k, v in per_category.items()}, weights)
