"""Конфигурации каскада и политика статусов — одно место для CLI, демо и ноутбуков.

Для каждой конфигурации `train` обучает свой верификатор: от набора источников зависят признаки кандидатов.
"""

from __future__ import annotations

# имя -> аргументы Components.stack(...)
CONFIGS: dict[str, dict] = {
    "patterns": dict(analogs=False),
    "patterns+analogs": dict(similar=False),
    "patterns+analogs+similar": dict(),
    # + судья (attrx.judge) на основе открытой модели; верификатор обучается, только если судья уже обучен
    "patterns+analogs+similar+judge": dict(judge="laya"),  # основа — выбранная на val в notebooks/05_judge.ipynb
}
MAIN = "patterns+analogs+similar"  # основная конфигурация: python -m attrx run, демо

TARGET_PRECISION = 0.95  # точность авто-публикации на val, по ней выбирается порог accept
REVIEW_THRESHOLD = 0.5  # ниже — значение не показываем (reject)

__all__ = ["CONFIGS", "MAIN", "TARGET_PRECISION", "REVIEW_THRESHOLD"]
