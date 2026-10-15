"""Стенд сравнения подходов: одни данные, одна оценка, разные гибриды.

Каждая конфигурация — фабрика пайплайна. Новые подходы (ML/NER, FRIDA, верификатор) добавляются
сюда одной строкой и автоматически попадают в отчёт reports/benchmark.md."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from attrx.evaluate import calibration_error, evaluate
from attrx.extractors.base import Card
from attrx.llm import LLMExtractor
from attrx.pipeline import Pipeline
from attrx.schema import Schema


@dataclass(frozen=True)
class Config:
    name: str
    title: str
    needs_llm: bool
    build: Callable[[Schema, LLMExtractor | None], Pipeline]


CONFIGS: list[Config] = [
    Config("seller", "Только атрибуты продавца", False,
           lambda s, _: Pipeline(s, use_rules=False)),
    Config("rules", "Только правила и словари", False,
           lambda s, _: Pipeline(s, use_seller=False)),
    Config("baseline", "Продавец + правила (базлайн)", False,
           lambda s, _: Pipeline(s)),
    Config("llm_only", "Только LLM по всем атрибутам", True,
           lambda s, llm: Pipeline(s, llm, use_seller=False, use_rules=False, llm_for_all=True)),
    Config("cascade", "Базлайн + LLM только для пропусков", True,
           lambda s, llm: Pipeline(s, llm)),
    # Подключаются по мере готовности (см. EXPERIMENTS.md):
    # ml_ner, frida_norm, frida_llm, verifier, distilled
]


@dataclass
class BenchRow:
    name: str
    title: str
    metrics: dict
    ece: float
    llm_calls: int
    tokens: int
    cards_to_llm: int
    ms_per_card: float
    cards: int

    def cost_per_million(self, price_per_1k_tokens: float | None) -> float | None:
        if price_per_1k_tokens is None or not self.cards:
            return None
        return self.tokens / self.cards * 1_000_000 / 1000 * price_per_1k_tokens


def run_bench(cards: list[Card], gold: dict, categories: dict, schema: Schema, llm_factory=None,
              only: list[str] | None = None) -> list[BenchRow]:
    rows = []
    for cfg in CONFIGS:
        if only and cfg.name not in only:
            continue
        llm = llm_factory() if cfg.needs_llm and llm_factory else None
        if cfg.needs_llm and llm is None:
            print(f"[bench] пропуск {cfg.name}: LLM недоступна")
            continue
        pipeline = cfg.build(schema, llm)
        started = time.perf_counter()
        results = [pipeline.run(card) for card in cards]
        elapsed = time.perf_counter() - started
        preds = {(r.card_id, p.attribute): (p.value, p.confidence) for r in results for p in r.predictions.values()}
        cats = {**{r.card_id: r.category for r in results if r.category}, **categories}
        report = evaluate(preds, gold, schema, cats, sweep=False)
        ece, _ = calibration_error(preds, gold, schema)
        stats = llm.stats if llm else None
        rows.append(BenchRow(
            cfg.name, cfg.title, report.as_dict(), ece,
            llm_calls=stats.calls if stats else 0,
            tokens=stats.tokens_equivalent if stats else 0,
            cards_to_llm=sum(r.llm_used for r in results),
            ms_per_card=round(1000 * elapsed / max(len(cards), 1), 2),
            cards=len(cards),
        ))
    return rows


def to_markdown(rows: list[BenchRow], price_per_1k_tokens: float | None = None) -> str:
    lines = [
        "| Подход | Precision | Recall | F1 | F1 (трафик) | Покрытие | Без подтверждения | ECE | Карточек в LLM | Токенов/карточку | мс/карточку | ₽ на 1 млн карточек |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        m = r.metrics["micro"]
        unsupported = sum(a["unsupported"] for a in r.metrics["per_attribute"].values())
        cost = r.cost_per_million(price_per_1k_tokens)
        lines.append(
            f"| {r.title} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {r.metrics['traffic_weighted_f1']:.3f} "
            f"| {m['coverage']:.3f} | {unsupported} | {r.ece:.3f} | {r.cards_to_llm / max(r.cards, 1):.0%} "
            f"| {r.tokens / max(r.cards, 1):.0f} | {r.ms_per_card} | {'—' if cost is None else f'{cost:,.0f}'} |"
        )
    lines += ["", "ECE — ошибка калибровки уверенности (0 — уверенность совпадает с реальной точностью).",
              "«Без подтверждения» — заполненные значения, которых нет в эталоне (риск ложных фильтров)."]
    return "\n".join(lines) + "\n"


def save(rows: list[BenchRow], path: Path | str, price_per_1k_tokens: float | None = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    markdown = to_markdown(rows, price_per_1k_tokens)
    path.write_text(markdown, encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps([r.__dict__ for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")
    return markdown
