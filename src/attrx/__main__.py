"""CLI.

Реальный каталог:   python -m attrx build-data | train | run | evaluate   (см. attrx/cli_real.py)
Синтетический пример без данных: python -m attrx predict | eval | bench
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from attrx.evaluate import evaluate
from attrx.io import load_cards, load_gold, load_predictions, write_predictions
from attrx.llm import LLMExtractor
from attrx.merge import REVIEW_THRESHOLD
from attrx.pipeline import Pipeline
from attrx.schema import DEFAULT_CONFIG_DIR, Schema


def cmd_predict(args: argparse.Namespace) -> int:
    schema = Schema.load(args.config)
    llm = LLMExtractor.from_env(schema, args.llm_model, max_calls=args.max_llm_calls) if args.llm else None
    pipeline = Pipeline(schema, llm)
    cards = load_cards(args.cards)[: args.limit or None]
    started = time.perf_counter()
    results = [pipeline.run(card) for card in cards]
    elapsed = time.perf_counter() - started
    rows = write_predictions(results, args.out)
    required = sum(len(r.required) for r in results)
    filled = sum(len(r.required) - len(r.missing) for r in results)
    stats = {
        "cards": len(results),
        "predictions": rows,
        "required_slots": required,
        "filled_share": round(filled / required, 4) if required else 0.0,
        "cards_sent_to_llm": sum(r.llm_used for r in results),
        "llm": llm.stats.as_dict() if llm else None,
        "seconds": round(elapsed, 2),
        "ms_per_card": round(1000 * elapsed / max(len(results), 1), 2),
    }
    stats_path = Path(args.out).with_name("run_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    schema = Schema.load(args.config)
    predictions, pred_categories = load_predictions(args.pred)
    gold, gold_categories = load_gold(args.gold)
    report = evaluate(predictions, gold, schema, {**pred_categories, **gold_categories}, args.threshold)
    markdown = report.to_markdown(schema)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(markdown, encoding="utf-8")
        Path(args.report).with_suffix(".json").write_text(
            json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(markdown)
    return 0


def cmd_bench(args: argparse.Namespace) -> int:
    from attrx import bench

    schema = Schema.load(args.config)
    cards = load_cards(args.cards)[: args.limit or None]
    gold, categories = load_gold(args.gold)
    llm_factory = (lambda: LLMExtractor.from_env(schema, args.llm_model, max_calls=args.max_llm_calls)) if args.llm else None
    only = args.configs.split(",") if args.configs else None
    rows = bench.run_bench(cards, gold, categories, schema, llm_factory, only)
    print(bench.save(rows, args.report, args.price_per_1k_tokens))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="attrx", description="Извлечение атрибутов товаров из карточек")
    parser.add_argument("--config", default=DEFAULT_CONFIG_DIR, help="папка со schema.yaml и dictionaries/")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("predict", help="извлечь атрибуты из карточек")
    p.add_argument("--cards", required=True, help="карточки: .jsonl / .json / .csv")
    p.add_argument("--out", default="results/predictions.csv")
    p.add_argument("--llm", action="store_true", help="включить LLM для пропущенных атрибутов")
    p.add_argument("--llm-model", default=None, help="имя модели в шлюзе (или ATTRX_LLM_MODEL)")
    p.add_argument("--max-llm-calls", type=int, default=None, help="бюджет вызовов LLM на запуск")
    p.add_argument("--limit", type=int, default=0, help="обработать первые N карточек")
    p.set_defaults(func=cmd_predict)

    e = sub.add_parser("eval", help="посчитать метрики по эталону")
    e.add_argument("--pred", default="results/predictions.csv")
    e.add_argument("--gold", required=True)
    e.add_argument("--threshold", type=float, default=REVIEW_THRESHOLD)
    e.add_argument("--report", default="reports/metrics.md")
    e.set_defaults(func=cmd_eval)

    b = sub.add_parser("bench", help="сравнить подходы (базлайн, LLM, гибриды) на одном эталоне")
    b.add_argument("--cards", required=True)
    b.add_argument("--gold", required=True)
    b.add_argument("--configs", default="", help="через запятую; по умолчанию все доступные")
    b.add_argument("--llm", action="store_true", help="включить конфигурации с LLM")
    b.add_argument("--llm-model", default=None)
    b.add_argument("--max-llm-calls", type=int, default=None)
    b.add_argument("--price-per-1k-tokens", type=float, default=None, help="цена шлюза, ₽ за 1000 токенов")
    b.add_argument("--limit", type=int, default=0)
    b.add_argument("--report", default="reports/benchmark.md")
    b.set_defaults(func=cmd_bench)

    from attrx import cli_real

    cli_real.register(sub)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # консоль Windows по умолчанию в cp1251
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
