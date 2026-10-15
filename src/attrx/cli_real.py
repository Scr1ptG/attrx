"""Команды CLI для реального каталога М.Видео: build-data → train → run → evaluate.

  python -m attrx build-data   --data-dir D   # рабочий набор, групповое разбиение, схема категорий, индекс аналогов
  python -m attrx train        --data-dir D   # справочники, шаблоны, аналоги (train) + верификатор и порог (val)
  python -m attrx run          --data-dir D --sample test_core | --cards файл  [--llm | --local-llm путь]
  python -m attrx evaluate     --data-dir D --pred results/predictions.csv --sample test_core
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from attrx.configs import CONFIGS, MAIN, REVIEW_THRESHOLD  # noqa: F401 — CONFIGS/MAIN импортируют ноутбуки

DEFAULT_DATA_DIR = os.environ.get("ATTRX_DATA_DIR", "data/real")
ASKABLE_FILE = "askable.json"  # какие поля категории спрашивать у LLM (пишет train)
CARD_ALIASES = {"id": "sku_id", "offer_id": "sku_id", "name": "title", "categoryId": "category_id", "category": "category_id"}


def _paths(args):
    from attrx.catalog import DataPaths

    return DataPaths(Path(args.data_dir))


def cmd_build_data(args: argparse.Namespace) -> int:
    from attrx.catalog import build

    t = time.time()
    stats = build(_paths(args))
    print(json.dumps(stats | {"seconds": round(time.time() - t, 1)}, ensure_ascii=False, indent=2))
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from attrx.train import build_stack, fit_askable, fit_components, fit_verifier, load_artifact, load_judge, save

    paths = _paths(args)
    comp = load_artifact(paths, "components") if args.reuse_components else None
    if args.reuse_components and comp is None:
        raise SystemExit("--reuse-components: нет components.pkl, запустите train без этого флага")
    if comp is None:
        comp = fit_components(paths)
        save(comp, paths, "components")
    verifiers = {}
    for name, kw in CONFIGS.items():
        if args.configs and name not in args.configs:
            continue
        if kw.get("judge") and load_judge(paths, kw["judge"]) is None:
            print(f"пропуск {name}: судья {kw['judge']} не обучен (notebooks/05_judge.ipynb)")
            continue
        verifiers[name] = fit_verifier(build_stack(comp, name, {}, paths, verifier=False), paths)
    save(verifiers, paths, "verifiers")
    askable = fit_askable(paths)
    (paths.work / "artifacts" / ASKABLE_FILE).write_text(json.dumps(askable, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"timings": comp.timings, "patterns": comp.patterns.stats, "analogs": comp.analogs.stats,
                      "main_config": MAIN, "llm_askable": {"categories": len(askable),
                                                           "attributes": sum(map(len, askable.values()))},
                      "verifiers": {k: v[0].report | {"accept_threshold": round(v[1], 4)} for k, v in verifiers.items()}},
                     ensure_ascii=False, indent=2))
    return 0


def _load_cards(args, paths):
    import polars as pl

    from attrx import metrics as M
    from attrx.catalog import load

    if args.sample:
        cards, cats = load(paths, "cards"), load(paths, "categories")
        presets = {"test_core": dict(split="test"), "test_full": dict(split="test", top_k=None, per_category=None, per_group=200),
                   "val_core": dict(split="val")}
        return M.eval_sample(cards, cats, **presets[args.sample])
    path = Path(args.cards)
    if path.suffix == ".parquet":
        df = pl.read_parquet(path)
    elif path.suffix == ".csv":
        df = pl.read_csv(path, infer_schema=False)
    else:
        df = pl.read_ndjson(path, infer_schema_length=None)
    df = df.rename({k: v for k, v in CARD_ALIASES.items() if k in df.columns and v not in df.columns})
    if "description" not in df.columns:
        df = df.with_columns(pl.lit("").alias("description"))
    return df.select(pl.col("sku_id").cast(pl.String), pl.col("category_id").cast(pl.String), "title",
                     pl.col("description").fill_null(""))


def cmd_run(args: argparse.Namespace) -> int:
    from attrx.train import assign_status, build_stack, load_artifact

    if args.llm:  # ошибка конфигурации — сразу, а не после минут обработки
        missing = [v for v in ("OPENAI_API_KEY",) if not os.getenv(v)]
        if not (args.llm_model or os.getenv("ATTRX_LLM_MODEL")):
            missing.append("ATTRX_LLM_MODEL (или --llm-model)")
        if missing:
            raise SystemExit(f"--llm: не заданы {', '.join(missing)}")
    paths = _paths(args)
    comp, verifiers = load_artifact(paths, "components"), load_artifact(paths, "verifiers")
    if comp is None or verifiers is None:
        raise SystemExit("Нет обученных артефактов: сначала python -m attrx train --data-dir ...")
    name = args.config or (MAIN if not args.no_analogs else "patterns")
    if name not in verifiers:
        raise SystemExit(f"Нет верификатора для {name}: python -m attrx train (доступны: {', '.join(verifiers)})")
    thr = verifiers[name][1]
    stack = build_stack(comp, name, verifiers, paths)
    llm_stats = llm_config = None
    if args.llm or args.local_llm:
        stack.gap_filler = _llm_filler(args, paths, comp)
        llm_config = {"model": stack.gap_filler.model, "max_calls": args.max_llm_calls, "min_expected": args.min_expected,
                      "askable_categories": len(stack.gap_filler.askable)}
    cards = _load_cards(args, paths)
    if args.limit:
        cards = cards.head(args.limit)
    t = time.perf_counter()
    frame = assign_status(stack.predict_frame(cards), thr, review=args.review_threshold, text_required=not args.publish_analogs,
                          judge_accept=verifiers[name][0].judge_threshold)
    elapsed = time.perf_counter() - t
    if stack.gap_filler is not None:
        llm_stats = stack.gap_filler.stats.as_dict()
        if llm_stats["calls"] + llm_stats["cache_hits"] == 0 and llm_stats["cards"]:
            print("ВНИМАНИЕ: LLM-режим включён, но ни одного ответа LLM не получено")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["sku_id", "category_id", "attribute", "value", "unit", "confidence", "status", "source", "evidence_field",
            "evidence_text"]
    frame.select(cols).sort("sku_id", "status", "confidence", descending=[False, False, True]).write_csv(out)
    counts = dict(frame.group_by("status").len().iter_rows())
    known = cards["category_id"].is_in(list(comp.lexicon.slots)).sum()
    stats = {"config": name, "cards": cards.height, "cards_unknown_category": int(cards.height - known),
             "cards_with_predictions": frame["sku_id"].n_unique(), "predictions": frame.height, "status": counts,
             "accept_threshold": round(thr, 4), "seconds": round(elapsed, 2),
             "ms_per_card": round(1000 * elapsed / max(cards.height, 1), 2), "llm_config": llm_config, "llm": llm_stats}
    out.with_name("run_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


def _llm_filler(args, paths, comp):
    from attrx.catalog import load
    from attrx.llm_fill import LLMFiller

    askable_path = paths.work / "artifacts" / ASKABLE_FILE
    if not askable_path.exists():
        raise SystemExit(f"Нет {ASKABLE_FILE}: переобучите python -m attrx train (список полей для LLM строится на val)")
    askable = json.loads(askable_path.read_text(encoding="utf-8"))
    if not askable:
        raise SystemExit(f"{ASKABLE_FILE} пуст: LLM нечего спрашивать")
    if args.local_llm:
        from attrx.local_llm import LocalChat

        client, model = LocalChat(args.local_llm), Path(args.local_llm).name
    else:
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL"))
        model = args.llm_model or os.environ["ATTRX_LLM_MODEL"]
    names = dict(load(paths, "categories").select("id", "name").iter_rows())
    return LLMFiller(comp.lexicon, client, model, askable, names, cache_dir=paths.work / "llm_cache",
                     max_calls=args.max_llm_calls, min_expected=args.min_expected)


def cmd_evaluate(args: argparse.Namespace) -> int:
    import polars as pl

    from attrx import metrics as M
    from attrx.catalog import load
    from attrx.train import status_summary

    paths = _paths(args)
    gold, schema, cats = (load(paths, n) for n in ("gold", "schema", "categories"))
    cards = _load_cards(args, paths)
    if "seller" not in cards.columns:  # свой файл карточек: продавец нужен только для среза 1P/3P
        sellers = load(paths, "cards").select("sku_id", "seller")
        cards = cards.join(sellers, on="sku_id", how="left").with_columns(pl.col("seller").fill_null(""))
    preds = pl.read_csv(args.pred, infer_schema=False).with_columns(pl.col("confidence").cast(pl.Float64))
    preds = preds.filter(pl.col("status") != "reject").join(cards.select("sku_id"), on="sku_id", how="semi")
    covered = preds["sku_id"].n_unique() / max(cards.height, 1)
    if covered < 0.5:
        print(f"ВНИМАНИЕ: предсказания есть только у {covered:.0%} карточек выборки — "
              "файл предсказаний сделан для другой выборки или с --limit?")
    pairs = M.score_pairs(preds, gold, cards, schema, M.supported_flags(cards, gold, schema))
    head = M.headline(pairs, cats)
    lo, hi = M.bootstrap_ci(pairs, lambda p: M.traffic_weighted_f1(p, cats), n=args.bootstrap)
    status = status_summary(pairs)
    names = cats.select(pl.col("id").alias("category_id"), "name", "traffic_share")
    by_cat = (M.summarize(pairs, "category_id").join(names, on="category_id").sort("traffic_share", descending=True)
              .select("name", "traffic_share", "n_gold", "precision", "recall", "f1"))
    by_kind = M.summarize(pairs, "kind").select("kind", "n_gold", "precision", "recall", "f1", "coverage")
    lines = [f"# Метрики · {args.sample or args.cards}", "",
             f"Карточек: {cards.height}. Эталонных пар «карточка × характеристика схемы»: {head['n_gold']}.", "",
             "| Метрика | Значение |", "|---|---|"]
    for k in ("precision", "recall", "f1", "recall_supported", "coverage", "ece"):
        lines.append(f"| {k} | {head[k]:.4f} |")
    lines.append(f"| traffic_weighted_f1 | {head['traffic_weighted_f1']:.4f} [{lo:.4f}; {hi:.4f}] |")
    lines += ["", "## Статусы (нагрузка на модератора)", "", _md(status), "", "## По типу характеристики", "", _md(by_kind),
              "", "## Топ-15 категорий по трафику", "", _md(by_cat.head(15))]
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report.with_suffix(".json").write_text(json.dumps(head | {"traffic_weighted_f1_ci": [lo, hi]}, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("\n".join(lines))
    return 0


def _md(df) -> str:
    cols = df.columns
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.iter_rows():
        rows.append("| " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in r) + " |")
    return "\n".join(rows)


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    os.environ["ATTRX_DATA_DIR"] = args.data_dir
    uvicorn.run("attrx.app:app", host=args.host, port=args.port, log_level="info")
    return 0


def register(sub) -> None:
    b = sub.add_parser("build-data", help="реальные данные: рабочий набор, разбиение, схема, индекс аналогов")
    b.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    b.set_defaults(func=cmd_build_data)

    t = sub.add_parser("train", help="реальные данные: обучить каскад и верификатор")
    t.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    t.add_argument("--reuse-components", action="store_true",
                   help="взять готовый components.pkl и переобучить только верификаторы (после изменения признаков или судьи)")
    t.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=None,
                   help="обучить верификаторы только этих конфигураций (по умолчанию все)")
    t.set_defaults(func=cmd_train)

    r = sub.add_parser("run", help="реальные данные: заполнить характеристики карточек")
    r.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample", choices=["test_core", "test_full", "val_core"])
    src.add_argument("--cards", help="parquet / csv / jsonl: sku_id, category_id, title, description")
    r.add_argument("--out", default="results/predictions.csv")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--config", choices=list(CONFIGS), default=None, help=f"конфигурация каскада (по умолчанию {MAIN})")
    r.add_argument("--no-analogs", action="store_true", help="без аналогов и похожих товаров из каталога")
    r.add_argument("--publish-analogs", action="store_true", help="публиковать значения аналогов без проверки модератором")
    r.add_argument("--review-threshold", type=float, default=REVIEW_THRESHOLD)
    llm = r.add_mutually_exclusive_group()
    llm.add_argument("--llm", action="store_true", help="LLM для пропусков через OpenAI-совместимый шлюз (OPENAI_API_KEY)")
    llm.add_argument("--local-llm", default=None, help="путь к локальной модели (transformers, 4 бита)")
    r.add_argument("--llm-model", default=None)
    r.add_argument("--max-llm-calls", type=int, default=None, help="дневной бюджет вызовов LLM")
    r.add_argument("--min-expected", type=float, default=0.0,
                   help="маршрутизация: вызывать LLM, только если ожидаемое число находимых полей ≥ значения")
    r.set_defaults(func=cmd_run)

    e = sub.add_parser("evaluate", help="реальные данные: метрики по эталону каталога")
    e.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    e.add_argument("--pred", default="results/predictions.csv")
    src = e.add_mutually_exclusive_group(required=True)
    src.add_argument("--sample", choices=["test_core", "test_full", "val_core"])
    src.add_argument("--cards")
    e.add_argument("--bootstrap", type=int, default=200)
    e.add_argument("--report", default="reports/metrics.md")
    e.set_defaults(func=cmd_evaluate)

    v = sub.add_parser("serve", help="демо для модератора (FastAPI): http://host:8000")
    v.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    v.add_argument("--host", default="0.0.0.0")
    v.add_argument("--port", type=int, default=8000)
    v.set_defaults(func=cmd_serve)
