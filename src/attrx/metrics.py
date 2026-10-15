"""Оценка на реальном эталоне каталога (docs/DATA_CONTRACT.md, разделы 4–5).

Единица оценки — пара «карточка × характеристика из схемы категории». Неверное значение = FP и FN,
пропуск при наличии эталона = FN, значение без эталона не штрафуется (учитывается как outside).
"""

from __future__ import annotations

import random
from collections.abc import Iterable

import polars as pl

from attrx.lexicon import BRAND, FLAG_STOP
from attrx.values import _SPLIT, card_text, key, same, tokens

ACCEPT = 0.8
TOP_K = 30


# ------------------------------------------------------------------ выборки
def eval_sample(cards: pl.DataFrame, categories: pl.DataFrame, split: str = "test", top_k: int | None = TOP_K,
                per_category: int | None = 60, per_group: int = 20, seed: int = 13) -> pl.DataFrame:
    """Стратифицированная выборка: топ-K категорий по трафику, ограничение на группу (продавца) и категорию."""
    sample = cards.filter(pl.col("split") == split).sort("sku_id")  # порядок строк после join не гарантирован
    if top_k:
        top = categories.sort("clicks", descending=True).head(top_k).select(pl.col("id").alias("category_id"))
        sample = sample.join(top, on="category_id", how="semi")
    sample = (sample.with_columns(pl.int_range(pl.len()).shuffle(seed=seed).alias("_r"))
              .sort("_r")
              .with_columns(pl.int_range(pl.len()).over("category_id", "group").alias("_g"))
              .filter(pl.col("_g") < per_group))
    if per_category:
        sample = sample.with_columns(pl.int_range(pl.len()).over("category_id").alias("_c")).filter(pl.col("_c") < per_category)
    return sample.drop([c for c in ("_r", "_g", "_c") if c in sample.columns]).sort("sku_id")


# ------------------------------------------------------------------ подтверждаемость
def _ngram_set(text: str, n_max: int = 6, limit: int = 4000) -> set[str]:
    tk = [t.norm for t in tokens(text)][:limit]
    return {" ".join(tk[i:i + n]) for i in range(len(tk)) for n in range(1, n_max + 1) if i + n <= len(tk)}


def _supported(grams: set[str], attribute: str, kind: str | None, value: str) -> bool:
    if kind == "bool":
        if key(value) != "да":
            return False
        name = [t for t in key(attribute).split() if t not in FLAG_STOP] or key(attribute).split()
        return " ".join(name) in grams
    parts = [key(p) for p in _SPLIT.split(value) if p.strip()]
    return bool(parts) and all(p in grams for p in parts if p)


def supported_flags(cards: pl.DataFrame, gold: pl.DataFrame, schema: pl.DataFrame) -> pl.DataFrame:
    """Для каждой эталонной пары схемы: подтверждается ли значение текстом карточки (контракт, раздел 2)."""
    pairs = (gold.join(cards.select("sku_id", "category_id"), on="sku_id", how="inner")
             .join(schema.select("category_id", "attribute", "kind"), on=["category_id", "attribute"], how="inner"))
    texts = {r["sku_id"]: card_text(r["title"]) + " ¶ " + card_text(r["description"])
             for r in cards.select("sku_id", "title", "description").iter_rows(named=True)}
    flags = []
    cache_id, grams = None, set()
    for r in pairs.sort("sku_id").iter_rows(named=True):
        if r["sku_id"] != cache_id:
            cache_id, grams = r["sku_id"], _ngram_set(texts[r["sku_id"]])
        flags.append((r["sku_id"], r["attribute"], _supported(grams, r["attribute"], r["kind"], r["value"])))
    return pl.DataFrame(flags, schema={"sku_id": pl.String, "attribute": pl.String, "supported": pl.Boolean}, orient="row")


# ------------------------------------------------------------------ пары и метрики
def score_pairs(preds: pl.DataFrame, gold: pl.DataFrame, cards: pl.DataFrame, schema: pl.DataFrame,
                supported: pl.DataFrame | None = None) -> pl.DataFrame:
    """preds: sku_id, attribute, value, unit, confidence[, source]. Возвращает таблицу пар с флагом correct."""
    targets = cards.select("sku_id", "category_id", "seller").join(
        schema.select("category_id", "attribute", "kind"), on="category_id", how="inner")
    g = gold.select("sku_id", "attribute", pl.col("value").alias("gold"), pl.col("unit").alias("gold_unit"))
    p = preds.select("sku_id", "attribute", pl.col("value").alias("pred"), pl.col("unit").alias("pred_unit"),
                     "confidence", *[c for c in ("source", "status", "evidence_field") if c in preds.columns])
    pairs = targets.join(g, on=["sku_id", "attribute"], how="left").join(p, on=["sku_id", "attribute"], how="left")
    pairs = pairs.with_columns(
        pl.struct("pred", "pred_unit", "gold", "gold_unit").map_elements(
            lambda r: same(r["pred"], r["pred_unit"], r["gold"], r["gold_unit"]), return_dtype=pl.Boolean).alias("correct"),
        pl.when(pl.col("seller") == "М.Видео").then(pl.lit("1P")).otherwise(pl.lit("3P")).alias("seller_type"))
    if supported is not None:
        pairs = pairs.join(supported, on=["sku_id", "attribute"], how="left").with_columns(pl.col("supported").fill_null(False))
    return pairs


def summarize(pairs: pl.DataFrame, by: str | list[str] | None = None, accept: float = ACCEPT) -> pl.DataFrame:
    """Метрики по парам. precision_accept и auto_fill берутся из колонки status (политика train.assign_status),
    а если её нет — по фиксированному порогу уверенности `accept` (для сырых источников без статусов)."""
    has_gold, has_pred = pl.col("gold").is_not_null(), pl.col("pred").is_not_null()
    accepted = (pl.col("status") == "accept").fill_null(False) if "status" in pairs.columns else pl.col("confidence") >= accept
    tp = (pl.col("correct") & has_gold & has_pred).sum()
    pred_in_gold = (has_pred & has_gold).sum()
    aggs = [
        has_gold.sum().alias("n_gold"),
        pred_in_gold.alias("n_pred"),
        (has_pred & ~has_gold).sum().alias("outside"),
        tp.alias("tp"),
        (tp / pred_in_gold).alias("precision"),
        (tp / has_gold.sum()).alias("recall"),
        (pred_in_gold / has_gold.sum()).alias("coverage"),
        ((pl.col("correct") & has_gold & accepted).sum() / (has_pred & has_gold & accepted).sum()).alias("precision_accept"),
        ((has_pred & has_gold & accepted).sum() / has_gold.sum()).alias("auto_fill"),
    ]
    if "supported" in pairs.columns:
        aggs += [(pl.col("supported") & has_gold).sum().alias("n_supported"),
                 ((pl.col("correct") & has_pred & pl.col("supported")).sum() / (pl.col("supported") & has_gold).sum()).alias("recall_supported")]
    out = pairs.group_by(by).agg(aggs) if by else pairs.select(aggs)
    out = out.with_columns((2 * pl.col("precision") * pl.col("recall") / (pl.col("precision") + pl.col("recall")))
                           .fill_nan(0.0).alias("f1"))
    return out.sort("n_gold", descending=True) if by else out


def traffic_weighted_f1(pairs: pl.DataFrame, categories: pl.DataFrame) -> float:
    per = summarize(pairs, "category_id").join(categories.select(pl.col("id").alias("category_id"), "clicks"),
                                               on="category_id", how="left").fill_null(0)
    per = per.filter(pl.col("n_gold") > 0)
    w = per["clicks"].sum()
    return float((per["f1"].fill_null(0) * per["clicks"]).sum() / w) if w else float("nan")


def bootstrap_ci(pairs: pl.DataFrame, fn, n: int = 200, seed: int = 13, alpha: float = 0.05) -> tuple[float, float]:
    """Доверительный интервал метрики: ресэмплинг карточек (а не пар — пары одной карточки зависимы)."""
    ids = pairs["sku_id"].unique().sort().to_list()
    rng = random.Random(seed)
    grouped = {k[0]: df for k, df in pairs.partition_by("sku_id", as_dict=True).items()}
    stats = []
    for _ in range(n):
        pick = [grouped[rng.choice(ids)] for _ in ids]
        stats.append(fn(pl.concat(pick)))
    stats.sort()
    return stats[int(alpha / 2 * n)], stats[int((1 - alpha / 2) * n) - 1]


def calibration(pairs: pl.DataFrame, bins: int = 10) -> tuple[float, pl.DataFrame]:
    """ECE по предсказаниям, у которых есть эталон."""
    d = pairs.filter(pl.col("pred").is_not_null() & pl.col("gold").is_not_null())
    if d.height == 0:
        return float("nan"), d
    d = d.with_columns((pl.col("confidence") * bins).floor().clip(0, bins - 1).alias("bin"))
    table = d.group_by("bin").agg(pl.len().alias("n"), pl.col("confidence").mean().alias("conf"),
                                  pl.col("correct").mean().alias("acc")).sort("bin")
    ece = float(((table["conf"] - table["acc"]).abs() * table["n"]).sum() / d.height)
    return ece, table


def threshold_sweep(pairs: pl.DataFrame, thresholds: Iterable[float] = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)) -> pl.DataFrame:
    rows = []
    for t in thresholds:
        s = summarize(pairs.with_columns(pl.when(pl.col("confidence") >= t).then(pl.col("pred")).alias("pred")))
        rows.append({"threshold": t, **{k: s[k][0] for k in ("precision", "recall", "coverage", "f1")}})
    return pl.DataFrame(rows)


def headline(pairs: pl.DataFrame, categories: pl.DataFrame) -> dict[str, float]:
    s = summarize(pairs).row(0, named=True)
    ece, _ = calibration(pairs)
    return {**{k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()},
            "traffic_weighted_f1": round(traffic_weighted_f1(pairs, categories), 4), "ece": round(ece, 4)}


__all__ = ["eval_sample", "supported_flags", "score_pairs", "summarize", "traffic_weighted_f1", "bootstrap_ci",
           "calibration", "threshold_sweep", "headline", "BRAND"]
