"""Сборка размеченного набора из выгрузки М.Видео (контракт: docs/DATA_CONTRACT.md).

Вход (ATTRX_DATA_DIR):
  cu_ws/sku_desc.parquet, cu_ws/query_clicks.parquet       — выданные файлы
  processed/offers, processed/params, processed/categories — результат scripts/convert_skus.py

Выход (ATTRX_DATA_DIR/work):
  cards.parquet      sku_id, category_id, seller, group, split, title, description — размеченные карточки
  gold.parquet       sku_id, attribute, value, unit                                   — эталон без служебных полей
  reference.parquet  category_id, attribute, value, unit, count                       — справочник значений train-каталога
  schema.parquet     category_id, attribute, kind, unit, fill, n_cards                — схема категории по train
  categories.parquet id, parent_id, name, clicks, traffic_share                       — веса категорий по кликам
  analog_offers.parquet, analog_facts.parquet                                          — индекс аналогов (train-товары каталога)

Разбиение групповое и детерминированное: группа 3P-карточки — продавец, 1P («М.Видео») — бренд.
Справочник и схема строятся только по train-группам всего каталога (в том числе по товарам без описания).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import polars as pl

BRAND = "Бренд"
FIRST_PARTY = "М.Видео"
SELLER_PARAM = "Продавец"
# Не описывают товар и не выводятся из текста карточки.
SERVICE_ATTRIBUTES = frozenset({
    "Гарантия предоставляется", "Вид выпуска товара", "Признак 18+", "Потратить бонусы", SELLER_PARAM,
    "Подборки", "Состояние", "Внешний вид",
})
SPLIT_BOUNDS = (70, 80)  # md5(group) % 100: [0,70) train, [70,80) val, [80,100) test
MIN_CARDS = 20
MIN_FILL = 0.5
BOOL_VALUES = ("да", "нет")
NUMBER_RE = r"^-?\d+([.,]\d+)?$"


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @classmethod
    def from_env(cls, default: str | Path = "data/real") -> DataPaths:
        return cls(Path(os.environ.get("ATTRX_DATA_DIR", default)))

    @property
    def raw(self) -> Path:
        return self.root / "cu_ws"

    @property
    def processed(self) -> Path:
        return self.root / "processed"

    @property
    def work(self) -> Path:
        return self.root / "work"


def split_of(group: str) -> str:
    """Детерминированное назначение группы в выборку (не зависит от версии библиотек и порядка строк)."""
    bucket = int(hashlib.md5(group.encode("utf-8")).hexdigest(), 16) % 100
    if bucket < SPLIT_BOUNDS[0]:
        return "train"
    return "val" if bucket < SPLIT_BOUNDS[1] else "test"


def group_expr() -> pl.Expr:
    vendor = pl.col("vendor").fill_null("").str.to_lowercase().str.strip_chars()
    return (pl.when(pl.col("seller") == FIRST_PARTY).then(pl.lit("1p:") + vendor)
            .when(pl.col("seller").is_null()).then(pl.lit("na:") + vendor)
            .otherwise(pl.lit("3p:") + pl.col("seller"))).alias("group")


def kind_expr() -> list[pl.Expr]:
    """Тип характеристики по её значениям: bool / num / enum / text (см. контракт)."""
    value = pl.col("value").str.to_lowercase().str.strip_chars()
    weight = pl.col("count")
    return [
        ((value.is_in(BOOL_VALUES).cast(pl.Float64) * weight).sum() / weight.sum()).alias("p_bool"),
        ((value.str.contains(NUMBER_RE).cast(pl.Float64) * weight).sum() / weight.sum()).alias("p_num"),
        pl.col("value").n_unique().alias("nuniq"),
    ]


def _offers(paths: DataPaths, split_by: str = "group") -> pl.DataFrame:
    """split_by='group' — по продавцу/бренду (основной режим), 'random' — по sku (только для замера утечки)."""
    sellers = (pl.scan_parquet(paths.processed / "params" / "*.parquet")
               .filter(pl.col("name") == SELLER_PARAM)
               .group_by("offer_id").agg(pl.col("value").min().alias("seller")))  # детерминированно при двух продавцах
    offers = (pl.scan_parquet(paths.processed / "offers" / "*.parquet")
              .select("offer_id", "vendor", "category_id", "name")
              .join(sellers, on="offer_id", how="left")
              .with_columns(group_expr())
              .collect())
    unit = "group" if split_by == "group" else "offer_id"
    units = offers.select(unit).unique()
    units = units.with_columns(pl.col(unit).map_elements(split_of, return_dtype=pl.String).alias("split"))
    return offers.join(units, on=unit, how="left")


def _search_params(paths: DataPaths) -> pl.LazyFrame:
    return (pl.scan_parquet(paths.processed / "params" / "*.parquet")
            .filter(pl.col("name").str.ends_with("_search"))
            .with_columns(pl.col("name").str.replace(r"_search$", "").alias("attribute"))
            .filter(~pl.col("attribute").is_in(list(SERVICE_ATTRIBUTES)) & (pl.col("value").str.strip_chars() != ""))
            .select("offer_id", "attribute", "value", "unit"))


def build(paths: DataPaths, min_cards: int = MIN_CARDS, min_fill: float = MIN_FILL, split_by: str = "group",
          out: Path | None = None) -> dict[str, int]:
    """Строит все файлы рабочего набора. Возвращает размеры для журнала."""
    out = out or paths.work
    out.mkdir(parents=True, exist_ok=True)
    offers = _offers(paths, split_by)

    desc = (pl.scan_parquet(paths.raw / "sku_desc.parquet")
            .select(pl.col("sku_id").cast(pl.String).alias("offer_id"), "title", "description"))
    cards = (offers.lazy().join(desc, on="offer_id", how="inner")
             .select(pl.col("offer_id").alias("sku_id"), "category_id", "seller", "group", "split", "title",
                     pl.col("description").fill_null(""))
             .sort("sku_id")
             .collect())
    cards.write_parquet(out / "cards.parquet")

    params = _search_params(paths)
    brand_rows = offers.lazy().filter(pl.col("vendor").is_not_null()).select(
        "offer_id", pl.lit(BRAND).alias("attribute"), pl.col("vendor").alias("value"), pl.lit(None, pl.String).alias("unit"))
    facts = pl.concat([params, brand_rows])

    labeled = cards.lazy().select(pl.col("sku_id").alias("offer_id"))
    gold = facts.join(labeled, on="offer_id", how="semi").rename({"offer_id": "sku_id"}).sort("sku_id", "attribute").collect()
    gold.write_parquet(out / "gold.parquet")

    train = offers.lazy().filter(pl.col("split") == "train").select("offer_id", "category_id")
    train_facts = facts.join(train, on="offer_id", how="inner")
    reference = (train_facts.group_by("category_id", "attribute", "value")
                 .agg(pl.len().alias("count"), pl.col("unit").drop_nulls().mode().first().alias("unit"))
                 .collect())
    reference.write_parquet(out / "reference.parquet")

    kinds = reference.group_by("attribute").agg(*kind_expr()).with_columns(
        pl.when(pl.col("p_bool") > 0.9).then(pl.lit("bool"))
        .when(pl.col("p_num") > 0.9).then(pl.lit("num"))
        .when(pl.col("nuniq") <= 300).then(pl.lit("enum"))
        .otherwise(pl.lit("text")).alias("kind")).select("attribute", "kind")

    n_cards = train.group_by("category_id").agg(pl.len().alias("n_cards")).collect()
    schema = (train_facts.group_by("category_id", "attribute")
              .agg(pl.col("offer_id").n_unique().alias("n"), pl.col("unit").drop_nulls().mode().first().alias("unit"))
              .collect()
              .join(n_cards, on="category_id")
              .with_columns((pl.col("n") / pl.col("n_cards")).alias("fill"))
              .filter((pl.col("n_cards") >= min_cards) & ((pl.col("fill") >= min_fill) | (pl.col("attribute") == BRAND)))
              .join(kinds, on="attribute", how="left")
              .select("category_id", "attribute", "kind", "unit", "fill", "n_cards")
              .sort("category_id", "fill", descending=[False, True]))
    schema.write_parquet(out / "schema.parquet")

    # Индекс аналогов: train-товары всего каталога (в том числе без описания) и их характеристики из схемы.
    analog_offers = (offers.lazy().filter(pl.col("split") == "train")
                     .join(schema.lazy().select("category_id").unique(), on="category_id", how="semi")
                     .select("offer_id", "category_id", "vendor", "name").collect())
    analog_offers.write_parquet(out / "analog_offers.parquet")
    (train_facts.join(schema.lazy().select("category_id", "attribute"), on=["category_id", "attribute"], how="semi")
     .select("offer_id", "attribute", "value", "unit").sort("offer_id").collect()
     .write_parquet(out / "analog_facts.parquet"))

    clicks = (pl.scan_parquet(paths.raw / "query_clicks.parquet")
              .select(pl.col("sku_seo_id").cast(pl.String).alias("id"))
              .group_by("id").agg(pl.len().alias("clicks")).collect())
    categories = (pl.read_parquet(paths.processed / "categories.parquet")
                  .join(clicks, on="id", how="left").fill_null(0)
                  .with_columns((pl.col("clicks") / pl.col("clicks").sum()).alias("traffic_share")))
    categories.write_parquet(out / "categories.parquet")

    return {"offers": offers.height, "analog_offers": analog_offers.height, "cards": cards.height, "gold": gold.height, "reference": reference.height,
            "schema_rows": schema.height, "schema_categories": schema["category_id"].n_unique()}


def load(paths: DataPaths, name: str, work: Path | None = None) -> pl.DataFrame:
    return pl.read_parquet((work or paths.work) / f"{name}.parquet")
