"""Потоковая конвертация skus.pkl (YML-каталог, сконвертированный в dict) в parquet.

Почему не pickle.load:
  * безопасность — pickle может исполнить код при загрузке. Здесь мы сами интерпретируем
    ограниченный набор инструкций (словари, списки, строки, числа); любая инструкция импорта
    (GLOBAL, REDUCE и т.п.) немедленно прерывает разбор;
  * память — файл 1.5 ГБ разворачивается в 6+ ГБ объектов Python. Конвертер держит в памяти
    только текущую пачку товаров и словарь повторно используемых строк.

Два прохода: 1) какие ячейки memo реально переиспользуются (битовая карта);
2) разбор с сохранением только этих ячеек и выгрузкой товаров пачками.

Выход:
  <out>/categories.parquet          id, parent_id, name
  <out>/offers/part-*.parquet       offer_id, available, vendor, model, name, category_id, extra (JSON прочих полей)
  <out>/params/part-*.parquet       offer_id, name, unit, value

Запуск: python scripts/convert_skus.py <skus.pkl> <out_dir>
"""

from __future__ import annotations

import json
import pickletools
import sys
import time
from pathlib import Path

import polars as pl

BATCH = 50_000
STRING_OPS = {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "SHORT_BINSTRING", "BINSTRING"}
SCALAR_OPS = STRING_OPS | {"BININT", "BININT1", "BININT2", "LONG1", "BINFLOAT", "INT", "LONG"}
GET_OPS = {"BINGET", "LONG_BINGET", "GET"}
PUT_OPS = {"BINPUT", "LONG_BINPUT", "PUT"}


class OfferSink:
    """Подменяет список offer: вместо накопления отдаёт товары на запись."""

    def __init__(self, writer: Writer):
        self.writer = writer

    def extend(self, items):
        for item in items:
            self.writer.add(item)

    def append(self, item):
        self.writer.add(item)


class Writer:
    def __init__(self, out: Path):
        self.out = out
        (out / "offers").mkdir(parents=True, exist_ok=True)
        (out / "params").mkdir(parents=True, exist_ok=True)
        self.offers: list[dict] = []
        self.params: list[dict] = []
        self.part = 0
        self.total = 0

    def add(self, offer):
        if not isinstance(offer, dict):
            return
        offer_id = str(offer.get("@id", ""))
        extra = {}
        for k, v in offer.items():
            if k in ("@id", "@available", "vendor", "model", "name", "categoryId", "categories", "param"):
                continue
            extra[k] = v
        # в выгрузке категория вложена: {"categories": {"categoryId": "6128"}}; бывает и список
        category = offer.get("categoryId") or (offer.get("categories") or {}).get("categoryId")
        if isinstance(category, list):
            category = category[0] if category else None
        params = offer.get("param") or []
        if isinstance(params, dict):
            params = [params]
        for p in params:
            if isinstance(p, dict):
                self.params.append({"offer_id": offer_id, "name": _s(p.get("@name")),
                                    "unit": _s(p.get("@unit")), "value": _s(p.get("#text"))})
        self.offers.append({
            "offer_id": offer_id, "available": _s(offer.get("@available")), "vendor": _s(offer.get("vendor")),
            "model": _s(offer.get("model")), "name": _s(offer.get("name")),
            "category_id": _s(_text(category)),
            "extra": json.dumps(extra, ensure_ascii=False) if extra else None,
        })
        if len(self.offers) >= BATCH:
            self.flush()

    def flush(self):
        if not self.offers:
            return
        pl.DataFrame(self.offers, schema={c: pl.String for c in self.offers[0]}).write_parquet(
            self.out / "offers" / f"part-{self.part:04d}.parquet")
        if self.params:
            pl.DataFrame(self.params, schema={c: pl.String for c in self.params[0]}).write_parquet(
                self.out / "params" / f"part-{self.part:04d}.parquet")
        self.total += len(self.offers)
        self.part += 1
        self.offers, self.params = [], []
        print(f"  записано товаров: {self.total:,}", flush=True)


def _text(v):
    return v.get("#text") if isinstance(v, dict) else v


def _s(v):
    return None if v is None else str(v)


def referenced_memo(path: Path) -> bytearray:
    """Проход 1: битовая карта индексов memo, на которые есть ссылки (BINGET)."""
    bits = bytearray(1 << 20)
    with path.open("rb") as f:
        for op, arg, _ in pickletools.genops(f):
            if op.name in GET_OPS:
                idx = int(arg)
                while idx // 8 >= len(bits):
                    bits.extend(bytearray(len(bits)))
                bits[idx // 8] |= 1 << (idx % 8)
    return bits


def convert(path: Path, out: Path) -> None:
    t0 = time.time()
    print("проход 1: ссылки memo…", flush=True)
    bits = referenced_memo(path)
    print(f"  готово за {time.time() - t0:.0f} c", flush=True)

    def is_ref(idx: int) -> bool:
        return idx // 8 < len(bits) and bits[idx // 8] >> (idx % 8) & 1

    writer = Writer(out)
    stack: list = []
    marks: list[int] = []  # позиции MARK в стеке (без копирования стека при поиске)
    memo: dict[int, object] = {}
    memo_next = 0
    categories = None
    print("проход 2: разбор и выгрузка…", flush=True)
    with path.open("rb") as f:
        for op, arg, _ in pickletools.genops(f):
            name = op.name
            if name in SCALAR_OPS:
                stack.append(arg)
            elif name == "MEMOIZE":
                if is_ref(memo_next):
                    memo[memo_next] = stack[-1]
                memo_next += 1
            elif name in PUT_OPS:
                if is_ref(int(arg)):
                    memo[int(arg)] = stack[-1]
            elif name in GET_OPS:
                stack.append(memo[int(arg)])
            elif name == "EMPTY_DICT":
                stack.append({})
            elif name == "EMPTY_LIST":
                stack.append(OfferSink(writer) if stack and stack[-1] == "offer" else [])
            elif name == "EMPTY_TUPLE":
                stack.append(())
            elif name == "MARK":
                marks.append(len(stack))
            elif name == "SETITEM":
                v = stack.pop()
                k = stack.pop()
                stack[-1][k] = v
                if k == "category" and isinstance(v, list):
                    categories = v
            elif name == "SETITEMS":
                i = marks.pop()
                items = stack[i:]
                del stack[i:]
                target = stack[-1]
                for j in range(0, len(items), 2):
                    target[items[j]] = items[j + 1]
                    if items[j] == "category" and isinstance(items[j + 1], list):
                        categories = items[j + 1]
            elif name == "APPEND":
                v = stack.pop()
                stack[-1].append(v)
            elif name == "APPENDS":
                i = marks.pop()
                items = stack[i:]
                del stack[i:]
                stack[-1].extend(items)
            elif name in ("TUPLE", "TUPLE1", "TUPLE2", "TUPLE3"):
                if name == "TUPLE":
                    i = marks.pop()
                    items = tuple(stack[i:])
                    del stack[i:]
                else:
                    n = int(name[-1])
                    items = tuple(stack[-n:])
                    del stack[-n:]
                stack.append(items)
            elif name == "NONE":
                stack.append(None)
            elif name in ("NEWTRUE", "NEWFALSE"):
                stack.append(name == "NEWTRUE")
            elif name in ("PROTO", "FRAME"):
                continue
            elif name == "STOP":
                break
            else:
                raise RuntimeError(f"Неожиданная инструкция pickle: {name} — разбор остановлен из соображений безопасности")
    writer.flush()
    if categories:
        rows = [{"id": _s(c.get("@id")), "parent_id": _s(c.get("@parentId")), "name": _s(c.get("#text"))}
                for c in categories if isinstance(c, dict)]
        pl.DataFrame(rows, schema={"id": pl.String, "parent_id": pl.String, "name": pl.String}).write_parquet(
            out / "categories.parquet")
        print(f"категорий: {len(rows):,}")
    print(f"готово: {writer.total:,} товаров за {time.time() - t0:.0f} c")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
