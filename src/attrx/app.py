"""Демо для модератора: карточка → характеристики со статусом, вероятностью и подтверждением.

Запуск: python -m attrx serve --port 8000  (нужны артефакты python -m attrx train)
API:    POST /api/extract {title, description, category_id}  ·  GET /api/example  ·  GET /api/categories  ·  GET /health

Модели (~1 ГБ) загружаются один раз в фоне при старте; /health отвечает сразу и сообщает готовность.
Запросы до окончания загрузки ждут её, а не грузят модели повторно.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from attrx.catalog import DataPaths, load
from attrx.configs import MAIN
from attrx.train import assign_status, build_stack, load_artifact
from attrx.values import card_text

STATIC = Path(__file__).with_name("static")
MAX_TITLE, MAX_DESCRIPTION = 500, 20000
log = logging.getLogger("attrx.app")


class CardIn(BaseModel):
    title: str = Field(min_length=3, max_length=MAX_TITLE)
    description: str = Field(default="", max_length=MAX_DESCRIPTION)
    category_id: str = Field(min_length=1, max_length=64)


class Item(BaseModel):
    attribute: str
    value: str
    unit: str | None
    confidence: float
    status: str
    source: str
    evidence_field: str | None
    evidence_text: str | None


class CardOut(BaseModel):
    category: str | None
    ms: float
    accept_threshold: float
    title: str
    description: str
    items: list[Item]
    missing: list[str]
    schema_size: int


class State:
    def __init__(self, paths: DataPaths):
        comp, verifiers = load_artifact(paths, "components"), load_artifact(paths, "verifiers")
        if comp is None or verifiers is None:
            raise RuntimeError(f"Нет артефактов в {paths.work / 'artifacts'}: сначала python -m attrx train")
        self.verifier, self.threshold = verifiers[MAIN]
        self.stack = build_stack(comp, MAIN, verifiers, paths)
        self.slots = comp.lexicon.slots
        cats = load(paths, "categories")
        self.names = {c: n for c, n in cats.select("id", "name").iter_rows()}
        self.traffic = {c: t for c, t in cats.select("id", "traffic_share").iter_rows()}
        cards = load(paths, "cards")
        cards = cards.filter(cards["split"] == "test")  # примеры только из отложенного теста
        top = sorted(self.slots, key=lambda c: -self.traffic.get(c, 0))[:40]
        self.examples = cards.filter(cards["category_id"].is_in(top)).select("sku_id", "category_id", "title", "description")
        self.gold = load(paths, "gold").join(self.examples.select("sku_id"), on="sku_id", how="semi")
        self.categories = [{"id": c, "name": self.names.get(c, c), "slots": len(self.slots[c])} for c in top]


class Loader:
    """Загружает State один раз (под блокировкой); ошибка загрузки запоминается и отдаётся как 503."""

    def __init__(self, paths: DataPaths | None):
        self.paths, self.state, self.error = paths, None, None
        self.lock = threading.Lock()

    def get(self) -> State:
        if self.state is None:
            with self.lock:
                if self.state is None and self.error is None:
                    t = time.perf_counter()
                    try:
                        self.state = State(self.paths or DataPaths.from_env())
                        self.state.stack.predict_frame(_frame(CardIn(title="прогрев модели", category_id=self.state.categories[0]["id"])))
                        log.info("модели загружены за %.1f с", time.perf_counter() - t)
                    except Exception as e:  # отдаём причину в /health и 503, а не падаем на каждом запросе
                        self.error = f"{type(e).__name__}: {e}"
                        log.exception("не удалось загрузить модели")
        if self.state is None:
            raise HTTPException(503, f"модели не загружены: {self.error}")
        return self.state


def create_app(paths: DataPaths | None = None) -> FastAPI:
    loader = Loader(paths)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        threading.Thread(target=lambda: _safe(loader.get), daemon=True).start()  # прогрев до первого пользователя
        yield

    app = FastAPI(title="attrx — заполнение характеристик", version="1.0", lifespan=lifespan)

    @app.exception_handler(Exception)
    async def internal_error(_request, exc: Exception):
        log.exception("ошибка запроса")
        return JSONResponse({"detail": "внутренняя ошибка"}, status_code=500)

    @app.get("/health")
    def health():
        ready = loader.state is not None
        return {"status": "ok" if loader.error is None else "error", "ready": ready, "error": loader.error}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/categories")
    def categories():
        return loader.get().categories

    @app.get("/api/example")
    def example(category_id: str | None = None):
        s = loader.get()
        df = s.examples if not category_id else s.examples.filter(s.examples["category_id"] == category_id)
        if df.height == 0:
            raise HTTPException(404, "нет примеров для категории")
        row = df.row(random.randrange(df.height), named=True)
        gold = {a: v for a, v in s.gold.filter(s.gold["sku_id"] == row["sku_id"]).select("attribute", "value").iter_rows()}
        # текст карточки без HTML-разметки и в пределах ограничений формы
        title = card_text(row["title"])[:MAX_TITLE]
        description = card_text(row["description"]).replace(" ¶ ", "\n")[:MAX_DESCRIPTION]
        return row | {"title": title, "description": description, "gold": gold}

    @app.post("/api/extract", response_model=CardOut)
    def extract(card: CardIn):
        s = loader.get()
        if card.category_id not in s.slots:
            raise HTTPException(404, f"категория {card.category_id} не поддерживается")
        t = time.perf_counter()
        frame = assign_status(s.stack.predict_frame(_frame(card)), s.threshold,  # та же политика, что в CLI
                              judge_accept=s.verifier.judge_threshold)
        ms = 1000 * (time.perf_counter() - t)
        items = [{"attribute": r["attribute"], "value": r["value"], "unit": r["unit"], "confidence": round(r["confidence"], 3),
                  "status": r["status"], "source": r["source"], "evidence_field": r["evidence_field"],
                  "evidence_text": r["evidence_text"]} for r in frame.iter_rows(named=True)]
        order = {"accept": 0, "review": 1, "reject": 2}
        items.sort(key=lambda x: (order[x["status"]], -x["confidence"]))
        shown = {i["attribute"] for i in items if i["status"] != "reject"}
        missing = [a for a in s.slots[card.category_id] if a not in shown]
        return {"category": s.names.get(card.category_id), "ms": round(ms, 1), "accept_threshold": round(s.threshold, 3),
                "title": card_text(card.title), "description": card_text(card.description), "items": items,
                "missing": missing, "schema_size": len(s.slots[card.category_id])}

    return app


def _safe(fn) -> None:
    try:
        fn()
    except HTTPException:
        pass  # причина уже в loader.error и в /health


def _frame(card: CardIn):
    import polars as pl

    return pl.DataFrame({"sku_id": ["demo"], "category_id": [card.category_id], "title": [card.title],
                         "description": [card.description]})


app = create_app()
