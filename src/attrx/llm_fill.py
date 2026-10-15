"""LLM для пропусков на реальном каталоге (каскад: LLM последняя и самая дорогая ступень).

- Спрашиваем только слоты категории без уверенного кандидата от дешёвых источников.
- Спрашиваем только «текстовые» характеристики: те, что на val хотя бы иногда встречаются в тексте
  (доля подтверждаемости ≥ порога). Страну или гарантию у LLM не спрашиваем: в тексте их нет, будет галлюцинация.
- Ответ обязан содержать дословную цитату. Цитата ищется в карточке, значение — в цитате.
  Не подтвердилось — значение отбрасывается и учитывается в статистике.
- Маршрутизация: карточку отправляем в LLM, только если ожидаемое число находимых полей
  (сумма долей подтверждаемости непокрытых слотов) не ниже `min_expected`. Так доля вызовов LLM
  остаётся в бюджете, а вызовы тратятся на карточки с длинным описанием и многими пропусками.
- Кэш по хэшу (модель + промпт) и бюджет вызовов — дневной лимит ключа хакатона.

Защита от галлюцинаций — каждое значение LLM проходит все проверки, иначе отбрасывается и считается в FillStats:
1. Поле должно быть из списка, о котором спрашивали: добавить новое поле LLM не может (unknown_attr).
2. Значение должно подходить полю: «Да/Нет» для флагов, число для числовых, значение из справочника категории
   для перечислений (rejected_value).
3. Цитата должна найтись в карточке, значение — в цитате. Для «Да/Нет» цитата должна говорить о самой
   характеристике (слово из её названия), а «Нет» — содержать отрицание отдельным словом (rejected_ungrounded).
4. LLM одна значение не публикует: без согласия дешёвых источников оно уходит модератору (train.ACCEPT_SOURCES).

Клиент — любой OpenAI-совместимый: шлюз хакатона, vLLM на ВМ или attrx.local_llm.LocalChat на своей GPU.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from attrx.extractors.base import Candidate, Evidence
from attrx.lexicon import Lexicon, Slot
from attrx.values import canon, card_text, key, number

log = logging.getLogger(__name__)

MAX_DESCRIPTION_CHARS = 2500
MAX_SLOTS = 25
EXAMPLES = 4
NEGATION = re.compile(r"(?:^|\s)(?:нет|без|не|отсутств\w*)(?=\s|$)")
OBJECT = re.compile(r'"([^"\n]{1,120})"\s*:\s*(\{[^{}]*\})')  # «характеристика»: {...} — разбор битого JSON по частям
TRANSIENT = frozenset({"APIConnectionError", "APITimeoutError", "RateLimitError", "InternalServerError"})
RETRIES = 3

SYSTEM_PROMPT = """Ты — модератор каталога интернет-магазина электроники и товаров для дома.
Извлеки характеристики товара ТОЛЬКО из текста карточки.
Правила:
1. Используй только то, что прямо написано в названии или описании. Не используй свои знания о модели.
2. Для каждой найденной характеристики верни значение и короткую дословную цитату из карточки (3–8 слов), где оно указано.
3. Если характеристики в тексте нет — не включай её в ответ.
4. Для числовых характеристик value — только число в указанной единице. Для да/нет — "Да" или "Нет".
5. Если даны примеры значений, пиши значение в том же формате.
Ответ — только JSON-объект {"<характеристика>": {"value": "...", "quote": "..."}} без пояснений."""


@dataclass
class FillStats:
    cards: int = 0                # карточек со слотами для вопроса
    skipped: int = 0              # маршрутизация: карточка не стоит вызова LLM
    budget_skipped: int = 0       # бюджет вызовов исчерпан — карточка осталась без LLM
    calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0         # без клиента (повтор по кэшу): ответа нет — карточка без LLM
    retries: int = 0              # повторы после временных ошибок шлюза
    prompt_tokens: int = 0
    completion_tokens: int = 0
    truncated: int = 0            # ответ обрезан по max_tokens (в кэш не кладём)
    proposed: int = 0             # значений в ответах по запрошенным полям
    accepted: int = 0             # прошли все проверки
    rejected_ungrounded: int = 0  # цитаты нет в карточке или значение из неё не следует
    rejected_value: int = 0       # значение не подходит полю: не «Да/Нет», не число, вне справочника категории
    unknown_attr: int = 0         # попытка добавить поле, о котором не спрашивали
    bad_item: int = 0             # элемент ответа без значения или не объект
    no_json: int = 0              # в ответе нет JSON
    salvaged: int = 0             # битый JSON разобран по частям
    parse_errors: int = 0         # ответ не разобран совсем

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        rejected = self.rejected_ungrounded + self.rejected_value
        d["rejected_share"] = round(rejected / self.proposed, 4) if self.proposed else 0.0
        return d


@dataclass
class LLMFiller:
    lexicon: Lexicon
    client: object
    model: str
    # категория -> характеристики, которые имеет смысл спрашивать. Словарь {характеристика: доля подтверждаемости
    # на val} включает маршрутизацию по min_expected; множество — спрашивать всегда.
    askable: Mapping[str, Mapping[str, float] | set[str]]
    category_names: dict[str, str] = field(default_factory=dict)
    cache_dir: Path = Path(".cache/llm_fill")
    max_calls: int | None = None
    min_expected: float = 0.0
    confidence: dict[str, float] = field(default_factory=lambda: {"bool": 0.7, "num": 0.7, "enum": 0.7, "text": 0.7})
    stats: FillStats = field(default_factory=FillStats)
    _vocab: dict[tuple[str, str], set[str]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- промпт
    def slots_to_ask(self, category: str | None, covered: set[str]) -> list[Slot]:
        slots = self.lexicon.slots.get(category or "", {})
        ask = self.askable.get(category or "", set())
        return [s for a, s in slots.items() if a in ask and a not in covered][:MAX_SLOTS]

    def expected_fields(self, category: str | None, slots: list[Slot]) -> float:
        """Ожидаемое число полей, которые LLM может найти в тексте: сумма долей подтверждаемости слотов."""
        ask = self.askable.get(category or "", set())
        return sum(ask.get(s.attribute, 0.0) for s in slots) if isinstance(ask, Mapping) else float(len(slots))

    def route(self, category: str | None, covered: set[str]) -> list[Slot]:
        """Слоты для вопроса или пустой список, если карточка не стоит вызова LLM."""
        slots = self.slots_to_ask(category, covered)
        if slots and self.expected_fields(category, slots) < self.min_expected:
            self.stats.skipped += 1
            return []
        return slots

    def prompt(self, title: str, description: str, category: str | None, slots: list[Slot]) -> list[dict]:
        vocab = self.lexicon.values.get(category or "", {})
        examples: dict[str, list[str]] = {}
        for entries in vocab.values():
            for attribute, display, _ in entries:
                if len(examples.setdefault(attribute, [])) < EXAMPLES and display not in examples[attribute]:
                    examples[attribute].append(display)
        lines = []
        for s in slots:
            spec = {"bool": "да/нет", "num": f"число{', ' + s.unit if s.unit else ''}", "enum": "значение из списка",
                    "text": "текст"}[s.kind]
            ex = examples.get(s.attribute)
            lines.append(f"- {s.attribute} — {spec}" + (f"; примеры: {', '.join(ex)}" if ex and s.kind != "num" else ""))
        user = (f"Категория: {self.category_names.get(category or '', category)}\n"
                f"Характеристики:\n" + "\n".join(lines) +
                f"\n\nНазвание: {card_text(title)}\nОписание: {card_text(description)[:MAX_DESCRIPTION_CHARS]}")
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]

    # ---------------------------------------------------------------- вызов с кэшем
    def ask(self, messages: list[dict]) -> str | None:
        path = self._cache_path(messages)
        cached = _read_cache(path)
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        if self.client is None:  # режим повтора по кэшу (ноутбуки, офлайн-проверка): новых вызовов нет
            self.stats.cache_misses += 1
            return None
        if self.max_calls is not None and self.stats.calls >= self.max_calls:
            if not self.stats.budget_skipped:
                log.warning("бюджет LLM исчерпан (%s вызовов): остальные карточки идут без LLM", self.max_calls)
            self.stats.budget_skipped += 1
            return None
        resp = self._create(messages)
        self.stats.calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.stats.prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
            self.stats.completion_tokens += getattr(usage, "completion_tokens", 0) or 0
        choice = resp.choices[0]
        answer = choice.message.content or ""
        if getattr(choice, "finish_reason", None) == "length":
            self.stats.truncated += 1  # обрезанный ответ разбираем, но не кэшируем: повторный запуск спросит снова
        elif answer.strip():
            _write_cache(path, answer)
        return answer

    def _create(self, messages: list[dict]):
        """Вызов шлюза. Повторяем только временные ошибки (сеть, лимит, 5xx); ключ или запрос с ошибкой — сразу наружу."""
        for attempt in range(RETRIES):
            try:
                return self.client.chat.completions.create(model=self.model, messages=messages, temperature=0,
                                                           max_tokens=700)
            except Exception as e:
                if type(e).__name__ not in TRANSIENT or attempt == RETRIES - 1:
                    raise
                self.stats.retries += 1
                log.warning("LLM: %s, повтор %d из %d", type(e).__name__, attempt + 1, RETRIES - 1)
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")

    def store(self, messages: list[dict], answer: str) -> None:
        """Для пакетной генерации: ответ получен снаружи (LocalChat.batch), кладём в тот же кэш."""
        if answer.strip():
            _write_cache(self._cache_path(messages), answer)

    def _cache_path(self, messages: list[dict]) -> Path:
        digest = hashlib.sha256(json.dumps([self.model, messages], ensure_ascii=False).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{digest}.json"

    # ---------------------------------------------------------------- разбор и проверка
    def parse(self, answer: str | None, title: str, description: str, slots: list[Slot],
              category: str | None = None) -> list[Candidate]:
        data = self._load(answer)
        if not data:
            return []
        by_name = {s.attribute: s for s in slots}
        texts = {"title": card_text(title), "description": card_text(description)}
        out = []
        for attribute, item in data.items():
            slot = by_name.get(attribute)
            if slot is None:
                self.stats.unknown_attr += 1  # поле, о котором не спрашивали, не добавляем
                continue
            if not isinstance(item, dict) or item.get("value") in (None, ""):
                self.stats.bad_item += 1
                continue
            self.stats.proposed += 1
            value, quote = str(item["value"]).strip(), str(item.get("quote") or "").strip()
            if not self._valid_value(value, slot, category):
                self.stats.rejected_value += 1
                continue
            ev = _ground(texts, quote, value, slot)
            if ev is None:
                self.stats.rejected_ungrounded += 1
                continue
            self.stats.accepted += 1
            out.append(Candidate(attribute, value, "|".join(sorted(canon(value))), self.confidence.get(slot.kind, 0.7), "llm", ev))
        return out

    def _load(self, answer: str | None) -> dict | None:
        """JSON из ответа. Битый JSON (пропущенные запятые и скобки — частая ошибка малых моделей) разбираем по
        объектам «характеристика: {...}», чтобы не терять весь ответ из-за одной ошибки."""
        if not answer:
            return None
        m = re.search(r"\{.*\}", answer, re.S)
        if not m:
            self.stats.no_json += 1
            return None
        try:
            data = json.loads(m.group())
            if isinstance(data, dict):
                return data
        except json.JSONDecodeError:
            pass
        data = {}
        for name, body in OBJECT.findall(m.group()):
            try:
                data[name] = json.loads(body)
            except json.JSONDecodeError:
                continue
        if data:
            self.stats.salvaged += 1
            return data
        self.stats.parse_errors += 1
        return None

    def _valid_value(self, value: str, slot: Slot, category: str | None) -> bool:
        """Значение подходит полю по типу; для перечислений — есть в справочнике категории (если он известен)."""
        if slot.kind == "bool":
            return key(value) in ("да", "нет")
        if slot.kind == "num":
            return number(value) is not None
        if slot.kind == "enum" and category is not None:
            vocab = self._vocabulary(category, slot.attribute)
            parts = {p for p in canon(value) if p}
            return not vocab or bool(parts) and parts <= vocab
        return True

    def _vocabulary(self, category: str, attribute: str) -> set[str]:
        k = (category, attribute)
        if k not in self._vocab:
            parts: set[str] = set()
            for entries in self.lexicon.values.get(category, {}).values():
                for a, display, _ in entries:
                    if a == attribute:
                        parts |= {p for p in canon(display) if p}
            self._vocab[k] = parts
        return self._vocab[k]

    def candidates_for(self, title: str, description: str, category: str | None, covered: set[str]) -> list[Candidate]:
        slots = self.route(category, covered)
        if not slots:
            return []
        self.stats.cards += 1
        messages = self.prompt(title, description, category, slots)
        return self.parse(self.ask(messages), title, description, slots, category)


def _ground(texts: dict[str, str], quote: str, value: str, slot: Slot) -> Evidence | None:
    """Цитата есть в карточке (по нормализованным токенам), значение следует из цитаты."""
    qk = key(quote)
    if not qk:
        return None
    for fld, text in texts.items():
        tk = key(text)
        pos = (" " + tk + " ").find(" " + qk + " ")
        if pos < 0:
            continue
        if not _value_in_quote(value, quote, slot):
            return None
        start = text.lower().find(quote.lower())
        if start < 0:  # цитата совпала только после нормализации: точного фрагмента для подсветки нет
            return Evidence(fld, 0, 0, quote)
        return Evidence(fld, start, start + len(quote), text[start:start + len(quote)])
    return None


def _value_in_quote(value: str, quote: str, slot: Slot) -> bool:
    qk = key(quote)
    q = " " + qk + " "
    if slot.kind == "bool":
        # «Да» и «Нет» подтверждает только цитата о самой характеристике: в ней есть слово из её названия.
        # «Нет» требует отрицания отдельным словом («без», «нет», «не»), а не подстроки («интернет», «безопасный»).
        if not any(t.startswith(s) for t in qk.split() for s in slot.stems):
            return False
        negated = bool(NEGATION.search(qk))
        v = key(value)
        return (v == "да" and not negated) or (v == "нет" and negated)
    parts = [p for p in canon(value) if p]
    if not parts:
        return False
    for p in parts:
        if ":" in p:  # число с измерением: ищем само число в цитате
            num = p.split(":", 1)[1]
            if f" {num} " not in q and f" {num}.0 " not in q:
                return False
        elif f" {p} " not in q:
            return False
    return True


def _read_cache(path: Path) -> str | None:
    """Ответ из кэша или None. Повреждённая запись (прерванная запись, чужой формат) считается промахом."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))["answer"]
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _write_cache(path: Path, answer: str) -> None:
    """Атомарная запись: временный файл и переименование — прерванный запуск не оставит битую запись."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"answer": answer}, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)
