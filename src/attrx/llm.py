"""LLM-слой каскада: вызывается только для пропущенных атрибутов, с кэшем, бюджетом и проверкой источника.

Работает с любым OpenAI-совместимым шлюзом (на ВМ хакатона заданы OPENAI_API_KEY и OPENAI_BASE_URL).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from attrx.extractors.base import Candidate, Card
from attrx.grounding import locate
from attrx.normalize import canonicalize
from attrx.schema import Schema

log = logging.getLogger(__name__)

LLM_CONFIDENCE = 0.65
UNKNOWN_VALUE_PENALTY = 0.1
MAX_DESCRIPTION_CHARS = 3000
RETRIES = 3

SYSTEM_PROMPT = """Ты извлекаешь характеристики товара из карточки маркетплейса электроники.
Правила:
- Используй только то, что явно написано в карточке. Если характеристики нет — верни null. Не угадывай.
- Для каждой характеристики верни value и quote — дословный фрагмент карточки, из которого взято значение.
- Если указан список допустимых значений, выбирай из него; если в тексте значение вне списка — верни его как в тексте.
- Ответ — только JSON-объект вида {"<атрибут>": {"value": "...", "quote": "..."} | null}."""


@dataclass
class LLMStats:
    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    proposed: int = 0
    rejected_ungrounded: int = 0  # LLM предложила значение, которого нет в карточке
    errors: int = 0
    cached_tokens: int = 0  # токены ответов из кэша: в продакшене за них пришлось бы заплатить

    @property
    def tokens_equivalent(self) -> int:
        """Токены без учёта кэша — для честной оценки стоимости подхода."""
        return self.prompt_tokens + self.completion_tokens + self.cached_tokens

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class LLMExtractor:
    schema: Schema
    model: str
    cache_dir: Path = Path(".cache/llm")
    max_calls: int | None = None  # бюджет на запуск (дневной лимит ключа хакатона)
    client: object | None = None  # OpenAI-совместимый клиент; в тестах подменяется
    stats: LLMStats = field(default_factory=LLMStats)

    def __post_init__(self) -> None:
        if self.client is None:
            from openai import OpenAI  # необязательная зависимость: без неё работает офлайн-режим

            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.getenv("OPENAI_BASE_URL"))
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls, schema: Schema, model: str | None = None, **kwargs) -> LLMExtractor | None:
        model = model or os.getenv("ATTRX_LLM_MODEL")
        if not os.getenv("OPENAI_API_KEY") or not model:
            log.warning("LLM выключен: нужны OPENAI_API_KEY и модель (--llm-model или ATTRX_LLM_MODEL)")
            return None
        return cls(schema=schema, model=model, **kwargs)

    def extract(self, card: Card, attributes: list[str]) -> list[Candidate]:
        if not attributes:
            return []
        prompt = self._prompt(card, attributes)
        answer = self._complete(prompt)
        if answer is None:
            return []
        return self._to_candidates(card, attributes, answer)

    def _prompt(self, card: Card, attributes: list[str]) -> str:
        lines = ["Нужные характеристики:"]
        for name in attributes:
            spec = self.schema.attributes[name]
            line = f"- {name}: {spec.title}"
            if spec.type == "enum" and spec.dictionary:
                line += f"; допустимые значения: {', '.join(self.schema.dictionaries.get(spec.dictionary, {}))}"
            lines.append(line)
        seller = "\n".join(f"{k}: {v}" for k, v in card.seller_attrs.items()) or "—"
        lines += [
            "", f"Заголовок: {card.title}",
            f"Описание: {card.description[:MAX_DESCRIPTION_CHARS] or '—'}",
            f"Атрибуты продавца:\n{seller}",
        ]
        return "\n".join(lines)

    def _complete(self, prompt: str) -> dict | None:
        digest = hashlib.sha256(f"{self.model}\n{SYSTEM_PROMPT}\n{prompt}".encode()).hexdigest()
        cache_file = self.cache_dir / f"{digest}.json"
        if cache_file.exists():
            cached = json.loads(cache_file.read_text(encoding="utf-8"))
            self.stats.cache_hits += 1
            self.stats.cached_tokens += cached.get("tokens", 0)
            return cached["answer"]
        if self.max_calls is not None and self.stats.calls >= self.max_calls:
            return None
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        for attempt in range(RETRIES):
            try:
                self.stats.calls += 1
                response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0)
                usage = getattr(response, "usage", None)
                tokens = 0
                if usage:
                    self.stats.prompt_tokens += usage.prompt_tokens or 0
                    self.stats.completion_tokens += usage.completion_tokens or 0
                    tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)
                answer = _parse_json(response.choices[0].message.content or "")
                cache_file.write_text(json.dumps({"answer": answer, "tokens": tokens}, ensure_ascii=False),
                                      encoding="utf-8")
                return answer
            except ValueError as exc:  # модель вернула не-JSON: повтор не поможет
                log.warning("LLM вернула некорректный JSON: %s", exc)
                self.stats.errors += 1
                return None
            except Exception as exc:  # сетевые ошибки и лимиты шлюза — повтор с паузой
                log.warning("LLM ошибка (попытка %d): %s", attempt + 1, exc)
                self.stats.errors += 1
                time.sleep(2 ** attempt)
        return None

    def _to_candidates(self, card: Card, attributes: list[str], answer: dict) -> list[Candidate]:
        out = []
        for name in attributes:
            item = answer.get(name)
            if not isinstance(item, dict) or not item.get("value"):
                continue
            self.stats.proposed += 1
            spec = self.schema.attributes[name]
            canon = canonicalize(spec, str(item["value"]), self.schema)
            evidence = locate(card, spec, canon.key, self.schema) if canon else None
            if canon is None or evidence is None:
                self.stats.rejected_ungrounded += 1
                continue
            confidence = LLM_CONFIDENCE - (0 if canon.known else UNKNOWN_VALUE_PENALTY)
            out.append(Candidate(name, canon.display, canon.key, confidence, "llm", evidence, canon.known))
        return out


def _parse_json(content: str) -> dict:
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        raise ValueError(content[:200])
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("ожидался JSON-объект")
    return data
