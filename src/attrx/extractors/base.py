"""Общие структуры: карточка товара и кандидат значения атрибута."""

from __future__ import annotations

from dataclasses import dataclass, field

from attrx.text import clean


@dataclass(frozen=True)
class Card:
    id: str
    title: str
    description: str = ""
    seller_attrs: dict[str, str] = field(default_factory=dict)
    category: str | None = None

    @classmethod
    def build(cls, id: str, title: str, description: str = "", seller_attrs: dict | None = None,
              category: str | None = None) -> Card:
        attrs = {clean(str(k)): clean(str(v)) for k, v in (seller_attrs or {}).items() if v not in (None, "")}
        return cls(str(id), clean(title), clean(description), attrs, category or None)

    def field_text(self, source_field: str) -> str:
        return {"title": self.title, "description": self.description}.get(source_field, "")


@dataclass(frozen=True)
class Evidence:
    """Где в карточке найдено значение: поле и позиция (для подсветки в демо и проверки галлюцинаций)."""

    field: str  # title | description | seller:<ключ>
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Candidate:
    attribute: str
    value: str  # каноническое отображение
    key: str  # ключ сравнения
    confidence: float  # априорная уверенность источника в [0, 1]
    source: str  # seller | rules | heuristic | llm
    evidence: Evidence | None = None
    known: bool = True  # значение из справочника/распознанного формата
