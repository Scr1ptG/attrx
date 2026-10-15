"""Схема атрибутов и справочники из configs/."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import yaml

from attrx.text import fold

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


@dataclass(frozen=True)
class AttributeSpec:
    name: str
    title: str
    type: str  # enum | quantity | pattern | text
    traffic_weight: float = 1.0
    dictionary: str | None = None
    unit_family: str | None = None
    context: tuple[str, ...] = ()
    anti_context: tuple[str, ...] = ()


@dataclass(frozen=True)
class CategorySpec:
    name: str
    title: str
    keywords: tuple[str, ...]
    attributes: tuple[str, ...]
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)


@dataclass
class Schema:
    attributes: dict[str, AttributeSpec]
    categories: dict[str, CategorySpec]
    seller_key_synonyms: dict[str, tuple[str, ...]]
    dictionaries: dict[str, dict[str, tuple[str, ...]]]

    @classmethod
    def load(cls, config_dir: Path | str = DEFAULT_CONFIG_DIR) -> Schema:
        config_dir = Path(config_dir)
        raw = yaml.safe_load((config_dir / "schema.yaml").read_text(encoding="utf-8"))
        attributes = {
            name: AttributeSpec(
                name=name,
                title=spec.get("title", name),
                type=spec["type"],
                traffic_weight=float(spec.get("traffic_weight", 1.0)),
                dictionary=spec.get("dictionary"),
                unit_family=spec.get("unit_family"),
                context=tuple(spec.get("context", ())),
                anti_context=tuple(spec.get("anti_context", ())),
            )
            for name, spec in raw["attributes"].items()
        }
        categories = {}
        for name, spec in raw["categories"].items():
            attrs = spec.get("attributes") or {}
            unknown = set(attrs) - set(attributes)
            if unknown:
                raise ValueError(f"Категория {name}: неизвестные атрибуты {sorted(unknown)}")
            categories[name] = CategorySpec(
                name=name,
                title=spec.get("title", name),
                keywords=tuple(spec.get("keywords", ())),
                attributes=tuple(attrs),
                ranges={a: tuple(v["range"]) for a, v in attrs.items() if v and "range" in v},
            )
        dictionaries = {}
        for path in sorted((config_dir / "dictionaries").glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            dictionaries[path.stem] = {str(k): tuple(str(s) for s in v) for k, v in data.items()}
        return cls(
            attributes=attributes,
            categories=categories,
            seller_key_synonyms={k: tuple(v) for k, v in raw.get("seller_key_synonyms", {}).items()},
            dictionaries=dictionaries,
        )

    def required(self, category: str | None) -> tuple[str, ...]:
        """Обязательные атрибуты категории; для неизвестной категории — объединение всех."""
        if category in self.categories:
            return self.categories[category].attributes
        return tuple(dict.fromkeys(a for c in self.categories.values() for a in c.attributes))

    def value_range(self, category: str | None, attribute: str) -> tuple[float, float] | None:
        cat = self.categories.get(category or "")
        return cat.ranges.get(attribute) if cat else None

    def detect_category(self, title: str) -> str | None:
        """Запасной вариант, если в данных нет поля категории: по ключевым словам заголовка."""
        folded = fold(title)
        for cat in self.categories.values():
            if any(fold(k) in folded for k in cat.keywords):
                return cat.name
        return None

    @cached_property
    def seller_key_index(self) -> list[tuple[str, str]]:
        """(свёрнутый синоним ключа, атрибут), длинные синонимы первыми — они специфичнее."""
        pairs = [(fold(s), attr) for attr, syns in self.seller_key_synonyms.items() for s in syns]
        return sorted(pairs, key=lambda p: -len(p[0]))
