"""Проверка, что значение действительно есть в карточке (защита от галлюцинаций LLM)."""

from __future__ import annotations

from attrx.extractors.base import Card, Evidence
from attrx.normalize import find_in_dictionary, find_patterns, find_quantities
from attrx.schema import AttributeSpec, Schema
from attrx.text import fold


def locate(card: Card, spec: AttributeSpec, key: str, schema: Schema) -> Evidence | None:
    """Ищет в заголовке, описании и атрибутах продавца фрагмент, который нормализуется в тот же ключ.
    Нет фрагмента — значит, значение выдумано, и его нельзя пускать в каталог."""
    fields = [("title", card.title), ("description", card.description)]
    fields += [(f"seller:{k}", v) for k, v in card.seller_attrs.items()]
    for name, text in fields:
        span = _find_span(text, spec, key, schema)
        if span:
            return Evidence(name, span[0], span[1], text[span[0]:span[1]])
    return None


def _find_span(text: str, spec: AttributeSpec, key: str, schema: Schema) -> tuple[int, int] | None:
    if not text:
        return None
    if spec.type == "quantity":
        for q in find_quantities(text, spec.unit_family or ""):
            if q.key == key:
                return q.start, q.end
        return None
    if spec.type == "enum":
        for m in find_in_dictionary(text, schema.dictionaries.get(spec.dictionary or "", {})):
            if fold(m.canonical) == key:
                return m.start, m.end
        # значение вне справочника (например, «титановый синий») ищем как подстроку
        return _substring(text, key)
    if spec.type == "pattern":
        for m in find_patterns(text, spec.name):
            if fold(m.canonical) == key:
                return m.start, m.end
        return _substring(text, key)
    return _substring(text, key, compact=True)


def _substring(text: str, key: str, compact: bool = False) -> tuple[int, int] | None:
    """Поиск ключа в тексте без учёта регистра/пунктуации (и пробелов, если compact)."""
    if not key:
        return None
    # строим свёрнутую копию текста с картой позиций, чтобы вернуть span в исходном тексте
    chars, positions = [], []
    for i, ch in enumerate(text.lower().replace("ё", "е")):
        if ch.isalnum() or ch == "." or (ch.isspace() and not compact):
            if ch.isspace() and chars and chars[-1] == " ":
                continue
            chars.append(" " if ch.isspace() else ch)
            positions.append(i)
    folded = "".join(chars)
    target = key.replace(" ", "") if compact else key
    idx = folded.find(target)
    if idx < 0:
        return None
    return positions[idx], positions[idx + len(target) - 1] + 1
