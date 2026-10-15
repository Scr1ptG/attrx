from attrx.extractors.base import Card
from attrx.extractors.rules import extract_rules
from attrx.extractors.seller import extract_seller, map_seller_key


def _values(candidates):
    return {c.attribute: c.value for c in candidates}


def test_ram_and_storage_follow_nearest_keyword(schema):
    card = Card.build("1", "Ноутбук X", "8 ГБ ОЗУ, SSD 512 ГБ", category="laptop")
    assert _values(extract_rules(card, schema, ("ram", "storage"))) == {"ram": "8 ГБ", "storage": "512 ГБ"}


def test_phone_memory_pair_gives_storage(schema):
    card = Card.build("1", "Смартфон Samsung Galaxy A55 8/256 ГБ", category="smartphone")
    assert _values(extract_rules(card, schema, ("storage",)))["storage"] == "256 ГБ"


def test_value_outside_category_range_is_ignored(schema):
    card = Card.build("1", "Смартфон кнопочный, память 2 ГБ", category="smartphone")
    assert "storage" not in _values(extract_rules(card, schema, ("storage",)))


def test_every_rule_value_has_evidence_in_text(schema):
    card = Card.build("1", "Смартфон Samsung Galaxy S24 Ultra 12/256 ГБ черный", "Экран 6,8\", 5000 мАч", category="smartphone")
    for c in extract_rules(card, schema, schema.required("smartphone")):
        assert c.evidence is not None
        assert card.field_text(c.evidence.field)[c.evidence.start:c.evidence.end] == c.evidence.text


def test_model_heuristic_stops_at_specs(schema):
    card = Card.build("1", "Смартфон Samsung Galaxy S24 Ultra 12/256 ГБ черный", category="smartphone")
    assert _values(extract_rules(card, schema, ("model",)))["model"] == "Galaxy S24 Ultra"


def test_seller_key_matching_uses_whole_words(schema):
    assert map_seller_key("Цвет корпуса", schema) == "color"
    assert map_seller_key("Модель процессора", schema) == "cpu"
    assert map_seller_key("Тип аккумулятора", schema) is None


def test_seller_values_are_normalized(schema):
    card = Card.build("1", "x", seller_attrs={"Объём встроенной памяти": "256Gb", "Цвет": "Black"})
    assert _values(extract_seller(card, schema, ("storage", "color"))) == {"storage": "256 ГБ", "color": "черный"}
