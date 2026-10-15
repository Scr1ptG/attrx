import pytest

from attrx.normalize import canonicalize, find_in_dictionary, find_quantities, parse_quantity


def test_data_units_are_equivalent():
    assert parse_quantity("1 ТБ", "data").key == parse_quantity("1024GB", "data").key


@pytest.mark.parametrize("raw, display", [
    ("256GB", "256 ГБ"), ("256 гб", "256 ГБ"), ("1 TB", "1 ТБ"), ("512 гигабайт", "512 ГБ"), ("128", "128 ГБ"),
])
def test_data_display(raw, display):
    assert parse_quantity(raw, "data").display == display


@pytest.mark.parametrize("raw, display", [('6,7"', "6.7 дюйма"), ("15.6 дюймов", "15.6 дюйма"), ("6 дюймов", "6 дюймов")])
def test_diagonal_display(raw, display):
    assert parse_quantity(raw, "length_in").display == display


@pytest.mark.parametrize("raw", ["3349 мА·ч", "3349mAh", "3349 мАч"])
def test_battery_units(raw):
    assert parse_quantity(raw, "charge").display == "3349 мАч"


def test_memory_pair_keeps_order():
    found = find_quantities("Смартфон 12/256 ГБ", "data")
    assert [(q.value, q.hint) for q in found] == [(12, "first"), (256, "second")]


def test_no_quantity_inside_model_code():
    assert find_quantities("Vivobook X1504ZA-BQ1143", "data") == []


@pytest.mark.parametrize("raw", ["чёрная", "черного", "Black", "Midnight", "Полночный черный"])
def test_color_forms_map_to_canonical(schema, raw):
    assert canonicalize(schema.attributes["color"], raw, schema).display == "черный"


def test_color_synonym(schema):
    assert canonicalize(schema.attributes["color"], "алый", schema).display == "красный"


def test_color_stem_does_not_match_other_words(schema):
    assert find_in_dictionary("белорусский производитель", schema.dictionaries["colors"]) == []


def test_unknown_color_is_kept_but_marked(schema):
    canon = canonicalize(schema.attributes["color"], "Титановый", schema)
    assert canon.display == "Титановый" and not canon.known


def test_cpu_spellings_are_equal(schema):
    spec = schema.attributes["cpu"]
    keys = {canonicalize(spec, raw, schema).key for raw in ["Intel Core i5 1235U", "i5-1235U", "intel core i5-1235u"]}
    assert len(keys) == 1


def test_model_key_ignores_leading_brand(schema):
    spec = schema.attributes["model"]
    assert canonicalize(spec, "Samsung Galaxy S24", schema).key == canonicalize(spec, "Galaxy S24", schema).key
