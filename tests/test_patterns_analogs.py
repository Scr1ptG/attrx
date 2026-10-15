import pytest

pl = pytest.importorskip("polars")

from attrx.analogs import AnalogIndex, SimilarItems, title_words  # noqa: E402
from attrx.patterns import WHOLE, Target, TuplePatterns, find_tuples, render  # noqa: E402

SCHEMA = pl.DataFrame({
    "category_id": ["761"] * 4,
    "attribute": ["Оперативная память (RAM)", "Встроенная память (ROM)", "Оперативная и встроенная память", "Вес"],
    "kind": ["enum", "enum", "enum", "num"],
    "unit": [None, None, None, "г"],
    "fill": [1.0] * 4,
    "n_cards": [100] * 4,
})


def test_find_tuples_shares_trailing_unit():
    (m,) = find_tuples("Смартфон Galaxy A57 8/128Gb Icy Blue", "title")
    assert m.key == ("/", 2) and m.numbers == (8.0, 128.0) and m.units == ("gb", "gb")
    (d,) = find_tuples("Габариты 42,5x72.2х6.3 см", "description")
    assert d.key == ("x", 3) and d.numbers == (42.5, 72.2, 6.3) and d.units == ("см", "см", "см")


def test_render_converts_units_and_whole_tuple():
    (m,) = find_tuples("8/256GB", "title")
    assert render(Target("RAM", "enum", None, "ГБ"), m, 0) == "8 ГБ"
    assert render(Target("RAM+ROM", "enum", None, "ГБ"), m, WHOLE) == "8 ГБ/256 ГБ"
    (d,) = find_tuples("16x7x0.8 см", "title")
    assert render(Target("Высота", "num", "мм", None), d, 0) == "160"
    assert render(Target("Вес", "num", "г", None), d, 0) is None  # длина в слот массы не подставляется


def test_patterns_learn_position_to_attribute():
    cards = pl.DataFrame({"sku_id": [str(i) for i in range(8)], "category_id": ["761"] * 8,
                          "title": [f"Смартфон X{i} {r}/{m}Gb" for i, (r, m) in enumerate([(8, 128), (12, 256)] * 4)],
                          "description": [""] * 8})
    rows = []
    for i, (r, m) in enumerate([(8, 128), (12, 256)] * 4):
        rows += [(str(i), "Оперативная память (RAM)", f"{r} ГБ", None), (str(i), "Встроенная память (ROM)", f"{m} ГБ", None),
                 (str(i), "Оперативная и встроенная память", f"{r} ГБ/{m} ГБ", None)]
    gold = pl.DataFrame(rows, schema=["sku_id", "attribute", "value", "unit"], orient="row")
    ref = gold.with_columns(pl.lit("761").alias("category_id"), pl.lit(1).alias("count")).select(
        "category_id", "attribute", "value", "count", "unit")
    pat = TuplePatterns.fit(cards, gold, SCHEMA, ref, min_support=3)
    got = {c.attribute: c.value for c in pat.candidates("Смартфон Y 6/64GB", "", "761")}
    assert got == {"Оперативная память (RAM)": "6 ГБ", "Встроенная память (ROM)": "64 ГБ",
                   "Оперативная и встроенная память": "6 ГБ/64 ГБ"}
    assert all(c.evidence.text == "6/64GB" for c in pat.candidates("Смартфон Y 6/64GB", "", "761"))


def test_title_words_normalize_model_codes():
    assert "kt2239" in title_words("Аэрогриль Kitfort КТ-2239") and "kt2239" in title_words("Kitfort KT2239")


@pytest.fixture()
def index():
    offers = pl.DataFrame({"offer_id": ["a", "b", "c"], "category_id": ["761"] * 3,
                           "name": ["Смартфон Samsung Galaxy A57 8/128Gb черный", "Смартфон Samsung Galaxy A57 8/256Gb синий",
                                    "Смартфон Samsung Galaxy S25 12/256Gb"]})
    facts = pl.DataFrame({"offer_id": ["a", "a", "b", "b", "c"],
                          "attribute": ["Вес", "Страна", "Вес", "Страна", "Вес"],
                          "value": ["0.2", "Вьетнам", "200", "Вьетнам", "160"],
                          "unit": ["кг", None, "г", None, "г"]})
    return AnalogIndex.build(offers, facts, SCHEMA)


def test_analogs_vote_and_convert_units(index):
    cands = {c.attribute: c for c in index.candidates("Смартфон Samsung Galaxy A57 8/256Gb белый", "", "761")}
    assert cands["Страна"].value == "Вьетнам" and cands["Страна"].source == "analog"
    assert cands["Вес"].value == "200"  # 0.2 кг у одного аналога и 200 г у другого — одно значение в единицах слота
    assert cands["Вес"].evidence.field.startswith("analog:")


def test_analogs_need_shared_model_code(index):
    assert index.candidates("Смартфон Samsung Galaxy без кода", "", "761") == []
    assert index.candidates("Смартфон Samsung Galaxy A57", "", "999") == []


@pytest.fixture()
def similar():
    offers = pl.DataFrame({"offer_id": ["d", "e", "f", "g"], "category_id": ["761"] * 4,
                           "name": ["Чайник Tefal Loft белый", "Чайник Tefal Loft черный", "Чайник Tefal Loft серый",
                                    "Чайник Bosch KT2239 белый"]})
    facts = pl.DataFrame({"offer_id": ["d", "d", "e", "e", "f", "f", "g"],
                          "attribute": ["Страна", "Вес", "Страна", "Вес", "Страна", "Вес", "Страна"],
                          "value": ["Китай", "1.2", "Китай", "1.2", "Китай", "1.5", "Германия"],
                          "unit": [None, "кг", None, "кг", None, "кг", None]})
    return SimilarItems.build(AnalogIndex.build(offers, facts, SCHEMA), offers)


def test_similar_items_publish_only_unanimous_values(similar):
    cands = {c.attribute: c for c in similar.candidates("Чайник Tefal Loft красный", "", "761")}
    assert cands["Страна"].value == "Китай" and cands["Страна"].source == "similar"
    assert cands["Страна"].evidence.field.startswith("similar:")
    assert "Вес" not in cands  # 1.2 / 1.2 / 1.5: согласие 0.67 ниже порога


def test_similar_items_defer_to_exact_analog_and_need_three_neighbors(similar):
    assert similar.candidates("Чайник Bosch KT2239 красный", "", "761") == []  # есть аналог по коду модели
    assert similar.candidates("Чайник Bosch Styline", "", "761") == []  # один похожий товар — мало
