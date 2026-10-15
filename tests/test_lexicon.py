import pytest

pl = pytest.importorskip("polars")

from attrx.lexicon import BRAND, Lexicon  # noqa: E402

SCHEMA = pl.DataFrame({
    "category_id": ["761"] * 6,
    "attribute": [BRAND, "Цвет", "Встроенная память (ROM)", "Емкость аккумулятора", "Поддержка NFC", "Материал корпуса"],
    "kind": ["text", "text", "enum", "num", "bool", "text"],
    "unit": [None, None, None, "мАч", None, None],
    "fill": [1.0] * 6,
    "n_cards": [100] * 6,
})
REFERENCE = pl.DataFrame({
    "category_id": ["761"] * 7,
    "attribute": [BRAND, BRAND, "Цвет", "Цвет", "Встроенная память (ROM)", "Встроенная память (ROM)", "Материал корпуса"],
    "value": ["Samsung", "Apple", "черный", "голубой", "256 ГБ", "128 ГБ", "металл"],
    "count": [50, 40, 30, 20, 25, 15, 10],
    "unit": [None] * 7,
})


@pytest.fixture(scope="module")
def lex():
    return Lexicon.fit(SCHEMA, REFERENCE)


def test_extracts_with_evidence(lex):
    preds = lex.predict("Смартфон Samsung Galaxy A57 8/128Gb голубой",
                        "<p>Ёмкость аккумулятора — 5000 мАч. Поддержка NFC для оплаты.</p>", "761")
    assert preds[BRAND].value == "Samsung"
    assert preds["Цвет"].value == "голубой"
    assert preds["Встроенная память (ROM)"].value == "128 ГБ"
    assert preds["Емкость аккумулятора"].value == "5000"
    assert preds["Поддержка NFC"].value == "Да"
    for p in preds.values():  # нет значений без подтверждения в тексте
        assert p.evidence is not None and p.evidence.text


def test_negated_flag_is_not_extracted(lex):
    preds = lex.predict("Смартфон Apple", "Модель без NFC", "761")
    assert "Поддержка NFC" not in preds


def test_key_value_pairs(lex):
    preds = lex.predict("Смартфон Apple", "Характеристики: ¶ Материал корпуса: металл ¶ Емкость аккумулятора: 4,5 тыс", "761")
    assert preds["Материал корпуса"].value == "металл"
    assert "kv" in preds["Материал корпуса"].sources


def test_unknown_category_gives_brand_only(lex):
    assert set(lex.predict("Samsung чехол черный", "", "999")) == {BRAND}


def test_brand_filter_drops_ambiguous_ngrams():
    ref = REFERENCE.vstack(pl.DataFrame({"category_id": ["761"], "attribute": [BRAND], "value": ["СМ"], "count": [3], "unit": [None]}))
    titles = pl.DataFrame({"title": [f"Стиральная машина СМ {i}" for i in range(5)], "brand": ["Beko"] * 5})
    assert ("см",) not in Lexicon.fit(SCHEMA, ref, titles=titles).brands
