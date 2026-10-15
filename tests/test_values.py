from attrx.values import canon, card_text, key, quantity, same, tokens


def test_tokens_split_digits_and_units():
    assert [t.norm for t in tokens("Galaxy A57 8/128Gb, 6,7\"")] == ["galaxy", "a", "57", "8", "128", "гб", "6.7", '"']


def test_key_ignores_case_yo_and_punctuation():
    assert key("USB Type-C") == key("usb type c")
    assert key("Чёрный") == "черный"


def test_quantity_units():
    assert quantity("256 ГБ") == ("data", 256.0)
    assert quantity("1 ТБ") == ("data", 1024.0)
    assert quantity("6.7", '"') == ("inch", 6.7)
    assert quantity("1 год") is None


def test_same_numbers_with_unit_conversion():
    assert same("1024", "ГБ", "1 ТБ", None)
    assert same("150", "см", "1500", "мм")
    assert same("6,7", None, "6.7", '"')
    assert not same("6.5", None, "6.7", '"')
    assert not same("5", "кг", "5", "см")


def test_same_multivalue_is_order_insensitive():
    assert same("стекло, металл", None, "металл/ стекло", None)
    assert canon("8 ГБ/256 ГБ") == {"data:8", "data:256"}
    assert not same("металл", None, "металл/ стекло", None)


def test_card_text_strips_html_and_keeps_lines():
    text = card_text("<p>Цвет: <b>белый</b><br>Вес: 2 кг</p>")
    assert "<" not in text and "Цвет: белый" in text and "¶" in text
