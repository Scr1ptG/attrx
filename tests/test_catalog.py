import pytest

pl = pytest.importorskip("polars")

from attrx.catalog import group_expr, split_of  # noqa: E402


def test_split_is_stable():
    # Хэши зафиксированы: изменение разбиения — это новая версия данных, а не случайность.
    groups = ("3p:ООО \"ВЕРТЕКС\"", "1p:samsung", "1p:apple", "3p:М.Видео тест", "g1", "g2", "g3", "g4")
    assert [split_of(g) for g in groups] == ["train", "train", "train", "test", "train", "train", "test", "train"]
    assert {split_of(f"g{i}") for i in range(200)} == {"train", "val", "test"}
    share = sum(split_of(f"g{i}") == "train" for i in range(5000)) / 5000
    assert 0.66 < share < 0.74


def test_group_first_party_by_brand_third_party_by_seller():
    df = pl.DataFrame({"seller": ["М.Видео", "ООО Ромашка", None], "vendor": ["Samsung", "Samsung", "LG"]})
    assert df.select(group_expr())["group"].to_list() == ["1p:samsung", "3p:ООО Ромашка", "na:lg"]
