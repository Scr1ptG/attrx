from attrx.evaluate import evaluate
from attrx.io import load_cards, load_gold
from attrx.pipeline import Pipeline

from conftest import SAMPLE_DIR


def test_counts_follow_slot_filling_rules(schema):
    gold = {"1": {"brand": "Samsung", "color": "черный"}, "2": {"brand": "Apple"}}
    preds = {
        ("1", "brand"): ("Samsung", 0.9),
        ("1", "color"): ("белый", 0.9),  # неверно: FP и FN
        ("2", "brand"): ("Apple", 0.3),  # ниже порога: не публикуется -> FN
        ("2", "color"): ("красный", 0.9),  # в эталоне пусто: FP без подтверждения
    }
    report = evaluate(preds, gold, schema, {"1": "smartphone", "2": "smartphone"}, threshold=0.5, sweep=False)
    brand, color = report.per_attribute["brand"], report.per_attribute["color"]
    assert (brand.tp, brand.fp, brand.fn) == (1, 0, 1)
    assert (color.tp, color.fp, color.fn, color.unsupported) == (0, 2, 1, 1)


def test_equivalent_spellings_count_as_match(schema):
    report = evaluate({("1", "storage"): ("1024GB", 0.9)}, {"1": {"storage": "1 ТБ"}}, schema,
                      {"1": "laptop"}, sweep=False)
    assert report.per_attribute["storage"].tp == 1


def test_sample_regression(schema):
    """Страховка от регрессий на синтетическом сэмпле (не замена оценке на датасете кейса)."""
    pipeline = Pipeline(schema)
    results = [pipeline.run(card) for card in load_cards(SAMPLE_DIR / "cards.jsonl")]
    preds = {(r.card_id, p.attribute): (p.value, p.confidence) for r in results for p in r.predictions.values()}
    gold, categories = load_gold(SAMPLE_DIR / "gold.jsonl")
    micro = evaluate(preds, gold, schema, categories, sweep=False).micro
    assert micro.precision >= 0.9 and micro.recall >= 0.9
