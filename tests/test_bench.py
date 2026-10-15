from attrx import bench
from attrx.evaluate import calibration_error
from attrx.io import load_cards, load_gold

from conftest import SAMPLE_DIR


def test_offline_configs_run_without_llm(schema):
    cards = load_cards(SAMPLE_DIR / "cards.jsonl")
    gold, categories = load_gold(SAMPLE_DIR / "gold.jsonl")
    rows = {r.name: r for r in bench.run_bench(cards, gold, categories, schema)}
    assert set(rows) == {"seller", "rules", "baseline"}  # LLM-конфигурации пропущены без ключа
    assert all(r.llm_calls == 0 and r.tokens == 0 for r in rows.values())
    # покрытие базлайна не меньше, чем у любого источника по отдельности
    coverage = {name: r.metrics["micro"]["coverage"] for name, r in rows.items()}
    assert coverage["baseline"] >= max(coverage["seller"], coverage["rules"])
    assert "| Продавец + правила (базлайн) |" in bench.to_markdown(list(rows.values()))


def test_calibration_error_bounds(schema):
    gold = {"1": {"brand": "Samsung"}, "2": {"brand": "Apple"}}
    perfect = {("1", "brand"): ("Samsung", 1.0), ("2", "brand"): ("Xiaomi", 0.0)}
    ece, _ = calibration_error(perfect, gold, schema)
    assert ece == 0.0
    overconfident = {("1", "brand"): ("Xiaomi", 0.95), ("2", "brand"): ("Huawei", 0.95)}
    assert calibration_error(overconfident, gold, schema)[0] > 0.9
