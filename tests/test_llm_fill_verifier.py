import json
from types import SimpleNamespace

import pytest

pl = pytest.importorskip("polars")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from attrx.lexicon import Lexicon  # noqa: E402
from attrx.llm_fill import LLMFiller  # noqa: E402
from attrx.stack import Stack  # noqa: E402
from attrx.verifier import Verifier  # noqa: E402

SCHEMA = pl.DataFrame({"category_id": ["666"] * 3, "attribute": ["Цвет", "Объем морозильной камеры", "No Frost"],
                       "kind": ["text", "num", "bool"], "unit": [None, "л", None], "fill": [1.0] * 3, "n_cards": [50] * 3})
REFERENCE = pl.DataFrame({"category_id": ["666"], "attribute": ["Цвет"], "value": ["белый"], "count": [10], "unit": [None]})
TITLE = "Холодильник Haier C2F636"
DESC = "Объем морозильной камеры — 108 л. Корпус серебристого цвета. Система без No Frost."


class FakeClient:
    def __init__(self, answer: dict):
        self.answer, self.calls = json.dumps(answer, ensure_ascii=False), 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **_):
        self.calls += 1
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.answer))],
                               usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))


def make_filler(tmp_path, answer):
    lex = Lexicon.fit(SCHEMA, REFERENCE)
    askable = {"666": {"Цвет", "Объем морозильной камеры", "No Frost"}}
    return LLMFiller(lex, FakeClient(answer), "fake", askable, cache_dir=tmp_path)


def test_llm_values_are_grounded_by_quote(tmp_path):
    answer = {"Объем морозильной камеры": {"value": "108", "quote": "Объем морозильной камеры — 108 л"},
              "Цвет": {"value": "черный", "quote": "Корпус черного цвета"},          # цитаты нет в карточке
              "No Frost": {"value": "Нет", "quote": "без No Frost"}}
    f = make_filler(tmp_path, answer)
    got = {c.attribute: c.value for c in f.candidates_for(TITLE, DESC, "666", covered=set())}
    assert got == {"Объем морозильной камеры": "108", "No Frost": "Нет"}
    assert f.stats.rejected_ungrounded == 1 and f.stats.accepted == 2


def test_llm_value_must_follow_from_quote(tmp_path):
    f = make_filler(tmp_path, {"Объем морозильной камеры": {"value": "293", "quote": "Объем морозильной камеры — 108 л"}})
    assert f.candidates_for(TITLE, DESC, "666", covered=set()) == []


def test_llm_cache_and_covered_slots(tmp_path):
    f = make_filler(tmp_path, {"Цвет": {"value": "серебристый", "quote": "серебристого цвета"}})
    f.candidates_for(TITLE, DESC, "666", covered=set())
    f.candidates_for(TITLE, DESC, "666", covered=set())
    assert f.client.calls == 1 and f.stats.cache_hits == 1
    assert f.candidates_for(TITLE, DESC, "666", covered={"Цвет", "Объем морозильной камеры", "No Frost"}) == []


def test_stack_keeps_llm_confidence_when_verifier_has_not_seen_llm(tmp_path):
    class ConstVerifier:
        def predict_proba(self, frame):
            return np.full(frame.height, 0.01)

    f = make_filler(tmp_path, {"Объем морозильной камеры": {"value": "108", "quote": "морозильной камеры — 108"}})
    f.confidence = {"num": 0.77}
    cards = pl.DataFrame({"sku_id": ["1"], "category_id": ["666"], "title": [TITLE], "description": [DESC]})
    only_llm = Stack(f.lexicon, [], ConstVerifier(), gap_filler=f, gap_threshold=2.0, use_lexicon=False)
    row = only_llm.predict_frame(cards).filter(pl.col("attribute") == "Объем морозильной камеры").row(0, named=True)
    assert row["source"] == "llm" and row["confidence"] == 0.77
    mixed = Stack(f.lexicon, [], ConstVerifier(), gap_filler=f, gap_threshold=2.0)  # LLM согласна с правилом
    row = mixed.predict_frame(cards).filter(pl.col("attribute") == "Объем морозильной камеры").row(0, named=True)
    assert row["source"] == "llm+number" and row["confidence"] == 0.01


def test_verifier_learns_and_calibrates():
    rng = np.random.default_rng(0)
    n = 600
    good = rng.random(n) < 0.5
    frame = pl.DataFrame({
        "category_id": ["666"] * n, "attribute": ["Цвет"] * n, "kind": ["text"] * n,
        "value": ["белый"] * n, "raw_confidence": np.where(good, 0.8, 0.6) + rng.normal(0, 0.05, n),
        "n_sources": np.where(good, 2, 1), "n_values": [1] * n, "conflict": [False] * n,
        "evidence_field": ["title"] * n, "source": np.where(good, "lexicon+analog", "lexicon"),
    })
    groups = np.arange(n) % 10
    ver = Verifier.from_reference(SCHEMA, REFERENCE).fit(frame, good.astype(int), groups)
    p = ver.predict_proba(frame)
    assert ver.report["oof_auc"] > 0.9
    assert p[good].mean() > 0.8 > 0.2 > p[~good].mean()
    assert 0.0 <= ver.threshold_for(good.astype(int), 0.95) <= 1.0


def test_llm_routing_skips_cards_with_low_expected_yield(tmp_path):
    f = make_filler(tmp_path, {})
    f.askable = {"666": dict(zip(SCHEMA["attribute"].to_list(), (0.2, 0.5, 0.1)))}
    f.min_expected = 0.7  # все три слота открыты: 0.2 + 0.5 + 0.1 = 0.8 — спрашиваем
    f.candidates_for(TITLE, DESC, "666", covered=set())
    assert f.client.calls == 1
    covered = {SCHEMA["attribute"][1]}  # закрыт самый «находимый» слот: 0.3 < 0.7 — LLM не вызывается
    assert f.candidates_for(TITLE, DESC, "666", covered=covered) == []
    assert f.client.calls == 1 and f.stats.skipped == 1


def test_llm_malformed_json_is_salvaged_and_unknown_fields_are_dropped(tmp_path):
    f = make_filler(tmp_path, {})
    # пропущена запятая между объектами (19% ответов Qwen3-8B), а «Вес» не спрашивали — поле не добавляем
    f.client.answer = ('{"Объем морозильной камеры": {"value": "108", "quote": "Объем морозильной камеры — 108 л"}'
                       '  "Вес": {"value": "60", "quote": "108 л"}}')
    got = {c.attribute: c.value for c in f.candidates_for(TITLE, DESC, "666", covered=set())}
    assert got == {"Объем морозильной камеры": "108"}
    assert f.stats.salvaged == 1 and f.stats.unknown_attr == 1 and f.stats.parse_errors == 0


def test_llm_flag_needs_quote_about_the_flag_and_negation_as_a_word(tmp_path):
    desc = "Есть доступ в интернет. Система без No Frost."
    no = make_filler(tmp_path / "no", {"No Frost": {"value": "Нет", "quote": "Есть доступ в интернет"}})
    assert no.candidates_for(TITLE, desc, "666", covered=set()) == []  # цитата не о No Frost; «интернет» — не «нет»
    yes = make_filler(tmp_path / "yes", {"No Frost": {"value": "Да", "quote": "Система без No Frost"}})
    assert yes.candidates_for(TITLE, desc, "666", covered=set()) == []  # «Да» при отрицании в цитате
    assert no.stats.rejected_ungrounded == 1 and yes.stats.rejected_ungrounded == 1


def test_llm_enum_value_must_be_in_category_vocabulary(tmp_path):
    schema = pl.DataFrame({"category_id": ["666"], "attribute": ["Тип управления"], "kind": ["enum"], "unit": [None],
                           "fill": [1.0], "n_cards": [50]})
    ref = pl.DataFrame({"category_id": ["666"] * 2, "attribute": ["Тип управления"] * 2,
                        "value": ["сенсорное", "механическое"], "count": [10, 5], "unit": [None, None]})
    answer = {"Тип управления": {"value": "голосовое", "quote": "голосовое управление"}}
    f = LLMFiller(Lexicon.fit(schema, ref), FakeClient(answer), "fake", {"666": {"Тип управления"}}, cache_dir=tmp_path)
    assert f.candidates_for("Плита", "Есть голосовое управление.", "666", covered=set()) == []
    assert f.stats.rejected_value == 1


def test_llm_cache_skips_empty_answers_and_survives_corrupt_entries(tmp_path):
    f = make_filler(tmp_path, {})
    f.client.answer = ""
    f.candidates_for(TITLE, DESC, "666", covered=set())
    assert list(tmp_path.glob("*.json")) == []  # пустой ответ не кэшируем: следующий запуск спросит снова
    f.client.answer = json.dumps({"Цвет": {"value": "серебристый", "quote": "серебристого цвета"}}, ensure_ascii=False)
    f.candidates_for(TITLE, DESC, "666", covered=set())
    next(tmp_path.glob("*.json")).write_text("{обрыв записи", encoding="utf-8")
    f.candidates_for(TITLE, DESC, "666", covered=set())  # повреждённая запись — промах, а не падение
    assert f.client.calls == 3 and f.stats.cache_hits == 0


def test_verifier_does_not_credit_agreement_with_llm(tmp_path):
    class SpyVerifier:
        def predict_proba(self, frame):
            self.seen = frame
            return np.full(frame.height, 0.5)

    spy = SpyVerifier()
    f = make_filler(tmp_path, {"Объем морозильной камеры": {"value": "108", "quote": "морозильной камеры — 108"}})
    cards = pl.DataFrame({"sku_id": ["1"], "category_id": ["666"], "title": [TITLE], "description": [DESC]})
    Stack(f.lexicon, [], spy, gap_filler=f, gap_threshold=2.0).predict_frame(cards)
    row = spy.seen.filter(pl.col("attribute") == "Объем морозильной камеры").row(0, named=True)
    assert row["source"] == "number" and row["n_sources"] == 1  # верификатор видит строку так, будто LLM нет
