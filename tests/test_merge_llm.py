import json
from types import SimpleNamespace

from attrx.extractors.base import Candidate, Card
from attrx.llm import LLMExtractor
from attrx.merge import merge


def _cand(value, conf, source):
    return Candidate("storage", value, value, conf, source)


def test_agreeing_sources_boost_confidence():
    pred = merge([_cand("256", 0.9, "seller"), _cand("256", 0.8, "rules")])["storage"]
    assert pred.confidence == 0.98 and pred.status == "accept" and pred.sources == ("rules", "seller")


def test_same_source_is_not_double_counted():
    assert merge([_cand("256", 0.8, "rules"), _cand("256", 0.8, "rules")])["storage"].confidence == 0.8


def test_conflict_goes_to_review_and_is_reported():
    pred = merge([_cand("128", 0.9, "seller"), _cand("256", 0.8, "rules")])["storage"]
    assert pred.value == "128" and pred.status == "review" and pred.conflict == "256"


class FakeClient:
    """Имитация OpenAI-совместимого клиента: всегда возвращает заданный ответ."""

    def __init__(self, answer: dict):
        self.calls = 0
        message = SimpleNamespace(content=json.dumps(answer, ensure_ascii=False))
        response = SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=None)

        def create(**_):
            self.calls += 1
            return response

        self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))


CARD = Card.build("1", "Смартфон Samsung Galaxy A55 8/256 ГБ", "Яркий экран", category="smartphone")
ANSWER = {
    "storage": {"value": "256 ГБ", "quote": "256 ГБ"},
    "color": {"value": "синий", "quote": "синий"},  # цвета в карточке нет — галлюцинация
    "battery_capacity": {"value": "5000 мАч", "quote": ""},  # тоже выдумано
}


def test_llm_values_without_evidence_are_rejected(schema, tmp_path):
    llm = LLMExtractor(schema, model="fake", cache_dir=tmp_path, client=FakeClient(ANSWER))
    out = llm.extract(CARD, ["storage", "color", "battery_capacity"])
    assert [(c.attribute, c.value, c.source) for c in out] == [("storage", "256 ГБ", "llm")]
    assert llm.stats.proposed == 3 and llm.stats.rejected_ungrounded == 2


def test_llm_answers_are_cached(schema, tmp_path):
    client = FakeClient(ANSWER)
    llm = LLMExtractor(schema, model="fake", cache_dir=tmp_path, client=client)
    llm.extract(CARD, ["storage"])
    llm.extract(CARD, ["storage"])
    assert client.calls == 1 and llm.stats.cache_hits == 1


def test_llm_budget_is_respected(schema, tmp_path):
    client = FakeClient(ANSWER)
    llm = LLMExtractor(schema, model="fake", cache_dir=tmp_path, client=client, max_calls=0)
    assert llm.extract(CARD, ["storage"]) == [] and client.calls == 0


def test_llm_disabled_without_key(schema, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert LLMExtractor.from_env(schema, "any-model") is None
