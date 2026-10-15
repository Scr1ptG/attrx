"""Судья: разметка из каталога, правила текстового подтверждения и вето LLM (модель подменена)."""

from pathlib import Path

import pytest

pl = pytest.importorskip("polars")
np = pytest.importorskip("numpy")

from attrx.extractors.base import Candidate, Evidence  # noqa: E402
from attrx.judge import CONFIRM, CONTRA, NONE, Backbone, Judge, build_examples, locate, trivial_rule, window  # noqa: E402
from attrx.lexicon import Lexicon  # noqa: E402
from attrx.metrics import supported_flags  # noqa: E402
from attrx.stack import Stack  # noqa: E402
from attrx.values import key, tokens  # noqa: E402

ROM, RAM, COUNTRY, WEIGHT = "Встроенная память (ROM)", "Оперативная память (RAM)", "Страна производства", "Вес"
SCHEMA = pl.DataFrame({"category_id": ["761"] * 5, "attribute": [ROM, RAM, "NFC", COUNTRY, WEIGHT],
                       "kind": ["enum", "enum", "bool", "text", "num"], "unit": [None, None, None, None, "г"],
                       "fill": [1.0] * 5, "n_cards": [50] * 5})
REF = [(ROM, "256 ГБ", 10), (ROM, "128 ГБ", 8), (RAM, "8 ГБ", 9), (RAM, "6 ГБ", 7), (COUNTRY, "Китай", 9),
       (COUNTRY, "Вьетнам", 7), (WEIGHT, "180", 9), (WEIGHT, "2", 8), (WEIGHT, "150", 7)]
REFERENCE = pl.DataFrame({"category_id": ["761"] * len(REF), "attribute": [a for a, _, _ in REF],
                          "value": [v for _, v, _ in REF], "count": [n for _, _, n in REF],
                          "unit": ["г" if a == WEIGHT else None for a, _, _ in REF]})
CARDS = pl.DataFrame({
    "sku_id": ["1", "2", "3"], "category_id": ["761"] * 3,
    "title": ["Смартфон Galaxy, встроенная память 256 ГБ, оперативная память 8 ГБ, черный", "Смартфон Lite без NFC", "Чехол"],
    "description": ["Модуль NFC для оплаты. Страна производства уточняется у продавца. Гарантия 2 года.",
                    "Вес 150 г.", "Подходит для 2 моделей."]})
GOLD = pl.DataFrame({
    "sku_id": ["1"] * 5 + ["2"] * 2 + ["3"],
    "attribute": [ROM, RAM, "NFC", COUNTRY, WEIGHT, "NFC", WEIGHT, WEIGHT],
    "value": ["256 ГБ", "8 ГБ", "Да", "Китай", "180", "Нет", "150", "2"],
    "unit": [None, None, None, None, "г", None, "г", "г"]})


def _examples():
    lex = Lexicon.fit(SCHEMA, REFERENCE)
    return build_examples(CARDS, GOLD, SCHEMA, REFERENCE, lex, supported_flags(CARDS, GOLD, SCHEMA), per_card=3)


def test_locate_finds_value_and_flag_name():
    text = "Смартфон Galaxy 256 ГБ черный ¶ Есть модуль NFC"
    tk = tokens(text)
    s, e = locate(tk, ROM, "enum", "256 ГБ")
    assert text[s:e] == "256 ГБ"
    s, e = locate(tk, "NFC", "bool", "Да")
    assert text[s:e] == "NFC"
    assert locate(tk, COUNTRY, "text", "Китай") is None
    assert "256 ГБ" in window(text, (16, 22), width=10)


def test_examples_need_context_not_substring():
    got = {(r["sku_id"], r["hypothesis"], r["label"], r["recipe"]) for r in _examples().iter_rows(named=True)}
    pairs = {(s, h, lab) for s, h, lab, _ in got}
    assert ("1", f"{ROM}: 256 ГБ", CONFIRM) in pairs and ("1", f"{RAM}: 8 ГБ", CONFIRM) in pairs
    assert ("1", f"{ROM}: 128 ГБ", CONTRA) in pairs  # то же окно, другое значение из справочника
    # соседняя характеристика в том же окне: «8 ГБ» — это ОЗУ, не встроенная память
    assert ("1", f"{ROM}: 8 ГБ", CONTRA) in pairs or ("1", f"{RAM}: 256 ГБ", CONTRA) in pairs
    assert ("1", "NFC: Да", CONFIRM) in pairs and ("1", "NFC: Нет", CONTRA) in pairs
    assert ("2", "NFC: Нет", CONFIRM, "negation") in got and ("2", "NFC: Да", CONTRA) in pairs  # «без NFC»
    assert ("1", f"{COUNTRY}: Китай", NONE, "mentioned") in got  # страна упомянута, но не названа
    assert ("1", f"{WEIGHT}: 2 г", NONE, "coincidence") in got  # «Гарантия 2 года» не про вес
    assert ("2", f"{WEIGHT}: 150 г", CONFIRM) in pairs  # «Вес 150 г» — число рядом с названием
    assert not any(s == "3" and lab == CONFIRM for s, _, lab in pairs)  # «для 2 моделей» вес не подтверждает


def test_trivial_rule_fails_on_hard_examples():
    ex = _examples()
    hard = ex.with_columns(pl.Series("pred", trivial_rule(ex))).filter(pl.col("recipe").is_in(["coincidence", "cross", "negation"]))
    assert (hard["pred"] != hard["label"]).any()  # «значение в окне» не равно «текст подтверждает»


def _fake_judge(rule):
    judge = Judge(net=None, tokenizer=None, backbone=Backbone("fake", Path(".")))
    judge.probs = lambda hyps, wins: np.array([rule(h) for h in hyps])
    return judge


def test_annotate_grants_text_evidence_only_for_found_value_in_context():
    judge = _fake_judge(lambda h: [0.05, 0.9, 0.05] if "128" in h else [0.9, 0.05, 0.05])
    frame = pl.DataFrame({
        "sku_id": ["1"] * 5, "category_id": ["761"] * 5,
        "attribute": [ROM, ROM, COUNTRY, WEIGHT, "Цвет"], "kind": ["enum", "enum", "text", "num", "enum"],
        "value": ["256 ГБ", "128 ГБ", "Китай", "2", "синий"], "unit": [None, None, None, "г", None],
        "source": ["analog", "llm", "similar", "analog", "similar"], "n_sources": [1] * 5,
        "evidence_field": ["analog:9", "description", "similar:7", "analog:3", "similar:2"], "evidence_text": ["x"] * 5})
    out = judge.annotate(frame, CARDS, Lexicon.fit(SCHEMA, REFERENCE)).to_dicts()
    assert out[0]["source"] == "analog+judge" and out[0]["evidence_field"] == "title" and "256 ГБ" in out[0]["evidence_text"]
    assert out[1]["source"] == "llm" and out[1]["judge_contra"] > 0.5  # LLM судья только проверяет
    assert out[2]["judge_confirm"] > 0.5 and out[2]["source"] == "similar"  # упомянута характеристика, не значение
    assert out[3]["judge_confirm"] > 0.5 and out[3]["source"] == "analog"  # «2 года» — не вес
    assert out[4]["judge_confirm"] is None  # ни значения, ни характеристики в тексте — судью не спрашиваем
    assert judge.report["last_annotate"] == {"rows": 5, "judged": 4, "granted": 1}


class _FakeLLM:
    def candidates_for(self, title, description, category, covered):
        return [Candidate(RAM, "6 ГБ", key("6 ГБ"), 0.7, "llm", Evidence("title", 0, 5, "x")),
                Candidate(COUNTRY, "Китай", key("Китай"), 0.7, "llm", Evidence("description", 0, 5, "y"))]


def test_stack_vetoes_llm_value_contradicted_by_text():
    judge = _fake_judge(lambda h: [0.05, 0.9, 0.05] if "6 ГБ" in h else [0.3, 0.3, 0.4])
    judge.veto = 0.8
    # только LLM: иначе значение из текста («8 ГБ») выиграло бы ещё при слиянии кандидатов
    stack = Stack(Lexicon.fit(SCHEMA, REFERENCE), gap_filler=_FakeLLM(), judge=judge, use_lexicon=False)
    out = stack.predict_frame(CARDS.head(1))
    llm = {r["attribute"]: r for r in out.filter(pl.col("source") == "llm").to_dicts()}
    assert llm[RAM]["confidence"] == 0.0  # текст говорит «8 ГБ»: значение LLM не показываем
    assert llm[COUNTRY]["confidence"] == 0.7  # «не указано» — не вето


class _CharTokenizer:
    """Токенизатор-заглушка: пара «гипотеза, окно» -> id символов с разделителем, как у настоящих."""

    def __call__(self, a, b, truncation=None, max_length=32, padding=True, return_tensors="pt"):
        import torch
        from transformers import BatchEncoding

        seqs = [([1] + [3 + ord(c) % 60 for c in x][:10] + [2] + [3 + ord(c) % 60 for c in y])[:max_length] for x, y in zip(a, b)]
        n = max(map(len, seqs))
        ids = torch.tensor([s + [0] * (n - len(s)) for s in seqs])
        return BatchEncoding({"input_ids": ids, "attention_mask": (ids != 0).long()})


def test_fit_trains_top_blocks_and_calibrates_on_tiny_encoder(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from attrx.judge import make_net

    torch.manual_seed(0)
    try:
        enc = transformers.T5EncoderModel(transformers.T5Config(vocab_size=64, d_model=16, d_kv=4, num_heads=2, d_ff=32,
                                                                num_layers=3, decoder_start_token_id=0))
    except (RuntimeError, ModuleNotFoundError) as e:  # CUDA-сборка torch 2.6 без доступной видеокарты
        chain, c = [], e
        while c is not None:
            chain.append(str(c))
            c = c.__cause__
        if not any("accelerator" in m for m in chain):  # transformers 5 не импортирует модель — это окружение, не код
            raise
        pytest.skip(f"окружение: {chain[-1]}")
    judge = Judge(net=make_net(enc, 16), tokenizer=_CharTokenizer(), backbone=Backbone("tiny", Path(".")), max_len=48)
    rows = [("NFC: Да", "есть NFC", CONFIRM), ("NFC: Нет", "есть NFC", CONTRA), ("Цвет: черный", "смартфон", NONE)] * 6
    df = pl.DataFrame(rows, schema=["hypothesis", "window", "label"], orient="row")
    judge.fit(df, df, batch=4, grad_accum=2, train_top=1, log_every=1000)
    frozen = [p.requires_grad for b in enc.encoder.block[:2] for p in b.parameters()]
    assert not any(frozen) and all(p.requires_grad for p in enc.encoder.block[2].parameters())
    assert judge.report["train"]["steps"] == 3 and judge.temperature > 0  # 5 пакетов по 2 + хвост эпохи
    p = judge.probs(df["hypothesis"].to_list(), df["window"].to_list())
    assert p.shape == (18, 3) and np.allclose(p.sum(1), 1.0)
    assert set(judge.evaluate(df)) >= {"accuracy", "nll", "macro_f1", "operating"}
    judge.save(tmp_path / "judge.pt")
    blob = torch.load(tmp_path / "judge.pt", map_location="cpu", weights_only=True)
    assert blob["report"]["id"] and blob["veto"] == judge.veto


def test_negation_and_context_stay_within_sentence():
    from attrx.judge import _negated, _span_in_context
    from attrx.lexicon import make_slot

    for text, attr, neg in (("Есть NFC. Нет зарядки в комплекте", "NFC", False), ("Смартфон без NFC", "NFC", True),
                            ("NFC: нет", "NFC", True), ("Зарядки нет. NFC есть", "NFC", False)):
        tk = tokens(text)
        span = next((i, i + 1) for i, t in enumerate(tk) if t.norm == "nfc")
        assert _negated(tk, text, span) is neg, text
    slot = make_slot(WEIGHT, "num", "г")
    text = "Гарантия 2 года. Вес: 5 кг"
    tk = tokens(text)
    assert _span_in_context(tk, text, [t.norm for t in tk], slot, "num", "2", "г") is None  # «вес» — в другой фразе
    text = "Подходит для 2 моделей, вес 2 г"
    tk = tokens(text)
    s, e = _span_in_context(tk, text, [t.norm for t in tk], slot, "num", "2", "г")
    assert tk[s].start > text.index("вес")  # взято второе вхождение — то, что в контексте
