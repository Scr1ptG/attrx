"""Политика статусов и порог авто-публикации: обещание «точность accept ≥ 0.95» должно выполняться честно."""

import pytest

pl = pytest.importorskip("polars")
np = pytest.importorskip("numpy")
pytest.importorskip("sklearn")

from attrx import metrics as M  # noqa: E402
from attrx.train import assign_status  # noqa: E402
from attrx.verifier import Verifier  # noqa: E402


def test_threshold_respects_ties_in_calibrated_scores():
    ver = Verifier()
    ver.oof_ = np.array([0.9] * 10 + [0.5] * 10)  # изотоническая калибровка даёт ступени
    y = np.array([1] * 14 + [0] * 6)  # ступень 0.9: 10 из 10 верны; ступень 0.5: 4 из 10
    # порог внутри ступени 0.5 принял бы всю ступень: точность 0.70 вместо обещанных 0.95
    assert ver.threshold_for(y, 0.95, min_support=5) == 0.9


def test_threshold_disables_auto_publication_when_target_is_unreachable():
    ver = Verifier()
    ver.oof_ = np.array([0.6] * 40)
    with pytest.warns(UserWarning):
        assert ver.threshold_for(np.array([1, 0] * 20), 0.95) == 1.0


def test_threshold_counts_only_candidates_that_policy_can_publish():
    ver = Verifier()
    ver.oof_ = np.array([0.9] * 40 + [0.95] * 40)
    y = np.array([1] * 40 + [0] * 40)  # самые уверенные — значения аналогов, которые всё равно не публикуются
    eligible = np.array([True] * 40 + [False] * 40)
    assert ver.threshold_for(y, 0.95, eligible=eligible) == 0.9


def test_status_policy_requires_text_evidence():
    frame = pl.DataFrame({"source": ["lexicon", "analog", "similar", "llm", "lexicon+llm", "analog+pattern"],
                          "confidence": [0.97] * 6})
    assert assign_status(frame, accept=0.9)["status"].to_list() == ["accept", "review", "review", "review", "accept", "accept"]
    assert assign_status(frame, accept=0.9, text_required=False)["status"].to_list() == ["accept"] * 6
    assert assign_status(frame.with_columns(pl.lit(0.4).alias("confidence")), accept=0.9)["status"].unique().to_list() == ["reject"]


def test_summarize_takes_accept_metrics_from_status():
    pairs = pl.DataFrame({"gold": ["a", "b", "c"], "pred": ["a", "x", "c"], "correct": [True, False, True],
                          "confidence": [0.99, 0.99, 0.6], "status": ["accept", "review", "review"]})
    s = M.summarize(pairs).row(0, named=True)
    assert s["precision_accept"] == 1.0 and abs(s["auto_fill"] - 1 / 3) < 1e-9
    no_status = M.summarize(pairs.drop("status")).row(0, named=True)  # сырые источники: фиксированный порог 0.8
    assert no_status["precision_accept"] == 0.5
