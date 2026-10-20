"""Unit tests for the evaluation metrics, with hand-computable fixtures.

The golden-set numbers are only worth publishing if the metric code is
provably right. These fixtures are small enough to verify by hand:

  results: ranks [1, 2, None], precisions [0.2, 0.4, 0.0]
  hit@1 = 1/3, hit@3 = 2/3, hit@5 = 2/3
  MRR   = (1/1 + 1/2 + 0) / 3 = 0.5
  citation precision = (0.2 + 0.4 + 0.0) / 3 = 0.2
"""

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src import eval as eval_module  # noqa: E402


def _ranked(*doc_ids):
    return [({"doc_id": doc_id}, 0.0) for doc_id in doc_ids]


def test_rank_of_expected_finds_first_match():
    ranked = _ranked("a", "b", "a", "c")
    assert eval_module.rank_of_expected(ranked, ["b"]) == 2
    assert eval_module.rank_of_expected(ranked, ["a"]) == 1
    assert eval_module.rank_of_expected(ranked, ["b", "c"]) == 2


def test_rank_of_expected_returns_none_on_miss():
    assert eval_module.rank_of_expected(_ranked("a", "b"), ["zzz"]) is None
    assert eval_module.rank_of_expected([], ["a"]) is None


def test_summarize_matches_hand_computed_values():
    results = [
        {"rank": 1, "precision": 0.2, "retrieval_ms": 100},
        {"rank": 2, "precision": 0.4, "retrieval_ms": 300},
        {"rank": None, "precision": 0.0, "retrieval_ms": 200},
    ]
    summary = eval_module.summarize(results)
    assert summary["n"] == 3
    assert summary["hit@1"] == pytest.approx(1 / 3)
    assert summary["hit@3"] == pytest.approx(2 / 3)
    assert summary["hit@5"] == pytest.approx(2 / 3)
    assert summary["mrr"] == pytest.approx(0.5)
    assert summary["citation_precision"] == pytest.approx(0.2)
    assert summary["mean_retrieval_ms"] == 200
    assert summary["p50_retrieval_ms"] == 200
