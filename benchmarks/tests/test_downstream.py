from __future__ import annotations

import random

import pytest
from sklearn.metrics import accuracy_score

from benchmarks.cad import CADPair
from benchmarks.pn2t import downstream
from pntx.types import NEGATIVE, POSITIVE


def _pairs() -> list[CADPair]:
    pairs = [CADPair(f"p{i}", f"pos text {i}", f"neg rev {i}", POSITIVE) for i in range(5)]
    pairs += [CADPair(f"n{i}", f"neg text {i}", f"pos rev {i}", NEGATIVE) for i in range(8)]
    pairs.append(CADPair("long", "x" * 2000, "y", NEGATIVE))
    return pairs


def test_sample_training_pairs_respects_labels_counts_and_length() -> None:
    pos, neg = downstream.sample_training_pairs(
        _pairs(), n_pos=2, n_neg=6, max_chars=100, rng=random.Random(0)
    )
    assert len(pos) == 2 and all(p.original_label == POSITIVE for p in pos)
    assert len(neg) == 6 and all(p.original_label == NEGATIVE for p in neg)
    assert all(len(p.original) <= 100 for p in pos + neg)


def test_sample_training_pairs_raises_when_too_few() -> None:
    with pytest.raises(ValueError, match="available"):
        downstream.sample_training_pairs(
            _pairs(), n_pos=6, n_neg=1, max_chars=100, rng=random.Random(0)
        )


def test_duplicate_and_human_cad_positives() -> None:
    dup = downstream.duplicate_positives(["a", "b"], 5, random.Random(0))
    assert len(dup) == 5 and set(dup) <= {"a", "b"}
    _, neg = downstream.sample_training_pairs(
        _pairs(), n_pos=1, n_neg=4, max_chars=100, rng=random.Random(0)
    )
    revs = downstream.human_cad_positives(neg, 3)
    assert revs == [p.revised for p in neg[:3]]
    with pytest.raises(ValueError):
        downstream.human_cad_positives(neg, 5)


def test_equalize_truncates_to_smallest_condition() -> None:
    augs, n, empty = downstream.equalize(
        {"a": ["1", "2", "3"], "b": ["1", "2"], "c": ["1", "2", "3"]}
    )
    assert n == 2
    assert all(len(v) == 2 for v in augs.values())
    assert empty == []


def test_equalize_drops_empty_conditions_instead_of_zeroing_everything() -> None:
    # Regression: a sampler that produced nothing used to truncate all others to 0.
    augs, n, empty = downstream.equalize({"a": ["1", "2"], "b": [], "c": ["1", "2", "3"]})
    assert empty == ["b"]
    assert n == 2
    assert set(augs) == {"a", "c"}


def test_bootstrap_ci_brackets_point_estimate() -> None:
    y_true = [POSITIVE, NEGATIVE] * 50
    y_pred = [POSITIVE, NEGATIVE] * 40 + [NEGATIVE, POSITIVE] * 10
    point, low, high = downstream.bootstrap_ci(y_true, y_pred, accuracy_score, seed=0)
    assert point == pytest.approx(0.8)
    assert low <= point <= high
    assert 0.6 < low < 0.8 < high < 1.0


def test_format_table() -> None:
    row = {
        "condition": "original",
        "n_train": 150,
        **{
            c: [0.5, 0.4, 0.6]
            for c in (
                "original_test_acc",
                "original_test_f1",
                "revised_test_acc",
                "revised_test_f1",
            )
        },
    }
    table = downstream.format_table([row])
    assert "| original | 150 | 0.500 [0.400, 0.600]" in table
    assert table.count(" - ") == 4  # auc/pred_pos absent in this row -> "-"
    row2 = {**row, "original_test_pred_pos": 0.25}
    assert "| 0.25 |" in downstream.format_table([row2])
