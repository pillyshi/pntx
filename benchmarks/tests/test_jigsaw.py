from __future__ import annotations

from datasets import Dataset

from benchmarks import jigsaw
from pntx.types import NEGATIVE, POSITIVE

_N_PER_LABEL = 20


def _make_dataset() -> Dataset:
    # 20 clearly non-toxic, 20 clearly toxic, 5 ambiguous (should be excluded).
    texts = (
        [f"clean-{i}" for i in range(_N_PER_LABEL)]
        + [f"toxic-{i}" for i in range(_N_PER_LABEL)]
        + [f"ambiguous-{i}" for i in range(5)]
    )
    toxicity = (
        [0.0] * _N_PER_LABEL
        + [1.0] * _N_PER_LABEL
        + [0.3] * 5
    )
    return Dataset.from_dict({"text": texts, "toxicity": toxicity})


def test_label_for_thresholds() -> None:
    assert jigsaw.label_for(0.0) == POSITIVE
    assert jigsaw.label_for(0.1) == POSITIVE
    assert jigsaw.label_for(0.5) == NEGATIVE
    assert jigsaw.label_for(1.0) == NEGATIVE
    assert jigsaw.label_for(0.3) is None


def test_sample_pools_draws_from_correct_labels() -> None:
    dataset = _make_dataset()
    positive, negative = jigsaw.sample_pools(dataset, seed=0, n_per_side=5)

    assert len(positive) == 5
    assert len(negative) == 5
    assert all(text.startswith("clean-") for text in positive)
    assert all(text.startswith("toxic-") for text in negative)


def test_sample_pools_is_deterministic() -> None:
    dataset = _make_dataset()
    first = jigsaw.sample_pools(dataset, seed=42, n_per_side=5)
    second = jigsaw.sample_pools(dataset, seed=42, n_per_side=5)
    assert first == second


def test_sample_eval_set_is_balanced_and_disjoint_from_pools() -> None:
    dataset = _make_dataset()
    n_per_side = 5
    n_eval = 10
    positive, negative = jigsaw.sample_pools(dataset, seed=0, n_per_side=n_per_side)
    eval_set = jigsaw.sample_eval_set(dataset, seed=0, n_per_side=n_per_side, n_eval=n_eval)

    assert len(eval_set) == n_eval
    labels = [label for _, label in eval_set]
    assert labels.count(POSITIVE) == n_eval // 2
    assert labels.count(NEGATIVE) == n_eval - n_eval // 2

    pool_texts = set(positive) | set(negative)
    eval_texts = {text for text, _ in eval_set}
    assert pool_texts.isdisjoint(eval_texts)


def test_sample_eval_set_never_returns_ambiguous_rows() -> None:
    dataset = _make_dataset()
    eval_set = jigsaw.sample_eval_set(dataset, seed=0, n_per_side=0, n_eval=10)
    for text, _ in eval_set:
        assert not text.startswith("ambiguous-")
