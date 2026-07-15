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


def test_sample_pairs_draws_from_correct_pools() -> None:
    dataset = _make_dataset()
    pairs = jigsaw.sample_pairs(dataset, seed=0, n_pairs=5)

    assert len(pairs) == 5
    for positive_text, negative_text in pairs:
        assert positive_text.startswith("clean-")
        assert negative_text.startswith("toxic-")


def test_sample_pairs_is_deterministic() -> None:
    dataset = _make_dataset()
    first = jigsaw.sample_pairs(dataset, seed=42, n_pairs=5)
    second = jigsaw.sample_pairs(dataset, seed=42, n_pairs=5)
    assert first == second


def test_sample_eval_set_is_balanced_and_disjoint_from_pairs() -> None:
    dataset = _make_dataset()
    n_pairs = 5
    n_eval = 10
    pairs = jigsaw.sample_pairs(dataset, seed=0, n_pairs=n_pairs)
    eval_set = jigsaw.sample_eval_set(dataset, seed=0, n_pairs_used=n_pairs, n_eval=n_eval)

    assert len(eval_set) == n_eval
    labels = [label for _, label in eval_set]
    assert labels.count(POSITIVE) == n_eval // 2
    assert labels.count(NEGATIVE) == n_eval - n_eval // 2

    pair_texts = {text for pair in pairs for text in pair}
    eval_texts = {text for text, _ in eval_set}
    assert pair_texts.isdisjoint(eval_texts)


def test_sample_eval_set_never_returns_ambiguous_rows() -> None:
    dataset = _make_dataset()
    eval_set = jigsaw.sample_eval_set(dataset, seed=0, n_pairs_used=0, n_eval=10)
    for text, _ in eval_set:
        assert not text.startswith("ambiguous-")
