"""Load, label, and sample the Jigsaw/Civil Comments dataset for pntx benchmarks.

Source: https://huggingface.co/datasets/google/civil_comments (CC0). The
``toxicity`` field is a continuous [0, 1] fraction of raters who flagged a
comment as toxic; we threshold it into pntx's binary ``Label``. By
convention, non-toxic text maps to ``"positive"`` and toxic text maps to
``"negative"`` -- this mapping is otherwise arbitrary (pntx itself treats
positive/negative as opaque), so it is fixed here rather than left to each
caller.

Shared by the t2pn (classification) benchmark; the sampling functions are
generic enough to reuse for a later pn2t (generation) benchmark, so keep
this module free of anything t2pn-specific.
"""

from __future__ import annotations

import random

from datasets import Dataset, concatenate_datasets
from datasets import load_dataset as _load_dataset

from pntx.types import NEGATIVE, POSITIVE, Label, Pair

DATASET_NAME = "google/civil_comments"

DEFAULT_CLEAN_THRESHOLD = 0.1
DEFAULT_TOXIC_THRESHOLD = 0.5


def load_dataset(cache_dir: str | None = None) -> Dataset:
    """Load every split of ``google/civil_comments``, concatenated into one Dataset.

    Splits are merged because exemplar-vs-eval sampling is done here (by
    shuffling with an explicit seed), not by relying on the dataset's own
    train/test split.
    """
    raw = _load_dataset(DATASET_NAME, cache_dir=cache_dir)
    return concatenate_datasets([raw[split] for split in raw])


def label_for(
    toxicity: float,
    *,
    clean_threshold: float = DEFAULT_CLEAN_THRESHOLD,
    toxic_threshold: float = DEFAULT_TOXIC_THRESHOLD,
) -> Label | None:
    """Map a continuous toxicity score to a pntx ``Label``, or ``None`` if ambiguous.

    Scores strictly between the two thresholds are dropped rather than
    forced into a label, so the resulting ground truth is less noisy.
    """
    if toxicity <= clean_threshold:
        return POSITIVE
    if toxicity >= toxic_threshold:
        return NEGATIVE
    return None


def sample_pairs(
    dataset: Dataset,
    seed: int,
    n_pairs: int,
    *,
    clean_threshold: float = DEFAULT_CLEAN_THRESHOLD,
    toxic_threshold: float = DEFAULT_TOXIC_THRESHOLD,
) -> list[Pair]:
    """Sample ``n_pairs`` (positive, negative) pairs for use as ``PNTX.fit`` exemplars.

    Each side is sampled independently and zipped together -- these are not
    paraphrase pairs of the same underlying comment, just unrelated examples
    of each label (Civil Comments doesn't provide toxic/non-toxic rewrites
    of the same text). Pass the same ``seed`` to ``sample_eval_set`` (with a
    matching ``n_pairs_used``) to draw its rows from the same shuffled
    ordering without overlapping these.
    """
    positive_texts = _shuffled_label_texts(
        dataset, POSITIVE, seed, clean_threshold, toxic_threshold
    )
    negative_texts = _shuffled_label_texts(
        dataset, NEGATIVE, seed, clean_threshold, toxic_threshold
    )
    return list(zip(positive_texts[:n_pairs], negative_texts[:n_pairs], strict=True))


def sample_eval_set(
    dataset: Dataset,
    seed: int,
    n_pairs_used: int,
    n_eval: int,
    *,
    clean_threshold: float = DEFAULT_CLEAN_THRESHOLD,
    toxic_threshold: float = DEFAULT_TOXIC_THRESHOLD,
) -> list[tuple[str, Label]]:
    """Sample ``n_eval`` held-out ``(text, label)`` rows, balanced between labels.

    ``seed`` and ``n_pairs_used`` must match the call to ``sample_pairs``
    this is paired with, so this draws from the same per-label shuffled
    ordering starting right after the rows ``sample_pairs`` already used
    (no overlap between exemplars and eval set).
    """
    positive_texts = _shuffled_label_texts(
        dataset, POSITIVE, seed, clean_threshold, toxic_threshold
    )
    negative_texts = _shuffled_label_texts(
        dataset, NEGATIVE, seed, clean_threshold, toxic_threshold
    )
    n_positive = n_eval // 2
    n_negative = n_eval - n_positive
    eval_set: list[tuple[str, Label]] = [
        (text, POSITIVE)
        for text in positive_texts[n_pairs_used : n_pairs_used + n_positive]
    ] + [
        (text, NEGATIVE)
        for text in negative_texts[n_pairs_used : n_pairs_used + n_negative]
    ]
    random.Random(seed).shuffle(eval_set)
    return eval_set


def _shuffled_label_texts(
    dataset: Dataset,
    label: Label,
    seed: int,
    clean_threshold: float,
    toxic_threshold: float,
) -> list[str]:
    def matches_label(batch: dict[str, list[float]]) -> list[bool]:
        return [
            label_for(t, clean_threshold=clean_threshold, toxic_threshold=toxic_threshold) == label
            for t in batch["toxicity"]
        ]

    subset = dataset.filter(matches_label, batched=True).shuffle(seed=seed)
    return list(subset["text"])
