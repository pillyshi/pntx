"""Out-of-domain binary sentiment test sets for the downstream benchmark.

Kaushik et al. (2020) checked that counterfactually-augmented IMDb training
generalises to other review domains (Amazon, Yelp, Twitter). This module
provides balanced samples of two such domains so ``benchmarks/pn2t/downstream.py
evaluate --ood ...`` can do the same:

- ``yelp``: ``fancyzhx/yelp_polarity`` (Zhang et al. 2015). Its class names are
  ``"1"``/``"2"``; per that dataset's definition, polarity 1 is negative (1-2
  stars) and polarity 2 is positive (4-5 stars).
- ``amazon``: ``fancyzhx/amazon_polarity`` (Zhang et al. 2015), class names
  ``"negative"``/``"positive"``; the text is title and content joined.

The positive class is named explicitly per dataset and checked against the
dataset's own ``ClassLabel`` names at load time, so a relabelled upstream
dataset fails loudly instead of silently inverting the evaluation. The sample
is drawn with its own fixed seed, independent of the training seed, so every
run is compared on the same test items. Loading needs ``datasets`` (the
``benchmark`` dependency group) and network access to the Hugging Face Hub.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pntx.types import NEGATIVE, POSITIVE, Label


@dataclass(frozen=True)
class OODSpec:
    dataset: str
    class_names: tuple[str, str]
    """Expected ``ClassLabel`` names, in index order."""
    positive_name: str
    text: Callable[[dict[str, Any]], str]


SPECS: dict[str, OODSpec] = {
    "yelp": OODSpec(
        dataset="fancyzhx/yelp_polarity",
        class_names=("1", "2"),
        positive_name="2",
        text=lambda row: row["text"],
    ),
    "amazon": OODSpec(
        dataset="fancyzhx/amazon_polarity",
        class_names=("negative", "positive"),
        positive_name="positive",
        text=lambda row: f"{row['title']}. {row['content']}",
    ),
}


def label_mapper(spec: OODSpec, class_names: Sequence[str]) -> Callable[[int], Label]:
    """Map a dataset label index to pntx's Label, after checking the dataset's
    class names are exactly what ``spec`` expects."""
    if tuple(class_names) != spec.class_names:
        raise ValueError(
            f"{spec.dataset}: expected class names {spec.class_names}, got "
            f"{tuple(class_names)}; refusing to guess which one is positive"
        )
    positive_index = spec.class_names.index(spec.positive_name)
    return lambda index: POSITIVE if index == positive_index else NEGATIVE


def sample_balanced(
    texts: Sequence[str], labels: Sequence[Label], n: int, seed: int
) -> tuple[list[str], list[Label]]:
    """``n`` items, half of each label, drawn with ``seed`` and shuffled."""
    if n % 2:
        raise ValueError(f"n must be even for a balanced sample, got {n}")
    rng = random.Random(seed)
    picked: list[tuple[str, Label]] = []
    for label in (POSITIVE, NEGATIVE):
        pool = [t for t, y in zip(texts, labels, strict=True) if y == label]
        if len(pool) < n // 2:
            raise ValueError(f"only {len(pool)} {label} items, need {n // 2}")
        picked += [(t, label) for t in rng.sample(pool, n // 2)]
    rng.shuffle(picked)
    return [t for t, _ in picked], [y for _, y in picked]


def load_ood(
    name: str, n: int = 1000, seed: int = 0, cache_dir: str | None = None
) -> tuple[list[str], list[Label]]:
    """A balanced ``n``-item sample of the named domain's test split."""
    from datasets import load_dataset

    if name not in SPECS:
        raise ValueError(f"unknown OOD set {name!r}; choose from {sorted(SPECS)}")
    spec = SPECS[name]
    ds = load_dataset(spec.dataset, split="test", cache_dir=cache_dir)
    to_label = label_mapper(spec, ds.features["label"].names)
    texts = [spec.text(row) for row in ds]
    labels = [to_label(i) for i in ds["label"]]
    return sample_balanced(texts, labels, n, seed)
