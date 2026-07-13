from __future__ import annotations

import random
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from . import dedup
from .types import Pair

SimilarityFn = Callable[[str, str], float]


@runtime_checkable
class Selector(Protocol):
    """Chooses which fitted pairs to embed in a prompt.

    ``query`` is the text being classified, for selectors that pick pairs
    relevant to it (e.g. ``NearestSelector``); selectors that don't use it
    (e.g. ``RandomSelector``) simply ignore the argument.
    """

    def select(self, pairs: list[Pair], k: int, query: str | None = None) -> list[Pair]: ...


class RandomSelector:
    """Selects a uniform random subset of ``pairs``, ignoring ``query``."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, pairs: list[Pair], k: int, query: str | None = None) -> list[Pair]:
        if k >= len(pairs):
            return list(pairs)
        return self._rng.sample(pairs, k)


class NearestSelector:
    """Selects the ``k`` pairs whose positive or negative text is most
    similar to ``query``.

    ``similarity_fn`` defaults to ``dedup.similarity`` (dependency-free
    character n-grams); pass e.g. ``pntx.embeddings.cosine_similarity_fn()``
    for semantic similarity instead.

    ``query=None`` (e.g. when called from a batch context that has no single
    query to be "nearest" to) has nothing to rank by, so this falls back to
    the first ``k`` pairs in ``pairs`` order; use ``RandomSelector`` or
    ``DiversitySelector`` if that's not the fallback you want.
    """

    def __init__(self, similarity_fn: SimilarityFn = dedup.similarity) -> None:
        self.similarity_fn = similarity_fn

    def select(self, pairs: list[Pair], k: int, query: str | None = None) -> list[Pair]:
        if k >= len(pairs):
            return list(pairs)
        if k <= 0:
            return []
        if query is None:
            return list(pairs[:k])

        def pair_similarity(pair: Pair) -> float:
            pos, neg = pair
            return max(self.similarity_fn(query, pos), self.similarity_fn(query, neg))

        ranked = sorted(range(len(pairs)), key=lambda i: pair_similarity(pairs[i]), reverse=True)
        return [pairs[i] for i in sorted(ranked[:k])]


class DiversitySelector:
    """Greedily selects ``k`` pairs that are maximally different from each
    other, ignoring ``query``.

    Starts from ``pairs[0]`` and repeatedly adds whichever remaining pair
    has the lowest similarity to its most-similar already-selected pair
    (a farthest-point / greedy diversity heuristic), representing each pair
    as its concatenated positive and negative text.

    ``similarity_fn`` defaults to ``dedup.similarity`` (dependency-free
    character n-grams); pass e.g. ``pntx.embeddings.cosine_similarity_fn()``
    for semantic similarity instead.
    """

    def __init__(self, similarity_fn: SimilarityFn = dedup.similarity) -> None:
        self.similarity_fn = similarity_fn

    def select(self, pairs: list[Pair], k: int, query: str | None = None) -> list[Pair]:
        if k >= len(pairs):
            return list(pairs)
        if k <= 0:
            return []

        representations = [f"{pos} {neg}" for pos, neg in pairs]
        selected = [0]
        while len(selected) < k:
            best_index = -1
            best_distance = -1.0
            for i in range(len(pairs)):
                if i in selected:
                    continue
                similarity_to_selected = max(
                    self.similarity_fn(representations[i], representations[j]) for j in selected
                )
                distance = 1.0 - similarity_to_selected
                if distance > best_distance:
                    best_distance = distance
                    best_index = i
            selected.append(best_index)

        return [pairs[i] for i in selected]
