from __future__ import annotations

import random
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from . import dedup

SimilarityFn = Callable[[str, str], float]


@runtime_checkable
class Selector(Protocol):
    """Chooses which fitted texts (from one side's pool) to embed in a prompt.

    ``pool`` is a single side's fitted texts (``PNTX.positive`` or
    ``PNTX.negative``); callers select from each side independently. ``query``
    is the text being classified, for selectors that pick texts relevant to
    it (e.g. ``NearestSelector``); selectors that don't use it (e.g.
    ``RandomSelector``) simply ignore the argument.
    """

    def select(self, pool: list[str], k: int, query: str | None = None) -> list[str]: ...


class RandomSelector:
    """Selects a uniform random subset of ``pool``, ignoring ``query``."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, pool: list[str], k: int, query: str | None = None) -> list[str]:
        if k >= len(pool):
            return list(pool)
        return self._rng.sample(pool, k)


class NearestSelector:
    """Selects the ``k`` texts in ``pool`` most similar to ``query``.

    ``similarity_fn`` defaults to ``dedup.similarity`` (dependency-free
    character n-grams); pass e.g. ``pntx.embeddings.cosine_similarity_fn()``
    for semantic similarity instead.

    ``query=None`` (e.g. when called from a batch context that has no single
    query to be "nearest" to) has nothing to rank by, so this falls back to
    the first ``k`` texts in ``pool`` order; use ``RandomSelector`` or
    ``DiversitySelector`` if that's not the fallback you want.
    """

    def __init__(self, similarity_fn: SimilarityFn = dedup.similarity) -> None:
        self.similarity_fn = similarity_fn

    def select(self, pool: list[str], k: int, query: str | None = None) -> list[str]:
        if k >= len(pool):
            return list(pool)
        if k <= 0:
            return []
        if query is None:
            return list(pool[:k])

        ranked = sorted(
            range(len(pool)), key=lambda i: self.similarity_fn(query, pool[i]), reverse=True
        )
        return [pool[i] for i in sorted(ranked[:k])]


class DiversitySelector:
    """Greedily selects ``k`` texts from ``pool`` that are maximally different
    from each other, ignoring ``query``.

    Starts from ``pool[0]`` and repeatedly adds whichever remaining text has
    the lowest similarity to its most-similar already-selected text (a
    farthest-point / greedy diversity heuristic).

    ``similarity_fn`` defaults to ``dedup.similarity`` (dependency-free
    character n-grams); pass e.g. ``pntx.embeddings.cosine_similarity_fn()``
    for semantic similarity instead.
    """

    def __init__(self, similarity_fn: SimilarityFn = dedup.similarity) -> None:
        self.similarity_fn = similarity_fn

    def select(self, pool: list[str], k: int, query: str | None = None) -> list[str]:
        if k >= len(pool):
            return list(pool)
        if k <= 0:
            return []

        selected = [0]
        while len(selected) < k:
            best_index = -1
            best_distance = -1.0
            for i in range(len(pool)):
                if i in selected:
                    continue
                similarity_to_selected = max(
                    self.similarity_fn(pool[i], pool[j]) for j in selected
                )
                distance = 1.0 - similarity_to_selected
                if distance > best_distance:
                    best_distance = distance
                    best_index = i
            selected.append(best_index)

        return [pool[i] for i in selected]
