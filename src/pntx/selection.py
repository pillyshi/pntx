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


class BudgetSelector:
    """Selects as many texts from ``pool`` as fit within a token budget.

    Ported from ``semaxis``'s ``HardPositiveOverSampler`` sampling strategy
    (``sample_texts_within_budget``): ``pool`` is shuffled, then texts are
    added one by one until the next one would push the running token total
    over ``token_budget`` -- at which point selection stops (remaining,
    possibly-shorter texts are *not* tried, matching the ported behavior).

    Unlike the other selectors, this one does *not* short-circuit to
    ``list(pool)`` when ``k >= len(pool)``: the budget is the real
    constraint here, and the whole point is to keep prompts within it even
    when every text in the pool would otherwise be included. ``k`` still
    acts as a secondary cap on top of the budget (e.g. from
    ``PNTX(max_exemplars=...)``); ``query`` is ignored.

    ``tokenizer_fn`` should match whatever backend is doing the actual
    tokenizing (e.g. ``LlamaCppBackend.count_tokens``), so the budget
    reflects real token counts rather than an approximation.
    """

    def __init__(
        self, tokenizer_fn: Callable[[str], int], token_budget: int, seed: int | None = None
    ) -> None:
        if token_budget <= 0:
            raise ValueError(f"token_budget must be > 0, got {token_budget}")
        self.tokenizer_fn = tokenizer_fn
        self.token_budget = token_budget
        self._rng = random.Random(seed)

    def select(self, pool: list[str], k: int, query: str | None = None) -> list[str]:
        if k <= 0:
            return []

        indices = list(range(len(pool)))
        self._rng.shuffle(indices)

        selected: list[str] = []
        total_tokens = 0
        for idx in indices:
            if len(selected) >= k:
                break
            text = pool[idx]
            count = self.tokenizer_fn(text)
            if total_tokens + count > self.token_budget:
                break
            selected.append(text)
            total_tokens += count
        return selected


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
