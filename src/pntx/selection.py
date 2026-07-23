from __future__ import annotations

import random
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from . import dedup

if TYPE_CHECKING:
    from numpy.typing import NDArray

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
    (``sample_texts_within_budget``), with one deviation: ``pool`` is
    shuffled, then texts are scanned in that order and added whenever they
    still fit under the running token total; a candidate that would push the
    total over ``token_budget`` is skipped (not selected), and scanning
    continues with the next one, so a single oversized candidate can't stop
    shorter ones later in the shuffle from being picked up.

    Unlike the other selectors, this one does *not* short-circuit to
    ``list(pool)`` when ``k >= len(pool)``: the budget is the real
    constraint here, and the whole point is to keep prompts within it even
    when every text in the pool would otherwise be included. ``k`` still
    acts as a secondary cap on top of the budget (e.g. from
    ``t2pn.Classifier(max_exemplars=...)``); ``query`` is ignored.

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
                continue
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


_SAMPLE_METHODS: tuple[str, ...] = ("random", "kmeans", "votek")


def sample_texts_kmeans(
    texts: list[str], n: int, embeddings: NDArray[np.float64], rng: random.Random | None = None
) -> list[str]:
    """Selects up to ``n`` texts by picking the closest text to each K-Means
    cluster centroid.

    Ported from ``semaxis``'s ``sampling.sample_texts_kmeans``. ``embeddings``
    is a pre-computed matrix of shape ``(len(texts), dim)``, aligned by index
    with ``texts``.
    """
    from sklearn.cluster import KMeans

    n = min(n, len(texts))
    if n == 0:
        return []
    if n == len(texts):
        return list(texts)

    seed = rng.randint(0, 2**31 - 1) if rng is not None else None
    km = KMeans(n_clusters=n, random_state=seed, n_init="auto")
    km.fit(embeddings)

    selected_indices: list[int] = []
    for center in km.cluster_centers_:
        dists = np.linalg.norm(embeddings - center, axis=1)
        # Exclude already-selected indices to avoid duplicates when clusters share a medoid
        dists[selected_indices] = np.inf
        selected_indices.append(int(np.argmin(dists)))

    return [texts[i] for i in selected_indices]


def sample_texts_votek(
    texts: list[str],
    n: int,
    embeddings: NDArray[np.float64],
    k: int = 10,
    rng: random.Random | None = None,
) -> list[str]:
    """Selects up to ``n`` texts using the Vote-K algorithm (Su et al. 2022).

    Ported from ``semaxis``'s ``sampling.sample_texts_votek``. Vote-K balances
    representativeness (high vote count = many neighbours) and diversity
    (selected texts' neighbours are suppressed from future picks). ``rng`` is
    accepted for API symmetry with ``sample_texts_kmeans`` but unused --
    Vote-K's selection is deterministic given ``embeddings``.
    """
    from sklearn.metrics.pairwise import cosine_similarity

    n = min(n, len(texts))
    if n == 0:
        return []
    if n == len(texts):
        return list(texts)

    k = min(k, len(texts) - 1)
    sim = cosine_similarity(embeddings)  # (N, N)

    # votes[i] = how many texts consider i among their top-k neighbours
    votes = np.zeros(len(texts), dtype=float)
    for i in range(len(texts)):
        sim_row = sim[i].copy()
        sim_row[i] = -np.inf  # exclude self
        top_k = np.argpartition(sim_row, -k)[-k:]
        votes[top_k] += 1.0

    selected_indices: list[int] = []
    remaining = np.ones(len(texts), dtype=bool)

    while len(selected_indices) < n and remaining.any():
        # Among remaining texts, pick the one with the most votes
        masked_votes = np.where(remaining, votes, -np.inf)
        best = int(np.argmax(masked_votes))
        selected_indices.append(best)
        remaining[best] = False

        # Suppress votes of k nearest neighbours
        sim_row = sim[best].copy()
        sim_row[best] = -np.inf
        top_k = np.argpartition(sim_row, -k)[-k:]
        votes[top_k] = 0.0

    return [texts[i] for i in selected_indices]


def _estimate_n(texts: list[str], token_budget: int, tokenizer_fn: Callable[[str], int]) -> int:
    """Estimates how many texts fit in ``token_budget`` based on average token length."""
    if not texts:
        return 0
    probe = texts[: min(20, len(texts))]
    avg = sum(tokenizer_fn(t) for t in probe) / len(probe)
    return max(1, int(token_budget / avg))


def _trim_to_budget(
    texts: list[str], token_budget: int, tokenizer_fn: Callable[[str], int]
) -> list[str]:
    """Trims a list of texts to fit within the token budget, preserving order.

    Skips (rather than stops at) any text that alone would push the running
    total over ``token_budget``, so a single oversized text can't crowd out
    shorter ones later in ``texts``.
    """
    result: list[str] = []
    total = 0
    for text in texts:
        count = tokenizer_fn(text)
        if total + count > token_budget:
            continue
        result.append(text)
        total += count
    return result


def sample_group(
    texts: list[str],
    budget: int,
    tokenizer_fn: Callable[[str], int],
    method: str,
    embedding_model: str,
    rng: random.Random,
) -> list[str]:
    """Dispatches to a budget-constrained sampling strategy by name.

    Ported from ``semaxis``'s ``supervised._sample_group``. ``method`` must be
    one of ``_SAMPLE_METHODS`` (the caller is expected to validate this, as
    semaxis's own callers do). ``"random"`` delegates to ``BudgetSelector``
    (no embeddings needed); ``"kmeans"``/``"votek"`` embed ``texts`` via
    ``pntx.embeddings.embed`` (requires the ``pntx[embeddings]`` extra --
    imported lazily here so importing this module doesn't require it) and
    cluster/vote over the resulting vectors, then trim the result back to
    ``budget`` as a final guarantee (the embedding-based methods only
    estimate how many texts fit).
    """
    if method == "random":
        return BudgetSelector(
            tokenizer_fn=tokenizer_fn, token_budget=budget, seed=rng.randint(0, 2**31 - 1)
        ).select(texts, k=len(texts))

    from . import embeddings

    vectors = np.array(embeddings.embed(texts, embedding_model))
    n = _estimate_n(texts, budget, tokenizer_fn)
    if method == "kmeans":
        sampled = sample_texts_kmeans(texts, n, vectors, rng)
    else:
        sampled = sample_texts_votek(texts, n, vectors, rng=rng)
    return _trim_to_budget(sampled, budget, tokenizer_fn)
