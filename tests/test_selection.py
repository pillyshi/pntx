from __future__ import annotations

import random
import sys
import types

import numpy as np
import pytest

from pntx.selection import (
    BudgetSelector,
    DiversitySelector,
    NearestSelector,
    RandomSelector,
    _trim_to_budget,
    sample_group,
    sample_texts_kmeans,
    sample_texts_votek,
)

from .conftest import SAMPLE_POSITIVE

_TOPIC_POOL: list[str] = [
    "apple pie recipe",
    "great customer service",
    "mountain hiking trip",
]


def _fake_embeddings_module(embed_fn: object) -> types.ModuleType:
    """Stands in for ``pntx.embeddings`` in tests so ``sample_group``'s
    ``"kmeans"``/``"votek"`` branches don't require sentence-transformers."""
    module = types.ModuleType("pntx.embeddings")
    module.embed = embed_fn  # type: ignore[attr-defined]
    return module


def test_select_fewer_than_available_returns_k_distinct_texts() -> None:
    selector = RandomSelector(seed=0)
    selected = selector.select(SAMPLE_POSITIVE, k=2)
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert all(text in SAMPLE_POSITIVE for text in selected)


def test_select_k_greater_than_available_returns_all_texts() -> None:
    selector = RandomSelector(seed=0)
    selected = selector.select(SAMPLE_POSITIVE, k=100)
    assert set(selected) == set(SAMPLE_POSITIVE)


def test_select_k_zero_returns_empty() -> None:
    selector = RandomSelector(seed=0)
    assert selector.select(SAMPLE_POSITIVE, k=0) == []


def test_select_is_reproducible_with_same_seed() -> None:
    a = RandomSelector(seed=42).select(SAMPLE_POSITIVE, k=2)
    b = RandomSelector(seed=42).select(SAMPLE_POSITIVE, k=2)
    assert a == b


def test_select_ignores_query() -> None:
    selector = RandomSelector(seed=0)
    without_query = selector.select(SAMPLE_POSITIVE, k=100)
    with_query = selector.select(SAMPLE_POSITIVE, k=100, query="anything")
    assert set(without_query) == set(with_query)


def test_nearest_selector_picks_most_similar_text_to_query() -> None:
    selector = NearestSelector()
    selected = selector.select(_TOPIC_POOL, k=1, query="apple pie recipe was amazing")
    assert selected == [_TOPIC_POOL[0]]


def test_nearest_selector_picks_most_similar_text_japanese() -> None:
    selector = NearestSelector()
    selected = selector.select(SAMPLE_POSITIVE, k=1, query="この映画は最高だった!")
    assert selected == [SAMPLE_POSITIVE[0]]


def test_nearest_selector_k_zero_returns_empty() -> None:
    selector = NearestSelector()
    assert selector.select(_TOPIC_POOL, k=0, query="anything") == []


def test_nearest_selector_k_greater_than_available_returns_all() -> None:
    selector = NearestSelector()
    assert set(selector.select(_TOPIC_POOL, k=100, query="apple")) == set(_TOPIC_POOL)


def test_nearest_selector_query_none_falls_back_to_prefix() -> None:
    selector = NearestSelector()
    assert selector.select(_TOPIC_POOL, k=2, query=None) == _TOPIC_POOL[:2]


def test_nearest_selector_uses_custom_similarity_fn() -> None:
    calls: list[tuple[str, str]] = []

    def fake_similarity(a: str, b: str) -> float:
        calls.append((a, b))
        return 1.0 if "target" in b else 0.0

    pool = ["target text", "other1", "other2"]
    selector = NearestSelector(similarity_fn=fake_similarity)

    selected = selector.select(pool, k=1, query="query")

    assert selected == [pool[0]]
    assert calls


def test_budget_selector_stops_when_budget_is_exhausted() -> None:
    pool = ["a", "b", "c", "d", "e"]
    selector = BudgetSelector(tokenizer_fn=lambda t: 10, token_budget=25, seed=0)

    selected = selector.select(pool, k=100)

    # budget(25) // per-item cost(10) == 2 items fit; a 3rd would be 30 > 25.
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert all(text in pool for text in selected)


def test_budget_selector_skips_oversized_candidate_instead_of_stopping() -> None:
    pool = ["huge", "a", "b", "c", "d", "e"]
    costs = {"huge": 100, "a": 1, "b": 1, "c": 1, "d": 1, "e": 1}
    # Regardless of shuffle order, "huge" alone blows the budget and must be
    # skipped rather than aborting the whole scan -- every cost-1 text still
    # fits and should all be selected.
    for seed in range(10):
        selector = BudgetSelector(tokenizer_fn=lambda t: costs[t], token_budget=10, seed=seed)
        selected = selector.select(pool, k=100)
        assert "huge" not in selected
        assert set(selected) == {"a", "b", "c", "d", "e"}


def test_budget_selector_does_not_shortcut_when_k_covers_whole_pool() -> None:
    pool = ["a", "b", "c", "d", "e"]
    selector = BudgetSelector(tokenizer_fn=lambda t: 10, token_budget=25, seed=0)

    # Unlike RandomSelector/NearestSelector/DiversitySelector, k >= len(pool)
    # must not bypass the budget check.
    selected = selector.select(pool, k=len(pool))
    assert len(selected) == 2


def test_budget_selector_respects_k_cap_even_when_budget_allows_more() -> None:
    pool = ["a", "b", "c", "d", "e"]
    selector = BudgetSelector(tokenizer_fn=lambda t: 1, token_budget=1000, seed=0)
    assert len(selector.select(pool, k=2)) == 2


def test_budget_selector_k_zero_returns_empty() -> None:
    selector = BudgetSelector(tokenizer_fn=lambda t: 1, token_budget=10, seed=0)
    assert selector.select(["a", "b"], k=0) == []


def test_budget_selector_empty_pool_returns_empty() -> None:
    selector = BudgetSelector(tokenizer_fn=lambda t: 1, token_budget=10, seed=0)
    assert selector.select([], k=5) == []


def test_budget_selector_is_reproducible_with_same_seed() -> None:
    pool = ["a", "b", "c", "d", "e"]
    a = BudgetSelector(tokenizer_fn=lambda t: 10, token_budget=25, seed=42).select(pool, k=100)
    b = BudgetSelector(tokenizer_fn=lambda t: 10, token_budget=25, seed=42).select(pool, k=100)
    assert a == b


def test_budget_selector_ignores_query() -> None:
    pool = ["a", "b", "c"]
    without_query = BudgetSelector(tokenizer_fn=lambda t: 1, token_budget=100, seed=0).select(
        pool, k=100
    )
    with_query = BudgetSelector(tokenizer_fn=lambda t: 1, token_budget=100, seed=0).select(
        pool, k=100, query="anything"
    )
    assert without_query == with_query


def test_budget_selector_rejects_non_positive_token_budget() -> None:
    with pytest.raises(ValueError, match="token_budget must be > 0"):
        BudgetSelector(tokenizer_fn=lambda t: 1, token_budget=0)


def test_diversity_selector_k_greater_than_available_returns_all() -> None:
    selector = DiversitySelector()
    assert set(selector.select(_TOPIC_POOL, k=100)) == set(_TOPIC_POOL)


def test_diversity_selector_k_zero_returns_empty() -> None:
    selector = DiversitySelector()
    assert selector.select(_TOPIC_POOL, k=0) == []


def test_diversity_selector_returns_k_distinct_texts() -> None:
    selector = DiversitySelector()
    selected = selector.select(_TOPIC_POOL, k=2)
    assert len(selected) == 2
    assert len(set(selected)) == 2


def test_diversity_selector_prefers_the_most_different_text_next() -> None:
    # near_duplicate is a near-duplicate of _TOPIC_POOL[0]; a genuinely
    # diverse selector picking a second text should prefer the unrelated
    # pool[2] over the near-duplicate.
    near_duplicate = "apple pie recipe!!"
    pool = [_TOPIC_POOL[0], near_duplicate, _TOPIC_POOL[2]]
    selector = DiversitySelector()

    selected = selector.select(pool, k=2)

    assert selected[0] == pool[0]
    assert selected[1] == _TOPIC_POOL[2]


def test_diversity_selector_ignores_query() -> None:
    selector = DiversitySelector()
    without_query = selector.select(_TOPIC_POOL, k=2)
    with_query = selector.select(_TOPIC_POOL, k=2, query="irrelevant")
    assert without_query == with_query


_TWO_CLUSTER_TEXTS = ["a1", "a2", "a3", "b1", "b2", "b3"]
_TWO_CLUSTER_EMBEDDINGS = np.array(
    [
        [0.0, 0.0],
        [0.1, 0.0],
        [0.0, 0.1],
        [10.0, 10.0],
        [10.1, 10.0],
        [10.0, 10.1],
    ]
)


def test_sample_texts_kmeans_picks_one_text_per_cluster() -> None:
    selected = sample_texts_kmeans(
        _TWO_CLUSTER_TEXTS, 2, _TWO_CLUSTER_EMBEDDINGS, rng=random.Random(0)
    )
    assert len(selected) == 2
    assert any(t in _TWO_CLUSTER_TEXTS[:3] for t in selected)
    assert any(t in _TWO_CLUSTER_TEXTS[3:] for t in selected)


def test_sample_texts_kmeans_n_zero_returns_empty() -> None:
    assert sample_texts_kmeans(_TWO_CLUSTER_TEXTS, 0, _TWO_CLUSTER_EMBEDDINGS) == []


def test_sample_texts_kmeans_n_covers_whole_pool_returns_all() -> None:
    selected = sample_texts_kmeans(_TWO_CLUSTER_TEXTS, 100, _TWO_CLUSTER_EMBEDDINGS)
    assert selected == _TWO_CLUSTER_TEXTS


def test_sample_texts_votek_picks_one_text_per_cluster() -> None:
    selected = sample_texts_votek(
        _TWO_CLUSTER_TEXTS, 2, _TWO_CLUSTER_EMBEDDINGS, k=2, rng=random.Random(0)
    )
    assert len(selected) == 2
    assert any(t in _TWO_CLUSTER_TEXTS[:3] for t in selected)
    assert any(t in _TWO_CLUSTER_TEXTS[3:] for t in selected)


def test_sample_texts_votek_n_zero_returns_empty() -> None:
    assert sample_texts_votek(_TWO_CLUSTER_TEXTS, 0, _TWO_CLUSTER_EMBEDDINGS) == []


def test_sample_texts_votek_n_covers_whole_pool_returns_all() -> None:
    selected = sample_texts_votek(_TWO_CLUSTER_TEXTS, 100, _TWO_CLUSTER_EMBEDDINGS)
    assert selected == _TWO_CLUSTER_TEXTS


def test_sample_group_random_respects_budget() -> None:
    pool = ["a", "b", "c", "d", "e"]
    selected = sample_group(
        pool, budget=25, tokenizer_fn=lambda t: 10, method="random",
        embedding_model="unused", rng=random.Random(0),
    )
    # budget(25) // per-item cost(10) == 2 items fit.
    assert len(selected) == 2
    assert all(text in pool for text in selected)


def test_sample_group_kmeans_uses_embeddings_module(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_embed(texts: list[str], model_name: str) -> list[list[float]]:
        assert model_name == "fake-model"
        return [_TWO_CLUSTER_EMBEDDINGS[_TWO_CLUSTER_TEXTS.index(t)].tolist() for t in texts]

    monkeypatch.setitem(
        sys.modules, "pntx.embeddings", _fake_embeddings_module(fake_embed)
    )

    selected = sample_group(
        _TWO_CLUSTER_TEXTS, budget=10_000, tokenizer_fn=lambda t: 1, method="kmeans",
        embedding_model="fake-model", rng=random.Random(0),
    )
    assert any(t in _TWO_CLUSTER_TEXTS[:3] for t in selected)
    assert any(t in _TWO_CLUSTER_TEXTS[3:] for t in selected)


def test_sample_group_votek_uses_embeddings_module(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_embed(texts: list[str], model_name: str) -> list[list[float]]:
        return [_TWO_CLUSTER_EMBEDDINGS[_TWO_CLUSTER_TEXTS.index(t)].tolist() for t in texts]

    monkeypatch.setitem(
        sys.modules, "pntx.embeddings", _fake_embeddings_module(fake_embed)
    )

    selected = sample_group(
        _TWO_CLUSTER_TEXTS, budget=10_000, tokenizer_fn=lambda t: 1, method="votek",
        embedding_model="fake-model", rng=random.Random(0),
    )
    assert any(t in _TWO_CLUSTER_TEXTS[:3] for t in selected)
    assert any(t in _TWO_CLUSTER_TEXTS[3:] for t in selected)


def test_sample_group_trims_kmeans_result_to_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_embed(texts: list[str], model_name: str) -> list[list[float]]:
        return [_TWO_CLUSTER_EMBEDDINGS[_TWO_CLUSTER_TEXTS.index(t)].tolist() for t in texts]

    monkeypatch.setitem(
        sys.modules, "pntx.embeddings", _fake_embeddings_module(fake_embed)
    )

    # Budget only fits 1 text (cost 10 each); the estimated n from _estimate_n
    # may pick more, but _trim_to_budget must cut the result back down.
    selected = sample_group(
        _TWO_CLUSTER_TEXTS, budget=10, tokenizer_fn=lambda t: 10, method="kmeans",
        embedding_model="fake-model", rng=random.Random(0),
    )
    assert len(selected) <= 1


def test_trim_to_budget_skips_oversized_text_instead_of_stopping() -> None:
    costs = {"huge": 100, "a": 1, "b": 1, "c": 1}
    # "huge" leads the list and alone exceeds the budget; it must be skipped
    # rather than aborting the scan, so the shorter texts after it still get
    # included.
    trimmed = _trim_to_budget(["huge", "a", "b", "c"], 10, lambda t: costs[t])
    assert trimmed == ["a", "b", "c"]
