from __future__ import annotations

import pytest

from pntx.selection import BudgetSelector, DiversitySelector, NearestSelector, RandomSelector

from .conftest import SAMPLE_POSITIVE

_TOPIC_POOL: list[str] = [
    "apple pie recipe",
    "great customer service",
    "mountain hiking trip",
]


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
