from __future__ import annotations

from pntx.selection import DiversitySelector, NearestSelector, RandomSelector
from pntx.types import Pair

from .conftest import SAMPLE_PAIRS


def test_select_fewer_than_available_returns_k_distinct_pairs() -> None:
    selector = RandomSelector(seed=0)
    selected = selector.select(SAMPLE_PAIRS, k=2)
    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert all(pair in SAMPLE_PAIRS for pair in selected)


def test_select_k_greater_than_available_returns_all_pairs() -> None:
    selector = RandomSelector(seed=0)
    selected = selector.select(SAMPLE_PAIRS, k=100)
    assert set(selected) == set(SAMPLE_PAIRS)


def test_select_k_zero_returns_empty() -> None:
    selector = RandomSelector(seed=0)
    assert selector.select(SAMPLE_PAIRS, k=0) == []


def test_select_is_reproducible_with_same_seed() -> None:
    a = RandomSelector(seed=42).select(SAMPLE_PAIRS, k=2)
    b = RandomSelector(seed=42).select(SAMPLE_PAIRS, k=2)
    assert a == b


def test_select_ignores_query() -> None:
    selector = RandomSelector(seed=0)
    without_query = selector.select(SAMPLE_PAIRS, k=100)
    with_query = selector.select(SAMPLE_PAIRS, k=100, query="anything")
    assert set(without_query) == set(with_query)


_TOPIC_PAIRS: list[Pair] = [
    ("apple pie recipe", "apple pie disaster"),
    ("great customer service", "terrible customer service"),
    ("mountain hiking trip", "mountain hiking injury"),
]


def test_nearest_selector_picks_most_similar_pair_to_query() -> None:
    selector = NearestSelector()
    selected = selector.select(_TOPIC_PAIRS, k=1, query="apple pie recipe was amazing")
    assert selected == [_TOPIC_PAIRS[0]]


def test_nearest_selector_picks_most_similar_pair_japanese() -> None:
    selector = NearestSelector()
    selected = selector.select(SAMPLE_PAIRS, k=1, query="この映画は最高だった!")
    assert selected == [SAMPLE_PAIRS[0]]


def test_nearest_selector_k_zero_returns_empty() -> None:
    selector = NearestSelector()
    assert selector.select(_TOPIC_PAIRS, k=0, query="anything") == []


def test_nearest_selector_k_greater_than_available_returns_all() -> None:
    selector = NearestSelector()
    assert set(selector.select(_TOPIC_PAIRS, k=100, query="apple")) == set(_TOPIC_PAIRS)


def test_nearest_selector_query_none_falls_back_to_prefix() -> None:
    selector = NearestSelector()
    assert selector.select(_TOPIC_PAIRS, k=2, query=None) == _TOPIC_PAIRS[:2]


def test_nearest_selector_uses_custom_similarity_fn() -> None:
    calls: list[tuple[str, str]] = []

    def fake_similarity(a: str, b: str) -> float:
        calls.append((a, b))
        return 1.0 if "target" in b else 0.0

    pairs: list[Pair] = [("target text", "other1"), ("other2", "other3")]
    selector = NearestSelector(similarity_fn=fake_similarity)

    selected = selector.select(pairs, k=1, query="query")

    assert selected == [pairs[0]]
    assert calls


def test_diversity_selector_k_greater_than_available_returns_all() -> None:
    selector = DiversitySelector()
    assert set(selector.select(_TOPIC_PAIRS, k=100)) == set(_TOPIC_PAIRS)


def test_diversity_selector_k_zero_returns_empty() -> None:
    selector = DiversitySelector()
    assert selector.select(_TOPIC_PAIRS, k=0) == []


def test_diversity_selector_returns_k_distinct_pairs() -> None:
    selector = DiversitySelector()
    selected = selector.select(_TOPIC_PAIRS, k=2)
    assert len(selected) == 2
    assert len(set(selected)) == 2


def test_diversity_selector_prefers_the_most_different_pair_next() -> None:
    # near_duplicate is a near-duplicate of _TOPIC_PAIRS[0]; a genuinely
    # diverse selector picking a second pair should prefer the unrelated
    # pairs[2] over the near-duplicate.
    near_duplicate: Pair = ("apple pie recipe!!", "apple pie disaster!!")
    pairs = [_TOPIC_PAIRS[0], near_duplicate, _TOPIC_PAIRS[2]]
    selector = DiversitySelector()

    selected = selector.select(pairs, k=2)

    assert selected[0] == pairs[0]
    assert selected[1] == _TOPIC_PAIRS[2]


def test_diversity_selector_ignores_query() -> None:
    selector = DiversitySelector()
    without_query = selector.select(_TOPIC_PAIRS, k=2)
    with_query = selector.select(_TOPIC_PAIRS, k=2, query="irrelevant")
    assert without_query == with_query
