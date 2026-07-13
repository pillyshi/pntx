from __future__ import annotations

from pntx.selection import RandomSelector

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
