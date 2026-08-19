from __future__ import annotations

import pytest

from pntx._labels import resolve_binary_labels


def test_numeric_labels_greater_value_is_positive() -> None:
    assert resolve_binary_labels([0, 1, 0, 1]) == (0, 1)
    assert resolve_binary_labels([-1, 1, -1, 1]) == (-1, 1)
    assert resolve_binary_labels([False, True, False]) == (False, True)


def test_positive_negative_strings_resolve_directly() -> None:
    assert resolve_binary_labels(["positive", "negative", "positive"]) == ("negative", "positive")


def test_ambiguous_labels_require_pos_label() -> None:
    with pytest.raises(ValueError, match="pass pos_label"):
        resolve_binary_labels(["spam", "ham", "spam"])


def test_pos_label_overrides_auto_resolution() -> None:
    assert resolve_binary_labels(["spam", "ham"], pos_label="spam") == ("ham", "spam")
    # explicit pos_label wins even when it disagrees with the numeric/string rules
    assert resolve_binary_labels([0, 1], pos_label=0) == (1, 0)
    assert resolve_binary_labels(["positive", "negative"], pos_label="negative") == (
        "positive",
        "negative",
    )


def test_pos_label_must_be_one_of_the_classes() -> None:
    with pytest.raises(ValueError, match="not one of the classes"):
        resolve_binary_labels(["spam", "ham"], pos_label="eggs")


def test_requires_exactly_two_classes() -> None:
    with pytest.raises(ValueError, match="exactly 2 classes"):
        resolve_binary_labels([0, 1, 2])
    with pytest.raises(ValueError, match="exactly 2 classes"):
        resolve_binary_labels([0, 0, 0])
