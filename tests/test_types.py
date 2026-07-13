from __future__ import annotations

from pntx.types import ClassifyResult


def test_eq_against_str() -> None:
    result = ClassifyResult(label="positive", confidence=0.9)
    assert result == "positive"
    assert result != "negative"
    assert "positive" == result
    assert "negative" != result


def test_eq_against_other_result() -> None:
    a = ClassifyResult(label="positive", confidence=0.9)
    b = ClassifyResult(label="positive", confidence=0.9)
    c = ClassifyResult(label="positive", confidence=0.5)
    assert a == b
    assert a != c


def test_eq_against_unrelated_type() -> None:
    result = ClassifyResult(label="positive", confidence=0.9)
    assert result != 42
