from __future__ import annotations

import json

import pytest
from pydantic import BaseModel, ValidationError

from pntx.pn2t._structured import complete_structured

from .conftest import FakeBackend, FakeStructuredBackend


class _Point(BaseModel):
    x: int
    y: int


def test_complete_structured_parses_plain_json() -> None:
    backend = FakeBackend(complete_responses=[json.dumps({"x": 1, "y": 2})])
    result = complete_structured(backend, "system", "user", _Point)
    assert result == _Point(x=1, y=2)


def test_complete_structured_strips_markdown_fence() -> None:
    fenced = "Here you go:\n```json\n" + json.dumps({"x": 3, "y": 4}) + "\n```"
    backend = FakeBackend(complete_responses=[fenced])
    result = complete_structured(backend, "system", "user", _Point)
    assert result == _Point(x=3, y=4)


def test_complete_structured_extracts_braces_from_surrounding_prose() -> None:
    raw = "Sure, here's the JSON: " + json.dumps({"x": 5, "y": 6}) + " hope that helps!"
    backend = FakeBackend(complete_responses=[raw])
    result = complete_structured(backend, "system", "user", _Point)
    assert result == _Point(x=5, y=6)


def test_complete_structured_retries_once_on_invalid_json_then_succeeds() -> None:
    backend = FakeBackend(complete_responses=["not json at all", json.dumps({"x": 7, "y": 8})])
    result = complete_structured(backend, "system", "user", _Point)
    assert result == _Point(x=7, y=8)
    assert len(backend.complete_calls) == 2


def test_complete_structured_raises_after_exhausting_retries() -> None:
    backend = FakeBackend(complete_responses=["not json", "still not json"])
    with pytest.raises((ValidationError, json.JSONDecodeError)):
        complete_structured(backend, "system", "user", _Point)
    assert len(backend.complete_calls) == 2


def test_complete_structured_prefers_complete_json_for_structured_backend() -> None:
    backend = FakeStructuredBackend(json_responses=[json.dumps({"x": 1, "y": 2})])
    result = complete_structured(backend, "system", "user", _Point)
    assert result == _Point(x=1, y=2)
    assert backend.complete_calls == []
    assert len(backend.complete_json_calls) == 1
    prompt, schema = backend.complete_json_calls[0]
    assert "system" in prompt and "user" in prompt
    assert schema == _Point.model_json_schema()


def test_complete_structured_retries_via_complete_json_on_invalid_json() -> None:
    backend = FakeStructuredBackend(
        json_responses=["not json at all", json.dumps({"x": 7, "y": 8})]
    )
    result = complete_structured(backend, "system", "user", _Point)
    assert result == _Point(x=7, y=8)
    assert len(backend.complete_json_calls) == 2
