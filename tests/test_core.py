from __future__ import annotations

import pytest

import pntx.core as core
from pntx import PNTX

from .conftest import SAMPLE_PAIRS, FakeBackend


def test_accepts_backend_instance_directly() -> None:
    backend = FakeBackend()
    model = PNTX(backend=backend)
    assert model.backend is backend


def test_backend_instance_rejects_extra_kwargs() -> None:
    with pytest.raises(TypeError, match="backend_kwargs"):
        PNTX(backend=FakeBackend(), model_path="unused")


def test_unknown_backend_string_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        PNTX(backend="not-a-real-backend")


def test_unimplemented_backend_string_raises_import_error() -> None:
    with pytest.raises(ImportError, match=r"pip install 'pntx\[anthropic\]'"):
        PNTX(backend="anthropic")


def test_backend_string_forwards_kwargs_to_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(core._BACKEND_REGISTRY, "fake", ("tests.conftest", "FakeBackend"))
    model = PNTX(backend="fake", choice_scores={"p": [1.0]})
    assert isinstance(model.backend, FakeBackend)
    assert model.backend.choice_scores == {"p": [1.0]}


def test_fit_stores_pairs_and_returns_self() -> None:
    model = PNTX(backend=FakeBackend())
    returned = model.fit(SAMPLE_PAIRS)
    assert returned is model
    assert model.pairs == SAMPLE_PAIRS


def test_fit_rejects_empty_pairs() -> None:
    model = PNTX(backend=FakeBackend())
    with pytest.raises(ValueError, match="non-empty"):
        model.fit([])


def test_methods_require_fit_first() -> None:
    model = PNTX(backend=FakeBackend())
    with pytest.raises(RuntimeError, match="fit"):
        model.classify("some text")
    with pytest.raises(RuntimeError, match="fit"):
        model.classify_batch(["some text"])
    with pytest.raises(RuntimeError, match="fit"):
        model.generate(n=1, side="positive")


def test_unimplemented_methods_raise_after_fit() -> None:
    model = PNTX(backend=FakeBackend()).fit(SAMPLE_PAIRS)
    with pytest.raises(NotImplementedError):
        model.classify("some text")
    with pytest.raises(NotImplementedError):
        model.classify_batch(["some text"])
    with pytest.raises(NotImplementedError):
        model.generate(n=1, side="positive")
