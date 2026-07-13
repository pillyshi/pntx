from __future__ import annotations

import pytest

from pntx import PNTX

from .conftest import SAMPLE_PAIRS, FakeBackend


def test_accepts_backend_instance_directly() -> None:
    backend = FakeBackend()
    model = PNTX(backend=backend)
    assert model.backend is backend


def test_unknown_backend_string_raises_value_error() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        PNTX(backend="not-a-real-backend")


def test_unimplemented_backend_string_raises_import_error() -> None:
    with pytest.raises(ImportError, match=r"pip install 'pntx\[llama\]'"):
        PNTX(backend="llama")


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
