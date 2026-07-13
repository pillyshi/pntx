from __future__ import annotations

import math

import pytest

import pntx.core as core
from pntx import PNTX, prompts
from pntx.selection import RandomSelector

from .conftest import SAMPLE_PAIRS, CompleteOnlyBackend, FakeBackend, FakeBatchBackend


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


def test_classify_and_classify_batch_require_scoring_backend() -> None:
    model = PNTX(backend=CompleteOnlyBackend()).fit(SAMPLE_PAIRS)
    with pytest.raises(NotImplementedError, match="ScoringBackend"):
        model.classify("some text")
    with pytest.raises(NotImplementedError, match="ScoringBackend"):
        model.classify_batch(["some text"])


def test_classify_picks_the_higher_scoring_label() -> None:
    prompt = prompts.build_classify_prompt(SAMPLE_PAIRS, "great service")
    backend = FakeBackend(choice_scores={prompt: [2.0, 0.0]})
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    result = model.classify("great service")

    assert result == "positive"
    expected_confidence = math.exp(2.0) / (math.exp(2.0) + math.exp(0.0))
    assert result.confidence == pytest.approx(expected_confidence)
    assert backend.score_calls == [(prompt, prompts.classify_choice_texts())]


def test_classify_prefers_negative_when_it_scores_higher() -> None:
    prompt = prompts.build_classify_prompt(SAMPLE_PAIRS, "bad service")
    backend = FakeBackend(choice_scores={prompt: [0.0, 3.0]})
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    result = model.classify("bad service")

    assert result == "negative"


def test_classify_caps_exemplars_with_max_exemplars() -> None:
    backend = FakeBackend()
    model = PNTX(backend=backend, selector=RandomSelector(seed=0), max_exemplars=1).fit(
        SAMPLE_PAIRS
    )

    model.classify("some text")

    (prompt, _choices) = backend.score_calls[0]
    assert prompt.count("Label: positive") == 1
    assert prompt.count("Label: negative") == 1


def test_classify_batch_empty_texts_returns_empty_list() -> None:
    model = PNTX(backend=FakeBackend()).fit(SAMPLE_PAIRS)
    assert model.classify_batch([]) == []


def test_classify_batch_uses_batch_scoring_backend_when_available() -> None:
    prefix = prompts.build_exemplar_prefix(SAMPLE_PAIRS)
    query_a = prompts.build_query_suffix("great service")
    query_b = prompts.build_query_suffix("bad service")
    backend = FakeBatchBackend(
        batch_scores={
            (prefix, query_a): [2.0, 0.0],
            (prefix, query_b): [0.0, 2.0],
        }
    )
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    results = model.classify_batch(["great service", "bad service"])

    assert [r.label for r in results] == ["positive", "negative"]
    # The batch path must be used, and the shared prefix evaluated once.
    assert len(backend.batch_calls) == 1
    assert backend.score_calls == []
    called_prefix, called_queries, called_choices = backend.batch_calls[0]
    assert called_prefix == prefix
    assert called_queries == [query_a, query_b]
    assert called_choices == prompts.classify_choice_texts()


def test_classify_batch_falls_back_to_sequential_scoring_without_batch_backend() -> None:
    prefix = prompts.build_exemplar_prefix(SAMPLE_PAIRS)
    prompt_a = prefix + prompts.build_query_suffix("great service")
    prompt_b = prefix + prompts.build_query_suffix("bad service")
    backend = FakeBackend(
        choice_scores={
            prompt_a: [2.0, 0.0],
            prompt_b: [0.0, 2.0],
        }
    )
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    results = model.classify_batch(["great service", "bad service"])

    assert [r.label for r in results] == ["positive", "negative"]
    assert [call[0] for call in backend.score_calls] == [prompt_a, prompt_b]


def test_classify_batch_matches_classify_for_each_text() -> None:
    backend = FakeBackend(
        choice_scores={
            prompts.build_classify_prompt(SAMPLE_PAIRS, "great service"): [2.0, 0.0],
        }
    )
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    single = model.classify("great service")
    batch = model.classify_batch(["great service"])[0]

    assert single == batch.label
    assert single.confidence == pytest.approx(batch.confidence)
