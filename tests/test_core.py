from __future__ import annotations

import math

import pytest

import pntx.core as core
from pntx import PNTX, prompts
from pntx.selection import RandomSelector

from .conftest import (
    SAMPLE_NEGATIVE,
    SAMPLE_POSITIVE,
    CompleteOnlyBackend,
    FakeBackend,
    FakeBatchBackend,
)


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


def test_unimplemented_backend_string_raises_import_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(
        core._BACKEND_REGISTRY, "missing", ("pntx._no_such_module", "NoSuchBackend")
    )
    with pytest.raises(ImportError, match=r"pip install 'pntx\[missing\]'"):
        PNTX(backend="missing")


def test_backend_string_forwards_kwargs_to_constructor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(core._BACKEND_REGISTRY, "fake", ("tests.conftest", "FakeBackend"))
    model = PNTX(backend="fake", choice_scores={"p": [1.0]})
    assert isinstance(model.backend, FakeBackend)
    assert model.backend.choice_scores == {"p": [1.0]}


def test_fit_stores_pools_and_returns_self() -> None:
    model = PNTX(backend=FakeBackend())
    returned = model.fit(positive=SAMPLE_POSITIVE, negative=SAMPLE_NEGATIVE)
    assert returned is model
    assert model.positive == SAMPLE_POSITIVE
    assert model.negative == SAMPLE_NEGATIVE


def test_fit_rejects_both_pools_empty() -> None:
    model = PNTX(backend=FakeBackend())
    with pytest.raises(ValueError, match="non-empty"):
        model.fit()


def test_fit_accepts_positive_only_pool() -> None:
    model = PNTX(backend=FakeBackend())
    returned = model.fit(positive=SAMPLE_POSITIVE)
    assert returned is model
    assert model.positive == SAMPLE_POSITIVE
    assert model.negative == []


def test_fit_accepts_negative_only_pool() -> None:
    model = PNTX(backend=FakeBackend())
    model.fit(negative=SAMPLE_NEGATIVE)
    assert model.positive == []
    assert model.negative == SAMPLE_NEGATIVE


def test_methods_require_fit_first() -> None:
    model = PNTX(backend=FakeBackend())
    with pytest.raises(RuntimeError, match="fit"):
        model.classify("some text")
    with pytest.raises(RuntimeError, match="fit"):
        model.classify_batch(["some text"])
    with pytest.raises(RuntimeError, match="fit"):
        model.generate(n=1, side="positive")


def test_classify_falls_back_to_parsing_completion_for_plain_backend() -> None:
    backend = CompleteOnlyBackend(complete_responses=["This text is positive."])
    model = PNTX(backend=backend).fit(positive=SAMPLE_POSITIVE, negative=SAMPLE_NEGATIVE)

    result = model.classify("some text")

    assert result == "positive"
    assert result.confidence == 1.0
    assert backend.complete_calls == [
        prompts.build_classify_prompt(SAMPLE_POSITIVE, SAMPLE_NEGATIVE, "some text")
    ]


def test_classify_batch_falls_back_to_parsing_completions_for_plain_backend() -> None:
    backend = CompleteOnlyBackend(complete_responses=["positive", "negative"])
    model = PNTX(backend=backend).fit(positive=SAMPLE_POSITIVE, negative=SAMPLE_NEGATIVE)

    results = model.classify_batch(["good text", "bad text"])

    assert [r.label for r in results] == ["positive", "negative"]
    assert backend.complete_calls == [
        prompts.build_classify_prompt(SAMPLE_POSITIVE, SAMPLE_NEGATIVE, "good text"),
        prompts.build_classify_prompt(SAMPLE_POSITIVE, SAMPLE_NEGATIVE, "bad text"),
    ]


def test_classify_picks_the_higher_scoring_label() -> None:
    prompt = prompts.build_classify_prompt(SAMPLE_POSITIVE, SAMPLE_NEGATIVE, "great service")
    backend = FakeBackend(choice_scores={prompt: [2.0, 0.0]})
    model = PNTX(backend=backend).fit(positive=SAMPLE_POSITIVE, negative=SAMPLE_NEGATIVE)

    result = model.classify("great service")

    assert result == "positive"
    expected_confidence = math.exp(2.0) / (math.exp(2.0) + math.exp(0.0))
    assert result.confidence == pytest.approx(expected_confidence)
    assert backend.score_calls == [(prompt, prompts.classify_choice_texts())]


def test_classify_prefers_negative_when_it_scores_higher() -> None:
    prompt = prompts.build_classify_prompt(SAMPLE_POSITIVE, SAMPLE_NEGATIVE, "bad service")
    backend = FakeBackend(choice_scores={prompt: [0.0, 3.0]})
    model = PNTX(backend=backend).fit(positive=SAMPLE_POSITIVE, negative=SAMPLE_NEGATIVE)

    result = model.classify("bad service")

    assert result == "negative"


def test_classify_caps_exemplars_with_max_exemplars() -> None:
    backend = FakeBackend()
    model = PNTX(backend=backend, selector=RandomSelector(seed=0), max_exemplars=1).fit(
        positive=SAMPLE_POSITIVE, negative=SAMPLE_NEGATIVE
    )

    model.classify("some text")

    (prompt, _choices) = backend.score_calls[0]
    assert prompt.count("Label: positive") == 1
    assert prompt.count("Label: negative") == 1


def test_classify_works_with_positive_only_pool() -> None:
    backend = FakeBackend()
    model = PNTX(backend=backend).fit(positive=SAMPLE_POSITIVE)

    model.classify("some text")

    (prompt, _choices) = backend.score_calls[0]
    assert "Label: positive" in prompt
    assert "Label: negative" not in prompt


def test_classify_batch_empty_texts_returns_empty_list() -> None:
    model = PNTX(backend=FakeBackend()).fit(positive=SAMPLE_POSITIVE, negative=SAMPLE_NEGATIVE)
    assert model.classify_batch([]) == []


def test_classify_batch_uses_batch_scoring_backend_when_available() -> None:
    prefix = prompts.build_exemplar_prefix(SAMPLE_POSITIVE, SAMPLE_NEGATIVE)
    query_a = prompts.build_query_suffix("great service")
    query_b = prompts.build_query_suffix("bad service")
    backend = FakeBatchBackend(
        batch_scores={
            (prefix, query_a): [2.0, 0.0],
            (prefix, query_b): [0.0, 2.0],
        }
    )
    model = PNTX(backend=backend).fit(positive=SAMPLE_POSITIVE, negative=SAMPLE_NEGATIVE)

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
    prefix = prompts.build_exemplar_prefix(SAMPLE_POSITIVE, SAMPLE_NEGATIVE)
    prompt_a = prefix + prompts.build_query_suffix("great service")
    prompt_b = prefix + prompts.build_query_suffix("bad service")
    backend = FakeBackend(
        choice_scores={
            prompt_a: [2.0, 0.0],
            prompt_b: [0.0, 2.0],
        }
    )
    model = PNTX(backend=backend).fit(positive=SAMPLE_POSITIVE, negative=SAMPLE_NEGATIVE)

    results = model.classify_batch(["great service", "bad service"])

    assert [r.label for r in results] == ["positive", "negative"]
    assert [call[0] for call in backend.score_calls] == [prompt_a, prompt_b]


def test_classify_batch_matches_classify_for_each_text() -> None:
    backend = FakeBackend(
        choice_scores={
            prompts.build_classify_prompt(
                SAMPLE_POSITIVE, SAMPLE_NEGATIVE, "great service"
            ): [2.0, 0.0],
        }
    )
    model = PNTX(backend=backend).fit(positive=SAMPLE_POSITIVE, negative=SAMPLE_NEGATIVE)

    single = model.classify("great service")
    batch = model.classify_batch(["great service"])[0]

    assert single == batch.label
    assert single.confidence == pytest.approx(batch.confidence)
