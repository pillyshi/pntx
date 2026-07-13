from __future__ import annotations

import pytest

from pntx import PNTX, prompts

from .conftest import SAMPLE_PAIRS, CompleteOnlyBackend, FakeBackend


def test_generate_zero_returns_empty_list_without_warning(recwarn: pytest.WarningsRecorder) -> None:
    model = PNTX(backend=FakeBackend()).fit(SAMPLE_PAIRS)
    assert model.generate(n=0, side="positive") == []
    assert len(recwarn) == 0


def test_generate_rejects_negative_n() -> None:
    model = PNTX(backend=FakeBackend()).fit(SAMPLE_PAIRS)
    with pytest.raises(ValueError, match="n must be >= 0"):
        model.generate(n=-1, side="positive")


def test_generate_without_verify_or_dedup_returns_parsed_candidates() -> None:
    backend = FakeBackend(complete_responses=["a\n2. b\n3. c"])
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    result = model.generate(n=3, side="positive", verify=False, dedup=False)

    assert result == ["a", "b", "c"]
    assert len(backend.complete_calls) == 1


def test_generate_works_without_scoring_backend_when_verify_false() -> None:
    backend = CompleteOnlyBackend(complete_responses=["a\n2. b"])
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    result = model.generate(n=2, side="positive", verify=False, dedup=False)

    assert result == ["a", "b"]


def test_generate_requires_scoring_backend_when_verify_true() -> None:
    backend = CompleteOnlyBackend()
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    with pytest.raises(NotImplementedError, match="ScoringBackend"):
        model.generate(n=1, side="positive", verify=True)
    assert backend.complete_calls == []  # fails fast, before generating anything


def test_generate_dedup_filters_near_duplicate_within_batch() -> None:
    backend = FakeBackend(
        complete_responses=[
            "This movie was amazing and fun\n"
            "2. This movie was amazing and fun!\n"
            "3. A completely unrelated sentence about weather"
        ]
    )
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    result = model.generate(n=2, side="positive", verify=False, dedup=True)

    assert result == [
        "This movie was amazing and fun",
        "A completely unrelated sentence about weather",
    ]


def test_generate_dedup_filters_against_seed_pairs(recwarn: pytest.WarningsRecorder) -> None:
    seed_text = SAMPLE_PAIRS[0][0]
    backend = FakeBackend(complete_responses=[seed_text, seed_text])
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    result = model.generate(n=1, side="positive", verify=False, dedup=True, max_attempts=2)

    assert result == []
    assert len(backend.complete_calls) == 2
    assert any("only generated 0" in str(w.message) for w in recwarn)


def test_generate_verify_rejects_then_retries_then_succeeds() -> None:
    wrong_label_text = "wrong label candidate"
    low_confidence_text = "low confidence candidate"
    good_text = "good candidate"

    wrong_label_prompt = prompts.build_classify_prompt(SAMPLE_PAIRS, wrong_label_text)
    low_confidence_prompt = prompts.build_classify_prompt(SAMPLE_PAIRS, low_confidence_text)
    good_prompt = prompts.build_classify_prompt(SAMPLE_PAIRS, good_text)

    backend = FakeBackend(
        choice_scores={
            wrong_label_prompt: [0.0, 3.0],  # classifies negative
            low_confidence_prompt: [0.05, 0.0],  # classifies positive, low confidence
            good_prompt: [3.0, 0.0],  # classifies positive, high confidence
        },
        complete_responses=[wrong_label_text, low_confidence_text, good_text],
    )
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    result = model.generate(
        n=1, side="positive", verify=True, dedup=False, min_confidence=0.8, max_attempts=5
    )

    assert result == [good_text]
    assert len(backend.complete_calls) == 3


def test_generate_hits_max_attempts_and_warns(recwarn: pytest.WarningsRecorder) -> None:
    bad_text = "always rejected"
    bad_prompt = prompts.build_classify_prompt(SAMPLE_PAIRS, bad_text)
    backend = FakeBackend(
        choice_scores={bad_prompt: [0.0, 3.0]},  # always classifies negative
        complete_responses=[bad_text, bad_text],
    )
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    result = model.generate(n=3, side="positive", verify=True, dedup=False, max_attempts=2)

    assert result == []
    assert len(backend.complete_calls) == 2
    assert any("only generated 0" in str(w.message) for w in recwarn)


def test_generate_uses_selected_side_in_prompt() -> None:
    backend = FakeBackend(complete_responses=["x"])
    model = PNTX(backend=backend).fit(SAMPLE_PAIRS)

    model.generate(n=1, side="negative", verify=False, dedup=False)

    assert "negative texts" in backend.complete_calls[0]
