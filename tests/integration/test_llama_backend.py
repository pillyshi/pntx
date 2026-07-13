from __future__ import annotations

import math

from pntx.backends.llama import LlamaCppBackend


def test_complete_returns_text(llama_backend: LlamaCppBackend) -> None:
    text = llama_backend.complete("The capital of France is", max_tokens=8, temperature=0.0)
    assert isinstance(text, str)
    assert text != ""


def test_score_choices_returns_one_finite_score_per_choice(
    llama_backend: LlamaCppBackend,
) -> None:
    scores = llama_backend.score_choices("The capital of France is", ["Paris", "a banana"])
    assert len(scores) == 2
    assert all(math.isfinite(s) for s in scores)


def test_score_choices_rolls_back_kv_cache_between_choices(
    llama_backend: LlamaCppBackend,
) -> None:
    # If the prefix rollback between choices were broken, evaluating the same
    # choice twice in a row would pick up leftover KV state from the first
    # pass and the two scores would drift apart.
    scores = llama_backend.score_choices("The weather today is", ["sunny", "sunny"])
    assert math.isclose(scores[0], scores[1], rel_tol=1e-4)


def test_score_choices_is_deterministic_across_calls(llama_backend: LlamaCppBackend) -> None:
    first = llama_backend.score_choices("The capital of France is", ["Paris", "a banana"])
    second = llama_backend.score_choices("The capital of France is", ["Paris", "a banana"])
    assert first == second
