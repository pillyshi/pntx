from __future__ import annotations

from pntx.backends.anthropic import AnthropicBackend


def test_complete_returns_text(anthropic_backend: AnthropicBackend) -> None:
    text = anthropic_backend.complete(
        "Reply with exactly one word: hello", max_tokens=8, temperature=0.0
    )
    assert isinstance(text, str)
    assert text.strip() != ""


def test_complete_batch_returns_one_result_per_prompt(anthropic_backend: AnthropicBackend) -> None:
    prompts = [
        "Reply with exactly one word: one",
        "Reply with exactly one word: two",
        "Reply with exactly one word: three",
    ]
    results = anthropic_backend.complete_batch(prompts, max_tokens=8, temperature=0.0)
    assert len(results) == 3
    assert all(isinstance(r, str) and r.strip() for r in results)
