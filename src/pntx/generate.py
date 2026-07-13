from __future__ import annotations

from collections.abc import Callable

from . import dedup as dedup_module
from . import prompts
from .backends.base import Backend
from .selection import Selector
from .types import ClassifyResult, Label, Pair

_DEFAULT_MAX_ATTEMPTS_MULTIPLIER = 3
_MIN_DEFAULT_MAX_ATTEMPTS = 3
_TOKENS_PER_TEXT_ESTIMATE = 64


def run_generation_loop(
    *,
    backend: Backend,
    pairs: list[Pair],
    selector: Selector,
    exemplar_count: int,
    classify: Callable[[str], ClassifyResult],
    n: int,
    side: Label,
    temperature: float,
    dedup: bool,
    verify: bool,
    min_confidence: float,
    max_attempts: int | None,
) -> list[str]:
    """Generate up to ``n`` new ``side`` texts, retrying failed attempts.

    On each attempt, asks the backend for however many texts are still
    needed, then accepts candidates that pass dedup (against both the seed
    pairs and texts already accepted) and verify (self-classifies as
    ``side`` with at least ``min_confidence``). Stops after ``max_attempts``
    attempts even if ``n`` texts were never reached; the caller is
    responsible for warning about a shortfall.
    """
    if n <= 0:
        return []

    attempts = (
        max_attempts
        if max_attempts is not None
        else max(n * _DEFAULT_MAX_ATTEMPTS_MULTIPLIER, _MIN_DEFAULT_MAX_ATTEMPTS)
    )
    seed_texts = [text for pair in pairs for text in pair] if dedup else []
    accepted: list[str] = []

    for _attempt in range(attempts):
        remaining = n - len(accepted)
        if remaining <= 0:
            break

        exemplars = selector.select(pairs, exemplar_count, query=None)
        prompt = prompts.build_generate_prompt(exemplars, side, remaining)
        raw = backend.complete(
            prompt,
            temperature=temperature,
            max_tokens=remaining * _TOKENS_PER_TEXT_ESTIMATE,
        )

        for candidate in prompts.parse_generated_texts(raw):
            if len(accepted) >= n:
                break
            if dedup and dedup_module.is_near_duplicate(candidate, seed_texts + accepted):
                continue
            if verify:
                result = classify(candidate)
                if result.label != side or result.confidence < min_confidence:
                    continue
            accepted.append(candidate)

    return accepted
