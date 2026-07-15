from __future__ import annotations

from collections.abc import Callable

from . import dedup as dedup_module
from . import prompts
from .backends.base import Backend
from .selection import Selector
from .types import ClassifyResult, Label

_DEFAULT_MAX_ATTEMPTS_MULTIPLIER = 3
_MIN_DEFAULT_MAX_ATTEMPTS = 3
_TOKENS_PER_TEXT_ESTIMATE = 64


def run_generation_loop(
    *,
    backend: Backend,
    positive: list[str],
    negative: list[str],
    selector: Selector,
    max_exemplars: int | None,
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
    pools and texts already accepted) and verify (self-classifies as
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
    seed_texts = positive + negative if dedup else []
    accepted: list[str] = []

    for _attempt in range(attempts):
        remaining = n - len(accepted)
        if remaining <= 0:
            break

        positive_k = max_exemplars if max_exemplars is not None else len(positive)
        negative_k = max_exemplars if max_exemplars is not None else len(negative)
        positive_exemplars = selector.select(positive, positive_k, query=None)
        negative_exemplars = selector.select(negative, negative_k, query=None)
        prompt = prompts.build_generate_prompt(
            positive_exemplars, negative_exemplars, side, remaining
        )
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
