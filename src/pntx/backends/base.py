from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """A text-completion backend.

    Every backend (llama.cpp, ...) must implement at least this.
    """

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> str: ...


@runtime_checkable
class ScoringBackend(Backend, Protocol):
    """A backend that can score candidate continuations directly.

    This is the primary path for classification: instead of generating text
    and parsing a label out of it, the backend returns a log-likelihood for
    each candidate choice appended to ``prompt``.
    """

    def score_choices(self, prompt: str, choices: list[str]) -> list[float]:
        """Return the log-likelihood of each choice in ``choices`` as a
        continuation of ``prompt``."""
        ...


@runtime_checkable
class BatchScoringBackend(ScoringBackend, Protocol):
    """A ``ScoringBackend`` that can batch-score many queries sharing one prefix.

    Used by ``classify_batch`` to warm a common prefix (e.g. the few-shot
    exemplar block) once and reuse it across every query, instead of
    re-evaluating it per item. Backends without this (e.g. remote API
    backends) are scored one item at a time via ``score_choices``.
    """

    def score_choices_batch(
        self, prefix: str, queries: list[str], choices: list[str]
    ) -> list[list[float]]:
        """For each query in ``queries``, score ``choices`` as a continuation
        of ``prefix + query``. Returns one score list per query, in order."""
        ...


@runtime_checkable
class StructuredBackend(Backend, Protocol):
    """A backend that can constrain generation to match a JSON schema directly
    (e.g. via llama.cpp's grammar-constrained decoding), instead of relying on
    prompt instructions plus a parse-and-retry loop.

    Used by ``pn2t._structured.complete_structured``, which prefers this over
    plain ``complete()`` + text parsing whenever the backend implements it.
    Backends without it (e.g. remote chat APIs with no grammar/schema
    support) fall back to the prompt-and-parse path unchanged.
    """

    def complete_json(
        self,
        prompt: str,
        *,
        schema: dict[str, Any],
        temperature: float = 1.0,
        max_tokens: int = 512,
    ) -> str:
        """Complete ``prompt``, constraining generation to syntactically valid
        JSON matching ``schema`` (a JSON Schema dict, typically from
        ``pydantic.BaseModel.model_json_schema()``).

        Guarantees JSON-syntax validity, not full schema conformance (e.g.
        numeric ranges or enum membership) -- callers should still validate
        the result (e.g. via ``pydantic.BaseModel.model_validate_json()``).
        """
        ...


@runtime_checkable
class BatchBackend(Backend, Protocol):
    """A ``Backend`` that can complete many prompts concurrently.

    Used by ``classify_batch``'s parse-based path (for backends that don't
    implement ``ScoringBackend``, e.g. remote chat APIs): instead of
    completing each prompt one at a time, ``complete_batch`` can run them
    concurrently (e.g. via asyncio) and return results in the same order.
    """

    def complete_batch(
        self,
        prompts: list[str],
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> list[str]:
        """Complete every prompt in ``prompts``, returning results in order."""
        ...
