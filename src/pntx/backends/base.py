from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Backend(Protocol):
    """A text-completion backend.

    Every backend (llama.cpp, Anthropic, ...) must implement at least this.
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
