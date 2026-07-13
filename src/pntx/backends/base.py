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
