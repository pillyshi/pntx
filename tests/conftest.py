from __future__ import annotations

from pntx.types import Pair


class FakeBackend:
    """A canned-response ScoringBackend for testing, per CLAUDE.md's test conventions."""

    def __init__(
        self,
        choice_scores: dict[str, list[float]] | None = None,
        complete_responses: list[str] | None = None,
    ) -> None:
        self.choice_scores = choice_scores or {}
        self._complete_responses = list(complete_responses or [])
        self.complete_calls: list[str] = []
        self.score_calls: list[tuple[str, list[str]]] = []

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> str:
        self.complete_calls.append(prompt)
        if self._complete_responses:
            return self._complete_responses.pop(0)
        return ""

    def score_choices(self, prompt: str, choices: list[str]) -> list[float]:
        self.score_calls.append((prompt, choices))
        try:
            return self.choice_scores[prompt]
        except KeyError:
            return [0.0 for _ in choices]


class FakeBatchBackend(FakeBackend):
    """A FakeBackend that also implements BatchScoringBackend."""

    def __init__(
        self,
        choice_scores: dict[str, list[float]] | None = None,
        batch_scores: dict[tuple[str, str], list[float]] | None = None,
    ) -> None:
        super().__init__(choice_scores)
        self.batch_scores = batch_scores or {}
        self.batch_calls: list[tuple[str, list[str], list[str]]] = []

    def score_choices_batch(
        self, prefix: str, queries: list[str], choices: list[str]
    ) -> list[list[float]]:
        self.batch_calls.append((prefix, list(queries), list(choices)))
        return [self.batch_scores.get((prefix, query), [0.0 for _ in choices]) for query in queries]


class CompleteOnlyBackend:
    """A Backend that does *not* implement ScoringBackend."""

    def __init__(self, complete_responses: list[str] | None = None) -> None:
        self._complete_responses = list(complete_responses or [])
        self.complete_calls: list[str] = []

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> str:
        self.complete_calls.append(prompt)
        if self._complete_responses:
            return self._complete_responses.pop(0)
        return ""


SAMPLE_PAIRS: list[Pair] = [
    ("この映画は最高だった", "この映画は退屈だった"),
    ("サポートが丁寧で助かった", "サポートの対応が雑だった"),
    ("店員さんの笑顔が素敵だった", "店員さんの態度が冷たかった"),
]
