from __future__ import annotations

from pntx.types import Pair


class FakeBackend:
    """A canned-response ScoringBackend for testing, per CLAUDE.md's test conventions."""

    def __init__(self, choice_scores: dict[str, list[float]] | None = None) -> None:
        self.choice_scores = choice_scores or {}
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
        return ""

    def score_choices(self, prompt: str, choices: list[str]) -> list[float]:
        self.score_calls.append((prompt, choices))
        try:
            return self.choice_scores[prompt]
        except KeyError:
            return [0.0 for _ in choices]


SAMPLE_PAIRS: list[Pair] = [
    ("この映画は最高だった", "この映画は退屈だった"),
    ("サポートが丁寧で助かった", "サポートの対応が雑だった"),
    ("店員さんの笑顔が素敵だった", "店員さんの態度が冷たかった"),
]
