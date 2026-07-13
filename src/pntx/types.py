from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Label = Literal["positive", "negative"]

POSITIVE: Label = "positive"
NEGATIVE: Label = "negative"

Pair = tuple[str, str]
"""A single (positive, negative) example pair."""


@dataclass(frozen=True, eq=False)
class ClassifyResult:
    """Result of classifying a single text.

    ``result == "positive"`` compares against ``label`` directly, so callers
    don't need to write ``result.label == "positive"``.
    """

    label: Label
    confidence: float

    def __eq__(self, other: object) -> bool:
        if isinstance(other, str):
            return self.label == other
        if isinstance(other, ClassifyResult):
            return self.label == other.label and self.confidence == other.confidence
        return NotImplemented

    def __hash__(self) -> int:
        return hash((self.label, self.confidence))
