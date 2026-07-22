from __future__ import annotations

from pydantic import BaseModel, Field


class HardPositive(BaseModel):
    """One generated positive-class text plus the LLM's rationale for it."""

    text: str
    positive_evidence: list[str]
    confusing_evidence: list[str]


class BoundaryFeature(BaseModel):
    """A feature the LLM judged relevant to the positive/negative boundary."""

    feature: str
    importance: float = Field(ge=0.0, le=1.0)


class HardPositiveGenerationResult(BaseModel):
    """Full structured output of one ``OverSampler`` generation batch."""

    positive_features: list[str]
    negative_features: list[str]
    boundary_features: list[BoundaryFeature]
    hard_positives: list[HardPositive]
