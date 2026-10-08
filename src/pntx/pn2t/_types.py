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
    """Full structured output of one ``HardPositiveOverSampler`` generation batch."""

    positive_features: list[str]
    negative_features: list[str]
    boundary_features: list[BoundaryFeature]
    hard_positives: list[HardPositive]


class SyntheticText(BaseModel):
    """One generated synthetic-positive text plus an auditable anonymization note."""

    text: str
    generalized_from: list[str]


class SyntheticGenerationResult(BaseModel):
    """Full structured output of one ``TypicalPositiveOverSampler`` generation batch."""

    style_features: list[str]
    content_features: list[str]
    synthetic_texts: list[SyntheticText]


class CounterfactualEditOutput(BaseModel):
    """One edit as returned by the LLM (``pivot_id`` indexes the batch's
    numbered negative texts, not ``X``)."""

    pivot_id: int
    edited_text: str
    changed_spans: list[str]
    is_positive: bool


class CounterfactualBatch(BaseModel):
    """Structured output the LLM must return for one ``CounterfactualOverSampler`` batch."""

    edits: list[CounterfactualEditOutput]


class CounterfactualEdit(BaseModel):
    """One accepted counterfactual edit: ``X[source_index]`` (a negative) minimally
    edited into ``text`` (a positive)."""

    source_index: int
    source_text: str
    text: str
    changed_spans: list[str]
    self_assessed_positive: bool
    edit_ratio: float


class RejectedEdit(BaseModel):
    """A candidate edit that was not accepted, with the reason (for auditing
    rejection rates per filter/verification strategy)."""

    source_index: int | None
    source_text: str | None
    text: str
    reason: str


class CounterfactualGenerationResult(BaseModel):
    """Full fitted state of a ``CounterfactualOverSampler``."""

    edits: list[CounterfactualEdit]
    rejected: list[RejectedEdit]
