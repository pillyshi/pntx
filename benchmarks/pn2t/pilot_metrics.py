"""Pure evaluation logic for the CounterfactualOverSampler ``verify`` pilot.

Kept separate from the runner (``counterfactual_pilot.py``) so it can be
unit-tested without a model. See ``research/ideas/counterfactual-edit-sampler.md``
("Choosing the ``verify`` default") for what the pilot measures and why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

STRATEGIES = ("none", "self", "classifier")


@dataclass(frozen=True)
class Candidate:
    """One generated edit that passed the format/no-op/minimality/dedup
    filters -- i.e. a candidate the ``verify`` strategies disagree about."""

    source_text: str
    text: str
    edit_ratio: float
    self_positive: bool
    """The editor LLM's own ``is_positive`` (what ``verify="self"`` uses)."""
    verifier_positive: bool
    """The cross-fitted classifier verifier's verdict (``verify=<classifier>``)."""
    judge_positive: bool
    """Reference label from an independent judge model (proxy for a human)."""
    downstream_positive: bool
    """Prediction of a shallow downstream classifier trained on original data
    only; ``True`` means the edit is "easy" for it."""


def keeps(candidate: Candidate, strategy: str) -> bool:
    """Whether ``strategy`` would accept ``candidate``."""
    if strategy == "none":
        return True
    if strategy == "self":
        return candidate.self_positive
    if strategy == "classifier":
        return candidate.verifier_positive
    raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize(candidates: list[Candidate], strategy: str) -> dict[str, Any]:
    """Per-strategy metrics, all against the judge's reference label.

    - ``precision``: share of kept edits the judge calls positive (label
      validity of what ends up in the training data).
    - ``catch_rate``: share of judge-negative candidates (failed flips) the
      strategy rejects.
    - ``false_reject_rate``: share of judge-positive candidates (good flips)
      it rejects.
    - ``yield``: share of candidates kept (inverse of the retry cost).
    - ``downstream_easy``: share of kept edits the shallow downstream
      classifier already gets right; lower means the kept data is harder,
      i.e. more informative (the model-in-the-loop bias concern).
    """
    kept = [c for c in candidates if keeps(c, strategy)]
    bad = [c for c in candidates if not c.judge_positive]
    good = [c for c in candidates if c.judge_positive]
    return {
        "strategy": strategy,
        "n_candidates": len(candidates),
        "n_kept": len(kept),
        "yield": _ratio(len(kept), len(candidates)),
        "precision": _ratio(sum(c.judge_positive for c in kept), len(kept)),
        "catch_rate": _ratio(sum(not keeps(c, strategy) for c in bad), len(bad)),
        "false_reject_rate": _ratio(sum(not keeps(c, strategy) for c in good), len(good)),
        "downstream_easy": _ratio(sum(c.downstream_positive for c in kept), len(kept)),
        "valid_and_hard": sum(c.judge_positive and not c.downstream_positive for c in kept),
    }


def format_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table of ``summarize`` rows."""

    def fmt(value: Any) -> str:
        if value is None:
            return "–"
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    columns = [
        "strategy",
        "n_kept",
        "yield",
        "precision",
        "catch_rate",
        "false_reject_rate",
        "downstream_easy",
        "valid_and_hard",
    ]
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(row.get(c)) for c in columns) + " |" for row in rows]
    return "\n".join(lines)
