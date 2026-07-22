"""Classification metrics for the t2pn benchmark, stdlib only (no sklearn)."""

from __future__ import annotations

from dataclasses import dataclass

from pntx.types import NEGATIVE, POSITIVE, Label

_LABELS: tuple[Label, Label] = (POSITIVE, NEGATIVE)


@dataclass(frozen=True)
class ClassificationMetrics:
    accuracy: float
    precision: dict[Label, float]
    recall: dict[Label, float]
    f1: dict[Label, float]
    confusion: dict[tuple[Label, Label], int]
    """Keyed by ``(true_label, predicted_label)``."""
    mean_confidence: float


def compute_metrics(
    y_true: list[Label], y_pred: list[Label], y_proba: list[float]
) -> ClassificationMetrics:
    """Compute metrics from predictions.

    ``y_proba`` is the predicted probability of *each item's predicted*
    label (i.e. ``predict_proba(X)[i, class_index_of(y_pred[i])]``), used
    only for ``mean_confidence``.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true has {len(y_true)} items but y_pred has {len(y_pred)}")
    if len(y_true) != len(y_proba):
        raise ValueError(f"y_true has {len(y_true)} items but y_proba has {len(y_proba)}")
    if not y_true:
        raise ValueError("y_true must be non-empty")

    confusion: dict[tuple[Label, Label], int] = {
        (t, p): 0 for t in _LABELS for p in _LABELS
    }
    correct = 0
    for true_label, predicted_label in zip(y_true, y_pred, strict=True):
        confusion[(true_label, predicted_label)] += 1
        if predicted_label == true_label:
            correct += 1

    precision: dict[Label, float] = {}
    recall: dict[Label, float] = {}
    f1: dict[Label, float] = {}
    for label in _LABELS:
        true_positive = confusion[(label, label)]
        predicted_positive = sum(confusion[(t, label)] for t in _LABELS)
        actual_positive = sum(confusion[(label, p)] for p in _LABELS)
        precision[label] = true_positive / predicted_positive if predicted_positive else 0.0
        recall[label] = true_positive / actual_positive if actual_positive else 0.0
        f1[label] = (
            2 * precision[label] * recall[label] / (precision[label] + recall[label])
            if (precision[label] + recall[label])
            else 0.0
        )

    return ClassificationMetrics(
        accuracy=correct / len(y_true),
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=confusion,
        mean_confidence=sum(y_proba) / len(y_proba),
    )
