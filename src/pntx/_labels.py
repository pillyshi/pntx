from __future__ import annotations

from collections.abc import Iterable
from numbers import Real
from typing import Any

from .types import NEGATIVE, POSITIVE

__all__ = ["resolve_binary_labels"]


def resolve_binary_labels(y: Iterable[Any], *, pos_label: Any = None) -> tuple[Any, Any]:
    """Resolve which of the two distinct values in ``y`` means "positive".

    Shared by every ``t2pn``/``pn2t`` component that consumes a binary
    ``y`` (``t2pn.LLMPromptingClassifier``, ``t2pn.FineTuningClassifier``,
    ``pn2t.OverSampler``, ``pn2t.SyntheticSampler``), so all four resolve
    label encodings the same way instead of each hardcoding its own rule.

    Resolution order:

    1. ``pos_label``, if given, wins outright: it must be one of the two
       values in ``y``.
    2. If both values are numbers (``int``/``float``/``bool``), the greater
       one is positive (e.g. ``{0, 1}`` -> ``1``, ``{-1, 1}`` -> ``1``).
    3. If the two values are exactly ``"positive"``/``"negative"``
       (``pntx.types.POSITIVE``/``NEGATIVE``), they're used directly.
    4. Otherwise resolution is ambiguous -- ``pos_label`` must be passed
       explicitly (e.g. arbitrary string pairs like ``{"spam", "ham"}``,
       where no encoding-level convention says which one is "positive").

    Returns:
        ``(negative_label, positive_label)``, in that order (so
        ``resolve_binary_labels(...)`` unpacks the same way
        ``sorted(set(y))`` used to for the numeric/positive-negative cases,
        but generalizes correctly when ``pos_label`` picks a value that
        wouldn't sort last).

    Raises:
        ValueError: if ``y`` doesn't contain exactly 2 distinct values, if
            ``pos_label`` isn't one of them, or if resolution is ambiguous
            without an explicit ``pos_label``.
    """
    classes = set(y)
    if len(classes) != 2:
        raise ValueError(
            f"y must contain exactly 2 classes, got {len(classes)}: {sorted(classes, key=repr)}"
        )

    if pos_label is not None:
        if pos_label not in classes:
            raise ValueError(f"pos_label={pos_label!r} is not one of the classes in y: {classes}")
        negative_label = next(c for c in classes if c != pos_label)
        return negative_label, pos_label

    if all(isinstance(c, Real) for c in classes):
        return min(classes), max(classes)

    if classes == {POSITIVE, NEGATIVE}:
        return NEGATIVE, POSITIVE

    raise ValueError(
        f"cannot infer which of {sorted(classes, key=repr)} is the positive class; "
        "pass pos_label explicitly to say which one"
    )
