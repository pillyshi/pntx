from __future__ import annotations

from .types import NEGATIVE, POSITIVE, Label, Pair

CLASSIFY_LABELS: list[Label] = [POSITIVE, NEGATIVE]
"""Fixed label order used everywhere a classify score list is produced."""

CLASSIFY_LABEL_CHOICES: dict[Label, str] = {
    POSITIVE: " positive",
    NEGATIVE: " negative",
}
"""Text appended after the prompt to score each label (leading space matters
for most tokenizers: it keeps the label as its own token(s) rather than
merging with the preceding colon)."""

_EXEMPLAR_TEMPLATE = "Text: {text}\nLabel: {label}\n\n"
_QUERY_TEMPLATE = "Text: {text}\nLabel:"


def build_exemplar_prefix(pairs: list[Pair]) -> str:
    """Render the few-shot block for ``pairs``. This is the part of the
    classification prompt shared across every query, so it should come
    first (see ``build_query_suffix``) to keep it a stable KV-cache prefix.
    """
    return "".join(
        _EXEMPLAR_TEMPLATE.format(text=pos, label=POSITIVE)
        + _EXEMPLAR_TEMPLATE.format(text=neg, label=NEGATIVE)
        for pos, neg in pairs
    )


def build_query_suffix(text: str) -> str:
    """Render the query-specific tail of the classification prompt."""
    return _QUERY_TEMPLATE.format(text=text)


def build_classify_prompt(pairs: list[Pair], query: str) -> str:
    """Render the full classification prompt for a single ``query``."""
    return build_exemplar_prefix(pairs) + build_query_suffix(query)


def classify_choice_texts() -> list[str]:
    """``CLASSIFY_LABEL_CHOICES`` values in ``CLASSIFY_LABELS`` order."""
    return [CLASSIFY_LABEL_CHOICES[label] for label in CLASSIFY_LABELS]
