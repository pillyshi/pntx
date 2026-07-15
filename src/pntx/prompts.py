from __future__ import annotations

import itertools
import re

from .types import NEGATIVE, POSITIVE, Label

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


def build_exemplar_prefix(positive: list[str], negative: list[str]) -> str:
    """Render the few-shot block for ``positive``/``negative`` exemplars,
    interleaved (positive[0], negative[0], positive[1], ...; whichever side
    runs out first just stops contributing lines). This is the part of the
    classification prompt shared across every query, so it should come
    first (see ``build_query_suffix``) to keep it a stable KV-cache prefix.
    """
    lines = []
    for pos, neg in itertools.zip_longest(positive, negative):
        if pos is not None:
            lines.append(_EXEMPLAR_TEMPLATE.format(text=pos, label=POSITIVE))
        if neg is not None:
            lines.append(_EXEMPLAR_TEMPLATE.format(text=neg, label=NEGATIVE))
    return "".join(lines)


def build_query_suffix(text: str) -> str:
    """Render the query-specific tail of the classification prompt."""
    return _QUERY_TEMPLATE.format(text=text)


def build_classify_prompt(positive: list[str], negative: list[str], query: str) -> str:
    """Render the full classification prompt for a single ``query``."""
    return build_exemplar_prefix(positive, negative) + build_query_suffix(query)


def classify_choice_texts() -> list[str]:
    """``CLASSIFY_LABEL_CHOICES`` values in ``CLASSIFY_LABELS`` order."""
    return [CLASSIFY_LABEL_CHOICES[label] for label in CLASSIFY_LABELS]


CLASSIFY_COMPLETION_MAX_TOKENS = 8
"""max_tokens for the parse-based classify path: just enough for a bare label."""


def parse_classify_label(raw: str) -> tuple[Label, float]:
    """Parse a freeform completion of ``build_classify_prompt`` into a label.

    Used for backends without ``score_choices`` (e.g. ``AnthropicBackend``):
    rather than comparing log-likelihoods, we ask the model to name the
    label and look for it in the response text.

    Confidence here is *not* a calibrated probability, just a fixed
    convention value: ``1.0`` when exactly one label word was found, ``0.5``
    when the response was ambiguous (both or neither label word present) and
    we fell back to a default guess.
    """
    lowered = raw.lower()
    positive_at = lowered.find(POSITIVE)
    negative_at = lowered.find(NEGATIVE)
    found_positive = positive_at != -1
    found_negative = negative_at != -1

    if found_positive and not found_negative:
        return POSITIVE, 1.0
    if found_negative and not found_positive:
        return NEGATIVE, 1.0
    if found_positive and found_negative:
        return (POSITIVE if positive_at < negative_at else NEGATIVE), 0.5
    return POSITIVE, 0.5  # neither label word found; arbitrary fallback guess


_GENERATE_PREAMBLE = (
    'Below are example pairs contrasting "positive" and "negative" texts; '
    "their exact meaning is defined only by these examples.\n\n"
)
_GENERATE_EXEMPLAR_LINE = "- {label}: {text}\n"
_GENERATE_INSTRUCTION = (
    "\nWrite {n} new, diverse {side} texts. Do not copy or closely paraphrase "
    "the examples above. Output one text per line, numbered starting at 1, "
    "with no other commentary.\n\n1. "
)

_NUMBERED_LINE = re.compile(r"^\s*\d+[.)]\s*(.+)$")


def build_generate_prompt(positive: list[str], negative: list[str], side: Label, n: int) -> str:
    """Render the generation prompt asking for ``n`` new ``side`` texts.

    ``positive``/``negative`` exemplars are interleaved the same way as
    ``build_exemplar_prefix``. Ends primed with ``"1. "`` so a plain
    text-completion backend continues directly into the first generated item.
    """
    lines = []
    for pos, neg in itertools.zip_longest(positive, negative):
        if pos is not None:
            lines.append(_GENERATE_EXEMPLAR_LINE.format(label=POSITIVE, text=pos))
        if neg is not None:
            lines.append(_GENERATE_EXEMPLAR_LINE.format(label=NEGATIVE, text=neg))
    return _GENERATE_PREAMBLE + "".join(lines) + _GENERATE_INSTRUCTION.format(n=n, side=side)


def parse_generated_texts(raw: str) -> list[str]:
    """Parse a numbered-list completion (continuing after a primed ``"1. "``)
    into a flat list of generated texts.

    This is a best-effort heuristic, not a strict parser: lines that aren't
    numbered (e.g. a wrapped continuation of the previous item) are kept
    as-is rather than dropped, since backends aren't guaranteed to always
    produce clean single-line items.
    """
    lines = [line for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return []
    texts = [lines[0].strip()]
    for line in lines[1:]:
        match = _NUMBERED_LINE.match(line)
        texts.append(match.group(1).strip() if match else line.strip())
    return [text for text in texts if text]
