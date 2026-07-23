from __future__ import annotations

import json

from ._types import HardPositiveGenerationResult, SyntheticGenerationResult

SYSTEM = """\
You are a data augmentation expert.

Analyze the provided Positive and Negative samples and generate new texts \
that belong to the Positive class.

Rules:
- No paraphrases of existing positive samples
- Preserve the essential features of the Positive class
- Extract only the essential Positive-class criteria; exclude proper nouns \
and coincidental commonalities
- Prioritize texts that experts would label Positive but that simple \
rule-based classifiers or untrained humans might label Negative
- Maximize diversity; avoid duplicating existing samples

Before generating, analyze positive_features, negative_features, and \
boundary_features. Each generated text must differ in situation, \
expression, and context.
boundary_features.importance is in [0.0, 1.0]; higher = more critical for \
the Positive/Negative boundary.
hard_positives must contain exactly the requested count.
Respond with a single JSON object matching this schema and nothing else \
(no prose, no markdown code fences):

{schema}"""

_USER_TEMPLATE = """\
Positive:
{positive_list}

Negative:
{negative_list}

Count: {n_synthesized_texts}"""

_LANGUAGE_INSTRUCTION = """\

Language constraint:
- Write only hard_positives[].text in {language}.
- Do not force positive_features, negative_features, boundary_features, \
positive_evidence, or confusing_evidence to be written in {language}."""


def build_system_message() -> str:
    """Render ``SYSTEM`` with ``HardPositiveGenerationResult``'s JSON schema inlined.

    Unlike semaxis (which relies on the LLM client enforcing the schema via
    grammar-constrained decoding), pntx's ``Backend`` protocol only exposes
    plain text completion, so the schema is spelled out in the prompt itself
    and the response is validated (with a retry) after the fact -- see
    ``pn2t._structured``.
    """
    return SYSTEM.format(schema=json.dumps(HardPositiveGenerationResult.model_json_schema()))


def build_user_message(
    pos_texts: list[str],
    neg_texts: list[str],
    n_synthesized: int,
    language: str | None = None,
) -> str:
    positive_list = (
        "\n".join(f"{i + 1}. {t}" for i, t in enumerate(pos_texts)) if pos_texts else "(none)"
    )
    negative_list = (
        "\n".join(f"{i + 1}. {t}" for i, t in enumerate(neg_texts)) if neg_texts else "(none)"
    )
    message = _USER_TEMPLATE.format(
        positive_list=positive_list,
        negative_list=negative_list,
        n_synthesized_texts=n_synthesized,
    )
    if language:
        message += _LANGUAGE_INSTRUCTION.format(language=language)
    return message


SYNTHETIC_SYSTEM = """\
You are a privacy-preserving synthetic data generation expert.

Analyze the provided Positive samples and generate new texts that belong to \
the Positive class.

Rules:
- Generate typical, ordinary, representative Positive-class texts -- NOT \
boundary or edge cases
- No paraphrases of existing positive samples; each text must describe an \
independent new situation
- Preserve style, register, topic, and structure
- Generalize away every specific identifying detail: proper nouns, personal \
names, brand/organization names, exact dates, numbers, amounts, and \
locations, and any phrase copied verbatim from a sample. Where a detail is \
essential to the meaning, replace it with a generic category rather than \
dropping it silently
- Maximize diversity; avoid duplicating existing samples

Before generating, analyze style_features and content_features -- describe \
only general patterns across the samples, never quote or restate a specific \
identifying detail from any one sample.
For each generated text, generalized_from must list the *categories* of \
details you removed or generalized (e.g. "removed a brand name", "replaced \
an exact date with a relative one") and must NEVER quote or restate the \
actual original identifying content -- that would leak the very information \
this field exists to keep out.
synthetic_texts must contain exactly the requested count.
Respond with a single JSON object matching this schema and nothing else \
(no prose, no markdown code fences):

{schema}"""

_SYNTHETIC_USER_TEMPLATE = """\
Positive:
{positive_list}

Count: {n_synthesized_texts}"""

_SYNTHETIC_LANGUAGE_INSTRUCTION = """\

Language constraint:
- Write only synthetic_texts[].text in {language}.
- Do not force style_features, content_features, or generalized_from to be \
written in {language}."""


def build_synthetic_system_message() -> str:
    """Render ``SYNTHETIC_SYSTEM`` with ``SyntheticGenerationResult``'s JSON schema inlined."""
    return SYNTHETIC_SYSTEM.format(
        schema=json.dumps(SyntheticGenerationResult.model_json_schema())
    )


def build_synthetic_user_message(
    pos_texts: list[str],
    n_synthesized: int,
    language: str | None = None,
) -> str:
    """Render the user message for a ``SyntheticSampler`` batch.

    Unlike ``build_user_message``, this takes only positive exemplars --
    negatives are deliberately excluded to avoid steering generation toward
    a boundary/adversarial framing (see ``SyntheticSampler``'s docstring).
    """
    positive_list = (
        "\n".join(f"{i + 1}. {t}" for i, t in enumerate(pos_texts)) if pos_texts else "(none)"
    )
    message = _SYNTHETIC_USER_TEMPLATE.format(
        positive_list=positive_list,
        n_synthesized_texts=n_synthesized,
    )
    if language:
        message += _SYNTHETIC_LANGUAGE_INSTRUCTION.format(language=language)
    return message
