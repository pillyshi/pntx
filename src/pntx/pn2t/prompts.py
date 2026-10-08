from __future__ import annotations

import json

from ._types import CounterfactualBatch, HardPositiveGenerationResult, SyntheticGenerationResult

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

    The schema is spelled out in the prompt regardless of backend: for
    backends without grammar-constrained decoding it's the only thing
    keeping output on-schema (validated, with a retry, after the fact -- see
    ``pn2t._structured``); for backends that do constrain decoding (e.g.
    ``LlamaCppBackend``), the grammar only enforces JSON syntax, not field
    semantics or the requested item count, so the prompt still carries that.
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
    return SYNTHETIC_SYSTEM.format(schema=json.dumps(SyntheticGenerationResult.model_json_schema()))


def build_synthetic_user_message(
    pos_texts: list[str],
    n_synthesized: int,
    language: str | None = None,
) -> str:
    """Render the user message for a ``TypicalPositiveOverSampler`` batch.

    Unlike ``build_user_message``, this takes only positive exemplars --
    negatives are deliberately excluded to avoid steering generation toward
    a boundary/adversarial framing (see ``TypicalPositiveOverSampler``'s docstring).
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


COUNTERFACTUAL_SYSTEM = """\
You are a counterfactual data editing expert.

You are given reference examples of the Positive class and a numbered list of \
Negative texts. For each Negative text, make the smallest edit that makes the \
Positive label clearly apply to it.

Rules (counterfactual revision, after Kaushik et al. 2020):
- The edited text must clearly belong to the Positive class, as the Positive \
reference examples define it
- The edited text must stay coherent and natural
- Make no unnecessary changes: keep everything not needed to change the label \
(topic, entities, length, style, wording) exactly as it is
- Keep the original text's language
- Do not copy from the Positive reference examples; they only show what \
Positive means

Typical minimal edits include: replacing or inserting modifiers, negating or \
removing a negation, recasting a fact as hoped-for (or the reverse), \
diminishing via qualifiers, adding or removing sarcasm, changing a stated \
rating or outcome.

Return exactly one edit per Negative text. pivot_id is the number of the \
Negative text the edit is based on. edited_text is the complete edited text \
and nothing else -- not the original, no arrows, no explanation. \
changed_spans lists each change briefly as "original -> edited". \
is_positive is your honest judgment of whether the edited text now clearly \
belongs to the Positive class; set it to false if a small edit could not \
achieve that.
Respond with a single JSON object matching this schema and nothing else \
(no prose, no markdown code fences):

{schema}"""

_COUNTERFACTUAL_USER_TEMPLATE = """\
Positive reference examples:
{positive_list}

Negative texts to edit:
{pivot_list}

Count: {n_edits}"""


def build_counterfactual_system_message() -> str:
    """Render ``COUNTERFACTUAL_SYSTEM`` with ``CounterfactualBatch``'s JSON schema inlined."""
    return COUNTERFACTUAL_SYSTEM.format(schema=json.dumps(CounterfactualBatch.model_json_schema()))


def build_counterfactual_user_message(pos_texts: list[str], pivots: list[str]) -> str:
    """Render the user message for a ``CounterfactualOverSampler`` batch.

    ``pivots`` are the negative texts to edit, numbered from 0 as
    ``[i]`` so the response's ``pivot_id`` can refer back to them;
    ``pos_texts`` are shown only as a reference for what "positive" means
    (they are never edited).
    """
    positive_list = "\n".join(f"- {t}" for t in pos_texts) if pos_texts else "(none)"
    pivot_list = "\n".join(f"[{i}] {t}" for i, t in enumerate(pivots))
    return _COUNTERFACTUAL_USER_TEMPLATE.format(
        positive_list=positive_list,
        pivot_list=pivot_list,
        n_edits=len(pivots),
    )
