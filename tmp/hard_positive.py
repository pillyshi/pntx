from __future__ import annotations

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
Output strict JSON matching the schema; no prose."""

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


def build_user_message(
    pos_texts: list[str],
    neg_texts: list[str],
    n_synthesized: int,
    language: str | None = None,
) -> str:
    positive_list = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(pos_texts)) if pos_texts else "(none)"
    negative_list = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(neg_texts)) if neg_texts else "(none)"
    message = _USER_TEMPLATE.format(
        positive_list=positive_list,
        negative_list=negative_list,
        n_synthesized_texts=n_synthesized,
    )
    if language:
        message += _LANGUAGE_INSTRUCTION.format(language=language)
    return message
