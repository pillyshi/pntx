from __future__ import annotations

from pntx import prompts

from .conftest import SAMPLE_PAIRS


def test_build_exemplar_prefix_includes_every_pair_with_its_label() -> None:
    prefix = prompts.build_exemplar_prefix(SAMPLE_PAIRS)
    for pos, neg in SAMPLE_PAIRS:
        assert f"Text: {pos}\nLabel: positive" in prefix
        assert f"Text: {neg}\nLabel: negative" in prefix


def test_build_query_suffix_has_no_trailing_label() -> None:
    suffix = prompts.build_query_suffix("some query")
    assert suffix == "Text: some query\nLabel:"


def test_build_classify_prompt_is_prefix_then_suffix() -> None:
    prompt = prompts.build_classify_prompt(SAMPLE_PAIRS, "some query")
    assert prompt == prompts.build_exemplar_prefix(SAMPLE_PAIRS) + prompts.build_query_suffix(
        "some query"
    )


def test_classify_choice_texts_matches_label_order() -> None:
    choices = prompts.classify_choice_texts()
    assert choices == [
        prompts.CLASSIFY_LABEL_CHOICES[label] for label in prompts.CLASSIFY_LABELS
    ]
    assert len(choices) == 2
