from __future__ import annotations

from pntx import prompts

from .conftest import SAMPLE_NEGATIVE, SAMPLE_POSITIVE


def test_build_exemplar_prefix_includes_every_text_with_its_label() -> None:
    prefix = prompts.build_exemplar_prefix(SAMPLE_POSITIVE, SAMPLE_NEGATIVE)
    for pos in SAMPLE_POSITIVE:
        assert f"Text: {pos}\nLabel: positive" in prefix
    for neg in SAMPLE_NEGATIVE:
        assert f"Text: {neg}\nLabel: negative" in prefix


def test_build_exemplar_prefix_with_positive_only_pool_has_no_negative_lines() -> None:
    prefix = prompts.build_exemplar_prefix(SAMPLE_POSITIVE, [])
    assert "Label: negative" not in prefix
    for pos in SAMPLE_POSITIVE:
        assert f"Text: {pos}\nLabel: positive" in prefix


def test_build_exemplar_prefix_with_unequal_pool_sizes_keeps_every_text() -> None:
    prefix = prompts.build_exemplar_prefix(SAMPLE_POSITIVE, SAMPLE_NEGATIVE[:1])
    assert prefix.count("Label: positive") == len(SAMPLE_POSITIVE)
    assert prefix.count("Label: negative") == 1


def test_build_query_suffix_has_no_trailing_label() -> None:
    suffix = prompts.build_query_suffix("some query")
    assert suffix == "Text: some query\nLabel:"


def test_build_classify_prompt_is_prefix_then_suffix() -> None:
    prompt = prompts.build_classify_prompt(SAMPLE_POSITIVE, SAMPLE_NEGATIVE, "some query")
    assert prompt == prompts.build_exemplar_prefix(
        SAMPLE_POSITIVE, SAMPLE_NEGATIVE
    ) + prompts.build_query_suffix("some query")


def test_classify_choice_texts_matches_label_order() -> None:
    choices = prompts.classify_choice_texts()
    assert choices == [
        prompts.CLASSIFY_LABEL_CHOICES[label] for label in prompts.CLASSIFY_LABELS
    ]
    assert len(choices) == 2


def test_build_generate_prompt_includes_exemplars_and_is_primed_for_item_one() -> None:
    prompt = prompts.build_generate_prompt(SAMPLE_POSITIVE, SAMPLE_NEGATIVE, "positive", 5)
    for pos in SAMPLE_POSITIVE:
        assert f"positive: {pos}" in prompt
    for neg in SAMPLE_NEGATIVE:
        assert f"negative: {neg}" in prompt
    assert "5" in prompt
    assert "positive texts" in prompt
    assert prompt.endswith("1. ")


def test_build_generate_prompt_with_negative_only_pool_has_no_positive_exemplars() -> None:
    prompt = prompts.build_generate_prompt([], SAMPLE_NEGATIVE, "negative", 3)
    assert "- positive:" not in prompt
    for neg in SAMPLE_NEGATIVE:
        assert f"negative: {neg}" in prompt


def test_parse_generated_texts_handles_numbered_list() -> None:
    raw = "first item\n2. second item\n3. third item"
    assert prompts.parse_generated_texts(raw) == ["first item", "second item", "third item"]


def test_parse_generated_texts_accepts_parenthesized_numbering() -> None:
    raw = "first item\n2) second item"
    assert prompts.parse_generated_texts(raw) == ["first item", "second item"]


def test_parse_generated_texts_ignores_blank_lines() -> None:
    raw = "first item\n\n2. second item\n\n"
    assert prompts.parse_generated_texts(raw) == ["first item", "second item"]


def test_parse_generated_texts_empty_input_returns_empty_list() -> None:
    assert prompts.parse_generated_texts("") == []
    assert prompts.parse_generated_texts("   \n  ") == []


def test_parse_generated_texts_single_item_no_numbering() -> None:
    assert prompts.parse_generated_texts("just one item") == ["just one item"]
