from __future__ import annotations

import json

from pntx.pn2t import prompts
from pntx.pn2t._types import HardPositiveGenerationResult


def test_build_system_message_inlines_json_schema() -> None:
    message = prompts.build_system_message()
    schema = HardPositiveGenerationResult.model_json_schema()
    assert json.dumps(schema) in message
    assert "hard_positives" in message


def test_build_user_message_lists_positive_and_negative_texts() -> None:
    message = prompts.build_user_message(["p1", "p2"], ["n1"], n_synthesized=3)
    assert "1. p1" in message
    assert "2. p2" in message
    assert "1. n1" in message
    assert "Count: 3" in message


def test_build_user_message_handles_empty_pools() -> None:
    message = prompts.build_user_message([], [], n_synthesized=1)
    assert "(none)" in message


def test_build_user_message_appends_language_constraint_when_given() -> None:
    without = prompts.build_user_message(["p1"], ["n1"], n_synthesized=1)
    with_lang = prompts.build_user_message(["p1"], ["n1"], n_synthesized=1, language="Japanese")
    assert "Language constraint" not in without
    assert "Language constraint" in with_lang
    assert "Japanese" in with_lang
