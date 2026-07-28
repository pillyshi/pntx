from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("llama_cpp")

import llama_cpp  # noqa: E402

from pntx.backends.llama import LlamaCppBackend  # noqa: E402


class _FakeLlama:
    def __init__(self, **kwargs: Any) -> None:
        self.init_kwargs = kwargs

    @classmethod
    def from_pretrained(cls, **kwargs: Any) -> _FakeLlama:
        instance = cls()
        instance.init_kwargs = {"from_pretrained": True, **kwargs}
        return instance


@pytest.fixture(autouse=True)
def _fake_llama(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(llama_cpp, "Llama", _FakeLlama)


def test_requires_model_path_or_repo_id() -> None:
    with pytest.raises(ValueError, match="model_path or repo_id"):
        LlamaCppBackend()


def test_rejects_both_model_path_and_repo_id() -> None:
    with pytest.raises(ValueError, match="model_path or repo_id"):
        LlamaCppBackend(model_path="model.gguf", repo_id="org/repo")


def test_model_path_constructs_llama_directly() -> None:
    backend = LlamaCppBackend(model_path="model.gguf", n_ctx=4096, flash_attn=True)

    kwargs = backend._llm.init_kwargs  # type: ignore[attr-defined]
    assert kwargs["model_path"] == "model.gguf"
    assert kwargs["n_ctx"] == 4096
    assert kwargs["flash_attn"] is True
    assert kwargs["logits_all"] is True
    assert "from_pretrained" not in kwargs


def test_repo_id_uses_from_pretrained() -> None:
    backend = LlamaCppBackend(repo_id="org/repo", filename="*q4_k_m.gguf", n_gpu_layers=-1)

    kwargs = backend._llm.init_kwargs  # type: ignore[attr-defined]
    assert kwargs["from_pretrained"] is True
    assert kwargs["repo_id"] == "org/repo"
    assert kwargs["filename"] == "*q4_k_m.gguf"
    assert kwargs["n_gpu_layers"] == -1
    assert kwargs["logits_all"] is True


def test_repo_id_without_filename_is_allowed() -> None:
    backend = LlamaCppBackend(repo_id="org/repo")

    kwargs = backend._llm.init_kwargs  # type: ignore[attr-defined]
    assert kwargs["filename"] is None


def test_fit_to_context_returns_tokens_unchanged_when_they_fit() -> None:
    backend = LlamaCppBackend(model_path="model.gguf")
    backend._llm.n_ctx = lambda: 10  # type: ignore[method-assign]

    tokens = [1, 2, 3]
    assert backend._fit_to_context(tokens, reserve=2) == tokens


def test_fit_to_context_trims_from_front_keeping_bos() -> None:
    backend = LlamaCppBackend(model_path="model.gguf")
    backend._llm.n_ctx = lambda: 5  # type: ignore[method-assign]

    tokens = [100, 1, 2, 3, 4, 5]  # 100 stands in for the leading BOS token
    with pytest.warns(UserWarning, match="exceeds the available context budget"):
        trimmed = backend._fit_to_context(tokens, reserve=1)

    # budget = n_ctx(5) - reserve(1) = 4: BOS kept, plus the last 3 tokens.
    assert trimmed == [100, 3, 4, 5]


def test_fit_to_context_raises_when_reserve_exceeds_context() -> None:
    backend = LlamaCppBackend(model_path="model.gguf")
    backend._llm.n_ctx = lambda: 5  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="leaves no room"):
        backend._fit_to_context([1, 2, 3], reserve=5)


def test_max_choice_tokens_returns_longest_choice_length() -> None:
    backend = LlamaCppBackend(model_path="model.gguf")

    def _tokenize(text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        return list(text)

    backend._llm.tokenize = _tokenize  # type: ignore[method-assign]

    assert backend._max_choice_tokens([" positive", " neg"]) == len(b" positive")


def test_max_choice_tokens_empty_choices_is_zero() -> None:
    backend = LlamaCppBackend(model_path="model.gguf")
    assert backend._max_choice_tokens([]) == 0


def test_complete_routes_through_chat_completion_as_a_single_user_message() -> None:
    """``complete``/``complete_json`` go through ``create_chat_completion``
    (not ``create_completion``) so the model's own chat template applies --
    without it, generation is more prone to never converging to a stop (see
    the repeat_penalty test below for the concrete failure mode this feeds
    into).
    """
    backend = LlamaCppBackend(model_path="model.gguf")
    backend._llm.n_ctx = lambda: 4096  # type: ignore[method-assign]
    backend._llm.tokenize = (  # type: ignore[method-assign]
        lambda text, add_bos=True, special=False: [1, 2, 3]
    )
    calls: list[dict[str, Any]] = []

    def _create_chat_completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    backend._llm.create_chat_completion = _create_chat_completion  # type: ignore[method-assign]

    result = backend.complete("prompt", max_tokens=64)

    assert result == "ok"
    assert calls[0]["messages"] == [{"role": "user", "content": "prompt"}]


def test_complete_applies_default_repeat_penalty() -> None:
    """``create_chat_completion``'s own default (``repeat_penalty=1.0``, i.e.
    no penalty) lets grammar-constrained generation collapse into repeating
    the same character forever instead of closing -- e.g. a JSON string that
    never finds its closing quote and runs to ``max_tokens``. Regression
    test for defaulting to llama.cpp's own CLI/server default (1.1) instead.
    """
    backend = LlamaCppBackend(model_path="model.gguf")
    backend._llm.n_ctx = lambda: 4096  # type: ignore[method-assign]
    backend._llm.tokenize = (  # type: ignore[method-assign]
        lambda text, add_bos=True, special=False: [1, 2, 3]
    )
    calls: list[dict[str, Any]] = []

    def _create_chat_completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    backend._llm.create_chat_completion = _create_chat_completion  # type: ignore[method-assign]

    backend.complete("prompt", max_tokens=64)

    assert calls[0]["repeat_penalty"] == 1.1


def test_complete_uses_custom_repeat_penalty() -> None:
    backend = LlamaCppBackend(model_path="model.gguf", repeat_penalty=1.3)
    backend._llm.n_ctx = lambda: 4096  # type: ignore[method-assign]
    backend._llm.tokenize = (  # type: ignore[method-assign]
        lambda text, add_bos=True, special=False: [1, 2, 3]
    )
    calls: list[dict[str, Any]] = []

    def _create_chat_completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "ok"}}]}

    backend._llm.create_chat_completion = _create_chat_completion  # type: ignore[method-assign]

    backend.complete("prompt", max_tokens=64)

    assert calls[0]["repeat_penalty"] == 1.3


def test_repeat_penalty_not_forwarded_to_llama_constructor() -> None:
    backend = LlamaCppBackend(model_path="model.gguf", repeat_penalty=1.3)
    assert "repeat_penalty" not in backend._llm.init_kwargs  # type: ignore[attr-defined]


def test_fit_prompt_to_context_returns_prompt_unchanged_when_it_fits() -> None:
    backend = LlamaCppBackend(model_path="model.gguf")
    backend._llm.n_ctx = lambda: 200  # type: ignore[method-assign]
    backend._llm.tokenize = (  # type: ignore[method-assign]
        lambda text, add_bos=True, special=False: list(text)
    )

    assert backend._fit_prompt_to_context("short prompt", reserve=10) == "short prompt"


def test_fit_prompt_to_context_trims_from_front_and_detokenizes() -> None:
    backend = LlamaCppBackend(model_path="model.gguf")
    # budget = n_ctx(100) - reserve(10) - _CHAT_TEMPLATE_OVERHEAD(64) = 26
    backend._llm.n_ctx = lambda: 100  # type: ignore[method-assign]
    backend._llm.tokenize = (  # type: ignore[method-assign]
        lambda text, add_bos=True, special=False: list(text)
    )
    backend._llm.detokenize = (  # type: ignore[method-assign]
        lambda tokens, prev_tokens=None, special=False: bytes(tokens)
    )

    prompt = "x" * 30
    with pytest.warns(UserWarning, match="exceeds the available context budget"):
        trimmed = backend._fit_prompt_to_context(prompt, reserve=10)

    assert trimmed == "x" * 26


def test_fit_prompt_to_context_raises_when_reserve_and_overhead_exceed_context() -> None:
    backend = LlamaCppBackend(model_path="model.gguf")
    backend._llm.n_ctx = lambda: 50  # type: ignore[method-assign]

    with pytest.raises(ValueError, match="leaves no room"):
        backend._fit_prompt_to_context("prompt", reserve=10)


class _FakeGrammar:
    def __init__(self, schema_json: str, verbose: bool = True) -> None:
        self.schema_json = schema_json


def test_complete_json_constrains_decoding_with_grammar_from_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = LlamaCppBackend(model_path="model.gguf")
    backend._llm.n_ctx = lambda: 4096  # type: ignore[method-assign]
    backend._llm.tokenize = (  # type: ignore[method-assign]
        lambda text, add_bos=True, special=False: [1, 2, 3]
    )

    monkeypatch.setattr(
        llama_cpp,
        "LlamaGrammar",
        type("_LlamaGrammar", (), {"from_json_schema": staticmethod(_FakeGrammar)}),
    )

    calls: list[dict[str, Any]] = []

    def _create_chat_completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": '{"x": 1}'}}]}

    backend._llm.create_chat_completion = _create_chat_completion  # type: ignore[method-assign]

    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    result = backend.complete_json("prompt", schema=schema, max_tokens=64)

    assert result == '{"x": 1}'
    grammar = calls[0]["grammar"]
    assert isinstance(grammar, _FakeGrammar)
    assert grammar.schema_json == json.dumps(schema)
    assert calls[0]["max_tokens"] == 64


def test_complete_json_inlines_refs_before_building_grammar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """pydantic emits ``$ref``/``$defs`` for nested ``BaseModel`` fields (e.g.
    ``pn2t``'s ``list[HardPositive]``), and llama.cpp's schema-to-grammar
    converter doesn't reliably resolve ``$ref`` in every version. Left
    unresolved, the referenced item type ends up under-constrained in the
    grammar -- notably, list fields lose their element schema -- which lets
    generation run to ``max_tokens`` instead of converging. Regression test
    for that: the schema handed to ``LlamaGrammar.from_json_schema`` must
    have every ``$ref`` inlined and no leftover ``$defs``.
    """
    backend = LlamaCppBackend(model_path="model.gguf")
    backend._llm.n_ctx = lambda: 4096  # type: ignore[method-assign]
    backend._llm.tokenize = (  # type: ignore[method-assign]
        lambda text, add_bos=True, special=False: [1, 2, 3]
    )

    monkeypatch.setattr(
        llama_cpp,
        "LlamaGrammar",
        type("_LlamaGrammar", (), {"from_json_schema": staticmethod(_FakeGrammar)}),
    )
    calls: list[dict[str, Any]] = []

    def _create_chat_completion(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        return {"choices": [{"message": {"content": "{}"}}]}

    backend._llm.create_chat_completion = _create_chat_completion  # type: ignore[method-assign]

    schema = {
        "$defs": {
            "Item": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            }
        },
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"$ref": "#/$defs/Item"}}},
        "required": ["items"],
    }
    backend.complete_json("prompt", schema=schema, max_tokens=64)

    grammar = calls[0]["grammar"]
    assert isinstance(grammar, _FakeGrammar)
    resolved_schema = json.loads(grammar.schema_json)
    assert "$defs" not in resolved_schema
    assert resolved_schema["properties"]["items"]["items"] == {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }


def test_count_tokens_returns_token_count_without_bos() -> None:
    backend = LlamaCppBackend(model_path="model.gguf")
    calls: list[tuple[bytes, bool]] = []

    def _tokenize(text: bytes, add_bos: bool = True, special: bool = False) -> list[int]:
        calls.append((text, add_bos))
        return list(text)

    backend._llm.tokenize = _tokenize  # type: ignore[method-assign]

    assert backend.count_tokens("hello") == len(b"hello")
    assert calls == [(b"hello", False)]
