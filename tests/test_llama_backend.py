from __future__ import annotations

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
    backend = LlamaCppBackend(
        repo_id="org/repo", filename="*q4_k_m.gguf", n_gpu_layers=-1
    )

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
