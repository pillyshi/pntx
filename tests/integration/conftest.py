from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pntx.backends.anthropic import AnthropicBackend
    from pntx.backends.llama import LlamaCppBackend

MODEL_PATH_ENV_VAR = "PNTX_LLAMA_MODEL_PATH"
ANTHROPIC_MODEL_ENV_VAR = "PNTX_ANTHROPIC_MODEL"
_DEFAULT_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"


@pytest.fixture
def llama_backend() -> Iterator[LlamaCppBackend]:
    pytest.importorskip("llama_cpp")
    model_path = os.environ.get(MODEL_PATH_ENV_VAR)
    if not model_path or not Path(model_path).is_file():
        pytest.skip(f"set {MODEL_PATH_ENV_VAR} to a local .gguf model file to run this test")

    from pntx.backends.llama import LlamaCppBackend as _LlamaCppBackend

    yield _LlamaCppBackend(model_path=model_path, n_ctx=512, verbose=False)


@pytest.fixture
def anthropic_backend() -> Iterator[AnthropicBackend]:
    pytest.importorskip("anthropic")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("set ANTHROPIC_API_KEY to run this test (this calls the real Anthropic API)")

    from pntx.backends.anthropic import AnthropicBackend as _AnthropicBackend

    model = os.environ.get(ANTHROPIC_MODEL_ENV_VAR, _DEFAULT_ANTHROPIC_MODEL)
    yield _AnthropicBackend(model=model)
