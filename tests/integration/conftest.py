from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pntx.backends.llama import LlamaCppBackend

MODEL_PATH_ENV_VAR = "PNTX_LLAMA_MODEL_PATH"


@pytest.fixture
def llama_backend() -> Iterator[LlamaCppBackend]:
    pytest.importorskip("llama_cpp")
    model_path = os.environ.get(MODEL_PATH_ENV_VAR)
    if not model_path or not Path(model_path).is_file():
        pytest.skip(f"set {MODEL_PATH_ENV_VAR} to a local .gguf model file to run this test")

    from pntx.backends.llama import LlamaCppBackend as _LlamaCppBackend

    yield _LlamaCppBackend(model_path=model_path, n_ctx=4096, verbose=False)
