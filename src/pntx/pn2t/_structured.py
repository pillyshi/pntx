from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from ..backends.base import Backend

T = TypeVar("T", bound=BaseModel)

_MAX_ATTEMPTS = 2
_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def complete_structured(
    backend: Backend,
    system: str,
    user: str,
    response_model: type[T],
    *,
    temperature: float = 1.0,
    max_tokens: int = 4096,
) -> T:
    """Ask ``backend`` to complete a JSON object matching ``response_model`` and validate it.

    Builds one flat prompt (``system`` + ``user``, matching the rest of
    pntx's prompt-building convention of plain strings rather than chat
    messages) and asks the backend to complete it, then parses the result as
    ``response_model``. Retries once (with a fresh completion) on invalid
    JSON or a schema mismatch; raises the underlying ``ValidationError``/
    ``JSONDecodeError`` if the second attempt also fails, letting the
    caller's own retry loop (a fresh batch, different exemplars) take over.

    Unlike semaxis's ``LlamaCppClient.complete_structured`` (which uses
    llama.cpp's grammar-constrained JSON decoding), this works over any
    ``Backend`` via plain-text completion + validation, since ``pntx``'s
    ``Backend`` protocol intentionally has no structured-output method.
    """
    prompt = f"{system}\n\n{user}\n\nJSON:\n"
    last_error: Exception | None = None
    for _attempt in range(_MAX_ATTEMPTS):
        raw = backend.complete(prompt, temperature=temperature, max_tokens=max_tokens)
        try:
            return response_model.model_validate_json(_extract_json(raw))
        except (ValidationError, json.JSONDecodeError) as e:
            last_error = e
            continue
    assert last_error is not None
    raise last_error


def _extract_json(raw: str) -> str:
    """Best-effort extraction of a JSON object from a completion.

    Strips a ```json fenced block if present, otherwise falls back to the
    substring between the first ``{`` and the last ``}``.
    """
    match = _JSON_FENCE.search(raw)
    if match:
        return match.group(1)
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end < start:
        return raw
    return raw[start : end + 1]
