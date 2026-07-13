from __future__ import annotations

from collections.abc import Iterable
from importlib import import_module
from typing import Any

from .backends.base import Backend
from .selection import RandomSelector, Selector
from .types import ClassifyResult, Label, Pair

_BACKEND_REGISTRY: dict[str, tuple[str, str]] = {
    "llama": ("pntx.backends.llama", "LlamaCppBackend"),
    "anthropic": ("pntx.backends.anthropic", "AnthropicBackend"),
}


def _resolve_backend(backend: Backend | str, backend_kwargs: dict[str, Any]) -> Backend:
    if not isinstance(backend, str):
        if backend_kwargs:
            raise TypeError(
                "backend_kwargs are only used when backend is given as a string "
                "(e.g. PNTX(backend='llama', model_path=...)); construct the "
                "Backend instance directly instead"
            )
        return backend
    try:
        module_name, class_name = _BACKEND_REGISTRY[backend]
    except KeyError:
        raise ValueError(
            f"Unknown backend {backend!r}; expected a Backend instance or one "
            f"of {sorted(_BACKEND_REGISTRY)}"
        ) from None
    try:
        module = import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Backend {backend!r} requires the optional '{backend}' extra. "
            f"Install it with: pip install 'pntx[{backend}]'"
        ) from e
    return getattr(module, class_name)(**backend_kwargs)  # type: ignore[no-any-return]


class PNTX:
    """Generates and classifies text from user-defined (positive, negative) pairs.

    The pairs' meaning is entirely up to the caller (sentiment, formality,
    policy compliance, ...); this class never interprets it, it only uses the
    pairs as few-shot/scoring material.
    """

    def __init__(
        self,
        backend: Backend | str,
        *,
        selector: Selector | None = None,
        **backend_kwargs: Any,
    ) -> None:
        """Create a PNTX model.

        ``backend`` is either a ready-made ``Backend`` instance, or the name
        of a built-in backend (e.g. ``"llama"``) to construct lazily; any
        ``backend_kwargs`` are then forwarded to that backend's constructor
        (e.g. ``PNTX(backend="llama", model_path="model.gguf")``).
        """
        self.backend = _resolve_backend(backend, backend_kwargs)
        self.selector: Selector = selector if selector is not None else RandomSelector()
        self._pairs: list[Pair] = []

    @property
    def pairs(self) -> list[Pair]:
        return list(self._pairs)

    def fit(self, pairs: Iterable[Pair]) -> PNTX:
        """Store the (positive, negative) pairs used as generation/classification material.

        This only stores and validates ``pairs``; no training happens here.
        """
        pairs = list(pairs)
        if not pairs:
            raise ValueError("pairs must be non-empty")
        self._pairs = pairs
        return self

    def generate(
        self,
        n: int,
        side: Label,
        *,
        temperature: float = 1.0,
        dedup: bool = True,
        verify: bool = True,
        min_confidence: float = 0.8,
    ) -> list[str]:
        self._check_fitted()
        raise NotImplementedError("PNTX.generate() is not implemented yet")

    def classify(self, text: str) -> ClassifyResult:
        self._check_fitted()
        raise NotImplementedError("PNTX.classify() is not implemented yet")

    def classify_batch(self, texts: list[str]) -> list[ClassifyResult]:
        self._check_fitted()
        raise NotImplementedError("PNTX.classify_batch() is not implemented yet")

    def _check_fitted(self) -> None:
        if not self._pairs:
            raise RuntimeError("PNTX.fit(pairs) must be called before this method")
