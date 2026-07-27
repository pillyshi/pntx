from __future__ import annotations

from importlib import import_module
from typing import Any

from .backends.base import Backend

_BACKEND_REGISTRY: dict[str, tuple[str, str]] = {
    "llama": ("pntx.backends.llama", "LlamaCppBackend"),
}


def resolve_backend(backend: Backend | str, backend_kwargs: dict[str, Any] | None) -> Backend:
    """Resolve ``backend`` (an instance, or the name of a built-in backend) to a ``Backend``.

    ``backend_kwargs`` are only used -- and forwarded to the backend's
    constructor -- when ``backend`` is given as a string (e.g.
    ``resolve_backend("llama", {"model_path": ...})``); passing them together
    with an already-constructed ``Backend`` instance is a ``TypeError``.
    """
    if not isinstance(backend, str):
        if backend_kwargs:
            raise TypeError(
                "backend_kwargs are only used when backend is given as a string "
                "(e.g. backend='llama', backend_kwargs={'model_path': ...}); "
                "construct the Backend instance directly instead"
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
    return getattr(module, class_name)(**(backend_kwargs or {}))  # type: ignore[no-any-return]
