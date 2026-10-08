from __future__ import annotations

import warnings
from typing import Any

from ._counterfactual import CounterfactualOverSampler
from ._hard_positive import HardPositiveOverSampler
from ._typical_positive import TypicalPositiveOverSampler

__all__ = [
    "CounterfactualOverSampler",
    "HardPositiveOverSampler",
    "TypicalPositiveOverSampler",
]

# Names kept importable for backward compatibility (deprecated in 0.16.0,
# removed in 0.18.0). Resolved lazily via PEP 562 so the warning fires on
# use, not on `import pntx.pn2t`; the alias *is* the new class, so
# isinstance checks and `.load()` keep working unchanged.
_DEPRECATED_ALIASES: dict[str, type[Any]] = {
    "OverSampler": HardPositiveOverSampler,
    "SyntheticSampler": TypicalPositiveOverSampler,
}


def __getattr__(name: str) -> Any:
    if name in _DEPRECATED_ALIASES:
        new = _DEPRECATED_ALIASES[name]
        warnings.warn(
            f"pntx.pn2t.{name} was renamed to {new.__name__} in 0.16.0; the old name "
            "will be removed in 0.18.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
