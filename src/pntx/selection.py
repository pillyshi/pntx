from __future__ import annotations

import random
from typing import Protocol, runtime_checkable

from .types import Pair


@runtime_checkable
class Selector(Protocol):
    """Chooses which fitted pairs to embed in a prompt.

    ``query`` is the text being classified, for selectors that pick pairs
    relevant to it (e.g. ``NearestSelector``); selectors that don't use it
    (e.g. ``RandomSelector``) simply ignore the argument.
    """

    def select(self, pairs: list[Pair], k: int, query: str | None = None) -> list[Pair]: ...


class RandomSelector:
    """Selects a uniform random subset of ``pairs``, ignoring ``query``."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def select(self, pairs: list[Pair], k: int, query: str | None = None) -> list[Pair]:
        if k >= len(pairs):
            return list(pairs)
        return self._rng.sample(pairs, k)
