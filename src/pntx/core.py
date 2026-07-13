from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from importlib import import_module
from typing import Any

from . import generate as generate_module
from . import prompts
from .backends.base import Backend, BatchScoringBackend, ScoringBackend
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
        max_exemplars: int | None = None,
        **backend_kwargs: Any,
    ) -> None:
        """Create a PNTX model.

        ``backend`` is either a ready-made ``Backend`` instance, or the name
        of a built-in backend (e.g. ``"llama"``) to construct lazily; any
        ``backend_kwargs`` are then forwarded to that backend's constructor
        (e.g. ``PNTX(backend="llama", model_path="model.gguf")``).

        ``max_exemplars`` caps how many fitted pairs ``selector`` is asked to
        pick for a single prompt; ``None`` means "as many as are fitted".
        """
        self.backend = _resolve_backend(backend, backend_kwargs)
        self.selector: Selector = selector if selector is not None else RandomSelector()
        self.max_exemplars = max_exemplars
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
        max_attempts: int | None = None,
    ) -> list[str]:
        """Generate up to ``n`` new ``side`` texts from the fitted pairs.

        If ``verify`` keeps rejecting candidates (wrong self-classified
        label, or confidence below ``min_confidence``), fewer than ``n``
        texts may come back; this warns rather than looping forever, capped
        at ``max_attempts`` retries (default: ``max(n * 3, 3)``).
        """
        self._check_fitted()
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if verify:
            self._scoring_backend()  # fail fast, before generating anything

        texts = generate_module.run_generation_loop(
            backend=self.backend,
            pairs=self._pairs,
            selector=self.selector,
            exemplar_count=self._exemplar_count(),
            classify=self.classify,
            n=n,
            side=side,
            temperature=temperature,
            dedup=dedup,
            verify=verify,
            min_confidence=min_confidence,
            max_attempts=max_attempts,
        )
        if len(texts) < n:
            warnings.warn(
                f"requested n={n} {side!r} texts but only generated {len(texts)}",
                UserWarning,
                stacklevel=2,
            )
        return texts

    def classify(self, text: str) -> ClassifyResult:
        self._check_fitted()
        backend = self._scoring_backend()
        exemplars = self.selector.select(self._pairs, self._exemplar_count(), query=text)
        prompt = prompts.build_classify_prompt(exemplars, text)
        scores = backend.score_choices(prompt, prompts.classify_choice_texts())
        return _result_from_scores(scores)

    def classify_batch(self, texts: list[str]) -> list[ClassifyResult]:
        self._check_fitted()
        backend = self._scoring_backend()
        if not texts:
            return []

        exemplars = self.selector.select(self._pairs, self._exemplar_count(), query=None)
        choices = prompts.classify_choice_texts()
        prefix = prompts.build_exemplar_prefix(exemplars)

        if isinstance(backend, BatchScoringBackend):
            queries = [prompts.build_query_suffix(text) for text in texts]
            all_scores = backend.score_choices_batch(prefix, queries, choices)
        else:
            # No batch-optimized path for this backend; score one prompt at a
            # time. (LlamaCppBackend implements BatchScoringBackend and takes
            # the branch above; a future non-batching ScoringBackend falls
            # back to this.)
            all_scores = [
                backend.score_choices(prefix + prompts.build_query_suffix(text), choices)
                for text in texts
            ]
        return [_result_from_scores(scores) for scores in all_scores]

    def _exemplar_count(self) -> int:
        return self.max_exemplars if self.max_exemplars is not None else len(self._pairs)

    def _scoring_backend(self) -> ScoringBackend:
        if not isinstance(self.backend, ScoringBackend):
            raise NotImplementedError(
                "this requires a ScoringBackend (classify(), classify_batch(), "
                "and generate(verify=True) all classify via score_choices); "
                "parse-based classification for plain Backend instances is not "
                "implemented yet"
            )
        return self.backend

    def _check_fitted(self) -> None:
        if not self._pairs:
            raise RuntimeError("PNTX.fit(pairs) must be called before this method")


def _result_from_scores(scores: list[float]) -> ClassifyResult:
    probs = _softmax(scores)
    best = max(range(len(scores)), key=scores.__getitem__)
    return ClassifyResult(label=prompts.CLASSIFY_LABELS[best], confidence=probs[best])


def _softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(score - top) for score in scores]
    total = sum(exps)
    return [exp / total for exp in exps]
