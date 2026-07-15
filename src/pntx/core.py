from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from importlib import import_module
from typing import Any

from . import generate as generate_module
from . import prompts
from .backends.base import Backend, BatchBackend, BatchScoringBackend, ScoringBackend
from .selection import RandomSelector, Selector
from .types import ClassifyResult, Label

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
    """Generates and classifies text from user-defined positive/negative example pools.

    The pools' meaning is entirely up to the caller (sentiment, formality,
    policy compliance, ...); this class never interprets it, it only uses the
    pools as few-shot/scoring material. ``positive`` and ``negative`` are
    independent pools, not aligned pairs -- they don't need to be the same
    length or otherwise correspond to each other.
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

        ``max_exemplars`` caps how many fitted texts ``selector`` is asked to
        pick *per side* for a single prompt; ``None`` means "as many as are
        fitted".
        """
        self.backend = _resolve_backend(backend, backend_kwargs)
        self.selector: Selector = selector if selector is not None else RandomSelector()
        self.max_exemplars = max_exemplars
        self._positive: list[str] = []
        self._negative: list[str] = []

    @property
    def positive(self) -> list[str]:
        return list(self._positive)

    @property
    def negative(self) -> list[str]:
        return list(self._negative)

    def fit(self, positive: Iterable[str] = (), negative: Iterable[str] = ()) -> PNTX:
        """Store the positive/negative example pools used as
        generation/classification material.

        ``positive`` and ``negative`` are independent pools -- they don't
        need to be aligned pairs or even the same length. At least one of
        them must be non-empty; fitting just one side is valid (e.g. to
        smoke-test ``generate(side=..., verify=False)`` from a single
        example) but degrades classification/verify quality since there's
        nothing to contrast against. This only stores and validates the
        pools; no training happens here.
        """
        positive = list(positive)
        negative = list(negative)
        if not positive and not negative:
            raise ValueError("at least one of positive or negative must be non-empty")
        self._positive = positive
        self._negative = negative
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
        """Generate up to ``n`` new ``side`` texts from the fitted pools.

        If ``verify`` keeps rejecting candidates (wrong self-classified
        label, or confidence below ``min_confidence``), fewer than ``n``
        texts may come back; this warns rather than looping forever, capped
        at ``max_attempts`` retries (default: ``max(n * 3, 3)``).
        """
        self._check_fitted()
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")

        texts = generate_module.run_generation_loop(
            backend=self.backend,
            positive=self._positive,
            negative=self._negative,
            selector=self.selector,
            max_exemplars=self.max_exemplars,
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
        """Classify ``text`` as ``"positive"`` or ``"negative"``.

        If the backend implements ``ScoringBackend``, this compares the
        log-likelihood of each label as a continuation of the prompt
        (``confidence`` is a calibrated softmax over those log-likelihoods).
        Otherwise it falls back to asking the backend to write the label and
        parsing it out of the response text; see
        ``prompts.parse_classify_label`` for that path's confidence
        convention (not a calibrated probability).
        """
        self._check_fitted()
        positive = self.selector.select(
            self._positive, self._exemplar_count(self._positive), query=text
        )
        negative = self.selector.select(
            self._negative, self._exemplar_count(self._negative), query=text
        )
        if isinstance(self.backend, ScoringBackend):
            prompt = prompts.build_classify_prompt(positive, negative, text)
            scores = self.backend.score_choices(prompt, prompts.classify_choice_texts())
            return _result_from_scores(scores)
        prompt = prompts.build_classify_prompt(positive, negative, text)
        raw = self.backend.complete(
            prompt, temperature=0.0, max_tokens=prompts.CLASSIFY_COMPLETION_MAX_TOKENS
        )
        return _result_from_completion(raw)

    def classify_batch(self, texts: list[str]) -> list[ClassifyResult]:
        """Classify each text in ``texts``; see ``classify()`` for the two
        scoring strategies this dispatches between."""
        self._check_fitted()
        if not texts:
            return []
        positive = self.selector.select(
            self._positive, self._exemplar_count(self._positive), query=None
        )
        negative = self.selector.select(
            self._negative, self._exemplar_count(self._negative), query=None
        )

        if isinstance(self.backend, ScoringBackend):
            choices = prompts.classify_choice_texts()
            prefix = prompts.build_exemplar_prefix(positive, negative)
            if isinstance(self.backend, BatchScoringBackend):
                queries = [prompts.build_query_suffix(text) for text in texts]
                all_scores = self.backend.score_choices_batch(prefix, queries, choices)
            else:
                # No batch-optimized path for this backend; score one prompt
                # at a time. (LlamaCppBackend implements BatchScoringBackend
                # and takes the branch above; a future non-batching
                # ScoringBackend falls back to this.)
                all_scores = [
                    self.backend.score_choices(prefix + prompts.build_query_suffix(text), choices)
                    for text in texts
                ]
            return [_result_from_scores(scores) for scores in all_scores]

        completion_prompts = [
            prompts.build_classify_prompt(positive, negative, text) for text in texts
        ]
        if isinstance(self.backend, BatchBackend):
            raw_completions = self.backend.complete_batch(
                completion_prompts,
                temperature=0.0,
                max_tokens=prompts.CLASSIFY_COMPLETION_MAX_TOKENS,
            )
        else:
            # No concurrent-batch path for this backend; complete one prompt
            # at a time. (AnthropicBackend implements BatchBackend and takes
            # the branch above.)
            raw_completions = [
                self.backend.complete(
                    prompt, temperature=0.0, max_tokens=prompts.CLASSIFY_COMPLETION_MAX_TOKENS
                )
                for prompt in completion_prompts
            ]
        return [_result_from_completion(raw) for raw in raw_completions]

    def _exemplar_count(self, pool: list[str]) -> int:
        return self.max_exemplars if self.max_exemplars is not None else len(pool)

    def _check_fitted(self) -> None:
        if not self._positive and not self._negative:
            raise RuntimeError(
                "PNTX.fit(positive=..., negative=...) must be called before this method"
            )


def _result_from_scores(scores: list[float]) -> ClassifyResult:
    probs = _softmax(scores)
    best = max(range(len(scores)), key=scores.__getitem__)
    return ClassifyResult(label=prompts.CLASSIFY_LABELS[best], confidence=probs[best])


def _result_from_completion(raw: str) -> ClassifyResult:
    label, confidence = prompts.parse_classify_label(raw)
    return ClassifyResult(label=label, confidence=confidence)


def _softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(score - top) for score in scores]
    total = sum(exps)
    return [exp / total for exp in exps]
