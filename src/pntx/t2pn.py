from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from . import prompts
from ._backend_resolve import resolve_backend
from ._sklearn import LLMEstimatorMixin
from .backends.base import Backend, BatchBackend, BatchScoringBackend, ScoringBackend
from .selection import BudgetSelector, Selector, _trim_to_budget, default_tokenizer
from .types import NEGATIVE, POSITIVE

__all__ = ["Classifier"]


class Classifier(LLMEstimatorMixin, ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """scikit-learn ``ClassifierMixin`` that labels text against user-defined
    positive/negative example pools (text → positive/negative, "t2pn").

    The pools' meaning is entirely up to the caller (sentiment, formality,
    policy compliance, ...); this class never interprets it, it only uses the
    pools as few-shot/scoring material. ``fit(X, y)`` groups ``X`` by ``y``
    into two independent pools -- no training happens, and ``X``/``y`` don't
    need to be aligned pairs beyond the usual per-sample correspondence.
    Exactly two classes must be present in ``y``.

    ``y`` may be any two hashable, orderable values (e.g. ``0``/``1``,
    ``-1``/``1``, or ``"positive"``/``"negative"``); by scikit-learn's
    binary-classification convention, the greater of the two
    (``classes_[1]``, e.g. ``1`` or ``"positive"``) is treated as the
    prompt's "positive" role and the lesser (``classes_[0]``) as "negative".
    ``predict_proba``'s columns follow ``classes_`` order.

    ``context_limit`` is the token budget for the *whole* per-call prompt
    (the few-shot exemplar block, shared across every text being classified
    in one ``predict``/``predict_proba`` call, plus that call's query text
    and label choices/response) -- pass a backend's actual context window
    (e.g. ``LlamaCppBackend``'s ``n_ctx``) as ``context_limit``. Exemplars
    are sampled to fit within ``(context_limit - reserve) // 2`` tokens per
    class, where ``reserve`` is computed per call from the longest text in
    that call's ``X`` plus the label choices (``ScoringBackend`` path) or
    the completion response (fallback path) -- unlike ``pn2t``'s samplers,
    which reserve a fixed ``max_tokens`` for an LLM response of unknown
    length, ``Classifier`` already knows every query it needs to score, so
    it can size the reservation exactly instead of conservatively. After
    budget-fitting, the larger class is randomly trimmed down to the
    smaller class's count, so positive and negative are represented
    equally in the few-shot prompt regardless of how lopsided the fitted
    pools are (mirrors ``pn2t.OverSampler``'s per-class budget split and
    count balancing). This replaces relying on the backend's own
    last-resort context trimming, which -- being backend-level -- knows
    nothing about exemplar boundaries or class balance and can silently
    truncate mid-exemplar.
    """

    def __init__(
        self,
        backend: Backend | str,
        *,
        backend_kwargs: dict[str, Any] | None = None,
        selector: Selector | None = None,
        max_exemplars: int | None = None,
        context_limit: int = 100_000,
        temperature: float = 0.0,
    ) -> None:
        """``backend`` is either a ready-made ``Backend`` instance, or the name
        of a built-in backend (e.g. ``"llama"``) to construct lazily from
        ``backend_kwargs`` (e.g.
        ``Classifier(backend="llama", backend_kwargs={"model_path": "model.gguf"})``).

        ``backend_kwargs`` is a single dict rather than ``**kwargs`` so this
        estimator stays compatible with scikit-learn's ``get_params()``/
        ``clone()`` (which requires every ``__init__`` parameter to be
        individually named).

        ``selector`` picks which fitted texts (up to ``max_exemplars`` per
        class) are candidates for the few-shot prompt; ``None`` (default)
        samples a random subset that fits the per-class token budget
        derived from ``context_limit`` (see the class docstring). Whatever
        ``selector`` returns -- default or user-supplied (e.g.
        ``NearestSelector``/``DiversitySelector``) -- is still trimmed to
        that budget as a final guarantee, so any selector stays safe to use
        with pools too large to fit.

        ``max_exemplars`` caps how many fitted texts ``selector`` is asked to
        pick *per class* for a single prompt, on top of the token budget;
        ``None`` means "as many as fit the budget".

        ``temperature`` is only used by the generation-based fallback path
        (backends that don't implement ``ScoringBackend``); it defaults to
        ``0.0`` since classification should be as deterministic as possible,
        unlike ``pn2t``'s samplers, which default to ``1.0`` to encourage
        varied generation.
        """
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.selector = selector
        self.max_exemplars = max_exemplars
        self.context_limit = context_limit
        self.temperature = temperature

    def fit(self, X: Iterable[str], y: Iterable[Any]) -> Classifier:
        """Store the ``X`` texts, grouped by ``y`` into two independent pools.

        This only stores and validates the pools; no training happens here
        (mirrors the pntx-wide convention that "fit" means "hold example
        material", not "learn parameters"). ``y`` must contain exactly two
        distinct classes.
        """
        X = list(X)
        y = list(y)
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length, got len(X)={len(X)} and len(y)={len(y)}"
            )
        classes = sorted(set(y))
        if len(classes) != 2:
            raise ValueError(f"y must contain exactly 2 classes, got {len(classes)}: {classes}")

        self.classes_ = np.array(classes)
        self.positive_ = [text for text, label in zip(X, y, strict=True) if label == classes[1]]
        self.negative_ = [text for text, label in zip(X, y, strict=True) if label == classes[0]]
        self.backend_ = resolve_backend(self.backend, self.backend_kwargs)
        return self

    def predict(self, X: Iterable[str]) -> NDArray[Any]:
        """Predict the most likely class (from ``classes_``) for each text in ``X``."""
        proba = self.predict_proba(X)
        result: NDArray[Any] = np.asarray(self.classes_)[np.argmax(proba, axis=1)]
        return result

    def predict_proba(self, X: Iterable[str]) -> NDArray[Any]:
        """Predict class probabilities for each text in ``X``.

        Columns follow ``classes_`` order. If the backend implements
        ``ScoringBackend``, probabilities come from a calibrated softmax over
        the log-likelihood of each label token as a continuation of the
        few-shot prompt. Otherwise this falls back to asking the backend to
        write the label and parsing it out of the response text; see
        ``prompts.parse_classify_label`` for that path's confidence
        convention (not a calibrated probability).

        Not a naive per-item loop: batched via ``BatchScoringBackend``/
        ``BatchBackend`` when the backend implements one.
        """
        check_is_fitted(self, "classes_")
        texts = list(X)
        if not texts:
            return np.empty((0, 2))

        tokenizer_fn = getattr(self.backend_, "count_tokens", default_tokenizer)
        max_query_tokens = max(tokenizer_fn(text) for text in texts)
        if isinstance(self.backend_, ScoringBackend):
            reserve = max_query_tokens + max(
                tokenizer_fn(choice) for choice in prompts.classify_choice_texts()
            )
        else:
            reserve = max_query_tokens + prompts.CLASSIFY_COMPLETION_MAX_TOKENS

        budget = (self.context_limit - reserve) // 2
        if budget < 1:
            raise ValueError(
                f"context_limit ({self.context_limit}) leaves no token budget for exemplars "
                f"after reserving {reserve} tokens for the longest text being classified "
                "(plus label choices/response); raise context_limit"
            )

        positive = self._select_exemplars(self.positive_, budget, tokenizer_fn)
        negative = self._select_exemplars(self.negative_, budget, tokenizer_fn)
        n_balanced = min(len(positive), len(negative))
        if len(positive) > n_balanced:
            positive = random.sample(positive, n_balanced)
        if len(negative) > n_balanced:
            negative = random.sample(negative, n_balanced)

        if isinstance(self.backend_, ScoringBackend):
            choices = prompts.classify_choice_texts()
            prefix = prompts.build_exemplar_prefix(positive, negative)
            if isinstance(self.backend_, BatchScoringBackend):
                queries = [prompts.build_query_suffix(text) for text in texts]
                all_scores = self.backend_.score_choices_batch(prefix, queries, choices)
            else:
                # No batch-optimized path for this backend; score one prompt
                # at a time. (LlamaCppBackend implements BatchScoringBackend
                # and takes the branch above; a future non-batching
                # ScoringBackend falls back to this.)
                all_scores = [
                    self.backend_.score_choices(prefix + prompts.build_query_suffix(text), choices)
                    for text in texts
                ]
            return np.array([_proba_from_scores(scores) for scores in all_scores])

        completion_prompts = [
            prompts.build_classify_prompt(positive, negative, text) for text in texts
        ]
        if isinstance(self.backend_, BatchBackend):
            raw_completions = self.backend_.complete_batch(
                completion_prompts,
                temperature=self.temperature,
                max_tokens=prompts.CLASSIFY_COMPLETION_MAX_TOKENS,
            )
        else:
            # No concurrent-batch path for this backend; complete one prompt
            # at a time. (A BatchBackend implementation takes the branch above.)
            raw_completions = [
                self.backend_.complete(
                    prompt,
                    temperature=self.temperature,
                    max_tokens=prompts.CLASSIFY_COMPLETION_MAX_TOKENS,
                )
                for prompt in completion_prompts
            ]
        return np.array([_proba_from_completion(raw) for raw in raw_completions])

    def _exemplar_count(self, pool: list[str]) -> int:
        return self.max_exemplars if self.max_exemplars is not None else len(pool)

    def _select_exemplars(
        self, pool: list[str], budget: int, tokenizer_fn: Callable[[str], int]
    ) -> list[str]:
        """Pick candidates from ``pool`` (via ``selector``, or a random
        budget-fit subset if none was configured), then trim the result to
        ``budget`` tokens as a final guarantee -- applied uniformly whether
        ``selector`` is the default or user-supplied, so a selector that
        isn't itself budget-aware (e.g. ``NearestSelector``) can't blow the
        budget.
        """
        k = self._exemplar_count(pool)
        if self.selector is None:
            candidates = BudgetSelector(tokenizer_fn=tokenizer_fn, token_budget=budget).select(
                pool, k
            )
        else:
            candidates = self.selector.select(pool, k)
        return _trim_to_budget(candidates, budget, tokenizer_fn)

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags.string = True
        tags.input_tags.two_d_array = False
        tags.target_tags.required = True
        return tags


def _proba_from_scores(scores: list[float]) -> list[float]:
    probs = _softmax(scores)
    positive_idx = prompts.CLASSIFY_LABELS.index(POSITIVE)
    negative_idx = prompts.CLASSIFY_LABELS.index(NEGATIVE)
    return [probs[negative_idx], probs[positive_idx]]  # column order: [classes_[0], classes_[1]]


def _proba_from_completion(raw: str) -> list[float]:
    label, confidence = prompts.parse_classify_label(raw)
    if label == POSITIVE:
        return [1.0 - confidence, confidence]
    return [confidence, 1.0 - confidence]


def _softmax(scores: list[float]) -> list[float]:
    top = max(scores)
    exps = [math.exp(score - top) for score in scores]
    total = sum(exps)
    return [exp / total for exp in exps]
