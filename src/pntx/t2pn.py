from __future__ import annotations

import math
import random
import warnings
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
    (the few-shot exemplar block plus that call's query text and label
    choices/response) -- pass a backend's actual context window (e.g.
    ``LlamaCppBackend``'s ``n_ctx``) as ``context_limit``. Exemplars are
    sampled to fit within ``(context_limit - reserve) // 2`` tokens per
    class, where ``reserve`` is the longer of the label choices
    (``ScoringBackend`` path) or the completion response (fallback path)
    plus the longest text the exemplar block needs to leave room for --
    unlike ``pn2t``'s samplers, which reserve a fixed ``max_tokens`` for an
    LLM response of unknown length, ``Classifier`` can size this
    exactly instead of conservatively. After budget-fitting, the larger
    class is randomly trimmed down to the smaller class's count, so
    positive and negative are represented equally in the few-shot prompt
    regardless of how lopsided the fitted pools are (mirrors
    ``pn2t.OverSampler``'s per-class budget split and count balancing).
    This replaces relying on the backend's own last-resort context
    trimming, which -- being backend-level -- knows nothing about
    exemplar boundaries or class balance and can silently truncate
    mid-exemplar.

    Whether exemplar selection happens once in ``fit()`` or is redone per
    text in ``predict``/``predict_proba`` depends on whether ``selector``
    is *query-aware* (``getattr(selector, "query_aware", False)``, e.g.
    ``NearestSelector``):

    - **Static** (default ``selector=None``, or any selector that ignores
      ``query``, e.g. ``RandomSelector``/``DiversitySelector``): exemplar
      selection, ordering, and (for ``ScoringBackend`` backends, unless
      ``calibrate=False``) content-free calibration are all resolved once in
      ``fit()`` from the fitted pools -- not from the classification
      target(s), which aren't known yet -- and reused by every later
      ``predict``/``predict_proba`` call. ``reserve`` is sized from the
      longest text across the *fitted pools* rather than the actual query,
      since the real query isn't known until ``predict``. This is what
      lets ``fit()``'s one-time selection be reused as a stable
      shared-prefix for ``BatchScoringBackend.score_choices_batch`` across
      an entire ``predict``/``predict_proba`` call, but it also means
      ``fit()`` is no longer free of backend calls: it makes one
      ``score_choices`` call (for calibration) when the backend implements
      ``ScoringBackend`` and ``calibrate`` is true.
    - **Dynamic** (query-aware ``selector``, e.g. ``NearestSelector``):
      exemplars genuinely relevant to each text can't be known until that
      text is seen, so selection (and calibration) reruns per text inside
      ``predict``/``predict_proba``, with ``reserve`` sized from the
      longest text in that call's ``X`` (mirrors the old, pre-``fit``-time
      behavior). For a ``BatchScoringBackend``, this forfeits the shared
      KV-cache prefix ``score_choices_batch`` provides -- each text gets
      its own ``score_choices`` call instead, and a ``UserWarning`` is
      raised once per call to flag the latency trade-off. Backends without
      that optimization to begin with (plain ``ScoringBackend``,
      ``BatchBackend``, plain ``Backend``) already loop per text, so
      dynamic selection costs nothing extra there.

    Content-free calibration (Zhao et al. 2021, "Calibrate Before Use")
    corrects for the few-shot prompt's own label bias (an artifact of which
    exemplars ended up in it and in what order) by scoring
    ``prompts.CONTENT_FREE_QUERY`` -- a query with no real content -- against
    the same exemplar prefix, then dividing each real query's raw label
    probabilities by that content-free baseline and renormalizing. It's
    applied only on the ``ScoringBackend`` path: the fallback (parse-based)
    path's "confidence" is already a fixed heuristic value rather than a
    calibrated probability, so there's nothing principled to correct.
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
        calibrate: bool = True,
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

        ``calibrate`` enables content-free calibration (see the class
        docstring) on the ``ScoringBackend`` path; ``True`` by default. Set
        ``False`` to get the raw, uncalibrated softmax over label
        log-likelihoods instead (e.g. to compare against pre-calibration
        behavior, or to skip the extra ``score_choices`` call it costs).
        """
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.selector = selector
        self.max_exemplars = max_exemplars
        self.context_limit = context_limit
        self.temperature = temperature
        self.calibrate = calibrate

    def fit(self, X: Iterable[str], y: Iterable[Any]) -> Classifier:
        """Store the ``X`` texts, grouped by ``y`` into two independent pools.

        No parameter learning happens here (mirrors the pntx-wide convention
        that "fit" means "hold example material"). ``y`` must contain
        exactly two distinct classes.

        For a *static* ``selector`` (see the class docstring), this also
        resolves exemplar selection, ordering, and -- for ``ScoringBackend``
        backends, unless ``calibrate=False`` -- content-free calibration
        once, so every later ``predict``/``predict_proba`` call can reuse
        them as a stable shared prefix; that calibration step makes one
        ``score_choices`` call. A *dynamic* (query-aware) ``selector``
        leaves all of that to predict time instead, since it can't be
        resolved without knowing the text being classified.
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

        self.exemplar_positive_: list[str] | None = None
        self.exemplar_negative_: list[str] | None = None
        self.exemplar_prefix_: str | None = None
        self.calibration_weights_: list[float] | None = None
        if not self._is_dynamic():
            tokenizer_fn = getattr(self.backend_, "count_tokens", default_tokenizer)
            budget = self._compute_budget(self.positive_ + self.negative_, tokenizer_fn)
            self.exemplar_positive_, self.exemplar_negative_ = self._select_balanced(
                budget, tokenizer_fn
            )
            self.exemplar_prefix_ = prompts.build_exemplar_prefix(
                self.exemplar_positive_, self.exemplar_negative_
            )
            if self.calibrate and isinstance(self.backend_, ScoringBackend):
                choices = prompts.classify_choice_texts()
                content_free_scores = self.backend_.score_choices(
                    self.exemplar_prefix_
                    + prompts.build_query_suffix(prompts.CONTENT_FREE_QUERY),
                    choices,
                )
                self.calibration_weights_ = _label_probs_from_scores(content_free_scores)
        return self

    def predict(self, X: Iterable[str]) -> NDArray[Any]:
        """Predict the most likely class (from ``classes_``) for each text in ``X``."""
        proba = self.predict_proba(X)
        result: NDArray[Any] = np.asarray(self.classes_)[np.argmax(proba, axis=1)]
        return result

    def predict_proba(self, X: Iterable[str]) -> NDArray[Any]:
        """Predict class probabilities for each text in ``X``.

        Columns follow ``classes_`` order. If the backend implements
        ``ScoringBackend``, probabilities come from a softmax over the
        log-likelihood of each label token as a continuation of the few-shot
        prompt, then (unless ``calibrate=False``) content-free-calibrated
        (see the class docstring). Otherwise this falls back to asking the
        backend to write the label and parsing it out of the response text;
        see ``prompts.parse_classify_label`` for that path's confidence
        convention (not a calibrated probability).

        Not a naive per-item loop when ``selector`` is static (the default):
        batched via ``BatchScoringBackend``/``BatchBackend`` when the
        backend implements one, reusing the exemplar selection fixed in
        ``fit()``. A dynamic (query-aware) ``selector`` (e.g.
        ``NearestSelector``) always loops per item to reselect exemplars for
        each text -- see the class docstring for the batching trade-off this
        implies.
        """
        check_is_fitted(self, "classes_")
        texts = list(X)
        if not texts:
            return np.empty((0, 2))

        tokenizer_fn = getattr(self.backend_, "count_tokens", default_tokenizer)
        dynamic = self._is_dynamic()
        budget = self._compute_budget(texts, tokenizer_fn) if dynamic else None

        if isinstance(self.backend_, ScoringBackend):
            choices = prompts.classify_choice_texts()
            if dynamic:
                assert budget is not None
                label_probs = self._score_dynamic(texts, budget, tokenizer_fn, choices)
            else:
                label_probs = self._score_static(texts, choices)
            return np.array([_proba_from_label_probs(lp) for lp in label_probs])

        if dynamic:
            assert budget is not None
            completion_prompts = []
            for text in texts:
                positive, negative = self._select_balanced(budget, tokenizer_fn, query=text)
                completion_prompts.append(prompts.build_classify_prompt(positive, negative, text))
        else:
            assert self.exemplar_positive_ is not None
            assert self.exemplar_negative_ is not None
            completion_prompts = [
                prompts.build_classify_prompt(
                    self.exemplar_positive_, self.exemplar_negative_, text
                )
                for text in texts
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

    def _is_dynamic(self) -> bool:
        """Whether exemplar selection must be redone per text in
        ``predict``/``predict_proba`` (true when ``selector`` is
        query-aware, e.g. ``NearestSelector``) rather than once in
        ``fit()`` -- see the class docstring."""
        return bool(self.selector is not None and getattr(self.selector, "query_aware", False))

    def _compute_budget(
        self, reference_texts: list[str], tokenizer_fn: Callable[[str], int]
    ) -> int:
        """Compute the per-class exemplar token budget for ``context_limit``,
        reserving room for the longest of ``reference_texts`` plus the label
        choices (``ScoringBackend`` path) or the completion response
        (fallback path).

        ``reference_texts`` is the actual ``predict``/``predict_proba``
        batch for a dynamic selector, or the fitted pools themselves (the
        best available stand-in, since the real query isn't known at
        ``fit()`` time) for a static one -- see the class docstring.
        """
        max_tokens = max((tokenizer_fn(text) for text in reference_texts), default=0)
        if isinstance(self.backend_, ScoringBackend):
            reserve = max_tokens + max(
                tokenizer_fn(choice) for choice in prompts.classify_choice_texts()
            )
        else:
            reserve = max_tokens + prompts.CLASSIFY_COMPLETION_MAX_TOKENS

        budget = (self.context_limit - reserve) // 2
        if budget < 1:
            raise ValueError(
                f"context_limit ({self.context_limit}) leaves no token budget for exemplars "
                f"after reserving {reserve} tokens for the longest text "
                "(plus label choices/response); raise context_limit"
            )
        return budget

    def _exemplar_count(self, pool: list[str]) -> int:
        return self.max_exemplars if self.max_exemplars is not None else len(pool)

    def _select_exemplars(
        self,
        pool: list[str],
        budget: int,
        tokenizer_fn: Callable[[str], int],
        query: str | None = None,
    ) -> list[str]:
        """Pick candidates from ``pool`` (via ``selector``, or a random
        budget-fit subset if none was configured), then trim the result to
        ``budget`` tokens as a final guarantee -- applied uniformly whether
        ``selector`` is the default or user-supplied, so a selector that
        isn't itself budget-aware (e.g. ``NearestSelector``) can't blow the
        budget. ``query`` is passed through to ``selector`` and is only
        meaningful for a query-aware one; the default budget-fit selector
        ignores it.
        """
        k = self._exemplar_count(pool)
        if self.selector is None:
            candidates = BudgetSelector(tokenizer_fn=tokenizer_fn, token_budget=budget).select(
                pool, k
            )
        else:
            candidates = self.selector.select(pool, k, query=query)
        return _trim_to_budget(candidates, budget, tokenizer_fn)

    def _select_balanced(
        self, budget: int, tokenizer_fn: Callable[[str], int], query: str | None = None
    ) -> tuple[list[str], list[str]]:
        """Select positive/negative exemplars, then randomly trim the larger
        side down to the smaller side's count so both classes are
        represented equally regardless of pool size (see the class
        docstring)."""
        positive = self._select_exemplars(self.positive_, budget, tokenizer_fn, query)
        negative = self._select_exemplars(self.negative_, budget, tokenizer_fn, query)
        n_balanced = min(len(positive), len(negative))
        if len(positive) > n_balanced:
            positive = random.sample(positive, n_balanced)
        if len(negative) > n_balanced:
            negative = random.sample(negative, n_balanced)
        return positive, negative

    def _score_static(self, texts: list[str], choices: list[str]) -> list[list[float]]:
        """Score every text against the exemplar prefix fixed in ``fit()``,
        batched via ``BatchScoringBackend`` when available so the prefix is
        eval'd once and its KV cache reused across the whole call."""
        assert isinstance(self.backend_, ScoringBackend)  # narrows for mypy; caller already checked
        prefix = self.exemplar_prefix_
        assert prefix is not None  # set in fit() for the static path
        if isinstance(self.backend_, BatchScoringBackend):
            queries = [prompts.build_query_suffix(text) for text in texts]
            all_scores = self.backend_.score_choices_batch(prefix, queries, choices)
        else:
            # No batch-optimized path for this backend; score one prompt at
            # a time. (LlamaCppBackend implements BatchScoringBackend and
            # takes the branch above; a future non-batching ScoringBackend
            # falls back to this.)
            all_scores = [
                self.backend_.score_choices(prefix + prompts.build_query_suffix(text), choices)
                for text in texts
            ]
        label_probs = [_label_probs_from_scores(scores) for scores in all_scores]
        if self.calibration_weights_ is not None:
            label_probs = [_calibrate(lp, self.calibration_weights_) for lp in label_probs]
        return label_probs

    def _score_dynamic(
        self,
        texts: list[str],
        budget: int,
        tokenizer_fn: Callable[[str], int],
        choices: list[str],
    ) -> list[list[float]]:
        """Reselect exemplars per text (a query-aware ``selector`` is
        configured) and score each against its own prefix.

        For a ``BatchScoringBackend`` this forfeits the shared-prefix
        KV-cache batching ``_score_static`` gets, since every text's prefix
        differs; warns once per call to flag the trade-off (backends
        without that optimization already loop per text, so there's nothing
        extra to warn about there). When ``calibrate`` is enabled, the
        content-free calibration query is batched together with the real
        query under the same per-text prefix via ``score_choices_batch``
        where available, so calibration doesn't cost a second prefix eval on
        top of losing cross-item batching.
        """
        assert isinstance(self.backend_, ScoringBackend)  # narrows for mypy; caller already checked
        if isinstance(self.backend_, BatchScoringBackend):
            warnings.warn(
                "selector is query-aware (e.g. NearestSelector), so exemplars are "
                "reselected per text; this forfeits the shared-prefix KV-cache batching "
                "score_choices_batch normally provides, so latency scales with len(X) "
                "instead of being ~constant. Use a non-query-aware selector (the default, "
                "or e.g. RandomSelector/DiversitySelector) if that's not the trade-off "
                "you want.",
                UserWarning,
                stacklevel=3,
            )

        label_probs: list[list[float]] = []
        for text in texts:
            positive, negative = self._select_balanced(budget, tokenizer_fn, query=text)
            prefix = prompts.build_exemplar_prefix(positive, negative)
            query_suffix = prompts.build_query_suffix(text)

            if not self.calibrate:
                scores = self.backend_.score_choices(prefix + query_suffix, choices)
                label_probs.append(_label_probs_from_scores(scores))
                continue

            content_free_suffix = prompts.build_query_suffix(prompts.CONTENT_FREE_QUERY)
            if isinstance(self.backend_, BatchScoringBackend):
                content_free_scores, query_scores = self.backend_.score_choices_batch(
                    prefix, [content_free_suffix, query_suffix], choices
                )
            else:
                content_free_scores = self.backend_.score_choices(
                    prefix + content_free_suffix, choices
                )
                query_scores = self.backend_.score_choices(prefix + query_suffix, choices)

            content_free_probs = _label_probs_from_scores(content_free_scores)
            calibrated = _calibrate(_label_probs_from_scores(query_scores), content_free_probs)
            label_probs.append(calibrated)
        return label_probs

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags.string = True
        tags.input_tags.two_d_array = False
        tags.target_tags.required = True
        return tags


def _label_probs_from_scores(scores: list[float]) -> list[float]:
    """Softmax over raw label scores, in ``prompts.CLASSIFY_LABELS`` order
    (``[POSITIVE, NEGATIVE]``) -- kept separate from ``classes_`` column
    order (``_proba_from_label_probs``) so content-free calibration, which
    also operates in this order, can slot in between."""
    return _softmax(scores)


def _proba_from_label_probs(label_probs: list[float]) -> list[float]:
    positive_idx = prompts.CLASSIFY_LABELS.index(POSITIVE)
    negative_idx = prompts.CLASSIFY_LABELS.index(NEGATIVE)
    return [
        label_probs[negative_idx],
        label_probs[positive_idx],
    ]  # column order: [classes_[0], classes_[1]]


def _calibrate(label_probs: list[float], content_free_probs: list[float]) -> list[float]:
    """Content-free calibration (Zhao et al. 2021, "Calibrate Before Use"):
    divide each label's raw probability by its content-free-query baseline
    and renormalize, undoing the few-shot prompt's own label bias (from
    exemplar choice/order) regardless of query content. Both lists must be
    in the same label order (``prompts.CLASSIFY_LABELS``)."""
    eps = 1e-12
    corrected = [p / max(cf, eps) for p, cf in zip(label_probs, content_free_probs, strict=True)]
    total = sum(corrected)
    if total <= 0:
        return label_probs
    return [c / total for c in corrected]


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
