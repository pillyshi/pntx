from __future__ import annotations

import json
import logging
import os
import random
import warnings
from collections.abc import Callable, Mapping
from math import ceil
from numbers import Integral, Real
from typing import TYPE_CHECKING, Any, ClassVar, Generic, Protocol, TypeAlias, TypeVar, cast

import numpy as np
from pydantic import BaseModel
from sklearn.base import BaseEstimator

from .._backend_resolve import resolve_backend
from .._labels import resolve_binary_labels
from .._sklearn import LLMEstimatorMixin
from ..backends.base import Backend
from ..selection import _SAMPLE_METHODS, default_tokenizer, sample_group
from ._structured import complete_structured

if TYPE_CHECKING:
    from typing_extensions import Self

__all__ = ["BaseLLMOverSampler", "SamplingStrategy"]

PROMPT_OVERHEAD = 500

SamplingStrategy: TypeAlias = (
    str | float | Mapping[Any, int] | Callable[[list[Any]], Mapping[Any, int]]
)

_SAMPLING_STRATEGY_STRINGS = ("auto", "minority", "not minority", "not majority", "all")


class _Logger(Protocol):
    def debug(self, msg: object, /, *args: object, **kwargs: object) -> None: ...


class _HasText(Protocol):
    text: str


ResultT = TypeVar("ResultT", bound=BaseModel)
BatchT = TypeVar("BatchT", bound=BaseModel)
ItemT = TypeVar("ItemT", bound=_HasText)


def resolve_n_to_generate(
    sampling_strategy: SamplingStrategy,
    y: list[Any],
    *,
    positive_label: Any,
    negative_label: Any,
) -> int:
    """How many positive texts to generate, per imbalanced-learn's
    ``sampling_strategy`` semantics restricted to positive-only generation.

    Mirrors ``imblearn.utils.check_sampling_strategy(..., "over-sampling")``
    for a binary ``y`` (without importing imbalanced-learn):

    - ``"auto"``/``"not majority"``/``"minority"``/``"not minority"``/``"all"``:
      the targeted classes are brought up to the majority count. Only the
      positive class's entry is used; if the strategy targets the negative
      class instead (e.g. ``"auto"`` when positive is already the majority),
      nothing is generated, since pn2t never generates negatives.
    - ``float`` in ``(0, 1]``: desired ``n_minority / n_majority`` after
      resampling. Only valid when positive is the minority class.
    - ``dict``: ``{label: n_samples_after_resampling}``. Only the positive
      label may ask for more samples than it has; a negative entry must equal
      the current negative count (or be omitted).
    - ``callable``: ``f(y) -> dict``, interpreted as above.
    """
    n_pos = sum(1 for v in y if v == positive_label)
    n_neg = len(y) - n_pos
    if isinstance(sampling_strategy, str):
        if sampling_strategy not in _SAMPLING_STRATEGY_STRINGS:
            raise ValueError(
                f"sampling_strategy string must be one of {_SAMPLING_STRATEGY_STRINGS}, "
                f"got {sampling_strategy!r}"
            )
        n_majority = max(n_pos, n_neg)
        positive_is_minority = n_pos < n_neg
        positive_is_majority = n_pos > n_neg
        targets_positive = {
            "auto": not positive_is_majority,
            "not majority": not positive_is_majority,
            "minority": positive_is_minority or n_pos == n_neg,
            "not minority": not positive_is_minority,
            "all": True,
        }[sampling_strategy]
        return n_majority - n_pos if targets_positive else 0
    if callable(sampling_strategy):
        return _n_from_dict(
            sampling_strategy(list(y)),
            n_pos=n_pos,
            n_neg=n_neg,
            positive_label=positive_label,
            negative_label=negative_label,
        )
    if isinstance(sampling_strategy, Mapping):
        return _n_from_dict(
            sampling_strategy,
            n_pos=n_pos,
            n_neg=n_neg,
            positive_label=positive_label,
            negative_label=negative_label,
        )
    if isinstance(sampling_strategy, Real) and not isinstance(sampling_strategy, bool):
        ratio = float(sampling_strategy)
        if not 0 < ratio <= 1:
            raise ValueError(
                "When 'sampling_strategy' is a float, it should be in the range (0, 1]. "
                f"Got {sampling_strategy} instead."
            )
        if n_pos > n_neg:
            raise ValueError(
                "A float sampling_strategy targets the minority class, but the positive "
                f"class is not the minority ({n_pos} positive vs {n_neg} negative); pn2t "
                "only generates positives. Pass a dict {pos_label: n} instead."
            )
        n_target = int(ratio * n_neg)
        if n_target < n_pos:
            raise ValueError(
                f"sampling_strategy={sampling_strategy} asks for {n_target} positive "
                f"samples, fewer than the {n_pos} already present; over-sampling cannot "
                "remove samples. Increase the ratio."
            )
        return n_target - n_pos
    raise ValueError(
        "sampling_strategy must be a str, a float in (0, 1], a dict "
        f"{{label: n_samples}}, or a callable returning such a dict; got {sampling_strategy!r}"
    )


def _n_from_dict(
    strategy: Mapping[Any, int],
    *,
    n_pos: int,
    n_neg: int,
    positive_label: Any,
    negative_label: Any,
) -> int:
    unknown = [k for k in strategy if k != positive_label and k != negative_label]
    if unknown:
        raise ValueError(f"The {set(unknown)} target class is/are not present in the data.")
    if negative_label in strategy and strategy[negative_label] != n_neg:
        raise ValueError(
            f"sampling_strategy asks for {strategy[negative_label]} samples of the negative "
            f"class {negative_label!r} (currently {n_neg}); pn2t only generates the positive "
            "class, so a negative entry must equal its current count or be omitted."
        )
    if positive_label not in strategy:
        return 0
    n_target = strategy[positive_label]
    if not isinstance(n_target, Integral) or isinstance(n_target, bool):
        raise ValueError(
            f"sampling_strategy dict values must be integers, got {n_target!r} for "
            f"{positive_label!r}"
        )
    if n_target < n_pos:
        raise ValueError(
            "With over-sampling methods, the number of samples in a class should be greater "
            f"or equal to the original number of samples. Originally, there is {n_pos} "
            f"samples and {n_target} samples are asked."
        )
    return int(n_target) - n_pos


def make_rng(random_state: Any) -> random.Random:
    """``random.Random`` from an sklearn-style ``random_state``.

    An ``int``/``None`` seeds ``random.Random`` directly (so an int gives the
    same results as the pre-0.16.0 ``seed=`` parameter); a
    ``numpy.random.RandomState`` seeds it with one draw from that generator.
    """
    if random_state is None or (
        isinstance(random_state, Integral) and not isinstance(random_state, bool)
    ):
        return random.Random(None if random_state is None else int(random_state))
    if isinstance(random_state, np.random.RandomState):
        return random.Random(int(random_state.randint(0, 2**31 - 1)))
    raise ValueError(
        f"random_state must be an int, a numpy.random.RandomState, or None; got {random_state!r}"
    )


class BaseLLMOverSampler(
    LLMEstimatorMixin,
    BaseEstimator,  # type: ignore[misc]
    Generic[ResultT, BatchT, ItemT],
):
    """Shared machinery for pn2t's LLM-based over-samplers.

    Plays the role of imbalanced-learn's ``BaseOverSampler``: subclasses
    implement one generation algorithm each (``HardPositiveOverSampler``,
    ``TypicalPositiveOverSampler``, ``CounterfactualOverSampler``, ...) and
    only supply the prompt, the pydantic schemas, and any extra acceptance
    filter. Everything else -- ``sampling_strategy``/``random_state``
    resolution, label handling, the batch-generation/retry loop, exact-match
    dedup, ``save``/``load`` -- lives here.

    Type parameters: ``ResultT`` is the fitted ``generation_result_`` model
    (what ``save``/``load`` round-trip), ``BatchT`` the schema the LLM must
    return for one batch (often the same model), and ``ItemT`` one generated
    item (anything with a ``text`` attribute).

    Each candidate item goes through, in order:

    1. ``_validity_reason`` -- always applied (e.g. a minimality check);
    2. exact-match dedup against ``X`` and already-accepted texts, then
       ``_dedup_extra_reason`` -- only when ``deduplicate=True``;
    3. ``_verify_candidates`` -- one call per batch on the survivors, for
       checks that are expensive per call (e.g. a classifier ``predict``).

    Rejected items are passed to ``_record_rejection`` with a short reason.

    Not meant to be instantiated directly. Subclasses define their own
    ``__init__`` listing every parameter explicitly (sklearn's ``get_params``
    introspects the concrete class's signature).
    """

    # Set by subclasses.
    _result_model: ClassVar[type[BaseModel]]
    _batch_model: ClassVar[type[BaseModel]]
    _progress_desc: ClassVar[str]
    _items_name: ClassVar[str]

    # Common __init__ parameters (assigned by each subclass's __init__).
    backend: Backend | str
    backend_kwargs: dict[str, Any] | None
    sampling_strategy: SamplingStrategy
    batch_size: int
    deduplicate: bool
    context_limit: int
    max_tokens: int
    language: str | None
    random_state: int | np.random.RandomState | None
    sample_method: str
    embedding_model: str
    temperature: float
    verbose: bool
    logger: _Logger | None
    pos_label: Any

    generation_result_: ResultT

    # ---- hooks for subclasses -------------------------------------------

    def _validate_extra_params(self) -> None:
        """Validate subclass-specific parameters; raise ``ValueError``."""

    def _prepare_fit(
        self,
        X: list[str],
        y: list[Any],
        *,
        positive_label: Any,
        negative_label: Any,
        random_state: Any,
    ) -> None:
        """Per-fit setup that needs the data (called before generation, even
        when nothing is to be generated, so data-dependent validation always
        runs)."""

    def _exemplar_budget(self) -> int:
        """Token budget for one side's exemplars in a single prompt."""
        raise NotImplementedError

    def _check_exemplars_fit(
        self,
        pos_texts: list[str],
        neg_texts: list[str],
        budget: int,
        tokenizer_fn: Callable[[str], int],
    ) -> None:
        raise NotImplementedError

    def _new_result(self) -> ResultT:
        raise NotImplementedError

    def _build_prompt(
        self,
        pos_texts: list[str],
        neg_texts: list[str],
        batch_count: int,
        budget: int,
        tokenizer_fn: Callable[[str], int],
        rng: random.Random,
    ) -> tuple[str, str]:
        """Return ``(system, user)`` messages for one batch."""
        raise NotImplementedError

    def _merge_batch_analysis(self, result: BatchT) -> None:
        """Fold one batch's analysis fields into ``generation_result_``."""
        raise NotImplementedError

    def _batch_items(self, result: BatchT) -> list[ItemT]:
        """Candidate items of one batch, converted to ``ItemT``."""
        raise NotImplementedError

    def _accepted_items(self) -> list[ItemT]:
        """The list inside ``generation_result_`` that accepted items go to."""
        raise NotImplementedError

    def _validity_reason(self, item: ItemT) -> str | None:
        """Rejection reason applied regardless of ``deduplicate``; ``None`` = ok."""
        return None

    def _dedup_extra_reason(self, item: ItemT, pos_texts: list[str]) -> str | None:
        """Extra rejection reason, applied only when ``deduplicate=True``."""
        return None

    def _verify_candidates(self, items: list[ItemT]) -> list[str | None]:
        """Batch-level check on items that passed every other filter; returns
        one rejection reason (or ``None``) per item."""
        return [None] * len(items)

    def _record_rejection(self, item: ItemT, reason: str) -> None:
        """Called for every rejected candidate (default: discard)."""

    # ---- shared implementation ------------------------------------------

    def fit_resample(self, X: list[str], y: Any) -> tuple[list[str], list[Any]]:
        """Generate positive texts and append them to the dataset.

        Args:
            X: Raw texts. Must be a list of strings, not a numeric feature matrix.
            y: Binary labels; see ``pos_label`` and
                ``pntx._labels.resolve_binary_labels`` for how the positive
                value is determined.

        Returns:
            Tuple of (augmented texts, augmented labels) where the appended
            texts are the generated positives and the appended labels are all
            the positive label resolved from ``y``.
        """
        self._validate_params()

        y_list = list(y)
        if len(X) != len(y_list):
            raise ValueError(
                f"X and y must have the same length, got len(X)={len(X)} and len(y)={len(y_list)}"
            )
        negative_label, positive_label = resolve_binary_labels(y_list, pos_label=self.pos_label)
        pos_texts = [t for t, yi in zip(X, y_list, strict=True) if yi == positive_label]
        neg_texts = [t for t, yi in zip(X, y_list, strict=True) if yi == negative_label]

        target_count = resolve_n_to_generate(
            self.sampling_strategy,
            y_list,
            positive_label=positive_label,
            negative_label=negative_label,
        )

        self.backend_ = resolve_backend(self.backend, self.backend_kwargs)

        if self.verbose:
            try:
                from tqdm.auto import tqdm as _tqdm
            except ImportError:
                raise ImportError(
                    "tqdm is required when verbose=True. Install it with: pip install tqdm"
                ) from None

        self._prepare_fit(
            list(X),
            y_list,
            positive_label=positive_label,
            negative_label=negative_label,
            random_state=self.random_state,
        )

        self.generation_result_ = self._new_result()
        if target_count == 0:
            return list(X), y_list

        budget = self._exemplar_budget()
        tokenizer_fn: Callable[[str], int] = getattr(
            self.backend_, "count_tokens", default_tokenizer
        )
        self._check_exemplars_fit(pos_texts, neg_texts, budget, tokenizer_fn)

        rng = make_rng(self.random_state)

        accepted = self._accepted_items()
        original_texts = set(X)
        accepted_texts: set[str] = set()
        max_batches = max(target_count, ceil(target_count / self.batch_size) * 3)
        warned_exception = False

        pbar = _tqdm(total=target_count, desc=self._progress_desc) if self.verbose else None
        try:
            for batch_idx in range(max_batches):
                remaining = target_count - len(accepted)
                if remaining <= 0:
                    break
                batch_count = min(self.batch_size, remaining)

                system, user = self._build_prompt(
                    pos_texts, neg_texts, batch_count, budget, tokenizer_fn, rng
                )
                try:
                    result = cast(
                        "BatchT",
                        complete_structured(
                            self.backend_,
                            system,
                            user,
                            self._batch_model,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens,
                        ),
                    )
                except Exception as e:
                    if not warned_exception:
                        warnings.warn(
                            f"Skipping batch due to {type(e).__name__}: {e}",
                            UserWarning,
                            stacklevel=2,
                        )
                        warned_exception = True
                    continue

                self._merge_batch_analysis(result)
                # loguru lacks isEnabledFor; when it's absent debug_log is always
                # True and loguru does its own level filtering inside debug().
                debug_log = self.logger is not None and (
                    not hasattr(self.logger, "isEnabledFor")
                    or self.logger.isEnabledFor(logging.DEBUG)
                )
                n_before = len(accepted)
                candidates = self._batch_items(result)
                reasons = self._screen_candidates(
                    candidates, pos_texts, original_texts, accepted_texts
                )
                for item, reason in zip(candidates, reasons, strict=True):
                    if len(accepted) >= target_count:
                        break
                    if reason is not None:
                        self._record_rejection(item, reason)
                        continue
                    accepted.append(item)
                    accepted_texts.add(item.text)
                    if pbar is not None:
                        pbar.update(1)
                if debug_log:
                    self.logger.debug(  # type: ignore[union-attr]
                        f"Batch {batch_idx + 1}/{max_batches}: accepted "
                        f"{len(accepted) - n_before} new sample(s) "
                        f"({len(accepted)}/{target_count} total)"
                    )
        finally:
            if pbar is not None:
                pbar.close()

        actual = len(accepted)
        if actual != target_count:
            warnings.warn(
                f"LLM returned {actual} accepted {self._items_name}, expected {target_count}",
                UserWarning,
                stacklevel=2,
            )

        generated_texts = [item.text for item in accepted]
        X_aug = list(X) + generated_texts
        y_aug = y_list + [positive_label] * len(generated_texts)
        return X_aug, y_aug

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save fitted state (``generation_result_``) to a JSON file.

        The backend is not saved; re-supply it to :meth:`load`.

        Args:
            path: Destination file path.
        """
        if not hasattr(self, "generation_result_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError(
                f"This {type(self).__name__} instance is not fitted yet. "
                "Call 'fit_resample' before using this method."
            )
        with open(path, "w") as f:
            json.dump(self.generation_result_.model_dump(), f)

    @classmethod
    def load(cls, path: str | os.PathLike[str], backend: Backend | str, **kwargs: Any) -> Self:
        """Load fitted state from a JSON file written by :meth:`save`.

        Args:
            path: Path to the JSON file.
            backend: Backend instance or name (must be re-supplied; not stored in the file).
            **kwargs: Additional init parameters (e.g. ``sampling_strategy``, ``batch_size``).

        Returns:
            A fitted instance with restored ``generation_result_``.
        """
        with open(path) as f:
            data = json.load(f)
        obj = cls(backend=backend, **kwargs)
        obj.generation_result_ = cast("ResultT", cls._result_model.model_validate(data))
        return obj

    def _validate_params(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.sample_method not in _SAMPLE_METHODS:
            raise ValueError(
                f"sample_method must be one of {_SAMPLE_METHODS}, got {self.sample_method!r}"
            )
        self._validate_extra_params()
        if self._exemplar_budget() < 1:
            raise ValueError(
                f"context_limit ({self.context_limit}) leaves no token budget for exemplars "
                f"after reserving overhead ({PROMPT_OVERHEAD}) and max_tokens "
                f"({self.max_tokens}) for the generation response; lower max_tokens or "
                "raise context_limit"
            )

    def _sample_prompt_examples(
        self,
        texts: list[str],
        budget: int,
        tokenizer_fn: Callable[[str], int],
        rng: random.Random,
        limit: int | None,
    ) -> list[str]:
        sampled = sample_group(
            texts, budget, tokenizer_fn, self.sample_method, self.embedding_model, rng
        )
        if limit is not None and len(sampled) > limit:
            sampled = rng.sample(sampled, limit)
        return sampled

    def _screen_candidates(
        self,
        candidates: list[ItemT],
        pos_texts: list[str],
        original_texts: set[str],
        accepted_texts: set[str],
    ) -> list[str | None]:
        """One rejection reason (or ``None``) per candidate; see the class
        docstring for the order checks run in. Cheap per-item checks run
        first so ``_verify_candidates`` only sees items that would otherwise
        be accepted."""
        reasons: list[str | None] = []
        seen_in_batch: set[str] = set()
        for item in candidates:
            reason = self._validity_reason(item)
            if reason is None and self.deduplicate:
                if (
                    item.text in original_texts
                    or item.text in accepted_texts
                    or item.text in seen_in_batch
                ):
                    reason = "duplicate"
                else:
                    reason = self._dedup_extra_reason(item, pos_texts)
            if reason is None:
                seen_in_batch.add(item.text)
            reasons.append(reason)

        pending = [i for i, reason in enumerate(reasons) if reason is None]
        if pending:
            verdicts = self._verify_candidates([candidates[i] for i in pending])
            if len(verdicts) != len(pending):
                raise RuntimeError(
                    "_verify_candidates must return one verdict per item, got "
                    f"{len(verdicts)} for {len(pending)}"
                )
            for i, verdict in zip(pending, verdicts, strict=True):
                reasons[i] = verdict
        return reasons
