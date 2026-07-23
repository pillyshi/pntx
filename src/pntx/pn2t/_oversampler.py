from __future__ import annotations

import json
import logging
import os
import warnings
from math import ceil
from typing import Any, Protocol

from sklearn.base import BaseEstimator

from .._backend_resolve import resolve_backend
from .._sklearn import LLMEstimatorMixin
from ..backends.base import Backend
from ..selection import BudgetSelector
from . import prompts
from ._structured import complete_structured
from ._types import HardPositiveGenerationResult

__all__ = ["OverSampler"]

_PROMPT_OVERHEAD = 500
_DEFAULT_CHARS_PER_TOKEN = 4


class _Logger(Protocol):
    def debug(self, msg: object, /, *args: object, **kwargs: object) -> None: ...


def _default_tokenizer(text: str) -> int:
    """Rough token-count fallback for backends without ``count_tokens``
    (e.g. ``LlamaCppBackend`` has one; ``AnthropicBackend`` doesn't)."""
    return len(text) // _DEFAULT_CHARS_PER_TOKEN + 1


class OverSampler(LLMEstimatorMixin, BaseEstimator):  # type: ignore[misc]
    """LLM-based oversampler that generates hard positives for binary
    classification (positive/negative → text, "pn2t").

    Full port of semaxis's ``HardPositiveOverSampler``
    (github.com/pillyshi/semaxis), with LLM calls routed through pntx's own
    ``Backend`` abstraction instead of semaxis's separate LLM client, so
    ``t2pn.Classifier`` and ``OverSampler`` can share one loaded model
    instead of each loading their own.

    Implements the imbalanced-learn ``fit_resample(X, y)`` interface for
    text arrays. ``imbalanced-learn`` itself is not imported or required --
    this only duck-types the method, so ``imblearn.pipeline.Pipeline`` still
    works with it if imbalanced-learn happens to be installed separately.
    ``X`` is a list of raw strings, not a numeric feature matrix. Only
    binary labels (``0``/``1``) are supported, and only the positive
    (``1``) side is generated (v1 scope -- negative-side generation and
    quality-focused "generation as deliverable" are out of scope).

    The LLM analyzes positive and negative examples to identify
    boundary-defining features, then synthesizes texts that carry all
    essential positive-class criteria while being superficially ambiguous --
    texts that experts would label positive but that shallow classifiers or
    untrained humans might label negative.

    ``context_limit`` is the token budget for the *whole* per-batch prompt
    (exemplars + fixed overhead), and ``max_tokens`` is reserved out of it
    for the generation response -- pass a backend's actual context window
    (e.g. ``LlamaCppBackend``'s ``n_ctx``) as ``context_limit`` rather than
    subtracting an output reservation yourself; ``max_tokens`` already
    accounts for that split (exemplar budget = ``(context_limit -
    overhead - max_tokens) / 2``, per class).

    Fitted attributes:
        generation_result_: Full LLM response including feature analysis and
            per-sample evidence. Useful for auditing boundary-defining
            features and why each generated text is considered a hard
            positive.

    Example::

        from pntx.pn2t import OverSampler

        sampler = OverSampler(backend="llama", backend_kwargs={"model_path": "model.gguf"})
        X_aug, y_aug = sampler.fit_resample(texts, labels)

        for hp in sampler.generation_result_.hard_positives:
            print(hp.text)
            print("  evidence:", hp.positive_evidence)
    """

    def __init__(
        self,
        backend: Backend | str,
        *,
        backend_kwargs: dict[str, Any] | None = None,
        n_synthesized: int | None = None,
        batch_size: int = 3,
        max_examples_per_class: int | None = None,
        deduplicate: bool = True,
        context_limit: int = 100_000,
        max_tokens: int = 1024,
        language: str | None = None,
        seed: int | None = None,
        verbose: bool = False,
        logger: _Logger | None = None,
    ) -> None:
        self.backend = backend
        self.backend_kwargs = backend_kwargs
        self.n_synthesized = n_synthesized
        self.batch_size = batch_size
        self.max_examples_per_class = max_examples_per_class
        self.deduplicate = deduplicate
        self.context_limit = context_limit
        self.max_tokens = max_tokens
        self.language = language
        self.seed = seed
        self.verbose = verbose
        self.logger = logger

    def fit_resample(self, X: list[str], y: Any) -> tuple[list[str], list[int]]:
        """Generate hard positives and append them to the dataset.

        Args:
            X: Raw texts. Must be a list of strings, not a numeric feature matrix.
            y: Binary labels. Positive class must be 1; negative class must be 0.

        Returns:
            Tuple of (augmented texts, augmented labels) where the appended
            texts are the generated hard positives and the appended labels
            are all 1.
        """
        if self.n_synthesized is not None and self.n_synthesized < 0:
            raise ValueError(f"n_synthesized must be >= 0 or None, got {self.n_synthesized}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_examples_per_class is not None and self.max_examples_per_class < 1:
            raise ValueError(
                "max_examples_per_class must be >= 1 or None, "
                f"got {self.max_examples_per_class}"
            )
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if (self.context_limit - _PROMPT_OVERHEAD - self.max_tokens) // 2 < 1:
            raise ValueError(
                f"context_limit ({self.context_limit}) leaves no token budget for exemplars "
                f"after reserving overhead ({_PROMPT_OVERHEAD}) and max_tokens "
                f"({self.max_tokens}) for the generation response; lower max_tokens or "
                "raise context_limit"
            )

        y_list = list(y)
        if len(X) != len(y_list):
            raise ValueError(
                f"X and y must have the same length, got len(X)={len(X)} and len(y)={len(y_list)}"
            )
        label_set = set(y_list)
        if not label_set <= {0, 1}:
            raise ValueError(
                f"y must contain only binary labels {{0, 1}}, got {label_set - {0, 1}}"
            )

        pos_texts = [t for t, yi in zip(X, y_list, strict=True) if yi == 1]
        neg_texts = [t for t, yi in zip(X, y_list, strict=True) if yi == 0]
        if not pos_texts:
            raise ValueError("y must contain at least one positive (1) sample")
        if not neg_texts:
            raise ValueError("y must contain at least one negative (0) sample")

        self.backend_ = resolve_backend(self.backend, self.backend_kwargs)

        pbar = None
        if self.verbose:
            try:
                from tqdm.auto import tqdm as _tqdm
            except ImportError:
                raise ImportError(
                    "tqdm is required when verbose=True. Install it with: pip install tqdm"
                ) from None

        target_count = (
            max(0, len(neg_texts) - len(pos_texts))
            if self.n_synthesized is None
            else self.n_synthesized
        )
        self.generation_result_ = HardPositiveGenerationResult(
            positive_features=[],
            negative_features=[],
            boundary_features=[],
            hard_positives=[],
        )
        if target_count == 0:
            return list(X), y_list

        budget = (self.context_limit - _PROMPT_OVERHEAD - self.max_tokens) // 2
        tokenizer_fn = getattr(self.backend_, "count_tokens", _default_tokenizer)
        selector = BudgetSelector(tokenizer_fn=tokenizer_fn, token_budget=budget, seed=self.seed)

        original_texts = set(X)
        accepted_texts: set[str] = set()
        max_batches = max(target_count, ceil(target_count / self.batch_size) * 3)
        warned_exception = False

        if self.verbose:
            pbar = _tqdm(total=target_count, desc="Generating hard positives")

        try:
            for batch_idx in range(max_batches):
                remaining = target_count - len(self.generation_result_.hard_positives)
                if remaining <= 0:
                    break

                pos_sampled = self._sample_prompt_examples(pos_texts, selector)
                neg_sampled = self._sample_prompt_examples(neg_texts, selector)
                batch_count = min(self.batch_size, remaining)

                system = prompts.build_system_message()
                user = prompts.build_user_message(
                    pos_texts=pos_sampled,
                    neg_texts=neg_sampled,
                    n_synthesized=batch_count,
                    language=self.language,
                )
                try:
                    result = complete_structured(
                        self.backend_,
                        system,
                        user,
                        HardPositiveGenerationResult,
                        max_tokens=self.max_tokens,
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

                self.generation_result_.positive_features.extend(result.positive_features)
                self.generation_result_.negative_features.extend(result.negative_features)
                self.generation_result_.boundary_features.extend(result.boundary_features)
                # loguru lacks isEnabledFor; when it's absent debug_log is always
                # True and loguru does its own level filtering inside debug().
                debug_log = self.logger is not None and (
                    not hasattr(self.logger, "isEnabledFor")
                    or self.logger.isEnabledFor(logging.DEBUG)
                )
                if debug_log:
                    n_before = len(self.generation_result_.hard_positives)
                for hp in result.hard_positives:
                    if len(self.generation_result_.hard_positives) >= target_count:
                        break
                    if self._accept_generated_text(hp.text, original_texts, accepted_texts):
                        self.generation_result_.hard_positives.append(hp)
                        if pbar is not None:
                            pbar.update(1)
                if debug_log:
                    n_accepted = len(self.generation_result_.hard_positives) - n_before
                    n_total = len(self.generation_result_.hard_positives)
                    self.logger.debug(  # type: ignore[union-attr]
                        f"Batch {batch_idx + 1}/{max_batches}: accepted {n_accepted} "
                        f"new sample(s) ({n_total}/{target_count} total)"
                    )
        finally:
            if pbar is not None:
                pbar.close()

        actual = len(self.generation_result_.hard_positives)
        if actual != target_count:
            warnings.warn(
                f"LLM returned {actual} accepted hard positives, expected {target_count}",
                UserWarning,
                stacklevel=2,
            )

        generated_texts = [hp.text for hp in self.generation_result_.hard_positives]
        X_aug = list(X) + generated_texts
        y_aug = y_list + [1] * len(generated_texts)
        return X_aug, y_aug

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save fitted state to a JSON file.

        Args:
            path: Destination file path.
        """
        if not hasattr(self, "generation_result_"):
            from sklearn.exceptions import NotFittedError

            raise NotFittedError(
                "This OverSampler instance is not fitted yet. "
                "Call 'fit_resample' before using this method."
            )
        with open(path, "w") as f:
            json.dump(self.generation_result_.model_dump(), f)

    @classmethod
    def load(
        cls, path: str | os.PathLike[str], backend: Backend | str, **kwargs: Any
    ) -> OverSampler:
        """Load fitted state from a JSON file.

        Args:
            path: Path to the JSON file written by :meth:`save`.
            backend: Backend instance or name (must be re-supplied; not stored in the file).
            **kwargs: Additional init parameters (e.g. ``n_synthesized``, ``batch_size``).

        Returns:
            A fitted :class:`OverSampler` instance with restored generation results.
        """
        with open(path) as f:
            data = json.load(f)
        obj = cls(backend=backend, **kwargs)
        obj.generation_result_ = HardPositiveGenerationResult.model_validate(data)
        return obj

    def _sample_prompt_examples(self, texts: list[str], selector: BudgetSelector) -> list[str]:
        k = self.max_examples_per_class if self.max_examples_per_class is not None else len(texts)
        return selector.select(texts, k)

    def _accept_generated_text(
        self,
        text: str,
        original_texts: set[str],
        accepted_texts: set[str],
    ) -> bool:
        if not self.deduplicate:
            return True
        if text in original_texts or text in accepted_texts:
            return False
        accepted_texts.add(text)
        return True
