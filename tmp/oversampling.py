from __future__ import annotations

import json
import logging
import os
import random
import warnings
from math import ceil
from typing import Any, Protocol

from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator

from ._base import _LLMTransformerMixin
from .llm import BaseLLMClient, LLMClient
from .supervised import _PROMPT_OVERHEAD, _SAMPLE_METHODS, _sample_group
from .prompts import hard_positive as prompts


class HardPositive(BaseModel):
    text: str
    positive_evidence: list[str]
    confusing_evidence: list[str]


class BoundaryFeature(BaseModel):
    feature: str
    importance: float = Field(ge=0.0, le=1.0)


class HardPositiveGenerationResult(BaseModel):
    positive_features: list[str]
    negative_features: list[str]
    boundary_features: list[BoundaryFeature]
    hard_positives: list[HardPositive]


class _Logger(Protocol):
    def debug(self, msg: object, /, *args: object, **kwargs: object) -> None: ...


class HardPositiveOverSampler(_LLMTransformerMixin, BaseEstimator):
    """LLM-based oversampler that generates hard positives for binary classification.

    Implements the imbalanced-learn ``fit_resample(X, y)`` interface for text arrays.
    Unlike standard oversamplers, ``X`` is a list of raw strings, not a numeric feature
    matrix. Only binary labels (0/1) are supported.

    The LLM analyzes positive and negative examples to identify boundary-defining
    features, then synthesizes texts that carry all essential positive-class criteria
    while being superficially ambiguous — texts that experts would label positive but
    that shallow classifiers or untrained humans might label negative.

    Fitted attributes:
        generation_result_: Full LLM response including feature analysis and
            per-sample evidence. Useful for auditing boundary-defining features
            and why each generated text is considered a hard positive.

    Example::

        from imblearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression

        sampler = HardPositiveOverSampler(llm="gpt-4o", n_synthesized=20)
        X_aug, y_aug = sampler.fit_resample(texts, labels)

        for hp in sampler.generation_result_.hard_positives:
            print(hp.text)
            print("  evidence:", hp.positive_evidence)
    """

    def __init__(
        self,
        llm: BaseLLMClient | str,
        n_synthesized: int | None = None,
        batch_size: int = 3,
        max_examples_per_class: int | None = None,
        deduplicate: bool = True,
        context_limit: int = 100_000,
        language: str | None = None,
        seed: int | None = None,
        sample_method: str = "random",
        embedding_model: str = "paraphrase-albert-small-v2",
        verbose: bool = False,
        logger: _Logger | None = None,
    ) -> None:
        self.llm = llm
        self.n_synthesized = n_synthesized
        self.batch_size = batch_size
        self.max_examples_per_class = max_examples_per_class
        self.deduplicate = deduplicate
        self.context_limit = context_limit
        self.language = language
        self.seed = seed
        self.sample_method = sample_method
        self.embedding_model = embedding_model
        self.verbose = verbose
        self.logger = logger

    def fit_resample(
        self,
        X: list[str],
        y: Any,
    ) -> tuple[list[str], list[int]]:
        """Generate hard positives and append them to the dataset.

        Args:
            X: Raw texts. Must be a list of strings, not a numeric feature matrix.
            y: Binary labels. Positive class must be 1; negative class must be 0.

        Returns:
            Tuple of (augmented texts, augmented labels) where the appended texts
            are the generated hard positives and the appended labels are all 1.
        """
        if self.sample_method not in _SAMPLE_METHODS:
            raise ValueError(
                f"sample_method must be one of {_SAMPLE_METHODS}, got {self.sample_method!r}"
            )
        if self.n_synthesized is not None and self.n_synthesized < 0:
            raise ValueError(f"n_synthesized must be >= 0 or None, got {self.n_synthesized}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_examples_per_class is not None and self.max_examples_per_class < 1:
            raise ValueError(
                "max_examples_per_class must be >= 1 or None, "
                f"got {self.max_examples_per_class}"
            )
        if (self.context_limit - _PROMPT_OVERHEAD) // 2 < 1:
            raise ValueError(
                f"context_limit ({self.context_limit}) leaves no token budget after overhead ({_PROMPT_OVERHEAD})"
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

        pos_texts = [t for t, yi in zip(X, y_list) if yi == 1]
        neg_texts = [t for t, yi in zip(X, y_list) if yi == 0]

        if not pos_texts:
            raise ValueError("y must contain at least one positive (1) sample")
        if not neg_texts:
            raise ValueError("y must contain at least one negative (0) sample")

        _llm = LLMClient(self.llm) if isinstance(self.llm, str) else self.llm
        _rng = random.Random(self.seed)

        if self.verbose:
            try:
                from tqdm.auto import tqdm as _tqdm
            except ImportError:
                raise ImportError(
                    "tqdm is required when verbose=True. "
                    "Install it with: pip install tqdm"
                )

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

        budget = (self.context_limit - _PROMPT_OVERHEAD) // 2
        original_texts = set(X)
        accepted_texts: set[str] = set()
        max_batches = max(target_count, ceil(target_count / self.batch_size) * 3)
        warned_empty_pos = False
        warned_empty_neg = False
        warned_exception = False

        pbar = None
        if self.verbose:
            pbar = _tqdm(total=target_count, desc="Generating hard positives")  # type: ignore[possibly-undefined]

        try:
            for batch_idx in range(max_batches):
                remaining = target_count - len(self.generation_result_.hard_positives)
                if remaining <= 0:
                    break

                pos_sampled = self._sample_prompt_examples(
                    pos_texts, budget, _llm.count_tokens, _rng
                )
                neg_sampled = self._sample_prompt_examples(
                    neg_texts, budget, _llm.count_tokens, _rng
                )

                if not pos_sampled and not warned_empty_pos:
                    warnings.warn(
                        "All positive texts exceed the per-group token budget; "
                        "the LLM will receive no positive examples.",
                        UserWarning,
                        stacklevel=2,
                    )
                    warned_empty_pos = True
                if not neg_sampled and not warned_empty_neg:
                    warnings.warn(
                        "All negative texts exceed the per-group token budget; "
                        "the LLM will receive no negative examples.",
                        UserWarning,
                        stacklevel=2,
                    )
                    warned_empty_neg = True

                batch_count = min(self.batch_size, remaining)
                messages = [
                    {"role": "system", "content": prompts.SYSTEM},
                    {"role": "user", "content": prompts.build_user_message(
                        pos_texts=pos_sampled,
                        neg_texts=neg_sampled,
                        n_synthesized=batch_count,
                        language=self.language,
                    )},
                ]
                try:
                    result = _llm.complete_structured(messages, HardPositiveGenerationResult)
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
                    or self.logger.isEnabledFor(logging.DEBUG)  # type: ignore[union-attr]
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
                        f"Batch {batch_idx + 1}/{max_batches}: accepted {n_accepted} new sample(s) ({n_total}/{target_count} total)"
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

    def save(self, path: str | os.PathLike) -> None:
        """Save fitted state to a JSON file.

        Args:
            path: Destination file path.
        """
        if not hasattr(self, "generation_result_") or self.generation_result_ is None:
            from sklearn.exceptions import NotFittedError
            raise NotFittedError(
                "This HardPositiveOverSampler instance is not fitted yet. "
                "Call 'fit_resample' before using this method."
            )
        with open(path, "w") as f:
            json.dump(self.generation_result_.model_dump(), f)

    @classmethod
    def load(cls, path: str | os.PathLike, llm: BaseLLMClient | str, **kwargs: Any) -> "HardPositiveOverSampler":
        """Load fitted state from a JSON file.

        Args:
            path: Path to the JSON file written by :meth:`save`.
            llm: LLM client or model name string (must be re-supplied; not stored in the file).
            **kwargs: Additional init parameters (e.g. ``n_synthesized``, ``batch_size``).

        Returns:
            A fitted :class:`HardPositiveOverSampler` instance with restored generation results.
        """
        with open(path) as f:
            data = json.load(f)
        obj = cls(llm=llm, **kwargs)
        obj.generation_result_ = HardPositiveGenerationResult.model_validate(data)
        return obj

    def _sample_prompt_examples(
        self,
        texts: list[str],
        budget: int,
        tokenizer_fn: Any,
        rng: random.Random,
    ) -> list[str]:
        sampled = _sample_group(
            texts, budget, tokenizer_fn,
            self.sample_method, self.embedding_model, rng,
        )
        if (
            self.max_examples_per_class is not None
            and len(sampled) > self.max_examples_per_class
        ):
            sampled = rng.sample(sampled, self.max_examples_per_class)
        return sampled

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
