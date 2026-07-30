from __future__ import annotations

import json
import logging
import os
import random
import warnings
from math import ceil
from typing import Any, Protocol

from sklearn.base import BaseEstimator

from .. import dedup
from .._backend_resolve import resolve_backend
from .._sklearn import LLMEstimatorMixin
from ..backends.base import Backend
from ..selection import _SAMPLE_METHODS, sample_group
from . import prompts
from ._structured import complete_structured
from ._types import SyntheticGenerationResult

__all__ = ["SyntheticSampler"]

_PROMPT_OVERHEAD = 500
_DEFAULT_CHARS_PER_TOKEN = 4


class _Logger(Protocol):
    def debug(self, msg: object, /, *args: object, **kwargs: object) -> None: ...


def _default_tokenizer(text: str) -> int:
    """Rough token-count fallback for backends without ``count_tokens``
    (e.g. ``LlamaCppBackend`` has one; other backends may not)."""
    return len(text) // _DEFAULT_CHARS_PER_TOKEN + 1


class SyntheticSampler(LLMEstimatorMixin, BaseEstimator):  # type: ignore[misc]
    """LLM-based sampler that generates anonymized, representative synthetic
    positives (positive/negative → text, "pn2t").

    Unlike ``OverSampler`` (which generates *hard positives* -- boundary-
    adjacent texts meant to challenge a classifier -- for data augmentation),
    ``SyntheticSampler`` generates *typical, ordinary* positive-class texts
    that carry no specific identifying detail from the original exemplars
    (proper nouns, names, exact dates/numbers, locations, verbatim phrases
    are generalized away). The purpose isn't classifier augmentation but
    publishable data: when the original positive pool can't be shared for
    privacy reasons, a synthetic pool that reproduces its style/content
    distribution can be.

    Implements the imbalanced-learn ``fit_resample(X, y)`` interface, same as
    ``OverSampler``, for API consistency (and so it drops into an
    ``imblearn.pipeline.Pipeline`` the same way). Both classes are validated
    the same way and require both a positive and a negative pool -- but
    unlike ``OverSampler``, the negative pool here is used only for that
    validation; it is never included in the generation prompt, since showing
    negatives would frame generation around the positive/negative boundary,
    which is exactly what this class must avoid (see the algorithm notes
    below). Only binary labels (``0``/``1``) are supported, and only the
    positive (``1``) side is generated.

    Anonymity is a best-effort property of the prompt plus a lightweight
    verbatim-substring filter (``min_verbatim_span``); this is a heuristic,
    not a guarantee -- it catches exact copied spans but not paraphrased
    leaks (e.g. "John Smith" → "the customer named John" would not be
    caught).

    ``context_limit`` is the token budget for the *whole* per-batch prompt
    (exemplars + fixed overhead), and ``max_tokens`` is reserved out of it
    for the generation response. Unlike ``OverSampler`` (which splits its
    budget between positive and negative exemplars), ``SyntheticSampler``
    only samples the positive side, so the entire remainder goes to positive
    exemplars: ``budget = context_limit - overhead - max_tokens`` (no
    halving) -- a ``context_limit`` tuned for ``OverSampler`` will admit more
    exemplars per batch here.

    ``sample_method`` picks how positive exemplars are chosen from the pool
    within that token budget: ``"random"`` (default) is a uniform random
    subset; ``"kmeans"``/``"votek"`` embed the pool via ``embedding_model``
    (requires the ``pntx[embeddings]`` extra) for a more representative/
    diverse exemplar set.

    ``temperature`` defaults to ``1.0`` (unlike ``t2pn.Classifier``, which
    defaults to ``0.0``) since generation benefits from varied output across
    batches, whereas classification should be as deterministic as possible.

    Fitted attributes:
        generation_result_: Full LLM response including style/content
            feature analysis and, for each generated text, a note on what
            categories of detail were generalized away (for auditing).

    Example::

        from pntx.pn2t import SyntheticSampler

        sampler = SyntheticSampler(
            backend="llama", backend_kwargs={"model_path": "model.gguf"}, n_synthesized=10
        )
        X_syn, y_syn = sampler.fit_resample(texts, labels)

        for st in sampler.generation_result_.synthetic_texts:
            print(st.text)
            print("  generalized:", st.generalized_from)
    """

    def __init__(
        self,
        backend: Backend | str,
        *,
        n_synthesized: int,
        backend_kwargs: dict[str, Any] | None = None,
        batch_size: int = 3,
        max_examples: int | None = None,
        deduplicate: bool = True,
        min_verbatim_span: int = 20,
        context_limit: int = 100_000,
        max_tokens: int = 1024,
        language: str | None = None,
        seed: int | None = None,
        sample_method: str = "random",
        embedding_model: str = "paraphrase-albert-small-v2",
        temperature: float = 1.0,
        verbose: bool = False,
        logger: _Logger | None = None,
    ) -> None:
        self.backend = backend
        self.n_synthesized = n_synthesized
        self.backend_kwargs = backend_kwargs
        self.batch_size = batch_size
        self.max_examples = max_examples
        self.deduplicate = deduplicate
        self.min_verbatim_span = min_verbatim_span
        self.context_limit = context_limit
        self.max_tokens = max_tokens
        self.language = language
        self.seed = seed
        self.sample_method = sample_method
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.verbose = verbose
        self.logger = logger

    def fit_resample(self, X: list[str], y: Any) -> tuple[list[str], list[int]]:
        """Generate anonymized synthetic positives and append them to the dataset.

        Args:
            X: Raw texts. Must be a list of strings, not a numeric feature matrix.
            y: Binary labels. Positive class must be 1; negative class must be 0.
                The negative pool is only used to validate ``y`` -- it is never
                shown to the backend.

        Returns:
            Tuple of (augmented texts, augmented labels) where the appended
            texts are the generated synthetic positives and the appended
            labels are all 1.
        """
        if self.n_synthesized < 0:
            raise ValueError(f"n_synthesized must be >= 0, got {self.n_synthesized}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.max_examples is not None and self.max_examples < 1:
            raise ValueError(f"max_examples must be >= 1 or None, got {self.max_examples}")
        if self.max_tokens < 1:
            raise ValueError(f"max_tokens must be >= 1, got {self.max_tokens}")
        if self.min_verbatim_span < 1:
            raise ValueError(f"min_verbatim_span must be >= 1, got {self.min_verbatim_span}")
        if self.sample_method not in _SAMPLE_METHODS:
            raise ValueError(
                f"sample_method must be one of {_SAMPLE_METHODS}, got {self.sample_method!r}"
            )
        if self.context_limit - _PROMPT_OVERHEAD - self.max_tokens < 1:
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

        target_count = self.n_synthesized
        self.generation_result_ = SyntheticGenerationResult(
            style_features=[],
            content_features=[],
            synthetic_texts=[],
        )
        if target_count == 0:
            return list(X), y_list

        budget = self.context_limit - _PROMPT_OVERHEAD - self.max_tokens
        tokenizer_fn = getattr(self.backend_, "count_tokens", _default_tokenizer)

        shortest_pos = min(tokenizer_fn(t) for t in pos_texts)
        if shortest_pos > budget:
            raise ValueError(
                f"no positive example fits within the exemplar budget ({budget} tokens); "
                f"the shortest positive text is {shortest_pos} tokens. Raise context_limit, "
                "lower max_tokens, or remove the outlier text."
            )

        rng = random.Random(self.seed)

        original_texts = set(X)
        accepted_texts: set[str] = set()
        max_batches = max(target_count, ceil(target_count / self.batch_size) * 3)
        warned_exception = False

        if self.verbose:
            pbar = _tqdm(total=target_count, desc="Generating synthetic positives")

        try:
            for batch_idx in range(max_batches):
                remaining = target_count - len(self.generation_result_.synthetic_texts)
                if remaining <= 0:
                    break

                pos_sampled = self._sample_prompt_examples(pos_texts, budget, tokenizer_fn, rng)
                batch_count = min(self.batch_size, remaining)

                system = prompts.build_synthetic_system_message()
                user = prompts.build_synthetic_user_message(
                    pos_texts=pos_sampled,
                    n_synthesized=batch_count,
                    language=self.language,
                )
                try:
                    result = complete_structured(
                        self.backend_,
                        system,
                        user,
                        SyntheticGenerationResult,
                        temperature=self.temperature,
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

                self.generation_result_.style_features.extend(result.style_features)
                self.generation_result_.content_features.extend(result.content_features)
                # loguru lacks isEnabledFor; when it's absent debug_log is always
                # True and loguru does its own level filtering inside debug().
                debug_log = self.logger is not None and (
                    not hasattr(self.logger, "isEnabledFor")
                    or self.logger.isEnabledFor(logging.DEBUG)
                )
                if debug_log:
                    n_before = len(self.generation_result_.synthetic_texts)
                for st in result.synthetic_texts:
                    if len(self.generation_result_.synthetic_texts) >= target_count:
                        break
                    if self._accept_generated_text(
                        st.text, pos_texts, original_texts, accepted_texts
                    ):
                        self.generation_result_.synthetic_texts.append(st)
                        if pbar is not None:
                            pbar.update(1)
                if debug_log:
                    n_accepted = len(self.generation_result_.synthetic_texts) - n_before
                    n_total = len(self.generation_result_.synthetic_texts)
                    self.logger.debug(  # type: ignore[union-attr]
                        f"Batch {batch_idx + 1}/{max_batches}: accepted {n_accepted} "
                        f"new sample(s) ({n_total}/{target_count} total)"
                    )
        finally:
            if pbar is not None:
                pbar.close()

        actual = len(self.generation_result_.synthetic_texts)
        if actual != target_count:
            warnings.warn(
                f"LLM returned {actual} accepted synthetic texts, expected {target_count}",
                UserWarning,
                stacklevel=2,
            )

        generated_texts = [st.text for st in self.generation_result_.synthetic_texts]
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
                "This SyntheticSampler instance is not fitted yet. "
                "Call 'fit_resample' before using this method."
            )
        with open(path, "w") as f:
            json.dump(self.generation_result_.model_dump(), f)

    @classmethod
    def load(
        cls, path: str | os.PathLike[str], backend: Backend | str, **kwargs: Any
    ) -> SyntheticSampler:
        """Load fitted state from a JSON file.

        Args:
            path: Path to the JSON file written by :meth:`save`.
            backend: Backend instance or name (must be re-supplied; not stored in the file).
            **kwargs: Additional init parameters (e.g. ``n_synthesized``, ``batch_size``).

        Returns:
            A fitted :class:`SyntheticSampler` instance with restored generation results.
        """
        with open(path) as f:
            data = json.load(f)
        kwargs.setdefault("n_synthesized", 0)
        obj = cls(backend=backend, **kwargs)
        obj.generation_result_ = SyntheticGenerationResult.model_validate(data)
        return obj

    def _sample_prompt_examples(
        self,
        texts: list[str],
        budget: int,
        tokenizer_fn: Any,
        rng: random.Random,
    ) -> list[str]:
        sampled = sample_group(
            texts, budget, tokenizer_fn, self.sample_method, self.embedding_model, rng
        )
        if self.max_examples is not None and len(sampled) > self.max_examples:
            sampled = rng.sample(sampled, self.max_examples)
        return sampled

    def _accept_generated_text(
        self,
        text: str,
        pos_texts: list[str],
        original_texts: set[str],
        accepted_texts: set[str],
    ) -> bool:
        if not self.deduplicate:
            return True
        if text in original_texts or text in accepted_texts:
            return False
        if dedup.contains_verbatim_span(text, pos_texts, min_len=self.min_verbatim_span):
            return False
        accepted_texts.add(text)
        return True
