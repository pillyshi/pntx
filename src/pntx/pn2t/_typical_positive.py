from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np

from .. import dedup
from ..backends.base import Backend
from . import prompts
from ._base import (
    BaseLLMOverSampler,
    SamplingStrategy,
    _Logger,
)
from ._types import TypicalPositive, TypicalPositiveGenerationResult

__all__ = ["TypicalPositiveOverSampler"]


class TypicalPositiveOverSampler(
    BaseLLMOverSampler[
        TypicalPositiveGenerationResult, TypicalPositiveGenerationResult, TypicalPositive
    ]
):
    """LLM-based over-sampler that generates typical, representative
    positives for classifier data augmentation (positive/negative → text,
    "pn2t"). (Named ``SyntheticSampler`` before 0.16.0.)

    The counterpart of ``HardPositiveOverSampler``: where that class
    generates *hard positives* -- boundary-adjacent texts meant to challenge
    a classifier -- ``TypicalPositiveOverSampler`` generates *typical,
    ordinary* positive-class texts that reproduce the style and content of
    the positive pool, each describing a new situation rather than
    paraphrasing an exemplar.

    Implements the imbalanced-learn ``fit_resample(X, y)`` interface, same as
    ``HardPositiveOverSampler``, and returns the original data with the
    generated positives appended. Both classes are validated the same way
    and require both a positive and a negative pool -- but here the negative
    pool is used only for that validation; it is never included in the
    generation prompt, since showing negatives would frame generation around
    the positive/negative boundary rather than the typical positive. Only
    binary ``y`` is supported, and only the positive side is generated.
    Which of the two values in ``y`` means "positive" is resolved by
    ``pntx._labels.resolve_binary_labels`` (see the ``pos_label`` parameter
    below) -- the same rule ``HardPositiveOverSampler``/
    ``t2pn.LLMPromptingClassifier``/``t2pn.FineTuningClassifier`` use.

    ``sampling_strategy`` follows imbalanced-learn's semantics, restricted to
    generating the positive class (see ``HardPositiveOverSampler``); the
    default ``"auto"`` generates until the positive class is as large as the
    negative class.

    With ``deduplicate=True`` (default), a generated text is rejected and
    regenerated when it exactly matches an existing or already accepted text
    (as in ``HardPositiveOverSampler``) or when it copies a substring of at
    least ``min_verbatim_span`` characters from a positive exemplar -- a
    cheap guard against near-copies that add little to the training data.
    The default of 50 characters is long enough that stock phrases shared by
    many texts (e.g. "The chemistry between the leads") don't count as
    copies; at 20 characters, typical English reviews were rejected for such
    phrases alone.

    This class does not anonymize: generated texts may carry details from
    the exemplars, which are placed in the prompt verbatim. It is not meant
    for producing a publishable stand-in for private data.

    ``context_limit`` and ``max_tokens`` work as in
    ``HardPositiveOverSampler`` (capped by the backend's ``context_window``,
    fixed prompt measured, final prompt checked and fitted by dropping
    exemplars). Unlike ``HardPositiveOverSampler``, which splits its budget
    between positive and negative exemplars, this class only samples the
    positive side, so the whole remainder goes to positive exemplars (no
    halving).

    ``sample_method`` picks how positive exemplars are chosen from the pool
    within that token budget: ``"random"`` (default) is a uniform random
    subset; ``"kmeans"``/``"votek"`` embed the pool via ``embedding_model``
    (requires the ``pntx[embeddings]`` extra) for a more representative/
    diverse exemplar set.

    ``temperature`` defaults to ``1.0`` (unlike ``t2pn.LLMPromptingClassifier``, which
    defaults to ``0.0``) since generation benefits from varied output across
    batches, whereas classification should be as deterministic as possible.

    Fitted attributes:
        generation_result_: Full LLM response: the style/content feature
            analysis and the generated ``typical_positives``.

    Example::

        from pntx.pn2t import TypicalPositiveOverSampler

        sampler = TypicalPositiveOverSampler(
            backend="llama",
            backend_kwargs={"model_path": "model.gguf"},
        )
        X_aug, y_aug = sampler.fit_resample(texts, labels)

        for item in sampler.generation_result_.typical_positives:
            print(item.text)
    """

    _n_prompt_sides = 1
    _result_model = TypicalPositiveGenerationResult
    _batch_model = TypicalPositiveGenerationResult
    _progress_desc = "Generating typical positives"
    _items_name = "typical positives"

    def __init__(
        self,
        backend: Backend | str,
        *,
        sampling_strategy: SamplingStrategy = "auto",
        backend_kwargs: dict[str, Any] | None = None,
        batch_size: int = 3,
        max_examples: int | None = None,
        deduplicate: bool = True,
        min_verbatim_span: int = 50,
        context_limit: int = 100_000,
        max_tokens: int = 1024,
        language: str | None = None,
        random_state: int | np.random.RandomState | None = None,
        sample_method: str = "random",
        embedding_model: str = "paraphrase-albert-small-v2",
        temperature: float = 1.0,
        verbose: bool = False,
        logger: _Logger | None = None,
        pos_label: Any = None,
    ) -> None:
        """``pos_label`` says which of the two values in ``fit_resample``'s
        ``y`` means "positive"; ``None`` (default) auto-resolves it
        (numeric: greater value; ``"positive"``/``"negative"``: used
        directly) and raises ``ValueError`` if ``y``'s two values don't fit
        either rule. See ``pntx._labels.resolve_binary_labels``.
        """
        self.backend = backend
        self.sampling_strategy = sampling_strategy
        self.backend_kwargs = backend_kwargs
        self.batch_size = batch_size
        self.max_examples = max_examples
        self.deduplicate = deduplicate
        self.min_verbatim_span = min_verbatim_span
        self.context_limit = context_limit
        self.max_tokens = max_tokens
        self.language = language
        self.random_state = random_state
        self.sample_method = sample_method
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.verbose = verbose
        self.logger = logger
        self.pos_label = pos_label

    def _validate_extra_params(self) -> None:
        if self.max_examples is not None and self.max_examples < 1:
            raise ValueError(f"max_examples must be >= 1 or None, got {self.max_examples}")
        if self.min_verbatim_span < 1:
            raise ValueError(f"min_verbatim_span must be >= 1, got {self.min_verbatim_span}")

    def _exemplar_budget(self, available: int) -> int:
        return available

    def _render_messages(self, sides: list[list[str]], batch_count: int) -> tuple[str, str]:
        (pos_texts,) = sides
        system = prompts.build_typical_positive_system_message()
        user = prompts.build_typical_positive_user_message(
            pos_texts=pos_texts,
            n_synthesized=batch_count,
            language=self.language,
        )
        return system, user

    def _check_exemplars_fit(
        self,
        pos_texts: list[str],
        neg_texts: list[str],
        budget: int,
        tokenizer_fn: Callable[[str], int],
    ) -> None:
        shortest_pos = min(tokenizer_fn(t) for t in pos_texts)
        if shortest_pos > budget:
            raise ValueError(
                f"no positive example fits within the exemplar budget ({budget} tokens); "
                f"the shortest positive text is {shortest_pos} tokens. Raise context_limit, "
                "lower max_tokens, or remove the outlier text."
            )

    def _new_result(self) -> TypicalPositiveGenerationResult:
        return TypicalPositiveGenerationResult(
            style_features=[],
            content_features=[],
            typical_positives=[],
        )

    def _build_prompt(
        self,
        pos_texts: list[str],
        neg_texts: list[str],
        batch_count: int,
        budget: int,
        tokenizer_fn: Callable[[str], int],
        rng: random.Random,
    ) -> tuple[str, str]:
        # neg_texts is deliberately unused: negatives never reach the prompt.
        pos_sampled = self._sample_prompt_examples(
            pos_texts, budget, tokenizer_fn, rng, self.max_examples
        )
        messages, _ = self._fit_prompt([pos_sampled], [1], batch_count, tokenizer_fn)
        return messages

    def _merge_batch_analysis(self, result: TypicalPositiveGenerationResult) -> None:
        self.generation_result_.style_features.extend(result.style_features)
        self.generation_result_.content_features.extend(result.content_features)

    def _batch_items(self, result: TypicalPositiveGenerationResult) -> list[TypicalPositive]:
        return result.typical_positives

    def _accepted_items(self) -> list[TypicalPositive]:
        return self.generation_result_.typical_positives

    def _dedup_extra_reason(self, item: TypicalPositive, pos_texts: list[str]) -> str | None:
        if dedup.contains_verbatim_span(item.text, pos_texts, min_len=self.min_verbatim_span):
            return "verbatim_copy"
        return None
