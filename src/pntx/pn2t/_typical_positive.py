from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np

from .. import dedup
from ..backends.base import Backend
from . import prompts
from ._base import (
    DEPRECATED,
    PROMPT_OVERHEAD,
    BaseLLMOverSampler,
    SamplingStrategy,
    _Logger,
)
from ._types import SyntheticGenerationResult, SyntheticText

__all__ = ["TypicalPositiveOverSampler"]


class TypicalPositiveOverSampler(BaseLLMOverSampler[SyntheticGenerationResult, SyntheticText]):
    """LLM-based over-sampler that generates typical, representative
    positives with specific details generalized away (positive/negative →
    text, "pn2t"). (Named ``SyntheticSampler`` before 0.16.0; that name
    remains as a deprecated alias.)

    Unlike ``HardPositiveOverSampler`` (which generates *hard positives* --
    boundary-adjacent texts meant to challenge a classifier -- for data
    augmentation), ``TypicalPositiveOverSampler`` generates *typical,
    ordinary* positive-class texts that carry no specific identifying detail
    from the original exemplars (proper nouns, names, exact dates/numbers,
    locations, verbatim phrases are generalized away). The intended use is
    publishable data: when the original positive pool can't be shared as-is,
    a synthetic pool that reproduces its style/content distribution can be.

    Implements the imbalanced-learn ``fit_resample(X, y)`` interface, same as
    ``HardPositiveOverSampler``, and returns the original data with the
    generated positives appended. Both classes are validated the same way
    and require both a positive and a negative pool -- but here the negative
    pool is used only for that validation; it is never included in the
    generation prompt, since showing negatives would frame generation around
    the positive/negative boundary, which is exactly what this class must
    avoid. Only binary ``y`` is supported, and only the positive side is
    generated. Which of the two values in ``y`` means "positive" is
    resolved by ``pntx._labels.resolve_binary_labels`` (see the
    ``pos_label`` parameter below) -- the same rule
    ``HardPositiveOverSampler``/``t2pn.LLMPromptingClassifier``/
    ``t2pn.FineTuningClassifier`` use.

    ``sampling_strategy`` follows imbalanced-learn's semantics, restricted to
    generating the positive class (see ``HardPositiveOverSampler``), but has
    **no default**: there is no natural target size for a publishable
    synthetic set, so it must be given explicitly -- typically as a dict
    ``{pos_label: n_positive_after_resampling}``.

    Removing identifying details is a best-effort property of the prompt
    plus a lightweight verbatim-substring filter (``min_verbatim_span``);
    this is a heuristic, not a guarantee -- it catches exact copied spans
    but not paraphrased leaks (e.g. "John Smith" → "the customer named John"
    would not be caught).

    In particular, this is **not** differentially private: positive
    exemplars are placed in the generation prompt verbatim, so nothing
    formally bounds how much any single exemplar can influence (or leak
    into) the output. Methods with a formal (ε, δ)-DP guarantee keep private
    text out of the prompt entirely -- e.g. Aug-PE (Xie et al. 2024), which
    only lets private samples cast noisy nearest-neighbor votes over
    LLM-generated candidates, or DP fine-tuning of the generator (Yue et al.
    2023). Use one of those when a formal guarantee is required.

    ``context_limit`` is the token budget for the *whole* per-batch prompt
    (exemplars + fixed overhead), and ``max_tokens`` is reserved out of it
    for the generation response. Unlike ``HardPositiveOverSampler`` (which
    splits its budget between positive and negative exemplars), this class
    only samples the positive side, so the entire remainder goes to positive
    exemplars: ``budget = context_limit - overhead - max_tokens`` (no
    halving) -- a ``context_limit`` tuned for ``HardPositiveOverSampler``
    will admit more exemplars per batch here.

    ``sample_method`` picks how positive exemplars are chosen from the pool
    within that token budget: ``"random"`` (default) is a uniform random
    subset; ``"kmeans"``/``"votek"`` embed the pool via ``embedding_model``
    (requires the ``pntx[embeddings]`` extra) for a more representative/
    diverse exemplar set.

    ``temperature`` defaults to ``1.0`` (unlike ``t2pn.LLMPromptingClassifier``, which
    defaults to ``0.0``) since generation benefits from varied output across
    batches, whereas classification should be as deterministic as possible.

    ``n_synthesized`` and ``seed`` are deprecated aliases (since 0.16.0,
    removed in 0.18.0) for ``sampling_strategy`` and ``random_state``.

    Fitted attributes:
        generation_result_: Full LLM response including style/content
            feature analysis and, for each generated text, a note on what
            categories of detail were generalized away (for auditing).

    Example::

        from pntx.pn2t import TypicalPositiveOverSampler

        n_pos = sum(1 for label in labels if label == 1)
        sampler = TypicalPositiveOverSampler(
            backend="llama",
            backend_kwargs={"model_path": "model.gguf"},
            sampling_strategy={1: n_pos + 10},  # generate 10 positives
        )
        X_aug, y_aug = sampler.fit_resample(texts, labels)

        for st in sampler.generation_result_.synthetic_texts:
            print(st.text)
            print("  generalized:", st.generalized_from)
    """

    _result_model = SyntheticGenerationResult
    _progress_desc = "Generating typical positives"
    _items_name = "typical positives"
    _default_sampling_strategy = None

    def __init__(
        self,
        backend: Backend | str,
        *,
        sampling_strategy: SamplingStrategy | None = None,
        backend_kwargs: dict[str, Any] | None = None,
        batch_size: int = 3,
        max_examples: int | None = None,
        deduplicate: bool = True,
        min_verbatim_span: int = 20,
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
        n_synthesized: int | str = DEPRECATED,
        seed: int | None | str = DEPRECATED,
    ) -> None:
        """``sampling_strategy`` is required (``None`` only so the deprecated
        ``n_synthesized`` can stand in for it until 0.18.0); ``fit_resample``
        raises ``ValueError`` if neither is given.

        ``pos_label`` says which of the two values in ``fit_resample``'s
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
        self.n_synthesized = n_synthesized
        self.seed = seed

    def _validate_extra_params(self) -> None:
        if self.max_examples is not None and self.max_examples < 1:
            raise ValueError(f"max_examples must be >= 1 or None, got {self.max_examples}")
        if self.min_verbatim_span < 1:
            raise ValueError(f"min_verbatim_span must be >= 1, got {self.min_verbatim_span}")

    def _exemplar_budget(self) -> int:
        return self.context_limit - PROMPT_OVERHEAD - self.max_tokens

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

    def _new_result(self) -> SyntheticGenerationResult:
        return SyntheticGenerationResult(
            style_features=[],
            content_features=[],
            synthetic_texts=[],
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
        system = prompts.build_synthetic_system_message()
        user = prompts.build_synthetic_user_message(
            pos_texts=pos_sampled,
            n_synthesized=batch_count,
            language=self.language,
        )
        return system, user

    def _merge_batch_analysis(self, result: SyntheticGenerationResult) -> None:
        self.generation_result_.style_features.extend(result.style_features)
        self.generation_result_.content_features.extend(result.content_features)

    def _batch_items(self, result: SyntheticGenerationResult) -> list[SyntheticText]:
        return result.synthetic_texts

    def _accepted_items(self) -> list[SyntheticText]:
        return self.generation_result_.synthetic_texts

    def _passes_extra_checks(self, text: str, pos_texts: list[str]) -> bool:
        return not dedup.contains_verbatim_span(text, pos_texts, min_len=self.min_verbatim_span)
