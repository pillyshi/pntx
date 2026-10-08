from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

import numpy as np

from ..backends.base import Backend
from . import prompts
from ._base import (
    PROMPT_OVERHEAD,
    BaseLLMOverSampler,
    SamplingStrategy,
    _Logger,
)
from ._types import HardPositive, HardPositiveGenerationResult

__all__ = ["HardPositiveOverSampler"]


class HardPositiveOverSampler(
    BaseLLMOverSampler[HardPositiveGenerationResult, HardPositiveGenerationResult, HardPositive]
):
    """LLM-based over-sampler that generates hard positives for binary
    classification (positive/negative → text, "pn2t").

    Full port of semaxis's ``HardPositiveOverSampler``
    (github.com/pillyshi/semaxis), with LLM calls routed through pntx's own
    ``Backend`` abstraction instead of semaxis's separate LLM client, so
    ``t2pn.LLMPromptingClassifier`` and ``HardPositiveOverSampler`` can share
    one loaded model instead of each loading their own. (Named
    ``OverSampler`` before 0.16.0.)

    Implements the imbalanced-learn ``fit_resample(X, y)`` interface for
    text arrays. ``imbalanced-learn`` itself is not imported or required --
    this only duck-types the method, so ``imblearn.pipeline.Pipeline`` still
    works with it if imbalanced-learn happens to be installed separately.
    ``X`` is a list of raw strings, not a numeric feature matrix. Only
    binary ``y`` is supported, and only the positive side is generated (v1
    scope -- negative-side generation is out of scope). Which of the two
    values in ``y`` means "positive" is resolved by
    ``pntx._labels.resolve_binary_labels`` (see the ``pos_label`` parameter
    below) -- the same rule ``t2pn.LLMPromptingClassifier``/
    ``t2pn.FineTuningClassifier`` use, so a dataset's label encoding doesn't
    need to change between ``t2pn`` and ``pn2t``. Generated texts are
    appended to ``X_aug`` under that positive label (not hardcoded ``1``).

    The LLM analyzes positive and negative examples to identify
    boundary-defining features, then synthesizes texts that carry all
    essential positive-class criteria while being superficially ambiguous --
    texts that experts would label positive but that shallow classifiers or
    untrained humans might label negative.

    Hard positives are *not* contrast sets (Gardner et al. 2020) or
    counterfactually-augmented data (Kaushik et al. 2020): those are minimal
    edits of an existing example that flip its gold label, whereas every
    text generated here is a new example whose gold label stays positive
    and only its surface is meant to look negative. No classifier is in the
    loop either -- "hard" is the LLM's judgment from the boundary-feature
    analysis, not something verified against a model's predictions.

    ``sampling_strategy`` follows imbalanced-learn's semantics (see
    ``pntx.pn2t._base.resolve_n_to_generate``), restricted to generating the
    positive class: ``"auto"`` (default) generates until positives match
    negatives; a float in ``(0, 1]`` is the desired positive/negative ratio
    after resampling (positive must be the minority); a dict
    ``{pos_label: n}`` is the desired *total* number of positives after
    resampling; a callable ``f(y) -> dict`` is interpreted the same way.

    ``context_limit`` is the token budget for the *whole* per-batch prompt
    (exemplars + fixed overhead), and ``max_tokens`` is reserved out of it
    for the generation response -- pass a backend's actual context window
    (e.g. ``LlamaCppBackend``'s ``n_ctx``) as ``context_limit`` rather than
    subtracting an output reservation yourself; ``max_tokens`` already
    accounts for that split (exemplar budget = ``(context_limit -
    overhead - max_tokens) / 2``, per class).

    ``sample_method`` picks how exemplars are chosen from each side's pool
    within that per-class token budget: ``"random"`` (default) is a uniform
    random subset; ``"kmeans"``/``"votek"`` embed the pool via
    ``embedding_model`` (a sentence-transformers model name, requires the
    ``pntx[embeddings]`` extra) and pick one representative text per K-Means
    cluster, or run the Vote-K algorithm (Su et al. 2022), respectively --
    both aim for a more representative/diverse exemplar set than a random
    subset.

    ``temperature`` defaults to ``1.0`` (unlike ``t2pn.LLMPromptingClassifier``, which
    defaults to ``0.0``) since hard-positive generation benefits from varied
    output across batches, whereas classification should be as deterministic
    as possible.

    Fitted attributes:
        generation_result_: Full LLM response including feature analysis and
            per-sample evidence. Useful for auditing boundary-defining
            features and why each generated text is considered a hard
            positive.

    Example::

        from pntx.pn2t import HardPositiveOverSampler

        sampler = HardPositiveOverSampler(
            backend="llama", backend_kwargs={"model_path": "model.gguf"}
        )
        X_aug, y_aug = sampler.fit_resample(texts, labels)

        for hp in sampler.generation_result_.hard_positives:
            print(hp.text)
            print("  evidence:", hp.positive_evidence)
    """

    _result_model = HardPositiveGenerationResult
    _batch_model = HardPositiveGenerationResult
    _progress_desc = "Generating hard positives"
    _items_name = "hard positives"

    def __init__(
        self,
        backend: Backend | str,
        *,
        backend_kwargs: dict[str, Any] | None = None,
        sampling_strategy: SamplingStrategy = "auto",
        batch_size: int = 3,
        max_examples_per_class: int | None = None,
        deduplicate: bool = True,
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
        self.backend_kwargs = backend_kwargs
        self.sampling_strategy = sampling_strategy
        self.batch_size = batch_size
        self.max_examples_per_class = max_examples_per_class
        self.deduplicate = deduplicate
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
        if self.max_examples_per_class is not None and self.max_examples_per_class < 1:
            raise ValueError(
                f"max_examples_per_class must be >= 1 or None, got {self.max_examples_per_class}"
            )

    def _exemplar_budget(self) -> int:
        return (self.context_limit - PROMPT_OVERHEAD - self.max_tokens) // 2

    def _check_exemplars_fit(
        self,
        pos_texts: list[str],
        neg_texts: list[str],
        budget: int,
        tokenizer_fn: Callable[[str], int],
    ) -> None:
        for side, texts in (("positive", pos_texts), ("negative", neg_texts)):
            shortest = min(tokenizer_fn(t) for t in texts)
            if shortest > budget:
                raise ValueError(
                    f"no {side} example fits within the per-class exemplar budget "
                    f"({budget} tokens); the shortest {side} text is {shortest} "
                    "tokens. Raise context_limit, lower max_tokens, or remove the "
                    "outlier text."
                )

    def _new_result(self) -> HardPositiveGenerationResult:
        return HardPositiveGenerationResult(
            positive_features=[],
            negative_features=[],
            boundary_features=[],
            hard_positives=[],
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
        limit = self.max_examples_per_class
        pos_sampled = self._sample_prompt_examples(pos_texts, budget, tokenizer_fn, rng, limit)
        neg_sampled = self._sample_prompt_examples(neg_texts, budget, tokenizer_fn, rng, limit)
        # Each side is sampled within the same token budget, but text
        # length differs per class, so item counts can end up
        # unequal; trim the larger side down so the boundary-feature
        # analysis sees a balanced number of examples per class.
        n_balanced = min(len(pos_sampled), len(neg_sampled))
        if len(pos_sampled) > n_balanced:
            pos_sampled = rng.sample(pos_sampled, n_balanced)
        if len(neg_sampled) > n_balanced:
            neg_sampled = rng.sample(neg_sampled, n_balanced)

        system = prompts.build_system_message()
        user = prompts.build_user_message(
            pos_texts=pos_sampled,
            neg_texts=neg_sampled,
            n_synthesized=batch_count,
            language=self.language,
        )
        return system, user

    def _merge_batch_analysis(self, result: HardPositiveGenerationResult) -> None:
        self.generation_result_.positive_features.extend(result.positive_features)
        self.generation_result_.negative_features.extend(result.negative_features)
        self.generation_result_.boundary_features.extend(result.boundary_features)

    def _batch_items(self, result: HardPositiveGenerationResult) -> list[HardPositive]:
        return result.hard_positives

    def _accepted_items(self) -> list[HardPositive]:
        return self.generation_result_.hard_positives
