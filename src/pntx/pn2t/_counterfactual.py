from __future__ import annotations

import random
from collections.abc import Callable
from numbers import Integral
from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

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
from ._types import (
    CounterfactualBatch,
    CounterfactualEdit,
    CounterfactualGenerationResult,
    RejectedEdit,
)

__all__ = ["CounterfactualOverSampler"]

_VERIFY_STRATEGIES = ("none", "self")
_ARROWS = ("->", "→", "=>")


class CounterfactualOverSampler(
    BaseLLMOverSampler[CounterfactualGenerationResult, CounterfactualBatch, CounterfactualEdit]
):
    """LLM-based over-sampler that makes positives by minimally editing
    negatives (counterfactually-augmented data; positive/negative → text,
    "pn2t").

    Each generated text is the smallest edit of one existing *negative*
    example (its "pivot") that makes the positive label apply, following the
    counterfactual-revision instructions of Kaushik et al. (2020): the target
    label must apply, the text must stay coherent, and nothing unnecessary may
    change. Because the pivot is already in ``X``, appending the edited
    positive yields an original/counterfactual *pair* -- the mechanism
    Kaushik et al. show breaks spurious correlations (features that never
    change across a pair stop being predictive) -- while only ever generating
    positive-side text. Positive examples are shown in the prompt only as a
    reference for what "positive" means here; they are never edited.

    Unlike ``HardPositiveOverSampler`` (brand-new texts meant to look
    negative), every output here stays anchored to a real negative, so it
    also probes the classifier's *local* decision boundary in the sense of
    Gardner et al.'s (2020) contrast sets.

    Every candidate edit goes through, in order:

    1. a format check (an arrow such as ``->`` that isn't in the pivot means
       the model echoed ``"original -> edited"`` into the text), a no-op
       check, and a minimality check -- ``dedup.edit_ratio(pivot, edit)``
       must not exceed ``max_edit_ratio`` (always applied, even with
       ``deduplicate=False``);
    2. exact-match dedup against ``X`` and earlier accepted edits (when
       ``deduplicate=True``);
    3. label verification per ``verify`` (below).

    Rejected candidates are recorded in ``generation_result_.rejected`` with
    a reason (``"malformed_text"``, ``"no_op"``, ``"edit_too_large"``, ``"duplicate"``,
    ``"self_check_failed"``, ``"verifier_rejected"``, ``"invalid_pivot_id"``),
    so rejection rates per filter are observable.

    ``verify`` decides how an edit that did not actually flip the label is
    caught. The default ``"self"`` was chosen by a pilot benchmark on
    Kaushik et al.'s IMDb data (``benchmarks/pn2t/counterfactual_pilot.py``;
    results in ``research/ideas/counterfactual-edit-sampler.md``): it was
    more precise than ``"none"`` at no extra cost and without losing the
    hard, valid edits, whereas a classifier verifier was the most precise
    but mostly kept edits a shallow classifier already gets right:

    - ``"none"``: trust the edit prompt.
    - ``"self"`` (default): reject edits the LLM itself marks
      ``is_positive=false`` in the same response. No extra backend calls, but
      the editor grades its own work, so it only catches some failed flips.
    - a classifier (any object with sklearn-style ``predict``, e.g.
      ``t2pn.LLMPromptingClassifier`` on the same backend, or
      ``t2pn.FineTuningClassifier``): reject edits it doesn't predict as the
      positive label. The verifier represents the *pre-augmentation* model:
      it is only ever fitted on the original ``(X, y)``, never on edits.
      Note that a model in the loop biases accepted edits toward ones that
      model already gets right (Gardner et al. 2020, §2.4) -- choose it when
      label precision matters more than hardness.

    ``verify_cv`` controls how a classifier verifier is fitted:

    - an int ``K`` (default ``5``): cross-fitting, like
      ``cross_val_predict``. ``(X, y)`` is split with
      ``StratifiedGroupKFold(K)`` (identical texts kept in one fold), one
      ``clone(verify)`` is fitted per fold on the other ``K - 1`` folds, and
      each edit is judged by the clone whose training data excludes its
      pivot. Without this, the verifier would have been trained on the very
      negative it is judging a minimal edit of, and would tend to reject
      correct flips. Clones are fitted lazily (only for folds that supply
      pivots); the passed instance itself is never fitted. ``K`` must not
      exceed the smaller class count. This costs up to ``K`` fits per
      ``fit_resample`` -- negligible for ``LLMPromptingClassifier`` (fit only
      stores pools), but ``K`` full training runs for
      ``FineTuningClassifier``; use a small ``K`` or ``"prefit"`` there.
    - ``"prefit"``: use ``verify`` as-is (already fitted elsewhere), without
      cloning or fitting, like ``CalibratedClassifierCV(cv="prefit")``.
      Avoiding pivot leakage is then up to you.

    ``verify_cv`` is ignored when ``verify`` is ``"none"`` or ``"self"``.

    ``max_edit_ratio`` defaults to ``0.5``, a deliberately lenient,
    provisional value: one-word sentiment flips score around ``0.1``-``0.15``
    and full rewrites ``0.7`` or more. Long texts and a calibrated default are
    open questions in the idea file above.

    ``sampling_strategy`` follows imbalanced-learn's semantics restricted to
    the positive class (see ``HardPositiveOverSampler``); the default
    ``"auto"`` edits negatives until the classes balance. Pivots are drawn
    from the negatives without replacement (via ``sample_method`` within the
    token budget) and only reused once every negative has been tried.

    ``context_limit``/``max_tokens`` work as in ``HardPositiveOverSampler``:
    the exemplar budget ``(context_limit - overhead - max_tokens) / 2`` is
    used once for the pivots and once for the positive references.

    Fitted attributes:
        generation_result_: Accepted edits (each with its pivot's index in
            ``X``, the pivot text, the changed spans, the LLM's
            self-assessment, and the edit ratio) and rejected candidates
            with reasons.

    The editor model matters more than the verifier: in the pilot a 7B model
    fully flipped about half of long (≤1000-char) reviews (versus ~0.8-0.9 for
    human revisions, by the same LLM judge) and a 3B model mostly rewrote
    instead of editing. Expect to need a ≥7B-class model for
    review-length texts.

    Example::

        from pntx.pn2t import CounterfactualOverSampler
        from pntx.t2pn import LLMPromptingClassifier
        from pntx.backends.llama import LlamaCppBackend

        backend = LlamaCppBackend(model_path="model.gguf")
        sampler = CounterfactualOverSampler(
            backend=backend,
            verify=LLMPromptingClassifier(backend=backend),  # cross-fitted
        )
        X_aug, y_aug = sampler.fit_resample(texts, labels)

        for edit in sampler.generation_result_.edits:
            print(edit.source_text, "->", edit.text)
    """

    _result_model = CounterfactualGenerationResult
    _batch_model = CounterfactualBatch
    _progress_desc = "Generating counterfactual edits"
    _items_name = "counterfactual edits"
    _default_sampling_strategy = "auto"

    # Introduced after the n_synthesized/seed deprecation, so it never
    # accepted them; the base class still reads these attributes.
    n_synthesized: Any = DEPRECATED
    seed: Any = DEPRECATED

    def __init__(
        self,
        backend: Backend | str,
        *,
        verify: str | Any = "self",
        verify_cv: int | str = 5,
        sampling_strategy: SamplingStrategy = "auto",
        backend_kwargs: dict[str, Any] | None = None,
        batch_size: int = 3,
        max_examples: int | None = None,
        max_edit_ratio: float = 0.5,
        deduplicate: bool = True,
        context_limit: int = 100_000,
        max_tokens: int = 1024,
        random_state: int | np.random.RandomState | None = None,
        sample_method: str = "random",
        embedding_model: str = "paraphrase-albert-small-v2",
        temperature: float = 1.0,
        verbose: bool = False,
        logger: _Logger | None = None,
        pos_label: Any = None,
    ) -> None:
        """``max_examples`` caps the number of positive reference examples
        per prompt (``None``: as many as fit the budget).

        ``pos_label`` says which of the two values in ``fit_resample``'s
        ``y`` means "positive"; ``None`` (default) auto-resolves it
        (numeric: greater value; ``"positive"``/``"negative"``: used
        directly) and raises ``ValueError`` if ``y``'s two values don't fit
        either rule. See ``pntx._labels.resolve_binary_labels``.
        """
        self.backend = backend
        self.verify = verify
        self.verify_cv = verify_cv
        self.sampling_strategy = sampling_strategy
        self.backend_kwargs = backend_kwargs
        self.batch_size = batch_size
        self.max_examples = max_examples
        self.max_edit_ratio = max_edit_ratio
        self.deduplicate = deduplicate
        self.context_limit = context_limit
        self.max_tokens = max_tokens
        self.random_state = random_state
        self.sample_method = sample_method
        self.embedding_model = embedding_model
        self.temperature = temperature
        self.verbose = verbose
        self.logger = logger
        self.pos_label = pos_label

    # ---- validation and per-fit state ---------------------------------------

    def _uses_classifier_verifier(self) -> bool:
        return not isinstance(self.verify, str)

    def _validate_extra_params(self) -> None:
        if isinstance(self.verify, str):
            if self.verify not in _VERIFY_STRATEGIES:
                raise ValueError(
                    f"verify must be one of {_VERIFY_STRATEGIES} or a classifier with a "
                    f"predict method, got {self.verify!r}"
                )
        else:
            if not callable(getattr(self.verify, "predict", None)):
                raise ValueError(
                    "verify must be one of "
                    f"{_VERIFY_STRATEGIES} or a classifier with a predict method, got "
                    f"{type(self.verify).__name__}"
                )
            if self.verify_cv != "prefit" and (
                not isinstance(self.verify_cv, Integral)
                or isinstance(self.verify_cv, bool)
                or self.verify_cv < 2
            ):
                raise ValueError(
                    f"verify_cv must be an int >= 2 or 'prefit', got {self.verify_cv!r}"
                )
        if not 0 < self.max_edit_ratio <= 1:
            raise ValueError(f"max_edit_ratio must be in (0, 1], got {self.max_edit_ratio}")
        if self.max_examples is not None and self.max_examples < 1:
            raise ValueError(f"max_examples must be >= 1 or None, got {self.max_examples}")

    def _prepare_fit(
        self,
        X: list[str],
        y: list[Any],
        *,
        positive_label: Any,
        negative_label: Any,
        random_state: Any,
    ) -> None:
        self._X = X
        self._y = y
        self._positive_label = positive_label
        self._neg_indices = [i for i, yi in enumerate(y) if yi == negative_label]
        self._unused_pivots = list(self._neg_indices)
        self._batch_pivots: list[int] = []
        self._fold_of: list[int] | None = None
        self._fold_verifiers: dict[int, Any] = {}

        if not self._uses_classifier_verifier() or self.verify_cv == "prefit":
            return
        n_splits = int(self.verify_cv)
        n_pos = len(y) - len(self._neg_indices)
        smaller = min(n_pos, len(self._neg_indices))
        if n_splits > smaller:
            raise ValueError(
                f"verify_cv={n_splits} needs at least {n_splits} samples of each class for "
                f"cross-fitting, but the smaller class has {smaller}. Lower verify_cv or "
                "pass verify_cv='prefit' with an already-fitted verifier."
            )
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
        )
        fold_of = [0] * len(X)
        # Grouping by text keeps identical texts in one fold, so a duplicate of
        # a pivot can't leak into the clone that judges its edit.
        for fold, (_, test_idx) in enumerate(splitter.split(X, y, groups=X)):
            for i in test_idx:
                fold_of[int(i)] = fold
        self._fold_of = fold_of

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
                    f"no {side} example fits within the per-side exemplar budget "
                    f"({budget} tokens); the shortest {side} text is {shortest} "
                    "tokens. Raise context_limit, lower max_tokens, or remove the "
                    "outlier text."
                )

    def _new_result(self) -> CounterfactualGenerationResult:
        return CounterfactualGenerationResult(edits=[], rejected=[])

    # ---- prompt -------------------------------------------------------------

    def _choose_pivots(
        self,
        batch_count: int,
        budget: int,
        tokenizer_fn: Callable[[str], int],
        rng: random.Random,
    ) -> list[int]:
        """Pick up to ``batch_count`` not-yet-tried negatives (indices into
        ``X``) that fit the budget; start over once every negative was tried."""
        for _ in range(2):
            if not self._unused_pivots:
                self._unused_pivots = list(self._neg_indices)
            texts = [self._X[i] for i in self._unused_pivots]
            sampled = self._sample_prompt_examples(texts, budget, tokenizer_fn, rng, batch_count)
            by_text: dict[str, list[int]] = {}
            for i in self._unused_pivots:
                by_text.setdefault(self._X[i], []).append(i)
            chosen = [by_text[t].pop(0) for t in sampled]
            if chosen:
                chosen_set = set(chosen)
                self._unused_pivots = [i for i in self._unused_pivots if i not in chosen_set]
                return chosen
            # Every remaining untried negative is too long for the budget on
            # its own; fall back to the full pool (which has one that fits,
            # per _check_exemplars_fit).
            self._unused_pivots = []
        raise RuntimeError("could not select any negative pivot within the token budget")

    def _build_prompt(
        self,
        pos_texts: list[str],
        neg_texts: list[str],
        batch_count: int,
        budget: int,
        tokenizer_fn: Callable[[str], int],
        rng: random.Random,
    ) -> tuple[str, str]:
        self._batch_pivots = self._choose_pivots(batch_count, budget, tokenizer_fn, rng)
        references = self._sample_prompt_examples(
            pos_texts, budget, tokenizer_fn, rng, self.max_examples
        )
        system = prompts.build_counterfactual_system_message()
        user = prompts.build_counterfactual_user_message(
            pos_texts=references,
            pivots=[self._X[i] for i in self._batch_pivots],
        )
        return system, user

    # ---- candidates ---------------------------------------------------------

    def _merge_batch_analysis(self, result: CounterfactualBatch) -> None:
        pass

    def _batch_items(self, result: CounterfactualBatch) -> list[CounterfactualEdit]:
        items: list[CounterfactualEdit] = []
        for out in result.edits:
            if not 0 <= out.pivot_id < len(self._batch_pivots):
                self.generation_result_.rejected.append(
                    RejectedEdit(
                        source_index=None,
                        source_text=None,
                        text=out.edited_text,
                        reason="invalid_pivot_id",
                    )
                )
                continue
            source_index = self._batch_pivots[out.pivot_id]
            source_text = self._X[source_index]
            items.append(
                CounterfactualEdit(
                    source_index=source_index,
                    source_text=source_text,
                    text=out.edited_text,
                    changed_spans=out.changed_spans,
                    self_assessed_positive=out.is_positive,
                    edit_ratio=dedup.edit_ratio(source_text, out.edited_text),
                )
            )
        return items

    def _accepted_items(self) -> list[CounterfactualEdit]:
        return self.generation_result_.edits

    def _validity_reason(self, item: CounterfactualEdit) -> str | None:
        if any(arrow in item.text and arrow not in item.source_text for arrow in _ARROWS):
            # Models echo the "original -> edited" changed_spans format into
            # the text field; the prompt asks them not to, but can't guarantee it.
            return "malformed_text"
        if item.text.strip() == item.source_text.strip():
            return "no_op"
        if item.edit_ratio > self.max_edit_ratio:
            return "edit_too_large"
        if self.verify == "self" and not item.self_assessed_positive:
            return "self_check_failed"
        return None

    def _verify_candidates(self, items: list[CounterfactualEdit]) -> list[str | None]:
        if not self._uses_classifier_verifier():
            return [None] * len(items)

        verdicts: list[str | None] = [None] * len(items)
        if self._fold_of is None:  # verify_cv="prefit"
            groups: dict[int, list[int]] = {-1: list(range(len(items)))}
        else:
            groups = {}
            for pos, item in enumerate(items):
                groups.setdefault(self._fold_of[item.source_index], []).append(pos)

        for fold, positions in groups.items():
            verifier: Any = self.verify if fold == -1 else self._fold_verifier(fold)
            predictions = verifier.predict([items[p].text for p in positions])
            for p, pred in zip(positions, predictions, strict=True):
                if pred != self._positive_label:
                    verdicts[p] = "verifier_rejected"
        return verdicts

    def _fold_verifier(self, fold: int) -> Any:
        """The clone fitted on every fold except ``fold`` (fitted on first use)."""
        if fold not in self._fold_verifiers:
            assert self._fold_of is not None
            train = [i for i, f in enumerate(self._fold_of) if f != fold]
            self._fold_verifiers[fold] = clone(self.verify).fit(
                [self._X[i] for i in train], [self._y[i] for i in train]
            )
        return self._fold_verifiers[fold]

    def _record_rejection(self, item: CounterfactualEdit, reason: str) -> None:
        self.generation_result_.rejected.append(
            RejectedEdit(
                source_index=item.source_index,
                source_text=item.source_text,
                text=item.text,
                reason=reason,
            )
        )
