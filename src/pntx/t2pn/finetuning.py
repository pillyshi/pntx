from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Iterable
from contextlib import nullcontext
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from .._labels import resolve_binary_labels

try:
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_constant_schedule_with_warmup,
    )
except ImportError as e:
    raise ImportError(
        "FineTuningClassifier requires the 'finetuning' extra. "
        "Install it with: pip install 'pntx[finetuning]'"
    ) from e

__all__ = ["FineTuningClassifier"]

_CLASSES_FILENAME = "pntx_classes.json"


class FineTuningClassifier(ClassifierMixin, BaseEstimator):  # type: ignore[misc]
    """scikit-learn ``ClassifierMixin`` that fine-tunes a pretrained
    ``transformers`` encoder (text → positive/negative, "t2pn").

    Unlike ``t2pn.LLMPromptingClassifier`` (which does no training and
    classifies via LLM few-shot prompting/scoring through the ``Backend``
    abstraction), ``fit(X, y)`` here actually trains: it loads
    ``model_name`` via ``AutoModelForSequenceClassification`` and fine-tunes
    it as a binary classifier for the number of ``epochs`` given. This is the
    one place in ``t2pn`` where "fit" means real parameter learning rather
    than pool bookkeeping.

    ``AutoModelForSequenceClassification``/``AutoTokenizer`` are
    architecture-agnostic -- swapping ``model_name`` alone supports any
    encoder checkpoint on the Hugging Face Hub (BERT, RoBERTa, DeBERTa,
    ...), so this class isn't tied to a specific architecture despite the
    default checkpoint being a BERT model. It does not use ``pntx``'s
    ``Backend`` abstraction at all -- no LLM completion/scoring is involved,
    so there is no loaded model to share with ``LLMPromptingClassifier``/
    ``pn2t``.

    ``y`` may be any two distinct hashable values; which one means
    "positive" (the model's label ``1``) is resolved by
    ``pntx._labels.resolve_binary_labels`` (see the ``pos_label``
    parameter below), the same rule ``LLMPromptingClassifier`` uses:
    numeric pairs resolve to their greater value, the exact pair
    ``"positive"``/``"negative"`` resolves directly, anything else
    requires ``pos_label``. ``classes_`` is ``[negative_label,
    positive_label]``, and ``predict_proba``'s columns follow it.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        *,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        weight_decay: float = 0.01,
        warmup_ratio: float = 0.0,
        batch_size: int = 8,
        eval_batch_size: int | None = None,
        gradient_accumulation_steps: int = 1,
        fp16: bool = False,
        bf16: bool = False,
        max_length: int = 128,
        class_weight: str | dict[Any, float] | None = None,
        device: str | None = None,
        seed: int | None = None,
        pos_label: Any = None,
    ) -> None:
        """``model_name`` is a Hugging Face Hub checkpoint id or local path
        passed to ``AutoModelForSequenceClassification.from_pretrained``/
        ``AutoTokenizer.from_pretrained``; it defaults to a multilingual BERT
        checkpoint (``bert-base-multilingual-cased``) since the pools'
        language is user-defined and not assumed to be any one language.

        ``class_weight`` follows scikit-learn's convention: ``None`` (default)
        trains on the unweighted loss; ``"balanced"`` weights each class
        inversely proportional to its frequency in the fitted ``y`` (``n_samples
        / (2 * class_count)``), same formula as
        ``sklearn.utils.class_weight.compute_class_weight("balanced", ...)``;
        or a ``{class_label: weight}`` dict (keys must match the two distinct
        values seen in ``y``) for explicit weights. Unlike
        ``LLMPromptingClassifier``, which rebalances by trimming the larger
        exemplar pool per prompt, an imbalanced fitted pool here just skews the
        loss towards the majority class unless ``class_weight`` corrects for it.

        ``eval_batch_size`` controls the batch size used by ``predict``/
        ``predict_proba``; ``None`` (default) reuses ``batch_size``. Kept
        separate from ``batch_size`` since inference has no optimizer state
        to hold in memory and can typically afford a larger batch than
        training on the same VRAM budget.

        ``gradient_accumulation_steps`` (default ``1``, meaning no
        accumulation) delays the optimizer step for this many consecutive
        batches, summing their (scaled) gradients first -- the effective
        batch size for a gradient update becomes ``batch_size *
        gradient_accumulation_steps`` without raising the per-step memory
        footprint of ``batch_size`` itself. Each epoch flushes any
        incomplete trailing group of batches with its own optimizer step, so
        accumulation never carries across an epoch boundary.

        ``fp16``/``bf16`` (both default ``False``, mutually exclusive) run
        the forward pass under ``torch.autocast`` in half precision to
        reduce VRAM usage; ``fp16`` additionally uses a ``torch.amp.
        GradScaler`` to guard against gradient underflow (unneeded for
        ``bf16``, whose exponent range matches fp32). Both require a CUDA
        device: since the fitted/loaded ``device`` resolves to ``"cpu"``
        without a GPU present, ``fp16=True`` or ``bf16=True`` combined with
        a CPU device raises ``ValueError`` at ``fit``/``load`` time rather
        than silently falling back to full precision.

        ``weight_decay`` (default ``0.01``, matching ``torch.optim.AdamW``'s
        own default) is passed straight through to the optimizer's L2
        penalty.

        ``warmup_ratio`` (default ``0.0``) is the fraction of total
        optimizer steps (across all epochs) spent linearly ramping the
        learning rate up from ``0`` to ``learning_rate`` before holding it
        constant for the remainder of training (via ``transformers``'
        ``get_constant_schedule_with_warmup``); the default ``0.0`` means
        zero warmup steps, i.e. ``learning_rate`` is constant from the
        first step, reproducing training as it behaved before this
        scheduler was introduced. Must be in ``[0, 1]``.

        ``device`` defaults to ``None``, which resolves to ``"cuda"`` if
        available, else ``"cpu"``.

        ``pos_label`` says which of the two values in ``fit``'s ``y`` means
        "positive"; ``None`` (default) auto-resolves it (numeric: greater
        value; ``"positive"``/``"negative"``: used directly) and raises
        ``ValueError`` if ``y``'s two values don't fit either rule.
        """
        self.model_name = model_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.bf16 = bf16
        self.max_length = max_length
        self.class_weight = class_weight
        self.device = device
        self.seed = seed
        self.pos_label = pos_label

    def fit(self, X: Iterable[str], y: Iterable[Any]) -> FineTuningClassifier:
        """Fine-tune ``model_name`` as a binary classifier over ``(X, y)``.

        ``y`` must contain exactly two distinct classes. This actually
        trains (unlike every other ``fit``/``fit_resample`` in ``pntx``,
        which only holds example material) -- see the class docstring.
        """
        texts = list(X)
        labels = list(y)
        if len(texts) != len(labels):
            raise ValueError(
                f"X and y must have the same length, got len(X)={len(texts)} "
                f"and len(y)={len(labels)}"
            )
        if not 0.0 <= self.warmup_ratio <= 1.0:
            raise ValueError(f"warmup_ratio must be between 0 and 1, got {self.warmup_ratio!r}")
        negative_label, positive_label = resolve_binary_labels(labels, pos_label=self.pos_label)
        classes = [negative_label, positive_label]

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.classes_ = np.array(classes)
        self.device_ = self._resolve_device()
        self._validate_precision()
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self.model_.to(self.device_)

        target_ids = [0 if label == classes[0] else 1 for label in labels]
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.CrossEntropyLoss(weight=self._resolve_class_weight(classes, target_ids))
        scaler = torch.amp.GradScaler("cuda", enabled=self.fp16)

        # Optimizer-step count matches the flush-per-epoch accounting below
        # (a trailing partial accumulation group still gets its own step).
        num_batches_per_epoch = math.ceil(len(texts) / self.batch_size)
        steps_per_epoch = math.ceil(num_batches_per_epoch / self.gradient_accumulation_steps)
        total_steps = self.epochs * steps_per_epoch
        num_warmup_steps = round(self.warmup_ratio * total_steps)
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=num_warmup_steps
        )

        def optimizer_step() -> None:
            if self.fp16:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        self.model_.train()
        shuffle_rng = random.Random(self.seed)
        indices = list(range(len(texts)))
        optimizer.zero_grad()
        for _ in range(self.epochs):
            # Reshuffle every epoch so batches mix classes throughout training.
            # Without this, callers that append same-label examples in a block
            # (e.g. pn2t.OverSampler appends generated hard positives to the
            # tail of X) end training on a run of batches skewed toward one
            # class, biasing the model via recency rather than genuine signal.
            shuffle_rng.shuffle(indices)
            accumulated = 0
            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start : start + self.batch_size]
                batch_texts = [texts[i] for i in batch_idx]
                batch_targets = [target_ids[i] for i in batch_idx]
                encoded = self.tokenizer_(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device_)
                target_tensor = torch.tensor(batch_targets, device=self.device_)

                with self._autocast():
                    # Compute the loss ourselves (rather than the model's own
                    # labels= path) so class_weight can be applied --
                    # transformers' built-in loss for a labels= call doesn't
                    # accept per-class weights.
                    logits = self.model_(**encoded).logits
                    loss = loss_fn(logits, target_tensor) / self.gradient_accumulation_steps
                if self.fp16:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                accumulated += 1

                if accumulated == self.gradient_accumulation_steps:
                    optimizer_step()
                    accumulated = 0
            # Flush a trailing partial accumulation group so every epoch ends
            # with an optimizer step reflecting all its batches; accumulation
            # never carries over into the next epoch.
            if accumulated > 0:
                optimizer_step()
        self.model_.eval()
        return self

    def _resolve_class_weight(
        self, classes: list[Any], target_ids: list[int]
    ) -> torch.Tensor | None:
        """Resolve ``class_weight`` into a ``(2,)`` tensor ordered
        ``[weight for classes[0], weight for classes[1]]``, or ``None`` for
        the unweighted loss."""
        if self.class_weight is None:
            return None
        if isinstance(self.class_weight, str):
            if self.class_weight != "balanced":
                raise ValueError(
                    "class_weight must be None, 'balanced', or a "
                    f"{{class_label: weight}} dict, got {self.class_weight!r}"
                )
            counts = np.bincount(target_ids, minlength=2)
            balanced_weights = len(target_ids) / (2 * counts)
            return torch.tensor(balanced_weights, dtype=torch.float32, device=self.device_)
        if isinstance(self.class_weight, dict):
            try:
                dict_weights = [self.class_weight[classes[0]], self.class_weight[classes[1]]]
            except KeyError as e:
                raise ValueError(
                    f"class_weight dict must have an entry for each class in {classes}, "
                    f"missing {e}"
                ) from e
            return torch.tensor(dict_weights, dtype=torch.float32, device=self.device_)
        raise ValueError(
            "class_weight must be None, 'balanced', or a {class_label: weight} dict, "
            f"got {self.class_weight!r}"
        )

    def _validate_precision(self) -> None:
        """Validate ``fp16``/``bf16``/``gradient_accumulation_steps`` against
        the resolved ``device_``. Called from ``fit``/``load`` right after
        ``device_`` is resolved, so a CPU fallback is rejected explicitly
        rather than silently training/predicting in full precision."""
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                "gradient_accumulation_steps must be >= 1, got "
                f"{self.gradient_accumulation_steps!r}"
            )
        if self.fp16 and self.bf16:
            raise ValueError("fp16 and bf16 cannot both be True")
        if (self.fp16 or self.bf16) and self.device_ == "cpu":
            flag = "fp16" if self.fp16 else "bf16"
            raise ValueError(f"{flag}=True requires a CUDA device, got device='cpu'")

    def _autocast(self) -> Any:
        """Return a fresh ``torch.autocast`` context manager for ``fp16``/
        ``bf16``, or a no-op context when neither is set. Called anew per
        forward pass rather than reused, matching ``torch.autocast``'s usual
        usage pattern."""
        if self.fp16:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        if self.bf16:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def predict(self, X: Iterable[str]) -> NDArray[Any]:
        """Predict the most likely class (from ``classes_``) for each text in ``X``."""
        proba = self.predict_proba(X)
        result: NDArray[Any] = np.asarray(self.classes_)[np.argmax(proba, axis=1)]
        return result

    def predict_proba(self, X: Iterable[str]) -> NDArray[Any]:
        """Predict class probabilities for each text in ``X`` (softmax over
        the fine-tuned model's logits). Columns follow ``classes_`` order."""
        check_is_fitted(self, "classes_")
        texts = list(X)
        if not texts:
            return np.empty((0, 2))

        eval_batch_size = (
            self.eval_batch_size if self.eval_batch_size is not None else self.batch_size
        )
        probs: list[NDArray[Any]] = []
        with torch.no_grad():
            for start in range(0, len(texts), eval_batch_size):
                batch_texts = texts[start : start + eval_batch_size]
                encoded = self.tokenizer_(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device_)
                with self._autocast():
                    logits = self.model_(**encoded).logits
                probs.append(torch.softmax(logits, dim=-1).float().cpu().numpy())
        return np.concatenate(probs, axis=0)

    def save(self, path: str | os.PathLike[str]) -> None:
        """Save the fine-tuned model, tokenizer, and label mapping to a directory.

        Unlike ``LLMPromptingClassifier``/``OverSampler``/``SyntheticSampler``,
        there is no ``backend`` to exclude -- this class has none. Instead the
        actual trained weights are persisted (via ``transformers``'
        ``save_pretrained``), which is the expensive-to-reproduce state here.

        Args:
            path: Destination directory (created if missing).
        """
        check_is_fitted(self, "classes_")
        os.makedirs(path, exist_ok=True)
        self.model_.save_pretrained(path)
        self.tokenizer_.save_pretrained(path)
        with open(os.path.join(path, _CLASSES_FILENAME), "w") as f:
            json.dump(self.classes_.tolist(), f)

    @classmethod
    def load(cls, path: str | os.PathLike[str], **kwargs: Any) -> FineTuningClassifier:
        """Load a fine-tuned model, tokenizer, and label mapping from a directory
        written by :meth:`save`.

        Args:
            path: Directory written by :meth:`save`.
            **kwargs: Additional init parameters (e.g. ``batch_size``, ``max_length``);
                ``model_name`` defaults to ``path`` if not given.

        Returns:
            A fitted :class:`FineTuningClassifier` instance.
        """
        with open(os.path.join(path, _CLASSES_FILENAME)) as f:
            classes = json.load(f)
        kwargs.setdefault("model_name", str(path))
        obj = cls(**kwargs)
        obj.classes_ = np.array(classes)
        obj.device_ = obj._resolve_device()
        obj._validate_precision()
        obj.tokenizer_ = AutoTokenizer.from_pretrained(path)
        obj.model_ = AutoModelForSequenceClassification.from_pretrained(path)
        obj.model_.to(obj.device_)
        obj.model_.eval()
        return obj

    def _resolve_device(self) -> str:
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        tags.no_validation = True
        tags.input_tags.string = True
        tags.input_tags.two_d_array = False
        tags.target_tags.required = True
        return tags
