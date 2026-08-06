from __future__ import annotations

import json
import os
from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
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

    ``y`` may be any two hashable, orderable values (e.g. ``0``/``1``,
    ``"positive"``/``"negative"``); the greater of the two (``classes_[1]``)
    is treated as the model's label ``1`` and the lesser (``classes_[0]``)
    as label ``0``, mirroring ``LLMPromptingClassifier``'s convention.
    ``predict_proba``'s columns follow ``classes_`` order.
    """

    def __init__(
        self,
        model_name: str = "bert-base-multilingual-cased",
        *,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 8,
        max_length: int = 128,
        device: str | None = None,
        seed: int | None = None,
    ) -> None:
        """``model_name`` is a Hugging Face Hub checkpoint id or local path
        passed to ``AutoModelForSequenceClassification.from_pretrained``/
        ``AutoTokenizer.from_pretrained``; it defaults to a multilingual BERT
        checkpoint (``bert-base-multilingual-cased``) since the pools'
        language is user-defined and not assumed to be any one language.

        ``device`` defaults to ``None``, which resolves to ``"cuda"`` if
        available, else ``"cpu"``.
        """
        self.model_name = model_name
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.seed = seed

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
        classes = sorted(set(labels))
        if len(classes) != 2:
            raise ValueError(f"y must contain exactly 2 classes, got {len(classes)}: {classes}")

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.classes_ = np.array(classes)
        self.device_ = self._resolve_device()
        self.tokenizer_ = AutoTokenizer.from_pretrained(self.model_name)
        self.model_ = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self.model_.to(self.device_)

        target_ids = [0 if label == classes[0] else 1 for label in labels]
        optimizer = torch.optim.AdamW(self.model_.parameters(), lr=self.learning_rate)

        self.model_.train()
        for _ in range(self.epochs):
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                batch_targets = target_ids[start : start + self.batch_size]
                encoded = self.tokenizer_(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device_)
                target_tensor = torch.tensor(batch_targets, device=self.device_)

                optimizer.zero_grad()
                outputs = self.model_(**encoded, labels=target_tensor)
                outputs.loss.backward()
                optimizer.step()
        self.model_.eval()
        return self

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

        probs: list[NDArray[Any]] = []
        with torch.no_grad():
            for start in range(0, len(texts), self.batch_size):
                batch_texts = texts[start : start + self.batch_size]
                encoded = self.tokenizer_(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device_)
                logits = self.model_(**encoded).logits
                probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
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
