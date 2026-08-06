from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from sklearn.model_selection import cross_val_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.utils.validation import check_is_fitted  # noqa: E402
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer  # noqa: E402

from pntx.t2pn import FineTuningClassifier  # noqa: E402

_VOCAB = [
    "[PAD]",
    "[UNK]",
    "[CLS]",
    "[SEP]",
    "[MASK]",
    "great",
    "terrible",
    "movie",
    "service",
    "was",
    "the",
]

X = [
    "great movie",
    "great service",
    "the movie was great",
    "terrible movie",
    "terrible service",
    "the movie was terrible",
]
Y = ["positive", "positive", "positive", "negative", "negative", "negative"]


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build a randomly-initialized, offline-only tiny BERT checkpoint (no
    Hugging Face Hub access) so unit tests don't depend on a network
    download, per CLAUDE.md's testing policy for FineTuningClassifier."""
    path = tmp_path_factory.mktemp("tiny-bert")
    vocab_path = path / "vocab.txt"
    vocab_path.write_text("\n".join(_VOCAB))

    tokenizer = BertTokenizer(vocab_file=str(vocab_path))
    config = BertConfig(
        vocab_size=len(_VOCAB),
        hidden_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=64,
        num_labels=2,
    )
    model = BertForSequenceClassification(config)

    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return str(path)


def _classifier(tiny_checkpoint: str, **kwargs: object) -> FineTuningClassifier:
    kwargs.setdefault("epochs", 1)
    kwargs.setdefault("batch_size", 2)
    kwargs.setdefault("max_length", 16)
    kwargs.setdefault("seed", 0)
    return FineTuningClassifier(model_name=tiny_checkpoint, **kwargs)  # type: ignore[arg-type]


def test_fit_requires_matching_lengths(tiny_checkpoint: str) -> None:
    clf = _classifier(tiny_checkpoint)
    with pytest.raises(ValueError, match="same length"):
        clf.fit(["a", "b"], ["positive"])


def test_fit_requires_exactly_two_classes(tiny_checkpoint: str) -> None:
    clf = _classifier(tiny_checkpoint)
    with pytest.raises(ValueError, match="exactly 2 classes"):
        clf.fit(["a", "b", "c"], ["positive", "positive", "positive"])


def test_predict_before_fit_raises_not_fitted() -> None:
    from sklearn.exceptions import NotFittedError

    clf = _classifier("unused")
    with pytest.raises(NotFittedError):
        clf.predict(["some text"])


def test_fit_predict_predict_proba(tiny_checkpoint: str) -> None:
    clf = _classifier(tiny_checkpoint).fit(X, Y)
    check_is_fitted(clf, "classes_")
    assert list(clf.classes_) == ["negative", "positive"]

    proba = clf.predict_proba(X)
    assert proba.shape == (len(X), 2)
    np.testing.assert_allclose(proba.sum(axis=1), np.ones(len(X)), atol=1e-5)

    pred = clf.predict(X)
    assert set(pred) <= {"positive", "negative"}


def test_predict_proba_empty_input_returns_empty_array(tiny_checkpoint: str) -> None:
    clf = _classifier(tiny_checkpoint).fit(X, Y)
    proba = clf.predict_proba([])
    assert proba.shape == (0, 2)


def test_save_load_roundtrip(tiny_checkpoint: str, tmp_path: object) -> None:
    save_dir = os.path.join(str(tmp_path), "saved")
    clf = _classifier(tiny_checkpoint).fit(X, Y)
    proba_before = clf.predict_proba(X)

    clf.save(save_dir)
    loaded = FineTuningClassifier.load(save_dir, batch_size=2, max_length=16)
    assert list(loaded.classes_) == list(clf.classes_)

    proba_after = loaded.predict_proba(X)
    np.testing.assert_allclose(proba_before, proba_after, atol=1e-5)


def test_sklearn_pipeline_and_cross_val_score(tiny_checkpoint: str) -> None:
    pipe = Pipeline([("clf", _classifier(tiny_checkpoint))])
    scores = cross_val_score(pipe, X, Y, cv=2)
    assert len(scores) == 2
