from __future__ import annotations

from pntx.backends.anthropic import AnthropicBackend
from pntx.t2pn import Classifier

from ..conftest import SAMPLE_NEGATIVE, SAMPLE_POSITIVE


def _fit_texts_and_labels() -> tuple[list[str], list[str]]:
    X = SAMPLE_POSITIVE + SAMPLE_NEGATIVE
    y = ["positive"] * len(SAMPLE_POSITIVE) + ["negative"] * len(SAMPLE_NEGATIVE)
    return X, y


def test_predict_returns_a_valid_label(anthropic_backend: AnthropicBackend) -> None:
    X, y = _fit_texts_and_labels()
    clf = Classifier(backend=anthropic_backend).fit(X, y)

    pred = clf.predict([SAMPLE_POSITIVE[0]])

    assert pred[0] in ("positive", "negative")


def test_batched_predict_matches_per_item_predict(anthropic_backend: AnthropicBackend) -> None:
    X, y = _fit_texts_and_labels()
    clf = Classifier(backend=anthropic_backend).fit(X, y)
    texts = SAMPLE_POSITIVE

    batched = list(clf.predict(texts))
    individual = [clf.predict([text])[0] for text in texts]

    assert batched == individual
