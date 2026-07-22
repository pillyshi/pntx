from __future__ import annotations

from pntx.backends.llama import LlamaCppBackend
from pntx.t2pn import Classifier

from ..conftest import SAMPLE_NEGATIVE, SAMPLE_POSITIVE


def _fit_texts_and_labels() -> tuple[list[str], list[str]]:
    X = SAMPLE_POSITIVE + SAMPLE_NEGATIVE
    y = ["positive"] * len(SAMPLE_POSITIVE) + ["negative"] * len(SAMPLE_NEGATIVE)
    return X, y


def test_predict_returns_a_valid_label(llama_backend: LlamaCppBackend) -> None:
    X, y = _fit_texts_and_labels()
    clf = Classifier(backend=llama_backend).fit(X, y)

    pred = clf.predict([SAMPLE_POSITIVE[0]])

    assert pred[0] in ("positive", "negative")


def test_predict_proba_rows_sum_to_one(llama_backend: LlamaCppBackend) -> None:
    X, y = _fit_texts_and_labels()
    clf = Classifier(backend=llama_backend).fit(X, y)

    proba = clf.predict_proba(SAMPLE_POSITIVE)

    for row in proba:
        assert 0.0 <= row[0] <= 1.0
        assert 0.0 <= row[1] <= 1.0
        assert abs(row.sum() - 1.0) < 1e-6


def test_batched_predict_matches_per_item_predict(llama_backend: LlamaCppBackend) -> None:
    X, y = _fit_texts_and_labels()
    clf = Classifier(backend=llama_backend).fit(X, y)
    texts = SAMPLE_POSITIVE

    batched = list(clf.predict(texts))
    individual = [clf.predict([text])[0] for text in texts]

    assert batched == individual
