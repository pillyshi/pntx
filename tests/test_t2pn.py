from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from pntx.selection import RandomSelector
from pntx.t2pn import Classifier

from .conftest import (
    SAMPLE_NEGATIVE,
    SAMPLE_POSITIVE,
    CompleteOnlyBackend,
    FakeBackend,
    FakeBatchBackend,
)


def _fit_texts_and_labels() -> tuple[list[str], list[str]]:
    X = SAMPLE_POSITIVE + SAMPLE_NEGATIVE
    y = ["positive"] * len(SAMPLE_POSITIVE) + ["negative"] * len(SAMPLE_NEGATIVE)
    return X, y


def test_fit_requires_matching_lengths() -> None:
    clf = Classifier(backend=FakeBackend())
    with pytest.raises(ValueError, match="same length"):
        clf.fit(["a", "b"], ["positive"])


def test_fit_requires_exactly_two_classes() -> None:
    clf = Classifier(backend=FakeBackend())
    with pytest.raises(ValueError, match="exactly 2 classes"):
        clf.fit(["a", "b", "c"], ["positive", "positive", "positive"])
    with pytest.raises(ValueError, match="exactly 2 classes"):
        clf.fit(["a", "b", "c"], ["x", "y", "z"])


def test_predict_before_fit_raises_not_fitted() -> None:
    from sklearn.exceptions import NotFittedError

    clf = Classifier(backend=FakeBackend())
    with pytest.raises(NotFittedError):
        clf.predict(["some text"])


def test_fit_groups_pools_by_label_and_check_is_fitted_passes() -> None:
    X, y = _fit_texts_and_labels()
    clf = Classifier(backend=FakeBackend()).fit(X, y)
    check_is_fitted(clf, "classes_")
    assert sorted(clf.positive_) == sorted(SAMPLE_POSITIVE)
    assert sorted(clf.negative_) == sorted(SAMPLE_NEGATIVE)
    assert list(clf.classes_) == ["negative", "positive"]


@pytest.mark.parametrize(
    "y_positive,y_negative",
    [("positive", "negative"), (1, 0), (1, -1)],
)
def test_predict_proba_scoring_backend_path(y_positive: object, y_negative: object) -> None:
    X = SAMPLE_POSITIVE + SAMPLE_NEGATIVE
    y = [y_positive] * len(SAMPLE_POSITIVE) + [y_negative] * len(SAMPLE_NEGATIVE)
    backend = FakeBatchBackend()
    clf = Classifier(backend=backend, selector=RandomSelector(seed=0)).fit(X, y)

    proba = clf.predict_proba(["a new query"])
    assert proba.shape == (1, 2)
    np.testing.assert_allclose(proba.sum(axis=1), [1.0])
    assert len(backend.batch_calls) == 1  # batched, not a per-item loop

    pred = clf.predict(["a new query"])
    assert pred[0] in (y_positive, y_negative)


def test_predict_proba_scoring_backend_uses_batch_scoring_when_available() -> None:
    X, y = _fit_texts_and_labels()
    backend = FakeBatchBackend()
    clf = Classifier(backend=backend, selector=RandomSelector(seed=0)).fit(X, y)
    proba = clf.predict_proba(["q1", "q2", "q3"])
    assert proba.shape == (3, 2)
    assert len(backend.batch_calls) == 1
    assert len(backend.batch_calls[0][1]) == 3  # 3 queries scored in one batched call


def test_predict_proba_balances_unequal_pool_sizes() -> None:
    positive_texts = ["pos one", "pos two", "pos three", "pos four"]
    negative_texts = ["neg one"]
    X = positive_texts + negative_texts
    y = ["positive"] * len(positive_texts) + ["negative"] * len(negative_texts)
    backend = FakeBatchBackend()
    clf = Classifier(backend=backend).fit(X, y)

    clf.predict_proba(["query"])

    prefix = backend.batch_calls[0][0]
    # The negative pool only has 1 text, so positive must be trimmed down to
    # match rather than including all 4 of its texts.
    assert prefix.count("Label: positive") == prefix.count("Label: negative") == 1


def test_predict_proba_trims_exemplars_to_fit_context_limit() -> None:
    positive_texts = ["p" * 400 for _ in range(10)]
    negative_texts = ["n" * 400 for _ in range(10)]
    X = positive_texts + negative_texts
    y = ["positive"] * len(positive_texts) + ["negative"] * len(negative_texts)
    backend = FakeBatchBackend()
    clf = Classifier(backend=backend, context_limit=600).fit(X, y)

    clf.predict_proba(["query"])

    prefix = backend.batch_calls[0][0]
    positive_count = prefix.count("Label: positive")
    assert 0 < positive_count < len(positive_texts)
    assert positive_count == prefix.count("Label: negative")


def test_predict_proba_raises_when_context_limit_too_small_for_reserve() -> None:
    X, y = _fit_texts_and_labels()
    backend = FakeBatchBackend()
    clf = Classifier(backend=backend, context_limit=1).fit(X, y)
    with pytest.raises(ValueError, match="context_limit"):
        clf.predict_proba(["some reasonably long query text"])


def test_predict_proba_falls_back_to_per_item_scoring_without_batch_backend() -> None:
    X, y = _fit_texts_and_labels()
    backend = FakeBackend()  # ScoringBackend but not BatchScoringBackend
    clf = Classifier(backend=backend).fit(X, y)
    proba = clf.predict_proba(["q1", "q2"])
    assert proba.shape == (2, 2)
    assert len(backend.score_calls) == 2  # one score_choices call per item


def test_predict_proba_parse_fallback_path_for_non_scoring_backend() -> None:
    X, y = _fit_texts_and_labels()
    backend = CompleteOnlyBackend(complete_responses=["The label is positive.", "negative"])
    clf = Classifier(backend=backend).fit(X, y)
    pred = clf.predict(["q1", "q2"])
    assert list(pred) == ["positive", "negative"]
    assert len(backend.complete_calls) == 2  # no BatchBackend => one complete() call per item


def test_predict_proba_empty_input_returns_empty_array() -> None:
    X, y = _fit_texts_and_labels()
    clf = Classifier(backend=FakeBackend()).fit(X, y)
    proba = clf.predict_proba([])
    assert proba.shape == (0, 2)


def test_get_params_excludes_nothing_and_backend_kwargs_is_a_single_param() -> None:
    backend = FakeBackend()
    clf = Classifier(backend=backend, backend_kwargs=None, max_exemplars=5)
    params = clf.get_params()
    assert params["backend"] is backend
    assert params["backend_kwargs"] is None
    assert params["max_exemplars"] == 5


def test_clone_reuses_the_same_backend_instance_without_deepcopy() -> None:
    backend = FakeBackend()
    clf = Classifier(backend=backend)
    cloned = clone(clf)
    assert cloned.backend is backend
    assert cloned is not clf


def test_sklearn_pipeline_and_cross_val_score() -> None:
    X, y = _fit_texts_and_labels()
    backend = FakeBatchBackend()
    pipe = Pipeline([("clf", Classifier(backend=backend, selector=RandomSelector(seed=0)))])
    scores = cross_val_score(pipe, X, y, cv=2)
    assert len(scores) == 2
