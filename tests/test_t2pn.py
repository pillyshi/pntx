from __future__ import annotations

import warnings

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from pntx.selection import NearestSelector, RandomSelector
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


def test_fit_caches_exemplar_selection_and_calibration_for_static_selector() -> None:
    X, y = _fit_texts_and_labels()
    backend = FakeBatchBackend()
    clf = Classifier(backend=backend).fit(X, y)
    assert clf.exemplar_positive_ is not None
    assert clf.exemplar_negative_ is not None
    assert clf.exemplar_prefix_ is not None
    assert clf.calibration_weights_ is not None
    assert len(clf.calibration_weights_) == 2
    assert len(backend.score_calls) == 1  # the one content-free calibration call


def test_fit_defers_exemplar_selection_for_dynamic_selector() -> None:
    X, y = _fit_texts_and_labels()
    backend = FakeBatchBackend()
    clf = Classifier(backend=backend, selector=NearestSelector()).fit(X, y)
    assert clf.exemplar_positive_ is None
    assert clf.exemplar_negative_ is None
    assert clf.exemplar_prefix_ is None
    assert clf.calibration_weights_ is None
    assert len(backend.score_calls) == 0  # nothing to calibrate yet -- no query is known


def test_fit_skips_calibration_call_when_calibrate_false() -> None:
    X, y = _fit_texts_and_labels()
    backend = FakeBackend()
    clf = Classifier(backend=backend, calibrate=False).fit(X, y)
    assert clf.calibration_weights_ is None
    assert backend.score_calls == []


def test_predict_proba_content_free_calibration_corrects_label_bias() -> None:
    # A biased few-shot prompt: against a content-free (uninformative) query,
    # the raw model favors "positive" heavily. Calibration should counteract
    # that prefix-level bias, pulling a genuinely uninformative real query's
    # probability away from "positive" instead of letting it inherit the bias.
    X = ["pos text", "neg text"]
    y = ["positive", "negative"]
    prefix = "Text: pos text\nLabel: positive\n\nText: neg text\nLabel: negative\n\n"
    content_free_prompt = prefix + "Text: \nLabel:"
    query_prompt = prefix + "Text: neutral query\nLabel:"
    backend = FakeBackend(
        choice_scores={
            content_free_prompt: [10.0, 0.0],  # heavily biased toward positive
            query_prompt: [0.0, 0.0],  # the query itself carries no signal either way
        }
    )
    clf = Classifier(backend=backend).fit(X, y)

    proba = clf.predict_proba(["neutral query"])

    # classes_ == ["negative", "positive"]; raw (uncalibrated) softmax of
    # [0.0, 0.0] would be exactly 50/50 -- calibration should move it well
    # below that for "positive", which the prefix is biased toward.
    assert proba[0][1] < 0.5


def test_predict_proba_calibrate_false_leaves_raw_softmax_uncorrected() -> None:
    X = ["pos text", "neg text"]
    y = ["positive", "negative"]
    prefix = "Text: pos text\nLabel: positive\n\nText: neg text\nLabel: negative\n\n"
    query_prompt = prefix + "Text: neutral query\nLabel:"
    backend = FakeBackend(choice_scores={query_prompt: [0.0, 0.0]})
    clf = Classifier(backend=backend, calibrate=False).fit(X, y)

    proba = clf.predict_proba(["neutral query"])

    np.testing.assert_allclose(proba[0], [0.5, 0.5])


def test_predict_proba_dynamic_selector_reselects_exemplars_per_item_and_warns() -> None:
    positive_texts = ["APPLE_POS", "HIKE_POS"]
    negative_texts = ["APPLE_NEG", "HIKE_NEG"]
    X = positive_texts + negative_texts
    y = ["positive"] * 2 + ["negative"] * 2

    def topic_similarity(query: str, candidate: str) -> float:
        for topic in ("APPLE", "HIKE"):
            if topic in query and topic in candidate:
                return 1.0
        return 0.0

    backend = FakeBatchBackend()
    selector = NearestSelector(similarity_fn=topic_similarity)
    clf = Classifier(backend=backend, selector=selector, max_exemplars=1).fit(X, y)

    with pytest.warns(UserWarning, match="query-aware"):
        clf.predict_proba(["an APPLE query", "a HIKE query"])

    # One score_choices_batch call per item (content-free + real query
    # batched under that item's own prefix), not one shared-prefix call.
    assert len(backend.batch_calls) == 2
    first_prefix, second_prefix = backend.batch_calls[0][0], backend.batch_calls[1][0]
    assert "Text: APPLE_POS\nLabel: positive" in first_prefix
    assert "Text: APPLE_NEG\nLabel: negative" in first_prefix
    assert "Text: HIKE_POS\nLabel: positive" in second_prefix
    assert "Text: HIKE_NEG\nLabel: negative" in second_prefix
    assert first_prefix != second_prefix


def test_predict_proba_dynamic_selector_no_warning_without_batch_backend() -> None:
    positive_texts = ["apple pie recipe", "great customer service"]
    negative_texts = ["mountain hiking trip", "boring quarterly meeting"]
    X = positive_texts + negative_texts
    y = ["positive"] * 2 + ["negative"] * 2
    backend = FakeBackend()  # ScoringBackend but not BatchScoringBackend
    clf = Classifier(backend=backend, selector=NearestSelector(), max_exemplars=1).fit(X, y)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        clf.predict_proba(["apple pie recipe was amazing"])


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


def test_fit_raises_when_context_limit_too_small_for_reserve_static_selector() -> None:
    # Static (default) selector: the budget check now runs in fit() itself,
    # since exemplar selection is resolved there rather than in predict_proba.
    X, y = _fit_texts_and_labels()
    backend = FakeBatchBackend()
    with pytest.raises(ValueError, match="context_limit"):
        Classifier(backend=backend, context_limit=1).fit(X, y)


def test_predict_proba_raises_when_context_limit_too_small_for_reserve_dynamic_selector() -> None:
    # Dynamic (query-aware) selector: fit() doesn't select exemplars, so the
    # budget check stays deferred to predict_proba, sized from that call's X.
    X, y = _fit_texts_and_labels()
    backend = FakeBatchBackend()
    clf = Classifier(backend=backend, selector=NearestSelector(), context_limit=1).fit(X, y)
    with pytest.raises(ValueError, match="context_limit"):
        clf.predict_proba(["some reasonably long query text"])


def test_predict_proba_falls_back_to_per_item_scoring_without_batch_backend() -> None:
    X, y = _fit_texts_and_labels()
    backend = FakeBackend()  # ScoringBackend but not BatchScoringBackend
    clf = Classifier(backend=backend).fit(X, y)
    proba = clf.predict_proba(["q1", "q2"])
    assert proba.shape == (2, 2)
    # 1 content-free calibration call in fit() + 1 score_choices call per item (2 items).
    assert len(backend.score_calls) == 3


def test_predict_proba_falls_back_to_per_item_scoring_without_batch_backend_no_calibration() -> (
    None
):
    X, y = _fit_texts_and_labels()
    backend = FakeBackend()  # ScoringBackend but not BatchScoringBackend
    clf = Classifier(backend=backend, calibrate=False).fit(X, y)
    proba = clf.predict_proba(["q1", "q2"])
    assert proba.shape == (2, 2)
    assert len(backend.score_calls) == 2  # one score_choices call per item, no calibration call


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
