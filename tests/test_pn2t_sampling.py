from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import pytest
from sklearn.base import clone

import pntx.pn2t
from pntx.pn2t import HardPositiveOverSampler, TypicalPositiveOverSampler
from pntx.pn2t._base import make_rng, resolve_n_to_generate

from .conftest import SAMPLE_NEGATIVE, SAMPLE_POSITIVE, FakeBackend


def _n(strategy: Any, n_pos: int, n_neg: int) -> int:
    y = [1] * n_pos + [0] * n_neg
    return resolve_n_to_generate(strategy, y, positive_label=1, negative_label=0)


# ---- sampling_strategy resolution (mirrors imblearn's over-sampling rules) --


@pytest.mark.parametrize(
    ("strategy", "n_pos", "n_neg", "expected"),
    [
        # Strings: targeted classes go up to the majority count; only the
        # positive entry is used, so a strategy targeting the negative class
        # generates nothing.
        ("auto", 4, 10, 6),
        ("auto", 10, 4, 0),
        ("auto", 5, 5, 0),
        ("not majority", 4, 10, 6),
        ("minority", 4, 10, 6),
        ("minority", 10, 4, 0),
        ("not minority", 4, 10, 0),
        ("all", 4, 10, 6),
        # Float: desired n_pos / n_neg after resampling, int-truncated.
        (0.5, 4, 10, 1),
        (1.0, 4, 10, 6),
        (1.0, 5, 5, 0),
        # Dict: desired total positives after resampling.
        ({1: 8}, 4, 10, 4),
        ({1: 4}, 4, 10, 0),
        ({0: 10, 1: 7}, 4, 10, 3),
        ({0: 10}, 4, 10, 0),
    ],
)
def test_resolve_n_to_generate_matches_imblearn_semantics(
    strategy: Any, n_pos: int, n_neg: int, expected: int
) -> None:
    assert _n(strategy, n_pos, n_neg) == expected


def test_resolve_n_to_generate_accepts_callable() -> None:
    seen: list[list[Any]] = []

    def strategy(y: list[Any]) -> dict[Any, int]:
        seen.append(y)
        return {1: 6}

    assert _n(strategy, 4, 10) == 2
    assert seen == [[1] * 4 + [0] * 10]


@pytest.mark.parametrize(
    ("strategy", "match"),
    [
        ("bogus", "must be one of"),
        (1.5, r"range \(0, 1\]"),
        (0, r"range \(0, 1\]"),
        (-0.1, r"range \(0, 1\]"),
        (0.2, "cannot remove samples"),
        ({1: 3}, "greater or equal to the original"),
        ({0: 12}, "only generates the positive class"),
        ({2: 5}, "not present in the data"),
        ({1: 6.5}, "must be integers"),
        ({1: True}, "must be integers"),
        (True, "must be a str"),
        ([1, 2], "must be a str"),
    ],
)
def test_resolve_n_to_generate_rejects_invalid_strategies(strategy: Any, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        _n(strategy, 4, 10)


def test_float_strategy_requires_positive_minority() -> None:
    with pytest.raises(ValueError, match="positive class is not the minority"):
        _n(1.0, 10, 4)


def test_resolve_n_to_generate_uses_resolved_labels_not_0_1() -> None:
    y = ["spam"] * 2 + ["ham"] * 5
    assert resolve_n_to_generate({"spam": 4}, y, positive_label="spam", negative_label="ham") == 2
    assert resolve_n_to_generate("auto", y, positive_label="spam", negative_label="ham") == 3


# ---- sampling_strategy through fit_resample --------------------------------


def _canned_hp(texts: list[str]) -> str:
    return json.dumps(
        {
            "positive_features": [],
            "negative_features": [],
            "boundary_features": [],
            "hard_positives": [
                {"text": t, "positive_evidence": [], "confusing_evidence": []} for t in texts
            ],
        }
    )


def test_hard_positive_float_sampling_strategy_end_to_end() -> None:
    X = SAMPLE_POSITIVE[:1] + SAMPLE_NEGATIVE + ["追加の負例です"]  # 1 pos, 4 neg
    y = [1] + [0] * 4
    backend = FakeBackend(complete_responses=[_canned_hp(["gen 1"])])
    sampler = HardPositiveOverSampler(backend=backend, sampling_strategy=0.5, batch_size=1)
    X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == ["gen 1"]  # int(0.5 * 4) = 2 positives -> 1 new
    assert y_aug.count(1) == 2


def test_invalid_sampling_strategy_raises_before_backend_is_resolved() -> None:
    # A bad strategy must fail fast, before an expensive backend (model) load.
    sampler = HardPositiveOverSampler(backend="no-such-backend", sampling_strategy=2.0)
    with pytest.raises(ValueError, match=r"range \(0, 1\]"):
        sampler.fit_resample(SAMPLE_POSITIVE + SAMPLE_NEGATIVE, [1, 1, 1, 0, 0, 0])


def test_get_params_exposes_imblearn_style_names() -> None:
    params = HardPositiveOverSampler(backend=FakeBackend()).get_params()
    assert params["sampling_strategy"] == "auto"
    assert params["random_state"] is None
    params = TypicalPositiveOverSampler(backend=FakeBackend()).get_params()
    assert params["sampling_strategy"] is None


# ---- random_state ----------------------------------------------------------


def test_make_rng_int_matches_old_seed_behavior() -> None:
    # random_state=<int> must reproduce what seed=<int> did (random.Random(int)).
    assert make_rng(42).random() == random.Random(42).random()


def test_make_rng_accepts_numpy_random_state_deterministically() -> None:
    a = make_rng(np.random.RandomState(0)).random()
    b = make_rng(np.random.RandomState(0)).random()
    assert a == b


@pytest.mark.parametrize("bad", ["42", 1.5, True])
def test_make_rng_rejects_invalid_random_state(bad: Any) -> None:
    with pytest.raises(ValueError, match="random_state"):
        make_rng(bad)


# ---- deprecations ----------------------------------------------------------


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("OverSampler", HardPositiveOverSampler),
        ("SyntheticSampler", TypicalPositiveOverSampler),
    ],
)
def test_old_class_names_are_deprecated_aliases(old: str, new: type) -> None:
    with pytest.warns(DeprecationWarning, match=f"renamed to {new.__name__}"):
        alias = getattr(pntx.pn2t, old)
    assert alias is new


def test_old_class_name_from_import_warns() -> None:
    with pytest.warns(DeprecationWarning, match="renamed to HardPositiveOverSampler"):
        from pntx.pn2t import OverSampler  # noqa: F401


def test_unknown_attribute_still_raises_attribute_error() -> None:
    with pytest.raises(AttributeError):
        pntx.pn2t.NoSuchSampler  # noqa: B018


def _pools() -> tuple[list[str], list[int]]:
    return SAMPLE_POSITIVE + SAMPLE_NEGATIVE, [1] * 3 + [0] * 3


def test_n_synthesized_is_deprecated_but_still_generates_that_many() -> None:
    X, y = _pools()
    backend = FakeBackend(complete_responses=[_canned_hp(["gen 1", "gen 2"])])
    sampler = HardPositiveOverSampler(backend=backend, n_synthesized=2, batch_size=2)
    with pytest.warns(DeprecationWarning, match="'n_synthesized' was deprecated"):
        X_aug, _ = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == ["gen 1", "gen 2"]


def test_n_synthesized_none_means_auto() -> None:
    X = SAMPLE_POSITIVE[:1] + SAMPLE_NEGATIVE
    y = [1] + [0] * 3
    backend = FakeBackend(complete_responses=[_canned_hp(["gen 1", "gen 2"])])
    sampler = HardPositiveOverSampler(backend=backend, n_synthesized=None, batch_size=2)
    with pytest.warns(DeprecationWarning):
        X_aug, _ = sampler.fit_resample(X, y)
    assert len(X_aug) == len(X) + 2


def test_typical_positive_n_synthesized_still_works() -> None:
    X, y = _pools()
    backend = FakeBackend(
        complete_responses=[
            json.dumps(
                {
                    "style_features": [],
                    "content_features": [],
                    "synthetic_texts": [{"text": "gen 1", "generalized_from": []}],
                }
            )
        ]
    )
    sampler = TypicalPositiveOverSampler(backend=backend, n_synthesized=1, batch_size=1)
    with pytest.warns(DeprecationWarning, match="'n_synthesized' was deprecated"):
        X_aug, _ = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == ["gen 1"]


@pytest.mark.parametrize("bad", [-1, 1.5, True])
def test_invalid_n_synthesized_raises(bad: Any) -> None:
    sampler = HardPositiveOverSampler(backend=FakeBackend(), n_synthesized=bad)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="n_synthesized"):
        sampler.fit_resample(*_pools())


@pytest.mark.parametrize(
    "sampler",
    [
        HardPositiveOverSampler(backend=FakeBackend(), n_synthesized=1, sampling_strategy={1: 4}),
        TypicalPositiveOverSampler(
            backend=FakeBackend(), n_synthesized=1, sampling_strategy="auto"
        ),
    ],
)
def test_n_synthesized_and_sampling_strategy_together_raise(sampler: Any) -> None:
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="not both"):
        sampler.fit_resample(*_pools())


def test_seed_is_deprecated_and_maps_to_random_state(monkeypatch: pytest.MonkeyPatch) -> None:
    import pntx.pn2t._base as base_mod

    seen: list[Any] = []
    real_make_rng = base_mod.make_rng

    def recording_make_rng(random_state: Any) -> random.Random:
        seen.append(random_state)
        return real_make_rng(random_state)

    monkeypatch.setattr(base_mod, "make_rng", recording_make_rng)
    X, y = _pools()
    backend = FakeBackend(complete_responses=[_canned_hp(["gen 1"])])
    sampler = HardPositiveOverSampler(
        backend=backend, sampling_strategy={1: 4}, batch_size=1, seed=7
    )
    with pytest.warns(DeprecationWarning, match="'seed' was deprecated"):
        sampler.fit_resample(X, y)
    assert seen == [7]


def test_seed_and_random_state_together_raise() -> None:
    sampler = HardPositiveOverSampler(backend=FakeBackend(), seed=1, random_state=2)
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="not both"):
        sampler.fit_resample(*_pools())


def test_new_params_emit_no_deprecation_warning(recwarn: pytest.WarningsRecorder) -> None:
    X, y = _pools()
    HardPositiveOverSampler(backend=FakeBackend(), random_state=0).fit_resample(X, y)
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]


def test_clone_preserves_deprecated_params() -> None:
    sampler = HardPositiveOverSampler(backend=FakeBackend(), n_synthesized=3, seed=5)
    cloned = clone(sampler)
    assert cloned.n_synthesized == 3
    assert cloned.seed == 5
