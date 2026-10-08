from __future__ import annotations

import json
import random
from typing import Any

import numpy as np
import pytest

import pntx.pn2t
from pntx.pn2t import (
    CounterfactualOverSampler,
    HardPositiveOverSampler,
    TypicalPositiveOverSampler,
)
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
    params = TypicalPositiveOverSampler(
        backend=FakeBackend(), sampling_strategy={1: 4}
    ).get_params()
    assert params["sampling_strategy"] == {1: 4}


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


# ---- 0.18.0: deprecated names removed --------------------------------------


@pytest.mark.parametrize("old", ["OverSampler", "SyntheticSampler"])
def test_old_class_names_are_gone(old: str) -> None:
    assert not hasattr(pntx.pn2t, old)
    with pytest.raises(ImportError):
        exec(f"from pntx.pn2t import {old}")


@pytest.mark.parametrize(
    "cls", [HardPositiveOverSampler, TypicalPositiveOverSampler, CounterfactualOverSampler]
)
@pytest.mark.parametrize("old_param", ["n_synthesized", "seed"])
def test_old_params_are_rejected(cls: type, old_param: str) -> None:
    kwargs: dict[str, Any] = {"backend": FakeBackend(), old_param: 1}
    if cls is TypicalPositiveOverSampler:
        kwargs["sampling_strategy"] = {1: 4}
    with pytest.raises(TypeError, match=old_param):
        cls(**kwargs)
    assert old_param not in cls(**{k: v for k, v in kwargs.items() if k != old_param}).get_params()


def test_fit_resample_emits_no_deprecation_warning(recwarn: pytest.WarningsRecorder) -> None:
    X, y = SAMPLE_POSITIVE + SAMPLE_NEGATIVE, [1] * 3 + [0] * 3
    HardPositiveOverSampler(backend=FakeBackend(), random_state=0).fit_resample(X, y)
    assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]
