from __future__ import annotations

import json
from pathlib import Path

import pytest
from sklearn.base import clone

from pntx.pn2t import OverSampler

from .conftest import SAMPLE_NEGATIVE, SAMPLE_POSITIVE, FakeBackend


def _canned(hard_positives: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "positive_features": ["warmth"],
            "negative_features": ["coldness"],
            "boundary_features": [{"feature": "tone", "importance": 0.8}],
            "hard_positives": hard_positives,
        }
    )


def _hp(text: str) -> dict[str, object]:
    return {"text": text, "positive_evidence": ["a"], "confusing_evidence": ["b"]}


def _pools() -> tuple[list[str], list[int]]:
    X = SAMPLE_POSITIVE + SAMPLE_NEGATIVE
    y = [1] * len(SAMPLE_POSITIVE) + [0] * len(SAMPLE_NEGATIVE)
    return X, y


def test_fit_resample_requires_binary_labels() -> None:
    X, _ = _pools()
    sampler = OverSampler(backend=FakeBackend())
    with pytest.raises(ValueError, match="binary labels"):
        sampler.fit_resample(X, ["positive"] * len(X))


def test_fit_resample_requires_at_least_one_of_each_class() -> None:
    sampler = OverSampler(backend=FakeBackend())
    with pytest.raises(ValueError, match="at least one positive"):
        sampler.fit_resample(["a", "b"], [0, 0])
    with pytest.raises(ValueError, match="at least one negative"):
        sampler.fit_resample(["a", "b"], [1, 1])


def test_fit_resample_requires_matching_lengths() -> None:
    sampler = OverSampler(backend=FakeBackend())
    with pytest.raises(ValueError, match="same length"):
        sampler.fit_resample(["a", "b"], [1])


def test_n_synthesized_zero_returns_original_data_unchanged() -> None:
    X, y = _pools()
    sampler = OverSampler(backend=FakeBackend(), n_synthesized=0)
    X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug == X
    assert y_aug == y
    assert sampler.generation_result_.hard_positives == []


def test_fit_resample_happy_path_appends_generated_positives() -> None:
    X, y = _pools()
    backend = FakeBackend(
        complete_responses=[_canned([_hp("new hard positive 1"), _hp("new hard positive 2")])]
    )
    sampler = OverSampler(backend=backend, n_synthesized=2, batch_size=2)
    X_aug, y_aug = sampler.fit_resample(X, y)

    assert X_aug[: len(X)] == X
    assert X_aug[len(X) :] == ["new hard positive 1", "new hard positive 2"]
    assert y_aug == y + [1, 1]
    assert len(sampler.generation_result_.hard_positives) == 2
    assert sampler.generation_result_.positive_features == ["warmth"]


def test_n_synthesized_none_balances_classes() -> None:
    X = SAMPLE_POSITIVE[:1] + SAMPLE_NEGATIVE  # 1 positive, 3 negative -> need 2 to balance
    y = [1] + [0] * len(SAMPLE_NEGATIVE)
    backend = FakeBackend(
        complete_responses=[_canned([_hp("gen 1"), _hp("gen 2")])]
    )
    sampler = OverSampler(backend=backend, batch_size=2)
    X_aug, y_aug = sampler.fit_resample(X, y)
    assert len(X_aug) == len(X) + 2
    assert y_aug.count(1) == 3
    assert y_aug.count(0) == 3


def test_exact_match_dedup_rejects_and_retries() -> None:
    X, y = _pools()
    duplicate_of_existing = SAMPLE_POSITIVE[0]
    backend = FakeBackend(
        complete_responses=[
            _canned([_hp(duplicate_of_existing), _hp("accepted-1")]),
            _canned([_hp("accepted-2")]),
        ]
    )
    sampler = OverSampler(backend=backend, n_synthesized=2, batch_size=2)
    X_aug, y_aug = sampler.fit_resample(X, y)
    generated = X_aug[len(X) :]
    assert duplicate_of_existing not in generated
    assert generated == ["accepted-1", "accepted-2"]


def test_deduplicate_false_accepts_everything() -> None:
    X, y = _pools()
    duplicate_of_existing = SAMPLE_POSITIVE[0]
    backend = FakeBackend(
        complete_responses=[_canned([_hp(duplicate_of_existing), _hp("also new")])]
    )
    sampler = OverSampler(backend=backend, n_synthesized=2, batch_size=2, deduplicate=False)
    X_aug, _ = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == [duplicate_of_existing, "also new"]


def test_shortfall_after_max_batches_warns() -> None:
    X, y = _pools()
    backend = FakeBackend(complete_responses=[])  # complete() returns "" -> always fails to parse
    sampler = OverSampler(backend=backend, n_synthesized=2, batch_size=1)
    with pytest.warns(UserWarning, match="expected"):
        X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug == X
    assert y_aug == y


def test_max_tokens_is_forwarded_to_backend_complete() -> None:
    X, y = _pools()
    backend = FakeBackend(
        complete_responses=[_canned([_hp("new hard positive 1"), _hp("new hard positive 2")])]
    )
    sampler = OverSampler(backend=backend, n_synthesized=2, batch_size=2, max_tokens=777)
    sampler.fit_resample(X, y)
    assert backend.complete_max_tokens == [777]


def test_context_limit_too_small_for_max_tokens_raises() -> None:
    X, y = _pools()
    # Regression test: previously max_tokens for the generation call was
    # hardcoded independently of context_limit, so a small backend n_ctx
    # (e.g. 4096) combined with the old hardcoded max_tokens=4096 default
    # left zero room for the prompt and every batch silently failed. Now
    # this mismatch is caught up front with a clear error instead.
    sampler = OverSampler(
        backend=FakeBackend(), n_synthesized=2, context_limit=4096, max_tokens=4096
    )
    with pytest.raises(ValueError, match="leaves no token budget for exemplars"):
        sampler.fit_resample(X, y)


def test_backend_kwargs_with_instance_backend_raises() -> None:
    sampler = OverSampler(backend=FakeBackend(), backend_kwargs={"foo": "bar"})
    with pytest.raises(TypeError, match="backend_kwargs"):
        sampler.fit_resample(*_pools())


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    X, y = _pools()
    backend = FakeBackend(
        complete_responses=[_canned([_hp("new hard positive 1"), _hp("new hard positive 2")])]
    )
    sampler = OverSampler(backend=backend, n_synthesized=2, batch_size=2)
    sampler.fit_resample(X, y)

    path = tmp_path / "hard_positives.json"
    sampler.save(path)

    loaded = OverSampler.load(path, backend=backend)
    assert loaded.generation_result_ == sampler.generation_result_


def test_save_before_fit_raises_not_fitted() -> None:
    from sklearn.exceptions import NotFittedError

    sampler = OverSampler(backend=FakeBackend())
    with pytest.raises(NotFittedError):
        sampler.save("does-not-matter.json")


def test_clone_reuses_the_same_backend_instance_without_deepcopy() -> None:
    backend = FakeBackend()
    sampler = OverSampler(backend=backend)
    cloned = clone(sampler)
    assert cloned.backend is backend
    assert cloned is not sampler


def test_fit_resample_does_not_require_imbalanced_learn_installed() -> None:
    import sys

    assert "imblearn" not in sys.modules
    X, y = _pools()
    OverSampler(backend=FakeBackend(), n_synthesized=0).fit_resample(X, y)
    assert "imblearn" not in sys.modules
