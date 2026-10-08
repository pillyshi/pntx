from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import pytest
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from pntx.pn2t import CounterfactualOverSampler

from .conftest import FakeBackend

# Small pools where every pivot has an obvious one-word positive flip.
NEG_JA = ["この映画は退屈だった", "サポートの対応が雑だった", "店員さんの態度が冷たかった"]
POS_JA = ["この映画は最高だった", "サポートが丁寧で助かった", "店員さんの笑顔が素敵だった"]
FLIP_JA = {"退屈": "痛快", "雑": "丁寧", "冷たかった": "温かかった"}

NEG_EN = [
    "The movie was boring and the ending made no sense.",
    "Support was slow and unhelpful.",
    "The staff were rude to us.",
]
POS_EN = ["The movie was fantastic.", "Support was quick and helpful.", "The staff were lovely."]
FLIP_EN = {"boring": "gripping", "slow and unhelpful": "fast and helpful", "rude": "kind"}

EditFn = Callable[[str], "tuple[str, bool] | None"]


def flip(mapping: dict[str, str]) -> EditFn:
    """Minimal edit: replace the first matching negative word."""

    def edit(pivot: str) -> tuple[str, bool]:
        for old, new in mapping.items():
            if old in pivot:
                return pivot.replace(old, new), True
        return pivot + "!", True

    return edit


class EditingBackend(FakeBackend):
    """Reads the numbered pivots out of the prompt and answers with one edit
    per pivot, so tests don't depend on which negatives were sampled.

    ``edit_fns`` are used one per backend call (the last one repeats); an
    edit function returning ``None`` skips that pivot.
    """

    def __init__(self, *edit_fns: EditFn, extra_edits: list[dict[str, Any]] | None = None):
        super().__init__()
        self.edit_fns = list(edit_fns)
        self.extra_edits = extra_edits or []
        self.pivots_seen: list[list[str]] = []

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 1.0,
        max_tokens: int = 512,
        stop: list[str] | None = None,
    ) -> str:
        super().complete(prompt, temperature=temperature, max_tokens=max_tokens, stop=stop)
        section = prompt.split("Negative texts to edit:\n", 1)[1].split("\n\nCount:", 1)[0]
        pivots = [m.group(2) for m in re.finditer(r"^\[(\d+)\] (.*)$", section, re.M)]
        self.pivots_seen.append(pivots)
        fn = self.edit_fns[min(len(self.complete_calls) - 1, len(self.edit_fns) - 1)]
        edits: list[dict[str, Any]] = []
        for i, pivot in enumerate(pivots):
            out = fn(pivot)
            if out is None:
                continue
            text, is_positive = out
            edits.append(
                {
                    "pivot_id": i,
                    "edited_text": text,
                    "changed_spans": ["x -> y"],
                    "is_positive": is_positive,
                }
            )
        edits.extend(self.extra_edits)
        self.extra_edits = []
        return json.dumps({"edits": edits})


def _data(pos: list[str], neg: list[str]) -> tuple[list[str], list[int]]:
    return pos + neg, [1] * len(pos) + [0] * len(neg)


def _sampler(backend: Any, **kwargs: Any) -> CounterfactualOverSampler:
    kwargs.setdefault("verify", "none")
    kwargs.setdefault("random_state", 0)
    return CounterfactualOverSampler(backend=backend, **kwargs)


# ---- basic generation -------------------------------------------------------


def test_minimal_edits_are_accepted_and_recorded_as_pairs() -> None:
    X, y = _data(POS_JA[:1], NEG_JA)  # 1 pos, 3 neg -> "auto" needs 2 edits
    backend = EditingBackend(flip(FLIP_JA))
    sampler = _sampler(backend, batch_size=2)
    X_aug, y_aug = sampler.fit_resample(X, y)

    assert len(X_aug) == len(X) + 2
    assert y_aug == y + [1, 1]
    for edit in sampler.generation_result_.edits:
        assert edit.source_text == X[edit.source_index]
        assert y[edit.source_index] == 0
        assert edit.text in X_aug[len(X) :]
        assert 0 < edit.edit_ratio <= 0.5
    assert sampler.generation_result_.rejected == []


def test_english_minimal_edits_are_accepted() -> None:
    X, y = _data(POS_EN[:1], NEG_EN)
    sampler = _sampler(EditingBackend(flip(FLIP_EN)), batch_size=2)
    X_aug, _ = sampler.fit_resample(X, y)
    assert len(X_aug) == len(X) + 2


def test_positives_are_never_pivots_only_references() -> None:
    X, y = _data(POS_JA, NEG_JA)
    backend = EditingBackend(flip(FLIP_JA))
    sampler = _sampler(backend, sampling_strategy={1: 6}, batch_size=3)
    sampler.fit_resample(X, y)

    for pivots in backend.pivots_seen:
        assert pivots
        assert set(pivots) <= set(NEG_JA)
    for prompt in backend.complete_calls:
        references = prompt.split("Positive reference examples:\n", 1)[1].split("\n\n", 1)[0]
        assert any(p in references for p in POS_JA)
        assert not any(n in references for n in NEG_JA)


def test_pivots_are_used_without_replacement_before_repeating() -> None:
    X, y = _data(POS_JA, NEG_JA)
    backend = EditingBackend(flip(FLIP_JA))
    sampler = _sampler(backend, sampling_strategy={1: 6}, batch_size=1)
    sampler.fit_resample(X, y)
    first_three = [p for pivots in backend.pivots_seen[:3] for p in pivots]
    assert sorted(first_three) == sorted(NEG_JA)


def test_generated_texts_use_the_resolved_positive_label() -> None:
    X = POS_JA[:1] + NEG_JA
    y = ["spam"] + ["ham"] * 3
    sampler = _sampler(EditingBackend(flip(FLIP_JA)), batch_size=2, pos_label="spam")
    _, y_aug = sampler.fit_resample(X, y)
    assert y_aug[len(X) :] == ["spam", "spam"]


# ---- filters ----------------------------------------------------------------


def test_edit_too_large_is_rejected_and_retried() -> None:
    X, y = _data(POS_EN[:2], NEG_EN)  # 2 pos, 3 neg -> 1 edit
    rewrite: EditFn = lambda pivot: ("Absolutely loved it, best day ever!", True)  # noqa: E731
    backend = EditingBackend(rewrite, flip(FLIP_EN))
    sampler = _sampler(backend, batch_size=1)
    X_aug, _ = sampler.fit_resample(X, y)

    assert "Absolutely loved it, best day ever!" not in X_aug
    assert len(X_aug) == len(X) + 1
    [rejected] = sampler.generation_result_.rejected
    assert rejected.reason == "edit_too_large"
    assert rejected.source_text in NEG_EN


def test_max_edit_ratio_is_configurable() -> None:
    X, y = _data(POS_EN[:2], NEG_EN)
    sampler = _sampler(EditingBackend(flip(FLIP_EN)), batch_size=1, max_edit_ratio=0.01)
    with pytest.warns(UserWarning, match="expected"):
        X_aug, _ = sampler.fit_resample(X, y)
    assert X_aug == X
    assert {r.reason for r in sampler.generation_result_.rejected} == {"edit_too_large"}


def test_no_op_edit_is_rejected_even_without_dedup() -> None:
    X, y = _data(POS_JA[:2], NEG_JA)
    unchanged: EditFn = lambda pivot: (pivot, True)  # noqa: E731
    backend = EditingBackend(unchanged, flip(FLIP_JA))
    sampler = _sampler(backend, batch_size=1, deduplicate=False)
    X_aug, _ = sampler.fit_resample(X, y)
    assert len(X_aug) == len(X) + 1
    assert sampler.generation_result_.rejected[0].reason == "no_op"


@pytest.mark.parametrize("arrow", ["->", "→"])
def test_echoed_original_arrow_edit_text_is_rejected(arrow: str) -> None:
    # Regression: qwen2.5-7B put "original -> edited" (the changed_spans
    # format) into the text field; with a short pivot that still passed the
    # edit-ratio check and would have been appended to X.
    X, y = _data(POS_JA[:2], NEG_JA)
    echoed: EditFn = lambda pivot: (f"{pivot} {arrow} {flip(FLIP_JA)(pivot)[0]}", True)  # noqa: E731
    backend = EditingBackend(echoed, flip(FLIP_JA))
    sampler = _sampler(backend, batch_size=1, max_edit_ratio=1.0)
    X_aug, _ = sampler.fit_resample(X, y)
    assert not any(arrow in t for t in X_aug)
    assert sampler.generation_result_.rejected[0].reason == "malformed_text"


def test_arrow_already_in_the_pivot_is_not_malformed() -> None:
    X, y = ["良い -> 最高", "悪い -> 最悪"], [1, 0]
    sampler = _sampler(
        EditingBackend(lambda p: ("良い -> 最悪", True)), sampling_strategy={1: 2}, batch_size=1
    )
    X_aug, _ = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == ["良い -> 最悪"]


def test_exact_duplicate_of_existing_text_is_rejected() -> None:
    # Editing "この映画は退屈だった" into an existing positive is a duplicate.
    X, y = _data(POS_JA[:2], NEG_JA)
    to_existing: EditFn = lambda pivot: (  # noqa: E731
        ("この映画は最高だった", True) if "退屈" in pivot else flip(FLIP_JA)(pivot)
    )
    sampler = _sampler(EditingBackend(to_existing), sampling_strategy={1: 5}, batch_size=3)
    # 3 edits requested -> all 3 negatives are pivots in the first batch.
    with pytest.warns(UserWarning, match="expected"):
        X_aug, _ = sampler.fit_resample(X, y)
    assert X_aug[len(X) :].count("この映画は最高だった") == 0
    assert "duplicate" in {r.reason for r in sampler.generation_result_.rejected}


def test_deduplicate_false_accepts_duplicates_but_keeps_minimality() -> None:
    X, y = _data(POS_JA[:2], NEG_JA)
    to_existing: EditFn = lambda pivot: (  # noqa: E731
        ("この映画は最高だった", True) if "退屈" in pivot else None
    )
    sampler = _sampler(
        EditingBackend(to_existing), sampling_strategy={1: 3}, batch_size=3, deduplicate=False
    )
    X_aug, _ = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == ["この映画は最高だった"]


def test_invalid_pivot_id_is_recorded_and_skipped() -> None:
    X, y = _data(POS_JA[:2], NEG_JA)
    bogus = {"pivot_id": 99, "edited_text": "幽霊", "changed_spans": [], "is_positive": True}
    backend = EditingBackend(flip(FLIP_JA), extra_edits=[bogus])
    sampler = _sampler(backend, batch_size=1)
    X_aug, _ = sampler.fit_resample(X, y)
    assert "幽霊" not in X_aug
    [rejected] = sampler.generation_result_.rejected
    assert rejected.reason == "invalid_pivot_id"
    assert rejected.source_index is None


def test_shortfall_after_max_batches_warns() -> None:
    X, y = _data(POS_JA[:1], NEG_JA)
    sampler = _sampler(FakeBackend(complete_responses=[]), batch_size=1)
    with pytest.warns(UserWarning, match="expected"):
        X_aug, y_aug = sampler.fit_resample(X, y)
    assert (X_aug, y_aug) == (X, y)


def test_auto_strategy_balances_classes() -> None:
    X, y = _data(POS_EN[:1], NEG_EN)
    sampler = _sampler(EditingBackend(flip(FLIP_EN)), batch_size=3)
    _, y_aug = sampler.fit_resample(X, y)
    assert y_aug.count(1) == y_aug.count(0) == 3


# ---- verify="none"/"self" -----------------------------------------------------


def test_verify_none_accepts_edits_the_llm_doubts() -> None:
    X, y = _data(POS_JA[:2], NEG_JA)
    doubtful: EditFn = lambda pivot: (flip(FLIP_JA)(pivot)[0], False)  # noqa: E731
    sampler = _sampler(EditingBackend(doubtful), verify="none", batch_size=1)
    X_aug, _ = sampler.fit_resample(X, y)
    assert len(X_aug) == len(X) + 1
    assert sampler.generation_result_.edits[0].self_assessed_positive is False


def test_verify_self_rejects_edits_the_llm_doubts() -> None:
    X, y = _data(POS_JA[:2], NEG_JA)
    doubtful: EditFn = lambda pivot: (flip(FLIP_JA)(pivot)[0], False)  # noqa: E731
    backend = EditingBackend(doubtful, flip(FLIP_JA))
    sampler = _sampler(backend, verify="self", batch_size=1)
    X_aug, _ = sampler.fit_resample(X, y)
    assert len(X_aug) == len(X) + 1
    assert sampler.generation_result_.rejected[0].reason == "self_check_failed"
    assert sampler.generation_result_.edits[0].self_assessed_positive is True


# ---- verify=<classifier> ----------------------------------------------------


class RecordingClassifier(ClassifierMixin, BaseEstimator):
    """Predicts positive iff the text contains ``marker``; records every
    fit/predict call (class-level, since clone() builds new instances)."""

    fits: ClassVar[list[tuple[int, list[str]]]] = []
    predicts: ClassVar[list[tuple[int, list[str]]]] = []

    def __init__(self, marker: str = "丁寧") -> None:
        self.marker = marker

    def fit(self, X: list[str], y: list[Any]) -> RecordingClassifier:
        RecordingClassifier.fits.append((id(self), list(X)))
        self.classes_ = sorted(set(y))
        return self

    def predict(self, X: list[str]) -> list[int]:
        RecordingClassifier.predicts.append((id(self), list(X)))
        return [1 if self.marker in x else 0 for x in X]


@pytest.fixture(autouse=True)
def _reset_recorder() -> None:
    RecordingClassifier.fits.clear()
    RecordingClassifier.predicts.clear()


def test_classifier_verifier_rejects_edits_it_predicts_negative() -> None:
    X, y = _data(POS_JA, NEG_JA)
    sampler = _sampler(
        EditingBackend(flip(FLIP_JA)),
        verify=RecordingClassifier(marker="丁寧"),
        verify_cv=2,
        sampling_strategy={1: 6},
        batch_size=3,
    )
    with pytest.warns(UserWarning, match="expected"):
        X_aug, _ = sampler.fit_resample(X, y)
    # Only the "雑" -> "丁寧" edit contains the marker.
    assert X_aug[len(X) :] == ["サポートの対応が丁寧だった"]
    reasons = {r.reason for r in sampler.generation_result_.rejected}
    assert "verifier_rejected" in reasons


def test_cross_fitting_never_judges_an_edit_with_a_clone_trained_on_its_pivot() -> None:
    X, y = _data(POS_JA, NEG_JA)
    user_verifier = RecordingClassifier(marker="")  # accepts everything
    sampler = _sampler(
        EditingBackend(flip(FLIP_JA)),
        verify=user_verifier,
        verify_cv=3,
        sampling_strategy={1: 6},
        batch_size=3,
    )
    sampler.fit_resample(X, y)

    trained_on = dict(RecordingClassifier.fits)
    # The passed instance is never fitted -- only clones are.
    assert id(user_verifier) not in trained_on
    assert RecordingClassifier.fits
    pivot_of = {e.text: e.source_text for e in sampler.generation_result_.edits}
    generated = set(pivot_of)
    for clone_id, judged in RecordingClassifier.predicts:
        train = trained_on[clone_id]
        # No clone ever saw generated edits...
        assert not generated & set(train)
        # ...or the pivot of an edit it judged.
        for text in judged:
            assert pivot_of[text] not in train


def test_verify_prefit_uses_the_instance_without_clone_or_fit() -> None:
    X, y = _data(POS_JA, NEG_JA)
    verifier = RecordingClassifier(marker="")
    sampler = _sampler(
        EditingBackend(flip(FLIP_JA)),
        verify=verifier,
        verify_cv="prefit",
        sampling_strategy={1: 4},
        batch_size=1,
    )
    sampler.fit_resample(X, y)
    assert RecordingClassifier.fits == []
    assert {clone_id for clone_id, _ in RecordingClassifier.predicts} == {id(verifier)}


def test_verify_cv_larger_than_smaller_class_raises() -> None:
    X, y = _data(POS_JA[:2], NEG_JA)
    sampler = _sampler(FakeBackend(), verify=RecordingClassifier(), verify_cv=3)
    with pytest.raises(ValueError, match="verify_cv=3 needs at least 3 samples"):
        sampler.fit_resample(X, y)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"verify": "bogus"}, "verify must be one of"),
        ({"verify": object()}, "verify must be one of"),
        ({"verify": RecordingClassifier(), "verify_cv": 1}, "verify_cv must be"),
        ({"verify": RecordingClassifier(), "verify_cv": "loo"}, "verify_cv must be"),
        ({"verify": RecordingClassifier(), "verify_cv": True}, "verify_cv must be"),
        ({"max_edit_ratio": 0}, "max_edit_ratio"),
        ({"max_edit_ratio": 1.5}, "max_edit_ratio"),
        ({"max_examples": 0}, "max_examples"),
    ],
)
def test_invalid_params_raise(kwargs: dict[str, Any], match: str) -> None:
    X, y = _data(POS_JA, NEG_JA)
    with pytest.raises(ValueError, match=match):
        _sampler(FakeBackend(), **kwargs).fit_resample(X, y)


def test_verify_is_required() -> None:
    with pytest.raises(TypeError):
        CounterfactualOverSampler(backend=FakeBackend())  # type: ignore[call-arg]


def test_verify_cv_is_ignored_for_string_strategies() -> None:
    X, y = _data(POS_JA[:2], NEG_JA)
    sampler = _sampler(EditingBackend(flip(FLIP_JA)), verify="self", verify_cv=99)
    X_aug, _ = sampler.fit_resample(X, y)
    assert len(X_aug) == len(X) + 1


# ---- sklearn plumbing -------------------------------------------------------


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    X, y = _data(POS_EN[:2], NEG_EN)
    backend = EditingBackend(
        lambda pivot: ("Completely different text that shares nothing!", True),
        flip(FLIP_EN),
    )
    sampler = _sampler(backend, batch_size=1)
    sampler.fit_resample(X, y)
    assert sampler.generation_result_.rejected  # rejected edits round-trip too

    path = tmp_path / "counterfactual.json"
    sampler.save(path)
    loaded = CounterfactualOverSampler.load(path, backend=backend, verify="none")
    assert loaded.generation_result_ == sampler.generation_result_


def test_clone_keeps_backend_and_verifier_params() -> None:
    backend = FakeBackend()
    verifier = RecordingClassifier()
    sampler = _sampler(backend, verify=verifier, verify_cv=3)
    cloned = clone(sampler)
    assert cloned.backend is backend
    assert cloned.verify_cv == 3
    assert isinstance(cloned.verify, RecordingClassifier)


def test_new_class_has_no_deprecated_params() -> None:
    params = _sampler(FakeBackend()).get_params()
    assert "n_synthesized" not in params
    assert "seed" not in params
