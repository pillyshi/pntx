from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from sklearn.base import clone

from pntx.pn2t import SyntheticSampler

from .conftest import SAMPLE_NEGATIVE, SAMPLE_POSITIVE, FakeBackend


def _canned(synthetic_texts: list[dict[str, object]]) -> str:
    return json.dumps(
        {
            "style_features": ["polite tone"],
            "content_features": ["customer service"],
            "synthetic_texts": synthetic_texts,
        }
    )


def _st(text: str, generalized_from: list[str] | None = None) -> dict[str, object]:
    return {"text": text, "generalized_from": generalized_from or ["removed a name"]}


def _pools() -> tuple[list[str], list[int]]:
    X = SAMPLE_POSITIVE + SAMPLE_NEGATIVE
    y = [1] * len(SAMPLE_POSITIVE) + [0] * len(SAMPLE_NEGATIVE)
    return X, y


def test_fit_resample_requires_exactly_two_classes() -> None:
    X, _ = _pools()
    sampler = SyntheticSampler(backend=FakeBackend(), n_synthesized=1)
    with pytest.raises(ValueError, match="exactly 2 classes"):
        sampler.fit_resample(X, (["a", "b", "c"] * (len(X) // 3 + 1))[: len(X)])
    with pytest.raises(ValueError, match="exactly 2 classes"):
        sampler.fit_resample(["a", "b"], [0, 0])
    with pytest.raises(ValueError, match="exactly 2 classes"):
        sampler.fit_resample(["a", "b"], [1, 1])


def test_fit_resample_accepts_alternate_binary_label_encodings() -> None:
    X = SAMPLE_POSITIVE + SAMPLE_NEGATIVE
    y = [1] * len(SAMPLE_POSITIVE) + [-1] * len(SAMPLE_NEGATIVE)
    backend = FakeBackend(complete_responses=[_canned([_st("new synthetic text")])])
    sampler = SyntheticSampler(backend=backend, n_synthesized=1, batch_size=1)
    X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == ["new synthetic text"]
    assert y_aug[len(X) :] == [1]


def test_fit_resample_requires_pos_label_for_ambiguous_string_labels() -> None:
    X = SAMPLE_POSITIVE + SAMPLE_NEGATIVE
    y = ["spam"] * len(SAMPLE_POSITIVE) + ["ham"] * len(SAMPLE_NEGATIVE)
    sampler = SyntheticSampler(backend=FakeBackend(), n_synthesized=1)
    with pytest.raises(ValueError, match="pass pos_label"):
        sampler.fit_resample(X, y)

    backend = FakeBackend(complete_responses=[_canned([_st("new synthetic text")])])
    sampler = SyntheticSampler(
        backend=backend, n_synthesized=1, batch_size=1, pos_label="spam"
    )
    X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == ["new synthetic text"]
    assert y_aug[len(X) :] == ["spam"]


def test_fit_resample_requires_matching_lengths() -> None:
    sampler = SyntheticSampler(backend=FakeBackend(), n_synthesized=1)
    with pytest.raises(ValueError, match="same length"):
        sampler.fit_resample(["a", "b"], [1])


def test_constructing_without_n_synthesized_raises_type_error() -> None:
    with pytest.raises(TypeError):
        SyntheticSampler(backend=FakeBackend())  # type: ignore[call-arg]


def test_n_synthesized_negative_raises() -> None:
    X, y = _pools()
    sampler = SyntheticSampler(backend=FakeBackend(), n_synthesized=-1)
    with pytest.raises(ValueError, match="n_synthesized must be >= 0"):
        sampler.fit_resample(X, y)


def test_n_synthesized_zero_returns_original_data_unchanged() -> None:
    X, y = _pools()
    sampler = SyntheticSampler(backend=FakeBackend(), n_synthesized=0)
    X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug == X
    assert y_aug == y
    assert sampler.generation_result_.synthetic_texts == []


def test_fit_resample_happy_path_appends_generated_synthetic_texts() -> None:
    X, y = _pools()
    backend = FakeBackend(
        complete_responses=[_canned([_st("new synthetic 1"), _st("new synthetic 2")])]
    )
    sampler = SyntheticSampler(backend=backend, n_synthesized=2, batch_size=2)
    X_aug, y_aug = sampler.fit_resample(X, y)

    assert X_aug[: len(X)] == X
    assert X_aug[len(X) :] == ["new synthetic 1", "new synthetic 2"]
    assert y_aug == y + [1, 1]
    assert len(sampler.generation_result_.synthetic_texts) == 2
    assert sampler.generation_result_.style_features == ["polite tone"]


def test_negative_texts_are_not_included_in_the_prompt() -> None:
    X, y = _pools()
    backend = FakeBackend(complete_responses=[_canned([_st("new synthetic 1")])])
    sampler = SyntheticSampler(backend=backend, n_synthesized=1, batch_size=1)
    sampler.fit_resample(X, y)

    assert backend.complete_calls
    for prompt in backend.complete_calls:
        for neg in SAMPLE_NEGATIVE:
            assert neg not in prompt
        assert "Negative:" not in prompt


def test_exact_match_dedup_rejects_and_retries() -> None:
    X, y = _pools()
    duplicate_of_existing = SAMPLE_POSITIVE[0]
    backend = FakeBackend(
        complete_responses=[
            _canned([_st(duplicate_of_existing), _st("accepted-1")]),
            _canned([_st("accepted-2")]),
        ]
    )
    sampler = SyntheticSampler(backend=backend, n_synthesized=2, batch_size=2)
    X_aug, y_aug = sampler.fit_resample(X, y)
    generated = X_aug[len(X) :]
    assert duplicate_of_existing not in generated
    assert generated == ["accepted-1", "accepted-2"]


def test_deduplicate_false_accepts_everything() -> None:
    X, y = _pools()
    duplicate_of_existing = SAMPLE_POSITIVE[0]
    backend = FakeBackend(
        complete_responses=[_canned([_st(duplicate_of_existing), _st("also new")])]
    )
    sampler = SyntheticSampler(backend=backend, n_synthesized=2, batch_size=2, deduplicate=False)
    X_aug, _ = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == [duplicate_of_existing, "also new"]


def test_verbatim_leak_dedup_rejects_and_retries() -> None:
    pos_long = "田中太郎さんがサポートセンターに感謝のメッセージを送ってくれてとても嬉しかった"
    X = [pos_long, "ok"]
    y = [1, 0]
    leaking_text = pos_long[:25] + "、本当に助かりました"
    backend = FakeBackend(
        complete_responses=[
            _canned([_st(leaking_text)]),
            _canned([_st("全く新しい独立した合成テキストです")]),
        ]
    )
    sampler = SyntheticSampler(backend=backend, n_synthesized=1, batch_size=1)
    X_aug, _ = sampler.fit_resample(X, y)
    generated = X_aug[len(X) :]
    assert leaking_text not in generated
    assert generated == ["全く新しい独立した合成テキストです"]


def test_deduplicate_false_also_disables_verbatim_leak_check() -> None:
    pos_long = "田中太郎さんがサポートセンターに感謝のメッセージを送ってくれてとても嬉しかった"
    X = [pos_long, "ok"]
    y = [1, 0]
    leaking_text = pos_long[:25] + "、本当に助かりました"
    backend = FakeBackend(complete_responses=[_canned([_st(leaking_text)])])
    sampler = SyntheticSampler(backend=backend, n_synthesized=1, batch_size=1, deduplicate=False)
    X_aug, _ = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == [leaking_text]


def test_min_verbatim_span_is_configurable() -> None:
    pos_short = "サポートが丁寧で助かった"
    X = [pos_short, "ok"]
    y = [1, 0]
    # "サポートが丁寧で" is an 8-char verbatim prefix carried into the generated text.
    overlapping_text = "サポートが丁寧で本当に助かった、また利用したい"

    lenient = FakeBackend(
        complete_responses=[
            _canned([_st(overlapping_text)]),
            _canned([_st("別の独立したテキストです")]),
        ]
    )
    sampler_lenient = SyntheticSampler(
        backend=lenient, n_synthesized=1, batch_size=1, min_verbatim_span=20
    )
    X_aug_lenient, _ = sampler_lenient.fit_resample(X, y)
    assert overlapping_text in X_aug_lenient[len(X) :]

    strict = FakeBackend(
        complete_responses=[
            _canned([_st(overlapping_text)]),
            _canned([_st("別の独立したテキストです")]),
        ]
    )
    sampler_strict = SyntheticSampler(
        backend=strict, n_synthesized=1, batch_size=1, min_verbatim_span=8
    )
    X_aug_strict, _ = sampler_strict.fit_resample(X, y)
    assert overlapping_text not in X_aug_strict[len(X) :]


def test_shortfall_after_max_batches_warns() -> None:
    X, y = _pools()
    backend = FakeBackend(complete_responses=[])  # complete() returns "" -> always fails to parse
    sampler = SyntheticSampler(backend=backend, n_synthesized=2, batch_size=1)
    with pytest.warns(UserWarning, match="expected"):
        X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug == X
    assert y_aug == y


def test_max_tokens_is_forwarded_to_backend_complete() -> None:
    X, y = _pools()
    backend = FakeBackend(
        complete_responses=[_canned([_st("new synthetic 1"), _st("new synthetic 2")])]
    )
    sampler = SyntheticSampler(backend=backend, n_synthesized=2, batch_size=2, max_tokens=777)
    sampler.fit_resample(X, y)
    assert backend.complete_max_tokens == [777]


def test_context_limit_too_small_for_max_tokens_raises() -> None:
    X, y = _pools()
    sampler = SyntheticSampler(
        backend=FakeBackend(), n_synthesized=2, context_limit=4096, max_tokens=4096
    )
    with pytest.raises(ValueError, match="leaves no token budget for exemplars"):
        sampler.fit_resample(X, y)


def test_no_positive_example_fits_budget_raises() -> None:
    # context_limit=600, max_tokens=90 -> budget = 600-500-90 = 10 tokens
    # (no halving, unlike OverSampler, since only the positive side is
    # sampled). The default tokenizer is len(text)//4 + 1, so a 50-char
    # positive text costs 13 tokens and can never fit.
    X = ["x" * 50, "ok"]
    y = [1, 0]
    sampler = SyntheticSampler(
        backend=FakeBackend(), n_synthesized=1, context_limit=600, max_tokens=90
    )
    with pytest.raises(ValueError, match="no positive example fits"):
        sampler.fit_resample(X, y)


def test_exemplar_sampling_respects_max_examples(monkeypatch: pytest.MonkeyPatch) -> None:
    from pntx.pn2t import prompts as prompts_mod

    real_build_synthetic_user_message = prompts_mod.build_synthetic_user_message
    captured: dict[str, list[str]] = {}

    def _capturing_build_synthetic_user_message(
        *,
        pos_texts: list[str],
        n_synthesized: int,
        language: str | None = None,
    ) -> str:
        captured["pos"] = list(pos_texts)
        return real_build_synthetic_user_message(
            pos_texts=pos_texts, n_synthesized=n_synthesized, language=language
        )

    monkeypatch.setattr(
        prompts_mod, "build_synthetic_user_message", _capturing_build_synthetic_user_message
    )

    pos_texts = ["a", "b", "c", "d", "e"]
    X = pos_texts + ["neg text"]
    y = [1] * len(pos_texts) + [0]

    backend = FakeBackend(complete_responses=[_canned([_st("gen 1")])])
    sampler = SyntheticSampler(backend=backend, n_synthesized=1, batch_size=1, max_examples=2)
    sampler.fit_resample(X, y)

    assert captured["pos"]
    assert len(captured["pos"]) == 2


def test_invalid_sample_method_raises() -> None:
    X, y = _pools()
    sampler = SyntheticSampler(backend=FakeBackend(), n_synthesized=2, sample_method="bogus")
    with pytest.raises(ValueError, match="sample_method"):
        sampler.fit_resample(X, y)


def _install_fake_embeddings_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stands in for ``pntx.embeddings`` so ``sample_method="kmeans"/"votek"``
    tests don't require sentence-transformers to be installed."""

    def fake_embed(texts: list[str], model_name: str) -> list[list[float]]:
        return [[float(ord(t[0])), float(len(t))] for t in texts]

    module = types.ModuleType("pntx.embeddings")
    module.embed = fake_embed  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pntx.embeddings", module)


@pytest.mark.parametrize("sample_method", ["kmeans", "votek"])
def test_fit_resample_with_embedding_backed_sample_method(
    monkeypatch: pytest.MonkeyPatch, sample_method: str
) -> None:
    _install_fake_embeddings_module(monkeypatch)
    X, y = _pools()
    backend = FakeBackend(
        complete_responses=[_canned([_st("new synthetic 1"), _st("new synthetic 2")])]
    )
    sampler = SyntheticSampler(
        backend=backend,
        n_synthesized=2,
        batch_size=2,
        sample_method=sample_method,
        embedding_model="fake-model",
    )
    X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug[: len(X)] == X
    assert X_aug[len(X) :] == ["new synthetic 1", "new synthetic 2"]
    assert y_aug == y + [1, 1]


def test_backend_kwargs_with_instance_backend_raises() -> None:
    sampler = SyntheticSampler(
        backend=FakeBackend(), n_synthesized=1, backend_kwargs={"foo": "bar"}
    )
    with pytest.raises(TypeError, match="backend_kwargs"):
        sampler.fit_resample(*_pools())


def test_generation_result_contains_style_and_content_features() -> None:
    X, y = _pools()
    backend = FakeBackend(complete_responses=[_canned([_st("new synthetic 1")])])
    sampler = SyntheticSampler(backend=backend, n_synthesized=1, batch_size=1)
    sampler.fit_resample(X, y)
    assert sampler.generation_result_.style_features == ["polite tone"]
    assert sampler.generation_result_.content_features == ["customer service"]
    assert sampler.generation_result_.synthetic_texts[0].generalized_from == ["removed a name"]


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    X, y = _pools()
    backend = FakeBackend(
        complete_responses=[_canned([_st("new synthetic 1"), _st("new synthetic 2")])]
    )
    sampler = SyntheticSampler(backend=backend, n_synthesized=2, batch_size=2)
    sampler.fit_resample(X, y)

    path = tmp_path / "synthetic.json"
    sampler.save(path)

    loaded = SyntheticSampler.load(path, backend=backend)
    assert loaded.generation_result_ == sampler.generation_result_


def test_save_before_fit_raises_not_fitted() -> None:
    from sklearn.exceptions import NotFittedError

    sampler = SyntheticSampler(backend=FakeBackend(), n_synthesized=1)
    with pytest.raises(NotFittedError):
        sampler.save("does-not-matter.json")


def test_clone_reuses_the_same_backend_instance_without_deepcopy() -> None:
    backend = FakeBackend()
    sampler = SyntheticSampler(backend=backend, n_synthesized=1)
    cloned = clone(sampler)
    assert cloned.backend is backend
    assert cloned is not sampler


def test_fit_resample_does_not_require_imbalanced_learn_installed() -> None:
    assert "imblearn" not in sys.modules
    X, y = _pools()
    SyntheticSampler(backend=FakeBackend(), n_synthesized=0).fit_resample(X, y)
    assert "imblearn" not in sys.modules
