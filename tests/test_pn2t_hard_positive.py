from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from sklearn.base import clone

from pntx.pn2t import HardPositiveOverSampler
from pntx.pn2t._base import PROMPT_MARGIN
from pntx.pn2t._structured import render_prompt
from pntx.selection import default_tokenizer

from .conftest import SAMPLE_NEGATIVE, SAMPLE_POSITIVE, FakeBackend


def _context_for(sampler: HardPositiveOverSampler, per_side_budget: int) -> int:
    """context_limit giving exactly ``per_side_budget`` tokens per side, from
    the *measured* fixed prompt (FakeBackend -> default tokenizer)."""
    fixed = default_tokenizer(
        render_prompt(*sampler._render_messages([[], []], sampler.batch_size))
    )
    return fixed + sampler.max_tokens + PROMPT_MARGIN + 2 * per_side_budget


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


def _generate(
    n: int, n_pos: int = len(SAMPLE_POSITIVE), pos_label: object = 1
) -> dict[object, int]:
    """sampling_strategy asking for ``n`` new positives on top of ``n_pos``."""
    return {pos_label: n_pos + n}


def test_fit_resample_requires_exactly_two_classes() -> None:
    X, _ = _pools()
    sampler = HardPositiveOverSampler(backend=FakeBackend())
    with pytest.raises(ValueError, match="exactly 2 classes"):
        sampler.fit_resample(X, (["a", "b", "c"] * (len(X) // 3 + 1))[: len(X)])
    with pytest.raises(ValueError, match="exactly 2 classes"):
        sampler.fit_resample(["a", "b"], [0, 0])
    with pytest.raises(ValueError, match="exactly 2 classes"):
        sampler.fit_resample(["a", "b"], [1, 1])


def test_fit_resample_accepts_alternate_binary_label_encodings() -> None:
    X = SAMPLE_POSITIVE + SAMPLE_NEGATIVE
    y = [1] * len(SAMPLE_POSITIVE) + [-1] * len(SAMPLE_NEGATIVE)
    backend = FakeBackend(complete_responses=[_canned([_hp("new hard positive")])])
    sampler = HardPositiveOverSampler(backend=backend, sampling_strategy=_generate(1), batch_size=1)
    X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == ["new hard positive"]
    assert y_aug[len(X) :] == [1]


def test_fit_resample_requires_pos_label_for_ambiguous_string_labels() -> None:
    X = SAMPLE_POSITIVE + SAMPLE_NEGATIVE
    y = ["spam"] * len(SAMPLE_POSITIVE) + ["ham"] * len(SAMPLE_NEGATIVE)
    sampler = HardPositiveOverSampler(backend=FakeBackend())
    with pytest.raises(ValueError, match="pass pos_label"):
        sampler.fit_resample(X, y)

    backend = FakeBackend(complete_responses=[_canned([_hp("new hard positive")])])
    sampler = HardPositiveOverSampler(
        backend=backend,
        sampling_strategy=_generate(1, pos_label="spam"),
        batch_size=1,
        pos_label="spam",
    )
    X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == ["new hard positive"]
    assert y_aug[len(X) :] == ["spam"]


def test_fit_resample_requires_matching_lengths() -> None:
    sampler = HardPositiveOverSampler(backend=FakeBackend())
    with pytest.raises(ValueError, match="same length"):
        sampler.fit_resample(["a", "b"], [1])


def test_sampling_strategy_requesting_no_new_samples_returns_original_data() -> None:
    X, y = _pools()
    sampler = HardPositiveOverSampler(backend=FakeBackend(), sampling_strategy=_generate(0))
    X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug == X
    assert y_aug == y
    assert sampler.generation_result_.hard_positives == []


def test_fit_resample_happy_path_appends_generated_positives() -> None:
    X, y = _pools()
    backend = FakeBackend(
        complete_responses=[_canned([_hp("new hard positive 1"), _hp("new hard positive 2")])]
    )
    sampler = HardPositiveOverSampler(backend=backend, sampling_strategy=_generate(2), batch_size=2)
    X_aug, y_aug = sampler.fit_resample(X, y)

    assert X_aug[: len(X)] == X
    assert X_aug[len(X) :] == ["new hard positive 1", "new hard positive 2"]
    assert y_aug == y + [1, 1]
    assert len(sampler.generation_result_.hard_positives) == 2
    assert sampler.generation_result_.positive_features == ["warmth"]


def test_default_sampling_strategy_auto_balances_classes() -> None:
    X = SAMPLE_POSITIVE[:1] + SAMPLE_NEGATIVE  # 1 positive, 3 negative -> need 2 to balance
    y = [1] + [0] * len(SAMPLE_NEGATIVE)
    backend = FakeBackend(complete_responses=[_canned([_hp("gen 1"), _hp("gen 2")])])
    sampler = HardPositiveOverSampler(backend=backend, batch_size=2)
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
    sampler = HardPositiveOverSampler(backend=backend, sampling_strategy=_generate(2), batch_size=2)
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
    sampler = HardPositiveOverSampler(
        backend=backend, sampling_strategy=_generate(2), batch_size=2, deduplicate=False
    )
    X_aug, _ = sampler.fit_resample(X, y)
    assert X_aug[len(X) :] == [duplicate_of_existing, "also new"]


def test_shortfall_after_max_batches_warns() -> None:
    X, y = _pools()
    backend = FakeBackend(complete_responses=[])  # complete() returns "" -> always fails to parse
    sampler = HardPositiveOverSampler(backend=backend, sampling_strategy=_generate(2), batch_size=1)
    with pytest.warns(UserWarning, match="expected"):
        X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug == X
    assert y_aug == y


def test_max_tokens_is_forwarded_to_backend_complete() -> None:
    X, y = _pools()
    backend = FakeBackend(
        complete_responses=[_canned([_hp("new hard positive 1"), _hp("new hard positive 2")])]
    )
    sampler = HardPositiveOverSampler(
        backend=backend, sampling_strategy=_generate(2), batch_size=2, max_tokens=777
    )
    sampler.fit_resample(X, y)
    assert backend.complete_max_tokens == [777]


def test_context_limit_too_small_for_max_tokens_raises() -> None:
    X, y = _pools()
    # Regression test: previously max_tokens for the generation call was
    # hardcoded independently of context_limit, so a small backend n_ctx
    # (e.g. 4096) combined with the old hardcoded max_tokens=4096 default
    # left zero room for the prompt and every batch silently failed. Now
    # this mismatch is caught up front with a clear error instead.
    sampler = HardPositiveOverSampler(
        backend=FakeBackend(), sampling_strategy=_generate(2), context_limit=4096, max_tokens=4096
    )
    with pytest.raises(ValueError, match="leaves no token budget for exemplars"):
        sampler.fit_resample(X, y)


def test_no_positive_example_fits_budget_raises() -> None:
    # Per-side budget of 5 tokens; the default tokenizer is len(text)//4 + 1,
    # so a 50-char positive costs 13 tokens and can never fit -- caught up
    # front instead of silently sending "(none)" as the positive exemplars.
    X = ["x" * 50, "ok"]
    y = [1, 0]
    sampler = HardPositiveOverSampler(
        backend=FakeBackend(), sampling_strategy=_generate(1, n_pos=1), max_tokens=90
    )
    sampler.set_params(context_limit=_context_for(sampler, 5))
    with pytest.raises(ValueError, match="no positive example fits"):
        sampler.fit_resample(X, y)


def test_no_negative_example_fits_budget_raises() -> None:
    X = ["ok", "y" * 50]
    y = [1, 0]
    sampler = HardPositiveOverSampler(
        backend=FakeBackend(), sampling_strategy=_generate(1, n_pos=1), max_tokens=90
    )
    sampler.set_params(context_limit=_context_for(sampler, 5))
    with pytest.raises(ValueError, match="no negative example fits"):
        sampler.fit_resample(X, y)


def test_fixed_prompt_larger_than_context_raises_with_measured_size() -> None:
    sampler = HardPositiveOverSampler(
        backend=FakeBackend(), sampling_strategy=_generate(1), max_tokens=90, context_limit=400
    )
    with pytest.raises(ValueError, match="fixed prompt"):
        sampler.fit_resample(*_pools())


def test_exemplar_sampling_balances_positive_and_negative_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from pntx.pn2t import prompts as prompts_mod

    real_build_user_message = prompts_mod.build_user_message
    captured: dict[str, list[str]] = {}

    def _capturing_build_user_message(
        *,
        pos_texts: list[str],
        neg_texts: list[str],
        n_synthesized: int,
        language: str | None = None,
    ) -> str:
        captured["pos"] = list(pos_texts)
        captured["neg"] = list(neg_texts)
        return real_build_user_message(
            pos_texts=pos_texts, neg_texts=neg_texts, n_synthesized=n_synthesized, language=language
        )

    monkeypatch.setattr(prompts_mod, "build_user_message", _capturing_build_user_message)

    # Per-class budget of 12 tokens (measured fixed prompt + max_tokens + margin
    # + 2*12). 5 one-token positives all fit (5 <= 12); 5 five-token negatives only
    # let 2 fit (10 <= 12, a 3rd would be 15 > 12) -- an unbalanced 5-vs-2
    # split unless the post-sampling balancing step trims the positive side
    # down to match.
    pos_texts = ["a", "b", "c", "d", "e"]
    neg_texts = [f"{'n' * 16}{i}" for i in range(5)]
    X = pos_texts + neg_texts
    y = [1] * len(pos_texts) + [0] * len(neg_texts)

    backend = FakeBackend(complete_responses=[_canned([_hp("gen 1")])])
    sampler = HardPositiveOverSampler(
        backend=backend,
        sampling_strategy=_generate(1, n_pos=len(pos_texts)),
        batch_size=1,
        max_tokens=100,
    )
    sampler.set_params(context_limit=_context_for(sampler, 12))
    sampler.fit_resample(X, y)

    assert captured["pos"]
    assert captured["neg"]
    assert len(captured["pos"]) == len(captured["neg"]) == 2


def test_overflowing_prompt_drops_exemplars_but_keeps_instructions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Regression: LlamaCppBackend trims overflowing prompts from the FRONT,
    # which for pn2t's instruction-first prompts cut the role/rules/schema.
    # The sampler must make the prompt fit itself by dropping exemplars.
    pos = [f"positive example number {i} " * 3 for i in range(8)]
    neg = [f"negative example number {i} " * 3 for i in range(8)]
    X, y = pos + neg, [1] * 8 + [0] * 8
    backend = FakeBackend(complete_responses=[_canned([_hp("gen 1")])])
    sampler = HardPositiveOverSampler(
        backend=backend, sampling_strategy=_generate(1, n_pos=8), batch_size=1, max_tokens=90
    )
    limit = _context_for(sampler, 40)
    sampler.set_params(context_limit=limit)
    # Make budgeted sampling admit everything, so only the final fit can save us.
    monkeypatch.setattr(HardPositiveOverSampler, "_exemplar_budget", lambda self, a: 10**6)
    sampler.fit_resample(X, y)

    [prompt] = backend.complete_calls
    assert prompt.startswith("You are a data augmentation expert.")
    assert '"hard_positives"' in prompt  # schema intact
    assert default_tokenizer(prompt) + 90 + PROMPT_MARGIN <= limit
    assert sum(p in prompt for p in pos) < 8  # exemplars were dropped instead
    assert sum(p in prompt for p in pos) >= 1 and sum(n in prompt for n in neg) >= 1


def test_prompt_that_cannot_fit_even_minimally_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    X = ["p " * 400, "n " * 400]
    sampler = HardPositiveOverSampler(
        backend=FakeBackend(), sampling_strategy=_generate(1, n_pos=1), max_tokens=90
    )
    sampler.set_params(context_limit=_context_for(sampler, 30))
    monkeypatch.setattr(HardPositiveOverSampler, "_exemplar_budget", lambda self, a: 10**6)
    monkeypatch.setattr(HardPositiveOverSampler, "_check_exemplars_fit", lambda *a: None)
    with pytest.raises(ValueError, match="does not fit the context window"):
        sampler.fit_resample(X, [1, 0])


class _WindowedBackend(FakeBackend):
    context_window = 3000


def test_context_limit_is_capped_by_backend_context_window() -> None:
    backend = _WindowedBackend(complete_responses=[_canned([_hp("gen 1")])])
    sampler = HardPositiveOverSampler(
        backend=backend, sampling_strategy=_generate(1), batch_size=1, max_tokens=90
    )  # default context_limit=100_000
    sampler.fit_resample(*_pools())
    assert sampler.context_limit_ == 3000
    [prompt] = backend.complete_calls
    assert default_tokenizer(prompt) + 90 + PROMPT_MARGIN <= 3000


def test_invalid_sample_method_raises() -> None:
    X, y = _pools()
    sampler = HardPositiveOverSampler(
        backend=FakeBackend(), sampling_strategy=_generate(2), sample_method="bogus"
    )
    with pytest.raises(ValueError, match="sample_method"):
        sampler.fit_resample(X, y)


def _install_fake_embeddings_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stands in for ``pntx.embeddings`` so ``sample_method="kmeans"/"votek"``
    tests don't require sentence-transformers to be installed."""

    def fake_embed(texts: list[str], model_name: str) -> list[list[float]]:
        # Cheap deterministic embedding: (index of first char, length).
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
        complete_responses=[_canned([_hp("new hard positive 1"), _hp("new hard positive 2")])]
    )
    sampler = HardPositiveOverSampler(
        backend=backend,
        sampling_strategy=_generate(2),
        batch_size=2,
        sample_method=sample_method,
        embedding_model="fake-model",
    )
    X_aug, y_aug = sampler.fit_resample(X, y)
    assert X_aug[: len(X)] == X
    assert X_aug[len(X) :] == ["new hard positive 1", "new hard positive 2"]
    assert y_aug == y + [1, 1]


def test_backend_kwargs_with_instance_backend_raises() -> None:
    sampler = HardPositiveOverSampler(backend=FakeBackend(), backend_kwargs={"foo": "bar"})
    with pytest.raises(TypeError, match="backend_kwargs"):
        sampler.fit_resample(*_pools())


def test_save_and_load_round_trip(tmp_path: Path) -> None:
    X, y = _pools()
    backend = FakeBackend(
        complete_responses=[_canned([_hp("new hard positive 1"), _hp("new hard positive 2")])]
    )
    sampler = HardPositiveOverSampler(backend=backend, sampling_strategy=_generate(2), batch_size=2)
    sampler.fit_resample(X, y)

    path = tmp_path / "hard_positives.json"
    sampler.save(path)

    loaded = HardPositiveOverSampler.load(path, backend=backend)
    assert loaded.generation_result_ == sampler.generation_result_


def test_save_before_fit_raises_not_fitted() -> None:
    from sklearn.exceptions import NotFittedError

    sampler = HardPositiveOverSampler(backend=FakeBackend())
    with pytest.raises(NotFittedError):
        sampler.save("does-not-matter.json")


def test_clone_reuses_the_same_backend_instance_without_deepcopy() -> None:
    backend = FakeBackend()
    sampler = HardPositiveOverSampler(backend=backend)
    cloned = clone(sampler)
    assert cloned.backend is backend
    assert cloned is not sampler


def test_fit_resample_does_not_require_imbalanced_learn_installed() -> None:
    import sys

    assert "imblearn" not in sys.modules
    X, y = _pools()
    HardPositiveOverSampler(backend=FakeBackend(), sampling_strategy=_generate(0)).fit_resample(
        X, y
    )
    assert "imblearn" not in sys.modules
