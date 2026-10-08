from __future__ import annotations

from pntx.backends.llama import LlamaCppBackend
from pntx.pn2t import CounterfactualOverSampler
from pntx.t2pn import LLMPromptingClassifier

from ..conftest import SAMPLE_NEGATIVE, SAMPLE_POSITIVE


def test_fit_resample_edits_negatives_into_positives(llama_backend: LlamaCppBackend) -> None:
    X = SAMPLE_POSITIVE[:1] + SAMPLE_NEGATIVE
    y = [1] + [0] * len(SAMPLE_NEGATIVE)
    sampler = CounterfactualOverSampler(
        backend=llama_backend, verify="self", batch_size=2, random_state=0
    )

    X_aug, y_aug = sampler.fit_resample(X, y)

    generated = X_aug[len(X) :]
    assert y_aug == y + [1] * len(generated)
    for edit in sampler.generation_result_.edits:
        assert edit.source_text in SAMPLE_NEGATIVE
        assert edit.edit_ratio <= sampler.max_edit_ratio
        assert edit.text not in X


def test_classifier_verifier_shares_the_loaded_backend(llama_backend: LlamaCppBackend) -> None:
    X = SAMPLE_POSITIVE + SAMPLE_NEGATIVE
    y = [1] * len(SAMPLE_POSITIVE) + [0] * len(SAMPLE_NEGATIVE)
    sampler = CounterfactualOverSampler(
        backend=llama_backend,
        verify=LLMPromptingClassifier(backend=llama_backend),
        verify_cv=3,
        sampling_strategy={1: len(SAMPLE_POSITIVE) + 1},
        batch_size=1,
        random_state=0,
    )

    sampler.fit_resample(X, y)

    result = sampler.generation_result_
    assert len(result.edits) + len(result.rejected) >= 1
