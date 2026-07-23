from __future__ import annotations

from pntx.backends.llama import LlamaCppBackend
from pntx.pn2t import SyntheticSampler

from ..conftest import SAMPLE_NEGATIVE, SAMPLE_POSITIVE


def test_fit_resample_generates_requested_number_of_synthetic_texts(
    llama_backend: LlamaCppBackend,
) -> None:
    X = SAMPLE_POSITIVE + SAMPLE_NEGATIVE
    y = [1] * len(SAMPLE_POSITIVE) + [0] * len(SAMPLE_NEGATIVE)
    sampler = SyntheticSampler(backend=llama_backend, n_synthesized=2, batch_size=2)

    X_aug, y_aug = sampler.fit_resample(X, y)

    assert len(X_aug) == len(X) + 2
    assert y_aug == y + [1, 1]
    generated = X_aug[len(X) :]
    assert all(text not in X for text in generated)
