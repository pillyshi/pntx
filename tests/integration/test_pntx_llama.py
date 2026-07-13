from __future__ import annotations

from pntx import PNTX
from pntx.backends.llama import LlamaCppBackend

from ..conftest import SAMPLE_PAIRS


def test_classify_returns_a_valid_result(llama_backend: LlamaCppBackend) -> None:
    model = PNTX(backend=llama_backend).fit(SAMPLE_PAIRS)

    result = model.classify(SAMPLE_PAIRS[0][0])

    assert result.label in ("positive", "negative")
    assert 0.0 <= result.confidence <= 1.0


def test_classify_batch_matches_classify_per_item(llama_backend: LlamaCppBackend) -> None:
    model = PNTX(backend=llama_backend).fit(SAMPLE_PAIRS)
    texts = [pos for pos, _neg in SAMPLE_PAIRS]

    batch_results = model.classify_batch(texts)
    individual_results = [model.classify(text) for text in texts]

    assert [r.label for r in batch_results] == [r.label for r in individual_results]
