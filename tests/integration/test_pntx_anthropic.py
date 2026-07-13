from __future__ import annotations

from pntx import PNTX
from pntx.backends.anthropic import AnthropicBackend

from ..conftest import SAMPLE_PAIRS


def test_classify_returns_a_valid_result(anthropic_backend: AnthropicBackend) -> None:
    model = PNTX(backend=anthropic_backend).fit(SAMPLE_PAIRS)

    result = model.classify(SAMPLE_PAIRS[0][0])

    assert result.label in ("positive", "negative")
    assert 0.0 <= result.confidence <= 1.0


def test_classify_batch_matches_classify_per_item(anthropic_backend: AnthropicBackend) -> None:
    model = PNTX(backend=anthropic_backend).fit(SAMPLE_PAIRS)
    texts = [pos for pos, _neg in SAMPLE_PAIRS]

    batch_results = model.classify_batch(texts)
    individual_results = [model.classify(text) for text in texts]

    assert [r.label for r in batch_results] == [r.label for r in individual_results]
