from __future__ import annotations

import math
from collections.abc import Callable

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "pntx.embeddings requires the 'embeddings' extra. "
        "Install it with: pip install 'pntx[embeddings]'"
    ) from e

DEFAULT_MODEL = "paraphrase-albert-small-v2"

_model_cache: dict[str, SentenceTransformer] = {}


def _load_model(model_name: str) -> SentenceTransformer:
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def embed(texts: list[str], model_name: str = DEFAULT_MODEL) -> list[list[float]]:
    """Encode ``texts`` into embedding vectors using a sentence-transformers model."""
    model = _load_model(model_name)
    return [vector.tolist() for vector in model.encode(texts)]


def cosine_similarity_fn(model_name: str = DEFAULT_MODEL) -> Callable[[str, str], float]:
    """Build a ``(text_a, text_b) -> float`` similarity function backed by
    sentence-transformers embeddings, for use as ``NearestSelector`` or
    ``DiversitySelector``'s ``similarity_fn``.

    Per-text embeddings are cached within the returned function, since
    both selectors call it repeatedly with the same texts.
    """
    model = _load_model(model_name)
    vector_cache: dict[str, list[float]] = {}

    def embed_one(text: str) -> list[float]:
        if text not in vector_cache:
            vector_cache[text] = model.encode([text])[0].tolist()
        return vector_cache[text]

    def similarity(a: str, b: str) -> float:
        return _cosine_similarity(embed_one(a), embed_one(b))

    return similarity


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)
