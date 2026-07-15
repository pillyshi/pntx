from __future__ import annotations

import pytest

pytest.importorskip("sentence_transformers")

from pntx import embeddings  # noqa: E402
from pntx.selection import DiversitySelector, NearestSelector  # noqa: E402


def test_embed_returns_one_vector_per_text() -> None:
    vectors = embeddings.embed(["hello world", "goodbye world"])
    assert len(vectors) == 2
    assert len(vectors[0]) == len(vectors[1])
    assert len(vectors[0]) > 0


def test_cosine_similarity_fn_ranks_semantically_close_text_higher() -> None:
    similarity = embeddings.cosine_similarity_fn()
    close = similarity("The movie was great", "The film was excellent")
    far = similarity("The movie was great", "I need to buy groceries")
    assert close > far


def test_nearest_selector_with_embeddings_similarity() -> None:
    pool = [
        "The movie was great",
        "I need to buy groceries",
    ]
    selector = NearestSelector(similarity_fn=embeddings.cosine_similarity_fn())
    selected = selector.select(pool, k=1, query="This film was fantastic")
    assert selected == [pool[0]]


def test_diversity_selector_with_embeddings_similarity() -> None:
    pool = [
        "The movie was great",
        "The film was excellent",
        "I need to buy groceries",
    ]
    selector = DiversitySelector(similarity_fn=embeddings.cosine_similarity_fn())
    selected = selector.select(pool, k=2)
    assert selected[0] == pool[0]
    assert selected[1] == pool[2]
