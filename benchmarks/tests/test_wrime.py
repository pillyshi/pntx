from __future__ import annotations

import random

import pytest

from benchmarks import wrime
from benchmarks.pn2t import downstream
from pntx.types import NEGATIVE, POSITIVE

HEADER = "Sentence\tUserID\tDatetime\tTrain/Dev/Test\tWriter_Sentiment\tAvg. Readers_Sentiment"
TSV = "\n".join(
    [
        HEADER,
        "うれしい\t1\tt\ttrain\t2\t1",
        "かなしい\t1\tt\ttrain\t-1\t-2",
        "ふつう\t1\tt\ttrain\t0\t0",
        "書き手だけ前向き\t2\tt\ttrain\t1\t0",
        "テストの文\t2\tt\ttest\t-1\t-1",
    ]
)


def test_parse_readers_drops_neutral_and_maps_sign() -> None:
    X, y = wrime.parse(TSV, "train")
    assert X == ["うれしい", "かなしい"]
    assert y == [POSITIVE, NEGATIVE]


def test_parse_writer_source_and_split() -> None:
    X, y = wrime.parse(TSV, "train", source="writer")
    assert X == ["うれしい", "かなしい", "書き手だけ前向き"]
    assert y == [POSITIVE, NEGATIVE, POSITIVE]
    assert wrime.parse(TSV, "test") == (["テストの文"], [NEGATIVE])


@pytest.mark.parametrize(("split", "source"), [("validation", "readers"), ("train", "both")])
def test_parse_rejects_unknown_split_or_source(split: str, source: str) -> None:
    with pytest.raises(ValueError):
        wrime.parse(TSV, split, source)


def test_sample_training_texts() -> None:
    texts = ["p1", "p2", "n1", "n2", "n3", "x" * 50]
    labels = [POSITIVE, POSITIVE, NEGATIVE, NEGATIVE, NEGATIVE, NEGATIVE]
    pos, neg = downstream.sample_training_texts(
        texts, labels, n_pos=1, n_neg=3, max_chars=10, rng=random.Random(0)
    )
    assert len(pos) == 1 and pos[0] in {"p1", "p2"}
    assert sorted(neg) == ["n1", "n2", "n3"]  # the 50-char text is filtered out
    with pytest.raises(ValueError, match="available"):
        downstream.sample_training_texts(
            texts, labels, n_pos=1, n_neg=4, max_chars=10, rng=random.Random(0)
        )
