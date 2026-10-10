from __future__ import annotations

import pytest

from benchmarks import ood
from benchmarks.pn2t import downstream
from pntx.types import NEGATIVE, POSITIVE


def test_label_mapper_follows_declared_positive_class() -> None:
    to_label = ood.label_mapper(ood.SPECS["yelp"], ["1", "2"])
    assert (to_label(0), to_label(1)) == (NEGATIVE, POSITIVE)
    to_label = ood.label_mapper(ood.SPECS["amazon"], ["negative", "positive"])
    assert (to_label(0), to_label(1)) == (NEGATIVE, POSITIVE)


def test_label_mapper_refuses_unexpected_class_names() -> None:
    with pytest.raises(ValueError, match="refusing to guess"):
        ood.label_mapper(ood.SPECS["amazon"], ["positive", "negative"])


def test_amazon_text_joins_title_and_content() -> None:
    assert ood.SPECS["amazon"].text({"title": "Great CD", "content": "Loved it"}) == (
        "Great CD. Loved it"
    )


def test_sample_balanced_is_balanced_deterministic_and_validates() -> None:
    texts = [f"p{i}" for i in range(10)] + [f"n{i}" for i in range(10)]
    labels = [POSITIVE] * 10 + [NEGATIVE] * 10
    X, y = ood.sample_balanced(texts, labels, 6, seed=0)
    assert y.count(POSITIVE) == y.count(NEGATIVE) == 3
    assert all((x[0] == "p") == (lab == POSITIVE) for x, lab in zip(X, y, strict=True))
    assert ood.sample_balanced(texts, labels, 6, seed=0) == (X, y)
    with pytest.raises(ValueError, match="even"):
        ood.sample_balanced(texts, labels, 5, seed=0)
    with pytest.raises(ValueError, match="need"):
        ood.sample_balanced(texts, labels, 30, seed=0)


def test_format_table_with_extra_test_sets() -> None:
    row = {"condition": "original", "n_train": 150}
    for t in ("original_test", "revised_test", "yelp_test"):
        for m in ("acc", "f1", "auc"):
            row[f"{t}_{m}"] = [0.5, 0.4, 0.6]
        row[f"{t}_pred_pos"] = 0.5
    table = downstream.format_table(
        [row], test_names=["original_test", "revised_test", "yelp_test"]
    )
    assert "yelp_test_auc" in table.splitlines()[0]
    assert table.splitlines()[2].count("0.500 [0.400, 0.600]") == 9
