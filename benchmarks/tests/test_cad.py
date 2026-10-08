from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks import cad
from pntx.types import NEGATIVE, POSITIVE

PAIRED_TSV = (
    "Sentiment\tText\tbatch_id\n"
    "Negative\tThe film was dull.\t1\n"
    "Positive\tThe film was gripping.\t1\n"
    "Positive\tGreat acting.\t2\n"
    "Negative\tTerrible acting.\t2\n"
    "Negative\tOrphan row without its pair.\t3\n"
    "Negative\tNeither side is original.\t4\n"
    "Positive\tStill not original.\t4\n"
)
ORIG_TSV = (
    "Sentiment\tText\n"
    "Negative\tThe film was dull.\n"
    "Positive\tGreat acting.\n"
    "Negative\tOrphan row without its pair.\n"
)


def test_parse_tsv_reads_header_and_rows() -> None:
    rows = cad.parse_tsv(ORIG_TSV)
    assert rows[0] == {"Sentiment": "Negative", "Text": "The film was dull."}
    assert len(rows) == 3


def test_pair_up_identifies_original_side_by_orig_split_membership() -> None:
    originals = {r["Text"] for r in cad.parse_tsv(ORIG_TSV)}
    pairs = {p.pair_id: p for p in cad.pair_up(cad.parse_tsv(PAIRED_TSV), originals)}

    assert set(pairs) == {"1", "2"}  # 3: no partner, 4: no original
    assert pairs["1"].original == "The film was dull."
    assert pairs["1"].revised == "The film was gripping."
    assert pairs["1"].original_label == NEGATIVE
    assert pairs["1"].revised_label == POSITIVE
    # The original can be the second row of its group, too.
    assert pairs["2"].original == "Great acting."
    assert pairs["2"].original_label == POSITIVE


def test_load_pairs_and_original_read_from_cache_without_network(tmp_path: Path) -> None:
    (tmp_path / "orig").mkdir()
    (tmp_path / "combined" / "paired").mkdir(parents=True)
    (tmp_path / "orig" / "test.tsv").write_text(ORIG_TSV, encoding="utf-8")
    (tmp_path / "combined" / "paired" / "test_paired.tsv").write_text(PAIRED_TSV, encoding="utf-8")

    assert len(cad.load_pairs("test", cache_dir=tmp_path)) == 2
    X, y = cad.load_original("test", cache_dir=tmp_path)
    assert X[1] == "Great acting."
    assert y == [NEGATIVE, POSITIVE, NEGATIVE]


def test_unknown_split_raises() -> None:
    with pytest.raises(ValueError, match="split"):
        cad.load_pairs("validation")
