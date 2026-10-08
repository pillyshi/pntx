from __future__ import annotations

import pytest

from benchmarks.pn2t.pilot_metrics import Candidate, format_table, keeps, summarize


def _c(*, self_pos: bool, verifier_pos: bool, judge_pos: bool, easy: bool) -> Candidate:
    return Candidate(
        source_text="src",
        text="edit",
        edit_ratio=0.1,
        self_positive=self_pos,
        verifier_positive=verifier_pos,
        judge_positive=judge_pos,
        downstream_positive=easy,
    )


CANDIDATES = [
    _c(self_pos=True, verifier_pos=True, judge_pos=True, easy=True),  # good, easy
    _c(self_pos=True, verifier_pos=False, judge_pos=True, easy=False),  # good, hard
    _c(self_pos=True, verifier_pos=False, judge_pos=False, easy=False),  # failed flip
    _c(self_pos=False, verifier_pos=False, judge_pos=False, easy=False),  # failed flip
]


def test_keeps_per_strategy() -> None:
    assert [keeps(c, "none") for c in CANDIDATES] == [True] * 4
    assert [keeps(c, "self") for c in CANDIDATES] == [True, True, True, False]
    assert [keeps(c, "classifier") for c in CANDIDATES] == [True, False, False, False]
    with pytest.raises(ValueError):
        keeps(CANDIDATES[0], "judge")


def test_summarize_none() -> None:
    s = summarize(CANDIDATES, "none")
    assert s["n_kept"] == 4
    assert s["precision"] == 0.5
    assert s["catch_rate"] == 0.0
    assert s["false_reject_rate"] == 0.0
    assert s["downstream_easy"] == 0.25
    assert s["valid_and_hard"] == 1


def test_summarize_self_catches_one_of_two_failed_flips() -> None:
    s = summarize(CANDIDATES, "self")
    assert s["yield"] == 0.75
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["catch_rate"] == 0.5
    assert s["false_reject_rate"] == 0.0


def test_summarize_classifier_is_precise_but_keeps_only_easy_edits() -> None:
    s = summarize(CANDIDATES, "classifier")
    assert s["precision"] == 1.0
    assert s["catch_rate"] == 1.0
    assert s["false_reject_rate"] == 0.5
    assert s["downstream_easy"] == 1.0
    assert s["valid_and_hard"] == 0


def test_summarize_handles_empty_denominators() -> None:
    s = summarize([], "none")
    assert s["precision"] is None
    assert s["yield"] is None


def test_format_table_renders_rows() -> None:
    table = format_table([summarize(CANDIDATES, "none")])
    assert table.splitlines()[0].startswith("| strategy")
    assert "| none | 4 | 1.00 | 0.50 |" in table
