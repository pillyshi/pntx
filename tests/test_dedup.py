from __future__ import annotations

from pntx import dedup


def test_similarity_identical_strings_is_one() -> None:
    assert dedup.similarity("hello world", "hello world") == 1.0


def test_similarity_empty_others_is_zero_via_is_near_duplicate() -> None:
    assert not dedup.is_near_duplicate("anything", [])


def test_is_near_duplicate_english_trailing_addition() -> None:
    assert dedup.is_near_duplicate(
        "This movie was amazing and fun",
        ["This movie was amazing and fun!"],
    )


def test_is_near_duplicate_english_unrelated_text_is_false() -> None:
    assert not dedup.is_near_duplicate(
        "The support team was very helpful",
        ["This movie was boring"],
    )


def test_is_near_duplicate_english_same_template_different_subject_is_false() -> None:
    # A single substituted word can still make texts genuinely different;
    # n-gram similarity alone should not flag this as a duplicate.
    assert not dedup.is_near_duplicate(
        "I would recommend this restaurant to everyone",
        ["I would recommend this app to everyone"],
    )


def test_is_near_duplicate_japanese_trailing_addition() -> None:
    assert dedup.is_near_duplicate(
        "店員さんの笑顔が素敵だった",
        ["店員さんの笑顔が素敵だったです"],
    )


def test_is_near_duplicate_japanese_unrelated_text_is_false() -> None:
    assert not dedup.is_near_duplicate(
        "店員さんの笑顔が素敵だった",
        ["今日は雨が降っている"],
    )


def test_is_near_duplicate_japanese_opposite_sentiment_is_false() -> None:
    assert not dedup.is_near_duplicate(
        "この映画は最高だった",
        ["この映画は退屈だった"],
    )


def test_is_near_duplicate_checks_every_candidate() -> None:
    assert dedup.is_near_duplicate(
        "この映画は最高だった",
        ["まったく無関係な文章です", "この映画は最高だった!"],
    )


def test_contains_verbatim_span_detects_copied_substring() -> None:
    source = "田中太郎さんが2024年1月にサポートセンターに問い合わせをしてくれた"
    text = "先日、田中太郎さんが2024年1月にサポートセンターに問い合わせをしてくれて助かった"
    assert dedup.contains_verbatim_span(text, [source], min_len=20)


def test_contains_verbatim_span_false_for_short_or_no_overlap() -> None:
    source = "この映画は最高だった、また観に行きたい"
    text = "サポートの対応がとても丁寧で感動した"
    assert not dedup.contains_verbatim_span(text, [source], min_len=20)


def test_contains_verbatim_span_respects_min_len_threshold() -> None:
    source = "サポートが丁寧で助かった"
    text = "サポートが丁寧で本当に助かった、また利用したい"
    # "サポートが丁寧で" is a 8-char shared prefix; below a 20-char threshold
    # it isn't flagged, but a low enough threshold catches it.
    assert not dedup.contains_verbatim_span(text, [source], min_len=20)
    assert dedup.contains_verbatim_span(text, [source], min_len=8)
