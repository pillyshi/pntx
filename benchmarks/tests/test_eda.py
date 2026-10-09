from __future__ import annotations

import random

import pytest

from benchmarks import eda

SYN = {"boring": ["dull", "tedious"], "movie": ["film"], "plot": ["storyline"]}
STOP = {"the", "was", "and", "a"}


def synonyms(word: str) -> list[str]:
    return SYN.get(word, [])


def test_normalize_strips_markup_and_punctuation() -> None:
    assert eda.normalize("The movie's plot -- was BORING!<br />Really.") == (
        "the movies plot was boring really"
    )


def test_synonym_replacement_replaces_all_occurrences_of_chosen_word() -> None:
    words = "the boring movie was boring".split()
    out = eda.synonym_replacement(words, 1, random.Random(0), synonyms, STOP)
    assert len(out) == len(words)
    changed = {w for w, o in zip(words, out, strict=True) if w != o}
    assert len(changed) == 1  # one distinct word replaced, everywhere it occurs
    word = changed.pop()
    replacements = {o for w, o in zip(words, out, strict=True) if w == word}
    assert len(replacements) == 1 and replacements <= set(SYN[word])


def test_synonym_replacement_skips_stop_words_and_words_without_synonyms() -> None:
    words = "the was and".split()
    assert eda.synonym_replacement(words, 3, random.Random(0), synonyms, STOP) == words


def test_random_insertion_adds_n_synonyms() -> None:
    words = "the boring movie".split()
    out = eda.random_insertion(words, 2, random.Random(0), synonyms, STOP)
    assert len(out) == len(words) + 2
    inserted = list(out)
    for w in words:
        inserted.remove(w)
    assert set(inserted) <= {"dull", "tedious", "film"}


def test_random_swap_preserves_multiset() -> None:
    words = "a b c d e".split()
    out = eda.random_swap(words, 2, random.Random(0))
    assert sorted(out) == sorted(words)


def test_random_deletion_never_returns_empty() -> None:
    words = "a b c".split()
    assert eda.random_deletion(words, 1.0, random.Random(0)) in (["a"], ["b"], ["c"])
    assert eda.random_deletion(["only"], 1.0, random.Random(0)) == ["only"]


def test_eda_count_determinism_and_alpha_scaling() -> None:
    sentence = "The boring movie had a boring plot and the movie dragged on"
    kwargs = {"alpha": 0.1, "num_aug": 6, "synonyms": synonyms, "stop_words": STOP}
    a = eda.eda(sentence, rng=random.Random(1), **kwargs)
    b = eda.eda(sentence, rng=random.Random(1), **kwargs)
    assert a == b
    assert len(a) == 6
    assert eda.eda("", rng=random.Random(0), **kwargs) == []
    assert eda.eda(sentence, rng=random.Random(0), **{**kwargs, "num_aug": 0}) == []


def test_augment_pool_returns_exact_total_spread_over_texts() -> None:
    texts = ["the boring movie", "a boring plot", "the movie"]
    out = eda.augment_pool(
        texts, 7, alpha=0.5, rng=random.Random(0), synonyms=synonyms, stop_words=STOP
    )
    assert len(out) == 7
    assert (
        eda.augment_pool([], 5, alpha=0.1, rng=random.Random(0), synonyms=synonyms, stop_words=STOP)
        == []
    )


@pytest.mark.parametrize("n_total", [1, 4, 9])
def test_augment_pool_sizes(n_total: int) -> None:
    out = eda.augment_pool(
        ["the boring movie plot"],
        n_total,
        alpha=0.25,
        rng=random.Random(0),
        synonyms=synonyms,
        stop_words=STOP,
    )
    assert len(out) == n_total
