from __future__ import annotations

from collections.abc import Iterable

NGRAM_SIZE = 3
SIMILARITY_THRESHOLD = 0.75


def char_ngrams(text: str, n: int = NGRAM_SIZE) -> set[str]:
    """Character n-grams of ``text``.

    Character n-grams (rather than word n-grams) work for both Japanese,
    which has no whitespace word boundaries, and English, without needing a
    tokenizer or language detection.
    """
    if len(text) <= n:
        return {text}
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def similarity(a: str, b: str, n: int = NGRAM_SIZE) -> float:
    """Jaccard similarity between the character n-grams of ``a`` and ``b``."""
    if a == b:
        return 1.0
    grams_a = char_ngrams(a, n)
    grams_b = char_ngrams(b, n)
    union = grams_a | grams_b
    if not union:
        return 0.0
    return len(grams_a & grams_b) / len(union)


def is_near_duplicate(
    text: str,
    others: Iterable[str],
    *,
    threshold: float = SIMILARITY_THRESHOLD,
) -> bool:
    """Whether ``text`` is an approximate duplicate of any text in ``others``."""
    return any(similarity(text, other) >= threshold for other in others)
