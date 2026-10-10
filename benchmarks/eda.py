"""EDA (Easy Data Augmentation; Wei & Zou, EMNLP-IJCNLP 2019) as a baseline.

Implements the four operations as defined in the paper ([[wei-2019]] in
``research/catalog.yaml``): for an input sentence of ``l`` words, each
augmented sentence applies *one* randomly chosen operation with
``n = max(1, int(alpha * l))``:

- synonym replacement (SR): replace ``n`` distinct non-stop-words with a random
  WordNet synonym;
- random insertion (RI): ``n`` times, insert a random synonym of a random
  non-stop-word at a random position;
- random swap (RS): ``n`` times, swap two random words;
- random deletion (RD): delete each word with probability ``p = alpha``.

Text is lowercased and reduced to letters and spaces before augmenting, so
outputs are bags of plain words; this is fine for the TF-IDF classifier the
downstream benchmark uses first, and keeps the operations well defined.

This lives in ``benchmarks/`` on purpose, not in the pntx library: it needs
WordNet (``nltk`` plus a data download, in the ``benchmark`` dependency group)
and is effectively English-only (see
``research/ideas/downstream-augmentation-benchmark.md``). The synonym source is
injectable so tests run offline.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from pathlib import Path

SynonymFn = Callable[[str], list[str]]

DEFAULT_NLTK_DIR = Path.home() / ".cache" / "pntx-benchmarks" / "nltk_data"

_NON_LETTERS = re.compile(r"[^a-z ]+")
_SPACES = re.compile(r" +")


def normalize(text: str) -> str:
    """Lowercase; turn HTML line breaks, apostrophes and anything that isn't a
    letter into spaces; collapse whitespace."""
    text = text.lower().replace("<br />", " ").replace("'", "").replace("’", "")
    return _SPACES.sub(" ", _NON_LETTERS.sub(" ", text)).strip()


def wordnet_synonyms(nltk_dir: Path = DEFAULT_NLTK_DIR) -> tuple[SynonymFn, set[str]]:
    """``(synonyms_fn, stop_words)`` backed by NLTK's WordNet and English
    stop-word list, downloading the data into ``nltk_dir`` on first use."""
    import nltk

    nltk_dir.mkdir(parents=True, exist_ok=True)
    if nltk_dir == DEFAULT_NLTK_DIR:
        # NLTK refuses to download into a directory that is (or has an
        # ancestor that is) group- or world-writable, which a umask of 002
        # (e.g. Ubuntu/WSL defaults) produces. Keep our own cache dirs private.
        for directory in (nltk_dir.parent, nltk_dir):
            directory.chmod(0o700)
    if str(nltk_dir) not in nltk.data.path:
        nltk.data.path.insert(0, str(nltk_dir))
    for package, probe in (("wordnet", "corpora/wordnet"), ("stopwords", "corpora/stopwords")):
        try:
            nltk.data.find(probe)
        except LookupError:
            nltk.download(package, download_dir=str(nltk_dir), quiet=True)
    from nltk.corpus import stopwords, wordnet

    cache: dict[str, list[str]] = {}

    def synonyms(word: str) -> list[str]:
        if word not in cache:
            found = {
                normalize(lemma.name().replace("_", " "))
                for synset in wordnet.synsets(word)
                for lemma in synset.lemmas()
            }
            cache[word] = sorted(s for s in found if s and s != word)
        return cache[word]

    return synonyms, set(stopwords.words("english"))


def synonym_replacement(
    words: list[str], n: int, rng: random.Random, synonyms: SynonymFn, stop_words: set[str]
) -> list[str]:
    candidates = sorted({w for w in words if w not in stop_words})
    rng.shuffle(candidates)
    out = list(words)
    replaced = 0
    for word in candidates:
        options = synonyms(word)
        if not options:
            continue
        choice = rng.choice(options)
        out = [choice if w == word else w for w in out]
        replaced += 1
        if replaced >= n:
            break
    return out


def random_insertion(
    words: list[str], n: int, rng: random.Random, synonyms: SynonymFn, stop_words: set[str]
) -> list[str]:
    out = list(words)
    candidates = [w for w in words if w not in stop_words]
    for _ in range(n):
        for _attempt in range(10):
            if not candidates:
                return out
            options = synonyms(rng.choice(candidates))
            if options:
                out.insert(rng.randint(0, len(out)), rng.choice(options))
                break
    return out


def random_swap(words: list[str], n: int, rng: random.Random) -> list[str]:
    out = list(words)
    if len(out) < 2:
        return out
    for _ in range(n):
        i, j = rng.sample(range(len(out)), 2)
        out[i], out[j] = out[j], out[i]
    return out


def random_deletion(words: list[str], p: float, rng: random.Random) -> list[str]:
    if len(words) <= 1:
        return list(words)
    kept = [w for w in words if rng.random() > p]
    return kept if kept else [rng.choice(words)]


def eda(
    sentence: str,
    *,
    alpha: float,
    num_aug: int,
    rng: random.Random,
    synonyms: SynonymFn,
    stop_words: set[str],
) -> list[str]:
    """``num_aug`` augmented versions of ``sentence`` (the original is not
    included). Operations are spread evenly, then shuffled and truncated."""
    if num_aug < 1:
        return []
    words = normalize(sentence).split()
    if not words:
        return []
    n = max(1, int(alpha * len(words)))
    per_op = num_aug // 4 + 1
    augmented: list[str] = []
    for _ in range(per_op):
        augmented.append(" ".join(synonym_replacement(words, n, rng, synonyms, stop_words)))
        augmented.append(" ".join(random_insertion(words, n, rng, synonyms, stop_words)))
        augmented.append(" ".join(random_swap(words, n, rng)))
        augmented.append(" ".join(random_deletion(words, alpha, rng)))
    rng.shuffle(augmented)
    return augmented[:num_aug]


def augment_pool(
    texts: list[str],
    n_total: int,
    *,
    alpha: float,
    rng: random.Random,
    synonyms: SynonymFn,
    stop_words: set[str],
) -> list[str]:
    """``n_total`` EDA sentences drawn as evenly as possible from ``texts``
    (each text gets ``ceil(n_total / len(texts))`` augmentations, then the
    combined list is shuffled and truncated)."""
    if not texts or n_total <= 0:
        return []
    per_text = -(-n_total // len(texts))
    out: list[str] = []
    for text in texts:
        out.extend(
            eda(
                text,
                alpha=alpha,
                num_aug=per_text,
                rng=rng,
                synonyms=synonyms,
                stop_words=stop_words,
            )
        )
    rng.shuffle(out)
    return out[:n_total]
