"""WRIME Ver.2 (Japanese SNS posts with sentiment polarity) as a binary task.

Source: https://github.com/ids-cv/wrime, ``wrime-ver2.tsv``: 35,000 posts by 60
writers with sentiment polarity in {-2, ..., 2} from the writer and from three
readers, plus the readers' average, and an official train/dev/test split
(30,000 / 2,500 / 2,500).

**License: CC BY-NC-ND 4.0.** Non-commercial research use only, and adaptations
must not be shared. Generated augmentations are adaptations of WRIME text, so
they must stay local (``benchmarks/results/`` is git-ignored) and must never be
published.

Binary labels: polarity > 0 -> ``"positive"``, < 0 -> ``"negative"``; neutral
(0) posts are dropped. By default the label comes from the readers' average
(``Avg. Readers_Sentiment``), i.e. how third parties read the text rather than
the writer's own feeling; ``source="writer"`` uses ``Writer_Sentiment``
instead. Files are fetched with the standard library and cached in
``~/.cache/pntx-benchmarks/wrime``.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from pathlib import Path

from pntx.types import NEGATIVE, POSITIVE, Label

URL = "https://raw.githubusercontent.com/ids-cv/wrime/master/wrime-ver2.tsv"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pntx-benchmarks" / "wrime"
SPLITS = ("train", "dev", "test")
LABEL_FIELDS = {"readers": "Avg. Readers_Sentiment", "writer": "Writer_Sentiment"}


def fetch(cache_dir: Path | None = None) -> str:
    """Text of ``wrime-ver2.tsv``, downloading it once."""
    cache = (cache_dir or DEFAULT_CACHE_DIR) / "wrime-ver2.tsv"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(URL, timeout=120) as response:  # noqa: S310 (fixed https URL)
            cache.write_bytes(response.read())
    return cache.read_text(encoding="utf-8")


def parse(text: str, split: str, source: str = "readers") -> tuple[list[str], list[Label]]:
    """Non-neutral posts of ``split`` as ``(X, y)`` with binary polarity labels."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    if source not in LABEL_FIELDS:
        raise ValueError(f"source must be one of {sorted(LABEL_FIELDS)}, got {source!r}")
    field = LABEL_FIELDS[source]
    X: list[str] = []
    y: list[Label] = []
    for row in csv.DictReader(io.StringIO(text, newline=""), delimiter="\t"):
        if row["Train/Dev/Test"] != split:
            continue
        polarity = float(row[field])
        if polarity == 0:
            continue
        X.append(row["Sentence"])
        y.append(POSITIVE if polarity > 0 else NEGATIVE)
    return X, y


def load(
    split: str, source: str = "readers", cache_dir: Path | None = None
) -> tuple[list[str], list[Label]]:
    """Binary WRIME ``split`` (neutral posts dropped)."""
    return parse(fetch(cache_dir), split, source)
