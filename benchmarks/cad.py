"""Load the counterfactually-augmented IMDb data (CAD) of Kaushik et al. (2020).

Source: https://github.com/acmi-lab/counterfactually-augmented-data
(Apache-2.0), the data release of "Learning the Difference that Makes a
Difference with Counterfactually-Augmented Data" (ICLR 2020; see
``research/notes/kaushik-2019.md``). Each original IMDb review was revised by
a crowd worker into the opposite sentiment with as few changes as possible,
so the data gives real human counterfactual edits to compare pntx's
``CounterfactualOverSampler`` against.

Files are fetched with the standard library (no ``datasets`` dependency) and
cached under ``~/.cache/pntx-benchmarks/cad`` (override with ``cache_dir``).
Parsing is kept in pure functions (``parse_tsv``/``pair_up``) so it can be
unit-tested offline.

Labels follow ``benchmarks/jigsaw.py``'s convention of pntx's binary
``Label``: positive sentiment -> ``"positive"``, negative -> ``"negative"``.
"""

from __future__ import annotations

import csv
import io
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pntx.types import NEGATIVE, POSITIVE, Label

BASE_URL = "https://raw.githubusercontent.com/acmi-lab/counterfactually-augmented-data/master"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pntx-benchmarks" / "cad"
SPLITS = ("train", "dev", "test")

_SENTIMENT_TO_LABEL: dict[str, Label] = {"Positive": POSITIVE, "Negative": NEGATIVE}


@dataclass(frozen=True)
class CADPair:
    """One human counterfactual revision: ``original`` (labelled
    ``original_label``) minimally edited into ``revised`` (the opposite label)."""

    pair_id: str
    original: str
    revised: str
    original_label: Label

    @property
    def revised_label(self) -> Label:
        return POSITIVE if self.original_label == NEGATIVE else NEGATIVE


def fetch(relative_path: str, cache_dir: Path | None = None) -> str:
    """Return the text of ``sentiment/<relative_path>``, downloading it once."""
    cache = (cache_dir or DEFAULT_CACHE_DIR) / relative_path
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/sentiment/{relative_path}"
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 (fixed https URL)
            cache.write_bytes(response.read())
    return cache.read_text(encoding="utf-8")


def parse_tsv(text: str) -> list[dict[str, str]]:
    """Rows of a CAD ``.tsv`` file (header: ``Sentiment``, ``Text``[, ``batch_id``])."""
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter="\t"))


def pair_up(paired_rows: list[dict[str, str]], original_texts: set[str]) -> list[CADPair]:
    """Group ``combined/paired`` rows by ``batch_id`` into original/revised pairs.

    The paired files don't mark which side is the original, so it is
    identified by membership in the ``orig`` split's texts. Groups that aren't
    exactly one original plus one revision are skipped.
    """
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in paired_rows:
        groups[row["batch_id"]].append(row)

    pairs: list[CADPair] = []
    for pair_id, rows in groups.items():
        if len(rows) != 2:
            continue
        originals = [r for r in rows if r["Text"] in original_texts]
        if len(originals) != 1:
            continue
        original = originals[0]
        revised = rows[1] if rows[0] is original else rows[0]
        original_label = _SENTIMENT_TO_LABEL[original["Sentiment"]]
        if _SENTIMENT_TO_LABEL[revised["Sentiment"]] == original_label:
            continue
        pairs.append(
            CADPair(
                pair_id=pair_id,
                original=original["Text"],
                revised=revised["Text"],
                original_label=original_label,
            )
        )
    return pairs


def load_pairs(split: str, cache_dir: Path | None = None) -> list[CADPair]:
    """Original/revised pairs of one split (``"train"``, ``"dev"`` or ``"test"``)."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    originals = {r["Text"] for r in parse_tsv(fetch(f"orig/{split}.tsv", cache_dir))}
    paired = parse_tsv(fetch(f"combined/paired/{split}_paired.tsv", cache_dir))
    return pair_up(paired, originals)


def load_original(split: str, cache_dir: Path | None = None) -> tuple[list[str], list[Label]]:
    """The *original* (unrevised) reviews of one split as ``(X, y)``."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    rows = parse_tsv(fetch(f"orig/{split}.tsv", cache_dir))
    return [r["Text"] for r in rows], [_SENTIMENT_TO_LABEL[r["Sentiment"]] for r in rows]
