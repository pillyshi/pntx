"""Downstream benchmark: does pn2t augmentation help a classifier?

Design and rationale: ``research/ideas/downstream-augmentation-benchmark.md``.
Data: Kaushik et al.'s counterfactually-augmented IMDb (``benchmarks/cad.py``).

A low-resource, imbalanced training set (``n_pos`` positives, ``n_neg``
negatives) is drawn from the CAD *training* pairs, and ``n_neg - n_pos``
positives are added under each condition so every augmented training set ends
up balanced and the same size:

- ``original``: no augmentation (lower reference);
- ``duplicate``: positives re-sampled with replacement (same-size control,
  after Huang et al. 2020);
- ``eda``: EDA sentences from the positives (``benchmarks/eda.py``);
- ``hard_positive``: ``pn2t.HardPositiveOverSampler``;
- ``counterfactual``: ``pn2t.CounterfactualOverSampler`` (default ``verify``);
- ``human_cad``: the crowd workers' positive revisions of training negatives
  (upper reference; only possible on CAD data).

Two subcommands, so generation (slow, local LLM) and evaluation (fast for
TF-IDF, GPU for fine-tuning) can run on different machines::

    uv run --extra llama --group benchmark python -m benchmarks.pn2t.downstream \\
        generate --editor-model /path/to/model.gguf --output aug.json
    uv run --group benchmark python -m benchmarks.pn2t.downstream \\
        evaluate --augmentations aug.json [--classifier finetuning --model-name ...]

Evaluation reports accuracy and macro-F1 with bootstrap 95% CIs on the original
CAD test reviews and on their counterfactual revisions (Kaushik et al.'s
Table 5 setup). Not run in CI.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import cad
from pntx.types import NEGATIVE, POSITIVE

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

CONDITIONS = ("original", "duplicate", "eda", "hard_positive", "counterfactual", "human_cad")


# ---- pure helpers (unit-tested) -------------------------------------------------


def sample_training_pairs(
    pairs: Sequence[cad.CADPair], *, n_pos: int, n_neg: int, max_chars: int, rng: random.Random
) -> tuple[list[cad.CADPair], list[cad.CADPair]]:
    """``(positive_pairs, negative_pairs)`` sampled by their *original* label,
    keeping only originals up to ``max_chars`` (so they fit LLM prompts)."""
    pos = [p for p in pairs if p.original_label == POSITIVE and len(p.original) <= max_chars]
    neg = [p for p in pairs if p.original_label == NEGATIVE and len(p.original) <= max_chars]
    if n_pos > len(pos) or n_neg > len(neg):
        raise ValueError(f"requested {n_pos}/{n_neg} pos/neg, only {len(pos)}/{len(neg)} available")
    return rng.sample(pos, n_pos), rng.sample(neg, n_neg)


def duplicate_positives(positives: Sequence[str], n: int, rng: random.Random) -> list[str]:
    """``n`` positives re-sampled with replacement."""
    return [rng.choice(positives) for _ in range(n)] if positives else []


def human_cad_positives(negative_pairs: Sequence[cad.CADPair], n: int) -> list[str]:
    """The human positive revisions of the first ``n`` training negatives."""
    if n > len(negative_pairs):
        raise ValueError(f"need {n} revisions but only {len(negative_pairs)} negatives")
    return [p.revised for p in negative_pairs[:n]]


def equalize(
    augmentations: dict[str, list[str]],
) -> tuple[dict[str, list[str]], int, list[str]]:
    """Truncate every non-empty condition to the smallest non-empty count, so
    all augmented training sets have the same size (a sampler may fall
    short). Conditions that produced nothing are dropped and returned
    separately -- otherwise one failed sampler would truncate every other
    condition to zero."""
    empty = sorted(k for k, v in augmentations.items() if not v)
    kept = {k: v for k, v in augmentations.items() if v}
    n = min((len(v) for v in kept.values()), default=0)
    return {k: v[:n] for k, v in kept.items()}, n, empty


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Callable[[Sequence[str], Sequence[str]], float],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """``(point, low, high)``: the metric and its 95% percentile bootstrap CI
    over test examples."""
    yt, yp = np.asarray(y_true), np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, len(yt), len(yt))
        stats.append(metric(yt[idx], yp[idx]))
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(metric(yt, yp)), float(low), float(high)


# ---- generate ---------------------------------------------------------------------


def generate(args: argparse.Namespace) -> None:
    from benchmarks import eda
    from pntx.backends.llama import LlamaCppBackend
    from pntx.pn2t import CounterfactualOverSampler, HardPositiveOverSampler

    rng = random.Random(args.seed)
    pos_pairs, neg_pairs = sample_training_pairs(
        cad.load_pairs("train", args.cache_dir),
        n_pos=args.n_pos,
        n_neg=args.n_neg,
        max_chars=args.max_chars,
        rng=rng,
    )
    X = [p.original for p in pos_pairs] + [p.original for p in neg_pairs]
    y = [POSITIVE] * len(pos_pairs) + [NEGATIVE] * len(neg_pairs)
    n_add = args.n_neg - args.n_pos
    positives = [p.original for p in pos_pairs]
    print(f"train: {args.n_pos} pos / {args.n_neg} neg; adding {n_add} positives per condition")

    augmentations: dict[str, dict[str, Any]] = {}

    def record(name: str, texts: list[str], seconds: float, **meta: Any) -> None:
        augmentations[name] = {"texts": texts, "seconds": seconds, **meta}
        print(f"  {name}: {len(texts)} texts in {seconds:.0f}s")

    start = time.perf_counter()
    record("duplicate", duplicate_positives(positives, n_add, rng), time.perf_counter() - start)

    synonyms, stop_words = eda.wordnet_synonyms()
    start = time.perf_counter()
    record(
        "eda",
        eda.augment_pool(
            positives,
            n_add,
            alpha=args.eda_alpha,
            rng=rng,
            synonyms=synonyms,
            stop_words=stop_words,
        ),
        time.perf_counter() - start,
        alpha=args.eda_alpha,
    )
    record("human_cad", human_cad_positives(neg_pairs, n_add), 0.0)

    backend = LlamaCppBackend(
        model_path=args.editor_model,
        n_ctx=args.n_ctx,
        n_gpu_layers=args.n_gpu_layers,
        verbose=False,
    )
    common = {
        "backend": backend,
        "sampling_strategy": {POSITIVE: args.n_pos + n_add},
        "batch_size": args.batch_size,
        "context_limit": args.n_ctx,
        "max_tokens": args.max_tokens,
        "random_state": args.seed,
    }
    for name, sampler in (
        ("hard_positive", HardPositiveOverSampler(**common)),
        ("counterfactual", CounterfactualOverSampler(**common)),
    ):
        if name not in args.samplers:
            continue
        start = time.perf_counter()
        X_aug, _ = sampler.fit_resample(X, y)
        meta: dict[str, Any] = {}
        if name == "counterfactual":
            meta["rejected"] = [r.reason for r in sampler.generation_result_.rejected]
        record(name, X_aug[len(X) :], time.perf_counter() - start, **meta)

    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_add": n_add,
        "train": {"X": X, "y": y},
        "augmentations": augmentations,
    }
    output = args.output or _RESULTS_DIR / (
        "pn2t-downstream-aug-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote augmentations to {output}")


# ---- evaluate ---------------------------------------------------------------------


def _make_classifier(args: argparse.Namespace) -> Any:
    if args.classifier == "tfidf":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline

        return make_pipeline(
            TfidfVectorizer(),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        )
    from pntx.t2pn import FineTuningClassifier

    return FineTuningClassifier(model_name=args.model_name, class_weight="balanced", seed=args.seed)


def evaluate(args: argparse.Namespace) -> None:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    data = json.loads(Path(args.augmentations).read_text())
    X, y = data["train"]["X"], data["train"]["y"]
    raw = {k: v["texts"] for k, v in data["augmentations"].items()}
    augs, n_added, empty = equalize(raw)
    if empty:
        print(f"note: no augmentations for {empty}; these conditions are skipped")
    if any(len(v) != n_added for v in raw.values() if v):
        print(f"note: conditions truncated to {n_added} added positives (shortfall)")

    test_orig_X, test_orig_y = cad.load_original("test", args.cache_dir)
    test_pairs = cad.load_pairs("test", args.cache_dir)
    test_rev_X = [p.revised for p in test_pairs]
    test_rev_y = [p.revised_label for p in test_pairs]
    test_sets = {
        "original_test": (test_orig_X, test_orig_y),
        "revised_test": (test_rev_X, test_rev_y),
    }

    def macro_f1(t: Sequence[str], p: Sequence[str]) -> float:
        return float(f1_score(t, p, average="macro"))

    def auc(t: Sequence[str], scores: Sequence[float]) -> float:
        # Threshold-free: separates ranking quality from a shifted decision
        # threshold (low-resource TF-IDF models tend to predict one class).
        return float(roc_auc_score(np.asarray(t) == POSITIVE, scores))

    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        if condition != "original" and condition not in augs:
            continue
        extra = augs.get(condition, [])
        clf = _make_classifier(args)
        clf.fit(X + extra, y + [POSITIVE] * len(extra))
        row: dict[str, Any] = {"condition": condition, "n_train": len(X) + len(extra)}
        pos_col = list(clf.classes_).index(POSITIVE)
        for test_name, (tX, ty) in test_sets.items():
            pred = list(clf.predict(tX))
            scores = np.asarray(clf.predict_proba(tX))[:, pos_col]
            for metric_name, metric in (("acc", accuracy_score), ("f1", macro_f1)):
                point, low, high = bootstrap_ci(ty, pred, metric, seed=args.seed)
                row[f"{test_name}_{metric_name}"] = [point, low, high]
            row[f"{test_name}_auc"] = list(bootstrap_ci(ty, scores, auc, seed=args.seed))
            row[f"{test_name}_pred_pos"] = pred.count(POSITIVE) / len(pred)
        rows.append(row)
        print(f"  evaluated {condition}")

    table = format_table(rows)
    print(table)
    report = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_added_per_condition": n_added,
        "n_generated": {k: len(v) for k, v in raw.items()},
        "skipped_empty_conditions": empty,
        "generation_seconds": {k: v.get("seconds") for k, v in data["augmentations"].items()},
        "rows": rows,
        "table": table,
    }
    output = args.output or Path(args.augmentations).with_name(
        Path(args.augmentations).stem + f"-eval-{args.classifier}.json"
    )
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote results to {output}")


def format_table(rows: list[dict[str, Any]]) -> str:
    """Markdown table: point estimate with 95% CI for each test set and metric,
    plus the share of test items predicted positive (``pred_pos``)."""
    cols = [
        f"{t}_{m}"
        for t in ("original_test", "revised_test")
        for m in ("acc", "f1", "auc", "pred_pos")
    ]

    def cell(v: Any) -> str:
        if v is None:
            return "-"
        if isinstance(v, (int, float)):
            return f"{v:.2f}"
        return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"

    lines = [
        "| condition | n_train | " + " | ".join(cols) + " |",
        "|---|---|" + "---|" * len(cols),
    ]
    for r in rows:
        lines.append(
            f"| {r['condition']} | {r['n_train']} | "
            + " | ".join(cell(r.get(c)) for c in cols)
            + " |"
        )
    return "\n".join(lines)


# ---- CLI ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=None, help="CAD download cache")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="Build the training set and all augmentations")
    g.add_argument("--editor-model", required=True, help="GGUF model for the pn2t samplers")
    g.add_argument("--n-pos", type=int, default=25)
    g.add_argument("--n-neg", type=int, default=125)
    g.add_argument("--max-chars", type=int, default=1000)
    g.add_argument("--eda-alpha", type=float, default=0.05, help="Wei & Zou's small-data value")
    g.add_argument(
        "--samplers",
        nargs="*",
        default=["hard_positive", "counterfactual"],
        choices=["hard_positive", "counterfactual"],
    )
    g.add_argument("--batch-size", type=int, default=2)
    g.add_argument("--n-ctx", type=int, default=8192)
    g.add_argument("--max-tokens", type=int, default=2048)
    g.add_argument("--n-gpu-layers", type=int, default=-1)

    e = sub.add_parser("evaluate", help="Train/evaluate classifiers on saved augmentations")
    e.add_argument("--augmentations", required=True, type=Path)
    e.add_argument("--classifier", choices=["tfidf", "finetuning"], default="tfidf")
    e.add_argument("--model-name", default="bert-base-multilingual-cased")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "generate":
        generate(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
