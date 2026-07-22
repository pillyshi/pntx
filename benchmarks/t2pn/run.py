"""t2pn (text -> positive/negative) classification benchmark, llama.cpp only.

Fits a t2pn.Classifier on sampled Jigsaw/Civil Comments positive/negative
pools, runs predict/predict_proba over a held-out eval set, and reports
accuracy/F1/latency.

Not run as part of CI or by this repo's own test suite -- it downloads the
full Civil Comments dataset and needs a local gguf model, so it's meant to
be run manually (e.g. on a GPU server), as a module from the repo root:

    uv run python -m benchmarks.t2pn.run --model-path /path/to/model.gguf \\
        --n-gpu-layers -1 --selector random --n-per-side 50 --n-eval 200
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import jigsaw, metrics
from pntx.selection import DiversitySelector, NearestSelector, RandomSelector, Selector
from pntx.t2pn import Classifier

_SELECTORS: dict[str, type[Selector]] = {
    "random": RandomSelector,
    "nearest": NearestSelector,
    "diversity": DiversitySelector,
}

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_selector(name: str) -> Selector:
    return _SELECTORS[name]()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Path to a llama.cpp gguf model")
    parser.add_argument(
        "--n-gpu-layers", type=int, default=-1, help="Layers to offload to GPU (-1 = all)"
    )
    parser.add_argument("--n-ctx", type=int, default=4096)
    parser.add_argument("--selector", choices=sorted(_SELECTORS), default="random")
    parser.add_argument(
        "--n-per-side", type=int, default=50, help="Positive/negative texts to fit() per side"
    )
    parser.add_argument(
        "--max-exemplars",
        type=int,
        default=None,
        help="Defaults to --n-per-side (use all fitted texts)",
    )
    parser.add_argument("--n-eval", type=int, default=200, help="Held-out eval texts (balanced)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--clean-threshold", type=float, default=jigsaw.DEFAULT_CLEAN_THRESHOLD)
    parser.add_argument("--toxic-threshold", type=float, default=jigsaw.DEFAULT_TOXIC_THRESHOLD)
    parser.add_argument("--cache-dir", default=None, help="Override the HF datasets cache dir")
    parser.add_argument("--output", default=None, help="Where to write the JSON result")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    max_exemplars = args.max_exemplars if args.max_exemplars is not None else args.n_per_side

    print(f"Loading {jigsaw.DATASET_NAME} (cache_dir={args.cache_dir!r})...")
    dataset = jigsaw.load_dataset(cache_dir=args.cache_dir)

    positive, negative = jigsaw.sample_pools(
        dataset,
        args.seed,
        args.n_per_side,
        clean_threshold=args.clean_threshold,
        toxic_threshold=args.toxic_threshold,
    )
    eval_set = jigsaw.sample_eval_set(
        dataset,
        args.seed,
        args.n_per_side,
        args.n_eval,
        clean_threshold=args.clean_threshold,
        toxic_threshold=args.toxic_threshold,
    )
    print(
        f"Sampled {len(positive)} positive + {len(negative)} negative exemplars "
        f"and {len(eval_set)} eval texts."
    )

    clf = Classifier(
        backend="llama",
        backend_kwargs={
            "model_path": args.model_path,
            "n_gpu_layers": args.n_gpu_layers,
            "n_ctx": args.n_ctx,
        },
        selector=build_selector(args.selector),
        max_exemplars=max_exemplars,
    )
    X_fit = positive + negative
    y_fit = ["positive"] * len(positive) + ["negative"] * len(negative)
    clf.fit(X_fit, y_fit)

    eval_texts = [text for text, _ in eval_set]
    eval_labels = [label for _, label in eval_set]

    start = time.perf_counter()
    proba = clf.predict_proba(eval_texts)
    elapsed = time.perf_counter() - start

    pred_indices = np.argmax(proba, axis=1)
    y_pred = [clf.classes_[i] for i in pred_indices]
    y_proba = [proba[i, pred_indices[i]] for i in range(len(eval_texts))]

    computed = metrics.compute_metrics(eval_labels, y_pred, y_proba)
    report = _build_report(
        args, max_exemplars, len(positive), len(negative), len(eval_set), elapsed, computed
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))

    output_path = Path(args.output) if args.output else _default_output_path(args.selector)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote results to {output_path}")


def _default_output_path(selector_name: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return _RESULTS_DIR / f"t2pn-{timestamp}-{selector_name}.json"


def _build_report(
    args: argparse.Namespace,
    max_exemplars: int,
    n_positive: int,
    n_negative: int,
    n_eval: int,
    elapsed_seconds: float,
    computed: metrics.ClassificationMetrics,
) -> dict[str, Any]:
    return {
        "config": {
            "backend": "llama",
            "model_path": args.model_path,
            "n_gpu_layers": args.n_gpu_layers,
            "n_ctx": args.n_ctx,
            "selector": args.selector,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "max_exemplars": max_exemplars,
            "n_eval": n_eval,
            "seed": args.seed,
            "clean_threshold": args.clean_threshold,
            "toxic_threshold": args.toxic_threshold,
        },
        "metrics": {
            "accuracy": computed.accuracy,
            "precision": computed.precision,
            "recall": computed.recall,
            "f1": computed.f1,
            "confusion": {f"{t}->{p}": count for (t, p), count in computed.confusion.items()},
            "mean_confidence": computed.mean_confidence,
        },
        "latency": {
            "total_seconds": elapsed_seconds,
            "seconds_per_item": elapsed_seconds / n_eval if n_eval else 0.0,
        },
    }


if __name__ == "__main__":
    main()
