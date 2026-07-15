"""t2pn (text -> positive/negative) classification benchmark, llama.cpp only.

Fits a PNTX model on sampled Jigsaw/Civil Comments pairs, runs
``classify_batch`` over a held-out eval set, and reports accuracy/F1/latency.

Not run as part of CI or by this repo's own test suite -- it downloads the
full Civil Comments dataset and needs a local gguf model, so it's meant to
be run manually (e.g. on a GPU server), as a module from the repo root:

    uv run python -m benchmarks.t2pn.run --model-path /path/to/model.gguf \\
        --n-gpu-layers -1 --selector random --n-pairs 50 --n-eval 200
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks import jigsaw, metrics
from pntx import PNTX
from pntx.selection import DiversitySelector, NearestSelector, RandomSelector, Selector

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
    parser.add_argument("--n-pairs", type=int, default=50, help="Pairs to fit()")
    parser.add_argument(
        "--max-exemplars",
        type=int,
        default=None,
        help="Defaults to --n-pairs (use all fitted pairs)",
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
    max_exemplars = args.max_exemplars if args.max_exemplars is not None else args.n_pairs

    print(f"Loading {jigsaw.DATASET_NAME} (cache_dir={args.cache_dir!r})...")
    dataset = jigsaw.load_dataset(cache_dir=args.cache_dir)

    pairs = jigsaw.sample_pairs(
        dataset,
        args.seed,
        args.n_pairs,
        clean_threshold=args.clean_threshold,
        toxic_threshold=args.toxic_threshold,
    )
    eval_set = jigsaw.sample_eval_set(
        dataset,
        args.seed,
        args.n_pairs,
        args.n_eval,
        clean_threshold=args.clean_threshold,
        toxic_threshold=args.toxic_threshold,
    )
    print(f"Sampled {len(pairs)} exemplar pairs and {len(eval_set)} eval texts.")

    model = PNTX(
        backend="llama",
        model_path=args.model_path,
        n_gpu_layers=args.n_gpu_layers,
        n_ctx=args.n_ctx,
        selector=build_selector(args.selector),
        max_exemplars=max_exemplars,
    )
    model.fit(pairs)

    eval_texts = [text for text, _ in eval_set]
    eval_labels = [label for _, label in eval_set]

    start = time.perf_counter()
    results = model.classify_batch(eval_texts)
    elapsed = time.perf_counter() - start

    computed = metrics.compute_metrics(eval_labels, results)
    report = _build_report(args, max_exemplars, len(pairs), len(eval_set), elapsed, computed)

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
    n_pairs: int,
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
            "n_pairs": n_pairs,
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
