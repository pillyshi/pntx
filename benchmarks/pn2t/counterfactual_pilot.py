"""Pilot benchmark for choosing ``CounterfactualOverSampler``'s ``verify`` default.

See ``research/ideas/counterfactual-edit-sampler.md`` ("Choosing the
``verify`` default"). On negative IMDb reviews from Kaushik et al.'s CAD
release (``benchmarks/cad.py``), this:

1. generates counterfactual edits once with ``verify="none"`` (the editor's
   own ``is_positive`` is recorded, which is all ``verify="self"`` uses);
2. judges every resulting candidate with a cross-fitted
   ``LLMPromptingClassifier`` verifier on the *same* loaded editor model --
   through the sampler's own cross-fitting code, so it is exactly what
   ``verify=<classifier>`` would do;
3. labels every candidate with an independent judge model (zero-shot
   log-prob scoring), as a proxy for the human judgment the paper used. The
   judge is also run on the pivots and their human revisions, so its
   agreement with the human labels is reported alongside;
4. scores "hardness" with a TF-IDF + logistic regression classifier trained
   on the original (unrevised) CAD training reviews only.

Since a verifier only filters candidates, one generation run is enough to
compare all three strategies on identical candidates (``pilot_metrics``).
In real use a rejected edit triggers a retry; the pilot measures the filter,
not the retry loop.

Not part of CI. Needs ``--extra llama`` and two local GGUF models, e.g.:

    uv run --extra llama python -m benchmarks.pn2t.counterfactual_pilot \\
        --editor-model /path/to/editor.gguf --judge-model /path/to/judge.gguf
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from benchmarks import cad
from benchmarks.pn2t.pilot_metrics import STRATEGIES, Candidate, format_table, summarize
from pntx.backends.llama import LlamaCppBackend
from pntx.pn2t import CounterfactualOverSampler
from pntx.t2pn import LLMPromptingClassifier
from pntx.types import NEGATIVE, POSITIVE

_RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

JUDGE_PROMPT = (
    "Read the movie review and decide whether its overall sentiment is positive "
    "or negative.\n\nReview:\n{text}\n\nSentiment (positive or negative):"
)
JUDGE_CHOICES = [" positive", " negative"]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--editor-model", required=True, help="GGUF model that writes the edits")
    parser.add_argument("--judge-model", required=True, help="GGUF model used as reference judge")
    parser.add_argument("--n-pivots", type=int, default=40, help="Negative reviews to edit")
    parser.add_argument(
        "--n-references", type=int, default=20, help="Positive reference/fit reviews"
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=1000,
        help="Only use reviews up to this length (Kaushik et al. also dropped the longest)",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--verify-cv", type=int, default=5)
    parser.add_argument("--n-ctx", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cache-dir", type=Path, default=None, help="CAD download cache")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def _load_backend(model_path: str, args: argparse.Namespace) -> LlamaCppBackend:
    return LlamaCppBackend(
        model_path=model_path, n_ctx=args.n_ctx, n_gpu_layers=args.n_gpu_layers, verbose=False
    )


def judge(backend: LlamaCppBackend, texts: list[str]) -> list[bool]:
    """Zero-shot sentiment by comparing the log-probs of " positive"/" negative"."""
    verdicts = []
    for text in texts:
        pos, neg = backend.score_choices(JUDGE_PROMPT.format(text=text), JUDGE_CHOICES)
        verdicts.append(pos > neg)
    return verdicts


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    rng = random.Random(args.seed)

    # ---- data -----------------------------------------------------------------
    pairs = [
        p
        for p in cad.load_pairs("test", args.cache_dir)
        if p.original_label == NEGATIVE and len(p.original) <= args.max_chars
    ]
    pivots = rng.sample(pairs, args.n_pivots)
    X_train, y_train = cad.load_original("train", args.cache_dir)
    train_positives = [x for x, label in zip(X_train, y_train, strict=True) if label == POSITIVE]
    references = rng.sample(
        [x for x in train_positives if len(x) <= args.max_chars], args.n_references
    )

    X = references + [p.original for p in pivots]
    y = [POSITIVE] * len(references) + [NEGATIVE] * len(pivots)
    print(f"{len(pivots)} negative pivots, {len(references)} positive references")

    # ---- 1. generate with verify="none" -----------------------------------------
    editor = _load_backend(args.editor_model, args)
    sampler = CounterfactualOverSampler(
        backend=editor,
        verify="none",
        sampling_strategy={POSITIVE: len(references) + len(pivots)},
        batch_size=args.batch_size,
        context_limit=args.n_ctx,
        max_tokens=args.max_tokens,
        temperature=0.7,
        random_state=args.seed,
    )
    start = time.perf_counter()
    sampler.fit_resample(X, y)
    generation_seconds = time.perf_counter() - start
    edits = sampler.generation_result_.edits
    filter_rejections = Counter(r.reason for r in sampler.generation_result_.rejected)
    print(f"generated {len(edits)} candidates in {generation_seconds:.0f}s; {filter_rejections}")

    # ---- 2. cross-fitted classifier verifier (same code path as verify=<clf>) ----
    verifier_sampler = CounterfactualOverSampler(
        backend=editor,
        verify=LLMPromptingClassifier(backend=editor, context_limit=args.n_ctx),
        verify_cv=args.verify_cv,
        random_state=args.seed,
    )
    verifier_sampler._validate_params()
    verifier_sampler._prepare_fit(
        X, y, positive_label=POSITIVE, negative_label=NEGATIVE, random_state=args.seed
    )
    start = time.perf_counter()
    verdicts = verifier_sampler._verify_candidates(edits)
    verify_seconds = time.perf_counter() - start
    print(f"verified {len(edits)} candidates in {verify_seconds:.0f}s")

    del sampler, verifier_sampler, editor
    gc.collect()

    # ---- 3. independent judge ------------------------------------------------------
    judge_backend = _load_backend(args.judge_model, args)
    start = time.perf_counter()
    judge_edits = judge(judge_backend, [e.text for e in edits])
    judge_originals = judge(judge_backend, [p.original for p in pivots])
    judge_revisions = judge(judge_backend, [p.revised for p in pivots])
    judge_seconds = time.perf_counter() - start
    del judge_backend
    gc.collect()
    # Pivots are human-labelled negative, their revisions positive.
    judge_human_agreement = (sum(not v for v in judge_originals) + sum(judge_revisions)) / (
        2 * len(pivots)
    )
    print(f"judge agreement with human labels on CAD: {judge_human_agreement:.2f}")

    # ---- 4. shallow downstream classifier on original data only -----------------
    downstream = make_pipeline(TfidfVectorizer(min_df=2), LogisticRegression(max_iter=1000))
    downstream.fit(X_train, y_train)
    downstream_edits = [p == POSITIVE for p in downstream.predict([e.text for e in edits])]
    downstream_human = [p == POSITIVE for p in downstream.predict([p.revised for p in pivots])]

    # ---- metrics ------------------------------------------------------------------
    candidates = [
        Candidate(
            source_text=e.source_text,
            text=e.text,
            edit_ratio=e.edit_ratio,
            self_positive=e.self_assessed_positive,
            verifier_positive=verdict is None,
            judge_positive=judged,
            downstream_positive=easy,
        )
        for e, verdict, judged, easy in zip(
            edits, verdicts, judge_edits, downstream_edits, strict=True
        )
    ]
    rows = [summarize(candidates, s) for s in STRATEGIES]
    human_reference = {
        "strategy": "human revisions (reference)",
        "n_kept": len(pivots),
        "precision": sum(judge_revisions) / len(pivots),
        "downstream_easy": sum(downstream_human) / len(pivots),
        "valid_and_hard": sum(
            j and not e for j, e in zip(judge_revisions, downstream_human, strict=True)
        ),
    }
    table = format_table([*rows, human_reference])
    print(table)

    report: dict[str, Any] = {
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "n_pivots": len(pivots),
        "n_candidates": len(candidates),
        "filter_rejections": dict(filter_rejections),
        "judge_human_agreement": judge_human_agreement,
        "edit_ratio": {
            "llm_median": sorted(c.edit_ratio for c in candidates)[len(candidates) // 2]
            if candidates
            else None,
        },
        "timing_seconds": {
            "generation": generation_seconds,
            "classifier_verification": verify_seconds,
            "judge": judge_seconds,
        },
        "summaries": rows,
        "human_reference": human_reference,
        "candidates": [vars(c) for c in candidates],
        "table": table,
    }
    output = args.output or _RESULTS_DIR / (
        "pn2t-counterfactual-pilot-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + f"-{Path(args.editor_model).name[:16]}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Wrote results to {output}")


if __name__ == "__main__":
    main()
