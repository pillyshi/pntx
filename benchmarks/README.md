# pntx benchmarks

Not part of the published `pntx` package -- these are repo-internal scripts
for measuring pntx's own performance, kept out of `src/pntx` and out of the
`pyproject.toml` build so they never ship to users.

## Dataset

[`google/civil_comments`](https://huggingface.co/datasets/google/civil_comments)
(CC0 license), loaded via `benchmarks/jigsaw.py`. Each comment has a
continuous `toxicity` score in `[0, 1]`; we threshold it into pntx's binary
`Label`:

- `toxicity <= clean_threshold` (default `0.1`) -> `"positive"` (non-toxic)
- `toxicity >= toxic_threshold` (default `0.5`) -> `"negative"` (toxic)
- anything in between is dropped as ambiguous

The positive/negative assignment is otherwise arbitrary (pntx treats the
axis as opaque) -- "non-toxic = positive" is just the convention fixed here.

## Setup

```
uv sync --extra llama --group benchmark
```

## t2pn (classification) benchmark

Fits `PNTX` on sampled Jigsaw pairs, runs `classify_batch` over a balanced
held-out eval set, and reports accuracy/precision/recall/F1 plus latency.
llama.cpp only (the only backend `pntx` currently ships). Not run in CI --
it downloads the full dataset and needs a local gguf model, so run it
manually, e.g. on a GPU server:

```
uv run python -m benchmarks.t2pn.run \
    --model-path /path/to/model.gguf \
    --n-gpu-layers -1 \
    --selector random \
    --n-pairs 50 \
    --n-eval 200
```

Run `uv run python -m benchmarks.t2pn.run --help` for all options
(`--selector {random,nearest,diversity}`, thresholds, seed, `--cache-dir`
for the HF datasets cache, `--output` for the result JSON path). Results are
written to `benchmarks/results/` (git-ignored; not checked in).

## Planned: pn2t (generation) benchmark

Not implemented yet. The plan is to evaluate `generate()` as a data
augmentor: generate synthetic positive/negative texts, train a separate
downstream classifier on real + synthetic data, and compare against a
real-data-only baseline on the same held-out Jigsaw eval set.
