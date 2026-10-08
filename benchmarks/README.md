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

## pn2t: `CounterfactualOverSampler` `verify` pilot

`benchmarks/pn2t/counterfactual_pilot.py` measures which label-verification
strategy (`verify="none"`, `"self"`, or a cross-fitted classifier) should be
`CounterfactualOverSampler`'s default. The design and results are recorded in
`research/ideas/counterfactual-edit-sampler.md`.

**Dataset:** the counterfactually-augmented IMDb data of Kaushik et al. (2020)
([`acmi-lab/counterfactually-augmented-data`](https://github.com/acmi-lab/counterfactually-augmented-data),
Apache-2.0), loaded by `benchmarks/cad.py`. Files are downloaded with the
standard library and cached in `~/.cache/pntx-benchmarks/cad`, so no `datasets`
dependency is needed. Negative test-split reviews are used as pivots, with
their human revisions as a reference. The original training reviews provide
positive references and the shallow downstream classifier.

**What it does:** it generates edits once with `verify="none"`, then scores
every candidate in four ways:

- the editor's own `is_positive`, which is what `"self"` uses;
- a cross-fitted `LLMPromptingClassifier` on the same loaded model, run
  through the sampler's own cross-fitting code;
- a judge model from a *different* family, used as a reference label (its
  agreement with the human labels is reported too);
- a TF-IDF + logistic regression model trained on the original data only,
  which measures how "easy" kept edits are.

Each strategy then gets precision, catch rate, false-reject rate, yield and
downstream-easy share (`benchmarks/pn2t/pilot_metrics.py`).

```
uv run --extra llama python -m benchmarks.pn2t.counterfactual_pilot \
    --editor-model /path/to/editor.gguf --judge-model /path/to/judge.gguf \
    --n-pivots 40 --n-references 20
```

Like the t2pn benchmark, this is not run in CI. Results go to
`benchmarks/results/`, which is git-ignored.
