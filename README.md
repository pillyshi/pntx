# pntx

`pntx` is a Python library that turns user-supplied `positive`/`negative` text pools
into two independent components:

1. **`pntx.t2pn.Classifier`** (text → positive/negative) — a
   [scikit-learn](https://scikit-learn.org/) `Classifier`: label arbitrary text as
   `positive` or `negative`.
2. **`pntx.pn2t`** (positive/negative → text) — two
   [imbalanced-learn](https://imbalanced-learn.org/)-style oversamplers with different
   goals: **`OverSampler`** generates "hard positive" text to balance an imbalanced
   dataset for classifier training, and **`SyntheticSampler`** generates anonymized,
   representative synthetic positives for publishing data you can't share as-is.

The meaning of "positive" and "negative" is entirely up to you. It doesn't have to be
sentiment — it can be formal/casual, policy-compliant/violating, or any other contrast
you define with examples. `pntx` never interprets the pools; it only uses them as
few-shot and scoring material.

```python
from pntx.t2pn import Classifier
from pntx.pn2t import OverSampler

# --- t2pn: classification (a scikit-learn Classifier) ---
clf = Classifier(backend="llama", backend_kwargs={"model_path": "model.gguf"})

X = ["The movie was fantastic", "Support was quick and helpful",
     "The movie was boring", "Support was slow and unhelpful"]
y = ["positive", "positive", "negative", "negative"]  # 0/1 works too

clf.fit(X, y)
clf.predict(["The staff were incredibly friendly"])        # array(['positive'], dtype='<U8')
clf.predict_proba(["The staff were incredibly friendly"])  # shape (1, 2), columns follow clf.classes_

# drops straight into the scikit-learn ecosystem
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X, y, cv=5)

# --- pn2t: generation (an imbalanced-learn-style OverSampler) ---
sampler = OverSampler(backend="llama", backend_kwargs={"model_path": "model.gguf"})

X_aug, y_aug = sampler.fit_resample(X, [1, 1, 0, 0])  # binary labels only; positive class = 1
sampler.generation_result_.hard_positives  # generated texts + the LLM's rationale for each

# --- pn2t: anonymized synthetic data generation ---
from pntx.pn2t import SyntheticSampler

synth = SyntheticSampler(
    backend="llama", backend_kwargs={"model_path": "model.gguf"}, n_synthesized=10
)
X_syn, y_syn = synth.fit_resample(X, [1, 1, 0, 0])
synth.generation_result_.synthetic_texts  # generated texts + what was generalized away for each
```

`OverSampler.fit_resample` generates "hard positives" — texts an expert would label
positive but that shallow classifiers or untrained humans might mislabel negative — by
first asking the backend to analyze what distinguishes the two classes. It's a full
port of [`semaxis`](https://github.com/pillyshi/semaxis)'s `HardPositiveOverSampler`,
routed through `pntx`'s own `Backend` abstraction so it can share a loaded model with
`Classifier` instead of loading its own. v1 only generates the positive side and
supports binary `{0, 1}` labels; `imbalanced-learn` itself isn't required (`fit_resample`
is duck-typed, so `imblearn.pipeline.Pipeline` still works if it's installed
separately).

`SyntheticSampler.fit_resample` has a different goal: instead of hard positives for
classifier augmentation, it generates *typical* positive-class texts with specific
identifying details (names, exact dates/numbers, locations, verbatim phrases) generalized
away, so the result is safe to publish even when the original pool isn't. The negative
pool is still required (for the same binary-label validation as `OverSampler`), but it's
never shown to the backend — only positive exemplars inform generation, since contrasting
against negatives would frame generation around the boundary rather than the typical
case. Anonymity is best-effort: besides the prompt instructions, a lightweight verbatim-
substring check (`min_verbatim_span`, default 20 characters) rejects and retries any
generated text that copies a long span straight out of a positive exemplar — this catches
copy-through leaks but not paraphrased ones, so it's not a privacy guarantee.

## Installation

`pntx` uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
uv add pntx               # core (scikit-learn + pydantic)
uv add "pntx[llama]"      # + llama.cpp in-process backend
uv add "pntx[embeddings]" # + semantic similarity for selectors
```

`scikit-learn` and `pydantic` are core dependencies (`Classifier`'s scikit-learn
contract and `OverSampler`'s structured LLM output need them respectively). Each
backend/feature otherwise lives behind its own extra, and using one without installing
it raises a clear `ImportError` with the install command to run.

## Backends

`pntx` runs models via a `Backend` protocol, shared by `Classifier`, `OverSampler`, and
`SyntheticSampler`:

- **`LlamaCppBackend`** (`pntx[llama]`) — runs a GGUF model in-process via
  `llama-cpp-python`. This is the primary, most-tuned backend: classification uses
  token log-probabilities directly (`score_choices`), and batched classification
  reuses the shared few-shot prefix's KV cache across every item instead of
  re-evaluating it per item.

```python
clf = Classifier(backend="llama", backend_kwargs={"model_path": "model.gguf"})

# or pass a backend instance directly, e.g. for dependency injection in tests
from pntx.backends.llama import LlamaCppBackend
clf = Classifier(backend=LlamaCppBackend(model_path="model.gguf"))
```

A remote API backend can be added later by implementing the `Backend` protocol
(`pntx.backends.base.Backend`) and passing an instance directly — no built-in one
ships right now.

`backend_kwargs` is only used when `backend` is given as a string; it's a single dict
(rather than `**kwargs`) so `Classifier`/`OverSampler`/`SyntheticSampler` stay compatible
with scikit-learn's `get_params()`/`clone()`.

`LlamaCppBackend` accepts either a local `model_path` or a `repo_id` (optionally
narrowed to one file with `filename`) to pull a GGUF model from the Hugging Face Hub
via `Llama.from_pretrained`. Any other keyword — `n_ctx`, `n_gpu_layers`,
`flash_attn`, `verbose`, ... — is forwarded as-is to `llama_cpp.Llama`:

```python
clf = Classifier(
    backend="llama",
    backend_kwargs={
        "repo_id": "Qwen/Qwen2.5-1.5B-Instruct-GGUF",
        "filename": "*q4_k_m.gguf",
        "n_ctx": 4096,
        "n_gpu_layers": -1,  # offload all layers to GPU
        "flash_attn": True,
    },
)
```

To share one loaded model across `Classifier`, `OverSampler`, and `SyntheticSampler`
(recommended for local inference — avoids loading the same GGUF twice), construct the
backend once and pass the instance to each:

```python
from pntx.backends.llama import LlamaCppBackend

backend = LlamaCppBackend(model_path="model.gguf")
clf = Classifier(backend=backend)
sampler = OverSampler(backend=backend)
synth = SyntheticSampler(backend=backend, n_synthesized=10)
```

## Selecting exemplars

When there are more fitted texts (on either side) than comfortably fit in a prompt, a
`Selector` decides which ones to use — `Classifier` calls it independently for the
positive and negative pools:

- **`RandomSelector`** (default) — a uniform random subset.
- **`NearestSelector`** — picks texts most similar to the text being classified;
  dynamic, per-query selection.
- **`DiversitySelector`** — greedily picks a maximally diverse subset.
- **`BudgetSelector`** — picks as many texts as fit within a token budget (used
  internally by `OverSampler` for its exemplar sampling).

`NearestSelector` and `DiversitySelector` take a `similarity_fn`. It defaults to a
dependency-free character n-gram similarity (`pntx.dedup.similarity`); pass
`pntx.embeddings.cosine_similarity_fn()` (requires `pntx[embeddings]`) for semantic
similarity instead:

```python
from pntx.t2pn import Classifier
from pntx.selection import NearestSelector

clf = Classifier(backend="llama", backend_kwargs={"model_path": "model.gguf"}, selector=NearestSelector())
```

`OverSampler` and `SyntheticSampler` don't take a `Selector`; instead their
`sample_method` constructor argument picks a *budget-based* sampling strategy (a full
port of semaxis's own `sample_method`/`embedding_model` for `OverSampler`;
`SyntheticSampler` reuses the same mechanism for its positive-only exemplar sampling):

- **`"random"`** (default) — a uniform random subset, filled until the token budget
  runs out (`BudgetSelector` under the hood).
- **`"kmeans"`** — embeds the pool via `embedding_model` (requires
  `pntx[embeddings]`) and picks one representative text per K-Means cluster.
- **`"votek"`** — embeds the pool and runs the Vote-K algorithm (Su et al. 2022),
  balancing representativeness and diversity.

```python
sampler = OverSampler(
    backend="llama",
    backend_kwargs={"model_path": "model.gguf"},
    sample_method="votek",
    embedding_model="paraphrase-albert-small-v2",  # sentence-transformers model name
)
```

## Development

```bash
uv sync                          # install dev dependencies
uv run pytest                    # unit tests (integration tests are skipped by default)
uv run ruff check .
uv run mypy src tests
```

Integration tests that hit a real model or API are opt-in:

```bash
PNTX_LLAMA_MODEL_PATH=/path/to/model.gguf uv run pytest tests/integration
```
