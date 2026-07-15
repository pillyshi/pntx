# pntx

`pntx` is a Python library that turns user-supplied `positive`/`negative` text pools
into:

1. **Generation** — synthesize new text on either side.
2. **Classification** — label arbitrary text as `positive` or `negative`.

The meaning of "positive" and "negative" is entirely up to you. It doesn't have to be
sentiment — it can be formal/casual, policy-compliant/violating, or any other contrast
you define with examples. `pntx` never interprets the pools; it only uses them as
few-shot and scoring material. `positive` and `negative` are independent pools, not
aligned pairs — they don't need to be the same length or otherwise correspond to each
other (e.g. sampling straight from an existing labeled dataset works fine). Fitting just
one side is also valid, e.g. to smoke-test generation from a single example.

```python
from pntx import PNTX

model = PNTX(backend="llama", model_path="model.gguf")

model.fit(
    positive=["The movie was fantastic", "Support was quick and helpful"],
    negative=["The movie was boring", "Support was slow and unhelpful"],
)

# Generation
texts = model.generate(
    n=20,
    side="positive",
    temperature=1.0,
    dedup=True,          # filter near-duplicates (of each other and of the fitted pools)
    verify=True,         # self-classify and reject anything that doesn't match `side`
    min_confidence=0.8,  # confidence threshold used by verify
)

# Classification
result = model.classify("The staff were incredibly friendly")
result.label        # "positive" | "negative"
result.confidence   # float in [0.0, 1.0]
result == "positive"  # True

results = model.classify_batch(texts)  # batched, not a naive per-item loop
```

## Installation

`pntx` uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
uv add pntx              # core (zero dependencies)
uv add "pntx[llama]"     # + llama.cpp in-process backend
uv add "pntx[anthropic]" # + Anthropic API backend
uv add "pntx[embeddings]" # + semantic similarity for selectors
```

The core package has no runtime dependencies. Each backend/feature lives behind its
own extra, and using one without installing it raises a clear `ImportError` with the
install command to run.

## Backends

`pntx` runs models two ways:

- **`LlamaCppBackend`** (`pntx[llama]`) — runs a GGUF model in-process via
  `llama-cpp-python`. This is the primary, most-tuned backend: classification uses
  token log-probabilities directly (`score_choices`), and batched classification
  reuses the shared few-shot prefix's KV cache across every item instead of
  re-evaluating it per item.
- **`AnthropicBackend`** (`pntx[anthropic]`) — calls the Anthropic Messages API.
  Since that API doesn't expose log-probabilities, classification asks the model to
  name the label and parses it out of the response instead (confidence is then a
  fixed convention value, not a calibrated probability). Batched classification runs
  requests concurrently (`asyncio` + a semaphore), not in a sequential loop.

```python
model = PNTX(backend="llama", model_path="model.gguf")
model = PNTX(backend="anthropic", model="claude-...")

# or pass a backend instance directly, e.g. for dependency injection in tests
from pntx.backends.llama import LlamaCppBackend
model = PNTX(backend=LlamaCppBackend(model_path="model.gguf"))
```

`LlamaCppBackend` accepts either a local `model_path` or a `repo_id` (optionally
narrowed to one file with `filename`) to pull a GGUF model from the Hugging Face Hub
via `Llama.from_pretrained`. Any other keyword — `n_ctx`, `n_gpu_layers`,
`flash_attn`, `verbose`, ... — is forwarded as-is to `llama_cpp.Llama`:

```python
model = PNTX(
    backend="llama",
    repo_id="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    filename="*q4_k_m.gguf",
    n_ctx=4096,
    n_gpu_layers=-1,   # offload all layers to GPU
    flash_attn=True,
)
```

## Selecting exemplars

When there are more fitted texts (on either side) than comfortably fit in a prompt, a
`Selector` decides which ones to use — it's called independently for the `positive` and
`negative` pools:

- **`RandomSelector`** (default) — a uniform random subset.
- **`NearestSelector`** — picks texts most similar to the text being classified;
  dynamic, per-query selection.
- **`DiversitySelector`** — greedily picks a maximally diverse subset.

Both `NearestSelector` and `DiversitySelector` take a `similarity_fn`. It defaults to
a dependency-free character n-gram similarity (`pntx.dedup.similarity`); pass
`pntx.embeddings.cosine_similarity_fn()` (requires `pntx[embeddings]`) for semantic
similarity instead:

```python
from pntx import PNTX
from pntx.selection import NearestSelector

model = PNTX(backend="llama", model_path="model.gguf", selector=NearestSelector())
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
ANTHROPIC_API_KEY=... uv run pytest tests/integration/test_anthropic_backend.py
```
