# pntx

`pntx` is a Python library that turns user-supplied `positive`/`negative` text pools
into two independent components:

1. **`pntx.t2pn`** (text → positive/negative) — a family of
   [scikit-learn](https://scikit-learn.org/) `Classifier`s that label arbitrary text as
   `positive` or `negative`: **`LLMPromptingClassifier`** classifies via LLM few-shot
   prompting/scoring (no training), and **`FineTuningClassifier`** actually fine-tunes a
   pretrained `transformers` encoder (default: multilingual BERT).
2. **`pntx.pn2t`** (positive/negative → text) — a family of
   [imbalanced-learn](https://imbalanced-learn.org/)-style over-samplers with different
   goals: **`HardPositiveOverSampler`** generates "hard positive" text to balance an
   imbalanced dataset for classifier training, **`CounterfactualOverSampler`** minimally
   edits existing negatives into positives (counterfactually-augmented data), and
   **`TypicalPositiveOverSampler`** generates typical, representative positives with
   specific details generalized away, for publishing a stand-in for data you can't
   share as-is.

The meaning of "positive" and "negative" is entirely up to you. It doesn't have to be
sentiment — it can be formal/casual, policy-compliant/violating, or any other contrast
you define with examples. `pntx` never interprets the pools; it only uses them as
few-shot and scoring material.

```python
from pntx.t2pn import LLMPromptingClassifier
from pntx.pn2t import HardPositiveOverSampler

# --- t2pn: classification via LLM few-shot prompting (no training) ---
clf = LLMPromptingClassifier(backend="llama", backend_kwargs={"model_path": "model.gguf"})

X = ["The movie was fantastic", "Support was quick and helpful",
     "The movie was boring", "Support was slow and unhelpful"]
y = ["positive", "positive", "negative", "negative"]  # 0/1 works too

clf.fit(X, y)
clf.predict(["The staff were incredibly friendly"])        # array(['positive'], dtype='<U8')
clf.predict_proba(["The staff were incredibly friendly"])  # shape (1, 2), columns follow clf.classes_

# drops straight into the scikit-learn ecosystem
from sklearn.model_selection import cross_val_score
cross_val_score(clf, X, y, cv=5)

# persist the fitted pools (not the backend -- pass a fresh one back in on load)
clf.save("classifier.json")
loaded = LLMPromptingClassifier.load("classifier.json", backend="llama",
                                     backend_kwargs={"model_path": "model.gguf"})

# --- t2pn: classification via fine-tuning a pretrained encoder (pntx[finetuning]) ---
from pntx.t2pn import FineTuningClassifier

ft_clf = FineTuningClassifier(class_weight="balanced")  # default model_name is multilingual BERT
ft_clf.fit(X, y)                 # this one actually trains
ft_clf.predict_proba(["The staff were incredibly friendly"])
ft_clf.save("finetuned/")        # persists the trained weights, not just pooled text
loaded_ft = FineTuningClassifier.load("finetuned/")

# --- pn2t: hard-positive generation (an imbalanced-learn-style over-sampler) ---
sampler = HardPositiveOverSampler(
    backend="llama",
    backend_kwargs={"model_path": "model.gguf"},
    sampling_strategy="auto",  # default: generate positives until classes balance
    random_state=0,
)

X_aug, y_aug = sampler.fit_resample(X, [1, 1, 0, 0])  # binary labels only; positive class = 1
sampler.generation_result_.hard_positives  # generated texts + the LLM's rationale for each

# --- pn2t: typical-positive generation for publishable stand-in data ---
from pntx.pn2t import TypicalPositiveOverSampler

synth = TypicalPositiveOverSampler(
    backend="llama",
    backend_kwargs={"model_path": "model.gguf"},
    sampling_strategy={1: 2 + 10},  # required: 2 existing positives + 10 new ones
)
X_syn, y_syn = synth.fit_resample(X, [1, 1, 0, 0])
synth.generation_result_.synthetic_texts  # generated texts + what was generalized away for each

# --- pn2t: counterfactual minimal edits of negatives ---
from pntx.pn2t import CounterfactualOverSampler

cf = CounterfactualOverSampler(
    backend="llama",
    backend_kwargs={"model_path": "model.gguf"},
    verify="self",  # required: "none", "self", or a classifier (see below)
)
X_cf, y_cf = cf.fit_resample(X, [1, 1, 0, 0])
cf.generation_result_.edits     # (source_index, source_text) -> text, with changed spans
cf.generation_result_.rejected  # rejected candidates and why
```

`HardPositiveOverSampler.fit_resample` generates "hard positives" — texts an expert would label
positive but that shallow classifiers or untrained humans might mislabel negative — by
first asking the backend to analyze what distinguishes the two classes. It's a full
port of [`semaxis`](https://github.com/pillyshi/semaxis)'s class of the same name,
routed through `pntx`'s own `Backend` abstraction so it can share a loaded model with
`LLMPromptingClassifier` instead of loading its own. v1 only generates the positive side and
supports binary labels only (`0`/`1`, `-1`/`1`, `"positive"`/`"negative"`, or any other pair
with `pos_label`); `imbalanced-learn` itself isn't required (`fit_resample`
is duck-typed, so `imblearn.pipeline.Pipeline` still works if it's installed
separately). Note that hard positives are not contrast sets or counterfactual edits
(minimal edits of an existing example that flip its label): each one is a new example
whose label stays positive, and "hard" is the LLM's judgment, not verified against a
classifier.

`sampling_strategy` follows imbalanced-learn's semantics, restricted to generating the
positive class: `"auto"` (the `HardPositiveOverSampler` default) generates until
positives match negatives; a float in `(0, 1]` is the desired positive/negative ratio
after resampling; a dict `{pos_label: n}` is the desired *total* number of positives
after resampling; a callable `f(y)` returning such a dict also works. A strategy that
would require generating negatives raises `ValueError`. `random_state` (an int or a
`numpy.random.RandomState`) seeds exemplar sampling.

`TypicalPositiveOverSampler.fit_resample` has a different goal: instead of hard
positives for classifier augmentation, it generates *typical* positive-class texts with
specific identifying details (names, exact dates/numbers, locations, verbatim phrases)
generalized away, as a stand-in you can publish when the original pool can't be shared.
It has no default `sampling_strategy`, since there's no natural target size for such a
set. The negative pool is still required (for the same binary-label validation as
`HardPositiveOverSampler`), but it's
never shown to the backend — only positive exemplars inform generation, since contrasting
against negatives would frame generation around the boundary rather than the typical
case. Removing identifying details is best-effort: besides the prompt instructions, a lightweight verbatim-
substring check (`min_verbatim_span`, default 20 characters) rejects and retries any
generated text that copies a long span straight out of a positive exemplar — this catches
copy-through leaks but not paraphrased ones, so it's not a privacy guarantee. In
particular, it is **not differentially private**: positive exemplars go into the prompt
verbatim. If you need a formal guarantee, use a DP method that keeps private text out of
the prompt, such as [Aug-PE](https://arxiv.org/abs/2403.01749) (Xie et al., ICML 2024) or
DP fine-tuning of the generator ([Yue et al., ACL 2023](https://aclanthology.org/2023.acl-long.74/)).

`CounterfactualOverSampler.fit_resample` builds *pairs*: each generated positive is the
smallest edit of one existing negative (its "pivot") that makes the positive label apply,
following the counterfactual-revision instructions of
[Kaushik et al. (ICLR 2020)](https://arxiv.org/abs/1909.12434) — the label must flip,
the text must stay coherent, nothing else may change. Because the pivot is already in
`X`, training on the result sees both sides of each minimal change, the mechanism that
paper shows makes classifiers less reliant on spurious features. Every candidate must
pass a format check, a no-op check, and a minimality check (`max_edit_ratio`, the
character-level share of the pivot that changed, default `0.5`), then exact-match dedup,
then label verification:

- `verify="none"` trusts the edit; `verify="self"` drops edits the LLM itself marks as
  not positive in the same response (no extra calls, but the editor grades its own
  work — small models do this poorly).
- `verify=<classifier>` (e.g. `LLMPromptingClassifier(backend=backend)` on the same
  loaded model, or a `FineTuningClassifier`) drops edits it doesn't predict positive.
  It is cross-fitted by default (`verify_cv=5`): each edit is judged by a clone that was
  never trained on its pivot, since a verifier trained on the very negative it's judging
  a minimal edit of tends to reject correct flips. `verify_cv="prefit"` uses an
  already-fitted classifier as-is.

`verify` has no default yet: which strategy works best is to be decided by a pilot
benchmark. Rejected candidates are kept in `generation_result_.rejected` with a reason
(`"edit_too_large"`, `"self_check_failed"`, `"verifier_rejected"`, ...) so you can see what
each filter catches.

> **Renamed in 0.16.0.** `OverSampler` → `HardPositiveOverSampler`, `SyntheticSampler` →
> `TypicalPositiveOverSampler`, and their `n_synthesized`/`seed` parameters →
> `sampling_strategy`/`random_state` (imbalanced-learn's names). The old names still work
> with a `DeprecationWarning` and will be removed in 0.18.0. `n_synthesized=n` corresponds
> to `sampling_strategy={pos_label: n_positive + n}`; `n_synthesized=None` to `"auto"`.

## Installation

`pntx` uses [uv](https://docs.astral.sh/uv/) for package management.

```bash
uv add pntx                # core (scikit-learn + pydantic)
uv add "pntx[llama]"       # + llama.cpp in-process backend
uv add "pntx[finetuning]"  # + FineTuningClassifier (transformers + torch)
uv add "pntx[embeddings]"  # + semantic similarity for selectors
```

`scikit-learn` and `pydantic` are core dependencies (every `t2pn` classifier's
scikit-learn contract and the `pn2t` over-samplers' structured LLM output need them
respectively). Each backend/feature otherwise lives behind its own extra, and using
one without installing it raises a clear `ImportError` with the install command to run.

## Backends

`pntx` runs models via a `Backend` protocol, shared by `LLMPromptingClassifier`,
`HardPositiveOverSampler`, and `TypicalPositiveOverSampler`. `FineTuningClassifier` does **not** use this
abstraction at all -- it has no LLM calls, only a fine-tuned `transformers` encoder, so
there's no loaded model to share with the others:

- **`LlamaCppBackend`** (`pntx[llama]`) — runs a GGUF model in-process via
  `llama-cpp-python`. This is the primary, most-tuned backend: classification uses
  token log-probabilities directly (`score_choices`), and batched classification
  reuses the shared few-shot prefix's KV cache across every item instead of
  re-evaluating it per item.

```python
clf = LLMPromptingClassifier(backend="llama", backend_kwargs={"model_path": "model.gguf"})

# or pass a backend instance directly, e.g. for dependency injection in tests
from pntx.backends.llama import LlamaCppBackend
clf = LLMPromptingClassifier(backend=LlamaCppBackend(model_path="model.gguf"))
```

A remote API backend can be added later by implementing the `Backend` protocol
(`pntx.backends.base.Backend`) and passing an instance directly — no built-in one
ships right now.

`backend_kwargs` is only used when `backend` is given as a string; it's a single dict
(rather than `**kwargs`) so `LLMPromptingClassifier` and the `pn2t` over-samplers stay compatible
with scikit-learn's `get_params()`/`clone()`.

`LlamaCppBackend` accepts either a local `model_path` or a `repo_id` (optionally
narrowed to one file with `filename`) to pull a GGUF model from the Hugging Face Hub
via `Llama.from_pretrained`. Any other keyword — `n_ctx`, `n_gpu_layers`,
`flash_attn`, `verbose`, ... — is forwarded as-is to `llama_cpp.Llama`:

```python
clf = LLMPromptingClassifier(
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

To share one loaded model across `LLMPromptingClassifier` and the `pn2t` over-samplers
(recommended for local inference — avoids loading the same GGUF twice), construct the
backend once and pass the instance to each (including as a `CounterfactualOverSampler`
verifier):

```python
from pntx.backends.llama import LlamaCppBackend

backend = LlamaCppBackend(model_path="model.gguf")
clf = LLMPromptingClassifier(backend=backend)
sampler = HardPositiveOverSampler(backend=backend)
synth = TypicalPositiveOverSampler(backend=backend, sampling_strategy={1: 100})
cf = CounterfactualOverSampler(backend=backend, verify=LLMPromptingClassifier(backend=backend))
```

## Selecting exemplars

When there are more fitted texts (on either side) than comfortably fit in a prompt, a
`Selector` decides which ones to use — `LLMPromptingClassifier` calls it independently for the
positive and negative pools:

- **`RandomSelector`** (default) — a uniform random subset.
- **`NearestSelector`** — picks texts most similar to the text being classified;
  dynamic, per-query selection.
- **`DiversitySelector`** — greedily picks a maximally diverse subset.
- **`BudgetSelector`** — picks as many texts as fit within a token budget (used
  internally by the `pn2t` over-samplers for their exemplar sampling).

`NearestSelector` and `DiversitySelector` take a `similarity_fn`. It defaults to a
dependency-free character n-gram similarity (`pntx.dedup.similarity`); pass
`pntx.embeddings.cosine_similarity_fn()` (requires `pntx[embeddings]`) for semantic
similarity instead:

```python
from pntx.t2pn import LLMPromptingClassifier
from pntx.selection import NearestSelector

clf = LLMPromptingClassifier(
    backend="llama", backend_kwargs={"model_path": "model.gguf"}, selector=NearestSelector()
)
```

`LLMPromptingClassifier` treats a selector as **static** or **dynamic** based on its
`query_aware` attribute (`RandomSelector`/`DiversitySelector`/`BudgetSelector` are
static; `NearestSelector` is the only built-in dynamic one):

- **Static** (default): exemplar selection, ordering, and calibration (see below) are
  all resolved once in `fit()` and reused by every later `predict`/`predict_proba`
  call — this is what lets a `BatchScoringBackend` (e.g. `LlamaCppBackend`) evaluate
  the shared few-shot prefix once per call and reuse its KV cache across the whole
  batch.
- **Dynamic** (e.g. `NearestSelector`): exemplars genuinely relevant to each text
  can't be known ahead of time, so selection reruns per text inside
  `predict`/`predict_proba` instead. For a `BatchScoringBackend` this forfeits the
  shared-prefix KV-cache reuse above (each text gets its own prefix and its own
  backend call), and `LLMPromptingClassifier` raises a `UserWarning` once per call to flag the
  latency trade-off.

`LLMPromptingClassifier` also applies **content-free calibration** (Zhao et al. 2021, "Calibrate
Before Use") by default on the `ScoringBackend` path: it scores an empty placeholder
query against the same few-shot prefix to estimate the prefix's own label bias (an
artifact of which exemplars ended up in it and in what order — few-shot prompts are
known to be sensitive to this), then divides each real prediction by that baseline and
renormalizes. Pass `LLMPromptingClassifier(..., calibrate=False)` to disable it and get the raw,
uncalibrated softmax instead.

The `pn2t` over-samplers don't take a `Selector`; instead their `sample_method`
constructor argument picks a *budget-based* sampling strategy (a full port of semaxis's
own `sample_method`/`embedding_model` for `HardPositiveOverSampler`;
`TypicalPositiveOverSampler` reuses the same mechanism for its positive-only exemplar
sampling):

- **`"random"`** (default) — a uniform random subset, filled until the token budget
  runs out (`BudgetSelector` under the hood).
- **`"kmeans"`** — embeds the pool via `embedding_model` (requires
  `pntx[embeddings]`) and picks one representative text per K-Means cluster.
- **`"votek"`** — embeds the pool and runs the Vote-K algorithm (Su et al. 2022),
  balancing representativeness and diversity.

```python
sampler = HardPositiveOverSampler(
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
