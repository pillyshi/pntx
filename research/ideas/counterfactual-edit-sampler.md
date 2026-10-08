# Issue Candidate: Counterfactual minimal-edit sampler for pn2t

## Status

Draft.

## Motivation

`pn2t.OverSampler` generates *new* hard positives from a boundary-feature
analysis. There's no direct evidence that this specific mechanism improves a
classifier's robustness. Counterfactually-augmented data (CAD), on the other
hand, does have that evidence: take an existing example, make the smallest
edit that flips its gold label, and train on the original plus the edit.
That breaks spurious correlations (genre words, etc.) because the edit keeps
every non-causal feature fixed while the label changes.

Doing the edit with an LLM instead of crowd workers would give pntx a second,
evidence-backed augmentation strategy. It would reuse the same `Backend`, the
same `(X, y)` contract and the same `fit_resample` interface.

## Evidence

- [[kaushik-2019]] (`research/notes/kaushik-2019.md`): training on original +
  revised IMDb reviews (3.4k) performs well on both original and revised test
  sets. It generalises out of domain (Amazon/Twitter/Yelp) better than the
  same amount of original data. Adding 1.7k revised reviews to 19k originals
  substantially improves accuracy on revised data. Spurious features stop
  being predictive once pairs are combined. All edits there were
  human-written; LLM-generated edits are untested in that paper.
- [[gardner-2020]] (`research/notes/gardner-2020.md`): minimal label-flipping
  edits are exactly what probes a model's *local* decision boundary. Models
  near SOTA lose 5–25 points on such sets while humans don't. §2.4 notes that
  automating the flip "presupposes a model that can already perform the
  intended task", and an LLM can plausibly fill that role.

## Proposed Scope

A new, independent class `pn2t.CounterfactualSampler`. It is a separate
class rather than an `OverSampler` mode, following the `SyntheticSampler`
precedent: the objective differs, so it gets its own class.

- `fit_resample(X, y)`: same contract as the other samplers (binary `y`,
  `resolve_binary_labels`, `pos_label`, both classes ≥ 1).
- **Pivots are negatives; output is positives only.** Each generated text is
  a minimal edit of one negative exemplar that makes the positive label
  apply. The original negative is already in `X`, so appending the edited
  positive yields the original–counterfactual *pair* that CAD relies on
  without generating any negative-side text. This stays inside CLAUDE.md's
  "positive-side generation only" scope. The reverse direction
  (positive → negative) remains out of scope.
- Prompt in `pn2t/prompts.py` (user-replaceable), with the three CAD
  constraints from Kaushik et al.: (i) the target label applies, (ii) the
  text stays coherent, (iii) no unnecessary changes. Optionally list the
  paper's eight edit patterns (Table 2) as hints.
- Structured output per edit: `source_index`, `text`, `changed_spans` (what
  was edited and why), validated with pydantic via `complete_structured`.
- **Minimality filter:** reject edits whose character-level change ratio
  (e.g. `1 - difflib.SequenceMatcher.ratio()`) exceeds `max_edit_ratio`, and
  reject no-op edits. Dependency-free, language-agnostic, and matches the
  existing `pntx.dedup` style.
- Exact-match dedup against `X` and previously accepted outputs (same as
  `OverSampler`).
- `n_synthesized=None` → balance classes, same as `OverSampler`. Pivot
  selection uses the existing `sample_method` machinery on the negative pool.
- `generation_result_` (pydantic): list of
  `(source_text, edited_text, changed_spans)` so pairs are auditable.
- `save`/`load`: same hand-written JSON pattern as the other samplers.

## Acceptance Criteria

- Unit tests with `FakeBackend` cover:
  - minimal edit accepted;
  - edit rejected by `max_edit_ratio` and retried;
  - no-op edit rejected;
  - exact-duplicate rejected;
  - batch cap reached → warning;
  - `n_synthesized=None` balance computation;
  - `save`/`load` round trip;
  - Japanese and English cases for the minimality filter.
- Positives are never used as pivots (direct prompt-content test, analogous
  to `SyntheticSampler`'s "negatives never in prompt" test).
- Generated texts are labelled with the resolved positive label, not `1`.
- Works in `imblearn.pipeline.Pipeline` via duck-typed `fit_resample`.
- An integration test (skipped by default) runs it end-to-end on
  `LlamaCppBackend`.

## Out Of Scope

- Positive → negative edits (negative-side generation; CLAUDE.md scope).
- Classifier-in-the-loop filtering by default (see Open Questions).
- Benchmarking the effect on a downstream classifier. That should be its own
  idea/issue (counterfactual test set à la Kaushik Table 5 / Gardner
  contrast consistency).
- Semantic-similarity-based minimality (embeddings). The character-level ratio
  is the v1 heuristic.

## Open Questions

1. **Label verification.** Kaushik et al. still rejected ~2% (sentiment) /
   ~9% (NLI) of *human* edits. How should LLM edits that didn't actually
   flip the label be caught?
   - (a) No verification; trust the prompt. Cheapest, noisiest.
   - (b) Self-check field in the structured output. Cheap, but weak.
   - (c) Score with a `t2pn.LLMPromptingClassifier` on the shared backend.
     This is model-in-the-loop, which [[gardner-2020]] §2.4 warns against. It
     also biases *toward* edits the classifier already gets right, i.e.
     against the hard cases. This filters for label validity, not hardness,
     but the tension should be measured, not assumed away.
   - Leaning: (b) by default, with (c) as an opt-in `verifier=` parameter.
2. **Relationship to `OverSampler`.** Should `OverSampler` eventually offer
   this as a strategy, or should the two stay fully separate? Separate seems
   cleaner until benchmarks show which is more useful.
3. **Long texts.** Kaushik filtered out the longest 20% of reviews, and
   editor agreement drops with length. Should pivots be capped by token
   length, or should `max_edit_ratio` scale with length?
4. **Default `max_edit_ratio`.** It needs an empirical value; Kaushik doesn't
   report edit sizes directly. Pick it from a small pilot on IMDb CAD data
   (the released revised split gives real human edit ratios to calibrate
   against).
