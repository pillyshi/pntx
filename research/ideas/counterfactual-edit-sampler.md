# Issue Candidate: Counterfactual minimal-edit over-sampler for pn2t

## Status

Draft.

## Motivation

`pn2t.HardPositiveOverSampler` generates *new* hard positives from a boundary-feature
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
  human-written; LLM-generated edits are untested in that paper. Edits were
  always checked by someone other than the editor. For sentiment, the
  authors inspected every revision and rejected ~2%. For NLI, three other
  crowd workers voted, with the authors breaking ties, and ~9% was
  discarded.
- [[gardner-2020]] (`research/notes/gardner-2020.md`): minimal label-flipping
  edits are exactly what probes a model's *local* decision boundary. Models
  near SOTA lose 5–25 points on such sets while humans don't. §2.4 notes that
  automating the flip "presupposes a model that can already perform the
  intended task", and an LLM can plausibly fill that role.

## Proposed Scope

A new, independent class `pn2t.CounterfactualOverSampler`, a subclass of
`BaseLLMOverSampler`. This follows the pn2t family rule (CLAUDE.md, modelled
on imbalanced-learn): one class per generation algorithm, named
`<Name>OverSampler`. A different generation mechanism gets its own class
rather than a mode of `HardPositiveOverSampler`.

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
  `HardPositiveOverSampler`; inherited from `BaseLLMOverSampler`).
- `sampling_strategy="auto"` (default) → balance classes, same as
  `HardPositiveOverSampler`; inherited from the base. Pivot
  selection uses the existing `sample_method` machinery on the negative pool.
- `generation_result_` (pydantic): list of
  `(source_text, edited_text, changed_spans)` so pairs are auditable.
- `save`/`load`: same hand-written JSON pattern as the other samplers.
- **Label verification is a parameter, not a fixed choice.** All three
  strategies are supported; the default is decided by benchmark (see
  "Choosing the `verify` default" below):
  - `verify="none"`: trust the edit prompt; accept as-is.
  - `verify="self"`: the structured output carries a self-assessed label
    for each edit. Reject edits the LLM itself doesn't judge positive. No
    extra backend calls.
  - `verify=<classifier>`: any object with sklearn-style `predict`
    (duck-typed), e.g. a `t2pn.LLMPromptingClassifier` on the shared backend
    or a `t2pn.FineTuningClassifier`. Edits it doesn't predict as positive
    are rejected and retried. This is the closest analogue to the source
    papers, where an *independent* judge verified edits: Kaushik et al. used
    the authors (sentiment) or a separate set of crowd workers by majority
    vote (NLI), never the editor. The difference is that a model, unlike a
    human judge, carries the model-in-the-loop bias [[gardner-2020]] §2.4
    warns about.

    **Fitting: clone + cross-fitting, controlled by `verify_cv`** (decided).
    The verifier is judged as the *pre-augmentation* model: it is fitted only
    on the original `(X, y)`, never on generated edits. A naive
    `clone(verify).fit(X, y)` would leak the pivot, though. The verifier
    would have learned the very negative it is now judging a minimal edit
    of, and near-duplicates of training points tend to get their training
    label. This biases it toward rejecting correct flips. The bias is
    especially strong for `LLMPromptingClassifier` with `NearestSelector`,
    which will put the pivot itself in the prompt as a negative exemplar, and
    for an overfitted `FineTuningClassifier`. So, in the same spirit as
    `cross_val_predict`:
    - `verify_cv=<int K>` (default `5`): split `(X, y)` with
      `StratifiedKFold(K)` (seeded by `random_state`). Fit one `clone(verify)` per
      fold on the other K−1 folds. Judge each edit with the clone whose
      training data excludes the edit's pivot. No verifier ever saw the
      pivot it judges. Clones are fitted lazily, only for folds that
      actually supply pivots. If `K` exceeds the smaller class count,
      raise `ValueError` (same constraint as sklearn's `StratifiedKFold`).
    - `verify_cv="prefit"`: use the passed classifier as-is, without cloning
      or fitting (same convention as `CalibratedClassifierCV(cv="prefit")`).
      This is the escape hatch when re-fitting is expensive, e.g. a
      `FineTuningClassifier` trained elsewhere. The user is then responsible
      for leakage.
    - Cost note: K fits per `fit_resample`. Negligible for
      `LLMPromptingClassifier` (fit only stores pools); K full training runs
      for `FineTuningClassifier`, so documentation should point such users
      to a small K or `"prefit"`.
    - `verify_cv` is ignored for `verify="none"`/`"self"`.
  - Rejected edits and their reasons are kept in `generation_result_` so the
    rejection rate per strategy is observable.

### Choosing the `verify` default

Decide by a pilot benchmark before the first release, not by argument:

1. **Label-flip precision.** On a sample of IMDb negatives that have human
   counterfactual revisions (Kaushik et al. release), generate LLM edits.
   Human-judge (or use the paired human revision's label as reference) which
   edits actually flipped. For each strategy, report what fraction of
   failed flips it catches and what fraction of good edits it wrongly
   rejects.
2. **Hardness retained.** For `verify=<classifier>`, measure how much
   it skews accepted edits toward ones the downstream classifier already
   gets right (the bias concern), compared with `none`/`self`.
3. **Cost.** Backend calls and wall time per accepted edit on
   `LlamaCppBackend`.

Pick the default with the best precision/retention trade-off at acceptable
cost. If two strategies are close, prefer the cheaper one.

## Acceptance Criteria

- Unit tests with `FakeBackend` cover:
  - minimal edit accepted;
  - edit rejected by `max_edit_ratio` and retried;
  - no-op edit rejected;
  - exact-duplicate rejected;
  - batch cap reached → warning;
  - `sampling_strategy="auto"` balance computation;
  - `save`/`load` round trip;
  - Japanese and English cases for the minimality filter.
- Positives are never used as pivots (direct prompt-content test, analogous
  to `TypicalPositiveOverSampler`'s "negatives never in prompt" test).
- Generated texts are labelled with the resolved positive label, not `1`.
- Each `verify` strategy is unit-tested: `"none"` accepts everything that
  passes the minimality/dedup filters; `"self"` rejects edits self-assessed
  as not positive; a classifier verifier rejects edits it predicts as
  negative (using a stub classifier). Rejections are recorded in
  `generation_result_`. An invalid `verify` value raises `ValueError`.
- Cross-fitting is tested with a recording stub classifier:
  - each edit is judged by a clone whose training data excludes its pivot;
  - the user's `verify` instance itself is never fitted (only clones are);
  - generated edits are never in any clone's training data;
  - `verify_cv="prefit"` uses the instance unchanged, with no clone or fit;
  - `K` greater than the smaller class count raises `ValueError`.
- The pilot benchmark in "Choosing the `verify` default" has been run and
  its result recorded (e.g. in `benchmarks/` or `research/reports/`) before
  the default is fixed.
- Works in `imblearn.pipeline.Pipeline` via duck-typed `fit_resample`.
- An integration test (skipped by default) runs it end-to-end on
  `LlamaCppBackend`.

## Out Of Scope

- Positive → negative edits (negative-side generation; CLAUDE.md scope).
- Benchmarking the effect on a downstream classifier. That should be its own
  idea/issue (counterfactual test set à la Kaushik Table 5 / Gardner
  contrast consistency).
- Semantic-similarity-based minimality (embeddings). The character-level ratio
  is the v1 heuristic.

## Open Questions

Settled so far: verification strategy (all three supported; default chosen
by benchmark) and classifier fitting (clone + cross-fitting via `verify_cv`,
with `"prefit"` escape hatch), and the class's place in the family (its own
`CounterfactualOverSampler`, not a `HardPositiveOverSampler` mode). See
"Proposed Scope".

1. **Long texts.** Kaushik filtered out the longest 20% of reviews, and
   editor agreement drops with length. Should pivots be capped by token
   length, or should `max_edit_ratio` scale with length?
2. **Default `max_edit_ratio`.** It needs an empirical value; Kaushik doesn't
   report edit sizes directly. Pick it from a small pilot on IMDb CAD data
   (the released revised split gives real human edit ratios to calibrate
   against).
