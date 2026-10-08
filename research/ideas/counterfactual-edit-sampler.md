# Issue Candidate: Counterfactual minimal-edit over-sampler for pn2t

## Status

Implemented as `pn2t.CounterfactualOverSampler` (`src/pntx/pn2t/_counterfactual.py`,
released in 0.17.0). The pilot benchmark below has been run: `verify` now
defaults to `"self"` and `max_edit_ratio` stays `0.5` (both evidence-based). Open:
generation quality (partial flips; see the open questions at the end).

### First real-model observations (2026-10-08, 3 Japanese negatives, not a benchmark)

- **Llama-3.2-1B-Instruct:** mostly unusable for this task in Japanese. Most
  candidates were rewrites (`edit_too_large`), no-ops, or out-of-range
  `pivot_id`s, and the filters caught them. What got through shows each
  `verify` mode's weak spot. With `verify="none"`, it accepted 「冷たかった→寒かった」,
  which is still negative (the model itself marked it `is_positive=false`). With
  `verify="self"`, it accepted a truncated 「この映画は」 and 「この映画は無情でした」,
  which isn't positive, both self-marked positive. So a 1B model's self-check is not
  reliable.
- **qwen2.5-7B-Instruct:** clean minimal flips (edit ratio 0.28–0.40), e.g.
  「この映画は退屈だった→この映画は興奮した」. One label-ambiguous edit passed under
  `verify="none"`: 「サポートの対応が良かっただけでは足りない」. It also exposed a
  format bug. Its first run put the `"original -> edited"` changed_spans format
  into the text field, and that still passed `max_edit_ratio` on short pivots. This
  was fixed by renaming the LLM field to `edited_text`, adding an explicit prompt
  rule, and adding a `malformed_text` rejection for arrows not present in the pivot.
- Takeaway for the pilot: the filters handle *form*, and label validity is
  where the strategies differ. Include a small model in the pilot, since that is
  where `"self"` vs. classifier verification should diverge most.

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

### Pilot results (2026-10-08)

Setup: `benchmarks/pn2t/counterfactual_pilot.py` (see `benchmarks/README.md`).
It uses 40 negative pivots from the CAD test split (reviews ≤ 1000 chars) and
20 positive references from the original training split, with seed 0. Edits
are generated once with `verify="none"` at temperature 0.7, and all three
strategies are then scored on identical candidates. The reference label comes
from a judge model of a *different family* than the editor, zero-shot. Hardness
is measured with TF-IDF + logistic regression trained on the original training
reviews only. A judge was needed because gemma3-12B, the intended judge, does
not fit in this machine's 16 GB RAM. Raw results are in
`benchmarks/results/pn2t-cf-pilot-*.json` (git-ignored).

**Editor qwen2.5-7B, judge llama3.1-8B.** The judge agreed with the human CAD
labels 0.89 of the time. 40 candidates; 14 were rejected earlier as
`edit_too_large`. Median LLM edit ratio was 0.16, versus 0.08 for humans.

| verify | kept | precision | catch rate | false-reject rate | downstream-easy | valid & hard |
|---|---|---|---|---|---|---|
| none | 40 | 0.33 | 0.00 | 0.00 | 0.33 | 6 |
| self | 30 | 0.43 | 0.37 | 0.00 | 0.30 | 6 |
| classifier (cv=5) | 12 | 0.75 | 0.89 | 0.31 | 0.83 | 2 |
| *human revisions* | 40 | 0.82 | – | – | 0.50 | 17 |

**Editor llama3.2-3B, judge qwen2.5-7B.** The judge agreed with the human labels
0.93 of the time. 32 candidates; 77 were rejected earlier as `edit_too_large`.
Only 2 of the 32 candidates were judged positive.

| verify | kept | precision | catch rate | false-reject rate | downstream-easy | valid & hard |
|---|---|---|---|---|---|---|
| none | 32 | 0.06 | 0.00 | 0.00 | 0.28 | 0 |
| self | 9 | 0.11 | 0.73 | 0.50 (n=2) | 0.44 | 0 |
| classifier (cv=5) | 1 | 1.00 | 1.00 | 0.50 (n=2) | 1.00 | 0 |
| *human revisions* | 40 | 0.90 | – | – | 0.50 | 17 |

How to read the columns:

- **precision:** the share of kept edits the judge calls positive.
- **catch rate / false-reject rate:** the share of judge-negative / judge-positive
  candidates the strategy rejects.
- **downstream-easy:** the share of kept edits the shallow classifier already
  gets right.
- **valid & hard:** kept edits that are judge-positive *and* missed by the shallow
  classifier. These are the edits worth adding to training data.

**Findings**

1. **The editor is the bottleneck, not the verifier.** Even qwen2.5-7B flips
   only a third of the reviews (by the judge), against 0.82–0.90 for human
   revisions. The typical failure is a *partial flip* on a long review: the
   rating changes, or a "however, it has redeeming qualities" sentence is
   appended, while most negative statements stay. llama3.2-3B mostly rewrites
   instead of editing. No verification strategy fixes a weak editor.
2. **`"self"` dominates `"none"`.** It had higher precision in both runs
   (0.43 vs 0.33 and 0.11 vs 0.06), kept the same number of valid & hard edits
   with qwen (6 = 6), costs nothing extra, and wrongly rejected no good edit
   with qwen. Its catch rate is modest (0.37) with the stronger editor.
3. **The classifier verifier shows exactly the bias [[gardner-2020]] §2.4
   predicts.** It is the most precise, but 83% of what it keeps is already easy
   for the shallow model, and it keeps only 2 valid & hard edits (versus 6). It
   also rejects 31% of good flips, adds about 11 s per candidate (qwen run), and
   its low yield (0.30 and 0.03) means many more retries in real use. With a weak
   editor it would rarely reach the target count.

**Recommendation:** make `verify="self"` the default, and document the
classifier verifier as an opt-in for when label precision matters more than
hardness. Raise the generation quality next. The partial-flip failure points
at the prompt, e.g. requiring that *every* negative judgment in the text is
flipped, not just the verdict. Also document that the sampler needs a ≥7B-class
editor for review-length texts.

**Caveats:** 40 pivots per run, one domain (English movie reviews, mostly
600–1000 chars), one prompt, a single seed, and LLM judges in place of humans.
The judges agree with human labels 89–93% of the time, but they under-call
positives (0.82–0.90 on human revisions), so absolute precisions are
conservative. The comparison *between* strategies uses the same judge and the
same candidates, which is the robust part.

### Prompt iteration for partial flips (2026-10-08, one time-boxed round)

The prompt changes were:

1. Partial flips are named as failures: changing only the rating or verdict,
   or appending a concession, while other Negative statements remain.
2. *every* Negative judgment must be flipped, each by swapping or negating its
   evaluative words rather than by deleting or rewriting sentences.
3. `is_positive` is judged "as a reader who sees only the edited text".

The rerun used the same pilot, seed and models (qwen2.5-7B editor,
llama3.1-8B judge).

| verify | precision | catch rate | false-reject | downstream-easy | valid & hard | yield |
|---|---|---|---|---|---|---|
| none | 0.33 → **0.53** | – | – | 0.33 → 0.60 | 6 → 6 | 1.00 |
| self | 0.43 → **0.59** | 0.37 → 0.26 | 0.00 → 0.05 | 0.30 → 0.59 | 6 → 6 | 0.75 → 0.85 |
| classifier | 0.75 → 0.77 | 0.89 → 0.74 | 0.31 → 0.19 | 0.83 → 0.68 | 2 → 5 | 0.30 → 0.55 |

Rewrites rejected as `edit_too_large` fell from 14 to 10. The median edit ratio
went from 0.165 to 0.188, still far under 0.5. The new prompt is kept: it
improves precision with no loss of valid & hard edits.

Complete flips are also more lexically obvious, so a larger share of them is
easy for the shallow model (downstream-easy rose). That is the expected price
of fixing partial flips, not a regression in useful output: valid & hard is
unchanged.

`"self"` stays the default. It still beats `"none"` on precision for free. The
classifier verifier closed most of its hardness gap (5 vs 6 valid & hard) at
0.77 precision, but it costs about 11 s per candidate and more retries (yield
0.55). It remains the opt-in for precision-sensitive use.

Caveat: the precision gain for `"none"` is 21/40 vs 13/40 judge-positive, from a
single sampled run at temperature 0.7. That is a meaningful but not conclusive
difference. No further prompt rounds are planned, per the maintainer's direction
to not over-invest in small-model quality.

### Negative result: "blind" self-check (2026-10-08, reverted)

**Hypothesis**, from the literature survey
(`research/reports/2026-10-08-cad-contrast-set-followups-llm.md`): `"self"`
catches few failed flips because the editor is asked to confirm the target label
right after being told to produce it. LLM judges tend to agree with a provided
label ([[nguyen-2024]]). So asking for the label *without* stating the target
should help.

**Change tried.** The `is_positive: bool` field was replaced by
`reader_label: "Positive" | "Negative" | "Ambiguous"`, described as "the label a
reader would assign who sees ONLY edited_text … 'Ambiguous' if the text mixes
Positive and Negative judgments". This also dropped the old instruction to "set
it to false if any Negative judgment remains". Nothing else changed: same pilot,
same seed, same models (qwen2.5-7B editor, llama3.1-8B judge).

| | `is_positive` (prompt v2) | `reader_label` (blind) |
|---|---|---|
| self says positive | 34/40 | 39/40 |
| self catch rate | 0.26 | **0.06** |
| self precision (none, same run) | 0.59 (0.53) | 0.59 (0.57) |
| classifier precision / catch | 0.77 / 0.74 | 0.74 / 0.59 |
| `"Ambiguous"` used | – | 0 times |

**Outcome:** reverted. The neutral question did not make the judgment blind.
The editor wrote the text *intending* it to be positive and labelled 39 of 40 as
`"Positive"`. It never used `"Ambiguous"`, even on the partial flips the judge
called negative.

The catches under prompt v2 evidently came from the explicit failure criterion
("false if any Negative judgment remains"), not from neutral wording. A truly
blind check needs a *separate* call that doesn't see the editing instruction.
With the same backend, that is essentially `verify=<LLMPromptingClassifier>`,
which is already supported and measured.

So same-response self-assessment appears to have a low ceiling, and prompt v2's
`is_positive` wording is the better of the two tried. Single run; the catch-rate
difference (5/19 vs 1/17 judge-negative candidates caught) is large enough to act
on, but it is not a precise estimate.

### `max_edit_ratio` calibration (2026-10-08)

On the 363 human negative→positive revisions in the CAD test and dev splits,
`dedup.edit_ratio` has a median of 0.081, a p95 of 0.20 and a maximum of 0.40.
None exceed 0.5. The ratio does not grow with length: by length bucket, the p95
is 0.28 (<300 chars), 0.23, 0.20, 0.19 and 0.16 (>1500 chars), and short texts
vary the most. So the provisional default of 0.5 is kept, as a ceiling above
every human edit. A length-dependent threshold is not needed (closes open
questions 1 and 2).

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

1. ~~**Long texts.**~~ Closed: human edit ratios don't grow with length
   (see "`max_edit_ratio` calibration"). Long reviews do make *partial flips*
   more likely, but that is a generation-quality problem, not a threshold one.
2. ~~**Default `max_edit_ratio`.**~~ Closed: 0.5 kept (human max 0.40).
3. ~~**Default `verify`.**~~ Closed: `"self"`, adopted 2026-10-08 per the pilot.
4. **Partial flips.** One prompt round raised `"none"` precision from 0.33 to
   0.53 and `"self"` from 0.43 to 0.59 with qwen2.5-7B (see "Prompt iteration").
   This is still below human revisions (0.82 by the same judge). Further gains
   most likely come from a stronger editor model rather than more prompting;
   not pursued further for now.
