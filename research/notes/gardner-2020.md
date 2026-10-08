# gardner-2020

## Citation

Gardner, M., Artzi, Y., Basmova, V., Berant, J., Bogin, B., Chen, S., et al.
(2020). *Evaluating Models' Local Decision Boundaries via Contrast Sets.*
Findings of EMNLP 2020. DOI: 10.18653/v1/2020.findings-emnlp.117
(arXiv:2004.02709).

## Why It Matters For This Project

`pn2t.OverSampler` asks the LLM for `boundary_features` and then for hard
positives that "experts would label Positive but that simple rule-based
classifiers or untrained humans might label Negative". That is a claim about
the *local decision boundary*, which is exactly what this paper formalises.
It gives us (a) vocabulary and a definition for what a "boundary" example is
in discrete text space, (b) a sharp distinction between what OverSampler
produces and what a contrast set is, and (c) an evaluation metric
(contrast consistency) we could use to benchmark whether OverSampler output
actually moves a classifier's boundary.

## Method

- Decision boundary is defined as a partition `{(x, y)}`; a *local decision
  boundary* around a pivot `x` is `{(x', y') | d(x, x') < ε}`.
- A *contrast set* `C(x)` is a sample from that local boundary, typically
  with `y' ≠ y`: small, meaningful perturbations of a test instance that
  change the gold label while preserving its lexical/syntactic artifacts.
- *Contrast consistency*: a model is consistent on `C(x)` only if it is
  correct on every element (including the pivot).
- Construction is manual, by dataset authors (experts), **without a model in
  the loop**, after first listing the phenomena the dataset should test.
  Pivots are drawn from the i.i.d. test set; contrast sets are not given at
  training time.
- Built for 10 datasets (incl. IMDb sentiment, BoolQ, DROP, UD parsing);
  ~1–3 min per perturbation, ~1000 examples ≈ a person-week.

## Relevant Findings

- SOTA models drop sharply on contrast sets (e.g. IMDb BERT 93.8 → 84.2,
  consistency 77.8; DROP −25.7 F1; UD parsing consistency 17.3) while humans
  are roughly unaffected (IMDb 94.3 → 93.9). The gap is attributed to models
  learning simple rules that exploit systematic gaps in the data.
- §2.3: contrast sets are explicitly *not* adversarial examples. Adversarial
  examples change the model's decision but not the gold label; contrast sets
  change the gold label.
- §2.4: automatic construction is hard because "pushing pivots across a
  decision boundary ... presupposes a model that can already perform the
  intended task". Having a model in the loop biases data toward that model's
  idiosyncrasies.
- §4.5: tagging each perturbation with the phenomenon it targets enables
  fine-grained error reporting (e.g. MATRES appearance order 66.5% vs
  temporal conjunctions 60.0%). Authors recommend categorising perturbations
  up front.

## Limitations

- Contrast sets only have *negative* predictive power: they can show a model
  is misaligned locally, never that it is aligned.
- Construction is manual and dataset-specific; the paper offers no automated
  procedure. The distance `d` is left to expert judgement.
- Evaluation-only; the paper does not study training on contrast sets
  (it deliberately withholds them from training).

## Actionable Findings

1. **Terminology in docs.** OverSampler's hard positives are *not* contrast
   sets: they are new positives (gold label unchanged = positive) that a
   shallow classifier is expected to mislabel. By the paper's §2.3 taxonomy
   they sit closer to model-agnostic challenge examples than to contrast
   sets. Worth stating precisely in the README/docstring to avoid
   overclaiming.
2. **Contrast-set-style generation mode.** A natural variant is to use a
   *negative* exemplar as a pivot and ask the LLM for a minimal edit that
   flips it to positive (cf. [[kaushik-2019]]). The LLM plays the role §2.4
   says automation lacks — "a model that can already perform the task" — so
   it is plausible, but needs validation that labels actually flip.
3. **Link hard positives to boundary features.** `HardPositive` currently
   carries free-text `positive_evidence`/`confusing_evidence` but no pointer
   to which `BoundaryFeature` it targets. Adding that link would enable the
   per-phenomenon error analysis of §4.5 for free.
4. **Benchmark metric.** For pn2t benchmarks, report consistency on
   generated hard positives (or on a human-built contrast set such as the
   released IMDb one) for a classifier trained with vs. without OverSampler,
   not only i.i.d. accuracy.
5. **Model-in-the-loop caution.** OverSampler today has no classifier in the
   loop (good, per §2.4). If we ever add "keep only generations the current
   classifier gets wrong" filtering, note the bias risk the paper describes.

## Issue Candidate Impact

Candidates for `research/ideas/`: (2) negative-pivot minimal-edit mode for
OverSampler; (3) `BoundaryFeature` ↔ `HardPositive` linkage in
`pn2t/_types.py`; (4) contrast-consistency metric in a pn2t benchmark.
(1) is a docs fix and doesn't need an idea file.

## Sources Checked

- arXiv:2004.02709v2 full text (read §1–§6; Tables 1–3), downloaded
  2026-10-07 to `research/pdfs/gardner-2020.pdf`.
- Compared against `src/pntx/pn2t/prompts.py` (`SYSTEM`) and
  `src/pntx/pn2t/_types.py` at commit 9dc383d.
