# Issue Candidate: Downstream benchmark for pn2t augmentation (with EDA baseline)

## Status

Draft (2026-10-08). Not started.

## Motivation

pntx's existing benchmarks don't answer pn2t's central question: does training a
classifier on pn2t-augmented data *help*?

- `benchmarks/t2pn/run.py` measures classification.
- `benchmarks/pn2t/counterfactual_pilot.py` measures only whether
  `CounterfactualOverSampler` edits flip the label.

The literature also says such a benchmark needs baselines that are
trivial but strong:

- [[huang-2020]]: CAD can lose its advantage against a *same-size* unaugmented
  control.
- [[wei-2019]] (EDA) is the standard cheap text-augmentation baseline, used e.g.
  by [[jin-2024]].
- [[longpre-2020]]: EDA and back-translation fail to consistently improve
  *pretrained* transformers. That is `t2pn.FineTuningClassifier`'s setting, so EDA
  doing nothing there is the expected, not alarming, outcome.

## Evidence

- [[wei-2019]] (EMNLP-IJCNLP 2019, read):
  - Four label-preserving operations: synonym replacement (WordNet), random
    insertion, random swap and random deletion. The share of words changed is
    α, and each sentence gets `n_aug` augmented copies.
  - Average gain is +3.0% at N_train = 500 but only +0.8% on full data, with
    CNN/RNN models only.
  - The authors expect little gain with pretrained models.
- [[longpre-2020]] (Findings EMNLP 2020, abstract): a negative result for EDA and
  back-translation on BERT, XLNet and RoBERTa across 5 tasks and 6 datasets.
- [[kaushik-2019]] Table 5, [[sen-2023]] and [[nguyen-2024]]: CAD is evaluated on
  original *and* counterfactual/out-of-domain test sets, because its gains often
  show up only on the challenging ones.

## Proposed Scope

**Data.** CAD IMDb (`benchmarks/cad.py`, already implemented):

- Train on a low-resource subset of the original training reviews.
- Test on the original test split, on the revised (counterfactual) test split,
  and optionally on an out-of-domain sentiment set.

**Classifiers.** `t2pn.FineTuningClassifier` is the main one; it is the case
where [[longpre-2020]] predicts EDA won't help. A TF-IDF + logistic-regression
model is a cheap secondary classifier, matching [[kaushik-2019]]'s linear models.

**Conditions.** Every augmenting condition adds the *same number* of positives,
so the comparison is fair:

| condition | role |
|---|---|
| original only | lower reference |
| + duplicated positives (`imblearn`-style random over-sampling) | same-size control ([[huang-2020]]) |
| + **EDA** on positives | standard cheap text-augmentation baseline ([[wei-2019]]) |
| + `HardPositiveOverSampler` | pntx |
| + `CounterfactualOverSampler` | pntx |
| + human CAD revisions | upper reference (only possible on CAD data) |

**EDA lives in `benchmarks/`, not in the library.**

- It needs WordNet (`nltk` plus its data download) and is effectively
  English-only: synonym operations need a lexicon, and Japanese needs a word
  tokenizer for any of the four operations.
- It doesn't use the `Backend` abstraction, so it doesn't fit pn2t's LLM-based
  design.
- A library `EDAOverSampler` would add a dependency for little value to pntx
  users, who can use existing augmentation packages.
- Implement the four operations directly, following the paper's definitions
  (about 60 lines), seeded, applied to positives only, with the paper's
  recommended small α.

**Reporting.** Accuracy and F1 per test set, mean ± standard deviation over at
least 3 seeds. Also report generation cost (LLM calls and wall time) next to the
gains.

## Acceptance Criteria

- `benchmarks/eda.py`: the four operations, unit-tested offline. Use a stub
  synonym source so the tests need no WordNet download. Cover English
  tokenisation, determinism under a seed, `n_aug`, and α scaling with sentence
  length.
- `benchmarks/pn2t/downstream.py`: runs every condition with an equal added
  count. The tables are reproducible from a results JSON (git-ignored).
- Results and caveats are recorded here before any README claim is made about
  augmentation helping.

## Out Of Scope

- EDA (or any non-LLM augmentation) as a public pntx API.
- Japanese EDA (no tokenizer or WordNet setup). This also means the benchmark
  can't say anything about Japanese.
- Back-translation. It is the other baseline in [[longpre-2020]], but it needs a
  translation model.

## Open Questions

1. **Training size.** Which low-resource sizes should be used (e.g. 50 / 200 / 1000
   per class), given that augmentation effects shrink with data size
   ([[wei-2019]], [[jin-2024]])?
2. **Compute budget.** Each `FineTuningClassifier` run needs a GPU for
   reasonable time. This might require a smaller encoder, or running on a GPU
   server like the t2pn benchmark.
3. **AEDA.** AEDA (punctuation insertion, Findings EMNLP 2021) is a simpler,
   arguably more language-agnostic variant. It appeared in search but has not
   been read or catalogued. Should it be added as a second cheap baseline?
