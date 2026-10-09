# Issue Candidate: Downstream benchmark for pn2t augmentation (with EDA baseline)

## Status

In progress (2026-10-09). The local part is implemented:

- `benchmarks/eda.py`: EDA, unit-tested with a stub synonym source;
- `benchmarks/pn2t/downstream.py`: `generate` / `evaluate` with the TF-IDF
  classifier; the `finetuning` classifier option is meant for a GPU machine,
  since huggingface.co is unreachable here.

The first generation run uses qwen2.5-7B, 25 pos / 125 neg, seed 0. Results will
be recorded below.

## First results (2026-10-09; TF-IDF + logistic regression, single seed)

**Setup.** `generate` with qwen2.5-7B, seed 0. The training set is 25 positive /
125 negative CAD training reviews (at most 1000 characters), and every condition
adds 100 positives, giving 250 training examples. Test sets: the original CAD test
reviews (n = 488) and their counterfactual revisions (n = 486). Values are the
point estimate with a 95% bootstrap CI over test examples. "pred+" is the share of
test items predicted positive.

| condition | orig acc | orig AUC | orig pred+ | revised acc | revised AUC | revised pred+ |
|---|---|---|---|---|---|---|
| original | 0.514 | 0.746 [0.706, 0.785] | 0.05 | 0.512 | 0.483 [0.430, 0.533] | 0.03 |
| duplicate | 0.516 | 0.747 | 0.04 | 0.512 | 0.475 | 0.03 |
| eda | 0.525 | 0.690 [0.644, 0.739] | 0.07 | 0.514 | 0.457 | 0.04 |
| hard_positive | 0.525 | 0.677 [0.629, 0.722] | 0.06 | 0.498 | 0.451 | 0.05 |
| **counterfactual** | **0.643** | 0.786 [0.750, 0.824] | 0.24 | **0.588** | **0.647 [0.592, 0.696]** | 0.23 |
| human_cad | 0.793 | 0.872 [0.842, 0.901] | 0.50 | 0.681 | 0.768 [0.727, 0.807] | 0.54 |

**Reading the table.**

1. **Accuracy alone is misleading here.** With 25 positives, TF-IDF + LR predicts
   "positive" for only about 5% of test items, despite `class_weight="balanced"`.
   Accuracy differences largely track that bias, which is why ROC-AUC
   (threshold-free) and pred+ were added to `evaluate`.
2. **Kaushik et al.'s core finding replicates.** The original-data model ranks
   original reviews reasonably (AUC 0.75) but is *at chance or worse* on their
   counterfactual revisions (AUC 0.48). Human CAD fixes both (0.87 / 0.77).
3. **`CounterfactualOverSampler` is the only generated condition that helps.**
   - On the revised test its AUC rises from 0.48 to 0.65; the CIs don't overlap.
     That is about 58% of the human-CAD gain on that test.
   - On the original test the AUC gain is smaller (0.75 → 0.79, overlapping CIs).
     Its accuracy gain there is mostly from a less skewed threshold (pred+ 0.05 →
     0.24).
4. **EDA and `HardPositiveOverSampler` slightly *hurt* ranking** (original AUC
   0.69 and 0.68 vs 0.75) and do not help on the revised test. For HardPositive
   with a bag-of-words model this is plausible by design: hard positives carry
   negative-looking words with a positive label. Whether this holds for
   `FineTuningClassifier` is the open question for the GPU run.
   - Caveat: during generation, HardPositive prompts overflowed the 8192-token
     context and the backend dropped exemplars (see "Issues found").
5. **Duplication equals the original**, as expected under balanced class weights.
   It confirms that the gains above come from content, not from size or class
   balance.

**Caveats.**

- A single data and generation seed. The CIs cover test sampling only, not
  training-set or generation variance.
- One classifier family (TF-IDF + LR), one domain (English movie reviews), and
  one editor model.
- Conclusions for pretrained classifiers need the `--classifier finetuning` run
  on a GPU machine, reusing this augmentation JSON.

**Cost.** Generation took 7108 s for HardPositive and 5054 s for Counterfactual
(100 accepted texts each; Counterfactual rejected 54 candidates as
`edit_too_large` and 13 as `self_check_failed`). EDA, duplication and human CAD
are free.

**Issues found (library).**

- `HardPositiveOverSampler`'s fixed `PROMPT_OVERHEAD = 500` underestimates its
  system prompt, which embeds the JSON schema. With `context_limit = n_ctx = 8192`
  and `max_tokens = 2048`, prompts reached about 6290 tokens against a 6080 budget,
  so `LlamaCppBackend` silently dropped exemplars.
- The backend's truncation warning tells users to pass
  `t2pn.LLMPromptingClassifier(max_exemplars=...)` even when the caller is a pn2t
  sampler, which is misleading.

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
