# Issue Candidate: Downstream benchmark for pn2t augmentation (with EDA baseline)

## Status

First results in (2026-10-10; see "3-seed results", which supersedes the seed-0 tables below). Started 2026-10-09. The local part is implemented:

- `benchmarks/eda.py`: EDA, unit-tested with a stub synonym source;
- `benchmarks/pn2t/downstream.py`: `generate` / `evaluate` with the TF-IDF
  classifier; the `finetuning` classifier option is meant for a GPU machine,
  since huggingface.co is unreachable here.

The first generation run uses qwen2.5-7B, 25 pos / 125 neg, seed 0. Results will
be recorded below.

## (superseded) First results (2026-10-09; TF-IDF + logistic regression, single seed)

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

**⚠ The `hard_positive` row above is invalid; it must be regenerated.** When its
prompts overflowed, `LlamaCppBackend` trimmed them *from the front*. For pn2t's
instruction-first prompts that removed about 210 tokens of the system prompt: the
role and the core rules, such as "prioritize texts experts would label Positive but
shallow classifiers would label Negative". So these hard positives were generated
largely *without their instructions*.

Fixed after this run. pn2t samplers now:

- measure the fixed prompt;
- cap `context_limit` by the backend's `context_window`;
- fit every prompt by dropping exemplars instead of letting the backend cut
  instructions.

A post-fix check with the same settings showed no truncation. It was also about
3–4× slower per call (2 calls, 1231 s for 4 texts), presumably because the intact
instructions elicit the full analysis output. That is unmeasured, and the
regeneration should record it. The `counterfactual` row very likely stands, by inference
rather than direct observation (stdout buffering means the log cannot attribute
warnings):

- The 44 warnings report prompts of 6296–6324 tokens. That matches HardPositive
  filling *both* per-class budgets (about 2 × 2822 tokens) plus its system prompt.
- Counterfactual uses at most 2 pivots per batch (about 500 tokens) on one side,
  so its prompts stay around 4.2k tokens.
- 44 is close to HardPositive's call count (about 50), not to Counterfactual's
  (about 84).

**Issues found (library).**

- `HardPositiveOverSampler`'s fixed `PROMPT_OVERHEAD = 500` underestimates its
  system prompt, which embeds the JSON schema. With `context_limit = n_ctx = 8192`
  and `max_tokens = 2048`, prompts reached about 6290 tokens against a 6080 budget,
  so `LlamaCppBackend` silently dropped exemplars.
- The backend's truncation warning tells users to pass
  `t2pn.LLMPromptingClassifier(max_exemplars=...)` even when the caller is a pn2t
  sampler, which is misleading.

## 3-seed results (2026-10-10) — current

**Setup.** Generation on the home GPU (`make run-bg`, qwen2.5-7B Q4_K_M,
`--n-gpu-layers -1`) for seeds 0, 1 and 2. Each seed draws its own 25 pos / 125
neg training set and generates its own augmentations (100 per condition, every
condition complete). There were **0 prompt-truncation warnings**, so these
`hard_positive` runs have intact instructions, unlike the invalid seed-0 run
below. Generation took about 13.5 min (HardPositive) and 20 min (Counterfactual)
per seed on the RTX 4060 Ti, against 2 h and 1.4 h on the Mac.

Evaluation per seed, with `--seed` also seeding BERT training:

- TF-IDF + LR, locally;
- `FineTuningClassifier(bert-base-uncased)`, library defaults
  (3 epochs, `max_length=128`), on the GPU.

**How to read the tables.** Values are the mean over seeds [min, max].
Δdup is the *paired* per-seed difference from the same-size `duplicate`
control, and "wins" counts the seeds where the condition beats it. The orig and
revised AUC columns are on the original and counterfactual CAD tests.

**TF-IDF + logistic regression**

| condition | orig AUC | revised AUC | revised acc | Δdup orig AUC | Δdup revised AUC (wins) |
|---|---|---|---|---|---|
| original | 0.750 [0.734, 0.768] | 0.493 [0.443, 0.553] | 0.511 | −0.003 | +0.007 (2/3) |
| duplicate | 0.753 [0.744, 0.767] | 0.486 [0.447, 0.536] | 0.510 | – | – |
| eda | 0.700 [0.678, 0.731] | 0.466 [0.450, 0.490] | 0.510 | −0.053 | −0.020 (1/3) |
| hard_positive | 0.701 [0.683, 0.736] | 0.436 [0.421, 0.453] | 0.499 | −0.051 | −0.050 (0/3) |
| **counterfactual** | 0.815 [0.799, 0.830] | **0.657** [0.622, 0.701] | 0.608 | +0.062 | **+0.171 (3/3)** |
| human_cad | 0.881 [0.872, 0.889] | 0.766 [0.765, 0.768] | 0.680 | +0.129 | +0.281 (3/3) |

**BERT (`FineTuningClassifier`, bert-base-uncased)**

| condition | orig AUC | revised AUC | revised acc | Δdup orig AUC | Δdup revised AUC (wins) |
|---|---|---|---|---|---|
| original | 0.825 [0.756, 0.938] | 0.690 [0.551, 0.897] | 0.613 | −0.063 | −0.080 (1/3) |
| duplicate | 0.888 [0.876, 0.897] | 0.770 [0.692, 0.830] | 0.610 | – | – |
| eda | 0.794 [0.751, 0.823] | 0.606 [0.474, 0.682] | 0.509 | −0.094 | −0.165 (0/3) |
| hard_positive | 0.888 [0.837, 0.937] | 0.791 [0.647, 0.935] | 0.647 | +0.000 | +0.021 (2/3) |
| **counterfactual** | 0.921 [0.910, 0.930] | **0.966** [0.965, 0.967] | **0.889** | +0.033 | **+0.196 (3/3)** |
| human_cad | 0.919 [0.918, 0.920] | 0.978 [0.976, 0.982] | 0.907 | +0.031 | +0.208 (3/3) |

**Findings.**

1. **`CounterfactualOverSampler` reliably helps, in every seed and with both
   classifiers.** It beats the same-size control on the counterfactual test in
   3/3 seeds.
   - With BERT it recovers **about 94% of human CAD's gain** over `duplicate` on
     that test (+0.196 vs +0.208 AUC; revised accuracy 0.889 vs 0.907), and it
     matches human CAD on the original test (0.921 vs 0.919).
   - With TF-IDF it recovers about 61% (+0.171 vs +0.281).
   - Its BERT results are also strikingly stable across seeds (revised AUC
     0.965–0.967).
2. **`HardPositiveOverSampler` does not help** once regenerated with intact
   instructions.
   - It slightly hurts TF-IDF: −0.05 AUC on both tests, 0/3 wins.
   - With BERT it is on par with `duplicate` (+0.000 / +0.021, 2/3 wins), with
     high variance (revised AUC 0.65–0.94).
   - Hard positives carry negative-looking surface features under a positive
     label. That confuses bag-of-words models and gives fine-tuned encoders no
     consistent signal at this scale.
3. **EDA hurts both classifiers** (0–1/3 wins), consistent with
   [[longpre-2020]].
4. **The same-size control matters for BERT.** `duplicate` beats `original` by
   +0.06 / +0.08 AUC from extra gradient steps and rebalancing alone (with fixed
   3 epochs). Measuring against `original` would have credited every
   augmentation, EDA included, with gains it doesn't have. BERT on the 150
   original examples is also unstable across seeds (orig AUC 0.76–0.94).

**Caveats.**

- 3 seeds, varying data, generation and BERT training jointly.
- One domain (English IMDb sentiment), one editor model (qwen2.5-7B Q4_K_M),
  one training size (25/125 + 100).
- Untuned library defaults (`max_length=128` truncates reviews).
- The counterfactual test set has the same construction as `human_cad`, so that
  reference is favoured on it by design.
- Conclusions about clinical or Japanese text need their own data.

**Implications.**

- The README can now state, with this evidence, that
  `CounterfactualOverSampler` improves robustness to counterfactual inputs in
  this setting.
- `HardPositiveOverSampler`'s value is not supported here, so the topic × style
  idea for it should wait for a setting where it helps.
- Next experiments, in order: a second domain; tuned `max_length`; and
  `FineTuningClassifier` hidden-space augmentation (Cutoff) stacked on
  `counterfactual`.

## (superseded) BERT results (2026-10-10; same augmentations, `--classifier finetuning`, home GPU)

**Setup.** Evaluated on the home GPU server (RTX 4060 Ti) with `make
bench-eval-ft`, using the seed-0 augmentation JSON from the TF-IDF run, so
`hard_positive` is still the **invalid**, instruction-truncated generation.
`FineTuningClassifier(bert-base-uncased, class_weight="balanced", seed=0)`
with library defaults: 3 epochs, lr 2e-5, batch 8, `max_length=128`, so reviews
are truncated to about half their length. AUC is shown with a 95% bootstrap CI.

| condition | orig acc | orig AUC | revised acc | revised AUC |
|---|---|---|---|---|
| original (150) | 0.588 | 0.756 [0.710, 0.798] | 0.521 | 0.623 [0.571, 0.675] |
| duplicate (250) | 0.775 | 0.897 [0.867, 0.923] | 0.675 | 0.789 [0.750, 0.829] |
| eda | 0.535 | 0.823 [0.786, 0.859] | 0.506 | 0.661 [0.615, 0.713] |
| hard_positive *(invalid)* | 0.670 | 0.862 [0.828, 0.894] | 0.595 | 0.794 [0.755, 0.833] |
| **counterfactual** | 0.760 | 0.911 [0.886, 0.935] | **0.784** | **0.951 [0.929, 0.969]** |
| human_cad | 0.832 | 0.920 [0.894, 0.944] | 0.916 | 0.976 [0.964, 0.986] |

**Reading the table.**

1. **For BERT, the same-size `duplicate` control is the right baseline, not
   `original`.** With a fixed 3 epochs, 250 examples mean about 67% more
   gradient steps than 150, plus rebalanced classes. Duplication alone lifts
   the original-test AUC from 0.76 to 0.90. Any augmentation must beat
   `duplicate` to count; this is exactly the [[huang-2020]] control.
2. **`counterfactual` beats `duplicate` clearly on the counterfactual (revised)
   test**: AUC 0.951 vs 0.789, with non-overlapping CIs. That recovers about 87%
   of human CAD's gain over `duplicate` on that test (0.976). On the original
   test it is on par with `duplicate` and human CAD (overlapping CIs). Same
   pattern as TF-IDF, stronger.
3. **EDA is worse than `duplicate`** on both tests, consistent with
   [[longpre-2020]].
4. **The invalid `hard_positive` is about equal to `duplicate`.** Rerun pending.

**Caveats.** One seed for data, generation and BERT training. `max_length=128`
truncation. Library-default hyperparameters, which are not tuned. Fixed epochs
confound data size with training steps (handled by comparing against
`duplicate`). Multi-seed GPU regeneration is running (`make run-bg
SESSION=gen-gpu`, seeds 0–2, files `pn2t-downstream-aug-gpu-qwen25-7b-seed*.json`).

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
