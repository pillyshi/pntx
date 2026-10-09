# Hidden-space data augmentation during fine-tuning of pretrained models

*Literature survey, 2026-10-09.*

## Scope

This survey covers methods that augment data **inside the network** while a
pretrained model is fine-tuned. Instead of producing new text, they perturb,
interpolate or drop parts of the **embeddings or hidden states**. This is the
family relevant to `t2pn.FineTuningClassifier`.

It contrasts with:

- pntx's `pn2t` samplers, which generate *text* with an LLM before training;
- [[jin-2024]], which edits latents with a separate generator and *decodes them
  back to text* (see `research/ideas/soft-label-augmented-finetuning.md`).

## Method

- **Search:** OpenAlex keyword search across the main method families, with
  abstracts confirmed from OpenAlex.
- **Catalog:** 19 papers added with `littrail add-paper`; `verify` passes.
- **Read in the main text:** the TACL 2023 empirical survey ([[chen-2023-2]]),
  which is the most decision-relevant paper. Its experimental setup, findings and
  conclusions were read; its results tables did not extract cleanly from the
  PDF, so specific numbers from them are not quoted.
- **Everything else** is summarised from abstracts.

## A taxonomy of the methods

### 1. Interpolation (mixup in hidden space)

Mix two examples' hidden representations, and their labels, with weight λ.

- [[verma-2018]] **Manifold Mixup** (ICML 2019; arXiv): interpolating *hidden*
  states gives smoother decision boundaries and flatter class representations.
  This is the general, non-NLP origin.
- [[guo-2019]]: mixup on word embeddings or on sentence embeddings for sentence
  classification. It improved CNN and LSTM models (pre-BERT).
- [[chen-2020]] **MixText / TMix** (ACL 2020): interpolates BERT hidden states at
  a chosen layer. Used for semi-supervised classification, it is strongest when
  labels are extremely scarce.
- [[sun-2020]] **Mixup-Transformer** (COLING 2020): end-to-end mixup inside
  BERT-style fine-tuning, evaluated on GLUE and in low-resource settings.
- [[kong-2020]] (EMNLP 2020) targets **calibration**:
  - *on-manifold* interpolations act as smoothness regularisation, improving
    in-distribution calibration;
  - *off-manifold* samples pushed toward uniform predictions reduce out-of-
    distribution overconfidence.
- [[xiao-2026]] **SFTMix** (ACL 2026): mixup for LLM *instruction tuning*,
  weighting examples by the model's confidence (training dynamics). It is a
  generative-LM variant, outside `FineTuningClassifier`'s encoder setting.

### 2. Adversarial and smoothness perturbation in embedding space

Perturb embeddings in the *worst-case* (or a random) direction within a small
ball, and train the model to keep its prediction.

- [[miyato-2016]] (ICLR 2017; arXiv): adversarial and *virtual* adversarial
  training applied to **word embeddings** rather than one-hot inputs. This is the
  origin of the NLP line and also works semi-supervised.
- [[zhu-2019]] **FreeLB** (ICLR 2020; arXiv): multi-step adversarial
  perturbations of embeddings during fine-tuning only. BERT-base on GLUE goes from
  78.3 to 79.4.
- [[jiang-2020]] **SMART** (ACL 2020): smoothness-inducing adversarial
  regularisation plus Bregman proximal-point optimisation (a trust region),
  explicitly to stop aggressive fine-tuning from overfitting small downstream
  data.
- [[aghajanyan-2020]] **R3F/R4F** (ICLR 2021; arXiv): replaces the adversarial
  step with *random* parametric noise, which is cheaper and reduces
  "representational collapse".
- [[zhou-2021]] **Virtual Data Augmentation** (EMNLP 2021): a mixture of masked-LM
  token-embedding substitutes plus Gaussian noise, for robustness while keeping
  semantics.

### 3. Erasure and dropout views

Create restricted views of an example by erasing parts of its representation,
and enforce consistent predictions.

- [[shen-2020]] **Cutoff** (arXiv 2020): token, feature or span cutoff on the
  embedding matrix during fine-tuning, with a Jensen–Shannon consistency loss.
  Its selling point is near-zero extra compute compared with adversarial
  training.
- [[chen-2021]] **HiddenCut** (ACL 2021): drops contiguous spans of *hidden*
  states, preferring informative spans, for better out-of-distribution
  generalisation.
- [[liang-2021]] **R-Drop** (NeurIPS 2021; arXiv): two dropout passes per example
  with a bidirectional-KL consistency loss. It is universally effective across 5
  task types and 18 datasets.

### 4. Plain noise injection

- [[jain-2023]] **NEFTune** (ICLR 2024; arXiv): uniform noise added to embedding
  vectors during instruction fine-tuning. Llama-2-7B on Alpaca goes from 29.8% to
  64.7% on AlpacaEval. Like SFTMix, this is the generative-LM setting.

### 5. Feature-space augmentation as data

- [[devries-2017]]: noise, interpolation or *extrapolation* in a learned feature
  space, as a domain-agnostic alternative to input-space transforms.
- [[kumar-2019]] (Workshop on Deep Learning for Low-Resource NLP, 2019): six
  feature-space augmentation methods on BERT features for few-shot intent
  classification. They improve few-shot performance without hurting existing
  classes.

### Surveys

- [[feng-2021]] (Findings ACL 2021): the standard taxonomy of NLP data
  augmentation.
- **[[chen-2023-2]] (TACL 2023), main text read.**
  - *Setup:* BERT-base, **10 labelled examples per class** (100 in the appendix),
    3 data seeds, and the same hyperparameters for all methods. It compares
    token-level methods (synonym replacement, LM substitution, EDA operations,
    word replacement), round-trip translation and **hidden-space methods
    (adversarial training, Cutoff, mixup)**, supervised and semi-supervised.
  - *Result 1:* there is **no single winner, and augmentation can hurt.**
  - *Result 2:* supervised token-level methods (word replacement, random swap)
    work well overall with extremely little data.
  - *Result 3:* **on single-sentence tasks (SST-2-like), hidden-space Cutoff gave
    the biggest supervised boost.**
  - *Result 4:* round-trip translation is the most consistent in semi-supervised
    learning, and *"if the computation is limited, cutoff may be a better
    choice."*
  - *Caveat visible in its tables:* many 95% CIs are ±10–35 points at 10
    examples per class, so most method differences are not distinguishable.

## Cross-cutting findings

1. **Cheap and model-agnostic.** These methods add no generation cost. Cutoff and
   R-Drop add one extra forward pass, while adversarial methods (FreeLB, SMART)
   add a few gradient steps.
2. **Regularisation more than new information.** Unlike LLM-generated text, they
   don't add new content. They smooth the decision function, keep fine-tuned
   representations close to pretrained ones ([[aghajanyan-2020]],
   [[jiang-2020]]) or improve calibration ([[kong-2020]]). That makes them
   complements to pn2t, not substitutes. pn2t adds *content* (e.g. counterfactual
   pairs); hidden-space methods *regularise* how a small, augmented training set
   is fitted.
3. **Effects are small and noisy in the few-shot regime that matters.**
   [[chen-2023-2]]'s wide CIs and "can hurt" finding echo [[longpre-2020]] for
   input-space methods: on pretrained transformers, generic augmentation gains are
   inconsistent.
4. **Binary single-sentence classification is where Cutoff looked best**
   ([[chen-2023-2]]). That is pntx's typical `FineTuningClassifier` use (and the
   CAD IMDb benchmark task).

## Implications for pntx

1. **A natural `FineTuningClassifier` option.** One `augmentation` parameter could
   offer `None` (default) and a few cheap, well-cited regularisers:
   - `"cutoff"` ([[shen-2020]]): the supervised winner on single-sentence tasks in
     [[chen-2023-2]], with low compute;
   - `"rdrop"` ([[liang-2021]]): simplest; two dropout passes plus KL;
   - `"mixup"` (hidden-state mixup, [[sun-2020]]/[[verma-2018]]).

   All of them work inside the existing training loop on
   `model(**encoded).logits`, with no new dependencies.
2. **Mixup needs the soft-target loss from the parked idea.** Mixed labels are
   soft targets, which is exactly the "soft-target support in
   `FineTuningClassifier`" component of
   `research/ideas/soft-label-augmented-finetuning.md`. Implementing hidden mixup
   would deliver that reusable piece cheaply. The expensive part of the parked
   idea, the T5 generator from [[jin-2024]], would no longer be a prerequisite.
3. **Interaction with `class_weight` must be specified.** With mixup the target is
   a mixture, so the per-example weight must be defined, e.g. the λ-weighted class
   weight. Cutoff and R-Drop keep hard labels and are unaffected.
4. **Benchmarking.** The downstream benchmark's `--classifier finetuning` path
   (`benchmarks/pn2t/downstream.py`) is the right place to measure this: hidden
   augmentation alone, pn2t alone, and both. That requires the GPU run, which is
   not possible here because huggingface.co is unreachable. Given the wide CIs in
   [[chen-2023-2]], use multiple seeds.
5. **Out of scope for now:** SFTMix and NEFTune target generative instruction
   tuning, not `FineTuningClassifier`'s encoder classification head.

## Limitations of this survey

- Only [[chen-2023-2]] was read in the main text; the rest are summarised from
  abstracts. Its table values were not extractable reliably, so no per-task
  numbers from it are quoted.
- Several key papers are cited via arXiv DOIs. Their peer-reviewed venues
  (ICLR, ICML, NeurIPS) are noted from general knowledge where stated, and were
  not verified against the proceedings.
- 2024–2026 work on hidden-space augmentation for *encoder classifiers* appeared
  sparse in the searches (recent activity is in LLM tuning: SFTMix, NEFTune). That
  may reflect the search terms rather than the field.

## Papers added in this survey (catalog keys)

`verma-2018`, `guo-2019`, `chen-2020`, `sun-2020`, `kong-2020`, `xiao-2026`,
`miyato-2016`, `zhu-2019`, `jiang-2020`, `aghajanyan-2020`, `zhou-2021`,
`shen-2020`, `chen-2021`, `liang-2021`, `jain-2023`, `devries-2017`,
`kumar-2019`, `chen-2023-2`, `feng-2021`.
