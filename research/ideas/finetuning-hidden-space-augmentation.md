# Issue Candidate: Hidden-space augmentation option for t2pn.FineTuningClassifier

## Status

Draft (2026-10-09). Not started.

## Motivation

`t2pn.FineTuningClassifier` fine-tunes a pretrained encoder on small,
often imbalanced pools, exactly the regime where fine-tuning overfits. A family
of cheap, well-cited methods augments data *inside* the network during
fine-tuning, by erasing, perturbing or mixing embeddings and hidden states, with
no text generation and no new dependency (survey:
`research/reports/2026-10-09-hidden-space-augmentation.md`).

These methods complement pn2t rather than compete with it. pn2t adds *content*
(e.g. counterfactual pairs, the only generated condition that helped in the
downstream benchmark). Hidden-space methods *regularise* how the small, possibly
augmented training set is fitted.

## Evidence

- [[chen-2023-2]] (TACL 2023; main text read; BERT-base, 10 labelled examples
  per class):
  - No method wins everywhere, and augmentation can hurt.
  - **On single-sentence tasks, hidden-space Cutoff gave the largest supervised
    boost.**
  - "If the computation is limited, cutoff may be a better choice."
  - Caveat: 95% CIs often span ±10–35 points at this data size.
- [[shen-2020]] Cutoff: token, feature or span erasure in the embedding matrix
  plus a Jensen–Shannon consistency loss. Near-zero extra compute compared with
  adversarial training.
- [[liang-2021]] R-Drop: two dropout forward passes plus bidirectional KL;
  broadly effective across 18 datasets.
- [[sun-2020]] Mixup-Transformer and [[verma-2018]] Manifold Mixup: interpolating
  hidden states (and labels) during BERT-style fine-tuning. [[kong-2020]] adds
  calibration benefits.
- [[longpre-2020]]: input-space augmentation is inconsistent on pretrained
  transformers. This argues for measuring the effect, not assuming it.

## Proposed Scope

A new `FineTuningClassifier` parameter
`augmentation: None | "cutoff" | "rdrop" | "mixup" = None`, plus one strength
parameter `augmentation_strength: float` whose meaning depends on the method
(defaults taken from the respective papers). The default `None` is
byte-for-byte today's training (regression test).

All three methods live inside the existing training step, which already computes
the loss itself from `model(**encoded).logits` with
`torch.nn.CrossEntropyLoss(weight=...)`:

- **`"cutoff"`** ([[shen-2020]]): token cutoff, which zeroes the embedding rows of
  a random share of non-special tokens.
  - Feed `inputs_embeds` instead of `input_ids` for the augmented view, using
    `model.get_input_embeddings()`. Loss = CE(original) + CE(cutoff view) +
    β·JS(original, cutoff).
  - Labels stay hard, so `class_weight` applies unchanged.
  - Start with token cutoff only; span and feature cutoff can follow.
- **`"rdrop"`** ([[liang-2021]]): two forward passes in train mode (different
  dropout masks). Loss = mean CE of the two + α·symmetric KL.
  - The simplest option, with no change to the inputs.
  - Labels stay hard.
- **`"mixup"`** ([[sun-2020]]): mix pooled representations, or a chosen layer's
  hidden states, of shuffled pairs within the batch with λ ~ Beta(a, a), and mix
  the one-hot targets the same way.
  - This **requires a soft-target loss**, the same component as the parked
    `research/ideas/soft-label-augmented-finetuning.md`. Implementing mixup
    delivers that reusable piece without the expensive generator.
  - `class_weight` must be defined for mixed targets. Proposal: the per-example
    weight is λ·w(y_i) + (1−λ)·w(y_j), i.e. the class weight under the mixed
    target distribution, and it reduces to today's weighting when λ ∈ {0, 1}.

Cost to document: R-Drop and Cutoff need about 2 forward passes per step;
mixup needs about 1.

`fp16`/`bf16` and `gradient_accumulation_steps` must keep working (the losses
are computed inside the same autocast/scaler path).

## Acceptance Criteria

- `augmentation=None` reproduces current training exactly: same seed, same
  weights after one step.
- Unit tests on the tiny random-init model already used for
  `FineTuningClassifier` (`AutoConfig`, no download):
  - each mode trains and predicts;
  - the loss of each mode on a fixed batch matches a hand-computed reference;
  - mixup's soft-target CE equals hard CE when λ ∈ {0, 1}, and `class_weight`
    behaves as specified;
  - invalid `augmentation` or `augmentation_strength` values raise `ValueError`;
  - `get_params`/`clone`/`save`/`load` round-trip the new parameters.
- **Effect measured before any README claim.** Use the downstream benchmark's
  `--classifier finetuning` path on a GPU machine (huggingface.co is unreachable
  here), with ≥3 seeds, comparing:
  - hidden augmentation alone;
  - pn2t augmentation alone (reusing the saved generations);
  - both together;
  - neither.

## Out Of Scope

- Adversarial methods (FreeLB, SMART, R3F). They are effective but need extra
  gradient steps or trust-region optimisation, i.e. more complexity for this
  first step.
- Generative-LM variants (NEFTune, SFTMix); `FineTuningClassifier` is an encoder
  classifier.
- Semi-supervised use with unlabeled data (MixText-style). pntx's API has no
  unlabeled-data channel.

## Open Questions

1. **Mixup layer.** Mix at the pooled `[CLS]` representation (simplest, before
   the classifier head) or at an intermediate layer (MixText style)? The latter
   needs model-specific hooks, which works against `AutoModel` neutrality.
2. **Defaults.** Which defaults for `augmentation_strength` (cutoff ratio, R-Drop
   α, mixup Beta parameter) hold up across BERT, RoBERTa, DeBERTa and
   multilingual checkpoints?
3. **Japanese.** Does token cutoff behave differently with Japanese subword
   tokenisation (many short subwords per word)? Should it be span cutoff by
   default for Japanese?
