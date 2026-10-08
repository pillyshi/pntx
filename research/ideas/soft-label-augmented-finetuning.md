# Issue Candidate: Soft-label augmentation for t2pn.FineTuningClassifier

## Status

Parked, not planned (2026-10-08). The design is recorded so it can be picked up
later. The maintainer decided not to implement it for now. The motivating method
([[jin-2024]]) has a high compute cost relative to its modest, low-resource-only
gains, and no follow-up work builds on it (see "Follow-up check").

## Motivation

[[jin-2024]] (LREC-COLING 2024) augments text-classification data *near the
decision boundary* in five steps:

1. It trains an encoder plus classifier on `(X, y)`, then a reconstruction
   decoder on the frozen encoder (T5-large).
2. It shifts each sentence's latent `z` by gradient steps toward the decision
   boundary, i.e. the point of equal class probability (`{0.5, 0.5}` for binary).
3. It decodes the shifted latent back to text with "mid-K sampling" for
   diversity.
4. It labels the new sentence with the classifier's **soft** probabilities.
5. It fine-tunes a separate downstream classifier (BERT/DeBERTa) on the original
   data plus the soft-labelled augmentations.

Soft labels are essential to the method. In its Table 3 ablation, replacing them
with the source sentence's hard label lowers performance.

Note that the downstream network itself is *not* modified. The latent-space
manipulation happens in a separate generator, and the downstream fine-tuning is
standard apart from soft targets.

## Why this belongs in t2pn, not pn2t

- A pn2t over-sampler's `fit_resample` follows imbalanced-learn's contract and
  returns hard labels. A soft target cannot travel through `y`, nor through
  `imblearn.pipeline.Pipeline`.
- The generated sentences are *ambiguous by construction*, coming from both
  classes. They are neither "positive" nor "negative", so they fall outside pn2t's
  positive-only scope.
- Generation and training must therefore happen inside one estimator's `fit`.
  That means a t2pn classifier.

## Proposed Scope (option B, chosen over a separate class)

Two pieces:

1. **Soft-target support in `FineTuningClassifier`.** The loss accepts per-sample
   target *distributions*, not only class indices. Today it is
   `CrossEntropyLoss(weight=...)` over hard labels. Hard-labelled originals become
   one-hot rows, so `class_weight` semantics stay as they are. Soft targets would
   be internal; the public `fit(X, y)` contract is unchanged.
2. **A pluggable augmenter parameter**, `FineTuningClassifier(augmenter=None)`.
   The augmenter is any object with `augment(X, y) -> (X_aug, soft_targets)`. It
   is called inside `fit` on the training data only, so it is cross-validation
   safe. As an sklearn estimator it can be nested, which allows
   `GridSearchCV(..., {"augmenter__n_steps": [...]})`.
   - First concrete augmenter: `DecisionBoundaryAugmenter`, following [[jin-2024]].
     It needs the encoder/classifier/decoder training, the gradient shift with
     `n_steps`/`step_size`, and mid-K sampling.
   - A cheaper augmenter that reuses existing pieces: generate candidates with a
     pn2t over-sampler, then soft-label them with
     `t2pn.LLMPromptingClassifier.predict_proba`. This avoids training a T5
     generator.

Why B and not `AugmentedFineTuningClassifier` (option A)? The change only touches
the training data and the loss. Model, `predict`/`predict_proba` and
`save`/`load` are identical, so this is a training option in the same spirit as
`class_weight`. A subclass would have to re-list all of `FineTuningClassifier`'s
parameters in its own `__init__`, because sklearn's `get_params` reads the
concrete signature. A strategy object also lets augmentation methods be added
without adding classes. The pn2t rule of one class per algorithm concerns
generation algorithms, not training options of a classifier.

## Acceptance Criteria (if picked up)

- `augmenter=None` reproduces current training exactly (regression test).
- The soft-target loss equals the hard-label loss on one-hot targets, and
  `class_weight` still applies.
- The augmenter is called only inside `fit`, on the data passed to `fit`.
  A recording stub verifies this under `cross_val_score`.
- `clone`/`get_params` expose nested augmenter parameters.
- `DecisionBoundaryAugmenter` passes unit tests on a tiny randomly initialised
  T5 (`AutoConfig`). huggingface.co is unreachable here, so real checkpoints
  belong in `tests/integration/`.

## Out Of Scope

- Changing the downstream network architecture.
- Curriculum augmentation, which was not significantly effective in [[jin-2024]].

## Open Questions

1. **Cost.** T5-large (about 770M parameters) is trained per dataset, which is
   impractical on the 16 GB development Mac. Is a smaller generator (t5-base) good
   enough?
2. **When does it pay off?** [[jin-2024]] reports gains with 1% and 10% of the
   training data. On several datasets the standard deviations are as large as the
   gains. Is it worth anything beyond very low-resource settings?
3. **LLM soft-labelling instead?** Would the cheap augmenter (pn2t generation plus
   `predict_proba` soft labels) capture most of the benefit without a trained
   generator?

## Follow-up check (2026-10-08)

The paper has no citations in OpenAlex (either record). Semantic Scholar lists one
citing paper, a curriculum-learning taxonomy survey (Toborek et al. 2026, arXiv
2607.18984). It cites [[jin-2024]] only as an example of a "decision-boundary
distance" difficulty proxy.

The same group's later CoBA ([[jin-2025]], arXiv 2508.21083) is a counterbias
augmentation built on semantic-triple edits. It does not cite or extend the
decision-boundary method. No follow-up that builds on the method was found.
