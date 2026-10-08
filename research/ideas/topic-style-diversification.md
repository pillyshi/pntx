# Issue Candidate: Topic × style diversification for pn2t generation

## Status

Draft (2026-10-09). Not started.

## Motivation

The most consistently documented weakness of LLM-generated training text is
**low diversity**. Generations collapse onto a typical style and a narrow set of
situations, and the synthetic distribution drifts from the real one.

pntx's `TypicalPositiveOverSampler` and `HardPositiveOverSampler` generate from
the same prompt in every batch, varying only the sampled exemplars and the
temperature. They are exposed to exactly this failure.

The literature has a cheap, evidence-backed remedy. **Sample an explicit topic and
an explicit writing style per batch, and put them in the prompt**, so that diversity
comes from composition rather than from temperature alone. This is the strongest
idea candidate from the 2026-10-09 survey
(`research/reports/2026-10-09-local-llm-clinical-text-augmentation.md`). It also
applies outside the clinical domain.

## Evidence

- [[xu-2024]] ClinGen (Findings ACL 2024; main text read):
  - It diagnoses few-shot LLM generation of clinical text as suffering
    **distribution shift and limited diversity**. Synthetic text sticks to a
    uniform style, while real data includes urgent and informal registers.
  - Composing sampled *clinical topics* (from a knowledge graph or from an LLM)
    with sampled *writing styles* gave +7.7–8.7% on average across 8 tasks and
    18 datasets.
- [[xie-2024]] Aug-PE (ICML 2024; main text read; see `research/notes/xie-2024.md`
  item 3) reaches the same mechanism independently for DP synthetic text:
  - a sub-category keyword ("pseudo-class") sampled per generation;
  - a random tone phrase per generation;
  - an adaptive target length.
- [[li-2023-3]] (Findings EMNLP 2023; main text read): synthetic sentences
  generated from annotation guidelines were measurably less diverse than real
  ones, with length SD 4.7 vs 12.7 tokens.
- [[joshi-2022]] (ACL 2022): lack of perturbation diversity is what limits
  counterfactual augmentation's out-of-domain gains.
- [[li-2023-2]] (EMNLP 2023): synthetic-data usefulness depends on the task.
  Diversity will not rescue highly subjective tasks, so measure, don't assume.

## Proposed Scope

**Applies to** `TypicalPositiveOverSampler` and `HardPositiveOverSampler`.

**Does not apply to** `CounterfactualOverSampler`. A minimal edit must keep the
pivot's topic and style; varying them contradicts minimality. Its diversity
lever is pivot selection and perturbation type (see
`research/ideas/counterfactual-edit-sampler.md`).

New constructor parameters, identical in both samplers:

- `topics: list[str] | "auto" | None = None`
- `styles: list[str] | "auto" | None = None`

How each value behaves:

- **`None` (default): current behaviour, unchanged.**
- **A list:** each batch samples one topic and/or one style uniformly with
  `random_state` and injects it into the user message (e.g. "Situation: … /
  Writing style: …"). User-supplied lists are the **recommended setting for
  clinical or other private data**: the user controls exactly what enters the
  prompt, and the lists can come from public vocabularies (a terminology subset,
  note types such as triage note / discharge summary / nursing note) rather than
  from patient text. This matches strategies A and B in the clinical survey.
- **`"auto"`:** one extra structured LLM call at the start of `fit_resample`
  proposes a list of topics or styles from the sampled exemplars, with
  grammar-constrained output validated by pydantic.
  - Each batch then samples from that list.
  - The proposed lists are stored in `generation_result_` (new fields with
    defaults, so old saved JSON still loads) for auditing.
  - For `TypicalPositiveOverSampler`, `"auto"` topics are a **privacy risk**: a
    topic derived from private exemplars can carry identifying specifics. So
    `"auto"` topics must be described as general categories in the prompt (the
    same rule as `style_features`/`content_features`), and each proposed topic is
    passed through the existing `contains_verbatim_span` check against the
    positive pool. Topics that fail are dropped.

**Prompt changes stay in `pn2t/prompts.py`**, which users can replace:
`build_user_message` and `build_synthetic_user_message` get optional
`topic=None, style=None` arguments. An empty value renders today's message
exactly, so existing custom prompts keep working.

Interaction with existing behaviour:

- `HardPositiveOverSampler`: the topic and style apply only to the *generated*
  texts. The boundary-feature analysis still sees the sampled exemplars,
  unchanged.
- Topic and style are sampled **per batch**, not per item. This keeps one prompt
  per batch, consistent with the current batching and KV-cache behaviour.

## Acceptance Criteria

- `topics=None, styles=None` produces byte-identical prompts to the current
  implementation (regression test).
- With lists, prompts contain exactly one sampled topic and style per batch.
  Sampling is deterministic under `random_state`, and every list entry is
  eventually used over enough batches.
- With `"auto"`:
  - the proposal call uses the shared backend (no new model load);
  - proposed lists are validated and stored in `generation_result_`;
  - for `TypicalPositiveOverSampler`, topics containing verbatim spans from the
    positive pool are rejected (Japanese and English test cases);
  - saved results from earlier versions still load.
- `CounterfactualOverSampler` is untouched.
- **Measured effect before any README claim.** Run the existing pn2t pilot
  infrastructure, or the planned downstream benchmark
  (`research/ideas/downstream-augmentation-benchmark.md`), with diversity
  `None` vs list vs `"auto"`, and report:
  - diversity: length SD vs real data, distinct-n, mean pairwise embedding
    distance;
  - fidelity to the real distribution, e.g. embedding-space distance;
  - label validity: judge precision, as in the pilot, so that diversity is not
    bought with off-label text;
  - downstream classifier effect, when the downstream benchmark exists.

## Out Of Scope

- Knowledge-graph integration. ClinGen used a clinical KG for topics; pntx
  would take user-supplied lists instead, with no KG dependency.
- Per-item topic and style (one call per item); it breaks batching.
- Adaptive target length (the third Aug-PE trick). It could be a follow-up using
  the same mechanism.

## Open Questions

1. **Should `"auto"` topics be offered for `TypicalPositiveOverSampler` at all**,
   or should that sampler accept only user-supplied lists, given its
   privacy-oriented purpose?
2. **Does topic diversification fight `HardPositiveOverSampler`'s objective?**
   Pushing generations toward varied topics might pull them away from the
   decision boundary the boundary-feature analysis targets. The judge-precision
   and "valid & hard" metrics from the pilot can check this.
3. **How long should the lists be?** Too short gives little diversity; too long
   and topics may stray outside the real distribution (fidelity loss).
   ClinGen's topic/style counts are a starting point.
4. **Japanese.** Are LLM-proposed styles meaningful for Japanese clinical text,
   given that physician and LLM note styles differ measurably ([[arihisa-2026]])?
   User-supplied note-type lists may be the safer default there.
