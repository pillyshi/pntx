# LLM-based augmentation of clinical text under privacy constraints, with a focus on local models

*Literature survey, 2026-10-09.*

## Scope

The setting is the one pntx's `pn2t` samplers target in healthcare:

- annotated clinical text is **scarce**;
- privacy rules (data-use agreements, institutional policy) **forbid sending
  patient text to external LLM APIs**;
- generation happens with a **locally run** open-weight LLM.

Per the maintainer's framing, the target is **partial clinical text** (sentences,
snippets or short notes) feeding a **binary classifier**. Generating whole EHRs is
out of scope, because the risk of medical or epidemiological inconsistencies grows
with the amount generated.

## Method

- **Sources:** OpenAlex keyword search, web search for 2024–2026 work and policy
  documents, and OpenAlex abstracts. The arXiv API returned 503 during the search,
  so arXiv papers were also resolved via OpenAlex.
- **Catalog:** 15 papers added with `littrail add-paper`; `verify` passes on every
  entry.
- **LLMSYN, the paper that prompted the survey,** has no DOI or OpenAlex record
  (PMLR), so it is cited here by URL and is not in the catalog.
- **Read in the main text:**
  - LLMSYN (PMLR PDF);
  - [[li-2023-3]] ("Two directions");
  - [[li-2026]] (DualAlign);
  - [[xu-2024]] (ClinGen).
- **Everything else** is summarised from abstracts; treat those claims as starting
  points.

## Why local models: the policy constraint is real

PhysioNet's guidance on using MIMIC with LLMs ("Use of MIMIC Data with Large
Language Models and Online Services", 2025-09-24,
https://physionet.org/news/post/llm-responsible-use/) states:

- The Credentialed Data Use Agreement *"explicitly prohibits sharing access to the
  data with third parties, including sending it through APIs or using it on online
  platforms"*.
- It **strongly recommends locally deployed LLMs**.
- It does not endorse any cloud service. An older PhysioNet post listed specific
  "approved" services, but the 2025 post does not.

Clinical NLP work increasingly cites this kind of constraint as the reason to run
open models locally:

- local Llama 2 for information extraction on MIMIC-IV ([[wiest-2024]]);
- local Llama-2-70B for echocardiogram report review ([[vaid-2024]]);
- an adapted local Llama-2-13B for note generation, motivated by providers
  preferring small locally hosted models ([[wang-2024-2]]);
- distilling a 70B teacher into an 8B model that runs on cheaper local hardware
  ([[woo-2025]]).

## Three ways the literature handles privacy (and where pntx sits)

Most "privacy-preserving" synthetic clinical text work does **not** use local
models. Instead it avoids putting patient-level text into the prompt, which lets
it use API models.

| Strategy | What the LLM sees | Examples | Generator |
|---|---|---|---|
| **A. Aggregate statistics plus public knowledge only** | Distributions (demographics, diagnosis frequencies, symptom trajectories) and retrieved medical knowledge; no patient text | LLMSYN (structured codes); [[li-2026]] DualAlign (statistics derived from VA EHRs) | API models are acceptable. LLMSYN used GPT-3.5 and Llama-2-70B. In LLMSYN, **7B models (Llama-2-7B, Falcon-7B) could not execute the pipeline**. |
| **B. Task definition only** | Annotation guidelines or label definitions | [[li-2023-3]] "bronze" set (GPT-4 via Azure, from guidelines only); [[tang-2023]] (ChatGPT generates labelled data, then a *local* model is fine-tuned) | API |
| **C. Real patient examples in the prompt** | Few-shot exemplars from the private pool | [[li-2023-3]] "silver" set (**local Llama 65B** labelling real MIMIC-III sentences, chosen for privacy); [[frayling-2024]] (Llama 2, zero- vs few-shot); [[kang-2024]] (open-source LLM on interview transcripts) | **Local models are required** under DUAs like PhysioNet's |

**pntx's `pn2t` samplers are strategy C.**
`HardPositiveOverSampler`, `CounterfactualOverSampler` and
`TypicalPositiveOverSampler` all put real positive (and, for two of them,
negative) exemplars in the prompt. That is exactly why pntx's local llama.cpp
backend matters for this use case, and it is the least studied cell of the table.

## Evidence closest to the scope (partial text plus binary classification)

- **[[li-2023-3]] "Two Directions" (Findings EMNLP 2023), main text read.**
  - *Task:* sentence-level **binary** detection of Alzheimer's disease signs and
    symptoms in EHRs, plus a 9-class variant. Negatives were sampled 5:1.
  - *Gold:* about 16k expert-annotated sentences.
  - *Silver ("data-to-label"):* a local Llama 65B labels public MIMIC-III
    sentences.
  - *Bronze ("label-to-data"):* GPT-4 writes notes from the annotation guideline
    only.
  - *Training:* fine-tune on synthetic data first, then on gold (two-stage).
  - *Binary accuracy:* gold-only about 0.90 → +silver 0.91 → +bronze 0.93 →
    +bronze+silver **0.94**.
  - *Multi-class:* larger gains, especially for minority classes.
  - *Caveat:* bronze sentences are **much less diverse**, with length SD 4.7
    tokens vs 12.7 for real text.
- **[[li-2026]] DualAlign (Findings ACL 2026), same group, main text read.**
  - *Method:* conditions generation on real-world aggregate statistics
    (demographics, risk factors, longitudinal symptom trajectories) to produce
    symptom-level sentences.
  - *Binary F1* (Llama-3.1-8B classifier): gold 0.72 → +bronze 0.77 → +DualAlign
    size-matched 0.79 → +DualAlign full **0.84**. Recall rose most (0.61 → 0.77).
  - *Interpretation:* grounding generation in realistic distributions beats
    unconstrained generation *at equal size*.
  - *Notes:* the main text (as read) doesn't name the generator model. The
    authors report trajectories that are sometimes temporally compressed. This
    supports the maintainer's view that long, whole-record generation is where
    inconsistencies creep in.
- **[[xu-2024]] ClinGen (Findings ACL 2024), main text read.**
  - *Diagnosis:* few-shot LLM generation of clinical text suffers from
    **distribution shift and limited diversity**: uniform style, missing the
    informal or urgent registers of real data.
  - *Fix:* compose sampled *clinical topics* (from a knowledge graph or an LLM)
    with sampled *writing styles*.
  - *Result:* +7.7–8.7% average over 8 tasks and 18 datasets. The generator was
    gpt-3.5-turbo.
- **[[li-2023-2]] (EMNLP 2023), general domain, abstract only:** the usefulness of
  LLM-synthetic data falls as task **subjectivity** rises, at both task and
  instance level. Clinical binary tasks with crisp guideline definitions should
  benefit more than subjective ones.
- **[[tang-2023]] (abstract):** applying ChatGPT directly to clinical NER/RE was
  poor and raised privacy concerns. Generating labelled synthetic data and
  fine-tuning a local model helped substantially (NER F1 23 → 64).
- **[[kweon-2024]] Asclepius (Findings ACL 2024; abstract unavailable in
  OpenAlex):** synthetic clinical notes as training data for a publicly shareable
  clinical LLM. Shareability is the same goal as pntx's
  `TypicalPositiveOverSampler`.
- **[[hiebel-2025]] (Annual Review of Biomedical Data Science, abstract):** review
  of clinical text generation and its evaluation without references, including
  synthetic text for secondary use. It is the entry point for evaluation
  methodology.

## Japanese clinical text

Little directly addresses augmentation.

- [[watanabe-2024]]: a 1B Japanese clinical/medical small language model
  (NCVC-slm-1) that beat larger models on 6 of 8 JMED-LLM tasks after
  fine-tuning. It shows small *local* Japanese clinical models are viable.
- [[arihisa-2026]] (JMIR Formative Research): Japanese psychiatric notes written
  by physicians vs LLMs for identical cases differ measurably in style. This is a
  warning that LLM-generated Japanese clinical text has its own register.
- No study was found of augmentation with local LLMs for **Japanese** clinical
  *binary classification*. This matches the gap noted in the 2026-10-08
  CAD survey (no Japanese or ≤8B-model studies).

## Privacy of synthetic clinical text

- Synthetic text is not automatically safe. Memorisation of patient-specific
  content by medical LMs is a documented concern (several 2024–2026 works found).
  A Spanish-language study and a confidentiality evaluation (Estignard et al.
  2025) also turned up, but their abstracts were not available, so they were not
  catalogued.
- Formal guarantees need differential privacy ([[xie-2024]], [[yue-2023]], already
  in the catalog). Strategies A and B above avoid exposure by construction.
  Strategy C, which pntx uses, does not, which is why `TypicalPositiveOverSampler`
  documents that it is *not* DP.

## Implications for pntx

1. **pntx's niche is real and under-served.** Strategy C (few-shot with real
   exemplars) needs local models under PhysioNet-style DUAs. Most published work
   avoids it, either by using aggregate statistics or guidelines with API models,
   or by labelling public data. A llama.cpp-native, sklearn/imblearn-compatible
   library for this cell has few direct competitors in the literature found.
2. **The maintainer's scope (partial text, binary classification) matches the
   strongest evidence.** The clearest positive results are at the *sentence
   level* with *binary* labels ([[li-2023-3]], [[li-2026]]). The documented
   failure mode is long or longitudinal generation.
3. **Model size is a hard constraint.** LLMSYN's pipeline failed with 7B models,
   and the 2026-10-08 pilot found a 3B editor unusable. The local-model papers use
   8B–70B. pntx docs should state a ≥7–8B recommendation for clinical use.
4. **Diversity is the documented weakness of synthetic clinical text**
   ([[li-2023-3]] bronze, [[xu-2024]]). ClinGen's topic × style composition, and
   Aug-PE's sub-category and tone sampling (see `research/notes/xie-2024.md`), map
   directly onto `TypicalPositiveOverSampler` and `HardPositiveOverSampler`
   prompts. This is cheap and evidence-backed, and it is the strongest idea
   candidate from this survey.
5. **Hybrid "strategy A + C" is a natural extension.** Conditioning generation on
   aggregate statistics as well as on exemplars, as LLMSYN and DualAlign do,
   improved realism at equal size in [[li-2026]]. A pntx parameter that injects
   user-supplied attribute distributions into the prompt would capture this
   without needing more patient text.
6. **Two-stage training is worth benchmarking.** [[li-2023-3]] pre-fine-tunes on
   synthetic data and then fine-tunes on gold; pntx's `fit_resample` naturally
   *mixes* them. The planned downstream benchmark
   (`research/ideas/downstream-augmentation-benchmark.md`) could compare mixing
   with two-stage training for `FineTuningClassifier`.
7. **Expect larger gains on less subjective tasks** ([[li-2023-2]]). Clinical sign
   or symptom presence fits, while sentiment-like labels are a harder case for
   synthetic data. This should inform which benchmark tasks are chosen.

## Limitations of this survey

- Four papers were read in the main text; the rest are summarised from abstracts.
- LLMSYN could not be catalogued (no DOI or OpenAlex record), so it is cited by
  URL.
- Privacy-risk papers on synthetic clinical text were found but not read and not
  catalogued.
- MedSyn (LNCS 2024, GPT-4 plus a fine-tuned LLaMA with a medical knowledge graph)
  appeared in search, but its abstract was not available in OpenAlex, so it was
  not catalogued.
- The search was English-language. Japanese-language publications (e.g. in
  Japanese medical informatics venues) were not searched.

## Papers added in this survey (catalog keys)

`tang-2023`, `li-2023-2`, `li-2023-3`, `xu-2024`, `kweon-2024`, `frayling-2024`,
`wiest-2024`, `vaid-2024`, `woo-2025`, `li-2026`, `hiebel-2025`, `kang-2024`,
`wang-2024-2`, `watanabe-2024`, `arihisa-2026`.
Not in the catalog: LLMSYN (Hao, He, Ho, MLHC 2024, PMLR 252;
https://proceedings.mlr.press/v252/hao24a.html).
