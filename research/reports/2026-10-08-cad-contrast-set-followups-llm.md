# Follow-up work to Kaushik et al. (2020) and Gardner et al. (2020), focusing on LLMs

*Literature survey, 2026-10-08. It was prepared for `pn2t.CounterfactualOverSampler`
(see `research/ideas/counterfactual-edit-sampler.md`).*

## Scope and method

The survey asks two questions. What has happened since counterfactually augmented
data (CAD; [[kaushik-2019]]) and contrast sets ([[gardner-2020]])? In particular,
what is known about using **large language models** to produce such minimal,
label-flipping edits?

Candidates came from three sources:

- OpenAlex keyword search (`littrail search`).
- OpenAlex citation filters: works citing either paper whose text mentions
  "large language model".
- Web search, for 2024–2026 work.

Every paper listed here was resolved to a stable identifier and added with
`littrail add-paper`. `littrail verify` passes on all 44 catalog entries. Titles and
authors were checked against arXiv/OpenAlex metadata, never taken from memory.

**Depth of reading:**

- **All 26 new papers:** abstract read.
- **3 papers read in the main text** (results and conclusions): [[sen-2023]],
  [[nguyen-2024]] and [[wang-2025]], the ones closest to pntx's design questions.
  PDFs are cached in `research/pdfs/`.
- **Everything else** is summarised from abstracts. Treat those claims as
  starting points, not evidence (littrail workflow).

## Map of the literature

### 1. Re-examining CAD itself (2020–2022, human-written edits)

- [[huang-2020]]: on SNLI, CAD does not generalise better than the same amount of
  *unaugmented* data, and it can make models less robust on challenge sets.
- [[kaushik-2020]] ("Explaining the efficacy of CAD", ICLR 2021) gives a toy
  linear-Gaussian analysis of why CAD helps out of domain.
- [[joshi-2022]] (ACL 2022): CAD identifies robust features, but it can stop models
  from learning *unperturbed* robust features and can amplify existing spurious
  correlations. The authors conclude that the limiting factor is the **lack of
  perturbation diversity**.
- [[sen-2021]] (EMNLP 2021): for sentiment, sexism and hate speech, CAD lowers
  in-domain performance but improves out-of-domain performance. Which edit types
  help depends on the construct.
- [[gardner-2021]] (EMNLP 2021, "competency problems") is the contrast-set authors'
  own follow-up. It argues that for complex language tasks *all* simple
  feature–label correlations are spurious, and it gives a statistical test for
  dataset artifacts.

### 2. Automating counterfactuals before instruction-tuned LLMs (2021–2022)

- [[wu-2021]] Polyjuice (ACL 2021): GPT-2 fine-tuned on paired sentences, with
  control codes for perturbation type (negation, lexical, ...). It does not decide
  the label itself. [[sen-2023]] quotes its authors as reporting that 40–63% of its
  outputs don't flip the label.
- [[ross-2021]] MiCE (Findings ACL 2021): minimal contrastive edits that change a
  *model's* prediction. It is for explanation, not augmentation.
- [[wen-2022]] AutoCAD (Findings EMNLP 2022): a classifier finds rationale spans
  without supervision, and controllable generation rewrites them. Its
  out-of-domain gains are comparable to human-in-the-loop CAD.
- [[howard-2022]] NeuroCounterfactuals (Findings EMNLP 2022) deliberately *relaxes*
  minimality ("loose counterfactuals" with larger, more natural edits). For
  sentiment it reports in- and out-of-domain gains, "outperforming even manually
  curated counterfactuals, under select settings".

### 3. LLM-based CAD generation (2022–2025)

- [[dixit-2022]] CORE (Findings EMNLP 2022): retrieves related excerpts from an
  unlabeled corpus and puts them into few-shot GPT-3 editing prompts. Its aim is
  **more diverse** perturbations, which [[joshi-2022]] identified as the
  bottleneck. Tested on NLI and sentiment.
- [[chen-2023]] DISCO (ACL 2023): an LLM generates phrase-level perturbations, and
  a **task-specific teacher model filters** them. Small NLI students trained on the
  result are 6 points more robust on stress tests. This is the closest published
  analogue of pntx's `verify=<classifier>`.
- [[zhou-2023]]: ChatGPT assigns *concept* labels to find concept-level spurious
  correlations, and ChatGPT-generated counterfactuals rebalance the training
  data.
- [[feder-2023]] (NeurIPS 2023): counterfactual augmentation guided by a causal
  graph, with an LLM modelling the text distribution, for clinical notes.
- [[gat-2023]]: LLM-generated counterfactuals ("change concept X, keep
  confounders fixed") used for *explaining* black-box models, plus a cheaper
  matching method that approximates them.
- [[bhattacharjee-2024]]: zero-shot LLM counterfactuals for **stress-testing** NLP
  models. This is the contrast-set use case automated with an LLM.
- [[zhang-2025]] (ACL 2025, "dually self-improved CDA"): uses the task model's
  attention to locate the causal terms, then self-improves the quality of the LLM
  counterfactuals.
- [[bae-2025]] SALAD (NAACL 2025): LLM counterfactuals are used as *negative*
  samples for contrastive learning, with structure-preserving positives.
- [[gebreegziabher-2025]] (Findings ACL 2025): CDA based on Variation Theory.
  Neuro-symbolic analysis finds the key concept dimensions, and an LLM varies the
  text along them. It mainly helps active learning at cold start.

### 4. How good are LLMs at this? Empirical studies

- [[li-2023]] ("Prompting LLMs for counterfactual generation: an empirical study",
  LREC-COLING 2024 per arXiv): LLMs are promising in most tasks but are limited by
  their own task competence. Instruction tuning and RLHF may help, and "simply
  increasing the parameter size does not yield the desired improvements". Task
  guidelines in the prompt matter, while chain-of-thought does not consistently
  help.
- **[[sen-2023]] "People Make Better Edits" (EMNLP 2023)** — main text read.
  - *Setup:* sexism and hate-speech detection, with CAD from Polyjuice, ChatGPT and
    Flan-T5, compared against human CAD.
  - *Result:* **human CAD is the most effective**, and ChatGPT CAD comes a close
    second.
  - *Failure mode — over-editing:* ChatGPT edits are *larger* than human edits
    (token edit distance 11.6 / 14.5 vs 2.4 / 6.7 for humans) and semantically
    further from the original.
  - *Failure mode — under-editing:* Flan-T5 and Polyjuice edits are *too small to
    flip the label*, which leaves mislabeled training points (low pointwise
    information).
  - *Conclusion:* mixing human and automated CAD can work, but automated CAD
    **needs manual label vetting**.
- **[[nguyen-2024]] (Findings EMNLP 2024)** — main text read.
  - *Setup:* Llama-2, Mistral, GPT-3.5 and GPT-4 on sentiment, NLI and hate speech.
  - *Quality:* LLM counterfactuals are fluent but **not minimal**.
  - *Label flips:* flip rates are comparable to humans' for sentiment but much
    lower for NLI and hate speech.
  - *Augmentation:* LLM counterfactuals can replace human ones for **sentiment**,
    but on NLI and hate speech a large gap remains (about 9–16 points), and they
    can even hurt in-distribution performance.
  - *Minimality:* keeping changes minimal correlates positively with augmentation
    value.
  - *LLMs as judges:* LLMs judging counterfactuals **agree with whatever label they
    are told, even a wrong one**. GPT-4 resists this better, but it prefers its own
    generations.
- [[nguyen-2024-2]] CEval (INLG 2024): a benchmark that unifies counterfactual and
  text-quality metrics. No method wins on both. Simple LLM prompting gives
  high-quality text but scores poorly on the counterfactual criteria.
- **[[wang-2025]] "Truth or Twist?" (2025)** — main text read.
  - *Setup:* how the choice of **judge model** affects the label-flip rate, the
    main validity metric for LLM counterfactuals. It covers four generators, 15
    judges and a user study with n = 90.
  - *Best judge:* judges *independent of the generator and not fine-tuned on the
    target data* align best with humans. Using the same model or the same family as
    the generator is worse.
  - *Gap to humans:* even the best judge **differs from human judgments by about
    23% on average**, and judges with high downstream task accuracy are not
    necessarily good judges.
  - *Conclusion:* fully automated CDA is not yet sufficient, and human oversight is
    needed.
- [[wang-2024]] (Findings EMNLP 2024) is a survey of natural-language
  counterfactual generation with an LLM-centred taxonomy (four method groups) and
  a summary of evaluation metrics. It is the best entry point for anything not
  covered here.

### 5. Contrast sets in the LLM era

There is much less direct follow-up than for CAD. Most LLM work that cites
[[gardner-2020]] uses contrast sets as an *evaluation idea*, for robustness or
shortcut testing, rather than extending the method.

- [[lin-2025]] (single-author arXiv preprint): **LLM-generated contrast sets** for
  SNLI (3,000 examples). Fine-tuning on them helps on perturbed examples and keeps
  standard accuracy. This is weak evidence (not peer-reviewed).
- [[balepur-2024]]: builds a contrast set for multiple-choice QA by *graph mining
  existing data*, explicitly to avoid expensive human annotation and biased
  model-generated data. With it, 12 LLMs are shown *not* to rely on choices-only
  shortcuts.
- [[bhattacharjee-2024]] (above) is effectively automated contrast-set
  construction with an LLM.

## Cross-cutting findings

1. **Label flipping is the core failure, and it depends on the task.** Sentiment
   is the easiest. NLI and hate speech are much harder ([[nguyen-2024]],
   [[li-2023]]).
2. **LLMs miss minimality in both directions.** Strong instruction-tuned models
   over-edit (fluent but not minimal; [[sen-2023]], [[nguyen-2024]],
   [[nguyen-2024-2]]). Weaker models under-edit and leave the label unchanged
   ([[sen-2023]]). Minimality correlates with augmentation value
   ([[nguyen-2024]]). [[howard-2022]] is the dissenting view: in some settings,
   looser edits can do better.
3. **Human CAD still wins for training data.** The gap is small for sentiment and
   large elsewhere ([[sen-2023]], [[nguyen-2024]]).
4. **Verification is unsolved, and some cheap verifiers are biased.**
   - Teacher-model filtering works in practice ([[chen-2023]]).
   - LLM judges are biased toward the label they are told ([[nguyen-2024]]).
   - The generator–judge relationship matters, and even the best automatic judge
     is about 23% off human judgments ([[wang-2025]]).
   - Two groups independently recommend human vetting ([[sen-2023]],
     [[wang-2025]]).
5. **CAD's value has limits whatever produces it.** It needs diverse perturbations
   ([[joshi-2022]]), and against a same-size unaugmented baseline the gain can
   vanish ([[huang-2020]]).

## Implications for pntx

Each implication below is tied to evidence. None has been decided yet; candidates
belong in `research/ideas/` first.

1. **The pilot's results match the literature.** The 2026-10-08 pilot found that
   label validity is the bottleneck, that long reviews get partial flips, and that
   a 3B editor rewrites rather than edits. These match findings 1 and 2. The
   sentiment-only pilot is also the *easiest* domain ([[nguyen-2024]]), so
   precision on NLI-like or hate-speech-like pools should be expected to be lower.
2. **The weakness of `verify="self"` has a known cause.** The pilot's low catch
   rate (0.26–0.37) is what two findings predict. The model is asked "is this
   positive?" right after being told to make it positive, which is the
   label-agreement bias in [[nguyen-2024]]. The judge is also the generator itself,
   which is the worst relationship in [[wang-2025]].
   - *Cheap candidate fix:* make the self-check **blind**, asking which label
     applies without stating the target. Re-run the pilot to measure the effect.
3. **The pilot's judge design follows best practice, but it needs a human
   anchor.** Using a different family, zero-shot, is the best relationship found
   in [[wang-2025]]. Even so, about 23% disagreement with humans is typical, so
   the pilot's absolute precisions carry that uncertainty.
   - *Candidate:* hand-label a small sample (say 40 candidates) once to calibrate
     the judge.
4. **The classifier verifier matches what DISCO and Wang 2025 report.**
   [[chen-2023]] shows teacher filtering helps. [[wang-2025]] finds that judges
   strong on the downstream task are not necessarily well aligned with humans.
   Both fit the pilot result that the classifier verifier is the most precise but
   keeps mainly the "easy" edits. Keeping it as an opt-in is consistent with the
   literature.
5. **Diversity is the next lever after label validity.** [[joshi-2022]] identifies
   lack of perturbation diversity as CAD's limit. Three published ways to
   diversify the edits suit `CounterfactualOverSampler`:
   - retrieval-conditioned edits ([[dixit-2022]]);
   - typed perturbations ([[wu-2021]]);
   - concept dimensions ([[gebreegziabher-2025]]).
6. **A downstream benchmark must control for data size.** Any future "does
   augmentation help" benchmark needs the **same-size unaugmented baseline** of
   [[huang-2020]]. It should also use human CAD as a reference upper bound and
   out-of-domain / counterfactual test sets ([[sen-2023]], [[nguyen-2024]]).
7. **Japanese and small open models are largely unexplored.** All studies here are
   English-only. Apart from Llama-2-7B and Mistral in [[nguyen-2024]], the models
   evaluated are much larger than pntx's local llama.cpp setting. Japanese
   counterfactual editing with ≤8B models is effectively unexplored, which is both
   a risk and an opportunity for pntx.

## Limitations of this survey

- Most claims come from abstracts. Only three papers were read in the main text.
- The "cites Gardner 2020 + LLM" query mostly returned LLM testing and red-teaming
  papers that only cite contrast sets in passing. There may be relevant contrast-set
  work that the keyword filter missed.
- 2025–2026 coverage depends on OpenAlex/arXiv indexing at the time of search.
- The search was English-only, as is the literature found.

## Papers added in this survey (catalog keys)

`huang-2020`, `kaushik-2020`, `joshi-2022`, `sen-2021`, `gardner-2021`, `wu-2021`,
`ross-2021`, `wen-2022`, `howard-2022`, `dixit-2022`, `chen-2023`, `li-2023`,
`sen-2023`, `nguyen-2024`, `nguyen-2024-2`, `wang-2024`, `gat-2023`,
`feder-2023`, `zhou-2023`, `wang-2025`, `zhang-2025`, `bae-2025`,
`bhattacharjee-2024`, `gebreegziabher-2025`, `lin-2025`, `balepur-2024`.
