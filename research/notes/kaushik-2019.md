# kaushik-2019

## Citation

Kaushik, D., Hovy, E., & Lipton, Z. C. (2020). *Learning the Difference that
Makes a Difference with Counterfactually-Augmented Data.* ICLR 2020.
arXiv:1909.12434 (DOI 10.48550/arxiv.1909.12434).

## Why It Matters For This Project

This is the clearest evidence that adding *label-flipped, minimally edited*
examples to training data makes classifiers less reliant on spurious
features. That is close to OverSampler's goal (generate positives that
defeat a shallow classifier) and is direct prior art for a generation mode
OverSampler does not yet have: edit a negative into a positive. It also
grounds the "boundary features" idea causally: the features that matter are
the ones you *must* change to change the label.

## Method

- Humans (Mechanical Turk) revise each document so that it (i) accords with
  the counterfactual label, (ii) stays coherent, (iii) avoids unnecessary
  changes — a "least action" principle.
- Sentiment: 2.5k IMDb reviews (longest 20% filtered out, 50:50 balance),
  each revised by two workers; one revision chosen at random. Splits
  1707/245/488. ~2% of revisions rejected on manual inspection.
- NLI (SNLI): workers revise the premise (RP) or hypothesis (RH) toward each
  of the two counterfactual classes; a separate set of workers verifies
  labels by majority vote; ~9% discarded.
- Cost: ~5 min per revised review; $10,778 in total.
- Models: SVM / Naive Bayes on TF-IDF, Bi-LSTM, ELMo-LSTM, fine-tuned BERT.
- Eight common sentiment edit patterns identified (Table 2): recasting fact
  as hoped for, suggesting sarcasm, inserting/replacing modifiers, inserting
  phrases, diminishing via qualifiers, differing perspectives, changing
  ratings.

## Relevant Findings

- Classifiers trained on original data fail on revised data and vice versa
  (e.g. SVM 80.0 on original vs 51.0 on revised; Bi-LSTM 79.3 → 55.7).
  BERT is more robust (cited as a likely benefit of pretraining).
- Training on **original + revised** (3.4k) performs well on both, within
  ~3 points of a model trained on the same amount of original data alone.
- Adding 1.7k revised reviews to 19k original ones substantially improves
  accuracy on revised data while slightly improving original-test accuracy.
- Counterfactually-augmented training generalises better out of domain
  (Amazon, Twitter, Yelp) than equal amounts of original data in almost all
  cases.
- Spurious features (e.g. genre words like "horror", "romantic") stop being
  predictive once the paired data is combined, because humans never edit
  them.

## Limitations

- All edits are human-written; the paper does not test automated or
  LLM-generated counterfactuals.
- Small scale (1.7k reviews); long reviews were filtered out.
- Inter-editor agreement on *which spans* to edit is low (Jaccard ≈ 0.11–0.42 by length bucket, ≈ 0.14–0.26 overall,
  falling with review length), so "the minimal edit" isn't unique.
- The benefit relies on *pairs* (original + its counterfactual). OverSampler
  generates fresh positives with no paired negative, so the
  spurious-correlation-breaking mechanism doesn't transfer automatically.

## Actionable Findings

1. **Paired, minimal-edit generation mode** for `pn2t`: take negative
   exemplars as sources and ask the LLM to revise each minimally so the
   positive label applies (same three constraints as the paper's
   instructions). Output is (negative, edited positive) pairs. This is the
   mechanism the paper shows works; OverSampler's current "generate new
   hard positives" is a different, untested mechanism.
2. **Prompt material.** The eight edit categories in Table 2 are a ready-made
   checklist for a minimal-edit prompt or for auditing generated positives
   (e.g. "suggesting sarcasm" and "diminishing via qualifiers" are typical
   hard cases for lexical classifiers).
3. **Benchmark design.** To test whether OverSampler helps, compare a
   classifier trained on original vs original + OverSampler output on a
   *counterfactual* test set (the released IMDb revised split fits),
   mirroring Table 5. This also checks for [[gardner-2020]]-style local
   consistency.
4. **Label verification.** The paper needed a verification pass (~2%/9%
   rejections) even with human editors. An LLM-edit mode should validate
   flipped labels (e.g. score with `t2pn.LLMPromptingClassifier`) before
   accepting them, which raises the model-in-the-loop caveat from
   [[gardner-2020]].
5. **Scope note.** A paired mode naturally produces *negatives* too
   (positive → negative edits). CLAUDE.md currently scopes negative-side
   generation out of `pn2t`, so any such mode would need to stay positive-only
   or the scope would need revisiting.

## Issue Candidate Impact

Strongest idea candidate from the three notes: "counterfactual minimal-edit
mode for pn2t" (items 1, 2, 4), plus a benchmark candidate (item 3).
Both should go to `research/ideas/` before becoming issues.

## Sources Checked

- arXiv:1909.12434v2 full text (ICLR 2020 camera-ready; read §1–§5,
  Tables 1–7), downloaded 2026-10-07 to `research/pdfs/kaushik-2019.pdf`.
- Compared against `src/pntx/pn2t/prompts.py` and CLAUDE.md scope section at
  commit 9dc383d.
