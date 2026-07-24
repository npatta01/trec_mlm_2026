# TREC 2026 four-model reference-answer review

## Technical summary

**Use Gemini 3.6 Flash as the default generator for the remaining discovery questions.** In the frozen 400-question blind review, Gemini appeared in the top tier on 329 questions (82.3%) and had 28 major-or-fatal errors (7.0%). The next-best existing source, Perplexity Sonar, appeared in the top tier on 249 questions (62.3%) and had 86 serious errors (21.5%). Qwen3.6-27B and Gemma 4 31B were materially worse on the deliberately difficult portion of the sample.

The result is large enough to change the plan. The 20.0 percentage-point top-tier advantage and 14.5 percentage-point serious-error advantage over Perplexity both exceed the predeclared five-point practical threshold. Gemini also remained first in every challenge slice inspected, including time-sensitive, numeric, long/multi-hop, and known-ranking-failure analogue questions.

The cheapest sensible next run is not the cheapest model. Reusing the 400 cached Gemini answers, the remaining 7,359 questions are projected to cost approximately **$27.35 at standard pricing**, based on actual pilot tokens. This exceeds the currently authorized $10 Gemini cap, so the full run has not been started.

## Gemini separates most clearly on the hard questions

| System | Top-tier questions | Fractional top credit | Mean correctness (0–4) | Major/fatal errors | Mean tier |
|---|---:|---:|---:|---:|---:|
| Gemini 3.6 Flash | 329 / 400 (82.3%) | 36.8% | 3.73 | 28 / 400 (7.0%) | 1.21 |
| Perplexity Sonar | 249 / 400 (62.3%) | 25.8% | 3.30 | 86 / 400 (21.5%) | 1.50 |
| Qwen3.6-27B | 235 / 400 (58.8%) | 20.8% | 2.96 | 125 / 400 (31.3%) | 1.61 |
| Gemma 4 31B | 203 / 400 (50.8%) | 16.6% | 2.79 | 145 / 400 (36.3%) | 1.72 |

“Top tier” preserves ties. There were 118 questions where all four systems were tied and only 100 with a sole winner. Sole wins were Gemini 49, Perplexity 32, Qwen 12, and Gemma 7. Fractional credit divides one point among every system tied in the top tier.

The sample intentionally overweights difficult questions: 253 of 400 have at least two challenge indicators. The following table therefore describes this validation sample, not estimated corpus-wide accuracy.

| Slice | Questions | Gemini top tier / serious error | Perplexity | Qwen | Gemma |
|---|---:|---:|---:|---:|---:|
| At least two challenge indicators | 253 | 87.0% / 7.5% | 66.0% / 22.5% | 51.0% / 40.7% | 45.1% / 45.8% |
| Long or multi-hop | 75 | 90.7% / 8.0% | 76.0% / 17.3% | 57.3% / 40.0% | 54.7% / 42.7% |
| Numeric | 113 | 86.7% / 6.2% | 64.6% / 24.8% | 48.7% / 44.2% | 46.9% / 45.1% |
| Time-sensitive | 22 | 86.4% / 9.1% | 54.5% / 36.4% | 31.8% / 59.1% | 40.9% / 36.4% |
| Known failure analogues | 14 | 92.9% / 7.1% | 71.4% / 14.3% | 35.7% / 50.0% | 50.0% / 50.0% |
| Fewer than two challenge indicators | 147 | 74.1% / 6.1% | 55.8% / 19.7% | 72.1% / 15.0% | 60.5% / 19.7% |

Qwen was competitive on the easier slice but collapsed on difficult entity-linking and numeric questions. Its four generation failures were all on hard, multi-hop questions. This is a poor failure shape for the intended use even before cost and throughput are considered.

## What the review found

The recurring failure was not style. It was confident resolution of ambiguous or multi-hop entities to the wrong person, date, place, title, institution, or count. Typical disagreements included identifying an organization from a long chain of clues, distinguishing historical from current facts, and answering questions whose missing antecedent made a unique answer impossible.

Gemini more often did one of three useful things: resolved the chain correctly, identified a false premise, or asked for missing context. Perplexity was often strong, especially when its search result matched the intended source, but grounding did not prevent wrong entity selection, stale/current-time confusion, or synthesis errors. Qwen sometimes spent very large reasoning budgets and still produced the wrong entity. Gemma was inexpensive and fluent but materially less reliable.

A 43-question higher-risk queue combined low-confidence reviews, uncertain pairwise outcomes, time-sensitive questions, and generation failures. Even within that queue Gemini remained first: 67.4% top-tier with a 30.2% serious-error rate, versus Perplexity at 46.5% / 51.2%, Qwen at 23.3% / 72.1%, and Gemma at 25.6% / 60.5%. Those absolute error rates are a warning: none of the four systems should be treated as gold on the hardest questions.

## Scope and method

- Population: 7,759 discovery questions with existing Perplexity answers and no organizer correctness judgments.
- Sample: 400 questions drawn across 192 non-overlapping semantic clusters, with deliberate coverage of question types, time sensitivity, numeric questions, entity density, historical response disagreement/refusal, and known ranking-failure analogues.
- Systems: existing grounded Perplexity Sonar answers; Gemini 3.6 Flash with low thinking and no grounding; DeepInfra Qwen3.6-27B; and DeepInfra Gemma 4 31B.
- Output allowance: 16,384 tokens by default. Six Qwen hard cases received bounded 32,768-token retries; two recovered and four remained explicit failures. No response was silently omitted.
- Review: deterministic per-question A/B/C/D blinding; pointwise correctness/completeness/usefulness scores; all six pairwise comparisons; tie-preserving tiers; system map opened only after all 400 decisions were locked.
- Human/agent review: an independent advisor completed the first pass over all 400. Codex then inspected all 400 and revised 34 judgments, mostly replacing unsupported fine-grained ordering with ties. The final ledger contains exactly 400 validated records.

The review did not use candidate/expert answers or organizer qrels. Historical responses were used only as sampling difficulty signals, never as correctness labels. This prevents leakage into the later candidate-ranking stage.

## Actual generation cost and operational behavior

| System | Completed | Output or completion tokens | Pilot cost | Projected 7,759-question cost | Operational note |
|---|---:|---:|---:|---:|---|
| Gemini 3.6 Flash | 400 / 400 | 92,018 answer + 98,623 thinking | $1.49 | $28.84 standard | Zero final failures; four initial length stops repaired under the 16K ceiling |
| Qwen3.6-27B | 396 / 400 | 628,249 | $2.02 | $39.26 | Four hard-case failures after bounded retries; unusually high reasoning-token use |
| Gemma 4 31B | 400 / 400 | 81,643 | $0.037 | $0.72 | Cheapest by far, but 36.3% serious-error rate |

Costs use the verified listed rates and actual token counts from these caches. Projections assume the sample's average token use, so they are planning estimates rather than billing guarantees. The Gemini projection for only the 7,359 uncached questions is about $27.35 standard. A Batch/Flex run could be roughly half that price, but the existing runner is synchronous and the standard path is the lowest-risk way to finish within three days.

## Limitations and robustness

1. **There is no discovery gold set.** These are structured qualitative judgments, not organizer relevance or answer-correctness labels.
2. **The sample is purposive.** It covers all 192 clusters and overweights hard cases; its rates must not be presented as unbiased population accuracy.
3. **The deepest source verification is incomplete.** The 43-case queue was reviewed for internal consistency and answer conflict, but no new web evidence pass was performed. Existing Perplexity citations remain available for targeted resolution.
4. **The evaluator is not independent of model-family knowledge.** Blinding removes provider-name bias and the A/B labels are balanced, but it does not create ground truth.
5. **A reference answer is a pivot, not an oracle.** Pairwise candidate judgments should allow ties and uncertainty, and should escalate material disagreement rather than blindly follow the selected reference.

The result is nevertheless robust enough for the immediate model-selection decision: Gemini's advantage is large overall, larger on most difficult slices, and paired with fewer operational failures than Qwen.

## Recommended next steps

1. Raise the Gemini authorization to a **$35 standard-API cap** and generate the remaining 7,359 answers without grounding, reusing the 400 cached answers.
2. Use Gemini as the default reference. Retain the existing Perplexity answer and citations beside it rather than overwriting them.
3. Route time-sensitive questions, low-confidence questions, and material Gemini–Perplexity disagreements to selective evidence checking or synthesis. Do not pay for blanket search.
4. Calibrate the later pointwise/pairwise judge on the 400 reviewed questions. Randomize A/B order, reverse close comparisons, preserve ties, and track position instability.
5. Treat the 43 higher-risk questions as the first validation set for the selective escalation rule.

## Gap versus prior TREC teams

The repository has already tried several ideas seen in TREC 2025: sparse and
dense query-neighbor retrieval, response-level BM25/RM3 retrieval, semantic
candidate profiles, and reciprocal-rank fusion. The strongest untried parts of
the prior-team playbook are downstream of this reference work:

- **Weak labels from a strong answer judge.** Prior teams used LLM judges or
  rerankers to create pseudo-supervision. We have calibrated the reference
  generator, but have not yet scored all 150 candidate answers against those
  references.
- **Efficient pairwise ranking.** Prior systems used pairwise or
  Quicksort-like comparisons rather than a full 150-model tournament. Our
  tie-preserving, order-reversal judge is designed but not yet run.
- **Confidence/log-probability features.** Other teams used average
  likelihood, variance, and high-percentile confidence as model-capability
  signals. Those features have not been integrated into the current profile
  ranker.
- **Learned hybrid routers.** We have hand-designed retrieval and fusion, but
  have not trained a fold-local ranker that combines retrieval, judge,
  confidence, and global-prior features.
- **Neural response reranking.** The repository has a response BM25/RM3
  baseline, but has not completed the stronger response-retrieval plus neural
  reranker path used by competitive teams.

The immediate gap is therefore not another reference model. It is converting
the selected references into leakage-safe pointwise and selective pairwise
signals, then testing whether those signals improve model ranking on held-out
development queries.

## Further questions

- Can a cheap local or Flash-Lite classifier identify the escalation set without missing Gemini's serious errors?
- Does combining Gemini with Perplexity reduce the hard-queue error rate, or merely produce more persuasive unsupported synthesis?
- How stable are pairwise candidate rankings when the reference is replaced by another top-tier answer or omitted entirely?
- What fraction of the full 7,759 questions triggers the escalation rule, and does that keep the complete reference pipeline below the time and cost budget?
