# Four-model reference-answer review: senior evaluation advice

**Date:** 2026-07-21  
**Decision:** compare Perplexity Sonar, Gemini 3.6 Flash, Qwen3.6-27B, and Gemma 4 31B on the frozen 400-question pilot, then decide whether any challenger should be run over all 7,759 discovery questions.  
**Assessment:** the existing design is directionally sound, but it needs a richer per-question ranking protocol and explicit treatment of generation failures before it can support that deployment decision.

## Answer first

Use a **blind, two-stage evaluation**:

1. Score each answer independently (pointwise) to identify factual, relevance, completeness, uncertainty, and generation-quality failures.
2. Compare all six answer pairs (pairwise), then produce a **tiered ranking with ties** for every question. Reverse the presentation order for close, inconsistent, or consequential pairs.
3. Have Codex inspect and lock a consolidated decision for **all 400 questions**. Use targeted source checking—not blanket web search—for material factual conflicts, numeric/current claims, surprising specificity, and uncertain cases.

Do not force a winner. A defensible result can be `A = B > C > D`, all equivalent, or no trustworthy answer. Report a model as the deployment winner only if it has a practically meaningful advantage, a lower serious-error rate, and no important stratum in which it is materially worse.

The 400-question sample is a hard-case-enriched screening sample, not a probability sample. Its aggregate results can choose a practical model, but cannot be presented as an unbiased estimate of correctness across all 7,759 questions.

## What is actually available

The frozen sample manifest contains exactly 400 questions from 192 non-overlapping semantic clusters (seed 2026). It intentionally overrepresents difficult cases:

| Indicator | Questions |
|---|---:|
| Historical answer disagreement | 241 |
| High entity density | 177 |
| High refusal behavior | 166 |
| Numeric | 113 |
| Long or multi-hop | 75 |
| Unresolved Perplexity retry case | 50 |
| Time-sensitive | 22 |
| Known sparse-failure analogue | 14 |
| Underspecified | 12 |

There are 253 questions with challenge score at least 2; the mean score is 2.175. This is useful for exposing failure modes and is deliberately not corpus-representative.

The Gemini cache covers all 400 IDs. It has 396 normal stops and four length-terminated records. In each length case, Gemini spent about 1,964–1,966 hidden thinking tokens and emitted only 78–81 answer tokens; all four visible answers end mid-sentence. These are **budget failures, not quality verdicts**.

Qwen3.6-27B and Gemma 4 31B generations are not yet present. The adjudication and report files described in the implementation plan are also not implemented yet. Therefore no four-model ranking currently exists.

## Generation protocol before judging

Use the same question-only system prompt for the three challengers, with no search grounding or candidate-response/qrel leakage. Preserve Perplexity as produced, including its search-assisted provenance, because the practical comparison is between deployable answer sources. Be explicit that this is **not** a controlled comparison of intrinsic base-model knowledge: Perplexity had retrieval while the challengers did not.

Use a generous technical ceiling so reasoning tokens do not consume the answer budget:

- Gemini: retry the four truncated questions with at least an 8,192-token total output/thinking ceiling. The other 396 finished naturally and need not be regenerated merely because their ceiling was lower.
- Qwen and Gemma: allow at least 4,096 visible completion tokens (8,192 is acceptable if the API treats reasoning and visible output jointly). Keep the instruction to be concise and not pad.
- Treat a response as judgeable only after one high-cap retry for `length`, empty, malformed, or transport-failure outcomes. If the retry still fails, record `generation_failure`; do not rank it as an ordinary bad answer.
- Record raw and repaired generations separately. Never overwrite the first attempt.

A high cap is a ceiling, not a target. Verbosity must not earn a higher score, and a concise complete answer must be allowed to beat a longer one.

## Blind packet for each question

Create a deterministic packet with:

- question ID, question text, cluster, strata, and challenge indicators;
- four answers labeled with per-question anonymous IDs such as `A`, `B`, `C`, `D`;
- a random permutation derived from a saved seed;
- Perplexity answer text with citation markers removed in the first-pass view;
- finish reason and failure status hidden from the judge unless text is missing or truncated;
- a separately sealed system map and original provider metadata.

Do not show provider names, citations, token counts, response latency, formatting metadata, or historical candidate answers during the first-pass quality judgment. These can create authority, verbosity, or style bias. Reveal Perplexity citations only during targeted evidence checking.

## Stage 1: pointwise assessment

Score each anonymous answer independently before seeing pairwise judgments:

| Dimension | Scale | Rule |
|---|---:|---|
| Question understanding/relevance | 0–2 | Directly answers the intended question; handles ambiguity rather than silently choosing an interpretation |
| Factual correctness | 0–4 | 4 = no material error found; 2 = mixed/uncertain; 0 = central answer is false |
| Completeness | 0–2 | Covers the material parts required by the question, not every possible detail |
| Calibrated uncertainty | 0–1 | Qualifies genuine uncertainty and avoids fabricated precision |
| Reference usefulness | 0–1 | Clear enough to serve as a downstream comparison pivot |

Also assign severity flags: `none`, `minor`, `major`, or `fatal`, plus `unsupported_specificity`, `stale_or_time_sensitive`, `refusal`, `truncated`, `unanswerable`, and `needs_evidence` where applicable.

Correctness is a gate, not just one additive component: an answer with a major or fatal factual error cannot rank above a materially correct answer because it is longer or more complete. Numeric totals do not replace the written rationale.

## Stage 2: pairwise comparison and tiered ranking

Evaluate all six unordered pairs for each question after pointwise scoring. Pairwise outcomes are:

- `left_better`
- `right_better`
- `tie`
- `both_unacceptable`
- `uncertain_needs_evidence`

Judge correctness first, then directness/completeness, calibrated uncertainty, and reference usefulness. Accept correct information absent from other answers; do not reward wording overlap or answer length.

Randomize left/right order independently of the four-way display. Repeat the pair with reversed order when:

- the margin is close;
- the pointwise scores and pairwise result disagree;
- pairwise results form a cycle (A > B > C > A);
- either answer has a major-error or evidence-needed flag; or
- the question is numeric, time-sensitive, or selected for deep adjudication.

Resolve the six pair results into ordered **tiers**, allowing ties and partial uncertainty. Save both the tiered ranking and pairwise evidence; never silently manufacture a total order from a cycle. Examples:

- `A > B = C > D`
- `A = C > B`, with `D = generation_failure`
- `uncertain: A/B; C = D unacceptable`
- `all_unanswerable_or_underspecified`

This is stronger than pointwise-only scoring (which can hide a decisive factual difference) and more interpretable than pairwise-only judging (which can produce cycles without explaining why).

## Factuality without blanket web search

Agreement among models is evidence, but it is not ground truth; shared training data can produce shared errors. Use three evidence levels:

1. **No external check:** straightforward, stable claims where the answers agree materially and nothing looks suspicious.
2. **Targeted check:** material factual disagreement, exact dates/numbers/names, surprising specificity, time-sensitive claims, a proposed synthesis, or an adjudicator uncertainty. Check only the decisive claim, preferably using a primary/authoritative source. Perplexity citations may locate evidence but should not be trusted merely because they exist.
3. **Unresolved:** sources are inaccessible, conflicting, or the question is irreducibly ambiguous. Preserve `uncertain_needs_check` or `unanswerable_or_underspecified`.

All 22 time-sensitive questions should receive a targeted freshness check. The 50 Perplexity-unresolved cases must be inspected, but only those with a material unresolved claim require external retrieval. The four repaired Gemini cases require truncation validation, not automatically web research.

Record the exact claim checked, source URL, access date, conclusion, and whether evidence changed the initial ranking. Detailed citation-entailment annotation remains unnecessary.

## What “Codex manually reviews all 400” should mean

Codex should review every consolidated packet, not merely accept an automatic label. Work in fixed batches (for example 20 questions), and for every ID record:

- tiered ranking with ties;
- pointwise dimension scores and severity flags for all four answers;
- one concise comparative rationale naming the decisive issue;
- confidence: `high`, `medium`, or `low`;
- whether targeted evidence was checked;
- selected answer/system(s), or uncertainty/unanswerability;
- whether synthesis is required.

Codex is an independent model-family adjudicator, not a human gold annotator. Calling this “manual” means direct case-by-case Codex inspection. The final report must not call these human judgments.

Quality controls for this full review:

- lock the blinded verdict before revealing the system map;
- exact-ID validation must prove 400/400 coverage;
- repeat a hidden 10% audit set with new label/order permutations;
- investigate ranking instability rather than averaging it away;
- deeply evidence-check roughly 30–50 informative disagreements, expanding only if a systematic error appears;
- maintain a decision ledger so review can resume without changing earlier records silently.

## Aggregate model guidance

After unblinding, report at least:

- top-tier share, including fractional credit for ties;
- mean tier/rank, with ties handled explicitly;
- all six pairwise win/tie/loss rates;
- major/fatal error rate and unacceptable-answer rate;
- generation failure, refusal, and truncation rate;
- evidence-check reversal rate;
- mean response cost, latency, and output length (diagnostic only);
- every metric by question form, cluster/challenge score, numeric, time-sensitive, unresolved-Perplexity, and other prespecified strata.

Use a cluster-level bootstrap over the 192 semantic clusters for uncertainty intervals, because questions in the same cluster are not independent. Treat overlapping challenge indicators as slices, not mutually exclusive populations. Report macro-average performance across prespecified strata so a large easy slice cannot hide an important weakness.

Do **not** label these numbers “estimated corpus accuracy.” The purposive inclusion probabilities are not known. They describe performance on this frozen stress-test sample.

## Deployment decision for all 7,759

Choose a single challenger for full generation only if it satisfies all of the following:

1. It has the strongest or statistically indistinguishable top tier on the overall pilot.
2. Its major/fatal error rate is not worse than the alternatives.
3. It has no material regression in numeric, time-sensitive, multi-hop, unresolved-Perplexity, or high-disagreement strata.
4. Its advantage persists on the evidence-checked subset and after excluding generation failures.
5. Cost, latency, and operational reliability fit the three-day deadline.

Use a practical margin, not mere rank ordering: for example, require at least a 5 percentage-point top-tier advantage or a clearly lower major-error rate before paying to replace already-available Perplexity answers. If the leaders are within that margin, report them as operationally tied and choose the cheaper/reliable model. If no non-grounded challenger clearly improves on Perplexity, do **not** regenerate all 7,759 solely to obtain a nominal winner; retain Perplexity and use the best challenger for repairs or disagreement checks.

If performance crosses by stratum, a routed policy may be more defensible than one universal winner—for example, a cheap local/API model for stable factoids and Gemini or Perplexity for hard/current questions. That is a separate deployment choice and should be reported explicitly rather than hidden in a single aggregate rank.

## Required output records

For each question, retain:

```json
{
  "question_id": "...",
  "blind_order": ["A", "B", "C", "D"],
  "pointwise": {"A": {}, "B": {}, "C": {}, "D": {}},
  "pairwise": [],
  "tiers": [["A"], ["B", "C"], ["D"]],
  "confidence": "medium",
  "rationale": "...",
  "evidence_checks": [],
  "selected_systems": ["A"],
  "selected_answer": "...",
  "synthesis_required": false,
  "review_status": "locked_blind"
}
```

Only after the blinded review is locked should a derived artifact map anonymous labels back to Perplexity, Gemini, Qwen, and Gemma and calculate aggregate guidance.

## Methodological gaps this closes

- The approved design names only one winner/equivalence outcome; it does not preserve a four-way ranking.
- It does not specify how pointwise and pairwise evidence are reconciled.
- It does not protect the adjudicator from citation/provider/verbosity cues strongly enough.
- It does not define retry-before-penalty for truncated outputs.
- It does not define position-bias audits, cycles, tie handling, or aggregate uncertainty.
- It does not state a practical decision margin for spending money on all 7,759.

With these changes, the pilot is defensible as a **model-selection stress test and reference-construction pass**. It still does not become organizer ground truth, human evaluation, or a formal estimate of discovery-set accuracy.

## Evidence inspected

- `plans/trec-2026-reference-answer-validation-design.md`
- `plans/trec-2026-reference-answer-validation-implementation.md`
- `artifacts/reference_answer_validation/sample_manifest.json`
- `artifacts/reference_answer_validation/generations/gemini36_low_400.json`
- `data/llm_discovery/perplexity_full_cache.json`
- `trec_mlm/reference_generation.py`
- `reports/trec_2026_qwen_gemma_api_vs_vast_cost_2026-07-21.md`

