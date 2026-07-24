# TREC 2026 reference-answer validation design

**Status:** 400-question blind pilot complete; remaining 7,359-answer Gemini run pending a higher cost cap

**Date:** 2026-07-20

**Scope:** Validate and improve a practical reference-answer pivot for downstream candidate-response comparison.

## Objective

Use the completed Perplexity Sonar cache as the full-coverage starting point, compare it with three challenger models on a diverse sample of approximately 400 discovery questions, and select or synthesize the strongest practical reference response for each sampled question. The study is a screening and model-selection pass, not a formal gold-annotation exercise.

The resulting references will support a later pairwise judge that compares candidate model answers. They are helpful pivots, not authoritative gold answers.

## Available evidence and its limits

- The discovery set contains 7,759 questions and historical responses from 150 candidate models. It has no organizer correctness or relevance judgments.
- The development set contains 322 separate queries with organizer expert/model qrels. Those qrels can evaluate a later expert-ranking system, but they cannot validate the correctness of discovery-set reference answers.
- The Perplexity Sonar cache covers all 7,759 discovery questions and preserves answer text, citations, search results, usage, and retry history.
- Fifty Perplexity records remain unresolved after retry: three have no citations and 48 contain one or more citation markers outside the returned citation list. These are mandatory challenge cases where possible.
- Historical candidate responses can provide disagreement signals and later pairwise inputs, but they are not ground truth and must not be shown to generators during reference creation.

## Systems

The comparison includes:

1. Existing Perplexity Sonar response.
2. Gemini 3.6 Flash (stable, selected on 2026-07-21 after its release; no
   Search grounding in the blind challenger pass).
3. Qwen3.6-27B (DeepInfra exact FP8 endpoint).
4. Gemma 4 31B (`google/gemma-4-31B-it`, DeepInfra exact FP8 endpoint, no
   grounding). Google's free endpoint was canaried first but not mixed into
   this source after its quota stalled.

Exact provider/model identifiers, licenses, context limits, local-runtime compatibility, and availability must be verified before execution. Large model downloads and paid calls require an explicit recorded preflight with projected cost and storage requirements.

## Question sample

Select approximately 400 discovery questions reproducibly. The sample should cover:

- non-overlapping semantic topic clusters;
- broad explanation, factoid, multi-hop/entity, numeric, and underspecified question forms;
- practical difficulty proxies such as length, entity density, candidate disagreement, and historical refusal coverage;
- previously diagnosed ranking failures;
- time-sensitive questions; and
- the 50 unresolved Perplexity retry cases where possible.

This purposive stratification is intended to expose behavior and failure patterns. It must not be reported as an unbiased estimate of corpus-wide error prevalence. Record the selection seed, cluster, strata, and inclusion reason for every sampled question.

## Blind answer generation

- Give each challenger the same question-only task and comparable response constraints.
- Do not expose Perplexity, another challenger's response, historical candidate responses, or organizer qrels during generation.
- Ask for an answer that is correct, complete enough to serve as a reference, concise where possible, and explicit when the question is underspecified.
- Record the exact prompt, model identifier, decoding settings, elapsed time, token/usage metadata, failures, and local hardware/runtime details.
- Preserve raw outputs. Normalized views may remove citation markers or presentation artifacts but must remain derived and reproducible.

## Automatic comparison

Present responses under anonymized system labels and randomized order. Compare all sampled questions on:

- whether the response understood the question;
- likely factual correctness;
- completeness relative to what the question asks;
- usefulness as a downstream comparison pivot;
- appropriate handling of ambiguity or insufficient context; and
- material unsupported specificity.

For each question, score every answer pointwise, compare all six pairs, and
produce ordered tiers that preserve ties, cycles, unacceptable answers, and
uncertainty. A ranking may therefore be `A > B = C > D`, all equivalent, or
partially unresolved rather than a forced single winner.

Automatic comparison is a triage signal, not ground truth. Record per-dimension reasons and uncertainty rather than only the winning label.

## Codex qualitative review

Codex will review all approximately 400 sampled comparisons, considering the question and anonymized responses rather than delegating the review to the user. For every question, Codex will confirm or revise the automatic outcome and record a concise rationale and uncertainty.

Within that full review, approximately 30–50 informative disagreements will receive deeper adjudication and targeted evidence checking. Prioritize:

- conflicting factual claims;
- surprising or highly specific claims;
- time-sensitive and numeric answers;
- cases where the automatic comparison is uncertain;
- proposed combinations; and
- unresolved Perplexity records.

Detailed citation-entailment annotation remains out of scope. Consult Perplexity citations or targeted external evidence when needed to resolve a material disagreement, surprising claim, numeric claim, or time-sensitive answer. The full 400-question review includes agreement cases, but routine agreements do not require external source checking unless something appears questionable.

## Reference selection

For each sampled question:

1. Select the strongest existing response when one is clearly preferable.
2. Mark responses equivalent when differences are stylistic or immaterial.
3. Synthesize a combined response only when the components are mutually compatible and the combination clearly improves correctness or completeness.
4. Preserve `uncertain_needs_check` when no response is trustworthy.
5. Use `unanswerable_or_underspecified` rather than forcing an unsupported answer.

Store the selected reference, its source or component sources, the decision label, rationale, uncertainty, and any evidence consulted. The pilot does not automatically replace the remaining 7,359 Perplexity references; it produces evidence for that deployment decision.

## Model-selection decision

The pilot should answer a practical question: which challenger offers the best quality-throughput trade-off for future repair, regeneration, and judging work?

Report:

- automatic outcome counts and patterns by question stratum;
- Codex qualitative-review findings;
- characteristic major and minor failures;
- ambiguity and abstention behavior;
- end-to-end throughput, peak memory, setup burden, and API cost; and
- whether any challenger has a practical advantage over the already-complete
  Perplexity source, using roughly five percentage points of top-tier share or
  a clearly lower major-error rate as a decision threshold rather than a
  nominal rank difference.

The result is a pragmatic recommendation, not a statistically precise claim of model superiority.

### Pilot decision (2026-07-21)

Gemini 3.6 Flash cleared the practical decision threshold. It appeared in the
top tier on 329/400 questions (82.3%) with 28/400 major-or-fatal errors (7.0%).
Perplexity was second at 249/400 (62.3%) and 86/400 (21.5%). Qwen3.6-27B and
Gemma 4 31B were materially worse on the difficult slices. The default next
step is therefore to reuse the 400 cached Gemini answers and generate the
remaining 7,359 without grounding. Actual pilot token use projects to about
$27.35 for that remaining run at standard pricing, which exceeds the current
$10 Gemini authorization and has not been started.

### Grounding failure-rescue addendum (2026-07-21)

The 28 Gemini answers labeled major or fatal were regenerated with Google
Search enabled. All 28 calls completed; Gemini actually invoked Search on 26,
issuing 53 search queries. Manual re-review found 12 clear rescues, six partial
rescues, six cases with no rescue, one improvement caused by regeneration
without Search, and three false positives in the original serious-error
labels. The revised true-failure denominator is therefore 25, not 28.

Question-shape routing is not selective enough: the best tested high-recall
rule selected 272/400 pilot questions to catch 19/25 audited failures. The
preferred design is a two-pass disagreement route using the answers already
available: escalate when Perplexity is better than ungrounded Gemini or the
comparison is unresolved. The locked manual comparisons select 52/400 and
catch 19/25 true failures plus 14/18 clear-or-partial grounding rescues. Treat
those figures as an upper bound until a local judge reproduces the manual
labels. Validate the route on all 52 pilot questions—including the 33 selected
non-failures—before scaling it.

The sanitized analysis is in
`reports/trec_2026_gemini_grounding_failure_review_2026-07-21.md`; its
case-level ledger excludes question and answer text.

## Downstream pairwise judging

After reference selection, compare candidate answers using the question and selected reference as context. The judge must:

- decide which candidate answers the question more correctly and completely;
- allow `A`, `B`, `tie`, `both_incorrect`, and `uncertain` outcomes;
- accept valid wording and facts absent from the reference;
- permit a candidate to expose an incomplete or incorrect reference;
- avoid rewarding phrase overlap with the reference;
- randomize A/B presentation order; and
- reverse close comparisons to detect position bias.

Use pointwise filtering before pairwise judging so that pairwise comparisons focus on plausible or close candidates rather than a full tournament over all 150 models.

Organizer qrels may later evaluate whether this response-derived signal improves expert ranking on the separate 322-query development set. That downstream evaluation is not part of reference-answer validation.

## Outputs

- Reproducible 400-question sample manifest.
- Raw challenger responses with provenance and usage metadata.
- Blind automatic comparison records.
- Codex qualitative-review decisions for all approximately 400 cases.
- Deep adjudication records for approximately 30–50 informative disagreements.
- Selected or synthesized reference records with uncertainty.
- Quality, failure-pattern, throughput, memory, and cost summary.
- A deployment recommendation for Perplexity, Gemini, Qwen27, and Gemma31 roles.

## Guardrails and success criteria

- No candidate-response or qrel leakage into reference generation.
- No silent conversion of failed or uncertain cases into confident references.
- No claim that the purposive 400-question sample estimates corpus-wide correctness.
- No claim that organizer qrels validate discovery reference answers.
- Every sampled question has complete provenance or an explicit generation failure.
- Codex reviews all approximately 400 sampled comparisons; deeper evidence-backed adjudication stays near 30–50 informative disagreements unless a systematic failure requires expansion.
- Paid calls have a predeclared cap, and no unusually large model is downloaded without authorization.
- The final recommendation clearly separates verified observations, practical inferences, and unresolved unknowns.

## Out of scope

- Exhaustive citation-entailment annotation.
- Creation of human gold answers for all 7,759 questions.
- Formal population-error estimation.
- Full pairwise tournaments among 150 candidates.
- Treating the rough-pass winner as proven universally superior.
