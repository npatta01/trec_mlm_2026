# Pointwise Phase 0 Ranking Gate Design

**Issue:** GitHub #31, Phase 0 only
**Base:** `origin/main` at `eeea2d5cc69c0205f56806e13bb1ccf0bfa9096a`
**Audience:** technical reviewers of the TREC Million LLMs 2026 routing study

## Objective

Determine whether the audited pointwise judgments yield candidate rankings that
agree with the complete 322-query by 150-candidate organizer qrels. Compare the
existing frozen `dense_nested` control, a pointwise-only system, and a fixed
combined system with nDCG@10 and MRR. Preserve query- and candidate-level
evidence and decide whether any later pairwise pilot is justified.

No LLM API, pairwise-judging, pointwise-regeneration, or new embedding-inference
call is in scope.

## Approaches considered

1. **Frozen candidate-local E5 label transfer — selected.** Join the audited
   labels by `record_id` to the existing qrel-free 322-by-150-by-10 response
   retrieval tensor. Decompose an overall candidate prior from a genuinely
   query-dependent residual and combine fixed ranks with reciprocal-rank
   fusion (RRF).
2. **Fresh discovery-question MPNet transfer — rejected.** This is also
   qrel-free, but it adds avoidable inference and transfers labels from generic
   question neighbors rather than candidate-local response neighbors.
3. **Nested tuning or a learned router on the 322 qrels — rejected.** Either
   approach adds adaptive reuse of an already-seen development collection and
   obscures whether the raw audited labels carry signal.

## Data and integrity boundary

The evaluator accepts only the hashes frozen in
`protocols/trec_2026_pointwise_ranking_phase0.md`. The compact labels and E5
retrieval artifacts stay in private local custody and are never copied into
Git. The tracked result contains aggregate metrics, bounded slices, hashes, and
official query/candidate identifiers, but no response or reference text.

Labels and E5 records are joined strictly by `record_id`; their source ordinals
are not assumed to align. Development query IDs and normalized texts must not
overlap the pointwise source questions. Any overlap fails closed.

## Pointwise signals

For candidate `c`, the overall quality prior is the Beta(1,1)-smoothed support
rate `(supported_c + 1) / (audited_c + 2)`. Audited-response count and coverage
are reported separately so conditional quality is not confused with response
availability.

For development query `q` and candidate `c`, the local score is the unweighted
mean support label over the frozen ten candidate-local E5 response neighbors.
The routing residual is `local_qc - prior_c`. The prior vector and residual
matrix are separate artifacts.

The pointwise-only system uses equal RRF of prior rank and routing-residual rank
with `k=60`. The combined system uses equal RRF of pointwise-only rank and the
frozen `dense_nested` rank with the same `k=60`. All ties resolve by the input
rank orders and then candidate ID. No qrel selects a constant, representation,
neighbor count, weight, or tie break.

## Leakage boundary and frozen control

The E5 tensor and pointwise labels were constructed without organizer qrels.
The frozen dense predictions retain their exact nested leave-one-query-out
provenance: a target qrel cannot influence its encoder/neighborhood selection,
neighbor grades, fold prior, or ranking. Fixed pointwise and fusion operations
consume no qrels. Organizer qrels enter only after all 322 rankings exist.

The dense rankings must reproduce the tracked `dense_nested` nDCG@10, MRR, and
all per-query metrics within absolute tolerance `5e-12` before comparison.

## Metrics, slices, and decision

Use raw-gain nDCG@10 and positive-grade MRR. Report pointwise-only and combined
candidate-minus-control deltas, paired 95% percentile bootstrap intervals with
10,000 resamples and seed 31, win/tie/loss, and zero-nDCG transitions.

Prespecified query slices are organizer-qrel density (`<=5`, `6-20`, `>20`
positive candidates), frozen dense zero/nonzero nDCG, and qrel-free quartiles
of each query's arithmetic mean of its 10 frozen E5 similarities across all
candidate slots. Candidate slices are quartiles of audited-response count and
smoothed support prior. For every quartile feature, stably sort the finite
`(value, identifier)` pairs ascending, using the exact identifier string as the
secondary key, and assign zero-based sorted position `i` among `N` items to
quartile `1 + floor(4 * i / N)`. Thus the bins are deterministic, differ in
size by at most one, and run from low (1) to high (4). If `N < 4`, later bins
may be empty; retain all four labels, emit an empty slice with its count, and
report its metric as unavailable rather than borrowing or merging a bin. If
`N = 0`, all four bins are empty. Candidate-slice metrics filter rankings and
qrels to that candidate group before recomputation.

Pointwise evidence passes the Phase 0 gate only when:

1. combined minus control nDCG@10 is positive with a 95% lower bound above 0;
2. combined minus control MRR is at least `-0.005`;
3. combined has fewer than the control's 116 zero-nDCG queries; and
4. routing-only nDCG@10 is higher than prior-only nDCG@10.

Recommend a later pairwise pilot only if the gate passes and the frozen
qrel-free selector exposes at least 500 close same-tier supported pairs across
at least 100 discovery questions. Otherwise stop after Phase 0.

The selector version is the literal `pointwise-phase0-pair-selector-v1`. Its
final tie key is the SHA-256 digest of canonical UTF-8 JSON containing exactly
`candidate_ids`, `question_id`, and `selector_version`: keys are in ascending
Unicode code-point order; `candidate_ids` contains the two exact, unnormalised
candidate ID strings in ascending Unicode code-point order; string values are
unescaped except as required by JSON; and the serialization has no insignificant
whitespace. This is equivalent to Python `json.dumps(payload, ensure_ascii=False,
sort_keys=True, separators=(",", ":")).encode("utf-8")` for that payload.

Within each selector pass, rank eligible pairs by this full ascending key:
`(tier, abs(prior_a - prior_b), use_key, selector_hash)`, where `tier` is 0
for `partial/partial` and 1 for `correct/correct`, and `use_key` is exactly
`(max(use_a, use_b), use_a + use_b, use_min_id, use_max_id)`. Here `min_id`
and `max_id` are the lexical candidate-ID order, and `use_min_id` and
`use_max_id` are the current counts for those respectively ordered IDs. Recompute
the complete key after every selection. The first pass selects at most one pair
per source question; the later fill has no per-source limit. Both use the same
dynamic key and preserve the declared tier, prior-distance, use-key, then hash
precedence. Start at candidate-use cap 20. At each cap, complete every available
selection under that cap (first the one-per-source pass, then the fill), stopping
only at 500 pairs or exhaustion. If fewer than 500 pairs result, repeat at caps
21, 22, and so on until 500 pairs or no eligible unselected pair remains.

## Components and artifacts

- `trec_mlm/pointwise_ranking.py`: validation, signal construction, ranks,
  slices, and zero-call selector.
- `trec_mlm/pointwise_ranking_evaluation.py`: metrics, paired comparisons,
  decision, and sanitized receipts.
- `scripts/evaluate_pointwise_ranking_phase0.py`: local zero-call entry point.
- `artifacts/pointwise_ranking_phase0/`: ignored complete run bundle.
- `reports/trec_2026_pointwise_ranking_phase0_*`: tracked sanitized evidence.
- `research/hub/studies/pointwise-ranking-phase0.qmd`: canonical report.

## Verification

Use test-first development, real-input structural preflight, byte-identical
repeat execution, independent metric recomputation, full tests, Ruff, generated
output checks, Quarto rendering, static-site audit, secret/private-path scan,
and independent review before opening the PR.

## Self-review

- Phase 0 only, three requested systems, both requested metrics, and both query
  and candidate slices are covered.
- Overall prior and query-specific routing evidence remain distinct.
- No new effectiveness result was inspected or encoded in this design.
- No placeholder, unresolved threshold, or remote execution path remains.
