# Source-query-first v1 owner decision and revisit policy

Status: **concluded v1 experiment; archived for reproducibility; not selected for
default or submission use**

Owner decision date: 2026-07-11

This record is intentionally separate from the locked protocol, result snapshot, and
reader report. Those files preserve what was specified and observed before the owner
decision. This file records what merging the completed experiment means and when a
future experiment may revisit it.

The locked snapshot therefore continues to record that an owner decision was still
required: that field is part of the immutable pre-decision receipt. This separate,
later record supplies the requested decision rather than rewriting the receipt.

The frozen v1 evidence includes
`protocols/trec_2026_source_query_first.md`,
`reports/trec_2026_source_query_first_results.json`,
`reports/trec_2026_source_query_first.artifact.json`, and
`reports/trec_2026_source_query_first.html`. Corrections must add an explicit
invalidation or supersession notice; they must not silently edit these files in place.

## Owner decision

Merge the source-query-first v1 implementation, tests, protocol, and evidence as a
frozen research archive. The merge does **not** advance source-query-first into the
default ranking path, select it for a submission, or replace SumScore as the current
response-search control.

The implementation remains opt-in. It runs only when a user explicitly invokes
`trec-mlm source-query-first`; its experiment module is imported lazily from that
command and is not used by the other ranking commands.

The decision-relevant implementation was independently reviewed at commit
`bb573d47b0375a012d519622a891cdec9bea9893`. The locked preregistration was committed
as `459e545`; its qrel-free feasibility amendment was committed as `9a91362` before
effectiveness scoring. Both official runs reproduced decision receipt
`80e3b904e4f02bb0193bf2e01c039049a0a45e971437565aaec1a7f8212a68cb`.

## Locked confirmatory finding

Source-query-first v1 scored 0.2033 nDCG@10 and 0.2482 MRR on the 322 nested held-out
development queries. SumScore scored 0.2774 and 0.3175. The paired nDCG@10 difference
was -0.0741 with a 95% interval of [-0.1040, -0.0449]. Source-query-first passed none
of the four preregistered effectiveness and response-volume gates.

The prespecified paired comparisons recorded 57 nDCG@10 wins, 142 ties, and 123
losses against SumScore. The corresponding MRR counts were 104 wins, 42 ties, and 176
losses.

The confirmatory decision is therefore scoped as follows:

> Do not select source-query-first v1, under the Issue #7 protocol and current
> evidence, as the default or submission method. Retain SumScore as the current
> response-search control.

This is a decision about one frozen version and evidence set. It is not a permanent
prohibition on source-query-first ideas.

## Post-hoc joint and case diagnostic

The following cross-system, cross-metric, oracle, and subgroup observations were
computed only after the locked decision and are exploratory. They cannot change the
four-gate result:

- Source-query-first strictly beat SumScore, Top10Sum, dense, and TF-IDF together on
  24 queries.
- In 58 cases MRR improved against SumScore while nDCG@10 tied at zero; the first
  relevant model moved upward but remained outside the top 10.
- A hindsight per-query maximum of source-query-first and SumScore would score 0.3227
  nDCG@10 versus 0.2774 for SumScore. That number demonstrates complementarity on
  these 322 judged queries but is not deployable because choosing the winner uses
  judged outcomes.
- Near-clone comparison and relationship templates were plausible success cases, but
  close-template failures showed that source-query similarity alone is not a safe
  router.
- No examined qrel-free signal—including query length, retrieval concentration,
  missing-response share, or source-match-score thresholds—validated a positive
  segment. The apparent pockets were small, uncertain, or outlier-sensitive.

These values are reproducible from `artifacts/source_query_first/per_query.tsv`,
`artifacts/source_query_first/query_diagnostics.tsv`, and
`artifacts/source_query_first/selected_sources.tsv` after running the frozen command.
Subgroup intervals are post-hoc, unadjusted for multiple comparisons, and should not
be treated as confirmatory evidence. Any hypothesis derived from these 322 judged
queries requires genuinely unseen judgments for a new confirmatory claim; rerunning a
newly frozen method on the same judgments remains exploratory.

## Revisit triggers

A future source-query-first experiment is explicitly allowed when at least one
material trigger is documented:

1. A correctness bug is found that can change a decision-relevant score, ranking,
   opportunity count, nested selection, comparison, or gate.
2. New response-level judgments, a genuinely unseen evaluation split, or materially
   changed benchmark data become available.
3. A substantively different qrel-free method is proposed, such as a preregistered
   semantic-template router, a source-quality confidence rule, or a changed
   opportunity/normalization design.
4. The candidate-model or discovery-response landscape changes enough that the v1
   evidence no longer represents the intended deployment setting.

Finding an interesting subgroup on the same judged queries is not sufficient by
itself. Any router, threshold, feature, or method change motivated by this post-hoc
analysis must be frozen before its effectiveness is scored.

## Required procedure for a retest

1. Open a new issue that names the trigger and the exact v1 limitation being tested.
2. Use a new version, run ID, artifact directory, and decision receipt. A
   behavior-changing retest must use a versioned module or command rather than silently
   changing the `source-query-first` v1 behavior.
3. Obtain advisor approval for a leakage-safe protocol before inspecting new
   effectiveness results.
4. Compare the new method with the archived v1 result and the then-current default.
5. If a bug invalidates v1, add an explicit invalidation or supersession notice that
   identifies every affected artifact and metric; do not silently rewrite history.

Future maintainers and automated agents should read “stop the source-query-first
branch” in the locked result as **conclude v1 under its frozen gates**, not as “never
revisit this family of methods.”
