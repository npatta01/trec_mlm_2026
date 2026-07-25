# Similar-query diagnosis of the 75 sole-positive failures

## Bottom line

Similar questions do **not** usually make the organizer’s sole positive label
obvious. A deliberately generous, pre-qrel evidence screen finds a strong or
corroborated signal in only **11/75 cases (14.7%)**. Another 34 are weak or
non-unique, and 30 are not predictably identifiable from the released historical
evidence.

An exploratory receipt suggests that using other development qrels does not
close the gap. Across eight lexical and dense leave-one-query-out representations,
the most optimistic inspected
configuration places only **10/75** official positives in its top ten. That is a
post-hoc grid maximum, not an unbiased estimate. Thirty-seven targets have no
nearest-neighbor label agreement under any representation. The original sweep
producer was not retained, so these multi-representation numbers are
**receipt-only and not independently reproducible**; rerun them before using
them to select a system.

The failures split into two operational groups:

- **21/75 are fusion-dilution cases:** profile finds 18 in its top ten and dense
  finds three, but the 1:1 reciprocal-rank fusion pushes all 21 below ten.
- **54/75 are evidence failures:** neither component places the official
  positive in its top ten. Reweighting the existing two lists cannot recover
  these without a new signal.

## What “expert” means here

The qrels are a complete 322-query × 150-candidate matrix. A positive is an
official query–LLM expertise label with grade 1 or 2. The 2026 assessor, rubric,
reference answer, prompt, and grading pipeline are not documented locally or on
the official track page.

The actual candidate responses to the development targets are not released.
Therefore this analysis cannot prove that an official positive answered its
target well or that a grade-0 candidate answered badly.

## Similar development questions rarely share the same expert

This section preserves an exploratory transfer receipt. Its protocol is
described, and the frozen MPNet rows can be independently matched by the
repository ranker, but the complete lexical/dense sweep cannot be regenerated
byte-for-byte from a retained producer.

Pure one-neighbor transfer asks a narrow question: after excluding the target’s
qrels, does the closest other development query label the same candidate
positive?

| Representation | Nearest-label agreement | Hit@10 | Median official rank |
| --- | ---: | ---: | ---: |
| Word TF-IDF 1–2 gram | 12/75 | 5/75 | 78 |
| E5-small-v2 | 9/75 | 7/75 | 78 |
| MPNet-base-v2 | 7/75 | 7/75 | 78 |
| E5-Mistral contextual | 11/75 | 5/75 | 85 |

The expected agreement from a random other development query, after accounting
for each candidate’s prevalence, is 6.67/75. Word TF-IDF’s 12/75 is nominally
above that null before correction, but not after accounting for the eight
representations inspected. Mean neighbor-label-set Jaccard ranges from only
0.1% to 5.4%.

The frozen MPNet result is particularly revealing. The official positive appears
somewhere in the positive labels of the ten closest development queries for
43/75 targets, but similarity-weighted averaging ranks it in the top ten only
3/75 times. Max/support variants do not solve this: they improve the failure
slice only marginally and materially reduce full-set nDCG. Much of the 43-query
“support” is isolated, non-unique evidence rather than a stable local class.

## Historical responses usually do not identify the official positive either

For each failure, the diagnostic retrieved the three closest discovery queries
in frozen MPNet space and checked whether four candidates produced something
other than the exact default refusal. This is a coverage proxy only.

| Candidate selected for comparison | Non-exact refusals | Queries with any attempt | Queries with zero attempts |
| --- | ---: | ---: | ---: |
| Official sole positive | 36/225 (16.0%) | 28/75 | 47/75 |
| Primary rank 1 | 47/225 (20.9%) | 36/75 | 39/75 |
| Profile rank 1 | 50/225 (22.2%) | 37/75 | 38/75 |
| Dense rank 1 | 33/225 (14.7%) | 27/75 | 48/75 |

The official positive has zero attempts while at least one of the three system
leaders attempts a close analogue in 25/75 cases. This does not prove the
leaders are correct, but it means the released response history often gives no
clear behavioral reason to prefer the sole positive.

The independent lexical screen reaches the same conclusion:

- the official positive’s best attempted analogue has similarity below 0.10 in
  44/75 cases;
- its five-analogue aggregate ranks top ten in 11/75 and first in only 2/75;
- it beats all three primary grade-0 leaders in 17/75, but by at least 0.05 in
  only 5/75.

## Recurring patterns

1. **Small pockets of template coherence exist.** County-location questions and
   a few food/drink templates transfer cleanly. For example, “The Squire
   Ignatius Haskell House is located in what county?” has the near-template “In
   what county is The Troy Historic Village located?”; both label
   `Technology_systems` positive.
2. **Fusion sometimes suppresses an interpretable profile signal.** Pizza-chain
   comparisons point to `Companies`; the profile ranks it second while dense
   ranks it 117th, producing primary rank 22. Riemann-hypothesis and
   autosegmental-phonology cases similarly have profile rank 1.
3. **Strong semantic similarity often carries conflicting official labels.** A
   plant-genus comparison labeled `Education` is closest to another plant-genus
   query whose positives include `Flora`, `Organisms`, and
   `Biota_by_conservation_status`, not `Education`.
4. **Counter-domain positives are common in the deepest misses.** A documentary
   release-order query labels `Food_and_drink_by_country`; its closest
   documentary comparison labels `Culture_stubs`. A film/book question labels
   `Cuisine` while media candidates dominate every available signal.
5. **Some targets are missing their antecedent.** “What is that determination
   based on?”, “What gave them a sense of solidarity?”, “What do the routes
   consist of?”, and “What is the county’s payroll?” cannot be reliably routed
   from the released text alone.
6. **Entity-heavy multi-hop questions lack entity-matched history.** Obscure
   films, biographies, satellite launches, organizations, and location chains
   often retrieve only generic topic analogues.

## What this changes

Nearest-query transfer is not the missing silver bullet. The evidence supports
two narrower actions:

1. **Protect the 21 component hits.** Test a preregistered, fold-local gate that
   keeps a component’s high-margin top-ten candidate when profile and dense
   strongly disagree. Measure recovery against new false positives across all
   322 queries.
2. **Acquire a genuinely new signal for the other 54.** Prioritize judged
   historical response quality, entity-aware retrieval, query completeness, and
   organizer clarification of label semantics. More averaging or neighbor-count
   tuning is unlikely to make these labels predictable.

Any experiment should keep the 11 strong/corroborated cases separate from the
34 weak and 30 non-predictable cases. A method that improves only the first
group is useful but cannot be presented as a general solution to the sparse
failure problem.

## Reproducibility and caveats

- Case-level calculations: [`trec_2026_sparse_failure_analogue_cases.tsv`](trec_2026_sparse_failure_analogue_cases.tsv)
- Machine-readable diagnostic: [`trec_2026_sparse_failure_analogue_diagnostic.json`](trec_2026_sparse_failure_analogue_diagnostic.json)
- Cross-representation transfer results: [`trec_2026_sparse_failure_neighbor_transfer.tsv`](trec_2026_sparse_failure_neighbor_transfer.tsv)
- Builder: [`../scripts/build_sparse_failure_analogue_diagnostic.py`](../scripts/build_sparse_failure_analogue_diagnostic.py)

The evidence buckets are deterministic screening flags, not human correctness
judgments. Profile rank ≤3 is sufficient for a strong flag by construction, so
11/75 should be treated as an upper bound on clearly predictable cases.
