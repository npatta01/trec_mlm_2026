# Ranking diagnosis report source notes

## Reporting job

- **Question:** Why does the promoted system still have 99 zero-nDCG queries,
  and do similar queries make the official expert predictable in the 75
  sole-positive failures?
- **Audience:** Product stakeholders who need a decision-ready research update.
- **Scope:** The 322-query, 150-candidate TREC 2026 development set; the analogue
  deep dive uses the 75 promoted-run failures with exactly one positive qrel.
- **Decision supported:** Choose the next bounded experiment and avoid spending
  another cycle on unsupported neighbor or fixed-fusion tuning.
- **Success criterion:** Separate recoverable fusion misses from missing-evidence
  cases, preserve uncertainty about opaque labels, and provide inspectable cases.

## Required-structure mapping

| Executive-report role | Visible report section |
| --- | --- |
| Title | TREC 2026: why ranking performance has plateaued |
| Executive summary | Executive summary |
| Key findings with visual evidence | Where we are doing badly; system and density charts; similar-query evidence; neighbor-transfer chart; causal decomposition; static-weight sensitivity; spot checks; full 75-case drill-down |
| Recommended next steps | Recommended next experiment; Why clustering is not next |
| Further questions | Further questions |
| Caveats and assumptions | Caveats |

No required role is omitted. Existing sections, charts, tables, sources, and
datasets from the earlier report are preserved; the analogue layer is inserted
before the recommendation.

## Chart map

| Report segment | Analytical question | Family / type | Fields | Supported claim | Palette policy |
| --- | --- | --- | --- | --- | --- |
| System comparison | Which approaches have moved nDCG@10? | Comparison / horizontal bar | method, nDCG@10, evidence status | Profile construction produced the main measured gain | Existing categorical policy retained |
| Density slices | Where are failures concentrated? | Comparison / grouped bar | density bucket, system, nDCG@10 | Sparse queries contain nearly all remaining failures | Existing categorical policy retained |
| Analogue evidence | Can released history identify the sole positive? | Comparison / horizontal bar | evidence bucket, query count, share | Only 11/75 receive a generous strong flag | Single-root blue plus direct values |
| Neighbor transfer | Does a different query encoder repair coherence? | Receipt-only horizontal bar | representation, Hit@10, k, MRR, median rank | Unreproduced exploratory receipt reports 10/75 at top 10 | Single-root orange plus direct values |
| Failure causes | What observable mechanism or evidence gap best explains each failure? | Comparison / horizontal bar | mutually exclusive cause proxy, query count, evidence type | All 21 fusion cases are RRF vetoes; 47/54 no-signal cases lack close history or have nearby labels pointing elsewhere | Single-root olive plus direct values |

The three diagnostic charts are horizontal bars because they answer different
category-comparison questions with long labels: evidence strength, encoder
ranking, and mutually exclusive failure explanations. Static weight sensitivity
is a table because the reader needs exact recovered/lost tradeoffs and nDCG at
each discrete ratio. Manual archetypes are also a table because their status as
structured interpretation must remain explicit. The report includes a short
representative table followed by a dense lookup table covering all 75 failures.

## Validation and caveat map

- The diagnostic builder requires exactly 75 cases, 322 queries, 150 candidates,
  7,759 discovery queries, and the frozen 11/34/30 evidence-bucket counts.
- The notebook independently reloads the saved TSV/JSON and asserts the headline
  evidence, component, response-attempt, and best-Hit@10 totals.
- MPNet one-neighbor and frozen k=50 ranks were independently reproduced against
  the query explorer.
- Equal reciprocal-rank fusion reproduced all 48,300 stored ranks with zero
  mismatches and reproduced 0.4202139922 nDCG@10 using raw graded gains.
- The 21 fusion cases all have an opposing component rank of 21–150. Profile-only
  recovery starts above a 1.30 profile/dense ratio; dense-only recovery ends below
  0.78, so no static ratio protects both directions.
- The deterministic no-signal proxies reconcile exactly to 7 + 24 + 23 = 54.
- Manual archetypes reconcile exactly to 54 and remain labeled interpretive.
- Actual development-target responses are unavailable. Historical responses are
  never treated as correct.
- The multi-representation grid is post-hoc; its maxima are labeled accordingly.
- The original producer for the multi-representation transfer grid was not
  retained. Its TSV and chart are labeled receipt-only and must be rerun before
  model selection.
- Candidate IDs are model identifiers, not canonical topic labels.

## Canonical and derived artifacts

- Canonical plan: `plans/trec-2026-sparse-failure-analogue-diagnosis.md`
- Canonical report input: `reports/trec_2026_ranking_diagnosis.artifact.json`
- Calculation output: `reports/trec_2026_sparse_failure_analogue_diagnostic.json`
- Case table: `reports/trec_2026_sparse_failure_analogue_cases.tsv`
- Cross-representation results: `reports/trec_2026_sparse_failure_neighbor_transfer.tsv`
- Cause diagnostic: `reports/trec_2026_sparse_failure_cause_diagnostic.json`
- Cause case table: `reports/trec_2026_sparse_failure_cause_cases.tsv`
- Static-weight sensitivity: `reports/trec_2026_sparse_failure_weight_sensitivity.tsv`
- Companion notebook: `notebooks/trec_2026_sparse_failure_analogue_diagnostic.ipynb`
- Derived private HTML is intentionally outside the repository; rebuild it from
  the canonical artifact before publishing through the private portal.
