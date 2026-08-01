# Task 3 implementer report

## Outcome

Implemented the frozen Phase 0 evaluator, zero-call CLI, complete deterministic
TREC runs, paired receipts, query and candidate slices, exact decision gate,
atomic ignored bundle, and sanitized exports. Ranking construction and pair
selection completed before the development qrels were hashed or opened. No API,
judge, pairwise, pointwise-regeneration, or embedding-inference call occurred.

## Red/green evidence

- Core red: `uv run pytest -q tests/test_pointwise_ranking_evaluation.py`
  failed at collection because `trec_mlm.pointwise_ranking_evaluation` did not
  exist.
- Core green: 10 synthetic tests passed after implementing frozen control
  reproduction, raw-gain nDCG@10, positive-grade MRR, query-grain paired
  receipts, exact equal-RRF ties, all query/candidate slice families, empty-bin
  handling, zero transitions, and every decision condition.
- Output red: the suite next failed at collection because
  `write_evaluation_outputs` did not exist.
- Output green: 11 tests passed after atomic bundle/export writing, five
  complete runs, manifests, and deterministic TSV/JSON serialization.
- Matrix red: the suite next failed at collection because
  `validate_qrel_matrix` did not exist.
- Matrix green: 12 tests passed after exact query/candidate dimension and
  identity checks plus frozen orchestration were implemented.

## Verification commands

- `uv run pytest -q tests/test_pointwise_ranking_evaluation.py
  tests/test_pointwise_ranking.py tests/test_metrics.py tests/test_runio.py
  tests/test_artifact_io.py` — 55 passed.
- `uv run ruff check trec_mlm/pointwise_ranking_evaluation.py
  scripts/evaluate_pointwise_ranking_phase0.py
  tests/test_pointwise_ranking_evaluation.py` — passed.
- `uv run python -m py_compile ...` and CLI `--help` — passed.
- Frozen CLI run with the protocol paths — passed; exact repeat used separate
  ignored artifact and export destinations.
- `cmp` on the three sanitized exports and both manifests — all byte-identical.
- All five emitted TREC runs validated at 322 queries, 150 candidates, and
  48,300 rows. `git diff --check` and sanitized path/text/credential scans
  passed.

## Frozen execution evidence

- Dense reproduction: nDCG@10 `0.35985474409504775`, MRR
  `0.40406330811168817`, 116 zero queries, and all 322 tracked per-query values
  passed at absolute tolerance `5e-12`.
- Pointwise-only: nDCG@10 `0.11882482433832028`, MRR
  `0.1708636403214092`, 231 zero queries.
- Prior-only: nDCG@10 `0.10216984523370287`; routing-only: nDCG@10
  `0.15972944894483668`.
- Combined: nDCG@10 `0.2113240933733129`, MRR `0.2691192482181426`,
  172 zero queries. Combined-minus-control nDCG is
  `-0.14853065072173485`, paired 95% interval
  `[-0.18381098850790814, -0.1142669617790963]`; W/T/L is 55/109/158.
- Combined zero transitions: 100 zero/zero, 16 rescues, 72 regressions, and
  134 nonzero/nonzero. Selector availability passed with 500 pairs across 500
  source questions, but the effectiveness gate failed and the exact
  recommendation is `stop_pairwise_work`.
- Query slice counts: density 248/30/44; dense zero/nonzero 116/206;
  similarity quartiles 81/80/81/80. Both candidate quartile families are
  38/37/38/37.

## Deterministic hashes and result status

- Ignored artifact aggregate:
  `f79246acfef8c5c043898680b68b3def38dddf895aa7df5040421dc24b0c4530`.
- Sanitized results JSON:
  `f8a5ad36871592135935b68e3bc7e2fefbfe3eb0c9a41544f3b8815b302c5a0a`.
- Sanitized per-query TSV:
  `e2193404d8f1052199b6b6a396d78556e48bb84b145a6e7adda125634218d889`.
- Sanitized candidate-slice TSV:
  `4d56304bf73f6d0e6bfd3c5612b922c3c3ce6e43f349857fc180aab9d83f3329`.
- The three exact Task 4 report files exist as untracked files and were not
  staged or committed. Task 4 owns independent validation and their commit.

## Commit and concerns

- Implementation commit: `6eb7c1e` (`feat: evaluate pointwise ranking phase 0`).
- No correctness or leakage concern is known. The paired percentile interval
  uses the repository metric primitive with Python's seeded MT19937 and records
  that generator explicitly because the protocol freezes seed 31 but does not
  name a generator.
