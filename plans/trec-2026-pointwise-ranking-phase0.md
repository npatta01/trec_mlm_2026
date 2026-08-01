# TREC 2026 Pointwise Ranking Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete GitHub issue #31 Phase 0 by testing whether audited pointwise labels improve the frozen candidate-ranking control without new judge, embedding, or pairwise calls.

**Architecture:** Join audited labels to a frozen candidate-local E5 retrieval tensor by record ID, decompose an overall candidate prior from a query-dependent residual, and use fixed RRF for pointwise-only and combined rankings. Evaluate once against organizer qrels, preserve private complete artifacts plus sanitized tracked evidence, and register the result in the Research Hub and experiment ledger.

**Tech Stack:** Python 3.13, NumPy, pytest, repository metrics/run I/O, Quarto.

## Global Constraints

- Phase 0 only; no LLM API, new embedding, pairwise, or pointwise-regeneration call.
- The frozen contract is `protocols/trec_2026_pointwise_ranking_phase0.md`.
- Private pointwise questions, responses, references, and tensors stay outside Git and rendered artifacts.
- The existing `dense_nested` run remains the control and is not duplicated or redefined.
- Overall candidate prior and query-specific residual remain separate outputs.
- Effectiveness is previously seen development evidence only.
- No diagnostic may replace the primary frozen result.

---

### Task 1: Freeze protocol and canonical working record

**Files:**
- Create: `docs/superpowers/specs/2026-08-01-pointwise-phase0-ranking-gate-design.md`
- Create: `protocols/trec_2026_pointwise_ranking_phase0.md`
- Create: `plans/trec-2026-pointwise-ranking-phase0.md`
- Modify: `plans/README.md`
- Modify: `research/hub/protocols/index.qmd`

- [x] Add the protocol and plan to authored indexes.
- [x] Scan the records for placeholders, contradictions, unfrozen choices, and private paths.
- [x] Commit the outcome-independent freeze before executing effectiveness evaluation.

### Task 2: Qrel-free pointwise construction

**Files:**
- Create: `trec_mlm/pointwise_ranking.py`
- Test: `tests/test_pointwise_ranking.py`

**Interfaces:**
- Produces `load_pointwise_evidence(...) -> PointwiseEvidence`, `construct_pointwise_rankings(...) -> PointwisePredictions`, and `select_pairwise_candidates(...) -> PairSelectionReceipt`.

- [x] Write failing tests for identity/hash mismatch and supported/verdict inconsistency.
- [x] Run the focused tests and verify the expected failures.
- [x] Implement streaming validation and strict record-ID joins.
- [x] Add failing tests for Beta prior, top-10 local mean, residual, deterministic RRF, and selector diversity/caps.
- [x] Implement the minimal construction and make focused tests pass.
- [x] Run a real-input structural preflight without qrels and verify all frozen counts/hashes.

### Task 3: Leakage-safe evaluation and receipts

**Files:**
- Create: `trec_mlm/pointwise_ranking_evaluation.py`
- Create: `scripts/evaluate_pointwise_ranking_phase0.py`
- Create: `tests/test_pointwise_ranking_evaluation.py`

**Interfaces:**
- Consumes frozen pointwise predictions, dense rankings, and development qrels.
- Produces control, pointwise-only, and combined rankings; diagnostics; metrics; paired intervals; slices; decision; zero-call selector receipt.

- [x] Write failing synthetic tests for baseline reproduction, metric grain, slice partitions, zero transitions, and decision conditions.
- [x] Implement evaluation, complete TREC runs, atomic local artifacts, and sanitized exports.
- [x] Run focused tests and Ruff green.
- [x] Execute the frozen evaluation once into `artifacts/pointwise_ranking_phase0/`.
- [x] Re-run and require byte-identical sanitized outputs.

### Task 4: Independent analytical validation

**Files:**
- Create: `reports/trec_2026_pointwise_ranking_phase0_results.json`
- Create: `reports/trec_2026_pointwise_ranking_phase0_per_query.tsv`
- Create: `reports/trec_2026_pointwise_ranking_phase0_candidate_slices.tsv`

- [x] Recompute aggregate metrics independently from the emitted TREC runs.
- [x] Reconcile row counts, candidate sets, slice partitions, and paired differences.
- [x] Verify dense control against both tracked receipts at `5e-12`.
- [x] Scan tracked evidence for private paths/text, secrets, and API artifacts.
- [x] Record a validation assessment and mandatory caveats in the structured result.

### Task 5: Canonical QMD report and run registry

**Files:**
- Create: `research/hub/studies/pointwise-ranking-phase0.qmd`
- Modify: `research/hub/ideas/local-response-quality-judging.qmd`
- Modify: reciprocal question, approach, and dataset records selected by existing relationships.
- Modify: `experiments/ledger.json`
- Modify: ledger loaders/tests only if a truthful generic per-query source kind is required.
- Regenerate: Hub and ledger derived outputs.

- [x] Add failing ledger/Hub tests for the two new systems and study relationships.
- [x] Register pointwise-only and combined once; reference the existing control without automatic promotion.
- [x] Write the QMD with answer-first summary, definitions, method, aggregate evidence, query/candidate slices, robustness, decision, limitations, next step, and further questions.
- [x] Regenerate outputs, pass focused tests, render the Hub, and inspect the final report.

### Task 6: Review, verification, and PR

**Files:**
- Modify: this plan and `plans/README.md` with final status/evidence.

- [x] Obtain independent task and whole-branch reviews; fix and re-review critical/important findings.
- [x] Run generated-output checks, Hub render/audit, Ruff, full pytest, `git diff --check`, and tracked-file secret/private-path scans.
- [x] Produce sanitized derived HTML in the host-configured private rendered-artifact directory, update the private portal index, verify live tailnet-only HTTPS, and confirm no private data exposure.
- [x] Commit intended files, push `codex/issue-31-phase0`, and open a reviewable draft PR to `main` with the leakage boundary, results, checks, and issue linkage.

## Current status

- Phase: complete; draft PR #55 is open for review.
- Base: `eeea2d5cc69c0205f56806e13bb1ccf0bfa9096a` from current `origin/main`.
- Baseline verification: 1,035 tests passed and 3 skipped before implementation;
  the first whole-branch run passed 1,069 tests with 3 skipped before final
  review fixes.
- Latest result: the frozen dense control reproduced at nDCG@10
  `0.35985474409504775`, MRR `0.40406330811168817`, 116 zero queries, and all
  322 per-query receipt values within `5e-12`. The combined system reached
  nDCG@10 `0.2113240933733129`, MRR `0.2691192482181426`, and 172 zero queries.
  Its nDCG delta was `-0.14853065072173485` with paired 95% interval
  `[-0.18381098850790814, -0.1142669617790963]`; the gate failed and the frozen
  recommendation is to stop pairwise work despite selector availability.
- Determinism: the post-fix evaluation and independent repeat were
  byte-identical. The ignored artifact aggregate is
  `f5e7575e1ca83391c852555441a1ba0488e7eacd4b493da1cf0a8fd6aa4a5e4b`;
  the validated results, per-query, and candidate-slice hashes are respectively
  `48ffaa81c77fde9dafb67ff9bbc29de98e51b8735a9320b2117f5662e31a7e6d`,
  `e2193404d8f1052199b6b6a396d78556e48bb84b145a6e7adda125634218d889`,
  and `4d56304bf73f6d0e6bfd3c5612b922c3c3ce6e43f349857fc180aab9d83f3329`.
- Independent validation: all 3,220 per-query metric cells, five aggregates,
  paired intervals, nine query slices, and 40 candidate-slice rows reconciled;
  the committed stdlib-only finalizer recreates the validated result bytes and
  exact SHA-256 from the evaluator bundle.
- Whole-branch review: all important Standards/Spec findings have fixes
  committed and independently re-reviewed with both axes approved. The
  remaining minor module-structure observations are deferred because splitting
  the frozen, fully tested evaluation path would add risk without changing
  Phase 0 evidence.
- Final verification: 1,089 tests passed and 3 skipped; Ruff, ledger and Hub
  generated-output checks, the independent validated-result `--check`,
  `git diff --check`, tracked privacy/API scans, a clean 175-page Hub render,
  and the static-site audit all passed.
- Private handoff: the sanitized Research Hub copy and portal index are live at
  the authorized tailnet-only Tailscale Serve mapping; the report returned HTTP
  200, contained the final evidence/provenance wording, and exposed neither
  QMD/Markdown sources nor local private paths. No Funnel handler is configured.
- Pull request: [#55](https://github.com/npatta01/trec_mlm_2026/pull/55)
  targets `main`, closes issue #31 when merged, and preserves the worktree for
  review follow-ups.
- Next action: review draft PR #55. Do not run Phase 1 or any pairwise/API work;
  the negative Phase 0 gate is terminal unless a new outcome-independent study
  is separately authorized.
