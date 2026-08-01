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

- [ ] Write failing tests for identity/hash mismatch and supported/verdict inconsistency.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement streaming validation and strict record-ID joins.
- [ ] Add failing tests for Beta prior, top-10 local mean, residual, deterministic RRF, and selector diversity/caps.
- [ ] Implement the minimal construction and make focused tests pass.
- [ ] Run a real-input structural preflight without qrels and verify all frozen counts/hashes.

### Task 3: Leakage-safe evaluation and receipts

**Files:**
- Create: `trec_mlm/pointwise_ranking_evaluation.py`
- Create: `scripts/evaluate_pointwise_ranking_phase0.py`
- Create: `tests/test_pointwise_ranking_evaluation.py`

**Interfaces:**
- Consumes frozen pointwise predictions, dense rankings, and development qrels.
- Produces control, pointwise-only, and combined rankings; diagnostics; metrics; paired intervals; slices; decision; zero-call selector receipt.

- [ ] Write failing synthetic tests for baseline reproduction, metric grain, slice partitions, zero transitions, and decision conditions.
- [ ] Implement evaluation, complete TREC runs, atomic local artifacts, and sanitized exports.
- [ ] Run focused tests and Ruff green.
- [ ] Execute the frozen evaluation once into `artifacts/pointwise_ranking_phase0/`.
- [ ] Re-run and require byte-identical sanitized outputs.

### Task 4: Independent analytical validation

**Files:**
- Create: `reports/trec_2026_pointwise_ranking_phase0_results.json`
- Create: `reports/trec_2026_pointwise_ranking_phase0_per_query.tsv`
- Create: `reports/trec_2026_pointwise_ranking_phase0_candidate_slices.tsv`

- [ ] Recompute aggregate metrics independently from the emitted TREC runs.
- [ ] Reconcile row counts, candidate sets, slice partitions, and paired differences.
- [ ] Verify dense control against both tracked receipts at `5e-12`.
- [ ] Scan tracked evidence for private paths/text, secrets, and API artifacts.
- [ ] Record a validation assessment and mandatory caveats in the structured result.

### Task 5: Canonical QMD report and run registry

**Files:**
- Create: `research/hub/studies/pointwise-ranking-phase0.qmd`
- Modify: `research/hub/ideas/local-response-quality-judging.qmd`
- Modify: reciprocal question, approach, and dataset records selected by existing relationships.
- Modify: `experiments/ledger.json`
- Modify: ledger loaders/tests only if a truthful generic per-query source kind is required.
- Regenerate: Hub and ledger derived outputs.

- [ ] Add failing ledger/Hub tests for the two new systems and study relationships.
- [ ] Register pointwise-only and combined once; reference the existing control without automatic promotion.
- [ ] Write the QMD with answer-first summary, definitions, method, aggregate evidence, query/candidate slices, robustness, decision, limitations, next step, and further questions.
- [ ] Regenerate outputs, pass focused tests, render the Hub, and inspect the final report.

### Task 6: Review, verification, and PR

**Files:**
- Modify: this plan and `plans/README.md` with final status/evidence.

- [ ] Obtain independent task and whole-branch reviews; fix and re-review critical/important findings.
- [ ] Run generated-output checks, Hub render/audit, Ruff, full pytest, `git diff --check`, and tracked-file secret/private-path scans.
- [ ] Produce sanitized derived HTML under `/home/npatta01/codex-rendered/plans/`, update the private portal index, verify live tailnet-only HTTPS, and confirm no private data exposure.
- [ ] Commit intended files, push `codex/issue-31-phase0`, and open a reviewable draft PR to `main` with the leakage boundary, results, checks, and issue linkage.

## Current status

- Phase: qrel-free construction implementation.
- Base: `eeea2d5cc69c0205f56806e13bb1ccf0bfa9096a` from current `origin/main`.
- Baseline verification: 1,035 tests passed and 3 skipped after supplying optional test dependencies in the isolated environment.
- Latest result: frozen records passed the 25-test protocol/index check and the
  Research Hub generated-output check; no pointwise effectiveness evaluation
  has run.
- Next action: start Task 2 test-first implementation.
