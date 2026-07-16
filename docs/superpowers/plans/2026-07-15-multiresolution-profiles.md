# Multi-resolution Candidate Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce Track A and run the bounded qrel-free construction, viability, repeatability, and diagnostic evaluation for issue #18 from the exact PR #17 head.

**Architecture:** Extend the existing deterministic candidate-profile primitives with an explicit requested cluster count and top-N centroid aggregation. A focused multiresolution module constructs k=5/10/20 views from validated frozen inference arrays, freezes C0/A1/R1/X1 rankings, writes a hash-bound artifact, and computes qrel-free viability. A separate evaluator opens qrels only against the frozen artifact.

**Tech Stack:** Python 3.13, NumPy, SciPy, scikit-learn deterministic K-means, existing `trec_mlm` metrics/artifact utilities, pytest, Ruff.

## Global Constraints

- Source commit is `2db10261bac607d7f8c67c0aa0c01728484a42f9` on dedicated branch `codex/issue-18-multires-profiles`.
- PR #17 must remain unchanged.
- Reuse issue-12 inference arrays; no record embedding inference.
- New construction, scoring, ranking, fusion, and viability are qrel-free.
- Track A and all current-query effectiveness results are diagnostic only.
- Freeze all C0/A1/R1/X1 rankings before evaluation and report every system.
- Use config-driven commands, complete artifacts, deterministic repeats, an accessible report, and a final diagnostic receipt.

---

### Task 1: Generalize deterministic profile construction and aggregation

**Files:**
- Modify: `trec_mlm/profile_retrieval.py`
- Test: `tests/test_profile_retrieval.py`

**Interfaces:**
- `canonical_kmeans(..., requested_clusters: int | None)` accepts requested counts through the number of records while retaining default `min(10,n)`.
- `score_profiles(..., top_n: int = 1)` returns deterministic scores and primary winners; a new contributor array records the first `top_n` centroid IDs.

- [ ] Add a failing test proving `requested_clusters=20` is accepted for 20 or more records while the default remains capped at 10.
- [ ] Run `PYTHONPATH=.:.venv/lib/python3.13/site-packages pytest tests/test_profile_retrieval.py -k requested_clusters -v` and confirm the old upper-bound failure.
- [ ] Change the explicit requested-count validation without changing the default low-support rule.
- [ ] Add failing tests for top-two mean, one-centroid fallback, contributor ordering, finite inputs, and candidate-ID tie breaking.
- [ ] Run the focused tests and confirm failure because top-two aggregation is absent.
- [ ] Implement the minimal aggregation API using stable descending similarity order with centroid ID as the exact-tie key.
- [ ] Run the focused tests and the complete `tests/test_profile_retrieval.py` file.
- [ ] Commit as `feat: add multi-centroid profile scoring`.

### Task 2: Construct and freeze multi-resolution artifacts

**Files:**
- Create: `trec_mlm/profile_multiresolution.py`
- Create: `scripts/run_multiresolution_profiles.py`
- Create: `configs/trec_2026_multiresolution_profiles.json`
- Test: `tests/test_profile_multiresolution.py`

**Interfaces:**
- `MultiResolutionConfig` names the validated issue-12 construction, output root, resolutions `(5,10,20)`, top count 2, seed 2026, RRF k 60, and source commit.
- `construct_multiresolution(config)` validates the issue-12 artifact and embedded inference, reconstructs annotated profiles at each resolution, freezes all profile and dense-fusion rankings, and returns a complete payload.
- `write_multiresolution_artifact(result, output)` writes atomically with a manifest containing exact file inventory, sizes, hashes, and aggregate hash.
- `load_multiresolution_artifact(path)` rejects inventory, hash, schema, source, and semantic reconstruction drift.

- [ ] Write failing tests for config rejection, source-aggregate binding, exact C0 reconstruction, complete ranks, expected system inventory, and tamper rejection.
- [ ] Run `PYTHONPATH=.:.venv/lib/python3.13/site-packages pytest tests/test_profile_multiresolution.py -v` and confirm import/API failures.
- [ ] Implement construction with the existing issue-12 records, embeddings, label aggregation, K-means, scoring, and RRF primitives.
- [ ] Implement atomic artifact writing and strict loading with semantic reconstruction checks.
- [ ] Add the CLI and frozen JSON config; the CLI must refuse qrel arguments.
- [ ] Run the new test file and relevant issue-12 profile tests.
- [ ] Commit as `feat: freeze multi-resolution profile artifacts`.

### Task 3: Add qrel-free viability and repeatability receipts

**Files:**
- Modify: `trec_mlm/profile_multiresolution.py`
- Modify: `scripts/run_multiresolution_profiles.py`
- Test: `tests/test_profile_multiresolution.py`

**Interfaces:**
- `assess_multiresolution_viability(artifact)` returns every issue-18 threshold, cost evidence, failed thresholds, and `continue`, `stop-no-incremental-value`, or `reject-invalid`.
- `compare_multiresolution_artifacts(first, second)` requires independent aggregate hashes but identical canonical scientific files and reports maximum score difference.

- [ ] Add failing fixture tests for occupancy entropy, cluster dominance, duplicate centroids, query collapse, candidate dominance, incremental top-10 difference, and receipt classification.
- [ ] Run the focused tests and confirm missing viability APIs.
- [ ] Implement cluster/ranking diagnostics with exact formulas and threshold names copied from the protocol amendment.
- [ ] Add failing repeat tests for source reuse, identical scientific content, and deliberately changed scores.
- [ ] Implement repeat comparison and receipt serialization.
- [ ] Run all multiresolution tests and commit as `feat: gate multi-resolution profile viability`.

### Task 4: Add sealed diagnostic evaluation

**Files:**
- Create: `trec_mlm/profile_multiresolution_evaluation.py`
- Create: `scripts/evaluate_multiresolution_profiles.py`
- Test: `tests/test_profile_multiresolution_evaluation.py`

**Interfaces:**
- `evaluate_multiresolution(artifact_path, qrels_path, output_path)` loads a frozen valid artifact, validates exact qrel identities, computes metrics and paired comparisons for every frozen system, reproduces Track A from issue-12 rankings, and verifies construction immutability.

- [ ] Add failing tests for invalid/unfrozen artifacts, qrel inventory mismatch, complete system evaluation, paired intervals, zero transitions, Track-A diagnostic labeling, and post-evaluation immutability.
- [ ] Run the test file and confirm missing evaluation API.
- [ ] Implement evaluation by reusing `trec_mlm.metrics` and issue-12 paired-comparison functions without accepting tuning options.
- [ ] Add a CLI that writes evaluation only after the viability receipt authorizes evaluation.
- [ ] Run evaluation and profile regression tests.
- [ ] Commit as `feat: evaluate frozen multi-resolution profiles`.

### Task 5: Execute Track B and Track A

**Files:**
- Create under: `artifacts/candidate_semantic_profiles/issue18_pr17head_pass1/`
- Create under: `artifacts/candidate_semantic_profiles/issue18_pr17head_repeat/` only if authorized
- Create: `reports/trec_2026_issue18_pr17head_viability.json`
- Create: `reports/trec_2026_issue18_pr17head_repeatability.json` only if authorized
- Create: `reports/trec_2026_issue18_pr17head_evaluation.json` only after freeze

- [ ] Record clean status, source commit, dependency versions, input hashes, and exact command before construction.
- [ ] Run pass one with qrels absent from the runtime and network disabled where practical.
- [ ] Validate the complete artifact and write the qrel-free viability/cost receipt.
- [ ] If and only if the receipt says `continue`, run the fresh construction repeat and compare it.
- [ ] Freeze ranking hashes and verify construction immutability.
- [ ] Load qrels once and run the diagnostic evaluation for every frozen system plus Track A reproduction.
- [ ] Commit code and bounded receipts without committing large embedding arrays.

### Task 6: Report, explorer, and advisor review

**Files:**
- Create: `reports/trec_2026_issue18_pr17head_results.json`
- Create: `reports/trec_2026_issue18_pr17head_report.artifact.json`
- Create: `reports/trec_2026_issue18_pr17head_report.html`
- Create: `reports/trec_2026_issue18_pr17head_query_explorer.html`
- Create: `reports/trec_2026_issue18_pr17head_delivery_receipt.json`

- [ ] Build a concise technical report that leads with diagnostic status and separates resolution from aggregation effects.
- [ ] Build an all-pairs explorer with C0/A1/R1/X1 ranks, scores, contributing centroid evidence, and failure slices.
- [ ] Validate both artifacts at desktop and mobile widths and verify local navigation.
- [ ] Run the full pytest suite, Ruff, JSON/SQL checks, artifact validation, deterministic hash checks, and `git diff --check`.
- [ ] Obtain advisor code/result review and address bounded defects.
- [ ] Commit as `docs: report issue 18 diagnostics` and preserve the worktree for post-merge rebase.
