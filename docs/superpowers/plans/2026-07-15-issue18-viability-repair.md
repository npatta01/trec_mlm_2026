# Issue 18 Viability Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repair the qrel-free issue-18 receipt and evaluation authorization without changing any frozen profile, score, ranking, run, manifest, or artifact hash.

**Architecture:** Extend the existing strict viability calculation with two immutable external controls and execution provenance, then classify dominance failures separately from incremental-distinctness failures. Keep repeat authorization and evaluation authorization separate: only `continue` can authorize a repeat, while any strictly valid non-`reject-invalid` receipt can authorize the one frozen diagnostic evaluation.

**Tech Stack:** Python 3.13, NumPy, JSON, pytest, Ruff, existing `trec_mlm.profile_multiresolution` and `trec_mlm.profile_multiresolution_evaluation` modules.

## Global Constraints

- Preserve frozen artifact aggregate `2a3909cfea23af246eed59565557d57ec4c5b8eb1e2f1d5c2b11d2f9f8c6d5b9` exactly.
- Keep qrels sealed until both tasks pass independent advisor review.
- Never authorize or run the second construction repeat for this pass.
- Bind contextual E5 to the validated issue-12 source aggregate and exact 322-query/150-candidate inventories.
- Expose no tuning options and report every frozen system.

---

### Task 1: Supplemental qrel-free viability receipt

**Files:**
- Modify: `trec_mlm/profile_multiresolution.py`
- Modify: `scripts/run_multiresolution_profiles.py`
- Modify: `tests/test_profile_multiresolution.py`

**Interfaces:**
- Consumes: `assess_multiresolution_viability(artifact_path)` and the strictly validated issue-12 source rankings.
- Produces: a complete receipt containing `control_sources`, `control_comparisons`, `execution_provenance`, and recomputed decision `reject-invalid | stop-query-sensitivity | stop-no-incremental-value | continue`.

- [ ] **Step 1: Write failing comparison and decision tests**

Add tests that assert all eight systems have entries against `query_knn` and
`contextual_e5`, exact control hashes/inventories are enforced, dominance-only
failure returns `stop-query-sensitivity`, incremental-only failure returns
`stop-no-incremental-value`, and true integrity failure remains
`reject-invalid`.

```python
assert set(receipt["control_comparisons"]) == set(EXPECTED_SYSTEMS)
assert set(receipt["control_comparisons"]["x1_multires_top2_dense_equal"]) == {
    "query_knn", "contextual_e5"
}
assert receipt["decision"] == "stop-query-sensitivity"
```

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_profile_multiresolution.py -q`

Expected: failures because control comparisons, provenance, and the new decision classification are absent.

- [ ] **Step 3: Implement exact qrel-free comparisons**

For each frozen system/control pair compute deterministic top-10 statistics:

```python
{
    "mean_top10_jaccard": float,
    "top10_difference_query_share": float,
    "mean_top10_overlap_count": float,
    "mean_common_top10_absolute_rank_displacement": float,
}
```

Load contextual E5 only from the source artifact bound by
`source_aggregate_sha256`; require exact query/candidate inventories and record
the exact ranking-file hash. Treat `dense_reference` as query-kNN and verify it
equals the frozen artifact copy.

- [ ] **Step 4: Implement strict provenance and decision validation**

Require exact schema and internally recompute control metrics, decision, cost,
artifact binding, command, thread environment, dependency versions, actual
execution-code commit, and timestamps. Classification order is:

```python
if failed_integrity:
    decision = "reject-invalid"
elif failed_query_sensitivity:
    decision = "stop-query-sensitivity"
elif failed_incremental:
    decision = "stop-no-incremental-value"
else:
    decision = "continue"
```

`repeat_authorized` remains true only for `continue` with both cost booleans true.

- [ ] **Step 5: Verify GREEN and full regression**

Run:

```bash
.venv/bin/pytest tests/test_profile_multiresolution.py -q
.venv/bin/pytest -q
.venv/bin/ruff check trec_mlm/profile_multiresolution.py scripts/run_multiresolution_profiles.py tests/test_profile_multiresolution.py
git diff --check
```

Expected: all tests pass; changed files are clean.

- [ ] **Step 6: Commit**

```bash
git add trec_mlm/profile_multiresolution.py scripts/run_multiresolution_profiles.py tests/test_profile_multiresolution.py
git commit -m "fix: complete issue 18 viability receipt"
```

### Task 2: Decouple diagnostic evaluation from repeat authorization

**Files:**
- Modify: `trec_mlm/profile_multiresolution_evaluation.py`
- Modify: `scripts/evaluate_multiresolution_profiles.py`
- Modify: `tests/test_profile_multiresolution_evaluation.py`

**Interfaces:**
- Consumes: the strict Task-1 viability validator and exact receipt bytes.
- Produces: sealed diagnostic evaluation authorization for any valid decision except `reject-invalid`; repeat authorization remains unchanged.

- [ ] **Step 1: Write failing authorization-order tests**

Add tests proving a strict `stop-query-sensitivity` receipt permits evaluation,
`reject-invalid` and forged receipts are rejected, and no qrel hash/open occurs
until artifact and receipt validation plus receipt-byte recheck complete.

```python
result = evaluate_multiresolution(
    artifact_path, viability_receipt_path, qrels_path, output_path
)
assert result["viability_decision"] == "stop-query-sensitivity"
assert result["diagnostic_only"] is True
```

- [ ] **Step 2: Run focused tests and verify RED**

Run: `.venv/bin/pytest tests/test_profile_multiresolution_evaluation.py -q`

Expected: the valid-stop case fails because current validation requires `continue`.

- [ ] **Step 3: Implement minimal authorization split**

Keep complete receipt validation mandatory. Add a diagnostic authorization
predicate that accepts `continue`, `stop-query-sensitivity`, or
`stop-no-incremental-value`, rejects `reject-invalid`, and never changes
`repeat_authorized`. Persist the receipt decision and exact receipt SHA-256 in
the evaluation payload.

- [ ] **Step 4: Verify GREEN and full regression**

Run:

```bash
.venv/bin/pytest tests/test_profile_multiresolution_evaluation.py tests/test_profile_multiresolution.py -q
.venv/bin/pytest -q
.venv/bin/ruff check trec_mlm/profile_multiresolution_evaluation.py scripts/evaluate_multiresolution_profiles.py tests/test_profile_multiresolution_evaluation.py
.venv/bin/python -m py_compile trec_mlm/profile_multiresolution_evaluation.py scripts/evaluate_multiresolution_profiles.py
git diff --check
```

Expected: all tests pass and no qrel-order regression exists.

- [ ] **Step 5: Commit and review**

```bash
git add trec_mlm/profile_multiresolution_evaluation.py scripts/evaluate_multiresolution_profiles.py tests/test_profile_multiresolution_evaluation.py
git commit -m "fix: authorize valid-stop diagnostic evaluation"
```

Obtain independent advisor review of both commits. Only after approval,
regenerate the qrel-free receipt from the frozen artifact, verify its decision
and hashes, and run the one sealed Track-A/Track-B diagnostic evaluation.

### Task 3: Seal the frozen snapshot and one-shot paths

**Files:**
- Modify: `trec_mlm/profile_multiresolution.py`
- Modify: `trec_mlm/profile_multiresolution_evaluation.py`
- Modify: `scripts/run_multiresolution_profiles.py`
- Modify: `scripts/evaluate_multiresolution_profiles.py`
- Modify: `tests/test_profile_multiresolution.py`
- Modify: `tests/test_profile_multiresolution_evaluation.py`

**Interfaces:**
- Produces: `load_issue18_frozen_snapshot(artifact_path)` with no construction calls.
- Consumes: exact canonical artifact, receipt, qrel, and evaluation-output identities from the design.

- [ ] **Step 1: Write failing adversarial tests**

Prove the current code accepts an alternate self-consistent aggregate, invokes
construction transitively, permits receipt output inside the artifact, permits
alternate evaluation paths, and skips the manifest assertion. Instrument every
construction/scoring/fusion function to raise if called by the frozen loader.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `PYTHONPATH=. pytest tests/test_profile_multiresolution.py tests/test_profile_multiresolution_evaluation.py -q`

Expected: the new boundary tests fail for the reviewed reasons.

- [ ] **Step 3: Implement the frozen-snapshot loader**

Require canonical resolved root, exact config and aggregate, manifest hashes,
source binding, exact inventories, finite/normalized arrays, complete rankings,
score/rank consistency, and direct C0 equality with issue-12 stored artifacts.
Do not call construction, K-means, scoring, or fusion functions.

- [ ] **Step 4: Lock output and qrel identities**

Require canonical external receipt path, canonical receipt input, canonical
qrel path and locked SHA-256, and canonical non-existing evaluation output.
Check receipt/artifact authorization before the first qrel hash/read.

- [ ] **Step 5: Verify GREEN**

Run focused and full pytest, changed-file Ruff, `py_compile`, and
`git diff --check`. Expected: all tests pass.

- [ ] **Step 6: Commit and review**

Commit as `fix: seal issue 18 frozen evaluation`, package the complete repair
diff, and obtain independent advisor GO before any real receipt/evaluation run.
