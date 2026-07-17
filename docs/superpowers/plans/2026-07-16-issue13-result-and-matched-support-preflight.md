# Issue 13 Result and Matched-Support Preflight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a minimal Issue 13 rejection record and one exploratory matched-support test from current `main`.

**Architecture:** A standalone script loads frozen dense-query inputs, compares hard-cluster grade means with equal-support unweighted nearest-neighbor means under leave-one-query-out folds, and emits only sanitized aggregate JSON. A standalone Markdown report records the authoritative rejection, advisor corrections, preflight outcome, and next decision.

**Tech Stack:** Python 3.13, NumPy, scikit-learn, SciPy, pytest, Ruff, Markdown.

## Global Constraints

- The preflight is exploratory and cannot promote a ranking method.
- Use only the frozen MPNet embeddings, cosine matrix, complete development qrels, and canonical inventories.
- Run at most 644 KMeans fits: 322 targets times two cluster counts.
- Do not add HDF5, cache, migration, compatibility, or experiment-runner APIs.
- Output no query text, query IDs, qrels, rankings, or raw arrays.
- Stop cluster work unless hard clustering beats matched-support neighbors on mean nDCG@10.

---

### Task 1: Deterministic matched-support primitives

**Files:**
- Create: `tests/test_issue13_matched_support_preflight.py`
- Create: `scripts/run_issue13_matched_support_preflight.py`

**Interfaces:**
- Produces: `select_positive_neighbors(similarity_row, support_indices, query_ids, count) -> tuple[int, ...]`
- Produces: `mean_grades(grades, indices) -> numpy.ndarray`
- Produces: `rank_candidates(scores, prior, candidate_ids) -> tuple[str, ...]`

- [ ] Write synthetic tests requiring positive-cosine filtering, similarity/query-ID ordering, exact support count, uniform grade means, and prior/candidate-ID tie breaking.
- [ ] Run `python -m pytest tests/test_issue13_matched_support_preflight.py -q` and require failure because the module is absent.
- [ ] Implement only the three pure functions and strict finite/shape/index validation.
- [ ] Rerun the focused test and require it to pass.
- [ ] Commit the task as `test: define issue13 matched-support comparison`.

### Task 2: Frozen-input preflight and sanitized receipt

**Files:**
- Modify: `tests/test_issue13_matched_support_preflight.py`
- Modify: `scripts/run_issue13_matched_support_preflight.py`

**Interfaces:**
- Produces: `run_preflight(...) -> dict[str, object]`
- Produces: a deterministic JSON receipt at `reports/issue_13_matched_support_preflight.json`

- [ ] Add tests with a small synthetic embedding/qrel fixture requiring target exclusion, deterministic KMeans settings, matched support, aggregate metrics, explicit `exploratory_not_promotable` status, and absence of per-query/raw fields.
- [ ] Run the focused test and verify the new assertions fail for missing orchestration.
- [ ] Implement frozen-input validation, leave-one-out C=20/40 orchestration, metric evaluation, support/granularity summaries, hashes, CLI arguments, and pretty sorted JSON through `trec_mlm.artifact_io.write_pretty_json`.
- [ ] Run focused tests and Ruff; require both to pass.
- [ ] Run the real preflight twice to separate temporary outputs and require byte identity with `cmp`.
- [ ] Publish the verified receipt to `reports/issue_13_matched_support_preflight.json`.
- [ ] Commit the task as `feat: compare clusters with matched local support`.

### Task 3: Standalone result and advisor-corrected report

**Files:**
- Create: `reports/issue_13_dense_aggregation_result.md`

**Interfaces:**
- Consumes: authoritative rejection metrics and the Task 2 receipt.
- Produces: the mergeable human-readable Issue 13 record.

- [ ] Write the report from the approved design, including baseline/result metrics, estimator granularity, the matched-support outcome, limitations, and the single next decision.
- [ ] Verify every quantitative claim against the compact receipts with a read-only assertion script.
- [ ] Scan for query IDs, query text, qrel rows, rankings, absolute private paths, and unsupported promotion language.
- [ ] Commit the task as `docs: record issue13 result and next decision`.

### Task 4: Branch and publication gate

**Files:**
- Modify only files required by Tasks 1-3 plus this design and plan.

**Interfaces:**
- Produces: a small draft PR against current `main` and closes superseded PR #20.

- [ ] Run `TMPDIR=$PWD/artifacts/pytest_tmp PYTHONPATH=.venv/lib/python3.13/site-packages python -m pytest -q`; require all tests to pass with only documented skips.
- [ ] Run Ruff on the new script/test and `git diff --check`.
- [ ] Inspect `git diff --stat origin/main...HEAD` and reject any old dense-aggregation implementation, HDF5, cache, lockfile, or dependency changes.
- [ ] Render and validate the standalone report through the existing private tailnet portal; require exact live/local byte equality and no raw-data leakage.
- [ ] Push `agent/issue13-result-and-preflight`, open a draft PR against `main`, and make the PR body state that the result is negative and the preflight is exploratory.
- [ ] Close PR #20 as superseded, linking the new draft PR and preserving PR #20 as historical context.

