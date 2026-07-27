# Local Pointwise Judge 100-Case Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a frozen, identity-blind 100-response diagnostic set containing the existing ten smoke cases, run Qwen3.6-35B-A3B Q4_K_XL in thinking and direct modes, and report reliability, latency, and label differences without claiming model-reviewer consensus is human accuracy.

**Architecture:** Extend the existing bounded-memory response selector and private manifest format rather than create a second sampling stack. Preserve the ten smoke cases as a named reused slice, add nine distinct cases per diagnostic proxy, run both configurations through the existing frozen prompt/schema and resumable runner, then produce a comparison that is independent of reviewer labels until a later 100-case Advisor/Claude adjudication stage.

**Tech Stack:** Python 3.11+, pytest, existing `trec_mlm.pointwise_*` modules, Lemonade 11.5 llama.cpp ROCm backend, Qwen3.6-35B-A3B UD-Q4_K_XL GGUF.

## Global Constraints

- Reuse the frozen prompt and decision schema from `protocols/pointwise_judge_bakeoff/`; do not tune them against the 100 selected answers.
- Include all ten `smoke-v2` cases and identify them explicitly in the manifest.
- Select exactly ten distinct response-level cases for each of the ten frozen diagnostic reasons, for 100 unique response IDs and 100 unique question IDs.
- Selection must remain blind to candidate model identity; seal candidate IDs in a mode-`0600` sidecar.
- Keep manifests, raw model responses, logs, and reports private (`0700` directories, `0600` files) and gitignored.
- Use the same 100 cases, frozen prompt and decision contract, Lemonade transport-schema adjustment, context (8192), output allowance (4096), and temperature (0) for thinking and direct modes. For future runs, hash the adjusted transport schema separately from the frozen decision schema. The historical canonical runs predate that hardening: both omitted `rationale.maxLength` and `error_flags.uniqueItems`, recorded the same adjustment string, and contain no `transport_schema_sha256`.
- Report model-reviewer agreement only after independent review; the initial comparison reports schema/reliability, latency/token usage, label shifts, and diagnostic-stratum behavior.
- Do not call hosted shortlist models or scale to 500 records in this plan.

---

### Task 1: Generalize the bounded selector to ten cases per reason

**Files:**
- Modify: `trec_mlm/pointwise_sampling.py`
- Test: `tests/test_pointwise_sampling.py`

**Interfaces:**
- Consumes: `references`, streamed `responses`, the frozen smoke `Sequence[CandidateCase]`, and seed `2026`.
- Produces: `select_diagnostic_case_set(..., cases_per_reason: int, required_cases: Sequence[CandidateCase]) -> tuple[list[CandidateCase], dict[str, str]]`.

- [x] **Step 1: Write failing tests for 100 unique cases and required smoke inclusion**

Add fixtures containing at least eleven eligible responses per diagnostic reason. Assert exactly 100 cases, ten per reason, unique question/response IDs, deterministic output, identity-blind case serialization, and inclusion of every required smoke response ID with unchanged content.

- [x] **Step 2: Run the focused tests and confirm failure**

Run: `uv run pytest tests/test_pointwise_sampling.py -q`

Expected: FAIL because `select_diagnostic_case_set` does not exist.

- [x] **Step 3: Implement the generalized selector**

Keep `select_diagnostic_cases` as the ten-case compatibility wrapper. Add a bounded candidate pool sized for `cases_per_reason`, validate positive integer counts and required-case identities, reserve required cases first, and deterministically assign remaining distinct question/candidate pairs. Preserve `_candidate_features` single-pass computation and bounded memory.

- [x] **Step 4: Run focused sampling tests**

Run: `uv run pytest tests/test_pointwise_sampling.py -q`

Expected: PASS.

---

### Task 2: Build and validate the frozen private 100-case manifest

**Files:**
- Create: `scripts/build_pointwise_judge_100.py`
- Modify: `scripts/run_local_pointwise_judge.py`
- Test: `tests/test_pointwise_judge_100_cli.py`
- Create privately: `artifacts/pointwise_judge_bakeoff/local-100-v1/manifest.json`
- Create privately: `artifacts/pointwise_judge_bakeoff/local-100-v1/sealed_candidates.json`

**Interfaces:**
- Consumes: verified discovery/grounded sources and `smoke-v2/smoke_manifest.json`.
- Produces: manifest version 2 with `selection_reasons`, `cases_per_reason: 10`, `reused_smoke_response_ids`, `cases`, source hashes, and canonical `manifest_hash`.

- [x] **Step 1: Write failing CLI/loader tests**

Assert exact manifest keys, source hashes, canonical hash, mode `0600`, idempotent regeneration, rejection of changed smoke content, 100 unique cases, ten cases per reason, and exact preservation of the ten reused smoke cases.

- [x] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_pointwise_judge_100_cli.py -q`

Expected: FAIL because the builder and version-2 loader are absent.

- [x] **Step 3: Implement the builder and versioned loader**

Reuse the safe source snapshot, duplicate-key rejection, canonical hashing, atomic no-partial-output behavior, and private modes from `build_pointwise_judge_smoke.py`. Update `_load_cases` to accept version 1 smoke manifests and version 2 100-case manifests with exact per-version keys and cardinality checks.

- [x] **Step 4: Run focused and regression tests**

Run: `uv run pytest tests/test_pointwise_judge_100_cli.py tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py -q`

Expected: PASS.

- [x] **Step 5: Generate twice and freeze the real manifest**

Run the builder with `data/dev/llm_discovery_data.json`, `artifacts/reference_answer_validation/grounding/gemini36_grounded_all_7759.json`, seed `2026`, and `smoke-v2/smoke_manifest.json`; rerun it unchanged and verify identical manifest SHA-256. Inspect counts, modes, unique IDs, per-reason totals, and reused-slice equality before inference.

---

### Task 3: Run Qwen direct and thinking over the identical 100 cases

**Files:**
- Modify if required by regression tests: `scripts/run_lemonade_pointwise_smoke.py`
- Create privately: `artifacts/pointwise_judge_bakeoff/local-100-v1/runs/qwen36-35b-a3b-direct.json`
- Create privately: `artifacts/pointwise_judge_bakeoff/local-100-v1/runs/qwen36-35b-a3b-thinking.json`

**Interfaces:**
- Consumes: frozen manifest/hash, prompt/schema v1, already cached Qwen GGUF.
- Produces: two resumable run containers with exactly 100 terminal records each.

- [x] **Step 1: Verify hardware, cache, service state, disk, and secrets**

Run `rocminfo`, `free -h`, `df -h`, Lemonade status, exact checkpoint inspection, and a scan confirming no secrets will enter command output. Do not download another checkpoint.

- [x] **Step 2: Start the cached Qwen backend with 8192 context**

Use loopback-only Lemonade llama.cpp ROCm serving and verify `/v1/models` plus one schema canary before the substantial run.

- [x] **Step 3: Run direct mode**

Use `--max-output-tokens 4096`, temperature 0, thinking disabled, the frozen decision contract, and resumable private output. The historical canonical run used a Lemonade compatibility schema with `rationale.maxLength` and `error_flags.uniqueItems` omitted and no separate transport-schema hash; the local frozen parser remained authoritative. Future runs use the hardened runner from `7e78bcb`, which omits only `rationale.maxLength` and records the adjusted transport schema hash. Require 100/100 terminal records and preserve usage/latency/raw response bodies.

- [x] **Step 4: Run thinking mode**

Use the identical settings except thinking enabled. Require 100/100 terminal records and preserve usage/latency/raw response bodies.

- [x] **Step 5: Stop/unload the model and validate both artifacts**

Confirm zero live test containers/services holding GPU memory, both runs bind the same input, prompt, and frozen decision-schema hashes, both historical runtime observations record the same two-keyword transport adjustment without a `transport_schema_sha256`, no strict-parser failures are silently omitted, and no candidate identities appear in public/tracked files.

---

### Task 4: Compare modes and prepare independent-review inputs

**Files:**
- Create: `scripts/build_pointwise_judge_100_comparison.py`
- Test: `tests/test_pointwise_judge_100_comparison.py`
- Create privately: `artifacts/pointwise_judge_bakeoff/local-100-v1/comparison.json`
- Create privately: `artifacts/pointwise_judge_bakeoff/local-100-v1/comparison.md`

**Interfaces:**
- Consumes: manifest plus direct/thinking run artifacts.
- Produces: deterministic label-confusion matrix, exact/adjacent mode agreement, per-reason changes, reused-slice metrics, schema/failure counts, latency/token totals, and anonymized review packets.

- [x] **Step 1: Write failing comparison tests**

Use small fabricated run containers to assert disagreement direction, per-stratum counts, reused-slice isolation, latency/token aggregation, failed-record accounting, and absence of model identity from anonymized packets.

- [x] **Step 2: Run tests and confirm failure**

Run: `uv run pytest tests/test_pointwise_judge_100_comparison.py -q`

Expected: FAIL because the comparison builder is absent.

- [x] **Step 3: Implement the comparison builder**

Fail closed on mismatched hashes/coverage. Do not label either mode correct without reviewer evidence. Emit case-level differences privately and aggregate only non-sensitive metrics in any later public view.

- [x] **Step 4: Run focused tests and build the report**

Run: `uv run pytest tests/test_pointwise_judge_100_comparison.py -q`

Expected: PASS, followed by successful private comparison generation.

---

### Task 5: Verification and user checkpoint

**Files:**
- Modify: `.superpowers/sdd/2026-07-26-local-pointwise-judge-100/progress.md`

**Interfaces:**
- Consumes: code, tests, frozen artifacts, Qwen runs, comparison, Qwen-27 diagnostic log, and authorized Claude audit.
- Produces: evidence-backed recommendation on thinking, direct, or routing between them.

- [x] **Step 1: Run proportional verification**

Run focused tests, `python -m py_compile` for new scripts, `ruff check`/format checks on touched Python, `git diff --check`, private-mode checks, secret scans, and `git status --short`. Preserve unrelated user changes.

- [x] **Step 2: Record exact Qwen-27 diagnostic outcome**

Classify it as successful, confirmed runtime error, or bounded inconclusive. Cite the preserved startup log and elapsed readiness time; do not infer beyond it.

- [x] **Step 3: Retry GPT-OSS with the user-authorized larger generation budget**

After the Qwen diagnostic releases the GPU, start the same cached GPT-OSS-20B checkpoint/runtime offline and rerun the ten-case smoke with an 8,192-token output allowance and a request deadline long enough to reach that allowance. Preserve server logs and terminal failure evidence. Compare completion coverage, elapsed time, token use, and labels with the earlier high/medium artifacts; classify any remaining failure as model behavior, runtime behavior, or still inconclusive based only on captured evidence.

- [x] **Step 4: Update the canonical ledger**

Record manifest hash, code/test evidence, direct/thinking coverage, schema failures, elapsed time, token totals, label agreement/confusion, diagnostic-stratum shifts, and remaining reviewer limitations.

- [x] **Step 5: Present the checkpoint**

Explain GPT-OSS timeout evidence, Qwen-27 diagnosis, the exact fallible-reviewer meaning of the original 9/9 metric, Claude's authorized audit, and the 100-case direct/thinking findings. Do not call agreement accuracy.
