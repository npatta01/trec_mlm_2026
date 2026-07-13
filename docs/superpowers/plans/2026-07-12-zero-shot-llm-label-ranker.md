# Zero-Shot LLM Label Ranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, execute, and independently audit zero-shot pointwise and listwise LLM ranking over all 150 expert labels.

**Architecture:** Harden the shared embedding artifact loader first, then add a pure ranker core for prompts, permutations, parsing, scoring, and rankings. Keep ROCm/vLLM inference in a separate optional backend and orchestration/artifact writing in a third module. Rankings are fixed and hashed before a shared evaluation layer can load qrels.

**Tech Stack:** Python 3.12, NumPy, PyTorch/ROCm, Transformers, vLLM 0.24, JSON Schema structured output, pytest/unittest, TREC run format.

## Global Constraints

- Ranker: `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`, Apache-2.0.
- Container: `vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`.
- Use FP32, official chat-template token IDs exactly once, greedy decoding, and seed 2026.
- Use only cleaned labels and the fixed capability sentence; exclude qrel-trained context, profiles, responses, exemplars, and embedding shortlists.
- Pointwise scores all 48,300 pairs as `P(1) + 2 * P(2)` from exact constrained grade-token log probabilities.
- Listwise runs three balanced presentation orders, returns 20 unique handles, uses RRF `k=60`, and appends never-selected IDs lexicographically.
- Qrels remain unavailable until prompts, outputs, scores, rankings, and hashes are fixed.
- Run pointwise and listwise twice in fresh processes; report strict and practical repeat diagnostics.
- Do not automatically fuse rankings or select a deployment model.

---

### Task 1: Harden Existing Artifact Loading

**Files:**
- Modify: `trec_mlm/label_experiment.py`
- Modify: `tests/test_label_experiment.py`

**Interfaces:**
- Consumes: one label-run artifact directory.
- Produces: `load_label_experiment_artifacts()` that rejects duplicate, missing, non-finite, out-of-inventory, hash-mismatched, or run-inconsistent rows.

- [ ] **Step 1: Write failing corruption tests**

Create a valid fixture artifact, then independently remove a score row, duplicate a row,
write `nan`, change a model ID, modify a hashed file, and swap TREC ranks. Assert each
mutation raises a specific `ValueError`.

- [ ] **Step 2: Verify corruption tests fail for the expected missing validation**

Run `python -m pytest -q tests/test_label_experiment.py -k 'loader_rejects'` in the pinned
container. Expected: at least the missing-row and run-consistency fixtures are accepted.

- [ ] **Step 3: Implement strict loading**

Track `(query_id, model_id)` keys in a set, parse every score as finite float, require the
exact 322×150 Cartesian product, verify `validation.json` hashes where present, recreate
rankings with `rank_label_scores()`, and require equality with `run.trec`.

- [ ] **Step 4: Verify all label tests pass**

Run the three label test modules. Expected: zero failures.

### Task 2: Capture Provenance and Enforce Smoke Receipts

**Files:**
- Modify: `trec_mlm/label_experiment.py`
- Modify: `trec_mlm/cli.py`
- Create: `trec_mlm/artifact_manifest.py`
- Create: `tests/test_artifact_manifest.py`

**Interfaces:**
- Produces: `build_artifact_manifest(artifact_dirs, code_revision, worktree_status)`, `validate_smoke_receipt(path, spec)`, `trec-mlm label-artifact-manifest`, and a required `smoke_receipt` field for future GPU experiments.

- [ ] **Step 1: Write failing manifest and smoke tests**

Require deterministic file hashes, explicit commit `6275c43`, captured worktree status,
model revision/dtype parity, `passed: true`, and receipt SHA256. Assert absent/failed or
wrong-model smoke receipts block experiment construction.

- [ ] **Step 2: Verify tests fail because the module and CLI are absent**

Run `python -m pytest -q tests/test_artifact_manifest.py`.

- [ ] **Step 3: Implement the manifest and CLI**

Use sorted relative paths and streaming SHA256. Never infer host git data from inside the
container; accept it as explicit CLI input and store it in the manifest.

- [ ] **Step 4: Generate the embedding provenance manifest**

Link commit `6275c43` to all corrected/diverse run core-file hashes and smoke receipts.
Write compact output under `reports/` and verify it without changing existing scores.

### Task 3: Implement the Pure Ranker Core

**Files:**
- Create: `trec_mlm/llm_ranker.py`
- Create: `tests/test_llm_ranker.py`

**Interfaces:**
- Produces: `RankerSpec`, `format_pointwise_messages()`, `grade_token_ids()`, `expected_grade_score()`, `candidate_handles()`, `balanced_candidate_orders()`, `format_listwise_messages()`, `parse_listwise_handles()`, `rrf_rankings()`, and deterministic complete rankings.

- [ ] **Step 1: Write failing pointwise tests**

Freeze exact system/user messages, capability text, prompt-injection delimiters, distinct
single-token grades, normalized grade probabilities, expected score, and candidate-ID
tie behavior.

- [ ] **Step 2: Verify pointwise tests fail on missing imports**

Run `python -m pytest -q tests/test_llm_ranker.py -k pointwise`.

- [ ] **Step 3: Implement minimal pointwise pure functions**

Validate finite log probabilities for token IDs 0/1/2, normalize with log-sum-exp, and
return `p1 + 2*p2` as float64 before final float32 artifact storage.

- [ ] **Step 4: Write and verify failing listwise tests**

Require stable `C000`–`C149` handles, three 150-item unique orders, each candidate once
per presentation third, exact JSON length/membership/uniqueness, RRF `1/(60+r)`, and
lexical completion for never-selected candidates.

- [ ] **Step 5: Implement listwise pure functions and pass core tests**

Use SHA256 byte ordering exactly as specified; do not call random state or qrels.

### Task 4: Implement vLLM Pointwise and Listwise Inference

**Files:**
- Create: `trec_mlm/llm_ranker_inference.py`
- Create: `tests/test_llm_ranker_inference.py`

**Interfaces:**
- Consumes: formatted messages, frozen spec, smoke receipt.
- Produces: `run_pointwise_inference()` with exact grade log probabilities and `run_listwise_inference()` with raw/parsed structured outputs plus runtime metadata.

- [ ] **Step 1: Write failing adapter tests with faithful output fixtures**

Freeze official chat-template token IDs, vLLM `allowed_token_ids`, `logprob_token_ids`,
`max_tokens=1`, structured JSON schema, one syntax-only repair, and semantic validation.

- [ ] **Step 2: Verify tests fail on the absent backend**

Run `python -m pytest -q tests/test_llm_ranker_inference.py` in the pinned container.

- [ ] **Step 3: Implement optional imports and batched offline inference**

Do not import vLLM at module import time. Record resolved model/container/software,
token counts, throughput, elapsed time, device properties, and peak memory where the
worker API exposes it.

- [ ] **Step 4: Pass inference unit tests**

Run the backend tests without downloading a checkpoint by using serialized vLLM output
fixtures and tokenizer doubles.

### Task 5: Implement Qrel-Isolated Orchestration and Artifacts

**Files:**
- Create: `trec_mlm/llm_ranker_experiment.py`
- Create: `tests/test_llm_ranker_experiment.py`
- Modify: `trec_mlm/cli.py`

**Interfaces:**
- Produces: `trec-mlm llm-label-ranker --method pointwise|listwise`, qrel-free raw artifacts, post-ranking evaluation, repeat comparison, and valid TREC output.

- [ ] **Step 1: Write failing end-to-end qrel-mutation and artifact tests**

Use deterministic inference fixtures and two conflicting qrel files. Require prompts,
raw outputs, scores, rankings, and their bytes/hashes to remain identical while only
post-hoc metrics change. Require 322 complete rankings and 48,300 pointwise rows.

- [ ] **Step 2: Verify tests fail on the absent experiment module and CLI**

Run `python -m pytest -q tests/test_llm_ranker_experiment.py`.

- [ ] **Step 3: Implement two-phase orchestration**

Phase A reads queries/candidates, runs inference, validates, ranks, hashes, and writes a
qrel-free frozen receipt. Phase B accepts only that receipt plus qrels, evaluates it, and
writes metrics without mutating Phase A files.

- [ ] **Step 4: Pass all ranker and label tests**

Run all `test_llm_ranker*.py`, `test_artifact_manifest.py`, and `test_label*.py` modules.

### Task 6: Run ROCm Canonical Smoke Tests

**Files:**
- Create: `artifacts/llm_label_ranker/smoke/smoke.json`

**Interfaces:**
- Produces: a passed, hashed receipt required by Task 5 full inference.

- [ ] **Step 1: Verify device, image, disk, model revision, and memory**

Use read-only checks before the 8.06 GB checkpoint download.

- [ ] **Step 2: Validate pointwise parity**

On synthetic query/label pairs, compare official chat-template token IDs and all three
grade probabilities between production vLLM and a direct Transformers forward pass.
Freeze the observed tolerance only if it meets the design requirement.

- [ ] **Step 3: Validate listwise structured output and throughput**

Run 150 candidates through each balanced order, require 20 unique handles, exercise one
malformed-output fixture through syntax repair, and estimate full-run duration.

### Task 7: Execute Two Full Pointwise and Listwise Runs

**Files:**
- Create: `artifacts/llm_label_ranker/pointwise/run1/`
- Create: `artifacts/llm_label_ranker/pointwise/run2/`
- Create: `artifacts/llm_label_ranker/listwise/run1/`
- Create: `artifacts/llm_label_ranker/listwise/run2/`

**Interfaces:**
- Consumes: passed smoke receipt and frozen inputs.
- Produces: raw qrel-free receipts, complete rankings, evaluation, repeat diagnostics, timing, and memory.

- [ ] **Step 1: Run pointwise twice in fresh containers**

Validate 48,300 rows, exact grade token coverage, finite probabilities/scores, complete
rankings, byte hashes, and repeat score tolerance.

- [ ] **Step 2: Run listwise twice in fresh containers**

Validate 966 initial calls, bounded repair/fallback counts, each rotation and aggregate,
position diagnostics, complete rankings, and strict/practical repeat comparison.

- [ ] **Step 3: Evaluate only after all four ranking receipts are fixed**

Load qrels post-hoc and calculate headline/sliced metrics and nominal exploratory paired
intervals against MPNet, contextual E5, and contextual Qwen Embedding 4B.

### Task 8: Independent Audit and Final Receipt

**Files:**
- Create: `artifacts/llm_label_ranker/combined/ranker_summary.json`
- Create: `artifacts/llm_label_ranker/combined/decision.md`
- Create: `docs/results/2026-07-12-zero-shot-llm-label-ranker.md`

**Interfaces:**
- Produces: advisor-reviewed compact findings without automatic fusion or selection.

- [ ] **Step 1: Ask the existing independent advisor to review code and raw findings**

Require exact file/line findings on prompts, qrel isolation, parsing, aggregation,
repeatability, paired comparisons, and claims. Do not finalize while a high-severity
finding remains open.

- [ ] **Step 2: Incorporate validated corrections and rerun affected checks**

Preserve the original raw receipts; regenerate only derived artifacts when appropriate.

- [ ] **Step 3: Run final verification**

Run ranker/label tests in the pinned container, Ruff and formatting on changed files,
`git diff --check`, full run/score/hash audits, and verify advisor disposition.

- [ ] **Step 4: Write and capture the compact result report**

Clearly separate verified facts, nominal exploratory intervals, protocol failures, and
unknowns. Commit source/tests/protocol/report locally; do not publish or push.
