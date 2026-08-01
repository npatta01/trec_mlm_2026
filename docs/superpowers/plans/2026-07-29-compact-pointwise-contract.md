# Reference-Grounded Pointwise Judge v8 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-field reference-grounded v8 judge and validate DeepSeek V4 Flash direct on the same targeted 50 cases twice.

**Architecture:** v8 is additive: a frozen prompt/schema pair defines the model boundary, while `trec_mlm.pointwise_judge` performs strict Pydantic validation and transparent local recovery. A dedicated validation runner sends identical, cached direct requests through DeepInfra and writes private resumable artifacts; a local audit compares both repetitions and exposes every disagreement.

**Tech Stack:** Python 3.11, Pydantic v2, JSON Schema, pytest, DeepInfra OpenAI-compatible API.

## Global Constraints

- Preserve all v1–v7 prompt and schema files byte-for-byte.
- Model output has exactly `correctness`, `coverage`, and `rationale`.
- `acceptable` is derived only as `correctness == "correct"`; partial coverage remains acceptable.
- `unknown` means the provisional reference is silent, not that the candidate is false.
- No provider fallback and no semantic retry.
- Keep API keys out of source, requests, logs, and artifacts; private artifacts use mode 0600.
- Make no commit because this worktree contains overlapping uncommitted user work and the user did not request one.

---

### Task 1: Add and validate the v8 contract

**Files:**
- Create: `protocols/pointwise_judge_bakeoff/pointwise-v8.txt`
- Create: `protocols/pointwise_judge_bakeoff/decision-v8.schema.json`
- Modify: `trec_mlm/pointwise_judge.py`
- Modify: `tests/test_pointwise_judge.py`

**Interfaces:**
- Produces: `DecisionV8`, `normalized_decision_v8_schema()`, `normalize_judge_decision_v8(raw: str) -> dict[str, object]`, and `load_prompt_contract(..., version="v8")`.

- [x] **Step 1: Write failing behavioral tests**

  Cover clean parsing, derived `acceptable`/`needs_review`, partial-but-correct acceptance, unknown reference silence, case/whitespace repair, unknown-key removal, rationale truncation, and an unscorable malformed result. Assert the loaded JSON Schema equals `normalized_decision_v8_schema()`.

- [x] **Step 2: Run the focused tests and observe failure**

  Run: `pytest tests/test_pointwise_judge.py -k v8 -q`

  Expected: import or contract-loading failures because v8 does not exist.

- [x] **Step 3: Add the exact three-field prompt/schema**

  The schema is strict:

  ```json
  {
    "correctness": "correct | mixed | wrong | unknown",
    "coverage": "complete | partial | missing | unknown",
    "rationale": "string with maxLength 240"
  }
  ```

  The prompt defines correctness before coverage, states that omissions never lower correctness, and states that reference silence maps to `unknown`.

- [x] **Step 4: Implement Pydantic-backed normalization**

  Clean output is validated with `extra="forbid"` and strict enum fields. Recovery may extract one object, normalize strings, drop unknown keys, and truncate the rationale; every mutation gets a repair code. A failure to recover required enums returns null decision values with `parse_status="unscorable"`.

- [x] **Step 5: Run focused and compatibility tests**

  Run: `pytest tests/test_pointwise_judge.py -q`

  Expected: all tests pass, including unchanged v1–v7 behavior.

### Task 2: Add a resumable two-repetition validation runner

**Files:**
- Create: `scripts/run_deepinfra_pointwise_v8_validation.py`
- Create: `tests/test_deepinfra_pointwise_v8_validation.py`

**Interfaces:**
- Consumes: a validated 100-case manifest plus exactly 50 unique ordinals, v8 contract, and `DEEPINFRA_API_TOKEN`.
- Produces: one private `result.json` for each repetition with request identity, usage, cost, normalized decisions, and reference-grounding tier.

- [x] **Step 1: Write failing runner tests**

  Exercise the real request-body builder and local response parser. Assert model `deepseek-ai/DeepSeek-V4-Flash`, temperature 0, direct mode (no `reasoning_effort`), 8192 output tokens, stable `prompt_cache_key`, strict JSON schema, no fallback, correct custom identity, and preservation of provider failures.

- [x] **Step 2: Run the runner tests and observe failure**

  Run: `pytest tests/test_deepinfra_pointwise_v8_validation.py -q`

  Expected: module import failure because the runner does not exist.

- [x] **Step 3: Implement the bounded runner**

  Use eight workers with identical per-case requests, one same-provider transport retry only for 408/429/5xx/timeouts, atomic mode-0600 checkpoints, and cache identity including manifest, prompt, schema, model, ordinals, and repetition. Join each question to the frozen Gemini grounding record and store `search_grounded`, `generated_without_search`, or `unanswered_or_failed`.

- [x] **Step 4: Run runner and full focused tests**

  Run: `pytest tests/test_deepinfra_pointwise_v8_validation.py tests/test_pointwise_judge.py -q`

  Expected: all tests pass with no live API calls in tests.

### Task 3: Run 100 calls and audit the outcomes

**Files:**
- Create: `scripts/audit_pointwise_v8_validation.py`
- Create: `tests/test_audit_pointwise_v8_validation.py`
- Create privately: `artifacts/pointwise_production/deepinfra-v8-direct-targeted-a/result.json`
- Create privately: `artifacts/pointwise_production/deepinfra-v8-direct-targeted-b/result.json`
- Create privately: `artifacts/pointwise_production/deepinfra-v8-direct-targeted-a-vs-b/audit.json`
- Create: `artifacts/pointwise_production/deepinfra-v8-direct-targeted-a-vs-b/audit.md`

**Interfaces:**
- Consumes: two complete v8 result artifacts with identical case identities.
- Produces: agreement metrics plus full inputs and both judgments for every disagreement.

- [x] **Step 1: Write failing audit tests**

  Hand-build two tiny result fixtures and assert literal correctness, coverage, exact agreement counts, label distributions, parse/error counts, total cost, and disagreement records containing question/reference/candidate and both outputs.

- [x] **Step 2: Run the audit tests and observe failure**

  Run: `pytest tests/test_audit_pointwise_v8_validation.py -q`

  Expected: module import failure because the audit does not exist.

- [x] **Step 3: Implement deterministic auditing**

  Fail closed on mismatched identities or hashes. Count agreements directly, never infer accuracy from self-consistency, and include all disagreement inputs for manual review.

- [x] **Step 4: Run repetition A, inspect its checkpoint, then run B**

  Use the same 50 ordinals as the v7 replicate. Confirm repetition A has 50 terminal records and no schema/provider failures before starting B. Do not replace a failed record by changing model, prompt, or provider.

- [x] **Step 5: Build and manually review the audit**

  Inspect every disagreement against the supplied question/reference/candidate. Record obvious rubric violations separately from reasonable ambiguity and reference silence. Compare repeat correctness agreement with v7's 41/50 baseline.

- [x] **Step 6: Run final verification**

  Run: `pytest tests/test_pointwise_judge.py tests/test_deepinfra_pointwise_v8_validation.py tests/test_audit_pointwise_v8_validation.py -q`

  Expected: all tests pass. Verify artifact permissions with `stat`, verify both prompt/schema hashes match, and confirm no secret-like values appear in added files.

## Execution result — 2026-07-29

- DeepInfra repetition A: 50/50 clean, zero repairs/errors/retries, `$0.00560331`.
- DeepInfra repetition B: 50/50 clean, zero repairs/errors/retries, `$0.00343823`.
- Total provider-reported cost: `$0.009041544` for 100 candidate judgments.
- Repeat agreement: correctness `47/50`, coverage `48/50`, exact label pair `45/50`; v7 correctness baseline was `41/50`.
- Grounding tiers in the targeted sample: 20 `search_grounded`, 30 `generated_without_search`.
- Independent all-50 semantic audit: 29 obvious stable-label errors, 3 ambiguous, 18 defensible; preferred coverage `8 complete / 20 partial / 22 missing`, versus run A `0/10/40` and run B `0/8/42`.
- Recommendation: do not scale DeepSeek v8 direct; test against a balanced human-adjudicated key and try a thinking or stronger judge.
- Verification: `91 passed` across the v8 contract, runner, and audit suites; 100/100 cache identities valid; private artifacts mode 0600; scoped code re-review ready.
- Repository-wide `pytest -q` could not collect `tests/test_label_inference.py` because this shell lacks the optional `torch` dependency. No dependency was installed.
