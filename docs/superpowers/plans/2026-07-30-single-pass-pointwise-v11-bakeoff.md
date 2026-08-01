# Single-Pass Pointwise v11 Bakeoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Select one production judge by testing an ordered, central-claim v11 contract with DeepSeek V4 Flash thinking, GPT-5.4 Nano, and Qwen3.6-35B-A3B thinking against 282 already-audited records.

**Architecture:** Add v11 alongside v1-v10, with four model-emitted fields and a deterministic derived label. Reconstruct one hash-bound adjudication manifest from the completed v10 result and semantic-audit corrections. Use a private resumable runner with pinned provider routes and no fallback, first on 10 trigger cases and then on all 282 audited cases; evaluate every arm against identical gold labels and enforce the approved acceptance gates.

**Tech Stack:** Python 3.11, Pydantic v2, JSON Schema, pytest, DeepInfra OpenAI-compatible API, OpenRouter pinned-provider API.

## Global Constraints

- Use one model for every production judgment; do not implement a cascade or second-pass router.
- The model emits `central_claim`, `answer_present`, `supported_core`, and `material_conflict`; code derives the label and score.
- Correctness of the central answer matters more than completeness; hedges, omissions, unrelated tails, and peripheral evidence silence do not lower a supported core.
- Evidence silence is not contradiction; `material_conflict` requires an explicit evidence-backed conflict.
- A wrong subject or referent is not an answer.
- Candidate models never see Gemini evidence. Only the retrospective judge sees question, citation-free Gemini evaluator evidence, and historical response.
- Send no candidate/model identity, record ID, response ID, citation, URL, source list, or grounding metadata in provider messages.
- Bare-IDK responses receive deterministic `nonanswer` labels without provider calls.
- Pin DeepSeek to DeepInfra, GPT Nano to OpenAI, and Qwen to AkashML; set `allow_fallbacks=false` everywhere.
- Run exactly 10 provider-eligible smoke cases before the audited bakeoff.
- Keep private artifacts mode 0600 and directories mode 0700; never print or persist credentials.
- Preserve all existing dirty-worktree changes. Make no commit because the user did not request one and the worktree contains overlapping uncommitted work.

---

### Task 1: Add the v11 contract and deterministic mapper

**Files:**
- Create: `protocols/pointwise_judge_bakeoff/pointwise-v11.txt`
- Create: `protocols/pointwise_judge_bakeoff/decision-v11.schema.json`
- Modify: `trec_mlm/pointwise_judge.py`
- Modify: `tests/test_pointwise_judge.py`

**Interfaces:**
- Produces: `DecisionV11`, `normalize_judge_decision_v11(raw: str) -> dict[str, object]`, `derive_v11_label(decision: DecisionV11 | Mapping[str, object]) -> str`, and `load_prompt_contract(..., version="v11")`.

- [x] **Step 1: Write failing v11 tests**

  Cover the exact four-field schema; `central_claim` at most 20 whitespace-delimited words; empty claim iff `answer_present=false`; all five valid boolean combinations; rejection of the three impossible combinations; clean parsing; narrow outer-whitespace/key-preserving normalization; and deterministic labels `correct`, `partial`, `incorrect`, `not_established`, and `nonanswer`.

- [x] **Step 2: Run the tests and observe missing-v11 failures**

  Run: `uv run --extra dev pytest tests/test_pointwise_judge.py -k v11 -q`

  Expected: failures because v11 types/functions and prompt loading do not exist.

- [x] **Step 3: Implement the minimal v11 contract**

  The prompt uses the approved ordered tests and seven generic trigger examples. The provider schema is a flat strict object with one bounded string and three booleans; Pydantic enforces cross-field invariants locally. Existing v1-v10 behavior remains byte-for-byte compatible.

- [x] **Step 4: Verify contract and compatibility suites**

  Run: `uv run --extra dev pytest tests/test_pointwise_judge.py -q`

  Run: `uv run --extra dev ruff check trec_mlm/pointwise_judge.py tests/test_pointwise_judge.py`

---

### Task 2: Build the machine-readable 282-case adjudication set

**Files:**
- Create: `protocols/pointwise_judge_bakeoff/v11-adjudication-overrides.json`
- Create: `scripts/build_pointwise_v11_adjudication.py`
- Create: `tests/test_build_pointwise_v11_adjudication.py`
- Create privately: `artifacts/pointwise_judge_bakeoff/v11/adjudication/manifest.json`

**Interfaces:**
- Consumes: the merged v10 500-case result, original v10 100-case result, run plan, five slice-audit reports, the primary 100-case audit, and an explicit ordinal-to-preferred-pair override file.
- Produces: a private manifest with exactly 282 unique cases, each containing ordinal, question, evaluator evidence, historical response, expected v11-derived label, evidence tier, response-length bucket, and audit provenance hashes.

- [x] **Step 1: Write failing manifest-builder tests**

  Assert inclusion of all original 100 records, all 148 new grade-0/1 records, the 30 audited new grade-2 controls, and bare-IDK ordinals 55/57/58/60; no duplicate ordinal; exactly 282 records; exactly 102 explicit semantic overrides; expected numeric-grade totals of 174 grade-2, 66 grade-1, and 42 grade-0; source SHA-256 bindings; and private output modes.

- [x] **Step 2: Run the tests and observe missing-builder failures**

  Run: `uv run --extra dev pytest tests/test_build_pointwise_v11_adjudication.py -q`

- [x] **Step 3: Add the reviewed override ledger and minimal builder**

  Store each override as `{ordinal, preferred_pair, source_report}`. Convert preferred v10 pairs deterministically to v11 labels. Unlisted audited records retain their reviewed v10 pair; grade-2 controls and deterministic IDKs must be explicitly identified by the manifest selection metadata rather than silently inferred.

- [x] **Step 4: Build and independently validate the private manifest**

  Run the builder with the frozen source paths. Validate the 282-record set, label totals, source hashes, zero forbidden IDs in provider-facing message fixtures, and mode 0600/0700.

---

### Task 3: Add a pinned, resumable three-model v11 runner

**Files:**
- Create: `scripts/run_pointwise_v11_bakeoff.py`
- Create: `tests/test_run_pointwise_v11_bakeoff.py`

**Interfaces:**
- Consumes: the 282-case adjudication manifest, v11 prompt/schema, selected model arm, selected ordinals, API credentials from environment, and explicit call/cost caps.
- Produces: one private per-arm result directory with metadata, resumable per-record checkpoints, provider usage/cost, normalized v11 fields, and derived label.

- [ ] **Step 1: Write failing runner tests**

  Assert exact provider-facing message labels `ORIGINAL QUESTION`, `GEMINI EVALUATOR EVIDENCE — RETROSPECTIVE JUDGE ONLY`, and `HISTORICAL RESPONSE`; absence of IDs/model names/source metadata; strict JSON schema; temperature zero only where supported; private modes; cache identity coverage; one identical same-provider retry only for timeout/connection/408/429/5xx; and no fallback.

  Pin arms exactly:

  - `deepseek-v4-flash-thinking`: DeepInfra direct endpoint, model `deepseek-ai/DeepSeek-V4-Flash`, `reasoning_effort=high`, 8192 output tokens, eight workers.
  - `gpt-5.4-nano-direct`: OpenRouter, model `openai/gpt-5.4-nano`, provider `OpenAI`, no temperature field, 1024 output tokens, four workers.
  - `qwen3.6-35b-a3b-thinking`: OpenRouter, model `qwen/qwen3.6-35b-a3b`, provider `AkashML`, reasoning enabled, 4096 output tokens, four workers.

- [ ] **Step 2: Run tests and observe missing-runner failures**

  Run: `uv run --extra dev pytest tests/test_run_pointwise_v11_bakeoff.py -q`

- [ ] **Step 3: Implement the minimal runner**

  Reuse existing private-write, transport-validation, and evaluator-evidence validation helpers where their interfaces already match. Require explicit `--arm`, `--ordinals`, `--max-calls`, and `--max-cost-usd`. Fail before network access if selection size, schema identity, budget, evidence, or provider route is invalid. Never call an arm for a bare-IDK record.

- [ ] **Step 4: Verify runner tests and static checks**

  Run: `uv run --extra dev pytest tests/test_run_pointwise_v11_bakeoff.py tests/test_pointwise_judge.py -q`

  Run: `uv run --extra dev ruff check scripts/run_pointwise_v11_bakeoff.py tests/test_run_pointwise_v11_bakeoff.py`

---

### Task 4: Add bakeoff evaluation and hard gates

**Files:**
- Create: `scripts/evaluate_pointwise_v11_bakeoff.py`
- Create: `tests/test_evaluate_pointwise_v11_bakeoff.py`

**Interfaces:**
- Consumes: adjudication manifest and one or more complete per-arm results.
- Produces: precision, recall, exact-label agreement, confusion matrix, schema/provider status, short/long recall gap, cost, elapsed time, per-evidence-tier metrics, and deterministic pass/fail reasons.

- [ ] **Step 1: Write failing metric tests**

  Use hand-checked miniature fixtures to prove supported precision/recall, five-label exact agreement, confusion counts, evidence-tier splits, and the absolute correct-recall gap between records at or below versus above the manifest's median candidate-response character length.

- [ ] **Step 2: Run tests and observe missing-evaluator failures**

  Run: `uv run --extra dev pytest tests/test_evaluate_pointwise_v11_bakeoff.py -q`

- [ ] **Step 3: Implement deterministic evaluation**

  Gate an arm only when supported precision is at least 0.98, supported recall at least 0.95, exact label agreement at least 0.90, length gap at most 0.05, and provider/schema failures equal zero. Do not select a winner when no arm passes. Among passing arms, choose highest exact agreement, then highest supported recall, then lowest observed cost.

- [ ] **Step 4: Verify evaluation tests and compatibility**

  Run: `uv run --extra dev pytest tests/test_evaluate_pointwise_v11_bakeoff.py -q`

---

### Task 5: Run the 10-case smoke and audited bakeoff

**Files:**
- Create privately: `artifacts/pointwise_judge_bakeoff/v11/smoke-10/`
- Create privately: `artifacts/pointwise_judge_bakeoff/v11/audited-282/`
- Create privately: `artifacts/pointwise_judge_bakeoff/v11/evaluation.json`
- Create privately: `artifacts/pointwise_judge_bakeoff/v11/report.md`

**Interfaces:**
- Smoke ordinals: 46, 66, 330, 362 (`correct`); 384, 468 (`partial`); 263 (`incorrect`); 201 (`not_established`); 261, 213 (`nonanswer`).
- Produces: a clean-schema transport result for each arm, full audited-set metrics for smoke-clean arms, and one model recommendation or an explicit no-pass outcome.

- [ ] **Step 1: Run zero-cost preflight**

  Re-run all focused tests, validate credentials exist without printing them, confirm current OpenRouter provider/model availability, validate worst-case cost caps, and verify all 10 smoke ordinals have the expected adjudicated labels.

- [ ] **Step 2: Run three 10-case smoke arms**

  Run DeepSeek, GPT Nano, and Qwen thinking with no fallback. Inspect every response for schema validity, derived-label consistency, provider route, finish reason, usage, and unexpected truncation before any larger call set.

- [ ] **Step 3: Run all smoke-clean arms on 282 audited cases**

  Use private resumable checkpoints and bounded concurrency. Stop an arm if provider/schema failure appears rather than silently changing models or routes.

- [ ] **Step 4: Evaluate and inspect actual disagreements**

  Generate the automated metrics, then manually inspect every false `correct`, all 0↔2 disagreements, and a stratified sample of remaining errors. Preserve raw outputs and document any auditor ambiguity separately from model error.

- [ ] **Step 5: Final verification and handoff**

  Run all new and focused compatibility tests, Ruff, `git diff --check`, private-mode checks, result invariants, and a secret scan. Report the winning single model only if it passes every gate; otherwise state that no model is ready and identify the nearest failure.

## Status

- 2026-07-30: design approved by user after Opus and Sol review; production cascade explicitly rejected.
- 2026-07-30: existing provider routes verified locally: DeepSeek direct through DeepInfra; GPT Nano and Qwen thinking through pinned OpenRouter providers.
- Next action: implement Task 1 test-first.
