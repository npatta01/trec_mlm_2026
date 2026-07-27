# DeepSeek Pointwise Production Run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the sealed 182,764-record DeepSeek V4 Flash pointwise evaluation with a Claude Opus prompt gate, durable 100-record validation checkpoints, anomaly pauses, and a first-100 user review artifact.

**Architecture:** Extend the existing Pydantic pointwise v2 contract and hosted OpenRouter call path with a production inventory loader and a resumable block runner. Keep raw provider responses, terminal decisions, attempts, spend, block receipts, and Opus reviews private; generate only sanitized aggregate reports. The first block is a hard human gate, while later blocks proceed only when each structural and accounting receipt passes.

**Tech Stack:** Python 3.11+, Pydantic 2, pytest, urllib/OpenRouter Chat Completions, Anthropic Claude Opus through the authorized hosted provider, JSON/Markdown private artifacts.

## Global Constraints

- Exact response inventory: 182,764 filtered/deduplicated records from the frozen refinement artifact.
- Exact DeepSeek route: `deepseek/deepseek-v4-flash`, direct/non-thinking mode.
- Every completed block has exactly 100 records, except the final block of 64.
- Combined DeepSeek and Claude hard spend cap is USD 60; pause when projection exceeds USD 50.
- Every block must fail closed on missing, duplicate, unexpected, invalid, truncated, or unreconciled records.
- At most one targeted semantic repair is allowed for a malformed model result; bounded transient retry handling remains finite and ledgered.
- Raw inputs, outputs, ledgers, diagnostics, and review packets are private (`0700` directories, `0600` files); sanitized reports contain no secrets or raw records.
- The first 100 DeepSeek records and the exact production prompt/schema must be reviewed by Claude Opus before scaling.
- No external call occurs before the no-network dry run, focused tests, lint, format, and prompt-review input validation pass.
- The first-100 report is diagnostic and cannot claim general judge quality or reference correctness.

---

### Task 1: Seal production inventory, references, and Opus prompt-review packet

**Files:**
- Create: `trec_mlm/pointwise_production.py`
- Create: `scripts/build_pointwise_production_inputs.py`
- Create: `scripts/build_opus_pointwise_prompt_review.py`
- Create: `tests/test_pointwise_production.py`
- Create: `tests/test_build_pointwise_production_inputs.py`
- Create: `tests/test_build_opus_pointwise_prompt_review.py`

**Interfaces:**
- Consumes the existing `iter_discovery_responses`, `load_discovery_questions`, `load_grounded_references`, `CandidateCase`, and v2 prompt/schema loaders.
- Produces `ProductionRecord` values with stable `record_id`, `question_id`, `candidate_id`, question, provisional reference, candidate response, and source ordinal; a sealed manifest; and an Opus review packet that contains the exact proposed prompt, schema, and adjudication questions without provider/model identity leakage from the DeepSeek run.

- [ ] **Step 1: Write failing tests** for exact 182,764 coverage, deterministic ordinal/record IDs, duplicate and missing reference rejection, source-hash sealing, private permissions, and review-packet identity blinding.
- [ ] **Step 2: Run focused tests** with `uv run pytest tests/test_pointwise_production.py tests/test_build_pointwise_production_inputs.py tests/test_build_opus_pointwise_prompt_review.py -q`; confirm failures describe missing production interfaces.
- [ ] **Step 3: Implement streaming inventory construction** over the existing pretty-printed discovery source, joining the grounded reference by question ID and writing a private ordered JSONL snapshot plus manifest atomically.
- [ ] **Step 4: Implement the Opus packet builder** with exact prompt/schema hashes, review questions for ambiguity/reference evidence/invariants/repair behavior, and no DeepSeek identity or API secret.
- [ ] **Step 5: Run the focused tests and a no-network input build** against the frozen source; verify exactly 182,764 records and all private modes.
- [ ] **Step 6: Commit** with `git add trec_mlm/pointwise_production.py scripts/build_pointwise_production_inputs.py scripts/build_opus_pointwise_prompt_review.py tests/test_pointwise_production.py tests/test_build_pointwise_production_inputs.py tests/test_build_opus_pointwise_prompt_review.py && git commit -m "seal production pointwise inventory and Opus packet"`.

### Task 2: Add Claude Opus prompt-review execution and adjudication receipt

**Files:**
- Create: `scripts/run_opus_pointwise_prompt_review.py`
- Create: `trec_mlm/pointwise_opus_review.py`
- Create: `tests/test_pointwise_opus_review.py`
- Modify: `scripts/build_opus_pointwise_prompt_review.py`

**Interfaces:**
- Consumes the Task 1 review packet and an authorized Anthropic/OpenRouter credential without serializing it.
- Produces a private, schema-validated Opus review, an explicit accepted/rejected recommendation ledger, and a sealed production prompt/schema revision for Task 3.

- [ ] **Step 1: Write failing tests** for exact prompt/schema binding, strict Opus JSON parsing, refusal of malformed/identity-bearing review output, and accepted/rejected recommendation recording.
- [ ] **Step 2: Run the focused review tests** and confirm they fail before implementation.
- [ ] **Step 3: Implement the bounded Opus call** with a fixed model alias, one request plus one bounded transport retry, provider cost accounting, response-size limits, and Pydantic review validation.
- [ ] **Step 4: Implement adjudication** so only explicitly accepted prompt changes produce a new versioned contract; rejected suggestions remain recorded and cannot alter the prompt silently.
- [ ] **Step 5: Run tests, lint, and a credential-free dry run**; verify the review artifact is mode `0600` and no key appears in serialized files.
- [ ] **Step 6: Commit** with `git add scripts/run_opus_pointwise_prompt_review.py trec_mlm/pointwise_opus_review.py scripts/build_opus_pointwise_prompt_review.py tests/test_pointwise_opus_review.py && git commit -m "add sealed Opus prompt review gate"`.

### Task 3: Implement production DeepSeek block runner and durable ledgers

**Files:**
- Create: `scripts/run_deepseek_pointwise_production.py`
- Create: `trec_mlm/pointwise_production_runner.py`
- Create: `tests/test_pointwise_production_runner.py`
- Modify: `scripts/run_hosted_pointwise_smoke.py` only when a small reusable OpenRouter primitive is required and covered by existing tests.

**Interfaces:**
- Consumes the sealed Task 1 inventory and Task 2 prompt/schema receipt.
- Produces private raw response checkpoints, validated terminal decisions, physical-attempt markers, spend ledger, per-block receipts, anomaly diagnostics, and a resumable run manifest.

- [ ] **Step 1: Write failing tests** for block slicing (`100` and final `64`), crash-safe resume, exact identity coverage, durable attempt reservation, cost cap, repair/retry limits, and provider finish-reason handling.
- [ ] **Step 2: Run the focused runner tests** to verify the expected failures.
- [ ] **Step 3: Implement the production runner** using the existing strict OpenRouter transport and v2 Pydantic decision contract, with ordered inventory reads, atomic checkpoint writes, and no in-memory requirement to load all 182,764 records.
- [ ] **Step 4: Implement the block gate** so the runner refuses to send the next block until the previous receipt proves exact coverage, schema/invariant validity, accounting reconciliation, prompt/model/source hashes, and private permissions.
- [ ] **Step 5: Implement bounded soft anomaly checks** for label/score/flag/confidence distributions, repeated outputs, token/cost/latency shifts, repair/retry rates, and provider drift; anomaly findings pause without mutating prior outputs.
- [ ] **Step 6: Run focused tests plus a no-network 2-block fixture run**, confirming resume after interruption and a final partial block.
- [ ] **Step 7: Commit** with `git add scripts/run_deepseek_pointwise_production.py trec_mlm/pointwise_production_runner.py tests/test_pointwise_production_runner.py scripts/run_hosted_pointwise_smoke.py && git commit -m "add gated DeepSeek production runner"`.

### Task 4: Implement first-100 and milestone audit/report artifacts

**Files:**
- Create: `scripts/build_pointwise_production_report.py`
- Create: `scripts/build_pointwise_opus_audit_packet.py`
- Create: `tests/test_build_pointwise_production_report.py`
- Create: `tests/test_build_pointwise_opus_audit_packet.py`

**Interfaces:**
- Consumes private block receipts, terminal decisions, spend/attempt ledgers, and deterministic audit strata.
- Produces a private first-100 Opus audit packet, milestone audit packets, and sanitized Markdown reports with no raw question/response text.

- [ ] **Step 1: Write failing tests** for sanitized output, deterministic seeded sampling, inclusion of all repaired/invalid/low-confidence/rare-label records, cost projections, and claim-boundary wording.
- [ ] **Step 2: Run focused report tests** and confirm failures.
- [ ] **Step 3: Implement audit sampling** with first-block all-100 coverage, milestone strata, anomaly-triggered additions, and final seeded random sample behavior.
- [ ] **Step 4: Implement reports** for coverage, parse/invariant failures, retries/repairs, cost, latency, distribution checks, Opus findings, and projected total cost; redact raw records and secrets.
- [ ] **Step 5: Run tests, secret scans, and a fixture report build**; verify all private artifacts are `0600` and sanitized reports contain no candidate text or API-key prefix.
- [ ] **Step 6: Commit** with `git add scripts/build_pointwise_production_report.py scripts/build_pointwise_opus_audit_packet.py tests/test_build_pointwise_production_report.py tests/test_build_pointwise_opus_audit_packet.py && git commit -m "add production audit and progress reports"`.

### Task 5: End-to-end preflight and authorized first-100 execution

**Files:**
- Modify: `docs/superpowers/specs/2026-07-27-deepseek-pointwise-production-run-design.md` only for adjudicated Opus changes, if any.
- Create: `.superpowers/sdd/2026-07-27-deepseek-pointwise-production-run/task-5-execution.md`

**Interfaces:**
- Consumes the sealed Task 1–4 artifacts and authorized credentials.
- Produces the first-100 DeepSeek run, validated block receipt, Opus semantic audit, and a sanitized report for the user’s scale-up decision.

- [ ] **Step 1: Run the full focused suite, adjacent pointwise suite, Ruff, formatting, and no-network dry run.** Require zero failures before networking.
- [ ] **Step 2: Run the Opus prompt-review call and validate its adjudication receipt.** Stop if the output is malformed or if an accepted change requires a new contract not covered by tests.
- [ ] **Step 3: Seal the final prompt/schema/manifest and run exactly the first 100 DeepSeek records under a conservative first-block spend budget.** Do not start record 101.
- [ ] **Step 4: Run the exhaustive 100-record gate and build the private Opus semantic audit packet.** Audit all 100 records.
- [ ] **Step 5: Build and inspect the sanitized first-100 report**, including projected cost, surprising patterns, parsing/validation findings, representative Opus findings, and explicit claim limits.
- [ ] **Step 6: Record the execution result and remaining authorization boundary** in `.superpowers/sdd/2026-07-27-deepseek-pointwise-production-run/task-5-execution.md`.

### Task 6: Final verification and handoff

**Files:**
- Modify: `.superpowers/sdd/2026-07-27-deepseek-pointwise-production-run/progress.md`
- Modify: `reports/pointwise-judge-v2-hosted-canary-2026-07-27.md` only if cross-run comparison is needed.

- [ ] **Step 1: Run the complete pointwise production test suite and Ruff checks.**
- [ ] **Step 2: Verify the first-100 artifact counts, hashes, modes, secret scan, spend cap, and no unexpected external files.**
- [ ] **Step 3: Run a final whole-branch review against this plan and the design specification.**
- [ ] **Step 4: Update the SDD ledger with commits, tests, artifact paths, actual spend, and the explicit next action: user approval before record 101.**
- [ ] **Step 5: Commit only implementation/report/plan-ledger files belonging to this work; preserve unrelated dirty files.**

