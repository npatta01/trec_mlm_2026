# Muse Two-Call Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a direct Meta Muse Spark Contributor adapter that performs two grounded Responses API calls per question, preserves both call records and Search provenance, resumes safely from a private cache, and runs a bounded 50-question canary.

**Architecture:** Keep the existing provider adapters unchanged. Add `trec_mlm.muse_reference_generation` with request builders, Responses parsing, two-call fallback, atomic cache helpers, and deterministic canary selection. Add a thin CLI that reads `MODEL_API_KEY`, runs only the selected 50 development questions, and writes a sanitized report. Do not send credentials to logs or store arbitrary output/token caps in requests.

**Tech Stack:** Python 3.11+, stdlib HTTP/JSON, pytest, direct `https://api.meta.ai/v1/responses`.

## Constraints

- Use `muse-spark-1.2-contributor` and Meta's native `web_search` tool.
- Always attempt the follow-up call when the first response returns a usable response ID.
- Do not set `max_output_tokens`, `max_tokens`, or a Search-use cap.
- Treat a completed second call as final; if it fails, preserve the first answer with `final_stage=1` and the structured follow-up failure.
- Keep raw provider responses, request payloads, response IDs, citations, Search actions, usage, and timestamps in the private cache.
- Completed cache records must make a rerun perform zero API calls.
- Never print or serialize the API key.
- Run the 50-question canary only; do not start a full-dataset generation.

## File map

- Create: `trec_mlm/muse_reference_generation.py`
- Create: `scripts/run_muse_canary.py`
- Create: `tests/test_muse_reference_generation.py`
- Create: `tests/test_run_muse_canary.py`
- Create: `reports/muse_canary_50_2026-08-22.md` after the live run

### Task 1: Add failing adapter and cache tests

- [x] Test request builders omit arbitrary output/Search limits and include web search.
- [x] Test parser extracts final assistant text, URL annotations, Search queries/page opens, usage, status, and response ID.
- [x] Test two-call runner passes `previous_response_id`, keeps both responses, and chooses stage 2.
- [x] Test second-call failure falls back to stage 1 with an explicit flag.
- [x] Test deterministic selection and cache idempotency.

### Task 2: Implement the adapter

- [x] Add generic initial/follow-up prompts, direct HTTP requests with safe retries, and no output cap.
- [x] Add Responses parser and structured non-secret failure diagnostics.
- [x] Add two-call runner with stage-1 fallback and complete provenance/usage records.
- [x] Add atomic private cache load/save and deterministic 50-question selection helpers.
- [x] Add a structured reference-prompt variant for grading other LLM answers.

### Task 3: Add and verify the canary CLI

- [x] Read the TSV, select exactly 50 stable questions, skip completed cache records, and checkpoint each result.
- [x] Add bounded progress output and a report with completion, fallback, Search, citation, latency, answer-length, and token metrics.
- [x] Run focused tests and compile checks.

### Task 4: Execute and review the live canary

- [x] Run the original and revised prompt variants with the saved `MODEL_API_KEY`, using separate private caches.
- [x] Verify cache/idempotency behavior through the adapter and CLI tests.
- [x] Inspect report/cache for secret leakage, truncation/incomplete responses, source-host quality signals, and follow-up failures.
- [x] Stop at the canary and document the quality/cost comparison before any full-dataset run.

### Task 5: Pilot generic ambiguity handling before scaling

**Files:**
- Modify: `trec_mlm/muse_reference_generation.py`
- Modify: `tests/test_muse_reference_generation.py`
- Create: `scripts/run_muse_ambiguity_pilot.py`
- Create: `tests/test_run_muse_ambiguity_pilot.py`
- Create: `reports/muse_ambiguity_pilot_2026-08-22.md` after the live pilot

**Interfaces:**
- The new prompt variant is named `reference_generic_branches` and uses the existing `build_muse_initial_request`, `build_muse_followup_request`, and `run_muse_two_call` interfaces.
- The pilot runner exports `PILOT_QUESTIONS`, `load_pilot_questions`, `run_pilot`, and `build_pilot_report` for tests and the command line.
- Pilot cache records use keys `<repeat>:<question_id>` so repeated calls cannot overwrite one another and can resume safely.

- [x] **Step 1: Add failing prompt and runner tests**
- [x] **Step 2: Implement the generic branch prompt variant**
- [x] **Step 3: Implement the stratified three-repeat pilot runner**
- [x] **Step 4: Run focused tests and the live pilot**
- [x] **Step 5: Inspect outputs and write the go/no-go report**

## Result

The revised reference prompt completed all 50 questions with two-call stage-2 finals and no hard failures, but one semantic regression was found: Q10's canonical answer changed from the likely ecological/habitat dimension to a plant life-form dimension. Three-repeat pilots of the original, v2, and v3 generic branch contracts showed progressive format/control improvements, but Q10 still alternated between `SINGLE` and `MULTIPLE`. Keep the two-call design, but move ambiguity adjudication to an explicit downstream policy or human/benchmark-specific step before scaling; no full-dataset run is approved yet.
