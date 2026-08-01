# Simplified Pointwise v10 Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run a private 100-response DeepSeek V4 Flash thinking pilot that grades only substantive historical responses against citation-free Gemini evaluator evidence using the approved `0/1/2` contract.

**Architecture:** Add an isolated v10 prompt/schema and Pydantic normalizer without changing v1-v9. Extend the existing DeepInfra validation runner with a conservative bare-IDK prefilter, evaluator-only prompt labels, fail-closed citation validation, and a 100-case v10 boundary. Reuse private resumable artifacts, prompt caching, and the same-provider-only transport retry.

**Tech Stack:** Python 3.11, Pydantic v2, JSON Schema, pytest, DeepInfra OpenAI-compatible API.

## Global Constraints

- Judge only response records that actually exist.
- Make no API call for a bare “I don't know” response, including known punctuation/encoding variants; do not filter substantive answers that merely express uncertainty.
- The historical candidate never sees Gemini material. Only the separate judge sees the question, Gemini evaluator evidence, and historical response.
- Pass no model ID, response ID, citations, URLs, source list, or grounding metadata to the judge.
- Preserve the two verified meaningful bracketed years `[1971]` and `[1978]`; fail closed on unexpected citation-like markers rather than deleting text broadly.
- `2/supported`: central answer supported with no material contradiction; omissions and peripheral reference silence are allowed.
- `1/mixed`: useful support plus material contradiction.
- `1/reference_silent`: central answer is not established by Gemini evidence.
- `0/contradicted`: central answer contradicted with no useful supported answer.
- `0/nonanswer`: irrelevant or non-substantive answer that escaped deterministic screening.
- This measures agreement with Gemini-grounded evidence, not absolute correctness.
- Keep v1-v9 unchanged, use DeepSeek V4 Flash thinking through DeepInfra only, enable caching, allow at most one identical same-provider transport retry, and use no provider fallback.
- Keep private inputs/results mode 0600 and directories mode 0700; never print or persist secrets.
- Make no commit because this worktree contains overlapping uncommitted user work and the user did not request one.

---

### Task 1: Add the strict v10 contract

**Files:**
- Create: `protocols/pointwise_judge_bakeoff/pointwise-v10.txt`
- Create: `protocols/pointwise_judge_bakeoff/decision-v10.schema.json`
- Modify: `trec_mlm/pointwise_judge.py`
- Modify: `tests/test_pointwise_judge.py`

**Interfaces:**
- Produces: `DecisionV10`, `normalized_decision_v10_schema()`, `normalize_judge_decision_v10(raw: str) -> dict[str, object]`, and `load_prompt_contract(..., version="v10")`.

- [x] **Step 1: Write failing v10 behavioral tests**

  Cover exact schema fields, the grade/reason validity matrix, clean parsing, narrow casing/whitespace repair, malformed output as unscorable, partial-but-supported grade 2, mixed/reference-silent grade 1, and contradicted/nonanswer grade 0.

- [x] **Step 2: Run the focused tests and observe the expected missing-v10 failure**

  Run: `pytest tests/test_pointwise_judge.py -k v10 -q`

- [x] **Step 3: Add the approved prompt, five generic few-shot examples, strict schema, and Pydantic normalizer**

  The model emits exactly `{"grade": 0|1|2, "reason": "supported|mixed|reference_silent|contradicted|nonanswer"}`. The prompt explicitly says the historical response was written independently without the later evaluator evidence.

- [x] **Step 4: Run focused and compatibility tests**

  Run: `pytest tests/test_pointwise_judge.py -q`

### Task 2: Enforce citation and non-answer boundaries in the existing runner

**Files:**
- Modify: `scripts/run_deepinfra_pointwise_v8_validation.py`
- Modify: `tests/test_deepinfra_pointwise_v8_validation.py`

**Interfaces:**
- Produces: `_is_bare_idk_response(text: str) -> bool`, `_evaluator_evidence(question_id: str, answer: str) -> str`, v10 request construction, and exactly-100 v10 selection validation.

- [x] **Step 1: Write failing behavioral tests**

  Assert bare corrupted IDK variants are rejected while longer uncertainty is retained; the two verified bracketed years remain unchanged; unexpected bracket markers and URLs fail closed; v10 messages contain no IDs, model identity, sources, or old reference labels; the request uses thinking, strict schema, a v10-specific cache key, 8192 output tokens, and no fallback.

- [x] **Step 2: Run runner tests and observe expected failures**

  Run: `pytest tests/test_deepinfra_pointwise_v8_validation.py -k v10 -q`

- [x] **Step 3: Add minimal v10 runner support**

  Validate the raw Gemini artifact before deriving evaluator evidence. Preserve verified meaningful bracket notation, exclude known bare-IDK records before selection, store only the derived evaluator evidence in new private v10 records, and bind transform version plus selected ordinals into the cache identity.

- [x] **Step 4: Run the full focused runner and contract suites**

  Run: `pytest tests/test_pointwise_judge.py tests/test_deepinfra_pointwise_v8_validation.py -q`

### Task 3: Run and audit 100 substantive responses

**Files:**
- Create privately: `artifacts/pointwise_production/deepinfra-v10-thinking-pilot-100/result.json`
- Create privately: `artifacts/pointwise_production/deepinfra-v10-thinking-pilot-100/audit.md`

**Interfaces:**
- Consumes: the sealed 500-case validation inventory, its manifest, frozen Gemini artifact, and exactly 100 selected non-IDK ordinals.
- Produces: one private resumable result plus an audit with transport/schema quality, grade/reason distributions, cost, reference-silent coverage, and manual findings from the actual cases.

- [x] **Step 1: Select exactly 100 substantive cases reproducibly**

  Start from the prior seeded holdout for comparability, remove any bare-IDK ordinals, and deterministically replace them from unused eligible sealed-inventory ordinals. Validate 100 unique records before any call.

- [x] **Step 2: Run the DeepInfra thinking pilot**

  Use eight workers, temperature 0, `reasoning_effort=high`, 8192 output tokens, and one same-provider transport retry. Stop and inspect if transport or schema failures appear.

- [x] **Step 3: Audit every output**

  Inspect the full grade/reason matrix for impossible combinations, report parser/transport failures and cost, examine all grade 0/1 cases plus a sample of grade 2 cases against the supplied Gemini evidence, and document surprising or rubric-violating decisions without changing the frozen run.

- [x] **Step 4: Run final verification**

  Run: `pytest tests/test_pointwise_judge.py tests/test_deepinfra_pointwise_v8_validation.py -q`; run `git diff --check`; verify private artifact modes and ensure no secret-like values appear in changed source/artifacts.

## Status

- 2026-07-30: design approved in conversation; Opus and Sol reviews incorporated.
- 2026-07-30: read-only preflight verified that `[1971]` and `[1978]` are meaningful years, not citations. An expanded exact full-string audit found 733 bare-IDK encoding variants among 182,764 prepared candidate records; longer uncertainty remains eligible.
- 2026-07-30: Task 1 complete after transport-schema review; 27 v10 tests and 112 full pointwise tests pass, with the five grade/reason pairs enforced locally.
- 2026-07-30: Task 2 complete after CRLF fail-closed review; 13 v10 runner tests and 134 combined runner/contract tests pass.
- 2026-07-30: Task 3 steps 1-3 complete. The 100-case DeepInfra run produced 100 clean outputs with no provider errors or retries in 307.2 seconds for $0.01916. Full semantic review found 80/100 advisor exact-pair agreement (79/100 after one primary-review boundary adjustment); the dominant error is conservatively downgrading supported central answers from grade 2 to grade 1.
- 2026-07-30: Final verification passes: 134 focused tests, Ruff, and `git diff --check`; private run artifacts are mode 0600 under a mode-0700 directory, and the DeepInfra token is absent from reviewed source/artifacts.
- 2026-07-30: Follow-on 500-case run complete. Four bare-IDK responses were deterministically labeled without provider calls; 396 new DeepInfra calls plus 100 reused results produced 500 valid records with no errors or retries. Total observed cost was $0.08888. Full review of all 179 low-score records found strong conservative undergrading; 103 reviewed grade-2 records had 102 exact audit agreements.
- 2026-07-30: Follow-on verification passes: 500 unique ordinal-aligned records, valid decision pairs, 496 normal provider stops, 134 tests, Ruff, `git diff --check`, private modes, and no DeepInfra token in reviewed artifacts.
- Next action: use grade 2 as the high-confidence signal and route grades 0/1 to a second pass before using the results for model ranking.
