# Local Pointwise Judge Smoke Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and execute a reproducible ten-record pointwise-judge smoke test across four local models, obtain blind Advisor and Claude reviews, and stop with a case-level user review packet.

**Architecture:** Add small domain modules for the frozen prompt/schema, private corpus loading and diagnostic selection, resumable OpenAI-compatible inference, blind review packets, and reporting. Runtime orchestration remains outside the domain modules: Lemonade or a rootless Podman vLLM process exposes one loopback endpoint at a time, while the runner records exact runtime receipts and immutable raw responses.

**Tech Stack:** Python 3.11+, standard library (`dataclasses`, `hashlib`, `json`, `urllib`), pytest, Lemonade Server 11.5/OpenAI-compatible API, rootless Podman/vLLM where needed, Claude Code CLI, Codex Advisor subagent.

## Global Constraints

- Implement only the ten-record local smoke and review checkpoint; do not run 100, 500, hosted judges, or the full corpus.
- Use grounded Gemini as a provisional pivot, never unquestionable gold.
- Hide candidate-model identity from judges and reviewers; retain it only in a mode-0600 sealed map.
- Bind inference endpoints only to `127.0.0.1`; never publish or expose a service.
- Preserve raw outputs, allow at most one predefined JSON repair, and never overwrite failures.
- Freeze source hashes, prompt/schema hashes, checkpoint revision, quantization, runtime versions, decoding settings, and metrics.
- The only authorized new model download is `Qwen3.6-35B-A3B-GGUF`, approximately 23.3 GB.
- Exclude GLM-4.7-Flash, Nemotron-3-Nano, and Qwen3.5-122B-A10B.
- Stop after the Advisor/Claude-reviewed ten-case report and obtain user approval before any 100-record work.
- Keep case-level text private under `artifacts/pointwise_judge_bakeoff/`; committed reports must be sanitized.

---

## File map

- `trec_mlm/pointwise_judge.py`: prompt contract, case/decision types, strict decision validation, and prompt rendering.
- `trec_mlm/pointwise_sampling.py`: grounded-reference and discovery-response loading, diagnostic proxies, deterministic ten-case selection, and sealed blinding.
- `trec_mlm/pointwise_runner.py`: loopback-only OpenAI-compatible calls, one-repair policy, content-addressed resume cache, and run receipts.
- `trec_mlm/pointwise_review.py`: anonymous result permutations, two-pass reviewer packets, reviewer-record validation, and comparison rows.
- `trec_mlm/pointwise_report.py`: smoke metrics and private Markdown report rendering.
- `protocols/pointwise_judge_bakeoff/pointwise-v1.txt`: frozen judge instruction.
- `protocols/pointwise_judge_bakeoff/decision-v1.schema.json`: frozen structured-output schema.
- `protocols/pointwise_judge_bakeoff/local-models-v1.json`: exact local shortlist and runtime expectations.
- `protocols/pointwise_judge_bakeoff/reviewer-v1.txt`: frozen two-pass review instruction.
- `scripts/build_pointwise_judge_smoke.py`: private sample/sealed-map CLI.
- `scripts/preflight_local_pointwise_models.py`: fail-closed host, model-cache, endpoint, and runtime preflight.
- `scripts/run_local_pointwise_judge.py`: one-model-at-a-time resumable generation CLI.
- `scripts/build_pointwise_judge_review.py`: blind review packet and comparison CLI.
- `scripts/build_pointwise_judge_smoke_report.py`: private report/summary CLI.
- `tests/test_pointwise_judge.py`, `tests/test_pointwise_sampling.py`, `tests/test_pointwise_runner.py`, `tests/test_pointwise_review.py`, `tests/test_pointwise_report.py`: focused unit tests.
- `tests/test_pointwise_judge_cli.py`: CLI integration tests over fabricated inputs and a fake local endpoint.

---

### Task 1: Frozen prompt and strict decision contract

**Files:**
- Create: `protocols/pointwise_judge_bakeoff/pointwise-v1.txt`
- Create: `protocols/pointwise_judge_bakeoff/decision-v1.schema.json`
- Create: `trec_mlm/pointwise_judge.py`
- Test: `tests/test_pointwise_judge.py`

**Interfaces:**
- Produces: `CandidateCase`, `JudgeDecision`, `PromptContract` dataclasses.
- Produces: `load_prompt_contract(protocol_dir: Path) -> PromptContract`.
- Produces: `render_judge_messages(case: CandidateCase, contract: PromptContract, *, repair_raw: str | None = None) -> list[dict[str, str]]`.
- Produces: `parse_judge_decision(raw: str) -> JudgeDecision` and `decision_to_dict(decision: JudgeDecision) -> dict[str, object]`.

- [ ] **Step 1: Write failing contract tests**

```python
def test_parse_judge_decision_rejects_extra_keys() -> None:
    raw = json.dumps({
        "label": "good", "correctness": 4, "completeness": 2,
        "relevance": 2, "error_flags": [], "reference_status": "sound",
        "confidence": "high", "rationale": "No material error.", "extra": 1,
    })
    with pytest.raises(ValueError, match="exact keys"):
        parse_judge_decision(raw)


def test_render_messages_keeps_reference_provisional() -> None:
    messages = render_judge_messages(CASE, CONTRACT)
    content = messages[-1]["content"]
    assert "PROVISIONAL REFERENCE" in content
    assert "may be incomplete or wrong" in content
    assert "CANDIDATE RESPONSE" in content
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest tests/test_pointwise_judge.py -v`

Expected: collection fails because `trec_mlm.pointwise_judge` does not exist.

- [ ] **Step 3: Write the frozen rubric and schema**

The prompt must define the four labels, the 0–4/0–2/0–2 scales, correctness gating, the closed flags `entity_error`, `number_error`, `date_error`, `negation_error`, `unsupported_specificity`, `refusal`, `ambiguity`, and `reference_conflict`, and require a rationale of 1–80 words. The JSON schema must set `additionalProperties: false`, enumerate every categorical value, and bound every integer.

- [ ] **Step 4: Implement the strict domain contract**

```python
@dataclass(frozen=True, slots=True)
class CandidateCase:
    question_id: str
    response_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    strata: tuple[str, ...]
    selection_reason: str


@dataclass(frozen=True, slots=True)
class JudgeDecision:
    label: str
    correctness: int
    completeness: int
    relevance: int
    error_flags: tuple[str, ...]
    reference_status: str
    confidence: str
    rationale: str
```

Use exact-key validation, reject booleans as integers, reject duplicate flags,
require the closed vocabularies, enforce the rationale word bound, and reject a
`good` label when correctness is below 4 or relevance below 2.

- [ ] **Step 5: Run contract tests and the existing reference tests**

Run: `uv run pytest tests/test_pointwise_judge.py tests/test_reference_adjudication.py tests/test_reference_data.py -v`

Expected: all pass.

- [ ] **Step 6: Commit the contract**

```bash
git add protocols/pointwise_judge_bakeoff/pointwise-v1.txt protocols/pointwise_judge_bakeoff/decision-v1.schema.json trec_mlm/pointwise_judge.py tests/test_pointwise_judge.py
git commit -m "add pointwise judge contract"
```

---

### Task 2: Private input loading and deterministic diagnostic selection

**Files:**
- Create: `trec_mlm/pointwise_sampling.py`
- Create: `scripts/build_pointwise_judge_smoke.py`
- Test: `tests/test_pointwise_sampling.py`
- Test: `tests/test_pointwise_judge_cli.py`

**Interfaces:**
- Consumes: `CandidateCase` from Task 1.
- Produces: `GroundedReference(question_id: str, question: str, answer: str)` and `CandidateResponse(question_id: str, candidate_id: str, answer: str)`.
- Produces: `load_grounded_references(path: Path) -> dict[str, GroundedReference]`.
- Produces: `iter_discovery_responses(path: Path) -> Iterator[CandidateResponse]`.
- Produces: `select_diagnostic_cases(references: Mapping[str, GroundedReference], responses: Iterable[CandidateResponse], *, seed: int = 2026) -> tuple[list[CandidateCase], dict[str, str]]`.
- Produces: private `smoke_manifest.json` and mode-0600 `sealed_candidates.json`.

- [ ] **Step 1: Write failing loader and selection tests with fabricated data**

```python
def test_select_diagnostic_cases_is_exact_and_identity_blind(tmp_path: Path) -> None:
    references, responses = diagnostic_fixture()
    cases, sealed = select_diagnostic_cases(references, responses, seed=2026)
    assert len(cases) == 10
    assert len({case.response_id for case in cases}) == 10
    assert set(sealed) == {case.response_id for case in cases}
    assert all(candidate not in json.dumps([asdict(case) for case in cases])
               for candidate in sealed.values())
    assert {case.selection_reason for case in cases} == set(DIAGNOSTIC_REASONS)


def test_grounded_loader_requires_exact_discovery_coverage(tmp_path: Path) -> None:
    path = write_grounded_cache(tmp_path, finish_reason="stop")
    loaded = load_grounded_references(path)
    assert loaded["q1"].answer == "Grounded answer."
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py -v`

Expected: import failure for `trec_mlm.pointwise_sampling`.

- [ ] **Step 3: Implement validated input loaders**

The grounded loader accepts the preserved generation-cache shape with a
top-level `records` object and requires nonempty `question_id`, `question`, and
`answer`, a successful finish status, unique IDs, and 7,759 records for a real
run. The discovery loader streams the pretty-printed JSON format already parsed
by `trec_mlm.reference_sampling`, emitting one response at a time without
retaining the 1.16-million-cell corpus.

- [ ] **Step 4: Implement ten stable diagnostic proxies**

Use deterministic, explicitly proxy-labeled reasons:

```python
DIAGNOSTIC_REASONS = (
    "high_overlap_control", "short_incomplete_proxy", "low_overlap_error_proxy",
    "entity_mismatch_proxy", "number_or_date_mismatch_proxy",
    "negation_mismatch_proxy", "verbose_error_proxy", "refusal_proxy",
    "underspecified_proxy", "possible_reference_conflict_proxy",
)
```

Candidate pools must exclude exact duplicate question/candidate pairs and use
stable SHA-256 tie-breaking over `seed`, `question_id`, and `candidate_id`.
`response_id` is SHA-256 over `question_id`, candidate answer, and candidate ID;
only the sealed file maps that ID back to `candidate_id`.

- [ ] **Step 5: Implement the sample CLI and private permissions**

```bash
uv run python scripts/build_pointwise_judge_smoke.py \
  --discovery data/dev/llm_discovery_data.json \
  --grounded artifacts/reference_answer_validation/grounding/gemini36_grounded_all_7759.json \
  --output-dir artifacts/pointwise_judge_bakeoff/smoke-v2 \
  --seed 2026
```

The CLI writes atomically, records source hashes and selection reasons, sets the
sealed file to mode `0600`, and fails if output already exists with a different
manifest hash.

- [ ] **Step 6: Run tests and verify the CLI on fabricated fixtures**

Run: `uv run pytest tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py -v`

Expected: all pass; no fixture candidate IDs occur in the blind manifest.

- [ ] **Step 7: Commit sampling**

```bash
git add trec_mlm/pointwise_sampling.py scripts/build_pointwise_judge_smoke.py tests/test_pointwise_sampling.py tests/test_pointwise_judge_cli.py
git commit -m "add diagnostic pointwise smoke sampling"
```

---

### Task 3: Loopback-only resumable inference runner

**Files:**
- Create: `trec_mlm/pointwise_runner.py`
- Create: `scripts/run_local_pointwise_judge.py`
- Test: `tests/test_pointwise_runner.py`
- Modify: `tests/test_pointwise_judge_cli.py`

**Interfaces:**
- Consumes: `CandidateCase`, `PromptContract`, `JudgeDecision`, and the private smoke manifest.
- Produces: `validate_loopback_endpoint(url: str) -> str`.
- Produces: `call_openai_judge(endpoint: str, model: str, messages: Sequence[Mapping[str, str]], schema: Mapping[str, object], *, timeout_seconds: float) -> tuple[str, Mapping[str, object]]`.
- Produces: `run_cases(cases: Sequence[CandidateCase], *, model_alias: str, model_id: str, contract: PromptContract, output_path: Path, runtime_receipt: Mapping[str, object], call: JudgeCall) -> dict[str, object]`.

- [ ] **Step 1: Write failing endpoint, repair, and resume tests**

```python
@pytest.mark.parametrize("url", [
    "http://0.0.0.0:13305/v1", "http://192.168.1.2:8000/v1",
    "https://api.example.com/v1", "http://localhost.example.com/v1",
])
def test_endpoint_must_be_literal_loopback(url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        validate_loopback_endpoint(url)


def test_run_cases_repairs_malformed_json_once_and_preserves_raw(tmp_path: Path) -> None:
    call = ScriptedCall(["not-json", VALID_DECISION_JSON])
    result = run_cases([CASE], model_alias="judge-a", model_id="model-a",
                       contract=CONTRACT, output_path=tmp_path / "run.json",
                       runtime_receipt=RECEIPT, call=call)
    record = result["records"][CASE.response_id]
    assert record["attempts"][0]["raw"] == "not-json"
    assert record["attempts"][1]["parsed"]["label"] == "good"
    assert len(call.requests) == 2
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/test_pointwise_runner.py -v`

Expected: module import failure.

- [ ] **Step 3: Implement the safe OpenAI-compatible call**

Allow only literal `127.0.0.1` or `::1`, require `/v1`, use
`response_format={"type":"json_schema", ...}` on the first call, and retain the
entire response body only in the private run cache. Reject redirects, missing
choices, empty content, mismatched returned model IDs, and nonnumeric usage.

- [ ] **Step 4: Implement content-addressed resume and one repair**

Each record stores `input_sha256`, prompt/schema hashes, model alias and exact
ID, runtime receipt hash, attempt list, final parsed decision or explicit
failure, elapsed seconds, and usage. Reuse a cached record only when every hash
matches. The repair request includes the malformed raw text and says to return
only a corrected object under the unchanged schema.

- [ ] **Step 5: Implement the one-model CLI**

```bash
uv run python scripts/run_local_pointwise_judge.py \
  --manifest artifacts/pointwise_judge_bakeoff/smoke-v2/smoke_manifest.json \
  --manifest-sha256 <independently-reviewed-manifest-sha256> \
  --protocol-dir protocols/pointwise_judge_bakeoff \
  --registry protocols/pointwise_judge_bakeoff/local-models-v1.json \
  --registry-sha256 <independently-reviewed-registry-sha256> \
  --model-alias qwen36-35b-a3b-gguf \
  --model-id Qwen3.6-35B-A3B-GGUF \
  --base-url http://127.0.0.1:13305/v1 \
  --runtime-receipt artifacts/pointwise_judge_bakeoff/smoke-v2/runtime/qwen36-35b-a3b-gguf.json \
  --runtime-receipt-sha256 <independently-reviewed-receipt-sha256> \
  --output artifacts/pointwise_judge_bakeoff/smoke-v2/runs/qwen36-35b-a3b-gguf.json
```

- [ ] **Step 6: Run runner and CLI tests**

Run: `uv run pytest tests/test_pointwise_runner.py tests/test_pointwise_judge_cli.py -v`

Expected: all pass, including simulated interruption/resume and mismatched-hash rejection.

- [ ] **Step 7: Commit the runner**

```bash
git add trec_mlm/pointwise_runner.py scripts/run_local_pointwise_judge.py tests/test_pointwise_runner.py tests/test_pointwise_judge_cli.py
git commit -m "add resumable local pointwise runner"
```

---

### Task 4: Local model registry and fail-closed preflight

**Files:**
- Create: `protocols/pointwise_judge_bakeoff/local-models-v1.json`
- Create: `scripts/preflight_local_pointwise_models.py`
- Modify: `tests/test_pointwise_judge_cli.py`

**Interfaces:**
- Consumes: local Hugging Face cache, Lemonade 11.5 registry, `rocminfo`, memory/disk facts, and loopback model-list endpoints.
- Produces: one JSON runtime receipt per model under `artifacts/pointwise_judge_bakeoff/smoke-v2/runtime/`.
- Produces: exit 0 only when source hashes, model artifacts, GPU visibility, endpoint binding, and a deterministic schema canary all pass.

- [ ] **Step 1: Write failing registry/preflight tests**

```python
def test_registry_contains_exact_approved_local_pool() -> None:
    registry = load_registry(PROTOCOL_DIR / "local-models-v1.json")
    assert set(registry) == {
        "qwen36-35b-a3b-gguf", "qwen36-27b-fp8",
        "gpt-oss-20b", "gemma4-31b-q4",
    }


def test_preflight_fails_when_model_endpoint_is_not_loopback(tmp_path: Path) -> None:
    result = run_preflight(fixture_registry(base_url="http://0.0.0.0:8000/v1"))
    assert result["passed"] is False
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/test_pointwise_judge_cli.py -k preflight -v`

Expected: preflight loader/script is missing.

- [ ] **Step 3: Freeze the four-model registry**

Record these logical candidates without silently equating representations:

```json
{
  "qwen36-35b-a3b-gguf": {"model_id": "Qwen3.6-35B-A3B-GGUF", "runtime": "lemonade-llamacpp-rocm", "new_download_authorized": true},
  "qwen36-27b-fp8": {"model_id": "Qwen/Qwen3.6-27B-FP8", "runtime": "vllm-rocm", "new_download_authorized": false},
  "gpt-oss-20b": {"model_id": "openai/gpt-oss-20b", "runtime": "vllm-rocm", "new_download_authorized": false},
  "gemma4-31b-q4": {"model_id": "google/gemma-4-31B-it-qat-q4_0-gguf", "runtime": "lemonade-llamacpp-rocm-custom", "new_download_authorized": false}
}
```

Include exact cached revision or GGUF filename, expected byte size, context 4096,
temperature 0, maximum 384 output tokens, and one request at a time. The script
must compare actual artifacts with this registry and stop on drift.

- [ ] **Step 4: Implement the preflight receipt**

Capture Lemonade and backend versions, `gfx1151`, visible memory, free disk,
checkpoint path/revision/size, endpoint model list, context, decoding settings,
canary raw/parsed result, startup seconds, and whether any download occurred.
Never record environment variable values or `/etc/lemonade/conf.d/zz-secrets.conf`.

- [ ] **Step 5: Run tests and static secret checks**

Run: `uv run pytest tests/test_pointwise_judge_cli.py -k preflight -v`

Run: `rg -n '(AIza|sk-[A-Za-z0-9]|AQ\.)' protocols/pointwise_judge_bakeoff scripts/preflight_local_pointwise_models.py tests/test_pointwise_judge_cli.py`

Expected: tests pass; `rg` has no matches.

- [ ] **Step 6: Commit preflight support**

```bash
git add protocols/pointwise_judge_bakeoff/local-models-v1.json scripts/preflight_local_pointwise_models.py tests/test_pointwise_judge_cli.py
git commit -m "add local judge runtime preflight"
```

---

### Task 5: Blind two-pass review records

**Files:**
- Create: `protocols/pointwise_judge_bakeoff/reviewer-v1.txt`
- Create: `trec_mlm/pointwise_review.py`
- Create: `scripts/build_pointwise_judge_review.py`
- Test: `tests/test_pointwise_review.py`

**Interfaces:**
- Consumes: blind smoke manifest and all completed local run caches.
- Produces: `build_independent_review_packet(...) -> list[dict[str, object]]` without judge decisions.
- Produces: `build_critique_packet(...) -> tuple[list[dict[str, object]], dict[str, object]]` with per-case anonymous judge labels and a sealed alias map.
- Produces: `validate_reviewer_record(record: Mapping[str, object], expected_response_ids: Collection[str]) -> None`.
- Produces: `compare_reviews(...) -> list[dict[str, object]]`.

- [ ] **Step 1: Write failing blinding and coverage tests**

```python
def test_critique_packet_has_stable_anonymous_judges() -> None:
    packet, sealed = build_critique_packet(CASES, RUNS, seed=2026)
    serialized = json.dumps(packet)
    assert all(alias not in serialized for alias in RUNS)
    assert set(sealed["judge_aliases"].values()) == set(RUNS)
    assert all(set(row["judges"]) == {"judge_a", "judge_b"} for row in packet)


def test_reviewer_coverage_requires_every_case_once() -> None:
    with pytest.raises(ValueError, match="missing"):
        validate_reviewer_coverage(EXPECTED_IDS, REVIEW_ROWS[:-1])
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/test_pointwise_review.py -v`

Expected: module import failure.

- [ ] **Step 3: Implement deterministic blinding and two-pass packets**

Pass 1 contains only response ID, question, provisional reference, and candidate
response. Pass 2 adds anonymous judge decisions in an order derived from
SHA-256 over seed, response ID, and model alias. Runtime, latency, model names,
and verbosity metadata stay hidden. Store the alias map separately at mode
`0600`.

- [ ] **Step 4: Implement strict reviewer records**

Each reviewer independently returns the Task 1 decision fields plus
`reviewer`, `response_id`, `decisive_disagreements`, `prompt_findings`, and
`review_phase`. Require exactly ten pass-1 decisions and ten pass-2 critiques,
unique IDs, and no unknown judge aliases.

- [ ] **Step 5: Implement the packet CLI**

```bash
uv run python scripts/build_pointwise_judge_review.py packets \
  --manifest artifacts/pointwise_judge_bakeoff/smoke-v2/smoke_manifest.json \
  --runs-dir artifacts/pointwise_judge_bakeoff/smoke-v2/runs \
  --output-dir artifacts/pointwise_judge_bakeoff/smoke-v2/review \
  --seed 2026
```

- [ ] **Step 6: Run tests**

Run: `uv run pytest tests/test_pointwise_review.py -v`

Expected: all pass.

- [ ] **Step 7: Commit review packet support**

```bash
git add protocols/pointwise_judge_bakeoff/reviewer-v1.txt trec_mlm/pointwise_review.py scripts/build_pointwise_judge_review.py tests/test_pointwise_review.py
git commit -m "add blind pointwise review packets"
```

---

### Task 6: Smoke metrics and private report

**Files:**
- Create: `trec_mlm/pointwise_report.py`
- Create: `scripts/build_pointwise_judge_smoke_report.py`
- Test: `tests/test_pointwise_report.py`

**Interfaces:**
- Consumes: manifest, run caches, Advisor pass-1/pass-2 records, Claude pass-1/pass-2 records, and sealed judge aliases.
- Produces: `build_smoke_summary(...) -> dict[str, object]`.
- Produces: `render_private_report(summary: Mapping[str, object]) -> str`.
- Produces: private `smoke-review.md`, private `smoke-summary.json`, and sanitized `reports/pointwise_judge_bakeoff/local-smoke-summary.json`.

- [ ] **Step 1: Write failing metric and redaction tests**

```python
def test_summary_reports_schema_failures_and_false_confidence() -> None:
    summary = build_smoke_summary(FIXTURE_INPUTS)
    assert summary["judges"]["judge_a"]["schema_success"] == 0.9
    assert summary["judges"]["judge_a"]["false_confident_errors"] == 1


def test_sanitized_summary_excludes_case_text_and_candidate_ids() -> None:
    sanitized = sanitize_summary(PRIVATE_SUMMARY)
    text = json.dumps(sanitized)
    assert "question" not in text
    assert "candidate_answer" not in text
    assert "candidate_id" not in text
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest tests/test_pointwise_report.py -v`

Expected: module import failure.

- [ ] **Step 3: Implement metrics without manufacturing gold**

Report exact agreement separately with Advisor and Claude, agreement where both
reviewers concur, reviewer disagreement, adjacent `good`/`partial` agreement,
false-confident errors against reviewer consensus only, schema/repair/failure
rates, label distributions, reference-conflict flags, latency, throughput,
startup, memory, and every diagnostic proxy. Mark consensus-derived fields as
`model_reviewer_consensus`, never `gold` or `accuracy`.

- [ ] **Step 4: Implement the private case table and sanitized summary**

The private report shows all ten inputs, anonymous judge results, both review
passes, and proposed prompt changes. The sanitized JSON includes only counts,
rates, runtime facts, hashes, and unresolved-case counts.

- [ ] **Step 5: Run report tests and commit**

Run: `uv run pytest tests/test_pointwise_report.py -v`

```bash
git add trec_mlm/pointwise_report.py scripts/build_pointwise_judge_smoke_report.py tests/test_pointwise_report.py
git commit -m "add local judge smoke reporting"
```

---

### Task 7: Full harness verification before model work

**Files:**
- Modify only if a defect is found: files from Tasks 1–6.

**Interfaces:**
- Consumes: complete code harness.
- Produces: passing focused/full tests and no secret/private-text leakage in tracked changes.

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/test_pointwise_judge.py tests/test_pointwise_sampling.py tests/test_pointwise_runner.py tests/test_pointwise_review.py tests/test_pointwise_report.py tests/test_pointwise_judge_cli.py -v`

Expected: all pass.

- [ ] **Step 2: Run the full suite**

Run: `uv run pytest -q`

Expected: all tests pass.

- [ ] **Step 3: Check formatting, repository state, and secrets**

Run: `git diff --check`

Run: `git status --short`

Run: `git diff --cached --name-only | rg '^(artifacts/pointwise_judge_bakeoff|data/dev/llm_discovery|.*sealed)'`

Expected: no whitespace errors; only intended code/protocol/test files are tracked; the private-artifact search has no matches.

- [ ] **Step 4: Commit any verification-only repair**

If Task 7 required a code correction, stage only the corrected code/test files and commit with `git commit -m "fix pointwise smoke verification"`. If no correction was needed, do not create an empty commit.

---

### Task 8: Restore inputs and freeze the real ten-case sample

**Files:**
- Create privately: `artifacts/reference_answer_validation/grounding/gemini36_grounded_all_7759.json`
- Create privately: `artifacts/pointwise_judge_bakeoff/smoke-v2/smoke_manifest.json`
- Create privately: `artifacts/pointwise_judge_bakeoff/smoke-v2/sealed_candidates.json`

**Interfaces:**
- Consumes: private `reference-corpus-v1` bucket artifact and shared discovery corpus.
- Produces: source-verified, immutable ten-case manifest.

- [ ] **Step 1: Restore the worktree-local discovery symlink**

Run: `ln -s /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/llm_discovery_data.json data/dev/llm_discovery_data.json`

Expected: the symlink resolves to the recorded SHA-256 `b33c28aa2571cdf5b2867dcc00f0dc55e22198b90022dd23f748a03395301b25`.

- [ ] **Step 2: Restore only the grounded answer file from the private bundle**

Run:

```bash
install -d -m 700 artifacts/reference_answer_validation/grounding
hf buckets cp \
  hf://buckets/Npatta01/trec_mlm_2026/reference-corpus-v1/public/answers/gemini36_grounded_all_7759.json \
  artifacts/reference_answer_validation/grounding/gemini36_grounded_all_7759.json
chmod 600 artifacts/reference_answer_validation/grounding/gemini36_grounded_all_7759.json
```

Do not restore evidence mappings or credential receipts.

- [ ] **Step 3: Verify source counts and hashes**

Run: `sha256sum data/dev/llm_discovery_data.json artifacts/reference_answer_validation/grounding/gemini36_grounded_all_7759.json`

Run: `jq '[.records | to_entries[] | select((.value.answer // "") != "")] | length' artifacts/reference_answer_validation/grounding/gemini36_grounded_all_7759.json`

Expected: discovery hash matches the frozen value; grounded cache contains 7,759 unique successful records and its observed hash is written into the smoke manifest.

- [ ] **Step 4: Build and freeze the ten cases**

Run the Task 2 CLI exactly as documented, then run it a second time and confirm the manifest hash is unchanged.

- [ ] **Step 5: Inspect the blinded selections before inference**

Inspect each private question/reference/candidate triple plus response IDs,
proxy reasons, strata, lengths, and hashes without opening the sealed candidate
map. Confirm ten unique questions and responses and all ten proxy reasons. If
semantic inspection shows a proxy is unusable, change the selector and tests,
create a new prompt/sample version, and preserve the rejected manifest.

---

### Task 9: Preflight and run four local judges

**Files:**
- Create privately: runtime receipts and run caches under `artifacts/pointwise_judge_bakeoff/smoke-v2/`.

**Interfaces:**
- Consumes: frozen manifest and four-model registry.
- Produces: four complete or explicitly failed local run caches.

- [ ] **Step 1: Verify GPU/runtime before any large load**

Run: `rocminfo`

Run: `lemonade --version`

Run: `free -h`

Run: `df -h /home/npatta01/.cache/huggingface`

Expected: `gfx1151`, Lemonade 11.5.0, at least 100 GiB available system memory before load, and at least 50 GiB free disk.

- [ ] **Step 2: Start Lemonade on loopback and run the small backend canary**

Run `lemond --host 127.0.0.1 --port 13305` in a managed PTY, load the already downloaded `Qwen3-0.6B-GGUF` with `lemonade load --llamacpp rocm Qwen3-0.6B-GGUF`, run the preflight canary, record the receipt, and unload it.

- [ ] **Step 3: Pull and run Qwen3.6-35B-A3B GGUF**

Run: `lemonade pull Qwen3.6-35B-A3B-GGUF`

Run: `lemonade load --llamacpp rocm --ctx-size 4096 Qwen3.6-35B-A3B-GGUF`

Run the preflight and ten-case CLI, then `lemonade unload Qwen3.6-35B-A3B-GGUF`. If ROCm fails, preserve the failure and use exactly one predefined Vulkan repair attempt; record it as a distinct runtime representation.

- [ ] **Step 4: Run cached Qwen3.6-27B FP8**

Run the exact cached image and revision offline:

```bash
podman run --rm --name trec-pointwise-qwen36-27b-fp8 --pull=never \
  --network none --ipc=host --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri/card1 --device /dev/dri/renderD128 \
  --group-add keep-groups -p 127.0.0.1:8000:8000 \
  -v /home/npatta01/.cache/huggingface:/root/.cache/huggingface:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  docker.io/vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28 \
  Qwen/Qwen3.6-27B-FP8 \
  --revision e89b16ebf1988b3d6befa7de50abc2d76f26eb09 \
  --host 0.0.0.0 --port 8000 --max-model-len 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 --limit-mm-per-prompt '{"image":0,"video":0}' \
  --enforce-eager
```

The container is network-isolated and the published host socket is literal
loopback; `0.0.0.0` is confined to the container namespace. Run preflight and
the ten-case CLI, then stop the container. Preserve startup/kernel failures
without changing quantization.

- [ ] **Step 5: Run cached gpt-oss-20b**

Run:

```bash
podman run --rm --name trec-pointwise-gpt-oss-20b --pull=never \
  --network none --ipc=host --security-opt seccomp=unconfined \
  --device /dev/kfd --device /dev/dri/card1 --device /dev/dri/renderD128 \
  --group-add keep-groups -p 127.0.0.1:8000:8000 \
  -v /home/npatta01/.cache/huggingface:/root/.cache/huggingface:ro \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  docker.io/vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28 \
  openai/gpt-oss-20b \
  --revision 6cee5e81ee83917806bbde320786a8fb61efebee \
  --host 0.0.0.0 --port 8000 --max-model-len 4096 --max-num-seqs 1 \
  --gpu-memory-utilization 0.85 --enforce-eager
```

Use vLLM's checkpoint chat template, which renders the Harmony format. Run
preflight and ten cases, then stop the container. Reject output if analysis
channel text leaks into the JSON result.

- [ ] **Step 6: Run cached Gemma 4 31B Q4**

Run the cached GGUF directly through the existing llama.cpp ROCm server image:

```bash
podman run --rm --name trec-pointwise-gemma4-31b-q4 --pull=never \
  --network none --ipc=host --device /dev/kfd --device /dev/dri/card1 \
  --device /dev/dri/renderD128 --group-add keep-groups \
  -p 127.0.0.1:8000:8000 \
  -v /home/npatta01/.cache/huggingface/hub/models--google--gemma-4-31B-it-qat-q4_0-gguf/snapshots/59dde24573e7e61570dba08b18a2e1fe246955ed:/models:ro \
  ghcr.io/ggml-org/llama.cpp:server-rocm \
  -m /models/gemma-4-31B_q4_0-it.gguf --host 0.0.0.0 --port 8000 \
  --ctx-size 4096 --parallel 1 --n-gpu-layers 999
```

The container is offline and exposed only on host loopback. Run preflight and
ten cases, then stop it. Do not replace the cached Q4_0 checkpoint with
Lemonade's different Q4_K_M built-in.

- [ ] **Step 7: Validate run coverage and stop every service**

Require exactly ten terminal records per viable judge, immutable raw attempts,
matching input/prompt/schema hashes, and a runtime receipt. Stop vLLM containers,
unload Lemonade models, and terminate the temporary loopback `lemond` PTY.

---

### Task 10: Advisor and Claude blind reviews

**Files:**
- Create privately: review packets and reviewer outputs under `artifacts/pointwise_judge_bakeoff/smoke-v2/review/`.

**Interfaces:**
- Consumes: frozen inputs and anonymous judge results.
- Produces: two complete pass-1 ledgers and two complete pass-2 critiques.

- [ ] **Step 1: Build and validate both review packets**

Run the Task 5 packet CLI. Confirm the pass-1 packet has no judge decisions,
both packets have no local model names or candidate IDs, and the sealed alias
map is mode `0600`.

- [ ] **Step 2: Spawn a fresh Advisor subagent for pass 1**

Give the Advisor only `reviewer-v1.txt` and the pass-1 packet. Require exactly
ten schema-valid independent decisions and no external writes. Validate and
freeze the returned ledger before showing any judge results.

- [ ] **Step 3: Run Claude Code read-only for pass 1**

Invoke Claude non-interactively with Sonnet, `--permission-mode plan`, no tools,
no session persistence, a fixed JSON schema, and `--max-budget-usd 5`. Supply
only `reviewer-v1.txt` and the pass-1 packet. Validate and freeze exactly ten
decisions.

- [ ] **Step 4: Run Advisor pass 2**

Give the same fresh Advisor context the anonymized critique packet plus its own
frozen pass-1 decisions. Require case-level decisive disagreements and prompt
findings without model-identity guesses.

- [ ] **Step 5: Run Claude Code pass 2**

Use the same fixed Claude configuration with the anonymized critique packet and
Claude's frozen pass-1 decisions. Validate ten critiques and preserve raw CLI
output separately from parsed review records.

- [ ] **Step 6: Validate independence and coverage**

Require 10/10 records in every ledger, no judge identity leakage, and explicit
unresolved status wherever Advisor and Claude disagree. Do not synthesize a
third verdict.

- [ ] **Step 7: Check only decisive disputed facts**

For a material Advisor/Claude disagreement or suspected Gemini-reference error,
check the minimum decisive claim against a primary or authoritative source.
Record claim, URL, access date, finding, and whether it changes either review.
Do not ground every case and do not replace either frozen reviewer ledger.

---

### Task 11: Produce the user checkpoint and stop

**Files:**
- Create privately: `artifacts/pointwise_judge_bakeoff/smoke-v2/smoke-review.md`
- Create privately: `artifacts/pointwise_judge_bakeoff/smoke-v2/smoke-summary.json`
- Create sanitized: `reports/pointwise_judge_bakeoff/local-smoke-summary.json`

**Interfaces:**
- Consumes: all validated runs and reviews.
- Produces: case-level private review packet, sanitized aggregate summary, and prompt-change recommendations.

- [ ] **Step 1: Build reports**

Run: `uv run python scripts/build_pointwise_judge_smoke_report.py --artifact-dir artifacts/pointwise_judge_bakeoff/smoke-v2 --sanitized-output reports/pointwise_judge_bakeoff/local-smoke-summary.json`

- [ ] **Step 2: Verify privacy and integrity**

Confirm the private report has ten cases and every judge/reviewer result, the
sanitized output contains no question text, response text, candidate IDs, raw
outputs, evidence URLs, secrets, or sealed aliases, and all recorded hashes
match current inputs.

- [ ] **Step 3: Present the checkpoint in conversation**

Show the prompt, a compact per-case comparison, schema/runtime failures,
Advisor/Claude disagreements, likely prompt defects, and the recommended set of
surviving local candidates. Clearly distinguish verified facts, reviewer
consensus, and unresolved cases.

- [ ] **Step 4: Stop**

Do not implement or run the 100-record bake-off. Wait for the user's prompt,
schema, survivor, and 100-record selection approval.
