# Pointwise Judge v2 Hosted Canary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a strict v2 pointwise-judge canary over the same 10 frozen cases using DeepSeek V4 Flash, GPT-5.4 nano, and hosted Qwen3.6-35B-A3B in direct and thinking modes.

**Architecture:** Keep the frozen v1 protocol readable and reusable while adding an explicitly selected v2 protocol whose Pydantic model is the semantic source of truth. Refactor hosted model selection around a configuration identity so two runs may share a provider/model endpoint without sharing artifacts. Enforce both spend and physical-attempt ceilings durably, then compare all four v2 runs to the locked human-reviewed labels and to v1 evidence.

**Tech Stack:** Python 3.11+, Pydantic 2, pytest, urllib/OpenRouter Chat Completions, JSON/Markdown artifacts.

## Global Constraints

- The same 10 frozen cases must be used for every configuration.
- The four configurations are DeepSeek V4 Flash direct on DeepInfra, GPT-5.4 nano non-thinking on OpenAI, Qwen3.6-35B-A3B direct on AkashML, and that same Qwen endpoint with thinking enabled.
- Initial nominal work is exactly 40 calls; every retry and repair is a physical paid call and the durable hard ceiling is 100 without new authorization.
- The dollar cap is no higher than USD 0.10.
- Pydantic v2 is authoritative; provider-compatible schema omissions are explicit, tested, and receipt-bound.
- All API artifacts are private (`0600` files, `0700` directories); no secret is serialized.
- Stop after the 10-case canary. Do not start a 50- or 500-case run.
- Preserve all v1 protocol behavior and existing v1 artifacts.

## File Map

- `pyproject.toml`, `uv.lock`: declare and lock Pydantic 2.
- `trec_mlm/pointwise_judge.py`: versioned contract loading, Pydantic v2 decision model, and invariant validation.
- `protocols/pointwise_judge_bakeoff/pointwise-v2.txt`: exact approved v2 judging prompt.
- `protocols/pointwise_judge_bakeoff/decision-v2.schema.json`: normalized Pydantic-generated schema snapshot.
- `trec_mlm/pointwise_runner.py`: protocol-aware frozen-contract hash validation while retaining v1 constants.
- `scripts/run_hosted_pointwise_smoke.py`: configuration identity, Qwen modes, durable call ceiling, receipts, and private paths.
- `scripts/build_pointwise_judge_v2_report.py`: validate and summarize the four runs against locked review evidence.
- `tests/test_pointwise_judge.py`, `tests/test_pointwise_judge_cli.py`: v2 model/schema/protocol behavior and v1 compatibility.
- `tests/test_run_hosted_pointwise_smoke.py`: four configurations, routing, reasoning, path separation, attempt ledger.
- `tests/test_build_pointwise_judge_v2_report.py`: comparison metrics, hard gates, and rationale quote audit.

---

### Task 1: Pydantic v2 Decision Contract

**Files:**
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `trec_mlm/pointwise_judge.py`
- Create: `protocols/pointwise_judge_bakeoff/pointwise-v2.txt`
- Create: `protocols/pointwise_judge_bakeoff/decision-v2.schema.json`
- Modify: `tests/test_pointwise_judge.py`

**Interfaces:**
- Consumes: the exact prompt in `artifacts/pointwise_judge_bakeoff/hosted-smoke-review/reviews/opus-prompt-advice.md` section 4 and the invariants in the design spec.
- Produces: `DecisionV2`, `load_prompt_contract(protocol_dir, version="v1")`, `parse_judge_decision(raw, version="v1")`, and a normalized v2 schema matching `DecisionV2.model_json_schema()`.

- [ ] **Step 1: Add failing version-selection and strict-validation tests**

```python
def test_v1_remains_default_and_v2_is_explicit() -> None:
    assert load_prompt_contract(PROTOCOL_DIR).prompt_path.name == "pointwise-v1.txt"
    assert load_prompt_contract(PROTOCOL_DIR, version="v2").schema_path.name == "decision-v2.schema.json"

def test_v2_rejects_coercion_and_extra_keys() -> None:
    raw = dict(VALID_V2_DECISION, correctness="4")
    with pytest.raises(ValueError):
        parse_judge_decision(json.dumps(raw), version="v2")
```

- [ ] **Step 2: Add failing model/schema drift and invariant matrix tests**

```python
def test_v2_schema_snapshot_matches_model() -> None:
    committed = load_prompt_contract(PROTOCOL_DIR, version="v2").schema
    assert committed == normalized_decision_v2_schema()

@pytest.mark.parametrize("patch", [
    {"label": "good", "completeness": 1},
    {"label": "incorrect", "error_flags": []},
    {"label": "uncertain", "correctness": 1},
    {"error_flags": ["reference_conflict"], "reference_status": "sound"},
])
def test_v2_cross_field_invariants_reject(patch: dict[str, object]) -> None:
    value = VALID_V2_DECISION | patch
    with pytest.raises(ValueError):
        parse_judge_decision(json.dumps(value), version="v2")
```

- [ ] **Step 3: Run the focused tests and verify RED**

Run: `uv run pytest tests/test_pointwise_judge.py -q`

Expected: failures for missing v2 interfaces/files and absent Pydantic dependency.

- [ ] **Step 4: Declare Pydantic 2 and implement the strict model**

Use `ConfigDict(extra="forbid", strict=True)`, `Literal` values, strict constrained integers, `conlist`/validators for unique `error_flags` with `max_length=3`, `constr(min_length=60, max_length=600)`, and an `after` model validator implementing all nine design invariants. Keep `JudgeDecision` as the runner-facing value type; convert `DecisionV2` to it after validation so v1 callers remain compatible.

- [ ] **Step 5: Add the exact approved v2 prompt and generated normalized snapshot**

Copy the prompt verbatim from the Opus advisory section 4. Generate the schema from `DecisionV2.model_json_schema()`, recursively remove unstable presentation-only keys only through a named normalization function, serialize with sorted keys and a trailing newline, and assert regeneration equality in the test.

- [ ] **Step 6: Run focused and adjacent tests**

Run: `uv run pytest tests/test_pointwise_judge.py tests/test_pointwise_judge_cli.py -q`

Expected: all pass, including unchanged v1 expectations.

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock trec_mlm/pointwise_judge.py protocols/pointwise_judge_bakeoff/pointwise-v2.txt protocols/pointwise_judge_bakeoff/decision-v2.schema.json tests/test_pointwise_judge.py
git commit -m "add strict pointwise judge v2 contract"
```

### Task 2: Protocol-Aware Runner Validation

**Files:**
- Modify: `trec_mlm/pointwise_runner.py`
- Modify: `tests/test_pointwise_judge_cli.py`

**Interfaces:**
- Consumes: `PromptContract` from Task 1.
- Produces: `validate_judge_contract(contract, *, version: str) -> str`; existing `validate_frozen_judge_contract(contract)` remains a v1-compatible wrapper.

- [ ] **Step 1: Add failing tests for v1 and v2 hash validation**

```python
def test_validate_judge_contract_accepts_both_committed_versions() -> None:
    assert validate_judge_contract(load_prompt_contract(PROTOCOL_DIR), version="v1")
    assert validate_judge_contract(load_prompt_contract(PROTOCOL_DIR, version="v2"), version="v2")

def test_validate_judge_contract_rejects_cross_version_contract() -> None:
    with pytest.raises(ValueError, match="v2"):
        validate_judge_contract(load_prompt_contract(PROTOCOL_DIR), version="v2")
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `uv run pytest tests/test_pointwise_judge_cli.py -q`

- [ ] **Step 3: Implement versioned prompt/schema hash registries**

Retain `FROZEN_DECISION_SCHEMA_SHA256` as the v1 alias. Add immutable v2 prompt/schema hashes and validate loaded filenames, bytes, and normalized schema against the requested version. Do not infer v2 from directory contents.

- [ ] **Step 4: Run the complete pointwise runner suite**

Run: `uv run pytest tests/test_pointwise_judge_cli.py tests/test_pointwise_judge.py -q`

- [ ] **Step 5: Commit**

```bash
git add trec_mlm/pointwise_runner.py tests/test_pointwise_judge_cli.py
git commit -m "support versioned judge contracts"
```

### Task 3: Four Hosted Configurations and Durable Attempt Limit

**Files:**
- Modify: `scripts/run_hosted_pointwise_smoke.py`
- Modify: `tests/test_run_hosted_pointwise_smoke.py`

**Interfaces:**
- Consumes: `validate_judge_contract(..., version="v2")`, `load_prompt_contract(..., version="v2")`, and the frozen 10-case manifest.
- Produces: `ModelSpec(config_id, model_id, provider, ..., reasoning)`; CLI `--config`; `AttemptBudget(cap_calls=100, used_calls=...)`; separate runtime/run/diagnostic files keyed by `config_id`.

- [ ] **Step 1: Add failing registry and payload tests**

```python
def test_v2_registry_has_four_distinct_configurations() -> None:
    specs = hosted.V2_CONFIG_SPECS
    assert set(specs) == {"deepseek-v4-flash-direct", "gpt-5.4-nano-direct", "qwen3.6-35b-a3b-direct", "qwen3.6-35b-a3b-thinking"}
    assert specs["qwen3.6-35b-a3b-direct"].model_id == specs["qwen3.6-35b-a3b-thinking"].model_id
    assert specs["qwen3.6-35b-a3b-direct"].provider == specs["qwen3.6-35b-a3b-thinking"].provider == "AkashML"

def test_qwen_pair_differs_only_by_reasoning() -> None:
    direct = hosted._build_payload(hosted.V2_CONFIG_SPECS["qwen3.6-35b-a3b-direct"], MESSAGES, V2_SCHEMA)
    thinking = hosted._build_payload(hosted.V2_CONFIG_SPECS["qwen3.6-35b-a3b-thinking"], MESSAGES, V2_SCHEMA)
    assert direct["reasoning"] == {"enabled": False}
    assert thinking["reasoning"] == {"enabled": True}
    assert {k: v for k, v in direct.items() if k != "reasoning"} == {k: v for k, v in thinking.items() if k != "reasoning"}
```

- [ ] **Step 2: Add failing durable attempt-budget tests**

```python
def test_attempt_budget_reserves_before_each_physical_request() -> None:
    budget = hosted.AttemptBudget(cap_calls=2)
    budget.reserve(); budget.reserve()
    with pytest.raises(RuntimeError, match="call ceiling"):
        budget.reserve()

def test_attempt_ledger_restores_interrupted_count_privately(tmp_path: Path) -> None:
    path = tmp_path / "attempt-ledger.json"
    hosted._write_attempt_ledger(path, hosted.AttemptBudget(100, 17))
    assert hosted._load_attempt_ledger(path, cap_calls=100).used_calls == 17
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
```

- [ ] **Step 3: Run focused tests and verify RED**

Run: `uv run pytest tests/test_run_hosted_pointwise_smoke.py -q`

- [ ] **Step 4: Implement configuration identity and exact routes**

Keep the v1 registry usable. Add v2 configurations with current pinned prices: DeepInfra DeepSeek `0.09/0.18`, OpenAI GPT nano `0.20/1.25`, and AkashML Qwen `0.14/1.00` USD per million input/output tokens. Key every v2 artifact and `model_alias` by `config_id`; bind model, provider, reasoning object, prompt hash, schema hash, and transport hash in receipts.

- [ ] **Step 5: Implement the durable physical-attempt ceiling**

Add CLI `--call-cap` with maximum 100 and a private atomic `attempt-ledger.json`. Reserve and persist one attempt immediately before every network request, including retry and schema repair calls. Never decrement after failure. On resume, combine the ledger with validated terminal records without recounting or resetting attempts.

- [ ] **Step 6: Make v2 transport adaptation minimal and receipt-bound**

Allow only explicit removal of provider-incompatible schema keywords. Pydantic validation remains after response parsing. The receipt must list each omitted JSON path and the transport hash; fail if the source v2 schema hash differs.

- [ ] **Step 7: Run hosted-runner tests and lint**

Run: `uv run pytest tests/test_run_hosted_pointwise_smoke.py tests/test_pointwise_judge.py tests/test_pointwise_judge_cli.py -q`

Run: `uv run ruff check scripts/run_hosted_pointwise_smoke.py trec_mlm/pointwise_judge.py trec_mlm/pointwise_runner.py tests/test_run_hosted_pointwise_smoke.py`

- [ ] **Step 8: Commit**

```bash
git add scripts/run_hosted_pointwise_smoke.py tests/test_run_hosted_pointwise_smoke.py
git commit -m "add bounded four-way hosted judge canary"
```

### Task 4: Comparison and Claim-Safety Report

**Files:**
- Create: `scripts/build_pointwise_judge_v2_report.py`
- Create: `tests/test_build_pointwise_judge_v2_report.py`

**Interfaces:**
- Consumes: four private v2 run files, their receipts, spend/attempt ledgers, frozen manifest, locked `advisor.json` and `claude.json`, and v1 run evidence.
- Produces: a private JSON comparison and sanitized Markdown report with per-config metrics, paired Qwen direct/thinking differences, hard-gate failures, and permitted claims.

- [ ] **Step 1: Add failing metric and gate tests**

```python
def test_metrics_report_exact_agreement_jaccard_cost_latency() -> None:
    result = compare_runs(FIXTURE_INPUT)
    assert result["configs"]["deepseek-v4-flash-direct"]["label_agreement"] == 0.9
    assert result["configs"]["deepseek-v4-flash-direct"]["error_flag_mean_jaccard"] == pytest.approx(0.8)
    assert "cost_usd" in result["configs"]["deepseek-v4-flash-direct"]

def test_claim_is_exploratory_for_ten_cases() -> None:
    assert permitted_claim(10).startswith("On this 10-case canary")
    assert "generally" not in permitted_claim(10)
```

- [ ] **Step 2: Add failing rationale quote-fidelity tests**

For each non-empty flag set, identify quoted spans in the rationale and require at least one short span to occur verbatim in the candidate. Report a failure instead of silently dropping malformed or absent quotes.

- [ ] **Step 3: Run focused tests and verify RED**

Run: `uv run pytest tests/test_build_pointwise_judge_v2_report.py -q`

- [ ] **Step 4: Implement deterministic comparison and sanitized report rendering**

Report 10/10 completion, parse/schema/invariant validity, provider identity, repair/retry counts, label/correctness/completeness/relevance/reference-status exact agreement, flag Jaccard, cross-model agreement, Qwen paired changes, cost, latency, rationale audit, and v1 differences. The claim section must explicitly say the sample is diagnostic and too small for a population-quality claim.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run pytest tests/test_build_pointwise_judge_v2_report.py -q`

```bash
git add scripts/build_pointwise_judge_v2_report.py tests/test_build_pointwise_judge_v2_report.py
git commit -m "report pointwise judge v2 canary"
```

### Task 5: Preflight, Paid 40-Call Canary, and Verification

**Files:**
- Create privately under: `artifacts/pointwise_judge_bakeoff/hosted-smoke-v2/`
- Create: `reports/pointwise-judge-v2-hosted-canary-2026-07-27.md`
- Modify: `reports/pointwise-judge-smoke-v2.json` only if it is confirmed to be the intended tracked summary; otherwise leave this pre-existing untracked file untouched.

**Interfaces:**
- Consumes: Tasks 1–4 and `OPENROUTER_API_KEY` loaded from the private ignored `.env` without printing it.
- Produces: exactly four complete 10-case runs, private ledgers/comparison, and one sanitized report.

- [ ] **Step 1: Run all local checks before network access**

Run: `uv run pytest tests/test_pointwise_judge.py tests/test_pointwise_judge_cli.py tests/test_run_hosted_pointwise_smoke.py tests/test_build_pointwise_judge_v2_report.py -q`

Run: `uv run ruff check trec_mlm/pointwise_judge.py trec_mlm/pointwise_runner.py scripts/run_hosted_pointwise_smoke.py scripts/build_pointwise_judge_v2_report.py tests/test_pointwise_judge.py tests/test_pointwise_judge_cli.py tests/test_run_hosted_pointwise_smoke.py tests/test_build_pointwise_judge_v2_report.py`

- [ ] **Step 2: Perform a read-only authenticated route preflight**

Load `.env` without echo, verify the account-visible catalog contains all three model IDs and each pinned provider supports every required request parameter, then store only sanitized provider/model/price metadata. Abort before chat calls on any mismatch.

- [ ] **Step 3: Run one included canary case per configuration**

Invoke the runner with protocol `v2`, all four config IDs, dollar cap `0.10`, call cap `100`, and the exact manifest hash. The runner's built-in canary executes the first case for each configuration before its remaining nine; these four calls are part of the 40, not extra probes. Stop on any route, schema, Pydantic, invariant, permission, or accounting failure.

- [ ] **Step 4: Complete the remaining nine cases per configuration**

Resume the same output directory and ledgers. Expected nominal total after success: 40 physical attempts. Any recorded retries/repairs increase the ledger count but may never exceed 100.

- [ ] **Step 5: Build and manually audit the comparison**

Run the report builder. Inspect every non-good rationale for exact quote fidelity and factual faithfulness to the shown candidate/reference; record manual rulings in the private comparison rather than editing model outputs.

- [ ] **Step 6: Verify artifact privacy, identities, hashes, counts, and costs**

Require four 10-record runs, matching manifest/prompt/schema/transport hashes, pinned providers, distinct Qwen config paths, attempt ledger at least 40 and no more than 100, spend at most USD 0.10, no serialized key, `0600` files, and `0700` private directories.

- [ ] **Step 7: Run the full relevant suite from a fresh command**

Run: `uv run pytest tests/test_pointwise_judge.py tests/test_pointwise_judge_cli.py tests/test_run_hosted_pointwise_smoke.py tests/test_build_pointwise_judge_v2_report.py -q`

Run: `uv run ruff format --check trec_mlm/pointwise_judge.py trec_mlm/pointwise_runner.py scripts/run_hosted_pointwise_smoke.py scripts/build_pointwise_judge_v2_report.py tests/test_pointwise_judge.py tests/test_pointwise_judge_cli.py tests/test_run_hosted_pointwise_smoke.py tests/test_build_pointwise_judge_v2_report.py`

- [ ] **Step 8: Commit only sanitized code and report**

```bash
git add reports/pointwise-judge-v2-hosted-canary-2026-07-27.md
git commit -m "record four-way pointwise judge v2 canary"
```

Do not add the private artifact directory or any pre-existing unrelated untracked file.

