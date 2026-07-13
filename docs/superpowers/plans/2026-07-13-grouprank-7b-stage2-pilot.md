# GroupRank 7B Stage 2 Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a qrel-isolated 12-question, 20-candidate GroupRank 7B pilot and compare its validated rankings with the three captured Stage 2 baselines.

**Architecture:** Add a small pure ranking-contract module, a model-specific vLLM adapter, and a pilot orchestration/report module instead of extending the already large Qwen pilot module. Reuse the frozen Stage 2 selection validator and existing metric utilities, persist qrel-free artifacts atomically, and open sealed judgments only in a separate evaluation command.

**Tech Stack:** Python 3.12, vLLM 0.24, PyTorch/ROCm, Hugging Face Transformers, rootless Podman, JSON/JSONL, pytest, existing TREC metric utilities.

**Post-Task-3 structure amendment:** Independent review approved Task 3 but found
that its strict execution, atomic persistence, and semantic reload contract already
make `grouprank_pilot.py` a large focused artifact module. To prevent further
responsibility growth, Task 4 places smoke-receipt behavior in
`grouprank_smoke.py`, and Task 5 places sealed evaluation/reporting in
`grouprank_report.py`. This changes file ownership only; the frozen experiment,
interfaces, gates, and artifact formats are unchanged.

## Global Constraints

- Model: `AQ-MedAI/Diver-GroupRank-7B@42a9504b32616f155970d5b7f349b63dc24b3e27`.
- License: Apache 2.0; 7,615,616,512 BF16 parameters; 15.2 GB Hub download.
- Runtime: `vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`.
- Runtime settings: BF16, `max_model_len=8192`, `gpu_memory_utilization=0.40`, `TRITON_ATTN`.
- Sampling: `temperature=0.3`, `top_p=0.8`, `max_tokens=2048`; request seeds derive from seed 2026 plus query ID, repetition, and group.
- Input: the existing Stage 2 selection at SHA-256 `668515c30a80473aaaa2dc530f50dfd3b7731711e9f9adf70074ebb95aa98b9d`.
- Exactly 12 questions, 20 candidates per question, two repetitions, two groups of 10 per repetition, and 48 requests.
- Grouping seeds are 666 and 667. Every candidate occurs exactly once per repetition.
- Prompt content is limited to question text, candidate ID, cleaned label, and the frozen capability sentence.
- No qrels in Phase A; no few-shot examples, semantic repair, zero substitution, fallback ranking, fusion, 32B download, or automatic expansion.
- Persist every raw generation before parsing. Any malformed or length-truncated request makes the run unscorable and prevents qrel access.
- Rank by descending mean score and candidate ID for exact ties.

---

### Task 1: Implement the Pure GroupRank Contract

**Files:**
- Create: `trec_mlm/grouprank.py`
- Create: `tests/test_grouprank.py`

**Interfaces:**
- Consumes: one frozen Stage 2 selection object and `_capability(candidate_id)` from `trec_mlm.llm_ranker`.
- Produces: `GroupRankRequest`, `build_group_requests(selection)`, `format_grouprank_messages(...)`, `grouprank_score_schema(handles)`, `parse_grouprank_scores(...)`, and `aggregate_grouprank_scores(...)`.

- [ ] **Step 1: Write failing prompt and grouping tests**

Add tests that build a two-question fixture with twenty lexical candidate IDs, then assert four requests per question, group sizes of ten, exact per-repetition coverage, seeds 666/667, stable handles `C00`–`C19`, and byte-identical output across calls. Assert prompt rows use this exact shape:

```text
C03 | Food_and_drink | This LLM is most likely to correctly answer questions about Food and drink.
```

Also assert that question text and expert rows occur inside explicit data delimiters and that no grade, relevance role, qrel field, response, or exemplar appears.

- [ ] **Step 2: Run the tests and verify the missing-module failure**

Run: `python -m pytest -q tests/test_grouprank.py -k 'group or prompt'`

Expected: test collection fails because `trec_mlm.grouprank` does not exist.

- [ ] **Step 3: Implement the request contract**

Create the immutable request record and deterministic grouping:

```python
@dataclass(frozen=True)
class GroupRankRequest:
    query_id: str
    repetition: int
    group: int
    ordered_candidate_ids: tuple[str, ...]
    handle_to_candidate: Mapping[str, str]
    messages: tuple[Mapping[str, str], ...]
    sampling_seed: int

def request_seed(query_id: str, repetition: int, group: int) -> int:
    digest = hashlib.sha256(
        f"2026|{query_id}|{repetition}|{group}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")
```

For each query and grouping seed, sort candidate IDs by
`sha256(f"{seed}|{query_id}|{candidate_id}")`, split at ten, and assign handles from the candidate-ID lexical order before shuffling. Reject every inventory except Stage 2's exact 12×20 shape.

- [ ] **Step 4: Write failing strict-parser and aggregation tests**

Use `object_pairs_hook` fixtures to require exactly the expected ten keys and integer values 0–10. Test missing, extra, duplicate, boolean, float, string, and out-of-range values. Test that two observations per candidate are averaged and ties use candidate ID.

- [ ] **Step 5: Implement strict parsing and aggregation**

Use a duplicate-preserving object hook before constructing a dictionary:

```python
def parse_grouprank_scores(raw: str, *, expected_handles: Sequence[str]) -> dict[str, int]:
    if any(
        raw.count(tag) != 1
        for tag in ("<reason>", "</reason>", "<answer>", "</answer>")
    ):
        raise ValueError("GroupRank output must use the official GroupRank envelope")
    match = re.fullmatch(
        r"<reason>(.*?)</reason>\s*<answer>(.*?)</answer>",
        raw,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("GroupRank output must use the official GroupRank envelope")
    answer = match.group(2).strip()
    if "```" in answer:
        fenced = re.fullmatch(
            r"```json[ \t]*\r?\n(.*?)\r?\n```", answer, flags=re.DOTALL
        )
        if fenced is None:
            raise ValueError("GroupRank output must use the official GroupRank envelope")
        answer = fenced.group(1)
    pairs = json.loads(answer, object_pairs_hook=lambda values: values)
    if not isinstance(pairs, list) or any(not isinstance(pair, tuple) for pair in pairs):
        raise ValueError("GroupRank answer must be a JSON object")
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("GroupRank answer contains duplicate handles")
    if set(keys) != set(expected_handles):
        raise ValueError("GroupRank answer has missing or extra handles")
    result = {}
    for key, value in pairs:
        if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 10:
            raise ValueError("GroupRank scores must be integers from 0 through 10")
        result[key] = value
    return result
```

Require exactly one `<reason>` block followed by exactly one `<answer>` block,
with nothing outside them. Extract bare JSON or one optional `json`-fenced JSON
object from the answer. Reject absent, reordered, or repeated blocks and malformed
fences rather than scanning arbitrary text, truncating, or selecting the last
block. The prompt must show this output skeleton without a semantically scored
expert exemplar.

Return a deterministic fixed-key JSON Schema with every expected handle required,
integer bounds 0–10, and `additionalProperties: false`. This schema documents and
hashes the validation contract but is not passed to guided decoding because the
checkpoint's official response includes reasoning before the final answer.

- [ ] **Step 6: Run the pure tests and commit**

Run: `python -m pytest -q tests/test_grouprank.py`

Expected: all tests pass.

Commit: `git add trec_mlm/grouprank.py tests/test_grouprank.py && git commit -m 'feat: add strict GroupRank contract'`

### Task 2: Add the ROCm vLLM Adapter

**Files:**
- Create: `trec_mlm/grouprank_inference.py`
- Create: `tests/test_grouprank_inference.py`

**Interfaces:**
- Consumes: `Sequence[GroupRankRequest]`.
- Produces: `load_grouprank_engine() -> tuple[Any, Any, dict[str, Any]]` and `run_grouprank_requests(llm, requests, *, stats) -> list[dict[str, Any]]`.

- [ ] **Step 1: Write failing loader and sampling tests**

With module doubles, assert the loader passes the exact model, revision, dtype, context, memory fraction, attention backend, and `trust_remote_code=False`. Assert inference uses `llm.chat` so vLLM applies the checkpoint's chat template, and passes one `SamplingParams` per request with temperature 0.3, top-p 0.8, 2,048 maximum tokens, and its frozen request seed.

- [ ] **Step 2: Verify the adapter tests fail**

Run: `python -m pytest -q tests/test_grouprank_inference.py`

Expected: collection fails because `trec_mlm.grouprank_inference` does not exist.

- [ ] **Step 3: Implement model loading and raw-output capture**

Freeze these constants in the new module:

```python
MODEL_ID = "AQ-MedAI/Diver-GroupRank-7B"
MODEL_REVISION = "42a9504b32616f155970d5b7f349b63dc24b3e27"
MODEL_DTYPE = "bfloat16"
MODEL_LICENSE = "Apache-2.0"
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.40
ATTENTION_BACKEND = "TRITON_ATTN"
```

Return one raw record per request containing query ID, repetition, group, candidate order, handle mapping, raw text, prompt/output token counts, finish reason, and sampling seed. Do not parse inside the adapter. Reject missing token accounting or finish reason, and mark `finish_reason == "length"` for the Phase A failure path.

Runtime metadata must also include the resolved revision, parameter count, license,
vLLM version, Torch version, HIP version, and ROCm device properties.

- [ ] **Step 4: Run inference-adapter regression tests and commit**

Run: `python -m pytest -q tests/test_grouprank_inference.py tests/test_llm_ranker_inference.py`

Expected: all new tests and existing Qwen adapter tests pass.

Commit: `git add trec_mlm/grouprank_inference.py tests/test_grouprank_inference.py && git commit -m 'feat: add GroupRank vLLM adapter'`

### Task 3: Build Qrel-Free Phase A and Atomic Artifacts

**Files:**
- Create: `trec_mlm/grouprank_pilot.py`
- Create: `tests/test_grouprank_pilot.py`

**Interfaces:**
- Consumes: `preflight_pilot_inputs(...)` from `trec_mlm.llm_ranker_pilot`, the frozen Stage 2 selection, `build_group_requests`, and an injected inference callback.
- Produces: `execute_grouprank_phase_a(..., output_dir)`, `load_frozen_grouprank(...)`, and `validate_grouprank_artifact(...)`.

- [ ] **Step 1: Write failing isolation, inventory, and failure-persistence tests**

Use two contradictory qrel fixtures and assert they cannot be accepted by Phase A. Require exactly 48 prompts/raw records, two scores per candidate, and twelve complete rankings. Inject a duplicate-key response and a length finish reason; in both cases require `raw_outputs.jsonl` to exist and be flushed before the parser is invoked, followed by a failure receipt and no ranking or score artifact.

- [ ] **Step 2: Run tests and verify missing orchestration**

Run: `python -m pytest -q tests/test_grouprank_pilot.py -k 'phase_a or failure or qrel'`

Expected: failures identify missing Phase A and artifact functions.

- [ ] **Step 3: Implement Phase A without a qrel parameter**

Define explicit result types and make the execution function own the staging
directory so persistence cannot be skipped:

```python
@dataclass(frozen=True)
class GroupRankPhaseAResult:
    selection: Mapping[str, Any]
    prompts: tuple[Mapping[str, Any], ...]
    raw_outputs: tuple[Mapping[str, Any], ...]
    candidate_scores: Mapping[str, Mapping[str, float]]
    rankings: Mapping[str, Sequence[str]]
    config: Mapping[str, Any]
    runtime: Mapping[str, Any]

@dataclass(frozen=True)
class GroupRankFailureResult:
    selection: Mapping[str, Any]
    prompts: tuple[Mapping[str, Any], ...]
    raw_outputs: tuple[Mapping[str, Any], ...]
    failure: Mapping[str, Any]
    config: Mapping[str, Any]
    runtime: Mapping[str, Any]

class GroupRankParseFailure(ValueError):
    def __init__(self, message: str, *, result: GroupRankFailureResult):
        super().__init__(message)
        self.result = result

def execute_grouprank_phase_a(
    selection: Mapping[str, Any],
    *,
    infer_batch: Callable[[Sequence[GroupRankRequest]], Sequence[Mapping[str, Any]]],
    runtime_metadata: Mapping[str, Any],
    output_dir: str | Path,
) -> GroupRankPhaseAResult:
    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("GroupRank destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    requests = build_group_requests(selection)
    prompts = tuple(request_to_record(request) for request in requests)
    config = build_phase_a_config(selection, prompts, runtime_metadata)
    write_json(staging / "config.json", config)
    write_json(staging / "selection.json", selection)
    write_jsonl(staging / "prompts.jsonl", prompts, fsync=True)
    raw_outputs = tuple(infer_batch(requests))
    write_jsonl(staging / "raw_outputs.jsonl", raw_outputs, fsync=True)
    try:
        candidate_scores, rankings = parse_and_aggregate(requests, raw_outputs)
    except ValueError as error:
        result = build_failure_result(
            selection, prompts, raw_outputs, config, runtime_metadata, error
        )
        finalize_failure_artifact(staging, result)
        os.replace(staging, destination)
        raise GroupRankParseFailure(str(error), result=result) from error
    result = GroupRankPhaseAResult(
        selection=selection,
        prompts=prompts,
        raw_outputs=raw_outputs,
        candidate_scores=candidate_scores,
        rankings=rankings,
        config=config,
        runtime=runtime_metadata,
    )
    finalize_success_artifact(staging, result)
    os.replace(staging, destination)
    return result
```

Implement the shown function body without changing its signature: create a hidden
temporary sibling directory; write config, selection, and prompts; collect every
inference result; write `raw_outputs.jsonl`; flush and `os.fsync` it; and only then
call the parser. Parse all responses, then aggregate only if every request is valid.
Include the selection SHA, model metadata, prompt/schema hashes, code revision,
worktree status, and runtime versions in the config and receipt.
Define the module-private helpers named in the body with typed signatures and keep
all JSON serialization sorted and newline-terminated. On an exception before a
complete raw file exists, retain the staging directory and report its path instead
of deleting evidence.

- [ ] **Step 4: Implement atomic success and failure writers**

After parsing, complete one of these inventories in the same staging directory and
rename the directory atomically. Success inventory:

```text
config.json
selection.json
prompts.jsonl
raw_outputs.jsonl
scores.tsv
rankings.json
runtime.json
ranking_receipt.json
```

Failure inventory omits scores/rankings and adds `failure_receipt.json`. Hash every core file before the rename. Refuse an existing destination. Never mutate the earlier Stage 2 artifacts.

- [ ] **Step 5: Add corruption and deterministic-reload tests**

Mutate each core file, selection hash, model revision, prompt hash, runtime field, and ranking entry independently; require precise validation failures. Reload a valid fixture twice and require identical rankings and hashes.

- [ ] **Step 6: Run Phase A tests and commit**

Run: `python -m pytest -q tests/test_grouprank.py tests/test_grouprank_inference.py tests/test_grouprank_pilot.py`

Expected: all tests pass.

Commit: `git add trec_mlm/grouprank_pilot.py tests/test_grouprank_pilot.py && git commit -m 'feat: persist qrel-free GroupRank pilot'`

### Task 4: Add Smoke, CLI, and Provenance Gates

**Files:**
- Modify: `trec_mlm/cli.py`
- Create: `trec_mlm/grouprank_smoke.py`
- Create: `tests/test_grouprank_smoke.py`

**Interfaces:**
- Produces CLI commands `trec-mlm grouprank-pilot --phase smoke|rank` and `write_grouprank_smoke(...)` / `validate_grouprank_smoke(...)` from `trec_mlm.grouprank_smoke`.

- [ ] **Step 1: Write failing CLI and smoke tests**

Assert `smoke` uses the first frozen group, requires a valid ten-key result, and records exact model/runtime/prompt/raw-output metadata. Assert `rank` rejects a missing, failed, wrong-model, wrong-revision, or corrupted smoke receipt before loading the engine. Assert both phases reject a dirty or mismatched `--code-revision`.

- [ ] **Step 2: Verify the CLI tests fail**

Run: `python -m pytest -q tests/test_grouprank_smoke.py -k 'cli or smoke or provenance'`

Expected: parser does not recognize `grouprank-pilot`.

- [ ] **Step 3: Add the CLI surface and smoke receipt**

Add arguments:

```text
grouprank-pilot --phase smoke|rank
  --selection PATH
  --code-revision SHA
  --output-dir PATH
  [--smoke-receipt PATH]
```

`smoke` and `rank` call `preflight_pilot_inputs` before engine loading. `smoke`
writes and fsyncs its raw response before parsing it. `rank` validates the smoke
first, loads the model once, and delegates all 48 requests plus success/failure
persistence to `execute_grouprank_phase_a`; it returns nonzero after a persisted
failure receipt.

- [ ] **Step 4: Run CLI and full CPU regression tests**

Run: `python -m pytest -q tests/test_grouprank.py tests/test_grouprank_inference.py tests/test_grouprank_pilot.py tests/test_grouprank_smoke.py tests/test_llm_ranker.py tests/test_llm_ranker_inference.py tests/test_llm_ranker_pilot.py`

Expected: all tests pass.

Commit: `git add trec_mlm/cli.py trec_mlm/grouprank_smoke.py tests/test_grouprank_smoke.py && git commit -m 'feat: add guarded GroupRank pilot CLI'`

### Task 5: Add Sealed Evaluation and Result Reporting

**Files:**
- Create: `trec_mlm/grouprank_report.py`
- Create: `tests/test_grouprank_report.py`
- Modify: `trec_mlm/cli.py`

**Interfaces:**
- Consumes: a validated Phase A artifact, `artifacts/llm_label_ranker/pilot/stage2/selection_judgments.json`, and captured baseline data in `artifacts/llm_label_ranker/pilot/stage2/report.json`.
- Produces: `build_grouprank_report(...) -> dict[str, Any]` and deterministic JSON/Markdown writers from `trec_mlm.grouprank_report`; extends `grouprank-pilot --phase evaluate` without importing qrels in the smoke/rank branches.

- [ ] **Step 1: Write failing metric, gate, and report tests**

Build fixtures whose relevant ranks have known nDCG@10, MRR, and top-three counts. Require per-query top three, best relevant rank, repetition score differences, tie counts, and comparisons with contextual E5, contextual Qwen4B, and pointwise Qwen. Require the gate to pass only for 48 valid outputs, top-three count at least 8/12, and nDCG@10 at least 0.7232.

- [ ] **Step 2: Verify reporting tests fail**

Run: `python -m pytest -q tests/test_grouprank_report.py -k 'report or metric or gate'`

Expected: report functions are missing.

- [ ] **Step 3: Implement post-hoc evaluation**

Call `load_frozen_grouprank` before opening judgments. Validate that judgment query/candidate inventories exactly match Phase A, then use the existing `evaluate_rankings` utility. Read baseline values from the frozen Stage 2 report and reject any mismatch with 8/12, 0.7232, 0.6361 for E5; 8/12, 0.6648, 0.5585 for Qwen4B; and 5/12, 0.4820, 0.3209 for pointwise.

Extend the same CLI parser with:

```text
grouprank-pilot --phase evaluate
  --selection PATH
  --code-revision SHA
  --output-dir PHASE_A_DIR
  --judgments PATH
  --report-json PATH
  --report-md PATH
```

- [ ] **Step 4: Implement deterministic result writers and commit**

Write sorted JSON and a concise Markdown report to `docs/results/2026-07-13-grouprank-7b-stage2-pilot.md`. Label the result qrel-assisted and exploratory and state that no expansion was started.

Run: `python -m pytest -q tests/test_grouprank_report.py && python -m pytest -q`

Expected: GroupRank tests and the full repository suite pass.

Commit: `git add trec_mlm/cli.py trec_mlm/grouprank_report.py tests/test_grouprank_report.py && git commit -m 'feat: evaluate GroupRank pilot'`

### Task 6: Run the Guarded ROCm Pilot and Capture Results

**Files:**
- Create locally (gitignored): `artifacts/llm_label_ranker/pilot/stage2/grouprank_7b/smoke.json`
- Create locally (gitignored): `artifacts/llm_label_ranker/pilot/stage2/grouprank_7b/rank/`
- Create: `docs/results/2026-07-13-grouprank-7b-stage2-pilot.md`

**Interfaces:**
- Consumes: committed clean implementation, pinned Podman image, Hugging Face cache, and frozen Stage 2 files.
- Produces: one validated qrel-free ranking artifact or explicit failure artifact, followed by a report only for valid rankings.

- [ ] **Step 1: Verify the host is safe before the download**

Run:

```bash
ls -l /dev/kfd /dev/dri/renderD128
free -h
cat /proc/pressure/memory
cat /proc/pressure/io
podman ps --format '{{.ID}} {{.Names}} {{.Status}} {{.Image}}'
```

Expected: both devices are accessible, at least 35 GiB memory is available, memory/IO PSI `avg10` is below 1.00, and no other ROCm inference container is active. Otherwise stop and report the preflight rather than downloading or running.

- [ ] **Step 2: Run the single protocol-correction smoke retry**

The original strict smoke failed because its raw generation contained two
`<answer>` blocks. Preserve
`artifacts/llm_label_ranker/pilot/stage2/grouprank_7b/smoke.json` byte-for-byte.
After the official `<reason>` then `<answer>` protocol correction is committed,
launch the pinned image rootlessly for one explicit operator retry using the
distinct `smoke_protocol_v2.json` receipt below. This is not runtime retry
behavior. The command preserves the validated devices, supplementary groups,
shared IPC, cache, repository, and read-only git-common-dir mounts.

```bash
IMAGE='docker.io/vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28'
REPO="$(pwd -P)"
GIT_COMMON="$(cd "$(git rev-parse --git-common-dir)" && pwd -P)"
NAME="trec-grouprank7b-smoke-protocol-v2-$(git rev-parse --short HEAD)"
podman run --name "$NAME" \
  --entrypoint /bin/bash \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --ipc=host \
  -e HF_HUB_OFFLINE=0 -e TRANSFORMERS_OFFLINE=0 \
  -v /home/npatta01/.cache/huggingface:/root/.cache/huggingface \
  -v "$REPO:/workspace" -v "$GIT_COMMON:$GIT_COMMON:ro" \
  -w /workspace "$IMAGE" -lc \
  'python -m trec_mlm.cli grouprank-pilot \
    --phase smoke \
    --selection artifacts/llm_label_ranker/pilot/stage2/selection.json \
    --code-revision "$(git rev-parse HEAD)" \
    --output-dir artifacts/llm_label_ranker/pilot/stage2/grouprank_7b/smoke_protocol_v2.json'
```

Expected: the 15.2 GB checkpoint resolves to the frozen revision, ROCm loads it in BF16, one ten-candidate response validates, and the smoke receipt reports `passed: true`.

- [ ] **Step 3: Recheck resources and run Phase A offline**

Repeat Step 1, then run the cached checkpoint offline:

```bash
IMAGE='docker.io/vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28'
REPO="$(pwd -P)"
GIT_COMMON="$(cd "$(git rev-parse --git-common-dir)" && pwd -P)"
NAME="trec-grouprank7b-rank-$(git rev-parse --short HEAD)"
podman run --name "$NAME" \
  --entrypoint /bin/bash \
  --device /dev/kfd --device /dev/dri \
  --group-add keep-groups --ipc=host \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v /home/npatta01/.cache/huggingface:/root/.cache/huggingface \
  -v "$REPO:/workspace" -v "$GIT_COMMON:$GIT_COMMON:ro" \
  -w /workspace "$IMAGE" -lc \
  'python -m trec_mlm.cli grouprank-pilot \
    --phase rank \
    --selection artifacts/llm_label_ranker/pilot/stage2/selection.json \
    --smoke-receipt artifacts/llm_label_ranker/pilot/stage2/grouprank_7b/smoke_protocol_v2.json \
    --code-revision "$(git rev-parse HEAD)" \
    --output-dir artifacts/llm_label_ranker/pilot/stage2/grouprank_7b/rank'
```

Expected: exactly 48 requests and twelve validated rankings. On any invalid output, retain the failure receipt, do not evaluate, and report the exact query/repetition/group and raw generation.

- [ ] **Step 4: Evaluate only after Phase A validation**

Run on the host:

```bash
python -m trec_mlm.cli grouprank-pilot \
  --phase evaluate \
  --selection artifacts/llm_label_ranker/pilot/stage2/selection.json \
  --judgments artifacts/llm_label_ranker/pilot/stage2/selection_judgments.json \
  --output-dir artifacts/llm_label_ranker/pilot/stage2/grouprank_7b/rank \
  --report-json artifacts/llm_label_ranker/pilot/stage2/grouprank_7b/report.json \
  --report-md docs/results/2026-07-13-grouprank-7b-stage2-pilot.md \
  --code-revision "$(git rev-parse HEAD)"
```

Expected: metrics, per-query evidence, baseline comparisons, and an explicit pass/fail decision.

- [ ] **Step 5: Verify, commit the durable report, and stop**

Run:

```bash
python -m pytest -q
git diff --check
git status --short
```

Expected: all tests pass; only the intended Markdown result is tracked while raw artifacts remain local and gitignored.

Commit: `git add docs/results/2026-07-13-grouprank-7b-stage2-pilot.md && git commit -m 'docs: report GroupRank 7B pilot'`

Stop after reporting. Do not start a full sweep, fusion, or another model.
