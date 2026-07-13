# LLM Expert-Label Ranker Feasibility Pilots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a six-question, twelve-label diagnostic comparing pointwise and listwise Qwen3-4B ranking with contextual Qwen4B and E5 embeddings, then run a larger diagnostic only if the frozen feasibility rule passes.

**Architecture:** Add a pilot-specific selection and ranking module without weakening the existing canonical 150-label code. Freeze a qrel-assisted selection manifest first, run qrel-free inference from that manifest, and reveal grades only in a separate reporting phase. Reuse the validated vLLM engine and existing embedding artifacts while hardening receipts as required by the independent advisor.

**Tech Stack:** Python 3.12, NumPy, vLLM 0.24, PyTorch/ROCm, JSON Schema structured output, pytest, TREC-style metrics.

## Global Constraints

- Model: `Qwen/Qwen3-4B-Instruct-2507@cdbee75f17c01a7cc42f958dc650907174af0554`.
- Runtime: `vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`.
- FP32, official chat template exactly once, greedy decoding, seed 2026.
- Stage 1 has exactly six questions and twelve unique candidate labels per question.
- Pointwise runs exactly 72 pairs. Listwise runs exactly 18 calls and ranks all twelve handles.
- Qrels may select the pilot inventory but may not appear in prompts or Phase A inference inputs.
- Baselines are contextual Qwen Embedding 4B and contextual E5-Mistral from their existing run-1 artifacts.
- No repeat or full sweep is authorized. Stage 2 is conditional on the Stage 1 feasibility rule.

---

### Task 1: Freeze the Stage 1 Selection Manifest

**Files:**
- Create: `trec_mlm/llm_ranker_pilot.py`
- Create: `tests/test_llm_ranker_pilot.py`
- Create: `artifacts/llm_label_ranker/pilot/stage1/selection.json`
- Create: `artifacts/llm_label_ranker/pilot/stage1/selection_judgments.json`

**Interfaces:**
- Consumes: queries, complete qrels, candidate manifest, and contextual Qwen4B `run.trec`.
- Produces: `build_stage1_selection(*, query_path, qrels_path, candidate_path, qwen_run_path) -> tuple[dict[str, Any], dict[str, Any]]` and `validate_pilot_selection(payload, *, expected_stage, expected_candidate_count) -> dict[str, Any]`.

- [ ] **Step 1: Write failing deterministic-selection tests**

Freeze these query IDs in the test:

```python
STAGE1_QUERY_IDS = (
    "56fc3b342603e7140040a037",  # issue #11, linguistics
    "57265eeaf1498d1400e8dd76",  # issue #11, landscape
    "570a34484103511400d595c8",  # embedding success, food/drink
    "57298ef11d04691400779530",  # embedding success, mathematics
    "5726eb785951b619008f8276",  # embedding failure, bacterial immunity
    "5729068e1d04691400778f8f",  # embedding failure, Parc de la Villette
)
```

Assert that every entry contains one or two highest-grade labels, four qrel-zero Qwen4B
false-positive confounders, deterministic SHA256 negatives, and exactly twelve unique
labels. `selection.json` contains question text, candidate IDs, and non-qrel source hashes
only. `selection_judgments.json` contains roles, grades, qrel hash, and Qwen run hash;
neither roles nor grades may occur in the Phase A manifest.

- [ ] **Step 2: Run the selection tests and verify the missing module failure**

Run: `python -m pytest -q tests/test_llm_ranker_pilot.py -k selection`

Expected: collection fails because `trec_mlm.llm_ranker_pilot` does not exist.

- [ ] **Step 3: Implement the exact selection algorithm**

Implement:

```python
def build_stage1_selection(*, query_path, qrels_path, candidate_path, qwen_run_path):
    queries = load_queries_tsv(query_path)
    qrels = load_complete_qrels(qrels_path, expected_query_ids=queries)
    candidate_ids = load_candidate_manifest(candidate_path)
    qwen_rankings = rankings_from_run(qwen_run_path)
    entries = []
    judgment_entries = []
    for query_id in STAGE1_QUERY_IDS:
        grades = qrels[query_id]
        relevant = [
            candidate_id
            for candidate_id, grade in sorted(
                grades.items(), key=lambda item: (-item[1], item[0])
            )
            if grade > 0
        ][:2]
        confounders = [
            candidate_id
            for candidate_id in qwen_rankings[query_id]
            if grades[candidate_id] == 0 and candidate_id not in relevant
        ][:4]
        excluded = set(relevant) | set(confounders)
        negative_pool = [
            candidate_id
            for candidate_id in candidate_ids
            if grades[candidate_id] == 0 and candidate_id not in excluded
        ]
        negatives = sorted(
            negative_pool,
            key=lambda candidate_id: hashlib.sha256(
                f"2026|{query_id}|{candidate_id}".encode()
            ).digest(),
        )[: 12 - len(excluded)]
        entries.append(
            {
                "query_id": query_id,
                "question": queries[query_id],
                "candidate_ids": sorted([*relevant, *confounders, *negatives]),
            }
        )
        judgment_entries.append(
            {
                "query_id": query_id,
                "grades": {candidate_id: grades[candidate_id] for candidate_id in entries[-1]["candidate_ids"]},
                "roles": {
                    "relevant": relevant,
                    "confounder": confounders,
                    "negative": negatives,
                },
            }
        )
    return (
        {"stage": 1, "seed": 2026, "questions": entries},
        {"stage": 1, "questions": judgment_entries},
    )
```

Reject missing fixed queries, wrong candidate universes, fewer than one relevant label,
role overlap, non-12 candidate sets, and changed input hashes.

- [ ] **Step 4: Generate and inspect the Stage 1 manifest**

Run a small module entry point that writes sorted JSON to
`artifacts/llm_label_ranker/pilot/stage1/selection.json` and
`selection_judgments.json`. Confirm six questions and 72 total question-candidate pairs
from the qrel-free manifest, then confirm the sealed manifest has the expected relevant
labels for the two issue examples.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest -q tests/test_llm_ranker_pilot.py -k selection`

Expected: all selection tests pass.

Commit: `git commit -am 'feat: freeze LLM ranker pilot selection'` after explicitly adding
the new module, test, and compact selection manifest.

### Task 2: Add Variable-Size Pilot Pointwise and Listwise Ranking

**Files:**
- Modify: `trec_mlm/llm_ranker_pilot.py`
- Modify: `trec_mlm/llm_ranker_inference.py`
- Modify: `tests/test_llm_ranker_pilot.py`
- Modify: `tests/test_llm_ranker_inference.py`

**Interfaces:**
- Produces: `pilot_handles()`, `balanced_pilot_orders()`,
  `format_pilot_listwise_messages()`, `parse_complete_pilot_ranking()`, and
  `run_pilot_listwise_messages()`.

- [ ] **Step 1: Write failing pilot-ranking tests**

For twelve and twenty candidates, require stable lexical handles, three unique balanced
orders, every candidate appearing in the first/middle/last presentation region across
the rotations, exact complete JSON rankings, and rejection of duplicates, omissions, or
unknown handles. Freeze the pointwise prompt bytes to the existing canonical prompt.

- [ ] **Step 2: Verify tests fail on absent variable-size functions**

Run: `python -m pytest -q tests/test_llm_ranker_pilot.py -k 'handles or listwise or pointwise'`

Expected: failures identify the missing pilot functions.

- [ ] **Step 3: Implement pilot-only ranking functions**

Use lexical handles `C00` through `C11` or `C19`. Partition the deterministic SHA256 base
order into three near-equal groups and cycle the groups through three rotations. The
listwise user prompt must request every handle exactly once and must contain only the
question, candidate IDs, and frozen capability sentences.

Implement strict parsing:

```python
def parse_complete_pilot_ranking(raw_output, *, valid_handles):
    payload = json.loads(raw_output)
    expected = set(valid_handles)
    if not isinstance(payload, list) or len(payload) != len(expected):
        raise ValueError("pilot ranking must contain every handle exactly once")
    if len(set(payload)) != len(payload) or set(payload) != expected:
        raise ValueError("pilot ranking has duplicates, omissions, or unknown handles")
    return payload
```

Do not use the prior lexical fallback. A malformed pilot call is recorded as a failed
question/rotation and cannot satisfy the feasibility rule.

- [ ] **Step 4: Generalize structured decoding without changing canonical defaults**

Change `listwise_json_schema(handles, *, selection_count=20)` so canonical calls retain
20 items while pilot calls request `len(handles)`. Add a `selection_count` keyword to
`run_listwise_messages()` and test both schemas with tokenizer/vLLM doubles.

- [ ] **Step 5: Run ranking tests and commit**

Run: `python -m pytest -q tests/test_llm_ranker.py tests/test_llm_ranker_inference.py tests/test_llm_ranker_pilot.py`

Expected: all canonical and pilot tests pass.

Commit: `git commit -am 'feat: support compact LLM ranker pilots'`.

### Task 3: Implement Qrel-Isolated Pilot Artifacts and Harden Receipts

**Files:**
- Modify: `trec_mlm/llm_ranker_pilot.py`
- Modify: `trec_mlm/artifact_manifest.py`
- Modify: `tests/test_llm_ranker_pilot.py`
- Modify: `tests/test_artifact_manifest.py`

**Interfaces:**
- Produces: `run_pilot_phase_a(selection, *, method, infer_batch, runtime_metadata) -> PilotPhaseAResult`, `write_pilot_phase_a(result, output_dir) -> dict[str, str]`,
  `load_frozen_pilot(artifact_dir) -> PilotPhaseAResult`, and `evaluate_pilot(artifact_dir, qrels_path) -> dict[str, Any]`.

- [ ] **Step 1: Write failing isolation and corruption tests**

Use inference fixtures and two contradictory qrel fixtures. Assert identical selection,
prompts, raw outputs, scores, rankings, and Phase A hashes while post-hoc relevance ranks
change. Corrupt each core file, current query file, current candidate file, prompt hash,
smoke receipt, model revision, dtype, seed, and code revision; require a specific failure.

- [ ] **Step 2: Verify the new corruption tests fail**

Run: `python -m pytest -q tests/test_llm_ranker_pilot.py tests/test_artifact_manifest.py`

Expected: failures show absent pilot artifacts and weak smoke/input enforcement.

- [ ] **Step 3: Implement Phase A and exact receipt validation**

Phase A accepts only the frozen qrel-free `selection.json` and inference callbacks. It
must not accept or open `selection_judgments.json` or a qrel path.
Write these required files:

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

Require exactly that core-file set in `ranking_receipt.json`; hash every core file and
the current query/candidate inputs. Require the real smoke receipt to contain exact grade
token IDs `{0: 15, 1: 16, 2: 17}`, parity delta at most `1e-4`, 20 semantically valid
smoke handles, pinned model/revision/dtype, and recorded vLLM/Torch/HIP versions.

- [ ] **Step 4: Preserve raw pointwise diagnostics**

Extend the pointwise adapter record to include the emitted grade token ID/text alongside
all three requested log probabilities and normalized expected-grade score. Ranking still
uses only `P(1) + 2P(2)` with candidate-ID ties.

- [ ] **Step 5: Run artifact tests and commit**

Run: `python -m pytest -q tests/test_llm_ranker_pilot.py tests/test_artifact_manifest.py tests/test_llm_ranker_experiment.py`

Expected: all qrel-isolation and corruption tests pass.

Commit: `git commit -am 'feat: validate LLM ranker pilot receipts'`.

### Task 4: Add Pilot CLI and Run Stage 1 Once

**Files:**
- Modify: `trec_mlm/cli.py`
- Modify: `tests/test_llm_ranker_pilot.py`
- Create: `artifacts/llm_label_ranker/pilot/stage1/pointwise/`
- Create: `artifacts/llm_label_ranker/pilot/stage1/listwise/`

**Interfaces:**
- Produces: `trec-mlm llm-label-ranker-pilot --stage 1 --method pointwise|listwise`.

- [ ] **Step 1: Write failing CLI tests**

Require `--selection`, `--method`, `--code-revision`, `--output-dir`, and the passed smoke
receipt. Reject qrel arguments during Phase A and reject candidate counts other than the
selection manifest's frozen count.

- [ ] **Step 2: Implement the CLI using one engine lifecycle per invocation**

Load the frozen Qwen3-4B engine, invoke exactly 72 pointwise messages or 18 listwise
messages, accumulate exact prompt/output token counts, and write Phase A only after every
output validates. Do not expose a qrel path on this command.

- [ ] **Step 3: Run all pilot tests in the pinned container**

Run the pilot, canonical ranker, label, and artifact tests inside the pinned ROCm image.

Expected: zero failures and unchanged canonical 150-label behavior.

- [ ] **Step 4: Commit the clean implementation before GPU inference**

Commit: `git commit -am 'feat: run LLM ranker feasibility pilot'`. Record the full commit
SHA and require a clean launch worktree in both Stage 1 receipts.

- [ ] **Step 5: Run pointwise and listwise Stage 1 once**

Use the validated `/dev/kfd`, `/dev/dri`, render/video groups, shared IPC, Hugging Face
cache, and repository mounts. The expected work is 72 one-token pointwise calls and 18
short listwise calls. Validate all hashes and counts immediately after completion.

### Task 5: Compare Baselines and Decide the Stage 1 Gate

**Files:**
- Modify: `trec_mlm/llm_ranker_pilot.py`
- Modify: `tests/test_llm_ranker_pilot.py`
- Create: `artifacts/llm_label_ranker/pilot/stage1/report.json`
- Create: `docs/results/2026-07-13-llm-ranker-stage1-pilot.md`

**Interfaces:**
- Produces: `build_pilot_report(*, pointwise_dir, listwise_dir, qwen_dir, e5_dir, qrels_path) -> dict[str, Any]` and a six-row decision table.

- [ ] **Step 1: Write failing restricted-baseline and gate tests**

For each question, restrict contextual Qwen4B and E5 rankings to the same twelve labels.
Require relevant-label ranks, top-three IDs, LLM-minus-baseline rank differences,
rotation agreement, parse failures, and a deterministic gate calculation.

- [ ] **Step 2: Implement post-hoc evaluation**

Load qrels only after both Phase A receipts validate. A method passes when a relevant
label is top-three for at least four of six questions, successful questions span more
than one subject stratum, and no success depends on a parse failure. Label all findings
qrel-assisted and exploratory; do not report a confidence interval.

- [ ] **Step 3: Add manual error classifications**

Classify each miss as one of: label ambiguity, related-topic confusion, overly broad
label preference, overly narrow label preference, or ranking-format failure. Store the
classification and one-sentence evidence in the result document.

- [ ] **Step 4: Run report tests and write the Stage 1 receipt**

Run: `python -m pytest -q tests/test_llm_ranker_pilot.py`

Then generate `report.json` and the concise Markdown table. Verify the report can be
fully recomputed from hashed Phase A files, frozen embeddings, and qrels.

- [ ] **Step 5: Apply the conditional stop**

If neither LLM method passes, stop and discuss the six examples. If either passes,
continue to Task 6 without changing prompts, model, scoring, or candidate construction.

### Task 6: Run the Conditional Twelve-by-Twenty Pilot

**Files:**
- Create: `artifacts/llm_label_ranker/pilot/stage2/selection.json`
- Create: `artifacts/llm_label_ranker/pilot/stage2/selection_judgments.json`
- Create: `artifacts/llm_label_ranker/pilot/stage2/pointwise/`
- Create: `artifacts/llm_label_ranker/pilot/stage2/listwise/`
- Create: `docs/results/2026-07-13-llm-ranker-stage2-pilot.md`

**Interfaces:**
- Consumes: a passed Stage 1 decision receipt.
- Produces: twelve new questions, twenty labels each, 240 pointwise pairs, 36 listwise
  calls, and the same question-level comparison table.

- [ ] **Step 1: Refuse Stage 2 without a passing Stage 1 receipt**

Write a failing test that supplies a failed or hash-mutated Stage 1 report and assert
that Stage 2 selection construction raises `ValueError("Stage 1 feasibility gate did not pass")`.

- [ ] **Step 2: Freeze twelve non-overlapping Stage 2 questions**

Select four issue-like clear semantic mappings, four contextual-embedding successes,
and four contextual-embedding failures. Exclude all Stage 1 IDs. Per question, retain up
to two highest-grade labels, six Qwen4B qrel-zero confounders, and deterministic negatives
to twenty.

- [ ] **Step 3: Run the same Phase A methods once**

Run exactly 240 pointwise pairs and 36 listwise calls. Do not repeat, fuse, or change the
model. Validate every artifact before qrel evaluation.

- [ ] **Step 4: Generate the Stage 2 report and stop**

Apply the same restricted-baseline table and feasibility heuristic over twelve new
questions. Clearly distinguish Stage 1 selection evidence from Stage 2 generalization
evidence. Stop after reporting; do not start a full sweep or deployment selection.

### Task 7: Independent Review and Final Verification

**Files:**
- Modify only files implicated by validated review findings.

**Interfaces:**
- Produces: advisor disposition covering code, raw pilot artifacts, and claims.

- [ ] **Step 1: Ask the existing independent advisor to audit Stage 1 code and findings**

Require exact file/line findings for selection leakage, prompt bytes, qrel isolation,
variable-size structured output, pointwise token probabilities, restricted baselines,
gate calculation, and provenance. Do not proceed past an unresolved high-severity issue.

- [ ] **Step 2: Incorporate validated corrections and rerun affected tests**

Preserve raw inference receipts. Regenerate only derived artifacts unless the advisor
finds a raw-inference-invalidating bug.

- [ ] **Step 3: Run final verification**

Run all ranker/label tests in the pinned container, Ruff, formatting, `git diff --check`,
receipt hash validation, exact request-count checks, and report recomputation.

- [ ] **Step 4: Commit locally and report the decision**

Commit source, tests, compact manifests, and reports. Do not publish, push, fuse systems,
or select a deployment model.
