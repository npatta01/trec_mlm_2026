# E5 Raw Historical-Response Refinement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:subagent-driven-development` to implement this plan task-by-task.
> Steps use checkbox syntax for tracking.

**Goal:** Regenerate the complete durable E5 profile artifact and determine
whether fixed top-10 raw historical-response evidence improves the promoted
candidate-profile router.

**Architecture:** Reuse the pinned qrel-free E5 inference and deterministic
centroid constructors. Add one isolated qrel-free refinement artifact and one
separate qrel-aware evaluator. Large arrays live in the repository's shared Git
data area; tracked code and reports remain in this worktree.

**Tech stack:** Python 3.13, NumPy, existing TREC metrics and artifact helpers,
PyTorch/Transformers in the pinned rootless ROCm Podman image, pytest, Ruff.

## Global constraints

- Follow `docs/superpowers/specs/2026-07-22-e5-raw-response-refinement-design.md`.
- Encoder is `intfloat/e5-mistral-7b-instruct` revision
  `d62cc4417be0e3289c03dd459d1ef2d137e2554e`; batch 8, seed 2026,
  512 tokens, 4,096 dimensions, normalized float32 persistence.
- Construction accepts no qrel path or qrel object.
- Raw neighbor count is 10; hierarchical weights are exactly 0.8 and 0.2;
  equal RRF uses `k=60`.
- No parameter sweep or post-result variant.
- Preserve unrelated dirty worktree changes.
- Durable root is
  `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement`.

---

### Task 1: Exact candidate-local raw scorer

**Files:**
- Create: `trec_mlm/raw_response_refinement.py`
- Create: `tests/test_raw_response_refinement.py`

**Interfaces:**
- Produces `score_candidate_records(query_embeddings, record_embeddings,
  record_ids, *, top_k=10) -> CandidateRawScores`.
- Produces `combine_hierarchical_scores(broad_scores, refined_scores) ->
  np.ndarray`.
- Produces `rank_score_matrix(query_ids, candidate_ids, scores) -> rankings`.

- [x] **Step 1: Write failing tests for exact top-k and ties**

```python
def test_raw_scores_average_top_k_and_break_ties_by_record_id() -> None:
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    records = np.array([[0.8, 0.6], [0.8, -0.6], [1.0, 0.0]], dtype=np.float32)
    result = score_candidate_records(
        queries, records, ("b", "a", "c"), top_k=2
    )
    assert result.record_ids == (("c", "a"),)
    assert result.mean_scores[0] == pytest.approx(0.9)
```

Add separate failing tests for support below 10, non-finite inputs, row-count
mismatch, fixed float64 `0.8/0.2` arithmetic, and candidate-ID score ties.

- [x] **Step 2: Run RED**

```bash
.venv/bin/pytest tests/test_raw_response_refinement.py -q
```

Expected: collection fails because `trec_mlm.raw_response_refinement` does not
exist.

- [x] **Step 3: Implement the minimal pure scorer**

```python
@dataclass(frozen=True, slots=True)
class CandidateRawScores:
    mean_scores: NDArray[np.float64]
    record_indices: NDArray[np.int32]
    record_ids: tuple[tuple[str, ...], ...]
    similarities: NDArray[np.float64]

def score_candidate_records(
    query_embeddings: NDArray[np.float32],
    record_embeddings: NDArray[np.float32],
    record_ids: Sequence[str],
    *,
    top_k: int = 10,
) -> CandidateRawScores:
    similarities = np.asarray(query_embeddings @ record_embeddings.T,
                              dtype=np.float64)
    count = min(top_k, similarities.shape[1])
    # Exact deterministic order: descending similarity, then record ID.
    orders = [sorted(range(len(record_ids)),
                     key=lambda i: (-float(row[i]), record_ids[i]))[:count]
              for row in similarities]
    selected = np.asarray(orders, dtype=np.int32)
    selected_scores = np.take_along_axis(similarities, selected, axis=1)
    means = np.asarray(
        [math.fsum(map(float, row)) / count for row in selected_scores],
        dtype=np.float64,
    )
    selected_ids = tuple(
        tuple(record_ids[index] for index in row) for row in selected
    )
    return CandidateRawScores(means, selected, selected_ids, selected_scores)
```

Use `math.fsum` for each selected mean and `np.lexsort` with candidate ID as the
final ranking tie-break.

- [x] **Step 4: Run GREEN and focused lint**

```bash
.venv/bin/pytest tests/test_raw_response_refinement.py -q
.venv/bin/ruff check trec_mlm/raw_response_refinement.py tests/test_raw_response_refinement.py
```

Expected: all focused tests pass and Ruff exits 0.

- [x] **Step 5: Commit Task 1 files only**

```bash
git add trec_mlm/raw_response_refinement.py tests/test_raw_response_refinement.py
git commit -m "feat: add exact raw response refinement scorer"
```

### Task 2: Qrel-free construction artifact

**Files:**
- Modify: `trec_mlm/raw_response_refinement.py`
- Create: `tests/test_raw_response_refinement_artifact.py`

**Interfaces:**
- Produces `RawRefinementConfig(inference_path, construction_path,
  dense_artifact_path)` with no qrel field.
- Produces `construct_raw_response_refinement(config)`.
- Produces atomic `write_raw_response_refinement_artifact`, strict
  `load_raw_response_refinement_artifact`, and
  `compare_raw_response_refinement_repeats`.

- [x] **Step 1: Write failing artifact tests**

Create tiny inference/construction fixtures with two queries and two candidates.
Assert exact files and manifest closure, `(Q,C,10)` record-index and similarity
arrays, dependency aggregate hashes, complete rankings, atomic failure cleanup,
tamper rejection, and absence of any `qrel` field in `RawRefinementConfig`.

- [x] **Step 2: Run RED**

```bash
.venv/bin/pytest tests/test_raw_response_refinement_artifact.py -q
```

Expected: imports fail for the missing artifact API.

- [x] **Step 3: Implement candidate-block construction**

```python
for candidate_index, candidate_id in enumerate(candidate_ids):
    start, stop = candidate_slices[candidate_id]
    raw = score_candidate_records(
        query_embeddings,
        record_embeddings[start:stop],
        record_ids[start:stop],
        top_k=RAW_TOP_K,
    )
    refined_scores[:, candidate_index] = raw.mean_scores
    top_indices[:, candidate_index, :raw.record_indices.shape[1]] = (
        raw.record_indices + start
    )
```

Read the inference artifact with `load_profile_inference_artifacts`, broad
scores/rankings with `load_profile_artifacts`, and fuse with
`reciprocal_rank_fusion`. Persist arrays, JSON rankings, TREC runs, selected
record IDs, configuration/provenance, validation, and a lexicographically
hashed manifest through existing `artifact_io` helpers. Refuse overwrite.

- [x] **Step 4: Run GREEN, then block-size invariance**

```bash
.venv/bin/pytest tests/test_raw_response_refinement.py tests/test_raw_response_refinement_artifact.py -q
```

Expected: all tests pass; repeated fixture construction has identical selected
IDs and rankings and maximum score difference at most `1e-12`.

- [x] **Step 5: Commit Task 2 files only**

```bash
git add trec_mlm/raw_response_refinement.py tests/test_raw_response_refinement_artifact.py
git commit -m "feat: persist raw response refinement artifacts"
```

### Task 3: Sealed evaluator and decision receipt

**Files:**
- Create: `trec_mlm/raw_response_refinement_evaluation.py`
- Create: `tests/test_raw_response_refinement_evaluation.py`

**Interfaces:**
- Consumes one validated construction and its repeat receipt, then qrels.
- Produces metrics, paired comparisons, zero transitions, per-query rows, and
  the frozen `advance|diagnostic|stop|invalid` decision.

- [x] **Step 1: Write failing evaluator tests**

```python
def test_advance_requires_all_primary_gates() -> None:
    decision = primary_decision(
        ndcg_interval_lower=0.001,
        ndcg_point_delta=0.01,
        mrr_delta=-0.004,
        zero_count=98,
        integrity_checks={"manifest": True},
    )
    assert decision["decision"] == "advance"
```

Add independent failing cases for zero count 99, interval lower bound 0,
MRR delta below -0.005, invalid integrity, diagnostic rescue, qrel inventory
mismatch, and construction hash mutation.

- [x] **Step 2: Run RED**

```bash
.venv/bin/pytest tests/test_raw_response_refinement_evaluation.py -q
```

- [x] **Step 3: Implement by composing existing evaluation primitives**

Use `load_complete_qrels`, `evaluate_rankings`, `paired_interval`,
`win_tie_loss`, and `zero_transitions`. Hard-code the matched parent identities
and require the parent metrics in the rebuilt construction to reproduce the
tracked values within `1e-12` before comparing the new systems.

- [x] **Step 4: Run GREEN and focused lint**

```bash
.venv/bin/pytest tests/test_raw_response_refinement_evaluation.py -q
.venv/bin/ruff check trec_mlm/raw_response_refinement_evaluation.py tests/test_raw_response_refinement_evaluation.py
```

- [x] **Step 5: Commit Task 3 files only**

```bash
git add trec_mlm/raw_response_refinement_evaluation.py tests/test_raw_response_refinement_evaluation.py
git commit -m "feat: evaluate raw response refinement"
```

### Task 4: Terminal wrappers and experiment report

**Files:**
- Create: `scripts/run_raw_response_refinement.py`
- Create: `scripts/evaluate_raw_response_refinement.py`
- Create: `scripts/build_raw_response_refinement_report.py`
- Create: `scripts/run_profile_inference_smoke.py`
- Create: `tests/test_raw_response_refinement_cli.py`

**Interfaces:**
- Construction wrapper has only `construct`, `validate`, and `compare`
  operations and exposes no tunable `k` or weight.
- Evaluation wrapper is the only interface accepting `--qrels`.
- Smoke wrapper has `run` and `compare` operations. It selects short, median,
  and longest real records, then derives the `over_512` smoke-only input by
  repeating the longest response exactly three times separated by `\n\n` while
  preserving and binding its canonical source ID. It emits the exact receipt
  schema consumed by `validate_profile_smoke_receipt`.

- [x] **Step 1: Write failing CLI boundary tests**

Inspect parser actions and assert no construction option contains `qrel`,
`top-k`, `weight`, `encoder`, or a fixture bypass. Exercise subprocess help and
invalid-input boundaries; the successful production CLI canary uses official
artifacts after the sealed smoke and inference stages.

- [x] **Step 2: Run RED**

```bash
.venv/bin/pytest tests/test_raw_response_refinement_cli.py -q
```

- [x] **Step 3: Implement thin wrappers and deterministic report builder**

Wrappers call only the public module APIs. The smoke wrapper calls the existing
frozen profile formatter/encoder and writes normalized vectors plus runtime
metadata for one process; its compare operation hashes the two individual
receipts and checks maximum absolute embedding difference. The report builder writes
`reports/trec_2026_e5_raw_response_refinement_results.{md,json}` and states that
development effectiveness is post-hoc confirmation, not independent hidden-test
evidence.

- [x] **Step 4: Run GREEN**

```bash
.venv/bin/pytest tests/test_raw_response_refinement_cli.py -q
.venv/bin/ruff check scripts/run_raw_response_refinement.py scripts/evaluate_raw_response_refinement.py scripts/build_raw_response_refinement_report.py scripts/run_profile_inference_smoke.py
```

- [x] **Step 5: Commit Task 4 files only**

```bash
git add scripts/run_raw_response_refinement.py scripts/evaluate_raw_response_refinement.py scripts/build_raw_response_refinement_report.py scripts/run_profile_inference_smoke.py tests/test_raw_response_refinement_cli.py
git commit -m "feat: expose raw response refinement workflow"
```

### Task 5: Cheap verification and qrel-free GPU smoke

**Files:**
- Update this plan with exact receipts and commands after each check.
- Create durable artifacts under the shared-data root; do not commit arrays.

- [x] **Step 1: Run the complete preflight**

```bash
git diff --check
.venv/bin/pytest -q
.venv/bin/ruff check trec_mlm tests scripts
podman image inspect docker.io/vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28
rocminfo | rg 'gfx1151'
df -h /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data
```

Require zero test/lint failures, the pinned image, `gfx1151`, and at least
20 GiB free disk.

- [x] **Step 2: Run two fresh smoke processes without qrels or network**

Use two raw short/median records plus the audited smoke-only derivation from the
longest real record, the pinned container, read-only code/model mounts,
`--network none`, and separate outputs.
Require finite normalized `(4,4096)` matrices and maximum repeat difference
`<=1e-5`; write the combined smoke receipt under the durable root.

```bash
podman run --rm --entrypoint python --device /dev/kfd --device /dev/dri --security-opt label=disable \
  --network none -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/code \
  -v /home/npatta01/.codex/worktrees/28f304c6-3d64-4618-a7eb-4e626ca11605/trec_mlm_2026/trec_mlm:/code/trec_mlm:ro \
  -v /home/npatta01/.codex/worktrees/28f304c6-3d64-4618-a7eb-4e626ca11605/trec_mlm_2026/scripts:/code/scripts:ro \
  -v /home/npatta01/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/llm_discovery_data.json:/inputs/discovery.json:ro \
  -v /home/npatta01/.codex/worktrees/28f304c6-3d64-4618-a7eb-4e626ca11605/trec_mlm_2026/reports/trec_2026_discovery_profile_audit.json:/inputs/candidates.json:ro \
  -v /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement:/outputs:rw \
  docker.io/vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28 \
  python /code/scripts/run_profile_inference_smoke.py run \
  --discovery /inputs/discovery.json --candidates /inputs/candidates.json \
  --output /outputs/receipts/smoke1
```

Repeat with `--output /outputs/receipts/smoke2`, then run:

```bash
.venv/bin/python scripts/run_profile_inference_smoke.py compare \
  --first /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/receipts/smoke1 \
  --second /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/receipts/smoke2 \
  --receipt /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/receipts/smoke_repeat_receipt.json
```

- [x] **Step 3: Re-run the sealed workflow canaries**

Run the focused in-process fixture tests for construction, artifact reload,
repeat comparison, and evaluation, plus CLI help and invalid-input subprocess
checks. Production CLIs deliberately expose no fixture bypass; their first
successful end-to-end invocation uses the official artifacts below. Do not open
official qrels during construction canaries.

### Task 6: Full E5 inference and deterministic centroid reconstruction

**Files:**
- Durable inference: `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/inference`
- Durable construction: `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/construction`
- Durable execution logs/receipts: `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/receipts/`

- [x] **Step 1: Record exact input hashes and isolation receipt**

Record the query, discovery, candidate, contextual-E5 validation, source-tree,
container, model-revision, mount, environment, and initial-open-file identities.

- [x] **Step 2: Run full existing `profile-infer` in the pinned container**

Mount no qrels and disable networking. Point `--output-dir` directly at the
durable root. Preserve the console log separately. Expected official counts:
322 queries, 150 candidates, 182,764 records, and 7,759 label questions.

```bash
podman run --rm --entrypoint python --device /dev/kfd --device /dev/dri --security-opt label=disable \
  --network none -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/code \
  -v /home/npatta01/.codex/worktrees/28f304c6-3d64-4618-a7eb-4e626ca11605/trec_mlm_2026/trec_mlm:/code/trec_mlm:ro \
  -v /home/npatta01/.cache/huggingface:/root/.cache/huggingface:ro \
  -v /home/npatta01/.codex/worktrees/28f304c6-3d64-4618-a7eb-4e626ca11605/trec_mlm_2026/data/dev/llm_dev_data.tsv:/inputs/queries.tsv:ro \
  -v /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/llm_discovery_data.json:/inputs/discovery.json:ro \
  -v /home/npatta01/.codex/worktrees/28f304c6-3d64-4618-a7eb-4e626ca11605/trec_mlm_2026/reports/trec_2026_discovery_profile_audit.json:/inputs/candidates.json:ro \
  -v /home/npatta01/.codex/worktrees/expert-label-results-only/trec_mlm_2026/artifacts/contextual_e5_clean/final_run1:/inputs/contextual_e5:ro \
  -v /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement:/outputs:rw \
  docker.io/vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28 \
  python -m trec_mlm.cli profile-infer \
  --queries /inputs/queries.tsv --discovery /inputs/discovery.json \
  --candidates /inputs/candidates.json --e5-artifact /inputs/contextual_e5 \
  --smoke-repeat-receipt /outputs/receipts/smoke_repeat_receipt.json \
  --output-dir /outputs/inference \
  --execution-receipt /outputs/receipts/inference_execution_receipt.json \
  2>&1 | tee /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/receipts/inference.log
```

- [x] **Step 3: Reload and hash-verify the inference artifact**

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -c 'from trec_mlm.profile_inference_artifact import load_profile_inference_artifacts; import sys; a=load_profile_inference_artifacts(sys.argv[1]); print(a.query_embeddings.shape, a.record_embeddings.shape, a.manifest["aggregate_sha256"])' \
/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/inference
```

Expected shapes: `(322, 4096)` and `(182764, 4096)`.

- [x] **Step 4: Run existing `profile-construct` and validate parent metrics**

Use one-thread numerical environment and the frozen dense artifact. Require the
rebuilt `profile_only` and `dense_profile_equal` rankings and metrics to match
their tracked identities before raw refinement evaluation.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python -m trec_mlm.cli profile-construct \
  --queries data/dev/llm_dev_data.tsv \
  --discovery /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/llm_discovery_data.json \
  --candidates reports/trec_2026_discovery_profile_audit.json \
  --dense-artifact /home/npatta01/.codex/worktrees/88a454dd-e8be-4b89-923b-57f896689cf9/trec_mlm_2026/artifacts/dense_query_knn_baseline \
  --inference-artifact /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/inference \
  --output-dir /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/construction

.venv/bin/python -m trec_mlm.cli profile-validate \
  --construction /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/construction
```

### Task 7: Freeze, repeat, evaluate once, and report

**Files:**
- Durable runs: `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/refinement_run1` and `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/refinement_run2`
- Durable receipt/evaluation: `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/repeat_receipt.json` and `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/evaluation/`
- Create tracked results:
  `reports/trec_2026_e5_raw_response_refinement_results.md`
  and `.json`

- [x] **Step 1: Construct run 1 and run 2 without qrels**

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
.venv/bin/python scripts/run_raw_response_refinement.py construct \
  --inference /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/inference \
  --profiles /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/construction \
  --dense /home/npatta01/.codex/worktrees/88a454dd-e8be-4b89-923b-57f896689cf9/trec_mlm_2026/artifacts/dense_query_knn_baseline \
  --output-dir /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/refinement_run1
```

Repeat the exact command with `refinement_run2` as the output directory.

- [x] **Step 2: Require repeat agreement and freeze ranking hashes**

Require identical selected record IDs, complete rankings, byte-identical TREC
runs, and maximum component-score difference `<=1e-12`.

```bash
.venv/bin/python scripts/run_raw_response_refinement.py compare \
  --first /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/refinement_run1 \
  --second /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/refinement_run2 \
  --receipt /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/repeat_receipt.json
```

- [x] **Step 3: Apply the pre-qrel no-op gate**

If both new systems have unchanged top-10 candidate lists versus their matched
parents, record `stop-no-op` without opening official qrels. Otherwise continue
exactly once.

- [x] **Step 4: Evaluate with official qrels in the separate evaluator**

Confirm qrel SHA/inventory, construction immutability before/after, and emit the
sole primary comparison plus secondary diagnostics.

```bash
.venv/bin/python scripts/evaluate_raw_response_refinement.py \
  --construction /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/refinement_run1 \
  --repeat-receipt /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/repeat_receipt.json \
  --qrels data/dev/llm_dev_qrels.txt \
  --output-dir /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/evaluation
```

- [x] **Step 5: Build and verify the tracked report**

```bash
.venv/bin/python scripts/build_raw_response_refinement_report.py \
  --evaluation /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/evaluation/evaluation.json \
  --markdown reports/trec_2026_e5_raw_response_refinement_results.md \
  --json reports/trec_2026_e5_raw_response_refinement_results.json
git diff --check
.venv/bin/pytest -q
.venv/bin/ruff check trec_mlm tests scripts
```

Read the fresh outputs, verify zero failures, audit the report against the
frozen decision rule, and record hashes/commands in this canonical plan.

### Execution outcome — 2026-07-22

- Decision: **advance**.
- Full inference: 322 queries, 182,764 records, 7,759 label questions, and 150
  candidates at 4,096 dimensions; aggregate
  `d9ef709db6642d3f7d9f62f25186939378c130ac17a407ee736e383b2f7d7054`.
- Centroid construction aggregate:
  `67c66f5bdca6c113ddc936051f8b10db8ae77fa770a9588290bf248ef8568780`.
- Refinement aggregates: run 1
  `113fa6391df361bd161c2c9ad6a1d3ccb7f6e6f59a917b66135c6a87b0a33f83`;
  run 2 `6767170a374dffc0689f5b6af475759b8cee71c3a87106a375609f0770de94b1`.
- Repeat gate: identical selected record IDs and rankings; maximum absolute
  score difference `0.0` against tolerance `1e-12`.
- Primary system nDCG@10: `0.435625273018` versus `0.420213992194`;
  delta `+0.015411280824`, paired 95% CI
  `[+0.006964609515, +0.024355856294]`.
- Primary MRR: `0.465457752325` versus `0.453017393783`; delta
  `+0.012440358542`.
- Zero-nDCG queries: `92` versus `99`; eight prior zeros rescued and one new
  zero introduced.
- Secondary raw-hierarchical nDCG@10: `0.449022366769` versus profile-only
  `0.414924459979`; delta `+0.034097906791`.
- Sealed evaluation SHA-256:
  `9ac887a04faea18f00789d486f350ccca9253e3bceb2de2ee4bb77eb4f06685c`.
- Final verification before commit: `674 passed, 3 skipped`; full Ruff and
  `git diff --check` passed; independent audit found no Critical or Important
  issue and reproduced the decision statistics.

- [x] **Step 6: Commit only experiment code and compact reports**

```bash
git add plans/e5-raw-response-refinement.md \
  reports/trec_2026_e5_raw_response_refinement_results.md \
  reports/trec_2026_e5_raw_response_refinement_results.json
git commit -m "exp: evaluate E5 raw response refinement"
```

Do not add large arrays, raw logs, discovery data, or unrelated worktree files.
