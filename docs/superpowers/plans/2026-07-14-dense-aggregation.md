# Dense Aggregation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement, validate, and run the advisor-approved GitHub issue #13 local-domain-prior dense aggregation experiment exactly as locked in `protocols/trec_2026_dense_aggregation.md`, with a qrel-free feasibility stop gate, an isolated legacy-control lock, two reproducible authoritative CPU runs, and read-only evaluation/reporting.

**Architecture:** Build a new isolated aggregation stack rather than extending the existing dense experiment. Pure float64 scoring, framed hashing, pinned deterministic clustering, an append-only fit cache, nested orchestration, immutable artifacts, and post-freeze evaluation live in separate modules. A dedicated bootstrap sets numerical thread variables before NumPy imports. All implementation code is complete and hashed before the qrel-free gate. Qrels cannot be opened until that gate passes; the new aggregation experiment cannot begin until the subsequent qrel-using legacy lock also validates.

**Tech Stack:** Python 3.13.9, NumPy 2.5.1, SciPy 1.18.0, scikit-learn 1.9.0, threadpoolctl 3.6.0, pytest, existing `trec_mlm` qrel/metric/TREC helpers, flat binary shards plus JSONL receipts, Markdown, and standalone HTML.

## Global Constraints

- The locked protocol is authoritative. Its raw SHA-256 is `bbe6b50fd3d20efe3213a446f6665cbe0527114f34bf5c1e90236af4f998ad49`.
- Do not edit `trec_mlm/dense_query_knn.py`; its locked raw SHA-256 is `90f8db3abbb2887d8513fb881c631a3843319c29da02e935e1ecdb6c3c85e248`.
- Do not change the encoder, embeddings, candidate inventory, qrels, grid, formula, K-means settings, metrics, thresholds, threading, or fold membership.
- Do not use a GPU. Every authoritative fit uses one CPU process and one thread in every declared numerical library.
- The feasibility worker has no qrel path, grade field, qrel import, or scoring import.
- A failed or stale feasibility gate stops execution before qrels are opened. The predeclared fallback is never activated automatically.
- The legacy fixed-MPNet exporter imports no module whose name starts with `trec_mlm.dense_aggregation`.
- Freeze construction artifacts before opening outer effectiveness metrics, slices, zero transitions, deployment metrics, or promotion gates.
- All authoritative artifacts are atomically published, immutable, manifest-validated, and local/private.
- Do not publish, push, open a pull request, or change sharing permissions without the user's explicit authorization.

---

## Verified Starting Point

- Branch: `codex/issue-13-dense-aggregation` at preregistration commit `93f0925`.
- Queries: `data/dev/llm_dev_data.tsv`, raw SHA-256 `862ff85381b8055cd927ada25c56666594ab1ec3c21e182beb20c58f6c637f3c`.
- Qrels: `data/dev/llm_dev_qrels.txt`, raw SHA-256 `3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35`.
- Dense snapshot: `reports/trec_2026_dense_query_knn_results.json`, raw SHA-256 `3ca4252cb37f12e9272561422fd2a45d0dcbf741770d0f544efda8acf4ddf902`.
- Dense per-query reference: `reports/trec_2026_dense_query_knn_per_query.tsv`, raw SHA-256 `ae4a0cf8bc7eb0642e1c0d315cafa3a56cf60b3d6bfb6ab63e80fb2d0de56437`.
- A validated MPNet array exists at `/home/npatta01/.codex/worktrees/88a454dd-e8be-4b89-923b-57f896689cf9/trec_mlm_2026/artifacts/dense_query_knn_baseline/embeddings/mpnet_base_v2.npy`; its shape is `(322, 768)`, dtype is float32, and legacy logical hash is `6d197055eb62ab9809ae321b0f60072e20f17524861e6a989b49cd35c389055b`.
- The current general CLI imports NumPy before dispatch. It is not an authoritative entry point for this experiment.
- Existing dense artifact output is non-atomic and rounds scores; it is an equivalence reference only, not a writer to reuse.
- `artifacts/` is gitignored. The retained official binary roots remain local unless the user later authorizes another destination.

## Implementation Decisions Fixed Here

These decisions make protocol details executable without changing any statistical or numerical choice.

1. Copy the validated frozen MPNet `.npy` and encoder metadata into the ignored `artifacts/frozen_dense_input/` directory. The qrel-free config contains no legacy source config because that file embeds a qrel path; both public configs require explicit qrel-free input paths and validate content, with no path hard-coded in Python.
2. Define a configuration's selection rank as its 1-based ordinal after sorting by `(-inner_macro_ndcg_at_10, grid_index)`. Record both its gap to the winner and the winner-minus-runner-up margin.
3. Store all inner scores in one `<f8` open-memmap with logical shape `(322, 16, 321, 150)` and all ranks in one `<u2` open-memmap with the same logical shape. Each `(outer, config)` slice is the protocol's logical 321-by-150 artifact and receives its own framed hash.
4. Record the required fit-wide membership hash and an additional per-cluster membership hash made by applying the same canonical-membership tag to that cluster's one-row membership payload.
5. Freeze an explicit `IMPLEMENTATION_FILES` tuple in `dense_aggregation_contract.py`. It includes every new experiment, clustering, cache, evaluation, report, and entry-point Python file plus the modified explorer script. A test compares the tuple with the implemented file set so an unrecorded Python entry point fails.
6. Compute p95 with linear interpolation at index `(n - 1) * 0.95` in the sorted duration list and record the method string `linear_index_(n-1)*0.95`.
7. A measured fit operation includes receipt `flush()` before references to support/fit arrays are deleted. JSONL and binary shards are `fsync()`ed when a shard closes or the cache finalizes; `fsync()` is not performed per fit.
8. Require the operator flag `--attest-exclusive-resources` at feasibility coordination time. Record it in both receipts and reject the gate when absent.
9. Treat the raw SHA-256 of the lossless canonical `fixed_mpnet_k50_control.tsv` as the official legacy score/ranking lock because every score, prior, and rank is encoded in that file. The lock also records the TREC run hash.
10. Track a feasibility config at `configs/trec_2026_dense_aggregation_feasibility.json`. Generate `configs/trec_2026_dense_aggregation.json` only after the legacy lock exists; the official config embeds the feasibility aggregate hash and legacy TSV hash.
11. The construction command atomically publishes a manifested prediction root containing only the inner metrics required for selection and no outer-effectiveness/control/slice/deployment metrics. The evaluator is a separate process that opens this immutable root read-only and atomically publishes a separate evaluation root.
12. The construction root contains the stitched primary predictions plus the persisted singleton fits, but no full-development prediction matrices. The evaluator writes and hashes `primary_freeze_receipt.json` before it reloads those singleton fits to construct the sixteen full-development matrices and optimistic deployment hint without another K-means fit.
13. A third manifested comparison root names Run A's construction/evaluation as the decision authority and proves Run B is semantically identical outside the exact volatile-field allowlist.
14. Compute the MRR promotion delta as the canonical-query-ordered mean of per-query differences. Use published headline values only for displayed reference headlines.
15. Every atomic writer fails if its destination already exists. It never overwrites or merges an authoritative root.
16. Store cluster centers, labels, and inertia in flat append-only binary shards. This preserves the fitted centers without creating 104,006 directories and permits strict revalidation.
17. Join query-explorer evidence by `outer_query_id` and `candidate_id`; join query-level neighbor and cluster records by `outer_query_id`. Explorer output always lives outside a manifested run root.
18. Treat `pyproject.toml` and `uv.lock` as runtime provenance, and `trec_mlm/query_knn.py`, `trec_mlm/metrics.py`, and `trec_mlm/runio.py` as executable dependency provenance, alongside the new Python implementation allowlist. Any change to one of these files invalidates feasibility and every downstream artifact just like a new implementation change. The separately frozen `trec_mlm/dense_query_knn.py` hash remains in the same contract inventory.
19. Resolve every configured path against an explicit repository root and require the resolved protocol path to equal `<repo-root>/protocols/trec_2026_dense_aggregation.md`. Never hash a process-working-directory-relative protocol path.
20. Every official invocation uses a fresh immutable attempt namespace. The first execution uses the paths shown below. A new suffix is allowed only for a pre-qrel authorized invalidation or an operational interruption/collision that produced no validated gate, lock, construction, evaluation, or comparison result. A feasibility `stop`, legacy mismatch, or post-qrel integrity failure is terminal under this protocol and cannot be retried without the required advisor action. Aborted/stale roots are retained and never reused, deleted, merged, or overwritten.
21. Before constructing a cache or staging artifact, create only their parent directories and require `cache_dir.parent.stat().st_dev == output_dir.parent.stat().st_dev`. Fail preflight before qrel access when atomic cache transfer cannot stay on one filesystem.

## File Responsibility Map

### Existing files to reuse without modification

- `trec_mlm/dense_query_knn.py`: frozen embedding/similarity validators and the legacy scorer only.
- `trec_mlm/query_knn.py`: complete-qrel validation and fixed TF-IDF ranker after the gate. The qrel-free contract module implements its own tiny query-only TSV loader so the feasibility import graph never imports a module that also defines qrel/scoring APIs.
- `trec_mlm/metrics.py`: nDCG@10, reciprocal rank, evaluation, and paired bootstrap.
- `trec_mlm/runio.py`: TREC writing, validation, and ranking loading.
- `trec_mlm/dense_experiment.py`: behavior/equivalence reference only.
- `trec_mlm/label_experiment.py`: staging-directory plus `os.replace` pattern only.

### New tracked implementation files

- `trec_mlm/dense_aggregation_runtime.py`: pure-stdlib thread/bootstrap constants and checks.
- `trec_mlm/dense_aggregation_contract.py`: strict configs, source allowlist, environment/input provenance, and clustering-contract payload.
- `trec_mlm/dense_aggregation_hashing.py`: canonical/framed/logical hashes and streaming logical-array hashers.
- `trec_mlm/dense_aggregation.py`: pure float64 `P`, `L`, `D`, contributions, final scores, ranking, grid, and selection.
- `trec_mlm/dense_aggregation_clustering.py`: pinned K-means, canonical labels, target assignment, and inertia validation.
- `trec_mlm/dense_aggregation_cache.py`: append-only flat shards, exact fit keys, receipt writer, bounded reads, finalization, and corruption checks.
- `trec_mlm/dense_aggregation_feasibility.py`: qrel-free sampling, worker timings, repeat comparison, projection, RSS parsing, and gate.
- `trec_mlm/legacy_dense_control.py`: isolated fixed-MPNet `k=50` export and exact regression validation.
- `trec_mlm/dense_aggregation_experiment.py`: pair/singleton scheduling, nested selection, disk-backed matrices, prediction freeze, and exact fit accounting.
- `trec_mlm/dense_aggregation_artifacts.py`: atomic directory writer, strict manifest loader, immutable construction loader, and repeat comparator.
- `trec_mlm/dense_aggregation_evaluation.py`: post-freeze controls, ablations, diagnostics, deployment hint, provisional gate inputs, repeat comparison, and the sole final decision.
- `trec_mlm/dense_aggregation_cli.py`: strict lazy subcommands and private worker commands.
- `trec_mlm/dense_aggregation_entrypoint.py`: sets thread variables before importing the CLI/numerical stack.
- `configs/trec_2026_dense_aggregation_feasibility.json`: qrel-free official feasibility inputs.
- `configs/trec_2026_dense_aggregation.json`: generated only after the legacy lock.
- `scripts/export_fixed_mpnet_k50_control.py`: dedicated isolated legacy bootstrap.
- `scripts/build_dense_aggregation_report.py`: validated snapshot to tracked JSON/Markdown/HTML.

### Existing files to modify

- `pyproject.toml`, `uv.lock`: exact lightweight `dense-aggregation` extra and console entry point.
- `.agents/skills/inspect-query-results/SKILL.md`: document aggregation evidence support and immutable-root output rule.
- `.agents/skills/inspect-query-results/scripts/build_query_explorer.py`: preserve, join, validate, and render aggregation evidence.
- `tests/test_query_explorer_skill.py`: aggregation schema, joins, manifest, escaping, and output-location coverage.
- `README.md`: gated commands, stop behavior, artifact layout, exploratory claim scope, and final result after validation.

### New tests and fixtures

- `tests/test_dense_aggregation_runtime.py`
- `tests/test_dense_aggregation_hashing.py`
- `tests/test_dense_aggregation_contract.py`
- `tests/test_dense_aggregation.py`
- `tests/test_dense_aggregation_clustering.py`
- `tests/test_dense_aggregation_cache.py`
- `tests/test_dense_aggregation_feasibility.py`
- `tests/test_legacy_dense_control.py`
- `tests/test_dense_aggregation_experiment.py`
- `tests/test_dense_aggregation_artifacts.py`
- `tests/test_dense_aggregation_evaluation.py`
- `tests/test_dense_aggregation_cli.py`
- `tests/test_dense_aggregation_report.py`
- `tests/fixtures/dense_aggregation/hash_golden_vectors.json`
- `tests/fixtures/dense_aggregation/sampling_numpy_2_5_1.json`
- `tests/fixtures/dense_aggregation/time_v.txt`

### Local and tracked run products

- Ignored local binary roots: `artifacts/dense_aggregation_feasibility/`, `artifacts/dense_aggregation_legacy_control/`, `artifacts/dense_aggregation_run_a/`, `artifacts/dense_aggregation_run_b/`, and `artifacts/dense_aggregation_repeat_validation/`.
- Tracked compact products: `reports/trec_2026_dense_aggregation_advisor_receipt.json`, `reports/trec_2026_dense_aggregation_feasibility.json`, `reports/trec_2026_dense_aggregation_legacy_control.json`, create-exclusive `reports/trec_2026_dense_aggregation_terminal_decision.json` only on a terminal post-qrel integrity failure, and the successful atomic manifested directory `reports/trec_2026_dense_aggregation/` containing `results.json`, `decision.json`, `repeatability.json`, `report.md`, `report.html`, and `manifest.json`.

## Protocol Traceability

| Locked protocol section | Implemented/tested in |
| --- | --- |
| Motivation, scope, frozen references, external-automatic status | Tasks 3, 10A-10B, 11, 17 |
| Frozen inputs and legacy representation hashes | Tasks 3, 12, 13 |
| New framed hashes and manifest sole exclusion | Task 2; mutation/repeat validation in Tasks 9A-10B |
| Pre-effectiveness legacy-control lock | Tasks 8 and 14 |
| Exact `P`, `L`, `D`, weighted contributions, `F`, and tie ranking | Task 4; artifact rederivation in Task 10A |
| Pinned K-means, canonical clusters, assignments, inertia | Tasks 1, 3, 5 |
| Frozen grid/order and effective uniqueness | Tasks 4, 9A-10A |
| Strict nested evaluation and fit deduplication | Tasks 6 and 9A-9B |
| Full-development deployment hint after primary freeze | Task 10A |
| Qrel-free runtime/RSS gate and nonautomatic fallback | Tasks 7 and 13 |
| Fixed controls and selected descriptive ablations | Tasks 8 and 10A |
| Atomic artifacts, complete receipts, two clean repeats | Tasks 2, 6, 9B, 10A-10B, 15, 16 |
| Bootstrap, W/T/L, zeros, density slices, support/neighbor diagnostics | Task 10A |
| Leakage and integrity invariants | Tasks 3, 8, 9A-10B, 12 |
| Promotion/stop decision | Tasks 10B and 16 |
| Required pre-run tests | Tasks 1-12 |
| Exploratory claim scope and deliverables | Tasks 11 and 17 |

---

### Task 1: Pin the Runtime and Create the Pre-Import Bootstrap

**Files:**

- Create: `trec_mlm/dense_aggregation_runtime.py`
- Create: `trec_mlm/dense_aggregation_entrypoint.py`
- Create: `tests/test_dense_aggregation_runtime.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Required interfaces:**

```python
THREAD_ENVIRONMENT: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

def pin_single_thread_environment() -> None: ...
def validate_preimport_environment() -> dict[str, str]: ...
```

- [ ] Write subprocess tests that clear all six variables, import the dedicated entry point, and assert each is set to string `"1"` before `numpy` appears in `sys.modules` at CLI import time.
- [ ] Add a test that rejects a pre-existing non-`1` value instead of silently overwriting it. This prevents an authoritative run from claiming a different environment.

```python
def test_entrypoint_pins_threads_before_cli_import() -> None:
    code = r'''
import json, os, sys, types
names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
         "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for name in names:
    os.environ.pop(name, None)
fake = types.ModuleType("trec_mlm.dense_aggregation_cli")
fake.main = lambda argv=None: 0
sys.modules["trec_mlm.dense_aggregation_cli"] = fake
from trec_mlm.dense_aggregation_entrypoint import main
assert "numpy" not in sys.modules
assert main([]) == 0
print(json.dumps({name: os.environ[name] for name in names}, sort_keys=True))
'''
    completed = subprocess.run([sys.executable, "-c", code], check=True,
                               text=True, capture_output=True)
    assert set(json.loads(completed.stdout).values()) == {"1"}

def test_runtime_rejects_conflicting_thread_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS must be unset or 1"):
        pin_single_thread_environment()
```
- [ ] Run the new test and confirm the expected import failure:

```bash
uv run --python 3.13.9 --with pytest pytest -q tests/test_dense_aggregation_runtime.py
```

Expected: FAIL because the runtime and entry-point modules do not exist.

- [ ] Insert the `dense-aggregation` key into the existing `[project.optional-dependencies]` table and the console-script key into the existing `[project.scripts]` table. Do not redeclare either TOML table. The resulting edited portions are:

```toml
[project.optional-dependencies]
dense = [
    "sentence-transformers>=5.1.0,<6",
    "torch>=2.7.0",
]
dense-aggregation = [
    "numpy==2.5.1; python_version >= '3.12'",
    "scipy==1.18.0; python_version >= '3.12'",
    "scikit-learn==1.9.0",
    "threadpoolctl==3.6.0",
]
response = [
    "bm25s==0.3.9",
    "PyStemmer==3.1.0",
]

[project.scripts]
trec-mlm = "trec_mlm.cli:main"
trec-mlm-dense-aggregation = "trec_mlm.dense_aggregation_entrypoint:main"
```

- [ ] Implement the runtime and entry point with this exact pre-import control flow:

```python
# trec_mlm/dense_aggregation_runtime.py
from __future__ import annotations

import os
import sys

THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
NUMERICAL_MODULE_PREFIXES = ("numpy", "scipy", "sklearn", "threadpoolctl")

def pin_single_thread_environment() -> None:
    for name in THREAD_ENVIRONMENT:
        current = os.environ.get(name)
        if current not in (None, "1"):
            raise RuntimeError(f"{name} must be unset or 1, got {current!r}")
        os.environ[name] = "1"

def validate_preimport_environment() -> dict[str, str]:
    imported = sorted(
        name for name in sys.modules
        if name.split(".", 1)[0] in NUMERICAL_MODULE_PREFIXES
    )
    if imported:
        raise RuntimeError(f"numerical modules imported before bootstrap: {imported!r}")
    values = {name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT}
    if set(values.values()) != {"1"}:
        raise RuntimeError(f"single-thread environment is not pinned: {values!r}")
    return values

# trec_mlm/dense_aggregation_entrypoint.py
from __future__ import annotations

from collections.abc import Sequence

from trec_mlm.dense_aggregation_runtime import (
    pin_single_thread_environment,
    validate_preimport_environment,
)

def main(argv: Sequence[str] | None = None) -> int:
    pin_single_thread_environment()
    validate_preimport_environment()
    from trec_mlm.dense_aggregation_cli import main as cli_main
    return cli_main(argv)
```
- [ ] Regenerate the lock with `uv lock` and verify the locked environment exactly:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation python -c 'import sys,numpy,scipy,sklearn,threadpoolctl; print(sys.version.split()[0], numpy.__version__, scipy.__version__, sklearn.__version__, threadpoolctl.__version__)'
```

Expected single line: `3.13.9 2.5.1 1.18.0 1.9.0 3.6.0`.

- [ ] Run `tests/test_dense_aggregation_runtime.py` again; expected: PASS.
- [ ] Commit:

```bash
git add pyproject.toml uv.lock trec_mlm/dense_aggregation_runtime.py trec_mlm/dense_aggregation_entrypoint.py tests/test_dense_aggregation_runtime.py
git commit -m "build: pin dense aggregation runtime"
```

### Task 2: Implement Canonical Hashes and Atomic Manifests

**Files:**

- Create: `trec_mlm/dense_aggregation_hashing.py`
- Create: `trec_mlm/dense_aggregation_artifacts.py`
- Create: `tests/test_dense_aggregation_hashing.py`
- Create: `tests/test_dense_aggregation_artifacts.py`
- Create: `tests/fixtures/dense_aggregation/hash_golden_vectors.json`

**Required interfaces:**

```python
def canonical_json(value: object) -> bytes: ...
def framed_sha256(tag: str, *payloads: bytes) -> str: ...
def logical_array_sha256(
    array: NDArray[Any],
    *,
    tag: str,
    dtype: str,
    row_inventory: Sequence[str | int],
    column_inventory: Sequence[str | int],
) -> str: ...
def support_query_ids_sha256(query_ids: Sequence[str]) -> str: ...
def raw_file_sha256(path: str | Path) -> str: ...
def canonical_membership_sha256(
    memberships: Sequence[tuple[int, Sequence[str]]],
) -> str: ...

class StreamingLogicalArrayHasher:
    def update_row(self, row: NDArray[Any]) -> None: ...
    def hexdigest(self) -> str: ...

@contextmanager
def atomic_artifact_directory(
    destination: str | Path, *, quarantine_on_error: bool = False,
) -> Iterator[Path]: ...
def rename_noreplace(source: str | Path, destination: str | Path) -> None: ...
def write_manifest(root: str | Path) -> dict[str, object]: ...
def validate_manifest(root: str | Path) -> dict[str, object]: ...
def semantic_artifact_sha256(
    root: str | Path, *, normalized_files: Mapping[str, bytes], tag: str,
) -> str: ...
```

- [ ] Add independent literal golden vectors covering payload boundaries, Unicode canonical JSON, NaN rejection, endian/layout conversion, all five required logical-array tags, support inventory, full/per-cluster membership, clustering contract, and aggregate inventory.
- [ ] Include these initial literal values in the fixture:

```json
{
  "framed_example_alpha_empty": "95acd93c0f69c911f11e2a3525c1d2d258d399d5364ec154710f1a6209aa0b7d",
  "support_q1_q2": "29e28ebde61c6fe7c435c73b32eb4ee67d2d7541d52f2b748259f52a7a755984",
  "inner_score_2x2": "4af6b8651a99d6de0e405712653e16bae490e61b0473977cb9726147b67d7fed",
  "membership_two_clusters": "15ffd8d6e8b96f332d2d610bc32951486bca46344149a8aeab4c4fd5ab8951df"
}
```

- [ ] Write tests proving a streamed logical hash equals the same full materialized array hash, converts a different numeric input dtype to the declared little-endian dtype, and rejects the wrong row count, row width, or non-finite float.
- [ ] Hash the same logical matrix from two different outer/config receipt contexts and require equal digests; outer/config IDs belong only in the naming receipt. Conversely, mutate dtype, shape, row inventory, column inventory, or one array byte and require a different digest.
- [ ] Write manifest tests for exact recursive inventory, literal file SHA-256, framed aggregate digest, `manifest.json` as the sole exclusion, symlink rejection, missing/extra/reordered/mutated file rejection, and destination-exists failure.
- [ ] Test `semantic_artifact_sha256`: it first validates the raw manifest, requires `normalized_files` keys to be an explicit subset of manifested regular files, replaces only those files' bytes in a second sorted `[path, semantic_sha256]` inventory, and frames it under the supplied fixed tag. Raw manifests remain the local tamper check; semantic digests are the stable cross-run references and never weaken raw validation.
- [ ] Test both failure policies: ordinary small/report writers remove unpublished staging, while `quarantine_on_error=True` atomically renames a nonempty failed staging tree to a unique `.aborted-<uuid>` sibling without touching a racing destination. Quarantined roots are explicitly invalid/non-reusable and are never accepted by `validate_manifest` as authoritative output.

```python
def test_inner_score_golden_and_streaming_equivalence() -> None:
    value = np.asarray([[1.0, -0.0], [2.5, 3.0]], dtype="<f8")
    expected = "4af6b8651a99d6de0e405712653e16bae490e61b0473977cb9726147b67d7fed"
    assert logical_array_sha256(
        value, tag="inner-validation-score", dtype="<f8",
        row_inventory=("q1", "q2"), column_inventory=("m1", "m2"),
    ) == expected
    streamed = StreamingLogicalArrayHasher(
        tag="inner-validation-score", dtype="<f8", shape=(2, 2),
        row_inventory=("q1", "q2"), column_inventory=("m1", "m2"),
    )
    streamed.update_row(value[0])
    streamed.update_row(value[1])
    assert streamed.hexdigest() == expected

def test_manifest_rejects_extra_file(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    with atomic_artifact_directory(root) as staging:
        (staging / "payload.txt").write_text("locked\n", encoding="utf-8")
    (root / "extra.txt").write_text("mutation\n", encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        validate_manifest(root)

def test_atomic_publish_never_replaces_racing_destination(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    with pytest.raises(OSError):
        with atomic_artifact_directory(root) as staging:
            (staging / "payload.txt").write_text("ours\n", encoding="utf-8")
            root.mkdir()
            (root / "sentinel.txt").write_text("theirs\n", encoding="utf-8")
    assert (root / "sentinel.txt").read_text(encoding="utf-8") == "theirs\n"
    assert not (root / "payload.txt").exists()
```
- [ ] Run:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation --with pytest pytest -q tests/test_dense_aggregation_hashing.py tests/test_dense_aggregation_artifacts.py
```

Expected: FAIL because the modules do not exist.

- [ ] Implement the protocol framing byte-for-byte with the following core. `StreamingLogicalArrayHasher` uses the same `_feed_payload` framing and predeclared final byte length:

```python
_HASH_PREFIX = b"trec-mlm-dense-aggregation-hash-v1\0"
_LOGICAL_TAGS = {
    "canonical-centers", "inner-validation-score", "inner-validation-rank",
    "full-development-score", "full-development-rank",
}

def canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

def _feed_payload(digest: Any, payload: bytes) -> None:
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)

def framed_sha256(tag: str, *payloads: bytes) -> str:
    tag_bytes = tag.encode("ascii")
    digest = hashlib.sha256()
    digest.update(_HASH_PREFIX)
    for payload in (tag_bytes, *payloads):
        _feed_payload(digest, payload)
    return digest.hexdigest()

def logical_array_sha256(
    array: NDArray[Any], *, tag: str, dtype: str,
    row_inventory: Sequence[str | int],
    column_inventory: Sequence[str | int],
) -> str:
    if tag not in _LOGICAL_TAGS:
        raise ValueError(f"unsupported logical-array tag: {tag}")
    declared = np.dtype(dtype)
    if declared.str[0] not in ("<", "|"):
        raise ValueError(f"dtype must be explicitly little-endian: {dtype}")
    value = np.ascontiguousarray(array, dtype=declared)
    if value.ndim != 2:
        raise ValueError("logical array must be two-dimensional")
    if row_inventory and value.shape[0] != len(row_inventory):
        raise ValueError("array row count does not match semantic inventory")
    if column_inventory and value.shape[1] != len(column_inventory):
        raise ValueError("array column count does not match semantic inventory")
    if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
        raise ValueError("logical float array contains a non-finite value")
    return framed_sha256(
        tag, value.dtype.str.encode("ascii"), canonical_json(list(value.shape)),
        canonical_json(list(row_inventory)), canonical_json(list(column_inventory)),
        value.tobytes(order="C"),
    )

class StreamingLogicalArrayHasher:
    def __init__(
        self, *, tag: str, dtype: str, shape: tuple[int, int],
        row_inventory: Sequence[str | int],
        column_inventory: Sequence[str | int],
    ) -> None:
        if tag not in _LOGICAL_TAGS:
            raise ValueError(f"unsupported logical-array tag: {tag}")
        self._dtype = np.dtype(dtype)
        if self._dtype.str[0] not in ("<", "|"):
            raise ValueError("streamed dtype must be explicitly little-endian")
        if shape[0] != len(row_inventory) or (
            column_inventory and shape[1] != len(column_inventory)
        ):
            raise ValueError("streamed shape does not match semantic inventories")
        self._shape = shape
        self._row_count = 0
        self._byte_count = 0
        self._expected_bytes = math.prod(shape) * self._dtype.itemsize
        self._digest = hashlib.sha256()
        self._digest.update(_HASH_PREFIX)
        for payload in (
            tag.encode("ascii"), self._dtype.str.encode("ascii"),
            canonical_json(list(shape)), canonical_json(list(row_inventory)),
            canonical_json(list(column_inventory)),
        ):
            _feed_payload(self._digest, payload)
        self._digest.update(struct.pack("<Q", self._expected_bytes))

    def update_row(self, row: NDArray[Any]) -> None:
        if self._row_count >= self._shape[0]:
            raise ValueError("stream received too many rows")
        value = np.ascontiguousarray(row, dtype=self._dtype)
        if value.shape != (self._shape[1],):
            raise ValueError("stream row width mismatch")
        if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
            raise ValueError("stream row contains a non-finite value")
        payload = value.tobytes(order="C")
        self._digest.update(payload)
        self._row_count += 1
        self._byte_count += len(payload)

    def hexdigest(self) -> str:
        if self._row_count != self._shape[0] or self._byte_count != self._expected_bytes:
            raise ValueError("stream ended before the declared logical array was complete")
        return self._digest.hexdigest()
```

- [ ] Implement atomic publication and strict inventory with this order; helper `_inventory_rows()` must walk with `Path.rglob("*")`, reject every symlink/non-regular entry, exclude only root `manifest.json`, and sort POSIX relative paths:

```python
def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)

@contextmanager
def atomic_artifact_directory(
    destination: str | Path, *, quarantine_on_error: bool = False,
) -> Iterator[Path]:
    final = Path(destination)
    if final.exists():
        raise FileExistsError(f"artifact destination exists: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.", dir=final.parent))
    try:
        yield staging
        _fsync_regular_files(staging)
        write_manifest(staging)
        validate_manifest(staging)
        rename_noreplace(staging, final)
        _fsync_directory(final.parent)
    except BaseException:
        if quarantine_on_error and staging.exists():
            quarantine = final.parent / f"{final.name}.aborted-{uuid.uuid4().hex}"
            rename_noreplace(staging, quarantine)
            _fsync_directory(final.parent)
        else:
            shutil.rmtree(staging, ignore_errors=True)
        raise

def rename_noreplace(source: str | Path, destination: str | Path) -> None:
    source_path = Path(source)
    destination_path = Path(destination)
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("authoritative artifact publication requires renameat2")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100, os.fsencode(source_path), -100, os.fsencode(destination_path), 1
    )
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination_path)

def raw_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def _inventory_rows(root: Path) -> list[list[str | int]]:
    rows: list[list[str | int]] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"artifact symlink is forbidden: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"artifact entry is not a regular file: {path}")
        relative = path.relative_to(root).as_posix()
        if relative == "manifest.json":
            continue
        rows.append([relative, path.stat().st_size, raw_file_sha256(path)])
    return rows

def _write_json_fsync(path: Path, value: object) -> None:
    payload = canonical_json(value) + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())

def _fsync_regular_files(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"artifact symlink is forbidden: {path}")
        if path.is_file():
            with path.open("rb") as handle:
                os.fsync(handle.fileno())

def write_manifest(root: str | Path) -> dict[str, object]:
    artifact_root = Path(root)
    rows = _inventory_rows(artifact_root)
    payload = {
        "schema": "trec-mlm-dense-aggregation-manifest-v1",
        "files": rows,
        "artifact_inventory_sha256": framed_sha256(
            "artifact-inventory", canonical_json(rows)
        ),
    }
    _write_json_fsync(artifact_root / "manifest.json", payload)
    return payload
```
- [ ] Run the two test files again; expected: PASS.
- [ ] Commit:

```bash
git add trec_mlm/dense_aggregation_hashing.py trec_mlm/dense_aggregation_artifacts.py tests/test_dense_aggregation_hashing.py tests/test_dense_aggregation_artifacts.py tests/fixtures/dense_aggregation/hash_golden_vectors.json
git commit -m "feat: add aggregation artifact integrity primitives"
```

### Task 3: Validate Frozen Inputs and Build the Clustering Contract

**Files:**

- Create: `trec_mlm/dense_aggregation_contract.py`
- Create: `tests/test_dense_aggregation_contract.py`
- Create: `configs/trec_2026_dense_aggregation_feasibility.json`

**Required data types:**

```python
@dataclass(frozen=True, slots=True)
class FrozenDenseInputs:
    query_ids: tuple[str, ...]
    query_texts: tuple[str, ...]
    embeddings: NDArray[np.float32]
    similarities: NDArray[np.float32]
    query_input_sha256: str
    embedding_sha256: str
    similarity_sha256: str

@dataclass(frozen=True, slots=True)
class FeasibilityConfig:
    repo_root: Path  # derived by loader; not serialized
    source_path: Path  # derived by loader; not serialized
    source_sha256: str  # hash of the exact bytes parsed; not serialized
    query_path: Path
    embedding_path: Path
    encoder_metadata_path: Path
    protocol_path: Path

def load_feasibility_config(path: str | Path) -> FeasibilityConfig: ...
def load_and_validate_frozen_dense_inputs(config: FeasibilityConfig) -> FrozenDenseInputs: ...
def build_clustering_contract(
    inputs: FrozenDenseInputs, *, repo_root: Path, protocol_path: Path,
) -> tuple[dict[str, object], str]: ...
def contract_source_inventory_sha256(
    contract: Mapping[str, object], *,
    category: Literal["implementation", "dependency"],
) -> str: ...
def runtime_source_inventory_sha256(repo_root: Path) -> str: ...
```

- [ ] Write tests for exact config keys and unknown-key rejection. Assert no field name or serialized key contains `qrel` or `grade`.
- [ ] Test exact `Q` ordering/count/hash, `(322, 768)` shape, normalized finite C-contiguous float32 rows, legacy query-input hash, embedding hash, and recomputed cosine hash.
- [ ] Add an AST/import-graph test proving `dense_aggregation_contract.py` imports neither `dense_query_knn` nor `query_knn`. Its qrel-free legacy array hash and cosine helpers are local byte-for-byte copies of the two pure algorithms, guarded by the frozen source hash in dependency provenance and fixed golden hashes below.
- [ ] Test metadata fields: model revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`, empty prefix, mean pooling, dimension 768, normalized float32.
- [ ] Test the clustering-contract map has exactly the protocol's top-level keys and exact runtime, thread, K-means, center, target-distance, inertia, and convergence values.
- [ ] Define and test one explicit `IMPLEMENTATION_FILES` tuple containing the Python implementation files that exist after Task 3. Every later task updates the tuple in the same commit that creates a Python implementation file. Task 12 compares the final tuple with the complete implementation map and proves all paths exist and are unique.

```python
def test_feasibility_config_has_no_qrel_or_grade_key(tmp_path: Path) -> None:
    payload = json.loads(
        Path("configs/trec_2026_dense_aggregation_feasibility.json")
        .read_text(encoding="utf-8")
    )
    serialized_keys = " ".join(payload).lower()
    assert "qrel" not in serialized_keys
    assert "grade" not in serialized_keys
    payload["qrel_path"] = "forbidden"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="extra=.*qrel_path"):
        load_feasibility_config(bad)

def test_clustering_contract_has_exact_top_level_keys(valid_inputs: FrozenDenseInputs) -> None:
    repo_root = Path.cwd().resolve()
    payload, digest = build_clustering_contract(
        valid_inputs, repo_root=repo_root,
        protocol_path=repo_root / "protocols/trec_2026_dense_aggregation.md",
    )
    assert set(payload) == {
        "schema", "protocol_file_sha256", "implementation_files",
        "embedding_sha256", "similarity_sha256", "versions", "threads",
        "kmeans", "center", "target_distance", "inertia", "convergence",
    }
    assert payload["schema"] == "trec-mlm-dense-aggregation-clustering-v1"
    assert payload["versions"] == {
        "python": "3.13.9", "numpy": "2.5.1", "scipy": "1.18.0",
        "scikit_learn": "1.9.0",
    }
    assert digest == framed_sha256("clustering-contract", canonical_json(payload))
```
- [ ] Run the tests; expected: FAIL because the contract module/config are absent.
- [ ] Implement the strict config/input path with exact-key comparison and legacy hash validation:

```python
_QUERY_FILE_SHA256 = "862ff85381b8055cd927ada25c56666594ab1ec3c21e182beb20c58f6c637f3c"
_QUERY_INPUT_SHA256 = "98faa04f979e428933c3d8e67dac43cf4dded112bd2c16969b3482a1d52886d9"
_EMBEDDING_SHA256 = "6d197055eb62ab9809ae321b0f60072e20f17524861e6a989b49cd35c389055b"
_SIMILARITY_SHA256 = "80c69a8dca29cd5e9a4de08dc4e44553fabf813428dd4956f7ecbeb2852968ed"
_FEASIBILITY_KEYS = {
    "schema", "query_path", "embedding_path", "encoder_metadata_path",
    "protocol_path",
}

def load_feasibility_config(path: str | Path) -> FeasibilityConfig:
    source = Path(path)
    source_bytes = source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    payload = json.loads(source_bytes)
    if set(payload) != _FEASIBILITY_KEYS:
        raise ValueError(
            f"feasibility config keys differ: "
            f"missing={sorted(_FEASIBILITY_KEYS - set(payload))!r}, "
            f"extra={sorted(set(payload) - _FEASIBILITY_KEYS)!r}"
        )
    if payload["schema"] != "trec-mlm-dense-aggregation-feasibility-config-v1":
        raise ValueError("unsupported feasibility config schema")
    base = source.resolve().parent.parent
    resolved = {
        name: (base / payload[name]).resolve()
        for name in _FEASIBILITY_KEYS - {"schema"}
    }
    return FeasibilityConfig(
        repo_root=base, source_path=source.resolve(strict=True),
        source_sha256=source_sha256, **resolved
    )

def _legacy_query_input_sha256(
    query_ids: Sequence[str], query_texts: Sequence[str]
) -> str:
    digest = hashlib.sha256()
    for query_id, query_text in zip(query_ids, query_texts, strict=True):
        digest.update(query_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(query_text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()

def _legacy_array_sha256_qrel_free(array: NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()

def _cosine_similarity_matrix_qrel_free(
    embeddings: NDArray[Any],
) -> NDArray[np.float32]:
    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("embeddings must be a nonempty two-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = np.zeros_like(array, dtype=np.float32)
    np.divide(array, norms, out=normalized, where=norms > 0.0)
    similarities = np.asarray(normalized @ normalized.T, dtype=np.float32)
    np.clip(similarities, -1.0, 1.0, out=similarities)
    return similarities

def _load_queries_tsv_qrel_free(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t", maxsplit=1)
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected query_id<TAB>query_text")
            query_id, query_text = fields[0].strip(), fields[1].strip()
            if not query_id or not query_text:
                raise ValueError(f"{path}:{line_number}: blank query ID or text")
            if query_id in queries:
                raise ValueError(f"{path}:{line_number}: duplicate query ID {query_id!r}")
            queries[query_id] = query_text
    if not queries:
        raise ValueError(f"{path}: query file is empty")
    return queries

def _require_exact_mpnet_metadata(metadata: Mapping[str, object]) -> None:
    required_metadata = {
        "key": "mpnet_base_v2",
        "model_id": "sentence-transformers/all-mpnet-base-v2",
        "license": "Apache-2.0",
        "query_prefix": "",
        "requested_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        "resolved_revision": "e8c3b32edf5434bc2275fc9bab85f82640a19130",
        "pooling": "mean",
        "embedding_dimension": 768,
        "dtype": "float32",
        "normalized": True,
        "query_count": 322,
        "input_sha256": _QUERY_INPUT_SHA256,
        "embedding_sha256": _EMBEDDING_SHA256,
        "similarity_sha256": _SIMILARITY_SHA256,
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in required_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"frozen MPNet metadata mismatch: {mismatches!r}")

def load_and_validate_frozen_dense_inputs(
    config: FeasibilityConfig,
) -> FrozenDenseInputs:
    if raw_file_sha256(config.query_path) != _QUERY_FILE_SHA256:
        raise ValueError("development query file hash mismatch")
    query_map = _load_queries_tsv_qrel_free(config.query_path)
    query_ids = tuple(sorted(query_map))
    query_texts = tuple(query_map[query_id] for query_id in query_ids)
    if len(query_ids) != 322:
        raise ValueError(f"expected 322 canonical queries, got {len(query_ids)}")
    embeddings = np.load(config.embedding_path, allow_pickle=False)
    if embeddings.shape != (322, 768) or embeddings.dtype != np.dtype("float32"):
        raise ValueError("frozen MPNet array must have shape (322, 768) and float32 dtype")
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    if not np.isfinite(embeddings).all() or np.any(np.linalg.norm(embeddings, axis=1) == 0):
        raise ValueError("frozen MPNet rows must be finite and nonzero")
    if not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("frozen MPNet rows are not normalized")
    if _legacy_array_sha256_qrel_free(embeddings) != _EMBEDDING_SHA256:
        raise ValueError("frozen MPNet embedding logical hash mismatch")
    similarities = _cosine_similarity_matrix_qrel_free(embeddings)
    if _legacy_array_sha256_qrel_free(similarities) != _SIMILARITY_SHA256:
        raise ValueError("frozen MPNet cosine logical hash mismatch")
    metadata = json.loads(config.encoder_metadata_path.read_text(encoding="utf-8"))["mpnet_base_v2"]
    _require_exact_mpnet_metadata(metadata)
    input_hash = _legacy_query_input_sha256(query_ids, query_texts)
    if input_hash != _QUERY_INPUT_SHA256:
        raise ValueError("frozen MPNet query-input logical hash mismatch")
    return FrozenDenseInputs(
        query_ids, query_texts, embeddings, similarities,
        input_hash, _EMBEDDING_SHA256, _SIMILARITY_SHA256,
    )

def build_clustering_contract(
    inputs: FrozenDenseInputs, *, repo_root: Path, protocol_path: Path,
) -> tuple[dict[str, object], str]:
    root = repo_root.resolve(strict=True)
    expected_protocol = (root / "protocols/trec_2026_dense_aggregation.md").resolve(
        strict=True
    )
    resolved_protocol = protocol_path.resolve(strict=True)
    if resolved_protocol != expected_protocol:
        raise ValueError(
            f"protocol path mismatch: {resolved_protocol} != {expected_protocol}"
        )
    protocol_sha256 = raw_file_sha256(resolved_protocol)
    if protocol_sha256 != "bbe6b50fd3d20efe3213a446f6665cbe0527114f34bf5c1e90236af4f998ad49":
        raise ValueError("locked dense aggregation protocol hash mismatch")
    if raw_file_sha256(root / "trec_mlm/dense_query_knn.py") != (
        "90f8db3abbb2887d8513fb881c631a3843319c29da02e935e1ecdb6c3c85e248"
    ):
        raise ValueError("locked legacy dense scorer source hash mismatch")
    implementation_rows = [
        [relative_path, raw_file_sha256(root / relative_path)]
        for relative_path in sorted((
            *IMPLEMENTATION_FILES,
            *DEPENDENCY_SOURCE_FILES,
        ))
    ]
    payload: dict[str, object] = {
        "schema": "trec-mlm-dense-aggregation-clustering-v1",
        "protocol_file_sha256": protocol_sha256,
        "implementation_files": implementation_rows,
        "embedding_sha256": inputs.embedding_sha256,
        "similarity_sha256": inputs.similarity_sha256,
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "threads": {
            **{name: os.environ[name] for name in THREAD_ENVIRONMENT},
            "threadpool_limit": 1, "process_count": 1,
        },
        "kmeans": {
            "init": "k-means++", "random_state": 2026, "n_init": 10,
            "algorithm": "lloyd", "max_iter": 300, "tol": 1e-4,
        },
        "center": {
            "dtype": "<f4", "layout": "C",
            "canonical_order": "minimum_member_query_id_lexical",
        },
        "target_distance": {
            "dtype": "<f8",
            "reduction": "numpy.sum(delta[z] * delta[z], dtype=numpy.float64)",
            "tie": "minimum_canonical_cluster_id",
        },
        "inertia": {
            "dtype": "<f8",
            "cluster_reduction": "numpy.sum(member_delta * member_delta, dtype=numpy.float64)",
            "total_reduction": "math.fsum(cluster_inertia)",
            "rtol": 1e-5, "atol": 1e-8,
        },
        "convergence": {
            "n_iter_min": 1, "n_iter_max": 299,
            "warning_action": "reject",
        },
    }
    _require_exact_runtime(payload["versions"], payload["threads"])
    return payload, framed_sha256("clustering-contract", canonical_json(payload))
```

`DEPENDENCY_SOURCE_FILES` is exactly `("trec_mlm/dense_query_knn.py", "trec_mlm/metrics.py", "trec_mlm/query_knn.py", "trec_mlm/runio.py")`; the first row must also equal the independently locked legacy-source hash. The protocol's single `implementation_files` contract field contains the sorted union of `IMPLEMENTATION_FILES` and `DEPENDENCY_SOURCE_FILES`, all and only executable Python sources. `RUNTIME_PROVENANCE_FILES` is exactly `("pyproject.toml", "uv.lock")` and is deliberately outside that clustering-contract field; artifacts hash it separately.

`contract_source_inventory_sha256` selects rows by exact membership in the named constant tuple, requires each path exactly once and no category overlap, then returns `framed_sha256(f"{category}-source-inventory", canonical_json(selected_rows))`. `runtime_source_inventory_sha256` builds sorted `[path, raw_sha256]` rows for the two runtime files and frames them with tag `runtime-source-inventory`. Tests require all three tuples to be sorted internally, pairwise disjoint, complete, unique, repository-relative, free of `..`, and regular files under `repo_root`; mutate one file in each category and prove only the correct category digest changes, while any Python-source change also changes the clustering contract. `load_feasibility_config` parses one captured byte string and retains its raw hash, `repo_root`, and exact `source_path`; it requires every resolved input to remain under that root and does not build a contract before inputs exist. `run_feasibility_worker` first requires `raw_file_sha256(config.source_path) == config.source_sha256`, records the captured hash, loads/validates the frozen inputs, then calls `build_clustering_contract(inputs, repo_root=config.repo_root, protocol_path=config.protocol_path)`.

`_require_exact_runtime` compares the version map to the four exact strings in the task test, requires every thread environment value to be string `"1"`, and requires integer limits/process count of 1. The test defines `_require_exact_mpnet_metadata`'s accepted map literally: model ID, revision/requested/resolved revision, license, empty prefix, mean pooling, dimension 768, dtype `float32`, normalized true, query count 322, and the three logical hashes shown above. No other metadata value can relax one of those fields.
- [ ] Add this strict qrel-free config:

```json
{
  "schema": "trec-mlm-dense-aggregation-feasibility-config-v1",
  "query_path": "data/dev/llm_dev_data.tsv",
  "embedding_path": "artifacts/frozen_dense_input/mpnet_base_v2.npy",
  "encoder_metadata_path": "artifacts/frozen_dense_input/encoder_metadata.json",
  "protocol_path": "protocols/trec_2026_dense_aggregation.md"
}
```

- [ ] Run the tests again with a temporary synthetic bundle; expected: PASS. Do not yet copy or open the official qrels.
- [ ] Commit:

```bash
git add trec_mlm/dense_aggregation_contract.py tests/test_dense_aggregation_contract.py configs/trec_2026_dense_aggregation_feasibility.json
git commit -m "feat: validate frozen aggregation inputs"
```

### Task 4: Implement the Pure Float64 Aggregation Formula and Grid

**Files:**

- Create: `trec_mlm/dense_aggregation.py`
- Create: `tests/test_dense_aggregation.py`
- Modify: `trec_mlm/dense_aggregation_contract.py`

**Required interfaces:**

```python
@dataclass(frozen=True, slots=True)
class DenseAggregationCandidate:
    clusters: int
    neighbors: int
    local_weight: float
    prior_shrinkage: float

DENSE_AGGREGATION_GRID: tuple[DenseAggregationCandidate, ...]

@dataclass(frozen=True, slots=True)
class LocalEvidence:
    scores: NDArray[np.float64]
    neighbor_indices: tuple[int, ...]
    neighbor_similarities: tuple[float, ...]
    mass: float

@dataclass(frozen=True, slots=True)
class FoldEvidence:
    prior: NDArray[np.float64]
    local_by_k: Mapping[int, LocalEvidence]

@dataclass(frozen=True, slots=True)
class AggregationPrediction:
    local: NDArray[np.float64]
    domain: NDArray[np.float64]
    prior: NDArray[np.float64]
    local_contribution: NDArray[np.float64]
    domain_contribution: NDArray[np.float64]
    prior_contribution: NDArray[np.float64]
    final_scores: NDArray[np.float64]
    model_order: tuple[int, ...]
    ranks: NDArray[np.uint16]

def fold_global_prior(*, grades: NDArray[Any], support_indices: Sequence[int]) -> NDArray[np.float64]: ...
def local_evidence(
    *, target_index: int, support_indices: Sequence[int],
    similarities: NDArray[Any], grades: NDArray[Any], query_ids: Sequence[str],
    k: int, prior: NDArray[Any],
) -> LocalEvidence: ...
def build_fold_evidence(
    *, target_index: int, support_indices: Sequence[int],
    similarities: NDArray[Any], grades: NDArray[Any], query_ids: Sequence[str],
    neighbor_counts: Sequence[int],
) -> FoldEvidence: ...
def domain_scores(*, member_indices: Sequence[int], grades: NDArray[Any]) -> NDArray[np.float64]: ...
def combine_and_rank(
    *, local: NDArray[Any], domain: NDArray[Any], prior: NDArray[Any],
    model_ids: Sequence[str], candidate: DenseAggregationCandidate,
) -> AggregationPrediction: ...
def select_inner_candidate(ndcg_by_candidate: Sequence[float]) -> int: ...
```

- [ ] Write a hand-calculated test with support grades `[[2,0,1],[0,2,1]]`, similarities `[0.75,0.25]`, a one-member target domain containing the first support row, `lambda=0.5`, and `beta=0.1`. Require:

```text
P = [1.0, 1.0, 1.0]
L = [1.5, 0.5, 1.0]
D = [2.0, 0.0, 1.0]
local contribution  = [0.675, 0.225, 0.45]
domain contribution = [0.9,   0.0,   0.45]
prior contribution  = [0.1,   0.1,   0.1]
F = [1.675, 0.325, 1.0]
```

```python
def test_hand_calculated_components_and_ranking() -> None:
    grades = np.asarray([[2, 0, 1], [0, 2, 1], [0, 0, 0]], dtype=np.float64)
    prior = fold_global_prior(grades=grades, support_indices=(0, 1))
    local = local_evidence(
        target_index=2, support_indices=(0, 1),
        similarities=np.asarray([[1, 0, .75], [0, 1, .25], [.75, .25, 1]], dtype=np.float32),
        grades=grades, query_ids=("q1", "q2", "qt"), k=2, prior=prior,
    )
    domain = domain_scores(member_indices=(0,), grades=grades)
    candidate = DenseAggregationCandidate(20, 2, 0.5, 0.1)
    result = combine_and_rank(
        local=local.scores, domain=domain, prior=prior,
        model_ids=("m1", "m2", "m3"), candidate=candidate,
    )
    np.testing.assert_array_equal(prior, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(local.scores, [1.5, 0.5, 1.0])
    np.testing.assert_array_equal(domain, [2.0, 0.0, 1.0])
    np.testing.assert_allclose(result.local_contribution, [0.675, 0.225, 0.45], rtol=0, atol=0)
    np.testing.assert_allclose(result.domain_contribution, [0.9, 0.0, 0.45], rtol=0, atol=0)
    np.testing.assert_allclose(result.prior_contribution, [0.1, 0.1, 0.1], rtol=0, atol=0)
    np.testing.assert_allclose(result.final_scores, [1.675, 0.325, 1.0], rtol=0, atol=0)
    assert result.model_order == (0, 2, 1)
```

- [ ] Add tests for canonical query-ordered `math.fsum`, strict positive similarities, similarity-descending/query-ID-ascending neighbor order, no-positive fallback, `mass <= 0` fallback, exact component `float.hex()` identity, `[0,2]` bounds, and non-finite rejection.
- [ ] Add ranking tests for final score, then prior, then candidate ID; prove `beta=0` still uses prior as tie-break and candidate-ID renaming changes no numeric signal.
- [ ] Assert the exact grid order from nested loops `C=(20,40)`, `k=(20,50)`, `lambda=(0.5,0.75)`, `beta=(0.0,0.1)` and that an exact inner nDCG tie selects the lower grid index without deduplicating nominal candidates.
- [ ] Run:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation --with pytest pytest -q tests/test_dense_aggregation.py
```

Expected: FAIL because the module does not exist.

- [ ] Implement every per-candidate grade reduction with the exact scalar reduction/ranking core below. Do not call `rank_target_from_similarity` or use NumPy matrix multiplication for experimental local evidence:

```python
DENSE_AGGREGATION_GRID = tuple(
    DenseAggregationCandidate(c, k, local_weight, prior_shrinkage)
    for c in (20, 40)
    for k in (20, 50)
    for local_weight in (0.5, 0.75)
    for prior_shrinkage in (0.0, 0.1)
)

def _validate_signal(value: NDArray[Any], name: str) -> NDArray[np.float64]:
    result = np.ascontiguousarray(value, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    if np.any(result < 0.0) or np.any(result > 2.0):
        raise ValueError(f"{name} lies outside [0,2]")
    return result

def _validate_grades(value: NDArray[Any]) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("grades must be a finite two-dimensional matrix")
    if np.any(result < 0.0) or np.any(result > 2.0):
        raise ValueError("grades lie outside [0,2]")
    return result

def _ordered_mean(
    grades: NDArray[np.float64], row_indices: Sequence[int]
) -> NDArray[np.float64]:
    indices = tuple(row_indices)
    if not indices:
        raise ValueError("grade aggregation requires nonempty support")
    if indices != tuple(sorted(set(indices))):
        raise ValueError("grade support indices must be unique canonical order")
    result = np.empty(grades.shape[1], dtype=np.float64)
    for model_index in range(grades.shape[1]):
        result[model_index] = math.fsum(
            float(grades[row_index, model_index]) for row_index in indices
        ) / len(indices)
    return result

def fold_global_prior(
    *, grades: NDArray[Any], support_indices: Sequence[int]
) -> NDArray[np.float64]:
    matrix = _validate_grades(grades)
    return _validate_signal(_ordered_mean(matrix, tuple(support_indices)), "prior")

def local_evidence(
    *, target_index: int, support_indices: Sequence[int],
    similarities: NDArray[Any], grades: NDArray[Any], query_ids: Sequence[str],
    k: int, prior: NDArray[Any],
) -> LocalEvidence:
    matrix = _validate_grades(grades)
    if k < 1:
        raise ValueError("k must be positive")
    if len(query_ids) != np.asarray(similarities).shape[0]:
        raise ValueError("query inventory does not match similarity matrix")
    if matrix.shape[0] != len(query_ids):
        raise ValueError("query inventory does not match grade matrix")
    similarity_row = np.asarray(similarities[target_index], dtype=np.float32)
    if not np.isfinite(similarity_row).all():
        raise ValueError("target similarity row contains a non-finite value")
    positive = [index for index in support_indices if float(similarity_row[index]) > 0.0]
    positive.sort(key=lambda index: (-float(similarity_row[index]), query_ids[index]))
    neighbors = tuple(positive[:k])
    weights = tuple(float(similarity_row[index]) for index in neighbors)
    mass = math.fsum(weights)
    if not neighbors or mass <= 0.0:
        scores = np.asarray(prior, dtype=np.float64).copy()
    else:
        scores = np.empty(matrix.shape[1], dtype=np.float64)
        for model_index in range(matrix.shape[1]):
            scores[model_index] = math.fsum(
                weight * float(matrix[index, model_index])
                for index, weight in zip(neighbors, weights, strict=True)
            ) / mass
    return LocalEvidence(_validate_signal(scores, "local"), neighbors, weights, mass)

def build_fold_evidence(
    *, target_index: int, support_indices: Sequence[int],
    similarities: NDArray[Any], grades: NDArray[Any], query_ids: Sequence[str],
    neighbor_counts: Sequence[int],
) -> FoldEvidence:
    prior = fold_global_prior(grades=grades, support_indices=support_indices)
    local_by_k = {
        k: local_evidence(
            target_index=target_index, support_indices=support_indices,
            similarities=similarities, grades=grades, query_ids=query_ids,
            k=k, prior=prior,
        )
        for k in neighbor_counts
    }
    return FoldEvidence(prior=prior, local_by_k=local_by_k)

def domain_scores(
    *, member_indices: Sequence[int], grades: NDArray[Any]
) -> NDArray[np.float64]:
    matrix = _validate_grades(grades)
    return _validate_signal(_ordered_mean(matrix, tuple(member_indices)), "domain")

def combine_and_rank(
    *, local: NDArray[Any], domain: NDArray[Any], prior: NDArray[Any],
    model_ids: Sequence[str], candidate: DenseAggregationCandidate,
) -> AggregationPrediction:
    local64 = _validate_signal(np.asarray(local, dtype=np.float64), "local")
    domain64 = _validate_signal(np.asarray(domain, dtype=np.float64), "domain")
    prior64 = _validate_signal(np.asarray(prior, dtype=np.float64), "prior")
    local_part = (1.0 - candidate.prior_shrinkage) * candidate.local_weight * local64
    domain_part = (
        (1.0 - candidate.prior_shrinkage)
        * (1.0 - candidate.local_weight)
        * domain64
    )
    prior_part = candidate.prior_shrinkage * prior64
    final = np.asarray([
        math.fsum((float(local_part[i]), float(domain_part[i]), float(prior_part[i])))
        for i in range(len(model_ids))
    ], dtype=np.float64)
    _validate_signal(final, "final")
    order = tuple(sorted(
        range(len(model_ids)),
        key=lambda i: (-float(final[i]), -float(prior64[i]), model_ids[i]),
    ))
    ranks = np.empty(len(model_ids), dtype=np.uint16)
    for rank, model_index in enumerate(order, start=1):
        ranks[model_index] = rank
    return AggregationPrediction(
        local64, domain64, prior64, local_part, domain_part, prior_part,
        final, order, ranks,
    )

def select_inner_candidate(ndcg_by_candidate: Sequence[float]) -> int:
    if len(ndcg_by_candidate) != len(DENSE_AGGREGATION_GRID):
        raise ValueError("inner selector requires exactly 16 metrics")
    if not all(math.isfinite(value) for value in ndcg_by_candidate):
        raise ValueError("inner selector metric must be finite")
    return min(range(16), key=lambda index: (-ndcg_by_candidate[index], index))
```

- [ ] Run the test again; expected: PASS.
- [ ] Add `trec_mlm/dense_aggregation.py` to `IMPLEMENTATION_FILES` in the same change and run the contract completeness/hash test; expected: PASS.
- [ ] Commit:

```bash
git add trec_mlm/dense_aggregation.py tests/test_dense_aggregation.py trec_mlm/dense_aggregation_contract.py
git commit -m "feat: implement local domain prior scoring"
```

### Task 5: Implement Deterministic Canonical K-Means

**Files:**

- Create: `trec_mlm/dense_aggregation_clustering.py`
- Create: `tests/test_dense_aggregation_clustering.py`
- Modify: `trec_mlm/dense_aggregation_contract.py`

**Required interfaces:**

```python
@dataclass(frozen=True, slots=True)
class FitKey:
    support_query_ids: tuple[str, ...]
    cluster_count: int
    embedding_sha256: str
    clustering_contract_sha256: str

@dataclass(slots=True)
class CanonicalClusterFit:
    key: FitKey
    centers: NDArray[np.float32]
    labels: NDArray[np.uint16]
    memberships: tuple[tuple[str, ...], ...]
    fit_membership_sha256: str
    cluster_membership_sha256: tuple[str, ...]
    center_sha256: str
    cluster_inertia: tuple[float, ...]
    sklearn_inertia: float
    recomputed_total_inertia: float
    n_iter: int

@dataclass(frozen=True, slots=True)
class TargetAssignment:
    query_id: str
    canonical_cluster_id: int
    squared_distance: float
    member_indices: tuple[int, ...]
    cluster_support: int

@dataclass(frozen=True, slots=True)
class NumericalEnvironmentReceipt:
    thread_environment: Mapping[str, str]
    threadpool_info: tuple[Mapping[str, str | int | None], ...]
    threadpool_info_sha256: str
    schema: str = "trec-mlm-dense-aggregation-numerical-environment-v1"

def capture_numerical_environment() -> NumericalEnvironmentReceipt: ...
def require_same_numerical_environment(
    expected: NumericalEnvironmentReceipt,
) -> None: ...
def numerical_environment_file_bytes(
    receipt: NumericalEnvironmentReceipt,
) -> bytes: ...

def fit_canonical_clusters(
    *, embeddings: NDArray[np.float32], query_ids: Sequence[str],
    support_indices: Sequence[int], cluster_count: int,
    embedding_sha256: str, clustering_contract_sha256: str,
) -> CanonicalClusterFit: ...
def assign_targets(
    *, fit: CanonicalClusterFit, embeddings: NDArray[np.float32],
    query_ids: Sequence[str], target_indices: Sequence[int],
    support_index_by_id: Mapping[str, int],
) -> tuple[TargetAssignment, ...]: ...
```

- [ ] Write a fake-KMeans test that captures exact arguments: `init="k-means++"`, `random_state=2026`, `n_init=10`, `algorithm="lloyd"`, `max_iter=300`, `tol=1e-4`, inside `threadpool_limits(limits=1)`.
- [ ] `capture_numerical_environment` runs inside `threadpool_limits(limits=1)`, requires all six environment variables to equal `"1"`, canonicalizes each `threadpool_info()` row to exactly `user_api`, `internal_api`, `prefix`, `filepath`, `version`, and integer `num_threads`, sorts rows by the first five fields, requires every `num_threads == 1`, and frames the canonical rows with tag `threadpool-info`. `require_same_numerical_environment` captures again and requires exact receipt equality.
- [ ] `numerical_environment_file_bytes` is exactly `canonical_json(asdict(receipt)) + b"\n"`. Across feasibility, construction, cache headers, provenance, and loaders, `numerical_environment_sha256` always means raw SHA-256 of those complete bytes; `threadpool_info_sha256` remains a separately validated inner field and is never substituted for the file hash.
- [ ] Feasibility and construction workers capture once at worker start, compare immediately before the first fit and immediately after the final fit, and persist the original canonical receipt as manifested `environment.json`. Tests simulate a start/first-fit and final-fit mutation and require failure. This check is worker-scoped rather than an expensive `threadpool_info()` call inside each of 104,006 fits.
- [ ] Add golden pinned-runtime tests for canonical relabeling by each cluster's minimum query ID, C-contiguous little-endian float32 centers, float64 target distance, exact-distance tie, query-input permutation invariance, per-cluster inertia, and `math.fsum` total inertia.
- [ ] Add rejection tests for warning, non-finite/zero row, fewer than `C` rows, fewer than `C` distinct float32 row byte strings, empty/unoccupied cluster, non-finite center, `n_iter_` outside `1..299`, and sklearn/recomputed inertia mismatch beyond `rtol=1e-5, atol=1e-8`.

```python
def test_exact_target_distance_tie_uses_lower_canonical_id() -> None:
    fit = CanonicalClusterFit(
        key=FitKey(("q1", "q2"), 2, "e" * 64, "c" * 64),
        centers=np.asarray([[-1.0, 0.0], [1.0, 0.0]], dtype="<f4"),
        labels=np.asarray([0, 1], dtype="<u2"),
        memberships=(("q1",), ("q2",)),
        fit_membership_sha256="f" * 64,
        cluster_membership_sha256=("1" * 64, "2" * 64),
        center_sha256="3" * 64, cluster_inertia=(0.0, 0.0),
        sklearn_inertia=0.0, recomputed_total_inertia=0.0, n_iter=1,
    )
    embeddings = np.asarray([[-1, 0], [1, 0], [0, 0]], dtype=np.float32)
    (assignment,) = assign_targets(
        fit=fit, embeddings=embeddings, query_ids=("q1", "q2", "qt"),
        target_indices=(2,), support_index_by_id={"q1": 0, "q2": 1},
    )
    assert assignment.canonical_cluster_id == 0
    assert assignment.squared_distance.hex() == float(1.0).hex()
```
- [ ] Run the clustering tests; expected: FAIL because the module does not exist.
- [ ] Implement K-means with this exact fit/canonicalization sequence. Tests provide a fake estimator for argument capture and the real pinned estimator for golden vectors:

```python
_KMEANS_ARGUMENTS = {
    "init": "k-means++", "random_state": 2026, "n_init": 10,
    "algorithm": "lloyd", "max_iter": 300, "tol": 1e-4,
}

def fit_canonical_clusters(
    *, embeddings: NDArray[np.float32], query_ids: Sequence[str],
    support_indices: Sequence[int], cluster_count: int,
    embedding_sha256: str, clustering_contract_sha256: str,
) -> CanonicalClusterFit:
    support = tuple(sorted(support_indices, key=lambda index: query_ids[index]))
    support_ids = tuple(query_ids[index] for index in support)
    rows = np.ascontiguousarray(embeddings[np.asarray(support)], dtype=np.float32)
    if rows.shape[0] < cluster_count:
        raise ValueError("cluster support has fewer rows than C")
    if not np.isfinite(rows).all() or np.any(np.linalg.norm(rows, axis=1) == 0.0):
        raise ValueError("cluster support contains a non-finite or zero row")
    if len({row.tobytes() for row in rows}) < cluster_count:
        raise ValueError("cluster support has fewer than C distinct float32 rows")
    estimator = KMeans(**_KMEANS_ARGUMENTS)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with threadpool_limits(limits=1):
            original_labels = np.asarray(estimator.fit_predict(rows), dtype=np.int64)
    if any(issubclass(item.category, ConvergenceWarning) for item in caught):
        raise RuntimeError("K-means emitted a convergence warning")
    if not 1 <= int(estimator.n_iter_) <= 299:
        raise RuntimeError(f"K-means n_iter_ outside 1..299: {estimator.n_iter_}")
    present = tuple(sorted(set(int(label) for label in original_labels)))
    if len(present) != cluster_count:
        raise RuntimeError("K-means did not occupy every cluster")
    original_members = {
        label: tuple(i for i, assigned in enumerate(original_labels) if assigned == label)
        for label in present
    }
    canonical_original = tuple(sorted(
        present, key=lambda label: min(support_ids[i] for i in original_members[label])
    ))
    remap = {original: canonical for canonical, original in enumerate(canonical_original)}
    labels = np.asarray([remap[int(label)] for label in original_labels], dtype=np.uint16)
    centers = np.ascontiguousarray(
        np.asarray(estimator.cluster_centers_, dtype=np.float32)[list(canonical_original)],
        dtype=np.dtype("<f4"),
    )
    if not np.isfinite(centers).all():
        raise RuntimeError("K-means produced a non-finite center")
    memberships = tuple(
        tuple(support_ids[i] for i, label in enumerate(labels) if int(label) == cluster_id)
        for cluster_id in range(cluster_count)
    )
    inertia_values: list[float] = []
    for cluster_id in range(cluster_count):
        member_delta = np.ascontiguousarray(
            rows[np.flatnonzero(labels == cluster_id)], dtype=np.float64
        ) - np.ascontiguousarray(centers[cluster_id], dtype=np.float64)
        inertia_values.append(float(np.sum(
            member_delta * member_delta, dtype=np.float64
        )))
    cluster_inertia = tuple(inertia_values)
    if not all(math.isfinite(value) and value >= 0.0 for value in cluster_inertia):
        raise RuntimeError("canonical cluster inertia is non-finite or negative")
    recomputed = math.fsum(cluster_inertia)
    if not math.isclose(float(estimator.inertia_), recomputed, rel_tol=1e-5, abs_tol=1e-8):
        raise RuntimeError("sklearn inertia differs from canonical recomputation")
    key = FitKey(support_ids, cluster_count, embedding_sha256, clustering_contract_sha256)
    return CanonicalClusterFit(
        key=key, centers=centers, labels=labels, memberships=memberships,
        fit_membership_sha256=canonical_membership_sha256(tuple(enumerate(memberships))),
        cluster_membership_sha256=tuple(
            canonical_membership_sha256(((cluster_id, memberships[cluster_id]),))
            for cluster_id in range(cluster_count)
        ),
        center_sha256=logical_array_sha256(
            centers, tag="canonical-centers", dtype="<f4",
            row_inventory=tuple(range(cluster_count)), column_inventory=(),
        ),
        cluster_inertia=cluster_inertia, sklearn_inertia=float(estimator.inertia_),
        recomputed_total_inertia=recomputed, n_iter=int(estimator.n_iter_),
    )

def assign_targets(
    *, fit: CanonicalClusterFit, embeddings: NDArray[np.float32],
    query_ids: Sequence[str], target_indices: Sequence[int],
    support_index_by_id: Mapping[str, int],
) -> tuple[TargetAssignment, ...]:
    centers = np.ascontiguousarray(fit.centers, dtype=np.float64)
    assignments: list[TargetAssignment] = []
    for target_index in target_indices:
        target = np.ascontiguousarray(embeddings[target_index], dtype=np.float64)
        if not np.isfinite(target).all():
            raise ValueError("target embedding contains a non-finite value")
        delta = centers - target
        distances = np.asarray([
            np.sum(delta[z] * delta[z], dtype=np.float64)
            for z in range(fit.key.cluster_count)
        ], dtype=np.float64)
        if not np.isfinite(distances).all():
            raise RuntimeError("target-to-center distance is non-finite")
        cluster_id = int(np.argmin(distances))
        member_indices = tuple(
            support_index_by_id[query_id]
            for query_id in fit.memberships[cluster_id]
        )
        assignments.append(TargetAssignment(
            query_id=query_ids[target_index], canonical_cluster_id=cluster_id,
            squared_distance=float(distances[cluster_id]),
            member_indices=member_indices, cluster_support=len(member_indices),
        ))
    return tuple(assignments)
```
- [ ] Run the clustering tests again; expected: PASS.
- [ ] Add this module path to `IMPLEMENTATION_FILES`; run the contract test.
- [ ] Commit:

```bash
git add trec_mlm/dense_aggregation_clustering.py tests/test_dense_aggregation_clustering.py trec_mlm/dense_aggregation_contract.py
git commit -m "feat: add deterministic aggregation clustering"
```

### Task 6: Build the Append-Only Fit Cache and Streaming Receipts

**Files:**

- Create: `trec_mlm/dense_aggregation_cache.py`
- Create: `tests/test_dense_aggregation_cache.py`
- Modify: `trec_mlm/dense_aggregation_artifacts.py`
- Modify: `tests/test_dense_aggregation_artifacts.py`
- Modify: `trec_mlm/dense_aggregation_contract.py`

**On-disk schema:**

```text
fit-cache/
  header.json
  centers.f4
  labels.u2
  cluster_inertia.f8
  fit_receipts.jsonl
  timings.jsonl
```

**Required interface:**

```python
@dataclass(frozen=True, slots=True)
class StoredFitRecord:
    fit: CanonicalClusterFit
    assignments: tuple[TargetAssignment, ...]
    exclusion_kind: Literal["pair", "single"]
    inventory_index: int
    excluded_query_ids: tuple[str, ...]

class StreamingFitCache:
    @classmethod
    def create(cls, root: str | Path, *, header: Mapping[str, object]) -> "StreamingFitCache": ...
    @classmethod
    def open_readonly(cls, root: str | Path) -> "StreamingFitCache": ...
    def append_fit(
        self, fit: CanonicalClusterFit,
        assignments: Sequence[TargetAssignment], *,
        exclusion_kind: Literal["pair", "single"], inventory_index: int,
        excluded_query_ids: Sequence[str],
    ) -> None: ...
    def load_fit(self, key: FitKey) -> StoredFitRecord | None: ...
    def load_by_inventory(
        self, *, exclusion_kind: Literal["pair", "single"],
        inventory_index: int, cluster_count: Literal[20, 40],
    ) -> StoredFitRecord: ...
    def release(self, key: FitKey) -> None: ...
    def append_timing(
        self, *, exclusion_kind: Literal["pair", "single"],
        inventory_index: int, cluster_count: Literal[20, 40],
        elapsed_ns: int, n_iter: int,
    ) -> None: ...
    def finalize(self) -> dict[str, object]: ...
    def finalize_into(self, destination: str | Path) -> dict[str, object]: ...

def preflight_atomic_roots(*, cache_dir: Path, output_dir: Path) -> None: ...
```

- [ ] Write tests that bind the cache key to the complete ordered support tuple, `C`, embedding hash, and clustering-contract hash; a digest collision cannot bypass full tuple comparison.
- [ ] Require an exact versioned cache-header schema containing canonical query IDs/hash, embedding/similarity/contract hashes, implementation/dependency/runtime/numerical-environment hashes, shard schemas, and expected inventory/counts. Reject unknown keys and forbid absolute paths, cache/output roots, process IDs, timestamps, or elapsed values so two authoritative cache headers are scientifically identical.
- [ ] Test binary offsets/shapes/dtypes, center/membership/inertia rehashing on read, JSONL canonical field order, truncated/mutated/extra-byte rejection, empty-root enforcement, duplicate-key rejection, and final `fsync`.
- [ ] Test bounded residency with weak references: after both held-out assignments and receipt flush, releasing a fit removes the last cache-held array reference.
- [ ] Test `preflight_atomic_roots` with two parents on the same device, simulated different `st_dev` values, an existing cache destination, and an existing output destination. It may create parent directories, but neither authoritative destination. It must fail before a cache, staging directory, or qrel file is opened.
- [ ] Test naive materialization versus cache round-trip for centers, labels, memberships, assignments, domain scores, final scores, and ranks using exact bytes/`float.hex()`.

```python
def test_fit_key_binds_order_c_embedding_and_contract() -> None:
    base = FitKey(("q1", "q2"), 20, "e" * 64, "c" * 64)
    mutations = (
        FitKey(("q2", "q1"), 20, "e" * 64, "c" * 64),
        FitKey(("q1", "q2"), 40, "e" * 64, "c" * 64),
        FitKey(("q1", "q2"), 20, "x" * 64, "c" * 64),
        FitKey(("q1", "q2"), 20, "e" * 64, "x" * 64),
    )
    assert len({_fit_key_sha256(base), *map(_fit_key_sha256, mutations)}) == 5

def test_cache_rejects_truncated_center_shard(populated_cache: Path) -> None:
    center_path = populated_cache / "centers.f4"
    center_path.write_bytes(center_path.read_bytes()[:-1])
    with pytest.raises(ValueError, match="center.*byte length"):
        StreamingFitCache.open_readonly(populated_cache)
```

`populated_cache` is a fixture that creates one two-cluster `CanonicalClusterFit` with two target assignments through the public `create/append_fit/finalize` path; it may not write shard bytes directly.
- [ ] Run the cache tests; expected: FAIL because the module does not exist.
- [ ] Implement flat concatenated shards using this canonical record builder. `_append_array()` writes exact C-order bytes, returns the pre-write byte offset/byte count, and leaves the shard open; `_append_jsonl()` writes `canonical_json(record) + b"\n"` and flushes the Python file object:

```python
def _fit_key_sha256(key: FitKey) -> str:
    return framed_sha256("fit-cache-key", canonical_json({
        "schema": "trec-mlm-dense-aggregation-fit-key-v1",
        "support_query_ids": list(key.support_query_ids),
        "cluster_count": key.cluster_count,
        "embedding_sha256": key.embedding_sha256,
        "clustering_contract_sha256": key.clustering_contract_sha256,
    }))

def append_fit(
    self, fit: CanonicalClusterFit,
    assignments: Sequence[TargetAssignment], *,
    exclusion_kind: Literal["pair", "single"], inventory_index: int,
    excluded_query_ids: Sequence[str],
) -> None:
    key_sha256 = _fit_key_sha256(fit.key)
    if key_sha256 in self._index:
        raise ValueError(f"duplicate fit key: {key_sha256}")
    center_offset, center_bytes = self._append_array(
        self._center_handle, np.ascontiguousarray(fit.centers, dtype="<f4")
    )
    label_offset, label_bytes = self._append_array(
        self._label_handle, np.ascontiguousarray(fit.labels, dtype="<u2")
    )
    inertia_offset, inertia_bytes = self._append_array(
        self._inertia_handle,
        np.ascontiguousarray(fit.cluster_inertia, dtype="<f8"),
    )
    record_without_hash = {
        "schema": "trec-mlm-dense-aggregation-fit-receipt-v1",
        "sequence": self._fit_count,
        "exclusion_kind": exclusion_kind,
        "inventory_index": inventory_index,
        "excluded_query_ids": list(excluded_query_ids),
        "cluster_count": fit.key.cluster_count,
        "fit_key_sha256": key_sha256,
        "support_query_ids_sha256": support_query_ids_sha256(fit.key.support_query_ids),
        "embedding_sha256": fit.key.embedding_sha256,
        "clustering_contract_sha256": fit.key.clustering_contract_sha256,
        "centers": {"offset": center_offset, "bytes": center_bytes,
                    "shape": list(fit.centers.shape), "dtype": "<f4"},
        "labels": {"offset": label_offset, "bytes": label_bytes,
                   "shape": list(fit.labels.shape), "dtype": "<u2"},
        "cluster_inertia": {"offset": inertia_offset, "bytes": inertia_bytes,
                            "shape": [len(fit.cluster_inertia)], "dtype": "<f8"},
        "center_sha256": fit.center_sha256,
        "fit_membership_sha256": fit.fit_membership_sha256,
        "cluster_membership_sha256": list(fit.cluster_membership_sha256),
        "cluster_inertia_hex": [value.hex() for value in fit.cluster_inertia],
        "sklearn_inertia_hex": fit.sklearn_inertia.hex(),
        "recomputed_total_inertia_hex": fit.recomputed_total_inertia.hex(),
        "n_iter": fit.n_iter,
        "assignments": [
            {"query_id": item.query_id,
             "canonical_cluster_id": item.canonical_cluster_id,
             "squared_distance_hex": item.squared_distance.hex(),
             "cluster_support": item.cluster_support}
            for item in assignments
        ],
        "fit_count_increment": 1,
        "start_count_increment": 10,
    }
    record = {**record_without_hash, "record_sha256": framed_sha256(
        "fit-cache-record", canonical_json(record_without_hash)
    )}
    self._append_jsonl(self._receipt_handle, record)
    self._index[key_sha256] = record
    self._fit_count += 1
```

- [ ] Reconstruct support IDs on load from canonical `Q` minus `excluded_query_ids` instead of duplicating 320/321 IDs in every receipt. Verify the reconstructed full tuple, support hash, fit key, offsets, exact byte lengths, center hash, memberships reconstructed from labels, both membership hash forms, inertia hex values, assignments, and record hash before returning a fit.
- [ ] Implement `preflight_atomic_roots` in `dense_aggregation_artifacts.py`: resolve both destinations, reject equality/ancestor overlap and any existing destination, create only the two parent directories, then compare `cache_dir.parent.stat().st_dev` with `output_dir.parent.stat().st_dev`. Raise `OSError("cache and output must share one filesystem")` on mismatch. Both the feasibility worker and authoritative construction worker call it as their first filesystem action.
- [ ] `finalize()` flushes and `fsync()`s all five data/JSONL files, closes every handle, validates their sizes/offsets/hashes against the in-memory index, writes and `fsync()`s `header.json` last among cache writes, `fsync()`s the cache directory, and only then reopens the now-complete cache through `open_readonly` for an independent full validation. `finalize_into(destination)` then requires a nonexistent destination on the same filesystem and uses the tested `rename_noreplace(self.root, destination)` primitive so the initially separate cache becomes the staged artifact's `fits/` subtree without copying 9 GiB or clobbering a racing destination.
- [ ] Run the cache tests again; expected: PASS.
- [ ] Add the module path to `IMPLEMENTATION_FILES`; run the contract test.
- [ ] Commit:

```bash
git add trec_mlm/dense_aggregation_cache.py tests/test_dense_aggregation_cache.py trec_mlm/dense_aggregation_artifacts.py tests/test_dense_aggregation_artifacts.py trec_mlm/dense_aggregation_contract.py
git commit -m "feat: add streaming aggregation fit cache"
```

### Task 7: Implement the Qrel-Free Feasibility Gate

**Files:**

- Create: `trec_mlm/dense_aggregation_feasibility.py`
- Create: `trec_mlm/dense_aggregation_cli.py`
- Create: `tests/test_dense_aggregation_feasibility.py`
- Create: `tests/test_dense_aggregation_cli.py`
- Create: `tests/fixtures/dense_aggregation/sampling_numpy_2_5_1.json`
- Create: `tests/fixtures/dense_aggregation/time_v.txt`
- Modify: `trec_mlm/dense_aggregation_contract.py`

**Required interfaces:**

```python
@dataclass(frozen=True, slots=True)
class TimingCategorySummary:
    fit_count: int
    mean_seconds: float
    median_seconds: float
    p95_seconds: float
    maximum_seconds: float
    p95_method: Literal["linear_index_(n-1)*0.95"]
    n_iter_min: int
    n_iter_mean: float
    n_iter_max: int
    n_iter_values_sha256: str

@dataclass(frozen=True, slots=True)
class FeasibilityReceipt:
    fit_count: int
    start_count: int
    pair_indices: tuple[int, ...]
    single_indices: tuple[int, ...]
    timing_summary: Mapping[str, TimingCategorySummary]
    fit_receipt_sha256: str
    provenance: Mapping[str, object]
    exclusive_resource_attested: bool
    schema: str = "trec-mlm-dense-aggregation-feasibility-worker-v1"

@dataclass(frozen=True, slots=True)
class FeasibilityGate:
    status: Literal["pass", "stop"]
    failures: tuple[str, ...]
    projection_seconds: float
    maximum_rss_bytes: int
    schema: str = "trec-mlm-dense-aggregation-feasibility-gate-v1"

FEASIBILITY_PROVENANCE_KEYS = {
    "config_sha256", "protocol_sha256", "clustering_contract_sha256",
    "implementation_inventory_sha256", "dependency_inventory_sha256",
    "runtime_inventory_sha256", "query_file_sha256", "query_input_sha256",
    "embedding_sha256", "similarity_sha256", "sample_fixture_sha256",
    "numerical_environment_sha256",
    "python_version", "numpy_version", "scipy_version",
    "scikit_learn_version", "thread_environment", "cache_root",
    "output_root", "worker_pid",
}

FEASIBILITY_REPEAT_CHECKS = (
    "receipt_schema", "config", "protocol", "clustering_contract",
    "implementation_inventory", "dependency_inventory", "runtime_inventory",
    "query_input", "embedding", "similarity", "sample_fixture",
    "sample_indices", "category_counts", "fit_count", "start_count",
    "fit_supports", "fit_memberships", "fit_centers", "target_assignments",
    "manifests", "fresh_distinct_roots", "finite_timings",
    "exclusive_resource_attestation",
)

def official_sample_indices() -> tuple[NDArray[np.int64], NDArray[np.int64]]: ...
def run_feasibility_worker(
    config: FeasibilityConfig, *, cache_dir: Path, output_dir: Path,
    exclusive_resource_attested: bool,
) -> FeasibilityReceipt: ...
def parse_gnu_time_verbose(path: str | Path) -> int: ...
def evaluate_feasibility_repeats(
    repeat_a: FeasibilityReceipt, repeat_b: FeasibilityReceipt, *,
    repeat_a_rss_bytes: int, repeat_b_rss_bytes: int,
    exclusive_resource_attested: bool,
) -> FeasibilityGate: ...
def build_parser() -> argparse.ArgumentParser: ...
def main(argv: Sequence[str] | None = None) -> int: ...
```

- [ ] Under an independent NumPy 2.5.1 one-liner, verify and then add this literal fixture content. The canonical JSON SHA-256 of the map containing these two arrays is `6b74301b9edb16438739a2c46b455ff37985698a6795bd7a4b9d1f5530cf9d39`:

```json
{
  "pair_indices": [92,213,660,663,1219,1358,1564,1690,1751,1785,1936,2009,3028,3081,3496,3587,4002,4005,4036,4105,4334,4444,4714,4792,4817,5065,5267,5427,5446,6072,6090,6629,6894,6957,7495,7520,7987,8091,8266,8505,8523,8900,9004,9123,9202,9715,9732,9808,10058,10643,10782,11648,11739,11970,12041,12632,13162,13210,13472,13937,13944,14301,14462,14527,14534,14685,14955,15346,15357,15382,15445,15649,15655,16164,16203,16208,16447,16467,16595,16812,16833,17248,17286,17435,17465,17770,18193,18255,18346,18470,18486,18579,18641,18796,18798,19055,19176,19595,20799,21033,21215,21455,21482,21602,21652,21766,21804,22038,22178,22287,22411,22881,23045,23073,23227,23471,23678,23850,23977,24032,24047,24219,24347,24355,24484,24584,25143,25360,25403,25500,25773,26507,26663,26772,26856,27062,27224,27821,28187,28260,28659,28696,29013,29067,29105,29241,29540,29747,29882,30095,30528,30624,30645,30939,31019,31031,31085,31180,31241,31265,31335,31779,32024,32025,32284,32449,32716,32910,32983,33095,33165,33581,33760,33869,34046,34077,34134,34148,34272,34533,34658,34820,35848,36102,36145,36234,37098,37452,37478,37644,37675,37875,38000,38338,38357,38730,38801,39554,39628,39926,40174,40627,40656,40661,40722,40928,40929,41062,41084,41130,42036,42267,42360,42498,42737,42746,43807,43887,43941,44101,44160,44260,44410,45017,45033,45530,45818,45952,46375,46559,46642,47325,47350,47395,47850,48153,48160,48320,48518,48623,48758,48794,48797,48850,49275,49430,49552,49670,49747,50114,50522,50934,50997,51027,51090,51326],
  "single_indices": [5,12,14,18,20,29,62,64,70,91,103,119,133,134,137,153,161,166,173,199,215,241,242,251,259,275,277,279,289,297,320,321]
}
```

- [ ] Test that the feasibility parser and config reject `--qrels`, `--grades`, unknown fields, alternate sampling, alternate counts, alternate thresholds, and all formula/grid overrides.
- [ ] Add a subprocess import-graph test proving the feasibility worker imports no qrel loader, qrel path, grade object, scoring module, legacy control, or evaluation module.
- [ ] Test exact pair inventory order from `itertools.combinations(Q, 2)`, 512 pair fits, 64 singleton fits, 576 total fits, and 5,760 starts per worker.
- [ ] Test the timed window begins immediately before support materialization and ends only after assignment, hashing, binary append, receipt flush, and reference deletion.
- [ ] Test mean, median, linear p95, max, and convergence iterations separately for pair/single and C20/C40.
- [ ] Test the exact projection, selection of the larger repeat, GNU `time -v` parse under `LC_ALL=C`, KiB-to-bytes factor 1024, inclusive pass boundaries at 86,400 seconds and `16 * 1024**3` bytes, repeat hash equality, separate empty roots, and resource attestation.
- [ ] Test that every failure produces a stop receipt and never activates the fallback.

```python
@pytest.mark.parametrize("forbidden", ["--qrels", "--grades", "--cluster-count"])
def test_feasibility_parser_rejects_effectiveness_options(forbidden: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "feasibility", "--config", "config.json",
            "--output-root", "out", forbidden, "forbidden",
        ])

def test_gate_thresholds_are_inclusive(valid_repeat: FeasibilityReceipt) -> None:
    summaries = valid_repeat.timing_summary
    exact = replace(
        valid_repeat,
        timing_summary={
            "pair_c20": replace(
                summaries["pair_c20"], mean_seconds=86_400 / (2 * 51_681)
            ),
            "pair_c40": replace(
                summaries["pair_c40"], mean_seconds=86_400 / (2 * 51_681)
            ),
            "single_c20": replace(summaries["single_c20"], mean_seconds=0.0),
            "single_c40": replace(summaries["single_c40"], mean_seconds=0.0),
        },
    )
    gate = evaluate_feasibility_repeats(
        exact, exact, repeat_a_rss_bytes=16 * 1024**3,
        repeat_b_rss_bytes=16 * 1024**3, exclusive_resource_attested=True,
    )
    assert gate.status == "pass"
    too_large = replace(exact, timing_summary={
        **exact.timing_summary,
        "single_c40": replace(
            exact.timing_summary["single_c40"],
            mean_seconds=float.fromhex("0x1p-30"),
        ),
    })
    assert evaluate_feasibility_repeats(
        too_large, exact, repeat_a_rss_bytes=1, repeat_b_rss_bytes=1,
        exclusive_resource_attested=True,
    ).status == "stop"
```
- [ ] Run:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation --with pytest pytest -q tests/test_dense_aggregation_feasibility.py tests/test_dense_aggregation_cli.py
```

Expected: FAIL because the feasibility and CLI modules do not exist.

- [ ] Implement the sampling/statistics/gate functions exactly as follows; the receipt stores both full integer lists and all four category summaries:

```python
def official_sample_indices() -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    rng = np.random.Generator(np.random.PCG64(2026))
    pair_indices = np.sort(
        rng.choice(51_681, size=256, replace=False, shuffle=True), kind="stable"
    )
    single_indices = np.sort(
        rng.choice(322, size=32, replace=False, shuffle=True), kind="stable"
    )
    return pair_indices, single_indices

def _linear_p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("timing summary requires at least one value")
    position = (len(ordered) - 1) * 0.95
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return ordered[lower] + fraction * (ordered[upper] - ordered[lower])

def _projection_seconds(summary: Mapping[str, TimingCategorySummary]) -> float:
    return (
        51_681 * (summary["pair_c20"].mean_seconds
                  + summary["pair_c40"].mean_seconds)
        + 322 * (summary["single_c20"].mean_seconds
                 + summary["single_c40"].mean_seconds)
    )

def parse_gnu_time_verbose(path: str | Path) -> int:
    prefix = "Maximum resident set size (kbytes):"
    matches = [
        line.partition(":")[2].strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip().startswith(prefix)
    ]
    if len(matches) != 1 or not matches[0].isdigit():
        raise ValueError("GNU time output has no unique RSS KiB field")
    return int(matches[0]) * 1024

def evaluate_feasibility_repeats(
    repeat_a: FeasibilityReceipt, repeat_b: FeasibilityReceipt, *,
    repeat_a_rss_bytes: int, repeat_b_rss_bytes: int,
    exclusive_resource_attested: bool,
) -> FeasibilityGate:
    failures = list(_repeat_integrity_failures(repeat_a, repeat_b))
    if repeat_a.fit_count != 576 or repeat_b.fit_count != 576:
        failures.append("each repeat must contain exactly 576 fits")
    if repeat_a.start_count != 5_760 or repeat_b.start_count != 5_760:
        failures.append("each repeat must contain exactly 5760 starts")
    projection = max(
        _projection_seconds(repeat_a.timing_summary),
        _projection_seconds(repeat_b.timing_summary),
    )
    rss_bytes = max(repeat_a_rss_bytes, repeat_b_rss_bytes)
    if projection > 86_400.0:
        failures.append("projected sequential runtime exceeds 24 hours")
    if rss_bytes > 16 * 1024**3:
        failures.append("maximum resident set size exceeds 16 GiB")
    if not (
        exclusive_resource_attested
        and repeat_a.exclusive_resource_attested
        and repeat_b.exclusive_resource_attested
    ):
        failures.append("exclusive host resources were not attested")
    return FeasibilityGate(
        status="pass" if not failures else "stop",
        failures=tuple(failures), projection_seconds=projection,
        maximum_rss_bytes=rss_bytes,
    )
```

The `provenance` map must have exactly `FEASIBILITY_PROVENANCE_KEYS`; unknown or missing keys fail validation. `_repeat_integrity_failures` executes `FEASIBILITY_REPEAT_CHECKS` in that exact order and returns `"feasibility repeat mismatch: <name>"` once for each failed item. The detailed checks compare exact config/input/protocol/contract/source/runtime/sample hashes, literal sample lists, category counts, every sampled fit's support/membership/center/assignment hashes, finite timing values, manifests, and all three resource attestations. `cache_root`, `output_root`, and `worker_pid` are intentionally required to be pairwise distinct across repeats after normalized-absolute-path validation; they are not equality fields. No caller may filter or downgrade a failure.

The worker's first filesystem action is `preflight_atomic_roots(cache_dir=cache_dir, output_dir=output_dir)`. It then validates the qrel-free config/contract, captures the worker numerical environment, and only afterward creates the empty cache and output staging roots. It writes `environment.json` into the manifested worker output, records that file's raw hash as `numerical_environment_sha256`, checks it again immediately before the first fit and after the final fit, and binds the same hash into the cache header. Tests patch the qrel path opener with a hard failure and separately prove same-filesystem rejection happens before frozen inputs or numerical modules are opened.

The worker uses the same clustering/cache calls as the full run and this measured loop; `cache.append_timing` occurs after `end_ns` because elapsed time cannot be known earlier:

```python
pairs = tuple(itertools.combinations(range(322), 2))
for exclusion_kind, sampled_indices in (
    ("pair", pair_indices), ("single", single_indices)
):
    for inventory_index in map(int, sampled_indices):
        excluded = pairs[inventory_index] if exclusion_kind == "pair" else (inventory_index,)
        for cluster_count in (20, 40):
            start_ns = time.perf_counter_ns()
            support = tuple(index for index in range(322) if index not in excluded)
            fit = fit_canonical_clusters(
                embeddings=inputs.embeddings, query_ids=inputs.query_ids,
                support_indices=support, cluster_count=cluster_count,
                embedding_sha256=inputs.embedding_sha256,
                clustering_contract_sha256=clustering_contract_sha256,
            )
            assignments = assign_targets(
                fit=fit, embeddings=inputs.embeddings, query_ids=inputs.query_ids,
                target_indices=excluded, support_index_by_id=query_index_by_id,
            )
            cache.append_fit(
                fit, assignments, exclusion_kind=exclusion_kind,
                inventory_index=inventory_index,
                excluded_query_ids=tuple(inputs.query_ids[index] for index in excluded),
            )
            cache.release(fit.key)
            n_iter = fit.n_iter
            del assignments, fit, support
            end_ns = time.perf_counter_ns()
            cache.append_timing(
                exclusion_kind=exclusion_kind, inventory_index=inventory_index,
                cluster_count=cluster_count, elapsed_ns=end_ns - start_ns,
                n_iter=n_iter,
            )
```

- [ ] Implement only the qrel-free lazy CLI subcommands `build-feasibility-config`, `prepare-frozen-input`, `validate-frozen-input`, `feasibility`, private `feasibility-worker`, and `validate-feasibility`; later tasks add commands only after their handler types exist. The config builder requires explicit query/embedding/metadata/protocol paths, validates their locked values, and writes canonical JSON through a temporary sibling plus `fsync` and no-replace publication. `prepare-frozen-input` validates the source array/metadata first and then atomically publishes a manifested create-exclusive bundle; both validators are strictly read-only. The public feasibility coordinator launches two workers sequentially under:

```bash
LC_ALL=C /usr/bin/time -v \
  -o artifacts/.dense_aggregation_feasibility-repeat-a.time.txt \
  uv run --frozen --python 3.13.9 --extra dense-aggregation \
  trec-mlm-dense-aggregation feasibility-worker \
  --config configs/trec_2026_dense_aggregation_feasibility.json \
  --cache-dir artifacts/.dense_aggregation_feasibility-repeat-a-cache \
  --output-dir artifacts/.dense_aggregation_feasibility-repeat-a-output \
  --exclusive-resource-attested
```

- [ ] Have the coordinator require two fresh, nonexistent cache/output destinations and write an atomic aggregate gate artifact only after both worker manifests validate.
- [ ] `feasibility --compact-receipt reports/trec_2026_dense_aggregation_feasibility.json` is required for the official coordinator. It writes the canonical compact summary only when `gate.status == "pass"`, uses a temporary sibling plus create-exclusive destination check, and leaves no compact file on a stop/failure path.
- [ ] Run the tests again; expected: PASS.
- [ ] Add both module paths to `IMPLEMENTATION_FILES`; run the contract test.
- [ ] Commit:

```bash
git add trec_mlm/dense_aggregation_feasibility.py trec_mlm/dense_aggregation_cli.py trec_mlm/dense_aggregation_contract.py tests/test_dense_aggregation_feasibility.py tests/test_dense_aggregation_cli.py tests/fixtures/dense_aggregation/sampling_numpy_2_5_1.json tests/fixtures/dense_aggregation/time_v.txt
git commit -m "feat: add qrel-free aggregation feasibility gate"
```

### Task 8: Implement the Isolated Legacy-Control Lock

**Files:**

- Create: `trec_mlm/legacy_dense_control.py`
- Create: `scripts/export_fixed_mpnet_k50_control.py`
- Create: `tests/test_legacy_dense_control.py`
- Modify: `trec_mlm/dense_aggregation_contract.py`

**Required interface:**

```python
@dataclass(frozen=True, slots=True)
class LegacyControlValidation:
    tsv_sha256: str
    row_count: int
    mean_ndcg_at_10: float
    mean_reciprocal_rank: float

def export_fixed_mpnet_k50_control(
    *, query_path: Path, qrel_path: Path, embedding_path: Path, output_tsv: Path
) -> LegacyControlValidation: ...
def validate_legacy_control_lock(lock_path: str | Path, expected_tsv_sha256: str) -> LegacyControlValidation: ...
```

- [ ] Write an AST/import-graph test that rejects any import from `trec_mlm.dense_aggregation*` in both new files. Allow only stdlib, NumPy, `query_knn`, `dense_query_knn`, `metrics`, and `runio` in the worker path.
- [ ] Mock the legacy scorer and require one call per outer query with `k=50`, no additional exclusion, canonical `Q`, canonical `M`, and the untouched similarity matrix.
- [ ] Test exactly 48,300 rows in canonical query/candidate order and columns `query_id`, `candidate_id`, `raw_score_hex`, `fold_prior_hex`, `final_rank`.
- [ ] Test two-fresh-process orchestration with a deterministic synthetic worker fixture whose bytes are supplied by the test; require isolated destinations, byte equality, and mismatch failure. Unit-test the exact nDCG/MRR constant checks by injecting `EvaluationResult` values. Do not open the official qrel file or execute the official legacy exporter before Task 13 passes.
- [ ] Test lock rejection for the wrong feasibility receipt, qrel/source/input hash, row count, TSV hash, TREC hash, score, rank, order, duplicate, or non-finite decoded float.

```python
def test_legacy_modules_have_isolated_imports() -> None:
    for path in (Path("trec_mlm/legacy_dense_control.py"),
                 Path("scripts/export_fixed_mpnet_k50_control.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any(name.startswith("trec_mlm.dense_aggregation") for name in imported)

def test_legacy_tsv_is_lossless_and_canonical(legacy_tsv: Path) -> None:
    with legacy_tsv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 48_300
    assert list(rows[0]) == [
        "query_id", "candidate_id", "raw_score_hex", "fold_prior_hex", "final_rank"
    ]
    assert [(row["query_id"], row["candidate_id"]) for row in rows] == sorted(
        (row["query_id"], row["candidate_id"]) for row in rows
    )
    assert all(float.fromhex(row["raw_score_hex"]).hex() == row["raw_score_hex"] for row in rows)
    assert all(float.fromhex(row["fold_prior_hex"]).hex() == row["fold_prior_hex"] for row in rows)
```

`legacy_tsv` is generated from a synthetic 322-query/150-candidate matrix and a mocked `rank_target_from_similarity`; it never reads `data/dev/llm_dev_qrels.txt`. The official byte-parity and exact metric regression are executed in Task 14, after the feasibility gate.
- [ ] Run the legacy tests; expected: FAIL because the exporter is absent.
- [ ] Implement the legacy worker around the untouched scorer with this loop; the script duplicates the six pure-stdlib environment assignments before its first `trec_mlm` import, then launches this worker twice in separate subprocesses:

```python
def export_fixed_mpnet_k50_control(
    *, query_path: Path, qrel_path: Path, embedding_path: Path, output_tsv: Path,
) -> LegacyControlValidation:
    if _raw_file_sha256(Path(dense_query_knn.__file__)) != _LOCKED_LEGACY_SOURCE_SHA256:
        raise RuntimeError("legacy dense scorer source hash mismatch")
    data = load_development_data(query_path, qrel_path)
    query_ids = tuple(sorted(data.queries))
    model_ids = tuple(sorted(data.models))
    grades = np.asarray([
        [data.qrels[query_id][model_id] for model_id in model_ids]
        for query_id in query_ids
    ], dtype=np.float64)
    embeddings = np.ascontiguousarray(
        np.load(embedding_path, allow_pickle=False), dtype=np.float32
    )
    if array_sha256(embeddings) != _LOCKED_EMBEDDING_SHA256:
        raise RuntimeError("legacy control embedding hash mismatch")
    similarities = cosine_similarity_matrix(embeddings)
    if array_sha256(similarities) != _LOCKED_SIMILARITY_SHA256:
        raise RuntimeError("legacy control similarity hash mismatch")
    rankings: dict[str, list[str]] = {}
    with output_tsv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("query_id", "candidate_id", "raw_score_hex",
                         "fold_prior_hex", "final_rank"))
        for target_index, query_id in enumerate(query_ids):
            result = rank_target_from_similarity(
                target_index=target_index, additionally_excluded_indices=(),
                similarities=similarities, grades=grades,
                query_ids=query_ids, model_ids=model_ids, k=50,
            )
            rankings[query_id] = [model_ids[index] for index in result.model_order]
            rank_by_index = {index: rank for rank, index in enumerate(result.model_order, 1)}
            for model_index, model_id in enumerate(model_ids):
                writer.writerow((
                    query_id, model_id, float(result.scores[model_index]).hex(),
                    float(result.global_prior[model_index]).hex(),
                    rank_by_index[model_index],
                ))
        handle.flush()
        os.fsync(handle.fileno())
    metrics = evaluate_rankings(rankings, data.qrels, k=10)
    if metrics.macro_ndcg_at_k != 0.36045807293576454:
        raise RuntimeError("legacy fixed-MPNet nDCG regression mismatch")
    if metrics.mean_reciprocal_rank != 0.4039893654911617:
        raise RuntimeError("legacy fixed-MPNet MRR regression mismatch")
    return LegacyControlValidation(
        tsv_sha256=_raw_file_sha256(output_tsv), row_count=48_300,
        mean_ndcg_at_10=metrics.macro_ndcg_at_k,
        mean_reciprocal_rank=metrics.mean_reciprocal_rank,
    )
```

`legacy_dense_control.py` defines `_raw_file_sha256` locally with stdlib `hashlib` so it imports no aggregation module. The orchestrator compares `Path(export_a).read_bytes() == Path(export_b).read_bytes()`, validates both independently, writes one canonical TREC run, and records both process return codes/source hashes plus the TSV/TREC hashes in `fixed_mpnet_k50_control.lock.json`. Its required `--compact-receipt` path is written atomically only after the manifested lock root reloads successfully; an existing destination is an error.

After equality, copy the already-validated bytes from repeat A into the staged authoritative name `fixed_mpnet_k50_control.tsv`; retain compact `repeat_a.json`/`repeat_b.json` process receipts rather than duplicate TSVs. The lock hashes that canonical file, and the manifest covers the TSV, TREC run, both repeat receipts, and lock JSON.
- [ ] Do not execute the official exporter in this task; official qrel access occurs only after Task 13 passes.
- [ ] Run the tests again; expected: PASS.
- [ ] Add the two Python paths to `IMPLEMENTATION_FILES` as provenance only; keep their import graph isolated.
- [ ] Commit:

```bash
git add trec_mlm/legacy_dense_control.py scripts/export_fixed_mpnet_k50_control.py tests/test_legacy_dense_control.py trec_mlm/dense_aggregation_contract.py
git commit -m "feat: add isolated dense legacy control lock"
```

### Task 9A: Implement the Pure Nested Work Scheduler

**Files:**

- Create: `trec_mlm/dense_aggregation_experiment.py`
- Create: `tests/test_dense_aggregation_experiment.py`
- Modify: `trec_mlm/dense_aggregation_contract.py`

**Required interfaces:**

```python
@dataclass(frozen=True, slots=True)
class FitWorkItem:
    exclusion_kind: Literal["pair", "single"]
    inventory_index: int
    excluded_indices: tuple[int, ...]
    support_indices: tuple[int, ...]
    cluster_count: Literal[20, 40]

def iter_pair_fit_work(query_ids: Sequence[str]) -> Iterator[FitWorkItem]: ...
def iter_single_fit_work(query_ids: Sequence[str]) -> Iterator[FitWorkItem]: ...
def retained_inner_row(outer_index: int, inner_index: int) -> int: ...
```

- [ ] Write the inventory/row-order tests below first. Add generalized property cases for `n in {3, 4, 42, 322}`: exactly `2 * comb(n, 2) + 2 * n` work items, stable pair order from `itertools.combinations(range(n), 2)`, C order `(20, 40)`, canonical support order, no duplicate inventory key, and no materialized embedding/grade array in a `FitWorkItem`.

```python
def test_official_fit_inventory_and_membership() -> None:
    query_ids = tuple(f"q{index:03d}" for index in range(322))
    pair_work = tuple(iter_pair_fit_work(query_ids))
    single_work = tuple(iter_single_fit_work(query_ids))
    assert len(pair_work) == 103_362
    assert len(single_work) == 644
    assert len(pair_work) + len(single_work) == 104_006
    assert 10 * (len(pair_work) + len(single_work)) == 1_040_060
    for item in pair_work:
        assert len(item.excluded_indices) == 2
        assert item.support_indices == tuple(
            i for i in range(322) if i not in item.excluded_indices
        )
    for item in single_work:
        assert len(item.excluded_indices) == 1
        assert item.support_indices == tuple(
            i for i in range(322) if i not in item.excluded_indices
        )

def test_retained_inner_row_order() -> None:
    assert [retained_inner_row(2, inner) for inner in (0, 1, 3, 4)] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="cannot equal"):
        retained_inner_row(2, 2)
```
- [ ] Run `uv run --frozen --python 3.13.9 --extra dense-aggregation --with pytest pytest -q tests/test_dense_aggregation_experiment.py -k 'official_fit_inventory or retained_inner_row or work_scheduler'`; expected: FAIL because the experiment module is absent.
- [ ] Implement only the frozen work-item generators and retained-row mapping. Run the same selection; expected: PASS.
- [ ] Add the module to `IMPLEMENTATION_FILES`; run the contract tests.
- [ ] Commit:

```bash
git add trec_mlm/dense_aggregation_experiment.py tests/test_dense_aggregation_experiment.py trec_mlm/dense_aggregation_contract.py
git commit -m "feat: schedule nested aggregation fits"
```

### Task 9B: Construct and Atomically Freeze Nested Predictions

**Files:**

- Modify: `trec_mlm/dense_aggregation_experiment.py`
- Modify: `tests/test_dense_aggregation_experiment.py`
- Modify: `trec_mlm/dense_aggregation_artifacts.py`
- Modify: `trec_mlm/dense_aggregation_cli.py`
- Modify: `trec_mlm/dense_aggregation_contract.py`

**Required interfaces:**

```python
@dataclass(frozen=True, slots=True)
class DenseAggregationExperimentConfig:
    repo_root: Path  # derived by the loader; not a serialized key
    source_path: Path  # derived by the loader; not a serialized key
    source_sha256: str  # hash of the exact parsed bytes; not serialized
    query_path: Path
    qrel_path: Path
    embedding_path: Path
    encoder_metadata_path: Path
    protocol_path: Path
    feasibility_receipt_path: Path
    feasibility_receipt_sha256: str
    legacy_control_lock_path: Path
    legacy_control_tsv_path: Path
    legacy_control_tsv_sha256: str
    dense_snapshot_path: Path
    dense_snapshot_sha256: str
    dense_per_query_path: Path
    dense_per_query_sha256: str
    grid: tuple[DenseAggregationCandidate, ...]

@dataclass(frozen=True, slots=True)
class ConstructionReceipt:
    clustering_contract_sha256: str
    construction_provenance_sha256: str
    prediction_freeze_sha256: str
    fit_count: int
    start_count: int
    selected_config_indices: tuple[int, ...]
    inner_score_hashes: tuple[tuple[str, ...], ...]
    inner_rank_hashes: tuple[tuple[str, ...], ...]
    started_at_utc: str
    finished_at_utc: str
    wall_elapsed_ns: int
    schema: str = "trec-mlm-dense-aggregation-construction-v1"

@dataclass(frozen=True, slots=True)
class OfficialPreflight:
    inputs: FrozenDenseInputs
    clustering_contract: Mapping[str, object]
    clustering_contract_sha256: str
    feasibility_manifest_sha256: str
    legacy_control_tsv_sha256: str

@dataclass(frozen=True, slots=True)
class ConstructionProvenance:
    query_file_sha256: str
    qrel_file_sha256: str
    query_input_sha256: str
    candidate_inventory_sha256: str
    embedding_sha256: str
    similarity_sha256: str
    protocol_sha256: str
    clustering_contract_sha256: str
    implementation_inventory_sha256: str
    dependency_inventory_sha256: str
    runtime_inventory_sha256: str
    numerical_environment_sha256: str
    feasibility_manifest_sha256: str
    legacy_control_tsv_sha256: str
    official_config_sha256: str
    schema: str = "trec-mlm-dense-aggregation-construction-provenance-v1"

@dataclass(slots=True)
class ValidatedConstructionArtifact:
    root: Path
    raw_manifest_sha256: str
    semantic_sha256: str
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    qrels: Mapping[str, Mapping[str, float]]
    grades: NDArray[np.float64]
    similarities: NDArray[np.float32]
    fit_cache: StreamingFitCache
    def primary_rankings(self) -> dict[str, list[str]]: ...
    def load_singleton_fit(
        self, target_index: int, cluster_count: Literal[20, 40],
    ) -> StoredFitRecord: ...
    def release_fit(self, key: FitKey) -> None: ...

def validate_official_preflight(
    config: DenseAggregationExperimentConfig, *, repo_root: Path,
) -> OfficialPreflight: ...
def load_official_config(
    path: str | Path,
) -> DenseAggregationExperimentConfig: ...
def load_and_validate_construction(
    root: str | Path, *, config: DenseAggregationExperimentConfig,
) -> ValidatedConstructionArtifact: ...
def run_dense_aggregation_experiment(
    config: DenseAggregationExperimentConfig, *, cache_dir: Path, output_dir: Path,
) -> ConstructionReceipt: ...
def candidate_inventory_sha256(model_ids: Sequence[str]) -> str: ...

INNER_METRIC_COLUMNS = (
    "outer_query_id", "config_index", "clusters", "neighbors",
    "local_weight", "beta", "inner_macro_ndcg_at_10",
    "inner_macro_mrr", "selected", "selection_rank",
    "gap_to_winner", "winner_margin", "score_sha256", "rank_sha256",
)
SELECTION_COLUMNS = (
    "outer_query_id", "selected_config_index", "clusters", "neighbors",
    "local_weight", "beta", "winner_margin",
)
PREDICTION_COLUMNS = (
    "outer_query_id", "candidate_id", "selected_config_index", "clusters",
    "neighbors", "local_weight", "beta", "local_score", "domain_score",
    "prior_score", "local_contribution", "domain_contribution",
    "prior_contribution", "final_score", "aggregation_score",
    "prior_tie_value", "final_rank",
)
NEIGHBOR_COLUMNS = (
    "outer_query_id", "realized_neighbor_count", "neighbor_mass",
    "neighbor_ids", "neighbor_similarities", "neighbor_weights",
)
CLUSTER_ASSIGNMENT_COLUMNS = (
    "outer_query_id", "target_cluster_id", "cluster_support",
    "squared_distance_hex", "membership_sha256", "center_sha256",
)
CONSTRUCTION_VOLATILE_JSON_PATHS = (
    ("construction_receipt.json", "/started_at_utc"),
    ("construction_receipt.json", "/finished_at_utc"),
    ("construction_receipt.json", "/wall_elapsed_ns"),
    ("fits/timings.jsonl", "/*/elapsed_ns"),
)
```

- [ ] Build a `toy_42q_4m` fixture with complete grades, 42 distinct embeddings, and an injectable deterministic clusterer so both C20 and C40 are exercised without expensive K-means.
- [ ] Write support-membership tests for every local, domain, prior, and cluster operation: inner uses exactly `Q - {outer, inner}`; outer/full-development uses exactly `Q - {target}`.
- [ ] Prove each unordered pair/C fit is invoked once, assigns both excluded targets before release, and feeds both directed roles. Prove each singleton/C fit is invoked once, emits the stitched outer evidence, persists its target assignment, and can later serve all eight full-development configurations with that `C` without another K-means call.
- [ ] Compare a naive directed/materialized oracle against the optimized scheduler for all components, `float.hex()` scores, ranks, inner metrics, selections, fit hashes, and receipts.
- [ ] Mutate an outer query's complete qrel vector and require no change to its 16 inner metrics, selection, support sets, fits, memberships, component values, neighbors, final scores, or ranking. Add the corresponding excluded-qrel and excluded-embedding tests for inner folds.
- [ ] Assert selection receives only the 16 inner macro nDCG values. MRR, outer metrics, runtime, support, slices, and uniqueness must be unreachable from the selector.
- [ ] Test disk-backed inner shapes/dtypes/order, logical slice hashes, selected configuration frequency, margins, neighbor evidence, target cluster evidence, and complete 322-by-150 stitched outer predictions.
- [ ] Test exact `ConstructionReceipt` schema/version/keys and reject missing, extra, wrong-type, or unknown-version fields before any array/cache access.
- [ ] `neighbors.tsv` has exactly one row per outer query in canonical order. Its three plural fields are canonical compact JSON arrays in neighbor-rank order with equal length `realized_neighbor_count`; IDs are unique support queries, similarities are finite positive float representations, weights are finite nonnegative representations summing to one when the count is positive, and `neighbor_mass` is the pre-normalization positive-similarity sum. The zero-positive fallback is exactly count `0`, mass `0.0`, and three empty arrays.
- [ ] Test that construction emits no full-development scores, ranks, metrics, uniqueness result, or deployment selection. It must retain the two singleton fits/assignments per query so the evaluator can construct them after the primary freeze.
- [ ] Test that the public worker validates feasibility, protocol/code/dependency/runtime contract hashes, frozen inputs, and legacy lock before its first qrel `open()`.
- [ ] Test `ConstructionProvenance` exact-key serialization and mutations of every field. Define `candidate_inventory_sha256(M)` as `framed_sha256("candidate-inventory", canonical_json(list(M)))`; prove it binds order and all 150 unique IDs. After qrel loading, persist `construction_provenance.json` with the raw qrel hash, raw query hash, canonical candidate-inventory hash, query-input/embedding/cosine hashes, protocol/contract/source/runtime inventories, feasibility/legacy locks, and official-config raw hash. The construction receipt stores the raw SHA-256 of this file.
- [ ] Test the construction loader's public singleton path: `load_singleton_fit(t, C)` delegates to `fit_cache.load_by_inventory(exclusion_kind="single", inventory_index=t, cluster_count=C)`, requires exactly `excluded_query_ids == (query_ids[t],)` and exactly one assignment for that query, and returns the complete `StoredFitRecord`. `release_fit(record.fit.key)` delegates to the cache and is required in a `finally` block by every consumer.
- [ ] Test the stable construction digest: after raw-manifest validation, canonicalize only `CONSTRUCTION_VOLATILE_JSON_PATHS` to empty timestamp strings/zero elapsed integers, pass those normalized file bytes to `semantic_artifact_sha256(..., tag="construction-semantic-artifact")`, and expose both raw-manifest and semantic hashes. Two valid fixtures differing only there have different raw manifests but identical semantic digests; any other mutation fails.
- [ ] Run the full experiment/artifact/CLI tests; expected: FAIL because construction publication is absent.
- [ ] Add `build-official-config`, `run-worker`, and `validate-construction` CLI commands around the strict handlers in this task. Their parsers reject unknown options, require explicit fresh output/cache paths, and do not import the evaluation module. The config builder serializes to a temporary sibling, `fsync()`s and reparses/revalidates it, publishes with `rename_noreplace`, and `fsync()`s the parent; tests cover existing/racing destinations and prove no overwrite or partial config is visible.
- [ ] Make qrel opening visibly last in preflight. `validate_official_preflight` validates the feasibility manifest/status/hash, implementation/protocol/environment contract, frozen query/embedding/cosine bundle, legacy lock/TSV hash, and dense snapshot/per-query reference hashes without statting, hashing, or reading `config.qrel_path`:

```python
def run_dense_aggregation_experiment(
    config: DenseAggregationExperimentConfig, *,
    cache_dir: Path, output_dir: Path,
) -> ConstructionReceipt:
    preflight_atomic_roots(cache_dir=cache_dir, output_dir=output_dir)
    preflight = validate_official_preflight(config, repo_root=config.repo_root)
    environment = capture_numerical_environment()
    require_same_numerical_environment(environment)
    qrel_sha256 = raw_file_sha256(config.qrel_path)
    if qrel_sha256 != "3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35":
        raise ValueError("development qrel file hash mismatch")
    qrels = load_complete_qrels(
        config.qrel_path, expected_query_ids=preflight.inputs.query_ids
    )
    if raw_file_sha256(config.qrel_path) != qrel_sha256:
        raise ValueError("development qrel file changed while it was loaded")
    model_ids = validate_complete_qrels(
        qrels, expected_query_ids=preflight.inputs.query_ids
    )
    provenance = ConstructionProvenance(
        query_file_sha256=raw_file_sha256(config.query_path),
        qrel_file_sha256=qrel_sha256,
        query_input_sha256=preflight.inputs.query_input_sha256,
        candidate_inventory_sha256=candidate_inventory_sha256(model_ids),
        embedding_sha256=preflight.inputs.embedding_sha256,
        similarity_sha256=preflight.inputs.similarity_sha256,
        protocol_sha256=str(preflight.clustering_contract["protocol_file_sha256"]),
        clustering_contract_sha256=preflight.clustering_contract_sha256,
        implementation_inventory_sha256=contract_source_inventory_sha256(
            preflight.clustering_contract, category="implementation"
        ),
        dependency_inventory_sha256=contract_source_inventory_sha256(
            preflight.clustering_contract, category="dependency"
        ),
        runtime_inventory_sha256=runtime_source_inventory_sha256(config.repo_root),
        numerical_environment_sha256=hashlib.sha256(
            numerical_environment_file_bytes(environment)
        ).hexdigest(),
        feasibility_manifest_sha256=preflight.feasibility_manifest_sha256,
        legacy_control_tsv_sha256=preflight.legacy_control_tsv_sha256,
        official_config_sha256=config.source_sha256,
    )
    grades = np.asarray([
        [qrels[query_id][model_id] for model_id in model_ids]
        for query_id in preflight.inputs.query_ids
    ], dtype=np.float64)
    if grades.shape != (322, 150) or set(np.unique(grades)) != {0.0, 1.0, 2.0}:
        raise ValueError("official qrels must be a complete 322x150 matrix with grades 0/1/2")
    return _construct_validated(
        config=config, preflight=preflight, provenance=provenance,
        environment=environment, qrels=qrels,
        model_ids=model_ids, grades=grades,
        cache_dir=cache_dir, output_dir=output_dir,
    )
```

`preflight_atomic_roots` is deliberately before `validate_official_preflight` and qrel hashing. `load_official_config` parses one captured byte string, retains `source_path` plus `source_sha256`, derives the repository root as the config directory's parent, requires the protocol at the exact root-relative location, and lexically joins every repository-relative value without touching the qrel leaf. `validate_official_preflight` first requires the current config-file raw hash to match the captured hash, uses `config.repo_root` rather than the ambient working directory, and validates every other pre-qrel input. `load_and_validate_construction` first validates the root manifest, then exact artifact schemas/inventories/hashes, and only then opens `fits/` read-only. It owns the cache handle and exposes no mutating cache method.

- [ ] Implement the pair phase with the following scheduling skeleton. `build_fold_evidence()` calls only the pure scoring functions; `consume_inner_prediction()` writes the exact memmap row, updates that slice's streaming score/rank hash, and accumulates nDCG/MRR for that outer/config:

```python
pair_inventory = tuple(itertools.combinations(range(len(query_ids)), 2))
inner_scores = np.lib.format.open_memmap(
    staging / "inner_scores.npy", mode="w+", dtype="<f8",
    shape=(322, 16, 321, 150),
)
inner_ranks = np.lib.format.open_memmap(
    staging / "inner_ranks.npy", mode="w+", dtype="<u2",
    shape=(322, 16, 321, 150),
)

def retained_inner_row(outer_index: int, inner_index: int) -> int:
    if outer_index == inner_index:
        raise ValueError("inner target cannot equal outer target")
    return inner_index if inner_index < outer_index else inner_index - 1

for pair_index, (left, right) in enumerate(pair_inventory):
    support = tuple(
        index for index in range(322) if index not in (left, right)
    )
    evidence = {
        target: build_fold_evidence(
            target_index=target, support_indices=support,
            similarities=similarities, grades=grades,
            query_ids=query_ids, neighbor_counts=(20, 50),
        )
        for target in (left, right)
    }
    for cluster_count in (20, 40):
        fit_start_ns = time.perf_counter_ns()
        fit = fit_canonical_clusters(
            embeddings=embeddings, query_ids=query_ids,
            support_indices=support, cluster_count=cluster_count,
            embedding_sha256=embedding_sha256,
            clustering_contract_sha256=clustering_contract_sha256,
        )
        assignments = assign_targets(
            fit=fit, embeddings=embeddings, query_ids=query_ids,
            target_indices=(left, right), support_index_by_id=query_index_by_id,
        )
        cache.append_fit(
            fit, assignments, exclusion_kind="pair", inventory_index=pair_index,
            excluded_query_ids=(query_ids[left], query_ids[right]),
        )
        assignment_by_target = {
            query_index_by_id[item.query_id]: item for item in assignments
        }
        for outer_index, target_index in ((left, right), (right, left)):
            domain = domain_scores(
                member_indices=assignment_by_target[target_index].member_indices,
                grades=grades,
            )
            for config_index, candidate in enumerate(DENSE_AGGREGATION_GRID):
                if candidate.clusters != cluster_count:
                    continue
                fold = evidence[target_index]
                prediction = combine_and_rank(
                    local=fold.local_by_k[candidate.neighbors].scores, domain=domain,
                    prior=fold.prior, model_ids=model_ids, candidate=candidate,
                )
                consume_inner_prediction(
                    outer_index=outer_index, target_index=target_index,
                    retained_row=retained_inner_row(outer_index, target_index),
                    config_index=config_index, prediction=prediction,
                    target_qrels=qrels_by_index[target_index],
                    score_memmap=inner_scores, rank_memmap=inner_ranks,
                )
        n_iter = fit.n_iter
        cache.release(fit.key)
        del assignments, fit
        cache.append_timing(
            exclusion_kind="pair", inventory_index=pair_index,
            cluster_count=cluster_count,
            elapsed_ns=time.perf_counter_ns() - fit_start_ns, n_iter=n_iter,
        )
```

`build_fold_evidence` returns a frozen record containing the query-ordered prior and the two `LocalEvidence` values. `consume_inner_prediction` converts `prediction.model_order` to candidate IDs, calls `ndcg_at_k(ranking_ids, target_qrels, k=10)` and `reciprocal_rank(ranking_ids, target_qrels)`, and writes the model-index-aligned score/rank row. It cannot access an outer ranking or any diagnostic.
- [ ] Select configurations and implement singleton fits with this sequence. `inner_ndcg_values` and `inner_mrr_values` contain 321 values per outer/config from `consume_inner_prediction`:

```python
selected_config_indices: list[int] = []
for outer_index in range(322):
    macro_ndcg = [math.fsum(values) / 321 for values in inner_ndcg_values[outer_index]]
    macro_mrr = [math.fsum(values) / 321 for values in inner_mrr_values[outer_index]]
    selected_index = select_inner_candidate(macro_ndcg)
    selection_order = tuple(sorted(
        range(16), key=lambda index: (-macro_ndcg[index], index)
    ))
    selection_rank = {
        config_index: rank
        for rank, config_index in enumerate(selection_order, start=1)
    }
    winner_margin = macro_ndcg[selection_order[0]] - macro_ndcg[selection_order[1]]
    gap_to_winner = tuple(
        macro_ndcg[selection_order[0]] - value for value in macro_ndcg
    )
    selected_config_indices.append(selected_index)
    write_inner_metric_rows(
        outer_index, macro_ndcg, macro_mrr, selected_index,
        selection_rank=selection_rank, winner_margin=winner_margin,
        gap_to_winner=gap_to_winner,
        score_hashers=inner_score_hashers[outer_index],
        rank_hashers=inner_rank_hashers[outer_index],
    )

for target_index in range(322):
    support = tuple(index for index in range(322) if index != target_index)
    selected = DENSE_AGGREGATION_GRID[selected_config_indices[target_index]]
    fold = build_fold_evidence(
        target_index=target_index, support_indices=support,
        similarities=similarities, grades=grades,
        query_ids=query_ids, neighbor_counts=(20, 50),
    )
    selected_prediction: AggregationPrediction | None = None
    for cluster_count in (20, 40):
        fit_start_ns = time.perf_counter_ns()
        fit = fit_canonical_clusters(
            embeddings=embeddings, query_ids=query_ids,
            support_indices=support, cluster_count=cluster_count,
            embedding_sha256=embedding_sha256,
            clustering_contract_sha256=clustering_contract_sha256,
        )
        (assignment,) = assign_targets(
            fit=fit, embeddings=embeddings, query_ids=query_ids,
            target_indices=(target_index,), support_index_by_id=query_index_by_id,
        )
        cache.append_fit(
            fit, (assignment,), exclusion_kind="single",
            inventory_index=target_index,
            excluded_query_ids=(query_ids[target_index],),
        )
        if selected.clusters == cluster_count:
            selected_prediction = combine_and_rank(
                local=fold.local_by_k[selected.neighbors].scores,
                domain=domain_scores(member_indices=assignment.member_indices, grades=grades),
                prior=fold.prior, model_ids=model_ids, candidate=selected,
            )
            write_primary_prediction(
                target_index, selected, fold, assignment, fit, selected_prediction
            )
        n_iter = fit.n_iter
        cache.release(fit.key)
        del assignment, fit
        cache.append_timing(
            exclusion_kind="single", inventory_index=target_index,
            cluster_count=cluster_count,
            elapsed_ns=time.perf_counter_ns() - fit_start_ns, n_iter=n_iter,
        )
    if selected_prediction is None:
        raise RuntimeError("selected outer cluster count was not constructed")
```
- [ ] Write `environment.json`, `construction_provenance.json`, lossless component arrays, human-readable `predictions.tsv`, `neighbors.tsv`, `cluster_assignments.tsv`, `inner_metrics.tsv`, `selections.tsv`, TREC runs, fit shards/receipts, and `prediction_freeze_receipt.json`. Recheck the numerical environment after the final fit and before freeze/publication. The strict loader rederives and matches every provenance field and environment hash before exposing predictions or fits. Do not write full-development matrices or compute outer nDCG, MRR, zero/slice diagnostics, uniqueness, deployment selection, or decision.
- [ ] Flush both inner memmaps, `fsync()` their backing files, close/delete the writable mappings, reopen them read-only, and compute/validate every logical slice hash from the reopened mappings before writing the prediction-freeze receipt.
- [ ] Require exactly one full-run timing row per fit (103,362 pair plus 644 singleton rows), with category/inventory/C/n_iter/elapsed fields. Full-run elapsed values cover the actual fit work item through its required scoring/evidence serialization and release; they are descriptive runtime data, not substituted into the preregistered feasibility projection. The construction wall clock in `ConstructionReceipt` is the authoritative end-to-end runtime. Comparator normalization allows only these elapsed values and the three receipt timing fields to differ.
- [ ] Give `predictions.tsv` the exact score columns `local_score`, `domain_score`, `prior_score`, `local_contribution`, `domain_contribution`, `prior_contribution`, `final_score`, `aggregation_score`, `prior_tie_value`, and `final_rank`. Write score text with `repr(float(value))`; require `aggregation_score == final_score` both numerically and textually so the explorer's primary-score alias and detailed evidence schema cannot diverge.
- [ ] Finalize and validate the initially separate cache, then call `cache.finalize_into(staging / "fits")` before construction `manifest.json` is written. The output manifest therefore covers every center/label/inertia shard and receipt; no fit cache remains outside the immutable construction root after success.
- [ ] `construction_receipt.json` contains exactly the noncircular `ConstructionReceipt` fields above plus the three allowed volatile timing fields used by Task 10B. It never contains its own artifact-manifest hash. After publication, callers hash `manifest.json` directly; no file covered by a manifest embeds that manifest's digest.
- [ ] Atomically publish with `atomic_artifact_directory(output_dir, quarantine_on_error=True)` and immediately reload the construction root with strict manifest, shape, finite-value, component-sum, ranking, TREC, and fit-count validation. This preserves a cache already moved into staging if a late error occurs. Feasibility/control/run writers likewise quarantine nonempty official staging; ordinary test/report staging uses cleanup. A quarantined root is retained only for diagnosis and can never be an input or retry destination.
- [ ] Run the tests again; expected: PASS.
- [ ] Add all created/modified entry-point paths to `IMPLEMENTATION_FILES`; run the contract test.
- [ ] Commit:

```bash
git add trec_mlm/dense_aggregation_experiment.py trec_mlm/dense_aggregation_artifacts.py trec_mlm/dense_aggregation_cli.py trec_mlm/dense_aggregation_contract.py tests/test_dense_aggregation_experiment.py tests/test_dense_aggregation_artifacts.py
git commit -m "feat: construct nested dense aggregation predictions"
```

### Task 10A: Implement Read-Only Evaluation, Controls, and Deployment Matrices

**Files:**

- Create: `trec_mlm/dense_aggregation_evaluation.py`
- Create: `tests/test_dense_aggregation_evaluation.py`
- Modify: `trec_mlm/dense_aggregation_artifacts.py`
- Modify: `trec_mlm/dense_aggregation_cli.py`
- Modify: `trec_mlm/dense_aggregation_contract.py`

**Required interfaces:**

```python
REQUIRED_INTEGRITY_CHECKS: tuple[str, ...] = (
    "input",
    "leakage",
    "clustering",
    "regression",
    "repeatability",
    "completeness",
    "artifact",
)
PROVISIONAL_INTEGRITY_CHECKS: tuple[str, ...] = tuple(
    name for name in REQUIRED_INTEGRITY_CHECKS if name != "repeatability"
)

@dataclass(frozen=True, slots=True)
class PromotionInputs:
    integrity_checks: Mapping[str, bool]
    ndcg_ci_lower: float
    ndcg_ci_upper: float
    mean_mrr_delta: float
    zero_ndcg_count: int
    low_density_mean_ndcg_delta: float

@dataclass(frozen=True, slots=True)
class EvaluationReceipt:
    construction_semantic_sha256: str
    primary_freeze_sha256: str
    full_development_score_hashes: tuple[str, ...]
    full_development_rank_hashes: tuple[str, ...]
    provisional_gate_inputs: PromotionInputs
    per_run_integrity_passed: bool
    started_at_utc: str
    finished_at_utc: str
    wall_elapsed_ns: int
    schema: str = "trec-mlm-dense-aggregation-evaluation-v1"

@dataclass(frozen=True, slots=True)
class ValidatedEvaluationArtifact:
    root: Path
    raw_manifest_sha256: str
    semantic_sha256: str
    receipt: EvaluationReceipt

EVALUATION_VOLATILE_JSON_PATHS = (
    ("evaluation_receipt.json", "/started_at_utc"),
    ("evaluation_receipt.json", "/finished_at_utc"),
    ("evaluation_receipt.json", "/wall_elapsed_ns"),
)

def evaluate_dense_aggregation_artifact(
    construction_root: str | Path,
    *,
    config_path: str | Path,
    output_dir: str | Path,
) -> EvaluationReceipt: ...
```

The seven names in `REQUIRED_INTEGRITY_CHECKS` are the exact category mapping for the protocol's input, leakage, clustering, regression, repeatability, completeness, and artifact checks. The evaluation receipt's provisional map must have exactly `PROVISIONAL_INTEGRITY_CHECKS`, with Boolean values and no missing or extra keys; `per_run_integrity_passed` must equal `all(provisional_gate_inputs.integrity_checks[name] for name in PROVISIONAL_INTEGRITY_CHECKS)`. Only the repeat comparator may extend that validated map with the `repeatability` key.

- [ ] Test that evaluation opens the construction root read-only, validates its raw manifest first, computes the stable construction semantic digest defined in Task 9B, recomputes every component with `math.fsum`, recomputes every rank/TREC row/hash, and rejects missing, extra, reordered, mutated, duplicate, or non-finite data.
- [ ] Test exact `EvaluationReceipt` schema/version/keys and reject missing, extra, wrong-type, or unknown-version fields before metrics/matrices are exposed.
- [ ] Test exact frozen nested-reference raw hashes and exact stored query order. Pair against parsed stored decimals; display the snapshot's authoritative aggregate headlines separately.
- [ ] Generate and validate the fixed TF-IDF `k=10`, fold-local prior, locked fixed-MPNet `k=50`, selected local-only, selected domain-only, and selected pre-shrink-mixture controls. Assert none enters the 16-item selector or promotion decision.

Require exact snapshot regression values before comparison: frozen nested dense nDCG/MRR `0.35985474409504775`/`0.40406330811168817` with 116 zero-nDCG queries; fixed MPNet `0.36045807293576454`/`0.4039893654911617`; fixed TF-IDF `0.2176542461618029`/`0.2776956693601721`; fold-local prior `0.1385241823188521`/`0.20665310801034087`.

```python
def ablation_scores(
    selected: DenseAggregationCandidate,
    prediction: AggregationPrediction,
) -> dict[str, NDArray[np.float64]]:
    return {
        "selected_local_only": prediction.local.copy(),
        "selected_domain_only": prediction.domain.copy(),
        "selected_pre_shrink_mixture": np.asarray([
            math.fsum((
                selected.local_weight * float(prediction.local[index]),
                (1.0 - selected.local_weight) * float(prediction.domain[index]),
            ))
            for index in range(prediction.local.size)
        ], dtype=np.float64),
    }

def rank_ablation(
    scores: NDArray[np.float64], prior: NDArray[np.float64],
    model_ids: Sequence[str],
) -> tuple[int, ...]:
    return tuple(sorted(
        range(len(model_ids)),
        key=lambda index: (-float(scores[index]), -float(prior[index]), model_ids[index]),
    ))
```

Use the outer fold's selected `k` for `prediction.local`, selected `C` for `prediction.domain`, and selected `lambda` for the mixture. Selected `beta` does not enter these three descriptive scores.
- [ ] Test primary macro metrics, canonical per-query differences, win/tie/loss tolerance `1e-12`, 10,000-resample paired bootstrap at seed 2026, zero transitions, and mutually exhaustive qrel-density slices `<=5`, `6..20`, and `>20` positives.
- [ ] Test the evaluator writes/hash-locks and `fsync()`s `primary_nested_result.json` plus `primary_freeze_receipt.json` before it reloads singleton fit records, constructs all sixteen full-development score/rank matrices, hashes score/rank uniqueness separately, computes their metrics, and selects the future-test deployment hint by macro nDCG then grid order. Mock K-means and assert it is never called by evaluation.
- [ ] Test the stable evaluation digest after raw-manifest validation: normalize only `EVALUATION_VOLATILE_JSON_PATHS` and frame with tag `evaluation-semantic-artifact`. The evaluation root does not copy construction timing rows/summaries; it references only the stable construction semantic digest. Two runs with different raw timing manifests must therefore produce identical primary-freeze hashes and identical evaluation semantic digests.
- [ ] Record the deployment recipe with the selected `C/k/lambda/beta`: for a future unseen target, its support is all 322 development queries and selected `C` is fit exactly once on those rows before target assignment. Label this a recipe only; no full-322-support fit occurs in the development run or its 104,006-fit count.
- [ ] Test that a per-run evaluator can emit only `PromotionInputs` with the exact six-key `PROVISIONAL_INTEGRITY_CHECKS` map and `per_run_integrity_passed`; it has no final-outcome field and never calls `decide_promotion`. Reject a missing, extra, non-Boolean, or prematurely present `repeatability` key.
- [ ] Test claim strings always label the result `leakage-controlled exploratory development evidence`, and the deployment metric `optimistic future-test hint`.
- [ ] Run the evaluation tests; expected: FAIL because the module is absent.
- [ ] Add `evaluate` and `validate-evaluation` CLI commands only after the evaluator/loader types exist; their parsers require explicit config, construction, output, phase, failure, and compact-terminal-decision paths.
- [ ] Implement evaluation in this exact phase order. `_write_primary_freeze_and_fsync()` writes primary macro/per-query values, frozen-reference comparison inputs, the stable construction semantic digest (not the timing-sensitive raw manifest hash), and primary prediction/TREC hashes; only after it returns may `load_singleton_fit()` be called:

```python
config = load_official_config(config_path)
construction = load_and_validate_construction(construction_root, config=config)
primary_rankings = construction.primary_rankings()
primary_metrics = evaluate_rankings(primary_rankings, construction.qrels, k=10)
primary_freeze = build_primary_freeze_receipt(construction, primary_metrics)
_write_primary_result_and_fsync(staging / "primary_nested_result.json", primary_metrics)
_write_primary_freeze_and_fsync(staging / "primary_freeze_receipt.json", primary_freeze)

full_scores = np.lib.format.open_memmap(
    staging / "full_development_scores.npy", mode="w+", dtype="<f8",
    shape=(16, 322, 150),
)
full_ranks = np.lib.format.open_memmap(
    staging / "full_development_ranks.npy", mode="w+", dtype="<u2",
    shape=(16, 322, 150),
)
for target_index in range(322):
    support = tuple(index for index in range(322) if index != target_index)
    fold = build_fold_evidence(
        target_index=target_index, support_indices=support,
        similarities=construction.similarities, grades=construction.grades,
        query_ids=construction.query_ids, neighbor_counts=(20, 50),
    )
    for cluster_count in (20, 40):
        record = construction.load_singleton_fit(target_index, cluster_count)
        try:
            (assignment,) = record.assignments
            domain = domain_scores(
                member_indices=assignment.member_indices, grades=construction.grades
            )
            for config_index, candidate in enumerate(DENSE_AGGREGATION_GRID):
                if candidate.clusters != cluster_count:
                    continue
                prediction = combine_and_rank(
                    local=fold.local_by_k[candidate.neighbors].scores, domain=domain,
                    prior=fold.prior, model_ids=construction.model_ids,
                    candidate=candidate,
                )
                full_scores[config_index, target_index] = prediction.final_scores
                full_ranks[config_index, target_index] = prediction.ranks
        finally:
            construction.release_fit(record.fit.key)

full_score_hashes = tuple(
    logical_array_sha256(
        full_scores[index], tag="full-development-score", dtype="<f8",
        row_inventory=construction.query_ids,
        column_inventory=construction.model_ids,
    ) for index in range(16)
)
full_rank_hashes = tuple(
    logical_array_sha256(
        full_ranks[index], tag="full-development-rank", dtype="<u2",
        row_inventory=construction.query_ids,
        column_inventory=construction.model_ids,
    ) for index in range(16)
)
deployment_metrics = evaluate_all_full_development_rankings(
    full_ranks, construction.query_ids, construction.model_ids, construction.qrels
)
deployment_index = min(
    range(16),
    key=lambda index: (-deployment_metrics[index].macro_ndcg_at_k, index),
)
```

Before computing the two hash tuples, flush and `fsync()` both full-development memmaps, delete the writable mappings, reopen the files read-only with `np.load(..., mmap_mode="r", allow_pickle=False)`, and validate exact shape/dtype. All later metrics and publication validation use only those reopened mappings.

- [ ] Implement paired diagnostics with `paired_bootstrap_ci`; construct baseline and candidate dicts by inserting query IDs in canonical `Q` order, pass `confidence_level=0.95`, `n_resamples=10_000`, `seed=2026`, and compute MRR delta with `math.fsum(candidate[q] - baseline[q] for q in Q) / 322`.

```python
def win_tie_loss(differences: Sequence[float]) -> tuple[int, int, int]:
    wins = sum(value > 1e-12 for value in differences)
    losses = sum(value < -1e-12 for value in differences)
    ties = len(differences) - wins - losses
    return wins, ties, losses

def density_slice(positive_count: int) -> Literal["at_most_5", "6_through_20", "more_than_20"]:
    if positive_count <= 5:
        return "at_most_5"
    if positive_count <= 20:
        return "6_through_20"
    return "more_than_20"

def zero_transition(baseline: float, candidate: float) -> str:
    baseline_zero = baseline == 0.0
    candidate_zero = candidate == 0.0
    if baseline_zero and candidate_zero:
        return "zero_in_both"
    if baseline_zero:
        return "rescued_from_dense_zero"
    if candidate_zero:
        return "harmed_to_zero"
    return "nonzero_in_both"
```

For the low-density promotion gate, concatenate canonical queries from `at_most_5` and `6_through_20` and compute their candidate-minus-baseline nDCG mean with `math.fsum`.
- [ ] Atomically publish the evaluation root with its manifest, immediately reopen it read-only, and validate all schemas, source hashes, matrices, hashes, controls, diagnostics, and provisional inputs.
- [ ] Run the tests again; expected: PASS.
- [ ] Add the module path to `IMPLEMENTATION_FILES`; run the contract test.
- [ ] Commit:

```bash
git add trec_mlm/dense_aggregation_evaluation.py trec_mlm/dense_aggregation_artifacts.py trec_mlm/dense_aggregation_cli.py trec_mlm/dense_aggregation_contract.py tests/test_dense_aggregation_evaluation.py tests/test_dense_aggregation_artifacts.py tests/test_dense_aggregation_cli.py
git commit -m "feat: evaluate frozen aggregation predictions"
```

### Task 10B: Compare Fresh Runs and Make the Sole Final Decision

**Files:**

- Modify: `trec_mlm/dense_aggregation_evaluation.py`
- Modify: `tests/test_dense_aggregation_evaluation.py`
- Modify: `trec_mlm/dense_aggregation_artifacts.py`
- Modify: `tests/test_dense_aggregation_artifacts.py`
- Modify: `trec_mlm/dense_aggregation_cli.py`
- Modify: `tests/test_dense_aggregation_cli.py`
- Modify: `trec_mlm/dense_aggregation_contract.py`

**Required interfaces:**

```python
@dataclass(frozen=True, slots=True)
class Decision:
    outcome: Literal["promote", "reject", "reject-invalid"]
    failed_checks: tuple[str, ...]

@dataclass(frozen=True, slots=True)
class DeploymentHintReceipt:
    grid_index: int
    clusters: Literal[20, 40]
    neighbors: Literal[20, 50]
    local_weight_hex: str
    prior_shrinkage_hex: str
    macro_ndcg_at_10_hex: str
    mean_reciprocal_rank_hex: str
    label: Literal["optimistic future-test hint"]

@dataclass(frozen=True, slots=True)
class DecisionReceipt:
    integrity_checks: Mapping[str, bool]
    gate_observations: Mapping[str, str | int | None]
    gate_thresholds: Mapping[str, str | int | bool]
    gate_status: Mapping[str, Literal["pass", "fail", "not_evaluated"]]
    selected_configuration_frequency: tuple[int, ...] | None
    deployment_hint: DeploymentHintReceipt | None
    effective_unique_configuration_count: Mapping[str, int] | None
    outcome: Literal["promote", "reject", "reject-invalid"]
    failed_checks: tuple[str, ...]
    unavailable_fields: tuple[str, ...]
    claim_scope: str
    schema: str = "trec-mlm-dense-aggregation-decision-v1"

@dataclass(frozen=True, slots=True)
class RepeatabilityReceipt:
    run_fit_counts: tuple[int, int]
    run_start_counts: tuple[int, int]
    total_fit_count: int
    total_start_count: int
    semantic_equal: bool
    decision: Decision
    integrity_failures: tuple[str, ...]
    schema: str = "trec-mlm-dense-aggregation-repeatability-v1"

VOLATILE_JSON_PATHS: tuple[tuple[str, str], ...] = (
    *CONSTRUCTION_VOLATILE_JSON_PATHS,
    *EVALUATION_VOLATILE_JSON_PATHS,
)

def decide_promotion(inputs: PromotionInputs) -> Decision: ...
def build_decision_receipt(
    *, promotion_inputs: PromotionInputs | None,
    selected_configuration_frequency: Sequence[int] | None,
    deployment_hint: DeploymentHintReceipt | None,
    effective_unique_configuration_count: Mapping[str, int] | None,
    integrity_failures: Sequence[str],
) -> DecisionReceipt: ...
def write_terminal_reject_invalid(
    *, phase: Literal["run_a", "run_b", "comparison"],
    integrity_failures: Sequence[str], output_dir: Path,
    compact_receipt: Path,
) -> DecisionReceipt: ...
def compare_authoritative_repeats(
    run_a_construction: Path, run_a_evaluation: Path,
    run_b_construction: Path, run_b_evaluation: Path, *,
    config_path: Path,
    output_dir: Path,
) -> RepeatabilityReceipt: ...
```

`gate_observations` has exactly `ndcg_ci_lower_hex`, `ndcg_ci_upper_hex`, `mean_mrr_delta_hex`, `zero_ndcg_count`, and `low_density_mean_ndcg_delta_hex`. Every available float is `float.hex()`; unavailable observations are `None`. `gate_thresholds` persists exact values zero, `float(-0.005).hex()`, 116, zero, and all-true. `gate_status` has exactly `ndcg_ci_lower_gt_zero`, `mean_mrr_delta_gte_minus_0_005`, `zero_ndcg_count_lt_116`, `low_density_mean_ndcg_delta_gt_zero`, and `all_integrity_checks`. Tests reconstruct each comparison from observations/thresholds rather than trusting a stored Boolean. Every non-provisional `PromotionInputs` and every `DecisionReceipt.integrity_checks` map has exactly `REQUIRED_INTEGRITY_CHECKS`, all with values whose concrete type is `bool`; missing, extra, or non-Boolean values are themselves `reject-invalid` schema failures. `selected_configuration_frequency` is a 16-integer tuple summing to 322. `DeploymentHintReceipt` stores integer grid/C/k values and requires canonical `float.hex()` round trips for lambda, beta, and both metrics. `effective_unique_configuration_count` has exactly `score_matrices` and `rank_matrices`, each the distinct-hash count among the sixteen full-development configurations.

- [ ] Test promotion boundaries exactly: CI lower bound `==0` fails; MRR delta `==-0.005` passes; zero count `==116` fails; low-density mean delta `==0` fails; any integrity failure yields `reject-invalid`; all integrity checks plus all four effectiveness gates are required for `promote`. Independently prove that omitting any one required integrity key, adding an unknown key, or using a non-Boolean value can never promote.
- [ ] Test that only `compare_authoritative_repeats`, after proving repeatability, may add `integrity_checks["repeatability"]` and call `decide_promotion`. Static-call tests reject production calls from the per-run evaluator.
- [ ] Test the exact `DecisionReceipt` schema for `promote`, `reject`, and early/late `reject-invalid`. Require all five gate entries, full-precision hex round trips, 16 selection frequencies, deployment hint, effective-unique count, outcome, and failed checks. On an early integrity failure, unavailable result-dependent fields are explicitly `None`/`not_evaluated` and named in `unavailable_fields`; no effectiveness metric is computed or claimed.
- [ ] Test `write_terminal_reject_invalid` rejects an empty failure list, can only produce `reject-invalid`, atomically writes a manifested local failure root plus a create-exclusive compact decision receipt, names every failed invariant, and never calls `decide_promotion` or reads an invalid prediction artifact.
- [ ] Test repeat comparison permits differences only at `VOLATILE_JSON_PATHS`; reject an unknown timestamp/timing key as a schema error. Memberships, canonical IDs, assignments, center hashes, inner metrics, selections, raw/weighted components, final scores/ranks, TREC rows, per-query values, inputs, and inventories must match exactly.
- [ ] Test exact `RepeatabilityReceipt` and `DecisionReceipt` schema/version/keys and reject missing, extra, wrong-type, or unknown-version fields.
- [ ] Require 104,006 fits/1,040,060 starts in each run and 208,012/2,080,120 in the comparison receipt.
- [ ] Record each construction/evaluation receipt's manifested wall time in `authoritative_result.json`. The report's memory value is explicitly the preregistered feasibility-gate maximum RSS, not an unmeasured authoritative-run peak; neither descriptive value enters semantic equality, selection, or promotion.

```python
def passing_inputs() -> PromotionInputs:
    return PromotionInputs(
        integrity_checks={name: True for name in REQUIRED_INTEGRITY_CHECKS},
        ndcg_ci_lower=float.fromhex("0x1p-20"),
        ndcg_ci_upper=float.fromhex("0x1p-10"),
        mean_mrr_delta=0.0,
        zero_ndcg_count=115,
        low_density_mean_ndcg_delta=float.fromhex("0x1p-20"),
    )

@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"ndcg_ci_lower": 0.0}, "reject"),
        ({"mean_mrr_delta": -0.005}, "promote"),
        ({"zero_ndcg_count": 116}, "reject"),
        ({"low_density_mean_ndcg_delta": 0.0}, "reject"),
    ],
)
def test_promotion_boundaries(changes: dict[str, object], expected: str) -> None:
    assert decide_promotion(replace(passing_inputs(), **changes)).outcome == expected

@pytest.mark.parametrize("failed", REQUIRED_INTEGRITY_CHECKS)
def test_failed_integrity_check_cannot_promote(failed: str) -> None:
    checks = {name: name != failed for name in REQUIRED_INTEGRITY_CHECKS}
    assert decide_promotion(replace(passing_inputs(), integrity_checks=checks)) == Decision(
        "reject-invalid", (failed,)
    )

@pytest.mark.parametrize("missing", REQUIRED_INTEGRITY_CHECKS)
def test_missing_integrity_check_cannot_promote(missing: str) -> None:
    checks = {name: True for name in REQUIRED_INTEGRITY_CHECKS if name != missing}
    assert decide_promotion(replace(passing_inputs(), integrity_checks=checks)) == Decision(
        "reject-invalid", (f"missing-integrity-check:{missing}",)
    )

def test_repeat_counts_are_exact(valid_comparison: RepeatabilityReceipt) -> None:
    assert valid_comparison.run_fit_counts == (104_006, 104_006)
    assert valid_comparison.run_start_counts == (1_040_060, 1_040_060)
    assert valid_comparison.total_fit_count == 208_012
    assert valid_comparison.total_start_count == 2_080_120
```

- [ ] Run the comparator/decision tests; expected: FAIL because the production comparator is absent.
- [ ] Add the `compare-repeats` CLI command around the strict comparator and terminal-receipt writer in this task; its parser requires the official config, all four run roots, output root, and compact-terminal-decision path. It loads the config from captured bytes, rehashes it, and requires every construction/evaluation provenance record to bind that exact raw hash.
- [ ] Implement promotion as this pure function with no access to configurations or diagnostics; keep the repeat comparator as its only production caller:

```python
def decide_promotion(inputs: PromotionInputs) -> Decision:
    expected = frozenset(REQUIRED_INTEGRITY_CHECKS)
    actual = frozenset(inputs.integrity_checks)
    schema_failures = (
        *(f"missing-integrity-check:{name}" for name in REQUIRED_INTEGRITY_CHECKS if name not in actual),
        *(f"unknown-integrity-check:{name}" for name in sorted(actual - expected)),
        *(f"non-boolean-integrity-check:{name}" for name in REQUIRED_INTEGRITY_CHECKS
          if name in actual and type(inputs.integrity_checks[name]) is not bool),
    )
    if schema_failures:
        return Decision("reject-invalid", tuple(schema_failures))
    integrity_failures = tuple(
        name for name in REQUIRED_INTEGRITY_CHECKS
        if inputs.integrity_checks[name] is not True
    )
    if integrity_failures:
        return Decision("reject-invalid", integrity_failures)
    gates = {
        "ndcg_ci_lower_gt_zero": inputs.ndcg_ci_lower > 0.0,
        "mean_mrr_delta_gte_minus_0_005": inputs.mean_mrr_delta >= -0.005,
        "zero_ndcg_count_lt_116": inputs.zero_ndcg_count < 116,
        "low_density_mean_ndcg_delta_gt_zero": inputs.low_density_mean_ndcg_delta > 0.0,
    }
    failed = tuple(name for name, passed in gates.items() if not passed)
    return Decision("promote" if not failed else "reject", failed)
```

- [ ] Implement `_repeat_integrity_failures` as an ordered check table with these exact names: `construction_manifests`, `evaluation_manifests`, `source_provenance`, `runtime_provenance`, `input_provenance`, `fit_inventory`, `fit_counts`, `start_counts`, `memberships`, `centers`, `assignments`, `inner_arrays`, `inner_metrics`, `selections`, `primary_components`, `primary_scores`, `primary_ranks`, `trec_runs`, `primary_metrics`, `controls`, `ablations`, `diagnostics`, `full_development_arrays`, `deployment_hint`, and `provisional_gate_inputs`. Return `"repeat mismatch: <name>"` once per failed check in that order. No caller may suppress a failure.
- [ ] Implement the recursive comparator by parsing canonical JSON/JSONL, deleting only the seven exact volatile path patterns above, and requiring equality everywhere else. For manifests, recompute a semantic inventory in which hashes of only the three volatile-containing files are replaced by hashes of their normalized payloads; all other file bytes and raw hashes remain exact.
- [ ] After equality, first require Run A's provisional map to have exactly `PROVISIONAL_INTEGRITY_CHECKS`; construct final `PromotionInputs` by inserting only `integrity_checks["repeatability"] = True`, assert the resulting map has exactly `REQUIRED_INTEGRITY_CHECKS`, call `decide_promotion` once, and store that sole `Decision` in the comparison receipt.
- [ ] On any repeat mismatch, publish a manifested local comparison root containing `repeatability_receipt.json` and the full `decision_receipt.json` with outcome `reject-invalid`; atomically create the compact terminal decision receipt and stop. On equality, build the complete decision receipt from validated Run A plus repeatability before creating the authoritative result.
- [ ] The successful comparison writer first writes/fsyncs `repeatability_receipt.json` and `decision_receipt.json`, computes the repeatability receipt's raw SHA-256, places that value in `authoritative_result.json` under `comparison_receipt_sha256`, and writes the comparison root manifest last. Never put the comparison manifest's own hash inside a covered file.
- [ ] Atomically publish/reopen/validate the comparison root. Run the comparator/decision tests; expected: PASS.
- [ ] Run contract and full evaluation/artifact/CLI tests; expected: PASS.
- [ ] Commit:

```bash
git add trec_mlm/dense_aggregation_evaluation.py trec_mlm/dense_aggregation_artifacts.py trec_mlm/dense_aggregation_cli.py trec_mlm/dense_aggregation_contract.py tests/test_dense_aggregation_evaluation.py tests/test_dense_aggregation_artifacts.py tests/test_dense_aggregation_cli.py
git commit -m "feat: compare aggregation repeats and decide"
```

### Task 11: Extend the Query Explorer and Build Accessible Reports

**Files:**

- Modify: `.agents/skills/inspect-query-results/SKILL.md`
- Modify: `.agents/skills/inspect-query-results/scripts/build_query_explorer.py`
- Modify: `tests/test_query_explorer_skill.py`
- Create: `scripts/build_dense_aggregation_report.py`
- Create: `tests/test_dense_aggregation_report.py`
- Modify: `trec_mlm/dense_aggregation_contract.py`

- [ ] Add explorer tests for `aggregation_score` plus raw `local_score`, `domain_score`, `prior_score`, three weighted contributions, neighbor IDs/similarities/mass/count, cluster ID/support, membership hash, and center hash.
- [ ] Test evidence joins by canonical IDs, stable prediction order after qrels load, HTML escaping, labels that call values scores rather than probabilities, and graceful display of no positive neighbor fallback.
- [ ] Test that a manifested run is validated before read and that default explorer output is a sibling of—not a child of—the immutable artifact root. Building an explorer must leave the run manifest byte-identical.
- [ ] Add report tests for source hashes, metrics, full-precision differences/CIs, gate table, controls/ablations, selection frequency, effective uniqueness, runtime/RSS, integrity/repeat receipts, deployment warning, and exploratory claim scope.

```python
def test_manifested_run_defaults_explorer_outside_root(tmp_path: Path) -> None:
    run_root = tmp_path / "construction"
    run_root.mkdir()
    (run_root / "manifest.json").write_text("{}\n", encoding="utf-8")
    assert default_output_path(run_root) == tmp_path / "construction_query_explorer.html"

def test_aggregation_evidence_requires_complete_component_group() -> None:
    fields = ["local_score", "final_score"]
    with pytest.raises(ExplorerInputError, match="columns missing"):
        _aggregation_evidence(
            {"local_score": "1.0", "final_score": "1.0"},
            row_number=2, fieldnames=fields,
        )

def test_report_rejects_confirmatory_claim(tmp_path: Path) -> None:
    root = build_manifested_comparison_fixture(tmp_path)
    result_path = root / "authoritative_result.json"
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    payload["claim_scope"] = "confirmatory"
    result_path.write_text(json.dumps(payload), encoding="utf-8")
    rewrite_manifest_for_fixture(root)
    with pytest.raises(ValueError, match="invalid claim scope"):
        _validated_result_payload(root)
```

`build_manifested_comparison_fixture` creates the exact `RESULT_KEYS` map through the evaluation fixture builder, and `rewrite_manifest_for_fixture` is test-only mutation setup; neither helper is imported by production code.
- [ ] Run:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation --with pytest pytest -q tests/test_query_explorer_skill.py tests/test_dense_aggregation_report.py
```

Expected: FAIL because aggregation evidence/report support is absent.

- [ ] Preserve optional evidence with these exact field groups. If any candidate evidence column is present, require the entire candidate group; parse all score fields with `_finite_float`, keep candidate rank and `cluster_support` strictly positive, parse `realized_neighbor_count` with a separate nonnegative helper, and keep IDs/hashes as escaped text:

```python
SCORE_ALIASES = (
    "final_score", "aggregation_score", "raw_score", "sum_score",
    "primary_score", "score",
)
AGGREGATION_CANDIDATE_FLOAT_FIELDS = (
    "local_score", "domain_score", "prior_score",
    "local_contribution", "domain_contribution", "prior_contribution",
    "final_score", "prior_tie_value",
)
AGGREGATION_QUERY_FIELDS = (
    "realized_neighbor_count", "neighbor_mass", "neighbor_ids",
    "neighbor_similarities", "target_cluster_id", "cluster_support",
    "membership_sha256", "center_sha256",
)

def _aggregation_evidence(
    row: Mapping[str, str], *, row_number: int, fieldnames: Sequence[str]
) -> dict[str, object] | None:
    present = [name for name in AGGREGATION_CANDIDATE_FLOAT_FIELDS if name in fieldnames]
    if not present:
        return None
    if tuple(present) != AGGREGATION_CANDIDATE_FLOAT_FIELDS:
        missing = sorted(set(AGGREGATION_CANDIDATE_FLOAT_FIELDS) - set(present))
        raise ExplorerInputError(f"aggregation prediction columns missing: {missing!r}")
    return {
        name: _finite_float((row.get(name) or "").strip(),
                            row_number=row_number, column=name)
        for name in AGGREGATION_CANDIDATE_FLOAT_FIELDS
    }

def _nonnegative_int(value: str, *, row_number: int, column: str) -> int:
    parsed = _finite_float(value, row_number=row_number, column=column)
    if parsed < 0 or not parsed.is_integer():
        raise ExplorerInputError(
            f"row {row_number} has an invalid nonnegative integer {column}: {value!r}"
        )
    return int(parsed)

def default_output_path(run_dir: Path) -> Path:
    if (run_dir / "manifest.json").is_file():
        return run_dir.parent / f"{run_dir.name}_query_explorer.html"
    return run_dir / "query_explorer.html"
```

Keep `_positive_rank` unchanged for candidate ranks and positive cluster support. Parse only `realized_neighbor_count` with `_nonnegative_int`. Add `--neighbors` and `--clusters` paths. Require exactly one query-level record per construction query, validate their files against the construction manifest, join on `outer_query_id`, and attach the joined query evidence to every displayed candidate without changing prediction order. When both `final_score` and `aggregation_score` occur, require their source text and parsed float values to match. Render an accessible `<dl>` with raw signals, weighted contributions, neighbor summary, and cluster summary; use `html.escape`/JSON serialization already used by the script.

- [ ] Implement the report builder as a read-only consumer of a validated repeat-comparison root. Require these exact top-level snapshot keys before rendering:

```python
RESULT_KEYS = {
    "schema", "protocol_sha256", "construction_manifest_sha256",
    "evaluation_manifest_sha256", "comparison_receipt_sha256",
    "primary", "frozen_nested_dense", "paired_comparison", "controls",
    "ablations", "selection", "effective_uniqueness", "deployment_hint",
    "feasibility", "runtime", "memory", "integrity", "decision",
    "claim_scope",
}

def _validated_result_payload(comparison_root: Path) -> dict[str, object]:
    validate_manifest(comparison_root)
    payload = json.loads(
        (comparison_root / "authoritative_result.json").read_text(encoding="utf-8")
    )
    if set(payload) != RESULT_KEYS:
        raise ValueError("authoritative result snapshot keys differ from schema")
    if payload["claim_scope"] != "leakage-controlled exploratory development evidence":
        raise ValueError("authoritative result has invalid claim scope")
    return payload
```

The builder writes `results.json`, `decision.json`, `repeatability.json`, `report.md`, and standalone accessible `report.html` inside `atomic_artifact_directory(output_dir)`. `decision.json` is a canonical byte-for-byte copy of the validated comparison root's complete `decision_receipt.json`, not a reduced outcome summary; its tests require every full-precision gate, selection frequency, deployment hint, effective-unique count, and final outcome. The builder reopens all five payloads, requires `repeatability.json` to match the comparison root's validated `repeatability_receipt.json`, validates displayed numeric fields against the source payload, writes `manifest.json` last through the shared artifact writer, and publishes the directory with `rename_noreplace`.
- [ ] Run the tests again; expected: PASS.
- [ ] Add both Python paths to `IMPLEMENTATION_FILES`; run the contract test.
- [ ] Commit:

```bash
git add .agents/skills/inspect-query-results/SKILL.md .agents/skills/inspect-query-results/scripts/build_query_explorer.py tests/test_query_explorer_skill.py scripts/build_dense_aggregation_report.py tests/test_dense_aggregation_report.py trec_mlm/dense_aggregation_contract.py
git commit -m "feat: report dense aggregation evidence"
```

### Task 12: Complete the Pre-Gate Verification and Freeze Code Provenance

**Files:**

- Modify only if a test exposes a defect: files created in Tasks 1-11 and their tests.
- Create from the recorded locked review if not already present: `reports/trec_2026_dense_aggregation_advisor_receipt.json`

- [ ] Verify `IMPLEMENTATION_FILES` is sorted, complete, unique, and hashes the final source bytes. It must include every `dense_aggregation*.py` module, `legacy_dense_control.py`, both new scripts, and the explorer implementation. Also verify the exact `DEPENDENCY_SOURCE_FILES` and `RUNTIME_PROVENANCE_FILES` tuples above; the clustering contract hashes both Python tuples in its single protocol field, artifacts separately hash the runtime tuple, and Tasks 14–16 revalidate all three category inventories before any qrel/control/run action.
- [ ] Confirm the legacy source remains untouched:

```bash
sha256sum trec_mlm/dense_query_knn.py
```

Expected: `90f8db3abbb2887d8513fb881c631a3843319c29da02e935e1ecdb6c3c85e248`.

- [ ] Run all focused tests under the exact environment:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation --with pytest pytest -q \
  tests/test_dense_aggregation_runtime.py \
  tests/test_dense_aggregation_hashing.py \
  tests/test_dense_aggregation_contract.py \
  tests/test_dense_aggregation.py \
  tests/test_dense_aggregation_clustering.py \
  tests/test_dense_aggregation_cache.py \
  tests/test_dense_aggregation_feasibility.py \
  tests/test_legacy_dense_control.py \
  tests/test_dense_aggregation_experiment.py \
  tests/test_dense_aggregation_artifacts.py \
  tests/test_dense_aggregation_evaluation.py \
  tests/test_dense_aggregation_cli.py \
  tests/test_dense_aggregation_report.py \
  tests/test_query_explorer_skill.py
```

Expected: all PASS.

- [ ] Run the complete existing suite to detect regressions:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation --with pytest pytest -q \
  --deselect tests/test_dense_experiment.py::MergedControlRegressionTest::test_official_development_controls_match_merged_report
```

Expected: all selected tests PASS. The single deselected historical regression opens the official development qrels and is rerun in Task 14 after the gate; every other pre-gate test uses synthetic qrels.

- [ ] Run static/import/integrity checks:

```bash
git diff --check
python -m compileall -q trec_mlm scripts .agents/skills/inspect-query-results/scripts
sha256sum protocols/trec_2026_dense_aggregation.md data/dev/llm_dev_data.tsv reports/trec_2026_dense_query_knn_results.json reports/trec_2026_dense_query_knn_per_query.tsv
```

Expected hashes are the protocol, query, dense snapshot, and dense per-query values in this plan's verified starting point and global constraints. Do not hash, parse, stat for content, or otherwise open `data/dev/llm_dev_qrels.txt` during pre-gate verification; all pre-gate tests use synthetic qrel fixtures.

- [ ] Review disk capacity. Require at least 50 GiB free before retaining two complete runs because centers are about 8.93 GiB/run, membership labels about 64 MiB/run, and inner score/rank matrices about 2.31 GiB/run before receipts/reports:

```bash
df -B1 .
```

- [ ] Record the existing advisor approval and independent lockability verdict with this exact compact receipt; it cites the locked protocol as its evidence and does not invent an advisor identity, amendment, or new statistical choice:

```json
{
  "schema": "trec-mlm-dense-aggregation-advisor-receipt-v1",
  "issue": 13,
  "review_date": "2026-07-14",
  "protocol_path": "protocols/trec_2026_dense_aggregation.md",
  "protocol_sha256": "bbe6b50fd3d20efe3213a446f6665cbe0527114f34bf5c1e90236af4f998ad49",
  "advisor_verdict": "approve_with_no_remaining_blocker",
  "independent_consistency_verdict": "lockable_with_no_remaining_blocker",
  "authority": "protocol_status_and_review_preamble",
  "amendment": null
}
```
- [ ] Commit any verification-driven fixes and the review receipt. After this commit, any change to an implementation, dependency-source, or runtime-provenance file invalidates the future feasibility receipt and returns execution to the start of Task 13.

```bash
git add \
  pyproject.toml uv.lock \
  trec_mlm/dense_aggregation*.py trec_mlm/legacy_dense_control.py \
  scripts/export_fixed_mpnet_k50_control.py scripts/build_dense_aggregation_report.py \
  .agents/skills/inspect-query-results/SKILL.md \
  .agents/skills/inspect-query-results/scripts/build_query_explorer.py \
  configs/trec_2026_dense_aggregation_feasibility.json \
  tests/test_dense_aggregation*.py tests/test_legacy_dense_control.py \
  tests/test_query_explorer_skill.py tests/fixtures/dense_aggregation \
  reports/trec_2026_dense_aggregation_advisor_receipt.json
git diff --cached --check
git commit -m "test: freeze dense aggregation implementation"
```

Before committing, inspect `git diff --cached --name-only` and require every path to belong to Tasks 1–12. The explicit list captures any verification-driven fix in the scoped implementation/tests instead of committing only the advisor receipt; do not stage unrelated user work.

### Task 13: Run the Official Qrel-Free Feasibility Gate — Hard Checkpoint

**Files/artifacts:**

- Populate ignored: `artifacts/frozen_dense_input/`
- Populate ignored: `artifacts/dense_aggregation_feasibility/`
- Create tracked compact receipt only on successful validation: `reports/trec_2026_dense_aggregation_feasibility.json`

- [ ] Copy the already validated binary inputs into the ignored local bundle without copying qrels:

```bash
set -euo pipefail
if [[ -e artifacts/frozen_dense_input ]]; then
  uv run --frozen --python 3.13.9 --extra dense-aggregation \
    trec-mlm-dense-aggregation validate-frozen-input \
    --input-dir artifacts/frozen_dense_input
else
  uv run --frozen --python 3.13.9 --extra dense-aggregation \
    trec-mlm-dense-aggregation prepare-frozen-input \
    --source-embedding /home/npatta01/.codex/worktrees/88a454dd-e8be-4b89-923b-57f896689cf9/trec_mlm_2026/artifacts/dense_query_knn_baseline/embeddings/mpnet_base_v2.npy \
    --source-metadata /home/npatta01/.codex/worktrees/88a454dd-e8be-4b89-923b-57f896689cf9/trec_mlm_2026/artifacts/dense_query_knn_baseline/encoder_metadata.json \
    --output-dir artifacts/frozen_dense_input
fi
```

If the frozen-input root already exists from a prior attempt, do not copy over it: run the strict qrel-free validator and reuse it only if every inventory/content hash matches. Otherwise create a new `artifacts/frozen_dense_input-retry-N/` root and a correspondingly suffixed feasibility config; never mutate the old root.

For the first permitted pre-qrel replacement, create that config explicitly and substitute its three suffixed paths in the feasibility command:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation \
  trec-mlm-dense-aggregation build-feasibility-config \
  --queries data/dev/llm_dev_data.tsv \
  --embeddings artifacts/frozen_dense_input-retry-2/mpnet_base_v2.npy \
  --encoder-metadata artifacts/frozen_dense_input-retry-2/encoder_metadata.json \
  --protocol protocols/trec_2026_dense_aggregation.md \
  --output configs/trec_2026_dense_aggregation_feasibility-retry-2.json
```

- [ ] Run the qrel-free input validator and prove the qrel file was not opened through the worker audit receipt.
- [ ] Confirm no other experiment needs the host's CPU/memory, then execute exactly two sequential fresh workers:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation \
  trec-mlm-dense-aggregation feasibility \
  --config configs/trec_2026_dense_aggregation_feasibility.json \
  --output-root artifacts/dense_aggregation_feasibility \
  --compact-receipt reports/trec_2026_dense_aggregation_feasibility.json \
  --attest-exclusive-resources
```

- [ ] Validate both manifests, exact sample lists, 576 fits/5,760 starts per repeat, matching membership/center hashes, isolated empty cache/output roots, per-category timing summaries, exact projections, GNU-time RSS parse, environment, source hashes, and protocol hash.
- [ ] Apply the stop exactly:

  - If any integrity/repeatability check fails, projection is greater than 86,400 seconds, RSS is greater than `16 * 1024**3` bytes, or resource attestation is false: stop. Do not open qrels, execute Task 14, activate the fallback, change code, or claim effectiveness. Present the qrel-free receipt for advisor amendment discussion.
  - If all checks pass: the coordinator atomically writes `reports/trec_2026_dense_aggregation_feasibility.json` from the validated gate object and continues. It must refuse an existing compact-receipt path.

- [ ] Commit the successful compact receipt:

```bash
git add reports/trec_2026_dense_aggregation_feasibility.json
git commit -m "exp: record dense aggregation feasibility gate"
```

For a permitted suffixed attempt, stage the exact config and receipt that were used instead (example for the first replacement):

```bash
git add configs/trec_2026_dense_aggregation_feasibility-retry-2.json \
  reports/trec_2026_dense_aggregation_feasibility-retry-2.json
git commit -m "exp: record dense aggregation feasibility retry 2"
```

### Task 14: Lock the Official Legacy Control and Final Config

**Files/artifacts:**

- Populate ignored: `artifacts/dense_aggregation_legacy_control/`
- Create: `configs/trec_2026_dense_aggregation.json`

- [ ] Revalidate the successful feasibility aggregate against the current protocol, all three source/runtime inventories, numerical environment, and frozen input hashes with this exact read-only command immediately before the first qrel-opening command:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation \
  trec-mlm-dense-aggregation validate-feasibility \
  --config configs/trec_2026_dense_aggregation_feasibility.json \
  --feasibility-root artifacts/dense_aggregation_feasibility \
  --compact-receipt reports/trec_2026_dense_aggregation_feasibility.json
```

Expected: exit 0 and `status=pass`. Any mismatch/stop exits nonzero and ends Task 14 before the pytest below.
- [ ] Rerun the one embargoed historical control regression now that the gate has passed:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation --with pytest pytest -q \
  tests/test_dense_experiment.py::MergedControlRegressionTest::test_official_development_controls_match_merged_report
```

Expected: PASS with the already-published rounded TF-IDF/prior control values.
- [ ] Run two fresh isolated legacy exports:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation \
  python scripts/export_fixed_mpnet_k50_control.py \
  --queries data/dev/llm_dev_data.tsv \
  --qrels data/dev/llm_dev_qrels.txt \
  --embeddings artifacts/frozen_dense_input/mpnet_base_v2.npy \
  --feasibility-receipt artifacts/dense_aggregation_feasibility/manifest.json \
  --output-root artifacts/dense_aggregation_legacy_control \
  --compact-receipt reports/trec_2026_dense_aggregation_legacy_control.json
```

- [ ] Require byte-identical TSV exports, 48,300 canonical rows, exact score/prior round-trip from `float.hex()`, exact rankings, exact fixed-MPNet metrics, and unchanged source/input hashes.
- [ ] Generate the strict official config with the already-tested pre-gate command; it reads both manifested roots, computes the dynamic digests, and refuses a failed/stale gate or control:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation \
  trec-mlm-dense-aggregation build-official-config \
  --feasibility-root artifacts/dense_aggregation_feasibility \
  --legacy-control-root artifacts/dense_aggregation_legacy_control \
  --queries data/dev/llm_dev_data.tsv \
  --qrels data/dev/llm_dev_qrels.txt \
  --embeddings artifacts/frozen_dense_input/mpnet_base_v2.npy \
  --encoder-metadata artifacts/frozen_dense_input/encoder_metadata.json \
  --protocol protocols/trec_2026_dense_aggregation.md \
  --dense-snapshot reports/trec_2026_dense_query_knn_results.json \
  --dense-per-query reports/trec_2026_dense_query_knn_per_query.tsv \
  --output configs/trec_2026_dense_aggregation.json
```

The builder has no hard-coded data/report paths: all seven non-derived inputs are required CLI options above. It resolves them under the repository root, validates their exact locked hashes and both manifested roots, then writes only repository-relative paths. The generated canonical JSON has exactly these keys: `schema`, `query_path`, `qrel_path`, `embedding_path`, `encoder_metadata_path`, `protocol_path`, `feasibility_receipt_path`, `feasibility_receipt_sha256`, `legacy_control_lock_path`, `legacy_control_tsv_path`, `legacy_control_tsv_sha256`, `dense_snapshot_path`, `dense_snapshot_sha256`, `dense_per_query_path`, `dense_per_query_sha256`, and `grid`. `grid` is the sixteen `asdict()` rows in declared order; the loader requires equality to `DENSE_AGGREGATION_GRID`. The formula/K-means values remain protocol/code constants and have no config override.
- [ ] Run the official-config validation test and a qrel-open-order subprocess test.
- [ ] Commit the official config and the compact legacy lock receipt, but not the 48,300-row local control artifact unless a later repository-size review explicitly approves it:

```bash
git add configs/trec_2026_dense_aggregation.json reports/trec_2026_dense_aggregation_legacy_control.json
git commit -m "exp: lock fixed mpnet aggregation control"
```

### Task 15: Run and Validate Authoritative Run A

**Local outputs:**

- `artifacts/dense_aggregation_run_a/construction/`
- `artifacts/dense_aggregation_run_a/evaluation/`

- [ ] Reconfirm at least 25 GiB free for this run and no competing experiment requiring the CPU/memory.
- [ ] Start with nonexistent construction, evaluation, and cache paths. Run one process and one numerical thread:

```bash
mkdir -p artifacts/dense_aggregation_run_a
uv run --frozen --python 3.13.9 --extra dense-aggregation \
  trec-mlm-dense-aggregation run-worker \
  --phase run_a \
  --config configs/trec_2026_dense_aggregation.json \
  --cache-dir artifacts/dense_aggregation_run_a/cache \
  --output-dir artifacts/dense_aggregation_run_a/construction \
  --failure-dir artifacts/dense_aggregation_run_a/reject_invalid \
  --compact-terminal-decision reports/trec_2026_dense_aggregation_terminal_decision.json
```

- [ ] Require the construction loader to pass before any post-freeze evaluation. Verify exactly 104,006 fits, 1,040,060 starts, 322 selections, complete 322-by-150 primary predictions, every inner logical hash, both singleton fit records per query, source hashes, and manifest inventory. Confirm full-development matrices are absent at this stage.
- [ ] Run the separate read-only evaluator:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation \
  trec-mlm-dense-aggregation evaluate \
  --phase run_a \
  --config configs/trec_2026_dense_aggregation.json \
  --construction-root artifacts/dense_aggregation_run_a/construction \
  --output-dir artifacts/dense_aggregation_run_a/evaluation \
  --failure-dir artifacts/dense_aggregation_run_a/reject_invalid \
  --compact-terminal-decision reports/trec_2026_dense_aggregation_terminal_decision.json
```

- [ ] Validate the evaluation manifest, primary freeze receipt, all 16 full-development matrices/hashes, score/rank uniqueness, controls, ablations, paired diagnostics, deployment hint, and provisional gate inputs. Confirm there is no `promote`/`reject` outcome in Run A; Run B repeatability is required before the sole final decision.
- [ ] If construction or evaluation returns a validation failure, follow the Task 16 terminal-receipt commit step immediately and stop; do not start Run B.

### Task 16: Run Authoritative Run B and Compare Both Runs

**Local outputs:**

- `artifacts/dense_aggregation_run_b/construction/`
- `artifacts/dense_aggregation_run_b/evaluation/`
- `artifacts/dense_aggregation_repeat_validation/`

- [ ] Revalidate Run A completely before launching Run B. Start Run B from distinct nonexistent cache/output paths; it may read only the shared frozen inputs, qrels, protocol/code, feasibility receipt, and locked legacy control.
- [ ] Create only the `artifacts/dense_aggregation_run_b` parent and execute the same `run-worker` and `evaluate` commands with `--phase run_b`, distinct `run_b` paths, `artifacts/dense_aggregation_run_b/reject_invalid` as `--failure-dir`, and the same create-exclusive compact-terminal-decision path.
- [ ] Compare both runs:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation \
  trec-mlm-dense-aggregation compare-repeats \
  --config configs/trec_2026_dense_aggregation.json \
  --run-a-construction artifacts/dense_aggregation_run_a/construction \
  --run-a-evaluation artifacts/dense_aggregation_run_a/evaluation \
  --run-b-construction artifacts/dense_aggregation_run_b/construction \
  --run-b-evaluation artifacts/dense_aggregation_run_b/evaluation \
  --output-dir artifacts/dense_aggregation_repeat_validation \
  --compact-terminal-decision reports/trec_2026_dense_aggregation_terminal_decision.json
```

- [ ] Require exact semantic equality outside timestamp/elapsed fields and total counts of 208,012 fits and 2,080,120 starts.
- [ ] If either run or comparison fails integrity, its validation handler atomically writes the local failure root and `reports/trec_2026_dense_aggregation_terminal_decision.json`, with outcome `reject-invalid`, every failed invariant, and unavailable gates explicitly marked. Stop before effectiveness result/report generation, but retain and commit that compact protocol-required terminal decision receipt. An external kill that produces no validation result creates no decision and may follow the narrow operational-interruption rule.

```bash
git add reports/trec_2026_dense_aggregation_terminal_decision.json
git commit -m "exp: record invalid dense aggregation decision"
```

Run this conditional commit only on the terminal validation-failure path, then stop.
- [ ] If repeatability passes, designate Run A as the decision authority in the comparison receipt and retain both local roots.

### Task 17: Produce the Final Tracked Snapshot, Reports, Explorer, and Documentation

**Files:**

- Create atomically: `reports/trec_2026_dense_aggregation/results.json`
- Create atomically: `reports/trec_2026_dense_aggregation/decision.json`
- Create atomically: `reports/trec_2026_dense_aggregation/repeatability.json`
- Create atomically: `reports/trec_2026_dense_aggregation/report.md`
- Create atomically: `reports/trec_2026_dense_aggregation/report.html`
- Create last: `reports/trec_2026_dense_aggregation/manifest.json`
- Modify: `README.md`

- [ ] Build reports only from the validated comparison root:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation \
  python scripts/build_dense_aggregation_report.py \
  --comparison-root artifacts/dense_aggregation_repeat_validation \
  --output-dir reports/trec_2026_dense_aggregation
```

- [ ] Record the raw construction/evaluation manifest hashes, then build the query explorer outside both manifested roots with this exact post-comparison command:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation \
  python .agents/skills/inspect-query-results/scripts/build_query_explorer.py \
  --repo-root . \
  --run-dir artifacts/dense_aggregation_run_a/construction \
  --queries data/dev/llm_dev_data.tsv \
  --qrels data/dev/llm_dev_qrels.txt \
  --neighbors artifacts/dense_aggregation_run_a/construction/neighbors.tsv \
  --clusters artifacts/dense_aggregation_run_a/construction/cluster_assignments.tsv \
  --output artifacts/dense_aggregation_run_a/query_explorer.html \
  --title "TREC 2026 Dense Aggregation Query Explorer" \
  --top-candidates 10 \
  --no-responses
```

The modified explorer CLI requires both `--neighbors` and `--clusters` together for aggregation evidence and validates both against the construction manifest. Rehash both manifests after the command and require byte identity with the recorded values.
- [ ] Review the Markdown and standalone HTML for exact metrics/hashes, accessible tables, controls/ablations, gate outcomes, selection frequencies/margins, zero/slice diagnostics, full-development vs inner uniqueness, runtime/RSS, deployment warning, and exploratory claim scope.
- [ ] Update README with the exact command sequence, artifact locations, stop behavior, final decision, and links to the compact reports. Do not describe `promote` as confirmation; only unseen 2026 test qrels can confirm a gain.
- [ ] Run final verification:

```bash
uv run --frozen --python 3.13.9 --extra dense-aggregation --with pytest pytest -q
git diff --check
git status --short
```

Expected: all tests PASS; no whitespace errors; only intended code, tests, config, protocol/review receipts, compact reports, and documentation are tracked. Large local artifacts remain ignored.

- [ ] Perform a final source-to-protocol coverage review against all sections and all required-test bullets. Confirm no placeholder language, unplanned tuning, hidden qrel access, or missing deliverable.
- [ ] Commit the retained result set:

```bash
git add README.md reports/trec_2026_dense_aggregation
git commit -m "exp: report dense aggregation results"
```

- [ ] Stop with a clean local branch and present the result/decision plus integration choices. Do not push or open a pull request without explicit user authorization.

---

## Gate and Recovery Matrix

| Checkpoint | Pass action | Failure action |
| --- | --- | --- |
| Unit/equivalence/import tests | Freeze implementation hashes | Fix code, rerun all tests; no qrels opened |
| Qrel-free feasibility | Create compact receipt; proceed to legacy lock | Stop before qrels; seek advisor amendment using only qrel-free evidence |
| Legacy two-process lock | Freeze TSV hash into official config | Stop before experimental configuration construction |
| Run A construction validation | Evaluate immutable Run A | `reject-invalid`; diagnose without changing protocol |
| Run A evaluation validation | Start fresh Run B | `reject-invalid`; do not report effectiveness |
| Run B construction/evaluation | Compare repeats | `reject-invalid`; do not report effectiveness |
| Repeat comparison | Apply already-locked decision gates and report | `reject-invalid`; name exact invariant |
| Effectiveness gates with integrity passing | `promote` only if every gate passes; otherwise `reject` | Retain frozen dense baseline; no retuning |

Before the first qrel access, any change to the protocol, an `IMPLEMENTATION_FILES`, `DEPENDENCY_SOURCE_FILES`, or `RUNTIME_PROVENANCE_FILES` member, the numerical environment, or frozen input after Task 13 invalidates the feasibility receipt; return to Task 12 verification and rerun Task 13 from two new empty roots. Once Task 14 first opens qrels, those bytes are frozen and such a change is forbidden rather than a retry path. A pre-qrel legacy-exporter change invalidates Task 14 and every downstream artifact.

When—and only when—the preceding non-result retry rule permits it, use this namespace map. The full map applies to a pre-qrel replacement; phase-scoped rules below govern a post-qrel operational interruption. Subsequent permitted attempts increment the integer consistently:

| First-attempt path | Retry path |
| --- | --- |
| `artifacts/frozen_dense_input/` | `artifacts/frozen_dense_input-retry-2/` |
| `configs/trec_2026_dense_aggregation_feasibility.json` | `configs/trec_2026_dense_aggregation_feasibility-retry-2.json` |
| `artifacts/dense_aggregation_feasibility/` | `artifacts/dense_aggregation_feasibility-retry-2/` |
| `reports/trec_2026_dense_aggregation_feasibility.json` | `reports/trec_2026_dense_aggregation_feasibility-retry-2.json` |
| `artifacts/dense_aggregation_legacy_control/` | `artifacts/dense_aggregation_legacy_control-retry-2/` |
| `reports/trec_2026_dense_aggregation_legacy_control.json` | `reports/trec_2026_dense_aggregation_legacy_control-retry-2.json` |
| `configs/trec_2026_dense_aggregation.json` | `configs/trec_2026_dense_aggregation-retry-2.json` |
| `artifacts/dense_aggregation_run_a/cache/` | `artifacts/dense_aggregation_run_a/cache-retry-2/` |
| `artifacts/dense_aggregation_run_a/construction/` | `artifacts/dense_aggregation_run_a/construction-retry-2/` |
| `artifacts/dense_aggregation_run_a/evaluation/` | `artifacts/dense_aggregation_run_a/evaluation-retry-2/` |
| `artifacts/dense_aggregation_run_b/cache/` | `artifacts/dense_aggregation_run_b/cache-retry-2/` |
| `artifacts/dense_aggregation_run_b/construction/` | `artifacts/dense_aggregation_run_b/construction-retry-2/` |
| `artifacts/dense_aggregation_run_b/evaluation/` | `artifacts/dense_aggregation_run_b/evaluation-retry-2/` |
| `artifacts/dense_aggregation_repeat_validation/` | `artifacts/dense_aggregation_repeat_validation-retry-2/` |
| `reports/trec_2026_dense_aggregation/` | `reports/trec_2026_dense_aggregation-retry-2/` |

For a pre-qrel replacement, substitute every applicable row through all downstream outputs, and generate configs that embed only the replacement input/gate/control hashes. After qrels have first been opened, restart only the exact interrupted command, never a validated upstream component: a killed legacy export gets a suffixed legacy root/compact receipt; an interrupted config publication gets a suffixed config; a killed construction gets suffixed cache+construction paths; a killed evaluation reuses its validated construction and suffixes only evaluation; a killed comparison reuses all four validated roots and suffixes only comparison; a report collision suffixes only the report root. Every downstream command receives those exact explicit component paths. Quarantined/incomplete predecessor paths and the explicit downstream input paths provide the local audit trail; they are process metadata, not embedded in the scientific receipts or cross-run semantic digest. The CLI rejects any unvalidated upstream root. Superseded roots and compact receipts remain intact for audit; README links only the newest fully validated comparison/report root.

This namespace mechanism never turns a scientific or integrity failure into a rerun. A feasibility receipt with `status="stop"` ends execution before qrels and requires advisor amendment discussion. A byte-parity/control failure in Task 14 stops the experiment. After the first qrel access, source, protocol, environment, and input bytes remain frozen; any construction, evaluation, or comparison validation failure yields the declared `reject-invalid` and no replacement authoritative attempt. Only a killed process or publication collision that left no validated result may be restarted with identical frozen bytes in a new namespace; the retained quarantine/incomplete root is the local audit evidence and is never read by an authoritative command.
