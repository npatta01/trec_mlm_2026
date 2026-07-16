# Dense Aggregation HDF5 Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the custom streaming cache with one directly accessed HDF5 file, retain the experiment's scientific and qrel-safety invariants, remove obsolete compatibility/test code, and make PR #20 understandable to a newcomer.

**Architecture:** Feasibility and later construction code open `h5py.File` directly. One focused module contains only fixed-layout conversion and validation functions: initialize datasets, write a row, read a row, and validate a closed file. The writer closes and syncs a temporary file, independently validates it, then uses the existing no-replace rename and artifact-directory publication primitives.

**Tech Stack:** Python 3.13.9, NumPy 2.5.1, SciPy 1.18.0, scikit-learn 1.9.0, h5py 3.15.1, pytest, HDF5, canonical JSON, SHA-256.

## Global Constraints

- Do not open, hash, stat, resolve, or evaluate `data/dev/llm_dev_qrels.txt`.
- Do not run the official 104,006-fit construction or any qrel-using evaluation.
- Preserve locked protocol hash `bbe6b50fd3d20efe3213a446f6665cbe0527114f34bf5c1e90236af4f998ad49`.
- Preserve locked legacy scorer hash `90f8db3abbb2887d8513fb881c631a3843319c29da02e935e1ecdb6c3c85e248`.
- Support one final HDF5 layout only; do not add migration, compatibility readers, schema dispatch, or old-cache conversion.
- Use `h5py.File` directly; do not create a cache/store class, key index, resident-object cache, `release`, `append_timing`, `finalize`, or `finalize_into` API.
- The only project-specific HDF5 functions are `initialize_fit_datasets`, `write_fit_row`, `read_fit_row`, and `validate_fit_file`.
- Use one writer and no readers during writes; do not enable SWMR, distributed locking, or synchronization services.
- Keep filesystem race hardening, canonical config bytes, bounded reads, same-filesystem preflight, close/`fsync`/validate/no-replace-rename publication, and qrel embargo checks.
- Preserve the existing uncommitted feasibility hardening in `trec_mlm/dense_aggregation_feasibility.py` and `tests/test_dense_aggregation_feasibility.py`; edit on top of it and never reset or replace those files wholesale.
- Stage files explicitly. Do not stage unrelated user changes.

## File Structure

- Create `trec_mlm/dense_aggregation_fit_file.py`: fixed HDF5 constants, `FitInventoryRow`, `StoredFitRecord`, and the four conversion/validation functions.
- Create `tests/test_dense_aggregation_fit_file.py`: compact public-behavior tests for HDF5 round trips, validation, corruption, bounded reads, and incomplete writes.
- Modify `pyproject.toml` and `uv.lock`: add and lock `h5py==3.15.1` in `dense-aggregation`.
- Modify `trec_mlm/dense_aggregation_feasibility.py`: direct HDF5 worker writes, logical repeat evidence, final receipt fields, canonical configuration enforcement, and integration with existing race/qrel hardening.
- Modify `tests/test_dense_aggregation_feasibility.py`: migrate fixtures/assertions to one HDF5 file and remove the pretty-config compatibility case.
- Modify `trec_mlm/dense_aggregation_cli.py` and `tests/test_dense_aggregation_cli.py`: validate the final worker layout and receipt fields without importing the old cache.
- Modify `trec_mlm/legacy_dense_control.py` and `tests/test_legacy_dense_control.py`: remove the compatibility wrapper and use the rich final validator.
- Modify `trec_mlm/dense_aggregation_contract.py` and `tests/test_dense_aggregation_contract.py`: replace the implementation inventory path and bind h5py provenance.
- Rewrite `configs/trec_2026_dense_aggregation_feasibility.json`: exact canonical JSON bytes.
- Delete `trec_mlm/dense_aggregation_cache.py`, `tests/test_dense_aggregation_cache.py`, and `tests/test_dense_aggregation_cache_regressions.py`.
- Rewrite `reports/issue_13_dense_aggregation_foundation_review.md`: standalone project/research report and accurate readiness status.

There is currently no authoritative construction module or construction call
site in the branch. This cleanup therefore migrates the implemented feasibility
worker and leaves the same direct HDF5 functions ready for the later
construction task; it does not invent or run that unfinished pipeline here.

---

### Task 1: Add the Direct HDF5 Fit-File Primitive

**Files:**

- Create: `trec_mlm/dense_aggregation_fit_file.py`
- Create: `tests/test_dense_aggregation_fit_file.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**

- Consumes: `CanonicalClusterFit`, `FitKey`, and `TargetAssignment` from `trec_mlm.dense_aggregation_clustering`; canonical membership/array hashing from `trec_mlm.dense_aggregation_hashing`.
- Produces: `FitInventoryRow`, `StoredFitRecord`, `initialize_fit_datasets`, `write_fit_row`, `read_fit_row`, and `validate_fit_file` with the signatures below.

- [ ] **Step 1: Write the failing public round-trip test**

Create `tests/test_dense_aggregation_fit_file.py` with a small canonical inventory containing one pair row and one single row. Generate 322 query IDs, use embedding width 2, and construct fits with exact 320/321 support lengths. The test must interact through direct `h5py.File` handles:

```python
def test_direct_hdf5_pair_and_single_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "fits.partial.h5"
    query_ids = tuple(f"q{index:03d}" for index in range(322))
    inventory = (
        FitInventoryRow("pair", 0, (0, 1)),
        FitInventoryRow("single", 0, (0,)),
    )
    metadata = _metadata(query_ids)

    with h5py.File(path, "x") as h5:
        initialize_fit_datasets(
            h5, metadata=metadata, inventory=inventory, embedding_width=2
        )
        pair_fit, pair_assignments = _fit_and_assignments(
            query_ids=query_ids,
            excluded=(0, 1),
            cluster_count=20,
        )
        write_fit_row(
            h5, 0, pair_fit, pair_assignments, started_ns=time.perf_counter_ns()
        )
        single_fit, single_assignments = _fit_and_assignments(
            query_ids=query_ids,
            excluded=(0,),
            cluster_count=40,
        )
        write_fit_row(
            h5, 1, single_fit, single_assignments,
            started_ns=time.perf_counter_ns(),
        )

        assert h5["fits/c20/complete"][0] == 1
        assert h5["fits/c40/complete"][1] == 1
        pair_loaded = read_fit_row(h5, 0, 20)
        single_loaded = read_fit_row(h5, 1, 40)

    assert np.array_equal(pair_loaded.fit.centers, pair_fit.centers)
    assert np.array_equal(pair_loaded.fit.labels, pair_fit.labels)
    assert pair_loaded.fit.memberships == pair_fit.memberships
    assert pair_loaded.assignments == pair_assignments
    assert np.array_equal(single_loaded.fit.centers, single_fit.centers)
    assert single_loaded.fit.memberships == single_fit.memberships
    assert single_loaded.assignments == single_assignments
```

The `_metadata` helper must return exactly these keys: `query_ids`, `query_input_sha256`, `embedding_sha256`, `similarity_sha256`, `clustering_contract_sha256`, `implementation_inventory_sha256`, `dependency_inventory_sha256`, `runtime_inventory_sha256`, `numerical_environment_sha256`, and `protocol_sha256`. Every hash is lowercase 64-character hex.

- [ ] **Step 2: Run the round-trip test and verify the expected failure**

Run:

```bash
pytest -q tests/test_dense_aggregation_fit_file.py::test_direct_hdf5_pair_and_single_round_trip
```

Expected: collection fails with `ModuleNotFoundError: No module named 'trec_mlm.dense_aggregation_fit_file'`.

- [ ] **Step 3: Add the pinned HDF5 dependency**

Add this exact entry to `[project.optional-dependencies].dense-aggregation`:

```toml
"h5py==3.15.1",
```

Then run:

```bash
uv lock
```

Expected: `uv.lock` contains an `h5py` package record and the project `dense-aggregation` extra references it. Do not update unrelated package pins.

- [ ] **Step 4: Implement the domain records and exact four functions**

Create `trec_mlm/dense_aggregation_fit_file.py` with these public definitions and no storage class:

```python
FIT_FILE_SCHEMA = "trec-mlm-dense-aggregation-fit-file-final"
CLUSTER_COUNTS = (20, 40)

@dataclass(frozen=True, slots=True)
class FitInventoryRow:
    exclusion_kind: Literal["pair", "single"]
    inventory_index: int
    excluded_query_indices: tuple[int, ...]

@dataclass(frozen=True, slots=True)
class StoredFitRecord:
    fit: CanonicalClusterFit
    assignments: tuple[TargetAssignment, ...]
    exclusion_kind: Literal["pair", "single"]
    inventory_index: int
    excluded_query_ids: tuple[str, ...]
    elapsed_ns: int

def initialize_fit_datasets(
    h5: h5py.File,
    *,
    metadata: Mapping[str, object],
    inventory: Sequence[FitInventoryRow],
    embedding_width: int,
) -> None: ...

def write_fit_row(
    h5: h5py.File,
    inventory_row: int,
    fit: CanonicalClusterFit,
    assignments: Sequence[TargetAssignment],
    *,
    started_ns: int,
) -> None: ...

def read_fit_row(
    h5: h5py.File,
    inventory_row: int,
    cluster_count: Literal[20, 40],
) -> StoredFitRecord: ...

def validate_fit_file(
    path: Path,
    *,
    expected_metadata: Mapping[str, object],
    expected_inventory: Sequence[FitInventoryRow],
) -> None: ...
```

`initialize_fit_datasets` must reject a nonempty HDF5 file, validate exact metadata/inventory, write `canonical_json(metadata)` into `metadata/canonical_json` as `uint8`, and create the exact design-spec layout. Use explicit little-endian dtypes (`<f4`, `<f8`, `<u2`, `<u4`, `<i2`, `u1`), chunk the large fit datasets on the first dimension, enable `fletcher32=True`, and do not enable compression.

`write_fit_row` must derive the group from `fit.key.cluster_count`, validate the complete support tuple and assignment targets against inventory, validate canonical hashes before writing, write all data, set `elapsed_ns = time.perf_counter_ns() - started_ns`, and set the slot's `complete` byte last. It must not retain `fit` or array references.

`read_fit_row` must slice only one row, reject an incomplete slot, remove the pair-label sentinel, reconstruct support IDs and memberships, recompute hashes and inertia relationships, and return `StoredFitRecord`.

Initially implement `validate_fit_file` as a complete-file check that delegates row reconstruction to `read_fit_row`; Task 2 adds adversarial validation and bounded-read assertions.

- [ ] **Step 5: Run the focused tests**

Run:

```bash
pytest -q tests/test_dense_aggregation_fit_file.py
```

Expected: all initial round-trip, duplicate-slot, wrong-support, wrong-shape, and nonfinite-value tests pass.

- [ ] **Step 6: Confirm direct library use and dependency provenance**

Run:

```bash
rg -n "class .*Store|class .*Cache|diskcache|SWMR|swmr" trec_mlm/dense_aggregation_fit_file.py
uv run --extra dense-aggregation python -c 'import h5py; print(h5py.__version__)'
```

Expected: the search returns no matches and Python prints `3.15.1`.

- [ ] **Step 7: Commit Task 1**

Stage only:

```bash
git add pyproject.toml uv.lock trec_mlm/dense_aggregation_fit_file.py tests/test_dense_aggregation_fit_file.py
git diff --cached --check
git commit -m "feat: store aggregation fits directly in HDF5"
```

Expected: one commit containing only the four listed files.

---

### Task 2: Validate, Sync, and Publish One Immutable HDF5 File

**Files:**

- Modify: `trec_mlm/dense_aggregation_fit_file.py`
- Modify: `tests/test_dense_aggregation_fit_file.py`
- Modify: `trec_mlm/dense_aggregation_artifacts.py`
- Modify: `tests/test_dense_aggregation_artifacts.py`

**Interfaces:**

- Consumes: the four direct HDF5 functions from Task 1 and existing `rename_noreplace`/manifest helpers.
- Produces: strict full-file validation and a tested close/OS-`fsync`/validate/no-replace-rename sequence used inline by Task 3. No fifth HDF5 abstraction is introduced.

- [ ] **Step 1: Add failing final-file validation tests**

Add tests with these exact behavioral assertions:

```python
def test_validate_fit_file_rejects_incomplete_file(valid_partial_file: Path) -> None:
    with pytest.raises(ValueError, match="file is incomplete"):
        validate_fit_file(
            valid_partial_file,
            expected_metadata=_expected_metadata(),
            expected_inventory=_expected_inventory(),
        )

def test_validate_fit_file_rejects_mutated_center(valid_final_file: Path) -> None:
    with h5py.File(valid_final_file, "r+") as h5:
        h5["fits/c20/centers"][0, 0, 0] += np.float32(1.0)
    with pytest.raises(ValueError, match="center hash"):
        validate_fit_file(
            valid_final_file,
            expected_metadata=_expected_metadata(),
            expected_inventory=_expected_inventory(),
        )

def test_validate_fit_file_rejects_truncation(valid_final_file: Path) -> None:
    size = valid_final_file.stat().st_size
    with valid_final_file.open("r+b") as handle:
        handle.truncate(size - 1)
    with pytest.raises((OSError, ValueError)):
        validate_fit_file(
            valid_final_file,
            expected_metadata=_expected_metadata(),
            expected_inventory=_expected_inventory(),
        )
```

Also test unknown groups/datasets/attributes, wrong dtype/shape/chunk/checksum settings, noncanonical metadata bytes, wrong inventory, assignment mutation, membership-hash mutation, and an incomplete slot.

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```bash
pytest -q tests/test_dense_aggregation_fit_file.py -k 'validate or mutated or truncation'
```

Expected: new adversarial cases fail because Task 1 does not yet enforce every exact-layout and mutation boundary.

- [ ] **Step 3: Complete strict bounded validation**

Make `validate_fit_file`:

1. `lstat` the path and require a regular file.
2. Open only with `h5py.File(path, "r")`.
3. Require exact root attributes and exact group/dataset names.
4. Require schema `trec-mlm-dense-aggregation-fit-file-final`, root `complete == 1`, exact inventory count, and exact embedding width.
5. Require every dataset's dtype, shape, chunks, Fletcher32 setting, and absence of compression to match the final constants.
6. Require metadata bytes to equal `canonical_json(expected_metadata)` exactly.
7. Require stored inventory arrays to equal `expected_inventory` exactly.
8. Iterate inventory rows and both cluster counts, calling `read_fit_row` once per slot and retaining only the current record.
9. Convert HDF5 `OSError`/`KeyError`/`TypeError` into a contextual `ValueError` whose message begins `invalid HDF5 fit file:` without hiding the scientific mismatch message.

Instrument `read_fit_row` in a test and require full validation's maximum live record count to be one. Patch `h5py.Dataset.__getitem__` or a local slice helper and fail if any fit dataset is read with `[:]` or without a first-dimension integer index; completion/inventory/hash metadata arrays may be read as bounded small arrays.

- [ ] **Step 4: Add failing sync/publication ordering tests**

In `tests/test_dense_aggregation_artifacts.py`, exercise the concrete sequence used by callers:

```python
with h5py.File(partial, "x") as h5:
    _write_complete_fixture(h5)

with partial.open("rb") as handle:
    os.fsync(handle.fileno())
validate_fit_file(
    partial,
    expected_metadata=metadata,
    expected_inventory=inventory,
)
rename_noreplace(partial, destination)
```

Patch `os.fsync`, `validate_fit_file`, and `rename_noreplace` to record calls and assert the order is `close`, `file_fsync`, `validate`, `rename`, `directory_fsync`. Add an existing/racing destination case and require no overwrite.

- [ ] **Step 5: Implement the inline publication sequence without a new API**

Do not add a `publish_fit_store` or similar wrapper. Reuse `rename_noreplace` and the existing directory-`fsync` helper or add a narrowly named private `_fsync_directory(path: Path) -> None` to `dense_aggregation_artifacts.py` if the operation is currently duplicated. The exact caller sequence remains visible:

```python
with partial_path.open("rb") as handle:
    os.fsync(handle.fileno())
validate_fit_file(
    partial_path,
    expected_metadata=metadata,
    expected_inventory=inventory,
)
rename_noreplace(partial_path, staging / "fits.h5")
fsync_directory(staging)
```

- [ ] **Step 6: Run HDF5 and artifact tests**

Run:

```bash
pytest -q tests/test_dense_aggregation_fit_file.py tests/test_dense_aggregation_artifacts.py
```

Expected: all tests pass, including corruption, bounded read, interrupted write, and publication-race cases.

- [ ] **Step 7: Commit Task 2**

```bash
git add trec_mlm/dense_aggregation_fit_file.py tests/test_dense_aggregation_fit_file.py trec_mlm/dense_aggregation_artifacts.py tests/test_dense_aggregation_artifacts.py
git diff --cached --check
git commit -m "fix: validate HDF5 fit artifacts before publication"
```

---

### Task 3: Migrate the Qrel-Free Feasibility Pipeline and Final Config

**Files:**

- Modify: `trec_mlm/dense_aggregation_feasibility.py`
- Modify: `tests/test_dense_aggregation_feasibility.py`
- Modify: `trec_mlm/dense_aggregation_cli.py`
- Modify: `tests/test_dense_aggregation_cli.py`
- Rewrite: `configs/trec_2026_dense_aggregation_feasibility.json`

**Interfaces:**

- Consumes: direct HDF5 functions from Tasks 1–2, existing atomic artifact helpers, and the current uncommitted feasibility race/qrel validation work.
- Produces: one-file feasibility worker artifacts, final canonical configuration enforcement, `fit_file_sha256` for per-artifact integrity, and `fit_evidence_sha256` for cross-repeat scientific equivalence.

- [ ] **Step 1: Replace cache-shaped feasibility tests with final HDF5 expectations**

Change worker layout expectations to exactly:

```python
_WORKER_OUTPUT_FILES = (
    "environment.json",
    "feasibility_receipt.json",
    "fits.h5",
)
```

Update `FeasibilityReceipt` tests so `_RECEIPT_KEYS` contains
`fit_file_sha256` and `fit_evidence_sha256` instead of
`fit_receipt_sha256`. Require:

```python
assert receipt.fit_file_sha256 == raw_file_sha256(output_dir / "fits.h5")
assert len(receipt.fit_evidence_sha256) == 64
assert repeat_a.fit_evidence_sha256 == repeat_b.fit_evidence_sha256
assert len(repeat_a.fit_file_sha256) == len(repeat_b.fit_file_sha256) == 64
```

Raw HDF5 file hashes are validated against each repeat's manifest but are not
required to match across repeats. Only `fit_evidence_sha256` is a cross-repeat
scientific-equivalence field.

Replace `SyntheticCache` monkeypatches with small synthetic `fit_canonical_clusters` and `assign_targets` patches while allowing real direct HDF5 writes. Delete tests that inspect cache headers, JSONL rows, timing files, or `StreamingFitCache` methods; retain their scientific assertions through `read_fit_row` and logical evidence extracted from `fits.h5`.

Change the worker CLI and function surface rather than retaining cache terminology:

```python
def run_feasibility_worker(
    config: FeasibilityConfig,
    *,
    fit_work_file: Path,
    output_dir: Path,
    exclusive_resource_attested: bool,
) -> FeasibilityReceipt: ...
```

The `feasibility-worker` parser requires `--fit-work-file` and no longer accepts
`--cache-dir`. Tests must assert that the obsolete option is rejected.

- [ ] **Step 2: Add the canonical-config failing test and remove compatibility acceptance**

Delete `test_validate_feasibility_manifest_accepts_pretty_printed_task7_raw_hash`. Add:

```python
def test_load_feasibility_config_rejects_noncanonical_json(tmp_path: Path) -> None:
    path = tmp_path / "feasibility.json"
    path.write_text(json.dumps(_valid_config_value(), indent=2) + "\n")
    with pytest.raises(ValueError, match="canonical JSON bytes"):
        load_feasibility_config(path)
```

Require `load_feasibility_config` to compare captured source bytes with
`canonical_json(payload) + b"\n"` before constructing `FeasibilityConfig`.

- [ ] **Step 3: Run targeted tests and verify failures**

Run:

```bash
pytest -q \
  tests/test_dense_aggregation_feasibility.py \
  tests/test_dense_aggregation_cli.py \
  -k 'worker or repeat or config or manifest'
```

Expected: failures reference the old cache layout/API and acceptance of noncanonical configuration bytes.

- [ ] **Step 4: Implement direct HDF5 feasibility writing**

In `run_feasibility_worker`:

1. Replace `preflight_atomic_roots` with `preflight_atomic_paths(*, fit_work_file: Path, output_dir: Path)` and keep it as the first filesystem action. It requires a nonexistent `.partial.h5` file, a nonexistent output directory, nonoverlapping paths, and parents on the same filesystem.
2. Use the explicit `fit_work_file` directly and never create an old cache directory or shard tree.
3. Build `FitInventoryRow` values in the exact sampled pair-then-single order; each logical row receives `C=20` and `C=40` slots.
4. Build final file metadata from the former cache header without `schema`, `shard_schemas`, `expected_inventory`, or `expected_counts`; add locked `protocol_sha256`.
5. Open `h5py.File(partial_path, "x")`, call `initialize_fit_datasets`, execute the existing measured fit loop, and call `write_fit_row(h5, inventory_row, fit, assignments, started_ns=start_ns)` after assignment.
6. Read the just-written scalar `elapsed_ns` dataset cell and append that value with `fit.n_iter` to the in-memory `timing_rows`; do not write a timing receipt file or reconstruct the full fit merely to obtain timing.
7. After both cluster slots for every row are complete, set root `complete=1`, flush, exit the context, `fsync` the closed file, call `validate_fit_file`, no-replace rename it to `staging / "fits.h5"`, and `fsync` the staging directory.
8. Compute `fit_file_sha256 = raw_file_sha256(staging / "fits.h5")`.
9. Compute `fit_evidence_sha256` as SHA-256 of canonical JSON over the ordered logical evidence fields formerly in `_FIT_COMPARISON_FIELDS`, plus `elapsed_ns` excluded from this cross-repeat digest. Iterate with bounded `read_fit_row` calls.

Rename receipt provenance field `cache_root` to `fit_work_file`. Update freshness
checks to require the two repeat work files and output roots to be canonical,
distinct, nonoverlapping paths. Do not preserve the old provenance field.

Define the experiment-only evidence conversion in this module:

```python
def _logical_fit_evidence(record: StoredFitRecord) -> dict[str, object]:
    return {
        "exclusion_kind": record.exclusion_kind,
        "inventory_index": record.inventory_index,
        "cluster_count": record.fit.key.cluster_count,
        "support_query_ids_sha256": support_query_ids_sha256(
            record.fit.key.support_query_ids
        ),
        "embedding_sha256": record.fit.key.embedding_sha256,
        "clustering_contract_sha256": (
            record.fit.key.clustering_contract_sha256
        ),
        "fit_membership_sha256": record.fit.fit_membership_sha256,
        "cluster_membership_sha256": list(
            record.fit.cluster_membership_sha256
        ),
        "center_sha256": record.fit.center_sha256,
        "assignments": [
            {
                "query_id": assignment.query_id,
                "canonical_cluster_id": assignment.canonical_cluster_id,
                "squared_distance_hex": assignment.squared_distance.hex(),
                "cluster_support": assignment.cluster_support,
            }
            for assignment in record.assignments
        ],
    }
```

Frame each `canonical_json(_logical_fit_evidence(record))` with its eight-byte
big-endian length before updating the SHA-256 accumulator, preventing ambiguous
concatenation without creating a receipt file.

The loop must continue to check the numerical environment immediately before the first fit and after the final fit, and it must delete its local fit/assignment/support references after each write.

- [ ] **Step 5: Migrate repeat validation and CLI loading**

Replace every `StreamingFitCache.open_readonly`, cache-header read, JSONL receipt read, and timing-file read with:

```python
fits_path = output_root / "fits.h5"
validate_fit_file(
    fits_path,
    expected_metadata=expected_metadata,
    expected_inventory=expected_inventory,
)
with h5py.File(fits_path, "r") as h5:
    records = tuple(
        logical_fit_evidence(read_fit_row(h5, row, cluster_count))
        for row in range(len(expected_inventory))
        for cluster_count in (20, 40)
    )
```

Do not retain the tuple in production: stream each canonical evidence row into a SHA-256 accumulator and keep only the compact fields needed for pairwise repeat checks. Timing summaries come from `StoredFitRecord.elapsed_ns` and `fit.n_iter`.

Update CLI `_load_worker_output` to validate the three-file layout, raw HDF5 hash, receipt schema, and manifest before returning. It must not perform a second full HDF5 scan; repeat evaluation owns that validation boundary.

- [ ] **Step 6: Preserve and finish the existing feasibility hardening**

Review the pre-existing dirty diff line by line. Retain descriptor-based bounded reads, inode/race checks, lexical path validation, aggregate drift windows, qrel-leaf embargo, and the public `validate_feasibility_manifest`. Remove only logic/tests whose sole purpose is accepting pretty Task-7 config bytes or parsing old cache shards.

Run the previously added adversarial tests explicitly:

```bash
pytest -q tests/test_dense_aggregation_feasibility.py -k \
'fifo or inode_swap or qrel or drift or fresh_passing_retry or canonical'
```

Expected: all selected tests pass and no qrel leaf access occurs.

- [ ] **Step 7: Rewrite the tracked config as exact canonical bytes**

The complete file must be this one line plus newline:

```json
{"embedding_path":"artifacts/frozen_dense_input/mpnet_base_v2.npy","encoder_metadata_path":"artifacts/frozen_dense_input/encoder_metadata.json","protocol_path":"protocols/trec_2026_dense_aggregation.md","query_path":"data/dev/llm_dev_data.tsv","schema":"trec-mlm-dense-aggregation-feasibility-config-v1"}
```

Verify:

```bash
python - <<'PY'
from pathlib import Path
from trec_mlm.dense_aggregation_hashing import canonical_json
import json
p = Path("configs/trec_2026_dense_aggregation_feasibility.json")
b = p.read_bytes()
assert b == canonical_json(json.loads(b)) + b"\n"
print(len(b))
PY
```

Expected: the assertion passes.

- [ ] **Step 8: Run feasibility and CLI suites**

```bash
pytest -q tests/test_dense_aggregation_feasibility.py tests/test_dense_aggregation_cli.py
```

Expected: all tests pass without opening official qrels or invoking real 576-fit feasibility work.

- [ ] **Step 9: Commit Task 3 with explicit mixed-worktree staging**

```bash
git add \
  trec_mlm/dense_aggregation_feasibility.py \
  tests/test_dense_aggregation_feasibility.py \
  trec_mlm/dense_aggregation_cli.py \
  tests/test_dense_aggregation_cli.py \
  configs/trec_2026_dense_aggregation_feasibility.json
git diff --cached --check
git diff --cached --name-only
git commit -m "feat: run aggregation feasibility through HDF5"
```

Expected: the commit includes both the previously uncommitted feasibility hardening and its HDF5 migration, with no unrelated file.

---

### Task 4: Remove the Cache and Compatibility Surface

**Files:**

- Delete: `trec_mlm/dense_aggregation_cache.py`
- Delete: `tests/test_dense_aggregation_cache.py`
- Delete: `tests/test_dense_aggregation_cache_regressions.py`
- Modify: `trec_mlm/dense_aggregation_contract.py`
- Modify: `tests/test_dense_aggregation_contract.py`
- Modify: `trec_mlm/legacy_dense_control.py`
- Modify: `tests/test_legacy_dense_control.py`

**Interfaces:**

- Consumes: Task 3 has removed all production and CLI imports of `StreamingFitCache`.
- Produces: one final implementation inventory containing `dense_aggregation_fit_file.py`, and one rich legacy lock validator with no compatibility wrapper.

- [ ] **Step 1: Add failing final-surface tests**

Change contract expectations to:

```python
assert "trec_mlm/dense_aggregation_fit_file.py" in IMPLEMENTATION_FILES
assert "trec_mlm/dense_aggregation_cache.py" not in IMPLEMENTATION_FILES
```

Change the legacy public-surface test to require only:

```python
assert callable(module.validate_legacy_control_lock_bindings)
assert not hasattr(module, "validate_legacy_control_lock")
```

Update every caller/test of `validate_legacy_control_lock` to call:

```python
validate_legacy_control_lock_bindings(
    lock_path,
    expected_lock_sha256=raw_file_sha256(lock_path),
    expected_tsv_sha256=expected_tsv_sha256,
)
```

Assertions use the rich `LegacyControlLockValidation` fields directly.

- [ ] **Step 2: Run the final-surface tests and verify failure**

```bash
pytest -q \
  tests/test_dense_aggregation_contract.py \
  tests/test_legacy_dense_control.py \
  -k 'implementation or public or validate_legacy_control_lock'
```

Expected: failures show the old cache path and compatibility wrapper still exist.

- [ ] **Step 3: Delete old implementation and exhaustive tests**

Delete the three cache files in full. Do not preserve aliases, import shims, old constants, fixtures, format readers, mutation matrices, or regression compatibility tests.

Replace `trec_mlm/dense_aggregation_cache.py` with nothing; the path must be absent, not an empty forwarding module.

- [ ] **Step 4: Update the frozen implementation inventory**

In `IMPLEMENTATION_FILES`, replace exactly:

```python
"trec_mlm/dense_aggregation_cache.py",
```

with:

```python
"trec_mlm/dense_aggregation_fit_file.py",
```

Update the exact contract/inventory hash fixtures only after reviewing the diff and confirming the new file and dependency changes are the intended cause.

- [ ] **Step 5: Remove the legacy compatibility wrapper**

Delete `validate_legacy_control_lock`. Keep `validate_legacy_control_lock_bindings` as the sole validator and retain all filename/hash/manifest/race validation. Remove only tests specific to the old two-positional-argument wrapper; retain rich-validator mutation and race coverage.

- [ ] **Step 6: Prove there is no old cache or wrapper reference**

```bash
rg -n "StreamingFitCache|dense_aggregation_cache|validate_legacy_control_lock\(" trec_mlm tests || true
test ! -e trec_mlm/dense_aggregation_cache.py
test ! -e tests/test_dense_aggregation_cache.py
test ! -e tests/test_dense_aggregation_cache_regressions.py
```

Expected: `rg` prints no matches and all three absence checks succeed.

- [ ] **Step 7: Run the affected suites**

```bash
pytest -q \
  tests/test_dense_aggregation_fit_file.py \
  tests/test_dense_aggregation_contract.py \
  tests/test_legacy_dense_control.py \
  tests/test_dense_aggregation_feasibility.py \
  tests/test_dense_aggregation_cli.py
```

Expected: all tests pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add \
  trec_mlm/dense_aggregation_cache.py \
  tests/test_dense_aggregation_cache.py \
  tests/test_dense_aggregation_cache_regressions.py \
  trec_mlm/dense_aggregation_contract.py \
  tests/test_dense_aggregation_contract.py \
  trec_mlm/legacy_dense_control.py \
  tests/test_legacy_dense_control.py
git diff --cached --check
git commit -m "refactor: remove custom aggregation cache"
```

Expected: Git records three deletions and the final inventory/validator changes.

---

### Task 5: Rewrite the Standalone Pull Request Report

**Files:**

- Rewrite: `reports/issue_13_dense_aggregation_foundation_review.md`
- Modify: `.superpowers/sdd/issue13-pr-body.md` if the ignored local PR-body source exists

**Interfaces:**

- Consumes: verified test counts, diff statistics, design decisions, baseline metrics, and current official-run status.
- Produces: one mergeable tracked report understandable without issue history or this conversation, plus an accurate PR body linking it.

- [ ] **Step 1: Replace the report's opening with first-principles context**

The first screen must state:

```markdown
# Issue 13: Dense Aggregation Foundation Review

## Executive summary

This project ranks 150 candidate LLM responses for each of 322 development
queries. The current MPNet nearest-neighbor baseline reaches 0.3599 mean
nDCG@10 and 0.4041 mean reciprocal rank. It selected k=50 in 320 of 322 folds,
but 116 queries still have zero nDCG@10.

Issue 13 tests whether combining local-neighbor evidence, semantic-domain
evidence, and a global prior can improve that ranking without leakage. This PR
builds and validates the qrel-safe infrastructure needed to run that experiment.
It does not yet contain an effectiveness result.
```

Define qrels, nDCG@10, MRR, fold, leakage, MPNet, nearest neighbor, semantic domain, and global prior in plain language.

- [ ] **Step 2: Add exact “where we started / what we are testing” sections**

Include a table with baseline nDCG `0.3599`, MRR `0.4041`, `k=50` selected in `320/322` folds, and `116` zero-nDCG queries. State the hypothesis and locked success criteria exactly:

- paired nDCG confidence-interval lower bound above zero;
- MRR delta at least `-0.005`;
- fewer than `116` zero-nDCG queries;
- positive gain on low-density queries;
- all integrity checks pass.

Explain that “180k responses” is not the immediate unit: effectiveness is the locked `322 × 150` ranking task, while authoritative construction requires `104,006` fits and `1,040,060` clustering starts.

- [ ] **Step 3: Report implementation and evidence accurately**

Add a status table with these rows and no stronger claim:

| Area | Status |
|---|---|
| Scientific protocol | Frozen |
| Scoring/clustering primitives | Implemented and unit tested |
| Direct HDF5 fit artifact | Implemented and unit tested |
| Qrel-free feasibility software | Implemented and unit tested |
| Official feasibility gate | Not run |
| Authoritative construction | Not run |
| Ranking evaluation | Not run |
| Effectiveness finding | None yet |

Include the exact verification commands and observed counts from Task 6. Clearly separate infrastructure tests from ranking-quality evidence.

- [ ] **Step 4: Add limitations, next decision, and navigation**

State that merge means the experiment foundation is reviewable, not that the method wins. The next decision after merge is whether to authorize the two-repeat qrel-free feasibility run; qrels remain embargoed until that gate and the legacy control lock pass.

Link the frozen protocol, HDF5 design spec, implementation plan, relevant source entry points, and PR. Ensure every repository link is relative and readable on GitHub.

- [ ] **Step 5: Check report accessibility and unsupported claims**

```bash
rg -n "TBD|TODO|180,?000 responses evaluated|effectiveness.*pass|all tests pass" \
  reports/issue_13_dense_aggregation_foundation_review.md
python - <<'PY'
from pathlib import Path
p = Path("reports/issue_13_dense_aggregation_foundation_review.md")
text = p.read_text()
for required in (
    "0.3599", "0.4041", "320/322", "116", "104,006", "1,040,060",
    "Not run", "None yet", "nDCG", "MRR", "leakage",
):
    assert required in text, required
print(len(text.split()), "words")
PY
```

Expected: the first search has no matches unless “all tests pass” is explicitly tied to a dated command and count; the Python check passes.

- [ ] **Step 6: Commit Task 5**

```bash
git add reports/issue_13_dense_aggregation_foundation_review.md
git diff --cached --check
git commit -m "docs: make dense aggregation review standalone"
```

Do not stage `.superpowers/sdd/issue13-pr-body.md`; it is an ignored local helper for the later GitHub update.

---

### Task 6: Verify, Review, and Update the Single Draft PR

**Files:**

- Modify only if verification finds scoped defects: files changed in Tasks 1–5.
- Update remote metadata: draft PR #20 body.

**Interfaces:**

- Consumes: the completed implementation, tests, report, and existing branch history.
- Produces: evidence-backed merge recommendation for one draft PR, not multiple PRs.

- [ ] **Step 1: Run cheap static and safety checks first**

```bash
git diff --check origin/codex/issue-13-dense-aggregation...HEAD
rg -n "StreamingFitCache|dense_aggregation_cache|fit_receipts\.jsonl|timings\.jsonl|shard_schemas|fit_count_increment|start_count_increment" trec_mlm tests || true
rg -n "llm_dev_qrels\.txt" trec_mlm/dense_aggregation_fit_file.py tests/test_dense_aggregation_fit_file.py
python -m compileall -q trec_mlm tests
```

Expected: no whitespace errors; no old-cache references; the new HDF5 module/tests contain no qrel path; compilation succeeds.

- [ ] **Step 2: Run the focused final HDF5 path**

```bash
pytest -q \
  tests/test_dense_aggregation_fit_file.py \
  tests/test_dense_aggregation_artifacts.py \
  tests/test_dense_aggregation_feasibility.py \
  tests/test_dense_aggregation_cli.py \
  tests/test_dense_aggregation_contract.py \
  tests/test_legacy_dense_control.py
```

Expected: all selected tests pass. Record the exact test count and duration in the standalone report.

- [ ] **Step 3: Run the complete dense-aggregation/qrel-safe suite**

```bash
pytest -q \
  tests/test_dense_aggregation.py \
  tests/test_dense_aggregation_artifacts.py \
  tests/test_dense_aggregation_cli.py \
  tests/test_dense_aggregation_clustering.py \
  tests/test_dense_aggregation_contract.py \
  tests/test_dense_aggregation_experiment.py \
  tests/test_dense_aggregation_feasibility.py \
  tests/test_dense_aggregation_fit_file.py \
  tests/test_dense_aggregation_hashing.py \
  tests/test_dense_aggregation_runtime.py \
  tests/test_legacy_dense_control.py
```

Expected: all selected tests pass without accessing official qrels or running real clustering workloads.

- [ ] **Step 4: Run the full repository suite**

```bash
pytest -q
```

Expected: all tests pass. If collection is blocked only by an unavailable optional Torch environment, record the exact error and rely on the complete dense-aggregation/qrel-safe command rather than installing or changing Torch.

- [ ] **Step 5: Inspect scope, size reduction, and secrets**

```bash
git status --short
git diff --stat origin/main...HEAD
git diff --numstat origin/main...HEAD
git log --oneline origin/main..HEAD
rg -n -i '(api[_-]?key|secret|password|token)[[:space:]]*[:=]' \
  docs/superpowers/specs/2026-07-15-dense-aggregation-hdf5-fit-store-design.md \
  docs/superpowers/plans/2026-07-15-dense-aggregation-hdf5-cleanup.md \
  reports/issue_13_dense_aggregation_foundation_review.md || true
```

Expected: no unexpected uncommitted files, no secrets, one PR, old cache code/tests absent, and a material net reduction from the pre-cleanup cache surface.

- [ ] **Step 6: Request independent code review**

Use `superpowers:requesting-code-review` against the complete branch diff. The reviewer must check:

- direct `h5py` usage with no reinvented cache API;
- exact layout and bounded reads;
- scientific reconstruction/hash equivalence;
- interrupted-write and atomic publication behavior;
- preservation of feasibility race/qrel hardening;
- removal of compatibility paths and redundant tests;
- report accuracy and newcomer accessibility.

Address only verified actionable findings, rerunning the smallest relevant test followed by Steps 2–4 as appropriate.

- [ ] **Step 7: Commit verification-driven fixes or report updates**

If verification changes scoped files:

```bash
git add \
  pyproject.toml uv.lock \
  configs/trec_2026_dense_aggregation_feasibility.json \
  trec_mlm/dense_aggregation_artifacts.py \
  trec_mlm/dense_aggregation_cli.py \
  trec_mlm/dense_aggregation_contract.py \
  trec_mlm/dense_aggregation_feasibility.py \
  trec_mlm/dense_aggregation_fit_file.py \
  trec_mlm/legacy_dense_control.py \
  tests/test_dense_aggregation_artifacts.py \
  tests/test_dense_aggregation_cli.py \
  tests/test_dense_aggregation_contract.py \
  tests/test_dense_aggregation_feasibility.py \
  tests/test_dense_aggregation_fit_file.py \
  tests/test_legacy_dense_control.py \
  reports/issue_13_dense_aggregation_foundation_review.md
git diff --cached --check
git commit -m "fix: finalize HDF5 aggregation cleanup"
```

If no files changed, do not create an empty commit.

- [ ] **Step 8: Push and update the existing draft PR**

```bash
git push origin codex/issue-13-dense-aggregation
```

Update PR #20—not a new PR—with:

- one-paragraph research context and baseline;
- direct HDF5 simplification and deleted custom cache surface;
- exact test commands/counts;
- explicit “official gate/construction/evaluation not run” status;
- link to `reports/issue_13_dense_aggregation_foundation_review.md`;
- known optional full-suite blocker, if any.

Keep the PR draft unless every acceptance criterion is met and independent review finds no merge blocker.

- [ ] **Step 9: Apply the finishing-development-branch decision gate**

Use `superpowers:finishing-a-development-branch`. Report:

- where the branch started;
- what changed and how much code was removed;
- exact verification evidence;
- whether PR #20 is mergeable at GitHub;
- whether it is scientifically ready only for the next qrel-free feasibility run;
- any remaining blocker before changing the PR from draft to ready.
