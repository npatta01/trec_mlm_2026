# Dense Aggregation HDF5 Fit Store Design

**Date:** 2026-07-15

**Status:** Approved direction; implementation pending

**Issue:** #13

**Pull request:** #20

## Decision

Replace the custom append-only streaming cache with one single-writer HDF5 file
written and read directly through `h5py`.

This is a clean replacement. There is no migration, compatibility reader,
schema dispatch, or support for the existing cache directory layout. Existing
experimental cache artifacts are disposable because no official construction
run has been accepted from them.

The scientific protocol remains frozen. This change affects only how completed
cluster fits are stored, validated, and retrieved.

## Why the Existing Design Is Being Replaced

The current `StreamingFitCache` implements storage-engine concerns directly:

- five manually coordinated binary and JSONL shards;
- byte offsets, byte counts, and a second in-memory index;
- duplicated record and shard schemas;
- explicit resident-object release;
- separate finalize and move operations;
- extensive tests for custom parsing, offsets, mutation matrices, signatures,
  and internal compatibility behavior.

That design grew to roughly 1,900 production lines and roughly 6,000 cache test
lines. The experiment does not need a general cache or a concurrent database.
It has one writer, a known finite inventory, two cluster counts, bounded random
reads after construction, and immutable publication.

HDF5 already supplies typed multidimensional datasets, bounded slicing,
chunk-level checksums, structured metadata, and mature file handling. The
project should retain tests for its scientific invariants and publication
rules, not tests for a home-grown storage format.

## Scope

This change will:

1. Replace `StreamingFitCache` with direct `h5py.File` usage and a few small
   experiment-specific conversion and validation functions.
2. Add `h5py==3.15.1` to the `dense-aggregation` dependency group and update
   the lock file.
3. Update feasibility and construction to write and read the HDF5 file directly.
4. Remove the old cache implementation and its obsolete tests rather than
   retaining compatibility paths.
5. Require canonical JSON bytes for final configuration files.
6. Remove the legacy validator's compatibility wrapper and call the final rich
   validator directly.
7. Consolidate cache tests around externally meaningful behavior.
8. Rewrite the pull request report so a newcomer can understand the research
   question, baseline, implementation status, evidence, and remaining work.

The change will not:

- open or evaluate official qrels;
- run the official 104,006-fit construction;
- change folds, candidate sets, metrics, clustering, scoring, selection, or
  success criteria;
- support multiple writers or readers during a write;
- resume an interrupted file in place;
- preserve or convert the old cache layout;
- make raw HDF5 bytes part of cross-run scientific equivalence.

## Library Choice

`h5py` is used directly because HDF5 natively represents large typed NumPy
datasets and supports bounded slicing, chunking, and error-detection filters.
The project does not wrap those capabilities in another storage abstraction.

`diskcache` was considered and rejected. It is a SQLite-backed key/value cache
with serialization, expiration, eviction, and concurrency behavior. Those
features do not serve this single-writer immutable artifact. Saving 104,006
large Python values would also introduce per-value serialization and potentially
many external value files, while preventing direct bounded array slicing.

Individual `.npy` or `.npz` files were also rejected: one file per fit would
create 104,006 filesystem objects, while one combined archive cannot be built
and sliced as cleanly under the experiment's memory bound.

## Architecture

### Lifecycle

The authoritative lifecycle is:

1. Validate all preflight inputs and require a new temporary file path.
2. Create `fits.partial.h5` with its complete dataset shapes preallocated from
   the expected inventory.
3. Write each fit exactly once at its predetermined inventory row.
4. Mark that row complete only after all fit data and timing data have been
   written successfully.
5. After the scheduled loop, require every expected slot, set the file's root
   `complete` attribute to 1, and flush. Normal context exit then closes the
   HDF5 handle. On an exception, context exit closes it with `complete` still 0.
6. Open the file with the operating system, call `fsync`, and `fsync` its
   parent directory.
7. Reopen the file read-only with `h5py.File(path, "r")` and perform independent
   full validation.
8. Atomically rename the file to `fits.h5` inside the construction staging
   directory without overwriting an existing destination.
9. Include `fits.h5` in the construction manifest and atomically publish the
   construction directory using the existing artifact publication mechanism.

An interrupted or invalid `fits.partial.h5` is never accepted as input and is
never renamed to `fits.h5`. A retry uses a fresh path. This is simpler and more
reliable than in-place crash recovery for a one-off scientific construction.

### Concurrency and synchronization

There is exactly one writer and no reader while the file is being written.
HDF5 SWMR mode, distributed locks, file-lock coordination, and a synchronization
service are therefore out of scope.

`h5py.File.flush()` transfers HDF5 buffers to the operating system. Durability
is established only after the HDF5 file is closed and the underlying file is
`fsync`ed. Atomic visibility is established by the existing no-replace rename
and parent-directory `fsync` operations.

## Direct Library Use

There is no public cache or storage class. Construction owns an ordinary
`h5py.File` context and writes directly to precreated datasets:

```python
with h5py.File(partial_path, "x") as h5:
    initialize_fit_datasets(
        h5,
        metadata=metadata,
        inventory=inventory,
        embedding_width=embedding_width,
    )
    for inventory_row, cluster_count in scheduled_fits:
        started_ns = time.perf_counter_ns()
        fit, assignments = compute_fit(inventory_row, cluster_count)
        write_fit_row(
            h5,
            inventory_row,
            fit,
            assignments,
            started_ns=started_ns,
        )
    if not (
        np.all(h5["fits/c20/complete"][:])
        and np.all(h5["fits/c40/complete"][:])
    ):
        raise ValueError("fit construction is incomplete")
    h5.attrs["complete"] = np.uint8(1)
    h5.flush()
```

Only four module-level functions contain project-specific HDF5 logic:

```python
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

These functions do not form a general storage API. They encode only the
experiment's fixed HDF5 layout and scientific invariants. Consumers open the
file directly with `h5py.File(path, "r")` and pass that handle to
`read_fit_row`. The returned arrays become reclaimable when the caller drops
the returned record; there is no resident-object cache or `release` method.

Artifact publication remains a separate existing concern: after normal context
exit, the caller syncs the closed file, calls `validate_fit_file`, and uses the
existing no-replace rename helper. There is no `load_fit`, key index,
`append_timing`, `finalize`, or `finalize_into` method.

## Logical Inventory

The ordered inventory is supplied once by the experiment and stored once in the
file. Each logical inventory row identifies:

- `exclusion_kind`: pair or single;
- `inventory_index`: the canonical pair or query index;
- one or two excluded canonical query indices;
- the expected support count, 320 for a pair and 321 for a single.

Each inventory row has exactly two fit slots, one for `C=20` and one for
`C=40`. Callers address a fit by `(inventory_row, cluster_count)`. The store
does not maintain a second fit-key index because inventory position is already
the canonical experiment key.

Support query IDs are reconstructed as canonical query order minus the excluded
indices. They are not duplicated for every fit.

## HDF5 Layout

The file contains one final layout only. Dataset names and dtypes are constants
in the implementation; there is no version dispatcher.

```text
/
  attrs:
    schema_id                 fixed final schema identifier
    complete                  uint8, 0 while writing and 1 before validation
    inventory_count           uint64
    embedding_width           uint32
  metadata/
    canonical_json            uint8[N]
  inventory/
    exclusion_kind            uint8[I]       # 0 pair, 1 single
    inventory_index           uint32[I]
    excluded_query_indices    int16[I,2]     # -1 in unused single slot
    support_count             uint16[I]
  fits/c20/
    centers                   float32[I,20,D]
    labels                    uint16[I,321]   # sentinel in unused pair cell
    cluster_inertia           float64[I,20]
    cluster_membership_hash   uint8[I,20,32]
    center_hash               uint8[I,32]
    fit_membership_hash       uint8[I,32]
    sklearn_inertia           float64[I]
    recomputed_total_inertia  float64[I]
    n_iter                    uint32[I]
    elapsed_ns                uint64[I]
    assignment_query_index    int16[I,2]
    assignment_cluster_id     uint16[I,2]
    assignment_distance       float64[I,2]
    assignment_support        uint16[I,2]
    complete                  uint8[I]
  fits/c40/
    same fields, with centers[I,40,D], inertia[I,40], and hash[I,40,32]
```

`I` is the number of pair/single inventory rows, `D` is embedding width, and
the two cluster-count groups avoid padding every 20-center fit to 40 centers.
The single padded label and assignment cells use explicit sentinels and are
validated.

Large numerical datasets are chunked so a single fit read remains bounded.
Chunk sizes may group a small fixed number of adjacent fits for efficient I/O,
but a read must not materialize a complete dataset. Chunked datasets use the
HDF5 Fletcher32 checksum filter. Compression is disabled to avoid adding CPU
cost to the already expensive construction unless feasibility measurements
show a clear overall benefit.

The canonical metadata document contains only experiment-wide provenance:
canonical query IDs/hash, embedding/similarity/clustering-contract hashes,
implementation/dependency/runtime/numerical-environment hashes, and the locked
protocol hash. It does not contain paths, process IDs, timestamps, elapsed
values, duplicate dataset schemas, or expected counts derivable from inventory.

## Write Rules

`write_fit_row` validates before mutating the file:

- the inventory row and cluster count are valid;
- that slot is not already complete;
- excluded queries and assignments match the inventory;
- the support tuple is exactly canonical query order minus exclusions;
- centers, labels, inertia, memberships, and assignments have exact shapes,
  dtypes, finite values, and canonical ordering;
- the fit's embedding and clustering-contract hashes match file metadata;
- stored logical hashes match values reconstructed from the supplied fit.

The caller records `started_ns` immediately before beginning the cluster fit,
then passes it to `write_fit_row`. The function writes the fit datasets, reads
`time.perf_counter_ns()` after those writes, stores their difference as
`elapsed_ns`, and writes the slot's `complete` byte last. This preserves the
existing measurement boundary, which includes fitting, assignment, validation,
and storage but excludes later publication and full validation. A separate
timing receipt or `append_timing` method is unnecessary.

No application-level object is retained after the call. The caller may release
its arrays immediately.

## Read and Validation Rules

`validate_fit_file` opens the path read-only and rejects:

- a missing or non-regular file;
- an unknown schema identifier or `complete != 1`;
- missing or extra groups, datasets, or attributes;
- incorrect dtypes, ranks, shapes, chunk/checksum settings, or metadata bytes;
- noncanonical metadata JSON or provenance mismatches;
- an invalid inventory or any incomplete fit slot.

`read_fit_row` reads only the selected slot, reconstructs its support and
memberships, recomputes logical hashes and inertia relationships, validates
assignments, and returns the existing domain `StoredFitRecord` representation.

Full validation iterates over every inventory row and both cluster counts using
bounded reads. It validates all logical fit contents before publication. The
construction manifest additionally records the raw SHA-256 and byte size of
the closed `fits.h5` file for mutation/truncation detection within that
artifact. Raw file hashes are integrity evidence for one artifact, not a
cross-run reproducibility requirement; cross-run comparison uses logical
scientific hashes reconstructed from datasets.

## Configuration and Validator Cleanup

Final feasibility and official configuration files must equal
`canonical_json(payload) + b"\n"`. Their recorded source hash is the raw
SHA-256 of those exact bytes. Pretty-printed or otherwise reserialized JSON is
rejected even when it parses to the same object.

Only the final configuration shape is supported. There is no format migration
or schema-version dispatch. The tracked feasibility configuration will be
rewritten canonically.

The legacy control validation path will call the final rich validator directly.
Its compatibility signature/reflection wrapper and tests will be removed.
Filesystem race protection, no-replace publication, numerical-environment
checks, and the qrel embargo remain required.

## Test Strategy

Implementation follows test-driven development. Tests exercise public behavior
and scientific invariants rather than HDF5 internals.

Retained or rewritten coverage:

- exact fit round-trip for both exclusion kinds and both cluster counts;
- equivalence of reconstructed centers, labels, memberships, inertia,
  assignments, downstream scores, and ranks;
- duplicate, wrong-order, wrong-shape, nonfinite, mismatched-provenance, and
  incomplete-fit rejection;
- HDF5 truncation, dataset mutation, checksum failure, manifest mismatch, and
  unknown-layout rejection;
- bounded single-fit reads and bounded full-validation residency;
- interrupted writer never becoming a published `fits.h5`;
- file and directory sync ordering;
- no-replace publication races and same-filesystem requirements;
- canonical configuration bytes;
- qrel embargo and existing feasibility safety checks.

Removed coverage:

- custom binary offsets and byte spans;
- JSONL field-order and receipt parsing matrices;
- duplicated header/shard schemas;
- fit-key SHA collision monkeypatches;
- resident-cache release and private-handle inspection;
- deleted method signatures and reflection-based compatibility assertions;
- repeated mutation matrices that prove the same validation boundary.

Small fixtures will create valid stores exclusively through the public writer.
Focused mutations will cover one representative failure per validation boundary.
This should reduce the cache test suite substantially while preserving the
risks that matter to the experiment.

## Standalone Pull Request Report

The tracked report in the pull request will be rewritten for someone joining
the project with no prior conversation. It will explain:

1. The task: rank 150 candidate responses for each of 322 development queries.
2. The metrics and terminology: qrels, nDCG, MRR, folds, leakage, MPNet, and
   nearest-neighbor evidence.
3. The starting point: 0.3599 nDCG, 0.4041 MRR, MPNet `k=50` selected in
   320/322 folds, and 116 queries with zero nDCG.
4. The hypothesis: combine local neighbor evidence, semantic-domain evidence,
   and a global prior instead of relying only on one 50-neighbor weighted mean.
5. What the PR implements and what each safety mechanism protects.
6. What has actually been tested versus what remains unmeasured.
7. Why “180k responses” is not the immediate evaluation unit: the authoritative
   construction is 104,006 cluster fits and 1,040,060 clustering starts, while
   effectiveness is evaluated on the locked 322-by-150 ranking task.
8. The locked success criteria: paired nDCG confidence-interval lower bound
   above zero, MRR delta at least -0.005, fewer than 116 zero-nDCG queries,
   positive low-density-query gain, and all integrity checks passing.
9. A status table, reproduction commands, limitations, and the next decision.

The report will clearly state that passing infrastructure tests is not evidence
that ranking quality improved. No effectiveness finding will be claimed until
the authorized construction and evaluation complete.

## Acceptance Criteria

The cleanup is ready for pull-request review when:

- the old cache implementation and obsolete cache tests are absent;
- production code has one HDF5 layout and no cache compatibility path;
- storage uses direct `h5py.File` access plus only the four fixed-layout helper
  functions defined by this document;
- canonical configuration and direct final validation are enforced;
- representative cache, feasibility, publication, construction, and qrel-safe
  tests pass;
- the full available repository test suite passes, or optional-environment
  blockers are documented with the largest relevant passing subset;
- the standalone report accurately distinguishes completed engineering from
  unrun scientific evaluation;
- the PR contains only promoted implementation, tests, configuration,
  dependency changes, the design/plan, and the standalone report;
- no official qrels have been accessed.
