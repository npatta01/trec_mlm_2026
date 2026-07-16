from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path, PurePosixPath
import stat
import statistics
import time
from typing import Literal

import h5py
import numpy as np
from numpy.typing import NDArray

from trec_mlm.dense_aggregation_hashing import (
    canonical_json,
    framed_sha256,
    support_query_ids_sha256,
)
from trec_mlm.dense_aggregation_artifacts import (
    atomic_artifact_directory,
    raw_file_sha256,
    rename_noreplace,
    validate_manifest,
)
from trec_mlm.dense_aggregation_clustering import (
    assign_targets,
    capture_numerical_environment,
    fit_canonical_clusters,
    numerical_environment_file_bytes,
    require_same_numerical_environment,
)
from trec_mlm.dense_aggregation_contract import (
    FeasibilityConfig,
    build_clustering_contract,
    contract_source_inventory_sha256,
    load_and_validate_frozen_dense_inputs,
    runtime_source_inventory_sha256,
)
from trec_mlm.dense_aggregation_fit_file import (
    FitInventoryRow,
    StoredFitRecord,
    initialize_fit_datasets,
    read_fit_row,
    validate_fit_file,
    write_fit_row,
)
from trec_mlm.dense_aggregation_runtime import THREAD_ENVIRONMENT


_WORKER_SCHEMA = "trec-mlm-dense-aggregation-feasibility-worker-v1"
_GATE_SCHEMA = "trec-mlm-dense-aggregation-feasibility-gate-v1"
_FEASIBILITY_CONFIG_SCHEMA = "trec-mlm-dense-aggregation-feasibility-config-v1"
_MEASUREMENTS_SCHEMA = "trec-mlm-dense-aggregation-feasibility-repeat-measurements-v1"
_FROZEN_FEASIBILITY_PATHS = (
    ("query_path", "query", "data/dev/llm_dev_data.tsv"),
    (
        "embedding_path",
        "embedding",
        "artifacts/frozen_dense_input/mpnet_base_v2.npy",
    ),
    (
        "encoder_metadata_path",
        "encoder metadata",
        "artifacts/frozen_dense_input/encoder_metadata.json",
    ),
    (
        "protocol_path",
        "protocol",
        "protocols/trec_2026_dense_aggregation.md",
    ),
)
_SUMMARY_KEYS = {
    "fit_count",
    "mean_seconds",
    "median_seconds",
    "p95_seconds",
    "maximum_seconds",
    "p95_method",
    "n_iter_min",
    "n_iter_mean",
    "n_iter_max",
    "n_iter_values_sha256",
}
_RECEIPT_KEYS = {
    "schema",
    "fit_count",
    "start_count",
    "pair_indices",
    "single_indices",
    "timing_summary",
    "fit_file_sha256",
    "fit_evidence_sha256",
    "provenance",
    "exclusive_resource_attested",
}
_TIMING_CATEGORIES = ("pair_c20", "pair_c40", "single_c20", "single_c40")
_LOWERCASE_HEX = frozenset("0123456789abcdef")
_TASK7_JSON_BYTE_LIMIT = 1024 * 1024
_TASK7_MANIFEST_BYTE_LIMIT = 1024 * 1024
_MISSING_LAYOUT_ENTRY = object()
_WORKER_OUTPUT_FILES = (
    "environment.json",
    "feasibility_receipt.json",
    "fits.h5",
)
_AGGREGATE_OUTPUT_FILES = (
    "feasibility_gate.json",
    "repeat_a_receipt.json",
    "repeat_b_receipt.json",
    "repeat_measurements.json",
)


FEASIBILITY_PROVENANCE_KEYS = {
    "config_sha256",
    "protocol_sha256",
    "clustering_contract_sha256",
    "implementation_inventory_sha256",
    "dependency_inventory_sha256",
    "runtime_inventory_sha256",
    "query_file_sha256",
    "query_input_sha256",
    "embedding_sha256",
    "similarity_sha256",
    "sample_fixture_sha256",
    "numerical_environment_sha256",
    "python_version",
    "numpy_version",
    "scipy_version",
    "scikit_learn_version",
    "thread_environment",
    "fit_work_file",
    "output_root",
    "worker_pid",
}

FEASIBILITY_REPEAT_CHECKS = (
    "receipt_schema",
    "config",
    "protocol",
    "clustering_contract",
    "implementation_inventory",
    "dependency_inventory",
    "runtime_inventory",
    "query_input",
    "embedding",
    "similarity",
    "sample_fixture",
    "sample_indices",
    "category_counts",
    "fit_count",
    "start_count",
    "fit_supports",
    "fit_memberships",
    "fit_centers",
    "target_assignments",
    "manifests",
    "fresh_distinct_roots",
    "finite_timings",
    "exclusive_resource_attestation",
)


class _RepeatCheckContext:
    def __init__(
        self, repeat_a: FeasibilityReceipt, repeat_b: FeasibilityReceipt
    ) -> None:
        self.receipts = (repeat_a, repeat_b)
        self._evidence: tuple[_RepeatOutputEvidence, ...] | None = None
        self._evidence_failed = False

    def evidence(self) -> tuple[_RepeatOutputEvidence, ...] | None:
        if self._evidence is not None:
            return self._evidence
        if self._evidence_failed:
            return None
        loaded: list[_RepeatOutputEvidence] = []
        try:
            for receipt in self.receipts:
                if receipt.fit_count != 576:
                    raise ValueError("worker fit count is not the locked count")
                output_value = receipt.provenance.get("output_root")
                output_root = _require_lexical_absolute_path(
                    output_value,
                    label="worker output root",
                )
                if _is_embargoed_qrel_path(output_root):
                    raise ValueError("worker output root is embargoed")
                loaded.append(_read_repeat_output_evidence(output_root, receipt))
        except (KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
            self._evidence_failed = True
            return None
        self._evidence = tuple(loaded)
        return self._evidence

    def field_digests(self) -> tuple[Mapping[str, str], ...] | None:
        evidence = self.evidence()
        if evidence is None:
            return None
        return tuple(item.field_sha256 for item in evidence)


_ACTIVE_REPEAT_CONTEXT: ContextVar[_RepeatCheckContext | None] = ContextVar(
    "dense_aggregation_feasibility_repeat_context", default=None
)


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
    fit_file_sha256: str
    fit_evidence_sha256: str
    provenance: Mapping[str, object]
    exclusive_resource_attested: bool
    schema: str = _WORKER_SCHEMA


@dataclass(frozen=True, slots=True)
class FeasibilityGate:
    status: Literal["pass", "stop"]
    failures: tuple[str, ...]
    projection_seconds: float
    maximum_rss_bytes: int
    schema: str = _GATE_SCHEMA


@dataclass(frozen=True, slots=True)
class ValidatedFeasibilityGate:
    manifest_sha256: str
    status: Literal["pass"]
    failures: tuple[str, ...]
    freshness_passed: bool


@dataclass(frozen=True, slots=True)
class _QrelFreeFrozenInputPaths:
    query_path: Path
    embedding_path: Path
    encoder_metadata_path: Path


@dataclass(frozen=True, slots=True)
class _RepeatOutputEvidence:
    metadata: Mapping[str, object]
    inventory: tuple[FitInventoryRow, ...]
    field_sha256: Mapping[str, str]
    timing_summary: Mapping[str, TimingCategorySummary]


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _LOWERCASE_HEX for character in value)
    )


def _require_exact_int(value: object, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an exact integer >= {minimum}")
    return value


def _require_finite_float(value: object, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be a finite number")
    return converted


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _require_lexical_absolute_path(value: object, *, label: str) -> Path:
    if (
        type(value) is not str
        or not value
        or "\\" in value
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise ValueError(f"{label} is not a canonical absolute path")
    pure = PurePosixPath(value)
    candidate = Path(value)
    if (
        not pure.is_absolute()
        or pure.as_posix() != value
        or any(part in {"", ".", ".."} for part in pure.parts)
        or candidate.as_posix() != value
    ):
        raise ValueError(f"{label} is not a canonical absolute path")
    return candidate


def _is_embargoed_qrel_path(path: Path) -> bool:
    return "llm_dev_qrels.txt" in Path(path).parts


def _stat_snapshot(status: os.stat_result) -> tuple[int, ...]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_nlink,
        status.st_uid,
        status.st_gid,
        status.st_rdev,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _open_bounded_regular_descriptor(
    path: Path, *, byte_limit: int, label: str
) -> tuple[int, os.stat_result]:
    if type(byte_limit) is not int or byte_limit < 0:
        raise ValueError(f"{label} byte limit is invalid")
    source = Path(path)
    try:
        before = source.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing: {source}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} is not a regular file: {source}")
    if before.st_size > byte_limit:
        raise ValueError(f"{label} exceeds its {byte_limit}-byte limit")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            source,
            os.O_RDONLY
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _stat_snapshot(opened) != _stat_snapshot(
            before
        ):
            raise ValueError(f"{label} changed while opening: {source}")
        return descriptor, before
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise ValueError(f"{label} changed while opening: {source}") from error
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise


def _verify_bounded_regular_descriptor(
    path: Path,
    *,
    descriptor: int,
    before: os.stat_result,
    label: str,
) -> None:
    source = Path(path)
    expected = _stat_snapshot(before)
    try:
        after_handle = os.fstat(descriptor)
        after_path = source.lstat()
        resolved = source.resolve(strict=True)
        resolved_status = resolved.lstat()
    except OSError as error:
        raise ValueError(f"{label} changed while it was read: {source}") from error
    if any(
        _stat_snapshot(observed) != expected
        for observed in (after_handle, after_path, resolved_status)
    ):
        raise ValueError(f"{label} changed while it was read: {source}")


def _read_bounded_regular_file(path: Path, *, byte_limit: int, label: str) -> bytes:
    source = Path(path)
    descriptor, before = _open_bounded_regular_descriptor(
        source,
        byte_limit=byte_limit,
        label=label,
    )
    try:
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ValueError(f"{label} changed while it was read: {source}")
            chunks.append(block)
            remaining -= len(block)
        if os.read(descriptor, 1) != b"":
            raise ValueError(f"{label} grew while it was read: {source}")
        _verify_bounded_regular_descriptor(
            source,
            descriptor=descriptor,
            before=before,
            label=label,
        )
        return b"".join(chunks)
    except OSError as error:
        raise ValueError(f"{label} changed while it was read: {source}") from error
    finally:
        os.close(descriptor)


def _bounded_regular_file_sha256(path: Path, *, byte_limit: int, label: str) -> str:
    source = Path(path)
    descriptor, before = _open_bounded_regular_descriptor(
        source,
        byte_limit=byte_limit,
        label=label,
    )
    digest = hashlib.sha256()
    try:
        remaining = before.st_size
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                raise ValueError(f"{label} changed while it was hashed: {source}")
            digest.update(block)
            remaining -= len(block)
        if os.read(descriptor, 1) != b"":
            raise ValueError(f"{label} grew while it was hashed: {source}")
        _verify_bounded_regular_descriptor(
            source,
            descriptor=descriptor,
            before=before,
            label=label,
        )
        return digest.hexdigest()
    except OSError as error:
        raise ValueError(f"{label} changed while it was hashed: {source}") from error
    finally:
        os.close(descriptor)


def _read_bounded_canonical_json(
    path: Path, *, byte_limit: int = _TASK7_JSON_BYTE_LIMIT, label: str
) -> object:
    payload = _read_bounded_regular_file(Path(path), byte_limit=byte_limit, label=label)
    return _parse_canonical_json_bytes(payload, label=label)


def _parse_canonical_json_bytes(payload: bytes, *, label: str) -> object:
    try:
        value = json.loads(payload)
        expected = canonical_json(value) + b"\n"
    except (TypeError, UnicodeError, ValueError) as error:
        raise ValueError(f"{label} is not valid canonical JSON") from error
    if payload != expected:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _preflight_exact_physical_layout(
    root: Path, *, expected_files: Sequence[str], label: str
) -> None:
    tree: dict[str, object] = {}
    for relative in ("manifest.json", *expected_files):
        parts = relative.split("/")
        if (
            not relative
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError(f"{label} expected file layout is invalid")
        node = tree
        for part in parts[:-1]:
            if part not in node:
                node[part] = {}
            child = node[part]
            if type(child) is not dict:
                raise ValueError(f"{label} expected file layout is invalid")
            node = child
        if parts[-1] in node:
            raise ValueError(f"{label} expected file layout is invalid")
        node[parts[-1]] = None

    def scan(directory: Path, expected: dict[str, object]) -> None:
        try:
            status = directory.lstat()
        except OSError as error:
            raise ValueError(f"{label} has an unexpected physical layout") from error
        if not stat.S_ISDIR(status.st_mode):
            raise ValueError(f"{label} has an unexpected physical layout")

        observed: set[str] = set()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    child = expected.get(entry.name, _MISSING_LAYOUT_ENTRY)
                    if child is _MISSING_LAYOUT_ENTRY or entry.name in observed:
                        raise ValueError(f"{label} has an unexpected physical layout")
                    observed.add(entry.name)
                    if type(child) is dict:
                        valid_kind = entry.is_dir(follow_symlinks=False)
                    else:
                        valid_kind = entry.is_file(follow_symlinks=False)
                    if not valid_kind:
                        raise ValueError(f"{label} has an unexpected physical layout")
        except OSError as error:
            raise ValueError(f"{label} has an unexpected physical layout") from error
        if observed != set(expected):
            raise ValueError(f"{label} has an unexpected physical layout")

        for name, child in expected.items():
            if type(child) is dict:
                scan(directory / name, child)

    scan(Path(root), tree)


def _validate_task7_manifest(
    root: Path,
    *,
    expected_files: Sequence[str],
    label: str,
    expected_manifest_sha256: str | None = None,
    expected_manifest_bytes: bytes | None = None,
) -> dict[str, object]:
    artifact_root = Path(root)
    manifest_bytes = _read_bounded_regular_file(
        artifact_root / "manifest.json",
        byte_limit=_TASK7_MANIFEST_BYTE_LIMIT,
        label=f"{label} manifest",
    )
    if (
        expected_manifest_sha256 is not None
        and hashlib.sha256(manifest_bytes).hexdigest() != expected_manifest_sha256
    ):
        raise ValueError(f"{label} manifest differs from its expected SHA-256")
    if (
        expected_manifest_bytes is not None
        and manifest_bytes != expected_manifest_bytes
    ):
        raise ValueError(f"{label} manifest changed during validation")
    value = _parse_canonical_json_bytes(
        manifest_bytes,
        label=f"{label} manifest",
    )
    if type(value) is not dict:
        raise ValueError(f"{label} manifest has an unexpected layout")
    rows = value.get("files")
    if (
        type(rows) is not list
        or any(type(row) is not list or len(row) != 3 for row in rows)
        or [row[0] for row in rows] != list(expected_files)
    ):
        raise ValueError(f"{label} manifest has an unexpected layout")
    _preflight_exact_physical_layout(
        artifact_root,
        expected_files=expected_files,
        label=label,
    )
    return validate_manifest(artifact_root)


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


def _summarize_timing_rows(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, TimingCategorySummary]:
    elapsed_by_category: dict[str, list[float]] = {
        name: [] for name in _TIMING_CATEGORIES
    }
    n_iter_by_category: dict[str, list[int]] = {name: [] for name in _TIMING_CATEGORIES}
    for row in rows:
        kind = row.get("exclusion_kind")
        cluster_count = row.get("cluster_count")
        if type(kind) is not str or kind not in {"pair", "single"}:
            raise ValueError("timing row exclusion_kind is invalid")
        if type(cluster_count) is not int or cluster_count not in {20, 40}:
            raise ValueError("timing row cluster_count is invalid")
        elapsed_ns = _require_exact_int(
            row.get("elapsed_ns"), name="timing row elapsed_ns"
        )
        n_iter = _require_exact_int(
            row.get("n_iter"), name="timing row n_iter", minimum=1
        )
        if n_iter > 299:
            raise ValueError("timing row n_iter must be <= 299")
        category = f"{kind}_c{cluster_count}"
        elapsed_by_category[category].append(elapsed_ns / 1_000_000_000)
        n_iter_by_category[category].append(n_iter)

    summaries: dict[str, TimingCategorySummary] = {}
    for category in _TIMING_CATEGORIES:
        elapsed = elapsed_by_category[category]
        n_iters = n_iter_by_category[category]
        if not elapsed:
            raise ValueError(f"timing category {category} is empty")
        summaries[category] = TimingCategorySummary(
            fit_count=len(elapsed),
            mean_seconds=math.fsum(elapsed) / len(elapsed),
            median_seconds=float(statistics.median(elapsed)),
            p95_seconds=_linear_p95(elapsed),
            maximum_seconds=max(elapsed),
            p95_method="linear_index_(n-1)*0.95",
            n_iter_min=min(n_iters),
            n_iter_mean=math.fsum(n_iters) / len(n_iters),
            n_iter_max=max(n_iters),
            n_iter_values_sha256=framed_sha256(
                "feasibility-n-iter-values", canonical_json(n_iters)
            ),
        )
    return summaries


def _summary_to_dict(summary: TimingCategorySummary) -> dict[str, object]:
    return {
        "fit_count": summary.fit_count,
        "mean_seconds": summary.mean_seconds,
        "median_seconds": summary.median_seconds,
        "p95_seconds": summary.p95_seconds,
        "maximum_seconds": summary.maximum_seconds,
        "p95_method": summary.p95_method,
        "n_iter_min": summary.n_iter_min,
        "n_iter_mean": summary.n_iter_mean,
        "n_iter_max": summary.n_iter_max,
        "n_iter_values_sha256": summary.n_iter_values_sha256,
    }


def _summary_from_dict(value: object) -> TimingCategorySummary:
    if type(value) is not dict or set(value) != _SUMMARY_KEYS:
        raise ValueError("timing summary has an invalid contract")
    fit_count = _require_exact_int(value["fit_count"], name="fit_count", minimum=1)
    n_iter_min = _require_exact_int(value["n_iter_min"], name="n_iter_min", minimum=1)
    n_iter_max = _require_exact_int(value["n_iter_max"], name="n_iter_max", minimum=1)
    if n_iter_min > n_iter_max or n_iter_max > 299:
        raise ValueError("timing summary n_iter bounds are invalid")
    digest = value["n_iter_values_sha256"]
    if not _is_sha256(digest):
        raise ValueError("timing summary n_iter digest is invalid")
    if value["p95_method"] != "linear_index_(n-1)*0.95":
        raise ValueError("timing summary p95 method is invalid")
    return TimingCategorySummary(
        fit_count=fit_count,
        mean_seconds=_require_finite_float(value["mean_seconds"], name="mean_seconds"),
        median_seconds=_require_finite_float(
            value["median_seconds"], name="median_seconds"
        ),
        p95_seconds=_require_finite_float(value["p95_seconds"], name="p95_seconds"),
        maximum_seconds=_require_finite_float(
            value["maximum_seconds"], name="maximum_seconds"
        ),
        p95_method="linear_index_(n-1)*0.95",
        n_iter_min=n_iter_min,
        n_iter_mean=_require_finite_float(value["n_iter_mean"], name="n_iter_mean"),
        n_iter_max=n_iter_max,
        n_iter_values_sha256=digest,
    )


def _receipt_to_dict(receipt: FeasibilityReceipt) -> dict[str, object]:
    return {
        "schema": receipt.schema,
        "fit_count": receipt.fit_count,
        "start_count": receipt.start_count,
        "pair_indices": list(receipt.pair_indices),
        "single_indices": list(receipt.single_indices),
        "timing_summary": {
            name: _summary_to_dict(receipt.timing_summary[name])
            for name in _TIMING_CATEGORIES
        },
        "fit_file_sha256": receipt.fit_file_sha256,
        "fit_evidence_sha256": receipt.fit_evidence_sha256,
        "provenance": dict(receipt.provenance),
        "exclusive_resource_attested": receipt.exclusive_resource_attested,
    }


def _receipt_from_dict(value: object) -> FeasibilityReceipt:
    if type(value) is not dict or set(value) != _RECEIPT_KEYS:
        raise ValueError("feasibility receipt has an invalid top-level contract")
    if value["schema"] != _WORKER_SCHEMA:
        raise ValueError("unsupported feasibility receipt schema")
    pair_value = value["pair_indices"]
    single_value = value["single_indices"]
    if type(pair_value) is not list or type(single_value) is not list:
        raise ValueError("feasibility sample indices must be lists")
    pair_indices = tuple(
        _require_exact_int(item, name="pair sample index") for item in pair_value
    )
    single_indices = tuple(
        _require_exact_int(item, name="single sample index") for item in single_value
    )
    summary_value = value["timing_summary"]
    if type(summary_value) is not dict or tuple(summary_value) != _TIMING_CATEGORIES:
        raise ValueError("feasibility timing categories are invalid or unordered")
    provenance = value["provenance"]
    if type(provenance) is not dict or set(provenance) != FEASIBILITY_PROVENANCE_KEYS:
        raise ValueError("feasibility provenance keys differ from the exact contract")
    provenance_hash_fields = FEASIBILITY_PROVENANCE_KEYS - {
        "python_version",
        "numpy_version",
        "scipy_version",
        "scikit_learn_version",
        "thread_environment",
        "fit_work_file",
        "output_root",
        "worker_pid",
    }
    if any(not _is_sha256(provenance[field]) for field in provenance_hash_fields):
        raise ValueError("feasibility provenance contains an invalid SHA-256")
    expected_versions = {
        "python_version": "3.13.9",
        "numpy_version": "2.5.1",
        "scipy_version": "1.18.0",
        "scikit_learn_version": "1.9.0",
    }
    if any(
        provenance[field] != expected for field, expected in expected_versions.items()
    ):
        raise ValueError("feasibility provenance runtime versions are invalid")
    thread_environment = provenance["thread_environment"]
    if type(thread_environment) is not dict or thread_environment != {
        name: "1" for name in THREAD_ENVIRONMENT
    }:
        raise ValueError("feasibility provenance thread environment is invalid")
    normalized_roots: list[Path] = []
    for field in ("fit_work_file", "output_root"):
        root = _require_lexical_absolute_path(
            provenance[field],
            label=f"feasibility provenance {field}",
        )
        normalized_roots.append(root)
    if _paths_overlap(normalized_roots[0], normalized_roots[1]):
        raise ValueError("feasibility fit work file and output roots overlap")
    worker_pid = provenance["worker_pid"]
    if type(worker_pid) is not int or worker_pid <= 0:
        raise ValueError("feasibility provenance worker_pid is invalid")
    fit_file_sha256 = value["fit_file_sha256"]
    fit_evidence_sha256 = value["fit_evidence_sha256"]
    if not _is_sha256(fit_file_sha256) or not _is_sha256(fit_evidence_sha256):
        raise ValueError("fit file or evidence digest is invalid")
    attested = value["exclusive_resource_attested"]
    if type(attested) is not bool:
        raise ValueError("exclusive_resource_attested must be a bool")
    return FeasibilityReceipt(
        fit_count=_require_exact_int(value["fit_count"], name="fit_count"),
        start_count=_require_exact_int(value["start_count"], name="start_count"),
        pair_indices=pair_indices,
        single_indices=single_indices,
        timing_summary={
            name: _summary_from_dict(summary_value[name]) for name in _TIMING_CATEGORIES
        },
        fit_file_sha256=fit_file_sha256,
        fit_evidence_sha256=fit_evidence_sha256,
        provenance=dict(provenance),
        exclusive_resource_attested=attested,
    )


def _write_new_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _verify_environment_file(path: Path, expected_sha256: str) -> None:
    if raw_file_sha256(path) != expected_sha256:
        raise RuntimeError("persisted numerical environment changed during worker")


def preflight_atomic_paths(*, fit_work_file: Path, output_dir: Path) -> None:
    work_path = Path(fit_work_file)
    output_path = Path(output_dir)
    if work_path.suffixes[-2:] != [".partial", ".h5"]:
        raise ValueError("fit work file must end with .partial.h5")
    work_file = work_path.resolve()
    output_root = output_path.resolve()
    if _paths_overlap(work_file, output_root):
        raise ValueError("fit work file and output destinations overlap")
    for kind, argument, resolved in (
        ("fit work file", work_path, work_file),
        ("output", output_path, output_root),
    ):
        for candidate in (argument, resolved):
            try:
                candidate.lstat()
            except FileNotFoundError:
                continue
            raise FileExistsError(f"{kind} destination exists: {argument}")
    work_file.parent.mkdir(parents=True, exist_ok=True)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if work_file.parent.stat().st_dev != output_root.parent.stat().st_dev:
        raise OSError("fit work file and output must share one filesystem")


def load_feasibility_config(path: str | Path) -> FeasibilityConfig:
    source = Path(path)
    source_bytes = _read_bounded_regular_file(
        source,
        byte_limit=_TASK7_JSON_BYTE_LIMIT,
        label="feasibility config",
    )
    try:
        payload = json.loads(source_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("feasibility config is not valid JSON") from error
    if type(payload) is not dict:
        raise ValueError("feasibility config must be a JSON object")
    if source_bytes != canonical_json(payload) + b"\n":
        raise ValueError("feasibility config must use canonical JSON bytes")
    expected_keys = {
        "schema",
        "query_path",
        "embedding_path",
        "encoder_metadata_path",
        "protocol_path",
    }
    if set(payload) != expected_keys:
        raise ValueError("feasibility config keys differ from the exact contract")
    if payload["schema"] != _FEASIBILITY_CONFIG_SCHEMA:
        raise ValueError("unsupported feasibility config schema")
    source_path = source.resolve(strict=True)
    repo_root = source_path.parent.parent
    resolved: dict[str, Path] = {}
    for name in sorted(expected_keys - {"schema"}):
        raw_path = payload[name]
        if type(raw_path) is not str or not raw_path:
            raise ValueError(f"feasibility config {name} must be a nonempty string")
        candidate = (repo_root / raw_path).resolve()
        if not candidate.is_relative_to(repo_root):
            raise ValueError(f"feasibility config {name} resolves outside repository root")
        resolved[name] = candidate
    return FeasibilityConfig(
        repo_root=repo_root,
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        query_path=resolved["query_path"],
        embedding_path=resolved["embedding_path"],
        encoder_metadata_path=resolved["encoder_metadata_path"],
        protocol_path=resolved["protocol_path"],
    )


def _worker_inventory(
    pair_indices: Sequence[int], single_indices: Sequence[int]
) -> tuple[FitInventoryRow, ...]:
    pairs = tuple(itertools.combinations(range(322), 2))
    return tuple(
        FitInventoryRow(
            exclusion_kind,
            int(inventory_index),
            pairs[int(inventory_index)]
            if exclusion_kind == "pair"
            else (int(inventory_index),),
        )
        for exclusion_kind, sampled_indices in (
            ("pair", pair_indices),
            ("single", single_indices),
        )
        for inventory_index in sampled_indices
    )


def _fit_metadata(
    *,
    query_ids: Sequence[str],
    query_input_sha256: str,
    embedding_sha256: str,
    similarity_sha256: str,
    clustering_contract_sha256: str,
    implementation_inventory_sha256: str,
    dependency_inventory_sha256: str,
    runtime_inventory_sha256: str,
    numerical_environment_sha256: str,
    protocol_sha256: str,
) -> dict[str, object]:
    return {
        "query_ids": list(query_ids),
        "query_input_sha256": query_input_sha256,
        "embedding_sha256": embedding_sha256,
        "similarity_sha256": similarity_sha256,
        "clustering_contract_sha256": clustering_contract_sha256,
        "implementation_inventory_sha256": implementation_inventory_sha256,
        "dependency_inventory_sha256": dependency_inventory_sha256,
        "runtime_inventory_sha256": runtime_inventory_sha256,
        "numerical_environment_sha256": numerical_environment_sha256,
        "protocol_sha256": protocol_sha256,
    }


def _logical_fit_evidence(record: StoredFitRecord) -> dict[str, object]:
    return {
        "exclusion_kind": record.exclusion_kind,
        "inventory_index": record.inventory_index,
        "cluster_count": record.fit.key.cluster_count,
        "support_query_ids_sha256": support_query_ids_sha256(
            record.fit.key.support_query_ids
        ),
        "embedding_sha256": record.fit.key.embedding_sha256,
        "clustering_contract_sha256": record.fit.key.clustering_contract_sha256,
        "fit_membership_sha256": record.fit.fit_membership_sha256,
        "cluster_membership_sha256": list(record.fit.cluster_membership_sha256),
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


def _update_framed_digest(digest: object, value: object) -> None:
    payload = canonical_json(value)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _fit_evidence_sha256(path: Path, *, row_count: int) -> str:
    digest = hashlib.sha256()
    with h5py.File(path, "r") as h5:
        for row in range(row_count):
            for cluster_count in (20, 40):
                _update_framed_digest(
                    digest, _logical_fit_evidence(read_fit_row(h5, row, cluster_count))
                )
    return digest.hexdigest()


def _read_repeat_output_evidence(
    output_root: Path, receipt: FeasibilityReceipt
) -> _RepeatOutputEvidence:
    fits_path = output_root / "fits.h5"
    if raw_file_sha256(fits_path) != receipt.fit_file_sha256:
        raise ValueError("fit file digest mismatch")
    with h5py.File(fits_path, "r") as h5:
        raw_metadata = bytes(np.asarray(h5["metadata/canonical_json"], dtype="u1"))
    metadata = json.loads(raw_metadata)
    if type(metadata) is not dict or canonical_json(metadata) != raw_metadata:
        raise ValueError("fit metadata is not canonical")
    expected_metadata = {
        "query_ids": metadata.get("query_ids"),
        "query_input_sha256": receipt.provenance["query_input_sha256"],
        "embedding_sha256": receipt.provenance["embedding_sha256"],
        "similarity_sha256": receipt.provenance["similarity_sha256"],
        "clustering_contract_sha256": receipt.provenance[
            "clustering_contract_sha256"
        ],
        "implementation_inventory_sha256": receipt.provenance[
            "implementation_inventory_sha256"
        ],
        "dependency_inventory_sha256": receipt.provenance[
            "dependency_inventory_sha256"
        ],
        "runtime_inventory_sha256": receipt.provenance[
            "runtime_inventory_sha256"
        ],
        "numerical_environment_sha256": receipt.provenance[
            "numerical_environment_sha256"
        ],
        "protocol_sha256": receipt.provenance["protocol_sha256"],
    }
    inventory = _worker_inventory(receipt.pair_indices, receipt.single_indices)
    validate_fit_file(
        fits_path,
        expected_metadata=expected_metadata,
        expected_inventory=inventory,
    )
    full_digest = hashlib.sha256()
    field_digests = {
        name: hashlib.sha256()
        for name in (
            "fit_supports",
            "fit_memberships",
            "fit_centers",
            "target_assignments",
        )
    }
    timing_rows: list[dict[str, object]] = []
    with h5py.File(fits_path, "r") as h5:
        for row_index, inventory_row in enumerate(inventory):
            for cluster_count in (20, 40):
                record = read_fit_row(h5, row_index, cluster_count)
                logical = _logical_fit_evidence(record)
                _update_framed_digest(full_digest, logical)
                subsets = {
                    "fit_supports": {
                        key: logical[key]
                        for key in (
                            "exclusion_kind",
                            "inventory_index",
                            "cluster_count",
                            "support_query_ids_sha256",
                            "embedding_sha256",
                            "clustering_contract_sha256",
                        )
                    },
                    "fit_memberships": {
                        key: logical[key]
                        for key in (
                            "fit_membership_sha256",
                            "cluster_membership_sha256",
                        )
                    },
                    "fit_centers": {"center_sha256": logical["center_sha256"]},
                    "target_assignments": {"assignments": logical["assignments"]},
                }
                for name, value in subsets.items():
                    _update_framed_digest(field_digests[name], value)
                timing_rows.append(
                    {
                        "exclusion_kind": inventory_row.exclusion_kind,
                        "cluster_count": cluster_count,
                        "elapsed_ns": record.elapsed_ns,
                        "n_iter": record.fit.n_iter,
                    }
                )
                del record, logical, subsets
    if full_digest.hexdigest() != receipt.fit_evidence_sha256:
        raise ValueError("fit evidence digest mismatch")
    return _RepeatOutputEvidence(
        metadata=dict(metadata),
        inventory=inventory,
        field_sha256={name: digest.hexdigest() for name, digest in field_digests.items()},
        timing_summary=_summarize_timing_rows(timing_rows),
    )


def _contract_versions(contract: Mapping[str, object]) -> dict[str, str]:
    versions = contract.get("versions")
    required = {"python", "numpy", "scipy", "scikit_learn"}
    if type(versions) is not dict or set(versions) != required:
        raise ValueError("clustering contract versions are invalid")
    if any(type(value) is not str or not value for value in versions.values()):
        raise ValueError("clustering contract version values are invalid")
    return versions


def run_feasibility_worker(
    config: FeasibilityConfig,
    *,
    fit_work_file: Path,
    output_dir: Path,
    exclusive_resource_attested: bool,
) -> FeasibilityReceipt:
    preflight_atomic_paths(fit_work_file=fit_work_file, output_dir=output_dir)

    work_file = Path(fit_work_file).resolve()
    output_root = Path(output_dir).resolve()
    if raw_file_sha256(config.source_path) != config.source_sha256:
        raise ValueError("feasibility config changed after it was loaded")
    inputs = load_and_validate_frozen_dense_inputs(config)
    clustering_contract, clustering_contract_sha256 = build_clustering_contract(
        inputs,
        repo_root=config.repo_root,
        protocol_path=config.protocol_path,
    )
    implementation_inventory_sha256 = contract_source_inventory_sha256(
        clustering_contract, category="implementation"
    )
    dependency_inventory_sha256 = contract_source_inventory_sha256(
        clustering_contract, category="dependency"
    )
    runtime_inventory_sha256 = runtime_source_inventory_sha256(config.repo_root)
    versions = _contract_versions(clustering_contract)
    numerical_environment = capture_numerical_environment()
    environment_bytes = numerical_environment_file_bytes(numerical_environment)
    numerical_environment_sha256 = hashlib.sha256(environment_bytes).hexdigest()
    protocol_sha256 = raw_file_sha256(config.protocol_path)

    pair_array, single_array = official_sample_indices()
    pair_indices = tuple(map(int, pair_array))
    single_indices = tuple(map(int, single_array))
    sample_fixture_sha256 = hashlib.sha256(
        canonical_json(
            {
                "pair_indices": list(pair_indices),
                "single_indices": list(single_indices),
            }
        )
    ).hexdigest()
    metadata = _fit_metadata(
        query_ids=inputs.query_ids,
        query_input_sha256=inputs.query_input_sha256,
        embedding_sha256=inputs.embedding_sha256,
        similarity_sha256=inputs.similarity_sha256,
        clustering_contract_sha256=clustering_contract_sha256,
        implementation_inventory_sha256=implementation_inventory_sha256,
        dependency_inventory_sha256=dependency_inventory_sha256,
        runtime_inventory_sha256=runtime_inventory_sha256,
        numerical_environment_sha256=numerical_environment_sha256,
        protocol_sha256=protocol_sha256,
    )
    inventory = _worker_inventory(pair_indices, single_indices)
    timing_rows: list[dict[str, object]] = []

    with atomic_artifact_directory(output_root, quarantine_on_error=True) as staging:
        _write_new_file(staging / "environment.json", environment_bytes)
        with h5py.File(work_file, "x") as h5:
            initialize_fit_datasets(
                h5,
                metadata=metadata,
                inventory=inventory,
                embedding_width=int(inputs.embeddings.shape[1]),
            )
            require_same_numerical_environment(numerical_environment)
            _verify_environment_file(
                staging / "environment.json", numerical_environment_sha256
            )
            for inventory_row, row in enumerate(inventory):
                excluded = row.excluded_query_indices
                for cluster_count in (20, 40):
                    start_ns = time.perf_counter_ns()
                    support = tuple(
                        index for index in range(322) if index not in excluded
                    )
                    fit = fit_canonical_clusters(
                        embeddings=inputs.embeddings,
                        query_ids=inputs.query_ids,
                        support_indices=support,
                        cluster_count=cluster_count,
                        embedding_sha256=inputs.embedding_sha256,
                        clustering_contract_sha256=clustering_contract_sha256,
                    )
                    support_index_by_id = {
                        inputs.query_ids[index]: index for index in support
                    }
                    assignments = assign_targets(
                        fit=fit,
                        embeddings=inputs.embeddings,
                        query_ids=inputs.query_ids,
                        target_indices=excluded,
                        support_index_by_id=support_index_by_id,
                    )
                    write_fit_row(
                        h5,
                        inventory_row,
                        fit,
                        assignments,
                        started_ns=start_ns,
                    )
                    elapsed_ns = int(
                        h5[f"fits/c{cluster_count}/elapsed_ns"][inventory_row]
                    )
                    n_iter = fit.n_iter
                    del assignments, fit, support, support_index_by_id
                    timing_rows.append(
                        {
                            "exclusion_kind": row.exclusion_kind,
                            "inventory_index": row.inventory_index,
                            "cluster_count": cluster_count,
                            "elapsed_ns": elapsed_ns,
                            "n_iter": n_iter,
                        }
                    )
            completion_arrays = tuple(
                np.asarray(h5[f"fits/c{cluster_count}/complete"][...], dtype="u1")
                for cluster_count in (20, 40)
            )
            if any(np.any(completion != 1) for completion in completion_arrays):
                raise ValueError("fit construction is incomplete")
            h5.attrs.modify("complete", np.uint8(1))
            h5.flush()
        require_same_numerical_environment(numerical_environment)
        _verify_environment_file(
            staging / "environment.json", numerical_environment_sha256
        )
        with work_file.open("rb") as handle:
            os.fsync(handle.fileno())
        validate_fit_file(
            work_file,
            expected_metadata=metadata,
            expected_inventory=inventory,
        )
        rename_noreplace(work_file, staging / "fits.h5")
        staging_fd = os.open(
            staging, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        )
        try:
            os.fsync(staging_fd)
        finally:
            os.close(staging_fd)
        fit_file_sha256 = raw_file_sha256(staging / "fits.h5")
        fit_evidence_sha256 = _fit_evidence_sha256(
            staging / "fits.h5", row_count=len(inventory)
        )

        receipt = FeasibilityReceipt(
            fit_count=576,
            start_count=5_760,
            pair_indices=pair_indices,
            single_indices=single_indices,
            timing_summary=_summarize_timing_rows(timing_rows),
            fit_file_sha256=fit_file_sha256,
            fit_evidence_sha256=fit_evidence_sha256,
            provenance={
                "config_sha256": config.source_sha256,
                "protocol_sha256": protocol_sha256,
                "clustering_contract_sha256": clustering_contract_sha256,
                "implementation_inventory_sha256": implementation_inventory_sha256,
                "dependency_inventory_sha256": dependency_inventory_sha256,
                "runtime_inventory_sha256": runtime_inventory_sha256,
                "query_file_sha256": raw_file_sha256(config.query_path),
                "query_input_sha256": inputs.query_input_sha256,
                "embedding_sha256": inputs.embedding_sha256,
                "similarity_sha256": inputs.similarity_sha256,
                "sample_fixture_sha256": sample_fixture_sha256,
                "numerical_environment_sha256": numerical_environment_sha256,
                "python_version": versions["python"],
                "numpy_version": versions["numpy"],
                "scipy_version": versions["scipy"],
                "scikit_learn_version": versions["scikit_learn"],
                "thread_environment": dict(numerical_environment.thread_environment),
                "fit_work_file": str(work_file),
                "output_root": str(output_root),
                "worker_pid": os.getpid(),
            },
            exclusive_resource_attested=exclusive_resource_attested,
        )
        _write_new_file(
            staging / "feasibility_receipt.json",
            canonical_json(_receipt_to_dict(receipt)) + b"\n",
        )

    _validate_task7_manifest(
        output_root,
        expected_files=_WORKER_OUTPUT_FILES,
        label="worker output",
    )
    persisted_value = _read_bounded_canonical_json(
        output_root / "feasibility_receipt.json",
        label="worker feasibility receipt",
    )
    persisted = _receipt_from_dict(persisted_value)
    if persisted != receipt:
        raise ValueError("published feasibility receipt differs from worker receipt")
    if raw_file_sha256(output_root / "fits.h5") != persisted.fit_file_sha256:
        raise ValueError("published fit file digest differs from worker receipt")
    return persisted


def _projection_seconds(summary: Mapping[str, TimingCategorySummary]) -> float:
    return 51_681 * (
        summary["pair_c20"].mean_seconds + summary["pair_c40"].mean_seconds
    ) + 322 * (summary["single_c20"].mean_seconds + summary["single_c40"].mean_seconds)


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


def _repeat_check(
    name: str, repeat_a: FeasibilityReceipt, repeat_b: FeasibilityReceipt
) -> bool:
    provenance_fields = {
        "config": "config_sha256",
        "protocol": "protocol_sha256",
        "clustering_contract": "clustering_contract_sha256",
        "implementation_inventory": "implementation_inventory_sha256",
        "dependency_inventory": "dependency_inventory_sha256",
        "runtime_inventory": "runtime_inventory_sha256",
        "query_input": "query_input_sha256",
        "embedding": "embedding_sha256",
        "similarity": "similarity_sha256",
        "sample_fixture": "sample_fixture_sha256",
    }
    if name == "receipt_schema":
        try:
            return (
                repeat_a.schema == repeat_b.schema == _WORKER_SCHEMA
                and set(repeat_a.provenance) == FEASIBILITY_PROVENANCE_KEYS
                and set(repeat_b.provenance) == FEASIBILITY_PROVENANCE_KEYS
                and _receipt_from_dict(_receipt_to_dict(repeat_a)) == repeat_a
                and _receipt_from_dict(_receipt_to_dict(repeat_b)) == repeat_b
            )
        except (KeyError, TypeError, ValueError):
            return False
    if name in provenance_fields:
        field = provenance_fields[name]
        if repeat_a.provenance.get(field) != repeat_b.provenance.get(field):
            return False
        metadata_fields = {
            "clustering_contract": "clustering_contract_sha256",
            "implementation_inventory": "implementation_inventory_sha256",
            "dependency_inventory": "dependency_inventory_sha256",
            "runtime_inventory": "runtime_inventory_sha256",
            "query_input": "query_input_sha256",
            "embedding": "embedding_sha256",
            "similarity": "similarity_sha256",
        }
        if name in metadata_fields:
            context = _ACTIVE_REPEAT_CONTEXT.get()
            if context is None:
                context = _RepeatCheckContext(repeat_a, repeat_b)
            evidence = context.evidence()
            metadata_field = metadata_fields[name]
            if evidence is None or any(
                item.metadata.get(metadata_field) != receipt.provenance.get(field)
                for item, receipt in zip(evidence, (repeat_a, repeat_b), strict=True)
            ):
                return False
        if name == "query_input":
            return repeat_a.provenance.get(
                "query_file_sha256"
            ) == repeat_b.provenance.get("query_file_sha256")
        if name == "sample_fixture":
            return repeat_a.provenance.get(field) == (
                "6b74301b9edb16438739a2c46b455ff37985698a6795bd7a4b9d1f5530cf9d39"
            )
        if name == "runtime_inventory":
            runtime_fields = (
                "python_version",
                "numpy_version",
                "scipy_version",
                "scikit_learn_version",
                "thread_environment",
                "numerical_environment_sha256",
            )
            if any(
                repeat_a.provenance.get(runtime_field)
                != repeat_b.provenance.get(runtime_field)
                for runtime_field in runtime_fields
            ):
                return False
            context = _ACTIVE_REPEAT_CONTEXT.get()
            if context is None:
                context = _RepeatCheckContext(repeat_a, repeat_b)
            evidence = context.evidence()
            if evidence is None:
                return False
            for receipt, item in zip((repeat_a, repeat_b), evidence, strict=True):
                output_value = receipt.provenance.get("output_root")
                expected_hash = receipt.provenance.get("numerical_environment_sha256")
                if type(output_value) is not str or not _is_sha256(expected_hash):
                    return False
                if item.metadata.get("numerical_environment_sha256") != expected_hash:
                    return False
                try:
                    if (
                        _bounded_regular_file_sha256(
                            Path(output_value) / "environment.json",
                            byte_limit=_TASK7_JSON_BYTE_LIMIT,
                            label="worker numerical environment",
                        )
                        != expected_hash
                    ):
                        return False
                except OSError:
                    return False
        return True
    if name == "sample_indices":
        pair, single = official_sample_indices()
        expected_pair = tuple(map(int, pair))
        expected_single = tuple(map(int, single))
        receipt_samples_match = (
            repeat_a.pair_indices == repeat_b.pair_indices == expected_pair
            and repeat_a.single_indices == repeat_b.single_indices == expected_single
        )
        if not receipt_samples_match:
            return False
        context = _ACTIVE_REPEAT_CONTEXT.get()
        if context is None:
            context = _RepeatCheckContext(repeat_a, repeat_b)
        evidence = context.evidence()
        return evidence is not None and all(
            item.inventory == _worker_inventory(
                receipt.pair_indices, receipt.single_indices
            )
            for item, receipt in zip(evidence, (repeat_a, repeat_b), strict=True)
        )
    if name == "category_counts":
        expected = {
            "pair_c20": 256,
            "pair_c40": 256,
            "single_c20": 32,
            "single_c40": 32,
        }
        if not all(
            set(receipt.timing_summary) == set(expected)
            and all(
                receipt.timing_summary[category].fit_count == count
                for category, count in expected.items()
            )
            for receipt in (repeat_a, repeat_b)
        ):
            return False
        for category in _TIMING_CATEGORIES:
            left = repeat_a.timing_summary[category]
            right = repeat_b.timing_summary[category]
            if (
                left.n_iter_min,
                left.n_iter_mean,
                left.n_iter_max,
                left.n_iter_values_sha256,
            ) != (
                right.n_iter_min,
                right.n_iter_mean,
                right.n_iter_max,
                right.n_iter_values_sha256,
            ):
                return False
        return True
    if name == "fit_count":
        if repeat_a.fit_count != 576 or repeat_b.fit_count != 576:
            return False
        context = _ACTIVE_REPEAT_CONTEXT.get()
        if context is None:
            context = _RepeatCheckContext(repeat_a, repeat_b)
        evidence = context.evidence()
        return evidence is not None and all(
            2 * len(item.inventory) == receipt.fit_count
            for item, receipt in zip(evidence, (repeat_a, repeat_b), strict=True)
        )
    if name == "start_count":
        if repeat_a.start_count != 5_760 or repeat_b.start_count != 5_760:
            return False
        return True
    if name in {
        "fit_supports",
        "fit_memberships",
        "fit_centers",
        "target_assignments",
    }:
        context = _ACTIVE_REPEAT_CONTEXT.get()
        if context is None:
            context = _RepeatCheckContext(repeat_a, repeat_b)
        digests = context.field_digests()
        if digests is None:
            return False
        left, right = digests
        if name == "fit_supports" and (
            repeat_a.fit_evidence_sha256 != repeat_b.fit_evidence_sha256
        ):
            return False
        return left.get(name) == right.get(name)
    if name == "manifests":
        for receipt in (repeat_a, repeat_b):
            output_value = receipt.provenance.get("output_root")
            try:
                output_root = _require_lexical_absolute_path(
                    output_value,
                    label="worker output root",
                )
            except ValueError:
                return False
            if _is_embargoed_qrel_path(output_root):
                return False
            try:
                _validate_task7_manifest(
                    output_root,
                    expected_files=_WORKER_OUTPUT_FILES,
                    label="worker output",
                )
                persisted = _receipt_from_dict(
                    _read_bounded_canonical_json(
                        output_root / "feasibility_receipt.json",
                        label="worker feasibility receipt",
                    )
                )
                if persisted != receipt:
                    return False
                if raw_file_sha256(output_root / "fits.h5") != receipt.fit_file_sha256:
                    return False
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                return False
        return True
    if name == "fresh_distinct_roots":
        roots: list[Path] = []
        for receipt in (repeat_a, repeat_b):
            for field in ("fit_work_file", "output_root"):
                value = receipt.provenance.get(field)
                try:
                    candidate = _require_lexical_absolute_path(
                        value,
                        label=f"worker {field}",
                    )
                except ValueError:
                    return False
                if _is_embargoed_qrel_path(candidate):
                    return False
                roots.append(candidate)
        worker_pids = tuple(
            receipt.provenance.get("worker_pid") for receipt in (repeat_a, repeat_b)
        )
        return (
            all(
                not _paths_overlap(left, right)
                for index, left in enumerate(roots)
                for right in roots[index + 1 :]
            )
            and all(type(pid) is int and pid > 0 for pid in worker_pids)
            and worker_pids[0] != worker_pids[1]
        )
    if name == "finite_timings":
        context = _ACTIVE_REPEAT_CONTEXT.get()
        if context is None:
            context = _RepeatCheckContext(repeat_a, repeat_b)
        evidence = context.evidence()
        if evidence is None or any(
            dict(item.timing_summary) != dict(receipt.timing_summary)
            for item, receipt in zip(evidence, (repeat_a, repeat_b), strict=True)
        ):
            return False
        return all(
            all(
                math.isfinite(value) and value >= 0.0
                for summary in receipt.timing_summary.values()
                for value in (
                    summary.mean_seconds,
                    summary.median_seconds,
                    summary.p95_seconds,
                    summary.maximum_seconds,
                    summary.n_iter_mean,
                )
            )
            for receipt in (repeat_a, repeat_b)
        )
    if name == "exclusive_resource_attestation":
        return (
            repeat_a.exclusive_resource_attested
            and repeat_b.exclusive_resource_attested
        )
    return True


def _repeat_integrity_failures(
    repeat_a: FeasibilityReceipt, repeat_b: FeasibilityReceipt
) -> tuple[str, ...]:
    token = _ACTIVE_REPEAT_CONTEXT.set(_RepeatCheckContext(repeat_a, repeat_b))
    try:
        return tuple(
            f"feasibility repeat mismatch: {name}"
            for name in FEASIBILITY_REPEAT_CHECKS
            if not _repeat_check(name, repeat_a, repeat_b)
        )
    finally:
        _ACTIVE_REPEAT_CONTEXT.reset(token)


def evaluate_feasibility_repeats(
    repeat_a: FeasibilityReceipt,
    repeat_b: FeasibilityReceipt,
    *,
    repeat_a_rss_bytes: int,
    repeat_b_rss_bytes: int,
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
        failures=tuple(failures),
        projection_seconds=projection,
        maximum_rss_bytes=rss_bytes,
    )


def _gate_from_dict(value: object) -> FeasibilityGate:
    keys = {
        "schema",
        "status",
        "failures",
        "projection_seconds",
        "maximum_rss_bytes",
    }
    if type(value) is not dict or set(value) != keys:
        raise ValueError("feasibility gate has an invalid contract")
    if value["schema"] != _GATE_SCHEMA:
        raise ValueError("unsupported feasibility gate schema")
    status = value["status"]
    failures = value["failures"]
    projection = value["projection_seconds"]
    maximum_rss_bytes = value["maximum_rss_bytes"]
    if type(status) is not str or status not in {"pass", "stop"}:
        raise ValueError("feasibility gate status is invalid")
    if type(failures) is not list or any(type(item) is not str for item in failures):
        raise ValueError("feasibility gate failures are invalid")
    if type(projection) not in {int, float} or not math.isfinite(float(projection)):
        raise ValueError("feasibility gate projection is invalid")
    if type(maximum_rss_bytes) is not int or maximum_rss_bytes < 0:
        raise ValueError("feasibility gate RSS is invalid")
    if (status == "pass") != (not failures):
        raise ValueError("feasibility gate status and failures disagree")
    return FeasibilityGate(
        status=status,
        failures=tuple(failures),
        projection_seconds=float(projection),
        maximum_rss_bytes=maximum_rss_bytes,
    )


def _repeat_measurements_from_dict(
    value: object,
) -> tuple[str, str, int, int]:
    keys = {
        "schema",
        "repeat_a_manifest_sha256",
        "repeat_b_manifest_sha256",
        "repeat_a_rss_bytes",
        "repeat_b_rss_bytes",
    }
    if (
        type(value) is not dict
        or set(value) != keys
        or value["schema"] != _MEASUREMENTS_SCHEMA
    ):
        raise ValueError("repeat measurements have an invalid contract")
    repeat_a_manifest_sha256 = value["repeat_a_manifest_sha256"]
    repeat_b_manifest_sha256 = value["repeat_b_manifest_sha256"]
    if not _is_sha256(repeat_a_manifest_sha256) or not _is_sha256(
        repeat_b_manifest_sha256
    ):
        raise ValueError("repeat measurements contain an invalid manifest digest")
    repeat_a_rss_bytes = value["repeat_a_rss_bytes"]
    repeat_b_rss_bytes = value["repeat_b_rss_bytes"]
    if (
        type(repeat_a_rss_bytes) is not int
        or repeat_a_rss_bytes < 0
        or type(repeat_b_rss_bytes) is not int
        or repeat_b_rss_bytes < 0
    ):
        raise ValueError("repeat RSS values are invalid")
    return (
        repeat_a_manifest_sha256,
        repeat_b_manifest_sha256,
        repeat_a_rss_bytes,
        repeat_b_rss_bytes,
    )


def _load_validated_worker_receipt(
    output_root: Path, *, expected_manifest_sha256: str
) -> FeasibilityReceipt:
    root = Path(output_root)
    _validate_task7_manifest(
        root,
        expected_files=_WORKER_OUTPUT_FILES,
        label="worker output",
        expected_manifest_sha256=expected_manifest_sha256,
    )
    receipt = _receipt_from_dict(
        _read_bounded_canonical_json(
            root / "feasibility_receipt.json",
            label="worker feasibility receipt",
        )
    )
    if receipt.provenance.get("output_root") != str(root):
        raise ValueError("worker receipt output root is not self-binding")
    if receipt.fit_count != 576:
        raise ValueError("worker fit count is not the locked count")
    if raw_file_sha256(root / "fits.h5") != receipt.fit_file_sha256:
        raise ValueError("worker fit file digest mismatch")
    return receipt


def _validate_worker_binding(
    receipt: FeasibilityReceipt,
    *,
    label: str,
    expected_manifest_sha256: str,
) -> None:
    worker_root_value = receipt.provenance.get("output_root")
    worker_root = _require_lexical_absolute_path(
        worker_root_value,
        label=f"repeat {label} worker output root",
    )
    if _is_embargoed_qrel_path(worker_root):
        raise ValueError(f"repeat {label} worker output root is embargoed")
    persisted = _load_validated_worker_receipt(
        worker_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    if persisted != receipt:
        raise ValueError(f"aggregate repeat {label} receipt differs from worker output")


def _resolve_repository_root_argument(repo_root: Path) -> Path:
    candidate = _require_lexical_absolute_path(
        str(repo_root),
        label="repository root",
    )
    if candidate.suffix or _is_embargoed_qrel_path(candidate):
        raise ValueError("repository root is leaf-shaped or embargoed")
    try:
        before = candidate.lstat()
    except OSError as error:
        raise ValueError("repository root is missing") from error
    if not stat.S_ISDIR(before.st_mode):
        raise ValueError("repository root is not a directory")
    try:
        resolved = candidate.resolve(strict=True)
        after = candidate.lstat()
        resolved_status = resolved.lstat()
    except OSError as error:
        raise ValueError("repository root changed during validation") from error
    if (
        resolved != candidate
        or _stat_snapshot(after) != _stat_snapshot(before)
        or _stat_snapshot(resolved_status) != _stat_snapshot(before)
    ):
        raise ValueError("repository root changed during validation")
    return resolved


def _validate_receipt_artifact_roots(
    receipts: Sequence[FeasibilityReceipt], *, repo_root: Path
) -> None:
    protected_paths = tuple(
        repo_root / relative for _field, _label, relative in _FROZEN_FEASIBILITY_PATHS
    ) + (repo_root / "data/dev/llm_dev_qrels.txt",)
    for repeat_index, receipt in enumerate(receipts):
        for field in ("fit_work_file", "output_root"):
            candidate = _require_lexical_absolute_path(
                receipt.provenance.get(field),
                label=f"repeat {repeat_index} {field}",
            )
            if _is_embargoed_qrel_path(candidate) or any(
                _paths_overlap(candidate, protected) for protected in protected_paths
            ):
                raise ValueError(
                    f"repeat {repeat_index} {field} overlaps a protected or "
                    "embargoed input"
                )


def _resolve_current_feasibility_paths(
    *,
    repo_root: Path,
    query_path: Path,
    embedding_path: Path,
    encoder_metadata_path: Path,
    protocol_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    root = _resolve_repository_root_argument(repo_root)
    supplied_paths = {
        "query_path": Path(query_path),
        "embedding_path": Path(embedding_path),
        "encoder_metadata_path": Path(encoder_metadata_path),
        "protocol_path": Path(protocol_path),
    }
    expected_paths = {
        field: root / relative for field, _label, relative in _FROZEN_FEASIBILITY_PATHS
    }

    # Complete every lexical comparison before touching any supplied leaf.
    for field, label, _relative in _FROZEN_FEASIBILITY_PATHS:
        if supplied_paths[field] != expected_paths[field]:
            raise ValueError(f"{label} path is not the official {label} path")

    resolved_paths: dict[str, Path] = {}
    for field, label, _relative in _FROZEN_FEASIBILITY_PATHS:
        candidate = supplied_paths[field]
        try:
            status = candidate.lstat()
        except OSError as error:
            raise ValueError(f"official {label} path is missing") from error
        if not stat.S_ISREG(status.st_mode):
            raise ValueError(f"official {label} path is not a regular file")
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as error:
            raise ValueError(f"official {label} path is missing") from error
        if resolved != expected_paths[field]:
            raise ValueError(
                f"{label} path does not resolve to the official {label} path"
            )
        resolved_paths[field] = resolved
    return (
        root,
        resolved_paths["query_path"],
        resolved_paths["embedding_path"],
        resolved_paths["encoder_metadata_path"],
        resolved_paths["protocol_path"],
    )


def _task7_config_payload(
    *,
    repo_root: Path,
    query_path: Path,
    embedding_path: Path,
    encoder_metadata_path: Path,
    protocol_path: Path,
) -> dict[str, str]:
    return {
        "schema": _FEASIBILITY_CONFIG_SCHEMA,
        "query_path": query_path.relative_to(repo_root).as_posix(),
        "embedding_path": embedding_path.relative_to(repo_root).as_posix(),
        "encoder_metadata_path": encoder_metadata_path.relative_to(
            repo_root
        ).as_posix(),
        "protocol_path": protocol_path.relative_to(repo_root).as_posix(),
    }


def _frozen_task7_config_payload() -> dict[str, str]:
    return {
        "schema": _FEASIBILITY_CONFIG_SCHEMA,
        **{field: relative for field, _label, relative in _FROZEN_FEASIBILITY_PATHS},
    }


def _current_feasibility_provenance(
    *,
    repo_root: Path,
    query_path: Path,
    embedding_path: Path,
    encoder_metadata_path: Path,
    protocol_path: Path,
) -> dict[str, object]:
    input_paths = _QrelFreeFrozenInputPaths(
        query_path=query_path,
        embedding_path=embedding_path,
        encoder_metadata_path=encoder_metadata_path,
    )
    inputs = load_and_validate_frozen_dense_inputs(input_paths)  # type: ignore[arg-type]
    contract, contract_sha256 = build_clustering_contract(
        inputs,
        repo_root=repo_root,
        protocol_path=protocol_path,
    )
    versions = _contract_versions(contract)
    numerical_environment = capture_numerical_environment()
    numerical_environment_sha256 = hashlib.sha256(
        numerical_environment_file_bytes(numerical_environment)
    ).hexdigest()
    pair_indices, single_indices = official_sample_indices()
    sample_fixture_sha256 = hashlib.sha256(
        canonical_json(
            {
                "pair_indices": list(map(int, pair_indices)),
                "single_indices": list(map(int, single_indices)),
            }
        )
    ).hexdigest()
    return {
        "protocol_sha256": _bounded_regular_file_sha256(
            protocol_path,
            byte_limit=_TASK7_JSON_BYTE_LIMIT,
            label="current protocol",
        ),
        "clustering_contract_sha256": contract_sha256,
        "implementation_inventory_sha256": contract_source_inventory_sha256(
            contract, category="implementation"
        ),
        "dependency_inventory_sha256": contract_source_inventory_sha256(
            contract, category="dependency"
        ),
        "runtime_inventory_sha256": runtime_source_inventory_sha256(repo_root),
        "query_file_sha256": _bounded_regular_file_sha256(
            query_path,
            byte_limit=_TASK7_JSON_BYTE_LIMIT,
            label="current query input",
        ),
        "query_input_sha256": inputs.query_input_sha256,
        "embedding_sha256": inputs.embedding_sha256,
        "similarity_sha256": inputs.similarity_sha256,
        "sample_fixture_sha256": sample_fixture_sha256,
        "numerical_environment_sha256": numerical_environment_sha256,
        "python_version": versions["python"],
        "numpy_version": versions["numpy"],
        "scipy_version": versions["scipy"],
        "scikit_learn_version": versions["scikit_learn"],
        "thread_environment": dict(numerical_environment.thread_environment),
    }


def validate_feasibility_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str,
    repo_root: Path,
    query_path: Path,
    embedding_path: Path,
    encoder_metadata_path: Path,
    protocol_path: Path,
) -> ValidatedFeasibilityGate:
    """Validate a passing Task 7 artifact without accessing qrels."""
    if not _is_sha256(expected_sha256):
        raise ValueError("expected feasibility manifest digest is invalid")
    manifest_candidate = Path(manifest_path)
    if manifest_candidate.name != "manifest.json" or _is_embargoed_qrel_path(
        manifest_candidate
    ):
        raise ValueError("feasibility manifest path must name manifest.json")
    manifest_candidate = _require_lexical_absolute_path(
        str(manifest_candidate),
        label="feasibility manifest path",
    )
    validated_repo_root = _resolve_repository_root_argument(repo_root)
    try:
        resolved_manifest_parent = manifest_candidate.parent.resolve(strict=True)
    except OSError as error:
        raise ValueError("feasibility manifest is missing") from error
    if resolved_manifest_parent != manifest_candidate.parent:
        raise ValueError("feasibility manifest parent is not canonical")
    resolved_manifest_path = resolved_manifest_parent / "manifest.json"
    manifest_bytes = _read_bounded_regular_file(
        resolved_manifest_path,
        byte_limit=_TASK7_MANIFEST_BYTE_LIMIT,
        label="aggregate feasibility manifest",
    )
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha256 != expected_sha256:
        raise ValueError("feasibility manifest digest differs from expected SHA-256")
    aggregate_root = resolved_manifest_path.parent
    _validate_task7_manifest(
        aggregate_root,
        expected_files=_AGGREGATE_OUTPUT_FILES,
        label="aggregate feasibility output",
        expected_manifest_sha256=expected_sha256,
        expected_manifest_bytes=manifest_bytes,
    )

    gate = _gate_from_dict(
        _read_bounded_canonical_json(
            aggregate_root / "feasibility_gate.json",
            label="aggregate feasibility gate",
        )
    )
    repeat_a = _receipt_from_dict(
        _read_bounded_canonical_json(
            aggregate_root / "repeat_a_receipt.json",
            label="aggregate repeat A receipt",
        )
    )
    repeat_b = _receipt_from_dict(
        _read_bounded_canonical_json(
            aggregate_root / "repeat_b_receipt.json",
            label="aggregate repeat B receipt",
        )
    )
    (
        repeat_a_manifest_sha256,
        repeat_b_manifest_sha256,
        repeat_a_rss_bytes,
        repeat_b_rss_bytes,
    ) = _repeat_measurements_from_dict(
        _read_bounded_canonical_json(
            aggregate_root / "repeat_measurements.json",
            label="aggregate repeat measurements",
        )
    )

    _validate_receipt_artifact_roots(
        (repeat_a, repeat_b),
        repo_root=validated_repo_root,
    )

    for label, receipt, expected_worker_manifest_sha256 in (
        ("A", repeat_a, repeat_a_manifest_sha256),
        ("B", repeat_b, repeat_b_manifest_sha256),
    ):
        _validate_worker_binding(
            receipt,
            label=label,
            expected_manifest_sha256=expected_worker_manifest_sha256,
        )

    observed_gate = evaluate_feasibility_repeats(
        repeat_a,
        repeat_b,
        repeat_a_rss_bytes=repeat_a_rss_bytes,
        repeat_b_rss_bytes=repeat_b_rss_bytes,
        exclusive_resource_attested=True,
    )
    if observed_gate != gate or gate.status != "pass" or gate.failures:
        raise ValueError("feasibility gate is stale or stopped")

    current_paths = _resolve_current_feasibility_paths(
        repo_root=validated_repo_root,
        query_path=query_path,
        embedding_path=embedding_path,
        encoder_metadata_path=encoder_metadata_path,
        protocol_path=protocol_path,
    )
    (
        root,
        current_query_path,
        current_embedding_path,
        current_encoder_metadata_path,
        current_protocol_path,
    ) = current_paths
    current_config_payload = _task7_config_payload(
        repo_root=root,
        query_path=current_query_path,
        embedding_path=current_embedding_path,
        encoder_metadata_path=current_encoder_metadata_path,
        protocol_path=current_protocol_path,
    )
    frozen_config_payload = _frozen_task7_config_payload()
    if current_config_payload != frozen_config_payload:
        raise ValueError("Task 7 config differs from the frozen semantic payload")
    expected_provenance = _current_feasibility_provenance(
        repo_root=root,
        query_path=current_query_path,
        embedding_path=current_embedding_path,
        encoder_metadata_path=current_encoder_metadata_path,
        protocol_path=current_protocol_path,
    )
    for receipt in (repeat_a, repeat_b):
        if any(
            receipt.provenance.get(key) != value
            for key, value in expected_provenance.items()
        ):
            raise ValueError("feasibility provenance differs from current inputs")

    _validate_task7_manifest(
        aggregate_root,
        expected_files=_AGGREGATE_OUTPUT_FILES,
        label="aggregate feasibility output",
        expected_manifest_sha256=expected_sha256,
        expected_manifest_bytes=manifest_bytes,
    )
    for label, receipt, expected_worker_manifest_sha256 in (
        ("A", repeat_a, repeat_a_manifest_sha256),
        ("B", repeat_b, repeat_b_manifest_sha256),
    ):
        _validate_worker_binding(
            receipt,
            label=label,
            expected_manifest_sha256=expected_worker_manifest_sha256,
        )
    if (
        _read_bounded_regular_file(
            resolved_manifest_path,
            byte_limit=_TASK7_MANIFEST_BYTE_LIMIT,
            label="aggregate feasibility manifest",
        )
        != manifest_bytes
    ):
        raise ValueError("feasibility manifest changed during validation")
    final_paths = _resolve_current_feasibility_paths(
        repo_root=validated_repo_root,
        query_path=query_path,
        embedding_path=embedding_path,
        encoder_metadata_path=encoder_metadata_path,
        protocol_path=protocol_path,
    )
    (
        final_root,
        final_query_path,
        final_embedding_path,
        final_encoder_metadata_path,
        final_protocol_path,
    ) = final_paths
    final_config_payload = _task7_config_payload(
        repo_root=final_root,
        query_path=final_query_path,
        embedding_path=final_embedding_path,
        encoder_metadata_path=final_encoder_metadata_path,
        protocol_path=final_protocol_path,
    )
    final_provenance = _current_feasibility_provenance(
        repo_root=final_root,
        query_path=final_query_path,
        embedding_path=final_embedding_path,
        encoder_metadata_path=final_encoder_metadata_path,
        protocol_path=final_protocol_path,
    )
    if (
        final_paths != current_paths
        or final_config_payload != frozen_config_payload
        or final_config_payload != current_config_payload
        or final_provenance != expected_provenance
        or any(
            receipt.provenance.get(key) != value
            for receipt in (repeat_a, repeat_b)
            for key, value in final_provenance.items()
        )
    ):
        raise ValueError("current feasibility provenance changed during validation")

    # The terminal artifact check must follow the second current-provenance
    # capture so neither side can drift after the other side's last check.
    _validate_task7_manifest(
        aggregate_root,
        expected_files=_AGGREGATE_OUTPUT_FILES,
        label="aggregate feasibility output",
        expected_manifest_sha256=expected_sha256,
        expected_manifest_bytes=manifest_bytes,
    )
    for label, receipt, expected_worker_manifest_sha256 in (
        ("A", repeat_a, repeat_a_manifest_sha256),
        ("B", repeat_b, repeat_b_manifest_sha256),
    ):
        _validate_worker_binding(
            receipt,
            label=label,
            expected_manifest_sha256=expected_worker_manifest_sha256,
        )
    if (
        _read_bounded_regular_file(
            resolved_manifest_path,
            byte_limit=_TASK7_MANIFEST_BYTE_LIMIT,
            label="aggregate feasibility manifest",
        )
        != manifest_bytes
    ):
        raise ValueError("feasibility manifest changed during validation")
    return ValidatedFeasibilityGate(
        manifest_sha256=manifest_sha256,
        status="pass",
        failures=(),
        freshness_passed=True,
    )
