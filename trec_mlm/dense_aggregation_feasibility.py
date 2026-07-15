from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import stat
import statistics
import time
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from trec_mlm.dense_aggregation_hashing import canonical_json, framed_sha256
from trec_mlm.dense_aggregation_artifacts import (
    atomic_artifact_directory,
    preflight_atomic_roots,
    raw_file_sha256,
    validate_manifest,
)
from trec_mlm.dense_aggregation_cache import StreamingFitCache
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
from trec_mlm.dense_aggregation_runtime import THREAD_ENVIRONMENT


_WORKER_SCHEMA = "trec-mlm-dense-aggregation-feasibility-worker-v1"
_GATE_SCHEMA = "trec-mlm-dense-aggregation-feasibility-gate-v1"
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
    "fit_receipt_sha256",
    "provenance",
    "exclusive_resource_attested",
}
_TIMING_CATEGORIES = ("pair_c20", "pair_c40", "single_c20", "single_c40")
_LOWERCASE_HEX = frozenset("0123456789abcdef")
_FIT_RECEIPT_ROW_BYTE_LIMIT = 1024 * 1024
_TIMING_ROW_BYTE_LIMIT = 16 * 1024
_TASK7_JSON_BYTE_LIMIT = 1024 * 1024
_TASK7_MANIFEST_BYTE_LIMIT = 1024 * 1024
_MISSING_LAYOUT_ENTRY = object()
_WORKER_OUTPUT_FILES = (
    "environment.json",
    "feasibility_receipt.json",
    "fits/centers.f4",
    "fits/cluster_inertia.f8",
    "fits/fit_receipts.jsonl",
    "fits/header.json",
    "fits/labels.u2",
    "fits/timings.jsonl",
)
_AGGREGATE_OUTPUT_FILES = (
    "feasibility_gate.json",
    "repeat_a_receipt.json",
    "repeat_b_receipt.json",
    "repeat_measurements.json",
)
_FIT_COMPARISON_FIELDS = (
    "sequence",
    "exclusion_kind",
    "inventory_index",
    "cluster_count",
    "fit_key_sha256",
    "support_query_ids_sha256",
    "embedding_sha256",
    "clustering_contract_sha256",
    "fit_membership_sha256",
    "cluster_membership_sha256",
    "center_sha256",
    "assignments",
)
_TIMING_EVIDENCE_FIELDS = (
    "exclusion_kind",
    "cluster_count",
    "elapsed_ns",
    "n_iter",
)
_SHARD_SCHEMAS = {
    "centers": {
        "path": "centers.f4",
        "kind": "binary",
        "dtype": "<f4",
        "layout": "C",
    },
    "labels": {
        "path": "labels.u2",
        "kind": "binary",
        "dtype": "<u2",
        "layout": "C",
    },
    "cluster_inertia": {
        "path": "cluster_inertia.f8",
        "kind": "binary",
        "dtype": "<f8",
        "layout": "C",
    },
    "fit_receipts": {
        "path": "fit_receipts.jsonl",
        "kind": "canonical-jsonl",
        "record_schema": "trec-mlm-dense-aggregation-fit-receipt-v1",
    },
    "timings": {
        "path": "timings.jsonl",
        "kind": "canonical-jsonl",
        "record_schema": "trec-mlm-dense-aggregation-fit-timing-v1",
    },
}


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
    "cache_root",
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
                if type(output_value) is not str:
                    raise ValueError("worker output root is invalid")
                fits_root = Path(output_value) / "fits"
                reopened = StreamingFitCache.open_readonly(fits_root)
                header = getattr(reopened, "_header", None)
                if type(header) is not dict:
                    raise ValueError("validated cache header is unavailable")
                del reopened
                fit_rows, fit_receipt_sha256 = _read_exact_canonical_jsonl(
                    fits_root / "fit_receipts.jsonl",
                    expected_count=receipt.fit_count,
                    byte_limit=_FIT_RECEIPT_ROW_BYTE_LIMIT,
                    label="fit receipt",
                    retained_fields=_FIT_COMPARISON_FIELDS,
                )
                if fit_receipt_sha256 != receipt.fit_receipt_sha256:
                    raise ValueError("fit receipt stream digest mismatch")
                timing_rows, _timing_sha256 = _read_exact_canonical_jsonl(
                    fits_root / "timings.jsonl",
                    expected_count=receipt.fit_count,
                    byte_limit=_TIMING_ROW_BYTE_LIMIT,
                    label="timing receipt",
                    retained_fields=_TIMING_EVIDENCE_FIELDS,
                )
                loaded.append(
                    _RepeatOutputEvidence(
                        header=dict(header),
                        fit_rows=fit_rows,
                        timing_summary=_summarize_timing_rows(timing_rows),
                    )
                )
        except (KeyError, OSError, UnicodeError, ValueError, json.JSONDecodeError):
            self._evidence_failed = True
            return None
        self._evidence = tuple(loaded)
        return self._evidence

    def fit_rows(self) -> tuple[tuple[dict[str, object], ...], ...] | None:
        evidence = self.evidence()
        if evidence is None:
            return None
        return tuple(item.fit_rows for item in evidence)


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
    fit_receipt_sha256: str
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
class _RepeatOutputEvidence:
    header: Mapping[str, object]
    fit_rows: tuple[dict[str, object], ...]
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


def _read_bounded_regular_file(path: Path, *, byte_limit: int, label: str) -> bytes:
    source = Path(path)
    try:
        status = source.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing: {source}") from error
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"{label} is not a regular file: {source}")
    if status.st_size > byte_limit:
        raise ValueError(f"{label} exceeds its {byte_limit}-byte limit")
    with source.open("rb") as handle:
        payload = handle.read(status.st_size + 1)
    if len(payload) != status.st_size:
        raise ValueError(f"{label} size changed while it was read")
    return payload


def _read_bounded_canonical_json(
    path: Path, *, byte_limit: int = _TASK7_JSON_BYTE_LIMIT, label: str
) -> object:
    payload = _read_bounded_regular_file(Path(path), byte_limit=byte_limit, label=label)
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
    root: Path, *, expected_files: Sequence[str], label: str
) -> dict[str, object]:
    artifact_root = Path(root)
    value = _read_bounded_canonical_json(
        artifact_root / "manifest.json",
        byte_limit=_TASK7_MANIFEST_BYTE_LIMIT,
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


def _read_exact_canonical_jsonl(
    path: Path,
    *,
    expected_count: int,
    byte_limit: int,
    label: str,
    retained_fields: Sequence[str] | None = None,
) -> tuple[tuple[dict[str, object], ...], str]:
    if type(expected_count) is not int or expected_count < 0:
        raise ValueError(f"{label} expected count is invalid")
    if type(byte_limit) is not int or byte_limit <= 0:
        raise ValueError(f"{label} byte limit is invalid")
    rows: list[dict[str, object]] = []
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for sequence in range(expected_count):
            line = handle.readline(byte_limit + 1)
            if line == b"":
                raise ValueError(f"{label} stream ended before its expected count")
            if len(line) > byte_limit or not line.endswith(b"\n"):
                raise ValueError(f"{label} row is oversized or unterminated")
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError(f"{label} row is not an object")
            if canonical_json(value) + b"\n" != line:
                raise ValueError(f"{label} row is not canonical JSON")
            if value.get("sequence") != sequence:
                raise ValueError(f"{label} sequence is not canonical")
            digest.update(line)
            if retained_fields is None:
                rows.append(dict(value))
            else:
                try:
                    rows.append({field: value[field] for field in retained_fields})
                except KeyError as error:
                    raise ValueError(
                        f"{label} row is missing a retained field"
                    ) from error
        if handle.read(1) != b"":
            raise ValueError(f"{label} stream contains data after its expected count")
    return tuple(rows), digest.hexdigest()


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
        "fit_receipt_sha256": receipt.fit_receipt_sha256,
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
        "cache_root",
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
    for field in ("cache_root", "output_root"):
        root_value = provenance[field]
        if type(root_value) is not str:
            raise ValueError("feasibility provenance root is invalid")
        root = Path(root_value)
        if not root.is_absolute() or str(root.resolve()) != root_value:
            raise ValueError("feasibility provenance root is not normalized absolute")
        normalized_roots.append(root)
    if _paths_overlap(normalized_roots[0], normalized_roots[1]):
        raise ValueError("feasibility cache and output roots overlap")
    worker_pid = provenance["worker_pid"]
    if type(worker_pid) is not int or worker_pid <= 0:
        raise ValueError("feasibility provenance worker_pid is invalid")
    fit_receipt_sha256 = value["fit_receipt_sha256"]
    if not _is_sha256(fit_receipt_sha256):
        raise ValueError("fit receipt digest is invalid")
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
        fit_receipt_sha256=fit_receipt_sha256,
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


def _worker_expected_inventory(
    pair_indices: Sequence[int], single_indices: Sequence[int]
) -> list[dict[str, object]]:
    return [
        {
            "exclusion_kind": exclusion_kind,
            "inventory_index": int(inventory_index),
            "cluster_count": cluster_count,
        }
        for exclusion_kind, sampled_indices in (
            ("pair", pair_indices),
            ("single", single_indices),
        )
        for inventory_index in sampled_indices
        for cluster_count in (20, 40)
    ]


def _cache_header(
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
    pair_indices: Sequence[int],
    single_indices: Sequence[int],
) -> dict[str, object]:
    expected_inventory = _worker_expected_inventory(pair_indices, single_indices)
    pair_fit_count = 2 * len(pair_indices)
    single_fit_count = 2 * len(single_indices)
    fit_count = pair_fit_count + single_fit_count
    return {
        "schema": "trec-mlm-dense-aggregation-fit-cache-header-v1",
        "query_ids": list(query_ids),
        "query_input_sha256": query_input_sha256,
        "embedding_sha256": embedding_sha256,
        "similarity_sha256": similarity_sha256,
        "clustering_contract_sha256": clustering_contract_sha256,
        "implementation_inventory_sha256": implementation_inventory_sha256,
        "dependency_inventory_sha256": dependency_inventory_sha256,
        "runtime_inventory_sha256": runtime_inventory_sha256,
        "numerical_environment_sha256": numerical_environment_sha256,
        "shard_schemas": _SHARD_SCHEMAS,
        "expected_inventory": expected_inventory,
        "expected_counts": {
            "pair_fit_count": pair_fit_count,
            "single_fit_count": single_fit_count,
            "fit_count": fit_count,
            "start_count": 10 * fit_count,
            "timing_record_count": fit_count,
        },
    }


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
    cache_dir: Path,
    output_dir: Path,
    exclusive_resource_attested: bool,
) -> FeasibilityReceipt:
    preflight_atomic_roots(cache_dir=cache_dir, output_dir=output_dir)

    cache_root = Path(cache_dir).resolve()
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
    header = _cache_header(
        query_ids=inputs.query_ids,
        query_input_sha256=inputs.query_input_sha256,
        embedding_sha256=inputs.embedding_sha256,
        similarity_sha256=inputs.similarity_sha256,
        clustering_contract_sha256=clustering_contract_sha256,
        implementation_inventory_sha256=implementation_inventory_sha256,
        dependency_inventory_sha256=dependency_inventory_sha256,
        runtime_inventory_sha256=runtime_inventory_sha256,
        numerical_environment_sha256=numerical_environment_sha256,
        pair_indices=pair_indices,
        single_indices=single_indices,
    )
    timing_rows: list[dict[str, object]] = []
    pairs = tuple(itertools.combinations(range(322), 2))

    with atomic_artifact_directory(output_root, quarantine_on_error=True) as staging:
        _write_new_file(staging / "environment.json", environment_bytes)
        cache = StreamingFitCache.create(cache_root, header=header)
        require_same_numerical_environment(numerical_environment)
        _verify_environment_file(
            staging / "environment.json", numerical_environment_sha256
        )
        for exclusion_kind, sampled_indices in (
            ("pair", pair_indices),
            ("single", single_indices),
        ):
            for inventory_index in sampled_indices:
                excluded = (
                    pairs[inventory_index]
                    if exclusion_kind == "pair"
                    else (inventory_index,)
                )
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
                    cache.append_fit(
                        fit,
                        assignments,
                        exclusion_kind=exclusion_kind,
                        inventory_index=inventory_index,
                        excluded_query_ids=tuple(
                            inputs.query_ids[index] for index in excluded
                        ),
                    )
                    cache.release(fit.key)
                    n_iter = fit.n_iter
                    del assignments, fit, support, support_index_by_id
                    end_ns = time.perf_counter_ns()
                    elapsed_ns = end_ns - start_ns
                    cache.append_timing(
                        exclusion_kind=exclusion_kind,
                        inventory_index=inventory_index,
                        cluster_count=cluster_count,
                        elapsed_ns=elapsed_ns,
                        n_iter=n_iter,
                    )
                    timing_rows.append(
                        {
                            "exclusion_kind": exclusion_kind,
                            "inventory_index": inventory_index,
                            "cluster_count": cluster_count,
                            "elapsed_ns": elapsed_ns,
                            "n_iter": n_iter,
                        }
                    )
        require_same_numerical_environment(numerical_environment)
        _verify_environment_file(
            staging / "environment.json", numerical_environment_sha256
        )
        cache.finalize()
        fit_receipt_sha256 = raw_file_sha256(cache_root / "fit_receipts.jsonl")
        cache.finalize_into(staging / "fits")

        receipt = FeasibilityReceipt(
            fit_count=576,
            start_count=5_760,
            pair_indices=pair_indices,
            single_indices=single_indices,
            timing_summary=_summarize_timing_rows(timing_rows),
            fit_receipt_sha256=fit_receipt_sha256,
            provenance={
                "config_sha256": config.source_sha256,
                "protocol_sha256": raw_file_sha256(config.protocol_path),
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
                "cache_root": str(cache_root),
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
    reopened = StreamingFitCache.open_readonly(output_root / "fits")
    del reopened
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
        header_fields = {
            "clustering_contract": "clustering_contract_sha256",
            "implementation_inventory": "implementation_inventory_sha256",
            "dependency_inventory": "dependency_inventory_sha256",
            "runtime_inventory": "runtime_inventory_sha256",
            "query_input": "query_input_sha256",
            "embedding": "embedding_sha256",
            "similarity": "similarity_sha256",
        }
        if name in header_fields:
            context = _ACTIVE_REPEAT_CONTEXT.get()
            if context is None:
                context = _RepeatCheckContext(repeat_a, repeat_b)
            evidence = context.evidence()
            header_field = header_fields[name]
            if evidence is None or any(
                item.header.get(header_field) != receipt.provenance.get(field)
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
                if item.header.get("numerical_environment_sha256") != expected_hash:
                    return False
                try:
                    if (
                        raw_file_sha256(Path(output_value) / "environment.json")
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
            item.header.get("expected_inventory")
            == _worker_expected_inventory(receipt.pair_indices, receipt.single_indices)
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
            item.header.get("expected_counts")
            == {
                "pair_fit_count": 2 * len(receipt.pair_indices),
                "single_fit_count": 2 * len(receipt.single_indices),
                "fit_count": receipt.fit_count,
                "start_count": receipt.start_count,
                "timing_record_count": receipt.fit_count,
            }
            for item, receipt in zip(evidence, (repeat_a, repeat_b), strict=True)
        )
    if name == "start_count":
        if repeat_a.start_count != 5_760 or repeat_b.start_count != 5_760:
            return False
        context = _ACTIVE_REPEAT_CONTEXT.get()
        if context is None:
            context = _RepeatCheckContext(repeat_a, repeat_b)
        evidence = context.evidence()
        return evidence is not None and all(
            item.header.get("expected_counts", {}).get("start_count")
            == receipt.start_count
            for item, receipt in zip(evidence, (repeat_a, repeat_b), strict=True)
        )
    if name in {
        "fit_supports",
        "fit_memberships",
        "fit_centers",
        "target_assignments",
    }:
        context = _ACTIVE_REPEAT_CONTEXT.get()
        if context is None:
            context = _RepeatCheckContext(repeat_a, repeat_b)
        rows = context.fit_rows()
        if rows is None:
            return False
        left_rows, right_rows = rows
        fields = {
            "fit_supports": (
                "sequence",
                "exclusion_kind",
                "inventory_index",
                "cluster_count",
                "fit_key_sha256",
                "support_query_ids_sha256",
                "embedding_sha256",
                "clustering_contract_sha256",
            ),
            "fit_memberships": (
                "sequence",
                "fit_membership_sha256",
                "cluster_membership_sha256",
            ),
            "fit_centers": ("sequence", "center_sha256"),
            "target_assignments": ("sequence", "assignments"),
        }[name]
        try:
            left_values = tuple(
                tuple(row[field] for field in fields) for row in left_rows
            )
            right_values = tuple(
                tuple(row[field] for field in fields) for row in right_rows
            )
        except KeyError:
            return False
        if name == "fit_supports" and (
            repeat_a.fit_receipt_sha256 != repeat_b.fit_receipt_sha256
        ):
            return False
        return left_values == right_values
    if name == "manifests":
        for receipt in (repeat_a, repeat_b):
            output_value = receipt.provenance.get("output_root")
            if type(output_value) is not str:
                return False
            output_root = Path(output_value)
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
                if (
                    raw_file_sha256(output_root / "fits/fit_receipts.jsonl")
                    != receipt.fit_receipt_sha256
                ):
                    return False
            except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
                return False
        return True
    if name == "fresh_distinct_roots":
        roots: list[Path] = []
        for receipt in (repeat_a, repeat_b):
            for field in ("cache_root", "output_root"):
                value = receipt.provenance.get(field)
                if type(value) is not str:
                    return False
                candidate = Path(value)
                if not candidate.is_absolute() or str(candidate.resolve()) != value:
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
