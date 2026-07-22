"""Exact candidate-local scoring for raw historical response refinement."""

from __future__ import annotations

from collections.abc import Sequence
import ctypes
from dataclasses import asdict, dataclass, field, fields
import errno
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.artifact_io import (
    manifest_aggregate_sha256,
    manifest_entries,
    sha256_file,
    source_tree_sha256,
    write_pretty_json,
)
from trec_mlm.profile_experiment import (
    load_profile_artifacts,
    validate_complete_rankings,
)
from trec_mlm.profile_inference_artifact import load_profile_inference_artifacts
from trec_mlm.profile_retrieval import reciprocal_rank_fusion
from trec_mlm.runio import rankings_from_run, validate_trec_run, write_trec_run


ARTIFACT_VERSION = 1
RAW_TOP_K = 10
RRF_K = 60
RANKING_KEYS = ("raw_hierarchical", "raw_hierarchical_dense_equal")
METADATA_KEYS = frozenset(
    {
        "qrel_free",
        "source_tree_sha256",
        "dependency_aggregate_sha256",
        "raw_top_k",
        "hierarchical_weights",
        "rrf",
    }
)
ARTIFACT_FILES = frozenset(
    {
        "broad_scores.npy",
        "config.json",
        "diagnostics.json",
        "hierarchical_scores.npy",
        "inputs.json",
        "metadata.json",
        "rankings/dense_reference.json",
        "rankings/raw_hierarchical.json",
        "rankings/raw_hierarchical_dense_equal.json",
        "raw_scores.npy",
        "runs/dense_reference.trec",
        "runs/raw_hierarchical.trec",
        "runs/raw_hierarchical_dense_equal.trec",
        "scores/raw_hierarchical_dense_equal.npy",
        "selected_record_ids.json",
        "top_record_indices.npy",
        "top_record_similarities.npy",
        "validation.json",
    }
)


@dataclass(frozen=True, slots=True)
class RawRefinementConfig:
    """Immutable qrel-free inputs for raw-response refinement."""

    inference_path: str
    construction_path: str
    dense_artifact_path: str


@dataclass(frozen=True, slots=True)
class RawRefinementResult:
    """Complete in-memory representation of one refinement artifact."""

    config: RawRefinementConfig
    query_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    broad_scores: NDArray[np.float64]
    raw_scores: NDArray[np.float64]
    hierarchical_scores: NDArray[np.float64]
    top_record_indices: NDArray[np.int32]
    top_record_similarities: NDArray[np.float64]
    selected_record_ids: tuple[tuple[tuple[str, ...], ...], ...]
    rankings: dict[str, dict[str, list[str]]]
    fusion_scores: NDArray[np.float32]
    dense_rankings: dict[str, list[str]]
    diagnostics: dict[str, Any]
    metadata: dict[str, Any]
    _dependencies: Any = field(repr=False, compare=False)
    validation: dict[str, Any] | None = None
    manifest: dict[str, Any] | None = None
    root: Path | None = None


_write_json = write_pretty_json


def _validate_config(config: RawRefinementConfig) -> None:
    for descriptor in fields(config):
        value = getattr(config, descriptor.name)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{descriptor.name} must be a non-empty path")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_embedding_matrix(
    matrix: NDArray[np.floating], *, rows: int, label: str
) -> None:
    values = np.asarray(matrix)
    if values.dtype != np.dtype("<f4"):
        raise ValueError(f"{label} embeddings must be exact float32")
    if not values.flags.c_contiguous:
        raise ValueError(f"{label} embeddings must be C-contiguous")
    if values.ndim != 2 or values.shape[0] != rows or not values.shape[1]:
        raise ValueError(f"{label} embedding shape changed")
    if not np.isfinite(values).all():
        raise ValueError(f"{label} embeddings must be finite")
    if not np.allclose(
        np.linalg.norm(values, axis=1), 1.0, atol=1e-5, rtol=0.0
    ):
        raise ValueError(f"{label} embeddings must be unit-normalized")


def _closed_dense_dependency(
    artifact_path: str,
) -> tuple[dict[str, list[str]], str]:
    supplied = Path(artifact_path)
    root = supplied if supplied.is_dir() else supplied.parent
    run_path = root / "run.trec" if supplied.is_dir() else supplied
    if not run_path.is_file():
        raise ValueError("dense artifact does not contain run.trec")
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("files")
        actual = manifest_entries(root)
        if (
            not isinstance(entries, dict)
            or entries != actual
            or manifest.get("aggregate_sha256")
            != manifest_aggregate_sha256(entries)
        ):
            raise ValueError("dense artifact manifest closure changed")
        aggregate = manifest["aggregate_sha256"]
    else:
        entries = manifest_entries(root) if supplied.is_dir() else {
            run_path.name: {
                "byte_size": run_path.stat().st_size,
                "sha256": sha256_file(run_path),
            }
        }
        aggregate = manifest_aggregate_sha256(entries)
    return rankings_from_run(run_path), aggregate


def _load_dependencies(config: RawRefinementConfig, fixture: Any | None) -> Any:
    if fixture is not None:
        dependencies = fixture
    else:
        inference = load_profile_inference_artifacts(config.inference_path)
        construction = load_profile_artifacts(config.construction_path)
        dense_rankings, dense_aggregate = _closed_dense_dependency(
            config.dense_artifact_path
        )
        dependencies = SimpleNamespace(
            inference=inference,
            construction=construction,
            construction_query_embeddings=np.load(
                construction.root / "query_embeddings.npy", allow_pickle=False
            ),
            construction_record_embeddings=np.load(
                construction.root / "record_embeddings.npy", allow_pickle=False
            ),
            dense_rankings=dense_rankings,
            dense_aggregate_sha256=dense_aggregate,
            construction_metadata=json.loads(
                (construction.root / "metadata.json").read_text(encoding="utf-8")
            ),
            construction_records=json.loads(
                (construction.root / "records.json").read_text(encoding="utf-8")
            ),
        )

    inference = dependencies.inference
    construction = dependencies.construction
    construction_metadata = (
        dependencies.construction_metadata
        if hasattr(dependencies, "construction_metadata")
        else construction.metadata
    )
    construction_records = (
        dependencies.construction_records
        if hasattr(dependencies, "construction_records")
        else construction.records
    )
    if (
        tuple(inference.query_ids) != tuple(construction.query_ids)
        or tuple(inference.candidate_ids) != tuple(construction.candidate_ids)
    ):
        raise ValueError("inference and construction inventories differ")
    if (
        construction_metadata.get("inference_aggregate_sha256")
        != inference.manifest.get("aggregate_sha256")
    ):
        raise ValueError("construction inference aggregate differs from supplied inference")
    if construction_records != inference.records:
        raise ValueError("construction record identities differ from supplied inference")
    _validate_embedding_matrix(
        inference.query_embeddings, rows=len(inference.query_ids), label="query"
    )
    _validate_embedding_matrix(
        inference.record_embeddings, rows=len(inference.records), label="record"
    )
    construction_queries = (
        dependencies.construction_query_embeddings
        if hasattr(dependencies, "construction_query_embeddings")
        else construction.query_embeddings
    )
    construction_records = (
        dependencies.construction_record_embeddings
        if hasattr(dependencies, "construction_record_embeddings")
        else construction.record_embeddings
    )
    _validate_embedding_matrix(
        construction_queries,
        rows=len(construction.query_ids),
        label="construction query",
    )
    _validate_embedding_matrix(
        construction_records,
        rows=len(inference.records),
        label="construction record",
    )
    if not np.array_equal(
        inference.query_embeddings, construction_queries
    ) or not np.array_equal(
        inference.record_embeddings, construction_records
    ):
        raise ValueError("inference and construction embeddings differ")
    validate_complete_rankings(
        dependencies.dense_rankings,
        inference.query_ids,
        inference.candidate_ids,
    )
    if construction.source_rankings.get("dense_reference") != dependencies.dense_rankings:
        raise ValueError("dense dependency differs from construction reference")
    return dependencies


def _candidate_slices(
    records: Sequence[dict[str, Any]], candidate_ids: Sequence[str]
) -> tuple[dict[str, tuple[int, int]], tuple[str, ...]]:
    positions = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    record_ids: list[str] = []
    slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for candidate_id in candidate_ids:
        start = cursor
        while cursor < len(records) and records[cursor].get("candidate_id") == candidate_id:
            record_id = records[cursor].get("record_id")
            if not isinstance(record_id, str) or not record_id:
                raise ValueError("record IDs must be non-empty strings")
            record_ids.append(record_id)
            cursor += 1
        slices[candidate_id] = (start, cursor)
    if cursor != len(records) or any(
        record.get("candidate_id") not in positions for record in records
    ):
        raise ValueError("records are not contiguous candidate blocks")
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("record IDs must be globally unique")
    return slices, tuple(record_ids)


def _diagnostics(
    candidate_ids: Sequence[str], slices: dict[str, tuple[int, int]]
) -> dict[str, Any]:
    counts = {
        candidate_id: slices[candidate_id][1] - slices[candidate_id][0]
        for candidate_id in candidate_ids
    }
    values = list(counts.values())
    return {
        "record_counts_by_candidate": counts,
        "minimum_record_count": min(values),
        "maximum_record_count": max(values),
        "candidates_below_top_k": [
            candidate_id
            for candidate_id in candidate_ids
            if counts[candidate_id] < RAW_TOP_K
        ],
        "unavailable_candidate_ids": [
            candidate_id for candidate_id in candidate_ids if counts[candidate_id] == 0
        ],
    }


def construct_raw_response_refinement(
    config: RawRefinementConfig, *, _fixture_dependencies: Any | None = None
) -> RawRefinementResult:
    """Construct both frozen raw-response systems without loading judgments."""

    _validate_config(config)
    dependencies = _load_dependencies(config, _fixture_dependencies)
    inference = dependencies.inference
    construction = dependencies.construction
    query_ids = tuple(inference.query_ids)
    candidate_ids = tuple(inference.candidate_ids)
    slices, record_ids = _candidate_slices(inference.records, candidate_ids)
    shape = (len(query_ids), len(candidate_ids))
    broad_scores = np.asarray(construction.profile_scores, dtype=np.float64)
    if broad_scores.shape != shape or not np.isfinite(broad_scores).all():
        raise ValueError("broad score inventory changed")
    raw_scores = np.full(shape, -2.0, dtype=np.float64)
    top_indices = np.full((*shape, RAW_TOP_K), -1, dtype=np.int32)
    top_similarities = np.full((*shape, RAW_TOP_K), np.nan, dtype=np.float64)
    selected: list[list[tuple[str, ...]]] = [
        [tuple() for _ in candidate_ids] for _ in query_ids
    ]
    for candidate_index, candidate_id in enumerate(candidate_ids):
        start, stop = slices[candidate_id]
        if start == stop:
            continue
        raw = score_candidate_records(
            inference.query_embeddings,
            inference.record_embeddings[start:stop],
            record_ids[start:stop],
            top_k=RAW_TOP_K,
        )
        count = raw.record_indices.shape[1]
        raw_scores[:, candidate_index] = raw.mean_scores
        top_indices[:, candidate_index, :count] = raw.record_indices + start
        top_similarities[:, candidate_index, :count] = raw.similarities
        for query_index in range(len(query_ids)):
            selected[query_index][candidate_index] = raw.record_ids[query_index]
    hierarchical = combine_hierarchical_scores(broad_scores, raw_scores)
    raw_rankings = rank_score_matrix(query_ids, candidate_ids, hierarchical)
    diagnostics = _diagnostics(candidate_ids, slices)
    fused = reciprocal_rank_fusion(
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        rankings_by_signal={
            "profile": raw_rankings,
            "dense": dependencies.dense_rankings,
        },
        weights={"profile": 1.0, "dense": 1.0},
        unavailable_profile_ids=frozenset(diagnostics["unavailable_candidate_ids"]),
        k=RRF_K,
    )
    selected_ids = tuple(tuple(row) for row in selected)
    return RawRefinementResult(
        config=config,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        broad_scores=broad_scores,
        raw_scores=raw_scores,
        hierarchical_scores=hierarchical,
        top_record_indices=top_indices,
        top_record_similarities=top_similarities,
        selected_record_ids=selected_ids,
        rankings={
            "raw_hierarchical": raw_rankings,
            "raw_hierarchical_dense_equal": fused.rankings,
        },
        fusion_scores=fused.scores,
        dense_rankings={key: list(value) for key, value in dependencies.dense_rankings.items()},
        diagnostics=diagnostics,
        metadata={
            "qrel_free": True,
            "source_tree_sha256": source_tree_sha256(Path(__file__).resolve().parent),
            "dependency_aggregate_sha256": {
                "inference": inference.manifest["aggregate_sha256"],
                "construction": construction.manifest["aggregate_sha256"],
                "dense": dependencies.dense_aggregate_sha256,
            },
            "raw_top_k": RAW_TOP_K,
            "hierarchical_weights": {"broad": 0.8, "raw": 0.2},
            "rrf": {"k": RRF_K, "dense_weight": 1.0, "raw_weight": 1.0},
        },
        _dependencies=dependencies,
    )


def _validation_payload(result: RawRefinementResult) -> dict[str, Any]:
    return {
        "status": "passed",
        "qrel_free": True,
        "query_count": len(result.query_ids),
        "candidate_count": len(result.candidate_ids),
        "record_count": int(sum(result.diagnostics["record_counts_by_candidate"].values())),
        "ranking_keys": list(RANKING_KEYS),
        "integrity_checks": {
            "qrel_independence": True,
            "dependency_bindings": True,
            "embedding_contract": True,
            "inventory": True,
            "semantic_reconstruction": True,
            "trec_completeness": True,
        },
    }


def _validate_result_for_write(result: RawRefinementResult) -> None:
    _validate_config(result.config)
    shape = (len(result.query_ids), len(result.candidate_ids))
    if (
        not result.query_ids
        or not result.candidate_ids
        or len(set(result.query_ids)) != len(result.query_ids)
        or len(set(result.candidate_ids)) != len(result.candidate_ids)
    ):
        raise ValueError("raw refinement result inventories must be non-empty and unique")
    arrays = (
        ("broad", result.broad_scores, np.dtype("<f8"), shape),
        ("raw", result.raw_scores, np.dtype("<f8"), shape),
        ("hierarchical", result.hierarchical_scores, np.dtype("<f8"), shape),
        (
            "top-record index",
            result.top_record_indices,
            np.dtype("<i4"),
            (*shape, RAW_TOP_K),
        ),
        (
            "top-record similarity",
            result.top_record_similarities,
            np.dtype("<f8"),
            (*shape, RAW_TOP_K),
        ),
        ("equal-RRF", result.fusion_scores, np.dtype("<f4"), shape),
    )
    for label, values, dtype, expected_shape in arrays:
        if (
            values.dtype != dtype
            or values.shape != expected_shape
            or not values.flags.c_contiguous
        ):
            raise ValueError(f"{label} result array contract changed")
    if not all(
        np.isfinite(values).all()
        for values in (
            result.broad_scores,
            result.raw_scores,
            result.hierarchical_scores,
            result.fusion_scores,
        )
    ):
        raise ValueError("raw refinement result scores must be finite")
    valid = result.top_record_indices >= 0
    if (
        not np.isfinite(result.top_record_similarities[valid]).all()
        or not np.isnan(result.top_record_similarities[~valid]).all()
        or np.any(result.top_record_indices[~valid] != -1)
    ):
        raise ValueError("top-record padding or similarity contract changed")
    selected_shape = tuple(
        tuple(len(candidate) for candidate in query)
        for query in result.selected_record_ids
    )
    expected_selected_shape = tuple(
        tuple(int(np.count_nonzero(valid[q, c])) for c in range(shape[1]))
        for q in range(shape[0])
    )
    if selected_shape != expected_selected_shape:
        raise ValueError("selected record identity shape changed")
    recombined = combine_hierarchical_scores(result.broad_scores, result.raw_scores)
    if not np.array_equal(recombined, result.hierarchical_scores):
        raise ValueError("hierarchical score arithmetic changed before write")
    expected_raw_ranking = rank_score_matrix(
        result.query_ids, result.candidate_ids, result.hierarchical_scores
    )
    validate_complete_rankings(
        result.dense_rankings, result.query_ids, result.candidate_ids
    )
    if result.rankings.get("raw_hierarchical") != expected_raw_ranking:
        raise ValueError("raw hierarchical ranking changed before write")
    fused = reciprocal_rank_fusion(
        query_ids=result.query_ids,
        candidate_ids=result.candidate_ids,
        rankings_by_signal={
            "profile": expected_raw_ranking,
            "dense": result.dense_rankings,
        },
        weights={"profile": 1.0, "dense": 1.0},
        unavailable_profile_ids=frozenset(
            result.diagnostics["unavailable_candidate_ids"]
        ),
        k=RRF_K,
    )
    if (
        set(result.rankings) != set(RANKING_KEYS)
        or result.rankings["raw_hierarchical_dense_equal"] != fused.rankings
        or not np.array_equal(result.fusion_scores, fused.scores)
    ):
        raise ValueError("equal-RRF result changed before write")
    bindings = result.metadata.get("dependency_aggregate_sha256")
    if (
        set(result.metadata) != METADATA_KEYS
        or result.metadata.get("qrel_free") is not True
        or not _is_sha256(result.metadata.get("source_tree_sha256"))
        or not isinstance(bindings, dict)
        or set(bindings) != {"inference", "construction", "dense"}
        or any(not _is_sha256(value) for value in bindings.values())
    ):
        raise ValueError("raw refinement provenance changed before write")
    if result._dependencies is None:
        raise ValueError("raw refinement dependencies are unavailable before write")
    reconstructed = construct_raw_response_refinement(
        result.config, _fixture_dependencies=result._dependencies
    )
    if result.selected_record_ids != reconstructed.selected_record_ids:
        raise ValueError("selected record identity arithmetic changed before write")
    evidence_arrays = (
        (result.broad_scores, reconstructed.broad_scores),
        (result.raw_scores, reconstructed.raw_scores),
        (result.hierarchical_scores, reconstructed.hierarchical_scores),
        (result.top_record_indices, reconstructed.top_record_indices),
        (result.fusion_scores, reconstructed.fusion_scores),
    )
    if any(not np.array_equal(left, right) for left, right in evidence_arrays) or not np.array_equal(
        result.top_record_similarities,
        reconstructed.top_record_similarities,
        equal_nan=True,
    ):
        raise ValueError("raw refinement evidence arithmetic changed before write")
    if (
        result.rankings != reconstructed.rankings
        or result.dense_rankings != reconstructed.dense_rankings
        or result.diagnostics != reconstructed.diagnostics
        or result.metadata != reconstructed.metadata
    ):
        raise ValueError("raw refinement semantic reconstruction changed before write")


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically publish a directory only when the destination is absent."""

    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    )
    if result == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number, os.strerror(error_number), str(destination)
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def write_raw_response_refinement_artifact(
    result: RawRefinementResult, output_dir: str | Path
) -> dict[str, str]:
    """Atomically persist a complete refinement artifact without overwrite."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    _validate_result_for_write(result)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _write_json(
            staging / "config.json",
            {
                "artifact_type": "raw_response_refinement",
                "version": ARTIFACT_VERSION,
                "qrel_free": True,
                **asdict(result.config),
            },
        )
        _write_json(
            staging / "inputs.json",
            {"query_ids": result.query_ids, "candidate_ids": result.candidate_ids},
        )
        _write_json(staging / "metadata.json", result.metadata)
        _write_json(staging / "diagnostics.json", result.diagnostics)
        _write_json(staging / "selected_record_ids.json", result.selected_record_ids)
        arrays = {
            "broad_scores.npy": result.broad_scores,
            "raw_scores.npy": result.raw_scores,
            "hierarchical_scores.npy": result.hierarchical_scores,
            "top_record_indices.npy": result.top_record_indices,
            "top_record_similarities.npy": result.top_record_similarities,
            "scores/raw_hierarchical_dense_equal.npy": result.fusion_scores,
        }
        for name, values in arrays.items():
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, values, allow_pickle=False)
        all_rankings = {"dense_reference": result.dense_rankings, **result.rankings}
        for key, ranking in all_rankings.items():
            validate_complete_rankings(ranking, result.query_ids, result.candidate_ids)
            _write_json(staging / "rankings" / f"{key}.json", ranking)
            write_trec_run(staging / "runs" / f"{key}.trec", ranking, run_id=key)
            validate_trec_run(
                staging / "runs" / f"{key}.trec",
                expected_query_ids=result.query_ids,
                expected_model_ids=result.candidate_ids,
            )
        _write_json(staging / "validation.json", _validation_payload(result))
        entries = manifest_entries(staging)
        if set(entries) != ARTIFACT_FILES:
            raise ValueError("raw refinement file inventory changed before finalization")
        _write_json(
            staging / "manifest.json",
            {
                "artifact_type": "raw_response_refinement_manifest",
                "version": ARTIFACT_VERSION,
                "qrel_free": True,
                "files": entries,
                "aggregate_sha256": manifest_aggregate_sha256(entries),
            },
        )
        _rename_noreplace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        path.relative_to(destination).as_posix(): str(path)
        for path in destination.rglob("*")
        if path.is_file()
    }


def _load_array(
    root: Path, name: str, dtype: np.dtype[Any], shape: tuple[int, ...]
) -> NDArray[Any]:
    values = np.load(root / name, allow_pickle=False)
    if (
        values.dtype != dtype
        or values.shape != shape
        or not values.flags.c_contiguous
    ):
        raise ValueError(f"persisted array contract changed: {name}")
    return values


def load_raw_response_refinement_artifact(
    artifact_dir: str | Path, *, _fixture_dependencies: Any | None = None
) -> RawRefinementResult:
    """Strictly validate hashes, dependencies, and semantic reconstruction."""

    root = Path(artifact_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if (
        set(manifest) != {
            "artifact_type",
            "version",
            "qrel_free",
            "files",
            "aggregate_sha256",
        }
        or manifest.get("artifact_type") != "raw_response_refinement_manifest"
        or manifest.get("version") != ARTIFACT_VERSION
        or manifest.get("qrel_free") is not True
        or not isinstance(entries, dict)
        or set(entries) != ARTIFACT_FILES
        or manifest.get("aggregate_sha256") != manifest_aggregate_sha256(entries)
    ):
        raise ValueError("raw refinement manifest contract changed")
    actual = manifest_entries(root)
    if set(actual) != ARTIFACT_FILES:
        raise ValueError("raw refinement artifact file inventory changed")
    for name, expected in entries.items():
        if (
            not isinstance(expected, dict)
            or set(expected) != {"byte_size", "sha256"}
            or actual[name] != expected
        ):
            raise ValueError(f"raw refinement artifact hash mismatch: {name}")
    config_payload = json.loads((root / "config.json").read_text(encoding="utf-8"))
    expected_config_keys = {
        "artifact_type", "version", "qrel_free", *(field.name for field in fields(RawRefinementConfig))
    }
    if (
        set(config_payload) != expected_config_keys
        or config_payload["artifact_type"] != "raw_response_refinement"
        or config_payload["version"] != ARTIFACT_VERSION
        or config_payload["qrel_free"] is not True
    ):
        raise ValueError("raw refinement configuration schema changed")
    config = RawRefinementConfig(
        **{field.name: config_payload[field.name] for field in fields(RawRefinementConfig)}
    )
    _validate_config(config)
    dependencies = _load_dependencies(config, _fixture_dependencies)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    expected_bindings = {
        "inference": dependencies.inference.manifest["aggregate_sha256"],
        "construction": dependencies.construction.manifest["aggregate_sha256"],
        "dense": dependencies.dense_aggregate_sha256,
    }
    if (
        set(metadata) != METADATA_KEYS
        or metadata.get("qrel_free") is not True
        or not _is_sha256(metadata.get("source_tree_sha256"))
        or metadata.get("dependency_aggregate_sha256") != expected_bindings
        or metadata.get("raw_top_k") != RAW_TOP_K
        or metadata.get("hierarchical_weights") != {"broad": 0.8, "raw": 0.2}
        or metadata.get("rrf") != {"k": RRF_K, "dense_weight": 1.0, "raw_weight": 1.0}
    ):
        raise ValueError("raw refinement dependency or frozen configuration changed")
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    if set(inputs) != {"query_ids", "candidate_ids"}:
        raise ValueError("raw refinement input schema changed")
    query_ids = tuple(inputs["query_ids"])
    candidate_ids = tuple(inputs["candidate_ids"])
    if query_ids != tuple(dependencies.inference.query_ids) or candidate_ids != tuple(
        dependencies.inference.candidate_ids
    ):
        raise ValueError("raw refinement input inventory differs from dependencies")
    shape = (len(query_ids), len(candidate_ids))
    broad = _load_array(root, "broad_scores.npy", np.dtype("<f8"), shape)
    raw = _load_array(root, "raw_scores.npy", np.dtype("<f8"), shape)
    hierarchical = _load_array(root, "hierarchical_scores.npy", np.dtype("<f8"), shape)
    indices = _load_array(root, "top_record_indices.npy", np.dtype("<i4"), (*shape, RAW_TOP_K))
    similarities = _load_array(root, "top_record_similarities.npy", np.dtype("<f8"), (*shape, RAW_TOP_K))
    fusion = _load_array(root, "scores/raw_hierarchical_dense_equal.npy", np.dtype("<f4"), shape)
    if not all(np.isfinite(values).all() for values in (broad, raw, hierarchical, fusion)):
        raise ValueError("raw refinement scores must be finite")
    selected_payload = json.loads((root / "selected_record_ids.json").read_text(encoding="utf-8"))
    selected_ids = tuple(
        tuple(tuple(candidate) for candidate in query) for query in selected_payload
    )
    dense_rankings = json.loads((root / "rankings" / "dense_reference.json").read_text(encoding="utf-8"))
    rankings = {
        key: json.loads((root / "rankings" / f"{key}.json").read_text(encoding="utf-8"))
        for key in RANKING_KEYS
    }
    for key, ranking in {"dense_reference": dense_rankings, **rankings}.items():
        validate_complete_rankings(ranking, query_ids, candidate_ids)
        if rankings_from_run(root / "runs" / f"{key}.trec") != ranking:
            raise ValueError(f"TREC run disagrees with ranking: {key}")
    if dense_rankings != dependencies.dense_rankings:
        raise ValueError("stored dense ranking differs from dependency")
    diagnostics = json.loads((root / "diagnostics.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "validation.json").read_text(encoding="utf-8"))
    reconstructed = construct_raw_response_refinement(
        config, _fixture_dependencies=dependencies
    )
    if not np.array_equal(broad, reconstructed.broad_scores):
        raise ValueError("broad score arithmetic changed")
    if not np.array_equal(raw, reconstructed.raw_scores):
        raise ValueError("raw score arithmetic changed")
    if not np.array_equal(hierarchical, reconstructed.hierarchical_scores):
        raise ValueError("hierarchical score arithmetic changed")
    if not np.array_equal(indices, reconstructed.top_record_indices):
        raise ValueError("top-record index arithmetic changed")
    if not np.array_equal(
        similarities, reconstructed.top_record_similarities, equal_nan=True
    ):
        raise ValueError("top-record similarity arithmetic changed")
    if selected_ids != reconstructed.selected_record_ids:
        raise ValueError("selected record identity arithmetic changed")
    if rankings != reconstructed.rankings or not np.array_equal(
        fusion, reconstructed.fusion_scores
    ):
        raise ValueError("ranking or equal-RRF arithmetic changed")
    if diagnostics != reconstructed.diagnostics:
        raise ValueError("inventory diagnostics changed")
    if validation != _validation_payload(reconstructed):
        raise ValueError("raw refinement validation receipt changed")
    return RawRefinementResult(
        config=config,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        broad_scores=broad,
        raw_scores=raw,
        hierarchical_scores=hierarchical,
        top_record_indices=indices,
        top_record_similarities=similarities,
        selected_record_ids=selected_ids,
        rankings=rankings,
        fusion_scores=fusion,
        dense_rankings=dense_rankings,
        diagnostics=diagnostics,
        metadata=metadata,
        _dependencies=dependencies,
        validation=validation,
        manifest=manifest,
        root=root,
    )


def compare_raw_response_refinement_repeats(
    first_dir: str | Path,
    second_dir: str | Path,
    *,
    _fixture_dependencies: Any | None = None,
) -> dict[str, Any]:
    """Require exact selections/rankings and score agreement within 1e-12."""

    first = load_raw_response_refinement_artifact(
        first_dir, _fixture_dependencies=_fixture_dependencies
    )
    second = load_raw_response_refinement_artifact(
        second_dir, _fixture_dependencies=_fixture_dependencies
    )
    if first.root.resolve() == second.root.resolve():
        raise ValueError("refinement repeats must use distinct artifact roots")
    if first.query_ids != second.query_ids or first.candidate_ids != second.candidate_ids:
        raise ValueError("refinement repeat inventories differ")
    if (
        first.metadata.get("dependency_aggregate_sha256")
        != second.metadata.get("dependency_aggregate_sha256")
    ):
        raise ValueError("refinement repeats do not bind the same dependencies")
    selected_identical = first.selected_record_ids == second.selected_record_ids
    rankings_identical = (
        first.rankings == second.rankings
        and first.dense_rankings == second.dense_rankings
    )
    maximum = 0.0
    for left, right in (
        (first.broad_scores, second.broad_scores),
        (first.raw_scores, second.raw_scores),
        (first.hierarchical_scores, second.hierarchical_scores),
        (first.fusion_scores, second.fusion_scores),
    ):
        if left.shape != right.shape:
            raise ValueError("refinement repeat score shapes differ")
        maximum = max(maximum, float(np.max(np.abs(left - right))))
    if not selected_identical:
        raise ValueError("refinement repeat selected record IDs differ")
    if not np.array_equal(first.top_record_indices, second.top_record_indices) or not np.array_equal(
        first.top_record_similarities,
        second.top_record_similarities,
        equal_nan=True,
    ):
        raise ValueError("refinement repeat record evidence differs")
    if not rankings_identical:
        raise ValueError("refinement repeat rankings differ")
    if maximum > 1e-12:
        raise ValueError(f"refinement repeat score difference exceeds 1e-12: {maximum}")
    return {
        "status": "passed",
        "selected_record_ids_identical": True,
        "rankings_identical": True,
        "maximum_absolute_score_difference": maximum,
        "tolerance": 1e-12,
        "first_aggregate_sha256": first.manifest["aggregate_sha256"],
        "second_aggregate_sha256": second.manifest["aggregate_sha256"],
    }


@dataclass(frozen=True, slots=True)
class CandidateRawScores:
    """Top-record evidence and its exact mean score for each query."""

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
    """Average each query's exact top-k candidate-local record similarities."""

    queries = np.asarray(query_embeddings, dtype=np.float32)
    records = np.asarray(record_embeddings, dtype=np.float32)
    identities = tuple(record_ids)
    if queries.ndim != 2 or not queries.shape[0] or not queries.shape[1]:
        raise ValueError("query embeddings must be a non-empty two-dimensional matrix")
    if records.ndim != 2 or not records.shape[0] or not records.shape[1]:
        raise ValueError("record embeddings must be a non-empty two-dimensional matrix")
    if not np.isfinite(queries).all() or not np.isfinite(records).all():
        raise ValueError("query and record embeddings must be finite")
    if records.shape[0] != len(identities):
        raise ValueError("record embedding and record ID counts differ")
    if queries.shape[1] != records.shape[1]:
        raise ValueError("query and record embedding dimensions differ")
    if any(not isinstance(record_id, str) or not record_id for record_id in identities):
        raise ValueError("record IDs must be non-empty strings")
    if len(set(identities)) != len(identities):
        raise ValueError("record IDs must be unique")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    similarities = np.asarray(queries @ records.T, dtype=np.float64)
    if not np.isfinite(similarities).all():
        raise ValueError("record similarities must be finite")
    count = min(top_k, similarities.shape[1])
    orders = [
        sorted(
            range(len(identities)),
            key=lambda index: (-float(row[index]), identities[index]),
        )[:count]
        for row in similarities
    ]
    selected = np.asarray(orders, dtype=np.int32)
    selected_scores = np.take_along_axis(similarities, selected, axis=1)
    means = np.asarray(
        [math.fsum(map(float, row)) / count for row in selected_scores],
        dtype=np.float64,
    )
    selected_ids = tuple(
        tuple(identities[index] for index in row) for row in selected
    )
    return CandidateRawScores(means, selected, selected_ids, selected_scores)


def combine_hierarchical_scores(
    broad_scores: NDArray[np.floating],
    refined_scores: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Combine broad and raw-refined scores with the frozen 0.8/0.2 weights."""

    broad = np.asarray(broad_scores, dtype=np.float64)
    refined = np.asarray(refined_scores, dtype=np.float64)
    if broad.shape != refined.shape or broad.ndim != 2:
        raise ValueError("broad and refined scores must be same-shaped matrices")
    if not np.isfinite(broad).all() or not np.isfinite(refined).all():
        raise ValueError("broad and refined scores must be finite")
    return np.asarray(
        np.float64(0.8) * broad + np.float64(0.2) * refined,
        dtype=np.float64,
    )


def rank_score_matrix(
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    scores: NDArray[np.floating],
) -> dict[str, list[str]]:
    """Rank score rows descending, breaking exact ties by candidate ID."""

    queries = tuple(query_ids)
    candidates = tuple(candidate_ids)
    values = np.asarray(scores, dtype=np.float64)
    if len(set(queries)) != len(queries) or len(set(candidates)) != len(candidates):
        raise ValueError("query IDs and candidate IDs must each be unique")
    if any(not isinstance(value, str) or not value for value in queries + candidates):
        raise ValueError("query IDs and candidate IDs must be non-empty strings")
    if values.shape != (len(queries), len(candidates)):
        raise ValueError("score shape does not match query and candidate IDs")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    names = np.asarray(candidates)
    return {
        query_id: [candidates[int(index)] for index in np.lexsort((names, -values[row]))]
        for row, query_id in enumerate(queries)
    }


__all__ = [
    "CandidateRawScores",
    "RawRefinementConfig",
    "RawRefinementResult",
    "combine_hierarchical_scores",
    "compare_raw_response_refinement_repeats",
    "construct_raw_response_refinement",
    "load_raw_response_refinement_artifact",
    "rank_score_matrix",
    "score_candidate_records",
    "write_raw_response_refinement_artifact",
]
