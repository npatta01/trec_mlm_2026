from __future__ import annotations

import hashlib
import json
import os
import platform
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
import scipy
import sklearn

from trec_mlm.dense_aggregation_artifacts import raw_file_sha256
from trec_mlm.dense_aggregation_hashing import canonical_json, framed_sha256
from trec_mlm.dense_aggregation_runtime import THREAD_ENVIRONMENT


_QUERY_FILE_SHA256 = (
    "862ff85381b8055cd927ada25c56666594ab1ec3c21e182beb20c58f6c637f3c"
)
_QUERY_INPUT_SHA256 = (
    "98faa04f979e428933c3d8e67dac43cf4dded112bd2c16969b3482a1d52886d9"
)
_EMBEDDING_SHA256 = (
    "6d197055eb62ab9809ae321b0f60072e20f17524861e6a989b49cd35c389055b"
)
_SIMILARITY_SHA256 = (
    "80c69a8dca29cd5e9a4de08dc4e44553fabf813428dd4956f7ecbeb2852968ed"
)
_PROTOCOL_SHA256 = (
    "bbe6b50fd3d20efe3213a446f6665cbe0527114f34bf5c1e90236af4f998ad49"
)
_LEGACY_DENSE_SOURCE_SHA256 = (
    "90f8db3abbb2887d8513fb881c631a3843319c29da02e935e1ecdb6c3c85e248"
)
_FEASIBILITY_SCHEMA = "trec-mlm-dense-aggregation-feasibility-config-v1"
_FEASIBILITY_KEYS = {
    "schema",
    "query_path",
    "embedding_path",
    "encoder_metadata_path",
    "protocol_path",
}
_EXPECTED_VERSIONS = {
    "python": "3.13.9",
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit_learn": "1.9.0",
}
_LOWERCASE_HEX = frozenset("0123456789abcdef")
_FEASIBILITY_CONFIG_BYTE_LIMIT = 64 * 1024
_ENCODER_METADATA_BYTE_LIMIT = 256 * 1024
_FROZEN_EMBEDDING_BYTE_LIMIT = 4 * 1024 * 1024


IMPLEMENTATION_FILES: tuple[str, ...] = (
    "trec_mlm/dense_aggregation.py",
    "trec_mlm/dense_aggregation_artifacts.py",
    "trec_mlm/dense_aggregation_cache.py",
    "trec_mlm/dense_aggregation_cli.py",
    "trec_mlm/dense_aggregation_clustering.py",
    "trec_mlm/dense_aggregation_contract.py",
    "trec_mlm/dense_aggregation_entrypoint.py",
    "trec_mlm/dense_aggregation_feasibility.py",
    "trec_mlm/dense_aggregation_hashing.py",
    "trec_mlm/dense_aggregation_runtime.py",
)
DEPENDENCY_SOURCE_FILES: tuple[str, ...] = (
    "trec_mlm/dense_query_knn.py",
    "trec_mlm/metrics.py",
    "trec_mlm/query_knn.py",
    "trec_mlm/runio.py",
)
RUNTIME_PROVENANCE_FILES: tuple[str, ...] = ("pyproject.toml", "uv.lock")


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
    repo_root: Path
    source_path: Path
    source_sha256: str
    query_path: Path
    embedding_path: Path
    encoder_metadata_path: Path
    protocol_path: Path


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


def load_feasibility_config(path: str | Path) -> FeasibilityConfig:
    source = Path(path)
    source_bytes = _read_bounded_regular_file(
        source,
        byte_limit=_FEASIBILITY_CONFIG_BYTE_LIMIT,
        label="feasibility config",
    )
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    try:
        payload_value = json.loads(source_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("feasibility config is not valid JSON") from error
    if type(payload_value) is not dict:
        raise ValueError("feasibility config must be a JSON object")
    payload: dict[object, object] = payload_value
    payload_keys = set(payload)
    if payload_keys != _FEASIBILITY_KEYS:
        raise ValueError(
            "feasibility config keys differ: "
            f"missing={sorted(_FEASIBILITY_KEYS - payload_keys)!r}, "
            f"extra={sorted(payload_keys - _FEASIBILITY_KEYS)!r}"
        )
    if payload["schema"] != _FEASIBILITY_SCHEMA:
        raise ValueError("unsupported feasibility config schema")

    source_path = source.resolve(strict=True)
    repo_root = source_path.parent.parent
    resolved: dict[str, Path] = {}
    for name in sorted(_FEASIBILITY_KEYS - {"schema"}):
        raw_path = payload[name]
        if type(raw_path) is not str or not raw_path:
            raise ValueError(f"feasibility config {name} must be a nonempty string")
        candidate = (repo_root / raw_path).resolve()
        if not candidate.is_relative_to(repo_root):
            raise ValueError(
                f"feasibility config {name} resolves outside repository root"
            )
        resolved[name] = candidate
    return FeasibilityConfig(
        repo_root=repo_root,
        source_path=source_path,
        source_sha256=source_sha256,
        query_path=resolved["query_path"],
        embedding_path=resolved["embedding_path"],
        encoder_metadata_path=resolved["encoder_metadata_path"],
        protocol_path=resolved["protocol_path"],
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
                raise ValueError(
                    f"{path}:{line_number}: expected query_id<TAB>query_text"
                )
            query_id, query_text = fields[0].strip(), fields[1].strip()
            if not query_id or not query_text:
                raise ValueError(
                    f"{path}:{line_number}: blank query ID or text"
                )
            if query_id in queries:
                raise ValueError(
                    f"{path}:{line_number}: duplicate query ID {query_id!r}"
                )
            queries[query_id] = query_text
    if not queries:
        raise ValueError(f"{path}: query file is empty")
    return queries


def _same_exact_value(actual: object, expected: object) -> bool:
    return type(actual) is type(expected) and actual == expected


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
        if not _same_exact_value(metadata.get(key), expected)
    }
    if mismatches:
        raise ValueError(f"frozen MPNet metadata mismatch: {mismatches!r}")


def _load_frozen_embedding_qrel_free(path: Path) -> NDArray[np.float32]:
    source = Path(path)
    try:
        status = source.lstat()
    except OSError as error:
        raise ValueError(f"frozen MPNet array is missing: {source}") from error
    if not stat.S_ISREG(status.st_mode):
        raise ValueError(f"frozen MPNet array is not a regular file: {source}")
    if status.st_size > _FROZEN_EMBEDDING_BYTE_LIMIT:
        raise ValueError(
            f"frozen MPNet array exceeds its {_FROZEN_EMBEDDING_BYTE_LIMIT}-byte limit"
        )
    mapped = np.load(source, mmap_mode="r", allow_pickle=False)
    if mapped.shape != (322, 768) or mapped.dtype != np.dtype("float32"):
        raise ValueError(
            "frozen MPNet array must have shape (322, 768) and float32 dtype"
        )
    offset = getattr(mapped, "offset", None)
    if type(offset) is not int or status.st_size != offset + mapped.nbytes:
        raise ValueError("frozen MPNet array file size does not match its payload")
    embeddings = np.array(mapped, dtype=np.float32, order="C", copy=True)
    del mapped
    return embeddings


def _load_frozen_metadata_qrel_free(path: Path) -> dict[str, object]:
    metadata_bytes = _read_bounded_regular_file(
        Path(path),
        byte_limit=_ENCODER_METADATA_BYTE_LIMIT,
        label="frozen encoder metadata",
    )
    try:
        metadata_value = json.loads(metadata_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen encoder metadata is not valid JSON") from error
    if type(metadata_value) is not dict:
        raise ValueError("frozen encoder metadata must be a JSON object")
    return metadata_value


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

    embeddings = _load_frozen_embedding_qrel_free(config.embedding_path)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or np.any(norms == 0):
        raise ValueError("frozen MPNet rows must be finite and nonzero")
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("frozen MPNet rows are not normalized")
    if _legacy_array_sha256_qrel_free(embeddings) != _EMBEDDING_SHA256:
        raise ValueError("frozen MPNet embedding logical hash mismatch")

    similarities = _cosine_similarity_matrix_qrel_free(embeddings)
    if _legacy_array_sha256_qrel_free(similarities) != _SIMILARITY_SHA256:
        raise ValueError("frozen MPNet cosine logical hash mismatch")

    metadata_value = _load_frozen_metadata_qrel_free(config.encoder_metadata_path)
    metadata = metadata_value.get("mpnet_base_v2")
    if not isinstance(metadata, Mapping):
        raise ValueError("frozen encoder metadata is missing mpnet_base_v2")
    _require_exact_mpnet_metadata(metadata)

    input_hash = _legacy_query_input_sha256(query_ids, query_texts)
    if input_hash != _QUERY_INPUT_SHA256:
        raise ValueError("frozen MPNet query-input logical hash mismatch")
    return FrozenDenseInputs(
        query_ids=query_ids,
        query_texts=query_texts,
        embeddings=embeddings,
        similarities=similarities,
        query_input_sha256=input_hash,
        embedding_sha256=_EMBEDDING_SHA256,
        similarity_sha256=_SIMILARITY_SHA256,
    )


def _validate_relative_source_path(relative: object, *, python: bool) -> str:
    if type(relative) is not str or not relative or "\\" in relative:
        raise ValueError(f"invalid repository-relative source path: {relative!r}")
    path = PurePosixPath(relative)
    if (
        path.is_absolute()
        or path.as_posix() != relative
        or "." in path.parts
        or ".." in path.parts
        or (python and path.suffix != ".py")
    ):
        raise ValueError(f"invalid repository-relative source path: {relative!r}")
    return relative


def _validate_source_constants() -> None:
    groups = (
        ("implementation", IMPLEMENTATION_FILES, True),
        ("dependency", DEPENDENCY_SOURCE_FILES, True),
        ("runtime", RUNTIME_PROVENANCE_FILES, False),
    )
    for category, paths, python in groups:
        if type(paths) is not tuple or paths != tuple(sorted(paths)):
            raise ValueError(f"{category} source paths must be a sorted tuple")
        if len(paths) != len(set(paths)):
            raise ValueError(f"{category} source paths must be unique")
        for relative in paths:
            _validate_relative_source_path(relative, python=python)
    for index, (left_name, left, _left_python) in enumerate(groups):
        for right_name, right, _right_python in groups[index + 1 :]:
            overlap = sorted(set(left) & set(right))
            if overlap:
                raise ValueError(
                    f"{left_name} and {right_name} source categories overlap: "
                    f"{overlap!r}"
                )


def _resolve_repo_root(repo_root: Path) -> Path:
    root = repo_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"repository root is not a directory: {root}")
    return root


def _require_regular_source(root: Path, relative: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        mode = candidate.lstat().st_mode
    except OSError as error:
        raise ValueError(f"source file is missing: {relative}") from error
    if not stat.S_ISREG(mode):
        raise ValueError(f"source path is not a regular file: {relative}")
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"source path resolves outside repository root: {relative}")
    return candidate


def _source_inventory_rows(
    root: Path, relative_paths: Sequence[str]
) -> list[list[str]]:
    return [
        [relative, raw_file_sha256(_require_regular_source(root, relative))]
        for relative in relative_paths
    ]


def _require_exact_runtime(
    versions: Mapping[str, object], threads: Mapping[str, object]
) -> None:
    if (
        type(versions) is not dict
        or set(versions) != set(_EXPECTED_VERSIONS)
        or any(
            not _same_exact_value(versions.get(key), expected)
            for key, expected in _EXPECTED_VERSIONS.items()
        )
    ):
        raise ValueError(
            f"runtime version mismatch: {dict(versions)!r} != "
            f"{_EXPECTED_VERSIONS!r}"
        )
    expected_thread_keys = {
        *THREAD_ENVIRONMENT,
        "threadpool_limit",
        "process_count",
    }
    valid_threads = type(threads) is dict and set(threads) == expected_thread_keys
    if valid_threads:
        valid_threads = all(
            type(threads.get(name)) is str and threads.get(name) == "1"
            for name in THREAD_ENVIRONMENT
        ) and all(
            type(threads.get(name)) is int and threads.get(name) == 1
            for name in ("threadpool_limit", "process_count")
        )
    if not valid_threads:
        raise ValueError(f"thread contract mismatch: {dict(threads)!r}")


def build_clustering_contract(
    inputs: FrozenDenseInputs,
    *,
    repo_root: Path,
    protocol_path: Path,
) -> tuple[dict[str, object], str]:
    _validate_source_constants()
    root = _resolve_repo_root(repo_root)
    expected_protocol = (
        root / "protocols/trec_2026_dense_aggregation.md"
    ).resolve(strict=True)
    resolved_protocol = protocol_path.resolve(strict=True)
    if resolved_protocol != expected_protocol:
        raise ValueError(
            f"protocol path mismatch: {resolved_protocol} != {expected_protocol}"
        )
    protocol_sha256 = raw_file_sha256(resolved_protocol)
    if protocol_sha256 != _PROTOCOL_SHA256:
        raise ValueError("locked dense aggregation protocol hash mismatch")

    all_source_paths = tuple(
        sorted((*IMPLEMENTATION_FILES, *DEPENDENCY_SOURCE_FILES))
    )
    implementation_rows = _source_inventory_rows(root, all_source_paths)
    legacy_hash = dict(implementation_rows)["trec_mlm/dense_query_knn.py"]
    if legacy_hash != _LEGACY_DENSE_SOURCE_SHA256:
        raise ValueError("locked legacy dense scorer source hash mismatch")
    for relative in RUNTIME_PROVENANCE_FILES:
        _require_regular_source(root, relative)

    versions: dict[str, object] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }
    threads: dict[str, object] = {
        **{name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT},
        "threadpool_limit": 1,
        "process_count": 1,
    }
    payload: dict[str, object] = {
        "schema": "trec-mlm-dense-aggregation-clustering-v1",
        "protocol_file_sha256": protocol_sha256,
        "implementation_files": implementation_rows,
        "embedding_sha256": inputs.embedding_sha256,
        "similarity_sha256": inputs.similarity_sha256,
        "versions": versions,
        "threads": threads,
        "kmeans": {
            "init": "k-means++",
            "random_state": 2026,
            "n_init": 10,
            "algorithm": "lloyd",
            "max_iter": 300,
            "tol": 1e-4,
        },
        "center": {
            "dtype": "<f4",
            "layout": "C",
            "canonical_order": "minimum_member_query_id_lexical",
        },
        "target_distance": {
            "dtype": "<f8",
            "reduction": (
                "numpy.sum(delta[z] * delta[z], dtype=numpy.float64)"
            ),
            "tie": "minimum_canonical_cluster_id",
        },
        "inertia": {
            "dtype": "<f8",
            "cluster_reduction": (
                "numpy.sum(member_delta * member_delta, dtype=numpy.float64)"
            ),
            "total_reduction": "math.fsum(cluster_inertia)",
            "rtol": 1e-5,
            "atol": 1e-8,
        },
        "convergence": {
            "n_iter_min": 1,
            "n_iter_max": 299,
            "warning_action": "reject",
        },
    }
    _require_exact_runtime(versions, threads)
    return payload, framed_sha256(
        "clustering-contract",
        canonical_json(payload),
    )


def _validated_contract_rows(
    contract: Mapping[str, object],
) -> dict[str, list[str]]:
    _validate_source_constants()
    rows_value = contract.get("implementation_files")
    if type(rows_value) is not list:
        raise ValueError("contract source inventory must be a list")
    expected_paths = tuple(
        sorted((*IMPLEMENTATION_FILES, *DEPENDENCY_SOURCE_FILES))
    )
    rows_by_path: dict[str, list[str]] = {}
    observed_paths: list[str] = []
    for row in rows_value:
        if (
            type(row) is not list
            or len(row) != 2
            or type(row[0]) is not str
            or type(row[1]) is not str
        ):
            raise ValueError("contract source inventory row is invalid")
        relative, digest = row
        _validate_relative_source_path(relative, python=True)
        if len(digest) != 64 or any(
            character not in _LOWERCASE_HEX for character in digest
        ):
            raise ValueError("contract source inventory digest is invalid")
        observed_paths.append(relative)
        if relative in rows_by_path:
            raise ValueError("contract source inventory paths must appear exactly once")
        rows_by_path[relative] = [relative, digest]
    if tuple(observed_paths) != expected_paths:
        raise ValueError("contract source inventory paths must appear exactly once")
    return rows_by_path


def contract_source_inventory_sha256(
    contract: Mapping[str, object],
    *,
    category: Literal["implementation", "dependency"],
) -> str:
    if category == "implementation":
        selected_paths = IMPLEMENTATION_FILES
    elif category == "dependency":
        selected_paths = DEPENDENCY_SOURCE_FILES
    else:
        raise ValueError(f"unsupported source inventory category: {category!r}")
    rows_by_path = _validated_contract_rows(contract)
    selected_rows = [rows_by_path[relative] for relative in selected_paths]
    return framed_sha256(
        f"{category}-source-inventory",
        canonical_json(selected_rows),
    )


def runtime_source_inventory_sha256(repo_root: Path) -> str:
    _validate_source_constants()
    root = _resolve_repo_root(repo_root)
    rows = _source_inventory_rows(root, RUNTIME_PROVENANCE_FILES)
    return framed_sha256(
        "runtime-source-inventory",
        canonical_json(rows),
    )
