"""Isolated legacy dense-control export and artifact validation."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import struct
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm import dense_query_knn
from trec_mlm.dense_query_knn import (
    array_sha256,
    cosine_similarity_matrix,
    rank_target_from_similarity,
)
from trec_mlm.metrics import evaluate_rankings
from trec_mlm.query_knn import load_development_data


_LOCKED_QUERY_FILE_SHA256 = (
    "862ff85381b8055cd927ada25c56666594ab1ec3c21e182beb20c58f6c637f3c"
)
_LOCKED_QUERY_INPUT_SHA256 = (
    "98faa04f979e428933c3d8e67dac43cf4dded112bd2c16969b3482a1d52886d9"
)
_LOCKED_QREL_FILE_SHA256 = (
    "3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35"
)
_LOCKED_EMBEDDING_SHA256 = (
    "6d197055eb62ab9809ae321b0f60072e20f17524861e6a989b49cd35c389055b"
)
_LOCKED_SIMILARITY_SHA256 = (
    "80c69a8dca29cd5e9a4de08dc4e44553fabf813428dd4956f7ecbeb2852968ed"
)
_LOCKED_LEGACY_SOURCE_SHA256 = (
    "90f8db3abbb2887d8513fb881c631a3843319c29da02e935e1ecdb6c3c85e248"
)
_EXPECTED_NDCG = 0.36045807293576454
_EXPECTED_MRR = 0.4039893654911617
_QUERY_COUNT = 322
_CANDIDATE_COUNT = 150
_ROW_COUNT = _QUERY_COUNT * _CANDIDATE_COUNT
_EMBEDDING_BYTE_LIMIT = 4 * 1024 * 1024
_HASH_PREFIX = b"trec-mlm-dense-aggregation-hash-v1\0"
_SOURCE_RELATIVE_PATHS = (
    "scripts/export_fixed_mpnet_k50_control.py",
    "trec_mlm/dense_query_knn.py",
    "trec_mlm/legacy_dense_control.py",
    "trec_mlm/metrics.py",
    "trec_mlm/query_knn.py",
    "trec_mlm/runio.py",
)
_LOCK_FILENAME = "fixed_mpnet_k50_control.lock.json"
_TSV_FILENAME = "fixed_mpnet_k50_control.tsv"
_TREC_FILENAME = "fixed_mpnet_k50_control.trec"
_RUN_ID = "fixed_mpnet_k50_control"
_LOCK_SCHEMA = "trec-mlm-dense-aggregation-legacy-control-lock-v1"
_REPEAT_SCHEMA = "trec-mlm-dense-aggregation-legacy-control-repeat-v1"
_MANIFEST_SCHEMA = "trec-mlm-dense-aggregation-manifest-v1"
_ROOT_FILES = (
    _LOCK_FILENAME,
    _TREC_FILENAME,
    _TSV_FILENAME,
    "repeat_a.json",
    "repeat_b.json",
)
_LOCK_KEYS = {
    "schema",
    "feasibility_manifest_sha256",
    "source_sha256",
    "query_file_sha256",
    "qrel_file_sha256",
    "query_input_sha256",
    "candidate_inventory_sha256",
    "embedding_sha256",
    "similarity_sha256",
    "neighbor_count",
    "tsv_filename",
    "tsv_sha256",
    "trec_filename",
    "trec_sha256",
    "run_id",
    "row_count",
    "query_count",
    "candidate_count",
    "mean_ndcg_at_10",
    "mean_reciprocal_rank",
    "repeat_a_receipt_sha256",
    "repeat_b_receipt_sha256",
}
_REPEAT_KEYS = {
    "schema",
    "repeat",
    "return_code",
    "feasibility_manifest_sha256",
    "source_sha256",
    "query_file_sha256",
    "qrel_file_sha256",
    "query_input_sha256",
    "candidate_inventory_sha256",
    "embedding_sha256",
    "similarity_sha256",
    "neighbor_count",
    "tsv_sha256",
    "row_count",
    "query_count",
    "candidate_count",
    "mean_ndcg_at_10",
    "mean_reciprocal_rank",
}
_DIGEST_FIELDS = {
    "feasibility_manifest_sha256",
    "query_file_sha256",
    "qrel_file_sha256",
    "query_input_sha256",
    "candidate_inventory_sha256",
    "embedding_sha256",
    "similarity_sha256",
    "tsv_sha256",
}
_LOWERCASE_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True, slots=True)
class LegacyControlValidation:
    """Validated scientific summary of one legacy-control export."""

    tsv_sha256: str
    row_count: int
    mean_ndcg_at_10: float
    mean_reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class LegacyControlLockValidation:
    """Complete validated downstream bindings for one legacy-control lock."""

    manifest_sha256: str
    lock_sha256: str
    feasibility_manifest_sha256: str
    source_sha256: Mapping[str, str]
    query_file_sha256: str
    qrel_file_sha256: str
    query_input_sha256: str
    candidate_inventory_sha256: str
    embedding_sha256: str
    similarity_sha256: str
    tsv_sha256: str
    trec_sha256: str
    run_id: str
    neighbor_count: int
    query_count: int
    candidate_count: int
    row_count: int
    mean_ndcg_at_10: float
    mean_reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class _LegacyControlWorkerResult:
    validation: LegacyControlValidation
    source_sha256: dict[str, str]
    query_file_sha256: str
    qrel_file_sha256: str
    query_input_sha256: str
    candidate_inventory_sha256: str
    embedding_sha256: str
    similarity_sha256: str
    embedding_file_sha256: str
    neighbor_count: int
    query_count: int
    candidate_count: int


def _raw_file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _framed_sha256(tag: str, *payloads: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(_HASH_PREFIX)
    for payload in (tag.encode("ascii"), *payloads):
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _query_input_sha256(
    query_ids: tuple[str, ...], query_texts: tuple[str, ...]
) -> str:
    digest = hashlib.sha256()
    for query_id, query_text in zip(query_ids, query_texts, strict=True):
        digest.update(query_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(query_text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _candidate_inventory_sha256(model_ids: tuple[str, ...]) -> str:
    return _framed_sha256("candidate-inventory", _canonical_json(list(model_ids)))


def _repo_root() -> Path:
    return Path(__file__).resolve(strict=True).parents[1]


def _source_sha256() -> dict[str, str]:
    root = _repo_root()
    return {
        relative: _raw_file_sha256(root / relative)
        for relative in _SOURCE_RELATIVE_PATHS
    }


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _LOWERCASE_HEX for character in value)
    )


def _read_bounded_regular_file(path: Path, *, byte_limit: int, label: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} is not a regular file")
    if before.st_size > byte_limit:
        raise ValueError(f"{label} exceeds its byte limit")
    flags = os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} changed while it was read") from error
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
            or opened.st_size != before.st_size
            or opened.st_mtime_ns != before.st_mtime_ns
            or opened.st_ctime_ns != before.st_ctime_ns
        ):
            raise ValueError(f"{label} changed while it was read")
        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        payload = b"".join(chunks)
        opened_after = os.fstat(descriptor)
    except OSError as error:
        raise ValueError(f"{label} changed while it was read") from error
    finally:
        os.close(descriptor)
    try:
        after = path.lstat()
    except OSError as error:
        raise ValueError(f"{label} changed while it was read") from error
    if (
        len(payload) != before.st_size
        or not stat.S_ISREG(after.st_mode)
        or (opened_after.st_dev, opened_after.st_ino) != (before.st_dev, before.st_ino)
        or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
        or opened_after.st_size != before.st_size
        or after.st_size != before.st_size
        or opened_after.st_mtime_ns != before.st_mtime_ns
        or after.st_mtime_ns != before.st_mtime_ns
        or opened_after.st_ctime_ns != before.st_ctime_ns
        or after.st_ctime_ns != before.st_ctime_ns
    ):
        raise ValueError(f"{label} changed while it was read")
    return payload


def _bounded_regular_file_sha256(path: Path, *, byte_limit: int, label: str) -> str:
    payload = _read_bounded_regular_file(
        path,
        byte_limit=byte_limit,
        label=label,
    )
    return hashlib.sha256(payload).hexdigest()


def _read_canonical_json(path: Path, *, byte_limit: int, label: str) -> object:
    payload = _read_bounded_regular_file(path, byte_limit=byte_limit, label=label)
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    try:
        expected = _canonical_json(value) + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not canonical JSON") from error
    if payload != expected:
        raise ValueError(f"{label} is not canonical JSON")
    return value


def _require_exact_root_layout(root: Path) -> None:
    try:
        status = root.lstat()
    except OSError as error:
        raise ValueError(
            "legacy control root has an unexpected physical layout"
        ) from error
    if not stat.S_ISDIR(status.st_mode):
        raise ValueError("legacy control root has an unexpected physical layout")
    expected = {"manifest.json", *_ROOT_FILES}
    observed: set[str] = set()
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                if entry.name not in expected or entry.name in observed:
                    raise ValueError(
                        "legacy control root has an unexpected physical layout"
                    )
                if not entry.is_file(follow_symlinks=False):
                    raise ValueError(
                        "legacy control root has an unexpected physical layout"
                    )
                observed.add(entry.name)
    except OSError as error:
        raise ValueError(
            "legacy control root has an unexpected physical layout"
        ) from error
    if observed != expected:
        raise ValueError("legacy control root has an unexpected physical layout")


def _validate_local_manifest(root: Path) -> dict[str, object]:
    _require_exact_root_layout(root)
    value = _read_canonical_json(
        root / "manifest.json", byte_limit=1024 * 1024, label="legacy manifest"
    )
    if type(value) is not dict or set(value) != {
        "schema",
        "files",
        "artifact_inventory_sha256",
    }:
        raise ValueError("legacy manifest has an invalid contract")
    if value["schema"] != _MANIFEST_SCHEMA:
        raise ValueError("legacy manifest has an unsupported schema")
    rows = value["files"]
    if (
        type(rows) is not list
        or [row[0] for row in rows if type(row) is list and len(row) == 3]
        != list(_ROOT_FILES)
        or len(rows) != len(_ROOT_FILES)
    ):
        raise ValueError("legacy manifest inventory is invalid")
    observed_rows: list[list[str | int]] = []
    for index, (name, size, digest) in enumerate(rows):
        if (
            type(name) is not str
            or name != _ROOT_FILES[index]
            or type(size) is not int
            or isinstance(size, bool)
            or size < 0
            or not _is_sha256(digest)
        ):
            raise ValueError("legacy manifest inventory is invalid")
        path = root / name
        payload = _read_bounded_regular_file(
            path, byte_limit=32 * 1024 * 1024, label=f"legacy artifact {name}"
        )
        observed_rows.append([name, len(payload), hashlib.sha256(payload).hexdigest()])
    if rows != observed_rows:
        raise ValueError("legacy manifest inventory does not match files")
    expected_inventory = _framed_sha256(
        "artifact-inventory", _canonical_json(observed_rows)
    )
    if value["artifact_inventory_sha256"] != expected_inventory:
        raise ValueError("legacy manifest digest does not match inventory")
    return value


def _validate_source_map(value: object) -> dict[str, str]:
    if type(value) is not dict or tuple(value) != _SOURCE_RELATIVE_PATHS:
        raise ValueError("legacy control source inventory is invalid")
    if any(not _is_sha256(digest) for digest in value.values()):
        raise ValueError("legacy control source inventory is invalid")
    return dict(value)


def _validate_repeat(value: object, *, repeat: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != _REPEAT_KEYS:
        raise ValueError("legacy repeat receipt has an invalid contract")
    if value["schema"] != _REPEAT_SCHEMA or value["repeat"] != repeat:
        raise ValueError("legacy repeat receipt has an invalid schema or label")
    if value["return_code"] != 0 or type(value["return_code"]) is not int:
        raise ValueError("legacy repeat return code is invalid")
    for field in _DIGEST_FIELDS:
        if not _is_sha256(value[field]):
            raise ValueError(f"legacy repeat {field} is invalid")
    _validate_source_map(value["source_sha256"])
    for field, expected in (
        ("neighbor_count", 50),
        ("row_count", _ROW_COUNT),
        ("query_count", _QUERY_COUNT),
        ("candidate_count", _CANDIDATE_COUNT),
    ):
        if type(value[field]) is not int or value[field] != expected:
            raise ValueError(f"legacy repeat {field.replace('_', ' ')} mismatch")
    for field, expected in (
        ("mean_ndcg_at_10", _EXPECTED_NDCG),
        ("mean_reciprocal_rank", _EXPECTED_MRR),
    ):
        metric = value[field]
        if type(metric) is not float or not math.isfinite(metric) or metric != expected:
            raise ValueError(f"legacy repeat {field} mismatch")
    return dict(value)


def _validate_lock_contract(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != _LOCK_KEYS:
        raise ValueError("legacy control lock has an invalid contract")
    if value["schema"] != _LOCK_SCHEMA:
        raise ValueError("legacy control lock has an unsupported schema")
    for field in _DIGEST_FIELDS | {
        "trec_sha256",
        "repeat_a_receipt_sha256",
        "repeat_b_receipt_sha256",
    }:
        if not _is_sha256(value[field]):
            raise ValueError(f"legacy control {field} is invalid")
    source = _validate_source_map(value["source_sha256"])
    if source != _source_sha256():
        raise ValueError("legacy control source inventory differs from current bytes")
    locked_inputs = {
        "query_file_sha256": _LOCKED_QUERY_FILE_SHA256,
        "qrel_file_sha256": _LOCKED_QREL_FILE_SHA256,
        "query_input_sha256": _LOCKED_QUERY_INPUT_SHA256,
        "embedding_sha256": _LOCKED_EMBEDDING_SHA256,
        "similarity_sha256": _LOCKED_SIMILARITY_SHA256,
    }
    if any(value[field] != expected for field, expected in locked_inputs.items()):
        raise ValueError("legacy control locked input hash mismatch")
    for field, expected in (
        ("neighbor_count", 50),
        ("row_count", _ROW_COUNT),
        ("query_count", _QUERY_COUNT),
        ("candidate_count", _CANDIDATE_COUNT),
    ):
        if type(value[field]) is not int or value[field] != expected:
            raise ValueError(f"legacy control {field.replace('_', ' ')} mismatch")
    if value["tsv_filename"] != _TSV_FILENAME:
        raise ValueError("legacy control TSV filename mismatch")
    if value["trec_filename"] != _TREC_FILENAME:
        raise ValueError("legacy control TREC filename mismatch")
    if value["run_id"] != _RUN_ID:
        raise ValueError("legacy control run ID mismatch")
    for field, expected in (
        ("mean_ndcg_at_10", _EXPECTED_NDCG),
        ("mean_reciprocal_rank", _EXPECTED_MRR),
    ):
        metric = value[field]
        if type(metric) is not float or not math.isfinite(metric) or metric != expected:
            raise ValueError(f"legacy control {field} mismatch")
    return dict(value)


def _parse_legacy_tsv(
    source: Path | bytes,
) -> tuple[dict[str, list[str]], tuple[str, ...], int]:
    expected_header = (
        b"query_id\tcandidate_id\traw_score_hex\tfold_prior_hex\tfinal_rank\n"
    )
    rankings: dict[str, list[str]] = {}
    canonical_candidates: tuple[str, ...] | None = None
    current_query: str | None = None
    current_rows: list[tuple[str, float, float, int]] = []
    previous_candidate: str | None = None
    row_count = 0

    def finish_query() -> None:
        nonlocal canonical_candidates, current_rows
        if current_query is None:
            return
        if len(current_rows) != _CANDIDATE_COUNT:
            raise ValueError("legacy TSV candidate count mismatch")
        candidates = tuple(row[0] for row in current_rows)
        if canonical_candidates is None:
            canonical_candidates = candidates
        elif candidates != canonical_candidates:
            raise ValueError("legacy TSV candidate inventory mismatch")
        ranks = [row[3] for row in current_rows]
        if sorted(ranks) != list(range(1, _CANDIDATE_COUNT + 1)):
            raise ValueError("legacy TSV ranks are not a full permutation")
        derived = sorted(current_rows, key=lambda row: (-row[1], -row[2], row[0]))
        expected_rank = {row[0]: rank for rank, row in enumerate(derived, start=1)}
        if any(
            rank != expected_rank[candidate] for candidate, _, _, rank in current_rows
        ):
            raise ValueError("legacy TSV rank differs from score/prior/candidate order")
        rankings[current_query] = [
            row[0] for row in sorted(current_rows, key=lambda row: row[3])
        ]
        current_rows = []

    handle = io.BytesIO(source) if type(source) is bytes else source.open("rb")
    with handle:
        if handle.readline(1025) != expected_header:
            raise ValueError("legacy TSV header is not canonical")
        for line_number in range(2, _ROW_COUNT + 2):
            line = handle.readline(1025)
            if not line:
                raise ValueError("legacy TSV ended before its exact row count")
            if len(line) > 1024 or not line.endswith(b"\n") or b"\r" in line:
                raise ValueError("legacy TSV row is oversized or noncanonical")
            try:
                text = line[:-1].decode("utf-8")
            except UnicodeError as error:
                raise ValueError("legacy TSV row is not UTF-8") from error
            fields = text.split("\t")
            if len(fields) != 5:
                raise ValueError("legacy TSV row does not have five columns")
            query_id, candidate_id, score_hex, prior_hex, rank_text = fields
            if (
                not query_id
                or not candidate_id
                or any(character.isspace() for character in query_id + candidate_id)
            ):
                raise ValueError("legacy TSV identifier is invalid")
            try:
                score = float.fromhex(score_hex)
                prior = float.fromhex(prior_hex)
            except ValueError as error:
                raise ValueError("legacy TSV score is not hexadecimal") from error
            if (
                not math.isfinite(score)
                or not math.isfinite(prior)
                or score.hex() != score_hex
                or prior.hex() != prior_hex
            ):
                raise ValueError("legacy TSV score is non-finite or noncanonical")
            try:
                rank = int(rank_text)
            except ValueError as error:
                raise ValueError("legacy TSV rank is not decimal") from error
            if str(rank) != rank_text or not 1 <= rank <= _CANDIDATE_COUNT:
                raise ValueError("legacy TSV rank is not canonical")

            if current_query is None:
                current_query = query_id
            elif query_id != current_query:
                if query_id <= current_query:
                    raise ValueError("legacy TSV query order is not canonical")
                finish_query()
                current_query = query_id
                previous_candidate = None
            if previous_candidate is not None and candidate_id <= previous_candidate:
                raise ValueError("legacy TSV candidate order is not canonical")
            previous_candidate = candidate_id
            current_rows.append((candidate_id, score, prior, rank))
            row_count += 1
        if handle.read(1) != b"":
            raise ValueError("legacy TSV contains data after its exact row count")
    finish_query()
    if len(rankings) != _QUERY_COUNT or row_count != _ROW_COUNT:
        raise ValueError("legacy TSV query or row count mismatch")
    assert canonical_candidates is not None
    return rankings, canonical_candidates, row_count


def _trec_bytes_from_rankings(rankings: dict[str, list[str]]) -> bytes:
    chunks: list[bytes] = []
    byte_count = 0
    for query_id in sorted(rankings):
        model_ids = rankings[query_id]
        if len(model_ids) != _CANDIDATE_COUNT or len(set(model_ids)) != len(model_ids):
            raise ValueError("legacy ranking candidate inventory is invalid")
        for rank, model_id in enumerate(model_ids, start=1):
            line = (
                f"{query_id} Q0 {model_id} {rank} "
                f"{(_CANDIDATE_COUNT - rank + 1):.1f} {_RUN_ID}\n"
            ).encode("utf-8")
            byte_count += len(line)
            if byte_count > 16 * 1024 * 1024:
                raise ValueError("reconstructed legacy TREC run exceeds byte limit")
            chunks.append(line)
    return b"".join(chunks)


def _require_unchanged_arrays(
    *,
    grades: NDArray[np.float64],
    grades_bytes: bytes,
    similarities: NDArray[np.float32],
    similarity_bytes: bytes,
) -> None:
    if grades.tobytes(order="C") != grades_bytes:
        raise RuntimeError("legacy scorer mutated the canonical grade matrix")
    if similarities.tobytes(order="C") != similarity_bytes:
        raise RuntimeError("legacy scorer mutated the cosine similarity matrix")


def _validated_ranking(
    result: Any, *, model_ids: tuple[str, ...]
) -> tuple[tuple[int, ...], NDArray[np.float64], NDArray[np.float64]]:
    order = result.model_order
    if (
        type(order) is not tuple
        or len(order) != _CANDIDATE_COUNT
        or any(type(index) is not int for index in order)
        or set(order) != set(range(_CANDIDATE_COUNT))
    ):
        raise RuntimeError("legacy scorer returned an invalid model permutation")
    scores = np.asarray(result.scores, dtype=np.float64)
    priors = np.asarray(result.global_prior, dtype=np.float64)
    if scores.shape != (_CANDIDATE_COUNT,) or priors.shape != (_CANDIDATE_COUNT,):
        raise RuntimeError("legacy scorer returned an invalid score shape")
    if not np.isfinite(scores).all() or not np.isfinite(priors).all():
        raise RuntimeError("legacy scorer returned a non-finite score")
    expected_order = tuple(
        int(index) for index in np.lexsort((np.asarray(model_ids), -priors, -scores))
    )
    if order != expected_order:
        raise RuntimeError(
            "legacy scorer order differs from score/prior/model ordering"
        )
    return order, scores, priors


def _require_exact_metrics(metrics: Any) -> None:
    if metrics.cutoff != 10 or type(metrics.cutoff) is not int:
        raise RuntimeError("legacy fixed-MPNet metric cutoff mismatch")
    ndcg = metrics.macro_ndcg_at_k
    mrr = metrics.mean_reciprocal_rank
    if type(ndcg) is not float or not math.isfinite(ndcg) or ndcg != _EXPECTED_NDCG:
        raise RuntimeError("legacy fixed-MPNet nDCG regression mismatch")
    if type(mrr) is not float or not math.isfinite(mrr) or mrr != _EXPECTED_MRR:
        raise RuntimeError("legacy fixed-MPNet MRR regression mismatch")


def _preflight_embedding_file(path: Path) -> os.stat_result:
    try:
        status = path.lstat()
    except OSError as error:
        raise RuntimeError("legacy control embedding file is missing") from error
    if not stat.S_ISREG(status.st_mode):
        raise RuntimeError("legacy control embedding file is not regular")
    if status.st_size > _EMBEDDING_BYTE_LIMIT:
        raise RuntimeError("legacy control embedding file exceeds byte limit")
    return status


def _load_locked_embedding(path: Path) -> NDArray[np.float32]:
    status = _preflight_embedding_file(path)
    try:
        mapped = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise RuntimeError("legacy control embedding NPY is invalid") from error
    if mapped.shape != (_QUERY_COUNT, 768):
        raise RuntimeError("legacy control embedding array shape mismatch")
    if mapped.dtype != np.dtype("float32"):
        raise RuntimeError("legacy control embedding array dtype mismatch")
    offset = getattr(mapped, "offset", None)
    if type(offset) is not int or status.st_size != offset + mapped.nbytes:
        raise RuntimeError("legacy control embedding file size mismatch")
    if not np.isfinite(mapped).all():
        raise RuntimeError("legacy control embedding array contains non-finite values")
    embeddings = np.array(mapped, dtype=np.float32, order="C", copy=True)
    del mapped
    return embeddings


def _export_fixed_mpnet_k50_control_attested(
    *,
    query_path: Path,
    qrel_path: Path,
    embedding_path: Path,
    output_tsv: Path,
) -> _LegacyControlWorkerResult:
    query_path = Path(query_path)
    qrel_path = Path(qrel_path)
    embedding_path = Path(embedding_path)
    output_tsv = Path(output_tsv)

    source_before = _source_sha256()
    if (
        source_before["trec_mlm/dense_query_knn.py"] != _LOCKED_LEGACY_SOURCE_SHA256
        or _raw_file_sha256(Path(dense_query_knn.__file__))
        != _LOCKED_LEGACY_SOURCE_SHA256
    ):
        raise RuntimeError("legacy dense scorer source hash mismatch")

    query_file_sha256 = _raw_file_sha256(query_path)
    if query_file_sha256 != _LOCKED_QUERY_FILE_SHA256:
        raise RuntimeError("legacy control query file hash mismatch")
    qrel_file_sha256 = _raw_file_sha256(qrel_path)
    if qrel_file_sha256 != _LOCKED_QREL_FILE_SHA256:
        raise RuntimeError("legacy control qrel file hash mismatch")
    _preflight_embedding_file(embedding_path)
    embedding_file_sha256 = _raw_file_sha256(embedding_path)

    data = load_development_data(query_path, qrel_path)
    query_ids = tuple(sorted(data.queries))
    model_ids = tuple(sorted(data.models))
    if len(query_ids) != _QUERY_COUNT:
        raise RuntimeError("legacy control query count mismatch")
    if len(model_ids) != _CANDIDATE_COUNT:
        raise RuntimeError("legacy control candidate count mismatch")
    query_input_sha256 = _query_input_sha256(
        query_ids, tuple(data.queries[query_id] for query_id in query_ids)
    )
    if query_input_sha256 != _LOCKED_QUERY_INPUT_SHA256:
        raise RuntimeError("legacy control query-input hash mismatch")
    candidate_inventory_sha256 = _candidate_inventory_sha256(model_ids)
    grades = np.asarray(
        [
            [data.qrels[query_id][model_id] for model_id in model_ids]
            for query_id in query_ids
        ],
        dtype=np.float64,
    )
    if (
        grades.shape != (_QUERY_COUNT, _CANDIDATE_COUNT)
        or not np.isfinite(grades).all()
        or not np.isin(grades, (0.0, 1.0, 2.0)).all()
    ):
        raise RuntimeError("legacy control grade matrix is invalid")

    embeddings = _load_locked_embedding(embedding_path)
    embedding_sha256 = array_sha256(embeddings)
    if embedding_sha256 != _LOCKED_EMBEDDING_SHA256:
        raise RuntimeError("legacy control embedding hash mismatch")
    similarities = cosine_similarity_matrix(embeddings)
    similarity_sha256 = array_sha256(similarities)
    if similarity_sha256 != _LOCKED_SIMILARITY_SHA256:
        raise RuntimeError("legacy control similarity hash mismatch")

    grades_bytes = grades.tobytes(order="C")
    similarity_bytes = similarities.tobytes(order="C")
    rankings: dict[str, list[str]] = {}
    row_count = 0
    with output_tsv.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "query_id",
                "candidate_id",
                "raw_score_hex",
                "fold_prior_hex",
                "final_rank",
            )
        )
        for target_index, query_id in enumerate(query_ids):
            result = rank_target_from_similarity(
                target_index=target_index,
                additionally_excluded_indices=(),
                similarities=similarities,
                grades=grades,
                query_ids=query_ids,
                model_ids=model_ids,
                k=50,
            )
            _require_unchanged_arrays(
                grades=grades,
                grades_bytes=grades_bytes,
                similarities=similarities,
                similarity_bytes=similarity_bytes,
            )
            order, scores, priors = _validated_ranking(result, model_ids=model_ids)
            rankings[query_id] = [model_ids[index] for index in order]
            rank_by_index = {
                model_index: rank for rank, model_index in enumerate(order, start=1)
            }
            for model_index, model_id in enumerate(model_ids):
                writer.writerow(
                    (
                        query_id,
                        model_id,
                        float(scores[model_index]).hex(),
                        float(priors[model_index]).hex(),
                        rank_by_index[model_index],
                    )
                )
                row_count += 1
        handle.flush()
        os.fsync(handle.fileno())
    if row_count != _ROW_COUNT or len(rankings) != _QUERY_COUNT:
        raise RuntimeError("legacy control TSV row count mismatch")

    metrics = evaluate_rankings(rankings, data.qrels, k=10)
    _require_exact_metrics(metrics)
    validation = LegacyControlValidation(
        tsv_sha256=_raw_file_sha256(output_tsv),
        row_count=row_count,
        mean_ndcg_at_10=metrics.macro_ndcg_at_k,
        mean_reciprocal_rank=metrics.mean_reciprocal_rank,
    )

    source_after = _source_sha256()
    if source_after != source_before:
        raise RuntimeError("legacy control source files drifted during export")
    if _raw_file_sha256(query_path) != query_file_sha256:
        raise RuntimeError("legacy control query file drifted during export")
    if _raw_file_sha256(qrel_path) != qrel_file_sha256:
        raise RuntimeError("legacy control qrel file drifted during export")
    if _raw_file_sha256(embedding_path) != embedding_file_sha256:
        raise RuntimeError("legacy control embedding file drifted during export")
    return _LegacyControlWorkerResult(
        validation=validation,
        source_sha256=source_before,
        query_file_sha256=query_file_sha256,
        qrel_file_sha256=qrel_file_sha256,
        query_input_sha256=query_input_sha256,
        candidate_inventory_sha256=candidate_inventory_sha256,
        embedding_sha256=embedding_sha256,
        similarity_sha256=similarity_sha256,
        embedding_file_sha256=embedding_file_sha256,
        neighbor_count=50,
        query_count=len(query_ids),
        candidate_count=len(model_ids),
    )


def export_fixed_mpnet_k50_control(
    *,
    query_path: Path,
    qrel_path: Path,
    embedding_path: Path,
    output_tsv: Path,
) -> LegacyControlValidation:
    """Export the locked fixed-MPNet, k=50 legacy control."""

    return _export_fixed_mpnet_k50_control_attested(
        query_path=query_path,
        qrel_path=qrel_path,
        embedding_path=embedding_path,
        output_tsv=output_tsv,
    ).validation


def validate_legacy_control_lock_bindings(
    lock_path: str | Path,
    *,
    expected_lock_sha256: str,
    expected_tsv_sha256: str,
) -> LegacyControlLockValidation:
    """Validate and expose every downstream binding in one legacy-control root."""

    source_lock = Path(lock_path)
    if source_lock.name != _LOCK_FILENAME:
        raise ValueError("legacy control lock filename is invalid")
    if not _is_sha256(expected_lock_sha256):
        raise ValueError("expected lock hash is invalid")
    if not _is_sha256(expected_tsv_sha256):
        raise ValueError("expected TSV hash is invalid")
    try:
        resolved_lock = source_lock.resolve(strict=True)
    except OSError as error:
        raise ValueError("legacy control lock is missing") from error
    root = resolved_lock.parent
    if resolved_lock != root / _LOCK_FILENAME:
        raise ValueError("legacy control lock filename is invalid")
    _validate_local_manifest(root)
    manifest_path = root / "manifest.json"
    observed_manifest_sha256 = _bounded_regular_file_sha256(
        manifest_path,
        byte_limit=1024 * 1024,
        label="legacy manifest",
    )
    observed_lock_sha256 = _bounded_regular_file_sha256(
        resolved_lock,
        byte_limit=256 * 1024,
        label="legacy control lock",
    )
    if expected_lock_sha256 != observed_lock_sha256:
        raise ValueError("expected lock hash differs from observed bytes")

    lock = _validate_lock_contract(
        _read_canonical_json(
            resolved_lock,
            byte_limit=256 * 1024,
            label="legacy control lock",
        )
    )
    repeat_a_path = root / "repeat_a.json"
    repeat_b_path = root / "repeat_b.json"
    repeat_a = _validate_repeat(
        _read_canonical_json(
            repeat_a_path, byte_limit=256 * 1024, label="legacy repeat A receipt"
        ),
        repeat="a",
    )
    repeat_b = _validate_repeat(
        _read_canonical_json(
            repeat_b_path, byte_limit=256 * 1024, label="legacy repeat B receipt"
        ),
        repeat="b",
    )
    if {key: value for key, value in repeat_a.items() if key != "repeat"} != {
        key: value for key, value in repeat_b.items() if key != "repeat"
    }:
        raise ValueError("legacy repeat receipts disagree")
    if lock["repeat_a_receipt_sha256"] != _bounded_regular_file_sha256(
        repeat_a_path,
        byte_limit=256 * 1024,
        label="legacy repeat A receipt",
    ):
        raise ValueError("legacy repeat A receipt hash mismatch")
    if lock["repeat_b_receipt_sha256"] != _bounded_regular_file_sha256(
        repeat_b_path,
        byte_limit=256 * 1024,
        label="legacy repeat B receipt",
    ):
        raise ValueError("legacy repeat B receipt hash mismatch")

    bindings = (
        ("feasibility_manifest_sha256", "feasibility"),
        ("source_sha256", "source"),
        ("query_file_sha256", "query file"),
        ("qrel_file_sha256", "qrel"),
        ("query_input_sha256", "query-input"),
        ("candidate_inventory_sha256", "candidate inventory"),
        ("embedding_sha256", "embedding"),
        ("similarity_sha256", "similarity"),
        ("neighbor_count", "neighbor count"),
        ("tsv_sha256", "TSV hash"),
        ("row_count", "row count"),
        ("query_count", "query count"),
        ("candidate_count", "candidate count"),
        ("mean_ndcg_at_10", "nDCG"),
        ("mean_reciprocal_rank", "MRR"),
    )
    for field, label in bindings:
        if lock[field] != repeat_a[field]:
            raise ValueError(f"legacy control {label} cross-binding mismatch")

    tsv_path = root / _TSV_FILENAME
    tsv_bytes = _read_bounded_regular_file(
        tsv_path,
        byte_limit=32 * 1024 * 1024,
        label="legacy control TSV",
    )
    observed_tsv_sha256 = hashlib.sha256(tsv_bytes).hexdigest()
    if expected_tsv_sha256 != observed_tsv_sha256:
        raise ValueError("expected TSV hash differs from observed bytes")
    if lock["tsv_sha256"] != observed_tsv_sha256:
        raise ValueError("legacy control TSV hash mismatch")
    rankings, candidate_ids, row_count = _parse_legacy_tsv(tsv_bytes)
    if _candidate_inventory_sha256(candidate_ids) != lock["candidate_inventory_sha256"]:
        raise ValueError("legacy control candidate inventory hash mismatch")

    trec_path = root / _TREC_FILENAME
    observed_trec = _read_bounded_regular_file(
        trec_path, byte_limit=16 * 1024 * 1024, label="legacy TREC run"
    )
    observed_trec_sha256 = hashlib.sha256(observed_trec).hexdigest()
    if lock["trec_sha256"] != observed_trec_sha256:
        raise ValueError("legacy control TREC hash mismatch")
    reconstructed = _trec_bytes_from_rankings(rankings)
    if observed_trec != reconstructed:
        raise ValueError("legacy control TREC bytes are not canonical")

    _validate_local_manifest(root)
    if (
        _bounded_regular_file_sha256(
            manifest_path,
            byte_limit=1024 * 1024,
            label="legacy manifest",
        )
        != observed_manifest_sha256
    ):
        raise ValueError("legacy control manifest changed during validation")
    if (
        _bounded_regular_file_sha256(
            resolved_lock,
            byte_limit=256 * 1024,
            label="legacy control lock",
        )
        != observed_lock_sha256
    ):
        raise ValueError("legacy control lock changed during validation")

    return LegacyControlLockValidation(
        manifest_sha256=observed_manifest_sha256,
        lock_sha256=observed_lock_sha256,
        feasibility_manifest_sha256=lock["feasibility_manifest_sha256"],
        source_sha256=MappingProxyType(_validate_source_map(lock["source_sha256"])),
        query_file_sha256=lock["query_file_sha256"],
        qrel_file_sha256=lock["qrel_file_sha256"],
        query_input_sha256=lock["query_input_sha256"],
        candidate_inventory_sha256=lock["candidate_inventory_sha256"],
        embedding_sha256=lock["embedding_sha256"],
        similarity_sha256=lock["similarity_sha256"],
        tsv_sha256=observed_tsv_sha256,
        trec_sha256=observed_trec_sha256,
        run_id=lock["run_id"],
        neighbor_count=lock["neighbor_count"],
        query_count=lock["query_count"],
        candidate_count=lock["candidate_count"],
        row_count=row_count,
        mean_ndcg_at_10=float(lock["mean_ndcg_at_10"]),
        mean_reciprocal_rank=float(lock["mean_reciprocal_rank"]),
    )
