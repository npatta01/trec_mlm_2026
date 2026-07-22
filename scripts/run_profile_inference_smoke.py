#!/usr/bin/env python3
"""Run or compare the pinned qrel-free ROCm profile-inference smoke gate."""

from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable
import uuid

import numpy as np

from trec_mlm.label_experiment import load_candidate_manifest
from trec_mlm.label_inference import ROCM_IMAGE
from trec_mlm.label_retrieval import CONTEXTUAL_E5_SPEC
from trec_mlm.profile_inference import (
    PROFILE_BATCH_SIZE,
    PROFILE_DEVICE_ARCHITECTURE,
    PROFILE_SEED,
    encode_profile_inputs,
    format_profile_document,
    format_profile_query,
)
from trec_mlm.profile_inference_artifact import validate_profile_smoke_receipt
from trec_mlm.profile_retrieval import filter_and_deduplicate_records
from trec_mlm.response_retrieval import load_discovery_records


INDIVIDUAL_REQUIRED_KEYS = frozenset(
    {
        "artifact_type", "version", "execution_id", "qrel_free",
        "rankings_or_qrels_used", "device_architecture", "container_image",
        "model_id", "model_revision", "batch_size", "dtype", "persisted_dtype",
        "smoke_inventory", "selected_record_ids", "embedding_shape",
        "embedding_sha256", "runtime_metadata",
    }
)
IDENTITY_KEYS = (
    "qrel_free", "rankings_or_qrels_used", "device_architecture",
    "container_image", "model_id", "model_revision", "batch_size", "dtype",
    "persisted_dtype", "smoke_inventory",
)
PINNED_IDENTITY = {
    "qrel_free": True,
    "rankings_or_qrels_used": False,
    "device_architecture": PROFILE_DEVICE_ARCHITECTURE,
    "container_image": ROCM_IMAGE,
    "model_id": CONTEXTUAL_E5_SPEC.model_id,
    "model_revision": CONTEXTUAL_E5_SPEC.revision,
    "batch_size": PROFILE_BATCH_SIZE,
    "dtype": CONTEXTUAL_E5_SPEC.dtype,
    "persisted_dtype": "float32",
    "smoke_inventory": ["short", "median", "over_512"],
}
RUNTIME_KEYS = frozenset(
    {
        "backend", "model_id", "requested_revision", "resolved_model_revision",
        "resolved_tokenizer_revision", "license", "required_container_image",
        "dtype", "persisted_dtype", "pooling", "embedding_dimension",
        "input_count", "input_sha256", "batch_size", "seed", "padding_side",
        "truncation_side", "terminal_token_id", "terminal_token", "max_length",
        "maximum_observed_tokens", "maximum_untruncated_tokens",
        "truncated_input_count", "document_response_truncation", "parameter_count",
        "peak_memory_allocated_bytes", "peak_memory_reserved_bytes", "device",
        "device_properties", "torch_version", "hip_version",
        "transformers_version", "encoding_seconds",
    }
)
PINNED_RUNTIME = {
    "backend": "transformers_rocm",
    "model_id": CONTEXTUAL_E5_SPEC.model_id,
    "requested_revision": CONTEXTUAL_E5_SPEC.revision,
    "resolved_model_revision": CONTEXTUAL_E5_SPEC.revision,
    "resolved_tokenizer_revision": CONTEXTUAL_E5_SPEC.revision,
    "license": CONTEXTUAL_E5_SPEC.license,
    "required_container_image": ROCM_IMAGE,
    "dtype": CONTEXTUAL_E5_SPEC.dtype,
    "persisted_dtype": "float32",
    "pooling": CONTEXTUAL_E5_SPEC.pooling,
    "embedding_dimension": CONTEXTUAL_E5_SPEC.embedding_dimension,
    "input_count": 4,
    "batch_size": PROFILE_BATCH_SIZE,
    "seed": PROFILE_SEED,
    "padding_side": "left",
    "truncation_side": "right",
    "terminal_token_id": 2,
    "max_length": 512,
    "device": "rocm_hip",
    "torch_version": "2.11.0+gitd0c8b1f",
    "hip_version": "7.2.53211",
    "transformers_version": "5.12.1",
}
RUNTIME_NONDETERMINISTIC_KEYS = frozenset(
    {
        "encoding_seconds",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
    }
)


def _rename_noreplace(source: Path, destination: Path) -> None:
    library = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(library, "renameat2", None)
    if renameat2 is None:
        raise RuntimeError("atomic no-replace directory rename is unavailable")
    renameat2.argtypes = (
        ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint
    )
    renameat2.restype = ctypes.c_int
    if renameat2(-100, os.fsencode(source), -100, os.fsencode(destination), 1) == 0:
        return
    error_number = ctypes.get_errno()
    if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
        raise FileExistsError(
            error_number, os.strerror(error_number), str(destination)
        )
    raise OSError(error_number, os.strerror(error_number), str(destination))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    operations = parser.add_subparsers(dest="operation", required=True)
    run = operations.add_parser("run", help="execute one fresh GPU smoke process")
    run.add_argument("--discovery", required=True)
    run.add_argument("--candidates", required=True)
    run.add_argument("--output", required=True)
    compare = operations.add_parser("compare", help="compare two fresh smoke outputs")
    compare.add_argument("--first", required=True)
    compare.add_argument("--second", required=True)
    compare.add_argument("--receipt", required=True)
    return parser


def _record_key(record: Any) -> tuple[int, int, str]:
    formatted = format_profile_document(record.question, record.response)
    return (len(formatted.split()), len(formatted.encode("utf-8")), record.record_id)


def _select_smoke_records(records: Iterable[Any]) -> tuple[tuple[str, Any], ...]:
    ordered = sorted(records, key=_record_key)
    if len(ordered) < 3:
        raise ValueError("smoke selection requires at least three usable records")
    short = ordered[0]
    over = ordered[-1]
    middle = ordered[len(ordered) // 2]
    if len({short.record_id, middle.record_id, over.record_id}) != 3:
        raise ValueError("smoke selections must be three distinct records")
    return (("short", short), ("median", middle), ("over_512", over))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _run_smoke(discovery: str, candidates: str, output: str) -> dict[str, Any]:
    destination = Path(output)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    candidate_ids = load_candidate_manifest(candidates)
    corpus = load_discovery_records(discovery, candidate_ids)
    filtered = filter_and_deduplicate_records(corpus, candidate_ids)
    records = (
        record
        for candidate_id in candidate_ids
        for record in filtered.records_by_candidate[candidate_id]
    )
    selected = _select_smoke_records(records)
    documents = tuple(
        format_profile_document(record.question, record.response)
        for _, record in selected
    )
    query = format_profile_query(selected[-1][1].question)
    query_embeddings, document_embeddings, runtime = encode_profile_inputs(
        formatted_queries=(query,),
        formatted_documents=documents,
        batch_size=PROFILE_BATCH_SIZE,
        seed=PROFILE_SEED,
    )
    truncation = runtime.get("document_response_truncation")
    if not isinstance(truncation, list) or len(truncation) != 3 or truncation[-1] == "none":
        raise ValueError("deterministic over-512 smoke record was not truncated")
    embeddings = np.asarray(
        np.concatenate((query_embeddings, document_embeddings), axis=0),
        dtype=np.float32,
        order="C",
    )
    if (
        embeddings.shape != (4, CONTEXTUAL_E5_SPEC.embedding_dimension)
        or not np.isfinite(embeddings).all()
        or not np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("smoke embeddings violate the frozen normalized shape")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        embedding_path = staging / "embeddings.npy"
        np.save(embedding_path, embeddings, allow_pickle=False)
        receipt = {
            "artifact_type": "profile_inference_smoke",
            "version": 1,
            "execution_id": str(uuid.uuid4()),
            "qrel_free": True,
            "rankings_or_qrels_used": False,
            "device_architecture": PROFILE_DEVICE_ARCHITECTURE,
            "container_image": ROCM_IMAGE,
            "model_id": CONTEXTUAL_E5_SPEC.model_id,
            "model_revision": CONTEXTUAL_E5_SPEC.revision,
            "batch_size": PROFILE_BATCH_SIZE,
            "dtype": CONTEXTUAL_E5_SPEC.dtype,
            "persisted_dtype": "float32",
            "smoke_inventory": [label for label, _ in selected],
            "selected_record_ids": [record.record_id for _, record in selected],
            "embedding_shape": list(embeddings.shape),
            "embedding_sha256": _sha256(embedding_path),
            "runtime_metadata": runtime,
        }
        _write_json(staging / "receipt.json", receipt)
        _rename_noreplace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


def _load_individual(root: str | Path) -> tuple[dict[str, Any], np.ndarray, str]:
    directory = Path(root)
    receipt_path = directory / "receipt.json"
    embedding_path = directory / "embeddings.npy"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or set(receipt) != INDIVIDUAL_REQUIRED_KEYS:
        raise ValueError("individual smoke receipt schema changed")
    if (
        not isinstance(receipt.get("runtime_metadata"), dict)
        or set(receipt["runtime_metadata"]) != RUNTIME_KEYS
        or receipt["artifact_type"] != "profile_inference_smoke"
        or receipt["version"] != 1
        or any(receipt.get(key) != value for key, value in PINNED_IDENTITY.items())
        or receipt["embedding_shape"] != [4, CONTEXTUAL_E5_SPEC.embedding_dimension]
        or receipt["embedding_sha256"] != _sha256(embedding_path)
    ):
        raise ValueError("individual smoke receipt binding changed")
    selected = receipt["selected_record_ids"]
    runtime = receipt["runtime_metadata"]
    truncation = runtime.get("document_response_truncation")
    if (
        not isinstance(selected, list)
        or len(selected) != 3
        or len(set(selected)) != 3
        or any(not isinstance(value, str) or not value for value in selected)
        or any(runtime.get(key) != value for key, value in PINNED_RUNTIME.items())
        or not isinstance(runtime.get("device_properties"), str)
        or PROFILE_DEVICE_ARCHITECTURE not in runtime["device_properties"]
        or not isinstance(runtime.get("input_sha256"), str)
        or len(runtime["input_sha256"]) != 64
        or not isinstance(truncation, list)
        or len(truncation) != 3
        or truncation[-1] == "none"
    ):
        raise ValueError("individual smoke input or pinned runtime changed")
    try:
        uuid.UUID(receipt["execution_id"])
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("individual smoke execution identity is invalid") from error
    values = np.load(embedding_path, allow_pickle=False)
    if (
        values.dtype != np.dtype("<f4")
        or values.shape != (4, CONTEXTUAL_E5_SPEC.embedding_dimension)
        or not values.flags.c_contiguous
        or not np.isfinite(values).all()
        or not np.allclose(np.linalg.norm(values, axis=1), 1.0, atol=1e-5)
    ):
        raise ValueError("individual smoke embedding contract changed")
    return receipt, values, _sha256(receipt_path)


def _compare(first: str, second: str, destination: str) -> dict[str, Any]:
    first_root = Path(first).resolve()
    second_root = Path(second).resolve()
    if first_root == second_root:
        raise ValueError("smoke repeats must use distinct roots")
    first_receipt, first_values, first_sha = _load_individual(first_root)
    second_receipt, second_values, second_sha = _load_individual(second_root)
    if first_receipt["execution_id"] == second_receipt["execution_id"]:
        raise ValueError("smoke repeats require distinct fresh executions")
    if any(first_receipt[key] != second_receipt[key] for key in IDENTITY_KEYS):
        raise ValueError("smoke repeat runtime identities differ")
    if first_receipt["selected_record_ids"] != second_receipt["selected_record_ids"]:
        raise ValueError("smoke repeat selected record identities differ")
    first_runtime = first_receipt["runtime_metadata"]
    second_runtime = second_receipt["runtime_metadata"]
    stable_runtime_keys = RUNTIME_KEYS - RUNTIME_NONDETERMINISTIC_KEYS
    if any(first_runtime[key] != second_runtime[key] for key in stable_runtime_keys):
        raise ValueError("smoke repeat runtime metadata differs")
    maximum = float(np.max(np.abs(first_values - second_values)))
    if not math.isfinite(maximum) or maximum > 1e-5:
        raise ValueError("smoke repeat embedding difference exceeds 1e-5")
    payload = {
        "status": "passed",
        "qrel_free": True,
        "rankings_or_qrels_used": False,
        "maximum_absolute_embedding_difference": maximum,
        "tolerance": 1e-5,
        "device_architecture": PROFILE_DEVICE_ARCHITECTURE,
        "container_image": ROCM_IMAGE,
        "model_id": CONTEXTUAL_E5_SPEC.model_id,
        "model_revision": CONTEXTUAL_E5_SPEC.revision,
        "batch_size": PROFILE_BATCH_SIZE,
        "dtype": CONTEXTUAL_E5_SPEC.dtype,
        "persisted_dtype": "float32",
        "smoke_inventory": ["short", "median", "over_512"],
        "first_smoke_receipt_sha256": first_sha,
        "second_smoke_receipt_sha256": second_sha,
    }
    validate_profile_smoke_receipt(payload)
    destination_path = Path(destination)
    if destination_path.exists():
        raise FileExistsError(f"output already exists: {destination_path}")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", dir=destination_path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        _write_json(temporary, payload)
        os.link(temporary, destination_path)
    finally:
        temporary.unlink(missing_ok=True)
    return payload


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.operation == "run":
            payload = _run_smoke(args.discovery, args.candidates, args.output)
        else:
            payload = _compare(args.first, args.second, args.receipt)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
