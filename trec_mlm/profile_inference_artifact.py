"""Qrel-free, ROCm-only Stage 1 artifacts for candidate semantic profiles."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.artifact_io import (
    length_framed_sha256,
    manifest_aggregate_sha256,
    manifest_entries,
    sha256_file,
    source_tree_sha256,
    write_pretty_json,
)
from trec_mlm.label_experiment import (
    load_candidate_manifest,
    load_frozen_legacy_label_experiment_artifacts,
)
from trec_mlm.label_inference import (
    ROCM_IMAGE,
    encode_contextual_e5 as _encode_contextual_e5,
)
from trec_mlm.label_retrieval import CONTEXTUAL_E5_SPEC, format_query
from trec_mlm.profile_inference import (
    PROFILE_BATCH_SIZE,
    PROFILE_DEVICE_ARCHITECTURE,
    PROFILE_SEED,
    encode_profile_inputs,
    format_profile_document,
    format_profile_query,
)
from trec_mlm.profile_retrieval import (
    canonical_profile_record_id,
    filter_and_deduplicate_records,
)
from trec_mlm.query_knn import load_queries_tsv
from trec_mlm.response_retrieval import load_discovery_records


ARTIFACT_VERSION = 2
LEGACY_ARTIFACT_VERSION = 1
LEGACY_PASS_ONE_AGGREGATE_SHA256 = (
    "20a72f0cbb04ef1c1e58a7caf2d10c9357f3529c1bcd3e66f4a7ec5f371657c5"
)
RECORD_ID_AMENDMENT_SHA256 = (
    "a43e13518aa74365dc7bbe2022c4ee22fec47cb7b5f44ee65523f6ba6e5cb31a"
)
OFFICIAL_DISCOVERY_SHA256 = (
    "b33c28aa2571cdf5b2867dcc00f0dc55e22198b90022dd23f748a03395301b25"
)
FROZEN_E5_VALIDATION_SHA256 = (
    "0049f337106aa0bafa183d43cb8bd08f6426ffc2bb3a4d9d3991f5fb46806c35"
)
OFFICIAL_QUERY_COUNT = 322
OFFICIAL_CANDIDATE_COUNT = 150
OFFICIAL_RECORD_COUNT = 182_764
OFFICIAL_LABEL_QUESTION_COUNT = 7_759
ARRAY_FILES = (
    "query_embeddings.npy",
    "record_embeddings.npy",
    "label_question_embeddings.npy",
    "label_embeddings.npy",
)
NONDETERMINISTIC_METADATA_KEYS = frozenset(
    {
        "encoding_seconds",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
    }
)
REPAIR_PROVENANCE_KEYS = frozenset(
    {
        "qrel_free",
        "record_id_scheme",
        "source_artifact_version",
        "source_aggregate_sha256",
        "source_source_tree_sha256",
        "repair_source_tree_sha256",
        "protocol_amendment_sha256",
        "byte_identical_files",
    }
)
BYTE_IDENTICAL_REPAIR_FILES = (*ARRAY_FILES, "formatted_inputs.json")
FRESH_EXECUTION_RECEIPT_KEYS = frozenset(
    {
        "artifact_type",
        "version",
        "mode",
        "qrel_free",
        "embeddings_reused",
        "record_id_scheme",
        "artifact_aggregate_sha256",
        "source_tree_sha256",
        "container_image",
        "device_architecture",
        "model_id",
        "model_revision",
        "query_count",
        "candidate_count",
        "record_count",
    }
)


@dataclass(frozen=True, slots=True)
class ProfileInferenceArtifactConfig:
    query_path: str
    discovery_path: str
    candidate_path: str
    e5_artifact_path: str
    smoke_repeat_receipt_path: str = "smoke_repeat_receipt.json"
    batch_size: int = PROFILE_BATCH_SIZE
    seed: int = PROFILE_SEED


PROFILE_INFERENCE_CONFIG_KEYS = frozenset(
    {
        "artifact_type",
        "version",
        "qrel_free",
        *(field.name for field in fields(ProfileInferenceArtifactConfig)),
    }
)


@dataclass(frozen=True, slots=True)
class ProfileInferenceArtifactResult:
    config: ProfileInferenceArtifactConfig
    query_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    label_question_ids: tuple[str, ...]
    records: list[dict[str, Any]]
    audits: dict[str, dict[str, Any]]
    formatted_inputs: dict[str, list[str]]
    query_embeddings: NDArray[np.float32]
    record_embeddings: NDArray[np.float32]
    label_question_embeddings: NDArray[np.float32]
    label_embeddings: NDArray[np.float32]
    e5_rankings: dict[str, list[str]]
    metadata: dict[str, Any]
    manifest: dict[str, Any] | None = None
    root: Path | None = None


_sha256 = sha256_file


def _source_tree_sha256() -> str:
    return source_tree_sha256(Path(__file__).resolve().parent)


_length_framed = length_framed_sha256
_write_json = write_pretty_json
_manifest_entries = manifest_entries
_aggregate_hash = manifest_aggregate_sha256


def _canonical_repeat_metadata(value: Any) -> Any:
    """Remove only preregistered run-to-run telemetry from Stage 1 metadata."""

    if isinstance(value, Mapping):
        return {
            key: _canonical_repeat_metadata(item)
            for key, item in sorted(value.items())
            if key not in NONDETERMINISTIC_METADATA_KEYS
        }
    if isinstance(value, list):
        return [_canonical_repeat_metadata(item) for item in value]
    return value


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _repeat_metadata_with_explicit_lineage(
    artifact: ProfileInferenceArtifactResult, *, allow_pinned_repair: bool
) -> tuple[Any, bool]:
    metadata = artifact.metadata
    payload = json.loads(json.dumps(metadata))
    repair = payload.pop("record_id_repair", None)
    if repair is None:
        return _canonical_repeat_metadata(payload), False
    if not allow_pinned_repair:
        raise ValueError("only repeat one may be the pinned record-ID repair")
    if not isinstance(repair, dict) or set(repair) != REPAIR_PROVENANCE_KEYS:
        raise ValueError("record-ID repair provenance keys changed")
    expected = {
        "qrel_free": True,
        "record_id_scheme": "profile-record-v2",
        "source_artifact_version": LEGACY_ARTIFACT_VERSION,
        "source_aggregate_sha256": LEGACY_PASS_ONE_AGGREGATE_SHA256,
        "protocol_amendment_sha256": RECORD_ID_AMENDMENT_SHA256,
    }
    if any(repair.get(key) != value for key, value in expected.items()):
        raise ValueError("record-ID repair lineage differs from the approved amendment")
    locked = repair.get("byte_identical_files")
    if not isinstance(locked, dict) or set(locked) != set(BYTE_IDENTICAL_REPAIR_FILES):
        raise ValueError("record-ID repair byte-identity evidence changed")
    if artifact.manifest is None or any(
        not _is_sha256(value)
        or artifact.manifest["files"].get(name, {}).get("sha256") != value
        for name, value in locked.items()
    ):
        raise ValueError("record-ID repair byte-identity hashes changed")
    source_tree = payload.get("source_tree_sha256")
    repair_tree = repair.get("repair_source_tree_sha256")
    if (
        not _is_sha256(source_tree)
        or not _is_sha256(repair.get("source_source_tree_sha256"))
        or repair.get("source_source_tree_sha256") != source_tree
        or not _is_sha256(repair_tree)
    ):
        raise ValueError("record-ID repair source tree lineage changed")
    payload["source_tree_sha256"] = repair_tree
    return _canonical_repeat_metadata(payload), True


def _validate_config(config: ProfileInferenceArtifactConfig) -> None:
    for field in fields(config):
        value = getattr(config, field.name)
        if field.name.endswith("_path") and (not isinstance(value, str) or not value):
            raise ValueError(f"{field.name} must be a non-empty path")
    if config.batch_size != PROFILE_BATCH_SIZE or config.seed != PROFILE_SEED:
        raise ValueError("inference batch and seed differ from frozen protocol")


def _validate_record_ids(records: list[dict[str, Any]], *, fixture: bool) -> None:
    record_ids = [record.get("record_id") for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise ValueError("profile inference record IDs are not globally unique")
    if fixture:
        return
    for record in records:
        expected = canonical_profile_record_id(
            record["candidate_id"],
            record["source_query_id"],
            record["question"],
            record["response"],
        )
        if record["record_id"] != expected:
            raise ValueError("profile inference record ID is not canonical schema v2")


def validate_profile_smoke_receipt(payload: Mapping[str, Any]) -> None:
    """Validate the mandatory pre-full-run smoke gate against frozen identities."""

    expected = {
        "status": "passed",
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
    if any(payload.get(key) != value for key, value in expected.items()):
        raise ValueError("profile smoke receipt identity contract changed")
    tolerance = payload.get("tolerance")
    maximum = payload.get("maximum_absolute_embedding_difference")
    if tolerance != 1e-5 or not isinstance(maximum, (int, float)) or maximum > tolerance:
        raise ValueError("profile smoke receipt repeat tolerance failed")
    first_sha = payload.get("first_smoke_receipt_sha256")
    second_sha = payload.get("second_smoke_receipt_sha256")
    if any(not isinstance(value, str) or len(value) != 64 for value in (first_sha, second_sha)) or first_sha == second_sha:
        raise ValueError("profile smoke receipt does not prove two fresh runs")


def ordered_source_questions(corpus: Any) -> tuple[tuple[str, ...], dict[str, str]]:
    """Return every raw discovery question, including those with no usable response."""

    questions: dict[str, str] = {}
    for record in corpus:
        previous = questions.setdefault(record.source_query_id, record.query_text)
        if previous != record.query_text:
            raise ValueError("source query ID maps to multiple raw questions")
    identifiers = tuple(sorted(questions))
    return identifiers, questions


def run_profile_inference_artifact(
    config: ProfileInferenceArtifactConfig,
) -> ProfileInferenceArtifactResult:
    """Encode the complete official inventory without clustering or judgments."""

    _validate_config(config)
    smoke_receipt = json.loads(
        Path(config.smoke_repeat_receipt_path).read_text(encoding="utf-8")
    )
    validate_profile_smoke_receipt(smoke_receipt)
    queries = load_queries_tsv(config.query_path)
    candidate_ids = load_candidate_manifest(config.candidate_path)
    corpus = load_discovery_records(config.discovery_path, candidate_ids)
    if corpus.source_sha256 != OFFICIAL_DISCOVERY_SHA256:
        raise ValueError("discovery source SHA-256 differs from frozen protocol")
    filtered = filter_and_deduplicate_records(corpus, candidate_ids)
    flat_records = tuple(
        record
        for candidate_id in candidate_ids
        for record in filtered.records_by_candidate[candidate_id]
    )
    formatted_queries = tuple(format_profile_query(value) for value in queries.values())
    formatted_documents = tuple(
        format_profile_document(record.question, record.response)
        for record in flat_records
    )
    e5_validation = Path(config.e5_artifact_path) / "validation.json"
    if _sha256(e5_validation) != FROZEN_E5_VALIDATION_SHA256:
        raise ValueError("contextual E5 validation differs from frozen control")
    e5 = load_frozen_legacy_label_experiment_artifacts(config.e5_artifact_path)
    if e5.query_ids != tuple(queries) or e5.candidate_ids != candidate_ids:
        raise ValueError("contextual E5 inventory differs from inference inputs")
    query_embeddings, record_embeddings, profile_metadata = encode_profile_inputs(
        formatted_queries=formatted_queries,
        formatted_documents=formatted_documents,
        batch_size=config.batch_size,
        seed=config.seed,
    )
    truncation = profile_metadata.pop("document_response_truncation")
    label_question_ids, unique_questions = ordered_source_questions(corpus)
    formatted_label_questions = tuple(
        format_query(unique_questions[item]) for item in label_question_ids
    )
    label_question_embeddings, label_metadata = _encode_contextual_e5(
        formatted_label_questions, batch_size=PROFILE_BATCH_SIZE, seed=config.seed
    )
    records: list[dict[str, Any]] = []
    audits: dict[str, dict[str, Any]] = {}
    cursor = 0
    for candidate_id in candidate_ids:
        candidate_records = filtered.records_by_candidate[candidate_id]
        classes = truncation[cursor : cursor + len(candidate_records)]
        cursor += len(candidate_records)
        audits[candidate_id] = {
            **asdict(filtered.audit_by_candidate[candidate_id]),
            "not_truncated": classes.count("none"),
            "partial_response_truncation": classes.count("partial"),
            "full_response_truncation": classes.count("full"),
        }
        for record in candidate_records:
            formatted = format_profile_document(record.question, record.response)
            records.append(
                {
                    "record_id": record.record_id,
                    "candidate_id": candidate_id,
                    "source_query_id": record.source_query_id,
                    "question": record.question,
                    "response": record.response,
                    "raw_membership_sha256": _length_framed(
                        (record.question, record.response)
                    ),
                    "formatted_sha256": hashlib.sha256(
                        formatted.encode("utf-8")
                    ).hexdigest(),
                }
            )
    if cursor != len(flat_records):
        raise AssertionError("truncation audit did not cover all records")
    return ProfileInferenceArtifactResult(
        config=config,
        query_ids=tuple(queries),
        candidate_ids=candidate_ids,
        label_question_ids=label_question_ids,
        records=records,
        audits=audits,
        formatted_inputs={
            "target_queries": list(formatted_queries),
            "profile_documents": list(formatted_documents),
            "label_questions": list(formatted_label_questions),
        },
        query_embeddings=query_embeddings,
        record_embeddings=record_embeddings,
        label_question_embeddings=label_question_embeddings,
        label_embeddings=e5.label_embeddings,
        e5_rankings=e5.rankings,
        metadata={
            "qrel_free": True,
            "source_tree_sha256": _source_tree_sha256(),
            "discovery_sha256": corpus.source_sha256,
            "query_source_sha256": _sha256(Path(config.query_path)),
            "candidate_source_sha256": _sha256(Path(config.candidate_path)),
            "smoke_repeat_receipt_sha256": _sha256(
                Path(config.smoke_repeat_receipt_path)
            ),
            "e5_validation_sha256": FROZEN_E5_VALIDATION_SHA256,
            "profile_encoder": profile_metadata,
            "label_encoder": label_metadata,
            "record_count": len(records),
        },
    )


def write_profile_inference_artifacts(
    result: ProfileInferenceArtifactResult, output_dir: str | Path
) -> dict[str, str]:
    """Atomically persist the exact Stage 1 cross-environment handoff."""

    destination = Path(output_dir)
    _validate_record_ids(result.records, fixture=bool(result.metadata.get("fixture")))
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        _write_json(staging / "config.json", {"artifact_type": "profile_inference", "version": ARTIFACT_VERSION, "qrel_free": True, **asdict(result.config)})
        _write_json(staging / "inputs.json", {"query_ids": result.query_ids, "candidate_ids": result.candidate_ids, "label_question_ids": result.label_question_ids})
        _write_json(staging / "records.json", result.records)
        _write_json(staging / "audits.json", result.audits)
        _write_json(staging / "formatted_inputs.json", result.formatted_inputs)
        _write_json(staging / "e5_rankings.json", result.e5_rankings)
        _write_json(staging / "metadata.json", result.metadata)
        arrays = {
            "query_embeddings.npy": result.query_embeddings,
            "record_embeddings.npy": result.record_embeddings,
            "label_question_embeddings.npy": result.label_question_embeddings,
            "label_embeddings.npy": result.label_embeddings,
        }
        for name, values in arrays.items():
            matrix = np.asarray(values)
            if matrix.dtype != np.float32 or not matrix.flags.c_contiguous:
                raise ValueError(f"{name} must be C-contiguous float32")
            np.save(staging / name, matrix, allow_pickle=False)
        _write_json(staging / "validation.json", {"status": "passed", "qrel_free": True, "array_row_mappings": {"query_embeddings.npy": "inputs.query_ids", "record_embeddings.npy": "records order", "label_question_embeddings.npy": "inputs.label_question_ids", "label_embeddings.npy": "inputs.candidate_ids"}})
        entries = _manifest_entries(staging)
        _write_json(staging / "manifest.json", {"artifact_type": "profile_inference_manifest", "version": ARTIFACT_VERSION, "qrel_free": True, "files": entries, "aggregate_sha256": _aggregate_hash(entries)})
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        path.relative_to(destination).as_posix(): str(path)
        for path in destination.rglob("*")
        if path.is_file()
    }


def repair_legacy_profile_inference_artifact(
    source_dir: str | Path,
    output_dir: str | Path,
    amendment_path: str | Path,
) -> dict[str, Any]:
    """Atomically migrate only the pinned pass-one IDs to the schema-v2 contract."""

    source = Path(source_dir)
    destination = Path(output_dir)
    amendment = Path(amendment_path)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    if _sha256(amendment) != RECORD_ID_AMENDMENT_SHA256:
        raise ValueError("record-ID protocol amendment hash changed")
    source_manifest_bytes = (source / "manifest.json").read_bytes()
    manifest = json.loads(source_manifest_bytes)
    entries = manifest.get("files")
    if (
        manifest.get("artifact_type") != "profile_inference_manifest"
        or manifest.get("version") != LEGACY_ARTIFACT_VERSION
        or manifest.get("qrel_free") is not True
        or not isinstance(entries, dict)
        or manifest.get("aggregate_sha256") != _aggregate_hash(entries)
        or manifest.get("aggregate_sha256") != LEGACY_PASS_ONE_AGGREGATE_SHA256
    ):
        raise ValueError("legacy repair source is not the pinned pass-one artifact")
    actual = {
        path.relative_to(source).as_posix()
        for path in source.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != set(entries):
        raise ValueError("legacy repair source file inventory changed")
    for name, entry in entries.items():
        path = source / name
        if (
            path.stat().st_size != entry.get("byte_size")
            or _sha256(path) != entry.get("sha256")
        ):
            raise ValueError(f"legacy repair source hash mismatch: {name}")

    records = json.loads((source / "records.json").read_text(encoding="utf-8"))
    repaired_records: list[dict[str, Any]] = []
    for record in records:
        raw_hash = _length_framed((record["question"], record["response"]))
        legacy_id = f'{record["source_query_id"]}:{raw_hash[:16]}'
        if (
            record.get("record_id") != legacy_id
            or record.get("raw_membership_sha256") != raw_hash
        ):
            raise ValueError("legacy repair source record identity changed")
        repaired_records.append(
            {
                **record,
                "record_id": canonical_profile_record_id(
                    record["candidate_id"],
                    record["source_query_id"],
                    record["question"],
                    record["response"],
                ),
            }
        )
    _validate_record_ids(repaired_records, fixture=False)

    byte_identical_names = (*ARRAY_FILES, "formatted_inputs.json")
    source_hashes = {name: _sha256(source / name) for name in byte_identical_names}
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        config = json.loads((staging / "config.json").read_text(encoding="utf-8"))
        if (
            config.get("artifact_type") != "profile_inference"
            or config.get("version") != LEGACY_ARTIFACT_VERSION
            or config.get("qrel_free") is not True
        ):
            raise ValueError("legacy repair source config changed")
        config["version"] = ARTIFACT_VERSION
        _write_json(staging / "config.json", config)
        _write_json(staging / "records.json", repaired_records)
        metadata = json.loads((staging / "metadata.json").read_text(encoding="utf-8"))
        metadata["record_id_repair"] = {
            "qrel_free": True,
            "record_id_scheme": "profile-record-v2",
            "source_artifact_version": LEGACY_ARTIFACT_VERSION,
            "source_aggregate_sha256": LEGACY_PASS_ONE_AGGREGATE_SHA256,
            "source_source_tree_sha256": metadata.get("source_tree_sha256"),
            "repair_source_tree_sha256": _source_tree_sha256(),
            "protocol_amendment_sha256": RECORD_ID_AMENDMENT_SHA256,
            "byte_identical_files": source_hashes,
        }
        _write_json(staging / "metadata.json", metadata)
        repaired_entries = _manifest_entries(staging)
        repaired_manifest = {
            "artifact_type": "profile_inference_manifest",
            "version": ARTIFACT_VERSION,
            "qrel_free": True,
            "files": repaired_entries,
            "aggregate_sha256": _aggregate_hash(repaired_entries),
        }
        _write_json(staging / "manifest.json", repaired_manifest)
        for name, expected_hash in source_hashes.items():
            if _sha256(staging / name) != expected_hash:
                raise ValueError(f"legacy repair changed byte-locked file: {name}")
        load_profile_inference_artifacts(staging)
        final_source_files = {
            path.relative_to(source).as_posix()
            for path in source.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        source_still_valid = final_source_files == set(entries) and all(
            (source / name).stat().st_size == entry["byte_size"]
            and _sha256(source / name) == entry["sha256"]
            for name, entry in entries.items()
        )
        if (
            not source_still_valid
            or (source / "manifest.json").read_bytes() != source_manifest_bytes
        ):
            raise ValueError("legacy repair source artifact changed")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "passed",
        "qrel_free": True,
        "source_aggregate_sha256": LEGACY_PASS_ONE_AGGREGATE_SHA256,
        "repaired_aggregate_sha256": repaired_manifest["aggregate_sha256"],
        "protocol_amendment_sha256": RECORD_ID_AMENDMENT_SHA256,
        "record_count": len(repaired_records),
        "record_ids_globally_unique": True,
        "record_ids_canonical_v2": True,
        "non_id_record_fields_and_order_identical": True,
        "formatted_inputs_byte_identical": True,
        "embedding_arrays_byte_identical": True,
        "byte_identical_files": source_hashes,
    }


def build_profile_inference_execution_receipt(
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Bind the completed artifact to the fresh full-inference execution path."""

    artifact = load_profile_inference_artifacts(artifact_dir)
    if artifact.metadata.get("record_id_repair") is not None:
        raise ValueError("a repaired artifact is not a fresh inference execution")
    source_tree = artifact.metadata.get("source_tree_sha256")
    if not artifact.metadata.get("fixture") and not _is_sha256(source_tree):
        raise ValueError("fresh inference source tree receipt is missing")
    return {
        "artifact_type": "profile_inference_execution_receipt",
        "version": 1,
        "mode": "fresh_full_inference",
        "qrel_free": True,
        "embeddings_reused": False,
        "record_id_scheme": "profile-record-v2",
        "artifact_aggregate_sha256": artifact.manifest["aggregate_sha256"],
        "source_tree_sha256": source_tree,
        "container_image": ROCM_IMAGE,
        "device_architecture": PROFILE_DEVICE_ARCHITECTURE,
        "model_id": CONTEXTUAL_E5_SPEC.model_id,
        "model_revision": CONTEXTUAL_E5_SPEC.revision,
        "query_count": len(artifact.query_ids),
        "candidate_count": len(artifact.candidate_ids),
        "record_count": len(artifact.records),
    }


def load_profile_inference_artifacts(
    artifact_dir: str | Path,
) -> ProfileInferenceArtifactResult:
    """Validate the complete manifest before reading Stage 1 arrays."""

    root = Path(artifact_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if manifest.get("artifact_type") != "profile_inference_manifest" or manifest.get("version") != ARTIFACT_VERSION or manifest.get("qrel_free") is not True or not isinstance(entries, dict) or manifest.get("aggregate_sha256") != _aggregate_hash(entries):
        raise ValueError("profile inference manifest contract changed")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual != set(entries):
        raise ValueError("profile inference file inventory changed")
    for name, entry in entries.items():
        path = root / name
        if path.stat().st_size != entry.get("byte_size") or _sha256(path) != entry.get("sha256"):
            raise ValueError(f"profile inference artifact hash mismatch: {name}")
    config_payload = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if (
        not isinstance(config_payload, dict)
        or set(config_payload) != PROFILE_INFERENCE_CONFIG_KEYS
        or config_payload["artifact_type"] != "profile_inference"
        or config_payload["version"] != ARTIFACT_VERSION
        or config_payload["qrel_free"] is not True
    ):
        raise ValueError("profile inference configuration schema changed")
    config = ProfileInferenceArtifactConfig(**{field.name: config_payload[field.name] for field in fields(ProfileInferenceArtifactConfig)})
    _validate_config(config)
    inputs = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    records = json.loads((root / "records.json").read_text(encoding="utf-8"))
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    _validate_record_ids(records, fixture=bool(metadata.get("fixture")))
    if not metadata.get("fixture") and (
        len(inputs["query_ids"]) != OFFICIAL_QUERY_COUNT
        or len(inputs["candidate_ids"]) != OFFICIAL_CANDIDATE_COUNT
        or len(records) != OFFICIAL_RECORD_COUNT
        or len(inputs["label_question_ids"]) != OFFICIAL_LABEL_QUESTION_COUNT
    ):
        raise ValueError("profile inference official inventory counts changed")
    formatted = json.loads((root / "formatted_inputs.json").read_text(encoding="utf-8"))
    audits = json.loads((root / "audits.json").read_text(encoding="utf-8"))
    rankings = json.loads((root / "e5_rankings.json").read_text(encoding="utf-8"))
    if set(formatted) != {"target_queries", "profile_documents", "label_questions"} or (
        len(formatted["target_queries"]) != len(inputs["query_ids"])
        or len(formatted["profile_documents"]) != len(records)
        or len(formatted["label_questions"]) != len(inputs["label_question_ids"])
    ):
        raise ValueError("profile inference formatted-input inventory changed")
    expected_record_keys = {
        "record_id", "candidate_id", "source_query_id", "question", "response",
        "raw_membership_sha256", "formatted_sha256",
    }
    for index, record in enumerate(records):
        expected_formatted = format_profile_document(record["question"], record["response"])
        if set(record) != expected_record_keys or record["formatted_sha256"] != hashlib.sha256(expected_formatted.encode("utf-8")).hexdigest() or record["raw_membership_sha256"] != _length_framed((record["question"], record["response"])) or formatted["profile_documents"][index] != expected_formatted:
            raise ValueError("profile inference record correspondence changed")
    if set(audits) != set(inputs["candidate_ids"]):
        raise ValueError("profile inference audit inventory changed")
    for candidate_id in inputs["candidate_ids"]:
        audit = audits[candidate_id]
        truncation = [audit.get(key) for key in ("not_truncated", "partial_response_truncation", "full_response_truncation")]
        if not metadata.get("fixture") and (any(not isinstance(value, int) or value < 0 for value in truncation) or sum(truncation) != audit.get("usable")):
            raise ValueError("profile inference truncation audit arithmetic changed")
    if set(rankings) != set(inputs["query_ids"]) or any(len(values) != len(inputs["candidate_ids"]) or set(values) != set(inputs["candidate_ids"]) for values in rankings.values()):
        raise ValueError("profile inference E5 ranking inventory changed")
    if not metadata.get("fixture"):
        required_encoder = {
            "model_id": CONTEXTUAL_E5_SPEC.model_id,
            "requested_revision": CONTEXTUAL_E5_SPEC.revision,
            "resolved_model_revision": CONTEXTUAL_E5_SPEC.revision,
            "resolved_tokenizer_revision": CONTEXTUAL_E5_SPEC.revision,
            "required_container_image": ROCM_IMAGE,
            "dtype": CONTEXTUAL_E5_SPEC.dtype,
            "embedding_dimension": CONTEXTUAL_E5_SPEC.embedding_dimension,
            "batch_size": PROFILE_BATCH_SIZE,
            "seed": PROFILE_SEED,
        }
        for encoder_name in ("profile_encoder", "label_encoder"):
            encoder = metadata.get(encoder_name, {})
            if any(encoder.get(key) != value for key, value in required_encoder.items()) or PROFILE_DEVICE_ARCHITECTURE not in str(encoder.get("device_properties")):
                raise ValueError(f"profile inference pinned {encoder_name} identity changed")
    arrays = {name: np.load(root / name, allow_pickle=False) for name in ARRAY_FILES}
    expected_rows = {
        "query_embeddings.npy": len(inputs["query_ids"]),
        "record_embeddings.npy": len(records),
        "label_question_embeddings.npy": len(inputs["label_question_ids"]),
        "label_embeddings.npy": len(inputs["candidate_ids"]),
    }
    dimensions: set[int] = set()
    for name, matrix in arrays.items():
        if matrix.dtype.str != "<f4" or matrix.ndim != 2 or matrix.shape[0] != expected_rows[name] or (not metadata.get("fixture") and matrix.shape[1] != CONTEXTUAL_E5_SPEC.embedding_dimension) or not matrix.flags.c_contiguous or not np.isfinite(matrix).all():
            raise ValueError(f"profile inference array contract changed: {name}")
        if not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-5, rtol=0.0):
            raise ValueError(f"profile inference array is not normalized: {name}")
        dimensions.add(matrix.shape[1])
    if len(dimensions) != 1:
        raise ValueError("profile inference embedding dimensions differ")
    return ProfileInferenceArtifactResult(
        config=config,
        query_ids=tuple(inputs["query_ids"]),
        candidate_ids=tuple(inputs["candidate_ids"]),
        label_question_ids=tuple(inputs["label_question_ids"]),
        records=records,
        audits=audits,
        formatted_inputs=formatted,
        query_embeddings=np.asarray(arrays["query_embeddings.npy"]),
        record_embeddings=np.asarray(arrays["record_embeddings.npy"]),
        label_question_embeddings=np.asarray(arrays["label_question_embeddings.npy"]),
        label_embeddings=np.asarray(arrays["label_embeddings.npy"]),
        e5_rankings=rankings,
        metadata=metadata,
        manifest=manifest,
        root=root,
    )


def compare_profile_inference_repeats(
    first_dir: str | Path,
    second_dir: str | Path,
    second_execution_receipt_path: str | Path | None = None,
) -> dict[str, Any]:
    """Apply the approved Stage 1 repeat gate."""

    first = load_profile_inference_artifacts(first_dir)
    second = load_profile_inference_artifacts(second_dir)
    if first.root.resolve() == second.root.resolve():
        raise ValueError("inference repeats must use distinct artifact roots")
    if first.manifest["aggregate_sha256"] == second.manifest["aggregate_sha256"]:
        raise ValueError("inference repeats must have distinct aggregate digests")
    canonical = ("inputs.json", "records.json", "audits.json", "formatted_inputs.json", "e5_rankings.json")
    identical = all(first.manifest["files"][name] == second.manifest["files"][name] for name in canonical)
    first_metadata, first_is_repair = _repeat_metadata_with_explicit_lineage(
        first, allow_pinned_repair=True
    )
    second_metadata, second_is_repair = _repeat_metadata_with_explicit_lineage(
        second, allow_pinned_repair=False
    )
    execution_receipt = None
    execution_receipt_valid = False
    if second_execution_receipt_path is not None:
        execution_receipt = json.loads(
            Path(second_execution_receipt_path).read_text(encoding="utf-8")
        )
        expected_execution = {
            "artifact_type": "profile_inference_execution_receipt",
            "version": 1,
            "mode": "fresh_full_inference",
            "qrel_free": True,
            "embeddings_reused": False,
            "record_id_scheme": "profile-record-v2",
            "artifact_aggregate_sha256": second.manifest["aggregate_sha256"],
            "source_tree_sha256": second.metadata.get("source_tree_sha256"),
            "container_image": ROCM_IMAGE,
            "device_architecture": PROFILE_DEVICE_ARCHITECTURE,
            "model_id": CONTEXTUAL_E5_SPEC.model_id,
            "model_revision": CONTEXTUAL_E5_SPEC.revision,
            "query_count": len(second.query_ids),
            "candidate_count": len(second.candidate_ids),
            "record_count": len(second.records),
        }
        execution_receipt_valid = (
            isinstance(execution_receipt, dict)
            and set(execution_receipt) == FRESH_EXECUTION_RECEIPT_KEYS
            and execution_receipt == expected_execution
        )
    if first_is_repair and not second.metadata.get("fixture") and not execution_receipt_valid:
        raise ValueError("fresh second inference execution receipt is required")
    metadata_identical = first_metadata == second_metadata
    maximum = 0.0
    for name in ARRAY_FILES:
        left = np.load(first.root / name, allow_pickle=False)
        right = np.load(second.root / name, allow_pickle=False)
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"inference repeat shape/dtype differs for {name}")
        maximum = max(maximum, float(np.max(np.abs(left - right))))
    if not metadata_identical:
        raise ValueError("inference repeat metadata and token audits differ")
    if not identical or maximum > 1e-5:
        raise ValueError("inference repeat gate failed")
    return {
        "status": "passed",
        "canonical_inputs_identical": True,
        "maximum_absolute_embedding_difference": maximum,
        "tolerance": 1e-5,
        "first_is_pinned_id_repair": first_is_repair,
        "second_has_no_repair_lineage": not second_is_repair,
        "second_fresh_execution_receipt_valid": execution_receipt_valid,
        "second_execution_receipt": execution_receipt,
        "metadata_lineage_exceptions": (
            [
                "record_id_repair",
                "source_tree_sha256: source inference tree replaced by repair tree",
            ]
            if first_is_repair
            else []
        ),
        "first_aggregate_sha256": first.manifest["aggregate_sha256"],
        "second_aggregate_sha256": second.manifest["aggregate_sha256"],
        "integrity_checks": {
            "canonical_inputs_and_rankings_identical": True,
            "metadata_and_token_audits_identical": True,
            "embedding_tolerance": True,
        },
    }


__all__ = [
    "ProfileInferenceArtifactConfig",
    "ProfileInferenceArtifactResult",
    "build_profile_inference_execution_receipt",
    "compare_profile_inference_repeats",
    "load_profile_inference_artifacts",
    "ordered_source_questions",
    "repair_legacy_profile_inference_artifact",
    "run_profile_inference_artifact",
    "validate_profile_smoke_receipt",
    "write_profile_inference_artifacts",
]
