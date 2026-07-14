"""Qrel-free orchestration and durable artifacts for contextual E5 ranking."""

from __future__ import annotations

from collections.abc import Mapping
import csv
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.label_inference import ROCM_IMAGE
from trec_mlm.label_retrieval import (
    CONTEXTUAL_E5_SPEC,
    TASK_INSTRUCTION,
    ContextualE5Spec,
    format_document,
    format_query,
    rank_label_scores,
    rank_score_matrix,
)
from trec_mlm.query_knn import load_queries_tsv
from trec_mlm.runio import rankings_from_run, validate_trec_run, write_trec_run


_ARTIFACT_VERSION = 1
_CORE_FILES = (
    "config.json",
    "encoder_metadata.json",
    "inputs.json",
    "query_embeddings.npy",
    "label_embeddings.npy",
    "scores.npy",
    "predictions.tsv",
    "rankings.json",
    "run.trec",
)
_REQUIRED_RUNTIME_FIELDS = frozenset(
    {
        "backend",
        "model_id",
        "requested_revision",
        "resolved_model_revision",
        "resolved_tokenizer_revision",
        "license",
        "dtype",
        "pooling",
        "embedding_dimension",
        "input_count",
        "input_sha256",
        "transformers_version",
        "torch_version",
        "hip_version",
        "device",
        "device_properties",
        "required_container_image",
        "parameter_count",
        "peak_memory_allocated_bytes",
        "peak_memory_reserved_bytes",
        "batch_size",
        "seed",
        "padding_side",
        "maximum_observed_tokens",
        "maximum_untruncated_tokens",
        "truncated_input_count",
        "terminal_token_id",
        "terminal_token",
        "max_length",
        "encoding_seconds",
    }
)


@dataclass(frozen=True, slots=True)
class LabelExperimentConfig:
    """Python configuration for the single retained contextual E5 method."""

    query_path: str = "data/dev/llm_dev_data.tsv"
    candidate_path: str = "reports/trec_2026_discovery_profile_audit.json"
    encoder_key: str = CONTEXTUAL_E5_SPEC.key
    label_representation: str = "expert_capability"
    batch_size: int = 32
    seed: int = 2026
    run_id: str = "contextual_e5_labels"


@dataclass(frozen=True, slots=True)
class LabelExperimentResult:
    """One complete qrel-free contextual E5 ranking result."""

    config: LabelExperimentConfig
    encoder_spec: ContextualE5Spec
    query_ids: tuple[str, ...]
    query_texts: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    formatted_queries: tuple[str, ...]
    formatted_documents: tuple[str, ...]
    query_embeddings: NDArray[np.float32]
    label_embeddings: NDArray[np.float32]
    scores: NDArray[np.float32]
    rankings: dict[str, list[str]]
    encoder_metadata: dict[str, Any]
    summary: dict[str, Any]


def load_candidate_manifest(path: str | Path) -> tuple[str, ...]:
    """Load and sort one stable candidate universe."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    candidates: Any = payload.get("candidate_ids") if isinstance(payload, dict) else None
    if candidates is None and isinstance(payload, dict):
        plain = payload.get("plain")
        if isinstance(plain, dict):
            candidates = plain.get("candidate_ids")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{source}: candidate_ids must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in candidates):
        raise ValueError(f"{source}: candidate IDs must be non-empty strings")
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"{source}: candidate IDs must be unique")
    return tuple(sorted(candidates))


def run_label_experiment(
    config: LabelExperimentConfig,
) -> LabelExperimentResult:
    """Format, encode, and rank without accepting judgments or embeddings."""

    return _run_label_experiment(config)


def _run_label_experiment_with_embeddings(
    config: LabelExperimentConfig,
    *,
    query_embeddings: NDArray[np.floating[Any]],
    label_embeddings: NDArray[np.floating[Any]],
    encoder_metadata: Mapping[str, Any],
) -> LabelExperimentResult:
    """Private deterministic fixture boundary for tests and archival parity."""

    return _run_label_experiment(
        config,
        query_embeddings=query_embeddings,
        label_embeddings=label_embeddings,
        encoder_metadata=encoder_metadata,
    )


def _run_label_experiment(
    config: LabelExperimentConfig,
    *,
    query_embeddings: NDArray[np.floating[Any]] | None = None,
    label_embeddings: NDArray[np.floating[Any]] | None = None,
    encoder_metadata: Mapping[str, Any] | None = None,
) -> LabelExperimentResult:

    _validate_config(config)
    if (query_embeddings is None) != (label_embeddings is None):
        raise ValueError("query and label embeddings must be supplied together")
    queries = load_queries_tsv(config.query_path)
    query_ids = tuple(queries)
    query_texts = tuple(queries[query_id] for query_id in query_ids)
    candidate_ids = load_candidate_manifest(config.candidate_path)
    formatted_queries = tuple(format_query(text) for text in query_texts)
    formatted_documents = tuple(format_document(item) for item in candidate_ids)
    formatted_input_sha256 = _hash_texts(
        (*formatted_queries, *formatted_documents)
    )

    if query_embeddings is None or label_embeddings is None:
        from trec_mlm.label_inference import encode_label_inputs

        query_embeddings, label_embeddings, generated_metadata = encode_label_inputs(
            formatted_queries=formatted_queries,
            formatted_documents=formatted_documents,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        metadata = dict(generated_metadata)
    else:
        metadata = dict(encoder_metadata or {})
    _validate_encoder_metadata(
        metadata,
        config=config,
        expected_input_count=len(query_ids) + len(candidate_ids),
        expected_input_sha256=formatted_input_sha256,
    )
    scored = rank_label_scores(
        query_embeddings=query_embeddings,
        label_embeddings=label_embeddings,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
    )
    if (
        np.asarray(query_embeddings).shape[1] != CONTEXTUAL_E5_SPEC.embedding_dimension
        or np.asarray(label_embeddings).shape[1]
        != CONTEXTUAL_E5_SPEC.embedding_dimension
    ):
        raise ValueError("embedding dimensions differ from contextual E5")
    summary = {
        "status": "ranked",
        "qrel_free": True,
        "query_count": len(query_ids),
        "candidate_count": len(candidate_ids),
        "score_count": int(scored.scores.size),
        "encoder": CONTEXTUAL_E5_SPEC.key,
    }
    return LabelExperimentResult(
        config=config,
        encoder_spec=CONTEXTUAL_E5_SPEC,
        query_ids=query_ids,
        query_texts=query_texts,
        candidate_ids=candidate_ids,
        formatted_queries=formatted_queries,
        formatted_documents=formatted_documents,
        query_embeddings=np.asarray(query_embeddings, dtype=np.float32),
        label_embeddings=np.asarray(label_embeddings, dtype=np.float32),
        scores=scored.scores,
        rankings=scored.rankings,
        encoder_metadata=metadata,
        summary=summary,
    )


def write_label_experiment_artifacts(
    result: LabelExperimentResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Atomically write one complete, self-validating ranking artifact."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        _write_artifact_files(result, staging)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        name: str(destination / name) for name in (*_CORE_FILES, "validation.json")
    }


def _write_artifact_files(result: LabelExperimentResult, destination: Path) -> None:
    revision = _git_revision()
    worktree_status = _git_worktree_status()
    if revision is None or worktree_status is None:
        raise RuntimeError("could not resolve required git provenance")
    source_files = {
        "label_retrieval.py": Path(__file__).with_name("label_retrieval.py"),
        "label_inference.py": Path(__file__).with_name("label_inference.py"),
        "label_experiment.py": Path(__file__),
    }
    config_payload = {
        "artifact_type": "contextual_e5_label_ranking",
        "version": _ARTIFACT_VERSION,
        "qrel_free": True,
        **asdict(result.config),
        "encoder_spec": asdict(result.encoder_spec),
        "required_container_image": ROCM_IMAGE,
        "input_sha256": {
            result.config.query_path: _sha256(Path(result.config.query_path)),
            result.config.candidate_path: _sha256(Path(result.config.candidate_path)),
        },
        "formatted_input_sha256": _hash_texts(
            (*result.formatted_queries, *result.formatted_documents)
        ),
        "prompt_metadata": {
            "instruction": TASK_INSTRUCTION,
            "query_template": "Instruct: {instruction}\nQuery: {question}",
            "document_template": (
                "This LLM is most likely to correctly answer questions about "
                "{cleaned_label}."
            ),
            "label_cleaning": (
                "replace underscores and parentheses with spaces; collapse whitespace"
            ),
        },
        "code_revision": revision,
        "git_worktree_status": worktree_status,
        "source_code_sha256": {
            name: _sha256(path) for name, path in source_files.items()
        },
        "python_version": sys.version,
        "platform": platform.platform(),
    }
    _write_json(destination / "config.json", config_payload)
    _write_json(destination / "encoder_metadata.json", result.encoder_metadata)
    _write_json(
        destination / "inputs.json",
        {
            "artifact_type": "contextual_e5_label_inputs",
            "version": _ARTIFACT_VERSION,
            "queries": [
                {
                    "query_id": query_id,
                    "question": question,
                    "formatted_query": formatted,
                }
                for query_id, question, formatted in zip(
                    result.query_ids,
                    result.query_texts,
                    result.formatted_queries,
                    strict=True,
                )
            ],
            "candidates": [
                {"candidate_id": candidate_id, "formatted_document": formatted}
                for candidate_id, formatted in zip(
                    result.candidate_ids,
                    result.formatted_documents,
                    strict=True,
                )
            ],
        },
    )
    np.save(destination / "query_embeddings.npy", result.query_embeddings)
    np.save(destination / "label_embeddings.npy", result.label_embeddings)
    np.save(destination / "scores.npy", result.scores)
    _write_json(destination / "rankings.json", result.rankings)

    candidate_positions = {
        candidate_id: index for index, candidate_id in enumerate(result.candidate_ids)
    }
    with (destination / "predictions.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("query_id", "candidate_id", "score", "rank"))
        for query_index, query_id in enumerate(result.query_ids):
            for rank, candidate_id in enumerate(result.rankings[query_id], start=1):
                writer.writerow(
                    (
                        query_id,
                        candidate_id,
                        f"{float(result.scores[query_index, candidate_positions[candidate_id]]):.12g}",
                        rank,
                    )
                )
    write_trec_run(
        destination / "run.trec",
        result.rankings,
        run_id=result.config.run_id,
    )
    run_validation = validate_trec_run(
        destination / "run.trec",
        expected_query_ids=result.query_ids,
        expected_model_ids=result.candidate_ids,
    )
    validation = {
        "artifact_type": "contextual_e5_label_validation",
        "version": _ARTIFACT_VERSION,
        "status": "passed",
        "qrel_free": True,
        "query_count": len(result.query_ids),
        "candidate_count": len(result.candidate_ids),
        "score_shape": list(result.scores.shape),
        "score_count": int(result.scores.size),
        "embedding_shapes": {
            "query": list(result.query_embeddings.shape),
            "label": list(result.label_embeddings.shape),
        },
        "embeddings_finite": bool(
            np.isfinite(result.query_embeddings).all()
            and np.isfinite(result.label_embeddings).all()
        ),
        "run": run_validation,
        "artifact_sha256": {
            name: _sha256(destination / name) for name in _CORE_FILES
        },
    }
    _write_json(destination / "validation.json", validation)


def load_label_experiment_artifacts(
    artifact_dir: str | Path,
) -> LabelExperimentResult:
    """Load an artifact only after hashes and derived rankings validate."""

    return _load_label_experiment_artifacts(artifact_dir)


def _load_label_experiment_artifacts(
    artifact_dir: str | Path,
    *,
    allow_frozen_legacy_provenance: bool = False,
) -> LabelExperimentResult:
    """Load artifacts, optionally admitting the externally frozen legacy schema."""

    source = Path(artifact_dir)
    validation = _read_json(source / "validation.json")
    if (
        validation.get("artifact_type") != "contextual_e5_label_validation"
        or validation.get("version") != _ARTIFACT_VERSION
        or validation.get("status") != "passed"
        or validation.get("qrel_free") is not True
    ):
        raise ValueError("validation contract changed")
    hashes = validation.get("artifact_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(_CORE_FILES):
        raise ValueError("validation file inventory changed")
    for name, expected in hashes.items():
        if _sha256(source / name) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")

    config_payload = _read_json(source / "config.json")
    legacy_provenance = (
        allow_frozen_legacy_provenance
        and config_payload.get("required_container_image") is None
        and config_payload.get("container_image") == ROCM_IMAGE
    )
    if (
        config_payload.get("artifact_type") != "contextual_e5_label_ranking"
        or config_payload.get("version") != _ARTIFACT_VERSION
        or config_payload.get("qrel_free") is not True
        or (
            config_payload.get("required_container_image") != ROCM_IMAGE
            and not legacy_provenance
        )
        or not config_payload.get("code_revision")
        or not isinstance(config_payload.get("source_code_sha256"), dict)
    ):
        raise ValueError("resolved config contract changed")
    config = LabelExperimentConfig(
        **{
            field.name: config_payload[field.name]
            for field in fields(LabelExperimentConfig)
        }
    )
    _validate_config(config)
    if config_payload.get("encoder_spec") != asdict(CONTEXTUAL_E5_SPEC):
        raise ValueError("encoder specification changed")

    inputs = _read_json(source / "inputs.json")
    query_rows = inputs.get("queries")
    candidate_rows = inputs.get("candidates")
    if not isinstance(query_rows, list) or not isinstance(candidate_rows, list):
        raise ValueError("input inventory changed")
    query_ids = tuple(row["query_id"] for row in query_rows)
    query_texts = tuple(row["question"] for row in query_rows)
    candidate_ids = tuple(row["candidate_id"] for row in candidate_rows)
    formatted_queries = tuple(row["formatted_query"] for row in query_rows)
    formatted_documents = tuple(row["formatted_document"] for row in candidate_rows)
    if formatted_queries != tuple(format_query(text) for text in query_texts):
        raise ValueError("stored query formatting changed")
    if formatted_documents != tuple(format_document(item) for item in candidate_ids):
        raise ValueError("stored document formatting changed")
    if config_payload.get("formatted_input_sha256") != _hash_texts(
        (*formatted_queries, *formatted_documents)
    ):
        raise ValueError("formatted input hash changed")

    query_embeddings = np.load(source / "query_embeddings.npy", allow_pickle=False)
    label_embeddings = np.load(source / "label_embeddings.npy", allow_pickle=False)
    stored_scores = np.load(source / "scores.npy", allow_pickle=False)
    expected_query_shape = (len(query_ids), CONTEXTUAL_E5_SPEC.embedding_dimension)
    expected_label_shape = (
        len(candidate_ids),
        CONTEXTUAL_E5_SPEC.embedding_dimension,
    )
    if query_embeddings.shape != expected_query_shape:
        raise ValueError("query embedding shape changed")
    if label_embeddings.shape != expected_label_shape:
        raise ValueError("label embedding shape changed")
    if stored_scores.shape != (len(query_ids), len(candidate_ids)):
        raise ValueError("score shape changed")
    reconstructed = rank_label_scores(
        query_embeddings=query_embeddings,
        label_embeddings=label_embeddings,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
    )
    stored_scored = rank_score_matrix(
        scores=stored_scores,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
    )
    rankings = _read_json(source / "rankings.json")
    if rankings != stored_scored.rankings:
        raise ValueError("stored rankings disagree with stored scores")
    if rankings_from_run(source / "run.trec") != stored_scored.rankings:
        raise ValueError("TREC run disagrees with stored scores")
    run_validation = validate_trec_run(
        source / "run.trec",
        expected_query_ids=query_ids,
        expected_model_ids=candidate_ids,
    )
    if run_validation.get("row_count") != len(query_ids) * len(candidate_ids):
        raise ValueError("TREC run is incomplete")
    prediction_scores, prediction_rankings = _load_prediction_scores(
        source / "predictions.tsv",
        query_ids=query_ids,
        candidate_ids=candidate_ids,
    )
    if not np.allclose(prediction_scores, stored_scores, rtol=0.0, atol=1e-6):
        raise ValueError("prediction scores disagree with binary scores")
    if prediction_rankings != stored_scored.rankings:
        raise ValueError("prediction ranks disagree with stored scores")
    if not np.allclose(stored_scores, reconstructed.scores, rtol=0.0, atol=1e-6):
        raise ValueError("stored scores disagree with stored embeddings")
    metadata = _read_json(source / "encoder_metadata.json")
    metadata_to_validate = metadata
    if legacy_provenance:
        if (
            metadata.get("required_container_image") is not None
            or metadata.get("container_image") != ROCM_IMAGE
            or metadata.get("device") != "cuda_rocm"
        ):
            raise ValueError("legacy encoder provenance changed")
        metadata_to_validate = {
            **metadata,
            "required_container_image": metadata["container_image"],
            "device": "rocm_hip",
        }
    _validate_encoder_metadata(
        metadata_to_validate,
        config=config,
        expected_input_count=len(query_ids) + len(candidate_ids),
        expected_input_sha256=_hash_texts(
            (*formatted_queries, *formatted_documents)
        ),
    )
    return LabelExperimentResult(
        config=config,
        encoder_spec=CONTEXTUAL_E5_SPEC,
        query_ids=query_ids,
        query_texts=query_texts,
        candidate_ids=candidate_ids,
        formatted_queries=formatted_queries,
        formatted_documents=formatted_documents,
        query_embeddings=np.asarray(query_embeddings, dtype=np.float32),
        label_embeddings=np.asarray(label_embeddings, dtype=np.float32),
        scores=stored_scored.scores,
        rankings=stored_scored.rankings,
        encoder_metadata=metadata,
        summary={
            "status": "ranked",
            "qrel_free": True,
            "query_count": len(query_ids),
            "candidate_count": len(candidate_ids),
            "score_count": int(reconstructed.scores.size),
            "encoder": CONTEXTUAL_E5_SPEC.key,
        },
    )


def _load_prediction_scores(
    path: Path,
    *,
    query_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
) -> tuple[NDArray[np.float32], dict[str, list[str]]]:
    query_positions = {query_id: index for index, query_id in enumerate(query_ids)}
    candidate_positions = {
        candidate_id: index for index, candidate_id in enumerate(candidate_ids)
    }
    scores = np.full((len(query_ids), len(candidate_ids)), np.nan, dtype=np.float32)
    ranked_candidates: dict[str, list[str | None]] = {
        query_id: [None] * len(candidate_ids) for query_id in query_ids
    }
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["query_id", "candidate_id", "score", "rank"]:
            raise ValueError("prediction columns changed")
        for row in reader:
            query_id = row["query_id"]
            candidate_id = row["candidate_id"]
            key = (query_id, candidate_id)
            if (
                query_id not in query_positions
                or candidate_id not in candidate_positions
                or key in seen
            ):
                raise ValueError("prediction inventory changed")
            value = float(row["score"])
            if not np.isfinite(value):
                raise ValueError("prediction scores must be finite")
            try:
                rank = int(row["rank"])
            except ValueError as error:
                raise ValueError("prediction ranks must be integers") from error
            if (
                row["rank"] != str(rank)
                or not 1 <= rank <= len(candidate_ids)
                or ranked_candidates[query_id][rank - 1] is not None
            ):
                raise ValueError("prediction ranks are invalid or duplicated")
            seen.add(key)
            scores[query_positions[query_id], candidate_positions[candidate_id]] = value
            ranked_candidates[query_id][rank - 1] = candidate_id
    if len(seen) != scores.size or np.isnan(scores).any():
        raise ValueError("predictions are not a complete Cartesian product")
    if any(
        any(candidate is None for candidate in row)
        for row in ranked_candidates.values()
    ):
        raise ValueError("prediction ranks are incomplete")
    return scores, {
        query_id: [str(candidate) for candidate in row]
        for query_id, row in ranked_candidates.items()
    }


def _validate_config(config: LabelExperimentConfig) -> None:
    if config.encoder_key != CONTEXTUAL_E5_SPEC.key:
        raise ValueError("encoder_key must remain contextual E5")
    if config.label_representation != "expert_capability":
        raise ValueError("label_representation must remain expert_capability")
    if isinstance(config.batch_size, bool) or config.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not config.run_id or any(character.isspace() for character in config.run_id):
        raise ValueError("run_id must be a non-empty token")


def _validate_encoder_metadata(
    metadata: Mapping[str, Any],
    *,
    config: LabelExperimentConfig,
    expected_input_count: int,
    expected_input_sha256: str,
) -> None:
    if not _REQUIRED_RUNTIME_FIELDS.issubset(metadata):
        raise ValueError("encoder provenance is incomplete")
    exact = {
        "backend": "transformers",
        "model_id": CONTEXTUAL_E5_SPEC.model_id,
        "requested_revision": CONTEXTUAL_E5_SPEC.revision,
        "resolved_model_revision": CONTEXTUAL_E5_SPEC.revision,
        "resolved_tokenizer_revision": CONTEXTUAL_E5_SPEC.revision,
        "license": CONTEXTUAL_E5_SPEC.license,
        "dtype": CONTEXTUAL_E5_SPEC.dtype,
        "pooling": CONTEXTUAL_E5_SPEC.pooling,
        "embedding_dimension": CONTEXTUAL_E5_SPEC.embedding_dimension,
        "input_count": expected_input_count,
        "input_sha256": expected_input_sha256,
        "required_container_image": ROCM_IMAGE,
        "device": "rocm_hip",
        "batch_size": config.batch_size,
        "seed": config.seed,
        "padding_side": "left",
        "terminal_token_id": 2,
        "terminal_token": "</s>",
        "max_length": CONTEXTUAL_E5_SPEC.max_length,
    }
    for key, expected in exact.items():
        if metadata.get(key) != expected:
            raise ValueError(f"encoder provenance differs for {key}")


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_worktree_status() -> str | None:
    try:
        output = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return "clean" if not output else output


def _hash_texts(texts: tuple[str, ...]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON artifact: {path.name}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path.name}")
    return payload


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "LabelExperimentConfig",
    "LabelExperimentResult",
    "load_candidate_manifest",
    "load_label_experiment_artifacts",
    "run_label_experiment",
    "write_label_experiment_artifacts",
]
