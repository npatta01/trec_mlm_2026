"""Qrel-free unseen-query inference and TREC submission pipeline (issue #39).

One deterministic entry point ingests an organizer-style unseen-query release,
validates a frozen candidate inventory and frozen system configuration, runs one
pluggable qrel-free backend, and atomically emits a complete TREC run plus an
auditable execution receipt. No code path here accepts, opens, or names qrels;
inputs that look like qrels are rejected fail-closed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from typing import Any

import numpy as np

from trec_mlm.artifact_io import (
    length_framed_sha256,
    manifest_aggregate_sha256,
    manifest_entries,
    sha256_file,
    source_tree_sha256,
    write_pretty_json,
)
from trec_mlm.label_experiment import load_candidate_manifest
from trec_mlm.query_knn import load_queries_tsv
from trec_mlm.runio import validate_trec_run, write_trec_run


SUBMISSION_CONFIG_TYPE = "trec_submission_system"
SUBMISSION_CONFIG_VERSION = 1
SUBMISSION_RECEIPT_TYPE = "trec_submission_receipt"
SUBMISSION_RECEIPT_VERSION = 1
OFFICIAL_CANDIDATE_COUNT = 150
RUN_FILE_NAME = "run.trec"
RECEIPT_FILE_NAME = "receipt.json"
ARTIFACT_RESTORE_ISSUE = "issue #40"
_QREL_TOKEN = "qrel"

SUBMISSION_CONFIG_KEYS = frozenset(
    {
        "config_type",
        "version",
        "qrel_free",
        "fixture",
        "system_id",
        "run_id",
        "backend",
        "candidate_manifest",
        "backend_config",
    }
)
CANDIDATE_MANIFEST_KEYS = frozenset(
    {
        "candidate_count",
        "candidate_ids",
        "candidate_ids_sha256",
        "source_path",
        "source_sha256",
    }
)
PROFILE_SCORING_BACKEND_KEYS = frozenset({"construction_artifact"})
PROFILE_SCORING_CONSTRUCTION_KEYS = frozenset({"path", "expected_aggregate_sha256"})


@dataclass(frozen=True, slots=True)
class SubmissionConfig:
    """A validated frozen system configuration bound to its file hash."""

    path: str
    sha256: str
    fixture: bool
    system_id: str
    run_id: str
    backend: str
    candidate_count: int
    candidate_ids: tuple[str, ...]
    candidate_ids_sha256: str
    candidate_source_path: str
    candidate_source_sha256: str
    backend_config: dict[str, Any]


@dataclass(frozen=True, slots=True)
class BackendRequest:
    """Everything a submission backend may consume. Qrels are unrepresentable."""

    queries: dict[str, str]
    candidate_ids: tuple[str, ...]
    backend_config: dict[str, Any]
    artifact_path: str | None
    fixture: bool
    base_dir: Path


@dataclass(frozen=True, slots=True)
class BackendRun:
    """Complete per-query rankings plus the backend's audit receipt."""

    rankings: dict[str, list[str]]
    receipt: dict[str, Any]


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_token(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and not any(character.isspace() for character in value)
    )


def reject_qrel_material(value: Any, *, context: str) -> None:
    """Fail closed on any qrel-named key, value, or path anywhere in the input."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{context}: configuration keys must be strings")
            if key != "qrel_free" and _QREL_TOKEN in key.lower():
                raise ValueError(f"{context}: qrel material is forbidden: {key!r}")
            reject_qrel_material(item, context=context)
        return
    if isinstance(value, (list, tuple)):
        for item in value:
            reject_qrel_material(item, context=context)
        return
    if isinstance(value, str) and _QREL_TOKEN in value.lower():
        raise ValueError(f"{context}: qrel material is forbidden: {value!r}")


def _looks_like_qrels(text: str) -> bool:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    for line in lines:
        fields = line.split()
        if len(fields) != 4 or fields[1] != "0":
            return False
        try:
            int(fields[3])
        except ValueError:
            return False
    return True


def _load_topics_jsonl(path: Path, text: str) -> dict[str, str]:
    queries: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{path}:{line_number}: topics line is not valid JSON"
            ) from error
        if not isinstance(payload, dict) or set(payload) != {"topic_id", "question"}:
            raise ValueError(
                f"{path}:{line_number}: expected exactly topic_id and question"
            )
        topic_id = payload["topic_id"]
        question = payload["question"]
        if not isinstance(topic_id, str) or not isinstance(question, str):
            raise ValueError(
                f"{path}:{line_number}: topic_id and question must be strings"
            )
        topic_id = topic_id.strip()
        question = question.strip()
        if not topic_id or not question:
            raise ValueError(
                f"{path}:{line_number}: topic ID and question must be non-empty"
            )
        if topic_id in queries:
            raise ValueError(f"{path}:{line_number}: duplicate topic ID {topic_id!r}")
        queries[topic_id] = question
    if not queries:
        raise ValueError(f"{path}: topics file is empty")
    return queries


def load_unseen_queries(path: str | Path) -> dict[str, str]:
    """Load an unseen-query release as ``query_id -> text``, refusing qrels.

    Two organizer shapes are accepted: the development-style unheaded
    ``query_id<TAB>query_text`` TSV and the official test-topics JSONL with
    exactly ``topic_id`` and ``question`` keys per line.
    """

    source = Path(path)
    reject_qrel_material(str(source), context="unseen query path")
    text = source.read_text(encoding="utf-8")
    if _looks_like_qrels(text):
        raise ValueError(
            f"{source}: input is shaped like a qrels file; "
            "this pipeline never reads relevance judgments"
        )
    suffix = source.suffix.lower()
    if suffix == ".tsv":
        queries = dict(load_queries_tsv(source))
    elif suffix == ".jsonl":
        queries = _load_topics_jsonl(source, text)
    else:
        raise ValueError(
            f"{source}: unsupported query format {suffix!r}; expected .tsv or .jsonl"
        )
    for query_id in queries:
        if not _is_token(query_id):
            raise ValueError(
                f"{source}: query ID {query_id!r} is not a whitespace-free token"
            )
    return queries


def _validate_candidate_manifest(
    payload: Any, *, fixture: bool, base_dir: Path
) -> tuple[int, tuple[str, ...], str, str, str]:
    if not isinstance(payload, dict) or set(payload) != CANDIDATE_MANIFEST_KEYS:
        raise ValueError("candidate manifest schema changed")
    candidate_ids = payload["candidate_ids"]
    if (
        not isinstance(candidate_ids, list)
        or not candidate_ids
        or any(not _is_token(item) for item in candidate_ids)
    ):
        raise ValueError("candidate IDs must be non-empty whitespace-free tokens")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    if candidate_ids != sorted(candidate_ids):
        raise ValueError("candidate IDs must be sorted")
    count = payload["candidate_count"]
    if not isinstance(count, int) or isinstance(count, bool) or count != len(candidate_ids):
        raise ValueError("candidate_count disagrees with the candidate ID list")
    if not fixture and count != OFFICIAL_CANDIDATE_COUNT:
        raise ValueError(
            f"production candidate inventory must contain exactly "
            f"{OFFICIAL_CANDIDATE_COUNT} IDs"
        )
    ids_sha = payload["candidate_ids_sha256"]
    if length_framed_sha256(candidate_ids) != ids_sha:
        raise ValueError("candidate inventory hash disagrees with the ID list")
    source_path = payload["source_path"]
    source_sha = payload["source_sha256"]
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("candidate source_path must be a non-empty path")
    if not _is_sha256(source_sha):
        raise ValueError("candidate source_sha256 must be a SHA-256 hex digest")
    source = Path(source_path)
    if not source.is_absolute():
        source = base_dir / source
    if not source.is_file():
        raise FileNotFoundError(
            f"candidate manifest source is missing: {source}"
        )
    if sha256_file(source) != source_sha:
        raise ValueError("candidate manifest source file hash changed")
    if load_candidate_manifest(source) != tuple(candidate_ids):
        raise ValueError("candidate manifest source disagrees with frozen IDs")
    return count, tuple(candidate_ids), ids_sha, source_path, source_sha


def _validate_profile_scoring_backend_config(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != PROFILE_SCORING_BACKEND_KEYS:
        raise ValueError("profile_scoring backend configuration schema changed")
    construction = payload["construction_artifact"]
    if (
        not isinstance(construction, dict)
        or set(construction) != PROFILE_SCORING_CONSTRUCTION_KEYS
    ):
        raise ValueError("profile_scoring construction binding schema changed")
    if not isinstance(construction["path"], str) or not construction["path"]:
        raise ValueError("construction artifact path must be a non-empty path")
    if not _is_sha256(construction["expected_aggregate_sha256"]):
        raise ValueError(
            "construction expected_aggregate_sha256 must be a SHA-256 hex digest"
        )


BACKEND_CONFIG_VALIDATORS: dict[str, Callable[[Any], None]] = {
    "profile_scoring": _validate_profile_scoring_backend_config,
}


def load_submission_config(
    path: str | Path, *, base_dir: str | Path | None = None
) -> SubmissionConfig:
    """Load and validate one frozen system configuration fail-closed."""

    source = Path(path)
    root = Path(base_dir) if base_dir is not None else Path.cwd()
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or set(payload) != SUBMISSION_CONFIG_KEYS:
        raise ValueError("submission configuration schema changed")
    if payload["config_type"] != SUBMISSION_CONFIG_TYPE:
        raise ValueError("submission configuration type changed")
    if payload["version"] != SUBMISSION_CONFIG_VERSION:
        raise ValueError("submission configuration version changed")
    if payload["qrel_free"] is not True:
        raise ValueError("submission configuration must declare qrel_free true")
    fixture = payload["fixture"]
    if not isinstance(fixture, bool):
        raise ValueError("submission configuration fixture flag must be boolean")
    reject_qrel_material(payload, context=f"submission configuration {source}")
    if not _is_token(payload["system_id"]):
        raise ValueError("system_id must be a non-empty whitespace-free token")
    if not _is_token(payload["run_id"]):
        raise ValueError("run_id must be a non-empty whitespace-free token")
    backend = payload["backend"]
    if backend not in BACKEND_CONFIG_VALIDATORS:
        raise ValueError(
            f"unknown submission backend {backend!r}; "
            f"choose from {sorted(BACKEND_CONFIG_VALIDATORS)!r}"
        )
    BACKEND_CONFIG_VALIDATORS[backend](payload["backend_config"])
    count, candidate_ids, ids_sha, source_path, source_sha = (
        _validate_candidate_manifest(
            payload["candidate_manifest"], fixture=fixture, base_dir=root
        )
    )
    return SubmissionConfig(
        path=str(path),
        sha256=hashlib.sha256(raw).hexdigest(),
        fixture=fixture,
        system_id=payload["system_id"],
        run_id=payload["run_id"],
        backend=backend,
        candidate_count=count,
        candidate_ids=candidate_ids,
        candidate_ids_sha256=ids_sha,
        candidate_source_path=source_path,
        candidate_source_sha256=source_sha,
        backend_config=payload["backend_config"],
    )


def _artifact_missing(role: str, detail: str) -> FileNotFoundError:
    return FileNotFoundError(
        f"{role} is missing: {detail}; restore or produce it first "
        f"(private artifact custody is tracked in {ARTIFACT_RESTORE_ISSUE}); "
        "this pipeline never substitutes a fallback ranking"
    )


def _profile_scoring_backend(request: BackendRequest) -> BackendRun:
    """Score unseen queries against frozen candidate-profile centroids.

    The preregistered qrel-free profile score is the maximum cosine between the
    query embedding and each candidate's frozen centroids. Query embeddings come
    from a validated Stage 1 profile inference artifact covering exactly the
    unseen queries; centroids come from the hash-pinned frozen construction.
    """

    from trec_mlm.profile_experiment import load_profile_artifacts
    from trec_mlm.profile_inference_artifact import load_profile_inference_artifacts
    from trec_mlm.profile_retrieval import CandidateProfile, score_profiles

    if request.artifact_path is None:
        raise _artifact_missing(
            "profile inference artifact",
            "no --artifact directory was provided; the unseen-query inference "
            "artifact is produced by the release-day profile-infer stage",
        )
    inference_dir = Path(request.artifact_path)
    if not inference_dir.is_absolute():
        inference_dir = request.base_dir / inference_dir
    if not (inference_dir / "manifest.json").is_file():
        raise _artifact_missing("profile inference artifact", str(inference_dir))
    construction_binding = request.backend_config["construction_artifact"]
    construction_dir = Path(construction_binding["path"])
    if not construction_dir.is_absolute():
        construction_dir = request.base_dir / construction_dir
    if not (construction_dir / "manifest.json").is_file():
        raise _artifact_missing(
            "frozen profile construction artifact", str(construction_dir)
        )

    inference = load_profile_inference_artifacts(inference_dir)
    construction = load_profile_artifacts(construction_dir)
    expected_aggregate = construction_binding["expected_aggregate_sha256"]
    if construction.manifest["aggregate_sha256"] != expected_aggregate:
        raise ValueError(
            "frozen construction aggregate hash differs from the submission "
            "configuration"
        )
    construction_metadata = json.loads(
        (construction.root / "metadata.json").read_text(encoding="utf-8")
    )
    if not request.fixture and (
        inference.metadata.get("fixture") or construction_metadata.get("fixture")
    ):
        raise ValueError("production submission cannot consume fixture artifacts")
    if inference.candidate_ids != request.candidate_ids:
        raise ValueError(
            "inference artifact candidate inventory differs from the frozen manifest"
        )
    if construction.candidate_ids != request.candidate_ids:
        raise ValueError(
            "construction artifact candidate inventory differs from the frozen "
            "manifest"
        )
    if set(inference.query_ids) != set(request.queries):
        missing = sorted(set(request.queries) - set(inference.query_ids))
        extra = sorted(set(inference.query_ids) - set(request.queries))
        raise ValueError(
            f"inference artifact query inventory differs from the release: "
            f"missing={missing!r}, extra={extra!r}"
        )

    profile_payload = json.loads(
        (construction.root / "profiles.json").read_text(encoding="utf-8")
    )
    profiles: dict[str, CandidateProfile | None] = {}
    unavailable = 0
    for candidate_id in request.candidate_ids:
        entry = profile_payload[candidate_id]
        if entry.get("available") is False:
            profiles[candidate_id] = None
            unavailable += 1
            continue
        profiles[candidate_id] = CandidateProfile(
            candidate_id=candidate_id,
            requested_clusters=int(entry["requested_clusters"]),
            centroids=np.asarray(entry["centroids"], dtype=np.float32),
            clusters=(),
            converged=bool(entry["converged"]),
            iteration_count=int(entry["iteration_count"]),
        )

    ordered_query_ids = tuple(sorted(request.queries))
    positions = {query_id: index for index, query_id in enumerate(inference.query_ids)}
    rows = np.asarray(
        inference.query_embeddings[
            np.asarray([positions[item] for item in ordered_query_ids], dtype=np.intp)
        ],
        dtype=np.float32,
    )
    scored = score_profiles(
        query_ids=ordered_query_ids,
        candidate_ids=request.candidate_ids,
        query_embeddings=rows,
        profiles=profiles,
    )
    if not np.isfinite(scored.scores).all():
        raise ValueError("profile scoring produced non-finite scores")
    return BackendRun(
        rankings={query_id: list(scored.rankings[query_id]) for query_id in ordered_query_ids},
        receipt={
            "backend": "profile_scoring",
            "qrel_free": True,
            "scoring": {"method": "max_centroid_cosine", "top_n": 1},
            "inference_artifact": {
                "path": request.artifact_path,
                "aggregate_sha256": inference.manifest["aggregate_sha256"],
                "query_count": len(inference.query_ids),
                "candidate_count": len(inference.candidate_ids),
                "record_count": len(inference.records),
            },
            "construction_artifact": {
                "path": construction_binding["path"],
                "aggregate_sha256": construction.manifest["aggregate_sha256"],
            },
            "unavailable_profile_count": unavailable,
        },
    )


SUBMISSION_BACKENDS: dict[str, Callable[[BackendRequest], BackendRun]] = {
    "profile_scoring": _profile_scoring_backend,
}


def _validate_complete_rankings(
    rankings: Mapping[str, Sequence[str]],
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> None:
    if set(rankings) != set(query_ids):
        raise ValueError("backend ranking query inventory mismatch")
    expected = set(candidate_ids)
    for query_id in query_ids:
        values = list(rankings[query_id])
        if len(values) != len(set(values)):
            raise ValueError(f"backend ranking for {query_id!r} contains duplicates")
        if len(values) != len(candidate_ids) or set(values) != expected:
            raise ValueError(
                f"backend ranking for {query_id!r} does not cover the frozen "
                "candidate inventory exactly"
            )


def run_submission(
    *,
    config_path: str | Path,
    queries_path: str | Path,
    output_dir: str | Path,
    artifact_path: str | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Produce one complete, validated, receipted TREC run atomically."""

    root = Path(base_dir) if base_dir is not None else Path.cwd()
    for name, value in (
        ("configuration path", str(config_path)),
        ("output path", str(output_dir)),
        ("artifact path", artifact_path or ""),
    ):
        reject_qrel_material(value, context=name)
    config = load_submission_config(config_path, base_dir=root)
    queries = load_unseen_queries(queries_path)
    queries_file = Path(queries_path)

    backend = SUBMISSION_BACKENDS[config.backend]
    result = backend(
        BackendRequest(
            queries=queries,
            candidate_ids=config.candidate_ids,
            backend_config=config.backend_config,
            artifact_path=artifact_path,
            fixture=config.fixture,
            base_dir=root,
        )
    )
    _validate_complete_rankings(result.rankings, tuple(queries), config.candidate_ids)

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"output directory already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        run_path = staging / RUN_FILE_NAME
        write_trec_run(run_path, result.rankings, run_id=config.run_id)
        validation = validate_trec_run(
            run_path,
            expected_query_ids=tuple(queries),
            expected_model_ids=config.candidate_ids,
        )
        if (
            validation["run_id"] != config.run_id
            or validation["query_count"] != len(queries)
            or validation["candidate_count"] != config.candidate_count
        ):
            raise ValueError("emitted run disagrees with the frozen configuration")
        entries = manifest_entries(staging)
        receipt = {
            "artifact_type": SUBMISSION_RECEIPT_TYPE,
            "version": SUBMISSION_RECEIPT_VERSION,
            "qrel_free": True,
            "fixture": config.fixture,
            "system_id": config.system_id,
            "backend": config.backend,
            "run_id": config.run_id,
            "config": {"path": config.path, "sha256": config.sha256},
            "queries": {
                "path": str(queries_path),
                "format": queries_file.suffix.lower().lstrip("."),
                "sha256": sha256_file(queries_file),
                "query_count": len(queries),
            },
            "candidate_manifest": {
                "candidate_count": config.candidate_count,
                "candidate_ids_sha256": config.candidate_ids_sha256,
                "source_path": config.candidate_source_path,
                "source_sha256": config.candidate_source_sha256,
            },
            "backend_receipt": result.receipt,
            "run": {
                "file": RUN_FILE_NAME,
                "sha256": entries[RUN_FILE_NAME]["sha256"],
                "byte_size": entries[RUN_FILE_NAME]["byte_size"],
                "row_count": validation["row_count"],
                "query_count": validation["query_count"],
                "candidate_count": validation["candidate_count"],
            },
            "files": entries,
            "aggregate_sha256": manifest_aggregate_sha256(entries),
            "tool_versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "trec_mlm_source_tree_sha256": source_tree_sha256(
                    Path(__file__).resolve().parent
                ),
            },
        }
        write_pretty_json(staging / RECEIPT_FILE_NAME, receipt)
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return receipt


__all__ = [
    "BACKEND_CONFIG_VALIDATORS",
    "BackendRequest",
    "BackendRun",
    "OFFICIAL_CANDIDATE_COUNT",
    "SUBMISSION_BACKENDS",
    "SUBMISSION_CONFIG_KEYS",
    "SUBMISSION_CONFIG_TYPE",
    "SUBMISSION_CONFIG_VERSION",
    "SubmissionConfig",
    "load_submission_config",
    "load_unseen_queries",
    "reject_qrel_material",
    "run_submission",
]
