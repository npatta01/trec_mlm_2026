"""Qrel-free execution and durable artifacts for the Stage 2 GroupRank pilot."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any, NoReturn

from trec_mlm.grouprank import (
    GROUPING_SEEDS,
    GroupRankRequest,
    aggregate_grouprank_scores,
    build_group_requests,
    grouprank_score_schema,
    parse_grouprank_scores,
)
from trec_mlm.grouprank_inference import (
    ATTENTION_BACKEND,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    ROCM_CONTAINER_IMAGE,
    finalize_grouprank_runtime,
)
from trec_mlm.llm_ranker_pilot import preflight_pilot_inputs


_MODEL_PARAMETER_COUNT = 7_615_616_512
_REQUEST_COUNT = 48
_QUESTION_COUNT = 12
_CANDIDATE_COUNT = 20
_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")

_RAW_OUTPUT_KEYS = frozenset(
    {
        "query_id",
        "repetition",
        "group",
        "ordered_candidate_ids",
        "handle_to_candidate",
        "raw_text",
        "prompt_token_count",
        "output_token_count",
        "finish_reason",
        "sampling_seed",
    }
)
_RUNTIME_REQUIRED_KEYS = frozenset(
    {
        "model_id",
        "revision",
        "resolved_revision",
        "dtype",
        "license",
        "parameter_count",
        "max_model_len",
        "gpu_memory_utilization",
        "attention_backend",
        "vllm_version",
        "torch_version",
        "hip_version",
        "device_properties",
        "code_revision",
        "worktree_status",
        "query_path",
        "candidate_path",
        "container_image",
    }
)
_RUNTIME_OPTIONAL_KEYS = frozenset(
    {
        "elapsed_seconds",
        "inference_stats",
    }
)
_CONFIG_KEYS = frozenset(
    {
        "version",
        "qrel_free_phase",
        "method",
        "stage",
        "seed",
        "grouping_seeds",
        "request_count",
        "input_sha256",
        "selection_sha256",
        "prompt_sha256",
        "schema_sha256",
        "model_id",
        "model_revision",
        "resolved_revision",
        "dtype",
        "license",
        "parameter_count",
        "max_model_len",
        "gpu_memory_utilization",
        "attention_backend",
        "code_revision",
        "worktree_status",
        "query_path",
        "candidate_path",
        "container_image",
        "runtime_versions",
    }
)
_SUCCESS_CORE_FILES = frozenset(
    {
        "config.json",
        "selection.json",
        "prompts.jsonl",
        "raw_outputs.jsonl",
        "scores.tsv",
        "rankings.json",
        "runtime.json",
    }
)
_FAILURE_CORE_FILES = frozenset(
    {
        "config.json",
        "selection.json",
        "prompts.jsonl",
        "raw_outputs.jsonl",
        "runtime.json",
    }
)
_SUCCESS_INVENTORY = (*sorted(_SUCCESS_CORE_FILES), "ranking_receipt.json")


@dataclass(frozen=True)
class GroupRankPhaseAResult:
    """One complete qrel-free GroupRank result."""

    selection: Mapping[str, Any]
    prompts: tuple[Mapping[str, Any], ...]
    raw_outputs: tuple[Mapping[str, Any], ...]
    candidate_scores: Mapping[str, Mapping[str, float]]
    rankings: Mapping[str, Sequence[str]]
    config: Mapping[str, Any]
    runtime: Mapping[str, Any]


@dataclass(frozen=True)
class GroupRankFailureResult:
    """Complete raw GroupRank evidence for an unscorable attempt."""

    selection: Mapping[str, Any]
    prompts: tuple[Mapping[str, Any], ...]
    raw_outputs: tuple[Mapping[str, Any], ...]
    failure: Mapping[str, Any]
    config: Mapping[str, Any]
    runtime: Mapping[str, Any]


class GroupRankParseFailure(ValueError):
    """Raised only after an atomic failure artifact has been published."""

    def __init__(self, message: str, *, result: GroupRankFailureResult):
        super().__init__(message)
        self.result = result


class _GroupRankValidationError(ValueError):
    def __init__(self, message: str, *, failures: Sequence[Mapping[str, Any]]):
        super().__init__(message)
        self.failures = tuple(dict(failure) for failure in failures)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(
                row,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_sha256(prompts: Sequence[Mapping[str, Any]]) -> str:
    schemas = tuple(
        {
            "query_id": prompt["query_id"],
            "repetition": prompt["repetition"],
            "group": prompt["group"],
            "score_schema": prompt["score_schema"],
        }
        for prompt in prompts
    )
    return _sha256_bytes(_jsonl_bytes(schemas))


def _write_bytes(path: Path, payload: bytes, *, fsync: bool) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        if fsync:
            os.fsync(handle.fileno())


def write_json(path: Path, payload: Any, *, fsync: bool = False) -> None:
    """Write deterministic, newline-terminated JSON."""

    _write_bytes(path, _json_bytes(payload), fsync=fsync)


def write_jsonl(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    fsync: bool,
) -> None:
    """Write deterministic JSONL and optionally force it to stable storage."""

    _write_bytes(path, _jsonl_bytes(rows), fsync=fsync)


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def request_to_record(request: GroupRankRequest) -> dict[str, Any]:
    """Serialize one immutable request, including its exact score schema."""

    return {
        "query_id": request.query_id,
        "repetition": request.repetition,
        "group": request.group,
        "ordered_candidate_ids": list(request.ordered_candidate_ids),
        "handle_to_candidate": dict(request.handle_to_candidate),
        "messages": [dict(message) for message in request.messages],
        "sampling_seed": request.sampling_seed,
        "score_schema": grouprank_score_schema(tuple(request.handle_to_candidate)),
    }


def _validate_inference_stats(payload: object) -> dict[str, int]:
    expected = {
        "request_count",
        "prompt_token_count",
        "output_token_count",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError("GroupRank runtime inference stats schema changed")
    result: dict[str, int] = {}
    for field, value in payload.items():
        if type(value) is not int or value < 0:
            raise ValueError(
                f"GroupRank runtime inference stats {field} must be an integer"
            )
        result[field] = value
    return result


def _validate_runtime_metadata(
    runtime_metadata: Mapping[str, Any],
    *,
    finalized: bool = False,
) -> dict[str, Any]:
    if not isinstance(runtime_metadata, Mapping):
        raise ValueError("GroupRank runtime metadata must be an object")
    fields = set(runtime_metadata)
    if any(
        "qrel" in str(field).lower() or "judgment" in str(field).lower()
        for field in fields
    ):
        raise ValueError("GroupRank Phase A does not accept qrel or judgment metadata")
    allowed = _RUNTIME_REQUIRED_KEYS | _RUNTIME_OPTIONAL_KEYS
    if fields - allowed or not _RUNTIME_REQUIRED_KEYS.issubset(fields):
        raise ValueError("GroupRank runtime metadata schema changed")

    runtime = dict(runtime_metadata)
    for field, label in (
        ("model_id", "model ID"),
        ("revision", "model revision"),
        ("resolved_revision", "resolved model revision"),
        ("dtype", "dtype"),
        ("license", "license"),
        ("attention_backend", "attention backend"),
        ("vllm_version", "vLLM version"),
        ("torch_version", "Torch version"),
        ("hip_version", "HIP version"),
        ("device_properties", "device properties"),
        ("code_revision", "code revision"),
        ("worktree_status", "worktree status"),
        ("query_path", "query path"),
        ("candidate_path", "candidate path"),
        ("container_image", "container image"),
    ):
        value = runtime[field]
        if type(value) is not str or not value.strip():
            raise ValueError(f"GroupRank runtime {label} must be a non-empty string")
    for field, expected in (
        ("model_id", MODEL_ID),
        ("revision", MODEL_REVISION),
        ("resolved_revision", MODEL_REVISION),
        ("dtype", MODEL_DTYPE),
        ("license", MODEL_LICENSE),
        ("parameter_count", _MODEL_PARAMETER_COUNT),
        ("max_model_len", MAX_MODEL_LEN),
        ("gpu_memory_utilization", GPU_MEMORY_UTILIZATION),
        ("attention_backend", ATTENTION_BACKEND),
        ("container_image", ROCM_CONTAINER_IMAGE),
    ):
        if runtime[field] != expected:
            labels = {
                "revision": "model revision",
                "resolved_revision": "resolved model revision",
                "parameter_count": "parameter count",
                "max_model_len": "maximum model length",
                "gpu_memory_utilization": "GPU memory utilization",
                "attention_backend": "attention backend",
            }
            raise ValueError(f"GroupRank runtime {labels.get(field, field)} changed")
    if _COMMIT_SHA_RE.fullmatch(runtime["code_revision"]) is None:
        raise ValueError(
            "GroupRank runtime code revision must be a full 40-hex commit SHA"
        )
    if runtime["worktree_status"] != "clean":
        raise ValueError("GroupRank runtime worktree status must be exactly 'clean'")
    if "elapsed_seconds" in runtime:
        elapsed = runtime["elapsed_seconds"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed <= 0
        ):
            raise ValueError(
                "GroupRank runtime elapsed seconds must be positive finite"
            )
    if "inference_stats" in runtime:
        runtime["inference_stats"] = _validate_inference_stats(
            runtime["inference_stats"]
        )
    if finalized and not {"elapsed_seconds", "inference_stats"}.issubset(runtime):
        raise ValueError("GroupRank finalized runtime metadata is incomplete")
    if finalized and runtime["inference_stats"]["request_count"] != _REQUEST_COUNT:
        raise ValueError("GroupRank runtime must record exactly 48 requests")
    return runtime


def build_phase_a_config(
    selection: Mapping[str, Any],
    prompts: Sequence[Mapping[str, Any]],
    runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Freeze selection, prompt, schema, model, source, and runtime provenance."""

    return {
        "version": 1,
        "qrel_free_phase": True,
        "method": "grouprank",
        "stage": selection["stage"],
        "seed": 2026,
        "grouping_seeds": list(GROUPING_SEEDS),
        "request_count": len(prompts),
        "input_sha256": dict(selection["input_sha256"]),
        "selection_sha256": _sha256_bytes(_json_bytes(selection)),
        "prompt_sha256": _sha256_bytes(_jsonl_bytes(prompts)),
        "schema_sha256": _schema_sha256(prompts),
        "model_id": runtime_metadata["model_id"],
        "model_revision": runtime_metadata["revision"],
        "resolved_revision": runtime_metadata["resolved_revision"],
        "dtype": runtime_metadata["dtype"],
        "license": runtime_metadata["license"],
        "parameter_count": runtime_metadata["parameter_count"],
        "max_model_len": runtime_metadata["max_model_len"],
        "gpu_memory_utilization": runtime_metadata["gpu_memory_utilization"],
        "attention_backend": runtime_metadata["attention_backend"],
        "code_revision": runtime_metadata["code_revision"],
        "worktree_status": runtime_metadata["worktree_status"],
        "query_path": runtime_metadata["query_path"],
        "candidate_path": runtime_metadata["candidate_path"],
        "container_image": runtime_metadata["container_image"],
        "runtime_versions": {
            "vllm": runtime_metadata["vllm_version"],
            "torch": runtime_metadata["torch_version"],
            "hip": runtime_metadata["hip_version"],
        },
    }


def _record_failure(
    *,
    index: int,
    request: GroupRankRequest | None,
    message: str,
) -> dict[str, Any]:
    return {
        "request_index": index,
        "query_id": None if request is None else request.query_id,
        "repetition": None if request is None else request.repetition,
        "group": None if request is None else request.group,
        "error_message": message,
    }


def _validate_raw_record(
    record: Mapping[str, Any],
    request: GroupRankRequest,
) -> None:
    if set(record) != _RAW_OUTPUT_KEYS:
        raise ValueError("GroupRank raw output schema changed")
    for field, expected in (
        ("query_id", request.query_id),
        ("repetition", request.repetition),
        ("group", request.group),
        ("ordered_candidate_ids", list(request.ordered_candidate_ids)),
        ("handle_to_candidate", dict(request.handle_to_candidate)),
        ("sampling_seed", request.sampling_seed),
    ):
        if record[field] != expected:
            raise ValueError(f"GroupRank raw output {field.replace('_', ' ')} changed")
    if type(record["raw_text"]) is not str:
        raise ValueError("GroupRank raw output text is missing")
    for field in ("prompt_token_count", "output_token_count"):
        if type(record[field]) is not int or record[field] < 0:
            raise ValueError(f"GroupRank raw output {field.replace('_', ' ')} changed")
    finish_reason = record["finish_reason"]
    if type(finish_reason) is not str or not finish_reason:
        raise ValueError("GroupRank raw output finish reason is missing")
    if finish_reason == "length":
        raise ValueError("GroupRank output was length-truncated")


def parse_and_aggregate(
    requests: Sequence[GroupRankRequest],
    raw_outputs: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    """Validate all raw records before allowing aggregate ranking construction."""

    failures: list[dict[str, Any]] = []
    parsed_scores: list[Mapping[str, int]] = []
    if len(raw_outputs) != len(requests):
        failures.append(
            _record_failure(
                index=min(len(raw_outputs), len(requests)),
                request=None,
                message=(
                    "GroupRank inference returned the wrong number of raw outputs"
                ),
            )
        )

    for index, request in enumerate(requests):
        if index >= len(raw_outputs):
            continue
        raw_record = raw_outputs[index]
        if not isinstance(raw_record, Mapping):
            failures.append(
                _record_failure(
                    index=index,
                    request=request,
                    message="GroupRank raw outputs must be objects",
                )
            )
            continue
        try:
            _validate_raw_record(raw_record, request)
            parsed_scores.append(
                parse_grouprank_scores(
                    raw_record["raw_text"],
                    expected_handles=tuple(request.handle_to_candidate),
                )
            )
        except ValueError as error:
            failures.append(
                _record_failure(
                    index=index,
                    request=request,
                    message=str(error),
                )
            )

    for index in range(len(requests), len(raw_outputs)):
        failures.append(
            _record_failure(
                index=index,
                request=None,
                message="GroupRank received an unexpected extra raw output",
            )
        )
    if failures:
        first = failures[0]
        location = (
            f"{first['query_id']}/{first['repetition']}/{first['group']}"
            if first["query_id"] is not None
            else f"request {first['request_index']}"
        )
        raise _GroupRankValidationError(
            f"GroupRank parse failure for {location}: {first['error_message']}",
            failures=failures,
        )
    return aggregate_grouprank_scores(requests, parsed_scores)


def build_failure_result(
    selection: Mapping[str, Any],
    prompts: tuple[Mapping[str, Any], ...],
    raw_outputs: tuple[Mapping[str, Any], ...],
    config: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any],
    error: ValueError,
) -> GroupRankFailureResult:
    failures = getattr(error, "failures", ())
    if not failures:
        failures = (
            {
                "request_index": None,
                "query_id": None,
                "repetition": None,
                "group": None,
                "error_message": str(error),
            },
        )
    return GroupRankFailureResult(
        selection=selection,
        prompts=prompts,
        raw_outputs=raw_outputs,
        failure={
            "error_type": "ValueError",
            "error_message": str(error),
            "parse_failures": list(failures),
        },
        config=config,
        runtime=runtime_metadata,
    )


def _model_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model_id": config["model_id"],
        "revision": config["model_revision"],
        "resolved_revision": config["resolved_revision"],
        "dtype": config["dtype"],
        "license": config["license"],
        "parameter_count": config["parameter_count"],
        "max_model_len": config["max_model_len"],
        "gpu_memory_utilization": config["gpu_memory_utilization"],
        "attention_backend": config["attention_backend"],
    }


def _receipt_provenance(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "selection_sha256": config["selection_sha256"],
        "prompt_sha256": config["prompt_sha256"],
        "schema_sha256": config["schema_sha256"],
        "model": _model_receipt(config),
        "code_revision": config["code_revision"],
        "worktree_status": config["worktree_status"],
        "runtime_versions": config["runtime_versions"],
        "container_image": config["container_image"],
    }


def _artifact_hashes(staging: Path, names: frozenset[str]) -> dict[str, str]:
    return {name: _sha256(staging / name) for name in sorted(names)}


def _write_scores(staging: Path, result: GroupRankPhaseAResult) -> None:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(("query_id", "candidate_id", "score", "rank"))
    for entry in result.selection["questions"]:
        query_id = entry["query_id"]
        ranking = list(result.rankings[query_id])
        ranks = {
            candidate_id: rank for rank, candidate_id in enumerate(ranking, start=1)
        }
        for candidate_id in entry["candidate_ids"]:
            score = result.candidate_scores[query_id][candidate_id]
            if not math.isfinite(score):
                raise ValueError("GroupRank scores contain a non-finite value")
            writer.writerow(
                (query_id, candidate_id, f"{score:.17g}", ranks[candidate_id])
            )
    _write_bytes(
        staging / "scores.tsv",
        stream.getvalue().encode("utf-8"),
        fsync=True,
    )


def finalize_success_artifact(
    staging: Path,
    result: GroupRankPhaseAResult,
) -> None:
    """Complete and force the success inventory before atomic publication."""

    write_json(staging / "runtime.json", result.runtime, fsync=True)
    _write_scores(staging, result)
    write_json(staging / "rankings.json", result.rankings, fsync=True)
    artifact_sha256 = _artifact_hashes(staging, _SUCCESS_CORE_FILES)
    receipt = {
        "version": 1,
        "artifact_type": "grouprank_phase_a",
        "qrel_free_phase": True,
        "method": "grouprank",
        "stage": result.selection["stage"],
        "rankings_available": True,
        "scores_available": True,
        "inventory": list(_SUCCESS_INVENTORY),
        "artifact_sha256": artifact_sha256,
        **_receipt_provenance(result.config),
        "execution": {
            "query_count": len(result.selection["questions"]),
            "candidate_count": len(result.selection["questions"][0]["candidate_ids"]),
            "planned_request_count": len(result.prompts),
            "persisted_raw_output_count": len(result.raw_outputs),
            "score_row_count": sum(
                len(entry["candidate_ids"]) for entry in result.selection["questions"]
            ),
            "observations_per_candidate": len(GROUPING_SEEDS),
        },
    }
    write_json(staging / "ranking_receipt.json", receipt, fsync=True)
    _fsync_directory(staging)


def finalize_failure_artifact(
    staging: Path,
    result: GroupRankFailureResult,
) -> None:
    """Complete and force an unscorable failure inventory before publication."""

    write_json(staging / "runtime.json", result.runtime, fsync=True)
    artifact_sha256 = _artifact_hashes(staging, _FAILURE_CORE_FILES)
    parse_failures = result.failure["parse_failures"]
    receipt = {
        "version": 1,
        "artifact_type": "grouprank_phase_a_failure",
        "qrel_free_phase": True,
        "method": "grouprank",
        "stage": result.selection["stage"],
        "disposition": "unscorable_parse_failure",
        "rankings_available": False,
        "scores_available": False,
        "inventory": [*sorted(_FAILURE_CORE_FILES), "failure_receipt.json"],
        "artifact_sha256": artifact_sha256,
        **_receipt_provenance(result.config),
        "execution": {
            "planned_request_count": len(result.prompts),
            "persisted_raw_output_count": len(result.raw_outputs),
            "parse_failure_count": len(parse_failures),
            "rankings_available": False,
            "scores_available": False,
        },
        "failure": result.failure,
    }
    write_json(staging / "failure_receipt.json", receipt, fsync=True)
    _fsync_directory(staging)


def _raise_retained_staging(
    staging: Path,
    error: BaseException,
    *,
    phase: str,
) -> NoReturn:
    raise RuntimeError(
        f"GroupRank {phase} failed; staging retained at {staging}: {error}"
    ) from error


def _publish_staging(staging: Path, destination: Path) -> None:
    if destination.exists():
        raise ValueError("GroupRank destination already exists")
    os.replace(staging, destination)
    _fsync_directory(destination.parent)


def execute_grouprank_phase_a(
    selection: Mapping[str, Any],
    *,
    infer_batch: Callable[[Sequence[GroupRankRequest]], Sequence[Mapping[str, Any]]],
    runtime_metadata: Mapping[str, Any],
    output_dir: str | Path,
) -> GroupRankPhaseAResult:
    """Run and atomically persist the qrel-free GroupRank Phase A inventory."""

    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("GroupRank destination already exists")
    runtime = _validate_runtime_metadata(runtime_metadata)
    frozen_selection, _input_sha256 = preflight_pilot_inputs(
        selection,
        query_path=runtime["query_path"],
        candidate_path=runtime["candidate_path"],
    )
    requests = build_group_requests(frozen_selection)
    prompts = tuple(request_to_record(request) for request in requests)
    config = build_phase_a_config(frozen_selection, prompts, runtime)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.",
            dir=destination.parent,
        )
    )
    try:
        write_json(staging / "config.json", config, fsync=True)
        write_json(staging / "selection.json", frozen_selection, fsync=True)
        write_jsonl(staging / "prompts.jsonl", prompts, fsync=True)
        started = time.perf_counter()
        raw_outputs = tuple(infer_batch(requests))
        elapsed_seconds = time.perf_counter() - started
        write_jsonl(
            staging / "raw_outputs.jsonl",
            raw_outputs,
            fsync=True,
        )
        runtime = finalize_grouprank_runtime(
            runtime,
            raw_outputs,
            elapsed_seconds=elapsed_seconds,
            expected_request_count=_REQUEST_COUNT,
        )
        runtime = _validate_runtime_metadata(runtime, finalized=True)
    except BaseException as error:
        _raise_retained_staging(
            staging,
            error,
            phase="inference or raw persistence",
        )

    try:
        candidate_scores, rankings = parse_and_aggregate(
            requests,
            raw_outputs,
        )
    except ValueError as error:
        failure_result = build_failure_result(
            frozen_selection,
            prompts,
            raw_outputs,
            config,
            runtime,
            error,
        )
        try:
            finalize_failure_artifact(staging, failure_result)
            _publish_staging(staging, destination)
        except BaseException as persistence_error:
            _raise_retained_staging(
                staging,
                persistence_error,
                phase="failure artifact publication",
            )
        raise GroupRankParseFailure(str(error), result=failure_result) from error

    success_result = GroupRankPhaseAResult(
        selection=frozen_selection,
        prompts=prompts,
        raw_outputs=raw_outputs,
        candidate_scores=candidate_scores,
        rankings=rankings,
        config=config,
        runtime=runtime,
    )
    try:
        finalize_success_artifact(staging, success_result)
        _publish_staging(staging, destination)
    except BaseException as error:
        _raise_retained_staging(
            staging,
            error,
            phase="success artifact publication",
        )
    return success_result


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _load_jsonl(path: Path, *, label: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    try:
        handle = path.open("r", encoding="utf-8")
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    with handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{label} contains a blank line")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{label} line {line_number} is not valid JSON"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"{label} rows must be objects")
            rows.append(row)
    return tuple(rows)


def _load_scores(
    path: Path,
    *,
    selection: Mapping[str, Any],
) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], int]]:
    expected_pairs = {
        (entry["query_id"], candidate_id)
        for entry in selection["questions"]
        for candidate_id in entry["candidate_ids"]
    }
    values: dict[str, dict[str, float]] = {
        entry["query_id"]: {} for entry in selection["questions"]
    }
    ranks: dict[tuple[str, str], int] = {}
    seen: set[tuple[str, str]] = set()
    try:
        handle = path.open("r", encoding="utf-8", newline="")
    except OSError as error:
        raise ValueError("GroupRank scores are missing") from error
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["query_id", "candidate_id", "score", "rank"]:
            raise ValueError("GroupRank score header changed")
        for row in reader:
            key = (row["query_id"], row["candidate_id"])
            if key in seen or key not in expected_pairs:
                raise ValueError("GroupRank scores contain a duplicate or unknown pair")
            try:
                score = float(row["score"])
                rank = int(row["rank"])
            except ValueError as error:
                raise ValueError("GroupRank score row is invalid") from error
            if not math.isfinite(score):
                raise ValueError("GroupRank scores contain a non-finite value")
            seen.add(key)
            values[key[0]][key[1]] = score
            ranks[key] = rank
    if seen != expected_pairs:
        raise ValueError("GroupRank scores are incomplete")
    return values, ranks


def _validate_config(config: Mapping[str, Any]) -> None:
    if set(config) != _CONFIG_KEYS:
        raise ValueError("GroupRank config schema changed")
    if (
        config.get("version") != 1
        or config.get("qrel_free_phase") is not True
        or config.get("method") != "grouprank"
        or config.get("stage") != 2
        or config.get("seed") != 2026
        or config.get("grouping_seeds") != list(GROUPING_SEEDS)
        or config.get("request_count") != _REQUEST_COUNT
    ):
        raise ValueError("GroupRank config contract changed")
    for field, expected in (
        ("model_id", MODEL_ID),
        ("model_revision", MODEL_REVISION),
        ("resolved_revision", MODEL_REVISION),
        ("dtype", MODEL_DTYPE),
        ("license", MODEL_LICENSE),
        ("parameter_count", _MODEL_PARAMETER_COUNT),
        ("max_model_len", MAX_MODEL_LEN),
        ("gpu_memory_utilization", GPU_MEMORY_UTILIZATION),
        ("attention_backend", ATTENTION_BACKEND),
        ("container_image", ROCM_CONTAINER_IMAGE),
    ):
        if config.get(field) != expected:
            labels = {
                "model_revision": "model revision",
                "resolved_revision": "resolved model revision",
                "parameter_count": "parameter count",
                "max_model_len": "maximum model length",
                "gpu_memory_utilization": "GPU memory utilization",
                "attention_backend": "attention backend",
            }
            raise ValueError(f"GroupRank {labels.get(field, field)} changed")
    code_revision = config.get("code_revision")
    if (
        not isinstance(code_revision, str)
        or _COMMIT_SHA_RE.fullmatch(code_revision) is None
    ):
        raise ValueError("GroupRank code revision changed")
    if config.get("worktree_status") != "clean":
        raise ValueError("GroupRank worktree status changed")
    versions = config.get("runtime_versions")
    if (
        not isinstance(versions, Mapping)
        or set(versions) != {"vllm", "torch", "hip"}
        or any(type(value) is not str or not value for value in versions.values())
    ):
        raise ValueError("GroupRank runtime versions changed")


def _validate_receipt_provenance(
    receipt: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    if receipt.get("selection_sha256") != config["selection_sha256"]:
        raise ValueError("GroupRank selection SHA256 changed")
    if receipt.get("prompt_sha256") != config["prompt_sha256"]:
        raise ValueError("GroupRank prompt SHA256 changed")
    if receipt.get("schema_sha256") != config["schema_sha256"]:
        raise ValueError("GroupRank schema SHA256 changed")
    if receipt.get("model") != _model_receipt(config):
        raise ValueError("GroupRank receipt model metadata changed")
    for field in (
        "code_revision",
        "worktree_status",
        "runtime_versions",
        "container_image",
    ):
        if receipt.get(field) != config[field]:
            raise ValueError(f"GroupRank receipt {field.replace('_', ' ')} changed")


def _validated_artifact(
    artifact_dir: str | Path,
) -> tuple[GroupRankPhaseAResult, dict[str, Any]]:
    source = Path(artifact_dir)
    try:
        directory_inventory = {path.name for path in source.iterdir()}
    except OSError as error:
        raise ValueError("GroupRank artifact directory is missing") from error
    if directory_inventory != set(_SUCCESS_INVENTORY):
        raise ValueError("GroupRank directory inventory changed")
    receipt_path = source / "ranking_receipt.json"
    receipt = _load_json_object(receipt_path, label="GroupRank ranking receipt")
    if receipt.get("inventory") != list(_SUCCESS_INVENTORY):
        raise ValueError("GroupRank receipt inventory changed")
    if (
        receipt.get("qrel_free_phase") is not True
        or receipt.get("artifact_type") != "grouprank_phase_a"
        or receipt.get("rankings_available") is not True
        or receipt.get("scores_available") is not True
    ):
        raise ValueError("GroupRank ranking receipt contract changed")
    artifact_sha256 = receipt.get("artifact_sha256")
    if (
        not isinstance(artifact_sha256, Mapping)
        or set(artifact_sha256) != _SUCCESS_CORE_FILES
    ):
        raise ValueError("GroupRank core artifact set changed")
    for name in sorted(_SUCCESS_CORE_FILES):
        path = source / name
        if not path.is_file() or _sha256(path) != artifact_sha256[name]:
            raise ValueError(f"artifact hash mismatch: {name}")
    if receipt.get("prompt_sha256") != artifact_sha256["prompts.jsonl"]:
        raise ValueError("GroupRank prompt SHA256 changed")

    config = _load_json_object(source / "config.json", label="GroupRank config")
    _validate_config(config)
    if config["prompt_sha256"] != artifact_sha256["prompts.jsonl"]:
        raise ValueError("GroupRank prompt SHA256 changed")
    _validate_receipt_provenance(receipt, config)

    selection = _load_json_object(
        source / "selection.json",
        label="GroupRank selection",
    )
    selection_sha256 = artifact_sha256["selection.json"]
    if (
        config["selection_sha256"] != selection_sha256
        or receipt["selection_sha256"] != selection_sha256
    ):
        raise ValueError("GroupRank selection SHA256 changed")
    frozen_selection, input_sha256 = preflight_pilot_inputs(
        selection,
        query_path=config["query_path"],
        candidate_path=config["candidate_path"],
    )
    if config["input_sha256"] != input_sha256:
        raise ValueError("GroupRank input SHA256 changed")

    requests = build_group_requests(frozen_selection)
    expected_prompts = tuple(request_to_record(request) for request in requests)
    prompts = _load_jsonl(source / "prompts.jsonl", label="GroupRank prompts")
    if prompts != expected_prompts:
        raise ValueError("GroupRank prompt content changed")
    if _schema_sha256(prompts) != config["schema_sha256"]:
        raise ValueError("GroupRank schema SHA256 changed")

    runtime = _load_json_object(source / "runtime.json", label="GroupRank runtime")
    validated_runtime = _validate_runtime_metadata(runtime, finalized=True)
    for runtime_field, config_field in (
        ("model_id", "model_id"),
        ("revision", "model_revision"),
        ("resolved_revision", "resolved_revision"),
        ("dtype", "dtype"),
        ("license", "license"),
        ("parameter_count", "parameter_count"),
        ("max_model_len", "max_model_len"),
        ("gpu_memory_utilization", "gpu_memory_utilization"),
        ("attention_backend", "attention_backend"),
        ("code_revision", "code_revision"),
        ("worktree_status", "worktree_status"),
        ("query_path", "query_path"),
        ("candidate_path", "candidate_path"),
        ("container_image", "container_image"),
    ):
        if validated_runtime[runtime_field] != config[config_field]:
            raise ValueError(
                f"GroupRank runtime {runtime_field.replace('_', ' ')} changed"
            )
    for runtime_field, config_field, label in (
        ("vllm_version", "vllm", "vLLM version"),
        ("torch_version", "torch", "Torch version"),
        ("hip_version", "hip", "HIP version"),
    ):
        if validated_runtime[runtime_field] != config["runtime_versions"][config_field]:
            raise ValueError(f"GroupRank runtime {label} changed")

    raw_outputs = _load_jsonl(
        source / "raw_outputs.jsonl",
        label="GroupRank raw outputs",
    )
    reconciled_runtime = finalize_grouprank_runtime(
        validated_runtime,
        raw_outputs,
        elapsed_seconds=validated_runtime["elapsed_seconds"],
        expected_request_count=_REQUEST_COUNT,
    )
    if reconciled_runtime["inference_stats"] != validated_runtime["inference_stats"]:
        raise ValueError("GroupRank inference stats disagree with raw outputs")
    candidate_scores, expected_rankings = parse_and_aggregate(
        requests,
        raw_outputs,
    )
    rankings_payload = _load_json_object(
        source / "rankings.json",
        label="GroupRank rankings",
    )
    expected_query_ids = {entry["query_id"] for entry in frozen_selection["questions"]}
    if set(rankings_payload) != expected_query_ids:
        raise ValueError("GroupRank ranking query inventory changed")
    rankings: dict[str, list[str]] = {}
    for entry in frozen_selection["questions"]:
        query_id = entry["query_id"]
        ranking = rankings_payload[query_id]
        if (
            not isinstance(ranking, list)
            or len(ranking) != _CANDIDATE_COUNT
            or len(set(ranking)) != _CANDIDATE_COUNT
            or set(ranking) != set(entry["candidate_ids"])
        ):
            raise ValueError("GroupRank ranking candidate inventory changed")
        rankings[query_id] = ranking

    stored_scores, stored_ranks = _load_scores(
        source / "scores.tsv",
        selection=frozen_selection,
    )
    for query_id, query_scores in candidate_scores.items():
        for candidate_id, expected_score in query_scores.items():
            if stored_scores[query_id][candidate_id] != expected_score:
                raise ValueError("GroupRank score disagrees with raw output")
        if rankings[query_id] != expected_rankings[query_id]:
            raise ValueError("GroupRank ranking disagrees with scores")
        for rank, candidate_id in enumerate(rankings[query_id], start=1):
            if stored_ranks[(query_id, candidate_id)] != rank:
                raise ValueError("GroupRank stored rank changed")

    execution = receipt.get("execution")
    if not isinstance(execution, Mapping) or execution != {
        "query_count": _QUESTION_COUNT,
        "candidate_count": _CANDIDATE_COUNT,
        "planned_request_count": _REQUEST_COUNT,
        "persisted_raw_output_count": _REQUEST_COUNT,
        "score_row_count": _QUESTION_COUNT * _CANDIDATE_COUNT,
        "observations_per_candidate": len(GROUPING_SEEDS),
    }:
        raise ValueError("GroupRank ranking receipt counts changed")

    result = GroupRankPhaseAResult(
        selection=frozen_selection,
        prompts=prompts,
        raw_outputs=raw_outputs,
        candidate_scores=candidate_scores,
        rankings=expected_rankings,
        config=config,
        runtime=validated_runtime,
    )
    return result, {**receipt, "receipt_sha256": _sha256(receipt_path)}


def validate_grouprank_artifact(artifact_dir: str | Path) -> dict[str, Any]:
    """Recompute and validate every frozen GroupRank artifact relationship."""

    _result, receipt = _validated_artifact(artifact_dir)
    return receipt


def load_frozen_grouprank(artifact_dir: str | Path) -> GroupRankPhaseAResult:
    """Load a GroupRank result only after complete hash and semantic validation."""

    result, _receipt = _validated_artifact(artifact_dir)
    return result
