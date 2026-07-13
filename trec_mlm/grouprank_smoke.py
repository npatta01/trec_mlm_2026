"""Durable smoke evidence for the frozen Stage 2 GroupRank pilot."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any

from trec_mlm.grouprank import (
    build_group_requests,
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
)
from trec_mlm.grouprank_pilot import request_to_record
from trec_mlm.llm_ranker_pilot import preflight_pilot_inputs


_MODEL_PARAMETER_COUNT = 7_615_616_512
_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_SHA256_RE = re.compile(r"[0-9a-f]{64}")

_MODEL_KEYS = frozenset(
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
    }
)
_RUNTIME_KEYS = frozenset(
    {
        *_MODEL_KEYS,
        "vllm_version",
        "torch_version",
        "hip_version",
        "device_properties",
        "code_revision",
        "worktree_status",
        "query_path",
        "candidate_path",
        "inference_stats",
    }
)
_RUNTIME_OPTIONAL_KEYS = frozenset({"container_digest", "elapsed_seconds"})
_PROMPT_KEYS = frozenset(
    {
        "query_id",
        "repetition",
        "group",
        "ordered_candidate_ids",
        "handle_to_candidate",
        "messages",
        "sampling_seed",
        "score_schema",
    }
)
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
_RECEIPT_KEYS = frozenset(
    {
        "version",
        "artifact_type",
        "qrel_free_phase",
        "method",
        "passed",
        "selection_sha256",
        "prompt_sha256",
        "raw_output_sha256",
        "model",
        "runtime",
        "prompt",
        "raw_output",
        "validation",
    }
)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_provisional(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist raw smoke evidence at its final name before semantic parsing."""

    with path.open("xb") as handle:
        handle.write(_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _replace_fsynced(path: Path, payload: Mapping[str, Any]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _model_contract() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
        "dtype": MODEL_DTYPE,
        "license": MODEL_LICENSE,
        "parameter_count": _MODEL_PARAMETER_COUNT,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "attention_backend": ATTENTION_BACKEND,
    }


def _model_label(field: str) -> str:
    return {
        "model_id": "model ID",
        "revision": "model revision",
        "resolved_revision": "resolved model revision",
        "parameter_count": "model parameter count",
        "max_model_len": "maximum model length",
        "gpu_memory_utilization": "GPU memory utilization",
        "attention_backend": "attention backend",
    }.get(field, field)


def _validate_model(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _MODEL_KEYS:
        raise ValueError("GroupRank smoke model schema changed")
    model = dict(payload)
    for field, expected in _model_contract().items():
        if model[field] != expected:
            raise ValueError(f"GroupRank smoke {_model_label(field)} changed")
    return model


def _validate_inference_stats(payload: object) -> dict[str, int]:
    fields = {
        "request_count",
        "prompt_token_count",
        "output_token_count",
    }
    if not isinstance(payload, Mapping) or set(payload) != fields:
        raise ValueError("GroupRank smoke inference stats schema changed")
    result: dict[str, int] = {}
    for field, value in payload.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"GroupRank smoke inference stat {field} changed")
        result[field] = value
    if result["request_count"] != 1:
        raise ValueError("GroupRank smoke must record exactly one request")
    return result


def _validate_runtime(payload: object) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("GroupRank smoke runtime metadata must be an object")
    fields = set(payload)
    if fields - (_RUNTIME_KEYS | _RUNTIME_OPTIONAL_KEYS) or not _RUNTIME_KEYS.issubset(
        fields
    ):
        raise ValueError("GroupRank smoke runtime metadata schema changed")
    runtime = dict(payload)
    model = _validate_model({field: runtime[field] for field in _MODEL_KEYS})
    for field, expected in model.items():
        if runtime[field] != expected:
            raise ValueError(f"GroupRank smoke runtime {_model_label(field)} changed")
    for field, label in (
        ("vllm_version", "vLLM version"),
        ("torch_version", "Torch version"),
        ("hip_version", "HIP version"),
        ("device_properties", "device properties"),
        ("query_path", "query path"),
        ("candidate_path", "candidate path"),
    ):
        value = runtime[field]
        if type(value) is not str or not value.strip():
            raise ValueError(f"GroupRank smoke runtime {label} is missing")
    code_revision = runtime["code_revision"]
    if (
        type(code_revision) is not str
        or _COMMIT_SHA_RE.fullmatch(code_revision) is None
    ):
        raise ValueError("GroupRank smoke code revision must be a full 40-hex SHA")
    if runtime["worktree_status"] != "clean":
        raise ValueError("GroupRank smoke worktree status must be exactly 'clean'")
    runtime["inference_stats"] = _validate_inference_stats(runtime["inference_stats"])
    if "container_digest" in runtime and (
        type(runtime["container_digest"]) is not str
        or not runtime["container_digest"].strip()
    ):
        raise ValueError("GroupRank smoke container digest is missing")
    if "elapsed_seconds" in runtime:
        elapsed = runtime["elapsed_seconds"]
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(float(elapsed))
            or elapsed < 0
        ):
            raise ValueError("GroupRank smoke elapsed seconds changed")
    return runtime


def _validate_raw_against_prompt(
    payload: object,
    prompt: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != _RAW_OUTPUT_KEYS:
        raise ValueError("GroupRank smoke raw output schema changed")
    raw = dict(payload)
    for field in (
        "query_id",
        "repetition",
        "group",
        "ordered_candidate_ids",
        "handle_to_candidate",
        "sampling_seed",
    ):
        if raw[field] != prompt[field]:
            raise ValueError(
                f"GroupRank smoke raw output {field.replace('_', ' ')} changed"
            )
    if type(raw["raw_text"]) is not str:
        raise ValueError("GroupRank smoke raw output text is missing")
    for field in ("prompt_token_count", "output_token_count"):
        if type(raw[field]) is not int or raw[field] < 0:
            raise ValueError(
                f"GroupRank smoke raw output {field.replace('_', ' ')} changed"
            )
    finish_reason = raw["finish_reason"]
    if type(finish_reason) is not str or not finish_reason:
        raise ValueError("GroupRank smoke finish reason is missing")
    if finish_reason == "length":
        raise ValueError("GroupRank smoke output was length-truncated")
    return raw


def _base_receipt(
    *,
    selection: Mapping[str, Any],
    prompt: Mapping[str, Any],
    raw_output: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "version": 1,
        "artifact_type": "grouprank_smoke",
        "qrel_free_phase": True,
        "method": "grouprank",
        "passed": False,
        "selection_sha256": _sha256_bytes(_json_bytes(selection)),
        "prompt_sha256": _sha256_bytes(_json_bytes(prompt)),
        "raw_output_sha256": _sha256_bytes(_json_bytes(raw_output)),
        "model": _model_contract(),
        "runtime": dict(runtime),
        "prompt": dict(prompt),
        "raw_output": dict(raw_output),
        "validation": {"state": "pending"},
    }


def write_grouprank_smoke(
    selection: Mapping[str, Any],
    *,
    raw_output: Mapping[str, Any],
    runtime_metadata: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Persist one first-group response before parsing and finalize its receipt."""

    destination = Path(output_path)
    if destination.exists():
        raise ValueError("GroupRank smoke destination already exists")
    runtime = _validate_runtime(runtime_metadata)
    frozen_selection, _input_sha256 = preflight_pilot_inputs(
        selection,
        query_path=runtime["query_path"],
        candidate_path=runtime["candidate_path"],
    )
    first_request = build_group_requests(frozen_selection)[0]
    prompt = request_to_record(first_request)
    raw = dict(raw_output)
    receipt = _base_receipt(
        selection=frozen_selection,
        prompt=prompt,
        raw_output=raw,
        runtime=runtime,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_provisional(destination, receipt)

    try:
        raw = _validate_raw_against_prompt(raw, prompt)
        stats = runtime["inference_stats"]
        if (
            stats["prompt_token_count"] != raw["prompt_token_count"]
            or stats["output_token_count"] != raw["output_token_count"]
        ):
            raise ValueError("GroupRank smoke inference stats disagree with raw output")
        scores = parse_grouprank_scores(
            raw["raw_text"],
            expected_handles=tuple(first_request.handle_to_candidate),
        )
    except ValueError as error:
        receipt["validation"] = {
            "state": "failed",
            "error_type": "ValueError",
            "error_message": str(error),
        }
        _replace_fsynced(destination, receipt)
        raise

    receipt["passed"] = True
    receipt["validation"] = {"state": "passed", "scores": scores}
    _replace_fsynced(destination, receipt)
    return validate_grouprank_smoke(
        destination,
        selection=frozen_selection,
        code_revision=runtime["code_revision"],
    )


def _load_receipt(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            "GroupRank smoke receipt is missing or not valid JSON"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError("GroupRank smoke receipt must be an object")
    return payload


def validate_grouprank_smoke(
    path: str | Path,
    *,
    selection: Mapping[str, Any] | None = None,
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Validate a passed smoke for the exact model, prompt, raw, and provenance."""

    source = Path(path)
    receipt = _load_receipt(source)
    if set(receipt) != _RECEIPT_KEYS:
        raise ValueError("GroupRank smoke receipt schema changed")
    for field, expected in (
        ("version", 1),
        ("artifact_type", "grouprank_smoke"),
        ("qrel_free_phase", True),
        ("method", "grouprank"),
    ):
        if receipt[field] != expected:
            raise ValueError(
                f"GroupRank smoke receipt {field.replace('_', ' ')} changed"
            )
    if receipt["passed"] is not True:
        raise ValueError("GroupRank smoke receipt did not pass")

    _validate_model(receipt["model"])
    runtime = _validate_runtime(receipt["runtime"])
    if receipt["model"] != {field: runtime[field] for field in _MODEL_KEYS}:
        raise ValueError("GroupRank smoke model and runtime disagree")
    if code_revision is not None and runtime["code_revision"] != code_revision:
        raise ValueError("GroupRank smoke receipt code revision changed")

    prompt = receipt["prompt"]
    if not isinstance(prompt, Mapping) or set(prompt) != _PROMPT_KEYS:
        raise ValueError("GroupRank smoke prompt schema changed")
    for field in ("selection_sha256", "prompt_sha256", "raw_output_sha256"):
        value = receipt[field]
        if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"GroupRank smoke {field.replace('_', ' ')} changed")
    if receipt["prompt_sha256"] != _sha256_bytes(_json_bytes(prompt)):
        raise ValueError("GroupRank smoke prompt SHA256 changed")

    raw = receipt["raw_output"]
    if receipt["raw_output_sha256"] != _sha256_bytes(_json_bytes(raw)):
        raise ValueError("GroupRank smoke raw output SHA256 changed")
    raw = _validate_raw_against_prompt(raw, prompt)
    stats = runtime["inference_stats"]
    if (
        stats["prompt_token_count"] != raw["prompt_token_count"]
        or stats["output_token_count"] != raw["output_token_count"]
    ):
        raise ValueError("GroupRank smoke inference stats disagree with raw output")

    expected_handles = tuple(prompt["handle_to_candidate"])
    scores = parse_grouprank_scores(
        raw["raw_text"],
        expected_handles=expected_handles,
    )
    validation = receipt["validation"]
    if (
        not isinstance(validation, Mapping)
        or set(validation) != {"state", "scores"}
        or validation["state"] != "passed"
        or validation["scores"] != scores
    ):
        raise ValueError("GroupRank smoke score validation changed")

    if selection is not None:
        frozen_selection, _input_sha256 = preflight_pilot_inputs(
            selection,
            query_path=runtime["query_path"],
            candidate_path=runtime["candidate_path"],
        )
        if receipt["selection_sha256"] != _sha256_bytes(_json_bytes(frozen_selection)):
            raise ValueError("GroupRank smoke selection SHA256 changed")
        expected_prompt = request_to_record(build_group_requests(frozen_selection)[0])
        if prompt != expected_prompt:
            raise ValueError("GroupRank smoke did not use the first frozen group")

    return {**receipt, "receipt_sha256": _sha256(source)}
