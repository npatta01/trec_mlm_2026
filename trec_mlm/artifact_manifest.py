"""Deterministic provenance manifests and enforced smoke receipts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

from trec_mlm.llm_ranker_inference import (
    ATTENTION_BACKEND,
    MAX_MODEL_LEN,
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_REVISION,
    PILOT_GPU_MEMORY_UTILIZATION,
)


_PILOT_GRADE_TOKEN_IDS = {0: 15, 1: 16, 2: 17}
_SMOKE_HANDLE_RE = re.compile(r"C(\d{3})")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_manifest(
    artifact_dirs: Sequence[str | Path],
    *,
    code_revision: str,
    worktree_status: str,
) -> dict[str, Any]:
    """Hash every file under explicitly supplied artifact directories."""

    if not code_revision.strip():
        raise ValueError("code revision must be non-empty")
    artifacts: list[dict[str, Any]] = []
    for value in sorted((Path(item) for item in artifact_dirs), key=lambda p: str(p)):
        if not value.is_dir():
            raise ValueError(f"artifact directory does not exist: {value}")
        files = {
            str(path.relative_to(value)): _sha256(path)
            for path in sorted(value.rglob("*"))
            if path.is_file()
        }
        artifacts.append({"directory": str(value), "files": files})
    return {
        "version": 1,
        "code_revision": code_revision,
        "worktree_status": worktree_status,
        "artifacts": artifacts,
    }


def validate_smoke_receipt(
    path: str | Path,
    *,
    model_id: str,
    revision: str,
    dtype: str,
) -> dict[str, Any]:
    """Require a passed smoke receipt for the exact frozen runtime contract."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise ValueError("smoke receipt did not pass")
    for field, expected in (
        ("model_id", model_id),
        ("revision", revision),
        ("dtype", dtype),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"smoke receipt {field} does not match")
    return {**payload, "receipt_sha256": _sha256(source)}


def _grade_mapping(payload: object, *, field: str) -> dict[int, float]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"smoke receipt {field} must contain grades 0, 1, and 2")
    try:
        values = {int(key): float(value) for key, value in payload.items()}
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"smoke receipt {field} must contain numeric grades 0, 1, and 2"
        ) from error
    if set(values) != {0, 1, 2} or not all(
        math.isfinite(value) for value in values.values()
    ):
        raise ValueError(
            f"smoke receipt {field} must contain finite grades 0, 1, and 2"
        )
    return values


def validate_pilot_smoke_receipt(
    path: str | Path,
    *,
    model_id: str = MODEL_ID,
    revision: str = MODEL_REVISION,
    dtype: str = MODEL_DTYPE,
    seed: int = 2026,
) -> dict[str, Any]:
    """Validate the complete pinned pointwise/listwise pilot smoke contract."""

    source = Path(path)
    payload = validate_smoke_receipt(
        source,
        model_id=model_id,
        revision=revision,
        dtype=dtype,
    )

    grade_tokens = payload.get("grade_token_ids")
    if (
        not isinstance(grade_tokens, Mapping)
        or set(grade_tokens) != {"0", "1", "2"}
        or any(type(token_id) is not int for token_id in grade_tokens.values())
    ):
        raise ValueError("smoke receipt grade token IDs changed")
    normalized_tokens = {
        int(grade): token_id for grade, token_id in grade_tokens.items()
    }
    if normalized_tokens != _PILOT_GRADE_TOKEN_IDS:
        raise ValueError("smoke receipt grade token IDs changed")

    transformers = _grade_mapping(
        payload.get("transformers_logprobs"), field="Transformers log probabilities"
    )
    vllm = _grade_mapping(
        payload.get("vllm_logprobs"), field="vLLM log probabilities"
    )
    observed_delta = max(
        abs(transformers[grade] - vllm[grade]) for grade in (0, 1, 2)
    )
    recorded_delta = payload.get("maximum_logprob_delta")
    if (
        isinstance(recorded_delta, bool)
        or not isinstance(recorded_delta, (int, float))
        or not math.isfinite(float(recorded_delta))
        or float(recorded_delta) > 1e-4
        or observed_delta > 1e-4
        or not math.isclose(
            float(recorded_delta), observed_delta, rel_tol=1e-9, abs_tol=1e-12
        )
    ):
        raise ValueError("smoke receipt parity delta is invalid")

    try:
        handles = json.loads(payload.get("listwise_output"))
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError("smoke receipt semantic handles are invalid") from error
    if (
        payload.get("listwise_valid_count") != 20
        or not isinstance(handles, list)
        or len(handles) != 20
        or not all(isinstance(handle, str) for handle in handles)
        or len(set(handles)) != 20
    ):
        raise ValueError("smoke receipt semantic handles are invalid")
    for handle in handles:
        match = _SMOKE_HANDLE_RE.fullmatch(handle)
        if match is None or int(match.group(1)) >= 150:
            raise ValueError("smoke receipt semantic handles are invalid")

    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("smoke receipt runtime metadata is missing")
    for field, expected in (
        ("model_id", model_id),
        ("revision", revision),
        ("dtype", dtype),
        ("seed", seed),
        ("gpu_memory_utilization", PILOT_GPU_MEMORY_UTILIZATION),
        ("max_model_len", MAX_MODEL_LEN),
        ("attention_backend", ATTENTION_BACKEND),
    ):
        if metadata.get(field) != expected:
            labels = {
                "revision": "model revision",
                "gpu_memory_utilization": "GPU memory utilization",
                "max_model_len": "maximum model length",
                "attention_backend": "attention backend",
            }
            label = labels.get(field, field)
            raise ValueError(f"smoke receipt metadata {label} does not match")
    for field, label in (
        ("vllm_version", "vLLM version"),
        ("torch_version", "Torch version"),
        ("hip_version", "HIP version"),
    ):
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"smoke receipt {label} is missing")

    return {
        **payload,
        "grade_token_ids": normalized_tokens,
        "receipt_sha256": _sha256(source),
    }
