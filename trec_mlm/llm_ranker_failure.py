"""Capture and validate terminal LLM-ranker failures without inventing rankings."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import tempfile
from typing import Any

from trec_mlm.label_retrieval import ROCM_IMAGE
from trec_mlm.llm_ranker_inference import MODEL_DTYPE, MODEL_ID, MODEL_REVISION


_OBSERVED_FILES = ("container.log", "expected_prompts.jsonl", "inspect.json")
_INVENTORY = (*_OBSERVED_FILES, "failure_receipt.json")
_SENSITIVE_ENV_NAME = re.compile(
    r"(?:^|_)(?:API_?KEY|AUTH|CREDENTIAL|PASSWORD|PASSWD|SECRET|TOKEN)(?:$|_)",
    re.IGNORECASE,
)
_RAW_SECRET_PATTERNS = (
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
    re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
_ERROR_RE = re.compile(
    r"ValueError: (pilot listwise parse failure for "
    r"(?P<query_id>[0-9a-f]+)/(?P<rotation>\d+): "
    r"pilot ranking has duplicates, omissions, or unknown handles)"
)
_PROGRESS_RE = re.compile(r"Processed prompts:[^\r\n]*?(\d+)/(\d+)")
_FORBIDDEN_RANKING_FILES = (
    "ranking_receipt.json",
    "rankings.json",
    "scores.tsv",
    "raw_outputs.jsonl",
)
_RECEIPT_KEYS = frozenset(
    {
        "version",
        "artifact_type",
        "qrel_free_phase",
        "method",
        "stage",
        "disposition",
        "rerun_permitted",
        "inventory",
        "artifact_sha256",
        "selection",
        "smoke_receipt",
        "model",
        "code_revision",
        "container",
        "capture_commands",
        "execution",
        "expected_prompts",
        "failure",
    }
)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_inspect(payload: bytes) -> dict[str, Any]:
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("podman inspect output is not valid JSON") from error
    if not isinstance(decoded, list) or len(decoded) != 1 or not isinstance(
        decoded[0], dict
    ):
        raise ValueError("podman inspect must contain exactly one container")
    return decoded[0]


def _reject_secrets(inspect_bytes: bytes, log_bytes: bytes) -> None:
    """Fail closed instead of modifying the exact observed byte streams."""

    inspected = _load_inspect(inspect_bytes)
    environment = inspected.get("Config", {}).get("Env", [])
    if not isinstance(environment, list):
        raise ValueError("podman inspect environment schema changed")
    for item in environment:
        if not isinstance(item, str) or "=" not in item:
            raise ValueError("podman inspect environment schema changed")
        name, value = item.split("=", 1)
        if value and name.upper() != "TOKENIZERS_PARALLELISM" and _SENSITIVE_ENV_NAME.search(name):
            raise ValueError(f"secret-bearing environment variable detected: {name}")
    for stream in (inspect_bytes, log_bytes):
        if any(pattern.search(stream) for pattern in _RAW_SECRET_PATTERNS):
            raise ValueError("secret-like credential detected in observed bytes")


def _container_facts(
    inspected: Mapping[str, Any],
    *,
    container_name: str,
    selection_path: str | Path,
    smoke_receipt_path: str | Path,
    expected_ranking_dir: str | Path,
) -> dict[str, Any]:
    state = inspected.get("State")
    config = inspected.get("Config")
    if not isinstance(state, Mapping) or not isinstance(config, Mapping):
        raise ValueError("podman inspect state/config schema changed")
    image = inspected.get("ImageName") or config.get("Image")
    accepted_images = {ROCM_IMAGE, f"docker.io/{ROCM_IMAGE}"}
    if image not in accepted_images or config.get("Image") not in accepted_images:
        raise ValueError("terminal failure container image changed")
    args = inspected.get("Args")
    if (
        inspected.get("Name") != container_name
        or not isinstance(inspected.get("Id"), str)
        or re.fullmatch(r"[0-9a-f]{64}", inspected["Id"]) is None
        or inspected.get("Path") != "/bin/bash"
        or not isinstance(args, list)
        or len(args) != 2
        or args[0] != "-lc"
        or not isinstance(args[1], str)
    ):
        raise ValueError("terminal failure container identity changed")
    try:
        command_tokens = shlex.split(args[1])
    except ValueError as error:
        raise ValueError("terminal failure command is invalid") from error
    expected_prefix = [
        "python",
        "-m",
        "trec_mlm.cli",
        "llm-label-ranker-pilot",
    ]
    if command_tokens[:4] != expected_prefix or len(command_tokens) != 16:
        raise ValueError("terminal failure command changed")
    options = command_tokens[4:]
    if len(options) % 2:
        raise ValueError("terminal failure command changed")
    parsed = dict(zip(options[::2], options[1::2], strict=True))
    expected_options = {
        "--stage": "2",
        "--method": "listwise",
    }
    if any(parsed.get(key) != value for key, value in expected_options.items()):
        raise ValueError("terminal failure command changed")
    for option, expected_path in (
        ("--selection", selection_path),
        ("--smoke-receipt", smoke_receipt_path),
        ("--output-dir", expected_ranking_dir),
    ):
        observed_path = parsed.get(option)
        if not isinstance(observed_path, str) or Path(observed_path).resolve(
            strict=False
        ) != Path(expected_path).resolve(strict=False):
            raise ValueError("terminal failure command changed")
    code_revision = parsed.get("--code-revision")
    if not isinstance(code_revision, str) or re.fullmatch(
        r"[0-9a-f]{40}", code_revision
    ) is None:
        raise ValueError("terminal failure code revision changed")
    if (
        state.get("Status") != "exited"
        or type(state.get("ExitCode")) is not int
        or state["ExitCode"] != 1
        or state.get("OOMKilled") is not False
        or not isinstance(state.get("StartedAt"), str)
        or not isinstance(state.get("FinishedAt"), str)
    ):
        raise ValueError("terminal failure exit state changed")
    return {
        "name": container_name,
        "id": inspected["Id"],
        "image": image,
        "path": inspected["Path"],
        "args": list(args),
        "command_tokens": command_tokens,
        "started_at": state["StartedAt"],
        "finished_at": state["FinishedAt"],
        "status": state["Status"],
        "exit_code": state["ExitCode"],
        "oom_killed": state["OOMKilled"],
        "code_revision": code_revision,
    }


def _failure_facts(log_bytes: bytes, *, planned_request_count: int) -> dict[str, Any]:
    try:
        text = log_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("podman logs are not UTF-8") from error
    observed = [
        (int(match.group(1)), int(match.group(2)))
        for match in _PROGRESS_RE.finditer(text)
    ]
    if (
        not observed
        or any(total != planned_request_count for _completed, total in observed)
        or max(completed for completed, _total in observed)
        != planned_request_count
    ):
        raise ValueError("terminal log request count changed")
    failures = list(_ERROR_RE.finditer(text))
    if len(failures) != 1:
        raise ValueError("terminal listwise error is missing or ambiguous")
    failure = failures[0]
    return {
        "error_type": "ValueError",
        "error_message": failure.group(1),
        "query_id": failure.group("query_id"),
        "rotation": int(failure.group("rotation")),
        "log_observed_request_count": max(
            completed for completed, _total in observed
        ),
    }


def _expected_prompt_rows(selection: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    from trec_mlm.llm_ranker_pilot import _expected_pilot_prompts

    return tuple(
        {
            "evidence_status": "expected_not_observed",
            "provenance": "regenerated_qrel_free",
            "prompt": prompt,
        }
        for prompt in _expected_pilot_prompts(selection, method="listwise")
    )


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def _jsonl_bytes(rows: tuple[dict[str, Any], ...]) -> bytes:
    return b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def capture_terminal_listwise_failure(
    *,
    container_name: str,
    output_dir: str | Path,
    selection_path: str | Path,
    smoke_receipt_path: str | Path,
    expected_ranking_dir: str | Path,
) -> dict[str, Any]:
    """Atomically capture exact inspect/log bytes for an unscorable final attempt."""

    destination = Path(output_dir)
    if destination.exists():
        raise ValueError("terminal failure destination already exists")
    inspect_command = ["podman", "inspect", container_name]
    log_command = ["podman", "logs", "--timestamps", container_name]
    try:
        inspect_result = subprocess.run(
            inspect_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        log_result = subprocess.run(
            log_command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("terminal failure evidence could not be captured") from error
    inspect_bytes = inspect_result.stdout
    log_bytes = log_result.stdout
    if not isinstance(inspect_bytes, bytes) or not isinstance(log_bytes, bytes):
        raise ValueError("terminal failure capture did not return bytes")
    _reject_secrets(inspect_bytes, log_bytes)

    from trec_mlm.llm_ranker_pilot import preflight_pilot_inputs

    selection, _ = preflight_pilot_inputs(selection_path)
    if selection.get("stage") != 2:
        raise ValueError("terminal failure requires the frozen Stage 2 selection")
    prompt_rows = _expected_prompt_rows(selection)
    planned_request_count = len(prompt_rows)
    if planned_request_count != 36:
        raise ValueError("terminal failure planned request count changed")
    container = _container_facts(
        _load_inspect(inspect_bytes),
        container_name=container_name,
        selection_path=selection_path,
        smoke_receipt_path=smoke_receipt_path,
        expected_ranking_dir=expected_ranking_dir,
    )
    failure = _failure_facts(
        log_bytes, planned_request_count=planned_request_count
    )
    if failure["query_id"] not in {
        entry["query_id"] for entry in selection["questions"]
    } or failure["rotation"] not in (0, 1, 2):
        raise ValueError("terminal failure query/rotation changed")
    ranking_dir = Path(expected_ranking_dir)
    persisted = [
        name for name in _FORBIDDEN_RANKING_FILES if (ranking_dir / name).exists()
    ]
    if persisted:
        raise ValueError("terminal failure cannot coexist with persisted ranking data")

    expected_prompt_bytes = _jsonl_bytes(prompt_rows)
    observed_payloads = {
        "container.log": log_bytes,
        "expected_prompts.jsonl": expected_prompt_bytes,
        "inspect.json": inspect_bytes,
    }
    receipt = {
        "version": 1,
        "artifact_type": "terminal_listwise_failure",
        "qrel_free_phase": True,
        "method": "listwise",
        "stage": 2,
        "disposition": "unscorable_final_attempt",
        "rerun_permitted": False,
        "inventory": list(_INVENTORY),
        "artifact_sha256": {
            name: _sha256_bytes(payload)
            for name, payload in sorted(observed_payloads.items())
        },
        "selection": {
            "path": str(selection_path),
            "sha256": _sha256(selection_path),
        },
        "smoke_receipt": {
            "path": str(smoke_receipt_path),
            "sha256": _sha256(smoke_receipt_path),
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "dtype": MODEL_DTYPE,
        },
        "code_revision": container.pop("code_revision"),
        "container": container,
        "capture_commands": {
            "inspect": inspect_command,
            "logs": log_command,
            "logs_combined_stdout_stderr": True,
        },
        "execution": {
            "planned_request_count": planned_request_count,
            "log_observed_request_count": failure.pop(
                "log_observed_request_count"
            ),
            "persisted_raw_output_count": 0,
            "rankings_available": False,
        },
        "expected_prompts": {
            "count": planned_request_count,
            "evidence_status": "expected_not_observed",
            "provenance": "regenerated_qrel_free",
        },
        "failure": failure,
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent)
    )
    try:
        for name, payload in observed_payloads.items():
            (temporary / name).write_bytes(payload)
        (temporary / "failure_receipt.json").write_bytes(_json_bytes(receipt))
        os.replace(temporary, destination)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return validate_terminal_listwise_failure(destination)


def validate_terminal_listwise_failure(
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Rehash and semantically cross-check a terminal listwise failure capture."""

    source = Path(artifact_dir)
    actual_inventory = sorted(path.name for path in source.iterdir())
    if actual_inventory != sorted(_INVENTORY):
        raise ValueError("terminal failure inventory changed")
    try:
        receipt = json.loads((source / "failure_receipt.json").read_text())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("terminal failure receipt is not valid JSON") from error
    if not isinstance(receipt, dict):
        raise ValueError("terminal failure receipt must be an object")
    if set(receipt) != _RECEIPT_KEYS:
        raise ValueError("terminal failure receipt schema changed")
    if receipt.get("inventory") != list(_INVENTORY):
        raise ValueError("terminal failure receipt inventory changed")
    expected_identity = {
        "version": 1,
        "artifact_type": "terminal_listwise_failure",
        "qrel_free_phase": True,
        "method": "listwise",
        "stage": 2,
        "disposition": "unscorable_final_attempt",
        "rerun_permitted": False,
    }
    if any(receipt.get(key) != value for key, value in expected_identity.items()):
        raise ValueError("terminal failure disposition changed")
    hashes = receipt.get("artifact_sha256")
    if not isinstance(hashes, Mapping) or set(hashes) != set(_OBSERVED_FILES):
        raise ValueError("terminal failure hash inventory changed")
    for name in _OBSERVED_FILES:
        if _sha256(source / name) != hashes[name]:
            raise ValueError(f"terminal failure hash mismatch: {name}")

    inspect_bytes = (source / "inspect.json").read_bytes()
    log_bytes = (source / "container.log").read_bytes()
    _reject_secrets(inspect_bytes, log_bytes)
    selection_record = receipt.get("selection")
    smoke_record = receipt.get("smoke_receipt")
    if not isinstance(selection_record, Mapping) or not isinstance(
        smoke_record, Mapping
    ) or set(selection_record) != {"path", "sha256"} or set(smoke_record) != {
        "path",
        "sha256",
    }:
        raise ValueError("terminal failure input receipt changed")
    for record, label in (
        (selection_record, "selection"),
        (smoke_record, "smoke receipt"),
    ):
        path = Path(str(record.get("path")))
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(f"terminal failure {label} hash mismatch")
    from trec_mlm.llm_ranker_pilot import preflight_pilot_inputs
    from trec_mlm.artifact_manifest import validate_pilot_smoke_receipt

    selection, _ = preflight_pilot_inputs(selection_record["path"])
    validate_pilot_smoke_receipt(smoke_record["path"])
    prompt_rows = _expected_prompt_rows(selection)
    if (source / "expected_prompts.jsonl").read_bytes() != _jsonl_bytes(prompt_rows):
        raise ValueError("terminal failure expected prompts changed")
    expected_prompts = receipt.get("expected_prompts")
    if expected_prompts != {
        "count": len(prompt_rows),
        "evidence_status": "expected_not_observed",
        "provenance": "regenerated_qrel_free",
    }:
        raise ValueError("terminal failure expected prompt receipt changed")

    inspected = _load_inspect(inspect_bytes)
    command_tokens = receipt.get("container", {}).get("command_tokens")
    if not isinstance(receipt.get("container"), Mapping) or set(
        receipt["container"]
    ) != {
        "name",
        "id",
        "image",
        "path",
        "args",
        "command_tokens",
        "started_at",
        "finished_at",
        "status",
        "exit_code",
        "oom_killed",
    } or not isinstance(command_tokens, list):
        raise ValueError("terminal failure command receipt changed")
    try:
        expected_ranking_dir = command_tokens[
            command_tokens.index("--output-dir") + 1
        ]
    except (ValueError, IndexError) as error:
        raise ValueError("terminal failure command receipt changed") from error
    container = _container_facts(
        inspected,
        container_name=str(receipt.get("container", {}).get("name")),
        selection_path=selection_record["path"],
        smoke_receipt_path=smoke_record["path"],
        expected_ranking_dir=expected_ranking_dir,
    )
    code_revision = container.pop("code_revision")
    if container != receipt.get("container"):
        raise ValueError("terminal failure container identity changed")
    if code_revision != receipt.get("code_revision"):
        raise ValueError("terminal failure code revision changed")
    model = receipt.get("model")
    if model != {
        "id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": MODEL_DTYPE,
    }:
        raise ValueError("terminal failure model identity changed")
    if receipt.get("capture_commands") != {
        "inspect": ["podman", "inspect", container["name"]],
        "logs": ["podman", "logs", "--timestamps", container["name"]],
        "logs_combined_stdout_stderr": True,
    }:
        raise ValueError("terminal failure capture command changed")
    failure = _failure_facts(log_bytes, planned_request_count=len(prompt_rows))
    observed_count = failure.pop("log_observed_request_count")
    if failure != receipt.get("failure"):
        raise ValueError("terminal failure log schema changed")
    if failure["query_id"] not in {
        entry["query_id"] for entry in selection["questions"]
    } or failure["rotation"] not in (0, 1, 2):
        raise ValueError("terminal failure query/rotation changed")
    if receipt.get("execution") != {
        "planned_request_count": len(prompt_rows),
        "log_observed_request_count": observed_count,
        "persisted_raw_output_count": 0,
        "rankings_available": False,
    }:
        raise ValueError("terminal failure execution receipt changed")
    ranking_dir = Path(expected_ranking_dir)
    if any((ranking_dir / name).exists() for name in _FORBIDDEN_RANKING_FILES):
        raise ValueError("terminal failure cannot coexist with persisted ranking data")
    return receipt
