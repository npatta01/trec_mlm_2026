"""Loopback-only OpenAI-compatible calls and resumable pointwise judging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import http.client
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time
from typing import Any, Protocol
import urllib.error
import urllib.parse
import urllib.request

from trec_mlm.pointwise_judge import (
    CandidateCase,
    PromptContract,
    decision_to_dict,
    parse_judge_decision,
    render_judge_messages,
)


# Pointwise decisions are small; one MiB allows full raw retention without an
# unbounded read from a malformed or hostile local server.
MAX_RESPONSE_BYTES = 1024 * 1024
_IDENTITY_KEYS = {
    "input_sha256",
    "prompt_sha256",
    "schema_sha256",
    "model_alias",
    "model_id",
    "runtime_receipt_sha256",
}
_RECORD_KEYS = _IDENTITY_KEYS | {
    "attempts",
    "decision",
    "failure",
    "elapsed_seconds",
    "usage",
}
_RUN_KEYS = {
    "version",
    "model_alias",
    "model_id",
    "prompt_sha256",
    "schema_sha256",
    "runtime_receipt_sha256",
    "records",
}
_RUNTIME_RECEIPT_KEYS = {
    "version",
    "passed",
    "model_alias",
    "model_id",
    "registry_sha256",
    "schema_sha256",
    "launch_evidence_sha256",
    "runtime",
    "hardware",
    "checkpoint",
    "endpoint",
    "configuration",
    "canary",
    "startup_seconds",
    "download_occurred",
    "receipt_sha256",
}
_LOCAL_MODEL_ALIASES = {
    "qwen36-35b-a3b-gguf",
    "qwen36-27b-fp8",
    "gpt-oss-20b",
    "gemma4-31b-q4",
}
_REGISTRY_ENTRY_KEYS = {
    "model_id",
    "runtime",
    "new_download_authorized",
    "checkpoint",
    "configuration",
}
_FROZEN_CONFIGURATION = {
    "context_tokens": 4096,
    "temperature": 0,
    "max_output_tokens": 384,
    "max_concurrency": 1,
}
FROZEN_DECISION_SCHEMA_SHA256 = (
    "84557061030827f09f94063a85ccbaa5324797b958963af503342ae1fce6c69b"
)


class JudgeCall(Protocol):
    """One schema-constrained call to an exact model."""

    def __call__(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, object],
    ) -> tuple[str, Mapping[str, object]]: ...


@dataclass(frozen=True, slots=True)
class _ReusableRecord:
    terminal: bool
    attempts: list[dict[str, object]]
    usage: dict[str, int | float]
    repair_raw: str | None
    elapsed_seconds: float


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


def _open_no_redirect(request: urllib.request.Request, timeout: float):
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _RejectRedirects
    ).open(request, timeout=timeout)


def validate_loopback_endpoint(url: str) -> str:
    """Return a normalized v1 base URL only for a literal loopback host."""
    if type(url) is not str:
        raise ValueError("endpoint must be a literal loopback URL")
    try:
        parsed = urllib.parse.urlsplit(url)
        parsed.port
    except ValueError as error:
        raise ValueError("endpoint must be a valid loopback URL") from error
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("endpoint must use HTTP on a literal loopback host")
    if parsed.hostname not in {"127.0.0.1", "::1"}:
        raise ValueError("endpoint host must be literal loopback 127.0.0.1 or ::1")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("endpoint must not contain userinfo")
    if parsed.query or parsed.fragment:
        raise ValueError("endpoint must not contain a query or fragment")
    if parsed.path.rstrip("/") != "/v1":
        raise ValueError("endpoint path must be /v1")
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, "/v1", "", ""))


def call_openai_judge(
    endpoint: str,
    model: str,
    messages: Sequence[Mapping[str, str]],
    schema: Mapping[str, object],
    *,
    timeout_seconds: float,
) -> tuple[str, Mapping[str, object]]:
    """Make one strict, non-redirecting OpenAI-compatible loopback request."""
    endpoint = validate_loopback_endpoint(endpoint)
    request_url = f"{endpoint}/chat/completions"
    payload = {
        "model": model,
        "messages": [dict(message) for message in messages],
        "temperature": 0,
        "max_tokens": 384,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pointwise_judge_decision",
                "strict": True,
                "schema": dict(schema),
            },
        },
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    request = urllib.request.Request(
        request_url,
        data=encoded,
        headers={"Accept": "application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with _open_no_redirect(request, timeout_seconds) as response:
            if response.geturl() != request_url:
                raise ValueError("redirect responses are not allowed")
            if not 200 <= response.status < 300:
                raise ValueError(f"judge endpoint returned HTTP {response.status}")
            declared_length = _declared_content_length(response.headers)
            try:
                raw_body = response.read(MAX_RESPONSE_BYTES + 1)
            except http.client.IncompleteRead as error:
                raise ValueError(
                    "judge response did not satisfy declared Content-Length"
                ) from error
            if len(raw_body) > MAX_RESPONSE_BYTES:
                raise ValueError("judge response body is too large")
            if declared_length is not None and len(raw_body) != declared_length:
                raise ValueError(
                    "judge response did not satisfy declared Content-Length"
                )
    except urllib.error.HTTPError as error:
        if 300 <= error.code < 400:
            raise ValueError("redirect responses are not allowed") from error
        raise RuntimeError(f"judge endpoint returned HTTP {error.code}") from error
    except (TimeoutError, urllib.error.URLError) as error:
        raise RuntimeError("judge endpoint request failed") from error

    try:
        body = json.loads(
            raw_body.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("judge response body is not valid UTF-8 JSON") from error
    if type(body) is not dict:
        raise ValueError("judge response body must be a JSON object")
    if type(body.get("model")) is not str or body["model"] != model:
        raise ValueError("judge response model does not match exact requested model")
    choices = body.get("choices")
    if type(choices) is not list or not choices:
        raise ValueError("judge response must contain nonempty choices")
    first = choices[0]
    if type(first) is not dict or type(first.get("message")) is not dict:
        raise ValueError("judge response choice must contain a message")
    content = first["message"].get("content")
    if type(content) is not str or not content.strip():
        raise ValueError("judge response content must not be empty")
    _validated_usage(body.get("usage"))
    return content, body


def _declared_content_length(headers: object) -> int | None:
    if not hasattr(headers, "get"):
        raise ValueError("judge response headers are unavailable")
    raw = headers.get("Content-Length")
    if raw is None:
        return None
    if type(raw) is not str or not raw.isdecimal():
        raise ValueError("judge response Content-Length is invalid")
    length = int(raw)
    if length > MAX_RESPONSE_BYTES:
        raise ValueError("judge response body is too large")
    return length


def validate_local_model_registry(
    registry: Mapping[str, object], *, model_alias: str, model_id: str
) -> dict[str, object]:
    """Validate Task 4's version-1 registry and return the selected descriptor."""
    value = dict(registry)
    if set(value) != {"version", "models"}:
        raise ValueError("local model registry must contain exact version-1 keys")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("local model registry version must be 1")
    models = value["models"]
    if type(models) is not dict or set(models) != _LOCAL_MODEL_ALIASES:
        raise ValueError("local model registry must contain the exact approved aliases")
    validated: dict[str, dict[str, object]] = {}
    for alias, untrusted_entry in models.items():
        entry = _exact_mapping(
            untrusted_entry, _REGISTRY_ENTRY_KEYS, f"local model registry entry {alias}"
        )
        _require_nonempty_strings(
            entry, {"model_id", "runtime"}, f"local model registry entry {alias}"
        )
        if type(entry["new_download_authorized"]) is not bool:
            raise ValueError("local model registry download authorization is invalid")
        checkpoint = _exact_mapping(
            entry["checkpoint"],
            {"revision", "gguf_filename", "expected_bytes", "sha256"},
            f"local model registry checkpoint {alias}",
        )
        revision = checkpoint["revision"]
        filename = checkpoint["gguf_filename"]
        if not (
            (type(revision) is str and bool(revision) and filename is None)
            or (revision is None and type(filename) is str and bool(filename))
        ):
            raise ValueError(
                "local model registry checkpoint must pin one revision or GGUF filename"
            )
        _require_nonnegative_integer(
            checkpoint["expected_bytes"],
            "local model registry expected checkpoint bytes",
            positive=True,
        )
        if not _is_sha256(checkpoint["sha256"]):
            raise ValueError("local model registry checkpoint hash is invalid")
        configuration = _exact_mapping(
            entry["configuration"],
            set(_FROZEN_CONFIGURATION),
            f"local model registry configuration {alias}",
        )
        if not _is_frozen_configuration(configuration):
            raise ValueError("local model registry configuration is not frozen")
        validated[alias] = entry
    if model_alias not in validated:
        raise ValueError("requested model alias is not in local model registry")
    selected = validated[model_alias]
    if selected["model_id"] != model_id:
        raise ValueError("local model registry does not match requested model ID")
    return selected


def validate_runtime_receipt(
    receipt: Mapping[str, object],
    *,
    model_alias: str,
    model_id: str,
    endpoint: str,
    registry_sha256: str,
    registry_entry: Mapping[str, object],
) -> dict[str, object]:
    """Validate the version-1 preflight evidence required before inference."""
    value = dict(receipt)
    if set(value) != _RUNTIME_RECEIPT_KEYS:
        raise ValueError("runtime receipt must contain exact version-1 keys")
    body = dict(value)
    stored_hash = body.pop("receipt_sha256")
    if not _is_sha256(stored_hash) or stored_hash != _sha256_json(body):
        raise ValueError("runtime receipt hash does not match its body")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("runtime receipt version must be 1")
    if value["passed"] is not True:
        raise ValueError("runtime receipt must record a successful preflight")
    if value["model_alias"] != model_alias or value["model_id"] != model_id:
        raise ValueError("runtime receipt does not match requested model identity")
    if (
        not _is_sha256(value["registry_sha256"])
        or value["registry_sha256"] != registry_sha256
    ):
        raise ValueError("runtime receipt registry hash is not externally anchored")
    if value["schema_sha256"] != FROZEN_DECISION_SCHEMA_SHA256:
        raise ValueError("runtime receipt schema hash is not frozen")
    if not _is_sha256(value["launch_evidence_sha256"]):
        raise ValueError("runtime receipt launch evidence hash is invalid")

    runtime = _exact_mapping(
        value["runtime"],
        {"name", "version", "backend_version"},
        "runtime receipt runtime",
    )
    _require_nonempty_strings(runtime, set(runtime), "runtime receipt runtime")
    if runtime["name"] != registry_entry["runtime"]:
        raise ValueError("runtime receipt runtime does not match registry")
    hardware = _exact_mapping(
        value["hardware"],
        {"gpu_arch", "visible_memory_bytes", "free_disk_bytes"},
        "runtime receipt hardware",
    )
    if hardware["gpu_arch"] != "gfx1151":
        raise ValueError("runtime receipt must prove gfx1151 visibility")
    _require_nonnegative_integer(
        hardware["visible_memory_bytes"],
        "runtime receipt visible memory",
        positive=True,
    )
    _require_nonnegative_integer(
        hardware["free_disk_bytes"], "runtime receipt free disk"
    )

    checkpoint = _exact_mapping(
        value["checkpoint"],
        {"path", "revision", "size_bytes", "sha256"},
        "runtime receipt checkpoint",
    )
    _require_nonempty_strings(
        checkpoint, {"path", "revision"}, "runtime receipt checkpoint"
    )
    _require_nonnegative_integer(
        checkpoint["size_bytes"], "runtime receipt checkpoint size", positive=True
    )
    if not _is_sha256(checkpoint["sha256"]):
        raise ValueError("runtime receipt checkpoint hash is invalid")
    registry_checkpoint = registry_entry["checkpoint"]
    if (
        checkpoint["size_bytes"] != registry_checkpoint["expected_bytes"]
        or checkpoint["sha256"] != registry_checkpoint["sha256"]
    ):
        raise ValueError("runtime receipt checkpoint does not match registry")
    registry_revision = registry_checkpoint["revision"]
    registry_filename = registry_checkpoint["gguf_filename"]
    if registry_revision is not None and checkpoint["revision"] != registry_revision:
        raise ValueError("runtime receipt checkpoint revision does not match registry")
    if (
        registry_filename is not None
        and Path(checkpoint["path"]).name != registry_filename
    ):
        raise ValueError("runtime receipt GGUF filename does not match registry")

    endpoint_record = _exact_mapping(
        value["endpoint"], {"base_url", "models"}, "runtime receipt endpoint"
    )
    try:
        receipt_endpoint = validate_loopback_endpoint(endpoint_record["base_url"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "runtime receipt endpoint is not literal loopback /v1"
        ) from error
    if receipt_endpoint != validate_loopback_endpoint(endpoint):
        raise ValueError("runtime receipt endpoint does not match requested endpoint")
    models = endpoint_record["models"]
    if type(models) is not list or models != [model_id]:
        raise ValueError("runtime receipt endpoint must list the exact requested model")

    configuration = _exact_mapping(
        value["configuration"],
        {"context_tokens", "temperature", "max_output_tokens", "max_concurrency"},
        "runtime receipt configuration",
    )
    if not _is_frozen_configuration(configuration):
        raise ValueError("runtime receipt does not match frozen decoding configuration")
    if configuration != registry_entry["configuration"]:
        raise ValueError("runtime receipt configuration does not match registry")

    canary = _exact_mapping(
        value["canary"], {"passed", "raw", "parsed"}, "runtime receipt canary"
    )
    if canary["passed"] is not True or type(canary["raw"]) is not str:
        raise ValueError("runtime receipt canary did not pass")
    try:
        parsed_canary = decision_to_dict(parse_judge_decision(canary["raw"]))
    except ValueError as error:
        raise ValueError("runtime receipt canary raw decision is invalid") from error
    try:
        canary_matches = _json_values_equal(canary["parsed"], parsed_canary)
    except ValueError as error:
        raise ValueError("runtime receipt canary parsed decision is invalid") from error
    if not canary_matches:
        raise ValueError("runtime receipt canary parsed decision does not match raw")
    startup = value["startup_seconds"]
    if type(startup) not in {int, float} or not math.isfinite(startup) or startup < 0:
        raise ValueError("runtime receipt startup_seconds is invalid")
    if type(value["download_occurred"]) is not bool:
        raise ValueError("runtime receipt download flag is invalid")
    if value["download_occurred"] and not registry_entry["new_download_authorized"]:
        raise ValueError("runtime receipt records an unauthorized download")
    return value


def _is_frozen_configuration(value: Mapping[str, object]) -> bool:
    return (
        value == _FROZEN_CONFIGURATION
        and all(
            type(value[key]) is int
            for key in ("context_tokens", "max_output_tokens", "max_concurrency")
        )
        and type(value["temperature"]) in {int, float}
    )


def run_cases(
    cases: Sequence[CandidateCase],
    *,
    model_alias: str,
    model_id: str,
    contract: PromptContract,
    output_path: Path,
    runtime_receipt: Mapping[str, object],
    call: JudgeCall,
) -> dict[str, object]:
    """Judge cases with atomic checkpoints and exact content-addressed resume."""
    output_path = Path(output_path)
    _prepare_private_parent(output_path.parent)
    existing = _load_existing_run(output_path)
    if existing is not None:
        _validate_run_container(existing)
    existing_records = existing.get("records", {}) if existing is not None else {}

    prompt_sha256 = hashlib.sha256(contract.prompt.encode("utf-8")).hexdigest()
    schema_sha256 = _sha256_json(contract.schema)
    runtime_receipt_sha256 = _sha256_json(dict(runtime_receipt))
    identities = {
        case.response_id: _record_identity(
            case,
            model_alias=model_alias,
            model_id=model_id,
            prompt_sha256=prompt_sha256,
            schema_sha256=schema_sha256,
            runtime_receipt_sha256=runtime_receipt_sha256,
        )
        for case in cases
    }
    if len(identities) != len(cases):
        raise ValueError("cases must have unique response IDs")

    records: dict[str, object] = {}
    reusable_records: dict[str, _ReusableRecord] = {}
    for response_id, identity in identities.items():
        cached = existing_records.get(response_id)
        reusable = _validate_reusable_record(cached, identity, response_id)
        if reusable is not None:
            if existing is None:
                raise RuntimeError("unreachable reusable record without existing run")
            _require_reusable_container_identity(existing, identity)
            records[response_id] = cached
            reusable_records[response_id] = reusable
    result: dict[str, object] = {
        "version": 1,
        "model_alias": model_alias,
        "model_id": model_id,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "runtime_receipt_sha256": runtime_receipt_sha256,
        "records": records,
    }
    _write_json_atomically(output_path, result)

    for case in cases:
        identity = identities[case.response_id]
        reusable = reusable_records.get(case.response_id)
        if reusable is not None and reusable.terminal:
            continue
        started = time.perf_counter()
        if reusable is None:
            attempts: list[dict[str, object]] = []
            usage: dict[str, int | float] = {}
            malformed_raw: str | None = None
            prior_elapsed = 0.0
        else:
            attempts = reusable.attempts
            usage = reusable.usage
            malformed_raw = reusable.repair_raw
            prior_elapsed = reusable.elapsed_seconds
        decision: dict[str, object] | None = None
        failure: dict[str, str] | None = None
        for attempt_number in range(len(attempts), 2):
            messages = render_judge_messages(
                case,
                contract,
                repair_raw=malformed_raw if attempt_number else None,
            )
            raw, response_body = call(model_id, messages, contract.schema)
            if type(raw) is not str or not isinstance(response_body, Mapping):
                raise ValueError("judge call returned an invalid result")
            body = dict(response_body)
            attempt: dict[str, object] = {
                "raw": raw,
                "response_body": body,
                "parsed": None,
            }
            _merge_usage(usage, _validated_usage(body.get("usage")))
            try:
                parsed = decision_to_dict(parse_judge_decision(raw))
            except ValueError as error:
                attempt["error"] = str(error)
                attempts.append(attempt)
                malformed_raw = raw
                if attempt_number == 1:
                    failure = {
                        "type": "invalid_decision_after_repair",
                        "message": str(error),
                    }
                else:
                    records[case.response_id] = {
                        **identity,
                        "attempts": attempts,
                        "decision": None,
                        "failure": None,
                        "elapsed_seconds": prior_elapsed
                        + time.perf_counter()
                        - started,
                        "usage": usage,
                    }
                    _write_json_atomically(output_path, result)
            else:
                attempt["parsed"] = parsed
                attempts.append(attempt)
                decision = parsed
                break

        record = {
            **identity,
            "attempts": attempts,
            "decision": decision,
            "failure": failure,
            "elapsed_seconds": prior_elapsed + time.perf_counter() - started,
            "usage": usage,
        }
        records[case.response_id] = record
        _write_json_atomically(output_path, result)
    return result


def _validated_usage(value: object) -> dict[str, int | float]:
    if type(value) is not dict:
        raise ValueError("judge response usage must be a JSON object")
    result: dict[str, int | float] = {}
    for key, amount in value.items():
        if type(key) is not str or type(amount) not in {int, float}:
            raise ValueError("judge response usage values must be numeric")
        if not math.isfinite(amount) or amount < 0:
            raise ValueError(
                "judge response usage values must be finite and nonnegative"
            )
        result[key] = amount
    return result


def _merge_usage(
    total: dict[str, int | float], current: Mapping[str, int | float]
) -> None:
    for key, amount in current.items():
        total[key] = total.get(key, 0) + amount


def _record_identity(
    case: CandidateCase,
    *,
    model_alias: str,
    model_id: str,
    prompt_sha256: str,
    schema_sha256: str,
    runtime_receipt_sha256: str,
) -> dict[str, str]:
    return {
        "input_sha256": _sha256_json(asdict(case)),
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "model_alias": model_alias,
        "model_id": model_id,
        "runtime_receipt_sha256": runtime_receipt_sha256,
    }


def _validate_run_container(value: dict[str, Any]) -> None:
    if set(value) != _RUN_KEYS:
        raise ValueError("corrupt existing run: expected exact top-level keys")
    if type(value["version"]) is not int or value["version"] != 1:
        raise ValueError("corrupt existing run: unsupported version")
    if any(
        type(value[key]) is not str or not value[key]
        for key in ("model_alias", "model_id")
    ):
        raise ValueError("corrupt existing run: invalid model identity")
    if any(
        not _is_sha256(value[key])
        for key in ("prompt_sha256", "schema_sha256", "runtime_receipt_sha256")
    ):
        raise ValueError("corrupt existing run: invalid content hash")
    if type(value["records"]) is not dict:
        raise ValueError("corrupt existing run: records must be an object")


def _require_reusable_container_identity(
    container: Mapping[str, object], identity: Mapping[str, str]
) -> None:
    expected = {
        "model_alias": identity["model_alias"],
        "model_id": identity["model_id"],
        "prompt_sha256": identity["prompt_sha256"],
        "schema_sha256": identity["schema_sha256"],
        "runtime_receipt_sha256": identity["runtime_receipt_sha256"],
    }
    if any(container[key] != value for key, value in expected.items()):
        raise ValueError(
            "corrupt existing run: container conflicts with reusable record"
        )


def _validate_reusable_record(
    value: object, identity: Mapping[str, str], response_id: str
) -> _ReusableRecord | None:
    if value is None:
        return None
    if type(value) is not dict or not _IDENTITY_KEYS <= set(value):
        raise ValueError(f"corrupt reusable record {response_id}: missing identity")
    if any(value.get(key) != expected for key, expected in identity.items()):
        return None
    try:
        return _validated_same_identity_record(value, identity)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"corrupt reusable record {response_id}: {error}") from error


def _validated_same_identity_record(
    value: dict[str, Any], identity: Mapping[str, str]
) -> _ReusableRecord:
    if set(value) != _RECORD_KEYS:
        raise ValueError("expected exact record keys")
    if any(value[key] != expected for key, expected in identity.items()):
        raise ValueError("identity changed during validation")
    if any(
        not _is_sha256(value[key])
        for key in (
            "input_sha256",
            "prompt_sha256",
            "schema_sha256",
            "runtime_receipt_sha256",
        )
    ):
        raise ValueError("invalid identity hash")
    if any(
        type(value[key]) is not str or not value[key]
        for key in ("model_alias", "model_id")
    ):
        raise ValueError("invalid model identity")
    elapsed = value["elapsed_seconds"]
    if type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0:
        raise ValueError("invalid elapsed_seconds")
    attempts = value["attempts"]
    if type(attempts) is not list or not 1 <= len(attempts) <= 2:
        raise ValueError("invalid attempt count")

    validated_attempts: list[dict[str, object]] = []
    stages: list[str] = []
    parsed_decisions: list[dict[str, object] | None] = []
    errors: list[str | None] = []
    aggregate_usage: dict[str, int | float] = {}
    for attempt in attempts:
        validated, stage, parsed, error, usage = _validated_cached_attempt(
            attempt, value["model_id"]
        )
        validated_attempts.append(validated)
        stages.append(stage)
        parsed_decisions.append(parsed)
        errors.append(error)
        _merge_usage(aggregate_usage, usage)

    recorded_usage = _validated_usage(value["usage"])
    if recorded_usage != aggregate_usage:
        raise ValueError("record usage does not equal attempt usage")
    decision = value["decision"]
    failure = value["failure"]
    if stages == ["valid"]:
        _require_success_outcome(decision, failure, parsed_decisions[0])
        terminal = True
        repair_raw = None
    elif stages == ["invalid"]:
        if decision is not None or failure is not None:
            raise ValueError("partial record must not contain an outcome")
        terminal = False
        repair_raw = validated_attempts[0]["raw"]
    elif stages == ["invalid", "valid"]:
        _require_success_outcome(decision, failure, parsed_decisions[1])
        terminal = True
        repair_raw = None
    elif stages == ["invalid", "invalid"]:
        if decision is not None:
            raise ValueError("failed record must not contain a decision")
        expected_failure = {
            "type": "invalid_decision_after_repair",
            "message": errors[1],
        }
        if failure != expected_failure:
            raise ValueError("invalid terminal failure")
        terminal = True
        repair_raw = None
    else:
        raise ValueError("invalid attempt stage sequence")
    return _ReusableRecord(
        terminal=terminal,
        attempts=validated_attempts,
        usage=recorded_usage,
        repair_raw=repair_raw,
        elapsed_seconds=float(elapsed),
    )


def _validated_cached_attempt(
    value: object, model_id: str
) -> tuple[
    dict[str, object],
    str,
    dict[str, object] | None,
    str | None,
    dict[str, int | float],
]:
    if type(value) is not dict or type(value.get("raw")) is not str:
        raise ValueError("invalid attempt structure")
    raw = value["raw"]
    body = value.get("response_body")
    usage = _validated_cached_response_body(body, model_id, raw)
    try:
        parsed = decision_to_dict(parse_judge_decision(raw))
    except ValueError as error:
        if set(value) != {"raw", "response_body", "parsed", "error"}:
            raise ValueError("invalid failed-attempt keys") from error
        if value["parsed"] is not None or value["error"] != str(error):
            raise ValueError("failed attempt does not match raw output") from error
        return dict(value), "invalid", None, str(error), usage
    if set(value) != {"raw", "response_body", "parsed"}:
        raise ValueError("invalid successful-attempt keys")
    if not _json_values_equal(value["parsed"], parsed):
        raise ValueError("parsed attempt does not match raw output")
    return dict(value), "valid", parsed, None, usage


def _validated_cached_response_body(
    value: object, model_id: str, raw: str
) -> dict[str, int | float]:
    if type(value) is not dict or value.get("model") != model_id:
        raise ValueError("attempt response has wrong model")
    choices = value.get("choices")
    if type(choices) is not list or not choices or type(choices[0]) is not dict:
        raise ValueError("attempt response has invalid choices")
    message = choices[0].get("message")
    if type(message) is not dict or message.get("content") != raw or not raw.strip():
        raise ValueError("attempt response content does not match raw output")
    return _validated_usage(value.get("usage"))


def _require_success_outcome(
    decision: object,
    failure: object,
    parsed: dict[str, object] | None,
) -> None:
    if (
        failure is not None
        or parsed is None
        or not _json_values_equal(decision, parsed)
    ):
        raise ValueError("successful outcome does not match parsed attempt")


def _json_values_equal(left: object, right: object) -> bool:
    """Compare JSON values structurally without Python's bool/number coercions."""
    for value in (left, right):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("non-finite number is not a valid trusted JSON value")
    if type(left) is not type(right):
        return False
    if left is None or type(left) in {str, bool, int, float}:
        return left == right
    if type(left) is list:
        return len(left) == len(right) and all(
            _json_values_equal(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    if type(left) is dict:
        if any(type(key) is not str for key in left) or any(
            type(key) is not str for key in right
        ):
            raise ValueError("trusted JSON object keys must be strings")
        return set(left) == set(right) and all(
            _json_values_equal(left[key], right[key]) for key in left
        )
    raise ValueError("unsupported value at trusted JSON boundary")


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _exact_mapping(value: object, keys: set[str], description: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{description} must contain exact keys")
    return value


def _require_nonempty_strings(
    value: Mapping[str, object], keys: set[str], description: str
) -> None:
    if any(type(value[key]) is not str or not value[key] for key in keys):
        raise ValueError(f"{description} values must be nonempty strings")


def _require_nonnegative_integer(
    value: object, description: str, *, positive: bool = False
) -> None:
    minimum = 1 if positive else 0
    if type(value) is not int or value < minimum:
        raise ValueError(f"{description} must be an integer at least {minimum}")


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prepare_private_parent(parent: Path) -> None:
    if parent.exists():
        mode = parent.lstat().st_mode
        if not stat.S_ISDIR(mode):
            raise ValueError("run output parent must be a directory")
        if stat.S_IMODE(mode) != 0o700:
            raise ValueError("run output parent must have mode 0700")
        return
    parent.mkdir(parents=True, mode=0o700)
    parent.chmod(0o700)


def _load_existing_run(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    mode = path.lstat().st_mode
    if not stat.S_ISREG(mode):
        raise ValueError("existing run output must be a regular file")
    if stat.S_IMODE(mode) != 0o600:
        raise ValueError("existing run output must have mode 0600")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise ValueError("existing run output is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError("existing run output must be a JSON object")
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key in run output: {key}")
        value[key] = item
    return value


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _write_json_atomically(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)
