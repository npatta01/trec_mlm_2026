"""Loopback-only OpenAI-compatible calls and resumable pointwise judging."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import hashlib
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


class JudgeCall(Protocol):
    """One schema-constrained call to an exact model."""

    def __call__(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, object],
    ) -> tuple[str, Mapping[str, object]]: ...


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
            raw_body = response.read()
    except urllib.error.HTTPError as error:
        if 300 <= error.code < 400:
            raise ValueError("redirect responses are not allowed") from error
        raise RuntimeError(f"judge endpoint returned HTTP {error.code}") from error
    except (TimeoutError, urllib.error.URLError) as error:
        raise RuntimeError("judge endpoint request failed") from error

    try:
        body = json.loads(raw_body.decode("utf-8"))
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
    existing_records = existing.get("records", {}) if existing is not None else {}
    if type(existing_records) is not dict:
        raise ValueError("existing run records must be a JSON object")

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
    for response_id, identity in identities.items():
        cached = existing_records.get(response_id)
        if _record_matches(cached, identity) or _partial_state(cached, identity):
            records[response_id] = cached
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
        cached = records.get(case.response_id)
        if _record_matches(cached, identity):
            continue
        started = time.perf_counter()
        partial = _partial_state(cached, identity)
        if partial is None:
            attempts: list[dict[str, object]] = []
            usage: dict[str, int | float] = {}
            malformed_raw: str | None = None
            prior_elapsed = 0.0
        else:
            attempts, usage, malformed_raw, prior_elapsed = partial
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


def _record_matches(value: object, identity: Mapping[str, str]) -> bool:
    if type(value) is not dict:
        return False
    if any(value.get(key) != expected for key, expected in identity.items()):
        return False
    attempts = value.get("attempts")
    has_outcome = value.get("decision") is not None or value.get("failure") is not None
    return type(attempts) is list and bool(attempts) and has_outcome


def _partial_state(
    value: object, identity: Mapping[str, str]
) -> tuple[list[dict[str, object]], dict[str, int | float], str, float] | None:
    if type(value) is not dict:
        return None
    if any(value.get(key) != expected for key, expected in identity.items()):
        return None
    attempts = value.get("attempts")
    if type(attempts) is not list or len(attempts) != 1:
        return None
    attempt = attempts[0]
    if (
        type(attempt) is not dict
        or type(attempt.get("raw")) is not str
        or type(attempt.get("response_body")) is not dict
        or attempt.get("parsed") is not None
        or value.get("decision") is not None
        or value.get("failure") is not None
    ):
        return None
    elapsed = value.get("elapsed_seconds")
    if type(elapsed) not in {int, float} or not math.isfinite(elapsed) or elapsed < 0:
        return None
    try:
        usage = _validated_usage(value.get("usage"))
    except ValueError:
        return None
    return [dict(attempt)], usage, attempt["raw"], float(elapsed)


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
        if not parent.is_dir():
            raise ValueError("run output parent must be a directory")
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
