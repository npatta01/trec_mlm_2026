#!/usr/bin/env python3
"""Run one pinned, resumable v11 retrospective-judge bakeoff arm.

The network boundary is deliberately injectable so that this module's tests do
not contact providers.  A run directory is an arm-specific, private cache:
metadata seals every request-affecting input and each record checkpoint binds
that metadata to one manifest record.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Callable

import requests

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_pointwise_v11_adjudication import provider_message_fixture
from trec_mlm.pointwise_judge import (
    PromptContract,
    load_prompt_contract,
    normalize_judge_decision_v11,
    normalized_decision_v11_schema,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHORIZED_OUTPUT_ROOT = Path("artifacts/pointwise_judge_bakeoff/v11")
_DIR_MODE = 0o700
_FILE_MODE = 0o600
_MAX_RESPONSE_BYTES = 1024 * 1024
_URL = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_NUMERIC_MARKER = re.compile(r"\[\d+\]")
_SOURCE_SECTION = re.compile(
    r"(?:^|\n)[ \t]*(?:sources|references)[ \t]*:", re.IGNORECASE
)
_FROZEN_MANIFEST_SHA256 = (
    "35948b68935a310cf0cc715927c0a0b31ab88a8f9fadbeaa391594f2ef1fc68f"
)
_FROZEN_PROMPT_SHA256 = (
    "4332319bdfb3e97518c11873b582b1e49a22516b4a618e5583baf7879870fe38"
)


@dataclass(frozen=True)
class Arm:
    """One provider route that may not be substituted or fall back."""

    name: str
    endpoint: str
    model: str
    provider: str
    provider_order: str | None
    reasoning_effort: str | None
    reasoning_enabled: bool
    max_tokens: int
    workers: int
    temperature: int | None
    api_key_env: str

    def as_identity(self) -> dict[str, object]:
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "provider": self.provider,
            "provider_order": self.provider_order,
            "reasoning_effort": self.reasoning_effort,
            "reasoning_enabled": self.reasoning_enabled,
            "max_tokens": self.max_tokens,
            "workers": self.workers,
            "temperature": self.temperature,
        }


ARMS = {
    "deepseek-v4-flash-thinking": Arm(
        name="deepseek-v4-flash-thinking",
        endpoint="https://api.deepinfra.com/v1/openai/chat/completions",
        model="deepseek-ai/DeepSeek-V4-Flash",
        provider="DeepInfra",
        provider_order=None,
        reasoning_effort="high",
        reasoning_enabled=False,
        max_tokens=8192,
        workers=8,
        temperature=0,
        api_key_env="DEEPINFRA_API_TOKEN",
    ),
    "gpt-5.4-nano-direct": Arm(
        name="gpt-5.4-nano-direct",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        model="openai/gpt-5.4-nano",
        provider="OpenRouter",
        provider_order="OpenAI",
        reasoning_effort=None,
        reasoning_enabled=False,
        max_tokens=1024,
        workers=4,
        temperature=None,
        api_key_env="OPENROUTER_API_KEY",
    ),
    "qwen3.6-35b-a3b-thinking": Arm(
        name="qwen3.6-35b-a3b-thinking",
        endpoint="https://openrouter.ai/api/v1/chat/completions",
        model="qwen/qwen3.6-35b-a3b",
        provider="OpenRouter",
        provider_order="AkashML",
        reasoning_effort=None,
        reasoning_enabled=True,
        max_tokens=4096,
        workers=4,
        temperature=0,
        api_key_env="OPENROUTER_API_KEY",
    ),
}


class ProviderHTTPError(RuntimeError):
    """Sanitized provider status; response bodies must never be persisted."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"provider HTTP {status}")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _private_dir(path: Path) -> None:
    if path.exists():
        details = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(details.st_mode) or stat.S_IMODE(details.st_mode) != _DIR_MODE:
            raise ValueError(f"private directory must be mode 0700: {path}")
        return
    path.mkdir(parents=True, mode=_DIR_MODE)
    path.chmod(_DIR_MODE)


def _write_private_json(path: Path, value: object) -> None:
    _private_dir(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, _FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(_FILE_MODE)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _read_private_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        details = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    if not stat.S_ISREG(details.st_mode) or stat.S_IMODE(details.st_mode) != _FILE_MODE:
        raise ValueError(f"{label} must be a regular mode-0600 file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must contain an object")
    return value


def _validate_provider_text(value: str, *, field: str) -> None:
    if not value:
        raise ValueError(f"{field} must be nonempty")
    if _URL.search(value) or _NUMERIC_MARKER.search(value) or _SOURCE_SECTION.search(value):
        raise ValueError(f"{field} contains forbidden provider-facing source metadata")


def _validate_manifest(manifest: dict[str, object]) -> dict[int, dict[str, object]]:
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("version") != 1
        or manifest.get("manifest_sha256") != _sha256_json(body)
        or manifest.get("manifest_sha256") != _FROZEN_MANIFEST_SHA256
        or manifest.get("record_count") != 282
    ):
        raise ValueError("adjudication manifest identity is invalid")
    records = manifest.get("records")
    if type(records) is not list or manifest.get("record_count") != len(records):
        raise ValueError("adjudication manifest records are invalid")
    transform = manifest.get("historical_response_transform")
    if (
        type(transform) is not dict
        or set(transform) != {"version", "sha256"}
        or type(transform["version"]) is not str
        or not transform["version"]
        or type(transform["sha256"]) is not str
        or len(transform["sha256"]) != 64
    ):
        raise ValueError("adjudication manifest transform identity is invalid")
    by_ordinal: dict[int, dict[str, object]] = {}
    for record in records:
        if type(record) is not dict or type(record.get("ordinal")) is not int:
            raise ValueError("adjudication manifest record ordinal is invalid")
        ordinal = record["ordinal"]
        if ordinal in by_ordinal:
            raise ValueError("adjudication manifest contains duplicate ordinal")
        if type(record.get("deterministic_no_provider_call")) is not bool:
            raise ValueError("adjudication manifest deterministic-call flag is invalid")
        if record.get("expected_v11_derived_label") not in {
            "correct", "partial", "incorrect", "not_established", "nonanswer"
        }:
            raise ValueError("adjudication manifest expected label is invalid")
        fixture = provider_message_fixture(record)
        for field, value in fixture.items():
            _validate_provider_text(value, field=field)
        by_ordinal[ordinal] = record
    return by_ordinal


def _validate_contract(contract: PromptContract) -> None:
    schema = contract.schema
    if not isinstance(contract.prompt, str) or not contract.prompt.strip() or not isinstance(schema, dict):
        raise ValueError("v11 contract is invalid")
    required = {"central_claim", "answer_present", "supported_core", "material_conflict"}
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("v11 schema must be a strict object")
    if set(schema.get("properties", {})) != required or set(schema.get("required", [])) != required:
        raise ValueError("v11 schema fields are invalid")
    if schema != normalized_decision_v11_schema():
        raise ValueError("v11 schema identity is not the frozen contract")
    if _sha256_text(contract.prompt) != _FROZEN_PROMPT_SHA256:
        raise ValueError("v11 prompt identity is not the frozen contract")


def _validate_arm(arm: Arm) -> None:
    if ARMS.get(arm.name) != arm:
        raise ValueError("provider route is not one of the pinned v11 arms")


def _user_message(record: dict[str, object]) -> str:
    fixture = provider_message_fixture(record)
    for field, value in fixture.items():
        _validate_provider_text(value, field=field)
    return (
        f"ORIGINAL QUESTION:\n{fixture['question']}\n\n"
        "GEMINI EVALUATOR EVIDENCE — RETROSPECTIVE JUDGE ONLY:\n"
        f"{fixture['evaluator_evidence']}\n\n"
        f"HISTORICAL RESPONSE:\n{fixture['historical_response']}"
    )


def _build_body(arm: Arm, contract: PromptContract, record: dict[str, object]) -> dict[str, object]:
    body: dict[str, object] = {
        "model": arm.model,
        "messages": [
            {"role": "system", "content": contract.prompt},
            {"role": "user", "content": _user_message(record)},
        ],
        "stream": False,
        "max_tokens": arm.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pointwise_judge_v11_decision",
                "strict": True,
                "schema": contract.schema,
            },
        },
    }
    if arm.temperature is not None:
        body["temperature"] = arm.temperature
    if arm.provider_order is not None:
        body["usage"] = {"include": True}
        body["provider"] = {
            "only": [arm.provider_order],
            "allow_fallbacks": False,
            "require_parameters": True,
        }
    if arm.reasoning_effort is not None:
        body["reasoning_effort"] = arm.reasoning_effort
    if arm.reasoning_enabled:
        body["reasoning"] = {"enabled": True}
    return body


def _retryable_status(status: int) -> bool:
    return status in {408, 429} or 500 <= status <= 599


def _post_once(
    body: dict[str, object], *, token: str, timeout_seconds: float, arm: Arm | None
) -> dict[str, object]:
    if arm is None:
        raise ValueError("provider arm is required")
    response = requests.post(
        arm.endpoint,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout_seconds,
    )
    if response.status_code >= 400:
        raise ProviderHTTPError(response.status_code)
    if len(response.content) > _MAX_RESPONSE_BYTES:
        raise ValueError("provider response exceeds private size limit")
    try:
        value = response.json()
    except requests.JSONDecodeError as error:
        raise ValueError("provider response is not JSON") from error
    if type(value) is not dict:
        raise ValueError("provider response must be an object")
    return value


def _post_with_one_retry(
    body: dict[str, object],
    *,
    token: str,
    timeout_seconds: float,
    arm: Arm | None = None,
) -> tuple[dict[str, object], int]:
    """Repeat the untouched request exactly once only for transport failures."""
    for attempt in (1, 2):
        try:
            return _post_once(body, token=token, timeout_seconds=timeout_seconds, arm=arm), attempt
        except ProviderHTTPError as error:
            if attempt == 2 or not _retryable_status(error.status):
                raise
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise
        time.sleep(1.0)
    raise AssertionError("retry loop did not return or raise")


def _content(response: dict[str, object]) -> str:
    choices = response.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise ValueError("provider response must contain exactly one choice")
    message = choices[0].get("message")
    if type(message) is not dict or type(message.get("content")) is not str or not message["content"].strip():
        raise ValueError("provider response content is invalid")
    return message["content"]


def _cost(response: dict[str, object]) -> float:
    usage = response.get("usage")
    if type(usage) is not dict:
        return 0.0
    value = usage.get("cost", usage.get("estimated_cost", 0.0))
    if type(value) is bool or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError("provider usage cost is invalid")
    return float(value)


def _metadata(
    *,
    arm: Arm,
    manifest: dict[str, object],
    contract: PromptContract,
    selected_ordinals: list[int],
    max_calls: int,
    max_cost_usd: float,
    timeout_seconds: float,
) -> dict[str, object]:
    transform = manifest["historical_response_transform"]
    assert isinstance(transform, dict)
    return {
        "version": 1,
        "contract_version": "v11",
        "arm": arm.name,
        "route": arm.as_identity(),
        "allow_fallbacks": False,
        "manifest_sha256": manifest["manifest_sha256"],
        "prompt_sha256": _sha256_text(contract.prompt),
        "schema_sha256": _sha256_json(contract.schema),
        "historical_response_transform_version": transform["version"],
        "historical_response_transform_sha256": transform["sha256"],
        "selected_ordinals": selected_ordinals,
        "max_calls": max_calls,
        "max_cost_usd": max_cost_usd,
        "timeout_seconds": timeout_seconds,
    }


def _record_cache_identity(metadata: dict[str, object], record: dict[str, object]) -> str:
    return _sha256_json({"metadata": metadata, "ordinal": record["ordinal"]})


def _record_matches_cache(record: dict[str, object], *, ordinal: int, identity: str) -> bool:
    return record.get("ordinal") == ordinal and record.get("cache_identity_sha256") == identity


def _deterministic_record(record: dict[str, object], identity: str) -> dict[str, object]:
    return {
        "ordinal": record["ordinal"],
        "cache_identity_sha256": identity,
        "provider_call": False,
        "normalized": {
            "decision": {
                "central_claim": "",
                "answer_present": False,
                "supported_core": False,
                "material_conflict": False,
            },
            "label": "nonanswer",
            "parse_status": "deterministic",
            "repair_codes": [],
        },
        "label": "nonanswer",
        "estimated_cost_usd": 0.0,
    }


def _network_record(
    record: dict[str, object],
    identity: str,
    response: dict[str, object],
    attempts: int,
    *,
    arm: Arm,
) -> dict[str, object]:
    if response.get("model") != arm.model:
        raise ValueError("provider returned an unexpected model")
    if arm.provider_order is not None and response.get("provider") != arm.provider_order:
        raise ValueError("OpenRouter returned an unexpected provider")
    choices = response.get("choices")
    if (
        type(choices) is not list
        or len(choices) != 1
        or type(choices[0]) is not dict
        or choices[0].get("finish_reason") != "stop"
    ):
        raise ValueError("provider response did not finish normally")
    normalized = normalize_judge_decision_v11(_content(response))
    result: dict[str, object] = {
        "ordinal": record["ordinal"],
        "cache_identity_sha256": identity,
        "provider_call": True,
        "attempt_count": attempts,
        "normalized": normalized,
        "label": normalized["label"],
        "estimated_cost_usd": _cost(response),
        "provider_model": response.get("model"),
        "provider": response.get("provider"),
    }
    if type(choices) is list and choices and type(choices[0]) is dict:
        result["finish_reason"] = choices[0].get("finish_reason")
    usage = response.get("usage")
    if type(usage) is dict:
        result["usage"] = usage
    if normalized["parse_status"] == "unscorable":
        result["schema_error"] = "unscorable"
    return result


Post = Callable[..., dict[str, object]]


def run_bakeoff(
    *,
    arm: Arm,
    manifest: dict[str, object],
    contract: PromptContract,
    selected_ordinals: list[int],
    output_dir: Path,
    max_calls: int,
    max_cost_usd: float,
    token: str,
    timeout_seconds: float,
    post: Post | None = None,
) -> dict[str, object]:
    """Run/resume one immutable arm selection without any route fallback."""
    records_by_ordinal = _validate_manifest(manifest)
    _validate_contract(contract)
    _validate_arm(arm)
    if not selected_ordinals or len(selected_ordinals) != len(set(selected_ordinals)):
        raise ValueError("selected ordinals must be nonempty and unique")
    if type(max_calls) is not int or max_calls < 0:
        raise ValueError("max-calls must be a nonnegative integer")
    if type(max_cost_usd) is bool or not isinstance(max_cost_usd, (int, float)) or not math.isfinite(max_cost_usd) or max_cost_usd <= 0:
        raise ValueError("max-cost-usd must be a positive finite number")
    if type(timeout_seconds) is bool or not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise ValueError("timeout-seconds must be positive")
    if type(token) is not str or not token.strip():
        raise ValueError("provider credential is required")
    try:
        selected = [records_by_ordinal[ordinal] for ordinal in selected_ordinals]
    except KeyError as error:
        raise ValueError("selected ordinal is absent from manifest") from error
    provider_cases = [record for record in selected if not record["deterministic_no_provider_call"]]
    if len(provider_cases) > max_calls:
        raise ValueError("max-calls is below provider-eligible selection size")

    metadata = _metadata(
        arm=arm,
        manifest=manifest,
        contract=contract,
        selected_ordinals=selected_ordinals,
        max_calls=max_calls,
        max_cost_usd=float(max_cost_usd),
        timeout_seconds=float(timeout_seconds),
    )
    output_dir = Path(output_dir)
    _private_dir(output_dir)
    records_dir = output_dir / "records"
    _private_dir(records_dir)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        if _read_private_json(metadata_path, label="run metadata") != metadata:
            raise ValueError("existing run cache identity does not match")
    else:
        _write_private_json(metadata_path, metadata)

    completed: dict[int, dict[str, object]] = {}
    pending: list[tuple[dict[str, object], str]] = []
    for record in selected:
        ordinal = record["ordinal"]
        assert isinstance(ordinal, int)
        identity = _record_cache_identity(metadata, record)
        path = records_dir / f"{ordinal:06d}.json"
        if path.exists():
            cached = _read_private_json(path, label=f"record {ordinal}")
            if not _record_matches_cache(cached, ordinal=ordinal, identity=identity):
                raise ValueError(f"record cache identity mismatch at ordinal {ordinal}")
            completed[ordinal] = cached
        elif record["deterministic_no_provider_call"]:
            deterministic = _deterministic_record(record, identity)
            _write_private_json(path, deterministic)
            completed[ordinal] = deterministic
        else:
            pending.append((record, identity))

    def invoke(record: dict[str, object], identity: str) -> tuple[int, dict[str, object]]:
        body = _build_body(arm, contract, record)
        try:
            if post is None:
                response, attempts = _post_with_one_retry(
                    body, token=token, timeout_seconds=float(timeout_seconds), arm=arm
                )
            else:
                response, attempts = post(
                    body, token=token, timeout_seconds=float(timeout_seconds), arm=arm
                ), 1
            result = _network_record(record, identity, response, attempts, arm=arm)
        except ProviderHTTPError as error:
            result = {"ordinal": record["ordinal"], "cache_identity_sha256": identity, "provider_call": True, "provider_error": f"http_{error.status}"}
        except requests.Timeout:
            result = {"ordinal": record["ordinal"], "cache_identity_sha256": identity, "provider_call": True, "provider_error": "timeout"}
        except requests.ConnectionError:
            result = {"ordinal": record["ordinal"], "cache_identity_sha256": identity, "provider_call": True, "provider_error": "connection_error"}
        except ValueError as error:
            result = {
                "ordinal": record["ordinal"],
                "cache_identity_sha256": identity,
                "provider_call": True,
                "provider_error": f"validation_error:{error}",
            }
        except Exception:
            result = {"ordinal": record["ordinal"], "cache_identity_sha256": identity, "provider_call": True, "provider_error": "unexpected_error"}
        return int(record["ordinal"]), result

    if pending:
        with ThreadPoolExecutor(max_workers=arm.workers) as executor:
            futures = [executor.submit(invoke, record, identity) for record, identity in pending]
            for future in as_completed(futures):
                ordinal, result = future.result()
                _write_private_json(records_dir / f"{ordinal:06d}.json", result)
                completed[ordinal] = result

    ordered = [completed[int(record["ordinal"])] for record in selected]
    provider_errors = sum("provider_error" in record for record in ordered)
    schema_errors = sum("schema_error" in record for record in ordered)
    observed_cost = sum(float(record.get("estimated_cost_usd", 0.0)) for record in ordered)
    summary = {
        "provider_calls": sum(record.get("provider_call") is True for record in ordered),
        "deterministic_nonanswers": sum(record.get("provider_call") is False for record in ordered),
        "provider_errors": provider_errors,
        "schema_errors": schema_errors,
        "estimated_cost_usd": observed_cost,
    }
    result = {"metadata": metadata, "summary": summary, "records": ordered}
    _write_private_json(output_dir / "result.json", result)
    if observed_cost > float(max_cost_usd) + 1e-12:
        raise RuntimeError("provider-reported cost exceeded max-cost-usd")
    return result


def _token(name: str) -> str:
    token = os.environ.get(name)
    if type(token) is not str or not token.strip():
        raise ValueError(f"{name} is required")
    return token.strip()


def _validate_output_dir(path: Path, *, arm: Arm) -> Path:
    root = (REPO_ROOT / AUTHORIZED_OUTPUT_ROOT).resolve()
    current = REPO_ROOT
    for part in AUTHORIZED_OUTPUT_ROOT.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("authorized v11 root must not contain symlinks")
    output = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    if output == root or not output.is_relative_to(root) or output.name != arm.name:
        raise ValueError("output-dir must be an arm-named directory under private v11 artifacts")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument("--ordinals", type=int, nargs="+", required=True)
    parser.add_argument("--max-calls", type=int, required=True)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=150.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    arm = ARMS[args.arm]
    manifest = _read_private_json(args.manifest, label="adjudication manifest")
    contract = load_prompt_contract(args.protocol_dir, version="v11")
    output_dir = _validate_output_dir(args.output_dir, arm=arm)
    token = _token(arm.api_key_env)
    result = run_bakeoff(
        arm=arm,
        manifest=manifest,
        contract=contract,
        selected_ordinals=args.ordinals,
        output_dir=output_dir,
        max_calls=args.max_calls,
        max_cost_usd=args.max_cost_usd,
        token=token,
        timeout_seconds=args.timeout_seconds,
    )
    print(json.dumps({"arm": arm.name, "summary": result["summary"]}, sort_keys=True))
    summary = result["summary"]
    return 0 if summary["provider_errors"] == 0 and summary["schema_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
