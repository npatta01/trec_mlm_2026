#!/usr/bin/env python3
"""Run the frozen ten-case pointwise smoke through pinned OpenRouter routes."""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import time
from typing import Callable, Mapping, Sequence
import urllib.error
import urllib.request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_local_pointwise_judge import (
    _load_cases,
    _load_json_from_opened_file,
)
from trec_mlm.pointwise_judge import CandidateCase, PromptContract, load_prompt_contract
from trec_mlm.pointwise_runner import (
    FROZEN_DECISION_SCHEMA_SHA256,
    FROZEN_V2_DECISION_SCHEMA_SHA256,
    MAX_RESPONSE_BYTES,
    _load_existing_run,
    _record_identity,
    _trusted_backend_response_model,
    _validate_reusable_record,
    _validate_run_container,
    run_cases,
    validate_judge_contract,
)


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_API_KEY_FILE = Path("/home/npatta01/.config/trec-mlm/openrouter.key")
DEFAULT_CAP_USD = 0.10
MAX_OUTPUT_TOKENS = 512
MAX_SERIALIZED_REQUEST_BYTES = 64 * 1024
CHAT_TEMPLATE_OVERHEAD_TOKENS = 2_048
_RETRYABLE_STATUS = {408, 429} | set(range(500, 600))


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    provider: str
    input_usd_per_million: float
    output_usd_per_million: float
    reasoning: Mapping[str, object] | None
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    temperature_supported: bool = True
    allowed_selected_models: tuple[str, ...] = ()
    config_id: str | None = None

    def worst_case_usd(self, input_token_ceiling: int) -> float:
        return (
            input_token_ceiling * self.input_usd_per_million
            + self.max_output_tokens * self.output_usd_per_million
        ) / 1_000_000


MODEL_SPECS = {
    spec.model_id: spec
    for spec in (
        ModelSpec(
            "deepseek/deepseek-v4-flash",
            "DeepInfra",
            0.09,
            0.18,
            {"enabled": False},
        ),
        ModelSpec(
            "nvidia/nemotron-3-super-120b-a12b",
            "DigitalOcean",
            0.21,
            0.455,
            {"enabled": False},
        ),
        ModelSpec(
            "qwen/qwen3-235b-a22b-2507",
            "DeepInfra",
            0.09,
            0.55,
            None,
            allowed_selected_models=("qwen/qwen3-235b-a22b-07-25",),
        ),
        ModelSpec(
            "openai/gpt-5.4-nano",
            "OpenAI",
            0.20,
            1.25,
            None,
            temperature_supported=False,
        ),
    )
}


V2_CONFIG_SPECS = {
    spec.config_id: spec
    for spec in (
        ModelSpec(
            model_id="deepseek/deepseek-v4-flash",
            provider="DeepInfra",
            input_usd_per_million=0.09,
            output_usd_per_million=0.18,
            reasoning={"enabled": False},
            config_id="deepseek-v4-flash-direct",
        ),
        ModelSpec(
            model_id="openai/gpt-5.4-nano",
            provider="OpenAI",
            input_usd_per_million=0.20,
            output_usd_per_million=1.25,
            reasoning=None,
            temperature_supported=False,
            config_id="gpt-5.4-nano-direct",
        ),
        ModelSpec(
            model_id="qwen/qwen3.6-35b-a3b",
            provider="AkashML",
            input_usd_per_million=0.14,
            output_usd_per_million=1.00,
            reasoning={"enabled": False},
            max_output_tokens=1_024,
            config_id="qwen3.6-35b-a3b-direct",
        ),
        ModelSpec(
            model_id="qwen/qwen3.6-35b-a3b",
            provider="AkashML",
            input_usd_per_million=0.14,
            output_usd_per_million=1.00,
            reasoning={"enabled": True},
            max_output_tokens=4_096,
            config_id="qwen3.6-35b-a3b-thinking",
        ),
    )
}


@dataclass(slots=True)
class SpendBudget:
    cap_usd: float
    spent_usd: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.cap_usd) or self.cap_usd <= 0:
            raise ValueError("budget cap must be finite and positive")
        if self.cap_usd > DEFAULT_CAP_USD:
            raise ValueError("budget exceeds the authorized hard cap")
        if (
            not math.isfinite(self.spent_usd)
            or self.spent_usd < 0
            or self.spent_usd > self.cap_usd
        ):
            raise ValueError("recorded spend must be within the budget cap")

    def reserve(self, worst_case_usd: float) -> None:
        if (
            not math.isfinite(worst_case_usd)
            or worst_case_usd < 0
            or self.spent_usd + worst_case_usd > self.cap_usd + 1e-12
        ):
            raise RuntimeError("request would exceed the hosted smoke budget")

    def charge(self, actual_usd: float) -> None:
        if not math.isfinite(actual_usd) or actual_usd < 0:
            raise ValueError("reported request cost is invalid")
        if self.spent_usd + actual_usd > self.cap_usd + 1e-12:
            raise RuntimeError(
                "provider-reported cost exceeded the hosted smoke budget"
            )
        self.spent_usd += actual_usd

    def debit_reservation(self, worst_case_usd: float) -> None:
        self.reserve(worst_case_usd)
        self.spent_usd += worst_case_usd

    def reconcile(self, reserved_usd: float, actual_usd: float) -> None:
        if not math.isfinite(actual_usd) or actual_usd < 0:
            raise ValueError("reported request cost is invalid")
        if actual_usd > reserved_usd + 1e-12:
            raise RuntimeError(
                "provider-reported cost exceeded its durable reservation"
            )
        self.spent_usd -= reserved_usd - actual_usd


@dataclass(slots=True)
class AttemptBudget:
    cap_calls: int = 100
    used_calls: int = 0
    ledger_path: Path | None = None
    canary_identity_sha256: str | None = None
    marker_dir: Path | None = None

    def __post_init__(self) -> None:
        if type(self.cap_calls) is not int or not 1 <= self.cap_calls <= 100:
            raise ValueError("call cap must be an integer from 1 through 100")
        if (
            type(self.used_calls) is not int
            or self.used_calls < 0
            or self.used_calls > self.cap_calls
        ):
            raise ValueError("used call count must be within the call cap")

    def reserve(self) -> None:
        self.require_available()
        next_count = self.used_calls + 1
        if self.is_durable:
            _write_attempt_marker(
                self.marker_dir,
                count=next_count,
                canary_identity_sha256=self.canary_identity_sha256,
            )
        self.used_calls += 1
        if self.is_durable:
            _write_attempt_ledger(self.ledger_path, self)

    def require_available(self) -> None:
        if self.used_calls >= self.cap_calls:
            raise RuntimeError("physical request would exceed the call ceiling")

    @property
    def is_durable(self) -> bool:
        return (
            self.ledger_path is not None
            and self.marker_dir is not None
            and _is_sha256(self.canary_identity_sha256)
        )


def _load_api_key(path: Path, *, environ: Mapping[str, str] | None = None) -> str:
    environment = os.environ if environ is None else environ
    value = environment.get("OPENROUTER_API_KEY")
    if value is not None:
        if not value.strip():
            raise ValueError("OPENROUTER_API_KEY must not be empty")
        return value.strip()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("OpenRouter API key is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError("API key file must be a regular file")
        if stat.S_IMODE(opened.st_mode) != 0o600:
            raise ValueError("API key file must have mode 0600")
        if opened.st_size > 16_384:
            raise ValueError("API key file is too large")
        raw = os.read(descriptor, 16_385)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if _file_identity(opened) != _file_identity(after) or _file_identity(
            after
        ) != _file_identity(current):
            raise ValueError("API key file changed during read")
    finally:
        os.close(descriptor)
    try:
        key = raw.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError("API key file must be UTF-8") from error
    if not key:
        raise ValueError("API key file must not be empty")
    return key


def _build_payload(
    spec: ModelSpec,
    messages: Sequence[Mapping[str, str]],
    schema: Mapping[str, object],
) -> dict[str, object]:
    if (
        spec.config_id is not None
        and _sha256_json(schema) != FROZEN_V2_DECISION_SCHEMA_SHA256
    ):
        raise ValueError("v2 hosted configuration requires the exact v2 schema")
    transport_schema = _openrouter_transport_schema(schema)
    payload: dict[str, object] = {
        "model": spec.model_id,
        "messages": [dict(message) for message in messages],
        "stream": False,
        "usage": {"include": True},
        "max_tokens": spec.max_output_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pointwise_judge_decision",
                "strict": True,
                "schema": transport_schema,
            },
        },
        "provider": {
            "only": [spec.provider],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {
                "prompt": spec.input_usd_per_million,
                "completion": spec.output_usd_per_million,
            },
        },
    }
    if spec.temperature_supported:
        payload["temperature"] = 0
    if spec.reasoning is not None:
        payload["reasoning"] = dict(spec.reasoning)
    return payload


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _openrouter_transport_schema(schema: Mapping[str, object]) -> dict[str, object]:
    if _sha256_json(schema) not in {
        FROZEN_DECISION_SCHEMA_SHA256,
        FROZEN_V2_DECISION_SCHEMA_SHA256,
    }:
        raise ValueError("hosted transport requires the exact frozen decision schema")
    transport_schema = copy.deepcopy(dict(schema))
    error_flags = transport_schema["properties"]["error_flags"]
    if error_flags.pop("uniqueItems", None) is not True:
        raise ValueError("frozen error_flags.uniqueItems constraint changed")
    return transport_schema


def _parse_response(
    body: object, spec: ModelSpec
) -> tuple[str, dict[str, object], float]:
    if type(body) is not dict:
        raise ValueError("OpenRouter response must be a JSON object")
    if spec.config_id is not None and "openrouter_metadata" not in body:
        raise ValueError("OpenRouter response router metadata is missing")
    if "openrouter_metadata" in body:
        _validate_router_metadata(body["openrouter_metadata"], spec)
    choices = body.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise ValueError("OpenRouter response must contain exactly one choice")
    choice = choices[0]
    finish_reason = choice.get("finish_reason")
    if finish_reason != "stop" and not (
        spec.config_id is not None and finish_reason == "length"
    ):
        raise ValueError("OpenRouter response finish_reason must be stop")
    message = choice.get("message")
    if type(message) is not dict or type(message.get("content")) is not str:
        raise ValueError("OpenRouter response content must be a string")
    usage = body.get("usage")
    if type(usage) is not dict or type(usage.get("cost")) not in {int, float}:
        raise ValueError("OpenRouter response must report numeric cost")
    cost = usage["cost"]
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("OpenRouter response cost is invalid")
    return message["content"], body, float(cost)


def _validate_router_metadata(value: object, spec: ModelSpec) -> None:
    if type(value) is not dict:
        raise ValueError("OpenRouter response router metadata is missing")
    if value.get("requested") != spec.model_id:
        raise ValueError("OpenRouter metadata requested model changed")
    if type(value.get("attempt")) is not int or value["attempt"] != 1:
        raise ValueError("OpenRouter metadata attempt must be exactly one")
    endpoints = value.get("endpoints")
    if type(endpoints) is not dict:
        raise ValueError("OpenRouter metadata endpoints are missing")
    available = endpoints.get("available")
    if type(available) is not list:
        raise ValueError("OpenRouter metadata available endpoints are missing")
    selected = [
        endpoint
        for endpoint in available
        if type(endpoint) is dict and endpoint.get("selected") is True
    ]
    if len(selected) != 1:
        raise ValueError(
            "OpenRouter metadata must identify exactly one selected endpoint"
        )
    endpoint = selected[0]
    if endpoint.get("provider") != spec.provider:
        raise ValueError("OpenRouter metadata selected the wrong provider")
    if not _matches_selected_model(
        spec.model_id,
        endpoint.get("model"),
        allowed_selected_models=spec.allowed_selected_models,
    ):
        raise ValueError("OpenRouter metadata selected the wrong model")
    if value.get("strategy") != "direct":
        raise ValueError("OpenRouter metadata strategy must be direct")


def _matches_selected_model(
    requested: str,
    selected: object,
    *,
    allowed_selected_models: Sequence[str] = (),
) -> bool:
    if selected == requested or selected in allowed_selected_models:
        return True
    prefix = f"{requested}-"
    if type(selected) is not str or not selected.startswith(prefix):
        return False
    suffix = selected[len(prefix) :]
    return len(suffix) == 8 and all("0" <= character <= "9" for character in suffix)


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


def _post_json(request: urllib.request.Request, *, timeout_seconds: float) -> object:
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _RejectRedirects()
    )
    with opener.open(request, timeout=timeout_seconds) as response:
        if response.geturl() != OPENROUTER_ENDPOINT:
            raise ValueError("redirect responses are not allowed")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("OpenRouter response body is too large")
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("OpenRouter response is not valid UTF-8 JSON") from error


def _reject_nonfinite(value: str) -> None:
    raise ValueError(f"non-finite JSON is forbidden: {value}")


def _request_with_retry(
    operation: Callable[..., object],
    request: object,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> object:
    for attempt in range(2):
        try:
            return operation(request)
        except urllib.error.HTTPError as error:
            if error.code not in _RETRYABLE_STATUS or attempt == 1:
                raise
            sleep(_retry_delay_seconds(error))
    raise RuntimeError("unreachable retry state")


def _retry_delay_seconds(error: urllib.error.HTTPError) -> float:
    headers = error.headers
    raw = headers.get("Retry-After") if hasattr(headers, "get") else None
    if type(raw) is not str:
        return 1.0
    try:
        delay = float(raw)
    except ValueError:
        return 1.0
    if not math.isfinite(delay) or delay < 0:
        return 1.0
    return min(60.0, max(1.0, delay))


class OpenRouterJudgeCall:
    def __init__(
        self,
        *,
        spec: ModelSpec,
        api_key: str,
        budget: SpendBudget,
        attempt_budget: AttemptBudget | None = None,
        timeout_seconds: float,
        checkpoint_spend: Callable[[], None] | None = None,
        checkpoint_attempt: Callable[[], None] | None = None,
        checkpoint_response: Callable[[object], None] | None = None,
    ) -> None:
        if spec.config_id is not None and (
            attempt_budget is None or not attempt_budget.is_durable
        ):
            raise ValueError("v2 hosted calls require a durable attempt budget")
        self.spec = spec
        self._api_key = api_key
        self.budget = budget
        self.attempt_budget = attempt_budget
        self.timeout_seconds = timeout_seconds
        self._checkpoint_spend = checkpoint_spend or (lambda: None)
        self._checkpoint_attempt = checkpoint_attempt or (lambda: None)
        self._checkpoint_response = checkpoint_response or (lambda _body: None)

    def __call__(
        self,
        model: str,
        messages: Sequence[Mapping[str, str]],
        schema: Mapping[str, object],
    ) -> tuple[str, Mapping[str, object]]:
        if model != self.spec.model_id:
            raise ValueError("logical model does not match the pinned route")
        payload = _build_payload(self.spec, messages, schema)
        encoded_payload = json.dumps(
            payload, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        if len(encoded_payload) > MAX_SERIALIZED_REQUEST_BYTES:
            raise ValueError("serialized input exceeds the hosted smoke ceiling")
        input_token_ceiling = len(encoded_payload) + CHAT_TEMPLATE_OVERHEAD_TOKENS
        reservation_usd = self.spec.worst_case_usd(input_token_ceiling)
        request = urllib.request.Request(
            OPENROUTER_ENDPOINT,
            data=encoded_payload,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
                "X-OpenRouter-Metadata": "enabled",
            },
            method="POST",
        )

        def send(value: urllib.request.Request) -> object:
            if self.attempt_budget is not None:
                self.attempt_budget.require_available()
            self.budget.debit_reservation(reservation_usd)
            self._checkpoint_spend()
            if self.attempt_budget is not None:
                self.attempt_budget.reserve()
                if self.spec.config_id is None:
                    self._checkpoint_attempt()
            return _post_json(value, timeout_seconds=self.timeout_seconds)

        body = _request_with_retry(send, request)
        self._checkpoint_response(body)
        content, raw_body, cost = _parse_response(body, self.spec)
        usage = raw_body["usage"]
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if (
            type(prompt_tokens) is not int
            or prompt_tokens < 0
            or prompt_tokens > input_token_ceiling
            or type(completion_tokens) is not int
            or completion_tokens < 0
            or completion_tokens > self.spec.max_output_tokens
        ):
            raise ValueError("provider usage exceeded the reserved token ceilings")
        self.budget.reconcile(reservation_usd, cost)
        self._checkpoint_spend()
        runner_body = dict(raw_body)
        runner_body["model"] = self.spec.model_id
        runner_body["openrouter_raw_response_body"] = raw_body
        return content, runner_body


def _write_private_json(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
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
        os.chmod(path, 0o600)
        directory_flags = (
            os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
        )
        directory_descriptor = os.open(path.parent, directory_flags)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_spend_ledger(path: Path, budget: SpendBudget) -> None:
    _write_private_json(
        path,
        {
            "version": 1,
            "cap_usd": budget.cap_usd,
            "spent_usd": budget.spent_usd,
        },
    )


def _write_attempt_ledger(path: Path, budget: AttemptBudget) -> None:
    if budget.is_durable:
        value = {
            "version": 2,
            "cap_calls": budget.cap_calls,
            "used_calls": budget.used_calls,
            "canary_identity_sha256": budget.canary_identity_sha256,
        }
    else:
        value = {
            "version": 1,
            "cap_calls": budget.cap_calls,
            "used_calls": budget.used_calls,
        }
    _write_private_json(path, value)


def _write_attempt_marker(
    marker_dir: Path | None,
    *,
    count: int,
    canary_identity_sha256: str | None,
) -> None:
    if marker_dir is None or not _is_sha256(canary_identity_sha256):
        raise ValueError("durable attempt marker identity is invalid")
    path = marker_dir / f"{count:03d}.json"
    if path.exists():
        raise ValueError("attempt marker already exists; ledger rollback detected")
    _write_private_json(
        path,
        {
            "version": 1,
            "attempt": count,
            "canary_identity_sha256": canary_identity_sha256,
        },
    )


def _write_response_checkpoint(
    path: Path,
    body: object,
    *,
    spec: ModelSpec | None = None,
    canary_identity_sha256: str | None = None,
    used_calls: int | None = None,
) -> None:
    if spec is not None:
        if (
            spec.config_id is None
            or not _is_sha256(canary_identity_sha256)
            or type(used_calls) is not int
            or used_calls < 1
        ):
            raise ValueError("v2 diagnostic identity is invalid")
        value = {
            "version": 2,
            "config_id": spec.config_id,
            "canary_identity_sha256": canary_identity_sha256,
            "used_calls": used_calls,
            "response_body": body,
        }
    else:
        value = {"version": 1, "response_body": body}
    _write_private_json(path, value)


def _load_spend_ledger(path: Path, *, cap_usd: float) -> SpendBudget:
    if not path.exists():
        return SpendBudget(cap_usd)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("spend ledger could not be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode) != 0o600:
            raise ValueError("spend ledger must be a regular mode-0600 file")
        raw = os.read(descriptor, 4097)
        if len(raw) > 4096:
            raise ValueError("spend ledger is too large")
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if _file_identity(opened) != _file_identity(after) or _file_identity(
            after
        ) != _file_identity(current):
            raise ValueError("spend ledger changed during read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("spend ledger is invalid JSON") from error
    if type(value) is not dict or set(value) != {"version", "cap_usd", "spent_usd"}:
        raise ValueError("spend ledger has an invalid shape")
    if value["version"] != 1 or value["cap_usd"] != cap_usd:
        raise ValueError("spend ledger does not match this run's cap")
    return SpendBudget(cap_usd, spent_usd=value["spent_usd"])


def _load_attempt_ledger(
    path: Path,
    *,
    cap_calls: int,
    canary_identity_sha256: str | None = None,
    marker_dir: Path | None = None,
    require_existing: bool = False,
) -> AttemptBudget:
    if not path.exists():
        if require_existing:
            raise ValueError("attempt ledger is missing for an existing v2 canary")
        return AttemptBudget(
            cap_calls,
            ledger_path=path if canary_identity_sha256 is not None else None,
            canary_identity_sha256=canary_identity_sha256,
            marker_dir=marker_dir,
        )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError("attempt ledger could not be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode) != 0o600:
            raise ValueError("attempt ledger must be a regular mode-0600 file")
        raw = os.read(descriptor, 4097)
        if len(raw) > 4096:
            raise ValueError("attempt ledger is too large")
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if _file_identity(opened) != _file_identity(after) or _file_identity(
            after
        ) != _file_identity(current):
            raise ValueError("attempt ledger changed during read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("attempt ledger is invalid JSON") from error
    expected_keys = {"version", "cap_calls", "used_calls"}
    expected_version = 1
    if canary_identity_sha256 is not None:
        expected_keys.add("canary_identity_sha256")
        expected_version = 2
    if type(value) is not dict or set(value) != expected_keys:
        raise ValueError("attempt ledger has an invalid shape")
    if value["version"] != expected_version or value["cap_calls"] != cap_calls:
        raise ValueError("attempt ledger does not match this run's cap")
    if (
        canary_identity_sha256 is not None
        and value["canary_identity_sha256"] != canary_identity_sha256
    ):
        raise ValueError("attempt ledger canary identity mismatch")
    budget = AttemptBudget(
        cap_calls,
        used_calls=value["used_calls"],
        ledger_path=path if canary_identity_sha256 is not None else None,
        canary_identity_sha256=canary_identity_sha256,
        marker_dir=marker_dir,
    )
    if budget.is_durable:
        _validate_attempt_markers(budget)
    return budget


def _validate_attempt_markers(budget: AttemptBudget) -> None:
    if not budget.is_durable:
        raise ValueError("attempt marker validation requires a durable budget")
    marker_dir = budget.marker_dir
    if not marker_dir.exists():
        if budget.used_calls:
            raise ValueError("attempt marker journal is missing")
        return
    if not marker_dir.is_dir():
        raise ValueError("attempt marker journal is not a directory")
    paths = sorted(marker_dir.iterdir())
    expected_names = [f"{count:03d}.json" for count in range(1, budget.used_calls + 1)]
    if [path.name for path in paths] != expected_names:
        raise ValueError("attempt ledger rollback conflicts with physical markers")
    for count, marker_path in enumerate(paths, start=1):
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker != {
            "version": 1,
            "attempt": count,
            "canary_identity_sha256": budget.canary_identity_sha256,
        }:
            raise ValueError("attempt marker identity mismatch")


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _run_canary_then_remaining(
    cases: Sequence[object],
    run: Callable[[list[object]], dict[str, object]],
    *,
    existing_terminal_records: int = 0,
) -> dict[str, object]:
    if len(cases) != 10:
        raise ValueError("hosted smoke requires exactly ten cases")
    if type(existing_terminal_records) is not int or existing_terminal_records < 0:
        raise ValueError("existing terminal record count is invalid")
    if existing_terminal_records:
        return run(list(cases))
    canary = run([cases[0]])
    records = canary.get("records")
    if type(records) is not dict or len(records) != 1:
        raise RuntimeError("hosted smoke canary did not produce one record")
    record = next(iter(records.values()))
    if (
        type(record) is not dict
        or record.get("decision") is None
        or record.get("failure") is not None
    ):
        raise RuntimeError("hosted smoke canary failed")
    return run(list(cases))


def _validated_existing_terminal_count(
    output_path: Path,
    *,
    cases: Sequence[CandidateCase],
    model_alias: str,
    model_id: str,
    contract: PromptContract,
    runtime_receipt: Mapping[str, object],
    version: str = "v1",
) -> int:
    terminal_count, _ = _validated_existing_counts(
        output_path,
        cases=cases,
        model_alias=model_alias,
        model_id=model_id,
        contract=contract,
        runtime_receipt=runtime_receipt,
        version=version,
    )
    return terminal_count


def _validated_existing_counts(
    output_path: Path,
    *,
    cases: Sequence[CandidateCase],
    model_alias: str,
    model_id: str,
    contract: PromptContract,
    runtime_receipt: Mapping[str, object],
    version: str,
) -> tuple[int, int]:
    existing = _load_existing_run(output_path)
    if existing is None:
        return 0, 0
    _validate_run_container(existing)
    prompt_sha256 = hashlib.sha256(contract.prompt.encode("utf-8")).hexdigest()
    schema_sha256 = validate_judge_contract(contract, version=version)
    runtime_receipt_sha256 = _sha256_json(dict(runtime_receipt))
    expected_container = {
        "model_alias": model_alias,
        "model_id": model_id,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "runtime_receipt_sha256": runtime_receipt_sha256,
    }
    if any(existing[key] != value for key, value in expected_container.items()):
        raise ValueError("existing run container identity mismatch")
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
    records = existing["records"]
    if not set(records) <= set(identities):
        raise ValueError("existing run contains an unexpected record identity")
    backend_model = _trusted_backend_response_model(runtime_receipt, model_id)
    terminal_count = 0
    semantic_attempt_count = 0
    for response_id, record in records.items():
        reusable = _validate_reusable_record(
            record,
            identities[response_id],
            response_id,
            backend_response_model=backend_model,
            version=version,
        )
        if reusable is None:
            raise ValueError("existing run record identity mismatch")
        terminal_count += int(reusable.terminal)
        semantic_attempt_count += len(reusable.attempts)
    return terminal_count, semantic_attempt_count


def _runtime_receipt(
    spec: ModelSpec,
    contract: PromptContract,
    *,
    version: str = "v1",
    canary_identity_sha256: str | None = None,
) -> dict[str, object]:
    transport_schema = _openrouter_transport_schema(contract.schema)
    receipt: dict[str, object] = {
        "schema_sha256": validate_judge_contract(contract, version=version),
        "transport_schema_sha256": _sha256_json(transport_schema),
        "schema_transport_adjustment": (
            "error_flags.uniqueItems omitted; frozen parser retained"
        ),
        "backend_response_model": spec.model_id,
        "transport": "openrouter-chat-completions",
        "endpoint": OPENROUTER_ENDPOINT,
        "provider": spec.provider,
        "model_id": spec.model_id,
        "reasoning": None if spec.reasoning is None else dict(spec.reasoning),
        "temperature": 0 if spec.temperature_supported else None,
        "max_output_tokens": spec.max_output_tokens,
        "max_serialized_request_bytes": MAX_SERIALIZED_REQUEST_BYTES,
        "chat_template_overhead_tokens": CHAT_TEMPLATE_OVERHEAD_TOKENS,
        "input_usd_per_million": spec.input_usd_per_million,
        "output_usd_per_million": spec.output_usd_per_million,
        "require_parameters": True,
        "allow_fallbacks": False,
    }
    if spec.allowed_selected_models:
        receipt["allowed_selected_models"] = list(spec.allowed_selected_models)
    if version == "v2":
        if spec.config_id is None:
            raise ValueError("v2 hosted receipt requires a configuration ID")
        if canary_identity_sha256 is not None and not _is_sha256(
            canary_identity_sha256
        ):
            raise ValueError("v2 canary identity is invalid")
        receipt.update(
            {
                "config_id": spec.config_id,
                "prompt_sha256": hashlib.sha256(
                    contract.prompt.encode("utf-8")
                ).hexdigest(),
                "transport_schema_omitted_json_paths": [
                    "$.properties.error_flags.uniqueItems"
                ],
            }
        )
        if canary_identity_sha256 is not None:
            receipt["canary_identity_sha256"] = canary_identity_sha256
    return receipt


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def _identity_name(spec: ModelSpec) -> str:
    return spec.config_id or _safe_name(spec.model_id)


def _diagnostic_response_path(output_dir: Path, spec: ModelSpec) -> Path:
    return output_dir / "diagnostics" / f"{_identity_name(spec)}-last-response.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--cap-usd", type=float, default=DEFAULT_CAP_USD)
    parser.add_argument("--call-cap", type=int, default=100)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--model",
        action="append",
        help="exact authorized model ID; repeat to select multiple models",
    )
    selection.add_argument(
        "--config",
        action="append",
        help="exact authorized v2 configuration ID; repeat to select multiple configs",
    )
    return parser


def _select_model_specs(requested: Sequence[str] | None) -> list[ModelSpec]:
    if requested is None:
        return list(MODEL_SPECS.values())
    if not requested:
        raise ValueError("model selection must not be empty")
    if any(type(model_id) is not str or not model_id for model_id in requested):
        raise ValueError("model selection must not contain an empty ID")
    if len(set(requested)) != len(requested):
        raise ValueError("model selection contains a duplicate ID")
    unknown = set(requested) - set(MODEL_SPECS)
    if unknown:
        raise ValueError("model selection contains an unknown ID")
    selected = set(requested)
    return [spec for model_id, spec in MODEL_SPECS.items() if model_id in selected]


def _select_config_specs(requested: Sequence[str] | None) -> list[ModelSpec]:
    if requested is None:
        return list(V2_CONFIG_SPECS.values())
    if not requested:
        raise ValueError("configuration selection must not be empty")
    if any(type(config_id) is not str or not config_id for config_id in requested):
        raise ValueError("configuration selection must not contain an empty ID")
    if len(set(requested)) != len(requested):
        raise ValueError("configuration selection contains a duplicate ID")
    unknown = set(requested) - set(V2_CONFIG_SPECS)
    if unknown:
        raise ValueError("configuration selection contains an unknown ID")
    selected = set(requested)
    return [
        spec for config_id, spec in V2_CONFIG_SPECS.items() if config_id in selected
    ]


def _v2_canary_identity(*, manifest_sha256: str, contract: PromptContract) -> str:
    if not _is_sha256(manifest_sha256):
        raise ValueError("v2 manifest identity is invalid")
    transport_schema = _openrouter_transport_schema(contract.schema)
    return _sha256_json(
        {
            "version": 1,
            "manifest_sha256": manifest_sha256,
            "prompt_sha256": hashlib.sha256(
                contract.prompt.encode("utf-8")
            ).hexdigest(),
            "schema_sha256": validate_judge_contract(contract, version="v2"),
            "transport_schema_sha256": _sha256_json(transport_schema),
            "configs": {
                config_id: {
                    "model_id": spec.model_id,
                    "provider": spec.provider,
                    "reasoning": dict(spec.reasoning) if spec.reasoning else None,
                    "input_usd_per_million": spec.input_usd_per_million,
                    "output_usd_per_million": spec.output_usd_per_million,
                }
                for config_id, spec in V2_CONFIG_SPECS.items()
            },
        }
    )


def _v2_canary_artifacts_exist(output_dir: Path) -> bool:
    paths = [
        output_dir / "spend-ledger.json",
        output_dir / "spend-receipt.json",
        output_dir / "attempt-markers",
    ]
    for spec in V2_CONFIG_SPECS.values():
        name = _identity_name(spec)
        paths.extend(
            [
                output_dir / "runtime" / f"{name}.json",
                output_dir / "runs" / f"{name}.json",
                _diagnostic_response_path(output_dir, spec),
            ]
        )
    return any(path.exists() for path in paths)


def _validate_all_v2_artifacts(
    output_dir: Path,
    *,
    cases: Sequence[CandidateCase],
    contract: PromptContract,
    canary_identity_sha256: str,
    attempt_budget: AttemptBudget,
) -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for config_id, spec in V2_CONFIG_SPECS.items():
        receipt = _runtime_receipt(
            spec,
            contract,
            version="v2",
            canary_identity_sha256=canary_identity_sha256,
        )
        receipt_path = output_dir / "runtime" / f"{config_id}.json"
        output_path = output_dir / "runs" / f"{config_id}.json"
        diagnostic_path = _diagnostic_response_path(output_dir, spec)
        if output_path.exists() and not receipt_path.exists():
            raise ValueError("existing v2 run is missing its runtime receipt")
        if receipt_path.exists():
            try:
                stored_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("existing v2 runtime receipt is invalid") from error
            if stored_receipt != receipt:
                raise ValueError("existing v2 runtime receipt identity mismatch")
        if output_path.exists():
            counts[config_id] = _validated_existing_counts(
                output_path,
                cases=cases,
                model_alias=config_id,
                model_id=spec.model_id,
                contract=contract,
                runtime_receipt=receipt,
                version="v2",
            )
        else:
            counts[config_id] = (0, 0)
        if diagnostic_path.exists():
            if not receipt_path.exists():
                raise ValueError(
                    "existing v2 diagnostic is missing its runtime receipt"
                )
            try:
                diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError("existing v2 diagnostic is invalid") from error
            if (
                type(diagnostic) is not dict
                or diagnostic.get("version") != 2
                or diagnostic.get("config_id") != config_id
                or diagnostic.get("canary_identity_sha256") != canary_identity_sha256
                or type(diagnostic.get("used_calls")) is not int
                or not 1 <= diagnostic["used_calls"] <= attempt_budget.used_calls
                or set(diagnostic)
                != {
                    "version",
                    "config_id",
                    "canary_identity_sha256",
                    "used_calls",
                    "response_body",
                }
            ):
                raise ValueError(
                    "existing v2 diagnostic identity or attempt-ledger rollback mismatch"
                )
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    AttemptBudget(args.call_cap)
    if args.config is not None:
        selected_specs = _select_config_specs(args.config)
        version = "v2"
    else:
        selected_specs = _select_model_specs(args.model)
        version = "v1"
    api_key = _load_api_key(args.api_key_file)
    manifest = _load_json_from_opened_file(
        args.manifest,
        "smoke manifest",
        expected_sha256=args.manifest_sha256,
        private=True,
    )
    cases = _load_cases(manifest)
    if len(cases) != 10:
        raise ValueError("hosted smoke accepts only a ten-case manifest")
    contract = load_prompt_contract(args.protocol_dir, version=version)
    validate_judge_contract(contract, version=version)
    spend_ledger_path = args.output_dir / "spend-ledger.json"
    budget = _load_spend_ledger(spend_ledger_path, cap_usd=args.cap_usd)
    attempt_ledger_path = args.output_dir / "attempt-ledger.json"
    canary_identity_sha256: str | None = None
    existing_counts: dict[str, tuple[int, int]] = {}
    if version == "v2":
        canary_identity_sha256 = _v2_canary_identity(
            manifest_sha256=args.manifest_sha256,
            contract=contract,
        )
        artifacts_exist = _v2_canary_artifacts_exist(args.output_dir)
        attempt_budget = _load_attempt_ledger(
            attempt_ledger_path,
            cap_calls=args.call_cap,
            canary_identity_sha256=canary_identity_sha256,
            marker_dir=args.output_dir / "attempt-markers",
            require_existing=artifacts_exist,
        )
        existing_counts = _validate_all_v2_artifacts(
            args.output_dir,
            cases=cases,
            contract=contract,
            canary_identity_sha256=canary_identity_sha256,
            attempt_budget=attempt_budget,
        )
        recorded_semantic_attempts = sum(
            semantic_attempts for _, semantic_attempts in existing_counts.values()
        )
        if attempt_budget.used_calls < recorded_semantic_attempts:
            raise ValueError(
                "attempt ledger is below validated existing semantic attempts"
            )
        if not attempt_ledger_path.exists():
            _write_attempt_ledger(attempt_ledger_path, attempt_budget)
    else:
        attempt_budget = _load_attempt_ledger(
            attempt_ledger_path, cap_calls=args.call_cap
        )
    summaries: list[dict[str, object]] = []
    for spec in selected_specs:
        receipt = _runtime_receipt(
            spec,
            contract,
            version=version,
            canary_identity_sha256=canary_identity_sha256,
        )
        name = _identity_name(spec)
        receipt_path = args.output_dir / "runtime" / f"{name}.json"
        output_path = args.output_dir / "runs" / f"{name}.json"
        _write_private_json(receipt_path, receipt)
        if version == "v2":
            existing_terminal_records = existing_counts[name][0]
        else:
            existing_terminal_records = _validated_existing_terminal_count(
                output_path,
                cases=cases,
                model_alias=name,
                model_id=spec.model_id,
                contract=contract,
                runtime_receipt=receipt,
            )
        call = OpenRouterJudgeCall(
            spec=spec,
            api_key=api_key,
            budget=budget,
            attempt_budget=attempt_budget,
            timeout_seconds=args.timeout_seconds,
            checkpoint_spend=lambda: _write_spend_ledger(spend_ledger_path, budget),
            checkpoint_attempt=(
                (lambda: _write_attempt_ledger(attempt_ledger_path, attempt_budget))
                if version == "v1"
                else None
            ),
            checkpoint_response=lambda body, selected=spec: _write_response_checkpoint(
                _diagnostic_response_path(args.output_dir, selected),
                body,
                spec=selected if version == "v2" else None,
                canary_identity_sha256=canary_identity_sha256,
                used_calls=(attempt_budget.used_calls if version == "v2" else None),
            ),
        )
        invoke = partial(
            run_cases,
            model_alias=name,
            model_id=spec.model_id,
            contract=contract,
            output_path=output_path,
            runtime_receipt=receipt,
            call=call,
            version=version,
        )
        result = _run_canary_then_remaining(
            cases,
            lambda selected: invoke(cases=selected),
            existing_terminal_records=existing_terminal_records,
        )
        summaries.append(
            {
                "config_id": spec.config_id,
                "model_id": spec.model_id,
                "provider": spec.provider,
                "record_count": len(result["records"]),
                "run_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
            }
        )
    _write_private_json(
        args.output_dir / "spend-receipt.json",
        {
            "version": 1,
            "cap_usd": budget.cap_usd,
            "spent_usd": budget.spent_usd,
            "call_cap": attempt_budget.cap_calls,
            "used_calls": attempt_budget.used_calls,
            "models": summaries,
            "api_key_serialized": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
