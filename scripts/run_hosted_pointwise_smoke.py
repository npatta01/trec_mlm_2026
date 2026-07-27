#!/usr/bin/env python3
"""Run the frozen ten-case pointwise smoke through pinned OpenRouter routes."""

from __future__ import annotations

import argparse
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
from trec_mlm.pointwise_judge import PromptContract, load_prompt_contract
from trec_mlm.pointwise_runner import (
    MAX_RESPONSE_BYTES,
    run_cases,
    validate_frozen_judge_contract,
)


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_API_KEY_FILE = Path("/home/npatta01/.config/trec-mlm/openrouter.key")
DEFAULT_CAP_USD = 0.10
MAX_OUTPUT_TOKENS = 512
MAX_SERIALIZED_REQUEST_BYTES = 64 * 1024
CHAT_TEMPLATE_OVERHEAD_TOKENS = 2_048
_RETRYABLE_STATUS = {408, 429} | set(range(500, 600))


@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    provider: str
    input_usd_per_million: float
    output_usd_per_million: float
    reasoning: Mapping[str, object] | None
    temperature_supported: bool = True

    def worst_case_usd(self, input_token_ceiling: int) -> float:
        return (
            input_token_ceiling * self.input_usd_per_million
            + MAX_OUTPUT_TOKENS * self.output_usd_per_million
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
            "DeepInfra",
            0.085,
            0.40,
            {"enabled": False},
        ),
        ModelSpec(
            "qwen/qwen3-235b-a22b-2507",
            "DeepInfra",
            0.09,
            0.55,
            None,
        ),
        ModelSpec(
            "openai/gpt-5.4-nano",
            "OpenAI",
            0.20,
            1.25,
            {"effort": "none"},
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
    payload: dict[str, object] = {
        "model": spec.model_id,
        "messages": [dict(message) for message in messages],
        "stream": False,
        "usage": {"include": True},
        "max_tokens": MAX_OUTPUT_TOKENS,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pointwise_judge_decision",
                "strict": True,
                "schema": dict(schema),
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


def _parse_response(
    body: object, spec: ModelSpec
) -> tuple[str, dict[str, object], float]:
    if type(body) is not dict:
        raise ValueError("OpenRouter response must be a JSON object")
    model = body.get("model")
    if model != spec.model_id:
        raise ValueError("OpenRouter response model changed from the pinned route")
    _validate_router_metadata(body.get("openrouter_metadata"), spec)
    choices = body.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise ValueError("OpenRouter response must contain exactly one choice")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
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
    if endpoint.get("model") != spec.model_id:
        raise ValueError("OpenRouter metadata selected the wrong model")


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
            sleep(1.0)
    raise RuntimeError("unreachable retry state")


class OpenRouterJudgeCall:
    def __init__(
        self,
        *,
        spec: ModelSpec,
        api_key: str,
        budget: SpendBudget,
        timeout_seconds: float,
        checkpoint_spend: Callable[[], None] | None = None,
    ) -> None:
        self.spec = spec
        self._api_key = api_key
        self.budget = budget
        self.timeout_seconds = timeout_seconds
        self._checkpoint_spend = checkpoint_spend or (lambda: None)

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
            self.budget.debit_reservation(reservation_usd)
            self._checkpoint_spend()
            return _post_json(value, timeout_seconds=self.timeout_seconds)

        body = _request_with_retry(send, request)
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
            or completion_tokens > MAX_OUTPUT_TOKENS
        ):
            raise ValueError("provider usage exceeded the reserved token ceilings")
        self.budget.reconcile(reservation_usd, cost)
        self._checkpoint_spend()
        return content, raw_body


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


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _run_canary_then_remaining(
    cases: Sequence[object], run: Callable[[list[object]], dict[str, object]]
) -> dict[str, object]:
    if len(cases) != 10:
        raise ValueError("hosted smoke requires exactly ten cases")
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


def _runtime_receipt(spec: ModelSpec, contract: PromptContract) -> dict[str, object]:
    return {
        "schema_sha256": validate_frozen_judge_contract(contract),
        "backend_response_model": spec.model_id,
        "transport": "openrouter-chat-completions",
        "endpoint": OPENROUTER_ENDPOINT,
        "provider": spec.provider,
        "model_id": spec.model_id,
        "reasoning": None if spec.reasoning is None else dict(spec.reasoning),
        "temperature": 0 if spec.temperature_supported else None,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "max_serialized_request_bytes": MAX_SERIALIZED_REQUEST_BYTES,
        "chat_template_overhead_tokens": CHAT_TEMPLATE_OVERHEAD_TOKENS,
        "input_usd_per_million": spec.input_usd_per_million,
        "output_usd_per_million": spec.output_usd_per_million,
        "require_parameters": True,
        "allow_fallbacks": False,
    }


def _safe_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--cap-usd", type=float, default=DEFAULT_CAP_USD)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
    contract = load_prompt_contract(args.protocol_dir)
    validate_frozen_judge_contract(contract)
    spend_ledger_path = args.output_dir / "spend-ledger.json"
    budget = _load_spend_ledger(spend_ledger_path, cap_usd=args.cap_usd)
    summaries: list[dict[str, object]] = []
    for spec in MODEL_SPECS.values():
        receipt = _runtime_receipt(spec, contract)
        name = _safe_name(spec.model_id)
        receipt_path = args.output_dir / "runtime" / f"{name}.json"
        output_path = args.output_dir / "runs" / f"{name}.json"
        _write_private_json(receipt_path, receipt)
        call = OpenRouterJudgeCall(
            spec=spec,
            api_key=api_key,
            budget=budget,
            timeout_seconds=args.timeout_seconds,
            checkpoint_spend=lambda: _write_spend_ledger(spend_ledger_path, budget),
        )
        invoke = partial(
            run_cases,
            model_alias=name,
            model_id=spec.model_id,
            contract=contract,
            output_path=output_path,
            runtime_receipt=receipt,
            call=call,
        )
        result = _run_canary_then_remaining(
            cases, lambda selected: invoke(cases=selected)
        )
        summaries.append(
            {
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
            "models": summaries,
            "api_key_serialized": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
