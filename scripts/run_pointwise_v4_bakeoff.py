#!/usr/bin/env python3
"""Run a bounded three-arm v4 pointwise-judge bakeoff through OpenRouter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
import urllib.error
import urllib.request

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_local_pointwise_judge import (  # noqa: E402
    _load_cases,
    _load_json_from_opened_file,
)
from trec_mlm.pointwise_judge import (  # noqa: E402
    CandidateCase,
    load_prompt_contract,
    normalize_judge_decision_v4,
    normalize_judge_decision_v5,
)


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MAX_RESPONSE_BYTES = 1024 * 1024


@dataclass(frozen=True, slots=True)
class Arm:
    name: str
    model_id: str
    provider: str
    reasoning: dict[str, object] | None
    max_tokens: int
    input_usd_per_million: float
    output_usd_per_million: float
    temperature_supported: bool = True


ARMS = {
    "deepseek-direct": Arm(
        name="deepseek-direct",
        model_id="deepseek/deepseek-v4-flash",
        provider="DeepInfra",
        reasoning=None,
        max_tokens=512,
        input_usd_per_million=0.14,
        output_usd_per_million=0.28,
    ),
    "deepseek-thinking": Arm(
        name="deepseek-thinking",
        model_id="deepseek/deepseek-v4-flash",
        provider="DeepInfra",
        reasoning={"effort": "high", "exclude": True},
        max_tokens=2048,
        input_usd_per_million=0.14,
        output_usd_per_million=0.28,
    ),
    "gpt-5.4-nano-direct": Arm(
        name="gpt-5.4-nano-direct",
        model_id="openai/gpt-5.4-nano",
        provider="OpenAI",
        reasoning=None,
        max_tokens=512,
        input_usd_per_million=0.20,
        output_usd_per_million=1.25,
        temperature_supported=False,
    ),
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--arm",
        action="append",
        choices=tuple(ARMS),
        help="arm to run; repeat for multiple arms (default: all three)",
    )
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--case-count",
        type=int,
        choices=(10, 100),
        default=10,
        help="expected manifest size (default: 10)",
    )
    parser.add_argument(
        "--ordinals",
        type=int,
        nargs="+",
        help="optional original case ordinals to run; requires the full manifest",
    )
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--version", choices=("v4", "v5", "v6"), default="v4")
    return parser


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_token(name: str) -> str:
    value = os.environ.get(name)
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _user_message(case: CandidateCase) -> str:
    return (
        f"QUESTION ID: {case.question_id}\n"
        f"RESPONSE ID: {case.response_id}\n\n"
        f"QUESTION:\n{case.question}\n\n"
        "PROVISIONAL REFERENCE (may be incomplete or wrong):\n"
        f"{case.reference_answer}\n\n"
        f"CANDIDATE RESPONSE:\n{case.candidate_answer}"
    )


def _build_payload(arm: Arm, *, system_prompt: str, schema: dict[str, object], case: CandidateCase) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": arm.model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": _user_message(case)},
        ],
        "stream": False,
        "max_tokens": arm.max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pointwise_judge_decision",
                "strict": True,
                "schema": schema,
            },
        },
        "provider": {
            "only": [arm.provider],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }
    if arm.temperature_supported:
        payload["temperature"] = 0
    if arm.reasoning is not None:
        payload["reasoning"] = dict(arm.reasoning)
    return payload


def _post_json(payload: dict[str, object], *, token: str, timeout_seconds: float) -> dict[str, object]:
    request = urllib.request.Request(
        ENDPOINT,
        data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8"),
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read(MAX_RESPONSE_BYTES + 1)
    except urllib.error.HTTPError as error:
        body = error.read(4096).decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body[:500]}") from error
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError("provider response exceeded the private size limit")
    try:
        body = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError("provider response was not JSON") from error
    if type(body) is not dict:
        raise RuntimeError("provider response was not an object")
    return body


def _content(body: dict[str, object]) -> str:
    choices = body.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise RuntimeError("provider response did not contain one choice")
    message = choices[0].get("message")
    if type(message) is not dict or type(message.get("content")) is not str:
        raise RuntimeError("provider response content was missing")
    return message["content"]


def _usage_cost(body: dict[str, object], arm: Arm) -> float | None:
    usage = body.get("usage")
    if type(usage) is not dict:
        return None
    reported = usage.get("cost")
    if type(reported) in {int, float} and reported >= 0:
        return float(reported)
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if type(prompt) is not int or type(completion) is not int:
        return None
    return (
        prompt * arm.input_usd_per_million
        + completion * arm.output_usd_per_million
    ) / 1_000_000


def _write_private(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _arm_summary(records: list[dict[str, object]]) -> dict[str, object]:
    clean = coerced = unscorable = errors = 0
    cost = 0.0
    for record in records:
        if "provider_error" in record:
            errors += 1
            continue
        status = record["normalized"]["parse_status"]
        clean += status == "clean"
        coerced += status == "coerced"
        unscorable += status == "unscorable"
        cost += float(record.get("estimated_cost_usd", 0.0))
    return {
        "request_count": len(records),
        "clean": clean,
        "coerced": coerced,
        "unscorable": unscorable,
        "provider_errors": errors,
        "estimated_cost_usd": cost,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    token = _load_token(args.api_key_env)
    manifest = _load_json_from_opened_file(
        args.manifest,
        "smoke manifest",
        expected_sha256=args.manifest_sha256,
        private=True,
    )
    cases = _load_cases(manifest)
    if len(cases) != args.case_count:
        raise ValueError(f"bakeoff requires exactly {args.case_count} cases")
    if args.ordinals is None:
        selected_cases = list(enumerate(cases))
    else:
        if not args.ordinals or len(set(args.ordinals)) != len(args.ordinals):
            raise ValueError("ordinals must be non-empty and unique")
        if any(ordinal < 0 or ordinal >= len(cases) for ordinal in args.ordinals):
            raise ValueError("ordinal is outside the manifest")
        selected_cases = [(ordinal, cases[ordinal]) for ordinal in args.ordinals]
    contract = load_prompt_contract(args.protocol_dir, version=args.version)
    prompt_hash = hashlib.sha256(contract.prompt.encode("utf-8")).hexdigest()
    schema_hash = _sha256_json(contract.schema)
    selected = [ARMS[name] for name in (args.arm or tuple(ARMS))]
    result: dict[str, object] = {
        "version": 1,
        "contract_version": args.version,
        "endpoint": ENDPOINT,
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "manifest_case_count": len(cases),
        "selected_ordinals": [ordinal for ordinal, _ in selected_cases],
        "arms": {},
    }
    started = time.perf_counter()
    for arm in selected:
        records: list[dict[str, object]] = []
        for ordinal, case in selected_cases:
            record: dict[str, object] = {
                "ordinal": ordinal,
                "question_id": case.question_id,
                "response_id": case.response_id,
                "selection_reason": case.selection_reason,
            }
            try:
                body = _post_json(
                    _build_payload(
                        arm,
                        system_prompt=contract.prompt,
                        schema=contract.schema,
                        case=case,
                    ),
                    token=token,
                    timeout_seconds=args.timeout_seconds,
                )
                raw = _content(body)
                record["normalized"] = (
                    normalize_judge_decision_v4(raw)
                    if args.version == "v4"
                    else normalize_judge_decision_v5(raw)
                )
                choices = body.get("choices")
                if type(choices) is list and choices and type(choices[0]) is dict:
                    record["finish_reason"] = choices[0].get("finish_reason")
                usage = body.get("usage")
                if type(usage) is dict:
                    record["usage"] = {
                        key: usage[key]
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                            "reasoning_tokens",
                            "cost",
                        )
                        if key in usage
                    }
                cost = _usage_cost(body, arm)
                if cost is not None:
                    record["estimated_cost_usd"] = cost
                record["provider_model"] = body.get("model")
            except Exception as error:  # one failed call must not hide other cases
                record["provider_error"] = str(error)
            records.append(record)
            result["arms"][arm.name] = {
                "model_id": arm.model_id,
                "provider": arm.provider,
                "reasoning": arm.reasoning,
                "records": records,
                "summary": _arm_summary(records),
            }
            _write_private(args.output, result)
        result["arms"][arm.name]["summary"] = _arm_summary(records)
    result["elapsed_seconds"] = time.perf_counter() - started
    result["total_estimated_cost_usd"] = sum(
        float(arm_result["summary"].get("estimated_cost_usd", 0.0))
        for arm_result in result["arms"].values()
    )
    _write_private(args.output, result)
    print(
        json.dumps(
            {
                "contract_version": args.version,
                "arms": {
                    name: arm_result["summary"]
                    for name, arm_result in result["arms"].items()
                },
                "total_estimated_cost_usd": result["total_estimated_cost_usd"],
                "elapsed_seconds": result["elapsed_seconds"],
            }
        )
    )
    return 0 if all(
        arm_result["summary"]["provider_errors"] == 0
        for arm_result in result["arms"].values()
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
