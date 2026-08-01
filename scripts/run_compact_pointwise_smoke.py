#!/usr/bin/env python3
"""Run a bounded DeepSeek Flash smoke against a compact pointwise contract."""

from __future__ import annotations

import argparse
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
    normalize_judge_decision_v3,
    normalize_judge_decision_v4,
    normalize_judge_decision_v5,
)


ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL_ID = "deepseek-ai/DeepSeek-V4-Flash"
INPUT_USD_PER_MILLION = 0.09
OUTPUT_USD_PER_MILLION = 0.18
MAX_OUTPUT_TOKENS = 384
THINKING_MAX_OUTPUT_TOKENS = 2048
MAX_RESPONSE_BYTES = 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--version", choices=("v3", "v4", "v5", "v6"), default="v3")
    parser.add_argument(
        "--thinking",
        action="store_true",
        help="enable the provider's native reasoning mode",
    )
    parser.add_argument(
        "--token-env", default="DEEPINFRA_API_TOKEN", help="environment variable"
    )
    return parser


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
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


def _request(
    *,
    token: str,
    prompt: str,
    schema: dict[str, object],
    case: CandidateCase,
    timeout_seconds: float,
    thinking: bool,
) -> tuple[str, dict[str, object]]:
    # DeepInfra's reliable transport for this model is json_object; the frozen
    # schema is still enforced locally by normalize_judge_decision_v3.
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": _user_message(case)},
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": THINKING_MAX_OUTPUT_TOKENS if thinking else MAX_OUTPUT_TOKENS,
        "response_format": {"type": "json_object"},
    }
    if thinking:
        # DeepInfra documents this as the portable OpenAI-compatible control;
        # using it avoids a provider-specific reasoning-object wrapper.
        payload["reasoning_effort"] = "high"
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
    choices = body.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise RuntimeError("provider response did not contain one choice")
    message = choices[0].get("message")
    if type(message) is not dict or type(message.get("content")) is not str:
        raise RuntimeError("provider response content was missing")
    return message["content"], body


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


def _usage_cost(body: dict[str, object]) -> float | None:
    usage = body.get("usage")
    if type(usage) is not dict:
        return None
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if type(prompt_tokens) is not int or type(completion_tokens) is not int:
        return None
    return (
        prompt_tokens * INPUT_USD_PER_MILLION
        + completion_tokens * OUTPUT_USD_PER_MILLION
    ) / 1_000_000


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.timeout_seconds <= 0:
        raise ValueError("timeout must be positive")
    token = _load_token(args.token_env)
    manifest = _load_json_from_opened_file(
        args.manifest,
        "smoke manifest",
        expected_sha256=args.manifest_sha256,
        private=True,
    )
    cases = _load_cases(manifest)
    if len(cases) != 10:
        raise ValueError("compact smoke requires exactly ten cases")
    contract = load_prompt_contract(args.protocol_dir, version=args.version)
    normalizer = {
        "v3": normalize_judge_decision_v3,
        "v4": normalize_judge_decision_v4,
        "v5": normalize_judge_decision_v5,
        "v6": normalize_judge_decision_v5,
    }[args.version]
    schema_hash = _sha256_json(contract.schema)
    prompt_hash = hashlib.sha256(contract.prompt.encode("utf-8")).hexdigest()
    records: list[dict[str, object]] = []
    started = time.perf_counter()
    for ordinal, case in enumerate(cases):
        record: dict[str, object] = {
            "ordinal": ordinal,
            "question_id": case.question_id,
            "response_id": case.response_id,
            "selection_reason": case.selection_reason,
        }
        try:
            raw, body = _request(
                token=token,
                prompt=contract.prompt,
                schema=contract.schema,
                case=case,
                timeout_seconds=args.timeout_seconds,
                thinking=args.thinking,
            )
            normalized = normalizer(raw)
            record["normalized"] = normalized
            record["finish_reason"] = (
                body.get("choices", [{}])[0].get("finish_reason")
            )
            usage = body.get("usage")
            if type(usage) is dict:
                record["usage"] = {
                    key: usage[key]
                    for key in ("prompt_tokens", "completion_tokens", "total_tokens")
                    if key in usage
                }
            cost = _usage_cost(body)
            if cost is not None:
                record["estimated_cost_usd"] = cost
        except Exception as error:  # one failed call must not hide the other nine
            record["provider_error"] = str(error)
        records.append(record)
        _write_private(
            args.output,
            {
                "version": 1,
                "contract_version": args.version,
                "model": MODEL_ID,
                "endpoint": ENDPOINT,
                "prompt_sha256": prompt_hash,
                "schema_sha256": schema_hash,
                "request_count": len(records),
                "records": records,
            },
        )
    clean = coerced = unscorable = provider_errors = 0
    estimated_cost = 0.0
    for record in records:
        if "provider_error" in record:
            provider_errors += 1
            continue
        status = record["normalized"]["parse_status"]
        clean += status == "clean"
        coerced += status == "coerced"
        unscorable += status == "unscorable"
        estimated_cost += float(record.get("estimated_cost_usd", 0.0))
    result = {
        "version": 1,
        "contract_version": args.version,
        "model": MODEL_ID,
        "endpoint": ENDPOINT,
        "reasoning_enabled": args.thinking,
        "max_output_tokens": (
            THINKING_MAX_OUTPUT_TOKENS if args.thinking else MAX_OUTPUT_TOKENS
        ),
        "prompt_sha256": prompt_hash,
        "schema_sha256": schema_hash,
        "request_count": len(records),
        "clean": clean,
        "coerced": coerced,
        "unscorable": unscorable,
        "provider_errors": provider_errors,
        "estimated_cost_usd": estimated_cost,
        "elapsed_seconds": time.perf_counter() - started,
        "records": records,
    }
    _write_private(args.output, result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))
    return 0 if provider_errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
