#!/usr/bin/env python3
"""Run the focused GPT-5.4 Nano medium-reasoning verdict validation."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
from typing import Any

import requests

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_pointwise_v11_adjudication import provider_message_fixture
from scripts.run_pointwise_v11_bakeoff import (
    AUTHORIZED_OUTPUT_ROOT,
    REPO_ROOT,
    _private_dir,
    _read_private_json,
    _sha256_json,
    _sha256_text,
    _validate_manifest,
    _validate_provider_text,
    _write_private_json,
)


ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "openai/gpt-5.4-nano"
PROVIDER = "OpenAI"
REASONING = {"effort": "medium"}
MAX_TOKENS = 4096
VERDICTS = frozenset(
    {"correct", "partial", "incorrect", "not_established", "nonanswer"}
)
PROMPT = """Judge a HISTORICAL RESPONSE against the supplied GEMINI EVALUATOR EVIDENCE. Use only that evidence, not outside knowledge. Return exactly one JSON object with verdict and explanation.

correct: central answer supported and no material contradiction; omissions, brevity, hedging, irrelevant tails, and peripheral evidence-silent details do not prevent correct.
partial: useful supported answer content plus a material claim explicitly contradicted by the evidence.
incorrect: substantive attempt at the intended question, but the central answer is explicitly contradicted and there is no useful supported answer.
not_established: substantive attempt at the intended question, but the evidence neither supports nor explicitly contradicts its central answer.
nonanswer: no substantive answer to the intended question, or wrong subject/referent.

Decision order: first verify the intended subject/referent; then explicit contradiction; then support. Evidence silence is not contradiction. Correctness matters more than completeness. Keep explanation to one short sentence.

Examples: Asking for two effects and giving one evidence-supported effect is correct. Lisbon 1913 against Lisbon 1912 is partial if Lisbon is useful support. Answering the western river when asked for the eastern river is nonanswer. Claiming an export when evidence only discusses railways is not_established. Saying service decreased when evidence says it increased, with no useful support, is incorrect."""
SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {"type": "string", "enum": sorted(VERDICTS)},
        "explanation": {"type": "string", "maxLength": 200},
    },
    "required": ["verdict", "explanation"],
}


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


def _body(record: dict[str, object]) -> dict[str, object]:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": PROMPT},
            {"role": "user", "content": _user_message(record)},
        ],
        "stream": False,
        "max_tokens": MAX_TOKENS,
        "usage": {"include": True},
        "reasoning": REASONING,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pointwise_simple_verdict",
                "strict": True,
                "schema": SCHEMA,
            },
        },
        "provider": {
            "only": [PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
        },
    }


def _parse_response(value: dict[str, Any]) -> tuple[dict[str, str], dict[str, Any]]:
    if value.get("model") != MODEL or value.get("provider") != PROVIDER:
        raise ValueError("OpenRouter returned an unexpected route")
    choices = value.get("choices")
    if (
        type(choices) is not list
        or len(choices) != 1
        or type(choices[0]) is not dict
        or choices[0].get("finish_reason") != "stop"
    ):
        raise ValueError("provider response did not finish normally")
    message = choices[0].get("message")
    if type(message) is not dict or type(message.get("content")) is not str:
        raise ValueError("provider response content is missing")
    try:
        decision = json.loads(message["content"])
    except json.JSONDecodeError as error:
        raise ValueError("provider decision is not JSON") from error
    if (
        type(decision) is not dict
        or set(decision) != {"verdict", "explanation"}
        or decision.get("verdict") not in VERDICTS
        or type(decision.get("explanation")) is not str
        or not decision["explanation"].strip()
        or len(decision["explanation"]) > 200
    ):
        raise ValueError("provider decision violates the simple contract")
    usage = value.get("usage")
    if type(usage) is not dict:
        raise ValueError("provider usage is missing")
    cost = usage.get("cost")
    if (
        type(cost) is bool
        or not isinstance(cost, (int, float))
        or not math.isfinite(cost)
        or cost < 0
    ):
        raise ValueError("provider cost is missing or invalid")
    return decision, usage


def _call(record: dict[str, object], *, token: str, timeout: float) -> dict[str, object]:
    response = requests.post(
        ENDPOINT,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        json=_body(record),
        timeout=timeout,
    )
    response.raise_for_status()
    value = response.json()
    if type(value) is not dict:
        raise ValueError("provider response is not an object")
    decision, usage = _parse_response(value)
    return {
        "ordinal": record["ordinal"],
        "manifest_expected": record["expected_v11_derived_label"],
        "verdict": decision["verdict"],
        "explanation": decision["explanation"],
        "provider": value["provider"],
        "model": value["model"],
        "finish_reason": value["choices"][0]["finish_reason"],
        "usage": usage,
        "cost_usd": float(usage["cost"]),
    }


def _output_dir(path: Path) -> Path:
    root = (REPO_ROOT / AUTHORIZED_OUTPUT_ROOT).resolve()
    output = path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()
    if output == root or not output.is_relative_to(root):
        raise ValueError("output must stay inside private v11 artifacts")
    return output


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cohort", choices=("original100",), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-cost-usd", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=150.0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    token = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not token:
        raise ValueError("OPENROUTER_API_KEY is required")
    if not 1 <= args.workers <= 8 or args.max_cost_usd <= 0:
        raise ValueError("workers or cost cap is invalid")
    manifest = _read_private_json(args.manifest, label="adjudication manifest")
    records = _validate_manifest(manifest)
    ordinals = manifest["selection"]["original_100_ordinals"]
    if type(ordinals) is not list or len(ordinals) != 100:
        raise ValueError("original100 cohort is invalid")
    output = _output_dir(args.output_dir)
    _private_dir(output)
    records_dir = output / "records"
    _private_dir(records_dir)
    metadata = {
        "version": 1,
        "manifest_sha256": manifest["manifest_sha256"],
        "model": MODEL,
        "provider": PROVIDER,
        "allow_fallbacks": False,
        "reasoning": REASONING,
        "max_tokens": MAX_TOKENS,
        "prompt_sha256": _sha256_text(PROMPT),
        "schema_sha256": _sha256_json(SCHEMA),
        "ordinals": ordinals,
    }
    metadata_path = output / "metadata.json"
    if metadata_path.exists():
        if _read_private_json(metadata_path, label="simple verdict metadata") != metadata:
            raise ValueError("existing output has a different run identity")
    else:
        _write_private_json(metadata_path, metadata)
    completed: dict[int, dict[str, object]] = {}
    pending: list[dict[str, object]] = []
    for ordinal in ordinals:
        path = records_dir / f"{ordinal:06d}.json"
        identity = _sha256_json({"metadata": metadata, "ordinal": ordinal})
        if path.exists():
            record = _read_private_json(path, label=f"simple verdict {ordinal}")
            if record.get("cache_identity_sha256") != identity:
                raise ValueError(f"cache identity mismatch for ordinal {ordinal}")
            completed[ordinal] = record
        else:
            pending.append(records[ordinal])
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _call, record, token=token, timeout=args.timeout_seconds
            ): record
            for record in pending
        }
        failures: list[dict[str, object]] = []
        for future in as_completed(futures):
            source = futures[future]
            ordinal = source["ordinal"]
            try:
                result = future.result()
            except Exception as error:
                failures.append(
                    {
                        "ordinal": ordinal,
                        "error": f"{type(error).__name__}:{error}",
                    }
                )
                continue
            result["cache_identity_sha256"] = _sha256_json(
                {"metadata": metadata, "ordinal": ordinal}
            )
            _write_private_json(records_dir / f"{ordinal:06d}.json", result)
            completed[ordinal] = result
            spent = sum(float(row["cost_usd"]) for row in completed.values())
            if spent > args.max_cost_usd + 1e-12:
                raise RuntimeError("provider cost exceeded max-cost-usd")
    if failures:
        partial = {
            "metadata": metadata,
            "summary": {
                "completed": len(completed),
                "failures": failures,
                "recorded_cost_usd": sum(
                    float(row["cost_usd"]) for row in completed.values()
                ),
            },
        }
        _write_private_json(output / "partial-result.json", partial)
        print(json.dumps(partial["summary"], sort_keys=True))
        return 1
    ordered = [completed[ordinal] for ordinal in ordinals]
    total_cost = sum(float(row["cost_usd"]) for row in ordered)
    agreement = sum(
        row["verdict"] == row["manifest_expected"] for row in ordered
    )
    result = {
        "metadata": metadata,
        "summary": {
            "records": len(ordered),
            "schema_valid": len(ordered),
            "manifest_exact_agreement": agreement,
            "cost_usd": total_cost,
        },
        "records": ordered,
    }
    _write_private_json(output / "result.json", result)
    print(json.dumps(result["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
