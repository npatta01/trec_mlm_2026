#!/usr/bin/env python3
"""Run the frozen pointwise smoke through a compatible local OpenAI backend."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import urllib.request

from scripts.run_local_pointwise_judge import _load_cases, _load_json_from_opened_file
from trec_mlm.pointwise_judge import load_prompt_contract
from trec_mlm.pointwise_runner import (
    MAX_RESPONSE_BYTES,
    FROZEN_DECISION_SCHEMA_SHA256,
    _open_no_redirect,
    _sha256_json,
    run_cases,
    validate_loopback_endpoint,
)


def _validate_backend_response_model(body: object, expected_model: str) -> None:
    if type(body) is not dict or body.get("model") != expected_model:
        raise ValueError("backend response model does not match checkpoint")


def _lemonade_transport_schema(schema: object) -> dict[str, object]:
    """Return the minimal Lemonade 11.5 adaptation of the frozen schema."""
    if (
        type(schema) is not dict
        or _sha256_json(schema) != FROZEN_DECISION_SCHEMA_SHA256
    ):
        raise ValueError("transport schema is not the exact frozen decision schema")
    transport_schema = copy.deepcopy(schema)
    if transport_schema["properties"]["rationale"].pop("maxLength") != 4000:
        raise ValueError("frozen rationale length constraint changed")
    return transport_schema


def _runtime_observation(
    schema: object,
    *,
    backend_model: str,
    backend_response_model: str,
    thinking: bool,
    thinking_control_omitted: bool,
    reasoning_effort: str | None,
    max_output_tokens: int,
) -> dict[str, object]:
    transport_schema = _lemonade_transport_schema(schema)
    return {
        "schema_sha256": FROZEN_DECISION_SCHEMA_SHA256,
        "transport_schema_sha256": _sha256_json(transport_schema),
        "transport": "lemonade-11.5-llamacpp-rocm",
        "backend_model": backend_model,
        "backend_response_model": backend_response_model,
        "thinking": thinking,
        "thinking_control_omitted": thinking_control_omitted,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "schema_transport_adjustment": (
            "rationale.maxLength omitted; frozen parser retained"
        ),
    }


def _parse_backend_response_body(
    raw_body: bytes, expected_model: str
) -> tuple[str, dict[str, object]]:
    try:
        body = json.loads(
            raw_body.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("backend response body is not valid UTF-8 JSON") from error
    if type(body) is not dict:
        raise ValueError("backend response body must be a JSON object")
    _validate_backend_response_model(body, expected_model)
    choices = body.get("choices")
    if type(choices) is not list or not choices or type(choices[0]) is not dict:
        raise ValueError("backend response must contain nonempty choices")
    message = choices[0].get("message")
    if type(message) is not dict or type(message.get("content")) is not str:
        raise ValueError("backend response choice must contain string content")
    return message["content"], body


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    body: dict[str, object] = {}
    for key, value in pairs:
        if key in body:
            raise ValueError("duplicate JSON key in backend response body")
        body[key] = value
    return body


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--backend-model", required=True)
    parser.add_argument("--backend-response-model")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-output-tokens", type=int, default=384)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--omit-thinking-control", action="store_true")
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"))
    parser.add_argument("--timeout-seconds", type=float, default=300.0)
    args = parser.parse_args()

    manifest = _load_json_from_opened_file(
        args.manifest,
        "smoke manifest",
        expected_sha256=args.manifest_sha256,
        private=True,
    )
    cases = _load_cases(manifest)
    contract = load_prompt_contract(args.protocol_dir)
    endpoint = validate_loopback_endpoint(args.base_url)

    def call(_model: str, messages: object, schema: object):
        transport_schema = _lemonade_transport_schema(schema)
        payload = {
            "model": args.backend_model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": args.max_output_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "pointwise_judge_decision",
                    "strict": True,
                    "schema": transport_schema,
                },
            },
        }
        if not args.omit_thinking_control:
            payload["chat_template_kwargs"] = {"enable_thinking": args.thinking}
        if args.reasoning_effort is not None:
            payload["reasoning_effort"] = args.reasoning_effort
        request = urllib.request.Request(
            f"{endpoint}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json"},
            method="POST",
        )
        with _open_no_redirect(request, args.timeout_seconds) as response:
            raw_body = response.read(MAX_RESPONSE_BYTES + 1)
        content, body = _parse_backend_response_body(
            raw_body, args.backend_response_model or args.backend_model
        )
        return content, body

    runtime_observation = _runtime_observation(
        contract.schema,
        backend_model=args.backend_model,
        backend_response_model=args.backend_response_model or args.backend_model,
        thinking=args.thinking,
        thinking_control_omitted=args.omit_thinking_control,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
    )
    run_cases(
        cases,
        model_alias=args.model_alias,
        model_id=args.model_id,
        contract=contract,
        output_path=args.output,
        runtime_receipt=runtime_observation,
        call=call,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
