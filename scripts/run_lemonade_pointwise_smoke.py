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
    run_cases,
    validate_loopback_endpoint,
)


def _validate_backend_response_model(body: object, expected_model: str) -> None:
    if type(body) is not dict or body.get("model") != expected_model:
        raise ValueError("backend response model does not match checkpoint")


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
        transport_schema = copy.deepcopy(schema)
        transport_schema["properties"]["rationale"].pop("maxLength")
        transport_schema["properties"]["error_flags"].pop("uniqueItems")
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
            body = json.loads(response.read(MAX_RESPONSE_BYTES + 1))
        _validate_backend_response_model(
            body, args.backend_response_model or args.backend_model
        )
        body["model"] = _model
        content = body["choices"][0]["message"]["content"]
        body["usage"] = {
            key: value
            for key, value in body.get("usage", {}).items()
            if type(value) in {int, float}
        }
        return content, body

    runtime_observation = {
        "schema_sha256": FROZEN_DECISION_SCHEMA_SHA256,
        "transport": "lemonade-11.5-llamacpp-rocm",
        "backend_model": args.backend_model,
        "backend_response_model": args.backend_response_model or args.backend_model,
        "thinking": args.thinking,
        "thinking_control_omitted": args.omit_thinking_control,
        "reasoning_effort": args.reasoning_effort,
        "max_output_tokens": args.max_output_tokens,
        "schema_transport_adjustment": (
            "rationale.maxLength and error_flags.uniqueItems omitted; "
            "frozen parser retained"
        ),
    }
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
