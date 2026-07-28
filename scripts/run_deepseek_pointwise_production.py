#!/usr/bin/env python3
"""Run (or credential-free validate) the sealed DeepSeek production runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import urllib.error
import urllib.request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_hosted_pointwise_smoke import (  # noqa: E402
    DEFAULT_API_KEY_FILE,
    OPENROUTER_ENDPOINT,
    V2_CONFIG_SPECS,
    _load_api_key,
    _post_json,
    _validate_router_metadata,
)
from trec_mlm.pointwise_production_runner import (  # noqa: E402
    MODEL_ID,
    PROVIDER,
    ProductionRunner,
    TransientProviderError,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-manifest", type=Path, required=True)
    parser.add_argument("--contract-receipt", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_API_KEY_FILE)
    parser.add_argument("--cap-usd", type=float, default=60.0)
    parser.add_argument("--worst-call-usd", type=float, default=0.01)
    parser.add_argument("--max-blocks", type=int)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate sealed inputs without network access",
    )
    return parser


class OpenRouterProductionCall:
    """One strict DeepSeek route call; retry policy belongs to ProductionRunner."""

    def __init__(self, *, api_key: str, prompt: str, schema: dict[str, object]) -> None:
        self._api_key, self._prompt, self._schema = api_key, prompt, schema
        self._spec = V2_CONFIG_SPECS["deepseek-v4-flash-direct"]

    def __call__(self, record: dict[str, object], repair: bool) -> dict[str, object]:
        user = (
            f"QUESTION ID: {record['question_id']}\nRESPONSE ID: {record['record_id']}\n\n"
            f"QUESTION:\n{record['question']}\n\n"
            "PROVISIONAL REFERENCE (may be incomplete or wrong):\n"
            f"{record['provisional_reference']}\n\nCANDIDATE RESPONSE:\n{record['candidate_response']}"
        )
        if repair:
            user += "\n\nYour preceding JSON was invalid. Return only one corrected JSON object under the unchanged schema."
        payload = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": user},
            ],
            "temperature": 0,
            "max_tokens": 512,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "pointwise_decision",
                    "strict": True,
                    "schema": self._schema,
                },
            },
            "provider": {"order": [PROVIDER], "allow_fallbacks": False},
            "reasoning": {"enabled": False},
        }
        request = urllib.request.Request(
            OPENROUTER_ENDPOINT,
            data=json.dumps(
                payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            ).encode("utf-8"),
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "X-OpenRouter-Metadata": "enabled",
            },
            method="POST",
        )
        try:
            body = _post_json(request, timeout_seconds=90.0)
        except urllib.error.HTTPError as error:
            if error.code in {408, 429} or error.code >= 500:
                raise TransientProviderError("retryable OpenRouter status") from error
            raise
        if type(body) is not dict:
            raise ValueError("OpenRouter response must be an object")
        _validate_router_metadata(body.get("openrouter_metadata"), self._spec)
        choices, usage = body.get("choices"), body.get("usage")
        if (
            type(choices) is not list
            or len(choices) != 1
            or type(choices[0]) is not dict
        ):
            raise ValueError("OpenRouter response must contain exactly one choice")
        choice = choices[0]
        message = choice.get("message")
        if (
            type(message) is not dict
            or type(message.get("content")) is not str
            or type(usage) is not dict
        ):
            raise ValueError("OpenRouter response content or usage is invalid")
        cost = usage.get("cost")
        if type(cost) not in {int, float}:
            raise ValueError("OpenRouter response cost is invalid")
        return {
            "content": message["content"],
            "finish_reason": choice.get("finish_reason"),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
            },
            "cost_usd": cost,
            "provider": PROVIDER,
            "model": MODEL_ID,
            "openrouter_response": body,
        }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_blocks is not None and args.max_blocks < 1:
        raise ValueError("--max-blocks must be positive")
    if args.dry_run:
        # Deliberately construct no transport or credential read on this path.
        runner = ProductionRunner(
            inventory=args.inventory,
            inventory_manifest=args.inventory_manifest,
            contract_receipt=args.contract_receipt,
            run_dir=args.run_dir,
            call=lambda _record, _repair: (_ for _ in ()).throw(
                RuntimeError("dry-run call")
            ),
            total_cap_usd=args.cap_usd,
            worst_call_usd=args.worst_call_usd,
        )
        print(
            json.dumps(
                {
                    "validated": True,
                    "record_count": runner._record_count,
                    "run_dir": str(args.run_dir),
                }
            )
        )
        return 0
    key = _load_api_key(args.api_key_file)
    # Construction validates all sealed identity inputs before any provider request.
    probe = ProductionRunner(
        inventory=args.inventory,
        inventory_manifest=args.inventory_manifest,
        contract_receipt=args.contract_receipt,
        run_dir=args.run_dir,
        call=lambda _record, _repair: (_ for _ in ()).throw(
            RuntimeError("transport not installed")
        ),
        total_cap_usd=args.cap_usd,
        worst_call_usd=args.worst_call_usd,
    )
    transport = OpenRouterProductionCall(
        api_key=key,
        prompt=str(probe.contract["prompt"]),
        schema=dict(probe.contract["schema"]),
    )
    runner = ProductionRunner(
        inventory=args.inventory,
        inventory_manifest=args.inventory_manifest,
        contract_receipt=args.contract_receipt,
        run_dir=args.run_dir,
        call=transport,
        total_cap_usd=args.cap_usd,
        worst_call_usd=args.worst_call_usd,
    )
    print(json.dumps(runner.run(max_blocks=args.max_blocks), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
