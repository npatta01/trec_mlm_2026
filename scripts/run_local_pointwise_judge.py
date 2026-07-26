#!/usr/bin/env python3
"""Run one local model over a private pointwise-judge smoke manifest."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import stat
from typing import Any, Sequence

from trec_mlm.pointwise_judge import CandidateCase, load_prompt_contract
from trec_mlm.pointwise_runner import (
    call_openai_judge,
    run_cases,
    validate_loopback_endpoint,
)


_CASE_KEYS = {
    "question_id",
    "response_id",
    "question",
    "reference_answer",
    "candidate_answer",
    "strata",
    "selection_reason",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def _load_private_json(path: Path, description: str) -> dict[str, Any]:
    if stat.S_IMODE(path.stat().st_mode) != 0o600:
        raise ValueError(f"{description} must have mode 0600")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{description} must be a JSON object")
    return value


def _load_cases(path: Path) -> list[CandidateCase]:
    manifest = _load_private_json(path, "smoke manifest")
    rows = manifest.get("cases")
    if type(rows) is not list or not rows:
        raise ValueError("smoke manifest must contain nonempty cases")
    cases: list[CandidateCase] = []
    for row in rows:
        if type(row) is not dict or set(row) != _CASE_KEYS:
            raise ValueError("smoke manifest case must contain exact keys")
        values = dict(row)
        strata = values["strata"]
        if type(strata) is not list or any(type(item) is not str for item in strata):
            raise ValueError("smoke manifest case strata must be strings")
        values["strata"] = tuple(strata)
        if any(
            type(values[key]) is not str or not values[key]
            for key in _CASE_KEYS - {"strata"}
        ):
            raise ValueError("smoke manifest case strings must not be empty")
        cases.append(CandidateCase(**values))
    if len({case.response_id for case in cases}) != len(cases):
        raise ValueError("smoke manifest response IDs must be unique")
    return cases


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key in private input: {key}")
        value[key] = item
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    endpoint = validate_loopback_endpoint(args.base_url)
    cases = _load_cases(args.manifest)
    receipt = _load_private_json(args.runtime_receipt, "runtime receipt")
    contract = load_prompt_contract(args.protocol_dir)
    call = partial(
        call_openai_judge,
        endpoint,
        timeout_seconds=args.timeout_seconds,
    )
    run_cases(
        cases,
        model_alias=args.model_alias,
        model_id=args.model_id,
        contract=contract,
        output_path=args.output,
        runtime_receipt=receipt,
        call=call,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
