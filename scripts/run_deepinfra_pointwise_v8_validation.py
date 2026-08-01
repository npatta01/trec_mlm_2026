#!/usr/bin/env python3
"""Run a resumable v8, v9, or v10 validation through DeepInfra direct calls."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import time
from typing import Any

import requests

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_deepinfra_pointwise_batch import _sha256_json, _write_private  # noqa: E402
from scripts.run_deepinfra_pointwise_production_batch import (  # noqa: E402
    ProductionCase,
    _load_inventory,
)
from trec_mlm.pointwise_judge import (  # noqa: E402
    PromptContract,
    load_prompt_contract,
    normalize_judge_decision_v8,
    normalize_judge_decision_v9,
    normalize_judge_decision_v10,
)
from trec_mlm.pointwise_production import FROZEN_GROUNDED_SHA256  # noqa: E402


ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
PROMPT_CACHE_KEY = "trec-mlm-pointwise-v8-reference-grounded"
PROMPT_CACHE_KEYS = {
    "v8": PROMPT_CACHE_KEY,
    "v9": "trec-mlm-pointwise-v9-few-shot",
    "v10": "trec-mlm-pointwise-v10-evaluator-evidence",
}
MAX_TOKENS = 8192
MAX_RESPONSE_BYTES = 1024 * 1024
TARGET_COUNT = 50
EVALUATOR_EVIDENCE_TRANSFORM_VERSION = "v10-evaluator-evidence-1"

_BARE_IDK_CANONICAL = frozenset(
    {
        "idontknow",
        "idonimgurtknow",
        "idonassistanttknow",
        "idonttknow",
        "idon39tknow",
        "idonnotknow",
        "idon0tknow",
        "idontmtknow",
        "idon92tknow",
        "idon42tknow",
        "idon2017tknow",
        "idonrestknow",
        "idonntknow",
        "idonimgurquoterightknow",
        "idonatknow",
        "idon5tknow",
        "idon57tknow",
        "idon2019tknow",
        "idonotknow",
    }
)
_VERIFIED_BRACKETED_YEARS = {
    "5a7286cf5542990c210a4129": "[1971]",
    "5ab29e0b554299194fa9346b": "[1978]",
}
_NUMERIC_SQUARE_BRACKET = re.compile(r"\[\d+\]")
_EXPLICIT_SOURCE_SECTION = re.compile(
    r"(?:^|\r?\n)[ \t]*(?:sources|references)[ \t]*:[ \t]*(?:\r?\n|$)",
    re.IGNORECASE,
)


class ProviderHTTPError(RuntimeError):
    """A sanitized provider error carrying only status and bounded body text."""

    def __init__(self, status: int) -> None:
        self.status = status
        super().__init__(f"DeepInfra HTTP {status}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-manifest", type=Path, required=True)
    parser.add_argument("--grounding-cache", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetition", choices=("a", "b"), required=True)
    parser.add_argument(
        "--contract-version", choices=("v8", "v9", "v10"), default="v8"
    )
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--ordinals", type=int, nargs="+", required=True)
    parser.add_argument("--api-key-env", default="DEEPINFRA_API_TOKEN")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=150.0)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _token(name: str) -> str:
    value = os.environ.get(name)
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _is_bare_idk_response(text: str) -> bool:
    """Return whether text is one frozen full-string bare-IDK encoding."""
    if type(text) is not str:
        raise TypeError("candidate response must be a string")
    canonical = "".join(
        character for character in text.casefold() if character.isascii() and character.isalnum()
    )
    return canonical in _BARE_IDK_CANONICAL


def _evaluator_evidence(question_id: str, answer: str) -> str:
    """Validate frozen Gemini text and return it without rewriting it."""
    if type(question_id) is not str or type(answer) is not str:
        raise TypeError("question_id and answer must be strings")
    lowered = answer.casefold()
    if any(marker in lowered for marker in ("http://", "https://", "www.")):
        raise ValueError("evaluator evidence contains a URL")
    if _EXPLICIT_SOURCE_SECTION.search(answer):
        raise ValueError("evaluator evidence contains an explicit source section")
    allowed_marker = _VERIFIED_BRACKETED_YEARS.get(question_id)
    for marker in _NUMERIC_SQUARE_BRACKET.findall(answer):
        if marker != allowed_marker:
            raise ValueError("evaluator evidence contains an unexpected numeric square-bracket marker")
    return answer


def _user_message(
    case: Any,
    *,
    contract_version: str = "v8",
    evaluator_evidence: str | None = None,
) -> str:
    if contract_version == "v10":
        if type(evaluator_evidence) is not str:
            raise ValueError("v10 requires validated evaluator evidence")
        return (
            f"ORIGINAL QUESTION:\n{case.question}\n\n"
            "GEMINI GROUNDED ANSWER — EVALUATOR EVIDENCE ONLY:\n"
            f"{evaluator_evidence}\n\n"
            "HISTORICAL MODEL RESPONSE TO SCORE:\n"
            f"{case.candidate_answer}"
        )
    return (
        f"QUESTION ID: {case.question_id}\n"
        f"RESPONSE ID: {case.response_id}\n\n"
        f"QUESTION:\n{case.question}\n\n"
        "PROVISIONAL GROUNDED REFERENCE:\n"
        f"{case.reference_answer}\n\n"
        f"CANDIDATE RESPONSE:\n{case.candidate_answer}"
    )


def _build_body(
    contract: PromptContract | Any,
    case: ProductionCase | Any,
    *,
    contract_version: str = "v8",
    reasoning_enabled: bool = False,
    evaluator_evidence: str | None = None,
) -> dict[str, Any]:
    """Build the frozen direct-mode DeepInfra request body."""
    reasoning_enabled = reasoning_enabled or contract_version == "v10"
    cache_key = PROMPT_CACHE_KEYS[contract_version]
    if reasoning_enabled:
        cache_key += "-thinking"
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": contract.prompt},
            {
                "role": "user",
                "content": _user_message(
                    case,
                    contract_version=contract_version,
                    evaluator_evidence=evaluator_evidence,
                ),
            },
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "prompt_cache_key": cache_key,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": f"pointwise_judge_{contract_version}_decision",
                "strict": True,
                "schema": contract.schema,
            },
        },
    }
    if reasoning_enabled:
        body["reasoning_effort"] = "high"
    return body


def _retryable_status(status: int) -> bool:
    return status in {408, 429} or 500 <= status <= 599


def _post_once(
    body: dict[str, Any], *, token: str, timeout_seconds: float
) -> dict[str, Any]:
    response = requests.post(
        ENDPOINT,
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
    if len(response.content) > MAX_RESPONSE_BYTES:
        raise RuntimeError("DeepInfra response exceeded the private size limit")
    try:
        value = response.json()
    except requests.JSONDecodeError as error:
        raise RuntimeError("DeepInfra response was not JSON") from error
    if type(value) is not dict:
        raise RuntimeError("DeepInfra response was not an object")
    return value


def _post_with_one_transport_retry(
    body: dict[str, Any], *, token: str, timeout_seconds: float
) -> tuple[dict[str, Any], int]:
    for attempt in (1, 2):
        try:
            return (
                _post_once(body, token=token, timeout_seconds=timeout_seconds),
                attempt,
            )
        except ProviderHTTPError as error:
            if attempt == 2 or not _retryable_status(error.status):
                raise
        except (requests.Timeout, requests.ConnectionError):
            if attempt == 2:
                raise
        time.sleep(1.0)
    raise AssertionError("retry loop did not return or raise")


def _content(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise RuntimeError("provider response did not contain exactly one choice")
    message = choices[0].get("message")
    if type(message) is not dict or type(message.get("content")) is not str:
        raise RuntimeError("provider response content was missing")
    return message["content"]


def _grounding_tier(record: dict[str, Any]) -> str:
    if record.get("uncertainty_status") != "answered":
        return "unanswered_or_failed"
    metadata = record.get("request_metadata")
    query_count = metadata.get("grounding_query_count") if type(metadata) is dict else None
    return (
        "search_grounded"
        if type(query_count) is int and query_count > 0
        else "generated_without_search"
    )


def _base_record(
    *,
    case: ProductionCase | Any,
    repetition: str,
    reference_tier: str,
    contract_version: str = "v8",
    evaluator_evidence: str | None = None,
) -> dict[str, Any]:
    record = {
        "ordinal": case.ordinal,
        "source_ordinal": case.source_ordinal,
        "record_id": case.record_id,
        "response_id": case.response_id,
        "question_id": case.question_id,
        "candidate_id": case.candidate_id,
        "selection_reason": "targeted_v8_validation",
        "repetition": repetition,
        "reference_tier": reference_tier,
        "question": case.question,
        "candidate_response": case.candidate_answer,
        "transport": "deepinfra-direct",
    }
    if contract_version == "v10":
        if type(evaluator_evidence) is not str:
            raise ValueError("v10 records require derived evaluator evidence")
        record["evaluator_evidence"] = evaluator_evidence
    else:
        record["provisional_reference"] = case.reference_answer
    return record


def _record_from_body(
    body: dict[str, Any],
    *,
    case: ProductionCase | Any,
    repetition: str,
    reference_tier: str,
    attempt_count: int,
    contract_version: str = "v8",
    evaluator_evidence: str | None = None,
) -> dict[str, Any]:
    record = _base_record(
        case=case,
        repetition=repetition,
        reference_tier=reference_tier,
        contract_version=contract_version,
        evaluator_evidence=evaluator_evidence,
    )
    record["attempt_count"] = attempt_count
    normalizer = {
        "v8": normalize_judge_decision_v8,
        "v9": normalize_judge_decision_v9,
        "v10": normalize_judge_decision_v10,
    }[contract_version]
    record["normalized"] = normalizer(_content(body))
    record["provider_model"] = body.get("model")
    choices = body.get("choices")
    if type(choices) is list and choices and type(choices[0]) is dict:
        record["finish_reason"] = choices[0].get("finish_reason")
    usage = body.get("usage")
    if type(usage) is dict:
        record["usage"] = usage
        estimated = usage.get("estimated_cost")
        if type(estimated) in {int, float} and estimated >= 0:
            record["estimated_cost_usd"] = float(estimated)
    return record


def _error_record(
    error: Exception,
    *,
    case: ProductionCase,
    repetition: str,
    reference_tier: str,
    contract_version: str = "v8",
    evaluator_evidence: str | None = None,
) -> dict[str, Any]:
    record = _base_record(
        case=case,
        repetition=repetition,
        reference_tier=reference_tier,
        contract_version=contract_version,
        evaluator_evidence=evaluator_evidence,
    )
    if isinstance(error, ProviderHTTPError):
        error_code = f"http_{error.status}"
    elif isinstance(error, requests.Timeout):
        error_code = "timeout"
    elif isinstance(error, requests.ConnectionError):
        error_code = "connection_error"
    else:
        error_code = "unexpected_error"
    record["provider_error"] = error_code
    return record


def _read_private_object(path: Path, *, label: str) -> dict[str, Any]:
    details = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(details.st_mode) or stat.S_IMODE(details.st_mode) != 0o600:
        raise ValueError(f"{label} must be a regular mode-0600 file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{label} must contain an object")
    return value


def _load_grounding(path: Path) -> dict[str, Any]:
    if _sha256_file(path) != FROZEN_GROUNDED_SHA256:
        raise ValueError("grounding cache does not match the frozen SHA-256")
    cache = _read_private_object(path, label="grounding cache")
    records = cache.get("records")
    if type(records) is not dict:
        raise ValueError("grounding cache records are missing")
    return records


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {
        "request_count": len(records),
        "clean": 0,
        "coerced": 0,
        "unscorable": 0,
        "provider_errors": 0,
        "retried_requests": 0,
        "estimated_cost_usd": 0.0,
    }
    for record in records:
        if "provider_error" in record:
            summary["provider_errors"] += 1
            continue
        parse_status = record["normalized"]["parse_status"]
        summary[parse_status] += 1
        summary["retried_requests"] += record.get("attempt_count") == 2
        summary["estimated_cost_usd"] += float(
            record.get("estimated_cost_usd", 0.0)
        )
    return summary


def _record_identity(metadata: dict[str, Any], case: ProductionCase | Any) -> str:
    return _sha256_json(
        {
            "metadata": metadata,
            "ordinal": case.ordinal,
            "record_id": case.record_id,
        }
    )


def _record_cache_matches(
    record: dict[str, Any],
    *,
    case: ProductionCase | Any,
    repetition: str,
    cache_identity_sha256: str,
) -> bool:
    return (
        record.get("ordinal") == case.ordinal
        and record.get("record_id") == case.record_id
        and record.get("repetition") == repetition
        and record.get("cache_identity_sha256") == cache_identity_sha256
    )


def _validate_ordinal_count(contract_version: str, ordinals: list[int]) -> None:
    """Enforce the frozen validation sizes for each additive contract."""
    if len(set(ordinals)) != len(ordinals):
        raise ValueError("validation requires unique ordinals")
    allowed_counts = {"v8": (50,), "v9": (10, 100), "v10": (100,)}[contract_version]
    if len(ordinals) not in allowed_counts:
        expected = " or ".join(str(count) for count in allowed_counts)
        raise ValueError(
            f"{contract_version} validation requires exactly {expected} unique ordinals"
        )


def _validate_v10_selection(cases: list[ProductionCase | Any]) -> None:
    """Prevent known bare-IDK candidates from reaching the v10 judge."""
    for case in cases:
        if _is_bare_idk_response(case.candidate_answer):
            raise ValueError(f"v10 selection contains bare IDK at ordinal {case.ordinal}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.workers < 1 or args.timeout_seconds <= 0:
        raise ValueError("workers and timeout must be positive")
    thinking_enabled = args.thinking or args.contract_version == "v10"
    _validate_ordinal_count(args.contract_version, args.ordinals)
    token = _token(args.api_key_env)
    all_cases, inventory_manifest = _load_inventory(
        args.inventory, args.inventory_manifest
    )
    if len(all_cases) != 500:
        raise ValueError("validation requires the sealed 500-case inventory")
    if any(ordinal < 0 or ordinal >= len(all_cases) for ordinal in args.ordinals):
        raise ValueError("ordinal is outside the sealed inventory")
    selected = [all_cases[ordinal] for ordinal in args.ordinals]
    if args.contract_version == "v10":
        _validate_v10_selection(selected)
    contract = load_prompt_contract(args.protocol_dir, version=args.contract_version)
    grounding = _load_grounding(args.grounding_cache)
    reference_tiers: dict[str, str] = {}
    evaluator_evidence_by_question_id: dict[str, str] = {}
    for case in selected:
        grounded = grounding.get(case.question_id)
        if type(grounded) is not dict:
            raise ValueError(f"missing grounded reference for {case.question_id}")
        if grounded.get("answer") != case.reference_answer:
            raise ValueError(f"grounded reference mismatch for {case.question_id}")
        reference_tiers[case.question_id] = _grounding_tier(grounded)
        if args.contract_version == "v10":
            evaluator_evidence_by_question_id[case.question_id] = _evaluator_evidence(
                case.question_id, grounded["answer"]
            )

    prompt_sha256 = hashlib.sha256(contract.prompt.encode("utf-8")).hexdigest()
    schema_sha256 = _sha256_json(contract.schema)
    metadata = {
        "contract_version": args.contract_version,
        "inventory_manifest_sha256": inventory_manifest["manifest_sha256"],
        "inventory_sha256": inventory_manifest["inventory_sha256"],
        "grounding_sha256": FROZEN_GROUNDED_SHA256,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "model": MODEL,
        "provider": "DeepInfra",
        "transport": "deepinfra-thinking" if thinking_enabled else "deepinfra-direct",
        "fallbacks": False,
        "reasoning_effort": "high" if thinking_enabled else None,
        "temperature": 0,
        "max_tokens": MAX_TOKENS,
        "prompt_cache_key": (
            PROMPT_CACHE_KEYS[args.contract_version]
            + ("-thinking" if thinking_enabled else "")
        ),
        "selected_ordinals": args.ordinals,
        "repetition": args.repetition,
        "workers": args.workers,
        "timeout_seconds": args.timeout_seconds,
    }
    if args.contract_version == "v10":
        metadata["evaluator_evidence_transform_version"] = (
            EVALUATOR_EVIDENCE_TRANSFORM_VERSION
        )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    records_dir = output_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    records_dir.chmod(0o700)
    metadata_path = output_dir / "metadata.json"
    if metadata_path.exists():
        if _read_private_object(metadata_path, label="metadata") != metadata:
            raise ValueError("existing validation cache identity does not match")
    else:
        _write_private(metadata_path, metadata)

    cache_identities = {
        case.ordinal: _record_identity(metadata, case) for case in selected
    }

    records: dict[int, dict[str, Any]] = {}
    pending: list[ProductionCase] = []
    for case in selected:
        path = records_dir / f"{case.ordinal:06d}.json"
        if not path.exists():
            pending.append(case)
            continue
        record = _read_private_object(path, label=f"record {case.ordinal}")
        if not _record_cache_matches(
            record,
            case=case,
            repetition=args.repetition,
            cache_identity_sha256=cache_identities[case.ordinal],
        ):
            raise ValueError(f"record cache identity mismatch at {case.ordinal}")
        records[case.ordinal] = record

    def run(case: ProductionCase) -> dict[str, Any]:
        tier = reference_tiers[case.question_id]
        evaluator_evidence = evaluator_evidence_by_question_id.get(case.question_id)
        try:
            body, attempts = _post_with_one_transport_retry(
                _build_body(
                    contract,
                    case,
                    contract_version=args.contract_version,
                    reasoning_enabled=thinking_enabled,
                    evaluator_evidence=evaluator_evidence,
                ),
                token=token,
                timeout_seconds=args.timeout_seconds,
            )
            record = _record_from_body(
                body,
                case=case,
                repetition=args.repetition,
                reference_tier=tier,
                attempt_count=attempts,
                contract_version=args.contract_version,
                evaluator_evidence=evaluator_evidence,
            )
            record["transport"] = metadata["transport"]
            return record
        except Exception as error:
            record = _error_record(
                error,
                case=case,
                repetition=args.repetition,
                reference_tier=tier,
                contract_version=args.contract_version,
                evaluator_evidence=evaluator_evidence,
            )
            record["transport"] = metadata["transport"]
            return record

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run, case): case for case in pending}
        for future in as_completed(futures):
            case = futures[future]
            record = future.result()
            record["cache_identity_sha256"] = cache_identities[case.ordinal]
            _write_private(records_dir / f"{case.ordinal:06d}.json", record)
            records[case.ordinal] = record
            ordered = [records[i] for i in args.ordinals if i in records]
            checkpoint = {
                **metadata,
                "elapsed_seconds": time.perf_counter() - started,
                "records": ordered,
                "summary": _summary(ordered),
            }
            _write_private(output_dir / "result.json", checkpoint)

    ordered = [records[ordinal] for ordinal in args.ordinals]
    result = {
        **metadata,
        "elapsed_seconds": time.perf_counter() - started,
        "records": ordered,
        "summary": _summary(ordered),
    }
    _write_private(output_dir / "result.json", result)
    print(json.dumps({"cached": not pending, "summary": result["summary"]}))
    return 0 if result["summary"]["provider_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
