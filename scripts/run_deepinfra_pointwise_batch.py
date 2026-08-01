#!/usr/bin/env python3
"""Run the v6 pointwise judge through DeepInfra's discounted Batch API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import requests

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_local_pointwise_judge import (  # noqa: E402
    _load_cases,
    _load_json_from_opened_file,
)
from scripts.run_pointwise_v4_bakeoff import _user_message  # noqa: E402
from trec_mlm.pointwise_judge import (  # noqa: E402
    load_prompt_contract,
    normalize_judge_decision_v5,
)


BASE_URL = "https://api.deepinfra.com/v1"
CHAT_ENDPOINT = "/v1/chat/completions"
MODEL = "deepseek-ai/DeepSeek-V4-Flash"
PROMPT_CACHE_KEY = "trec-mlm-pointwise-v6-deepseek-thinking"
MAX_RESPONSE_BYTES = 32 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="DEEPINFRA_API_TOKEN")
    parser.add_argument("--poll-seconds", type=float, default=20.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--ordinals",
        type=int,
        nargs="+",
        help="optional original case ordinals to run; the full manifest is still validated",
    )
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


def _write_private_text(path: Path, value: str) -> None:
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
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{path} must contain an object")
    return value


def _token(name: str) -> str:
    value = os.environ.get(name)
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
    }


def _json_request(
    method: str,
    url: str,
    *,
    token: str,
    timeout_seconds: float,
    **kwargs: Any,
) -> dict[str, Any]:
    response = requests.request(
        method,
        url,
        headers={**_headers(token), "Content-Type": "application/json"},
        timeout=timeout_seconds,
        **kwargs,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"DeepInfra HTTP {response.status_code}: {response.text[:500]}")
    value = response.json()
    if type(value) is not dict:
        raise RuntimeError("DeepInfra response was not an object")
    return value


def _build_body(contract: Any, case: Any, *, max_tokens: int) -> dict[str, Any]:
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": contract.prompt},
            {"role": "user", "content": _user_message(case)},
        ],
        "stream": False,
        "temperature": 0,
        "max_tokens": max_tokens,
        "reasoning_effort": "high",
        "prompt_cache_key": PROMPT_CACHE_KEY,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pointwise_judge_decision",
                "strict": True,
                "schema": contract.schema,
            },
        },
    }


def _request_lines(
    cases: list[tuple[int, Any]], contract: Any, *, max_tokens: int
) -> list[str]:
    return [
        json.dumps(
            {
                "custom_id": f"pointwise-{ordinal:03d}",
                "method": "POST",
                "url": CHAT_ENDPOINT,
                "body": _build_body(contract, case, max_tokens=max_tokens),
            },
            ensure_ascii=False,
            allow_nan=False,
        )
        for ordinal, case in cases
    ]


def _upload_batch_file(
    path: Path, *, token: str, timeout_seconds: float
) -> dict[str, Any]:
    with path.open("rb") as handle:
        response = requests.post(
            f"{BASE_URL}/files",
            headers=_headers(token),
            files={"file": (path.name, handle, "application/jsonl")},
            data={"purpose": "batch"},
            timeout=timeout_seconds,
        )
    if response.status_code >= 400:
        raise RuntimeError(f"DeepInfra file upload failed: {response.text[:500]}")
    value = response.json()
    if type(value) is not dict or type(value.get("id")) is not str:
        raise RuntimeError("DeepInfra file upload did not return a file ID")
    return value


def _create_batch(
    input_file_id: str,
    *,
    token: str,
    timeout_seconds: float,
    manifest_sha256: str,
    prompt_sha256: str,
    schema_sha256: str,
) -> dict[str, Any]:
    return _json_request(
        "POST",
        f"{BASE_URL}/batches",
        token=token,
        timeout_seconds=timeout_seconds,
        json={
            "input_file_id": input_file_id,
            "endpoint": CHAT_ENDPOINT,
            "completion_window": "24h",
            "metadata": {
                "purpose": "trec-mlm-pointwise-v6-thinking",
                "manifest_sha256": manifest_sha256,
                "prompt_sha256": prompt_sha256,
                "schema_sha256": schema_sha256,
                "model": MODEL,
            },
        },
    )


def _poll_batch(
    batch_id: str,
    *,
    token: str,
    poll_seconds: float,
    timeout_seconds: float,
) -> dict[str, Any]:
    while True:
        batch = _json_request(
            "GET",
            f"{BASE_URL}/batches/{batch_id}",
            token=token,
            timeout_seconds=timeout_seconds,
        )
        status = batch.get("status")
        print(json.dumps({"batch_id": batch_id, "status": status, "request_counts": batch.get("request_counts")}))
        if status in {"completed", "failed", "expired", "cancelled"}:
            return batch
        time.sleep(poll_seconds)


def _download_jsonl(
    file_id: str, *, token: str, timeout_seconds: float
) -> list[dict[str, Any]]:
    response = requests.get(
        f"{BASE_URL}/files/{file_id}/content",
        headers=_headers(token),
        timeout=timeout_seconds,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"DeepInfra output download failed: {response.text[:500]}")
    raw = response.content
    if len(raw) > MAX_RESPONSE_BYTES:
        raise RuntimeError("DeepInfra output exceeded the private size limit")
    rows: list[dict[str, Any]] = []
    for line in raw.decode("utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if type(value) is not dict:
            raise RuntimeError("DeepInfra batch output line was not an object")
        rows.append(value)
    return rows


def _content(body: dict[str, Any]) -> str:
    choices = body.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise RuntimeError("batch response did not contain one choice")
    message = choices[0].get("message")
    if type(message) is not dict or type(message.get("content")) is not str:
        raise RuntimeError("batch response content was missing")
    return message["content"]


def _record_from_line(line: dict[str, Any], *, ordinal: int, case: Any) -> dict[str, Any]:
    record: dict[str, Any] = {
        "ordinal": ordinal,
        "question_id": case.question_id,
        "response_id": case.response_id,
        "selection_reason": case.selection_reason,
    }
    response = line.get("response")
    if type(response) is not dict:
        record["provider_error"] = line.get("error", "batch response was missing")
        return record
    if response.get("status_code") != 200 or type(response.get("body")) is not dict:
        record["provider_error"] = response.get("body", "batch request failed")
        return record
    body = response["body"]
    try:
        record["normalized"] = normalize_judge_decision_v5(_content(body))
    except Exception as error:
        record["provider_error"] = str(error)
        return record
    record["provider_model"] = body.get("model")
    choices = body.get("choices")
    if type(choices) is list and choices and type(choices[0]) is dict:
        record["finish_reason"] = choices[0].get("finish_reason")
    usage = body.get("usage")
    if type(usage) is dict:
        record["usage"] = usage
        if type(usage.get("estimated_cost")) in {int, float}:
            record["estimated_cost_usd"] = float(usage["estimated_cost"])
    return record


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
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
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        raise ValueError("poll and request timeouts must be positive")
    if args.max_tokens < 2048:
        raise ValueError("max-tokens must be at least 2048 for thinking runs")
    token = _token(args.api_key_env)
    manifest = _load_json_from_opened_file(
        args.manifest,
        "100-case manifest",
        expected_sha256=args.manifest_sha256,
        private=True,
    )
    cases = _load_cases(manifest)
    if len(cases) != 100:
        raise ValueError("DeepInfra batch run requires exactly 100 cases")
    if args.ordinals is None:
        selected_cases = list(enumerate(cases))
    else:
        if not args.ordinals or len(set(args.ordinals)) != len(args.ordinals):
            raise ValueError("ordinals must be non-empty and unique")
        if any(ordinal < 0 or ordinal >= len(cases) for ordinal in args.ordinals):
            raise ValueError("ordinal is outside the manifest")
        selected_cases = [(ordinal, cases[ordinal]) for ordinal in args.ordinals]
    contract = load_prompt_contract(args.protocol_dir, version="v6")
    prompt_sha256 = hashlib.sha256(contract.prompt.encode("utf-8")).hexdigest()
    schema_sha256 = _sha256_json(contract.schema)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    metadata_path = output_dir / "metadata.json"
    request_path = output_dir / "requests.jsonl"
    batch_path = output_dir / "batch.json"
    result_path = output_dir / "result.json"
    metadata = {
        "manifest_sha256": args.manifest_sha256,
        "contract_version": "v6",
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "model": MODEL,
        "reasoning_effort": "high",
        "prompt_cache_key": PROMPT_CACHE_KEY,
        "provider": "DeepInfra",
        "fallbacks": False,
        "max_tokens": args.max_tokens,
        "selected_ordinals": [ordinal for ordinal, _ in selected_cases],
    }
    if metadata_path.exists():
        stored_metadata = _read_object(metadata_path)
        legacy_metadata = dict(metadata)
        legacy_metadata.pop("max_tokens")
        legacy_metadata.pop("selected_ordinals")
        if stored_metadata != metadata and not (
            args.max_tokens == 2048
            and selected_cases == list(enumerate(cases))
            and stored_metadata == legacy_metadata
        ):
            raise ValueError("existing batch cache identity does not match this run")
    _write_private(metadata_path, metadata)
    if result_path.exists():
        result = _read_object(result_path)
        print(json.dumps({"cached": True, "summary": result.get("summary")}))
        return 0 if result.get("summary", {}).get("provider_errors") == 0 else 1
    if not request_path.exists():
        lines = _request_lines(selected_cases, contract, max_tokens=args.max_tokens)
        _write_private_text(request_path, "\n".join(lines) + "\n")
    if batch_path.exists():
        batch = _read_object(batch_path)
    else:
        uploaded = _upload_batch_file(
            request_path, token=token, timeout_seconds=args.timeout_seconds
        )
        _write_private(output_dir / "input-file.json", uploaded)
        batch = _create_batch(
            uploaded["id"],
            token=token,
            timeout_seconds=args.timeout_seconds,
            manifest_sha256=args.manifest_sha256,
            prompt_sha256=prompt_sha256,
            schema_sha256=schema_sha256,
        )
        _write_private(batch_path, batch)
        print(json.dumps({"submitted": True, "batch_id": batch.get("id")}))
    batch_id = batch.get("id")
    if type(batch_id) is not str or not batch_id:
        raise ValueError("batch cache does not contain a batch ID")
    if batch.get("status") not in {"completed", "failed", "expired", "cancelled"}:
        batch = _poll_batch(
            batch_id,
            token=token,
            poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
        )
        _write_private(batch_path, batch)
    if batch.get("status") != "completed":
        raise RuntimeError(f"DeepInfra batch ended with status {batch.get('status')}")
    output_file_id = batch.get("output_file_id")
    if type(output_file_id) is not str or not output_file_id:
        raise RuntimeError("completed batch did not provide an output file")
    lines = _download_jsonl(
        output_file_id, token=token, timeout_seconds=args.timeout_seconds
    )
    by_ordinal: dict[int, dict[str, Any]] = {}
    for line in lines:
        custom_id = line.get("custom_id")
        if type(custom_id) is not str or not custom_id.startswith("pointwise-"):
            raise RuntimeError("batch output contained an invalid custom_id")
        ordinal = int(custom_id.removeprefix("pointwise-"))
        if ordinal in by_ordinal or not 0 <= ordinal < len(cases):
            raise RuntimeError("batch output contained duplicate or invalid ordinal")
        by_ordinal[ordinal] = _record_from_line(line, ordinal=ordinal, case=cases[ordinal])
    expected_ordinals = {ordinal for ordinal, _ in selected_cases}
    if set(by_ordinal) != expected_ordinals:
        raise RuntimeError("batch output did not cover the selected cases")
    records = [by_ordinal[i] for i in sorted(expected_ordinals)]
    result = {
        **metadata,
        "batch_id": batch_id,
        "input_file_id": batch.get("input_file_id"),
        "output_file_id": output_file_id,
        "manifest_case_count": len(cases),
        "selected_ordinals": sorted(expected_ordinals),
        "records": records,
        "summary": _summary(records),
    }
    _write_private(result_path, result)
    print(json.dumps({"cached": False, "summary": result["summary"], "batch_id": batch_id}))
    return 0 if result["summary"]["provider_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
