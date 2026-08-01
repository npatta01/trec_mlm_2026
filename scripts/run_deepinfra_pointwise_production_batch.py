#!/usr/bin/env python3
"""Run the v6 thinking judge over a private production inventory in batches."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.run_deepinfra_pointwise_batch import (
    CHAT_ENDPOINT,
    MODEL,
    PROMPT_CACHE_KEY,
    _build_body,
    _create_batch,
    _download_jsonl,
    _poll_batch,
    _record_from_line,
    _sha256_json,
    _summary,
    _upload_batch_file,
    _write_private,
    _write_private_text,
)
from scripts.run_local_pointwise_judge import _load_json_from_opened_file
from trec_mlm.pointwise_judge import PromptContract, load_prompt_contract


REVISED_PROMPT_CACHE_KEY = "trec-mlm-pointwise-v6-revised"


@dataclass(frozen=True, slots=True)
class ProductionCase:
    ordinal: int
    record_id: str
    source_ordinal: int
    question_id: str
    candidate_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    response_id: str
    selection_reason: str = "production_inventory"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--inventory-manifest", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-file",
        type=Path,
        help="optional revised prompt file; the v6 schema remains unchanged",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key-env", default="DEEPINFRA_API_TOKEN")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=16384)
    parser.add_argument("--poll-seconds", type=float, default=15.0)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    parser.add_argument(
        "--ordinals",
        type=int,
        nargs="+",
        help="optional inventory ordinals for a targeted validation run",
    )
    return parser


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _token(name: str) -> str:
    import os

    value = os.environ.get(name)
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} is required")
    return value.strip()


def _load_inventory(path: Path, manifest_path: Path) -> tuple[list[ProductionCase], dict[str, Any]]:
    manifest = _load_json_from_opened_file(
        manifest_path,
        "production inventory manifest",
        expected_sha256=_sha256_file(manifest_path),
        private=True,
    )
    expected_inventory_sha = manifest.get("inventory_sha256")
    if type(expected_inventory_sha) is not str:
        raise ValueError("production manifest inventory_sha256 is missing")
    actual_inventory_sha = _sha256_file(path)
    if actual_inventory_sha != expected_inventory_sha:
        raise ValueError("production inventory SHA-256 does not match its manifest")
    expected_count = manifest.get("record_count")
    if type(expected_count) is not int or expected_count <= 0:
        raise ValueError("production manifest record_count is invalid")
    cases: list[ProductionCase] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if type(record) is not dict:
                raise ValueError(f"inventory row {ordinal} is not an object")
            required = {
                "record_id",
                "source_ordinal",
                "question_id",
                "candidate_id",
                "question",
                "provisional_reference",
                "candidate_response",
            }
            if set(record) != required:
                raise ValueError(f"inventory row {ordinal} has unexpected keys")
            values = {key: record[key] for key in required}
            if any(type(values[key]) is not str or not values[key].strip() for key in required - {"source_ordinal"}):
                raise ValueError(f"inventory row {ordinal} contains an empty string")
            if type(values["source_ordinal"]) is not int:
                raise ValueError(f"inventory row {ordinal} source_ordinal is not an integer")
            record_id = values["record_id"]
            if record_id in seen_ids:
                raise ValueError(f"duplicate production record_id: {record_id}")
            seen_ids.add(record_id)
            cases.append(
                ProductionCase(
                    ordinal=ordinal,
                    record_id=record_id,
                    source_ordinal=values["source_ordinal"],
                    question_id=values["question_id"],
                    candidate_id=values["candidate_id"],
                    question=values["question"],
                    reference_answer=values["provisional_reference"],
                    candidate_answer=values["candidate_response"],
                    response_id=record_id,
                )
            )
    if len(cases) != expected_count:
        raise ValueError(f"inventory count {len(cases)} does not match manifest {expected_count}")
    return cases, manifest


def _request_lines(
    cases: list[ProductionCase],
    contract: Any,
    *,
    max_tokens: int,
    prompt_cache_key: str,
) -> list[str]:
    lines: list[str] = []
    for case in cases:
        body = _build_body(contract, case, max_tokens=max_tokens)
        body["prompt_cache_key"] = prompt_cache_key
        lines.append(
            json.dumps(
                {
                    "custom_id": f"production-{case.ordinal:06d}",
                    "method": "POST",
                    "url": CHAT_ENDPOINT,
                    "body": body,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
    return lines


def _run_chunk(
    cases: list[ProductionCase],
    *,
    chunk_index: int,
    output_dir: Path,
    contract: Any,
    inventory_manifest: dict[str, Any],
    prompt_sha256: str,
    schema_sha256: str,
    contract_version: str,
    prompt_cache_key: str,
    token: str,
    max_tokens: int,
    poll_seconds: float,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    metadata = {
        "contract_version": contract_version,
        "provider": "DeepInfra",
        "model": MODEL,
        "reasoning_effort": "high",
        "fallbacks": False,
        "prompt_cache_key": prompt_cache_key,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "inventory_sha256": inventory_manifest["inventory_sha256"],
        "inventory_manifest_sha256": inventory_manifest["manifest_sha256"],
        "chunk_index": chunk_index,
        "selected_ordinals": [case.ordinal for case in cases],
        "max_tokens": max_tokens,
    }
    metadata_path = output_dir / "metadata.json"
    request_path = output_dir / "requests.jsonl"
    batch_path = output_dir / "batch.json"
    result_path = output_dir / "result.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise ValueError(f"chunk {chunk_index} cache identity does not match")
    _write_private(metadata_path, metadata)
    if result_path.exists():
        result = json.loads(result_path.read_text())
        print(json.dumps({"chunk": chunk_index, "cached": True, "summary": result["summary"]}))
        return result["records"]
    if not request_path.exists():
        _write_private_text(
            request_path,
            "\n".join(
                _request_lines(
                    cases,
                    contract,
                    max_tokens=max_tokens,
                    prompt_cache_key=prompt_cache_key,
                )
            )
            + "\n",
        )
    if batch_path.exists():
        batch = json.loads(batch_path.read_text())
    else:
        uploaded = _upload_batch_file(request_path, token=token, timeout_seconds=timeout_seconds)
        _write_private(output_dir / "input-file.json", uploaded)
        batch = _create_batch(
            uploaded["id"],
            token=token,
            timeout_seconds=timeout_seconds,
            manifest_sha256=inventory_manifest["inventory_sha256"],
            prompt_sha256=prompt_sha256,
            schema_sha256=schema_sha256,
        )
        _write_private(batch_path, batch)
        print(json.dumps({"chunk": chunk_index, "submitted": True, "batch_id": batch.get("id")}))
    batch_id = batch.get("id")
    if type(batch_id) is not str or not batch_id:
        raise ValueError(f"chunk {chunk_index} has no batch ID")
    if batch.get("status") not in {"completed", "failed", "expired", "cancelled"}:
        batch = _poll_batch(
            batch_id,
            token=token,
            poll_seconds=poll_seconds,
            timeout_seconds=timeout_seconds,
        )
        _write_private(batch_path, batch)
    if batch.get("status") != "completed":
        raise RuntimeError(f"chunk {chunk_index} batch ended with {batch.get('status')}")
    output_file_id = batch.get("output_file_id")
    if type(output_file_id) is not str or not output_file_id:
        raise RuntimeError(f"chunk {chunk_index} has no output file")
    by_ordinal = {case.ordinal: case for case in cases}
    records: list[dict[str, Any]] = []
    for line in _download_jsonl(output_file_id, token=token, timeout_seconds=timeout_seconds):
        custom_id = line.get("custom_id")
        if type(custom_id) is not str or not custom_id.startswith("production-"):
            raise RuntimeError(f"chunk {chunk_index} output has invalid custom_id")
        ordinal = int(custom_id.removeprefix("production-"))
        case = by_ordinal.get(ordinal)
        if case is None:
            raise RuntimeError(f"chunk {chunk_index} output has unexpected ordinal {ordinal}")
        record = _record_from_line(line, ordinal=ordinal, case=case)
        record.update(
            {
                "record_id": case.record_id,
                "source_ordinal": case.source_ordinal,
                "candidate_id": case.candidate_id,
            }
        )
        records.append(record)
    if {record["ordinal"] for record in records} != set(by_ordinal):
        raise RuntimeError(f"chunk {chunk_index} output did not cover its inputs")
    records.sort(key=lambda record: int(record["ordinal"]))
    result = {
        **metadata,
        "batch_id": batch_id,
        "input_file_id": batch.get("input_file_id"),
        "output_file_id": output_file_id,
        "records": records,
        "summary": _summary(records),
    }
    _write_private(result_path, result)
    print(json.dumps({"chunk": chunk_index, "cached": False, "summary": result["summary"], "batch_id": batch_id}))
    return records


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.batch_size <= 0 or args.batch_size > 100:
        raise ValueError("batch-size must be between 1 and 100 for the DeepInfra batch contract")
    if args.max_tokens < 2048:
        raise ValueError("max-tokens must be at least 2048")
    if args.poll_seconds <= 0 or args.timeout_seconds <= 0:
        raise ValueError("poll and timeout values must be positive")
    cases, inventory_manifest = _load_inventory(args.inventory, args.inventory_manifest)
    if len(cases) != 500:
        raise ValueError("this run is intentionally limited to exactly 500 records")
    if args.ordinals is None:
        selected_cases = cases
    else:
        if not args.ordinals or len(set(args.ordinals)) != len(args.ordinals):
            raise ValueError("ordinals must be non-empty and unique")
        if any(ordinal < 0 or ordinal >= len(cases) for ordinal in args.ordinals):
            raise ValueError("ordinal is outside the inventory")
        selected_cases = [cases[ordinal] for ordinal in args.ordinals]
    token = _token(args.api_key_env)
    contract = load_prompt_contract(args.protocol_dir, version="v6")
    contract_version = "v6"
    prompt_cache_key = PROMPT_CACHE_KEY
    if args.prompt_file is not None:
        prompt_path = args.prompt_file
        prompt = prompt_path.read_text(encoding="utf-8")
        contract = PromptContract(
            prompt_path=prompt_path,
            schema_path=contract.schema_path,
            prompt=prompt,
            schema=contract.schema,
        )
        contract_version = "v6-revised"
        prompt_cache_key = REVISED_PROMPT_CACHE_KEY
    prompt_sha256 = hashlib.sha256(contract.prompt.encode("utf-8")).hexdigest()
    schema_sha256 = _sha256_json(contract.schema)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.chmod(0o700)
    all_records: list[dict[str, Any]] = []
    for chunk_index, start in enumerate(range(0, len(selected_cases), args.batch_size)):
        all_records.extend(
            _run_chunk(
                selected_cases[start : start + args.batch_size],
                chunk_index=chunk_index,
                output_dir=args.output_dir / f"chunk-{chunk_index:03d}",
                contract=contract,
                inventory_manifest=inventory_manifest,
                prompt_sha256=prompt_sha256,
                schema_sha256=schema_sha256,
                contract_version=contract_version,
                prompt_cache_key=prompt_cache_key,
                token=token,
                max_tokens=args.max_tokens,
                poll_seconds=args.poll_seconds,
                timeout_seconds=args.timeout_seconds,
            )
        )
    all_records.sort(key=lambda record: int(record["ordinal"]))
    expected_ordinals = {case.ordinal for case in selected_cases}
    if len(all_records) != len(selected_cases) or {record["ordinal"] for record in all_records} != expected_ordinals:
        raise RuntimeError("aggregate output does not cover exactly the selected inventory records")
    aggregate = {
        "contract_version": contract_version,
        "provider": "DeepInfra",
        "model": MODEL,
        "reasoning_effort": "high",
        "fallbacks": False,
        "prompt_cache_key": prompt_cache_key,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
        "inventory_sha256": inventory_manifest["inventory_sha256"],
        "inventory_manifest_sha256": inventory_manifest["manifest_sha256"],
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "selected_ordinals": sorted(expected_ordinals),
        "records": all_records,
        "summary": _summary(all_records),
    }
    _write_private(args.output_dir / "result.json", aggregate)
    print(json.dumps({"aggregate": str(args.output_dir / "result.json"), "summary": aggregate["summary"]}))
    return 0 if aggregate["summary"]["provider_errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
