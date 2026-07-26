#!/usr/bin/env python3
"""Run one local model over a private pointwise-judge smoke manifest."""

from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Any, Sequence

from trec_mlm.pointwise_judge import CandidateCase, load_prompt_contract
from trec_mlm.pointwise_runner import (
    call_openai_judge,
    run_cases,
    validate_local_model_registry,
    validate_loopback_endpoint,
    validate_runtime_receipt,
)
from trec_mlm.pointwise_sampling import DIAGNOSTIC_REASONS


_CASE_KEYS = {
    "question_id",
    "response_id",
    "question",
    "reference_answer",
    "candidate_answer",
    "strata",
    "selection_reason",
}
_MANIFEST_KEYS = {
    "version",
    "seed",
    "sources",
    "selection_reasons",
    "cases",
    "manifest_hash",
}
_MAX_INPUT_BYTES = 4 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--registry-sha256", required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--runtime-receipt-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def _load_json_from_opened_file(
    path: Path,
    description: str,
    *,
    expected_sha256: str,
    private: bool,
) -> dict[str, Any]:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{description} expected SHA-256 is invalid")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{description} could not be opened safely") from error
    try:
        os.set_inheritable(descriptor, False)
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode):
            raise ValueError(f"{description} must be a regular file")
        mode = stat.S_IMODE(opened_before.st_mode)
        if private and mode != 0o600:
            raise ValueError(f"{description} must have mode 0600")
        if not private and mode not in {0o600, 0o640, 0o644}:
            raise ValueError(f"{description} has an unsafe protocol-file mode")
        if opened_before.st_size > _MAX_INPUT_BYTES:
            raise ValueError(f"{description} is too large")
        chunks: list[bytes] = []
        total = 0
        while total <= _MAX_INPUT_BYTES:
            chunk = os.read(descriptor, min(65536, _MAX_INPUT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
        if total > _MAX_INPUT_BYTES:
            raise ValueError(f"{description} is too large")
        opened_after = os.fstat(descriptor)
        try:
            path_after = os.stat(path, follow_symlinks=False)
        except OSError as error:
            raise ValueError(f"{description} changed during read") from error
        identity_before = (
            opened_before.st_dev,
            opened_before.st_ino,
            opened_before.st_size,
            opened_before.st_mtime_ns,
            opened_before.st_ctime_ns,
        )
        identity_after = (
            opened_after.st_dev,
            opened_after.st_ino,
            opened_after.st_size,
            opened_after.st_mtime_ns,
            opened_after.st_ctime_ns,
        )
        path_identity = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(path_after.st_mode)
            or identity_before != identity_after
            or identity_after != path_identity
        ):
            raise ValueError(f"{description} changed during read")
        raw = b"".join(chunks)
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError(f"{description} does not match expected SHA-256")
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{description} is not valid JSON") from error
    finally:
        os.close(descriptor)
    if type(value) is not dict:
        raise ValueError(f"{description} must be a JSON object")
    return value


def _load_cases(manifest: dict[str, Any]) -> list[CandidateCase]:
    if set(manifest) != _MANIFEST_KEYS:
        raise ValueError("smoke manifest must contain exact Task 2 keys")
    body = dict(manifest)
    stored_hash = body.pop("manifest_hash")
    if not _is_sha256(stored_hash) or stored_hash != _canonical_hash(body):
        raise ValueError("smoke manifest hash does not match its body")
    if type(manifest["version"]) is not int or manifest["version"] != 1:
        raise ValueError("smoke manifest version must be 1")
    if type(manifest["seed"]) is not int:
        raise ValueError("smoke manifest seed must be an integer")
    _validate_sources(manifest["sources"])
    if manifest["selection_reasons"] != list(DIAGNOSTIC_REASONS):
        raise ValueError("smoke manifest selection reasons are not frozen")
    rows = manifest.get("cases")
    if type(rows) is not list or len(rows) != len(DIAGNOSTIC_REASONS):
        raise ValueError("smoke manifest must contain exactly ten cases")
    cases: list[CandidateCase] = []
    for row in rows:
        if type(row) is not dict or set(row) != _CASE_KEYS:
            raise ValueError("smoke manifest case must contain exact keys")
        values = dict(row)
        strata = values["strata"]
        if (
            type(strata) is not list
            or not strata
            or any(type(item) is not str or not item for item in strata)
        ):
            raise ValueError("smoke manifest case strata must be strings")
        values["strata"] = tuple(strata)
        if any(
            type(values[key]) is not str or not values[key]
            for key in _CASE_KEYS - {"strata"}
        ):
            raise ValueError("smoke manifest case strings must not be empty")
        if values["selection_reason"] not in strata:
            raise ValueError("smoke manifest selection reason must be a case stratum")
        cases.append(CandidateCase(**values))
    if len({case.response_id for case in cases}) != len(cases):
        raise ValueError("smoke manifest response IDs must be unique")
    if len({case.question_id for case in cases}) != len(cases):
        raise ValueError("smoke manifest question IDs must be unique")
    if [case.selection_reason for case in cases] != list(DIAGNOSTIC_REASONS):
        raise ValueError("smoke manifest cases do not match frozen reason order")
    return cases


def _validate_sources(value: object) -> None:
    if type(value) is not dict or set(value) != {"discovery", "grounded"}:
        raise ValueError("smoke manifest must contain exact source identities")
    for source in value.values():
        if type(source) is not dict or set(source) != {"sha256"}:
            raise ValueError("smoke manifest source must contain one hash")
        if not _is_sha256(source["sha256"]):
            raise ValueError("smoke manifest source hash is invalid")


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


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
    manifest = _load_json_from_opened_file(
        args.manifest,
        "smoke manifest",
        expected_sha256=args.manifest_sha256,
        private=True,
    )
    cases = _load_cases(manifest)
    registry = _load_json_from_opened_file(
        args.registry,
        "local model registry",
        expected_sha256=args.registry_sha256,
        private=False,
    )
    registry_entry = validate_local_model_registry(
        registry,
        model_alias=args.model_alias,
        model_id=args.model_id,
    )
    receipt = validate_runtime_receipt(
        _load_json_from_opened_file(
            args.runtime_receipt,
            "runtime receipt",
            expected_sha256=args.runtime_receipt_sha256,
            private=True,
        ),
        model_alias=args.model_alias,
        model_id=args.model_id,
        endpoint=endpoint,
        registry_sha256=args.registry_sha256,
        registry_entry=registry_entry,
    )
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
