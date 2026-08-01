#!/usr/bin/env python3
"""Fail-closed, resumable OpenAI Batch controller for pointwise production.

The provider boundary is deliberately injectable.  ``prepare_run`` and
``status_summary`` are fully offline; ``run_supervisor`` only mutates OpenAI
when given a transport (the CLI requires an explicit ``--provider`` opt-in).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any, Iterable, Protocol
import fcntl
from pydantic import BaseModel, ConfigDict, Field, field_validator


ROOT = Path(__file__).resolve().parents[1]
MODEL = "gpt-5.6-luna"
ENDPOINT = "/v1/chat/completions"
PRIMARY_MAX_COMPLETION_TOKENS = 8192
RETRY_MAX_COMPLETION_TOKENS = 16384
MAX_SHARD_REQUESTS = 500
MAX_SHARD_BYTES = 150 * 1024 * 1024
ACTIVE_BATCH_LIMIT = 3
AUTHORIZED_COST_CAP_USD = 50.00
MAX_IN_FLIGHT_WORST_CASE_USD = 10.00
PILOT_BATCH_COST_USD = 0.0987455
PILOT_CALLS = 500
PILOT_UNIT_COST_USD = PILOT_BATCH_COST_USD / PILOT_CALLS
MAX_RETRY_RECORDS = 1_000
MAX_RETRY_SHARD_RECORDS = 100
AMBIGUOUS_CREATE_WAIT_SECONDS = 26 * 60 * 60
PRICE_IDENTITY = {
    "source": "https://developers.openai.com/api/docs/pricing",
    "model": MODEL,
    "service": "batch",
    "context": "short",
    "input_per_million_usd": 0.10,
    "cached_input_per_million_usd": 0.01,
    "cache_write_per_million_usd": 0.125,
    "output_per_million_usd": 0.60,
}
PINNED_PROMPT_SHA256 = (
    "6adc8833a86fb05f1ceb174e4a0a02099bc22ab552839cb09b27b17c33a773a9"
)
PINNED_SCHEMA_SHA256 = (
    "3cf5401ba1af41362161d89cae1a0dbd38674a43f3855cb8c66e0ec4a85185e9"
)
DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["correct", "incorrect", "nonanswer", "not_established", "partial"],
        },
        "explanation": {"type": "string", "maxLength": 400},
    },
    "required": ["verdict", "explanation"],
}
_BARE_IDK = frozenset(
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
_TERMINAL = {"completed", "failed", "expired", "cancelled"}
_REMOTE_ACTIVE = {"validating", "in_progress", "finalizing", "cancelling"}


class _DecisionModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    verdict: str = Field(
        pattern="^(correct|incorrect|nonanswer|not_established|partial)$"
    )
    explanation: str = Field(min_length=1, max_length=400)

    @field_validator("explanation")
    @classmethod
    def reject_control_characters(cls, value: str) -> str:
        if any(
            ord(character) < 32 and character not in "\n\r\t" for character in value
        ):
            raise ValueError("explanation contains a control character")
        return value


class RunHalted(RuntimeError):
    """A persisted anomaly prevents any additional provider submission."""


class CapacityDeferred(RuntimeError):
    """The provider definitively rejected a create because queue capacity is full."""


class BatchTransport(Protocol):
    def find_batches(self, metadata: dict[str, str]) -> list[dict[str, object]]: ...
    def retrieve_batch(self, batch_id: str) -> dict[str, object]: ...
    def upload_file(self, path: Path) -> str: ...
    def create_batch(
        self, file_id: str, metadata: dict[str, str]
    ) -> dict[str, object]: ...
    def download_file(self, file_id: str) -> list[dict[str, object]]: ...


class OpenAIBatchTransport:
    """Thin SDK adapter; mutations have no implicit client-side retry."""

    def __init__(self, *, api_key: str, client: Any | None = None) -> None:
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:  # pragma: no cover - production dependency
                raise RuntimeError(
                    "OpenAI SDK is required only for provider supervision"
                ) from error
            client = OpenAI(api_key=api_key, max_retries=0)
        self._client = client

    @staticmethod
    def _plain(value: Any) -> dict[str, object]:
        data = value.model_dump() if hasattr(value, "model_dump") else dict(value)
        if type(data) is not dict:
            raise ValueError("OpenAI SDK returned a non-object")
        return data

    def find_batches(self, metadata: dict[str, str]) -> list[dict[str, object]]:
        matches: list[dict[str, object]] = []
        after: str | None = None
        while True:
            page = self._client.batches.list(limit=100, after=after)
            items = getattr(page, "data", None)
            if type(items) is not list:
                raise ValueError("OpenAI Batch listing is invalid")
            for item in items:
                row = self._plain(item)
                if row.get("metadata") == metadata:
                    matches.append(row)
            if not getattr(page, "has_more", False):
                return matches
            if not items:
                raise ValueError("OpenAI Batch pagination ended without a cursor")
            after = self._plain(items[-1]).get("id")
            if not isinstance(after, str) or not after:
                raise ValueError("OpenAI Batch pagination cursor is invalid")

    def retrieve_batch(self, batch_id: str) -> dict[str, object]:
        return self._plain(self._client.batches.retrieve(batch_id))

    def upload_file(self, path: Path) -> str:
        with Path(path).open("rb") as handle:
            file = self._client.files.create(file=handle, purpose="batch")
        file_id = self._plain(file).get("id")
        if not isinstance(file_id, str) or not file_id:
            raise ValueError("OpenAI file upload returned no ID")
        return file_id

    def create_batch(self, file_id: str, metadata: dict[str, str]) -> dict[str, object]:
        # SDK implicit POST retries are disabled in __init__; a lost accepted
        # response is resolved exclusively by paginated metadata discovery.
        try:
            created = self._client.batches.create(
                input_file_id=file_id,
                endpoint=ENDPOINT,
                completion_window="24h",
                metadata=metadata,
            )
        except Exception as error:
            if getattr(error, "status_code", None) == 429:
                raise CapacityDeferred(
                    "Batch queue capacity is currently full"
                ) from error
            raise
        return self._plain(created)

    def download_file(self, file_id: str) -> list[dict[str, object]]:
        content = self._client.files.content(file_id)
        text = getattr(content, "text", None)
        if not isinstance(text, str):
            raise ValueError("OpenAI output download is not UTF-8 text")
        rows: list[dict[str, object]] = []
        for line in text.splitlines():
            if not line:
                continue
            value = json.loads(line)
            if type(value) is not dict:
                raise ValueError("OpenAI output line is not an object")
            rows.append(value)
        return rows


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    )


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir() or stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError(f"{path} must be a private directory")
    path.chmod(0o700)


def _write_private(path: Path, value: object, *, text: bool = False) -> None:
    _private_dir(path.parent)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            if text:
                assert isinstance(value, str)
                handle.write(value)
            else:
                json.dump(
                    value,
                    handle,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _write_state(run_dir: Path, state: dict[str, Any]) -> None:
    # Source texts are reloaded from the sealed inventory; checkpoints never
    # rewrite all 182k source documents during a control-plane update.
    compact = {key: value for key, value in state.items() if key != "records"}
    _write_private(Path(run_dir) / "state.json", compact)


def _read_state(run_dir: Path) -> dict[str, Any]:
    return _read_json(Path(run_dir) / "state.json", "run state")


def _validate_state(state: dict[str, Any]) -> None:
    """Reject mutable/corrupt control-plane state before provider access."""
    if state.get("version") != 1 or type(state.get("identity")) is not dict:
        raise RunHalted("run state identity is invalid")
    identity = state["identity"]
    if (
        identity.get("model") != MODEL
        or identity.get("prompt_sha256") != PINNED_PROMPT_SHA256
        or identity.get("schema_sha256") != PINNED_SCHEMA_SHA256
        or identity.get("price_identity_sha256") != _sha256_json(PRICE_IDENTITY)
    ):
        raise RunHalted("run state contract identity is invalid")
    cap = state.get("spend", {}).get("cap_usd")
    if (
        type(cap) not in {int, float}
        or not math.isfinite(cap)
        or cap != identity.get("cost_cap_usd")
        or not 0 < cap <= AUTHORIZED_COST_CAP_USD
    ):
        raise RunHalted("run state cost authorization is invalid")
    spend = state["spend"]
    if spend.get("pricing") != PRICE_IDENTITY:
        raise RunHalted("run state pricing identity is invalid")
    failed_spend = spend.get("failed_attempt_usd")
    if (
        type(failed_spend) not in {int, float}
        or not math.isfinite(failed_spend)
        or failed_spend < 0
    ):
        raise RunHalted("run state failed-attempt spend is invalid")
    latest = spend.get("latest")
    if type(latest) is not dict:
        raise RunHalted("run state spend checkpoint is invalid")
    for key in (
        "completed_estimated_usd",
        "reserved_estimated_usd",
        "total_exposure_estimated_usd",
        "forecast_unit_usd",
    ):
        value = latest.get(key)
        if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
            raise RunHalted("run state monetary checkpoint is invalid")
    if type(latest.get("reserved_calls")) is not int or latest["reserved_calls"] < 0:
        raise RunHalted("run state reserved-call checkpoint is invalid")
    limits = state.get("limits")
    if (
        type(limits) is not dict
        or limits.get("requests") != identity.get("shard_request_limit")
        or limits.get("bytes") != identity.get("shard_byte_limit")
    ):
        raise RunHalted("run state limits differ from the sealed identity")
    if (
        not 1 <= limits["requests"] <= MAX_SHARD_REQUESTS
        or not 1 <= limits["bytes"] <= MAX_SHARD_BYTES
    ):
        raise RunHalted("run state limits exceed the production contract")
    contract = state.get("contract")
    if (
        type(contract) is not dict
        or type(contract.get("prompt")) is not str
        or contract.get("schema") != DECISION_SCHEMA
    ):
        raise RunHalted("run state prompt contract is invalid")
    if (
        _sha256_bytes(contract["prompt"].encode()) != PINNED_PROMPT_SHA256
        or _sha256_json(contract["schema"]) != PINNED_SCHEMA_SHA256
        or contract.get("prompt_sha256") != PINNED_PROMPT_SHA256
        or contract.get("schema_sha256") != PINNED_SCHEMA_SHA256
    ):
        raise RunHalted("run state prompt contract is invalid")


def _hydrate_records(run_dir: Path, state: dict[str, Any]) -> None:
    if "records" in state:
        return
    rows, manifest = _inventory(
        Path(state["inventory_path"]), Path(state["manifest_path"])
    )
    identity = state.get("identity", {})
    if identity.get("inventory_manifest_sha256") != manifest.get(
        "manifest_sha256"
    ) or identity.get("inventory_sha256") != manifest.get("inventory_sha256"):
        raise RunHalted("sealed inventory differs from the run identity")
    records = {row["record_id"]: row for row in rows}
    if len(records) != state["counts"]["records"]:
        raise ValueError("sealed inventory is incomplete")
    state["records"] = records


def _validate_hydrated_state(state: dict[str, Any]) -> None:
    _validate_state(state)
    records = state.get("records")
    results = state.get("results")
    shards = state.get("shards")
    if type(records) is not dict or len(records) != state.get("counts", {}).get(
        "records"
    ):
        raise RunHalted("run state inventory coverage is invalid")
    if type(results) is not dict or not set(results).issubset(records):
        raise RunHalted("run state result coverage is invalid")
    for record_id, result in results.items():
        source = records[record_id]
        if (
            type(result) is not dict
            or result.get("record_id") != record_id
            or result.get("source_ordinal") != source["source_ordinal"]
            or result.get("kind") not in {"cached", "deterministic", "provider"}
        ):
            raise RunHalted("run state contains a malformed result binding")
        _decision(
            {"verdict": result.get("verdict"), "explanation": result.get("explanation")}
        )
        _usage(result.get("usage"))
    if type(shards) is not list:
        raise RunHalted("run state shard collection is invalid")
    seen_records: set[str] = set()
    retry_records: set[str] = set()
    seen_names: set[str] = set()
    for shard in shards:
        if (
            type(shard) is not dict
            or type(shard.get("name")) is not str
            or shard["name"] in seen_names
        ):
            raise RunHalted("run state shard identity is invalid")
        seen_names.add(shard["name"])
        record_ids = shard.get("record_ids")
        if (
            type(record_ids) is not list
            or len(record_ids) != len(set(record_ids))
            or not set(record_ids).issubset(records)
        ):
            raise RunHalted("run state shard records are invalid")
        attempt = shard.get("attempt")
        expected_max = (
            RETRY_MAX_COMPLETION_TOKENS
            if attempt == 2
            else PRIMARY_MAX_COMPLETION_TOKENS
            if attempt == 1
            else None
        )
        expected_ids = {
            f"p{attempt}-{records[record_id]['source_ordinal']:06d}": record_id
            for record_id in record_ids
        }
        if (
            expected_max is None
            or shard.get("max_completion_tokens") != expected_max
            or shard.get("custom_ids") != expected_ids
        ):
            raise RunHalted("run state shard request contract is invalid")
        if (
            type(shard.get("input_sha256")) is not str
            or re.fullmatch(r"[0-9a-f]{64}", shard["input_sha256"]) is None
            or type(shard.get("input_bytes")) is not int
            or shard["input_bytes"] < 1
        ):
            raise RunHalted("run state shard byte identity is invalid")
        if shard.get("status") not in {
            "prepared",
            "upload-intent",
            "uploaded",
            "batch-intent",
            "validating",
            "failed",
            "in_progress",
            "finalizing",
            "completed",
            "expired",
            "cancelling",
            "cancelled",
            "reconciled",
            "anomalous",
        }:
            raise RunHalted("run state shard status is invalid")
        for field in ("input_file_id", "batch_id", "output_file_id", "error_file_id"):
            if shard.get(field) is not None and type(shard[field]) is not str:
                raise RunHalted("run state provider identity field is invalid")
        if shard.get("attempt") == 1:
            if seen_records.intersection(record_ids):
                raise RunHalted("primary shard records overlap")
            seen_records.update(record_ids)
        elif shard.get("attempt") == 2:
            if retry_records.intersection(record_ids):
                raise RunHalted("retry shard records overlap")
            retry_records.update(record_ids)
        else:
            raise RunHalted("run state shard attempt is invalid")
    base_results = {
        record_id
        for record_id, result in results.items()
        if result["kind"] in {"cached", "deterministic"}
    }
    if (
        seen_records.intersection(base_results)
        or seen_records | base_results != set(records)
        or not retry_records.issubset(seen_records)
    ):
        raise RunHalted("primary shard coverage differs from the sealed inventory")


def _validated_shard_bytes(
    run_dir: Path, state: dict[str, Any], shard: dict[str, Any]
) -> bytes:
    """Rebuild and bind every billable request immediately before upload."""
    attempt = shard.get("attempt")
    record_ids = shard.get("record_ids")
    if attempt not in {1, 2} or type(record_ids) is not list:
        raise RunHalted("shard attempt or records are invalid")
    expected_max = (
        RETRY_MAX_COMPLETION_TOKENS if attempt == 2 else PRIMARY_MAX_COMPLETION_TOKENS
    )
    if shard.get("max_completion_tokens") != expected_max:
        raise RunHalted("shard completion limit is invalid")
    if len(record_ids) > (
        MAX_RETRY_SHARD_RECORDS if attempt == 2 else state["limits"]["requests"]
    ):
        raise RunHalted("shard request count exceeds its safety limit")
    expected_ids = {
        f"p{attempt}-{state['records'][record_id]['source_ordinal']:06d}": record_id
        for record_id in record_ids
    }
    if shard.get("custom_ids") != expected_ids:
        raise RunHalted("shard custom-ID mapping is invalid")
    lines = [
        _request_line(
            state["records"][record_id],
            state["contract"],
            custom_id=f"p{attempt}-{state['records'][record_id]['source_ordinal']:06d}",
            max_tokens=expected_max,
        )
        for record_id in record_ids
    ]
    expected = ("\n".join(lines) + "\n").encode()
    path = Path(run_dir) / "shards" / f"{shard['name']}.jsonl"
    actual = path.read_bytes()
    if (
        actual != expected
        or shard.get("input_sha256") != _sha256_bytes(expected)
        or shard.get("input_bytes") != len(expected)
    ):
        raise RunHalted("shard bytes differ from the sealed request contract")
    descriptor = _read_json(
        Path(run_dir) / "shards" / f"{shard['name']}.json", "shard descriptor"
    )
    immutable = (
        "name",
        "record_ids",
        "custom_ids",
        "attempt",
        "max_completion_tokens",
        "input_sha256",
        "input_bytes",
    )
    if any(descriptor.get(key) != shard.get(key) for key in immutable):
        raise RunHalted("shard descriptor differs from durable state")
    return actual


def _inventory(
    inventory: Path, manifest: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    body = _read_json(manifest, "production manifest")
    sealed = dict(body)
    declared = sealed.pop("manifest_sha256", None)
    if declared != _sha256_json(sealed):
        raise ValueError("production manifest hash is invalid")
    if body.get("inventory") != Path(inventory).name:
        raise ValueError("production manifest inventory filename is invalid")
    raw = Path(inventory).read_bytes()
    if body.get("inventory_sha256") != _sha256_bytes(raw):
        raise ValueError("production inventory hash is invalid")
    if type(body.get("record_count")) is not int or body["record_count"] < 1:
        raise ValueError("production manifest record count is invalid")
    required = {
        "record_id",
        "source_ordinal",
        "question_id",
        "candidate_id",
        "question",
        "provisional_reference",
        "candidate_response",
    }
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line_no, line in enumerate(raw.decode("utf-8").splitlines()):
        if not line:
            raise ValueError(f"inventory row {line_no} is blank")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"inventory row {line_no} is invalid JSON") from error
        if type(row) is not dict or set(row) != required:
            raise ValueError(f"inventory row {line_no} has an invalid schema")
        if row["source_ordinal"] != line_no or type(row["source_ordinal"]) is not int:
            raise ValueError(f"inventory row {line_no} ordinal is not continuous")
        if any(
            type(row[key]) is not str or not row[key].strip()
            for key in required - {"source_ordinal"}
        ):
            raise ValueError(f"inventory row {line_no} contains an empty field")
        if row["record_id"] in seen:
            raise ValueError("production inventory has duplicate record IDs")
        seen.add(row["record_id"])
        records.append(row)
    if len(records) != body["record_count"]:
        raise ValueError("production inventory count does not match manifest")
    return records, body


def _contract(protocol: Path) -> dict[str, Any]:
    prompt_path = Path(protocol) / "pointwise-luna-v1.txt"
    schema_path = Path(protocol) / "decision-luna-v1.schema.json"
    # The validated Batch request did not include the customary terminal newline
    # in a text file.  Permit only that representation difference.
    prompt = prompt_path.read_text(encoding="utf-8").rstrip("\n")
    schema = _read_json(schema_path, "frozen Luna schema")
    prompt_sha256 = _sha256_bytes(prompt.encode())
    schema_sha256 = _sha256_json(schema)
    if schema != DECISION_SCHEMA:
        raise ValueError("frozen Luna schema is not the approved strict schema")
    if prompt_sha256 != PINNED_PROMPT_SHA256 or schema_sha256 != PINNED_SCHEMA_SHA256:
        raise ValueError("frozen Luna contract hash is not the approved pilot contract")
    return {
        "prompt": prompt,
        "schema": schema,
        "prompt_sha256": prompt_sha256,
        "schema_sha256": schema_sha256,
    }


def _bare_idk(text: str) -> bool:
    return (
        "".join(char for char in text.casefold() if char.isascii() and char.isalnum())
        in _BARE_IDK
    )


def _decision(value: object) -> dict[str, str]:
    try:
        parsed = _DecisionModel.model_validate(value)
    except Exception as error:
        raise ValueError("decision schema is invalid") from error
    verdict, explanation = parsed.verdict, parsed.explanation
    if not explanation.strip() or len(explanation) > 400 or "\n" in explanation:
        raise ValueError("decision explanation is invalid")
    return {"verdict": verdict, "explanation": explanation}


def _usage(value: object) -> dict[str, int]:
    if type(value) is not dict:
        raise ValueError("usage is missing")
    result: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        item = value.get(key)
        if type(item) is not int or item < 0:
            raise ValueError("usage is invalid")
        result[key] = item
    if result["total_tokens"] != result["prompt_tokens"] + result["completion_tokens"]:
        raise ValueError("usage total does not equal prompt plus completion tokens")
    details = value.get("prompt_tokens_details")
    cache_write_tokens = (
        details.get("cache_write_tokens", 0)
        if type(details) is dict
        else value.get("cache_write_tokens", 0)
    )
    cached_tokens = (
        details.get("cached_tokens", 0)
        if type(details) is dict
        else value.get("cached_tokens", 0)
    )
    if (
        type(cache_write_tokens) is not int
        or cache_write_tokens < 0
        or type(cached_tokens) is not int
        or cached_tokens < 0
        or cache_write_tokens + cached_tokens > result["prompt_tokens"]
    ):
        raise ValueError("prompt-cache usage is invalid")
    result["cache_write_tokens"] = cache_write_tokens
    result["cached_tokens"] = cached_tokens
    result["cache_details_known"] = (
        int(
            type(details) is dict
            and "cache_write_tokens" in details
            and "cached_tokens" in details
        )
        if "cache_details_known" not in value
        else int(value["cache_details_known"] == 1)
    )
    return result


def _usage_cost(usage: dict[str, int]) -> float:
    """Token-based Batch cost using mutually exclusive prompt-cache classes."""
    if usage.get("cache_details_known") != 1:
        prompt_cost = usage["prompt_tokens"] * max(
            PRICE_IDENTITY["input_per_million_usd"],
            PRICE_IDENTITY["cache_write_per_million_usd"],
        )
        return (
            prompt_cost
            + usage["completion_tokens"] * PRICE_IDENTITY["output_per_million_usd"]
        ) / 1_000_000
    cached = usage.get("cached_tokens", 0)
    cache_write = usage.get("cache_write_tokens", 0)
    ordinary = usage["prompt_tokens"] - cached - cache_write
    return (
        ordinary * PRICE_IDENTITY["input_per_million_usd"]
        + cached * PRICE_IDENTITY["cached_input_per_million_usd"]
        + cache_write * PRICE_IDENTITY["cache_write_per_million_usd"]
        + usage["completion_tokens"] * PRICE_IDENTITY["output_per_million_usd"]
    ) / 1_000_000


def _spend_exposure(state: dict[str, Any]) -> dict[str, float | int]:
    completed = (
        PILOT_BATCH_COST_USD
        + sum(
            _usage_cost(value["usage"])
            for value in state["results"].values()
            if value.get("kind") == "provider"
        )
        + float(state["spend"].get("failed_attempt_usd", 0.0))
    )
    completed_calls = PILOT_CALLS + sum(
        value.get("kind") == "provider" for value in state["results"].values()
    )
    observed_unit = completed / completed_calls if completed_calls else 0.0
    forecast_unit = max(PILOT_UNIT_COST_USD, observed_unit)
    reserved = 0.0
    reserved_calls = 0
    for shard in state["shards"]:
        if shard["status"] in {"reconciled", "anomalous"}:
            continue
        reserved += len(shard["record_ids"]) * forecast_unit
        reserved_calls += len(shard["record_ids"])
    return {
        "completed_estimated_usd": completed,
        "reserved_estimated_usd": reserved,
        "total_exposure_estimated_usd": completed + reserved,
        "forecast_unit_usd": forecast_unit,
        "reserved_calls": reserved_calls,
    }


def _enforce_cost_cap(state: dict[str, Any]) -> None:
    exposure = _spend_exposure(state)
    state["spend"]["latest"] = exposure
    if exposure["total_exposure_estimated_usd"] > state["spend"]["cap_usd"]:
        state["halted"] = True
        raise RunHalted("estimated spend exposure exceeds the authorized cap")


def _shard_worst_case_usd(shard: dict[str, Any]) -> float:
    """Conservative unobserved exposure: one input token per UTF-8 byte."""
    input_rate = (
        PRICE_IDENTITY["input_per_million_usd"]
        + PRICE_IDENTITY["cache_write_per_million_usd"]
    )
    input_cost = int(shard["input_bytes"]) * input_rate / 1_000_000
    output_cost = (
        len(shard["record_ids"])
        * int(shard["max_completion_tokens"])
        * PRICE_IDENTITY["output_per_million_usd"]
        / 1_000_000
    )
    return input_cost + output_cost


def _record_worst_case_usd(
    state: dict[str, Any], shard: dict[str, Any], record_id: str
) -> float:
    attempt = int(shard["attempt"])
    source = state["records"][record_id]
    custom_id = f"p{attempt}-{source['source_ordinal']:06d}"
    request_bytes = (
        len(
            _request_line(
                source,
                state["contract"],
                custom_id=custom_id,
                max_tokens=int(shard["max_completion_tokens"]),
            ).encode()
        )
        + 1
    )
    input_rate = (
        PRICE_IDENTITY["input_per_million_usd"]
        + PRICE_IDENTITY["cache_write_per_million_usd"]
    )
    return (
        request_bytes * input_rate
        + int(shard["max_completion_tokens"]) * PRICE_IDENTITY["output_per_million_usd"]
    ) / 1_000_000


def _enforce_in_flight_limit(state: dict[str, Any], candidate: dict[str, Any]) -> None:
    active = [
        shard
        for shard in state["shards"]
        if shard.get("batch_id") and shard.get("status") in _REMOTE_ACTIVE
    ]
    exposure = sum(
        _shard_worst_case_usd(shard) for shard in active
    ) + _shard_worst_case_usd(candidate)
    if exposure > MAX_IN_FLIGHT_WORST_CASE_USD:
        raise RunHalted(
            f"in-flight worst-case exposure ${exposure:.4f} exceeds the ${MAX_IN_FLIGHT_WORST_CASE_USD:.2f} safety ceiling"
        )
    completed = float(_spend_exposure(state)["completed_estimated_usd"])
    if completed + exposure > float(state["spend"]["cap_usd"]):
        raise RunHalted(
            f"completed spend plus worst-case in-flight exposure would exceed the ${state['spend']['cap_usd']:.2f} ceiling"
        )


def _pilot_records(
    pilot: Path,
    pilot_input: Path,
    inventory: dict[str, dict[str, Any]],
    contract: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    input_lines = [
        line
        for line in Path(pilot_input).read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(input_lines) != PILOT_CALLS:
        raise ValueError("pilot input must contain exactly 500 requests")
    try:
        requests = [json.loads(line) for line in input_lines]
    except json.JSONDecodeError as error:
        raise ValueError("pilot input contains invalid JSON") from error
    request_by_id: dict[str, dict[str, Any]] = {}
    for request in requests:
        custom_id = request.get("custom_id") if type(request) is dict else None
        if type(custom_id) is not str or custom_id in request_by_id:
            raise ValueError("pilot input custom IDs are invalid or duplicated")
        request_by_id[custom_id] = request
    payload = _read_json(pilot, "pilot result")
    if (
        payload.get("invalid") not in ([], None)
        or type(payload.get("records")) is not list
    ):
        raise ValueError("pilot result is not clean")
    if len(payload["records"]) != PILOT_CALLS:
        raise ValueError("pilot result must contain exactly 500 records")
    cached: dict[str, dict[str, Any]] = {}
    seen_result_ids: set[str] = set()
    for row in payload["records"]:
        if type(row) is not dict:
            raise ValueError("pilot record is invalid")
        record_id = row.get("record_id")
        source = inventory.get(record_id)
        if type(record_id) is not str or source is None or record_id in seen_result_ids:
            raise ValueError("pilot records are not unique source-bound records")
        seen_result_ids.add(record_id)
        if (
            row.get("source_ordinal") != source["source_ordinal"]
            or row.get("question_id") != source["question_id"]
            or row.get("candidate_id") != source["candidate_id"]
        ):
            raise ValueError("pilot record does not bind its source")
        pilot_index = row.get("pilot_index")
        if type(pilot_index) is not int or not 0 <= pilot_index < PILOT_CALLS:
            raise ValueError("pilot result index is invalid")
        custom_id = f"pilot-{pilot_index:06d}"
        request = request_by_id.pop(custom_id, None)
        if request is None:
            raise ValueError("pilot result has no uniquely matching request")
        expected = json.loads(
            _request_line(
                source,
                contract,
                custom_id=custom_id,
                max_tokens=PRIMARY_MAX_COMPLETION_TOKENS,
            )
        )
        cacheable = request == expected
        if row.get("model") != MODEL or row.get("finish_reason") != "stop":
            raise ValueError("pilot record is not a normal Luna completion")
        decision = _decision(
            {"verdict": row.get("verdict"), "explanation": row.get("explanation")}
        )
        usage = _usage(row.get("usage"))
        if cacheable:
            cached[record_id] = {
                "kind": "cached",
                "record_id": record_id,
                "source_ordinal": source["source_ordinal"],
                "provider_call": True,
                "model": MODEL,
                "finish_reason": "stop",
                "usage": usage,
                **decision,
            }
    if request_by_id:
        raise ValueError("pilot input contains requests without validated results")
    return cached


def _request_line(
    record: dict[str, Any], contract: dict[str, Any], *, custom_id: str, max_tokens: int
) -> str:
    value = {
        "custom_id": custom_id,
        "method": "POST",
        "url": ENDPOINT,
        "body": {
            "model": MODEL,
            "messages": [
                {"role": "system", "content": contract["prompt"]},
                {
                    "role": "user",
                    "content": "ORIGINAL QUESTION:\n"
                    + record["question"]
                    + "\n\nGEMINI EVALUATOR EVIDENCE — RETROSPECTIVE JUDGE ONLY:\n"
                    + record["provisional_reference"]
                    + "\n\nHISTORICAL RESPONSE:\n"
                    + record["candidate_response"],
                },
            ],
            "max_completion_tokens": max_tokens,
            "reasoning_effort": "medium",
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "pointwise_simple_verdict",
                    "strict": True,
                    "schema": DECISION_SCHEMA,
                },
            },
        },
    }
    encoded = json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
    )
    if json.loads(encoded) != value:
        raise ValueError("generated request line is invalid")
    return encoded


def _make_shard(
    state: dict[str, Any], record_ids: list[str], *, attempt: int
) -> dict[str, Any]:
    name = (
        f"retry-{attempt:02d}-{len(state['shards']):05d}"
        if attempt > 1
        else f"primary-{len(state['shards']):05d}"
    )
    custom_ids = {
        f"p{attempt}-{state['records'][record_id]['source_ordinal']:06d}": record_id
        for record_id in record_ids
    }
    if len(custom_ids) != len(record_ids):
        raise ValueError("shard ordinal mapping is not unique")
    return {
        "name": name,
        "record_ids": record_ids,
        "custom_ids": custom_ids,
        "attempt": attempt,
        "max_completion_tokens": RETRY_MAX_COMPLETION_TOKENS
        if attempt > 1
        else PRIMARY_MAX_COMPLETION_TOKENS,
        "status": "prepared",
        "input_file_id": None,
        "batch_id": None,
        "output_file_id": None,
        "error_file_id": None,
        "reconciled": {"valid": 0, "retry": 0},
    }


def _write_shard(run_dir: Path, state: dict[str, Any], shard: dict[str, Any]) -> None:
    records = state["records"]
    contract = state["contract"]
    lines = [
        _request_line(
            records[record_id],
            contract,
            custom_id=next(
                custom
                for custom, source in shard["custom_ids"].items()
                if source == record_id
            ),
            max_tokens=shard["max_completion_tokens"],
        )
        for record_id in shard["record_ids"]
    ]
    raw = ("\n".join(lines) + "\n").encode() if lines else b""
    if (
        not lines
        or len(lines) > state["limits"]["requests"]
        or len(raw) > state["limits"]["bytes"]
    ):
        raise ValueError("shard exceeds a configured limit")
    path = Path(run_dir) / "shards" / f"{shard['name']}.jsonl"
    _write_private(path, raw.decode(), text=True)
    shard["input_sha256"] = _sha256_bytes(raw)
    shard["input_bytes"] = len(raw)
    _write_private(Path(run_dir) / "shards" / f"{shard['name']}.json", shard)


def prepare_run(
    *,
    inventory: Path,
    manifest: Path,
    pilot: Path,
    pilot_input: Path,
    protocol: Path,
    run_dir: Path,
    shard_request_limit: int = MAX_SHARD_REQUESTS,
    shard_byte_limit: int = MAX_SHARD_BYTES,
    cost_cap_usd: float = AUTHORIZED_COST_CAP_USD,
) -> dict[str, Any]:
    """Validate sealed inputs and atomically create a resumable offline run."""
    if (
        not 1 <= shard_request_limit <= MAX_SHARD_REQUESTS
        or not 1 <= shard_byte_limit <= MAX_SHARD_BYTES
    ):
        raise ValueError("shard limits exceed the production contract")
    records, inventory_manifest = _inventory(Path(inventory), Path(manifest))
    contract = _contract(Path(protocol))
    by_id = {row["record_id"]: row for row in records}
    cached = _pilot_records(Path(pilot), Path(pilot_input), by_id, contract)
    if not 0 < cost_cap_usd <= AUTHORIZED_COST_CAP_USD:
        raise ValueError("cost cap exceeds the authorized production ceiling")
    identity = {
        "inventory_manifest_sha256": inventory_manifest["manifest_sha256"],
        "inventory_sha256": inventory_manifest["inventory_sha256"],
        "pilot_result_sha256": _sha256_bytes(Path(pilot).read_bytes()),
        "pilot_input_sha256": _sha256_bytes(Path(pilot_input).read_bytes()),
        "prompt_sha256": contract["prompt_sha256"],
        "schema_sha256": contract["schema_sha256"],
        "model": MODEL,
        "price_identity_sha256": _sha256_json(PRICE_IDENTITY),
        "cost_cap_usd": cost_cap_usd,
        "shard_request_limit": shard_request_limit,
        "shard_byte_limit": shard_byte_limit,
    }
    run_dir = Path(run_dir)
    if (run_dir / "state.json").exists():
        existing = _read_state(run_dir)
        if existing.get("identity") != identity:
            raise ValueError("existing run identity differs from this preparation")
        _validate_state(existing)
        return existing
    _private_dir(run_dir)
    results = dict(cached)
    pending: list[str] = []
    for row in records:
        record_id = row["record_id"]
        if record_id in results:
            continue
        if _bare_idk(row["candidate_response"]):
            results[record_id] = {
                "kind": "deterministic",
                "record_id": record_id,
                "source_ordinal": row["source_ordinal"],
                "provider_call": False,
                "verdict": "nonanswer",
                "explanation": "The historical response is an exact bare no-response.",
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        else:
            pending.append(record_id)
    state: dict[str, Any] = {
        "version": 1,
        "identity": identity,
        "inventory_path": str(Path(inventory).resolve()),
        "manifest_path": str(Path(manifest).resolve()),
        "contract": contract,
        "limits": {"requests": shard_request_limit, "bytes": shard_byte_limit},
        "records": {row["record_id"]: row for row in records},
        "results": results,
        "shards": [],
        "halted": False,
        "paused": False,
        "bootstrap_complete": False,
        "anomalies": [],
        "spend": {
            "cap_usd": cost_cap_usd,
            "pricing": PRICE_IDENTITY,
            "failed_attempt_usd": 0.0,
            "latest": {},
        },
        "counts": {
            "records": len(records),
            "cached": len(cached),
            "deterministic": sum(
                value["kind"] == "deterministic" for value in results.values()
            ),
            "pending": len(pending),
        },
    }
    batch: list[str] = []
    batch_bytes = 0
    for record_id in pending:
        line = (
            _request_line(
                state["records"][record_id],
                contract,
                custom_id=f"p1-{state['records'][record_id]['source_ordinal']:06d}",
                max_tokens=PRIMARY_MAX_COMPLETION_TOKENS,
            ).encode()
            + b"\n"
        )
        if len(line) > shard_byte_limit:
            raise ValueError("one request exceeds the shard byte limit")
        if batch and (
            len(batch) >= shard_request_limit
            or batch_bytes + len(line) > shard_byte_limit
        ):
            shard = _make_shard(state, batch, attempt=1)
            state["shards"].append(shard)
            _write_shard(run_dir, state, shard)
            batch, batch_bytes = [], 0
        batch.append(record_id)
        batch_bytes += len(line)
    if batch:
        shard = _make_shard(state, batch, attempt=1)
        state["shards"].append(shard)
        _write_shard(run_dir, state, shard)
    _enforce_cost_cap(state)
    _validate_state(state)
    _write_private(run_dir / "results.json", results)
    _write_state(run_dir, state)
    return state


def _response_line(
    custom_id: str, *, finish_reason: str = "stop", status_code: int = 200
) -> dict[str, object]:
    """Small public fixture helper used by the offline focused tests."""
    return {
        "custom_id": custom_id,
        "response": {
            "status_code": status_code,
            "body": {
                "model": MODEL,
                "choices": [
                    {
                        "finish_reason": finish_reason,
                        "message": {
                            "content": json.dumps(
                                {
                                    "verdict": "correct",
                                    "explanation": "The response is supported by the evidence.",
                                }
                            )
                        },
                    }
                ],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        },
    }


def _anomaly(
    state: dict[str, Any], message: str, *, shard: str, fatal: bool = True
) -> None:
    if fatal:
        state["halted"] = True
    else:
        state["paused"] = True
    state["anomalies"].append({"shard": shard, "message": message})


def reconcile_shard(
    run_dir: Path, shard_name: str, lines: Iterable[dict[str, object]]
) -> dict[str, Any]:
    """Reconcile a completed Batch by custom ID; invalid responses become one retry shard."""
    run_dir = Path(run_dir)
    state = _read_state(run_dir)
    _hydrate_records(run_dir, state)
    _validate_hydrated_state(state)
    shard = next((item for item in state["shards"] if item["name"] == shard_name), None)
    if shard is None:
        raise ValueError("unknown shard")
    _validated_shard_bytes(run_dir, state, shard)
    custom_to_record = shard["custom_ids"]
    expected = set(custom_to_record)
    found: set[str] = set()
    valid: dict[str, dict[str, Any]] = {}
    retry: list[str] = []
    raw_lines: list[str] = []
    for line in lines:
        raw_lines.append(json.dumps(line, ensure_ascii=False, separators=(",", ":")))
        custom_id = line.get("custom_id") if type(line) is dict else None
        if (
            type(custom_id) is not str
            or custom_id not in expected
            or custom_id in found
        ):
            _anomaly(
                state,
                "output has an extra, invalid, or duplicate custom ID",
                shard=shard_name,
            )
            continue
        found.add(custom_id)
        try:
            response = line.get("response")
            if type(response) is not dict or response.get("status_code") != 200:
                raise ValueError("non-200 response")
            body = response.get("body")
            if type(body) is not dict or body.get("model") != MODEL:
                raise ValueError("unexpected model")
            choices = body.get("choices")
            if (
                type(choices) is not list
                or len(choices) != 1
                or type(choices[0]) is not dict
                or choices[0].get("finish_reason") != "stop"
            ):
                raise ValueError("non-stop finish")
            message = choices[0].get("message")
            if type(message) is not dict or type(message.get("content")) is not str:
                raise ValueError("response content is invalid")
            decision = _decision(json.loads(message["content"]))
            record_id = custom_to_record[custom_id]
            usage = _usage(body.get("usage"))
            if usage["completion_tokens"] > shard["max_completion_tokens"]:
                raise ValueError("completion usage exceeds the request ceiling")
            valid[record_id] = {
                "kind": "provider",
                "record_id": record_id,
                "source_ordinal": state["records"][record_id]["source_ordinal"],
                "provider_call": True,
                "model": MODEL,
                "finish_reason": "stop",
                "usage": usage,
                **decision,
            }
        except (ValueError, TypeError, json.JSONDecodeError) as error:
            usage_recorded = False
            try:
                response = line.get("response")
                body = response.get("body") if type(response) is dict else None
                if type(body) is dict:
                    state["spend"]["failed_attempt_usd"] += _usage_cost(
                        _usage(body.get("usage"))
                    )
                    usage_recorded = True
            except ValueError:
                pass
            if not usage_recorded:
                state["spend"]["failed_attempt_usd"] += _record_worst_case_usd(
                    state, shard, custom_to_record[custom_id]
                )
            retry.append(custom_to_record[custom_id])
            _anomaly(state, str(error), shard=shard_name, fatal=False)
    missing = expected - found
    if missing:
        _anomaly(
            state,
            "output is missing requested custom IDs",
            shard=shard_name,
            fatal=False,
        )
        state["spend"]["failed_attempt_usd"] += sum(
            _record_worst_case_usd(state, shard, custom_to_record[custom])
            for custom in missing
        )
        retry.extend(sorted(custom_to_record[custom] for custom in missing))
    canonical_output = "\n".join(raw_lines) + ("\n" if raw_lines else "")
    _write_private(
        run_dir / "outputs" / f"{shard_name}.jsonl", canonical_output, text=True
    )
    shard["download_sha256"] = _sha256_bytes(canonical_output.encode())
    shard["download_records"] = len(raw_lines)
    if len(set(retry)) > MAX_RETRY_RECORDS:
        _anomaly(state, "retry population exceeds the safety ceiling", shard=shard_name)
    elif retry and shard["attempt"] >= 2:
        _anomaly(state, "a retry response failed a second time", shard=shard_name)
    elif retry:
        retry_ids = sorted(set(retry))
        for start in range(0, len(retry_ids), MAX_RETRY_SHARD_RECORDS):
            retry_shard = _make_shard(
                state, retry_ids[start : start + MAX_RETRY_SHARD_RECORDS], attempt=2
            )
            state["shards"].append(retry_shard)
            _write_shard(run_dir, state, retry_shard)
    state["results"].update(valid)
    shard["status"] = "reconciled" if not retry and not missing else "anomalous"
    shard["reconciled"] = {"valid": len(valid), "retry": len(set(retry))}
    state["counts"]["pending"] = state["counts"]["records"] - len(state["results"])
    if not retry:
        if shard["attempt"] == 1 or all(
            record_id in state["results"]
            for record_id in state["shards"][0]["record_ids"]
        ):
            state["bootstrap_complete"] = True
        if shard["attempt"] > 1:
            state["paused"] = any(
                item["attempt"] > 1 and item["status"] != "reconciled"
                for item in state["shards"]
            )
    try:
        _enforce_cost_cap(state)
    except RunHalted:
        _anomaly(
            state,
            "estimated spend exposure exceeds the authorized cap",
            shard=shard_name,
        )
    _write_private(run_dir / "results.json", state["results"])
    _write_state(run_dir, state)
    if state["halted"]:
        raise RunHalted("reconciliation anomaly; no new batches will be submitted")
    return state


def _metadata(state: dict[str, Any], shard: dict[str, Any]) -> dict[str, str]:
    return {
        "controller": "trec-mlm-openai-pointwise-v1",
        "run_identity": _sha256_json(state["identity"]),
        "shard": shard["name"],
        "input_sha256": shard["input_sha256"],
    }


def _validate_remote_batch(
    remote: dict[str, object],
    shard: dict[str, Any],
    metadata: dict[str, str],
    *,
    expected_id: str | None = None,
) -> None:
    if type(remote) is not dict or type(remote.get("id")) is not str:
        raise RunHalted("remote Batch identity is missing")
    if expected_id is not None and remote["id"] != expected_id:
        raise RunHalted("retrieved Batch ID differs from durable state")
    if (
        remote.get("input_file_id") != shard.get("input_file_id")
        or remote.get("endpoint") != ENDPOINT
        or remote.get("metadata") != metadata
    ):
        raise RunHalted("remote Batch is not bound to the sealed shard")
    if remote.get("status") not in {
        "validating",
        "failed",
        "in_progress",
        "finalizing",
        "completed",
        "expired",
        "cancelling",
        "cancelled",
    }:
        raise RunHalted("remote Batch status is invalid")


def _reconcile_terminal_remote(
    run_dir: Path,
    state: dict[str, Any],
    shard: dict[str, Any],
    remote: dict[str, object],
    transport: BatchTransport,
) -> None:
    request_counts = remote.get("request_counts")
    if type(request_counts) is not dict:
        raise RunHalted("terminal Batch request counts are missing")
    values = [request_counts.get(key) for key in ("total", "completed", "failed")]
    if (
        any(type(value) is not int or value < 0 for value in values)
        or values[0] != len(shard["record_ids"])
        or values[1] + values[2] > values[0]
    ):
        raise RunHalted("terminal Batch request counts are inconsistent")
    output = remote.get("output_file_id")
    errors = remote.get("error_file_id")
    if values[1] + values[2] > 0 and output is None and errors is None:
        raise RunHalted("terminal Batch supplied neither output nor error file")
    if (output is not None and type(output) is not str) or (
        errors is not None and type(errors) is not str
    ):
        raise RunHalted("terminal Batch file identity is invalid")
    shard["output_file_id"], shard["error_file_id"], shard["status"] = (
        output,
        errors,
        remote["status"],
    )
    _write_state(run_dir, state)
    lines: list[dict[str, object]] = []
    if isinstance(output, str):
        lines.extend(transport.download_file(output))
    if isinstance(errors, str):
        lines.extend(transport.download_file(errors))
    if len(lines) != values[1] + values[2]:
        raise RunHalted(
            "terminal Batch file cardinality differs from provider request counts"
        )
    reconcile_shard(run_dir, shard["name"], lines)


def run_supervisor(run_dir: Path, *, transport: BatchTransport) -> dict[str, Any]:
    """Perform one safe scheduling/reconciliation pass under the active-Batch limit."""
    run_dir = Path(run_dir)
    _private_dir(run_dir)
    lock_path = run_dir / "supervisor.lock"
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RunHalted("another supervisor owns this run") from error
        return _run_supervisor_locked(run_dir, transport=transport)
    finally:
        os.close(descriptor)


def _run_supervisor_locked(
    run_dir: Path, *, transport: BatchTransport
) -> dict[str, Any]:
    state = _read_state(run_dir)
    _hydrate_records(run_dir, state)
    _validate_hydrated_state(state)
    if state.get("halted"):
        raise RunHalted("run has a persisted anomaly")
    for shard in state["shards"]:
        if not shard.get("batch_id") or shard.get("status") in {
            "reconciled",
            "anomalous",
        }:
            continue
        _validated_shard_bytes(run_dir, state, shard)
        metadata = _metadata(state, shard)
        remote = transport.retrieve_batch(str(shard["batch_id"]))
        _validate_remote_batch(
            remote, shard, metadata, expected_id=str(shard["batch_id"])
        )
        shard["status"] = remote.get("status")
        shard["output_file_id"], shard["error_file_id"] = (
            remote.get("output_file_id"),
            remote.get("error_file_id"),
        )
        if remote.get("status") in _TERMINAL:
            _reconcile_terminal_remote(run_dir, state, shard, remote, transport)
            return status_summary(run_dir)
    active = sum(
        1
        for shard in state["shards"]
        if shard.get("batch_id") and shard.get("status") in _REMOTE_ACTIVE
    )
    try:
        _enforce_cost_cap(state)
    except RunHalted:
        _write_state(run_dir, state)
        raise
    for shard in state["shards"]:
        if active >= ACTIVE_BATCH_LIMIT or shard["status"] not in {
            "prepared",
            "upload-intent",
            "uploaded",
            "batch-intent",
        }:
            continue
        if state.get("paused") and shard["attempt"] == 1:
            continue
        metadata = _metadata(state, shard)
        discovered = transport.find_batches(metadata)
        if len(discovered) > 1:
            _anomaly(
                state,
                "multiple remote batches match durable intent",
                shard=shard["name"],
            )
            _write_state(run_dir, state)
            raise RunHalted("ambiguous remote batch discovery")
        if discovered:
            batch = discovered[0]
            _validate_remote_batch(batch, shard, metadata)
        else:
            if shard["status"] == "batch-intent":
                # The create may have been accepted before the process died.
                # Keep discovering for longer than the Batch completion window;
                # never create a duplicate while list visibility may lag.
                intent_at = shard.get("batch_intent_at")
                if (
                    type(intent_at) is not int
                    or time.time() - intent_at > AMBIGUOUS_CREATE_WAIT_SECONDS
                ):
                    _anomaly(
                        state,
                        "accepted-create window remained unresolved for 26 hours",
                        shard=shard["name"],
                    )
                    _write_state(run_dir, state)
                    raise RunHalted("ambiguous create outcome")
                shard["create_discovery_misses"] = (
                    int(shard.get("create_discovery_misses", 0)) + 1
                )
                _write_state(run_dir, state)
                return status_summary(run_dir)
            if shard["status"] != "uploaded":
                _validated_shard_bytes(run_dir, state, shard)
                shard["status"] = "upload-intent"
                _write_state(run_dir, state)
                file_id = transport.upload_file(
                    run_dir / "shards" / f"{shard['name']}.jsonl"
                )
                if not isinstance(file_id, str) or not file_id:
                    raise ValueError("upload did not return a file ID")
                shard["input_file_id"] = file_id
                shard["status"] = "uploaded"
                _write_state(run_dir, state)
        if not discovered:
            _validated_shard_bytes(run_dir, state, shard)
            _enforce_in_flight_limit(state, shard)
            shard["status"] = "batch-intent"
            shard["batch_intent_at"] = int(time.time())
            _write_state(run_dir, state)
            # Do not retry this mutation implicitly.  A thrown call leaves a
            # discoverable durable intent for the next invocation.
            try:
                batch = transport.create_batch(str(shard["input_file_id"]), metadata)
            except CapacityDeferred:
                shard["status"] = "uploaded"
                shard.pop("batch_intent_at", None)
                shard["capacity_deferred_at"] = int(time.time())
                _write_state(run_dir, state)
                break
            _validate_remote_batch(batch, shard, metadata)
        shard.pop("batch_intent_at", None)
        shard.pop("create_discovery_misses", None)
        shard["batch_id"], shard["status"] = (
            batch["id"],
            batch.get("status", "validating"),
        )
        _write_state(run_dir, state)
        if batch.get("status") in _TERMINAL:
            _reconcile_terminal_remote(run_dir, state, shard, batch, transport)
            return status_summary(run_dir)
        active += 1
        break
    return status_summary(run_dir)


def assemble_final(run_dir: Path) -> list[dict[str, Any]]:
    state = _read_state(Path(run_dir))
    _hydrate_records(Path(run_dir), state)
    _validate_hydrated_state(state)
    if state.get("halted") or len(state["results"]) != state["counts"]["records"]:
        raise RunHalted("final result coverage is incomplete or anomalous")
    output = [
        state["results"][record_id]
        for record_id, record in sorted(
            state["records"].items(), key=lambda item: item[1]["source_ordinal"]
        )
    ]
    _write_private(Path(run_dir) / "final-results.json", output)
    return output


def status_summary(run_dir: Path) -> dict[str, Any]:
    state = _read_state(Path(run_dir))
    _validate_state(state)
    active = sum(
        1
        for shard in state["shards"]
        if shard.get("batch_id") and shard.get("status") in _REMOTE_ACTIVE
    )
    return {
        "records": state["counts"]["records"],
        "cached": state["counts"]["cached"],
        "deterministic": state["counts"]["deterministic"],
        "resolved": len(state["results"]),
        "pending": state["counts"]["records"] - len(state["results"]),
        "shards": len(state["shards"]),
        "active": active,
        "halted": state["halted"],
        "paused": state["paused"],
        "bootstrap_complete": state["bootstrap_complete"],
        "anomalies": len(state["anomalies"]),
        "spend": state["spend"].get("latest", {}),
    }


def _is_transient_provider_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if type(status_code) is int and (
        status_code in {408, 409, 429} or status_code >= 500
    ):
        return True
    return type(error).__name__ in {
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "TimeoutError",
        "ConnectionError",
    }


def _run_supervisor_resilient(
    run_dir: Path, *, transport: BatchTransport, poll_seconds: float
) -> dict[str, Any]:
    """Retry only transport/provider failures; controller invariants remain fail-closed."""
    while True:
        try:
            return run_supervisor(run_dir, transport=transport)
        except Exception as error:
            if not _is_transient_provider_error(error):
                raise
            print(
                json.dumps(
                    {
                        "warning": "transient_provider_error",
                        "type": type(error).__name__,
                        "status_code": getattr(error, "status_code", None),
                        "retry_in_seconds": poll_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            time.sleep(poll_seconds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "status", "step", "supervise"))
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--inventory",
        type=Path,
        default=ROOT
        / "artifacts/pointwise_production/inputs-v1/production-records.jsonl",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT
        / "artifacts/pointwise_production/inputs-v1/production-manifest.json",
    )
    parser.add_argument(
        "--pilot",
        type=Path,
        default=ROOT
        / "artifacts/pointwise_judge_bakeoff/v11/openai-batch-production500/gpt-5.6-luna-medium/result.json",
    )
    parser.add_argument(
        "--pilot-input",
        type=Path,
        default=ROOT
        / "artifacts/pointwise_judge_bakeoff/v11/openai-batch-production500/gpt-5.6-luna-medium/input.jsonl",
    )
    parser.add_argument(
        "--protocol", type=Path, default=ROOT / "protocols/pointwise_judge_bakeoff"
    )
    parser.add_argument(
        "--execute-provider",
        action="store_true",
        help="required for step/supervise; permits OpenAI mutations",
    )
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare_run(
            inventory=args.inventory,
            manifest=args.manifest,
            pilot=args.pilot,
            pilot_input=args.pilot_input,
            protocol=args.protocol,
            run_dir=args.run_dir,
        )
        print(json.dumps(status_summary(args.run_dir), sort_keys=True))
        return 0
    if args.command == "status":
        print(json.dumps(status_summary(args.run_dir), sort_keys=True))
        return 0
    if not args.execute_provider:
        raise ValueError("step/supervise requires the explicit --execute-provider flag")
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for provider execution")
    if args.poll_seconds < 5:
        raise ValueError("poll interval must be at least 5 seconds")
    transport = OpenAIBatchTransport(api_key=api_key)
    while True:
        summary = (
            run_supervisor(args.run_dir, transport=transport)
            if args.command == "step"
            else _run_supervisor_resilient(
                args.run_dir, transport=transport, poll_seconds=args.poll_seconds
            )
        )
        print(json.dumps(summary, sort_keys=True), flush=True)
        if args.command == "step" or summary["halted"]:
            break
        if summary["resolved"] == summary["records"]:
            assemble_final(args.run_dir)
            break
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
