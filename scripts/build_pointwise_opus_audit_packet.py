#!/usr/bin/env python3
"""Build a deterministic private semantic audit packet for a production run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from typing import Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from trec_mlm.pointwise_judge import DecisionV2


DEFAULT_AUDIT_SEED = "pointwise-production-audit-v1"
_SAFE_DECISION_FIELDS = (
    "label",
    "correctness",
    "completeness",
    "relevance",
    "error_flags",
    "reference_status",
    "confidence",
)
_SAFE_LABELS = {"good", "partial", "incorrect", "uncertain"}
_SAFE_CONFIDENCE = {"high", "medium", "low"}
_SAFE_FLAGS = {
    "entity_error",
    "number_error",
    "date_error",
    "negation_error",
    "unsupported_specificity",
    "refusal",
    "ambiguity",
    "reference_conflict",
}
_SAFE_REASONS = {
    "first_100",
    "repaired",
    "invalid",
    "low_confidence",
    "rare_label",
    "anomaly",
    "seeded_ordinary",
}
_TERMINAL_STATUSES = {"valid", "invalid"}
_SYNTHETIC_RATIONALE = (
    "Structured audit validation uses the sealed production decision contract "
    "and intentionally omits the provider rationale from the packet."
)
_SECRET_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"(?i)(?:api[_-]?key|authorization|bearer)\s*[:=]\s*\S+"),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\b(?:hf|xai|pplx|xoxb)[-_][A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bAIza[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
)


class _AuditCaseContentModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=200_000)
    reference: str = Field(min_length=1, max_length=200_000)
    candidate_response: str = Field(min_length=1, max_length=200_000)

    @classmethod
    def model_validate(
        cls, obj: object, *args: object, **kwargs: object
    ) -> "_AuditCaseContentModel":
        model = super().model_validate(obj, *args, **kwargs)
        for value in (model.question, model.reference, model.candidate_response):
            if any(pattern.search(value) for pattern in _SECRET_PATTERNS):
                raise ValueError("semantic case content contains a secret")
        return model


def build_audit_packet(
    run_dir: Path,
    *,
    milestone_block: int,
    seed: str = DEFAULT_AUDIT_SEED,
    ordinary_sample_size: int = 20,
    inventory: Path | None = None,
) -> dict[str, object]:
    """Return a private semantic audit packet with structured decisions and case text.

    Block zero is deliberately exhaustive.  Later milestones carry every
    exception stratum, every record from an anomalous block, and a stable seeded
    ordinary sample.  This gives reviewers reproducible evidence without
    The case text is authorized private review content; raw provider output,
    rationales, and credentials are never copied.
    """
    if type(milestone_block) is not int or milestone_block < 0:
        raise ValueError("milestone block must be a nonnegative integer")
    _validate_seed(seed)
    if type(ordinary_sample_size) is not int or ordinary_sample_size < 0:
        raise ValueError("ordinary sample size must be a nonnegative integer")
    directory = Path(run_dir)
    manifest = _read_private(directory / "run-manifest.json", "run manifest")
    count = manifest.get("record_count")
    if type(count) is not int or count < 1:
        raise ValueError("run manifest record count is invalid")
    results = _results(directory)
    limit = min(count, (milestone_block + 1) * 100)
    selected = [row for row in results if _ordinal(row) < limit]
    ordinals = [_ordinal(row) for row in selected]
    if len(ordinals) != len(set(ordinals)):
        raise ValueError("terminal results contain duplicate source ordinals")
    receipts = _receipts_through(directory, milestone_block)
    selected_pairs = {str(row["record_id"]): _ordinal(row) for row in selected}
    selected_ids = set(selected_pairs)
    for receipt in receipts:
        receipt_ids = _receipt_ids(receipt)
        if any(record_id not in selected_ids for record_id in receipt_ids):
            raise ValueError("a block receipt lacks a terminal result")

    if milestone_block == 0:
        _validate_first_block(selected, receipts)
    inventory_path = _inventory_path(directory, manifest, inventory)
    _validate_inventory_binding(inventory_path, manifest, receipts)
    content = _inventory_content(inventory_path, selected_pairs)
    repairs = _repair_ordinals(directory)
    if milestone_block == 0:
        rows = [
            _packet_row(row, repairs, ["first_100"], content[str(row["record_id"])])
            for row in sorted(selected, key=_ordinal)
        ]
        scope = "first_100"
    else:
        rows = _milestone_rows(
            selected,
            repairs,
            receipts,
            directory,
            seed,
            ordinary_sample_size,
            content,
        )
        scope = "milestone"
    body = {
        "version": 1,
        "scope": scope,
        "milestone_block": milestone_block,
        "planned_records": count,
        "covered_records": len(selected),
        "expected_records_through_milestone": limit,
        "seed": seed,
        "records": rows,
        "safety": {
            "contains_semantic_case_content": True,
            "contains_provider_raw_output": False,
            "contains_credentials": False,
        },
    }
    if not _serialized_strings_are_safe(body):
        raise ValueError("audit packet contains a secret-like string")
    return body | {"packet_sha256": _json_hash(body)}


def write_packet(destination: Path, packet: dict[str, object]) -> None:
    """Atomically publish a packet as a mode-0600 private artifact."""
    if not _valid_packet(packet):
        raise ValueError("audit packet is invalid")
    _write_private(Path(destination), packet)


def _valid_packet(packet: object) -> bool:
    required = {
        "version",
        "scope",
        "milestone_block",
        "planned_records",
        "covered_records",
        "expected_records_through_milestone",
        "seed",
        "records",
        "safety",
        "packet_sha256",
    }
    if type(packet) is not dict or set(packet) != required:
        return False
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    if packet.get("packet_sha256") != _json_hash(body):
        return False
    if (
        packet.get("version") != 1
        or packet.get("scope") not in {"first_100", "milestone"}
        or type(packet.get("milestone_block")) is not int
        or type(packet.get("planned_records")) is not int
        or type(packet.get("covered_records")) is not int
        or type(packet.get("expected_records_through_milestone")) is not int
        or type(packet.get("seed")) is not str
        or not packet["seed"]
        or type(packet.get("records")) is not list
        or packet.get("safety")
        != {
            "contains_semantic_case_content": True,
            "contains_provider_raw_output": False,
            "contains_credentials": False,
        }
    ):
        return False
    try:
        _validate_seed(packet["seed"])
    except ValueError:
        return False
    seen: set[int] = set()
    for row in packet["records"]:
        if type(row) is not dict or set(row) != {
            "record_id",
            "source_ordinal",
            "status",
            "repaired",
            "decision",
            "reasons",
            "question",
            "reference",
            "candidate_response",
        }:
            return False
        ordinal = row.get("source_ordinal")
        if (
            type(row.get("record_id")) is not str
            or not row["record_id"]
            or any(pattern.search(row["record_id"]) for pattern in _SECRET_PATTERNS)
            or type(row.get("status")) is not str
            or row["status"] not in _TERMINAL_STATUSES
            or type(ordinal) is not int
            or ordinal < 0
            or ordinal in seen
            or type(row.get("repaired")) is not bool
            or type(row.get("reasons")) is not list
            or any(
                type(reason) is not str or reason not in _SAFE_REASONS
                for reason in row["reasons"]
            )
        ):
            return False
        seen.add(ordinal)
        try:
            _validated_content(
                {
                    "question": row["question"],
                    "reference": row["reference"],
                    "candidate_response": row["candidate_response"],
                }
            )
        except ValueError:
            return False
        decision = row.get("decision")
        if decision is None:
            continue
        if type(decision) is not dict or set(decision) != set(_SAFE_DECISION_FIELDS):
            return False
        try:
            _validated_decision(decision)
        except ValueError:
            return False
    if not _serialized_strings_are_safe(body):
        return False
    if packet["scope"] == "first_100" and (
        len(seen) != 100
        or sorted(seen) != list(range(100))
        or packet["covered_records"] != 100
        or packet["expected_records_through_milestone"] != 100
    ):
        return False
    return True


def _milestone_rows(
    selected: list[dict[str, object]],
    repairs: set[int],
    receipts: list[dict[str, object]],
    directory: Path,
    seed: str,
    ordinary_sample_size: int,
    content: dict[str, dict[str, str]],
) -> list[dict[str, object]]:
    labels = Counter(
        str(row["decision"].get("label"))
        for row in selected
        if row.get("status") == "valid" and type(row.get("decision")) is dict
    )
    rare_limit = max(1, (len(selected) + 99) // 100)
    anomalous_ids = _anomalous_record_ids(directory, receipts)
    reason_by_id: dict[str, list[str]] = defaultdict(list)
    for row in selected:
        record_id = str(row["record_id"])
        ordinal = _ordinal(row)
        decision = row.get("decision")
        if ordinal in repairs:
            reason_by_id[record_id].append("repaired")
        if row.get("status") != "valid":
            reason_by_id[record_id].append("invalid")
        if type(decision) is dict:
            if decision.get("confidence") == "low":
                reason_by_id[record_id].append("low_confidence")
            if labels.get(str(decision.get("label")), 0) <= rare_limit:
                reason_by_id[record_id].append("rare_label")
        if record_id in anomalous_ids:
            reason_by_id[record_id].append("anomaly")
    ordinary = [row for row in selected if str(row["record_id"]) not in reason_by_id]
    ordinary.sort(key=lambda row: _seed_key(seed, str(row["record_id"])))
    for row in ordinary[:ordinary_sample_size]:
        reason_by_id[str(row["record_id"])].append("seeded_ordinary")
    return [
        _packet_row(
            row,
            repairs,
            reason_by_id[str(row["record_id"])],
            content[str(row["record_id"])],
        )
        for row in sorted(selected, key=_ordinal)
        if str(row["record_id"]) in reason_by_id
    ]


def _packet_row(
    result: dict[str, object],
    repairs: set[int],
    reasons: list[str],
    content: dict[str, str],
) -> dict[str, object]:
    record_id = result.get("record_id")
    if (
        type(record_id) is not str
        or not record_id
        or any(pattern.search(record_id) for pattern in _SECRET_PATTERNS)
    ):
        raise ValueError("terminal result identity is invalid or secret-like")
    decision = result.get("decision")
    safe_decision = (
        _validated_decision(decision, require_rationale=True)
        if decision is not None
        else None
    )
    safe_content = _validated_content(content)
    ordinal = _ordinal(result)
    status = result.get("status")
    if type(status) is not str or status not in _TERMINAL_STATUSES:
        raise ValueError("terminal status is invalid")
    return {
        "record_id": result["record_id"],
        "source_ordinal": ordinal,
        "status": status,
        "repaired": ordinal in repairs,
        "decision": safe_decision,
        "reasons": sorted(set(reasons)),
        **safe_content,
    }


def _validated_decision(
    value: object, *, require_rationale: bool = False
) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("decision is not a valid audit decision")
    expected_keys = set(_SAFE_DECISION_FIELDS) | {"rationale"}
    if set(value) not in (
        set(_SAFE_DECISION_FIELDS),
        expected_keys,
    ):
        raise ValueError("decision is not a valid audit decision")
    if require_rationale and set(value) != expected_keys:
        raise ValueError("decision is not a valid production decision")
    payload = {field: value[field] for field in _SAFE_DECISION_FIELDS}
    payload["rationale"] = value.get("rationale", _SYNTHETIC_RATIONALE)
    try:
        validated = DecisionV2.model_validate(payload).model_dump(mode="json")
        return {field: validated[field] for field in _SAFE_DECISION_FIELDS}
    except ValidationError as error:
        raise ValueError("decision is not a valid audit decision") from error


def _validated_content(value: object) -> dict[str, str]:
    try:
        return _AuditCaseContentModel.model_validate(value).model_dump(mode="json")
    except ValueError:
        raise
    except (TypeError, ValidationError) as error:
        raise ValueError("semantic case content is invalid") from error


def _inventory_path(
    directory: Path, manifest: dict[str, object], inventory: Path | None
) -> Path:
    if inventory is not None:
        return Path(inventory)
    value = manifest.get("inventory")
    if type(value) is not str or not value:
        raise ValueError("private inventory is required for semantic audit")
    candidate = Path(value)
    return candidate if candidate.is_absolute() else directory / candidate


def _validate_seed(seed: object) -> None:
    if (
        type(seed) is not str
        or not seed
        or len(seed) > 128
        or any(character.isspace() for character in seed)
        or any(pattern.search(seed) for pattern in _SECRET_PATTERNS)
    ):
        raise ValueError("audit seed is invalid or secret-like")


def _validate_inventory_binding(
    path: Path,
    manifest: dict[str, object],
    receipts: list[dict[str, object]],
) -> None:
    observed = _bytes_hash(path)
    committed = manifest.get("inventory_sha256")
    if type(committed) is not str or observed != committed:
        raise ValueError("inventory hash does not match the sealed run manifest")
    for receipt in receipts:
        receipt_hash = receipt.get("inventory_sha256")
        if type(receipt_hash) is not str or observed != receipt_hash:
            raise ValueError("inventory hash does not match the block receipt")


def _inventory_content(
    path: Path, expected_pairs: dict[str, int]
) -> dict[str, dict[str, str]]:
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise ValueError("private inventory must be a mode-0600 regular file")
    expected = {
        "record_id",
        "source_ordinal",
        "question_id",
        "candidate_id",
        "question",
        "provisional_reference",
        "candidate_response",
    }
    found: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError("private inventory is not valid JSONL") from error
            if (
                type(row) is not dict
                or set(row) != expected
                or row.get("source_ordinal") != ordinal
            ):
                raise ValueError("private inventory row identity is invalid")
            record_id = row["record_id"]
            if type(record_id) is not str or record_id not in expected_pairs:
                continue
            if row["source_ordinal"] != expected_pairs[record_id]:
                raise ValueError("inventory row ordinal/record ID pair is invalid")
            if any(
                type(row[key]) is not str or not row[key]
                for key in expected - {"source_ordinal"}
            ):
                raise ValueError("private inventory row contains an invalid string")
            found[record_id] = _validated_content(
                {
                    "question": row["question"],
                    "reference": row["provisional_reference"],
                    "candidate_response": row["candidate_response"],
                }
            )
    if found.keys() != expected_pairs.keys():
        raise ValueError("private inventory is missing an audited record")
    return found


def _validate_first_block(
    selected: list[dict[str, object]], receipts: list[dict[str, object]]
) -> None:
    if len(receipts) != 1 or len(selected) != 100:
        raise ValueError("complete first block receipt is required for first_100")
    ordered = sorted(selected, key=_ordinal)
    if [_ordinal(row) for row in ordered] != list(range(100)):
        raise ValueError("complete first block receipt is required for first_100")
    receipt = receipts[0]
    if receipt.get("block_index") != 0:
        raise ValueError("complete first block receipt is required for first_100")
    if _receipt_ids(receipt) != [str(row["record_id"]) for row in ordered]:
        raise ValueError("complete first block receipt is required for first_100")
    receipt_hash = receipt.get("receipt_sha256")
    body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
    if type(receipt_hash) is not str or receipt_hash != _json_hash(body):
        raise ValueError("complete first block receipt is invalid")
    for row in ordered:
        if row.get("status") != "valid":
            raise ValueError("complete first block terminal coverage is invalid")
        _validated_decision(row.get("decision"), require_rationale=True)


def _receipts_through(directory: Path, milestone_block: int) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    for index in range(milestone_block + 1):
        path = directory / "blocks" / f"{index:06d}-receipt.json"
        if not path.exists():
            continue
        receipt = _read_private(path, "block receipt")
        if receipt.get("block_index") != index:
            raise ValueError("block receipt identity is invalid")
        receipt_hash = receipt.get("receipt_sha256")
        body = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if type(receipt_hash) is not str or receipt_hash != _json_hash(body):
            raise ValueError("block receipt identity is invalid")
        receipts.append(receipt)
    return receipts


def _receipt_ids(receipt: dict[str, object]) -> list[str]:
    values = receipt.get("record_ids")
    if type(values) is not list or any(
        type(value) is not str or not value for value in values
    ):
        raise ValueError("block receipt record identities are invalid")
    if receipt.get("expected_count") != len(values):
        raise ValueError("block receipt count is invalid")
    return list(values)


def _results(directory: Path) -> list[dict[str, object]]:
    rows = [
        _read_private(path, "terminal result")
        for path in sorted((directory / "results").glob("*.json"))
    ]
    seen: set[str] = set()
    for row in rows:
        if (
            type(row.get("record_id")) is not str
            or not row["record_id"]
            or row["record_id"] in seen
            or any(pattern.search(row["record_id"]) for pattern in _SECRET_PATTERNS)
        ):
            raise ValueError("terminal result identity is invalid")
        if (
            type(row.get("status")) is not str
            or row["status"] not in _TERMINAL_STATUSES
        ):
            raise ValueError("terminal status is invalid")
        _ordinal(row)
        seen.add(row["record_id"])
    return rows


def _repair_ordinals(directory: Path) -> set[int]:
    repaired: set[int] = set()
    for path in (directory / "attempts").glob("*.json"):
        marker = _read_private(path, "attempt marker")
        ordinal = marker.get("source_ordinal")
        if type(ordinal) is not int or ordinal < 0:
            raise ValueError("attempt marker ordinal is invalid")
        if marker.get("repair") is True:
            repaired.add(ordinal)
    return repaired


def _anomalous_record_ids(
    directory: Path, receipts: list[dict[str, object]]
) -> set[str]:
    selected: set[str] = set()
    for receipt in receipts:
        index = receipt["block_index"]
        anomaly = directory / "anomalies" / f"{int(index):06d}.json"
        if not anomaly.exists():
            continue
        diagnostics = _read_private(anomaly, "anomaly diagnostics").get("diagnostics")
        if type(diagnostics) is list and diagnostics:
            selected.update(_receipt_ids(receipt))
    return selected


def _ordinal(row: dict[str, object]) -> int:
    ordinal = row.get("source_ordinal")
    if type(ordinal) is not int or ordinal < 0:
        raise ValueError("source ordinal is invalid")
    return ordinal


def _seed_key(seed: str, record_id: str) -> str:
    return hashlib.sha256(f"{seed}:{record_id}".encode("utf-8")).hexdigest()


def _read_private(path: Path, label: str) -> dict[str, object]:
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise ValueError(f"{label} must be a mode-0600 regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _write_private(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(
                value, handle, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _serialized_strings_are_safe(value: object) -> bool:
    if isinstance(value, str):
        return not any(pattern.search(value) for pattern in _SECRET_PATTERNS)
    if isinstance(value, dict):
        return all(_serialized_strings_are_safe(item) for item in value.values())
    if isinstance(value, list):
        return all(_serialized_strings_are_safe(item) for item in value)
    return True


def _bytes_hash(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except (FileNotFoundError, OSError) as error:
        raise ValueError("private inventory is unavailable") from error
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--milestone-block", type=int, required=True)
    parser.add_argument("--seed", default=DEFAULT_AUDIT_SEED)
    parser.add_argument("--ordinary-sample-size", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    write_packet(
        args.output,
        build_audit_packet(
            args.run_dir,
            milestone_block=args.milestone_block,
            seed=args.seed,
            ordinary_sample_size=args.ordinary_sample_size,
            inventory=args.inventory,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
