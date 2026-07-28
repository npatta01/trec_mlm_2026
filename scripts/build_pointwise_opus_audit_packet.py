#!/usr/bin/env python3
"""Build a deterministic, text-free private audit packet for a production run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Sequence


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


def build_audit_packet(
    run_dir: Path,
    *,
    milestone_block: int,
    seed: str = DEFAULT_AUDIT_SEED,
    ordinary_sample_size: int = 20,
) -> dict[str, object]:
    """Return a review packet with structured decisions, never model/input text.

    Block zero is deliberately exhaustive.  Later milestones carry every
    exception stratum, every record from an anomalous block, and a stable seeded
    ordinary sample.  This gives reviewers reproducible evidence without
    placing questions, references, candidate responses, raw provider output,
    rationales, or credentials in the packet.
    """
    if type(milestone_block) is not int or milestone_block < 0:
        raise ValueError("milestone block must be a nonnegative integer")
    if type(seed) is not str or not seed:
        raise ValueError("audit seed must be a nonempty string")
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
    selected_ids = {str(row["record_id"]) for row in selected}
    for receipt in receipts:
        receipt_ids = _receipt_ids(receipt)
        if any(record_id not in selected_ids for record_id in receipt_ids):
            raise ValueError("a block receipt lacks a terminal result")

    repairs = _repair_ordinals(directory)
    if milestone_block == 0:
        rows = [
            _packet_row(row, repairs, ["first_100"])
            for row in sorted(selected, key=_ordinal)
        ]
        scope = "first_100"
    else:
        rows = _milestone_rows(
            selected, repairs, receipts, directory, seed, ordinary_sample_size
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
            "contains_raw_question_or_response_text": False,
            "contains_provider_raw_output": False,
            "contains_credentials": False,
        },
    }
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
            "contains_raw_question_or_response_text": False,
            "contains_provider_raw_output": False,
            "contains_credentials": False,
        }
    ):
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
        }:
            return False
        ordinal = row.get("source_ordinal")
        if (
            type(row.get("record_id")) is not str
            or not row["record_id"]
            or type(ordinal) is not int
            or ordinal < 0
            or ordinal in seen
            or type(row.get("repaired")) is not bool
            or type(row.get("reasons")) is not list
            or any(reason not in _SAFE_REASONS for reason in row["reasons"])
        ):
            return False
        seen.add(ordinal)
        decision = row.get("decision")
        if decision is None:
            continue
        if type(decision) is not dict or set(decision) != set(_SAFE_DECISION_FIELDS):
            return False
        if (
            decision.get("label") not in _SAFE_LABELS
            or decision.get("confidence") not in _SAFE_CONFIDENCE
            or type(decision.get("error_flags")) is not list
            or any(flag not in _SAFE_FLAGS for flag in decision["error_flags"])
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
        _packet_row(row, repairs, reason_by_id[str(row["record_id"])])
        for row in sorted(selected, key=_ordinal)
        if str(row["record_id"]) in reason_by_id
    ]


def _packet_row(
    result: dict[str, object], repairs: set[int], reasons: list[str]
) -> dict[str, object]:
    decision = result.get("decision")
    safe_decision = (
        {field: decision.get(field) for field in _SAFE_DECISION_FIELDS}
        if type(decision) is dict
        else None
    )
    ordinal = _ordinal(result)
    return {
        "record_id": result["record_id"],
        "source_ordinal": ordinal,
        "status": result.get("status"),
        "repaired": ordinal in repairs,
        "decision": safe_decision,
        "reasons": sorted(set(reasons)),
    }


def _receipts_through(directory: Path, milestone_block: int) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    for index in range(milestone_block + 1):
        path = directory / "blocks" / f"{index:06d}-receipt.json"
        if not path.exists():
            continue
        receipt = _read_private(path, "block receipt")
        if receipt.get("block_index") != index:
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
        ):
            raise ValueError("terminal result identity is invalid")
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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
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
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
