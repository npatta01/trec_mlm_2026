from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path

import pytest

from scripts import build_pointwise_opus_audit_packet as audit_module
from scripts.build_pointwise_opus_audit_packet import build_audit_packet, write_packet


def _private(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _result(
    ordinal: int,
    *,
    status: str = "valid",
    label: str = "good",
    confidence: str = "high",
    attempt: int | None = None,
) -> dict[str, object]:
    return {
        "version": 1,
        "record_id": f"record-{ordinal}",
        "source_ordinal": ordinal,
        "status": status,
        "attempt": ordinal + 1 if attempt is None else attempt,
        "decision": (
            None
            if status != "valid"
            else {
                "label": label,
                "correctness": 4,
                "completeness": 2,
                "relevance": 2,
                "error_flags": [],
                "reference_status": "sound",
                "confidence": confidence,
                "rationale": "private rationale and candidate response must not escape",
            }
        ),
        **({"reason": "private invalid response"} if status != "valid" else {}),
    }


def _run(tmp_path: Path, count: int = 120, *, first_block_valid: bool = False) -> Path:
    run = tmp_path / "run"
    inventory = run / "production-records.jsonl"
    run.mkdir(parents=True, exist_ok=True, mode=0o700)
    inventory.write_text(
        "".join(
            json.dumps(
                {
                    "record_id": f"record-{ordinal}",
                    "source_ordinal": ordinal,
                    "question_id": f"question-{ordinal}",
                    "candidate_id": f"candidate-{ordinal}",
                    "question": f"Question {ordinal}?",
                    "provisional_reference": f"Reference {ordinal}.",
                    "candidate_response": f"Candidate response {ordinal}.",
                }
            )
            + "\n"
            for ordinal in range(count)
        ),
        encoding="utf-8",
    )
    inventory.chmod(0o600)
    _private(
        run / "run-manifest.json",
        {"version": 1, "record_count": count, "inventory": str(inventory)},
    )
    for ordinal in range(count):
        _private(run / "results" / f"{ordinal:08d}.json", _result(ordinal))
        _private(
            run / "attempts" / f"{ordinal + 1:09d}.json",
            {
                "version": 1,
                "attempt": ordinal + 1,
                "source_ordinal": ordinal,
                "repair": False,
                "state": "reconciled",
            },
        )
        _private(
            run / "raw" / f"{ordinal:08d}-{ordinal + 1:09d}.json",
            {
                "response": {
                    "content": "RAW QUESTION: secret candidate response sk-test-never-emit",
                    "usage": {"prompt_tokens": 8, "completion_tokens": 4},
                    "cost_usd": 0.01,
                    "latency_seconds": 0.2,
                }
            },
        )
    first_receipt = {
        "version": 1,
        "block_index": 0,
        "record_ids": [f"record-{ordinal}" for ordinal in range(100)],
        "expected_count": 100,
    }
    _private(
        run / "blocks" / "000000-receipt.json",
        first_receipt | {"receipt_sha256": audit_module._json_hash(first_receipt)},
    )
    _private(
        run / "blocks" / "000001-receipt.json",
        {
            "version": 1,
            "block_index": 1,
            "record_ids": [f"record-{ordinal}" for ordinal in range(100, count)],
            "expected_count": count - 100,
        },
    )
    _private(
        run / "anomalies" / "000001.json",
        {
            "version": 1,
            "block_index": 1,
            "diagnostics": ["repeated provider output exceeds 95 percent of block"],
            "metrics": {},
        },
    )
    _private(run / "results" / "00000001.json", _result(1, attempt=1001))
    _private(
        run / "attempts" / "000001001.json",
        {
            "version": 1,
            "attempt": 1001,
            "source_ordinal": 1,
            "repair": True,
            "state": "reconciled",
        },
    )
    if not first_block_valid:
        _private(run / "results" / "00000002.json", _result(2, status="invalid"))
    _private(run / "results" / "00000003.json", _result(3, confidence="low"))
    _private(run / "results" / "00000004.json", _result(4, label="incorrect"))
    return run


def test_first_packet_covers_exactly_the_first_100_without_raw_text(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, first_block_valid=True)

    packet = build_audit_packet(run, milestone_block=0, seed="fixture-seed")

    assert packet["scope"] == "first_100"
    assert [row["source_ordinal"] for row in packet["records"]] == list(range(100))
    rendered = json.dumps(packet, sort_keys=True)
    assert packet["records"][0]["question"] == "Question 0?"
    assert packet["records"][0]["reference"] == "Reference 0."
    assert packet["records"][0]["candidate_response"] == "Candidate response 0."
    assert "RAW QUESTION" not in rendered
    assert "sk-test" not in rendered
    assert "rationale" not in rendered


def test_milestone_sampling_is_seeded_and_includes_every_required_stratum(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path)

    first = build_audit_packet(
        run, milestone_block=1, seed="same-seed", ordinary_sample_size=3
    )
    second = build_audit_packet(
        run, milestone_block=1, seed="same-seed", ordinary_sample_size=3
    )

    assert first == second
    records = {row["source_ordinal"]: row for row in first["records"]}
    assert {1, 2, 3, 4}.issubset(records)
    assert "repaired" in records[1]["reasons"]
    assert "invalid" in records[2]["reasons"]
    assert "low_confidence" in records[3]["reasons"]
    assert "rare_label" in records[4]["reasons"]
    assert any("anomaly" in row["reasons"] for row in records.values())
    assert any("seeded_ordinary" in row["reasons"] for row in records.values())


def test_packet_writer_is_private_mode_0600(tmp_path: Path) -> None:
    packet = build_audit_packet(
        _run(tmp_path, first_block_valid=True), milestone_block=0, seed="fixture-seed"
    )
    destination = tmp_path / "private" / "opus-audit.json"

    write_packet(destination, packet)

    assert stat.S_IMODE(destination.stat().st_mode) == 0o600


def test_packet_writer_refuses_an_unwhitelisted_rationale_field(tmp_path: Path) -> None:
    packet = build_audit_packet(
        _run(tmp_path, first_block_valid=True), milestone_block=0, seed="fixture-seed"
    )
    packet["records"][0]["decision"]["rationale"] = "private candidate answer"
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    packet["packet_sha256"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with pytest.raises(ValueError, match="audit packet is invalid"):
        write_packet(tmp_path / "private" / "bad-packet.json", packet)


def test_unpromoted_invalid_result_is_audited_without_a_block_receipt(
    tmp_path: Path,
) -> None:
    run = _run(tmp_path, first_block_valid=True)
    (run / "blocks" / "000000-receipt.json").unlink()

    with pytest.raises(ValueError, match="complete first block"):
        build_audit_packet(run, milestone_block=0, seed="fixture-seed")


@pytest.mark.parametrize(
    "field",
    [
        "label",
        "correctness",
        "completeness",
        "relevance",
        "error_flags",
        "reference_status",
        "confidence",
    ],
)
def test_packet_writer_rejects_secret_in_every_emitted_decision_field(
    tmp_path: Path, field: str
) -> None:
    packet = build_audit_packet(
        _run(tmp_path, first_block_valid=True), milestone_block=0, seed="fixture-seed"
    )
    packet["records"][0]["decision"][field] = "sk-do-not-emit-secret"
    body = {key: value for key, value in packet.items() if key != "packet_sha256"}
    packet["packet_sha256"] = audit_module._json_hash(body)

    with pytest.raises(ValueError, match="audit packet is invalid"):
        write_packet(tmp_path / "private" / "poisoned.json", packet)


def test_private_case_content_rejects_secret_value(tmp_path: Path) -> None:
    run = _run(tmp_path, first_block_valid=True)
    inventory = run / "production-records.jsonl"
    poisoned = inventory.read_text(encoding="utf-8").replace(
        "Question 0?", "Question 0? sk-do-not-emit-secret"
    )
    inventory.write_text(poisoned, encoding="utf-8")
    inventory.chmod(0o600)

    with pytest.raises(ValueError, match="secret"):
        build_audit_packet(run, milestone_block=0, seed="fixture-seed")


def test_packet_writer_fsyncs_parent_directory_after_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packet = build_audit_packet(
        _run(tmp_path, first_block_valid=True), milestone_block=0, seed="fixture-seed"
    )
    calls: list[int] = []
    original = audit_module.os.fsync
    monkeypatch.setattr(
        audit_module.os, "fsync", lambda fd: (calls.append(fd), original(fd))[1]
    )

    write_packet(tmp_path / "private" / "opus-audit.json", packet)

    assert len(calls) >= 2
