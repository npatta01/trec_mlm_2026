import hashlib
import json
import stat
from pathlib import Path

import pytest

from trec_mlm.pointwise_production_runner import (
    BLOCK_SIZE,
    BlockGateError,
    ProductionRunner,
    block_slices,
)


VALID = {
    "label": "good",
    "correctness": 4,
    "completeness": 2,
    "relevance": 2,
    "error_flags": [],
    "reference_status": "sound",
    "confidence": "high",
    "rationale": "The candidate directly and completely answers the question without a material error.",
}


def _sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _private(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _inputs(tmp_path: Path, count: int) -> tuple[Path, Path, Path]:
    inventory = tmp_path / "inputs" / "production-records.jsonl"
    inventory.parent.mkdir(mode=0o700)
    rows = [
        {
            "record_id": f"record-{index}",
            "source_ordinal": index,
            "question_id": f"q{index}",
            "candidate_id": "c",
            "question": "Question?",
            "provisional_reference": "Reference.",
            "candidate_response": "Answer.",
        }
        for index in range(count)
    ]
    inventory.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    inventory.chmod(0o600)
    manifest = inventory.parent / "production-manifest.json"
    body = {
        "version": 1,
        "record_count": count,
        "inventory": inventory.name,
        "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        "sources": {"fixture": {"sha256": "a" * 64}},
    }
    _private(manifest, body | {"manifest_sha256": _sha(body)})
    prompt = "Pointwise fixture prompt"
    schema = {"type": "object"}
    contract = {
        "version": "v2",
        "source_version": "v2",
        "prompt": prompt,
        "schema": schema,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "schema_sha256": _sha(schema),
    }
    receipt_body = {
        "version": 1,
        "packet_sha256": "b" * 64,
        "review_sha256": "c" * 64,
        "contract": contract,
        "recommendation_ledger": [],
    }
    receipt = inventory.parent / "adjudication.json"
    _private(receipt, receipt_body | {"adjudication_sha256": _sha(receipt_body)})
    return inventory, manifest, receipt


def _response(record: dict[str, object], repair: bool) -> dict[str, object]:
    decision = (
        VALID
        if int(record["source_ordinal"]) % 2 == 0
        else VALID
        | {
            "label": "partial",
            "correctness": 3,
            "confidence": "medium",
            "rationale": "The candidate is relevant but leaves a material part of the answer incomplete.",
        }
    )
    return {
        "content": json.dumps(decision),
        "finish_reason": "stop",
        "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        "cost_usd": 0.0,
        "provider": "DeepInfra",
        "model": "deepseek/deepseek-v4-flash",
    }


def test_block_slices_use_100_and_final_64() -> None:
    assert BLOCK_SIZE == 100
    assert block_slices(164) == [(0, 100), (100, 164)]


def test_runner_resumes_after_raw_checkpoint_crash_without_second_call(
    tmp_path: Path,
) -> None:
    inventory, manifest, receipt = _inputs(tmp_path, 2)
    calls: list[int] = []

    def crash_after_first(record: dict[str, object], repair: bool) -> dict[str, object]:
        calls.append(int(record["source_ordinal"]))
        if len(calls) == 1:
            return _response(record, repair) | {"crash_after_checkpoint": True}
        return _response(record, repair)

    runner = ProductionRunner(
        inventory=inventory,
        inventory_manifest=manifest,
        contract_receipt=receipt,
        run_dir=tmp_path / "run",
        call=crash_after_first,
        total_cap_usd=1.0,
    )
    with pytest.raises(RuntimeError, match="injected crash"):
        runner.run(max_blocks=1)
    assert calls == [0]
    resumed = ProductionRunner(
        inventory=inventory,
        inventory_manifest=manifest,
        contract_receipt=receipt,
        run_dir=tmp_path / "run",
        call=_response,
        total_cap_usd=1.0,
    )
    assert resumed.run(max_blocks=1)["completed_records"] == 2
    assert calls == [0]


def test_attempt_marker_and_spend_are_durable_before_send(tmp_path: Path) -> None:
    inventory, manifest, receipt = _inputs(tmp_path, 1)

    def inspect(record: dict[str, object], repair: bool) -> dict[str, object]:
        run_dir = tmp_path / "run"
        assert list((run_dir / "attempts").glob("*.json"))
        assert (run_dir / "spend-ledger.json").exists()
        return _response(record, repair)

    runner = ProductionRunner(
        inventory=inventory,
        inventory_manifest=manifest,
        contract_receipt=receipt,
        run_dir=tmp_path / "run",
        call=inspect,
        total_cap_usd=1.0,
    )
    runner.run(max_blocks=1)


def test_cost_cap_prevents_physical_send(tmp_path: Path) -> None:
    inventory, manifest, receipt = _inputs(tmp_path, 1)
    runner = ProductionRunner(
        inventory=inventory,
        inventory_manifest=manifest,
        contract_receipt=receipt,
        run_dir=tmp_path / "run",
        call=_response,
        total_cap_usd=0.000001,
    )
    with pytest.raises(RuntimeError, match="spend cap"):
        runner.run(max_blocks=1)


def test_invalid_response_gets_one_repair_then_gate_fails(tmp_path: Path) -> None:
    inventory, manifest, receipt = _inputs(tmp_path, 1)
    calls: list[bool] = []

    def invalid(record: dict[str, object], repair: bool) -> dict[str, object]:
        calls.append(repair)
        return _response(record, repair) | {"content": "{}"}

    runner = ProductionRunner(
        inventory=inventory,
        inventory_manifest=manifest,
        contract_receipt=receipt,
        run_dir=tmp_path / "run",
        call=invalid,
        total_cap_usd=1.0,
    )
    with pytest.raises(BlockGateError, match="terminal-invalid"):
        runner.run(max_blocks=1)
    assert calls == [False, True]


def test_length_finish_reason_is_terminal_invalid_and_never_promoted(
    tmp_path: Path,
) -> None:
    inventory, manifest, receipt = _inputs(tmp_path, 1)

    def truncated(record: dict[str, object], repair: bool) -> dict[str, object]:
        return _response(record, repair) | {"finish_reason": "length"}

    runner = ProductionRunner(
        inventory=inventory,
        inventory_manifest=manifest,
        contract_receipt=receipt,
        run_dir=tmp_path / "run",
        call=truncated,
        total_cap_usd=1.0,
    )
    with pytest.raises(BlockGateError, match="finish reason"):
        runner.run(max_blocks=1)
    terminal = json.loads((tmp_path / "run" / "results" / "00000000.json").read_text())
    assert terminal["status"] == "invalid"


def test_two_block_fixture_has_private_receipts_and_partial_final_block(
    tmp_path: Path,
) -> None:
    inventory, manifest, receipt = _inputs(tmp_path, 102)
    runner = ProductionRunner(
        inventory=inventory,
        inventory_manifest=manifest,
        contract_receipt=receipt,
        run_dir=tmp_path / "run",
        call=_response,
        total_cap_usd=1.0,
    )
    state = runner.run()
    assert state["completed_records"] == 102
    assert (
        json.loads((tmp_path / "run" / "blocks" / "000001-receipt.json").read_text())[
            "expected_count"
        ]
        == 2
    )
    assert (
        stat.S_IMODE(
            (tmp_path / "run" / "blocks" / "000001-receipt.json").stat().st_mode
        )
        == 0o600
    )
