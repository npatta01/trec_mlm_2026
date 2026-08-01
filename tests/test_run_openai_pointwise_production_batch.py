from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "run_openai_pointwise_production_batch.py"
)
_SPEC = importlib.util.spec_from_file_location("openai_production_batch", _SCRIPT)
assert _SPEC and _SPEC.loader
runner = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(runner)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _sources(tmp_path: Path, *, idk_index: int | None = 3) -> dict[str, Path]:
    protocol = tmp_path / "protocol"
    prompt = (
        (
            Path(__file__).parents[1]
            / "protocols/pointwise_judge_bakeoff/pointwise-luna-v1.txt"
        )
        .read_text(encoding="utf-8")
        .rstrip("\n")
    )
    _write(protocol / "decision-luna-v1.schema.json", runner.DECISION_SCHEMA)
    (protocol / "pointwise-luna-v1.txt").parent.mkdir(parents=True, exist_ok=True)
    (protocol / "pointwise-luna-v1.txt").write_text(prompt, encoding="utf-8")
    rows = []
    for ordinal in range(500):
        rows.append(
            {
                "record_id": f"record-{ordinal:04d}",
                "source_ordinal": ordinal,
                "question_id": f"question-{ordinal}",
                "candidate_id": "fixture",
                "question": "Who wrote Hamlet?",
                "provisional_reference": "William Shakespeare wrote Hamlet.",
                "candidate_response": "I don't know"
                if ordinal == idk_index
                else "Shakespeare wrote it.",
            }
        )
    inventory = tmp_path / "inputs" / "production-records.jsonl"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    manifest_body = {
        "version": 1,
        "record_count": len(rows),
        "inventory": inventory.name,
        "inventory_sha256": hashlib.sha256(inventory.read_bytes()).hexdigest(),
        "sources": {"fixture": {"sha256": "a" * 64}},
    }
    manifest = inventory.parent / "production-manifest.json"
    _write(manifest, manifest_body | {"manifest_sha256": _hash(manifest_body)})
    request_rows = []
    pilot_rows = []
    for index, row in enumerate(rows):
        custom_id = f"pilot-{index:06d}"
        contract = {"prompt": prompt, "schema": runner.DECISION_SCHEMA}
        request_rows.append(
            json.loads(
                runner._request_line(
                    row, contract, custom_id=custom_id, max_tokens=8192
                )
            )
        )
        pilot_rows.append(
            {
                "pilot_index": index,
                "record_id": row["record_id"],
                "source_ordinal": index,
                "question_id": row["question_id"],
                "candidate_id": row["candidate_id"],
                "model": runner.MODEL,
                "finish_reason": "stop",
                "verdict": "correct",
                "explanation": "The response is supported by the evidence.",
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 3,
                    "total_tokens": 5,
                },
            }
        )
    assert len(pilot_rows) == 500
    pilot_input = tmp_path / "pilot-input.jsonl"
    pilot_input.write_text(
        "".join(json.dumps(row) + "\n" for row in request_rows), encoding="utf-8"
    )
    pilot = tmp_path / "pilot-result.json"
    _write(pilot, {"records": pilot_rows, "invalid": []})
    return {
        "inventory": inventory,
        "manifest": manifest,
        "pilot": pilot,
        "pilot_input": pilot_input,
        "protocol": protocol,
    }


def test_prepare_partitions_cached_deterministic_and_pending_coverage(
    tmp_path: Path,
) -> None:
    paths = _sources(tmp_path)
    state = runner.prepare_run(
        **paths,
        run_dir=tmp_path / "run",
        shard_request_limit=2,
        shard_byte_limit=20_000,
    )

    assert state["counts"] == {
        "records": 500,
        "cached": 500,
        "deterministic": 0,
        "pending": 0,
    }
    assert len(state["shards"]) == 0
    assert (tmp_path / "run" / "state.json").exists()


def test_reconcile_uses_custom_ids_out_of_order_and_creates_one_retry_shard(
    tmp_path: Path,
) -> None:
    paths = _sources(tmp_path, idk_index=None)
    state = runner.prepare_run(
        **paths,
        run_dir=tmp_path / "run",
        shard_request_limit=500,
        shard_byte_limit=20_000_000,
    )
    # Make two cached records pending so reconciliation has an observable shard.
    state["results"] = {
        key: value
        for key, value in state["results"].items()
        if key not in {"record-0000", "record-0001"}
    }
    state["counts"] = {"records": 500, "cached": 498, "deterministic": 0, "pending": 2}
    state["shards"] = [
        runner._make_shard(state, ["record-0000", "record-0001"], attempt=1)
    ]
    runner._write_shard(tmp_path / "run", state, state["shards"][0])
    runner._write_state(tmp_path / "run", state)
    shard = state["shards"][0]
    good = runner._response_line("p1-000001")
    truncated = runner._response_line("p1-000000", finish_reason="length")

    runner.reconcile_shard(tmp_path / "run", shard["name"], [good, truncated])
    persisted = runner._read_state(tmp_path / "run")
    assert persisted["halted"] is False
    assert persisted["paused"] is True
    retry = [item for item in persisted["shards"] if item["attempt"] == 2]
    assert len(retry) == 1
    assert retry[0]["record_ids"] == ["record-0000"]


def test_clean_retry_unpauses_run_completes_bootstrap_and_preserves_first_success(
    tmp_path: Path,
) -> None:
    paths = _sources(tmp_path, idk_index=None)
    state = runner.prepare_run(
        **paths,
        run_dir=tmp_path / "run",
        shard_request_limit=500,
        shard_byte_limit=20_000_000,
    )
    state["results"] = {
        key: value
        for key, value in state["results"].items()
        if key not in {"record-0000", "record-0001"}
    }
    state["counts"] = {"records": 500, "cached": 498, "deterministic": 0, "pending": 2}
    primary = runner._make_shard(state, ["record-0000", "record-0001"], attempt=1)
    state["shards"] = [primary]
    runner._write_shard(tmp_path / "run", state, primary)
    runner._write_state(tmp_path / "run", state)

    runner.reconcile_shard(
        tmp_path / "run",
        primary["name"],
        [
            runner._response_line("p1-000001"),
            runner._response_line("p1-000000", finish_reason="length"),
        ],
    )
    retry = next(
        item
        for item in runner._read_state(tmp_path / "run")["shards"]
        if item["attempt"] == 2
    )
    runner.reconcile_shard(
        tmp_path / "run", retry["name"], [runner._response_line("p2-000000")]
    )

    persisted = runner._read_state(tmp_path / "run")
    assert persisted["halted"] is False
    assert persisted["paused"] is False
    assert persisted["bootstrap_complete"] is True
    assert len(persisted["results"]) == 500
    assert len(runner.assemble_final(tmp_path / "run")) == 500


def test_terminal_primary_with_invalid_output_does_not_block_retry_submission(
    tmp_path: Path,
) -> None:
    paths = _sources(tmp_path, idk_index=None)
    state = runner.prepare_run(
        **paths,
        run_dir=tmp_path / "run",
        shard_request_limit=500,
        shard_byte_limit=20_000_000,
    )
    state["results"] = {
        key: value
        for key, value in state["results"].items()
        if key not in {"record-0000", "record-0001"}
    }
    state["counts"] = {"records": 500, "cached": 498, "deterministic": 0, "pending": 2}
    primary = runner._make_shard(state, ["record-0000", "record-0001"], attempt=1)
    primary["batch_id"] = "batch-primary"
    state["shards"] = [primary]
    runner._write_shard(tmp_path / "run", state, primary)
    runner._write_state(tmp_path / "run", state)
    runner.reconcile_shard(
        tmp_path / "run",
        primary["name"],
        [
            runner._response_line("p1-000001"),
            runner._response_line("p1-000000", finish_reason="length"),
        ],
    )

    transport = _CapacityTransport()
    runner.run_supervisor(tmp_path / "run", transport=transport)
    result = runner.run_supervisor(tmp_path / "run", transport=transport)

    assert result["active"] == 1
    retry = next(
        item
        for item in runner._read_state(tmp_path / "run")["shards"]
        if item["attempt"] == 2
    )
    assert retry["batch_id"] == "batch-created"


def test_resume_discovers_created_batch_before_another_create(tmp_path: Path) -> None:
    paths = _sources(tmp_path, idk_index=None)
    state = runner.prepare_run(
        **paths,
        run_dir=tmp_path / "run",
        shard_request_limit=500,
        shard_byte_limit=20_000_000,
    )
    state["results"] = {
        key: value for key, value in state["results"].items() if key == "record-0002"
    }
    state["counts"] = {"records": 500, "cached": 1, "deterministic": 0, "pending": 499}
    shard = runner._make_shard(
        state, [f"record-{i:04d}" for i in range(500) if i != 2], attempt=1
    )
    shard["input_file_id"] = "file-existing"
    shard["status"] = "batch-intent"
    state["shards"] = [shard]
    runner._write_shard(tmp_path / "run", state, shard)
    runner._write_state(tmp_path / "run", state)

    transport = _DiscoveringTransport()
    result = runner.run_supervisor(tmp_path / "run", transport=transport)

    assert result["active"] == 1
    assert transport.created == 0
    assert (
        runner._read_state(tmp_path / "run")["shards"][0]["batch_id"]
        == "batch-existing"
    )


def test_definitive_capacity_rejection_reuses_upload_without_duplicate_batch(
    tmp_path: Path,
) -> None:
    paths = _sources(tmp_path, idk_index=None)
    state = runner.prepare_run(
        **paths,
        run_dir=tmp_path / "run",
        shard_request_limit=500,
        shard_byte_limit=20_000_000,
    )
    state["results"] = {
        key: value for key, value in state["results"].items() if key == "record-0499"
    }
    state["counts"] = {"records": 500, "cached": 1, "deterministic": 0, "pending": 499}
    shard = runner._make_shard(
        state, [f"record-{i:04d}" for i in range(499)], attempt=1
    )
    state["shards"] = [shard]
    runner._write_shard(tmp_path / "run", state, shard)
    runner._write_state(tmp_path / "run", state)
    transport = _CapacityTransport()

    first = runner.run_supervisor(tmp_path / "run", transport=transport)
    assert first["active"] == 0
    assert runner._read_state(tmp_path / "run")["shards"][0]["status"] == "uploaded"
    second = runner.run_supervisor(tmp_path / "run", transport=transport)

    assert second["active"] == 1
    assert transport.uploaded == 1
    assert transport.created == 2


def test_resilient_supervisor_retries_502_but_not_controller_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class Provider502(RuntimeError):
        status_code = 502

    calls = iter([Provider502("bad gateway"), {"active": 2, "halted": False}])
    sleeps: list[float] = []
    monkeypatch.setattr(
        runner,
        "run_supervisor",
        lambda *args, **kwargs: next(calls) if False else _raise_or_return(next(calls)),
    )
    monkeypatch.setattr(runner.time, "sleep", sleeps.append)

    result = runner._run_supervisor_resilient(
        tmp_path, transport=object(), poll_seconds=5
    )

    assert result == {"active": 2, "halted": False}
    assert sleeps == [5]
    assert runner._is_transient_provider_error(Provider502("bad gateway")) is True
    assert runner._is_transient_provider_error(runner.RunHalted("bad state")) is False


def test_decision_rejects_control_characters_in_explanation() -> None:
    with pytest.raises(ValueError, match="control character"):
        runner._DecisionModel.model_validate(
            {"verdict": "correct", "explanation": "supported\u0000text"}
        )


def _raise_or_return(value: object) -> object:
    if isinstance(value, BaseException):
        raise value
    return value


class _DiscoveringTransport:
    def __init__(self) -> None:
        self.created = 0

    def find_batches(self, metadata: dict[str, str]) -> list[dict[str, object]]:
        return [
            {
                "id": "batch-existing",
                "status": "in_progress",
                "metadata": metadata,
                "input_file_id": "file-existing",
                "endpoint": runner.ENDPOINT,
            }
        ]

    def retrieve_batch(self, batch_id: str) -> dict[str, object]:
        assert batch_id == "batch-existing"
        return {
            "id": batch_id,
            "status": "in_progress",
            "metadata": {},
            "input_file_id": "file-existing",
            "endpoint": runner.ENDPOINT,
        }

    def upload_file(self, path: Path) -> str:
        raise AssertionError("discovery must happen before upload")

    def create_batch(self, file_id: str, metadata: dict[str, str]) -> dict[str, object]:
        self.created += 1
        raise AssertionError("discovery must happen before create")

    def download_file(self, file_id: str) -> list[dict[str, object]]:
        raise AssertionError("not terminal")


class _CapacityTransport:
    def __init__(self) -> None:
        self.uploaded = 0
        self.created = 0

    def find_batches(self, metadata: dict[str, str]) -> list[dict[str, object]]:
        return []

    def retrieve_batch(self, batch_id: str) -> dict[str, object]:
        return {"id": batch_id, "status": "in_progress"}

    def upload_file(self, path: Path) -> str:
        self.uploaded += 1
        return "file-once"

    def create_batch(self, file_id: str, metadata: dict[str, str]) -> dict[str, object]:
        assert file_id == "file-once"
        self.created += 1
        if self.created == 1:
            raise runner.CapacityDeferred("fixture capacity")
        return {
            "id": "batch-created",
            "status": "validating",
            "metadata": metadata,
            "input_file_id": file_id,
            "endpoint": runner.ENDPOINT,
        }

    def download_file(self, file_id: str) -> list[dict[str, object]]:
        raise AssertionError("not terminal")
