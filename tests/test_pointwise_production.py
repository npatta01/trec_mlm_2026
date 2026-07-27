import hashlib
import json
import stat
from pathlib import Path

import pytest

from trec_mlm.pointwise_production import (
    EXPECTED_PRODUCTION_RECORD_COUNT,
    ProductionRecord,
    build_production_inventory,
)


def _write_discovery(path: Path, responses: dict[str, dict[str, str]]) -> None:
    path.write_text(
        json.dumps(
            {
                question_id: {
                    "query": f"Question for {question_id}?",
                    "responses": candidates,
                }
                for question_id, candidates in responses.items()
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _write_grounded(path: Path, question_ids: list[str]) -> None:
    path.write_text(
        json.dumps(
            {
                "records": {
                    question_id: {
                        "question_id": question_id,
                        "question": f"Question for {question_id}?",
                        "answer": f"Reference for {question_id}.",
                        "finish_reason": "stop",
                    }
                    for question_id in question_ids
                }
            }
        ),
        encoding="utf-8",
    )


def _write_refinement_records(
    path: Path, records: list[tuple[str, str, str, str]]
) -> None:
    path.write_text(
        json.dumps(
            [
                {
                    "record_id": record_id,
                    "source_query_id": question_id,
                    "candidate_id": candidate_id,
                    "question": f"Question for {question_id}?",
                    "response": response,
                    "formatted_sha256": "0" * 64,
                    "raw_membership_sha256": "1" * 64,
                }
                for record_id, question_id, candidate_id, response in records
            ]
        ),
        encoding="utf-8",
    )


def test_production_record_ids_are_frozen_and_ordinals_are_source_order() -> None:
    first = ProductionRecord.from_source(
        record_id="profile-record-v2:abc",
        source_ordinal=0,
        question_id="q1",
        candidate_id="candidate-a",
        question="Question?",
        provisional_reference="Reference.",
        candidate_response="Answer.",
    )
    same = ProductionRecord.from_source(
        record_id="profile-record-v2:abc",
        source_ordinal=0,
        question_id="q1",
        candidate_id="candidate-a",
        question="Question?",
        provisional_reference="Reference.",
        candidate_response="Answer.",
    )
    changed = ProductionRecord.from_source(
        record_id="profile-record-v2:def",
        source_ordinal=1,
        question_id="q1",
        candidate_id="candidate-a",
        question="Question?",
        provisional_reference="Reference.",
        candidate_response="Changed answer.",
    )

    assert first.source_ordinal == 0
    assert first.record_id == same.record_id == "profile-record-v2:abc"
    assert first.record_id != changed.record_id


def test_inventory_streams_frozen_membership_in_discovery_order_and_seals_private_snapshot(
    tmp_path: Path,
) -> None:
    discovery = tmp_path / "discovery.json"
    grounded = tmp_path / "grounded.json"
    refinement_records = tmp_path / "records.json"
    output_dir = tmp_path / "production"
    _write_discovery(
        discovery,
        {"q1": {"candidate-a": "First.", "candidate-b": "Second."}},
    )
    _write_grounded(grounded, ["q1"])
    _write_refinement_records(
        refinement_records,
        [
            ("profile-record-v2:second", "q1", "candidate-b", "Second."),
            ("profile-record-v2:first", "q1", "candidate-a", "First."),
        ],
    )

    manifest = build_production_inventory(
        discovery=discovery,
        grounded=grounded,
        refinement_records=refinement_records,
        output_dir=output_dir,
        expected_count=2,
    )

    records_path = output_dir / "production-records.jsonl"
    persisted = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert manifest["record_count"] == 2
    assert [row["source_ordinal"] for row in persisted] == [0, 1]
    assert [row["candidate_id"] for row in persisted] == ["candidate-a", "candidate-b"]
    assert [row["record_id"] for row in persisted] == [
        "profile-record-v2:first",
        "profile-record-v2:second",
    ]
    assert (
        manifest["inventory_sha256"]
        == hashlib.sha256(records_path.read_bytes()).hexdigest()
    )
    assert set(manifest["sources"]) == {
        "discovery",
        "grounded",
        "refinement_records",
    }
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(records_path.stat().st_mode) == 0o600
    assert (
        stat.S_IMODE((output_dir / "production-manifest.json").stat().st_mode) == 0o600
    )


def test_inventory_rejects_missing_reference_before_publishing(tmp_path: Path) -> None:
    discovery = tmp_path / "discovery.json"
    grounded = tmp_path / "grounded.json"
    refinement_records = tmp_path / "records.json"
    output_dir = tmp_path / "production"
    _write_discovery(discovery, {"q1": {"candidate-a": "Answer."}})
    _write_grounded(grounded, ["q2"])
    _write_refinement_records(
        refinement_records,
        [("profile-record-v2:one", "q1", "candidate-a", "Answer.")],
    )

    with pytest.raises(ValueError, match="missing grounded reference for q1"):
        build_production_inventory(
            discovery=discovery,
            grounded=grounded,
            refinement_records=refinement_records,
            output_dir=output_dir,
            expected_count=1,
        )

    assert not output_dir.exists()


def test_inventory_rejects_duplicate_grounded_reference_before_publishing(
    tmp_path: Path,
) -> None:
    discovery = tmp_path / "discovery.json"
    grounded = tmp_path / "grounded.json"
    refinement_records = tmp_path / "records.json"
    output_dir = tmp_path / "production"
    _write_discovery(discovery, {"q1": {"candidate-a": "Answer."}})
    duplicate = json.dumps(
        {
            "question_id": "q1",
            "question": "Question for q1?",
            "answer": "Reference.",
            "finish_reason": "stop",
        }
    )
    grounded.write_text(
        f'{{"records": {{"q1": {duplicate}, "q1": {duplicate}}}}}',
        encoding="utf-8",
    )
    _write_refinement_records(
        refinement_records,
        [("profile-record-v2:one", "q1", "candidate-a", "Answer.")],
    )

    with pytest.raises(ValueError, match="duplicate JSON key: q1"):
        build_production_inventory(
            discovery=discovery,
            grounded=grounded,
            refinement_records=refinement_records,
            output_dir=output_dir,
            expected_count=1,
        )

    assert not output_dir.exists()


def test_inventory_requires_exact_frozen_production_coverage(tmp_path: Path) -> None:
    discovery = tmp_path / "discovery.json"
    grounded = tmp_path / "grounded.json"
    refinement_records = tmp_path / "records.json"
    _write_discovery(discovery, {"q1": {"candidate-a": "Answer."}})
    _write_grounded(grounded, ["q1"])
    _write_refinement_records(
        refinement_records,
        [("profile-record-v2:one", "q1", "candidate-a", "Answer.")],
    )

    with pytest.raises(ValueError, match="expected 182764 production records, found 1"):
        build_production_inventory(
            discovery=discovery,
            grounded=grounded,
            refinement_records=refinement_records,
            output_dir=tmp_path / "production",
            expected_count=EXPECTED_PRODUCTION_RECORD_COUNT,
        )
