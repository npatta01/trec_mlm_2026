import hashlib
import json
import stat
from pathlib import Path

import pytest

from trec_mlm import pointwise_production as production
from trec_mlm.artifact_io import length_framed_sha256, manifest_aggregate_sha256
from trec_mlm.profile_inference import format_profile_document
from trec_mlm.profile_retrieval import canonical_profile_record_id
from trec_mlm.pointwise_production import ProductionRecord, build_production_inventory


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _write_refinement_records(path: Path, records: list[tuple[str, str, str]]) -> None:
    payload = []
    for question_id, candidate_id, response in records:
        question = f"Question for {question_id}?"
        payload.append(
            {
                "record_id": canonical_profile_record_id(
                    candidate_id, question_id, question, response
                ),
                "source_query_id": question_id,
                "candidate_id": candidate_id,
                "question": question,
                "response": response,
                "formatted_sha256": hashlib.sha256(
                    format_profile_document(question, response).encode("utf-8")
                ).hexdigest(),
                "raw_membership_sha256": length_framed_sha256((question, response)),
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
    entries = {path.name: {"byte_size": path.stat().st_size, "sha256": _sha256(path)}}
    (path.parent / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "profile_inference_manifest",
                "version": 2,
                "qrel_free": True,
                "files": entries,
                "aggregate_sha256": manifest_aggregate_sha256(entries),
            }
        ),
        encoding="utf-8",
    )


def _approve_fixture_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    discovery: Path,
    grounded: Path,
    refinement_records: Path,
    count: int,
) -> None:
    monkeypatch.setattr(production, "EXPECTED_PRODUCTION_RECORD_COUNT", count)
    monkeypatch.setattr(production, "FROZEN_DISCOVERY_SHA256", _sha256(discovery))
    monkeypatch.setattr(production, "FROZEN_GROUNDED_SHA256", _sha256(grounded))
    monkeypatch.setattr(
        production, "FROZEN_REFINEMENT_RECORDS_SHA256", _sha256(refinement_records)
    )
    monkeypatch.setattr(
        production,
        "FROZEN_REFINEMENT_MANIFEST_SHA256",
        _sha256(refinement_records.parent / "manifest.json"),
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
    assert first.source_ordinal == 0
    assert first.record_id == "profile-record-v2:abc"


def test_inventory_validates_frozen_membership_and_seals_private_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
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
        [("q1", "candidate-b", "Second."), ("q1", "candidate-a", "First.")],
    )
    _approve_fixture_sources(
        monkeypatch,
        discovery=discovery,
        grounded=grounded,
        refinement_records=refinement_records,
        count=2,
    )

    manifest = build_production_inventory(
        discovery=discovery,
        grounded=grounded,
        refinement_records=refinement_records,
        output_dir=output_dir,
    )

    records_path = output_dir / "production-records.jsonl"
    persisted = [json.loads(line) for line in records_path.read_text().splitlines()]
    assert manifest["record_count"] == 2
    assert [row["source_ordinal"] for row in persisted] == [0, 1]
    assert [row["candidate_id"] for row in persisted] == ["candidate-a", "candidate-b"]
    assert manifest["inventory_sha256"] == _sha256(records_path)
    assert set(manifest["sources"]) == {
        "discovery",
        "grounded",
        "refinement_records",
        "refinement_manifest",
    }
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(records_path.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "field", ["record_id", "raw_membership_sha256", "formatted_sha256"]
)
def test_inventory_rejects_noncanonical_or_unbound_refinement_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str
) -> None:
    discovery = tmp_path / "discovery.json"
    grounded = tmp_path / "grounded.json"
    refinement_records = tmp_path / "records.json"
    _write_discovery(discovery, {"q1": {"candidate-a": "Answer."}})
    _write_grounded(grounded, ["q1"])
    _write_refinement_records(refinement_records, [("q1", "candidate-a", "Answer.")])
    payload = json.loads(refinement_records.read_text())
    payload[0][field] = (
        "profile-record-v2:not-canonical" if field == "record_id" else "0" * 64
    )
    refinement_records.write_text(json.dumps(payload), encoding="utf-8")
    entries = {
        refinement_records.name: {
            "byte_size": refinement_records.stat().st_size,
            "sha256": _sha256(refinement_records),
        }
    }
    (refinement_records.parent / "manifest.json").write_text(
        json.dumps(
            {
                "artifact_type": "profile_inference_manifest",
                "version": 2,
                "qrel_free": True,
                "files": entries,
                "aggregate_sha256": manifest_aggregate_sha256(entries),
            }
        ),
        encoding="utf-8",
    )
    _approve_fixture_sources(
        monkeypatch,
        discovery=discovery,
        grounded=grounded,
        refinement_records=refinement_records,
        count=1,
    )

    with pytest.raises(ValueError, match="refinement record (ID|correspondence)"):
        build_production_inventory(
            discovery=discovery,
            grounded=grounded,
            refinement_records=refinement_records,
            output_dir=tmp_path / "production",
        )


@pytest.mark.parametrize("target", ["discovery", "refinement_records", "manifest"])
def test_inventory_rejects_substituted_approved_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    discovery = tmp_path / "discovery.json"
    grounded = tmp_path / "grounded.json"
    refinement_records = tmp_path / "records.json"
    _write_discovery(discovery, {"q1": {"candidate-a": "Answer."}})
    _write_grounded(grounded, ["q1"])
    _write_refinement_records(refinement_records, [("q1", "candidate-a", "Answer.")])
    _approve_fixture_sources(
        monkeypatch,
        discovery=discovery,
        grounded=grounded,
        refinement_records=refinement_records,
        count=1,
    )
    if target == "discovery":
        _write_discovery(discovery, {"q1": {"candidate-a": "Substituted."}})
    elif target == "refinement_records":
        refinement_records.write_text("[]", encoding="utf-8")
    else:
        (refinement_records.parent / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="approved .* hash"):
        build_production_inventory(
            discovery=discovery,
            grounded=grounded,
            refinement_records=refinement_records,
            output_dir=tmp_path / "production",
        )


def test_inventory_rejects_missing_reference_before_publishing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovery = tmp_path / "discovery.json"
    grounded = tmp_path / "grounded.json"
    refinement_records = tmp_path / "records.json"
    _write_discovery(discovery, {"q1": {"candidate-a": "Answer."}})
    _write_grounded(grounded, ["q2"])
    _write_refinement_records(refinement_records, [("q1", "candidate-a", "Answer.")])
    _approve_fixture_sources(
        monkeypatch,
        discovery=discovery,
        grounded=grounded,
        refinement_records=refinement_records,
        count=1,
    )

    with pytest.raises(ValueError, match="missing grounded reference for q1"):
        build_production_inventory(
            discovery=discovery,
            grounded=grounded,
            refinement_records=refinement_records,
            output_dir=tmp_path / "production",
        )

    assert not (tmp_path / "production").exists()
