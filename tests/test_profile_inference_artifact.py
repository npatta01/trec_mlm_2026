from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import numpy as np
import pytest

from trec_mlm.profile_inference_artifact import (
    ARTIFACT_VERSION,
    ProfileInferenceArtifactConfig,
    ProfileInferenceArtifactResult,
    build_profile_inference_execution_receipt,
    compare_profile_inference_repeats,
    load_profile_inference_artifacts,
    ordered_source_questions,
    repair_legacy_profile_inference_artifact,
    run_profile_inference_artifact,
    validate_profile_smoke_receipt,
    write_profile_inference_artifacts,
)
from trec_mlm.profile_retrieval import canonical_profile_record_id
from trec_mlm import profile_inference_artifact


def _smoke_receipt() -> dict[str, object]:
    return {
        "status": "passed",
        "qrel_free": True,
        "rankings_or_qrels_used": False,
        "maximum_absolute_embedding_difference": 0.0,
        "tolerance": 1e-5,
        "device_architecture": "gfx1151",
        "container_image": "vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28",
        "model_id": "intfloat/e5-mistral-7b-instruct",
        "model_revision": "d62cc4417be0e3289c03dd459d1ef2d137e2554e",
        "batch_size": 8,
        "dtype": "bfloat16",
        "persisted_dtype": "float32",
        "smoke_inventory": ["short", "median", "over_512"],
        "first_smoke_receipt_sha256": "a" * 64,
        "second_smoke_receipt_sha256": "b" * 64,
    }


def _result() -> ProfileInferenceArtifactResult:
    return ProfileInferenceArtifactResult(
        config=ProfileInferenceArtifactConfig(
            query_path="queries.tsv",
            discovery_path="discovery.json",
            candidate_path="candidates.json",
            e5_artifact_path="e5",
        ),
        query_ids=("q1",),
        candidate_ids=("A", "B"),
        label_question_ids=("s1",),
        records=[
            {
                "record_id": canonical_profile_record_id("A", "s1", "Q?", "A."),
                "candidate_id": "A",
                "source_query_id": "s1",
                "question": "Q?",
                "response": "A.",
                "raw_membership_sha256": hashlib.sha256(
                    len("Q?".encode()).to_bytes(8, "big")
                    + b"Q?"
                    + len("A.".encode()).to_bytes(8, "big")
                    + b"A."
                ).hexdigest(),
                "formatted_sha256": hashlib.sha256(
                    b"Question: Q?\nResponse: A."
                ).hexdigest(),
            }
        ],
        audits={"A": {"usable": 1}, "B": {"usable": 0}},
        formatted_inputs={
            "target_queries": ["query"],
            "profile_documents": ["Question: Q?\nResponse: A."],
            "label_questions": ["label question"],
        },
        query_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        record_embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
        label_question_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        label_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        e5_rankings={"q1": ["A", "B"]},
        metadata={
            "qrel_free": True,
            "fixture": True,
            "source_tree_sha256": profile_inference_artifact._source_tree_sha256(),
            "profile_encoder": {
                "input_sha256": "profile",
                "truncated_input_count": 1,
                "encoding_seconds": 10.0,
            },
            "label_encoder": {"input_sha256": "labels", "encoding_seconds": 5.0},
        },
    )


def _refresh_manifest(root: Path, relative_path: str) -> None:
    path = root / relative_path
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][relative_path] = {
        "byte_size": path.stat().st_size,
        "sha256": profile_inference_artifact._sha256(path),
    }
    manifest["aggregate_sha256"] = profile_inference_artifact._aggregate_hash(
        manifest["files"]
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def test_inference_artifact_public_boundary_has_no_qrels_or_sklearn_options() -> None:
    assert ARTIFACT_VERSION == 2
    assert not any("qrel" in field.name.lower() for field in fields(ProfileInferenceArtifactConfig))
    for function in (
        run_profile_inference_artifact,
        write_profile_inference_artifacts,
        load_profile_inference_artifacts,
    ):
        assert "qrel" not in str(inspect.signature(function)).lower()


def test_inference_writer_rejects_global_duplicate_record_ids(tmp_path: Path) -> None:
    result = _result()
    duplicate = {
        **result.records[0],
        "candidate_id": "B",
    }
    duplicate_result = replace(
        result,
        records=[result.records[0], duplicate],
        formatted_inputs={
            **result.formatted_inputs,
            "profile_documents": result.formatted_inputs["profile_documents"] * 2,
        },
        record_embeddings=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )

    with pytest.raises(ValueError, match="record IDs are not globally unique"):
        write_profile_inference_artifacts(duplicate_result, tmp_path / "duplicate")


def test_legacy_repair_is_pinned_atomic_and_embedding_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "legacy"
    destination = tmp_path / "repaired"
    amendment = (
        Path(__file__).parents[1]
        / "protocols/trec_2026_candidate_semantic_profiles_record_id_amendment.json"
    )
    base = _result()
    first = base.records[0]
    second = {
        **first,
        "candidate_id": "B",
        "record_id": canonical_profile_record_id("B", "s1", "Q?", "A."),
    }
    legacy_result = replace(
        base,
        records=[first, second],
        formatted_inputs={
            **base.formatted_inputs,
            "profile_documents": base.formatted_inputs["profile_documents"] * 2,
        },
        record_embeddings=np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32),
    )
    write_profile_inference_artifacts(legacy_result, source)
    records = json.loads((source / "records.json").read_text(encoding="utf-8"))
    legacy_id = f"s1:{records[0]['raw_membership_sha256'][:16]}"
    for record in records:
        record["record_id"] = legacy_id
    (source / "records.json").write_text(
        json.dumps(records, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config["version"] = 1
    (source / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    entries = profile_inference_artifact._manifest_entries(source)
    manifest = {
        "artifact_type": "profile_inference_manifest",
        "version": 1,
        "qrel_free": True,
        "files": entries,
        "aggregate_sha256": profile_inference_artifact._aggregate_hash(entries),
    }
    (source / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="manifest contract"):
        load_profile_inference_artifacts(source)
    changed_amendment = tmp_path / "changed-amendment.json"
    changed_amendment.write_bytes(amendment.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="amendment hash"):
        repair_legacy_profile_inference_artifact(
            source, tmp_path / "wrong-amendment", changed_amendment
        )
    with monkeypatch.context() as context:
        context.setattr(
            profile_inference_artifact,
            "LEGACY_PASS_ONE_AGGREGATE_SHA256",
            "0" * 64,
        )
        with pytest.raises(ValueError, match="pinned pass-one"):
            repair_legacy_profile_inference_artifact(
                source, tmp_path / "wrong-source", amendment
            )
    monkeypatch.setattr(
        profile_inference_artifact,
        "LEGACY_PASS_ONE_AGGREGATE_SHA256",
        manifest["aggregate_sha256"],
    )
    source_manifest_bytes = (source / "manifest.json").read_bytes()
    source_array_hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in profile_inference_artifact.ARRAY_FILES
    }

    receipt = repair_legacy_profile_inference_artifact(
        source, destination, amendment
    )

    assert (source / "manifest.json").read_bytes() == source_manifest_bytes
    repaired = load_profile_inference_artifacts(destination)
    assert len({record["record_id"] for record in repaired.records}) == 2
    assert receipt["source_aggregate_sha256"] == manifest["aggregate_sha256"]
    assert receipt["qrel_free"] is True
    assert receipt["embedding_arrays_byte_identical"] is True
    assert source_array_hashes == {
        name: hashlib.sha256((destination / name).read_bytes()).hexdigest()
        for name in profile_inference_artifact.ARRAY_FILES
    }
    for before, after in zip(records, repaired.records, strict=True):
        assert {key: value for key, value in before.items() if key != "record_id"} == {
            key: value for key, value in after.items() if key != "record_id"
        }
    fresh = tmp_path / "fresh"
    write_profile_inference_artifacts(legacy_result, fresh)
    execution_path = tmp_path / "fresh-execution.json"
    execution_path.write_text(
        json.dumps(
            build_profile_inference_execution_receipt(fresh),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    repeat = compare_profile_inference_repeats(
        destination, fresh, execution_path
    )
    assert repeat["status"] == "passed"
    assert repeat["first_is_pinned_id_repair"] is True
    assert repeat["second_has_no_repair_lineage"] is True
    assert repeat["second_fresh_execution_receipt_valid"] is True
    assert repeat["metadata_lineage_exceptions"] == [
        "record_id_repair",
        "source_tree_sha256: source inference tree replaced by repair tree",
    ]

    def mutate_repair_metadata(root: Path, key: str, value: object) -> None:
        metadata_path = root / "metadata.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["record_id_repair"][key] = value
        metadata_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        changed_entries = profile_inference_artifact._manifest_entries(root)
        changed_manifest = {
            "artifact_type": "profile_inference_manifest",
            "version": 2,
            "qrel_free": True,
            "files": changed_entries,
            "aggregate_sha256": profile_inference_artifact._aggregate_hash(
                changed_entries
            ),
        }
        (root / "manifest.json").write_text(
            json.dumps(changed_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    extra = tmp_path / "extra-provenance"
    shutil.copytree(destination, extra)
    mutate_repair_metadata(extra, "unapproved", True)
    with pytest.raises(ValueError, match="provenance keys"):
        compare_profile_inference_repeats(extra, fresh)

    changed_hash = tmp_path / "changed-byte-hash"
    shutil.copytree(destination, changed_hash)
    metadata_payload = json.loads(
        (changed_hash / "metadata.json").read_text(encoding="utf-8")
    )
    locked = dict(metadata_payload["record_id_repair"]["byte_identical_files"])
    locked["query_embeddings.npy"] = "0" * 64
    mutate_repair_metadata(changed_hash, "byte_identical_files", locked)
    with pytest.raises(ValueError, match="byte-identity hashes"):
        compare_profile_inference_repeats(changed_hash, fresh)

    with pytest.raises(ValueError, match="only repeat one"):
        compare_profile_inference_repeats(fresh, destination)

    failed_destination = tmp_path / "failed-atomic-repair"
    original_copytree = profile_inference_artifact.shutil.copytree

    def fail_copytree(source_path, destination_path, **kwargs):
        Path(destination_path, "partial").write_text("partial", encoding="utf-8")
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr(profile_inference_artifact.shutil, "copytree", fail_copytree)
    with pytest.raises(RuntimeError, match="injected"):
        repair_legacy_profile_inference_artifact(
            source, failed_destination, amendment
        )
    assert not failed_destination.exists()
    assert not list(tmp_path.glob(".failed-atomic-repair.*"))
    monkeypatch.setattr(profile_inference_artifact.shutil, "copytree", original_copytree)


def test_full_inference_requires_exact_qrel_free_gfx1151_smoke_receipt() -> None:
    validate_profile_smoke_receipt(_smoke_receipt())
    changed = {**_smoke_receipt(), "device_architecture": "gfx1100"}
    with pytest.raises(ValueError, match="smoke receipt"):
        validate_profile_smoke_receipt(changed)


def test_label_question_inventory_includes_raw_questions_with_no_usable_response() -> None:
    corpus = (
        SimpleNamespace(source_query_id="s2", query_text="second", response=None),
        SimpleNamespace(source_query_id="s1", query_text="first", response="usable"),
        SimpleNamespace(source_query_id="s2", query_text="second", response="usable"),
    )
    identifiers, questions = ordered_source_questions(corpus)
    assert identifiers == ("s1", "s2")
    assert questions == {"s1": "first", "s2": "second"}
    changed = {**_smoke_receipt(), "rankings_or_qrels_used": True}
    with pytest.raises(ValueError, match="smoke receipt"):
        validate_profile_smoke_receipt(changed)


def test_inference_artifact_round_trip_manifest_and_repeat(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    write_profile_inference_artifacts(_result(), first)
    second_result = _result()
    write_profile_inference_artifacts(
        replace(
            second_result,
            metadata={
                **second_result.metadata,
                "profile_encoder": {
                    **second_result.metadata["profile_encoder"],
                    "encoding_seconds": 11.0,
                },
            },
        ),
        second,
    )
    loaded = load_profile_inference_artifacts(first)
    assert loaded.query_ids == ("q1",)
    assert loaded.query_embeddings.dtype == np.float32
    assert loaded.query_embeddings.flags.c_contiguous
    assert loaded.manifest["aggregate_sha256"]
    receipt = compare_profile_inference_repeats(first, second)
    assert receipt["status"] == "passed"
    assert receipt["integrity_checks"]["metadata_and_token_audits_identical"] is True
    with pytest.raises(ValueError, match="distinct"):
        compare_profile_inference_repeats(first, first)

    changed = _result()
    changed_metadata = {
        **changed.metadata,
        "profile_encoder": {
            **changed.metadata["profile_encoder"],
            "truncated_input_count": 2,
        },
    }
    third = tmp_path / "third"
    write_profile_inference_artifacts(replace(changed, metadata=changed_metadata), third)
    with pytest.raises(ValueError, match="metadata"):
        compare_profile_inference_repeats(second, third)

    with (first / "query_embeddings.npy").open("ab") as handle:
        handle.write(b"mutation")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_profile_inference_artifacts(first)


@pytest.mark.parametrize(
    ("key", "value"),
    (("unexpected", True), ("artifact_type", "changed")),
)
def test_inference_loader_rejects_nonexact_configuration_schema(
    tmp_path: Path, key: str, value: object
) -> None:
    destination = tmp_path / f"artifact-{key}"
    write_profile_inference_artifacts(_result(), destination)
    config_path = destination / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[key] = value
    config_path.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _refresh_manifest(destination, "config.json")

    with pytest.raises(ValueError, match="configuration schema"):
        load_profile_inference_artifacts(destination)
