from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import inspect
from pathlib import Path

import numpy as np
import pytest

from trec_mlm.profile_inference_artifact import (
    ProfileInferenceArtifactConfig,
    ProfileInferenceArtifactResult,
    compare_profile_inference_repeats,
    load_profile_inference_artifacts,
    run_profile_inference_artifact,
    validate_profile_smoke_receipt,
    write_profile_inference_artifacts,
)


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
                "record_id": "s1:record",
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
            "profile_encoder": {
                "input_sha256": "profile",
                "truncated_input_count": 1,
                "encoding_seconds": 10.0,
            },
            "label_encoder": {"input_sha256": "labels", "encoding_seconds": 5.0},
        },
    )


def test_inference_artifact_public_boundary_has_no_qrels_or_sklearn_options() -> None:
    assert not any("qrel" in field.name.lower() for field in fields(ProfileInferenceArtifactConfig))
    for function in (
        run_profile_inference_artifact,
        write_profile_inference_artifacts,
        load_profile_inference_artifacts,
    ):
        assert "qrel" not in str(inspect.signature(function)).lower()


def test_full_inference_requires_exact_qrel_free_gfx1151_smoke_receipt() -> None:
    validate_profile_smoke_receipt(_smoke_receipt())
    changed = {**_smoke_receipt(), "device_architecture": "gfx1100"}
    with pytest.raises(ValueError, match="smoke receipt"):
        validate_profile_smoke_receipt(changed)
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
