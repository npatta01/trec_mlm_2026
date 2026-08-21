from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trec_mlm import profile_retrieval, submission
from trec_mlm.artifact_io import (
    length_framed_sha256,
    manifest_aggregate_sha256,
    sha256_file,
    write_pretty_json,
)
from trec_mlm.cli import build_parser
from trec_mlm.profile_experiment import (
    ProfileExperimentConfig,
    ProfileExperimentResult,
    freeze_profile_rankings,
    write_profile_artifacts,
)
from trec_mlm import profile_experiment
from trec_mlm.profile_inference_artifact import (
    ProfileInferenceArtifactConfig,
    ProfileInferenceArtifactResult,
    write_profile_inference_artifacts,
)
from trec_mlm.profile_retrieval import (
    CandidateProfile,
    aggregate_cluster_labels,
    canonical_profile_record_id,
    score_profiles,
)
from trec_mlm.runio import rankings_from_run, validate_trec_run
from trec_mlm.submission import (
    load_submission_config,
    load_unseen_queries,
    reject_qrel_material,
    run_submission,
)


REPO_ROOT = Path(__file__).parents[1]
QUERY_IDS = ("q1", "q2")
CANDIDATE_IDS = ("A", "B", "C")
DENSE = {"q1": ["A", "B", "C"], "q2": ["C", "B", "A"]}
E5 = {"q1": ["B", "A", "C"], "q2": ["A", "C", "B"]}
PROFILE = {"q1": ["C", "A", "B"], "q2": ["B", "C", "A"]}


def _fixture_question(candidate_id: str, index: int) -> str:
    return f"question {candidate_id}{index}"


def _fixture_response(candidate_id: str, index: int) -> str:
    return f"answer {candidate_id}{index}"


def _fixture_record_id(candidate_id: str, index: int) -> str:
    return canonical_profile_record_id(
        candidate_id,
        f"s{index}",
        _fixture_question(candidate_id, index),
        _fixture_response(candidate_id, index),
    )


def _fixture_singleton_dispersion(
    raw_vector: np.ndarray, centroid: np.ndarray
) -> float:
    similarities = np.asarray(raw_vector[None, :] @ centroid, dtype=np.float32)
    return float(np.mean(1.0 - similarities, dtype=np.float64))


def _construction_result() -> ProfileExperimentResult:
    raw_centroids = {
        "A": np.asarray(
            [[0.7, 0.0, np.sqrt(0.51)], [0.0, 0.1, np.sqrt(0.99)]], dtype=np.float32
        ),
        "B": np.asarray([[0.6, 0.0, 0.8], [0.0, 0.9, np.sqrt(0.19)]], dtype=np.float32),
        "C": np.asarray([[0.8, 0.0, 0.6], [0.0, 0.8, 0.6]], dtype=np.float32),
    }
    centroids = {
        candidate_id: np.stack(
            [
                profile_experiment._recompute_profile_centroid(row[None, :])
                for row in values
            ]
        )
        for candidate_id, values in raw_centroids.items()
    }
    query_embeddings = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    label_question_embeddings = query_embeddings.copy()
    label_embeddings = np.eye(3, dtype=np.float32)
    question_label_scores = label_question_embeddings @ label_embeddings.T
    profile = score_profiles(
        query_ids=QUERY_IDS,
        candidate_ids=CANDIDATE_IDS,
        query_embeddings=query_embeddings,
        profiles={
            candidate_id: CandidateProfile(candidate_id, 2, values, (), True, 1)
            for candidate_id, values in centroids.items()
        },
    )
    assert profile.rankings == PROFILE
    frozen = freeze_profile_rankings(
        query_ids=QUERY_IDS,
        candidate_ids=CANDIDATE_IDS,
        dense_rankings=DENSE,
        e5_rankings=E5,
        profile_scores=profile,
        unavailable_profile_ids=frozenset(),
    )
    return ProfileExperimentResult(
        config=ProfileExperimentConfig(
            query_path="queries.tsv",
            discovery_path="discovery.json",
            candidate_path="candidates.json",
            dense_artifact_path="dense",
        ),
        query_ids=QUERY_IDS,
        candidate_ids=CANDIDATE_IDS,
        profile_scores=profile.scores,
        winning_centroid_ids=profile.winning_centroid_ids,
        rankings=frozen.rankings,
        fusion_scores=frozen.scores,
        metadata={"qrel_free": True, "fixture": True},
        profile_payload={
            candidate_id: {
                "available": True,
                "candidate_id": candidate_id,
                "requested_clusters": 2,
                "actual_clusters": 2,
                "centroids": values.tolist(),
                "converged": True,
                "iteration_count": 1,
                "audit": {"usable": 2},
                "clusters": [
                    {
                        "cluster_id": index,
                        "member_record_ids": [_fixture_record_id(candidate_id, index)],
                        "support_count": 1,
                        "dispersion": _fixture_singleton_dispersion(
                            raw_centroids[candidate_id][index], values[index]
                        ),
                        "medoid_record_id": _fixture_record_id(candidate_id, index),
                        "representative_questions": [
                            _fixture_question(candidate_id, index)
                        ],
                        "membership_sha256": length_framed_sha256(
                            [_fixture_record_id(candidate_id, index)]
                        ),
                        "annotation": asdict(
                            aggregate_cluster_labels(
                                member_question_indices=[index],
                                question_label_scores=question_label_scores,
                                label_ids=CANDIDATE_IDS,
                            )
                        ),
                    }
                    for index in range(2)
                ],
            }
            for candidate_id, values in centroids.items()
        },
        source_rankings={"dense_reference": DENSE, "contextual_e5_control": E5},
        record_payload=[
            {
                "record_id": _fixture_record_id(candidate_id, index),
                "candidate_id": candidate_id,
                "source_query_id": f"s{index}",
                "raw_membership_sha256": length_framed_sha256(
                    [
                        _fixture_question(candidate_id, index),
                        _fixture_response(candidate_id, index),
                    ]
                ),
                "formatted_sha256": hashlib.sha256(
                    (
                        f"Question: {_fixture_question(candidate_id, index)}\n"
                        f"Response: {_fixture_response(candidate_id, index)}"
                    ).encode()
                ).hexdigest(),
                "question": _fixture_question(candidate_id, index),
                "response": _fixture_response(candidate_id, index),
            }
            for candidate_id in CANDIDATE_IDS
            for index in range(2)
        ],
        formatted_inputs={
            "target_queries": ["query 1", "query 2"],
            "profile_documents": [
                f"Question: {_fixture_question(candidate_id, index)}\n"
                f"Response: {_fixture_response(candidate_id, index)}"
                for candidate_id in CANDIDATE_IDS
                for index in range(2)
            ],
            "label_questions": ["label question 0", "label question 1"],
        },
        label_question_ids=("s0", "s1"),
        query_embeddings=query_embeddings,
        record_embeddings=np.concatenate(
            [raw_centroids[item] for item in CANDIDATE_IDS]
        ),
        label_question_embeddings=label_question_embeddings,
        label_embeddings=label_embeddings,
    )


def _inference_result(
    query_ids: tuple[str, ...] = QUERY_IDS,
) -> ProfileInferenceArtifactResult:
    question, response = "Q?", "A."
    formatted = f"Question: {question}\nResponse: {response}"
    query_rows = {
        "q1": [1.0, 0.0, 0.0],
        "q2": [0.0, 1.0, 0.0],
        "qX": [0.0, 0.0, 1.0],
    }
    return ProfileInferenceArtifactResult(
        config=ProfileInferenceArtifactConfig(
            query_path="queries.tsv",
            discovery_path="discovery.json",
            candidate_path="candidates.json",
            e5_artifact_path="e5",
        ),
        query_ids=query_ids,
        candidate_ids=CANDIDATE_IDS,
        label_question_ids=("s1",),
        records=[
            {
                "record_id": canonical_profile_record_id("A", "s1", question, response),
                "candidate_id": "A",
                "source_query_id": "s1",
                "question": question,
                "response": response,
                "raw_membership_sha256": length_framed_sha256((question, response)),
                "formatted_sha256": hashlib.sha256(formatted.encode()).hexdigest(),
            }
        ],
        audits={"A": {"usable": 1}, "B": {"usable": 0}, "C": {"usable": 0}},
        formatted_inputs={
            "target_queries": [f"query {item}" for item in query_ids],
            "profile_documents": [formatted],
            "label_questions": ["label question"],
        },
        query_embeddings=np.asarray(
            [query_rows[item] for item in query_ids], dtype=np.float32
        ),
        record_embeddings=np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
        label_question_embeddings=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        label_embeddings=np.eye(3, dtype=np.float32),
        e5_rankings={item: ["A", "B", "C"] for item in query_ids},
        metadata={"qrel_free": True, "fixture": True},
    )


def _write_queries_tsv(path: Path) -> Path:
    path.write_text("q1\tquery one\nq2\tquery two\n", encoding="utf-8")
    return path


def _write_candidate_source(path: Path, candidate_ids: list[str]) -> Path:
    write_pretty_json(path, {"candidate_ids": candidate_ids})
    return path


def _config_payload(
    tmp_path: Path,
    *,
    construction_dir: Path,
    expected_aggregate: str,
    candidate_ids: list[str] | None = None,
    fixture: bool = True,
) -> dict[str, Any]:
    ids = sorted(candidate_ids if candidate_ids is not None else list(CANDIDATE_IDS))
    source = _write_candidate_source(tmp_path / "candidates.json", ids)
    return {
        "config_type": "trec_submission_system",
        "version": 1,
        "qrel_free": True,
        "fixture": fixture,
        "system_id": "fixture_profile_scoring",
        "run_id": "fixture_profile_scoring_v1",
        "backend": "profile_scoring",
        "candidate_manifest": {
            "candidate_count": len(ids),
            "candidate_ids": ids,
            "candidate_ids_sha256": length_framed_sha256(ids),
            "source_path": str(source),
            "source_sha256": sha256_file(source),
        },
        "backend_config": {
            "construction_artifact": {
                "path": str(construction_dir),
                "expected_aggregate_sha256": expected_aggregate,
            }
        },
    }


def _write_fixture_world(tmp_path: Path) -> dict[str, Path]:
    inference_dir = tmp_path / "inference"
    construction_dir = tmp_path / "construction"
    write_profile_inference_artifacts(_inference_result(), inference_dir)
    write_profile_artifacts(_construction_result(), construction_dir)
    manifest = json.loads(
        (construction_dir / "manifest.json").read_text(encoding="utf-8")
    )
    config_path = tmp_path / "system.json"
    write_pretty_json(
        config_path,
        _config_payload(
            tmp_path,
            construction_dir=construction_dir,
            expected_aggregate=manifest["aggregate_sha256"],
        ),
    )
    return {
        "queries": _write_queries_tsv(tmp_path / "release.tsv"),
        "inference": inference_dir,
        "construction": construction_dir,
        "config": config_path,
    }


def test_submission_public_boundary_has_no_qrels() -> None:
    parser = build_parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    submit = subparsers.choices["submit-run"]
    options = [
        option for action in submit._actions for option in action.option_strings
    ]
    assert "--queries" in options and "--config" in options
    assert not any("qrel" in option.lower() for option in options)
    for function in (run_submission, load_submission_config, load_unseen_queries):
        assert "qrel" not in str(inspect.signature(function)).lower()
    assert not any(
        "qrel" in key.lower()
        for key in submission.SUBMISSION_CONFIG_KEYS - {"qrel_free"}
    )
    with pytest.raises(ValueError, match="forbidden"):
        reject_qrel_material({"qrels_path": "x"}, context="test")
    with pytest.raises(ValueError, match="forbidden"):
        reject_qrel_material({"paths": ["data/dev/llm_dev_qrels.txt"]}, context="test")


def test_unseen_query_loading_accepts_tsv_and_jsonl(tmp_path: Path) -> None:
    tsv = _write_queries_tsv(tmp_path / "release.tsv")
    assert load_unseen_queries(tsv) == {"q1": "query one", "q2": "query two"}

    jsonl = tmp_path / "topics.jsonl"
    jsonl.write_text(
        '{"topic_id": "MLLM_051", "question": "what?"}\n'
        '\n{"topic_id": "MLLM_052", "question": "who?"}\n',
        encoding="utf-8",
    )
    assert load_unseen_queries(jsonl) == {"MLLM_051": "what?", "MLLM_052": "who?"}


@pytest.mark.parametrize(
    ("name", "content", "match"),
    (
        ("dup.jsonl", '{"topic_id": "T1", "question": "a"}\n{"topic_id": "T1", "question": "b"}\n', "duplicate topic ID"),
        ("keys.jsonl", '{"topic_id": "T1", "question": "a", "answer": "b"}\n', "exactly topic_id and question"),
        ("types.jsonl", '{"topic_id": 1, "question": "a"}\n', "must be strings"),
        ("blank.jsonl", '{"topic_id": "T1", "question": "  "}\n', "non-empty"),
        ("broken.jsonl", "{not json}\n", "not valid JSON"),
        ("empty.jsonl", "\n", "empty"),
        ("release.txt", "q1\tquery one\n", "unsupported query format"),
        ("spaced.tsv", "q 1\tquery one\n", "whitespace-free token"),
    ),
)
def test_unseen_query_loading_fails_closed(
    tmp_path: Path, name: str, content: str, match: str
) -> None:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_unseen_queries(path)


def test_unseen_query_loading_refuses_qrels(tmp_path: Path) -> None:
    named = tmp_path / "llm_test_qrels.txt"
    named.write_text("q1\tquery one\n", encoding="utf-8")
    with pytest.raises(ValueError, match="forbidden"):
        load_unseen_queries(named)

    shaped = tmp_path / "release.tsv"
    shaped.write_text(
        "q1 0 A 2\nq1 0 B 0\nq2 0 A 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="shaped like a qrels file"):
        load_unseen_queries(shaped)


def test_official_release_files_load_without_qrels() -> None:
    topics = load_unseen_queries(REPO_ROOT / "data/test/2026-mllm-topics-final.jsonl")
    assert len(topics) == 50
    assert min(topics) == "MLLM_051" and max(topics) == "MLLM_100"
    development = load_unseen_queries(REPO_ROOT / "data/dev/llm_dev_data.tsv")
    assert len(development) == 322


def test_checked_in_production_config_is_frozen_and_blocked_on_artifacts() -> None:
    config = load_submission_config(
        REPO_ROOT / "configs/trec_2026_submission_profile_scoring.json",
        base_dir=REPO_ROOT,
    )
    assert config.fixture is False
    assert config.backend == "profile_scoring"
    assert config.candidate_count == submission.OFFICIAL_CANDIDATE_COUNT
    assert config.candidate_ids == tuple(sorted(config.candidate_ids))
    assert config.backend_config["construction_artifact"][
        "expected_aggregate_sha256"
    ] == "04d6be23d2f9046ed93eb1c01bb5ad56545bb51a9470d408b96d3241b1ee54f4"
    with pytest.raises(FileNotFoundError, match="issue #40"):
        run_submission(
            config_path=REPO_ROOT / "configs/trec_2026_submission_profile_scoring.json",
            queries_path=REPO_ROOT / "data/test/2026-mllm-topics-final.jsonl",
            output_dir=REPO_ROOT / "artifacts" / "never-created",
            artifact_path=None,
            base_dir=REPO_ROOT,
        )


def _mutate(payload: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
    target = payload
    for key in path[:-1]:
        target = target[key]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value


_DELETE = object()


@pytest.mark.parametrize(
    ("path", "value", "match"),
    (
        (("extra",), True, "schema changed"),
        (("config_type",), "changed", "type changed"),
        (("version",), 2, "version changed"),
        (("qrel_free",), False, "qrel_free"),
        (("fixture",), "yes", "must be boolean"),
        (("system_id",), "has space", "system_id"),
        (("run_id",), "", "run_id"),
        (("backend",), "unknown", "unknown submission backend"),
        (("backend_config", "qrels_path"), "x", "forbidden"),
        (
            ("backend_config", "construction_artifact", "path"),
            "artifacts/with_qrels_inside",
            "forbidden",
        ),
        (
            ("backend_config", "construction_artifact", "expected_aggregate_sha256"),
            "not-hex",
            "SHA-256",
        ),
        (("candidate_manifest", "candidate_ids"), ["B", "A", "C"], "sorted"),
        (("candidate_manifest", "candidate_ids"), ["A", "A", "B"], "unique"),
        (("candidate_manifest", "candidate_count"), 4, "disagrees"),
        (("candidate_manifest", "candidate_ids_sha256"), "0" * 64, "hash disagrees"),
        (("candidate_manifest", "source_sha256"), "0" * 64, "hash changed"),
        (("candidate_manifest", "source_path"), _DELETE, "schema changed"),
    ),
)
def test_configuration_validation_fails_closed(
    tmp_path: Path, path: tuple[str, ...], value: Any, match: str
) -> None:
    payload = _config_payload(
        tmp_path, construction_dir=tmp_path / "construction", expected_aggregate="0" * 64
    )
    _mutate(payload, path, value)
    config_path = tmp_path / "system.json"
    write_pretty_json(config_path, payload)
    with pytest.raises((ValueError, FileNotFoundError), match=match):
        load_submission_config(config_path, base_dir=tmp_path)


def test_production_configuration_requires_official_inventory(tmp_path: Path) -> None:
    payload = _config_payload(
        tmp_path,
        construction_dir=tmp_path / "construction",
        expected_aggregate="0" * 64,
        fixture=False,
    )
    config_path = tmp_path / "system.json"
    write_pretty_json(config_path, payload)
    with pytest.raises(ValueError, match="exactly 150"):
        load_submission_config(config_path, base_dir=tmp_path)


def test_configuration_rejects_changed_candidate_source(tmp_path: Path) -> None:
    payload = _config_payload(
        tmp_path, construction_dir=tmp_path / "construction", expected_aggregate="0" * 64
    )
    source = Path(payload["candidate_manifest"]["source_path"])
    write_pretty_json(source, {"candidate_ids": ["A", "B", "D"]})
    payload["candidate_manifest"]["source_sha256"] = sha256_file(source)
    config_path = tmp_path / "system.json"
    write_pretty_json(config_path, payload)
    with pytest.raises(ValueError, match="disagrees with frozen IDs"):
        load_submission_config(config_path, base_dir=tmp_path)


def test_end_to_end_fixture_run_is_deterministic_and_complete(tmp_path: Path) -> None:
    world = _write_fixture_world(tmp_path)
    first_dir = tmp_path / "out" / "first"
    receipt = run_submission(
        config_path=world["config"],
        queries_path=world["queries"],
        output_dir=first_dir,
        artifact_path=str(world["inference"]),
        base_dir=tmp_path,
    )

    validation = validate_trec_run(
        first_dir / "run.trec",
        expected_query_ids=QUERY_IDS,
        expected_model_ids=CANDIDATE_IDS,
    )
    assert validation["run_id"] == "fixture_profile_scoring_v1"
    assert validation["row_count"] == len(QUERY_IDS) * len(CANDIDATE_IDS)
    assert rankings_from_run(first_dir / "run.trec") == PROFILE

    stored = json.loads((first_dir / "receipt.json").read_text(encoding="utf-8"))
    assert stored == receipt
    assert receipt["qrel_free"] is True
    assert receipt["run_id"] == "fixture_profile_scoring_v1"
    assert receipt["system_id"] == "fixture_profile_scoring"
    assert receipt["config"]["sha256"] == sha256_file(world["config"])
    assert receipt["queries"]["sha256"] == sha256_file(world["queries"])
    assert receipt["queries"]["query_count"] == 2
    assert receipt["run"]["query_count"] == 2
    assert receipt["run"]["candidate_count"] == 3
    assert receipt["run"]["sha256"] == sha256_file(first_dir / "run.trec")
    assert receipt["aggregate_sha256"] == manifest_aggregate_sha256(receipt["files"])
    assert set(receipt["files"]) == {"run.trec"}
    inference_manifest = json.loads(
        (world["inference"] / "manifest.json").read_text(encoding="utf-8")
    )
    construction_manifest = json.loads(
        (world["construction"] / "manifest.json").read_text(encoding="utf-8")
    )
    backend_receipt = receipt["backend_receipt"]
    assert backend_receipt["inference_artifact"]["aggregate_sha256"] == (
        inference_manifest["aggregate_sha256"]
    )
    assert backend_receipt["construction_artifact"]["aggregate_sha256"] == (
        construction_manifest["aggregate_sha256"]
    )
    assert backend_receipt["scoring"] == {
        "method": "max_centroid_cosine",
        "top_n": 1,
    }
    assert "python" in receipt["tool_versions"]

    second_dir = tmp_path / "out" / "second"
    run_submission(
        config_path=world["config"],
        queries_path=world["queries"],
        output_dir=second_dir,
        artifact_path=str(world["inference"]),
        base_dir=tmp_path,
    )
    assert (first_dir / "run.trec").read_bytes() == (
        second_dir / "run.trec"
    ).read_bytes()
    assert (first_dir / "receipt.json").read_bytes() == (
        second_dir / "receipt.json"
    ).read_bytes()


def test_missing_artifacts_fail_closed_with_restore_pointer(tmp_path: Path) -> None:
    world = _write_fixture_world(tmp_path)
    with pytest.raises(FileNotFoundError, match="issue #40"):
        run_submission(
            config_path=world["config"],
            queries_path=world["queries"],
            output_dir=tmp_path / "out-a",
            artifact_path=None,
            base_dir=tmp_path,
        )
    with pytest.raises(FileNotFoundError, match="issue #40"):
        run_submission(
            config_path=world["config"],
            queries_path=world["queries"],
            output_dir=tmp_path / "out-b",
            artifact_path=str(tmp_path / "absent"),
            base_dir=tmp_path,
        )
    payload = json.loads(world["config"].read_text(encoding="utf-8"))
    payload["backend_config"]["construction_artifact"]["path"] = str(
        tmp_path / "gone"
    )
    write_pretty_json(world["config"], payload)
    with pytest.raises(FileNotFoundError, match="issue #40"):
        run_submission(
            config_path=world["config"],
            queries_path=world["queries"],
            output_dir=tmp_path / "out-c",
            artifact_path=str(world["inference"]),
            base_dir=tmp_path,
        )
    assert not any(tmp_path.glob("out-*"))


def test_construction_aggregate_pin_mismatch_fails_closed(tmp_path: Path) -> None:
    world = _write_fixture_world(tmp_path)
    payload = json.loads(world["config"].read_text(encoding="utf-8"))
    payload["backend_config"]["construction_artifact"][
        "expected_aggregate_sha256"
    ] = "0" * 64
    write_pretty_json(world["config"], payload)
    with pytest.raises(ValueError, match="aggregate hash differs"):
        run_submission(
            config_path=world["config"],
            queries_path=world["queries"],
            output_dir=tmp_path / "out",
            artifact_path=str(world["inference"]),
            base_dir=tmp_path,
        )


def test_inventory_mismatches_fail_closed(tmp_path: Path) -> None:
    world = _write_fixture_world(tmp_path)
    other_inference = tmp_path / "inference-other"
    write_profile_inference_artifacts(
        _inference_result(query_ids=("q1", "qX")), other_inference
    )
    with pytest.raises(ValueError, match="query inventory differs"):
        run_submission(
            config_path=world["config"],
            queries_path=world["queries"],
            output_dir=tmp_path / "out-a",
            artifact_path=str(other_inference),
            base_dir=tmp_path,
        )
    payload = json.loads(world["config"].read_text(encoding="utf-8"))
    ids = ["A", "B", "Z"]
    source = tmp_path / "candidates-z.json"
    write_pretty_json(source, {"candidate_ids": ids})
    payload["candidate_manifest"] = {
        "candidate_count": 3,
        "candidate_ids": ids,
        "candidate_ids_sha256": length_framed_sha256(ids),
        "source_path": str(source),
        "source_sha256": sha256_file(source),
    }
    write_pretty_json(world["config"], payload)
    with pytest.raises(ValueError, match="candidate inventory differs"):
        run_submission(
            config_path=world["config"],
            queries_path=world["queries"],
            output_dir=tmp_path / "out-b",
            artifact_path=str(world["inference"]),
            base_dir=tmp_path,
        )


def test_production_run_rejects_fixture_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(submission, "OFFICIAL_CANDIDATE_COUNT", 3)
    world = _write_fixture_world(tmp_path)
    payload = json.loads(world["config"].read_text(encoding="utf-8"))
    payload["fixture"] = False
    write_pretty_json(world["config"], payload)
    with pytest.raises(ValueError, match="cannot consume fixture artifacts"):
        run_submission(
            config_path=world["config"],
            queries_path=world["queries"],
            output_dir=tmp_path / "out",
            artifact_path=str(world["inference"]),
            base_dir=tmp_path,
        )


def test_pipeline_rejects_incomplete_backend_rankings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _write_fixture_world(tmp_path)

    def _bad_backend(request: submission.BackendRequest) -> submission.BackendRun:
        return submission.BackendRun(
            rankings={"q1": ["A", "B", "C"], "q2": ["A", "B"]},
            receipt={"backend": "stub"},
        )

    monkeypatch.setitem(submission.SUBMISSION_BACKENDS, "profile_scoring", _bad_backend)
    with pytest.raises(ValueError, match="does not cover the frozen candidate"):
        run_submission(
            config_path=world["config"],
            queries_path=world["queries"],
            output_dir=tmp_path / "out",
            artifact_path=str(world["inference"]),
            base_dir=tmp_path,
        )

    def _duplicate_backend(request: submission.BackendRequest) -> submission.BackendRun:
        return submission.BackendRun(
            rankings={"q1": ["A", "A", "C"], "q2": ["A", "B", "C"]},
            receipt={"backend": "stub"},
        )

    monkeypatch.setitem(
        submission.SUBMISSION_BACKENDS, "profile_scoring", _duplicate_backend
    )
    with pytest.raises(ValueError, match="contains duplicates"):
        run_submission(
            config_path=world["config"],
            queries_path=world["queries"],
            output_dir=tmp_path / "out",
            artifact_path=str(world["inference"]),
            base_dir=tmp_path,
        )

    def _missing_query_backend(
        request: submission.BackendRequest,
    ) -> submission.BackendRun:
        return submission.BackendRun(
            rankings={"q1": ["A", "B", "C"]}, receipt={"backend": "stub"}
        )

    monkeypatch.setitem(
        submission.SUBMISSION_BACKENDS, "profile_scoring", _missing_query_backend
    )
    with pytest.raises(ValueError, match="query inventory mismatch"):
        run_submission(
            config_path=world["config"],
            queries_path=world["queries"],
            output_dir=tmp_path / "out",
            artifact_path=str(world["inference"]),
            base_dir=tmp_path,
        )
    assert not (tmp_path / "out").exists()
    assert not list((tmp_path).glob(".out.*"))


def test_nonfinite_scores_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    world = _write_fixture_world(tmp_path)

    def _nan_scores(**kwargs: Any) -> SimpleNamespace:
        query_ids = tuple(kwargs["query_ids"])
        candidate_ids = tuple(kwargs["candidate_ids"])
        return SimpleNamespace(
            scores=np.full(
                (len(query_ids), len(candidate_ids)), np.nan, dtype=np.float32
            ),
            rankings={item: list(candidate_ids) for item in query_ids},
        )

    monkeypatch.setattr(profile_retrieval, "score_profiles", _nan_scores)
    with pytest.raises(ValueError, match="non-finite"):
        run_submission(
            config_path=world["config"],
            queries_path=world["queries"],
            output_dir=tmp_path / "out",
            artifact_path=str(world["inference"]),
            base_dir=tmp_path,
        )


def test_output_directory_is_never_overwritten(tmp_path: Path) -> None:
    world = _write_fixture_world(tmp_path)
    destination = tmp_path / "out"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        run_submission(
            config_path=world["config"],
            queries_path=world["queries"],
            output_dir=destination,
            artifact_path=str(world["inference"]),
            base_dir=tmp_path,
        )
