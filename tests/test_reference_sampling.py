import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from trec_mlm.artifact_io import sha256_file
from trec_mlm.reference_data import QuestionRecord
from trec_mlm.reference_sampling import (
    build_clustered_sample,
    historical_difficulty_features,
    question_strata,
)
from scripts.build_reference_validation_sample import _validate_cluster_labels


def _record(i: int, *, unresolved: bool = False) -> QuestionRecord:
    return QuestionRecord(
        str(i), f"What happened in {2000+i}?", "answer [1]", "answer", (), unresolved
    )


def test_question_strata_marks_numeric_time_sensitive_and_underspecified() -> None:
    assert set(question_strata("What is the latest population in 2026?")) >= {
        "factoid",
        "numeric",
        "time_sensitive",
    }
    assert "underspecified" in question_strata("What did the company do next?")


def test_sample_contains_every_cluster_and_required_id_and_is_repeatable() -> None:
    records = [_record(i, unresolved=i == 7) for i in range(12)]
    embeddings = np.asarray([[i, i % 3 + 1] for i in range(12)], dtype=np.float32)
    labels = np.asarray([0] * 4 + [1] * 4 + [2] * 4)

    first = build_clustered_sample(
        records, embeddings, labels, sample_size=7, seed=2026, required_ids={"7"}
    )
    second = build_clustered_sample(
        records, embeddings, labels, sample_size=7, seed=2026, required_ids={"7"}
    )

    assert first == second
    assert len(first) == 7
    assert {row["cluster_id"] for row in first} == {0, 1, 2}
    assert "7" in {row["question_id"] for row in first}


def test_historical_features_reduce_responses_to_aggregates(tmp_path: Path) -> None:
    path = tmp_path / "responses.json"
    path.write_text(
        json.dumps(
            {
                "q1": {
                    "query": "Where did Ada Lovelace meet Charles Babbage?",
                    "responses": {
                        "a": "I don't know",
                        "b": "At a party!",
                        "c": "at a party",
                        "d": "At a lecture",
                    },
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    features = historical_difficulty_features(path)
    repeated = historical_difficulty_features(path)

    assert features == repeated
    assert features.thresholds == repeated.thresholds
    assert features.thresholds == {
        "exact_refusal_share": pytest.approx(0.25),
        "normalized_answer_disagreement_entropy": pytest.approx(
            features["q1"]["normalized_answer_disagreement_entropy"]
        ),
        "entity_density": pytest.approx(2 / 7),
    }
    assert features["q1"]["response_count"] == 4
    assert features["q1"]["exact_refusal_share"] == pytest.approx(0.25)
    assert features["q1"]["non_refusal_share"] == pytest.approx(0.75)
    assert features["q1"]["distinct_normalized_non_refusal_answers"] == 2
    assert 0 < features["q1"]["normalized_answer_disagreement_entropy"] < 1
    assert features["q1"]["entity_density"] == pytest.approx(2 / 7)
    assert features["q1"]["high_refusal"] is True
    assert features["q1"]["high_disagreement"] is True
    assert features["q1"]["high_entity_density"] is True
    assert "At a party!" not in json.dumps(features)


def test_sample_rejects_invalid_arrays_and_missing_requirements() -> None:
    records = [_record(0), _record(1)]
    with pytest.raises(ValueError, match="finite"):
        build_clustered_sample(
            records,
            np.asarray([[1.0], [np.nan]]),
            np.asarray([0, 1]),
            sample_size=2,
            seed=2026,
        )
    with pytest.raises(ValueError, match="required question IDs"):
        build_clustered_sample(
            records,
            np.asarray([[1.0], [2.0]]),
            np.asarray([0, 1]),
            sample_size=2,
            seed=2026,
            required_ids={"missing"},
        )


def test_cli_rejects_collapsed_kmeans_labels() -> None:
    with pytest.raises(ValueError, match=r"missing=\[1\].*extra=\[\]"):
        _validate_cluster_labels(np.asarray([0, 0, 2]), cluster_count=3)


def test_sampling_cli_writes_private_aggregate_manifest_without_raw_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache.json"
    records = {
        f"q{i}": {
            "question": f"What happened to Place {i}?",
            "response": {"answer": f"baseline {i}", "citations": []},
        }
        for i in range(6)
    }
    cache.write_text(
        json.dumps(
            {
                "total_questions": 6,
                "records": records,
                "retry_summary": {"unresolved": ["q5"]},
            }
        ),
        encoding="utf-8",
    )
    discovery = tmp_path / "discovery.json"
    discovery.write_text(
        json.dumps(
            {
                f"q{i}": {
                    "query": f"What happened to Place {i}?",
                    "responses": {
                        "a": "I don't know",
                        "b": "RAW CANDIDATE SECRET" if i == 0 else f"answer {i}",
                    },
                }
                for i in range(6)
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    analogues = tmp_path / "analogues.tsv"
    analogues.write_text(
        "nearest_discovery_query_text\n"
        "What happened to Place 4?\n"
        "This does not match\n",
        encoding="utf-8",
    )

    class FakeEncoder:
        def __init__(self, model_id: str, *, revision: str, device: str) -> None:
            assert (model_id, revision, device) == ("fake/encoder", "abc123", "cpu")

        def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
            assert len(texts) == 6
            return np.asarray(
                [[1, 0], [2, 0], [3, 0], [0, 1], [0, 2], [0, 3]],
                dtype=np.float32,
            )

    class FakeKMeans:
        def __init__(self, *, n_clusters: int, random_state: int, n_init: int) -> None:
            assert (n_clusters, random_state, n_init) == (2, 2026, 10)

        def fit_predict(self, embeddings: np.ndarray) -> np.ndarray:
            assert np.allclose(np.linalg.norm(embeddings, axis=1), 1)
            return np.asarray([0, 0, 0, 1, 1, 1])

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeEncoder),
    )
    monkeypatch.setitem(
        sys.modules,
        "sklearn.cluster",
        types.SimpleNamespace(KMeans=FakeKMeans),
    )
    from scripts.build_reference_validation_sample import main

    output = tmp_path / "output"
    assert (
        main(
            [
                "--perplexity-cache",
                str(cache),
                "--expected-perplexity-sha256",
                sha256_file(cache),
                "--discovery-responses",
                str(discovery),
                "--analogue-cases",
                str(analogues),
                "--encoder",
                "fake/encoder",
                "--revision",
                "abc123",
                "--clusters",
                "2",
                "--sample-size",
                "4",
                "--seed",
                "2026",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    manifest_text = (output / "sample_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert len(manifest["questions"]) == 4
    assert {row["cluster_id"] for row in manifest["questions"]} == {0, 1}
    assert manifest["record_count"] == 6
    assert manifest["sample_size"] == 4
    assert manifest["cluster_count"] == 2
    assert manifest["seed"] == 2026
    assert manifest["encoder"] == {"id": "fake/encoder", "revision": "abc123"}
    assert manifest["historical_difficulty_thresholds"] == {
        "entity_density": pytest.approx(1 / 5),
        "exact_refusal_share": pytest.approx(0.5),
        "normalized_answer_disagreement_entropy": pytest.approx(0.0),
    }
    assert manifest["sources"]["perplexity_cache"]["sha256"] == sha256_file(cache)
    assert manifest["sources"]["discovery_responses"]["sha256"] == sha256_file(
        discovery
    )
    assert manifest["sources"]["analogue_cases"]["sha256"] == sha256_file(analogues)
    assert manifest["analogue_diagnostics"]["matched"] == [
        {
            "nearest_discovery_query_text": "What happened to Place 4?",
            "question_id": "q4",
        }
    ]
    assert manifest["analogue_diagnostics"]["unmatched"] == ["This does not match"]
    assert all(row["selection_reasons"] for row in manifest["questions"])
    assert "q5" in {row["question_id"] for row in manifest["questions"]}
    assert "RAW CANDIDATE SECRET" not in manifest_text
    assignments = (output / "cluster_assignments.tsv").read_text().splitlines()
    assert len(assignments) == 7
