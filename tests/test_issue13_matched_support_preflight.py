import json

import numpy as np
import pytest

import scripts.run_issue13_matched_support_preflight as preflight_module
from scripts.run_issue13_matched_support_preflight import (
    FrozenInputContract,
    mean_grades,
    rank_candidates,
    run_preflight,
    select_positive_neighbors,
    validate_frozen_inputs,
)
from trec_mlm.artifact_io import sha256_file
from trec_mlm.dense_query_knn import array_sha256, cosine_similarity_matrix


def test_select_positive_neighbors_uses_similarity_then_query_id() -> None:
    similarities = np.asarray([1.0, 0.4, 0.4, -0.1, 0.8], dtype=np.float32)

    selected = select_positive_neighbors(
        similarities,
        support_indices=(1, 2, 3, 4),
        query_ids=("target", "z", "a", "negative", "best"),
        count=3,
    )

    assert selected == (4, 2, 1)


def test_select_positive_neighbors_rejects_insufficient_positive_support() -> None:
    with pytest.raises(ValueError, match="positive-cosine support"):
        select_positive_neighbors(
            np.asarray([1.0, 0.2, 0.0], dtype=np.float32),
            support_indices=(1, 2),
            query_ids=("target", "positive", "zero"),
            count=2,
        )


def test_mean_grades_is_uniform_and_rejects_empty_support() -> None:
    grades = np.asarray([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])

    assert np.array_equal(mean_grades(grades, (0, 2)), np.asarray([1.5, 0.5]))
    with pytest.raises(ValueError, match="nonempty"):
        mean_grades(grades, ())


def test_rank_candidates_uses_prior_then_candidate_id_for_exact_ties() -> None:
    ranking = rank_candidates(
        np.asarray([1.0, 1.0, 0.5, 1.0]),
        np.asarray([0.1, 0.3, 0.9, 0.3]),
        ("d", "c", "a", "b"),
    )

    assert ranking == ("b", "c", "d", "a")


@pytest.mark.parametrize(
    ("function", "args"),
    [
        (mean_grades, (np.asarray([[np.nan]]), (0,))),
        (rank_candidates, (np.asarray([np.inf]), np.asarray([0.0]), ("a",))),
    ],
)
def test_primitives_reject_nonfinite_arrays(function, args) -> None:
    with pytest.raises(ValueError, match="finite"):
        function(*args)


def test_run_preflight_emits_only_aggregate_exploratory_results() -> None:
    embeddings = np.asarray(
        [
            [1.0, 0.0],
            [0.98, 0.20],
            [0.95, -0.31],
            [-1.0, 0.0],
            [-0.98, 0.20],
            [-0.95, -0.31],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    grades = np.asarray(
        [
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [1.0, 2.0, 0.0],
            [0.0, 0.0, 2.0],
            [0.0, 1.0, 2.0],
            [0.0, 2.0, 1.0],
        ]
    )

    result = run_preflight(
        embeddings=embeddings,
        grades=grades,
        query_ids=tuple(f"q{index}" for index in range(6)),
        candidate_ids=("a", "b", "c"),
        cluster_counts=(2,),
    )

    assert result["status"] == "exploratory_not_promotable"
    assert result["query_count"] == 6
    assert result["candidate_count"] == 3
    assert result["fit_count"] == 6
    assert result["completed_issue13_result"]["outcome"] == "reject"
    assert result["completed_issue13_result"]["candidate"]["ndcg_at_10"] == pytest.approx(
        0.34868084786664477
    )
    comparison = result["cluster_counts"]["2"]
    assert set(comparison) == {
        "cluster_mean",
        "matched_neighbor_mean",
        "paired_delta_cluster_minus_matched",
        "support",
        "score_granularity",
        "target_assignment",
    }
    assert comparison["support"]["count"] == 6
    assert result["decision"]["continue_clustering"] is False
    serialized = str(result).lower()
    assert "per_query" not in serialized
    assert "query_ids" not in serialized
    assert "rankings" not in serialized


def test_run_preflight_excludes_target_and_uses_frozen_kmeans_settings(
    monkeypatch,
) -> None:
    embeddings = np.asarray(
        [[1.0, 0.0], [0.9, 0.1], [0.8, -0.2], [-1.0, 0.0], [-0.9, 0.1], [-0.8, -0.2]],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    grades = np.eye(6, 3, dtype=np.float64)
    real_kmeans = preflight_module.KMeans
    fit_shapes = []
    settings = []
    neighbor_calls = []
    fit_rows = []

    class SpyKMeans:
        def __init__(self, **kwargs):
            settings.append(kwargs)
            self.delegate = real_kmeans(**kwargs)

        def fit_predict(self, rows):
            fit_shapes.append(rows.shape)
            fit_rows.append(rows.copy())
            labels = self.delegate.fit_predict(rows)
            self.cluster_centers_ = self.delegate.cluster_centers_
            return labels

    real_select = preflight_module.select_positive_neighbors

    def spy_select(similarity_row, support_indices, query_ids, count):
        selected = real_select(similarity_row, support_indices, query_ids, count)
        neighbor_calls.append((tuple(support_indices), count, selected))
        return selected

    monkeypatch.setattr(preflight_module, "KMeans", SpyKMeans)
    monkeypatch.setattr(preflight_module, "select_positive_neighbors", spy_select)

    run_preflight(
        embeddings=embeddings,
        grades=grades,
        query_ids=tuple(f"q{index}" for index in range(6)),
        candidate_ids=("a", "b", "c"),
        cluster_counts=(2,),
    )

    assert fit_shapes == [(5, 2)] * 6
    for target, rows in enumerate(fit_rows):
        assert not np.any(np.all(rows == embeddings[target], axis=1))
    assert settings == [
        {
            "n_clusters": 2,
            "init": "k-means++",
            "random_state": 2026,
            "n_init": 10,
            "algorithm": "lloyd",
            "max_iter": 300,
            "tol": 1e-4,
        }
    ] * 6
    assert all(len(support) == 5 for support, _, _ in neighbor_calls)
    assert all(len(selected) == count for _, count, selected in neighbor_calls)


def test_validate_frozen_inputs_rejects_permuted_embedding_rows(tmp_path) -> None:
    query_path = tmp_path / "queries.tsv"
    qrel_path = tmp_path / "qrels.txt"
    embedding_path = tmp_path / "embeddings.npy"
    metadata_path = tmp_path / "metadata.json"
    query_path.write_text("q1\tfirst\nq2\tsecond\n", encoding="utf-8")
    qrel_path.write_text("q1 0 a 2\nq2 0 a 1\n", encoding="utf-8")
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    np.save(embedding_path, embeddings, allow_pickle=False)
    metadata_path.write_text(
        json.dumps(
            {
                "mpnet_base_v2": {
                    "embedding_dimension": 2,
                    "embedding_sha256": array_sha256(embeddings),
                    "input_sha256": preflight_module.query_input_sha256(
                        ("q1", "q2"), ("first", "second")
                    ),
                    "normalized": True,
                    "query_count": 2,
                    "similarity_sha256": array_sha256(
                        cosine_similarity_matrix(embeddings)
                    ),
                }
            }
        ),
        encoding="utf-8",
    )
    contract = FrozenInputContract(
        query_file_sha256=sha256_file(query_path),
        qrel_file_sha256=sha256_file(qrel_path),
        embedding_file_sha256=sha256_file(embedding_path),
        encoder_metadata_sha256=sha256_file(metadata_path),
        query_input_sha256=preflight_module.query_input_sha256(
            ("q1", "q2"), ("first", "second")
        ),
        embedding_logical_sha256=array_sha256(embeddings),
        similarity_logical_sha256=array_sha256(cosine_similarity_matrix(embeddings)),
        embedding_shape=(2, 2),
        embedding_dtype="float32",
    )

    validate_frozen_inputs(
        query_path=query_path,
        qrel_path=qrel_path,
        embedding_path=embedding_path,
        metadata_path=metadata_path,
        query_ids=("q1", "q2"),
        query_texts=("first", "second"),
        embeddings=embeddings,
        contract=contract,
    )

    with pytest.raises(ValueError, match="embedding logical SHA-256"):
        validate_frozen_inputs(
            query_path=query_path,
            qrel_path=qrel_path,
            embedding_path=embedding_path,
            metadata_path=metadata_path,
            query_ids=("q1", "q2"),
            query_texts=("first", "second"),
            embeddings=embeddings[::-1],
            contract=contract,
        )
