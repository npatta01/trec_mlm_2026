import numpy as np
import pytest

from scripts.run_issue13_matched_support_preflight import (
    mean_grades,
    rank_candidates,
    run_preflight,
    select_positive_neighbors,
)


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
    serialized = str(result).lower()
    assert "per_query" not in serialized
    assert "query_ids" not in serialized
    assert "rankings" not in serialized
