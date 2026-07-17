import numpy as np
import pytest

from scripts.run_issue13_matched_support_preflight import (
    mean_grades,
    rank_candidates,
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
