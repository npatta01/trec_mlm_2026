import numpy as np
import pytest

from trec_mlm.raw_response_refinement import (
    combine_hierarchical_scores,
    rank_score_matrix,
    score_candidate_records,
)


def test_raw_scores_average_top_k_and_break_ties_by_record_id() -> None:
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    records = np.array(
        [[0.8, 0.6], [0.8, -0.6], [1.0, 0.0]], dtype=np.float32
    )

    result = score_candidate_records(queries, records, ("b", "a", "c"), top_k=2)

    assert result.record_ids == (("c", "a"),)
    np.testing.assert_array_equal(result.record_indices, [[2, 1]])
    assert result.mean_scores[0] == pytest.approx(0.9)
    np.testing.assert_allclose(result.similarities, [[1.0, 0.8]])


def test_raw_scores_average_all_records_when_support_is_below_ten() -> None:
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    records = np.array([[0.2, 0.0], [0.6, 0.0]], dtype=np.float32)

    result = score_candidate_records(queries, records, ("r1", "r2"))

    assert result.record_ids == (("r2", "r1"),)
    assert result.record_indices.shape == (1, 2)
    assert result.similarities.shape == (1, 2)
    assert result.mean_scores[0] == pytest.approx(0.4)


@pytest.mark.parametrize(
    ("queries", "records"),
    [
        (
            np.array([[np.nan, 0.0]], dtype=np.float32),
            np.array([[1.0, 0.0]], dtype=np.float32),
        ),
        (
            np.array([[1.0, 0.0]], dtype=np.float32),
            np.array([[np.inf, 0.0]], dtype=np.float32),
        ),
    ],
)
def test_raw_scores_reject_non_finite_inputs(
    queries: np.ndarray, records: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="finite"):
        score_candidate_records(queries, records, ("r1",))


def test_raw_scores_reject_record_id_row_count_mismatch() -> None:
    queries = np.array([[1.0, 0.0]], dtype=np.float32)
    records = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="record.*count"):
        score_candidate_records(queries, records, ("r1",))


def test_hierarchical_scores_use_fixed_float64_arithmetic() -> None:
    broad = np.array([[0.1, 0.7]], dtype=np.float32)
    refined = np.array([[0.9, -0.3]], dtype=np.float32)

    result = combine_hierarchical_scores(broad, refined)

    expected = (
        np.float64(0.8) * broad.astype(np.float64)
        + np.float64(0.2) * refined.astype(np.float64)
    )
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, expected)


def test_rank_score_matrix_breaks_candidate_score_ties_by_candidate_id() -> None:
    rankings = rank_score_matrix(
        ("q1", "q2"),
        ("beta", "alpha", "gamma"),
        np.array([[0.5, 0.5, 0.1], [-1.0, -1.0, 0.0]], dtype=np.float64),
    )

    assert rankings == {
        "q1": ["alpha", "beta", "gamma"],
        "q2": ["gamma", "alpha", "beta"],
    }
