from __future__ import annotations

from dataclasses import FrozenInstanceError
import math
import warnings

import numpy as np
import pytest

from trec_mlm.dense_aggregation import (
    DENSE_AGGREGATION_GRID,
    DenseAggregationCandidate,
    build_fold_evidence,
    combine_and_rank,
    domain_scores,
    fold_global_prior,
    local_evidence,
    select_inner_candidate,
)


def test_candidate_is_a_frozen_slotted_protocol_value() -> None:
    candidate = DenseAggregationCandidate(20, 50, 0.75, 0.1)

    assert candidate.__slots__ == (
        "clusters",
        "neighbors",
        "local_weight",
        "prior_shrinkage",
    )
    with pytest.raises(FrozenInstanceError):
        candidate.clusters = 40  # type: ignore[misc]


def test_grid_has_exact_nested_loop_order_without_deduplication() -> None:
    assert tuple(
        (
            candidate.clusters,
            candidate.neighbors,
            candidate.local_weight,
            candidate.prior_shrinkage,
        )
        for candidate in DENSE_AGGREGATION_GRID
    ) == (
        (20, 20, 0.5, 0.0),
        (20, 20, 0.5, 0.1),
        (20, 20, 0.75, 0.0),
        (20, 20, 0.75, 0.1),
        (20, 50, 0.5, 0.0),
        (20, 50, 0.5, 0.1),
        (20, 50, 0.75, 0.0),
        (20, 50, 0.75, 0.1),
        (40, 20, 0.5, 0.0),
        (40, 20, 0.5, 0.1),
        (40, 20, 0.75, 0.0),
        (40, 20, 0.75, 0.1),
        (40, 50, 0.5, 0.0),
        (40, 50, 0.5, 0.1),
        (40, 50, 0.75, 0.0),
        (40, 50, 0.75, 0.1),
    )
    assert len(DENSE_AGGREGATION_GRID) == 16


def test_hand_calculated_components_and_ranking() -> None:
    grades = np.asarray(
        [[2, 0, 1], [0, 2, 1], [0, 0, 0]], dtype=np.float64
    )
    prior = fold_global_prior(grades=grades, support_indices=(0, 1))
    local = local_evidence(
        target_index=2,
        support_indices=(0, 1),
        similarities=np.asarray(
            [[1, 0, 0.75], [0, 1, 0.25], [0.75, 0.25, 1]],
            dtype=np.float32,
        ),
        grades=grades,
        query_ids=("q1", "q2", "qt"),
        k=2,
        prior=prior,
    )
    domain = domain_scores(member_indices=(0,), grades=grades)
    candidate = DenseAggregationCandidate(20, 2, 0.5, 0.1)

    result = combine_and_rank(
        local=local.scores,
        domain=domain,
        prior=prior,
        model_ids=("m1", "m2", "m3"),
        candidate=candidate,
    )

    np.testing.assert_array_equal(prior, [1.0, 1.0, 1.0])
    np.testing.assert_array_equal(local.scores, [1.5, 0.5, 1.0])
    np.testing.assert_array_equal(domain, [2.0, 0.0, 1.0])
    np.testing.assert_allclose(
        result.local_contribution, [0.675, 0.225, 0.45], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        result.domain_contribution, [0.9, 0.0, 0.45], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        result.prior_contribution, [0.1, 0.1, 0.1], rtol=0, atol=0
    )
    np.testing.assert_allclose(
        result.final_scores, [1.675, 0.325, 1.0], rtol=0, atol=0
    )
    assert tuple(float(value).hex() for value in result.local_contribution) == (
        "0x1.599999999999ap-1",
        "0x1.ccccccccccccdp-3",
        "0x1.ccccccccccccdp-2",
    )
    assert tuple(float(value).hex() for value in result.domain_contribution) == (
        "0x1.ccccccccccccdp-1",
        "0x0.0p+0",
        "0x1.ccccccccccccdp-2",
    )
    assert tuple(float(value).hex() for value in result.prior_contribution) == (
        "0x1.999999999999ap-4",
        "0x1.999999999999ap-4",
        "0x1.999999999999ap-4",
    )
    assert tuple(float(value).hex() for value in result.final_scores) == (
        "0x1.acccccccccccdp+0",
        "0x1.4cccccccccccdp-2",
        "0x1.0000000000000p+0",
    )
    assert result.model_order == (0, 2, 1)
    np.testing.assert_array_equal(result.ranks, [1, 3, 2])
    assert result.ranks.dtype == np.dtype(np.uint16)
    for signal in (
        prior,
        local.scores,
        domain,
        result.local,
        result.domain,
        result.prior,
        result.local_contribution,
        result.domain_contribution,
        result.prior_contribution,
        result.final_scores,
    ):
        assert signal.dtype == np.dtype(np.float64)


def test_global_prior_uses_canonical_ordered_math_fsum() -> None:
    grades = np.asarray([[0.1], [0.2], [0.3]], dtype=np.float64)

    prior = fold_global_prior(grades=grades, support_indices=(0, 1, 2))

    assert float(prior[0]).hex() == "0x1.9999999999999p-3"
    assert float(grades.mean(axis=0)[0]).hex() == "0x1.999999999999bp-3"
    assert prior.flags.c_contiguous


@pytest.mark.parametrize("support_indices", ((), (1, 0, 2), (0, 0, 2)))
def test_grade_reductions_require_nonempty_unique_canonical_support(
    support_indices: tuple[int, ...],
) -> None:
    grades = np.asarray([[0.1], [0.2], [0.3]], dtype=np.float64)
    message = "nonempty support" if not support_indices else "unique canonical order"

    with pytest.raises(ValueError, match=message):
        fold_global_prior(grades=grades, support_indices=support_indices)
    with pytest.raises(ValueError, match=message):
        domain_scores(member_indices=support_indices, grades=grades)


def test_local_reduction_uses_similarity_then_query_id_order_and_math_fsum() -> None:
    grades = np.asarray([[0.1], [1.3], [0.0]], dtype=np.float64)
    similarities = np.asarray(
        [[1.0, 0.0, 0.1], [0.0, 1.0, 0.1], [0.1, 0.1, 1.0]],
        dtype=np.float32,
    )

    evidence = local_evidence(
        target_index=2,
        support_indices=(1, 0),
        similarities=similarities,
        grades=grades,
        query_ids=("q-a", "q-b", "q-target"),
        k=2,
        prior=np.asarray([0.0], dtype=np.float64),
    )

    weight = float(np.float32(0.1))
    assert evidence.neighbor_indices == (0, 1)
    assert evidence.neighbor_similarities == (weight, weight)
    assert evidence.mass.hex() == "0x1.99999a0000000p-3"
    assert float(evidence.scores[0]).hex() == "0x1.6666666666667p-1"
    matrix_product = float(
        np.dot(
            np.asarray(evidence.neighbor_similarities, dtype=np.float64),
            grades[np.asarray(evidence.neighbor_indices), 0],
        )
        / evidence.mass
    )
    assert matrix_product.hex() == "0x1.6666666666666p-1"


def test_local_neighbors_require_strictly_positive_similarity() -> None:
    grades = np.asarray([[2.0], [0.25], [1.75], [0.0]], dtype=np.float64)
    similarities = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, -0.25],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, -0.25, 0.5, 1.0],
        ],
        dtype=np.float32,
    )

    evidence = local_evidence(
        target_index=3,
        support_indices=(2, 1, 0),
        similarities=similarities,
        grades=grades,
        query_ids=("q0", "q1", "q2", "qt"),
        k=10,
        prior=np.asarray([1.0], dtype=np.float64),
    )

    assert evidence.neighbor_indices == (2,)
    assert evidence.neighbor_similarities == (0.5,)
    assert evidence.mass == 0.5
    np.testing.assert_array_equal(evidence.scores, [1.75])


def test_no_positive_neighbor_falls_back_to_an_owned_prior_copy() -> None:
    prior = np.asarray([0.25, 1.75], dtype=np.float64)
    evidence = local_evidence(
        target_index=2,
        support_indices=(0, 1),
        similarities=np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, -0.25], [0.0, -0.25, 1.0]],
            dtype=np.float32,
        ),
        grades=np.asarray([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
        query_ids=("q0", "q1", "qt"),
        k=2,
        prior=prior,
    )

    assert evidence.neighbor_indices == ()
    assert evidence.neighbor_similarities == ()
    assert evidence.mass == 0.0
    np.testing.assert_array_equal(evidence.scores, prior)
    assert not np.shares_memory(evidence.scores, prior)


def test_float64_positive_similarity_that_rounds_to_zero_is_not_a_neighbor() -> None:
    tiny_positive = np.nextafter(0.0, 1.0)
    assert tiny_positive > 0.0

    evidence = local_evidence(
        target_index=1,
        support_indices=(0,),
        similarities=np.asarray(
            [[1.0, tiny_positive], [tiny_positive, 1.0]], dtype=np.float64
        ),
        grades=np.asarray([[2.0], [0.0]], dtype=np.float64),
        query_ids=("q0", "qt"),
        k=1,
        prior=np.asarray([0.75], dtype=np.float64),
    )

    assert evidence.mass <= 0.0
    assert evidence.neighbor_indices == ()
    np.testing.assert_array_equal(evidence.scores, [0.75])


def test_every_admitted_nonempty_neighbor_set_has_strictly_positive_mass() -> None:
    positive = np.asarray(
        [
            np.nextafter(np.float32(0.0), np.float32(1.0)),
            np.float32(0.1),
            np.float32(1.0),
        ],
        dtype=np.float32,
    )
    similarities = np.eye(4, dtype=np.float32)
    similarities[3, :3] = positive

    evidence = local_evidence(
        target_index=3,
        support_indices=(0, 1, 2),
        similarities=similarities,
        grades=np.ones((4, 1), dtype=np.float64),
        query_ids=("q0", "q1", "q2", "qt"),
        k=3,
        prior=np.asarray([0.5], dtype=np.float64),
    )

    assert evidence.neighbor_indices
    assert all(weight > 0.0 for weight in evidence.neighbor_similarities)
    assert evidence.mass == math.fsum(evidence.neighbor_similarities)
    assert evidence.mass > 0.0


def test_build_fold_evidence_reuses_prior_for_each_nominal_k() -> None:
    evidence = build_fold_evidence(
        target_index=2,
        support_indices=(0, 1),
        similarities=np.asarray(
            [[1.0, 0.0, 0.75], [0.0, 1.0, 0.25], [0.75, 0.25, 1.0]],
            dtype=np.float32,
        ),
        grades=np.asarray([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0]]),
        query_ids=("q0", "q1", "qt"),
        neighbor_counts=(2, 1),
    )

    np.testing.assert_array_equal(evidence.prior, [1.0, 1.0])
    assert tuple(evidence.local_by_k) == (2, 1)
    np.testing.assert_array_equal(evidence.local_by_k[2].scores, [1.5, 0.5])
    np.testing.assert_array_equal(evidence.local_by_k[1].scores, [2.0, 0.0])


@pytest.mark.parametrize(
    ("grades", "message"),
    (
        (np.asarray([1.0]), "finite two-dimensional matrix"),
        (np.asarray([[math.nan]]), "finite two-dimensional matrix"),
        (np.asarray([[-0.01]]), r"outside \[0,2\]"),
        (np.asarray([[2.01]]), r"outside \[0,2\]"),
    ),
)
def test_grade_matrix_rejects_wrong_shape_nonfinite_and_out_of_bounds(
    grades: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        fold_global_prior(grades=grades, support_indices=(0,))


def test_local_evidence_rejects_invalid_k_and_inventories() -> None:
    grades = np.ones((2, 1), dtype=np.float64)
    similarities = np.eye(2, dtype=np.float32)
    arguments = {
        "target_index": 1,
        "support_indices": (0,),
        "similarities": similarities,
        "grades": grades,
        "query_ids": ("q0", "qt"),
        "k": 1,
        "prior": np.asarray([1.0]),
    }

    with pytest.raises(ValueError, match="k must be positive"):
        local_evidence(**{**arguments, "k": 0})
    with pytest.raises(ValueError, match="similarity matrix"):
        local_evidence(**{**arguments, "query_ids": ("q0",)})
    with pytest.raises(ValueError, match="grade matrix"):
        local_evidence(**{**arguments, "grades": grades[:1]})


def test_local_evidence_rejects_nonfinite_target_similarity_row() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        local_evidence(
            target_index=1,
            support_indices=(0,),
            similarities=np.asarray([[1.0, 0.0], [math.nan, 1.0]]),
            grades=np.ones((2, 1), dtype=np.float64),
            query_ids=("q0", "qt"),
            k=1,
            prior=np.asarray([1.0]),
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"target_index": -1}, "target index lies outside"),
        ({"target_index": 2}, "target index lies outside"),
        ({"support_indices": (-1,)}, "support index lies outside"),
        ({"support_indices": (2,)}, "support index lies outside"),
        ({"support_indices": (0, 0)}, "support indices must be unique"),
        ({"support_indices": (0, 1)}, "must not be included in support"),
        ({"similarities": np.ones(2)}, "two-dimensional matrix"),
        ({"similarities": np.ones((2, 1))}, "similarity matrix"),
        ({"prior": np.ones(2)}, "prior length"),
        ({"query_ids": ("same", "same")}, "IDs must be unique"),
    ),
)
def test_local_evidence_rejects_unsafe_indices_shapes_and_prior_width(
    updates: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "target_index": 1,
        "support_indices": (0,),
        "similarities": np.eye(2, dtype=np.float32),
        "grades": np.ones((2, 1), dtype=np.float64),
        "query_ids": ("q0", "qt"),
        "k": 1,
        "prior": np.ones(1, dtype=np.float64),
    }

    with pytest.raises(ValueError, match=message):
        local_evidence(**{**arguments, **updates})  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("neighbor_counts", "message"),
    (
        ((), "nonempty"),
        ((0,), "positive"),
        ((1, 1), "unique"),
        ((1.5,), "integer"),
    ),
)
def test_build_fold_evidence_rejects_invalid_neighbor_counts(
    neighbor_counts: tuple[object, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_fold_evidence(
            target_index=1,
            support_indices=(0,),
            similarities=np.eye(2, dtype=np.float32),
            grades=np.ones((2, 1), dtype=np.float64),
            query_ids=("q0", "qt"),
            neighbor_counts=neighbor_counts,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("signal_name", "invalid", "message"),
    (
        ("local", np.asarray([[1.0, 1.0]]), "one-dimensional"),
        ("domain", np.asarray([math.nan, 1.0]), "non-finite"),
        ("prior", np.asarray([-0.01, 1.0]), r"outside \[0,2\]"),
        ("local", np.asarray([2.01, 1.0]), r"outside \[0,2\]"),
    ),
)
def test_combination_rejects_invalid_float64_signals(
    signal_name: str, invalid: np.ndarray, message: str
) -> None:
    arguments: dict[str, object] = {
        "local": np.asarray([1.0, 1.0]),
        "domain": np.asarray([1.0, 1.0]),
        "prior": np.asarray([1.0, 1.0]),
        "model_ids": ("m0", "m1"),
        "candidate": DenseAggregationCandidate(20, 20, 0.5, 0.1),
    }
    arguments[signal_name] = invalid

    with pytest.raises(ValueError, match=message):
        combine_and_rank(**arguments)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("candidate", "message"),
    (
        (DenseAggregationCandidate(0, 20, 0.5, 0.0), "clusters.*positive integer"),
        (DenseAggregationCandidate(20, 0, 0.5, 0.0), "neighbors.*positive integer"),
        (
            DenseAggregationCandidate(20, 20, 1, 0.0),  # type: ignore[arg-type]
            "local_weight must be a float",
        ),
        (DenseAggregationCandidate(20, 20, 2.0, 0.0), r"outside \[0,1\]"),
        (DenseAggregationCandidate(20, 20, 0.5, -0.01), r"outside \[0,1\]"),
        (DenseAggregationCandidate(20, 20, math.nan, 0.0), "non-finite"),
    ),
)
def test_combination_rejects_invalid_candidate_fields(
    candidate: DenseAggregationCandidate, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        combine_and_rank(
            local=np.asarray([2.0]),
            domain=np.asarray([0.0]),
            prior=np.asarray([0.0]),
            model_ids=("m0",),
            candidate=candidate,
        )


def test_candidate_bounds_are_rejected_before_unsafe_arithmetic() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match=r"prior_shrinkage.*outside \[0,1\]"):
            combine_and_rank(
                local=np.asarray([2.0]),
                domain=np.asarray([2.0]),
                prior=np.asarray([2.0]),
                model_ids=("m0",),
                candidate=DenseAggregationCandidate(20, 20, 0.5, -1e308),
            )


@pytest.mark.parametrize(
    ("updates", "message"),
    (
        ({"local": np.asarray(1.0)}, "one-dimensional"),
        ({"domain": np.ones(1)}, "signal lengths"),
        ({"prior": np.ones(1)}, "signal lengths"),
        ({"model_ids": ()}, "model inventory must be nonempty"),
        ({"model_ids": ("m0", "")}, "nonempty string IDs"),
        ({"model_ids": ("m0", "m0")}, "IDs must be unique"),
        ({"model_ids": ("m0", 1)}, "nonempty string IDs"),
    ),
)
def test_combination_rejects_unsafe_shapes_lengths_and_model_ids(
    updates: dict[str, object], message: str
) -> None:
    arguments: dict[str, object] = {
        "local": np.ones(2),
        "domain": np.ones(2),
        "prior": np.ones(2),
        "model_ids": ("m0", "m1"),
        "candidate": DenseAggregationCandidate(20, 20, 0.5, 0.1),
    }

    with pytest.raises(ValueError, match=message):
        combine_and_rank(**{**arguments, **updates})  # type: ignore[arg-type]


def test_combination_rejects_model_inventory_beyond_uint16_rank_capacity() -> None:
    count = int(np.iinfo(np.uint16).max) + 1
    signal = np.zeros(count, dtype=np.float64)

    with pytest.raises(ValueError, match="uint16 rank capacity"):
        combine_and_rank(
            local=signal,
            domain=signal,
            prior=signal,
            model_ids=tuple(f"m{index}" for index in range(count)),
            candidate=DenseAggregationCandidate(20, 20, 0.5, 0.1),
        )


def test_ranking_uses_final_then_prior_then_id_with_beta_zero() -> None:
    candidate = DenseAggregationCandidate(20, 20, 0.5, 0.0)
    signals = {
        "local": np.asarray([2.0, 1.0, 1.0, 1.0]),
        "domain": np.asarray([2.0, 1.0, 1.0, 1.0]),
        "prior": np.asarray([0.0, 2.0, 1.0, 1.0]),
    }

    original = combine_and_rank(
        **signals,
        model_ids=("z", "x", "b", "a"),
        candidate=candidate,
    )
    renamed = combine_and_rank(
        **signals,
        model_ids=("renamed-z", "renamed-x", "a", "b"),
        candidate=candidate,
    )

    np.testing.assert_array_equal(original.final_scores, [2.0, 1.0, 1.0, 1.0])
    np.testing.assert_array_equal(original.prior_contribution, np.zeros(4))
    assert original.model_order == (0, 1, 3, 2)
    assert renamed.model_order == (0, 1, 2, 3)
    for field in (
        "local",
        "domain",
        "prior",
        "local_contribution",
        "domain_contribution",
        "prior_contribution",
        "final_scores",
    ):
        np.testing.assert_array_equal(getattr(original, field), getattr(renamed, field))


def test_exact_inner_tie_selects_lower_nominal_grid_index() -> None:
    all_tied = [0.5] * 16
    split_tie = [0.0] * 16
    split_tie[5] = 0.75
    split_tie[9] = 0.75

    assert select_inner_candidate(all_tied) == 0
    assert select_inner_candidate(split_tie) == 5


@pytest.mark.parametrize("metric_count", (0, 15, 17))
def test_inner_selector_requires_exactly_sixteen_metrics(
    metric_count: int,
) -> None:
    with pytest.raises(ValueError, match="exactly 16 metrics"):
        select_inner_candidate([0.0] * metric_count)


@pytest.mark.parametrize("nonfinite", (math.nan, math.inf, -math.inf))
def test_inner_selector_rejects_nonfinite_metrics(nonfinite: float) -> None:
    metrics = [0.0] * 16
    metrics[7] = nonfinite

    with pytest.raises(ValueError, match="metric must be finite"):
        select_inner_candidate(metrics)
