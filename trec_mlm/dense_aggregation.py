from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class DenseAggregationCandidate:
    clusters: int
    neighbors: int
    local_weight: float
    prior_shrinkage: float


DENSE_AGGREGATION_GRID: tuple[DenseAggregationCandidate, ...] = tuple(
    DenseAggregationCandidate(c, k, local_weight, prior_shrinkage)
    for c in (20, 40)
    for k in (20, 50)
    for local_weight in (0.5, 0.75)
    for prior_shrinkage in (0.0, 0.1)
)


@dataclass(frozen=True, slots=True)
class LocalEvidence:
    scores: NDArray[np.float64]
    neighbor_indices: tuple[int, ...]
    neighbor_similarities: tuple[float, ...]
    mass: float


@dataclass(frozen=True, slots=True)
class FoldEvidence:
    prior: NDArray[np.float64]
    local_by_k: Mapping[int, LocalEvidence]


@dataclass(frozen=True, slots=True)
class AggregationPrediction:
    local: NDArray[np.float64]
    domain: NDArray[np.float64]
    prior: NDArray[np.float64]
    local_contribution: NDArray[np.float64]
    domain_contribution: NDArray[np.float64]
    prior_contribution: NDArray[np.float64]
    final_scores: NDArray[np.float64]
    model_order: tuple[int, ...]
    ranks: NDArray[np.uint16]


def _validate_signal(value: NDArray[Any], name: str) -> NDArray[np.float64]:
    converted = np.asarray(value, dtype=np.float64)
    if converted.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    result = np.ascontiguousarray(converted, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} contains a non-finite value")
    if np.any(result < 0.0) or np.any(result > 2.0):
        raise ValueError(f"{name} lies outside [0,2]")
    return result


def _validate_grades(value: NDArray[Any]) -> NDArray[np.float64]:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError("grades must be a finite two-dimensional matrix")
    if np.any(result < 0.0) or np.any(result > 2.0):
        raise ValueError("grades lie outside [0,2]")
    return result


def _as_index(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _validate_indices(
    values: Sequence[int],
    *,
    size: int,
    name: str,
    require_canonical: bool,
    require_nonempty: bool,
) -> tuple[int, ...]:
    indices = tuple(_as_index(value, name=f"{name} index") for value in values)
    if require_nonempty and not indices:
        raise ValueError("grade aggregation requires nonempty support")
    if len(indices) != len(set(indices)):
        if require_canonical:
            raise ValueError("grade support indices must be unique canonical order")
        raise ValueError(f"{name} indices must be unique")
    if require_canonical and indices != tuple(sorted(indices)):
        raise ValueError("grade support indices must be unique canonical order")
    if any(index < 0 or index >= size for index in indices):
        raise ValueError(f"{name} index lies outside the query inventory")
    return indices


def _ordered_mean(
    grades: NDArray[np.float64], row_indices: Sequence[int]
) -> NDArray[np.float64]:
    indices = _validate_indices(
        row_indices,
        size=grades.shape[0],
        name="grade support",
        require_canonical=True,
        require_nonempty=True,
    )
    result = np.empty(grades.shape[1], dtype=np.float64)
    for model_index in range(grades.shape[1]):
        result[model_index] = math.fsum(
            float(grades[row_index, model_index]) for row_index in indices
        ) / len(indices)
    return result


def fold_global_prior(
    *, grades: NDArray[Any], support_indices: Sequence[int]
) -> NDArray[np.float64]:
    matrix = _validate_grades(grades)
    return _validate_signal(_ordered_mean(matrix, support_indices), "prior")


def _validate_query_inventory(query_ids: Sequence[str], size: int) -> tuple[str, ...]:
    inventory = tuple(query_ids)
    if len(inventory) != size:
        raise ValueError("query inventory does not match similarity matrix")
    if any(type(query_id) is not str or not query_id for query_id in inventory):
        raise ValueError("query inventory must contain nonempty string IDs")
    if len(inventory) != len(set(inventory)):
        raise ValueError("query inventory IDs must be unique")
    return inventory


def local_evidence(
    *,
    target_index: int,
    support_indices: Sequence[int],
    similarities: NDArray[Any],
    grades: NDArray[Any],
    query_ids: Sequence[str],
    k: int,
    prior: NDArray[Any],
) -> LocalEvidence:
    matrix = _validate_grades(grades)
    neighbor_count = _as_index(k, name="k")
    if neighbor_count < 1:
        raise ValueError("k must be positive")

    similarity_matrix = np.asarray(similarities)
    if similarity_matrix.ndim != 2:
        raise ValueError("similarities must be a two-dimensional matrix")
    inventory = _validate_query_inventory(query_ids, similarity_matrix.shape[0])
    if similarity_matrix.shape[1] != len(inventory):
        raise ValueError("query inventory does not match similarity matrix")
    if matrix.shape[0] != len(inventory):
        raise ValueError("query inventory does not match grade matrix")

    target = _as_index(target_index, name="target_index")
    if target < 0 or target >= len(inventory):
        raise ValueError("target index lies outside the query inventory")
    support = _validate_indices(
        support_indices,
        size=len(inventory),
        name="support",
        require_canonical=False,
        require_nonempty=False,
    )
    if target in support:
        raise ValueError("target index must not be included in support")

    prior64 = _validate_signal(np.asarray(prior, dtype=np.float64), "prior")
    if len(prior64) != matrix.shape[1]:
        raise ValueError("prior length does not match grade model inventory")

    similarity_row = np.asarray(similarity_matrix[target], dtype=np.float32)
    if not np.isfinite(similarity_row).all():
        raise ValueError("target similarity row contains a non-finite value")
    positive = [
        index for index in support if float(similarity_row[index]) > 0.0
    ]
    positive.sort(
        key=lambda index: (-float(similarity_row[index]), inventory[index])
    )
    neighbors = tuple(positive[:neighbor_count])
    weights = tuple(float(similarity_row[index]) for index in neighbors)
    mass = math.fsum(weights)
    if not neighbors or mass <= 0.0:
        scores = prior64.copy()
    else:
        scores = np.empty(matrix.shape[1], dtype=np.float64)
        for model_index in range(matrix.shape[1]):
            scores[model_index] = math.fsum(
                weight * float(matrix[index, model_index])
                for index, weight in zip(neighbors, weights, strict=True)
            ) / mass
    return LocalEvidence(_validate_signal(scores, "local"), neighbors, weights, mass)


def build_fold_evidence(
    *,
    target_index: int,
    support_indices: Sequence[int],
    similarities: NDArray[Any],
    grades: NDArray[Any],
    query_ids: Sequence[str],
    neighbor_counts: Sequence[int],
) -> FoldEvidence:
    support = tuple(support_indices)
    inventory = tuple(query_ids)
    counts = tuple(_as_index(value, name="neighbor count") for value in neighbor_counts)
    if not counts:
        raise ValueError("neighbor counts must be nonempty")
    if len(counts) != len(set(counts)):
        raise ValueError("neighbor counts must be unique")
    if any(value < 1 for value in counts):
        raise ValueError("neighbor counts must be positive")

    prior = fold_global_prior(grades=grades, support_indices=support)
    local_by_k = {
        count: local_evidence(
            target_index=target_index,
            support_indices=support,
            similarities=similarities,
            grades=grades,
            query_ids=inventory,
            k=count,
            prior=prior,
        )
        for count in counts
    }
    return FoldEvidence(prior=prior, local_by_k=local_by_k)


def domain_scores(
    *, member_indices: Sequence[int], grades: NDArray[Any]
) -> NDArray[np.float64]:
    matrix = _validate_grades(grades)
    return _validate_signal(_ordered_mean(matrix, member_indices), "domain")


def _validate_candidate_structure(candidate: DenseAggregationCandidate) -> None:
    if not isinstance(candidate, DenseAggregationCandidate):
        raise ValueError("candidate must be a DenseAggregationCandidate")
    for name in ("clusters", "neighbors"):
        value = getattr(candidate, name)
        if type(value) is not int or value < 1:
            raise ValueError(f"candidate {name} must be a positive integer")
    for name in ("local_weight", "prior_shrinkage"):
        value = getattr(candidate, name)
        if type(value) is not float:
            raise ValueError(f"candidate {name} must be a float")
        if not math.isfinite(value):
            raise ValueError(f"candidate {name} contains a non-finite value")


def _validate_candidate_bounds(candidate: DenseAggregationCandidate) -> None:
    if candidate.local_weight < 0.0 or candidate.local_weight > 1.0:
        raise ValueError("candidate local_weight lies outside [0,1]")
    if candidate.prior_shrinkage < 0.0 or candidate.prior_shrinkage > 1.0:
        raise ValueError("candidate prior_shrinkage lies outside [0,1]")


def combine_and_rank(
    *,
    local: NDArray[Any],
    domain: NDArray[Any],
    prior: NDArray[Any],
    model_ids: Sequence[str],
    candidate: DenseAggregationCandidate,
) -> AggregationPrediction:
    _validate_candidate_structure(candidate)
    _validate_candidate_bounds(candidate)
    local64 = _validate_signal(np.asarray(local, dtype=np.float64), "local")
    domain64 = _validate_signal(np.asarray(domain, dtype=np.float64), "domain")
    prior64 = _validate_signal(np.asarray(prior, dtype=np.float64), "prior")

    models = tuple(model_ids)
    if not models:
        raise ValueError("model inventory must be nonempty")
    if any(type(model_id) is not str or not model_id for model_id in models):
        raise ValueError("model inventory must contain nonempty string IDs")
    if len(models) != len(set(models)):
        raise ValueError("model inventory IDs must be unique")
    if not (len(local64) == len(domain64) == len(prior64) == len(models)):
        raise ValueError("signal lengths must match the model inventory")
    if len(models) > np.iinfo(np.uint16).max:
        raise ValueError("model inventory exceeds uint16 rank capacity")

    local_part = (
        (1.0 - candidate.prior_shrinkage)
        * candidate.local_weight
        * local64
    )
    domain_part = (
        (1.0 - candidate.prior_shrinkage)
        * (1.0 - candidate.local_weight)
        * domain64
    )
    prior_part = candidate.prior_shrinkage * prior64
    final = np.asarray(
        [
            math.fsum(
                (
                    float(local_part[index]),
                    float(domain_part[index]),
                    float(prior_part[index]),
                )
            )
            for index in range(len(models))
        ],
        dtype=np.float64,
    )
    _validate_signal(final, "final")

    order = tuple(
        sorted(
            range(len(models)),
            key=lambda index: (
                -float(final[index]),
                -float(prior64[index]),
                models[index],
            ),
        )
    )
    ranks = np.empty(len(models), dtype=np.uint16)
    for rank, model_index in enumerate(order, start=1):
        ranks[model_index] = rank
    return AggregationPrediction(
        local64,
        domain64,
        prior64,
        local_part,
        domain_part,
        prior_part,
        final,
        order,
        ranks,
    )


def select_inner_candidate(ndcg_by_candidate: Sequence[float]) -> int:
    if len(ndcg_by_candidate) != len(DENSE_AGGREGATION_GRID):
        raise ValueError("inner selector requires exactly 16 metrics")
    if not all(math.isfinite(value) for value in ndcg_by_candidate):
        raise ValueError("inner selector metric must be finite")
    return min(
        range(len(DENSE_AGGREGATION_GRID)),
        key=lambda index: (-ndcg_by_candidate[index], index),
    )
