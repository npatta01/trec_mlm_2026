#!/usr/bin/env python3
"""Run the exploratory Issue 13 matched-support clustering preflight."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _finite_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def select_positive_neighbors(
    similarity_row: ArrayLike,
    support_indices: Sequence[int],
    query_ids: Sequence[str],
    count: int,
) -> tuple[int, ...]:
    """Select exactly ``count`` positive-cosine support rows deterministically."""

    similarities = _finite_vector(similarity_row, name="similarity_row")
    identifiers = tuple(query_ids)
    support = tuple(support_indices)
    if len(identifiers) != len(similarities):
        raise ValueError("query_ids must match the similarity inventory")
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if len(set(support)) != len(support):
        raise ValueError("support indices must be unique")
    if any(type(index) is not int or index < 0 or index >= len(similarities) for index in support):
        raise ValueError("support index lies outside the similarity inventory")
    positive = [index for index in support if float(similarities[index]) > 0.0]
    positive.sort(key=lambda index: (-float(similarities[index]), identifiers[index]))
    if len(positive) < count:
        raise ValueError("insufficient positive-cosine support for requested count")
    return tuple(positive[:count])


def mean_grades(grades: ArrayLike, indices: Sequence[int]) -> NDArray[np.float64]:
    """Return a uniform candidate-grade mean over nonempty support rows."""

    matrix = np.asarray(grades, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("grades must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("grades must contain only finite values")
    support = tuple(indices)
    if not support:
        raise ValueError("grade support must be nonempty")
    if len(set(support)) != len(support):
        raise ValueError("grade support indices must be unique")
    if any(type(index) is not int or index < 0 or index >= matrix.shape[0] for index in support):
        raise ValueError("grade support index lies outside the matrix")
    return np.mean(matrix[np.asarray(support, dtype=np.intp)], axis=0, dtype=np.float64)


def rank_candidates(
    scores: ArrayLike,
    prior: ArrayLike,
    candidate_ids: Sequence[str],
) -> tuple[str, ...]:
    """Rank by score, then fold prior, then candidate ID."""

    score_vector = _finite_vector(scores, name="scores")
    prior_vector = _finite_vector(prior, name="prior")
    candidates = tuple(candidate_ids)
    if not (len(score_vector) == len(prior_vector) == len(candidates)):
        raise ValueError("scores, prior, and candidate_ids must have equal length")
    if len(set(candidates)) != len(candidates) or any(
        type(candidate) is not str or not candidate for candidate in candidates
    ):
        raise ValueError("candidate_ids must be nonempty and unique")
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -float(score_vector[index]),
            -float(prior_vector[index]),
            candidates[index],
        ),
    )
    return tuple(candidates[index] for index in order)
