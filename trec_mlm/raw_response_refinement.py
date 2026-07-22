"""Exact candidate-local scoring for raw historical response refinement."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CandidateRawScores:
    """Top-record evidence and its exact mean score for each query."""

    mean_scores: NDArray[np.float64]
    record_indices: NDArray[np.int32]
    record_ids: tuple[tuple[str, ...], ...]
    similarities: NDArray[np.float64]


def score_candidate_records(
    query_embeddings: NDArray[np.float32],
    record_embeddings: NDArray[np.float32],
    record_ids: Sequence[str],
    *,
    top_k: int = 10,
) -> CandidateRawScores:
    """Average each query's exact top-k candidate-local record similarities."""

    queries = np.asarray(query_embeddings, dtype=np.float32)
    records = np.asarray(record_embeddings, dtype=np.float32)
    identities = tuple(record_ids)
    if queries.ndim != 2 or not queries.shape[0] or not queries.shape[1]:
        raise ValueError("query embeddings must be a non-empty two-dimensional matrix")
    if records.ndim != 2 or not records.shape[0] or not records.shape[1]:
        raise ValueError("record embeddings must be a non-empty two-dimensional matrix")
    if not np.isfinite(queries).all() or not np.isfinite(records).all():
        raise ValueError("query and record embeddings must be finite")
    if records.shape[0] != len(identities):
        raise ValueError("record embedding and record ID counts differ")
    if queries.shape[1] != records.shape[1]:
        raise ValueError("query and record embedding dimensions differ")
    if any(not isinstance(record_id, str) or not record_id for record_id in identities):
        raise ValueError("record IDs must be non-empty strings")
    if len(set(identities)) != len(identities):
        raise ValueError("record IDs must be unique")
    if isinstance(top_k, bool) or not isinstance(top_k, int):
        raise TypeError("top_k must be an integer")
    if top_k < 1:
        raise ValueError("top_k must be positive")

    similarities = np.asarray(queries @ records.T, dtype=np.float64)
    if not np.isfinite(similarities).all():
        raise ValueError("record similarities must be finite")
    count = min(top_k, similarities.shape[1])
    orders = [
        sorted(
            range(len(identities)),
            key=lambda index: (-float(row[index]), identities[index]),
        )[:count]
        for row in similarities
    ]
    selected = np.asarray(orders, dtype=np.int32)
    selected_scores = np.take_along_axis(similarities, selected, axis=1)
    means = np.asarray(
        [math.fsum(map(float, row)) / count for row in selected_scores],
        dtype=np.float64,
    )
    selected_ids = tuple(
        tuple(identities[index] for index in row) for row in selected
    )
    return CandidateRawScores(means, selected, selected_ids, selected_scores)


def combine_hierarchical_scores(
    broad_scores: NDArray[np.floating],
    refined_scores: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Combine broad and raw-refined scores with the frozen 0.8/0.2 weights."""

    broad = np.asarray(broad_scores, dtype=np.float64)
    refined = np.asarray(refined_scores, dtype=np.float64)
    if broad.shape != refined.shape or broad.ndim != 2:
        raise ValueError("broad and refined scores must be same-shaped matrices")
    if not np.isfinite(broad).all() or not np.isfinite(refined).all():
        raise ValueError("broad and refined scores must be finite")
    return np.asarray(
        np.float64(0.8) * broad + np.float64(0.2) * refined,
        dtype=np.float64,
    )


def rank_score_matrix(
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    scores: NDArray[np.floating],
) -> dict[str, list[str]]:
    """Rank score rows descending, breaking exact ties by candidate ID."""

    queries = tuple(query_ids)
    candidates = tuple(candidate_ids)
    values = np.asarray(scores, dtype=np.float64)
    if len(set(queries)) != len(queries) or len(set(candidates)) != len(candidates):
        raise ValueError("query IDs and candidate IDs must each be unique")
    if any(not isinstance(value, str) or not value for value in queries + candidates):
        raise ValueError("query IDs and candidate IDs must be non-empty strings")
    if values.shape != (len(queries), len(candidates)):
        raise ValueError("score shape does not match query and candidate IDs")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    names = np.asarray(candidates)
    return {
        query_id: [candidates[int(index)] for index in np.lexsort((names, -values[row]))]
        for row, query_id in enumerate(queries)
    }


__all__ = [
    "CandidateRawScores",
    "combine_hierarchical_scores",
    "rank_score_matrix",
    "score_candidate_records",
]
