"""Qrel-free contextual E5 retrieval of expert topic labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating[Any]]

TASK_INSTRUCTION = (
    "Given a user question, retrieve topic labels describing the LLM experts "
    "most likely to answer it correctly."
)


@dataclass(frozen=True, slots=True)
class ContextualE5Spec:
    """The immutable model and representation contract selected by the study."""

    key: str = "e5_mistral_7b_contextual"
    model_id: str = "intfloat/e5-mistral-7b-instruct"
    revision: str = "d62cc4417be0e3289c03dd459d1ef2d137e2554e"
    license: str = "MIT"
    runtime: str = "transformers_rocm"
    dtype: str = "bfloat16"
    embedding_dimension: int = 4096
    pooling: str = "last_token_normalized"
    max_length: int = 512


CONTEXTUAL_E5_SPEC = ContextualE5Spec()


@dataclass(frozen=True, slots=True)
class LabelScoreResult:
    """A complete cosine-score matrix and its deterministic rankings."""

    scores: NDArray[np.float32]
    rankings: dict[str, list[str]]


def clean_candidate_label(candidate_id: str) -> str:
    """Convert a candidate ID to the frozen human-readable label text."""

    if not isinstance(candidate_id, str):
        raise TypeError("candidate ID must be a string")
    cleaned = " ".join(re.sub(r"[_()]+", " ", candidate_id).split())
    if not cleaned:
        raise ValueError("candidate label must be non-empty after cleaning")
    return cleaned


def format_query(question: str) -> str:
    """Apply E5-Mistral's query-only instruction format."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    return f"Instruct: {TASK_INSTRUCTION}\nQuery: {question.strip()}"


def format_document(candidate_id: str) -> str:
    """Represent one expert label with the selected capability sentence."""

    label = clean_candidate_label(candidate_id)
    return f"This LLM is most likely to correctly answer questions about {label}."


def normalize_embedding_rows(embeddings: FloatArray) -> NDArray[np.float32]:
    """Return finite, nonzero, unit-normalized float32 embedding rows."""

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("embedding rows must not have zero norm")
    return np.asarray(values / norms, dtype=np.float32)


def rank_label_scores(
    *,
    query_embeddings: FloatArray,
    label_embeddings: FloatArray,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> LabelScoreResult:
    """Rank every candidate by cosine score, breaking exact ties by ID."""

    if len(set(query_ids)) != len(query_ids) or len(set(candidate_ids)) != len(
        candidate_ids
    ):
        raise ValueError("query IDs and candidate IDs must each be unique")
    queries = normalize_embedding_rows(query_embeddings)
    labels = normalize_embedding_rows(label_embeddings)
    if queries.shape[0] != len(query_ids):
        raise ValueError("query embedding count does not match query IDs")
    if labels.shape[0] != len(candidate_ids):
        raise ValueError("label embedding count does not match candidate IDs")
    if queries.shape[1] != labels.shape[1]:
        raise ValueError("query and label embedding dimensions must match")

    scores = np.asarray(queries @ labels.T, dtype=np.float32)
    return rank_score_matrix(
        scores=scores,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
    )


def rank_score_matrix(
    *,
    scores: FloatArray,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> LabelScoreResult:
    """Rank a persisted score matrix without repeating its dot products."""

    values = np.asarray(scores, dtype=np.float32)
    if values.shape != (len(query_ids), len(candidate_ids)):
        raise ValueError("score shape does not match query and candidate IDs")
    if not np.isfinite(values).all():
        raise ValueError("scores must contain only finite values")
    names = np.asarray(candidate_ids)
    rankings: dict[str, list[str]] = {}
    for index, query_id in enumerate(query_ids):
        order = np.lexsort((names, -values[index]))
        rankings[query_id] = [candidate_ids[int(position)] for position in order]
    return LabelScoreResult(scores=values, rankings=rankings)


__all__ = [
    "CONTEXTUAL_E5_SPEC",
    "TASK_INSTRUCTION",
    "ContextualE5Spec",
    "LabelScoreResult",
    "clean_candidate_label",
    "format_document",
    "format_query",
    "normalize_embedding_rows",
    "rank_label_scores",
    "rank_score_matrix",
]
