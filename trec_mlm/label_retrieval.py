"""Zero-shot retrieval of expert labels with frozen text embedders.

Embedding backends are intentionally kept outside the scoring core.  This
module accepts precomputed query and label vectors, produces deterministic
complete rankings, and only then evaluates those rankings against qrels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import re
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.metrics import EvaluationResult, evaluate_rankings, paired_bootstrap_ci


FloatArray = NDArray[np.floating[Any]]

TASK_INSTRUCTION = (
    "Given a user question, retrieve topic labels describing the LLM experts "
    "most likely to answer it correctly."
)
ROCM_IMAGE = (
    "vllm/vllm-openai-rocm@sha256:"
    "3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28"
)


@dataclass(frozen=True, slots=True)
class LabelEncoderSpec:
    """One immutable encoder and its model-native formatting contract."""

    key: str
    model_id: str
    revision: str
    stage: int
    license: str
    runtime: str
    embedding_dimension: int
    dtype: str
    pooling: str
    query_style: str
    document_style: str


DEFAULT_LABEL_ENCODERS = (
    LabelEncoderSpec(
        key="qwen3_embedding_8b",
        model_id="Qwen/Qwen3-Embedding-8B",
        revision="14492eed37e1dbcfc6613802851c85204297f305",
        stage=1,
        license="Apache-2.0",
        runtime="vllm_rocm",
        embedding_dimension=4096,
        dtype="bfloat16",
        pooling="last_token_normalized",
        query_style="qwen_instruct",
        document_style="plain",
    ),
    LabelEncoderSpec(
        key="e5_mistral_7b_instruct",
        model_id="intfloat/e5-mistral-7b-instruct",
        revision="d62cc4417be0e3289c03dd459d1ef2d137e2554e",
        stage=1,
        license="MIT",
        runtime="transformers_rocm",
        embedding_dimension=4096,
        dtype="bfloat16",
        pooling="last_token_normalized",
        query_style="e5_instruct",
        document_style="plain",
    ),
    LabelEncoderSpec(
        key="qwen3_embedding_4b",
        model_id="Qwen/Qwen3-Embedding-4B",
        revision="5cf2132abc99cad020ac570b19d031efec650f2b",
        stage=2,
        license="Apache-2.0",
        runtime="vllm_rocm",
        embedding_dimension=2560,
        dtype="float32",
        pooling="last_token_normalized",
        query_style="qwen_instruct",
        document_style="plain",
    ),
    LabelEncoderSpec(
        key="qwen3_embedding_0_6b",
        model_id="Qwen/Qwen3-Embedding-0.6B",
        revision="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3",
        stage=2,
        license="Apache-2.0",
        runtime="vllm_rocm",
        embedding_dimension=1024,
        dtype="float32",
        pooling="last_token_normalized",
        query_style="qwen_instruct",
        document_style="plain",
    ),
    LabelEncoderSpec(
        key="jina_v5_small_retrieval",
        model_id="jinaai/jina-embeddings-v5-text-small-retrieval",
        revision="6856e76bb72982e58de0620458a4e8b3614da340",
        stage=2,
        license="CC-BY-NC-4.0",
        runtime="vllm_rocm",
        embedding_dimension=1024,
        dtype="float32",
        pooling="last_token_normalized",
        query_style="jina_query",
        document_style="jina_document",
    ),
    LabelEncoderSpec(
        key="bge_m3",
        model_id="BAAI/bge-m3",
        revision="5617a9f61b028005a4858fdac845db406aefb181",
        stage=2,
        license="MIT",
        runtime="transformers_rocm",
        embedding_dimension=1024,
        dtype="float32",
        pooling="dense_cls_normalized",
        query_style="plain",
        document_style="plain",
    ),
    LabelEncoderSpec(
        key="nomic_embed_text_v1_5",
        model_id="nomic-ai/nomic-embed-text-v1.5",
        revision="e9b6763023c676ca8431644204f50c2b100d9aab",
        stage=2,
        license="Apache-2.0",
        runtime="transformers_rocm",
        embedding_dimension=768,
        dtype="float32",
        pooling="mean_layernorm_normalized",
        query_style="nomic_search",
        document_style="nomic_search",
    ),
)

MPNET_LABEL_CONTROL = LabelEncoderSpec(
    key="mpnet_label_control",
    model_id="sentence-transformers/all-mpnet-base-v2",
    revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
    stage=0,
    license="Apache-2.0",
    runtime="sentence_transformers_cpu",
    embedding_dimension=768,
    dtype="float32",
    pooling="mean_normalized",
    query_style="plain",
    document_style="plain",
)


@dataclass(frozen=True, slots=True)
class LabelScoreResult:
    """Complete score matrix and deterministic rankings."""

    scores: NDArray[np.float32]
    rankings: dict[str, list[str]]


@dataclass(frozen=True, slots=True)
class LabelEvaluation:
    """Frozen rankings together with their post-hoc qrel evaluation."""

    rankings: dict[str, list[str]]
    evaluation: EvaluationResult


def encoder_specs_by_key() -> dict[str, LabelEncoderSpec]:
    """Return the frozen model registry keyed by command-line name."""

    return {spec.key: spec for spec in DEFAULT_LABEL_ENCODERS}


def encoder_spec_by_key(key: str) -> LabelEncoderSpec:
    """Resolve one frozen sweep encoder or the formal MPNet control."""

    if key == MPNET_LABEL_CONTROL.key:
        return MPNET_LABEL_CONTROL
    try:
        return encoder_specs_by_key()[key]
    except KeyError as error:
        raise ValueError(f"unknown label encoder: {key!r}") from error


def clean_candidate_label(model_id: str) -> str:
    """Convert a candidate ID into its frozen human-readable label text."""

    if not isinstance(model_id, str):
        raise TypeError("candidate model ID must be a string")
    cleaned = " ".join(re.sub(r"[_()]+", " ", model_id).split())
    if not cleaned:
        raise ValueError("candidate label must be non-empty after cleaning")
    return cleaned


def format_query(spec: LabelEncoderSpec, question: str) -> str:
    """Apply one encoder's frozen query-side retrieval format."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    text = question.strip()
    if spec.query_style == "plain":
        return text
    if spec.query_style == "qwen_instruct":
        return f"Instruct: {TASK_INSTRUCTION}\nQuery:{text}"
    if spec.query_style == "e5_instruct":
        return f"Instruct: {TASK_INSTRUCTION}\nQuery: {text}"
    if spec.query_style == "jina_query":
        return f"Query: {text}"
    if spec.query_style == "nomic_search":
        return f"search_query: {text}"
    raise ValueError(f"unsupported query style: {spec.query_style!r}")


def format_document(
    spec: LabelEncoderSpec,
    model_id: str,
    *,
    representation: str = "bare",
) -> str:
    """Apply one encoder's frozen document-side label format."""

    label = clean_candidate_label(model_id)
    if representation == "bare":
        document = label
    elif representation == "expert_capability":
        document = (
            f"This LLM is most likely to correctly answer questions about {label}."
        )
    else:
        raise ValueError(f"unsupported label representation: {representation!r}")
    if spec.document_style == "plain":
        return document
    if spec.document_style == "jina_document":
        return f"Document: {document}"
    if spec.document_style == "nomic_search":
        return f"search_document: {document}"
    raise ValueError(f"unsupported document style: {spec.document_style!r}")


def normalize_embedding_rows(embeddings: FloatArray) -> NDArray[np.float32]:
    """Return finite, nonzero, row-normalized float32 embeddings."""

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("embeddings must contain only finite values")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("embedding rows must not have zero norm")
    normalized = values / norms
    return np.asarray(normalized, dtype=np.float32)


def rank_label_scores(
    *,
    query_embeddings: FloatArray,
    label_embeddings: FloatArray,
    query_ids: Sequence[str],
    model_ids: Sequence[str],
) -> LabelScoreResult:
    """Rank every candidate label by cosine score, then model ID."""

    if len(set(query_ids)) != len(query_ids) or len(set(model_ids)) != len(model_ids):
        raise ValueError("query IDs and model IDs must each be unique")
    queries = normalize_embedding_rows(query_embeddings)
    labels = normalize_embedding_rows(label_embeddings)
    if queries.shape[0] != len(query_ids):
        raise ValueError("query embedding count does not match query IDs")
    if labels.shape[0] != len(model_ids):
        raise ValueError("label embedding count does not match model IDs")
    if queries.shape[1] != labels.shape[1]:
        raise ValueError("query and label embedding dimensions must match")

    scores = np.asarray(queries @ labels.T, dtype=np.float32)
    model_names = np.asarray(model_ids)
    rankings: dict[str, list[str]] = {}
    for index, query_id in enumerate(query_ids):
        order = np.lexsort((model_names, -scores[index]))
        rankings[query_id] = [model_ids[int(position)] for position in order]
    return LabelScoreResult(scores=scores, rankings=rankings)


def evaluate_label_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int | float]],
) -> LabelEvaluation:
    """Evaluate rankings after they have been fixed independently of qrels."""

    copied = {query_id: list(items) for query_id, items in rankings.items()}
    return LabelEvaluation(
        rankings=copied,
        evaluation=evaluate_rankings(copied, qrels, k=10),
    )


def advancement_gate(
    *,
    baseline: LabelEvaluation,
    candidate: LabelEvaluation,
    maximum_zero_ndcg: int,
    n_resamples: int = 10_000,
    seed: int = 2026,
) -> dict[str, Any]:
    """Apply the frozen Stage-1 nDCG confidence and coverage gate."""

    baseline_values = {
        query_id: value.ndcg_at_k
        for query_id, value in baseline.evaluation.per_query.items()
    }
    candidate_values = {
        query_id: value.ndcg_at_k
        for query_id, value in candidate.evaluation.per_query.items()
    }
    baseline_rr = {
        query_id: value.reciprocal_rank
        for query_id, value in baseline.evaluation.per_query.items()
    }
    candidate_rr = {
        query_id: value.reciprocal_rank
        for query_id, value in candidate.evaluation.per_query.items()
    }
    interval = paired_bootstrap_ci(
        baseline_values,
        candidate_values,
        n_resamples=n_resamples,
        seed=seed,
    )
    mrr_interval = paired_bootstrap_ci(
        baseline_rr,
        candidate_rr,
        n_resamples=n_resamples,
        seed=seed + 10_000,
    )
    differences = {
        query_id: candidate_values[query_id] - baseline_values[query_id]
        for query_id in baseline_values
    }
    tolerance = 1e-12
    wins = sum(value > tolerance for value in differences.values())
    losses = sum(value < -tolerance for value in differences.values())
    ties = len(differences) - wins - losses
    rr_differences = {
        query_id: candidate_rr[query_id] - baseline_rr[query_id]
        for query_id in baseline_rr
    }
    rr_wins = sum(value > tolerance for value in rr_differences.values())
    rr_losses = sum(value < -tolerance for value in rr_differences.values())
    rr_ties = len(rr_differences) - rr_wins - rr_losses
    zero_count = sum(value == 0.0 for value in candidate_values.values())
    ci_pass = interval.lower > 0.0
    coverage_pass = zero_count <= maximum_zero_ndcg
    return {
        "passed": ci_pass and coverage_pass,
        "ndcg_ci_lower_above_zero": ci_pass,
        "zero_count_within_limit": coverage_pass,
        "maximum_zero_ndcg": maximum_zero_ndcg,
        "candidate_zero_ndcg": zero_count,
        "ndcg_delta": {
            "mean_difference": interval.mean_difference,
            "lower": interval.lower,
            "upper": interval.upper,
            "confidence_level": interval.confidence_level,
            "n_resamples": interval.n_resamples,
        },
        "ndcg_win_tie_loss": {"wins": wins, "ties": ties, "losses": losses},
        "mrr_delta": {
            "mean_difference": mrr_interval.mean_difference,
            "lower": mrr_interval.lower,
            "upper": mrr_interval.upper,
            "confidence_level": mrr_interval.confidence_level,
            "n_resamples": mrr_interval.n_resamples,
        },
        "mrr_win_tie_loss": {
            "wins": rr_wins,
            "ties": rr_ties,
            "losses": rr_losses,
        },
        "zero_ndcg_transitions": {
            "baseline_zero_candidate_zero": sum(
                baseline_values[key] == 0.0 and candidate_values[key] == 0.0
                for key in baseline_values
            ),
            "baseline_zero_candidate_nonzero": sum(
                baseline_values[key] == 0.0 and candidate_values[key] != 0.0
                for key in baseline_values
            ),
            "baseline_nonzero_candidate_zero": sum(
                baseline_values[key] != 0.0 and candidate_values[key] == 0.0
                for key in baseline_values
            ),
            "baseline_nonzero_candidate_nonzero": sum(
                baseline_values[key] != 0.0 and candidate_values[key] != 0.0
                for key in baseline_values
            ),
        },
    }


__all__ = [
    "DEFAULT_LABEL_ENCODERS",
    "MPNET_LABEL_CONTROL",
    "ROCM_IMAGE",
    "TASK_INSTRUCTION",
    "LabelEncoderSpec",
    "LabelEvaluation",
    "LabelScoreResult",
    "advancement_gate",
    "clean_candidate_label",
    "encoder_specs_by_key",
    "encoder_spec_by_key",
    "evaluate_label_rankings",
    "format_document",
    "format_query",
    "normalize_embedding_rows",
    "rank_label_scores",
]
