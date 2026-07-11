"""Frozen dense query embeddings for leakage-safe query-neighbor retrieval.

The pretrained encoders in this module are inference-only.  They never see
qrels and never fit corpus statistics, so one normalized embedding can be
cached per query and encoder before the nested label splits are evaluated.
Only training-query qrels contribute expertise evidence or tie-breaking
priors.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import math
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True, slots=True)
class EncoderSpec:
    """Immutable pretrained encoder configuration."""

    key: str
    model_id: str
    revision: str
    query_prefix: str = ""
    license: str = ""


# Advisor-approved CPU-first grid, fixed before inspecting dense results.
DEFAULT_ENCODERS = (
    EncoderSpec(
        key="minilm_l6",
        model_id="sentence-transformers/all-MiniLM-L6-v2",
        revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
        license="Apache-2.0",
    ),
    EncoderSpec(
        key="e5_small_v2",
        model_id="intfloat/e5-small-v2",
        revision="ffb93f3bd4047442299a41ebb6fa998a38507c52",
        query_prefix="query: ",
        license="MIT",
    ),
    EncoderSpec(
        key="bge_small_en_v1_5",
        model_id="BAAI/bge-small-en-v1.5",
        revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
        license="MIT",
    ),
    EncoderSpec(
        key="mpnet_base_v2",
        model_id="sentence-transformers/all-mpnet-base-v2",
        revision="e8c3b32edf5434bc2275fc9bab85f82640a19130",
        license="Apache-2.0",
    ),
)


@dataclass(frozen=True, slots=True)
class DenseTargetRanking:
    """One dense query-kNN prediction and its fold-local diagnostics."""

    model_order: tuple[int, ...]
    scores: FloatArray
    global_prior: FloatArray
    neighbor_indices: tuple[int, ...]
    neighbor_similarities: tuple[float, ...]
    neighbor_mass: float


def normalize_embeddings(embeddings: FloatArray) -> NDArray[np.float32]:
    """Return finite row-wise L2-normalized float32 embeddings.

    A zero row remains zero.  Its cosine similarities are all zero, causing
    the ranker to use the training-only global-prior fallback.
    """

    array = np.asarray(embeddings, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("embeddings must be a two-dimensional array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError("embeddings must have at least one row and one column")
    if not np.isfinite(array).all():
        raise ValueError("embeddings must contain only finite values")

    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = np.zeros_like(array, dtype=np.float32)
    np.divide(array, norms, out=normalized, where=norms > 0.0)
    return normalized


def cosine_similarity_matrix(embeddings: FloatArray) -> NDArray[np.float32]:
    """Return the cosine matrix for row embeddings after L2 normalization."""

    normalized = normalize_embeddings(embeddings)
    similarities = np.asarray(normalized @ normalized.T, dtype=np.float32)
    # Bound tiny matrix-multiplication roundoff while preserving zero rows.
    np.clip(similarities, -1.0, 1.0, out=similarities)
    return similarities


def validate_similarity_matrix(
    similarities: FloatArray,
    *,
    expected_queries: int,
) -> NDArray[np.float32]:
    """Validate and return one finite, symmetric dense cosine matrix."""

    matrix = np.asarray(similarities, dtype=np.float32)
    if matrix.shape != (expected_queries, expected_queries):
        raise ValueError(
            "similarity matrix shape must be "
            f"{(expected_queries, expected_queries)!r}, got {matrix.shape!r}"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("similarity matrix must contain only finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-5, atol=1e-6):
        raise ValueError("similarity matrix must be symmetric")
    return matrix


def rank_target_from_similarity(
    *,
    target_index: int,
    additionally_excluded_indices: Sequence[int],
    similarities: FloatArray,
    grades: FloatArray,
    query_ids: Sequence[str],
    model_ids: Sequence[str],
    k: int,
) -> DenseTargetRanking:
    """Rank every model from positive-cosine training-query neighbors.

    ``target_index`` is always excluded from training.  Additional exclusions
    implement the inner fold of nested leave-one-out validation.  Model-score
    ties use the mean grade over the exact same training rows, then model ID.
    """

    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer")

    grade_matrix = np.asarray(grades, dtype=np.float64)
    query_count, candidate_count = grade_matrix.shape
    matrix = validate_similarity_matrix(
        similarities,
        expected_queries=query_count,
    )
    if len(query_ids) != query_count:
        raise ValueError("query_ids length does not match grades")
    if len(model_ids) != candidate_count:
        raise ValueError("model_ids length does not match grades")
    if not 0 <= target_index < query_count:
        raise IndexError("target_index is out of range")
    if not np.isfinite(grade_matrix).all():
        raise ValueError("grades must contain only finite values")

    excluded = {target_index}
    for index in additionally_excluded_indices:
        if not 0 <= index < query_count:
            raise IndexError("excluded query index is out of range")
        excluded.add(index)
    training_indices = np.asarray(
        [index for index in range(query_count) if index not in excluded],
        dtype=np.int64,
    )
    if training_indices.size == 0:
        raise ValueError("at least one training query is required")

    global_prior = grade_matrix[training_indices].mean(axis=0, dtype=np.float64)
    target_similarities = matrix[target_index, training_indices].astype(
        np.float64,
        copy=False,
    )
    positive_mask = target_similarities > 0.0
    positive_indices = training_indices[positive_mask]
    positive_similarities = target_similarities[positive_mask]
    if positive_indices.size:
        query_ties = np.asarray([query_ids[index] for index in positive_indices])
        order = np.lexsort((query_ties, -positive_similarities))
        selected = order[:k]
        neighbor_indices = positive_indices[selected]
        neighbor_similarities = positive_similarities[selected]
    else:
        neighbor_indices = np.asarray([], dtype=np.int64)
        neighbor_similarities = np.asarray([], dtype=np.float64)

    neighbor_mass = float(math.fsum(float(value) for value in neighbor_similarities))
    if neighbor_indices.size == 0 or neighbor_mass <= 0.0:
        scores = global_prior.copy()
    else:
        scores = (
            neighbor_similarities @ grade_matrix[neighbor_indices]
        ) / neighbor_mass

    model_names = np.asarray(model_ids)
    model_order = np.lexsort((model_names, -global_prior, -scores))
    return DenseTargetRanking(
        model_order=tuple(int(index) for index in model_order),
        scores=np.asarray(scores, dtype=np.float64),
        global_prior=np.asarray(global_prior, dtype=np.float64),
        neighbor_indices=tuple(int(index) for index in neighbor_indices),
        neighbor_similarities=tuple(float(value) for value in neighbor_similarities),
        neighbor_mass=neighbor_mass,
    )


def encode_frozen_queries(
    *,
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    specs: Sequence[EncoderSpec],
    device: str = "cpu",
    batch_size: int = 32,
    torch_threads: int | None = None,
    show_progress_bar: bool = False,
) -> tuple[dict[str, NDArray[np.float32]], dict[str, dict[str, Any]]]:
    """Encode each query once with pinned, eval-mode SentenceTransformers.

    ``sentence-transformers`` is an optional project dependency.  Importing
    this module and running matrix-only tests does not require PyTorch.
    """

    if len(query_ids) != len(query_texts):
        raise ValueError("query_ids and query_texts must have equal length")
    if not query_ids:
        raise ValueError("at least one query is required")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    _validate_encoder_specs(specs)

    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - exercised without dense extra
        raise RuntimeError(
            "dense inference requires the optional dependency group; run "
            "`uv sync --extra dense` or prefix the command with "
            "`uv run --extra dense`"
        ) from error

    if torch_threads is not None:
        if (
            isinstance(torch_threads, bool)
            or not isinstance(torch_threads, int)
            or torch_threads < 1
        ):
            raise ValueError("torch_threads must be a positive integer or None")
        torch.set_num_threads(torch_threads)
    torch.use_deterministic_algorithms(True)

    embeddings_by_encoder: dict[str, NDArray[np.float32]] = {}
    metadata_by_encoder: dict[str, dict[str, Any]] = {}
    for spec in specs:
        prepared_texts = [spec.query_prefix + text for text in query_texts]
        started = time.perf_counter()
        model = SentenceTransformer(
            spec.model_id,
            revision=spec.revision,
            device=device,
            trust_remote_code=False,
        )
        model.eval()
        token_lengths = [
            len(
                model.tokenizer.encode(
                    text,
                    add_special_tokens=True,
                    truncation=False,
                )
            )
            for text in prepared_texts
        ]
        with torch.inference_mode():
            encoded = model.encode(
                prepared_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        embeddings = normalize_embeddings(encoded)
        elapsed_seconds = time.perf_counter() - started
        maximum_tokens = int(model.max_seq_length)
        metadata_by_encoder[spec.key] = {
            "key": spec.key,
            "model_id": spec.model_id,
            "requested_revision": spec.revision,
            "resolved_revision": spec.revision,
            "license": spec.license,
            "query_prefix": spec.query_prefix,
            "trust_remote_code": False,
            "device": str(device),
            "dtype": str(embeddings.dtype),
            "batch_size": batch_size,
            "torch_threads": int(torch.get_num_threads()),
            "deterministic_algorithms": True,
            "eval_mode": not model.training,
            "normalized": True,
            "pooling": _pooling_description(model),
            "embedding_dimension": int(embeddings.shape[1]),
            "parameter_count": int(
                sum(parameter.numel() for parameter in model.parameters())
            ),
            "max_sequence_length": maximum_tokens,
            "maximum_observed_tokens": max(token_lengths),
            "truncated_query_count": sum(
                length > maximum_tokens for length in token_lengths
            ),
            "query_count": len(query_ids),
            "input_sha256": _query_input_sha256(
                query_ids,
                query_texts,
                prefix=spec.query_prefix,
            ),
            "embedding_sha256": array_sha256(embeddings),
            "encoding_seconds": elapsed_seconds,
            "sentence_transformers_version": _package_version("sentence-transformers"),
            "transformers_version": _package_version("transformers"),
            "torch_version": str(torch.__version__),
            "numpy_version": str(np.__version__),
        }
        embeddings_by_encoder[spec.key] = embeddings
        del model

    return embeddings_by_encoder, metadata_by_encoder


def array_sha256(array: FloatArray) -> str:
    """Hash array shape, dtype, and contiguous bytes deterministically."""

    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def encoder_specs_by_key(
    specs: Sequence[EncoderSpec] = DEFAULT_ENCODERS,
) -> Mapping[str, EncoderSpec]:
    """Return encoder specs keyed by their short CLI-safe names."""

    _validate_encoder_specs(specs)
    return {spec.key: spec for spec in specs}


def _validate_encoder_specs(specs: Sequence[EncoderSpec]) -> None:
    if not specs:
        raise ValueError("at least one encoder is required")
    keys = [spec.key for spec in specs]
    if len(set(keys)) != len(keys):
        raise ValueError("encoder keys must be unique")
    for spec in specs:
        if not spec.key or not spec.model_id or not spec.revision:
            raise ValueError("encoder key, model_id, and revision must be non-empty")
        if len(spec.revision) != 40 or any(
            character not in "0123456789abcdef" for character in spec.revision
        ):
            raise ValueError(
                f"encoder {spec.key!r} revision must be a 40-character lowercase SHA"
            )


def _pooling_description(model: Any) -> str:
    for module in model._modules.values():
        if module.__class__.__name__ == "Pooling":
            if isinstance(getattr(module, "pooling_mode", None), str):
                return str(module.pooling_mode)
            modes = [
                name.removeprefix("pooling_mode_")
                for name, enabled in vars(module).items()
                if name.startswith("pooling_mode_") and enabled is True
            ]
            return "+".join(sorted(modes)) or "sentence_transformers_pooling"
    return "model_defined"


def _query_input_sha256(
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    *,
    prefix: str,
) -> str:
    digest = hashlib.sha256()
    for query_id, query_text in zip(query_ids, query_texts, strict=True):
        digest.update(query_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update((prefix + query_text).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _package_version(package: str) -> str | None:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return None
