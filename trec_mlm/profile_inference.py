"""Pinned Transformers/ROCm inference for candidate semantic profiles."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib.metadata
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.label_inference import (
    ROCM_IMAGE,
    last_token_pool,
    snapshot_revision_from_path,
    validate_rocm_runtime,
)
from trec_mlm.label_retrieval import CONTEXTUAL_E5_SPEC


PROFILE_TASK_INSTRUCTION = (
    "Given a user question, retrieve historical question-and-response records "
    "from LLMs that are likely to answer it correctly."
)
PROFILE_BATCH_SIZE = 8
PROFILE_SEED = 2026
PROFILE_MAX_LENGTH = 512
PROFILE_EOS_TOKEN_ID = 2
PROFILE_TORCH_VERSION = "2.11.0+gitd0c8b1f"
PROFILE_HIP_VERSION = "7.2.53211"
PROFILE_TRANSFORMERS_VERSION = "5.12.1"
PROFILE_DEVICE_ARCHITECTURE = "gfx1151"


def classify_response_truncation(
    prefix_tokens: int, full_tokens: int, *, max_length: int = PROFILE_MAX_LENGTH
) -> str:
    """Classify whether truncation removes some or all response content."""

    if prefix_tokens < 0 or full_tokens < prefix_tokens:
        raise ValueError("token counts are inconsistent")
    content_budget = max_length - 1
    if full_tokens <= max_length:
        return "none"
    return "full" if prefix_tokens >= content_budget else "partial"


def format_profile_query(question: str) -> str:
    """Apply the frozen E5 profile-query instruction."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("target question must be a non-empty string")
    return f"Instruct: {PROFILE_TASK_INSTRUCTION}\nQuery: {question.strip()}"


def format_profile_document(question: str, response: str) -> str:
    """Format one discovery question and response exactly."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("discovery question must be a non-empty string")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("discovery response must be a non-empty string")
    return f"Question: {question.strip()}\nResponse: {response.strip()}"


def hash_formatted_inputs(texts: Sequence[str]) -> str:
    """Hash a sequence of formatted texts with unambiguous length framing."""

    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def validate_profile_runtime_options(*, batch_size: int, seed: int) -> None:
    """Reject operational changes to the preregistered inference contract."""

    if batch_size != PROFILE_BATCH_SIZE:
        raise ValueError(f"profile batch size must be {PROFILE_BATCH_SIZE}")
    if seed != PROFILE_SEED:
        raise ValueError(f"profile seed must be {PROFILE_SEED}")


def validate_profile_runtime_identity(
    torch_module: Any, *, transformers_version: str
) -> None:
    """Reject package/runtime drift from the frozen ROCm encoder contract."""

    if str(torch_module.__version__) != PROFILE_TORCH_VERSION:
        raise RuntimeError("profile Torch version differs from frozen contract")
    if str(torch_module.version.hip) != PROFILE_HIP_VERSION:
        raise RuntimeError("profile HIP version differs from frozen contract")
    if transformers_version != PROFILE_TRANSFORMERS_VERSION:
        raise RuntimeError("profile Transformers version differs from frozen contract")


def validate_profile_device_identity(properties: Any) -> None:
    """Require the exact preregistered AMD architecture, not merely any ROCm GPU."""

    if getattr(properties, "gcnArchName", None) != PROFILE_DEVICE_ARCHITECTURE:
        raise RuntimeError(
            f"profile inference requires {PROFILE_DEVICE_ARCHITECTURE}"
        )


def build_profile_token_ids(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int = PROFILE_MAX_LENGTH,
    response_prefixes: Sequence[str | None] | None = None,
) -> tuple[list[list[int]], dict[str, Any]]:
    """Right-truncate content and append exactly one terminal E5 EOS."""

    if not texts:
        raise ValueError("at least one profile input is required")
    if max_length < 2:
        raise ValueError("max_length must leave room for content and EOS")
    if tokenizer.eos_token_id != PROFILE_EOS_TOKEN_ID:
        raise ValueError("profile tokenizer must use EOS token ID 2")
    if getattr(tokenizer, "padding_side", None) != "left":
        raise ValueError("profile tokenizer must use left padding")
    if getattr(tokenizer, "truncation_side", None) != "right":
        raise ValueError("profile tokenizer must use right truncation")
    source = list(texts)
    raw_rows = tokenizer(
        source,
        add_special_tokens=True,
        padding=False,
        truncation=False,
        max_length=None,
    )["input_ids"]
    truncated_rows = tokenizer(
        source,
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=max_length - 1,
    )["input_ids"]

    def canonical(row: Sequence[int], limit: int) -> list[int]:
        values = list(row)
        while values and values[-1] == PROFILE_EOS_TOKEN_ID:
            values.pop()
        return [*values[: limit - 1], PROFILE_EOS_TOKEN_ID]

    raw_lengths = [len(canonical(row, len(row) + 1)) for row in raw_rows]
    encoded = [canonical(row, max_length) for row in truncated_rows]
    if any(len(row) > max_length or row[-1] != PROFILE_EOS_TOKEN_ID for row in encoded):
        raise AssertionError("profile EOS/truncation construction failed")
    metadata: dict[str, Any] = {
        "terminal_token_id": PROFILE_EOS_TOKEN_ID,
        "maximum_untruncated_tokens": max(raw_lengths),
        "maximum_encoded_tokens": max(map(len, encoded)),
        "truncated_input_count": sum(length > max_length for length in raw_lengths),
    }
    if response_prefixes is not None:
        if len(response_prefixes) != len(texts):
            raise ValueError("response prefix inventory differs from text inventory")
        active = [(index, value) for index, value in enumerate(response_prefixes) if value is not None]
        prefix_rows = (
            tokenizer(
                [value for _, value in active],
                add_special_tokens=True,
                padding=False,
                truncation=False,
                max_length=None,
            )["input_ids"]
            if active
            else []
        )
        prefix_lengths = [len(canonical(row, len(row) + 1)) for row in prefix_rows]
        classifications: list[str | None] = [None] * len(texts)
        for (index, _), prefix in zip(active, prefix_lengths, strict=True):
            classifications[index] = classify_response_truncation(
                prefix, raw_lengths[index], max_length=max_length
            )
        metadata["response_truncation"] = classifications
    return encoded, metadata


def normalize_persisted_embeddings(values: Any) -> NDArray[np.float32]:
    """Return finite nonzero normalized C-contiguous float32 rows."""

    matrix = np.asarray(values, dtype=np.float32, order="C")
    if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
        raise ValueError("profile embeddings must be a non-empty matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("profile embeddings must be finite")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0.0):
        raise ValueError("profile embeddings must be nonzero")
    return np.asarray(matrix / norms, dtype=np.float32, order="C")


def encode_profile_inputs(
    *,
    formatted_queries: Sequence[str],
    formatted_documents: Sequence[str],
    batch_size: int = PROFILE_BATCH_SIZE,
    seed: int = PROFILE_SEED,
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]]:
    """Encode all target queries and profile documents in one pinned runtime."""

    validate_profile_runtime_options(batch_size=batch_size, seed=seed)
    if not formatted_queries or not formatted_documents:
        raise ValueError("profile queries and documents must both be non-empty")
    combined, metadata = _encode_profile_texts(
        (*formatted_queries, *formatted_documents),
        batch_size=batch_size,
        seed=seed,
        response_prefixes=(
            *(None for _ in formatted_queries),
            *(
                text.split("\nResponse: ", maxsplit=1)[0] + "\nResponse: "
                for text in formatted_documents
            ),
        ),
    )
    count = len(formatted_queries)
    return combined[:count], combined[count:], metadata


def _encode_profile_texts(
    texts: Sequence[str],
    *,
    batch_size: int,
    seed: int,
    response_prefixes: Sequence[str | None] | None = None,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils.hub import cached_file
    except ImportError as error:  # pragma: no cover - container integration
        raise RuntimeError("profile inference requires torch and transformers") from error

    transformers_version = importlib.metadata.version("transformers")
    validate_profile_runtime_identity(
        torch, transformers_version=transformers_version
    )
    properties = validate_rocm_runtime(torch)
    validate_profile_device_identity(properties)
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    spec = CONTEXTUAL_E5_SPEC
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        padding_side="left",
        truncation_side="right",
        trust_remote_code=False,
    )
    tokenizer_path = cached_file(
        spec.model_id,
        "tokenizer_config.json",
        revision=spec.revision,
        local_files_only=True,
    )
    if tokenizer_path is None:
        raise RuntimeError("could not resolve pinned E5 tokenizer snapshot")
    tokenizer_revision = snapshot_revision_from_path(tokenizer_path)
    if tokenizer_revision != spec.revision:
        raise RuntimeError("resolved tokenizer revision differs from profile contract")
    model = AutoModel.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    ).to("cuda")
    model_revision = getattr(model.config, "_commit_hash", None)
    if model_revision != spec.revision:
        raise RuntimeError("resolved model revision differs from profile contract")
    model.eval()
    batches: list[NDArray[np.float32]] = []
    maximum_tokens = 0
    maximum_untruncated = 0
    truncated_count = 0
    response_truncation: list[str] = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            batch_prefixes = (
                response_prefixes[start : start + batch_size]
                if response_prefixes is not None
                else None
            )
            sequences, token_metadata = build_profile_token_ids(
                tokenizer, batch, response_prefixes=batch_prefixes
            )
            padded = tokenizer.pad(
                {
                    "input_ids": sequences,
                    "attention_mask": [[1] * len(row) for row in sequences],
                },
                padding=True,
                return_tensors="pt",
            )
            maximum_tokens = max(
                maximum_tokens, int(padded["attention_mask"].sum(dim=1).max().item())
            )
            maximum_untruncated = max(
                maximum_untruncated, token_metadata["maximum_untruncated_tokens"]
            )
            truncated_count += token_metadata["truncated_input_count"]
            response_truncation.extend(
                item
                for item in token_metadata.get("response_truncation", ())
                if item is not None
            )
            padded = {name: value.to("cuda") for name, value in padded.items()}
            output = model(**padded)
            pooled = functional.normalize(
                last_token_pool(output.last_hidden_state, padded["attention_mask"]),
                p=2,
                dim=1,
            )
            batches.append(pooled.float().cpu().numpy())
    matrix = normalize_persisted_embeddings(np.concatenate(batches, axis=0))
    if matrix.shape != (len(texts), spec.embedding_dimension):
        raise ValueError("profile embedding shape differs from frozen contract")
    return matrix, {
        "backend": "transformers_rocm",
        "model_id": spec.model_id,
        "requested_revision": spec.revision,
        "resolved_model_revision": model_revision,
        "resolved_tokenizer_revision": tokenizer_revision,
        "license": spec.license,
        "required_container_image": ROCM_IMAGE,
        "dtype": spec.dtype,
        "persisted_dtype": "float32",
        "pooling": spec.pooling,
        "embedding_dimension": spec.embedding_dimension,
        "input_count": len(texts),
        "input_sha256": hash_formatted_inputs(texts),
        "batch_size": batch_size,
        "seed": seed,
        "padding_side": "left",
        "truncation_side": "right",
        "terminal_token_id": PROFILE_EOS_TOKEN_ID,
        "terminal_token": tokenizer.eos_token,
        "max_length": PROFILE_MAX_LENGTH,
        "maximum_observed_tokens": maximum_tokens,
        "maximum_untruncated_tokens": maximum_untruncated,
        "truncated_input_count": truncated_count,
        "document_response_truncation": response_truncation,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "device": "rocm_hip",
        "device_properties": str(properties),
        "torch_version": str(torch.__version__),
        "hip_version": str(torch.version.hip),
        "transformers_version": transformers_version,
        "encoding_seconds": time.perf_counter() - started,
    }


__all__ = [
    "PROFILE_BATCH_SIZE",
    "PROFILE_DEVICE_ARCHITECTURE",
    "PROFILE_EOS_TOKEN_ID",
    "PROFILE_MAX_LENGTH",
    "PROFILE_SEED",
    "PROFILE_TASK_INSTRUCTION",
    "build_profile_token_ids",
    "classify_response_truncation",
    "encode_profile_inputs",
    "format_profile_document",
    "format_profile_query",
    "hash_formatted_inputs",
    "normalize_persisted_embeddings",
    "validate_profile_runtime_options",
    "validate_profile_runtime_identity",
    "validate_profile_device_identity",
]
