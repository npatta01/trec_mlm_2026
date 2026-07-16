"""Pinned Transformers/ROCm inference for contextual E5 label retrieval."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib.metadata
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.label_retrieval import (
    CONTEXTUAL_E5_SPEC,
    normalize_embedding_rows,
)


ROCM_IMAGE = (
    "vllm/vllm-openai-rocm@sha256:"
    "3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28"
)


def validate_rocm_runtime(torch_module: Any) -> Any:
    """Verify that the active Torch device is an AMD HIP runtime."""

    if not torch_module.cuda.is_available():
        raise RuntimeError("E5 ROCm inference requires an available HIP device")
    if not getattr(torch_module.version, "hip", None):
        raise RuntimeError("E5 ROCm inference requires a Torch HIP build")
    properties = torch_module.cuda.get_device_properties(0)
    architecture = getattr(properties, "gcnArchName", "")
    if not isinstance(architecture, str) or not architecture.startswith("gfx"):
        raise RuntimeError("E5 ROCm inference requires an AMD GCN device")
    return properties


def snapshot_revision_from_path(path: str | Path) -> str:
    """Recover the immutable Hub commit from a resolved snapshot file path."""

    parts = Path(path).parts
    try:
        revision = parts[parts.index("snapshots") + 1]
    except (ValueError, IndexError) as error:
        raise ValueError("resolved file is not inside a Hub snapshot") from error
    if len(revision) != 40 or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("resolved Hub snapshot revision is not a commit hash")
    return revision


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Pool the last non-padding token for left- or right-padded batches."""

    if last_hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden states and attention mask have invalid dimensions")
    if last_hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("attention mask shape must match hidden-state tokens")
    left_padded = bool(
        (attention_mask[:, -1].sum() == attention_mask.shape[0]).item()
    )
    if left_padded:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    if bool((sequence_lengths < 0).any().item()):
        raise ValueError("every input must contain at least one unmasked token")
    import torch

    return last_hidden_states[
        torch.arange(last_hidden_states.shape[0], device=last_hidden_states.device),
        sequence_lengths,
    ]


def build_e5_token_ids(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
) -> tuple[list[list[int]], dict[str, int]]:
    """Reproduce legacy E5 tokenization with exactly one terminal EOS token."""

    if tokenizer.eos_token_id is None:
        raise ValueError("E5 tokenizer must define an EOS token")
    if max_length <= 1:
        raise ValueError("max_length must leave room for content and EOS")
    terminal_token_id = int(tokenizer.eos_token_id)
    raw = tokenizer(
        list(texts),
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )["input_ids"]
    encoded = tokenizer(
        list(texts),
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=max_length - 1,
    )["input_ids"]
    sequences = [
        list(row) if row and row[-1] == terminal_token_id else [*row, terminal_token_id]
        for row in encoded
    ]
    if not sequences or any(not row or row[-1] != terminal_token_id for row in sequences):
        raise ValueError("EOS token construction failed")
    if any(len(row) > max_length for row in sequences):
        raise ValueError("EOS-terminated sequence exceeds max_length")
    untruncated_lengths = [
        len(row) if row and row[-1] == terminal_token_id else len(row) + 1
        for row in raw
    ]
    return sequences, {
        "terminal_token_id": terminal_token_id,
        "maximum_untruncated_tokens": max(untruncated_lengths),
        "maximum_encoded_tokens": max(map(len, sequences)),
        "truncated_input_count": sum(
            length > max_length for length in untruncated_lengths
        ),
    }


def encode_label_inputs(
    *,
    formatted_queries: Sequence[str],
    formatted_documents: Sequence[str],
    batch_size: int,
    seed: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]]:
    """Encode query and label texts once, then restore their separate matrices."""

    if not formatted_queries:
        raise ValueError("at least one formatted query is required")
    if not formatted_documents:
        raise ValueError("at least one formatted document is required")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    combined, metadata = _encode_contextual_e5(
        (*formatted_queries, *formatted_documents),
        batch_size=batch_size,
        seed=seed,
    )
    query_count = len(formatted_queries)
    return combined[:query_count], combined[query_count:], metadata


def _encode_contextual_e5(
    texts: Sequence[str],
    *,
    batch_size: int,
    seed: int,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Run the selected E5 checkpoint with its frozen pooling contract."""

    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
        from transformers.utils.hub import cached_file
    except ImportError as error:  # pragma: no cover - container integration path
        raise RuntimeError("E5 inference requires torch and transformers") from error
    device_properties = validate_rocm_runtime(torch)

    spec = CONTEXTUAL_E5_SPEC
    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        padding_side="left",
        trust_remote_code=False,
    )
    tokenizer_config_path = cached_file(
        spec.model_id,
        "tokenizer_config.json",
        revision=spec.revision,
        local_files_only=True,
    )
    if tokenizer_config_path is None:
        raise RuntimeError("could not resolve the pinned E5 tokenizer snapshot")
    tokenizer_revision = snapshot_revision_from_path(tokenizer_config_path)
    if tokenizer_revision != spec.revision:
        raise RuntimeError(
            "resolved E5 tokenizer revision differs from the frozen revision: "
            f"{tokenizer_revision!r}"
        )
    model = AutoModel.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    ).to("cuda")
    resolved_model_revision = getattr(model.config, "_commit_hash", None)
    if resolved_model_revision != spec.revision:
        raise RuntimeError(
            "resolved E5 model revision differs from the frozen revision: "
            f"{resolved_model_revision!r}"
        )
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    batches: list[NDArray[np.float32]] = []
    maximum_tokens = 0
    maximum_untruncated = 0
    truncation_count = 0
    terminal_token_id = int(tokenizer.eos_token_id)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            sequences, batch_metadata = build_e5_token_ids(
                tokenizer,
                batch,
                max_length=spec.max_length,
            )
            tokens = tokenizer.pad(
                {
                    "input_ids": sequences,
                    "attention_mask": [[1] * len(row) for row in sequences],
                },
                padding=True,
                return_tensors="pt",
            )
            truncation_count += batch_metadata["truncated_input_count"]
            maximum_untruncated = max(
                maximum_untruncated,
                batch_metadata["maximum_untruncated_tokens"],
            )
            maximum_tokens = max(
                maximum_tokens,
                int(tokens["attention_mask"].sum(dim=1).max().item()),
            )
            tokens = {key: value.to("cuda") for key, value in tokens.items()}
            output = model(**tokens)
            pooled = functional.normalize(
                last_token_pool(output.last_hidden_state, tokens["attention_mask"]),
                p=2,
                dim=1,
            )
            batches.append(pooled.float().cpu().numpy())

    matrix = normalize_embedding_rows(np.concatenate(batches, axis=0))
    if matrix.shape != (len(texts), spec.embedding_dimension):
        raise ValueError(
            "E5 embedding shape differs from the frozen contract: "
            f"{matrix.shape!r}"
        )
    return matrix, {
        "backend": "transformers",
        "model_id": spec.model_id,
        "requested_revision": spec.revision,
        "resolved_model_revision": resolved_model_revision,
        "resolved_tokenizer_revision": tokenizer_revision,
        "license": spec.license,
        "dtype": spec.dtype,
        "pooling": spec.pooling,
        "embedding_dimension": spec.embedding_dimension,
        "input_count": len(texts),
        "input_sha256": _hash_texts(texts),
        "transformers_version": importlib.metadata.version("transformers"),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device": "rocm_hip",
        "device_properties": str(device_properties),
        "required_container_image": ROCM_IMAGE,
        "parameter_count": int(parameter_count),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "batch_size": batch_size,
        "seed": seed,
        "padding_side": "left",
        "maximum_observed_tokens": maximum_tokens,
        "maximum_untruncated_tokens": maximum_untruncated,
        "truncated_input_count": truncation_count,
        "terminal_token_id": terminal_token_id,
        "terminal_token": tokenizer.eos_token,
        "max_length": spec.max_length,
        "encoding_seconds": time.perf_counter() - started,
    }


def _hash_texts(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


encode_contextual_e5 = _encode_contextual_e5


__all__ = [
    "ROCM_IMAGE",
    "build_e5_token_ids",
    "encode_contextual_e5",
    "encode_label_inputs",
    "last_token_pool",
    "snapshot_revision_from_path",
    "validate_rocm_runtime",
]
