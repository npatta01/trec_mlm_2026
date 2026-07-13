"""Optional embedding backends for the frozen expert-label sweep."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import importlib.metadata
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.label_retrieval import (
    ROCM_IMAGE,
    LabelEncoderSpec,
    normalize_embedding_rows,
)


def cls_token_pool(last_hidden_states: Any) -> Any:
    """Pool the first token from an encoder hidden-state tensor."""

    if last_hidden_states.ndim != 3 or last_hidden_states.shape[1] < 1:
        raise ValueError("hidden states must contain at least one token")
    return last_hidden_states[:, 0]


def mean_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Mean-pool only unmasked token states."""

    import torch

    if last_hidden_states.ndim != 3 or attention_mask.ndim != 2:
        raise ValueError("hidden states and attention mask have invalid dimensions")
    if last_hidden_states.shape[:2] != attention_mask.shape:
        raise ValueError("attention mask shape must match hidden-state tokens")
    counts = attention_mask.sum(dim=1, keepdim=True)
    if bool(torch.any(counts == 0).item()):
        raise ValueError("every input must contain at least one unmasked token")
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden_states.dtype)
    return (last_hidden_states * mask).sum(dim=1) / counts.to(
        dtype=last_hidden_states.dtype
    )


def pool_encoder_hidden_states(
    last_hidden_states: Any,
    attention_mask: Any,
    *,
    pooling: str,
) -> Any:
    """Apply a registered encoder pooler and return unit-normalized rows."""

    import torch.nn.functional as functional

    if pooling == "dense_cls_normalized":
        pooled = cls_token_pool(last_hidden_states)
    elif pooling == "mean_layernorm_normalized":
        pooled = mean_token_pool(last_hidden_states, attention_mask)
        pooled = functional.layer_norm(
            pooled,
            normalized_shape=(pooled.shape[1],),
        )
    else:
        raise ValueError(f"unsupported encoder pooling: {pooling!r}")
    return functional.normalize(pooled, p=2, dim=1)


def torch_dtype_for_spec(spec: LabelEncoderSpec) -> Any:
    """Resolve a frozen registry dtype to its PyTorch dtype."""

    import torch

    try:
        return {"bfloat16": torch.bfloat16, "float32": torch.float32}[spec.dtype]
    except KeyError as error:
        raise ValueError(f"unsupported torch dtype: {spec.dtype!r}") from error


def last_token_pool(last_hidden_states: Any, attention_mask: Any) -> Any:
    """Pool the final non-padding token for left- or right-padded batches."""

    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    import torch

    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device),
        sequence_lengths,
    ]


def embedding_outputs_to_matrix(
    outputs: Sequence[Any],
    *,
    expected_dimension: int,
) -> NDArray[np.float32]:
    """Extract and validate one normalized matrix from vLLM pooling outputs."""

    rows: list[Any] = []
    for output in outputs:
        payload = output.outputs
        value = getattr(payload, "embedding", None)
        if value is None:
            value = getattr(payload, "data", None)
        if value is None:
            raise ValueError("vLLM output does not contain embedding data")
        rows.append(value)
    matrix = normalize_embedding_rows(np.asarray(rows, dtype=np.float32))
    if matrix.shape[1] != expected_dimension:
        raise ValueError(
            "embedding dimension does not match frozen model spec: "
            f"observed={matrix.shape[1]}, expected={expected_dimension}"
        )
    return matrix


def validate_embedding_smoke(
    *,
    query_embeddings: Any,
    document_embeddings: Any,
    reference_scores: Any,
    expected_dimension: int,
    score_tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Validate a canonical embedding score matrix against a reference path."""

    queries = normalize_embedding_rows(query_embeddings)
    documents = normalize_embedding_rows(document_embeddings)
    if (
        queries.shape[1] != expected_dimension
        or documents.shape[1] != expected_dimension
    ):
        raise ValueError("smoke embedding dimension does not match the model registry")
    observed = np.asarray(queries @ documents.T, dtype=np.float32)
    reference = np.asarray(reference_scores, dtype=np.float32)
    if reference.shape != observed.shape or not np.isfinite(reference).all():
        raise ValueError("reference score matrix is invalid")
    maximum_difference = float(np.max(np.abs(observed - reference)))
    if maximum_difference > score_tolerance:
        raise ValueError(
            "embedding smoke scores disagree with the official reference: "
            f"maximum_difference={maximum_difference}, tolerance={score_tolerance}"
        )
    return {
        "passed": True,
        "expected_dimension": expected_dimension,
        "query_count": int(queries.shape[0]),
        "document_count": int(documents.shape[0]),
        "score_tolerance": score_tolerance,
        "maximum_absolute_score_difference": maximum_difference,
        "observed_scores": observed.tolist(),
        "reference_scores": reference.tolist(),
    }


def vllm_runtime_kwargs(spec: LabelEncoderSpec, *, seed: int) -> dict[str, Any]:
    """Return the frozen vLLM options, including deterministic ROCm attention."""

    return {
        "model": spec.model_id,
        "revision": spec.revision,
        "runner": "pooling",
        "convert": "embed",
        "trust_remote_code": False,
        "dtype": spec.dtype,
        "seed": seed,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.85,
        "attention_config": {"backend": "TRITON_ATTN"},
    }


def token_length_summary(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
) -> dict[str, int]:
    """Measure truncation without changing the inputs sent to the encoder."""

    tokenized = tokenizer(
        list(texts),
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    lengths = [len(row) for row in tokenized["input_ids"]]
    return {
        "maximum_untruncated_tokens": max(lengths),
        "truncated_input_count": sum(length > max_length for length in lengths),
    }


def _build_terminal_token_ids(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    terminal_token_id: int,
    reserved_tokens: int,
) -> tuple[list[list[int]], dict[str, int]]:
    if max_length <= reserved_tokens:
        raise ValueError("max_length must leave room for content and terminal token")
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
        max_length=max_length - reserved_tokens,
    )["input_ids"]
    sequences = [
        list(row) if row and row[-1] == terminal_token_id else [*row, terminal_token_id]
        for row in encoded
    ]
    if any(not row or row[-1] != terminal_token_id for row in sequences):
        raise ValueError("terminal token construction failed")
    if any(len(row) > max_length for row in sequences):
        raise ValueError("terminal-token sequence exceeds max_length")
    untruncated_lengths = [
        len(row) if row and row[-1] == terminal_token_id else len(row) + 1
        for row in raw
    ]
    return sequences, {
        "terminal_token_id": terminal_token_id,
        "maximum_untruncated_tokens": max(untruncated_lengths),
        "maximum_encoded_tokens": max(len(row) for row in sequences),
        "truncated_input_count": sum(
            raw_length > max_length for raw_length in untruncated_lengths
        ),
    }


def build_qwen_token_ids(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
) -> tuple[list[list[int]], dict[str, int]]:
    """Reproduce the pinned Qwen reference terminal-token preprocessing."""

    terminal = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    return _build_terminal_token_ids(
        tokenizer,
        texts,
        max_length=max_length,
        terminal_token_id=int(terminal),
        reserved_tokens=2,
    )


def build_e5_token_ids(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
) -> tuple[list[list[int]], dict[str, int]]:
    """Reproduce legacy E5 tokenization with exactly one terminal EOS token."""

    if tokenizer.eos_token_id is None:
        raise ValueError("E5 tokenizer must define an EOS token")
    return _build_terminal_token_ids(
        tokenizer,
        texts,
        max_length=max_length,
        terminal_token_id=int(tokenizer.eos_token_id),
        reserved_tokens=1,
    )


def make_token_prompts(
    sequences: Sequence[Sequence[int]],
    prompt_type: type[Any],
) -> list[Any]:
    """Wrap explicit token IDs for a vLLM pooling request."""

    return [prompt_type(prompt_token_ids=list(row)) for row in sequences]


def encode_label_inputs(
    spec: LabelEncoderSpec,
    *,
    formatted_queries: Sequence[str],
    formatted_documents: Sequence[str],
    batch_size: int,
    seed: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], dict[str, Any]]:
    """Encode frozen query/document texts with the declared backend."""

    if not formatted_queries or not formatted_documents:
        raise ValueError("at least one query and document are required")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be a positive integer")
    if spec.runtime == "sentence_transformers_cpu":
        combined, metadata = _encode_sentence_transformers(
            spec,
            (*formatted_queries, *formatted_documents),
            batch_size=batch_size,
            seed=seed,
        )
    elif spec.runtime == "transformers_rocm":
        encoder = (
            _encode_e5_transformers
            if spec.key == "e5_mistral_7b_instruct"
            else _encode_transformers_encoder
        )
        combined, metadata = encoder(
            spec,
            (*formatted_queries, *formatted_documents),
            batch_size=batch_size,
            seed=seed,
        )
    elif spec.runtime == "vllm_rocm":
        combined, metadata = _encode_vllm(
            spec,
            (*formatted_queries, *formatted_documents),
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported embedding runtime: {spec.runtime!r}")
    query_count = len(formatted_queries)
    return combined[:query_count], combined[query_count:], metadata


def _encode_sentence_transformers(
    spec: LabelEncoderSpec,
    texts: Sequence[str],
    *,
    batch_size: int,
    seed: int,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    try:
        import torch
        from sentence_transformers import SentenceTransformer
    except ImportError as error:  # pragma: no cover - optional dependency path
        raise RuntimeError(
            "MPNet label control requires `uv run --extra dense`"
        ) from error

    torch.manual_seed(seed)
    torch.set_num_threads(16)
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    model = SentenceTransformer(
        spec.model_id,
        revision=spec.revision,
        device="cpu",
        trust_remote_code=False,
    )
    model.eval()
    encoded = model.encode(
        list(texts),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    matrix = normalize_embedding_rows(encoded)
    _validate_dimension(matrix, spec)
    return matrix, {
        **_base_metadata(spec, texts),
        "backend": "sentence_transformers",
        "sentence_transformers_version": importlib.metadata.version(
            "sentence-transformers"
        ),
        "torch_version": torch.__version__,
        "device": "cpu",
        "batch_size": batch_size,
        "seed": seed,
        "encoding_seconds": time.perf_counter() - started,
        "deterministic_algorithms": True,
    }


def _encode_e5_transformers(
    spec: LabelEncoderSpec,
    texts: Sequence[str],
    *,
    batch_size: int,
    seed: int,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    try:
        import torch
        import torch.nn.functional as functional
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:  # pragma: no cover - container integration path
        raise RuntimeError("E5 inference requires torch and transformers") from error
    if not torch.cuda.is_available():
        raise RuntimeError("E5 ROCm inference requires an available HIP device")

    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        padding_side="left",
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    ).to("cuda")
    model.eval()
    batches: list[NDArray[np.float32]] = []
    maximum_tokens = 0
    truncation_count = 0
    maximum_untruncated = 0
    terminal_token_id = int(tokenizer.eos_token_id)
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            sequences, batch_metadata = build_e5_token_ids(
                tokenizer,
                batch,
                max_length=512,
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
                maximum_tokens, int(tokens["attention_mask"].sum(1).max())
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
    _validate_dimension(matrix, spec)
    return matrix, {
        **_base_metadata(spec, texts),
        "backend": "transformers",
        "transformers_version": importlib.metadata.version("transformers"),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device": "cuda_rocm",
        "container_image": ROCM_IMAGE,
        "device_properties": str(torch.cuda.get_device_properties(0)),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "batch_size": batch_size,
        "seed": seed,
        "maximum_observed_tokens": maximum_tokens,
        "maximum_untruncated_tokens": maximum_untruncated,
        "truncated_input_count": truncation_count,
        "terminal_token_id": terminal_token_id,
        "terminal_token": tokenizer.eos_token,
        "max_length": 512,
        "encoding_seconds": time.perf_counter() - started,
    }


def _encode_transformers_encoder(
    spec: LabelEncoderSpec,
    texts: Sequence[str],
    *,
    batch_size: int,
    seed: int,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    """Encode BERT-family inputs with the registry's explicit pooling contract."""

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as error:  # pragma: no cover - container integration path
        raise RuntimeError(
            "encoder inference requires torch and transformers"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("encoder ROCm inference requires an available HIP device")

    torch.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        trust_remote_code=False,
    )
    model = AutoModel.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        dtype=torch_dtype_for_spec(spec),
        trust_remote_code=False,
    ).to("cuda")
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    lengths = token_length_summary(tokenizer, texts, max_length=512)
    batches: list[NDArray[np.float32]] = []
    maximum_tokens = 0
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            tokens = tokenizer(
                batch,
                add_special_tokens=True,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            maximum_tokens = max(
                maximum_tokens,
                int(tokens["attention_mask"].sum(dim=1).max().item()),
            )
            tokens = {key: value.to("cuda") for key, value in tokens.items()}
            output = model(**tokens)
            pooled = pool_encoder_hidden_states(
                output.last_hidden_state,
                tokens["attention_mask"],
                pooling=spec.pooling,
            )
            batches.append(pooled.float().cpu().numpy())
    matrix = normalize_embedding_rows(np.concatenate(batches, axis=0))
    _validate_dimension(matrix, spec)
    return matrix, {
        **_base_metadata(spec, texts),
        "backend": "transformers_encoder",
        "transformers_version": importlib.metadata.version("transformers"),
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device": "cuda_rocm",
        "container_image": ROCM_IMAGE,
        "device_properties": str(torch.cuda.get_device_properties(0)),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "parameter_count": int(parameter_count),
        "batch_size": batch_size,
        "seed": seed,
        "maximum_observed_tokens": maximum_tokens,
        "max_length": 512,
        "pooling": spec.pooling,
        "encoding_seconds": time.perf_counter() - started,
        **lengths,
    }


def _encode_vllm(
    spec: LabelEncoderSpec,
    texts: Sequence[str],
    *,
    seed: int,
) -> tuple[NDArray[np.float32], dict[str, Any]]:
    try:
        import torch
        import vllm
        from vllm import LLM, TokensPrompt
        from vllm.config.pooler import PoolerConfig
        from transformers import AutoTokenizer
    except ImportError as error:  # pragma: no cover - container integration path
        raise RuntimeError(
            "Qwen/Jina inference requires the pinned vLLM image"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("vLLM ROCm inference requires an available HIP device")

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        spec.model_id,
        revision=spec.revision,
        trust_remote_code=False,
    )
    if spec.key.startswith("qwen3_embedding_"):
        sequences, length_metadata = build_qwen_token_ids(
            tokenizer,
            texts,
            max_length=512,
        )
        model_inputs: Sequence[Any] = make_token_prompts(sequences, TokensPrompt)
    else:
        length_metadata = token_length_summary(tokenizer, texts, max_length=512)
        model_inputs = list(texts)
    pooler = None
    if spec.key == "jina_v5_small_retrieval":
        pooler = PoolerConfig(task="embed", pooling_type="LAST")
    model = LLM(**vllm_runtime_kwargs(spec, seed=seed), pooler_config=pooler)
    outputs = model.encode(
        model_inputs,
        use_tqdm=False,
        pooling_task="embed",
    )
    matrix = embedding_outputs_to_matrix(
        outputs,
        expected_dimension=spec.embedding_dimension,
    )
    return matrix, {
        **_base_metadata(spec, texts),
        "backend": "vllm",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device": "cuda_rocm",
        "device_properties": str(torch.cuda.get_device_properties(0)),
        "container_image": ROCM_IMAGE,
        "seed": seed,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.85,
        "attention_backend": "TRITON_ATTN",
        **length_metadata,
        "terminal_token": (
            "<|endoftext|>" if spec.key.startswith("qwen3_embedding_") else None
        ),
        "determinism_reason": (
            "default ROCM_ATTN was non-repeatable on gfx1151; the pinned image's "
            "TRITON_ATTN backend was bit-identical in same-engine smoke testing"
        ),
        "encoding_seconds": time.perf_counter() - started,
    }


def _validate_dimension(matrix: NDArray[np.float32], spec: LabelEncoderSpec) -> None:
    if matrix.shape[1] != spec.embedding_dimension:
        raise ValueError(
            f"{spec.key} returned dimension {matrix.shape[1]}, "
            f"expected {spec.embedding_dimension}"
        )


def _base_metadata(
    spec: LabelEncoderSpec,
    texts: Sequence[str],
) -> dict[str, Any]:
    return {
        "model_id": spec.model_id,
        "requested_revision": spec.revision,
        "license": spec.license,
        "dtype": spec.dtype,
        "pooling": spec.pooling,
        "embedding_dimension": spec.embedding_dimension,
        "input_count": len(texts),
        "input_sha256": _hash_texts(texts),
    }


def _hash_texts(texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for text in texts:
        encoded = text.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


__all__ = [
    "ROCM_IMAGE",
    "build_e5_token_ids",
    "build_qwen_token_ids",
    "embedding_outputs_to_matrix",
    "encode_label_inputs",
    "last_token_pool",
    "make_token_prompts",
    "token_length_summary",
    "vllm_runtime_kwargs",
]
