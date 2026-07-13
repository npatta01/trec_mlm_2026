"""Optional vLLM adapter for the frozen GroupRank Stage 2 pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from trec_mlm.grouprank import GroupRankRequest


MODEL_ID = "AQ-MedAI/Diver-GroupRank-7B"
MODEL_REVISION = "42a9504b32616f155970d5b7f349b63dc24b3e27"
MODEL_DTYPE = "bfloat16"
MODEL_LICENSE = "Apache-2.0"
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.40
ATTENTION_BACKEND = "TRITON_ATTN"
ROCM_CONTAINER_IMAGE = (
    "docker.io/vllm/vllm-openai-rocm@sha256:"
    "3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28"
)

_MODEL_PARAMETER_COUNT = 7_615_616_512


def finalize_grouprank_runtime(
    runtime_metadata: Mapping[str, Any],
    raw_outputs: Sequence[Mapping[str, Any]],
    *,
    elapsed_seconds: float,
    expected_request_count: int,
) -> dict[str, Any]:
    """Build immutable post-inference provenance from captured raw records."""

    if runtime_metadata.get("container_image") != ROCM_CONTAINER_IMAGE:
        raise ValueError("GroupRank runtime container image changed")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(float(elapsed_seconds))
        or elapsed_seconds <= 0
    ):
        raise ValueError("GroupRank runtime elapsed seconds must be positive finite")
    if len(raw_outputs) != expected_request_count:
        raise ValueError(
            f"GroupRank runtime must record exactly {expected_request_count} requests"
        )

    prompt_token_count = 0
    output_token_count = 0
    for record in raw_outputs:
        if not isinstance(record, Mapping):
            raise ValueError("GroupRank raw outputs must be objects")
        for field in ("prompt_token_count", "output_token_count"):
            value = record.get(field)
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"GroupRank raw output {field.replace('_', ' ')} changed"
                )
        prompt_token_count += record["prompt_token_count"]
        output_token_count += record["output_token_count"]

    runtime = dict(runtime_metadata)
    runtime["elapsed_seconds"] = float(elapsed_seconds)
    runtime["inference_stats"] = {
        "request_count": len(raw_outputs),
        "prompt_token_count": prompt_token_count,
        "output_token_count": output_token_count,
    }
    return runtime


def load_grouprank_engine() -> tuple[Any, Any, dict[str, Any]]:
    """Load the pinned GroupRank model and expose its runtime provenance."""

    try:
        import torch
        import vllm
        from transformers import AutoTokenizer
        from vllm import LLM
    except ImportError as error:  # pragma: no cover - integration path
        raise RuntimeError(
            "GroupRank inference requires the pinned vLLM ROCm image"
        ) from error
    if not torch.cuda.is_available():
        raise RuntimeError("GroupRank inference requires an available ROCm device")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    llm = LLM(
        model=MODEL_ID,
        revision=MODEL_REVISION,
        dtype=MODEL_DTYPE,
        trust_remote_code=False,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        attention_config={"backend": ATTENTION_BACKEND},
    )
    return llm, tokenizer, {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
        "dtype": MODEL_DTYPE,
        "license": MODEL_LICENSE,
        "parameter_count": _MODEL_PARAMETER_COUNT,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "attention_backend": ATTENTION_BACKEND,
        "container_image": ROCM_CONTAINER_IMAGE,
        "vllm_version": vllm.__version__,
        "torch_version": str(torch.__version__),
        "hip_version": torch.version.hip,
        "device_properties": str(torch.cuda.get_device_properties(0)),
    }


def run_grouprank_requests(
    llm: Any,
    requests: Sequence[GroupRankRequest],
    *,
    stats: dict[str, int],
) -> list[dict[str, Any]]:
    """Generate and capture one unparsed raw record for every request."""

    from vllm import SamplingParams

    records: list[dict[str, Any]] = []
    prompt_token_count = 0
    output_token_count = 0
    for request in requests:
        sampling = SamplingParams(
            temperature=0.3,
            top_p=0.8,
            max_tokens=2048,
            seed=request.sampling_seed,
        )
        outputs = llm.chat(
            [[dict(message) for message in request.messages]],
            sampling_params=sampling,
            use_tqdm=False,
        )
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise ValueError(
                "vLLM must return exactly one GroupRank output per request"
            )
        output = outputs[0]
        completion = output.outputs[0]
        try:
            prompt_tokens = len(output.prompt_token_ids)
            output_tokens = len(completion.token_ids)
        except (AttributeError, TypeError) as error:
            raise ValueError(
                "vLLM GroupRank output is missing token accounting fields"
            ) from error
        try:
            finish_reason = completion.finish_reason
        except AttributeError as error:
            raise ValueError(
                "vLLM GroupRank output is missing a finish reason"
            ) from error
        if not isinstance(finish_reason, str) or not finish_reason:
            raise ValueError("vLLM GroupRank output is missing a finish reason")
        try:
            raw_text = completion.text
        except AttributeError as error:
            raise ValueError("vLLM GroupRank output is missing raw text") from error
        if not isinstance(raw_text, str):
            raise ValueError("vLLM GroupRank output is missing raw text")
        records.append(
            {
                "query_id": request.query_id,
                "repetition": request.repetition,
                "group": request.group,
                "ordered_candidate_ids": list(request.ordered_candidate_ids),
                "handle_to_candidate": dict(request.handle_to_candidate),
                "raw_text": raw_text,
                "prompt_token_count": prompt_tokens,
                "output_token_count": output_tokens,
                "finish_reason": finish_reason,
                "sampling_seed": request.sampling_seed,
            }
        )
        prompt_token_count += prompt_tokens
        output_token_count += output_tokens

    stats["request_count"] = stats.get("request_count", 0) + len(records)
    stats["prompt_token_count"] = (
        stats.get("prompt_token_count", 0) + prompt_token_count
    )
    stats["output_token_count"] = (
        stats.get("output_token_count", 0) + output_token_count
    )
    return records
