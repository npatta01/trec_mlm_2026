"""Optional vLLM adapters for zero-shot expert-label ranking."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
MODEL_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
MODEL_DTYPE = "float32"


def chat_template_input_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> Any:
    """Apply the official chat template once and unwrap its input IDs."""

    encoded = tokenizer.apply_chat_template(
        list(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    try:
        return encoded["input_ids"]
    except (KeyError, TypeError) as error:
        raise ValueError("chat template did not return input_ids") from error


def grade_token_ids(tokenizer: Any) -> dict[int, int]:
    """Resolve the three grades as distinct single tokens."""

    result: dict[int, int] = {}
    for grade in (0, 1, 2):
        token_ids = tokenizer.encode(str(grade), add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"grade {grade} is not exactly one token")
        result[grade] = int(token_ids[0])
    if len(set(result.values())) != 3:
        raise ValueError("grade token IDs must be distinct")
    return result


def pointwise_sampling_kwargs(
    grade_tokens: Mapping[int, int],
    *,
    seed: int,
) -> dict[str, Any]:
    """Return the frozen constrained one-token sampling parameters."""

    if set(grade_tokens) != {0, 1, 2}:
        raise ValueError("grade tokens must contain exactly grades 0, 1, and 2")
    allowed = [grade_tokens[grade] for grade in (0, 1, 2)]
    return {
        "temperature": 0.0,
        "seed": seed,
        "max_tokens": 1,
        "allowed_token_ids": allowed,
        "logprob_token_ids": allowed,
        "logprobs": 3,
    }


def extract_grade_log_probabilities(
    output: Any,
    grade_tokens: Mapping[int, int],
) -> dict[int, float]:
    """Extract all three requested grade log probabilities from one vLLM output."""

    try:
        token_logprobs = output.outputs[0].logprobs[0]
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("vLLM output is missing first-token log probabilities") from error
    values: dict[int, float] = {}
    for grade, token_id in grade_tokens.items():
        try:
            values[grade] = float(token_logprobs[token_id].logprob)
        except (KeyError, AttributeError, TypeError) as error:
            raise ValueError(f"vLLM output is missing grade {grade} log probability") from error
    return values


def listwise_json_schema(
    handles: Sequence[str],
    *,
    selection_count: int = 20,
) -> dict[str, Any]:
    """Return an exact-size handle array schema, defaulting to canonical top-20."""

    values = list(handles)
    if (
        (len(values), selection_count) not in ((150, 20), (12, 12), (20, 20))
        or len(set(values)) != len(values)
    ):
        raise ValueError(
            "listwise schema supports only 150 handles selecting 20, "
            "12 selecting 12, or 20 selecting 20, all with unique handles"
        )
    return {
        "type": "array",
        "items": {"type": "string", "enum": values},
        "minItems": selection_count,
        "maxItems": selection_count,
    }


def update_inference_stats(stats: dict[str, int], outputs: Sequence[Any]) -> None:
    """Accumulate request and token counts exposed by vLLM RequestOutput."""

    request_count = len(outputs)
    try:
        prompt_tokens = sum(len(output.prompt_token_ids) for output in outputs)
        generated_tokens = sum(len(output.outputs[0].token_ids) for output in outputs)
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError("vLLM outputs are missing token accounting fields") from error
    stats["request_count"] = stats.get("request_count", 0) + request_count
    stats["prompt_token_count"] = stats.get("prompt_token_count", 0) + prompt_tokens
    stats["output_token_count"] = stats.get("output_token_count", 0) + generated_tokens


def load_ranker_engine(*, seed: int = 2026) -> tuple[Any, Any, dict[str, Any]]:
    """Load the frozen FP32 ranker and tokenizer in the pinned runtime."""

    try:
        import torch
        import vllm
        from transformers import AutoTokenizer
        from vllm import LLM
    except ImportError as error:  # pragma: no cover - integration path
        raise RuntimeError("ranker inference requires the pinned vLLM ROCm image") from error
    if not torch.cuda.is_available():
        raise RuntimeError("ranker inference requires an available ROCm device")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=False,
    )
    llm = LLM(
        model=MODEL_ID,
        revision=MODEL_REVISION,
        dtype=MODEL_DTYPE,
        seed=seed,
        trust_remote_code=False,
        max_model_len=8192,
        gpu_memory_utilization=0.70,
        attention_config={"backend": "TRITON_ATTN"},
    )
    return llm, tokenizer, {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "dtype": MODEL_DTYPE,
        "seed": seed,
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "hip_version": torch.version.hip,
        "device_properties": str(torch.cuda.get_device_properties(0)),
    }


def run_pointwise_messages(
    llm: Any,
    tokenizer: Any,
    messages: Sequence[list[dict[str, str]]],
    *,
    seed: int = 2026,
    use_tqdm: bool = True,
    stats: dict[str, int] | None = None,
) -> tuple[list[dict[int, float]], dict[int, int]]:
    """Run constrained pointwise grading for preformatted chat messages."""

    from vllm import SamplingParams

    tokens = grade_token_ids(tokenizer)
    sampling = SamplingParams(**pointwise_sampling_kwargs(tokens, seed=seed))
    outputs = llm.chat(
        list(messages),
        sampling_params=sampling,
        use_tqdm=use_tqdm,
    )
    if stats is not None:
        update_inference_stats(stats, outputs)
    return [extract_grade_log_probabilities(output, tokens) for output in outputs], tokens


def run_listwise_messages(
    llm: Any,
    messages: Sequence[list[dict[str, str]]],
    *,
    handles: Sequence[str],
    selection_count: int = 20,
    seed: int = 2026,
    use_tqdm: bool = True,
    stats: dict[str, int] | None = None,
) -> list[str]:
    """Run structured exact-size generation for preformatted chat messages."""

    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    sampling = SamplingParams(
        temperature=0.0,
        seed=seed,
        max_tokens=256,
        structured_outputs=StructuredOutputsParams(
            json=listwise_json_schema(handles, selection_count=selection_count)
        ),
    )
    outputs = llm.chat(
        list(messages),
        sampling_params=sampling,
        use_tqdm=use_tqdm,
    )
    if stats is not None:
        update_inference_stats(stats, outputs)
    return [output.outputs[0].text for output in outputs]
