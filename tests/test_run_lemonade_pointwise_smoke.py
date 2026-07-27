import json
from pathlib import Path

import pytest

from scripts.run_lemonade_pointwise_smoke import (
    _lemonade_transport_schema,
    _parse_backend_response_body,
    _runtime_observation,
    _validate_backend_response_model,
)
from trec_mlm.pointwise_runner import FROZEN_DECISION_SCHEMA_SHA256


SCHEMA = json.loads(
    Path("protocols/pointwise_judge_bakeoff/decision-v1.schema.json").read_text(
        encoding="utf-8"
    )
)


def test_lemonade_transport_schema_omits_only_unsupported_max_length() -> None:
    frozen = json.loads(json.dumps(SCHEMA))

    transport = _lemonade_transport_schema(frozen)

    assert frozen == SCHEMA
    assert "maxLength" not in transport["properties"]["rationale"]
    assert transport["properties"]["error_flags"]["uniqueItems"] is True
    transport["properties"]["rationale"]["maxLength"] = 4000
    assert transport == SCHEMA


def test_runtime_observation_hashes_exact_adjusted_transport_schema() -> None:
    observation = _runtime_observation(
        SCHEMA,
        backend_model="friendly-model",
        backend_response_model="/verified/checkpoint.gguf",
        thinking=False,
        thinking_control_omitted=False,
        reasoning_effort=None,
        max_output_tokens=4096,
    )

    assert observation["schema_sha256"] == FROZEN_DECISION_SCHEMA_SHA256
    assert observation["transport_schema_sha256"] == (
        "40a725cf89740c377ba6b91fa3d1fbe4657127d5f0592b884110cae4e944bc67"
    )
    assert observation["schema_transport_adjustment"] == (
        "rationale.maxLength omitted; frozen parser retained"
    )


def test_validate_backend_response_model_accepts_pinned_canonical_path() -> None:
    response = {
        "model": (
            "/var/lib/lemonade/.cache/huggingface/hub/models--unsloth--"
            "Qwen3.6-35B-A3B-GGUF/snapshots/revision/"
            "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"
        )
    }

    _validate_backend_response_model(response, response["model"])


def test_validate_backend_response_model_rejects_unpinned_identity() -> None:
    with pytest.raises(ValueError, match="backend response model"):
        _validate_backend_response_model(
            {"model": "/unexpected/checkpoint.gguf"},
            "/verified/checkpoint.gguf",
        )


def test_parse_backend_response_body_preserves_canonical_model_and_nested_usage() -> (
    None
):
    raw_body = (
        b'{"id":"completion-1","model":"/verified/checkpoint.gguf",'
        b'"choices":[{"message":{"content":"{\\"label\\": \\"good\\"}"}}],'
        b'"usage":{"prompt_tokens":12,"completion_tokens":8,'
        b'"prompt_tokens_details":{"cached_tokens":3}}}'
    )

    content, body = _parse_backend_response_body(raw_body, "/verified/checkpoint.gguf")

    assert content == '{"label": "good"}'
    assert body == {
        "id": "completion-1",
        "model": "/verified/checkpoint.gguf",
        "choices": [{"message": {"content": '{"label": "good"}'}}],
        "usage": {
            "prompt_tokens": 12,
            "completion_tokens": 8,
            "prompt_tokens_details": {"cached_tokens": 3},
        },
    }


def test_parse_backend_response_body_rejects_duplicate_model_key() -> None:
    raw_body = (
        b'{"model":"/verified/checkpoint.gguf",'
        b'"model":"/foreign/checkpoint.gguf","choices":[]}'
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        _parse_backend_response_body(raw_body, "/verified/checkpoint.gguf")
