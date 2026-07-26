import pytest

from scripts.run_lemonade_pointwise_smoke import (
    _parse_backend_response_body,
    _validate_backend_response_model,
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
