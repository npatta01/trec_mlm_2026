import pytest

from scripts.run_lemonade_pointwise_smoke import _validate_backend_response_model


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
