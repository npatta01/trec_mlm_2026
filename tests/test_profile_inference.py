from __future__ import annotations

import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

from trec_mlm.profile_inference import (
    PROFILE_BATCH_SIZE,
    PROFILE_SEED,
    PROFILE_TASK_INSTRUCTION,
    build_profile_token_ids,
    classify_response_truncation,
    format_profile_document,
    format_profile_query,
    normalize_persisted_embeddings,
    validate_profile_runtime_options,
    validate_profile_runtime_identity,
    validate_profile_device_identity,
)


@pytest.mark.parametrize(
    ("prefix_tokens", "full_tokens", "expected"),
    [(40, 100, "none"), (40, 700, "partial"), (511, 700, "full")],
)
def test_response_truncation_distinguishes_partial_and_full(
    prefix_tokens: int, full_tokens: int, expected: str
) -> None:
    assert classify_response_truncation(prefix_tokens, full_tokens) == expected


class FakeTokenizer:
    eos_token_id = 2
    eos_token = "</s>"
    padding_side = "left"
    truncation_side = "right"

    def __init__(self, rows: dict[str, list[int]]) -> None:
        self.rows = rows

    def __call__(
        self,
        texts: list[str],
        *,
        add_special_tokens: bool,
        padding: bool,
        truncation: bool,
        max_length: int | None = None,
    ) -> dict[str, list[list[int]]]:
        assert add_special_tokens is True
        assert padding is False
        output = [list(self.rows[text]) for text in texts]
        if truncation:
            assert max_length is not None
            output = [row[:max_length] for row in output]
        return {"input_ids": output}


def test_profile_formatting_is_exact() -> None:
    assert PROFILE_TASK_INSTRUCTION == (
        "Given a user question, retrieve historical question-and-response records "
        "from LLMs that are likely to answer it correctly."
    )
    assert format_profile_query("  Why?  ") == (
        "Instruct: Given a user question, retrieve historical question-and-response "
        "records from LLMs that are likely to answer it correctly.\nQuery: Why?"
    )
    assert format_profile_document(" Q? ", " A. ") == "Question: Q?\nResponse: A."
    with pytest.raises(ValueError, match="non-empty"):
        format_profile_query("  ")


def test_token_builder_right_truncates_and_appends_exactly_one_eos() -> None:
    tokenizer = FakeTokenizer(
        {
            "short": [10, 11, 2, 2, 2],
            "long": list(range(10, 610)),
        }
    )

    rows, metadata = build_profile_token_ids(
        tokenizer, ("short", "long"), max_length=512
    )

    assert rows[0] == [10, 11, 2]
    assert len(rows[1]) == 512 and rows[1][-1] == 2
    assert rows[1][:-1] == list(range(10, 521))
    assert metadata["maximum_untruncated_tokens"] == 601
    assert metadata["maximum_encoded_tokens"] == 512
    assert metadata["truncated_input_count"] == 1
    assert metadata["terminal_token_id"] == 2


def test_token_builder_emits_per_document_truncation_audit() -> None:
    tokenizer = FakeTokenizer(
        {
            "partial": list(range(700)),
            "partial-prefix": list(range(40)),
            "full": list(range(700)),
            "full-prefix": list(range(511)),
        }
    )
    _, metadata = build_profile_token_ids(
        tokenizer,
        ("partial", "full"),
        response_prefixes=("partial-prefix", "full-prefix"),
    )
    assert metadata["response_truncation"] == ["partial", "full"]


def test_token_builder_validates_frozen_tokenizer_contract() -> None:
    tokenizer = FakeTokenizer({"x": [1]})
    tokenizer.padding_side = "right"
    with pytest.raises(ValueError, match="left padding"):
        build_profile_token_ids(tokenizer, ("x",), max_length=512)
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    with pytest.raises(ValueError, match="right truncation"):
        build_profile_token_ids(tokenizer, ("x",), max_length=512)
    tokenizer.truncation_side = "right"
    tokenizer.eos_token_id = 3
    with pytest.raises(ValueError, match="EOS token ID 2"):
        build_profile_token_ids(tokenizer, ("x",), max_length=512)


def test_persisted_embeddings_are_float32_contiguous_normalized() -> None:
    values = np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float16)[:, ::-1]
    result = normalize_persisted_embeddings(values)
    assert result.dtype == np.float32
    assert result.flags.c_contiguous
    np.testing.assert_allclose(np.linalg.norm(result, axis=1), 1.0, atol=1e-7)


def test_runtime_options_are_not_tunable() -> None:
    validate_profile_runtime_options(batch_size=PROFILE_BATCH_SIZE, seed=PROFILE_SEED)
    with pytest.raises(ValueError, match="batch size"):
        validate_profile_runtime_options(batch_size=16, seed=PROFILE_SEED)
    with pytest.raises(ValueError, match="seed"):
        validate_profile_runtime_options(batch_size=PROFILE_BATCH_SIZE, seed=7)


def test_runtime_identity_requires_exact_torch_hip_and_transformers() -> None:
    torch = SimpleNamespace(
        __version__="2.11.0+gitd0c8b1f", version=SimpleNamespace(hip="7.2.53211")
    )
    validate_profile_runtime_identity(torch, transformers_version="5.12.1")
    with pytest.raises(RuntimeError, match="Torch version"):
        validate_profile_runtime_identity(
            SimpleNamespace(__version__="other", version=torch.version),
            transformers_version="5.12.1",
        )
    with pytest.raises(RuntimeError, match="Transformers version"):
        validate_profile_runtime_identity(torch, transformers_version="other")


def test_device_identity_requires_exact_gfx1151_architecture() -> None:
    validate_profile_device_identity(SimpleNamespace(gcnArchName="gfx1151"))
    with pytest.raises(RuntimeError, match="gfx1151"):
        validate_profile_device_identity(SimpleNamespace(gcnArchName="gfx1100"))


def test_formatted_input_hash_is_length_framed() -> None:
    from trec_mlm.profile_inference import hash_formatted_inputs

    expected = hashlib.sha256()
    for text in ("ab", "c"):
        encoded = text.encode()
        expected.update(len(encoded).to_bytes(8, "big"))
        expected.update(encoded)
    assert hash_formatted_inputs(("ab", "c")) == expected.hexdigest()
