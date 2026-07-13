from __future__ import annotations

from dataclasses import dataclass
import unittest
from unittest import mock

import numpy as np
import torch

import trec_mlm.label_inference as label_inference

from trec_mlm.label_inference import (
    ROCM_IMAGE,
    cls_token_pool,
    embedding_outputs_to_matrix,
    build_e5_token_ids,
    build_qwen_token_ids,
    last_token_pool,
    make_token_prompts,
    mean_token_pool,
    pool_encoder_hidden_states,
    token_length_summary,
    torch_dtype_for_spec,
    validate_embedding_smoke,
    vllm_runtime_kwargs,
)
from trec_mlm.label_retrieval import encoder_spec_by_key


class LastTokenPoolingTest(unittest.TestCase):
    class _Tokenizer:
        eos_token_id = 2

        def convert_tokens_to_ids(self, token):
            return {"<|endoftext|>": 99}[token]

        def __call__(
            self,
            texts,
            *,
            add_special_tokens,
            padding,
            truncation,
            max_length=None,
        ):
            rows = [[1, *range(10, 10 + len(text))] for text in texts]
            if truncation:
                rows = [row[:max_length] for row in rows]
            return {"input_ids": rows, "attention_mask": [[1] * len(r) for r in rows]}

    def test_qwen_token_ids_reserve_space_and_append_endoftext(self) -> None:
        sequences, metadata = build_qwen_token_ids(
            self._Tokenizer(), ["ab", "abcdefghij"], max_length=8
        )

        self.assertEqual(99, sequences[0][-1])
        self.assertEqual(99, sequences[1][-1])
        self.assertLessEqual(len(sequences[1]), 7)
        self.assertEqual(1, metadata["truncated_input_count"])
        self.assertEqual(99, metadata["terminal_token_id"])

    def test_e5_token_ids_reserve_space_and_append_eos_once(self) -> None:
        sequences, metadata = build_e5_token_ids(
            self._Tokenizer(), ["ab", "abcdefghij"], max_length=8
        )

        self.assertEqual(2, sequences[0][-1])
        self.assertEqual(2, sequences[1][-1])
        self.assertLessEqual(len(sequences[1]), 8)
        self.assertEqual(1, metadata["truncated_input_count"])
        self.assertEqual(2, metadata["terminal_token_id"])

    def test_rocm_image_is_digest_pinned(self) -> None:
        self.assertEqual(
            "vllm/vllm-openai-rocm@sha256:"
            "3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28",
            ROCM_IMAGE,
        )

    def test_token_length_summary_counts_truncated_inputs(self) -> None:
        class Tokenizer:
            def __call__(self, texts, **kwargs):
                self.kwargs = kwargs
                return {"input_ids": [[1, 2], [1, 2, 3, 4, 5]]}

        tokenizer = Tokenizer()
        summary = token_length_summary(tokenizer, ["short", "long"], max_length=4)

        self.assertEqual(
            {"maximum_untruncated_tokens": 5, "truncated_input_count": 1},
            summary,
        )
        self.assertFalse(tokenizer.kwargs["truncation"])

    def test_right_padding_selects_last_unmasked_token(self) -> None:
        hidden = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0], [9.0, 9.0]],
                [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        pooled = last_token_pool(hidden, mask)

        torch.testing.assert_close(pooled, torch.tensor([[2.0, 0.0], [5.0, 0.0]]))


class EncoderPoolingTest(unittest.TestCase):
    def test_registered_float32_dtype_maps_to_torch(self) -> None:
        self.assertIs(
            torch.float32, torch_dtype_for_spec(encoder_spec_by_key("bge_m3"))
        )

    def test_cls_pool_returns_first_token(self) -> None:
        hidden = torch.tensor([[[1.0, 2.0], [9.0, 9.0]], [[3.0, 4.0], [8.0, 8.0]]])

        pooled = cls_token_pool(hidden)

        torch.testing.assert_close(pooled, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_mean_pool_excludes_masked_padding(self) -> None:
        hidden = torch.tensor(
            [
                [[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]],
                [[2.0, 6.0], [8.0, 4.0], [4.0, 2.0]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 1, 1]])

        pooled = mean_token_pool(hidden, mask)

        torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0], [14.0 / 3, 4.0]]))

    def test_nomic_pool_layer_normalizes_then_l2_normalizes(self) -> None:
        hidden = torch.tensor([[[1.0, 3.0, 8.0], [3.0, 5.0, 4.0]]])
        mask = torch.tensor([[1, 1]])

        pooled = pool_encoder_hidden_states(
            hidden,
            mask,
            pooling="mean_layernorm_normalized",
        )

        expected = torch.nn.functional.layer_norm(
            torch.tensor([[2.0, 4.0, 6.0]]),
            normalized_shape=(3,),
        )
        expected = torch.nn.functional.normalize(expected, p=2, dim=1)
        torch.testing.assert_close(pooled, expected)

    def test_mean_pool_rejects_empty_attention_row(self) -> None:
        with self.assertRaisesRegex(ValueError, "unmasked"):
            mean_token_pool(torch.ones((1, 2, 3)), torch.zeros((1, 2)))

    def test_bge_runtime_dispatches_to_generic_transformers_encoder(self) -> None:
        encoded = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        with mock.patch.object(
            label_inference,
            "_encode_transformers_encoder",
            return_value=(encoded, {"backend": "fixture"}),
        ) as encoder:
            queries, documents, metadata = label_inference.encode_label_inputs(
                encoder_spec_by_key("bge_m3"),
                formatted_queries=("query",),
                formatted_documents=("document",),
                batch_size=2,
                seed=7,
            )

        encoder.assert_called_once()
        np.testing.assert_array_equal(queries, encoded[:1])
        np.testing.assert_array_equal(documents, encoded[1:])
        self.assertEqual("fixture", metadata["backend"])

    def test_left_padding_selects_final_position(self) -> None:
        hidden = torch.tensor(
            [
                [[9.0, 9.0], [1.0, 0.0], [2.0, 0.0]],
                [[3.0, 0.0], [4.0, 0.0], [5.0, 0.0]],
            ]
        )
        mask = torch.tensor([[0, 1, 1], [1, 1, 1]])

        pooled = last_token_pool(hidden, mask)

        torch.testing.assert_close(pooled, torch.tensor([[2.0, 0.0], [5.0, 0.0]]))


@dataclass
class _Payload:
    embedding: list[float]


@dataclass
class _Output:
    outputs: _Payload


class VllmOutputTest(unittest.TestCase):
    def test_token_prompts_preserve_explicit_terminal_ids(self) -> None:
        @dataclass
        class Prompt:
            prompt_token_ids: list[int]

        prompts = make_token_prompts([[1, 2, 99], [3, 99]], Prompt)

        self.assertEqual([1, 2, 99], prompts[0].prompt_token_ids)
        self.assertEqual([3, 99], prompts[1].prompt_token_ids)

    def test_vllm_runtime_forces_deterministic_triton_attention(self) -> None:
        options = vllm_runtime_kwargs(
            encoder_spec_by_key("qwen3_embedding_8b"),
            seed=2026,
        )

        self.assertEqual({"backend": "TRITON_ATTN"}, options["attention_config"])
        self.assertEqual(2026, options["seed"])

    def test_embedding_outputs_become_finite_normalized_matrix(self) -> None:
        outputs = [
            _Output(_Payload([3.0, 4.0])),
            _Output(_Payload([4.0, -3.0])),
        ]

        matrix = embedding_outputs_to_matrix(outputs, expected_dimension=2)

        self.assertEqual(np.float32, matrix.dtype)
        np.testing.assert_allclose(matrix, [[0.6, 0.8], [0.8, -0.6]])

    def test_embedding_output_dimension_is_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "dimension"):
            embedding_outputs_to_matrix(
                [_Output(_Payload([1.0, 0.0]))],
                expected_dimension=3,
            )


class EmbeddingSmokeTest(unittest.TestCase):
    def test_canonical_scores_within_tolerance_pass(self) -> None:
        queries = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        documents = np.asarray([[0.8, 0.6], [0.0, 1.0]])
        reference = np.asarray([[0.8, 0.0], [0.6, 1.0]])

        receipt = validate_embedding_smoke(
            query_embeddings=queries,
            document_embeddings=documents,
            reference_scores=reference,
            expected_dimension=2,
            score_tolerance=1e-4,
        )

        self.assertTrue(receipt["passed"])
        self.assertLessEqual(receipt["maximum_absolute_score_difference"], 1e-4)

    def test_canonical_score_mismatch_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference"):
            validate_embedding_smoke(
                query_embeddings=np.eye(2),
                document_embeddings=np.eye(2),
                reference_scores=np.zeros((2, 2)),
                expected_dimension=2,
                score_tolerance=1e-4,
            )


if __name__ == "__main__":
    unittest.main()
