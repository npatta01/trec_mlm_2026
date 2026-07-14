from __future__ import annotations

import unittest
from unittest import mock

import numpy as np
import torch

import trec_mlm.label_inference as label_inference
from trec_mlm.label_inference import (
    ROCM_IMAGE,
    build_e5_token_ids,
    encode_label_inputs,
    last_token_pool,
    snapshot_revision_from_path,
)


class E5TokenContractTest(unittest.TestCase):
    class _Tokenizer:
        eos_token_id = 2

        def __call__(
            self,
            texts,
            *,
            add_special_tokens,
            padding,
            truncation,
            max_length=None,
        ):
            del add_special_tokens, padding
            rows = [
                [1, *range(10, 10 + len(text))]
                + ([self.eos_token_id] if text.startswith("!") else [])
                for text in texts
            ]
            if truncation:
                rows = [row[:max_length] for row in rows]
            return {"input_ids": rows}

    def test_eos_is_appended_once_with_space_reserved_for_it(self) -> None:
        sequences, metadata = build_e5_token_ids(
            self._Tokenizer(), ["ab", "!a", "abcdefghij"], max_length=8
        )

        self.assertTrue(all(row[-1] == 2 for row in sequences))
        self.assertEqual(1, sequences[1].count(2))
        self.assertLessEqual(max(map(len, sequences)), 8)
        self.assertEqual(1, metadata["truncated_input_count"])
        self.assertEqual(2, metadata["terminal_token_id"])

    def test_missing_eos_token_is_rejected(self) -> None:
        tokenizer = self._Tokenizer()
        tokenizer.eos_token_id = None

        with self.assertRaisesRegex(ValueError, "EOS"):
            build_e5_token_ids(tokenizer, ["text"], max_length=8)


class LastTokenPoolingTest(unittest.TestCase):
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


class E5EncodingDispatchTest(unittest.TestCase):
    def test_rocm_container_is_digest_pinned(self) -> None:
        self.assertEqual(
            "vllm/vllm-openai-rocm@sha256:"
            "3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28",
            ROCM_IMAGE,
        )

    def test_snapshot_revision_is_recovered_from_resolved_tokenizer_path(self) -> None:
        revision = "d62cc4417be0e3289c03dd459d1ef2d137e2554e"

        self.assertEqual(
            revision,
            snapshot_revision_from_path(
                "/cache/models--intfloat--e5-mistral-7b-instruct/"
                f"snapshots/{revision}/tokenizer_config.json"
            ),
        )
        with self.assertRaisesRegex(ValueError, "snapshot"):
            snapshot_revision_from_path("/cache/tokenizer_config.json")

    def test_fixed_backend_encodes_once_and_splits_queries_from_labels(self) -> None:
        encoded = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float32
        )
        with mock.patch.object(
            label_inference,
            "_encode_contextual_e5",
            return_value=(encoded, {"backend": "fixture"}),
        ) as backend:
            queries, labels, metadata = encode_label_inputs(
                formatted_queries=("query one", "query two"),
                formatted_documents=("label",),
                batch_size=2,
                seed=7,
            )

        backend.assert_called_once_with(
            ("query one", "query two", "label"), batch_size=2, seed=7
        )
        np.testing.assert_array_equal(queries, encoded[:2])
        np.testing.assert_array_equal(labels, encoded[2:])
        self.assertEqual("fixture", metadata["backend"])

    def test_empty_inputs_and_invalid_batch_size_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "query"):
            encode_label_inputs(
                formatted_queries=(),
                formatted_documents=("label",),
                batch_size=1,
                seed=7,
            )
        with self.assertRaisesRegex(ValueError, "batch_size"):
            encode_label_inputs(
                formatted_queries=("query",),
                formatted_documents=("label",),
                batch_size=0,
                seed=7,
            )


if __name__ == "__main__":
    unittest.main()
