from __future__ import annotations

import unittest

import numpy as np

from trec_mlm.label_retrieval import (
    CONTEXTUAL_E5_SPEC,
    TASK_INSTRUCTION,
    clean_candidate_label,
    format_document,
    format_query,
    normalize_embedding_rows,
    rank_label_scores,
)


class ContextualE5ContractTest(unittest.TestCase):
    def test_only_frozen_contextual_e5_contract_is_exposed(self) -> None:
        self.assertEqual("e5_mistral_7b_contextual", CONTEXTUAL_E5_SPEC.key)
        self.assertEqual(
            "intfloat/e5-mistral-7b-instruct", CONTEXTUAL_E5_SPEC.model_id
        )
        self.assertEqual(
            "d62cc4417be0e3289c03dd459d1ef2d137e2554e",
            CONTEXTUAL_E5_SPEC.revision,
        )
        self.assertEqual("MIT", CONTEXTUAL_E5_SPEC.license)
        self.assertEqual("bfloat16", CONTEXTUAL_E5_SPEC.dtype)
        self.assertEqual(4096, CONTEXTUAL_E5_SPEC.embedding_dimension)
        self.assertEqual("last_token_normalized", CONTEXTUAL_E5_SPEC.pooling)
        self.assertEqual(512, CONTEXTUAL_E5_SPEC.max_length)

    def test_query_and_capability_document_formats_are_exact(self) -> None:
        question = " Who founded autosegmental phonology? "

        self.assertEqual(
            f"Instruct: {TASK_INSTRUCTION}\n"
            "Query: Who founded autosegmental phonology?",
            format_query(question),
        )
        self.assertEqual(
            "This LLM is most likely to correctly answer questions about "
            "Industries economics.",
            format_document("Industries_(economics)"),
        )

    def test_candidate_label_cleaning_is_frozen(self) -> None:
        self.assertEqual(
            "Industries economics", clean_candidate_label("Industries_(economics)")
        )
        self.assertEqual("Food and drink", clean_candidate_label("Food_and_drink"))
        with self.assertRaisesRegex(ValueError, "non-empty"):
            clean_candidate_label("___")


class LabelScoringTest(unittest.TestCase):
    def test_normalization_requires_finite_nonzero_rows(self) -> None:
        values = normalize_embedding_rows(np.asarray([[3.0, 4.0], [4.0, -3.0]]))

        np.testing.assert_allclose(values, [[0.6, 0.8], [0.8, -0.6]])
        self.assertEqual(np.float32, values.dtype)
        with self.assertRaisesRegex(ValueError, "zero"):
            normalize_embedding_rows(np.asarray([[0.0, 0.0]]))
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_embedding_rows(np.asarray([[np.nan, 1.0]]))

    def test_cosine_rankings_use_candidate_id_for_exact_ties(self) -> None:
        result = rank_label_scores(
            query_embeddings=np.asarray([[1.0, 0.0], [0.0, 1.0]]),
            label_embeddings=np.asarray(
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
            ),
            query_ids=("q1", "q2"),
            candidate_ids=("beta", "alpha", "gamma"),
        )

        self.assertEqual(["alpha", "beta", "gamma"], result.rankings["q1"])
        self.assertEqual(["gamma", "alpha", "beta"], result.rankings["q2"])
        np.testing.assert_allclose(result.scores[0], [1.0, 1.0, 0.0])

    def test_scoring_rejects_shape_and_inventory_mismatches(self) -> None:
        with self.assertRaisesRegex(ValueError, "unique"):
            rank_label_scores(
                query_embeddings=np.ones((2, 2)),
                label_embeddings=np.ones((1, 2)),
                query_ids=("q", "q"),
                candidate_ids=("a",),
            )
        with self.assertRaisesRegex(ValueError, "dimensions"):
            rank_label_scores(
                query_embeddings=np.ones((1, 2)),
                label_embeddings=np.ones((1, 3)),
                query_ids=("q",),
                candidate_ids=("a",),
            )


if __name__ == "__main__":
    unittest.main()
