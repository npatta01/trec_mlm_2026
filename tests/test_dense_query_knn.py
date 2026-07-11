from __future__ import annotations

import unittest

import numpy as np

from trec_mlm.dense_query_knn import (
    DEFAULT_ENCODERS,
    cosine_similarity_matrix,
    normalize_embeddings,
    rank_target_from_similarity,
)


class DenseEmbeddingTest(unittest.TestCase):
    def test_normalizes_rows_and_preserves_zero_fallback_row(self) -> None:
        embeddings = np.asarray([[3.0, 4.0], [0.0, 0.0], [4.0, -3.0]])

        normalized = normalize_embeddings(embeddings)
        similarities = cosine_similarity_matrix(embeddings)

        np.testing.assert_allclose(normalized[0], [0.6, 0.8])
        np.testing.assert_allclose(normalized[1], [0.0, 0.0])
        np.testing.assert_allclose(np.diag(similarities), [1.0, 0.0, 1.0])
        np.testing.assert_allclose(similarities, similarities.T)

    def test_zero_similarity_target_uses_training_prior(self) -> None:
        similarities = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.2],
                [0.0, 0.2, 1.0],
            ],
            dtype=np.float32,
        )
        grades = np.asarray(
            [
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
                [0.0, 1.0, 2.0],
            ]
        )

        prediction = rank_target_from_similarity(
            target_index=0,
            additionally_excluded_indices=(),
            similarities=similarities,
            grades=grades,
            query_ids=("a", "b", "c"),
            model_ids=("m1", "m2", "m3"),
            k=2,
        )

        self.assertEqual((), prediction.neighbor_indices)
        np.testing.assert_allclose(prediction.scores, prediction.global_prior)
        self.assertEqual((1, 2, 0), prediction.model_order)

    def test_outer_exclusion_promotes_the_next_positive_neighbor(self) -> None:
        similarities = np.asarray(
            [
                [1.0, 0.9, 0.8, 0.7],
                [0.9, 1.0, 0.6, 0.5],
                [0.8, 0.6, 1.0, 0.4],
                [0.7, 0.5, 0.4, 1.0],
            ],
            dtype=np.float32,
        )
        grades = np.asarray(
            [
                [2.0, 0.0, 1.0],
                [0.0, 2.0, 1.0],
                [1.0, 0.0, 2.0],
                [0.0, 1.0, 2.0],
            ]
        )

        prediction = rank_target_from_similarity(
            target_index=0,
            additionally_excluded_indices=(1,),
            similarities=similarities,
            grades=grades,
            query_ids=("a", "b", "c", "d"),
            model_ids=("m1", "m2", "m3"),
            k=2,
        )

        self.assertEqual((2, 3), prediction.neighbor_indices)
        np.testing.assert_allclose(prediction.neighbor_similarities, (0.8, 0.7))

    def test_default_grid_is_pinned_and_e5_uses_symmetric_query_prefix(self) -> None:
        self.assertEqual(4, len(DEFAULT_ENCODERS))
        self.assertTrue(all(len(spec.revision) == 40 for spec in DEFAULT_ENCODERS))
        prefixes = {spec.key: spec.query_prefix for spec in DEFAULT_ENCODERS}
        self.assertEqual("query: ", prefixes["e5_small_v2"])
        self.assertEqual("", prefixes["bge_small_en_v1_5"])


if __name__ == "__main__":
    unittest.main()
