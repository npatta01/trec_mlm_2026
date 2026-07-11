from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np

from trec_mlm.dense_experiment import (
    DenseExperimentConfig,
    exact_nested_loo_inner_scores,
    run_nested_dense_experiment,
    write_dense_experiment_artifacts,
)
from trec_mlm.dense_query_knn import EncoderSpec, rank_target_from_similarity
from trec_mlm.experiment import ExperimentConfig, run_leave_one_out_experiment
from trec_mlm.metrics import ndcg_at_k, reciprocal_rank
from trec_mlm.runio import validate_trec_run


QUERY_IDS = ("a", "b", "c", "d", "e")
MODEL_IDS = ("m1", "m2", "m3")
GRADES = np.asarray(
    [
        [2.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
    ]
)
SIMILARITIES = np.asarray(
    [
        [1.0, 0.9, 0.8, 0.7, -0.2],
        [0.9, 1.0, 0.6, 0.5, -0.3],
        [0.8, 0.6, 1.0, 0.4, -0.4],
        [0.7, 0.5, 0.4, 1.0, -0.5],
        [-0.2, -0.3, -0.4, -0.5, 1.0],
    ],
    dtype=np.float32,
)


def naive_inner_scores(k: int) -> tuple[np.ndarray, np.ndarray]:
    ndcg_values = np.zeros(len(QUERY_IDS), dtype=np.float64)
    mrr_values = np.zeros(len(QUERY_IDS), dtype=np.float64)
    for outer_index in range(len(QUERY_IDS)):
        outer_ndcg: list[float] = []
        outer_rr: list[float] = []
        for inner_index in range(len(QUERY_IDS)):
            if inner_index == outer_index:
                continue
            prediction = rank_target_from_similarity(
                target_index=inner_index,
                additionally_excluded_indices=(outer_index,),
                similarities=SIMILARITIES,
                grades=GRADES,
                query_ids=QUERY_IDS,
                model_ids=MODEL_IDS,
                k=k,
            )
            ranking = [MODEL_IDS[index] for index in prediction.model_order]
            qrels = dict(zip(MODEL_IDS, GRADES[inner_index], strict=True))
            outer_ndcg.append(ndcg_at_k(ranking, qrels, k=10))
            outer_rr.append(reciprocal_rank(ranking, qrels))
        ndcg_values[outer_index] = sum(outer_ndcg) / len(outer_ndcg)
        mrr_values[outer_index] = sum(outer_rr) / len(outer_rr)
    return ndcg_values, mrr_values


class ExactNestedScoringTest(unittest.TestCase):
    def test_vectorized_scores_match_naive_materialized_folds(self) -> None:
        for k in (1, 2, 4):
            with self.subTest(k=k):
                expected_ndcg, expected_mrr = naive_inner_scores(k)
                actual_ndcg, actual_mrr = exact_nested_loo_inner_scores(
                    similarities=SIMILARITIES,
                    grades=GRADES,
                    query_ids=QUERY_IDS,
                    model_ids=MODEL_IDS,
                    k=k,
                )
                np.testing.assert_allclose(actual_ndcg, expected_ndcg, atol=1e-12)
                np.testing.assert_allclose(actual_mrr, expected_mrr, atol=1e-12)

    def test_neighbor_replacement_and_negative_similarity_fallback(self) -> None:
        expected = {
            1: (2, 3),  # exclude b: c,d replace the original b,c prefix
            2: (1, 3),  # exclude c: b,d
            3: (1, 2),  # exclude d outside the original b,c prefix
        }
        for outer_index, neighbors in expected.items():
            prediction = rank_target_from_similarity(
                target_index=0,
                additionally_excluded_indices=(outer_index,),
                similarities=SIMILARITIES,
                grades=GRADES,
                query_ids=QUERY_IDS,
                model_ids=MODEL_IDS,
                k=2,
            )
            self.assertEqual(neighbors, prediction.neighbor_indices)

        fallback = rank_target_from_similarity(
            target_index=4,
            additionally_excluded_indices=(0,),
            similarities=SIMILARITIES,
            grades=GRADES,
            query_ids=QUERY_IDS,
            model_ids=MODEL_IDS,
            k=4,
        )
        self.assertEqual((), fallback.neighbor_indices)
        np.testing.assert_allclose(fallback.scores, fallback.global_prior)


class NestedExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.query_path = root / "queries.tsv"
        self.qrels_path = root / "qrels.txt"
        self.query_path.write_text(
            "".join(f"{query_id}\tquery {query_id}\n" for query_id in QUERY_IDS),
            encoding="utf-8",
        )
        self._write_qrels(GRADES)
        self.specs = (
            EncoderSpec("encoder_a", "test/encoder-a", "a" * 40),
            EncoderSpec("encoder_b", "test/encoder-b", "b" * 40),
        )
        # Two duplicate encoders make the declared encoder-order tie observable.
        self.embeddings = {
            "encoder_a": np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0],
                    [0.7, 0.7, 0.0],
                    [0.1, 0.9, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
            "encoder_b": np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [0.9, 0.1, 0.0],
                    [0.7, 0.7, 0.0],
                    [0.1, 0.9, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            ),
        }

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _write_qrels(self, grades: np.ndarray) -> None:
        self.qrels_path.write_text(
            "".join(
                f"{query_id} 0 {model_id} {int(grades[q, m])}\n"
                for q, query_id in enumerate(QUERY_IDS)
                for m, model_id in enumerate(MODEL_IDS)
            ),
            encoding="utf-8",
        )

    def _run(self):
        return run_nested_dense_experiment(
            DenseExperimentConfig(
                query_path=str(self.query_path),
                qrels_path=str(self.qrels_path),
                encoder_specs=self.specs,
                neighbor_counts=(1, 2),
                bootstrap_resamples=20,
                torch_threads=None,
            ),
            precomputed_embeddings=self.embeddings,
        )

    def test_exact_config_ties_prefer_declared_encoder_order(self) -> None:
        result = self._run()

        self.assertTrue(
            all(
                selected.startswith("encoder_a@")
                for selected in result.selected_config_by_outer.values()
            )
        )
        self.assertEqual(
            result.rankings["global_prior"],
            {
                query_id: sorted(
                    MODEL_IDS,
                    key=lambda model_id: (
                        -sum(
                            GRADES[index, MODEL_IDS.index(model_id)]
                            for index in range(len(QUERY_IDS))
                            if QUERY_IDS[index] != query_id
                        ),
                        model_id,
                    ),
                )
                for query_id in QUERY_IDS
            },
        )

    def test_outer_qrel_mutation_cannot_change_that_outer_prediction(self) -> None:
        original = self._run()
        target_query = QUERY_IDS[0]

        mutated_grades = GRADES.copy()
        mutated_grades[0] = [0.0, 2.0, 0.0]
        self._write_qrels(mutated_grades)
        mutated = self._run()

        self.assertEqual(
            original.selected_config_by_outer[target_query],
            mutated.selected_config_by_outer[target_query],
        )
        self.assertEqual(
            original.rankings["dense_nested"][target_query],
            mutated.rankings["dense_nested"][target_query],
        )
        np.testing.assert_allclose(
            original.selected_rankings[target_query].scores,
            mutated.selected_rankings[target_query].scores,
        )
        np.testing.assert_allclose(
            original.selected_rankings[target_query].global_prior,
            mutated.selected_rankings[target_query].global_prior,
        )

    def test_writer_emits_three_complete_valid_runs(self) -> None:
        result = self._run()
        output_dir = Path(self.directory.name) / "artifacts"

        paths = write_dense_experiment_artifacts(result, output_dir)

        for filename in (
            "run.trec",
            "tfidf_control.trec",
            "global_prior_control.trec",
        ):
            validation = validate_trec_run(
                paths[filename],
                expected_query_ids=QUERY_IDS,
                expected_model_ids=MODEL_IDS,
            )
            self.assertEqual(15, validation["row_count"])
        self.assertTrue(Path(paths["inner_scores.tsv"]).is_file())
        self.assertTrue(Path(paths["embeddings/encoder_a.npy"]).is_file())


class MergedControlRegressionTest(unittest.TestCase):
    def test_official_development_controls_match_merged_report(self) -> None:
        root = Path(__file__).resolve().parents[1]
        result = run_leave_one_out_experiment(
            ExperimentConfig(
                query_path=str(root / "data/dev/llm_dev_data.tsv"),
                qrels_path=str(root / "data/dev/llm_dev_qrels.txt"),
                neighbor_counts=(10,),
                aggregations=("weighted_mean",),
                primary_neighbors=10,
                bootstrap_resamples=20,
            )
        )

        dense_control = result.evaluations["weighted_mean@10"]
        prior_control = result.evaluations["global_prior"]
        self.assertEqual(0.2177, round(dense_control.macro_ndcg_at_k, 4))
        self.assertEqual(0.2777, round(dense_control.mean_reciprocal_rank, 4))
        self.assertEqual(0.1385, round(prior_control.macro_ndcg_at_k, 4))
        self.assertEqual(0.2067, round(prior_control.mean_reciprocal_rank, 4))


if __name__ == "__main__":
    unittest.main()
