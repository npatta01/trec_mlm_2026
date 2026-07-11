from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
import unittest

import numpy as np

from trec_mlm.experiment import ExperimentConfig, run_leave_one_out_experiment
from trec_mlm.metrics import QueryMetrics, ndcg_at_k, reciprocal_rank
from trec_mlm.response_experiment import (
    ResponseExperimentConfig,
    exact_nested_response_inner_scores,
    fold_local_prior,
    rank_model_scores,
    run_nested_response_experiment,
    write_response_experiment_artifacts,
)
from trec_mlm.response_retrieval import ResponseCandidate
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
    ],
    dtype=np.float64,
)
SCORES_A = np.asarray(
    [
        [3.0, 0.0, 1.0],
        [0.0, 4.0, 1.0],
        [1.0, 0.0, 2.0],
        [0.0, 1.0, 2.0],
        [2.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
SCORES_B = np.asarray(
    [
        [0.0, 3.0, 1.0],
        [4.0, 0.0, 1.0],
        [0.0, 2.0, 1.0],
        [2.0, 1.0, 0.0],
        [0.0, 1.0, 2.0],
    ],
    dtype=np.float64,
)
CANDIDATES = (
    ResponseCandidate("bm25", "response_only", "basic", 100),
    ResponseCandidate("bm25_rm3", "query_response", "aggressive", 500),
)


@dataclass
class FakeScoreResult:
    candidates: tuple[ResponseCandidate, ...]
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    model_scores: np.ndarray
    hit_counts: np.ndarray
    mean_hit_scores: np.ndarray
    overlap_document_counts: np.ndarray
    inventory_filter_names: tuple[str, ...]
    retained_document_counts: np.ndarray
    retained_character_counts: np.ndarray
    retained_token_counts: np.ndarray
    corpus_sha256: str = "corpus"
    configuration_sha256: str = "configuration"


def fake_score_result(*, duplicate_candidates: bool = False) -> FakeScoreResult:
    scores = np.stack([SCORES_A, SCORES_A if duplicate_candidates else SCORES_B])
    hits = (scores > 0.0).astype(np.int32)
    return FakeScoreResult(
        candidates=CANDIDATES,
        query_ids=QUERY_IDS,
        model_ids=MODEL_IDS,
        model_scores=scores,
        hit_counts=hits,
        mean_hit_scores=scores.copy(),
        overlap_document_counts=np.zeros((2, len(QUERY_IDS)), dtype=np.int32),
        inventory_filter_names=("basic", "aggressive"),
        retained_document_counts=np.asarray([[10, 20, 30], [8, 16, 24]]),
        retained_character_counts=np.asarray([[100, 200, 300], [80, 160, 240]]),
        retained_token_counts=np.asarray([[20, 40, 60], [16, 32, 48]]),
    )


def naive_inner_scores(raw_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ndcg_values = np.zeros(len(QUERY_IDS), dtype=np.float64)
    mrr_values = np.zeros(len(QUERY_IDS), dtype=np.float64)
    for outer_index in range(len(QUERY_IDS)):
        outer_ndcg: list[float] = []
        outer_rr: list[float] = []
        for inner_index in range(len(QUERY_IDS)):
            if inner_index == outer_index:
                continue
            prior = fold_local_prior(GRADES, (outer_index, inner_index))
            order = rank_model_scores(raw_scores[inner_index], prior, MODEL_IDS)
            ranking = [MODEL_IDS[int(index)] for index in order]
            qrels = dict(zip(MODEL_IDS, GRADES[inner_index], strict=True))
            outer_ndcg.append(ndcg_at_k(ranking, qrels, k=10))
            outer_rr.append(reciprocal_rank(ranking, qrels))
        ndcg_values[outer_index] = sum(outer_ndcg) / len(outer_ndcg)
        mrr_values[outer_index] = sum(outer_rr) / len(outer_rr)
    return ndcg_values, mrr_values


class ExactNestedResponseScoringTest(unittest.TestCase):
    def test_vectorized_scores_match_naive_materialized_oracle(self) -> None:
        for scores in (SCORES_A, SCORES_B, np.zeros_like(SCORES_A)):
            with self.subTest(scores=scores.tolist()):
                expected_ndcg, expected_mrr = naive_inner_scores(scores)
                actual_ndcg, actual_mrr = exact_nested_response_inner_scores(
                    raw_scores=scores,
                    grades=GRADES,
                    query_ids=QUERY_IDS,
                    model_ids=MODEL_IDS,
                )
                np.testing.assert_allclose(actual_ndcg, expected_ndcg, atol=1e-12)
                np.testing.assert_allclose(actual_mrr, expected_mrr, atol=1e-12)

    def test_inner_prior_excludes_both_outer_and_inner_rows(self) -> None:
        prior = fold_local_prior(GRADES, (0, 1))
        np.testing.assert_allclose(prior, GRADES[[2, 3, 4]].mean(axis=0))

        # With raw ties, the hand-calculated prior controls the exact order.
        order = rank_model_scores(np.zeros(3), prior, MODEL_IDS)
        expected = sorted(
            MODEL_IDS,
            key=lambda model_id: (-prior[MODEL_IDS.index(model_id)], model_id),
        )
        self.assertEqual(expected, [MODEL_IDS[int(index)] for index in order])

    def test_raw_ties_then_prior_ties_use_model_id(self) -> None:
        order = rank_model_scores(
            np.asarray([1.0, 1.0, 1.0]),
            np.asarray([0.5, 0.7, 0.7]),
            ("z", "b", "a"),
        )
        self.assertEqual(["a", "b", "z"], [("z", "b", "a")[i] for i in order])


class NestedResponseExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.query_path = root / "queries.tsv"
        self.qrels_path = root / "qrels.txt"
        self.discovery_path = root / "discovery.json"
        self.query_path.write_text(
            "".join(f"{query_id}\tquery {query_id}\n" for query_id in QUERY_IDS),
            encoding="utf-8",
        )
        self.discovery_path.write_text("{}\n", encoding="utf-8")
        self._write_qrels(GRADES)

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

    def _config(self) -> ResponseExperimentConfig:
        return ResponseExperimentConfig(
            query_path=str(self.query_path),
            qrels_path=str(self.qrels_path),
            discovery_path=str(self.discovery_path),
            candidates=CANDIDATES,
            dense_per_query_path=None,
            bootstrap_resamples=20,
        )

    @staticmethod
    def _dense_reference() -> dict[str, QueryMetrics]:
        return {query_id: QueryMetrics(0.25, 0.5) for query_id in QUERY_IDS}

    def _run(self, *, duplicate_candidates: bool = False):
        return run_nested_response_experiment(
            self._config(),
            score_result=fake_score_result(duplicate_candidates=duplicate_candidates),
            dense_per_query_reference=self._dense_reference(),
        )

    def test_declared_candidate_order_resolves_exact_selection_ties(self) -> None:
        result = self._run(duplicate_candidates=True)
        first_key = CANDIDATES[0].key
        self.assertEqual(
            {first_key},
            set(result.selected_config_by_outer.values()),
        )

    def test_outer_qrel_mutation_cannot_change_outer_selection_or_prediction(
        self,
    ) -> None:
        original = self._run()
        target_query = QUERY_IDS[0]

        mutated = GRADES.copy()
        mutated[0] = [0.0, 2.0, 0.0]
        self._write_qrels(mutated)
        rerun = self._run()

        self.assertEqual(
            original.selected_config_by_outer[target_query],
            rerun.selected_config_by_outer[target_query],
        )
        self.assertEqual(
            original.rankings["response_nested"][target_query],
            rerun.rankings["response_nested"][target_query],
        )
        self.assertEqual(
            original.selected_raw_scores[target_query],
            rerun.selected_raw_scores[target_query],
        )
        self.assertEqual(
            original.selected_priors[target_query],
            rerun.selected_priors[target_query],
        )

    def test_every_system_ranking_is_complete_and_controls_match_fixture(self) -> None:
        result = self._run()
        direct = run_leave_one_out_experiment(
            ExperimentConfig(
                query_path=str(self.query_path),
                qrels_path=str(self.qrels_path),
                neighbor_counts=(10,),
                aggregations=("weighted_mean",),
                primary_neighbors=10,
                bootstrap_resamples=20,
            )
        )

        self.assertEqual(
            direct.rankings["weighted_mean@10"],
            result.rankings["tfidf_fixed_k10"],
        )
        self.assertEqual(
            direct.rankings["global_prior"],
            result.rankings["global_prior"],
        )
        for system_rankings in result.rankings.values():
            self.assertEqual(set(QUERY_IDS), set(system_rankings))
            for ranking in system_rankings.values():
                self.assertEqual(set(MODEL_IDS), set(ranking))
                self.assertEqual(len(MODEL_IDS), len(ranking))

    def test_writer_emits_complete_valid_runs_and_nested_evidence(self) -> None:
        result = self._run()
        output_dir = Path(self.directory.name) / "artifacts"

        paths = write_response_experiment_artifacts(result, output_dir)

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
        for filename in (
            "inner_scores.tsv",
            "selections.tsv",
            "predictions.tsv",
            "per_query.tsv",
            "diagnostics.json",
            "validation.json",
        ):
            self.assertTrue(Path(paths[filename]).is_file())


if __name__ == "__main__":
    unittest.main()
