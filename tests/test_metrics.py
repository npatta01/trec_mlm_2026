import math
import unittest

from trec_mlm.metrics import (
    evaluate_rankings,
    macro_ndcg_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    paired_bootstrap_ci,
    per_query_metrics,
    reciprocal_rank,
)


class NdcgTest(unittest.TestCase):
    def test_ideal_ranking_scores_one(self) -> None:
        qrels = {"strong": 2, "partial": 1, "none": 0}
        self.assertAlmostEqual(ndcg_at_k(["strong", "partial", "none"], qrels), 1.0)

    def test_raw_graded_gain_and_log_discount_match_trec_eval(self) -> None:
        qrels = {"a": 2, "b": 1, "c": 0, "d": 2}
        ranking = ["b", "unjudged", "a", "d"]

        actual_dcg_at_3 = 1.0 + 0.0 / math.log2(3) + 2.0 / math.log2(4)
        ideal_dcg_at_3 = 2.0 + 2.0 / math.log2(3) + 1.0 / math.log2(4)

        self.assertAlmostEqual(
            ndcg_at_k(ranking, qrels, k=3),
            actual_dcg_at_3 / ideal_dcg_at_3,
        )

    def test_cutoff_excludes_later_relevant_items_from_dcg(self) -> None:
        qrels = {"a": 2, "b": 1}
        ranking = ["unjudged", "a", "b"]
        expected = (2.0 / math.log2(3)) / (2.0 + 1.0 / math.log2(3))
        self.assertAlmostEqual(ndcg_at_k(ranking, qrels, k=2), expected)

    def test_no_positive_qrels_returns_zero(self) -> None:
        self.assertEqual(ndcg_at_k(["a"], {"a": 0, "b": -1}), 0.0)

    def test_invalid_cutoff_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ndcg_at_k(["a"], {"a": 1}, k=0)

    def test_duplicate_ranked_document_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ndcg_at_k(["a", "a"], {"a": 2})


class ReciprocalRankTest(unittest.TestCase):
    def test_any_positive_grade_is_relevant(self) -> None:
        qrels = {"zero": 0, "partial": 1, "strong": 2}
        self.assertAlmostEqual(
            reciprocal_rank(["unjudged", "zero", "partial", "strong"], qrels),
            1.0 / 3.0,
        )

    def test_no_relevant_result_returns_zero(self) -> None:
        self.assertEqual(reciprocal_rank(["a", "b"], {"a": 0}), 0.0)


class MacroEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.qrels = {
            "q1": {"a": 2, "b": 0},
            "q2": {"c": 1},
        }

    def test_missing_query_ranking_contributes_zero(self) -> None:
        rankings = {"q1": ["a"]}
        metrics = per_query_metrics(rankings, self.qrels)

        self.assertEqual(metrics["q1"].ndcg_at_k, 1.0)
        self.assertEqual(metrics["q2"].ndcg_at_k, 0.0)
        self.assertEqual(macro_ndcg_at_k(rankings, self.qrels), 0.5)
        self.assertEqual(mean_reciprocal_rank(rankings, self.qrels), 0.5)

    def test_evaluate_rankings_returns_per_query_and_macro_metrics(self) -> None:
        result = evaluate_rankings({"q1": ["b", "a"], "q2": ["c"]}, self.qrels)

        self.assertEqual(result.cutoff, 10)
        self.assertEqual(set(result.per_query), {"q1", "q2"})
        self.assertAlmostEqual(result.per_query["q1"].reciprocal_rank, 0.5)
        self.assertAlmostEqual(result.mean_reciprocal_rank, 0.75)
        self.assertAlmostEqual(
            result.macro_ndcg_at_k,
            (1.0 / math.log2(3) + 1.0) / 2.0,
        )

    def test_empty_qrels_cannot_form_macro_average(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_rankings({}, {})


class PairedBootstrapTest(unittest.TestCase):
    def test_constant_improvement_has_constant_interval(self) -> None:
        baseline = {"q1": 0.1, "q2": 0.2, "q3": 0.3}
        candidate = {"q1": 0.2, "q2": 0.3, "q3": 0.4}

        interval = paired_bootstrap_ci(baseline, candidate, n_resamples=500, seed=7)

        self.assertAlmostEqual(interval.mean_difference, 0.1)
        self.assertAlmostEqual(interval.lower, 0.1)
        self.assertAlmostEqual(interval.upper, 0.1)
        self.assertEqual(interval.confidence_level, 0.95)
        self.assertEqual(interval.n_resamples, 500)

    def test_seed_makes_interval_reproducible(self) -> None:
        baseline = {"q1": 0.1, "q2": 0.4, "q3": 0.2, "q4": 0.8}
        candidate = {"q1": 0.3, "q2": 0.2, "q3": 0.7, "q4": 0.9}

        first = paired_bootstrap_ci(baseline, candidate, n_resamples=200, seed=11)
        second = paired_bootstrap_ci(baseline, candidate, n_resamples=200, seed=11)

        self.assertEqual(first, second)

    def test_query_id_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "identical query IDs"):
            paired_bootstrap_ci({"q1": 0.1}, {"q2": 0.2})

    def test_empty_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            paired_bootstrap_ci({}, {})


if __name__ == "__main__":
    unittest.main()
