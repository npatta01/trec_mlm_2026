from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from trec_mlm.query_knn import (
    QueryKNNRanker,
    WordTfidfVectorizer,
    leave_one_out_rankings,
    load_development_data,
    load_queries_tsv,
    ranking_model_ids,
)


class QueryKNNTest(unittest.TestCase):
    def setUp(self) -> None:
        self.queries = {
            "q1": "red apple orchard",
            "q2": "red berry fruit",
            "q3": "blue ocean water",
        }
        self.qrels = {
            "q1": {"model-a": 2.0, "model-b": 0.0, "model-c": 1.0},
            "q2": {"model-a": 0.0, "model-b": 2.0, "model-c": 1.0},
            "q3": {"model-a": 0.0, "model-b": 1.0, "model-c": 2.0},
        }

    def test_self_query_is_excluded_from_neighbors(self) -> None:
        ranker = QueryKNNRanker(k=3, ngram_range=(1, 1)).fit(
            self.queries, self.qrels
        )

        neighbors = ranker.nearest_queries(
            self.queries["q1"], exclude_query_id="q1"
        )

        self.assertNotIn("q1", [neighbor.query_id for neighbor in neighbors])
        self.assertEqual("q2", neighbors[0].query_id)

    def test_weighted_sum_and_mean_have_identical_complete_qrel_ranking(self) -> None:
        ranker = QueryKNNRanker(k=3, ngram_range=(1, 1)).fit(
            self.queries, self.qrels
        )

        weighted_sum = ranker.rank("red fruit", aggregation="weighted_sum")
        weighted_mean = ranker.rank("red fruit", aggregation="weighted_mean")

        # Complete qrels give every model the same neighbor-weight denominator,
        # so mean rescales all model scores by one positive constant.
        self.assertEqual(
            ranking_model_ids(weighted_sum), ranking_model_ids(weighted_mean)
        )
        denominator = sum(
            neighbor.similarity
            for neighbor in ranker.nearest_queries("red fruit", k=3)
        )
        by_model_sum = {item.model_id: item.score for item in weighted_sum}
        by_model_mean = {item.model_id: item.score for item in weighted_mean}
        for model_id in ranker.models:
            self.assertAlmostEqual(
                by_model_sum[model_id] / denominator,
                by_model_mean[model_id],
            )

    def test_returns_deterministic_full_ranking_with_prior_tie_break(self) -> None:
        queries = {"q1": "shared alpha", "q2": "shared beta"}
        qrels = {
            "q1": {"z-model": 0.0, "a-model": 2.0, "m-model": 1.0},
            "q2": {"z-model": 2.0, "a-model": 0.0, "m-model": 1.0},
        }
        ranker = QueryKNNRanker(k=2, ngram_range=(1, 1)).fit(queries, qrels)

        first = ranker.rank("shared", aggregation="weighted_mean")
        second = ranker.rank("shared", aggregation="weighted_mean")

        self.assertEqual(first, second)
        self.assertEqual(3, len(first))
        self.assertEqual([1, 2, 3], [item.rank for item in first])
        # Scores and priors all tie, so model ID is the final stable key.
        self.assertEqual(
            ["a-model", "m-model", "z-model"], ranking_model_ids(first)
        )

    def test_oov_query_falls_back_to_global_prior(self) -> None:
        ranker = QueryKNNRanker(k=2, ngram_range=(1, 1)).fit(
            self.queries, self.qrels
        )

        ranking = ranker.rank("unseenword", aggregation="max")

        self.assertEqual(
            ["model-c", "model-b", "model-a"], ranking_model_ids(ranking)
        )
        for item in ranking:
            self.assertEqual(item.global_prior, item.score)

    def test_refit_drops_previous_fold_vocabulary(self) -> None:
        ranker = QueryKNNRanker(ngram_range=(1, 1)).fit(
            {"target": "heldouttoken", "train": "ordinary token"},
            {
                "target": {"a": 1.0, "b": 0.0},
                "train": {"a": 0.0, "b": 1.0},
            },
        )
        self.assertIn("heldouttoken", ranker.vectorizer.vocabulary_)

        ranker.fit(
            {"train": "ordinary token"},
            {"train": {"a": 0.0, "b": 1.0}},
        )

        self.assertNotIn("heldouttoken", ranker.vectorizer.vocabulary_)
        self.assertEqual(1, ranker.vectorizer.n_documents_)

    def test_leave_one_out_hides_the_target_qrel(self) -> None:
        # q1's grade strongly favors model-a.  Its held-out OOV query must use
        # only q2/q3 priors, which favor model-b.
        queries = {
            "q1": "targetonly",
            "q2": "shared train",
            "q3": "shared example",
        }
        qrels = {
            "q1": {"model-a": 100.0, "model-b": 0.0},
            "q2": {"model-a": 0.0, "model-b": 2.0},
            "q3": {"model-a": 0.0, "model-b": 2.0},
        }

        rankings = leave_one_out_rankings(
            queries, qrels, k=2, ngram_range=(1, 1)
        )

        self.assertEqual("model-b", rankings["q1"][0].model_id)


class DataLoadingTest(unittest.TestCase):
    def test_loads_queries_and_complete_qrels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path = root / "queries.tsv"
            qrels_path = root / "qrels.txt"
            query_path.write_text("q1\talpha query\nq2\tbeta query\n", encoding="utf-8")
            qrels_path.write_text(
                "q1 0 model-a 2\n"
                "q1 0 model-b 0\n"
                "q2 0 model-a 1\n"
                "q2 0 model-b 2\n",
                encoding="utf-8",
            )

            data = load_development_data(query_path, qrels_path)

            self.assertEqual(("model-a", "model-b"), data.models)
            self.assertEqual(2.0, data.qrels["q2"]["model-b"])

    def test_rejects_incomplete_qrel_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            query_path = root / "queries.tsv"
            qrels_path = root / "qrels.txt"
            query_path.write_text("q1\talpha\nq2\tbeta\n", encoding="utf-8")
            qrels_path.write_text(
                "q1 0 model-a 2\n"
                "q1 0 model-b 0\n"
                "q2 0 model-a 1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "incomplete qrels"):
                load_development_data(query_path, qrels_path)

    def test_query_loader_preserves_tabs_in_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "queries.tsv"
            path.write_text("q1\ttext with\tan internal tab\n", encoding="utf-8")

            self.assertEqual(
                {"q1": "text with\tan internal tab"}, load_queries_tsv(path)
            )


class WordTfidfTest(unittest.TestCase):
    def test_transform_does_not_learn_target_terms(self) -> None:
        vectorizer = WordTfidfVectorizer(ngram_range=(1, 1)).fit(
            ["training vocabulary"]
        )

        transformed = vectorizer.transform_one("targetonly training")

        self.assertNotIn("targetonly", vectorizer.vocabulary_)
        self.assertEqual({"training"}, set(transformed))


if __name__ == "__main__":
    unittest.main()
