from pathlib import Path
import tempfile
import unittest

from trec_mlm.experiment import (
    ExperimentConfig,
    configuration_key,
    run_leave_one_out_experiment,
    write_experiment_artifacts,
)
from trec_mlm.query_knn import leave_one_out_rankings, ranking_model_ids
from trec_mlm.runio import validate_trec_run


class ExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.query_path = root / "queries.tsv"
        self.qrels_path = root / "qrels.txt"
        self.query_path.write_text(
            "q1\tred apple orchard\n"
            "q2\tred berry fruit\n"
            "q3\tblue ocean water\n"
            "q4\tblue river water\n",
            encoding="utf-8",
        )
        self.qrels_path.write_text(
            "q1 0 a 2\nq1 0 b 0\nq1 0 c 1\n"
            "q2 0 a 0\nq2 0 b 2\nq2 0 c 1\n"
            "q3 0 a 0\nq3 0 b 1\nq3 0 c 2\n"
            "q4 0 a 1\nq4 0 b 0\nq4 0 c 2\n",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_optimized_sweep_matches_reference_leave_one_out(self) -> None:
        config = ExperimentConfig(
            query_path=str(self.query_path),
            qrels_path=str(self.qrels_path),
            neighbor_counts=(1, 2),
            primary_neighbors=1,
            bootstrap_resamples=50,
            seed=4,
        )
        result = run_leave_one_out_experiment(config)

        for aggregation in config.aggregations:
            for k in config.neighbor_counts:
                reference = leave_one_out_rankings(
                    result.data.queries,
                    result.data.qrels,
                    k=k,
                    aggregation=aggregation,
                    ngram_range=config.ngram_range,
                )
                expected = {
                    query_id: ranking_model_ids(ranking)
                    for query_id, ranking in reference.items()
                }
                self.assertEqual(
                    expected,
                    result.rankings[configuration_key(aggregation, k)],
                )
        self.assertTrue(result.summary["weighted_sum_mean_rankings_identical"])

    def test_artifacts_form_a_complete_valid_run(self) -> None:
        config = ExperimentConfig(
            query_path=str(self.query_path),
            qrels_path=str(self.qrels_path),
            neighbor_counts=(1,),
            primary_neighbors=1,
            bootstrap_resamples=20,
        )
        result = run_leave_one_out_experiment(config)
        output_dir = Path(self.directory.name) / "artifacts"

        paths = write_experiment_artifacts(result, output_dir)
        validation = validate_trec_run(
            paths["run.trec"],
            expected_query_ids=tuple(result.data.queries),
            expected_model_ids=result.data.models,
        )

        self.assertEqual(12, validation["row_count"])
        self.assertTrue(Path(paths["summary.json"]).is_file())
        self.assertIsNone(
            result.summary["density_slices"]["exactly_1_positive"][
                "primary_ndcg_at_10"
            ]
        )


if __name__ == "__main__":
    unittest.main()
