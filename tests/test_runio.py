from pathlib import Path
import tempfile
import unittest

from trec_mlm.runio import rankings_from_run, validate_trec_run, write_trec_run


class RunIOTest(unittest.TestCase):
    def test_writer_round_trip_and_strict_validation(self) -> None:
        rankings = {
            "q2": ["model-b", "model-a"],
            "q1": ["model-a", "model-b"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.trec"
            write_trec_run(path, rankings, run_id="test-run")

            validation = validate_trec_run(
                path,
                expected_query_ids=("q1", "q2"),
                expected_model_ids=("model-a", "model-b"),
            )

            self.assertTrue(validation["valid"])
            self.assertEqual(4, validation["row_count"])
            self.assertEqual(rankings, rankings_from_run(path))

    def test_rejects_score_that_increases_with_rank(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.trec"
            path.write_text(
                "q1 Q0 a 1 1.0 run\nq1 Q0 b 2 2.0 run\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "strictly decrease"):
                validate_trec_run(path)

    def test_rejects_different_candidate_sets(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.trec"
            path.write_text(
                "q1 Q0 a 1 2 run\n"
                "q1 Q0 b 2 1 run\n"
                "q2 Q0 a 1 2 run\n"
                "q2 Q0 c 2 1 run\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "different candidate set"):
                validate_trec_run(path)


if __name__ == "__main__":
    unittest.main()
