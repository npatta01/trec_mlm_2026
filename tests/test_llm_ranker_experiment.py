from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from trec_mlm.llm_ranker_experiment import (
    LLMRankerConfig,
    compare_frozen_ranker_artifacts,
    evaluate_frozen_ranker_artifact,
    run_listwise_phase_a,
    run_pointwise_phase_a,
    write_ranker_phase_a_artifacts,
)
from trec_mlm.cli import build_parser


class LLMRankerExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.query_path = self.root / "queries.tsv"
        self.candidate_path = self.root / "candidates.json"
        self.smoke_path = self.root / "smoke.json"
        self.query_ids = tuple(f"q{index:03d}" for index in range(322))
        self.candidate_ids = tuple(f"expert_{index:03d}" for index in range(150))
        self.query_path.write_text(
            "".join(f"{query_id}\tquestion {index}\n" for index, query_id in enumerate(self.query_ids)),
            encoding="utf-8",
        )
        self.candidate_path.write_text(
            json.dumps({"candidate_ids": list(reversed(self.candidate_ids))}),
            encoding="utf-8",
        )
        self.smoke_path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                    "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
                    "dtype": "float32",
                }
            ),
            encoding="utf-8",
        )
        self.config = LLMRankerConfig(
            query_path=str(self.query_path),
            candidate_path=str(self.candidate_path),
            smoke_receipt_path=str(self.smoke_path),
            method="pointwise",
            run_id="fixture_pointwise",
            code_revision="fixture-revision",
            worktree_status="fixture-clean",
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _infer(self, messages):
        output = []
        for item in messages:
            text = item[-1]["content"]
            candidate = text.split('<expert id="', 1)[1].split('">', 1)[0]
            grade = int(candidate.rsplit("_", 1)[1]) % 3
            log_probs = {0: -4.0, 1: -4.0, 2: -4.0}
            log_probs[grade] = -0.1
            output.append(log_probs)
        return output

    def _write_qrels(self, path: Path, *, reverse: bool) -> None:
        lines = []
        relevant = self.candidate_ids[-1] if reverse else self.candidate_ids[0]
        for query_id in self.query_ids:
            for candidate_id in self.candidate_ids:
                lines.append(
                    f"{query_id} 0 {candidate_id} {2 if candidate_id == relevant else 0}\n"
                )
        path.write_text("".join(lines), encoding="utf-8")

    def test_phase_a_is_complete_qrel_free_and_evaluation_is_post_hoc(self) -> None:
        result = run_pointwise_phase_a(self.config, infer_batch=self._infer, batch_size=997)
        self.assertEqual((322, 150), result.scores.shape)
        self.assertEqual(48_300, len(result.raw_records))
        self.assertEqual(set(self.query_ids), set(result.rankings))
        self.assertTrue(all(len(ranking) == 150 for ranking in result.rankings.values()))

        phase_a_dir = self.root / "phase_a"
        paths = write_ranker_phase_a_artifacts(result, phase_a_dir)
        before = {name: Path(path).read_bytes() for name, path in paths.items()}

        qrels_a = self.root / "qrels_a.txt"
        qrels_b = self.root / "qrels_b.txt"
        self._write_qrels(qrels_a, reverse=False)
        self._write_qrels(qrels_b, reverse=True)
        first = evaluate_frozen_ranker_artifact(phase_a_dir, qrels_a, self.root / "eval_a")
        second = evaluate_frozen_ranker_artifact(phase_a_dir, qrels_b, self.root / "eval_b")

        after = {name: Path(path).read_bytes() for name, path in paths.items()}
        self.assertEqual(before, after)
        self.assertNotEqual(first["metrics_sha256"], second["metrics_sha256"])

    def test_frozen_artifact_rejects_mutated_core_file(self) -> None:
        result = run_pointwise_phase_a(self.config, infer_batch=self._infer, batch_size=4096)
        phase_a_dir = self.root / "phase_a"
        write_ranker_phase_a_artifacts(result, phase_a_dir)
        with (phase_a_dir / "scores.tsv").open("a", encoding="utf-8") as handle:
            handle.write("corruption\n")
        qrels = self.root / "qrels.txt"
        self._write_qrels(qrels, reverse=False)
        with self.assertRaisesRegex(ValueError, "artifact hash mismatch"):
            evaluate_frozen_ranker_artifact(phase_a_dir, qrels, self.root / "eval")

    def test_cli_exposes_separate_rank_and_evaluate_phases(self) -> None:
        parser = build_parser()
        rank = parser.parse_args(
            [
                "llm-label-ranker",
                "--phase",
                "rank",
                "--method",
                "listwise",
                "--output-dir",
                "artifacts/listwise/run1",
            ]
        )
        evaluate = parser.parse_args(
            [
                "llm-label-ranker",
                "--phase",
                "evaluate",
                "--method",
                "listwise",
                "--input-dir",
                "artifacts/listwise/run1",
                "--qrels",
                "data/dev/llm_dev_qrels.txt",
                "--output-dir",
                "artifacts/listwise/run1/evaluation",
            ]
        )
        self.assertEqual("rank", rank.phase)
        self.assertEqual("listwise", rank.method)
        self.assertEqual("evaluate", evaluate.phase)
        self.assertEqual("artifacts/listwise/run1", evaluate.input_dir)

    def test_listwise_phase_a_writes_three_valid_calls_per_query(self) -> None:
        config = LLMRankerConfig(
            query_path=str(self.query_path),
            candidate_path=str(self.candidate_path),
            smoke_receipt_path=str(self.smoke_path),
            method="listwise",
            run_id="fixture_listwise",
            code_revision="fixture-revision",
            worktree_status="fixture-clean",
        )

        def infer(messages):
            outputs = []
            for item in messages:
                expert_rows = item[-1]["content"].split("<experts>\n", 1)[1]
                handles = [row.split(" | ", 1)[0] for row in expert_rows.splitlines()[:20]]
                outputs.append(json.dumps(handles))
            return outputs

        result = run_listwise_phase_a(config, infer_batch=infer, batch_size=41)
        self.assertEqual(966, len(result.prompts))
        self.assertEqual(966, len(result.raw_records))
        self.assertEqual(0, result.runtime_metadata["fallback_count"])
        self.assertEqual((322, 150), result.scores.shape)
        self.assertTrue(all(len(ranking) == 150 for ranking in result.rankings.values()))

    def test_repeat_comparison_reports_strict_and_practical_agreement(self) -> None:
        result = run_pointwise_phase_a(self.config, infer_batch=self._infer, batch_size=4096)
        first = self.root / "first"
        second = self.root / "second"
        write_ranker_phase_a_artifacts(result, first)
        write_ranker_phase_a_artifacts(result, second)

        comparison = compare_frozen_ranker_artifacts(first, second)

        self.assertTrue(comparison["strict_rankings_identical"])
        self.assertTrue(comparison["top_10_rankings_identical"])
        self.assertTrue(comparison["scores_within_tolerance"])
        self.assertEqual(0.0, comparison["maximum_absolute_score_difference"])


if __name__ == "__main__":
    unittest.main()
