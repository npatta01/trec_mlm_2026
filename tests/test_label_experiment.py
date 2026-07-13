from __future__ import annotations

import csv
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from trec_mlm.label_experiment import (
    LabelExperimentConfig,
    build_diversity_summary,
    build_sweep_summary,
    compare_repeated_results,
    load_label_experiment_artifacts,
    load_candidate_manifest,
    run_label_experiment,
    write_label_experiment_artifacts,
    write_diversity_summary,
    write_sweep_summary,
)
from trec_mlm.cli import build_parser
from trec_mlm.runio import validate_trec_run


class LabelExperimentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        root = Path(self.directory.name)
        self.query_path = root / "queries.tsv"
        self.qrels_path = root / "qrels.txt"
        self.candidate_path = root / "candidates.json"
        self.query_path.write_text(
            "q1\talpha question\nq2\tbeta question\n",
            encoding="utf-8",
        )
        self.qrels_path.write_text(
            "q1 0 Alpha_topic 2\n"
            "q1 0 Beta_topic 0\n"
            "q2 0 Alpha_topic 0\n"
            "q2 0 Beta_topic 2\n",
            encoding="utf-8",
        )
        self.candidate_path.write_text(
            json.dumps({"candidate_ids": ["Beta_topic", "Alpha_topic"]}),
            encoding="utf-8",
        )
        self.config = LabelExperimentConfig(
            query_path=str(self.query_path),
            qrels_path=str(self.qrels_path),
            candidate_path=str(self.candidate_path),
            encoder_key="qwen3_embedding_8b",
            bootstrap_resamples=100,
            seed=7,
            label_representation="expert_capability",
        )
        self.query_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])
        # Candidate manifest is sorted to Alpha_topic, Beta_topic.
        self.label_embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]])

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_candidate_manifest_accepts_snapshot_shape_and_sorts_ids(self) -> None:
        self.assertEqual(
            ("Alpha_topic", "Beta_topic"),
            load_candidate_manifest(self.candidate_path),
        )

        self.candidate_path.write_text(
            json.dumps({"plain": {"candidate_ids": ["z", "a"]}}),
            encoding="utf-8",
        )
        self.assertEqual(("a", "z"), load_candidate_manifest(self.candidate_path))

    def test_precomputed_embeddings_produce_complete_qrel_free_ranking(self) -> None:
        result = run_label_experiment(
            self.config,
            query_embeddings=self.query_embeddings,
            label_embeddings=self.label_embeddings,
            encoder_metadata={"backend": "fixture"},
        )

        self.assertEqual(["Alpha_topic", "Beta_topic"], result.rankings["q1"])
        self.assertEqual(["Beta_topic", "Alpha_topic"], result.rankings["q2"])
        self.assertEqual(1.0, result.evaluation.macro_ndcg_at_k)
        self.assertEqual(0, result.zero_ndcg_count)
        self.assertEqual(("q1", "q2"), result.query_ids)
        self.assertEqual(("Alpha_topic", "Beta_topic"), result.model_ids)
        self.assertEqual(
            "This LLM is most likely to correctly answer questions about Alpha topic.",
            result.formatted_documents[0],
        )

    def test_writer_emits_scores_embeddings_metrics_and_valid_run(self) -> None:
        result = run_label_experiment(
            self.config,
            query_embeddings=self.query_embeddings,
            label_embeddings=self.label_embeddings,
            encoder_metadata={"backend": "fixture"},
        )
        output_dir = Path(self.directory.name) / "artifacts"

        paths = write_label_experiment_artifacts(result, output_dir)

        expected = {
            "config.json",
            "encoder_metadata.json",
            "query_embeddings.npy",
            "label_embeddings.npy",
            "scores.tsv",
            "run.trec",
            "per_query.tsv",
            "summary.json",
            "validation.json",
        }
        self.assertEqual(expected, set(paths))
        validation = validate_trec_run(
            paths["run.trec"],
            expected_query_ids=("q1", "q2"),
            expected_model_ids=("Alpha_topic", "Beta_topic"),
        )
        self.assertEqual(4, validation["row_count"])
        with Path(paths["scores.tsv"]).open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(4, len(rows))
        self.assertEqual(
            {"query_id", "model_id", "score", "rank"},
            set(rows[0]),
        )

        loaded = load_label_experiment_artifacts(output_dir)
        self.assertEqual(result.rankings, loaded.rankings)
        np.testing.assert_allclose(result.scores, loaded.scores)
        self.assertEqual(result.summary, loaded.summary)
        config_payload = json.loads(
            Path(paths["config.json"]).read_text(encoding="utf-8")
        )
        self.assertIn("@sha256:", config_payload["container_image"])
        self.assertEqual(
            "Instruct: {instruction}\nQuery:{question}",
            config_payload["prompt_metadata"]["query_template"],
        )
        self.assertIn("LLM experts", config_payload["prompt_metadata"]["instruction"])
        self.assertEqual(
            "This LLM is most likely to correctly answer questions about "
            "{cleaned_label}.",
            config_payload["prompt_metadata"]["document_template"],
        )

    def test_writer_records_nomic_search_templates(self) -> None:
        config = replace(
            self.config,
            encoder_key="nomic_embed_text_v1_5",
            label_representation="bare",
        )
        result = run_label_experiment(
            config,
            query_embeddings=self.query_embeddings,
            label_embeddings=self.label_embeddings,
            encoder_metadata={"backend": "fixture"},
        )

        paths = write_label_experiment_artifacts(
            result,
            Path(self.directory.name) / "nomic_artifacts",
        )
        payload = json.loads(Path(paths["config.json"]).read_text(encoding="utf-8"))

        self.assertEqual(
            "search_query: {question}",
            payload["prompt_metadata"]["query_template"],
        )
        self.assertEqual(
            "search_document: {cleaned_label}",
            payload["prompt_metadata"]["document_template"],
        )

    def test_loader_rejects_missing_score_rows(self) -> None:
        result = run_label_experiment(
            self.config,
            query_embeddings=self.query_embeddings,
            label_embeddings=self.label_embeddings,
            encoder_metadata={"backend": "fixture"},
        )
        output_dir = Path(self.directory.name) / "missing_score"
        paths = write_label_experiment_artifacts(result, output_dir)
        score_path = Path(paths["scores.tsv"])
        lines = score_path.read_text(encoding="utf-8").splitlines()
        score_path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "score rows"):
            load_label_experiment_artifacts(output_dir)

    def test_loader_rejects_run_that_disagrees_with_scores(self) -> None:
        result = run_label_experiment(
            self.config,
            query_embeddings=self.query_embeddings,
            label_embeddings=self.label_embeddings,
            encoder_metadata={"backend": "fixture"},
        )
        output_dir = Path(self.directory.name) / "run_disagreement"
        paths = write_label_experiment_artifacts(result, output_dir)
        run_path = Path(paths["run.trec"])
        lines = run_path.read_text(encoding="utf-8").splitlines()
        first, second = lines[0].split(), lines[1].split()
        first[2], second[2] = second[2], first[2]
        lines[0], lines[1] = " ".join(first), " ".join(second)
        run_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "rankings disagree"):
            load_label_experiment_artifacts(output_dir)

    def test_sweep_summary_applies_gate_and_stage_two_decision(self) -> None:
        baseline_config = LabelExperimentConfig(
            query_path=str(self.query_path),
            qrels_path=str(self.qrels_path),
            candidate_path=str(self.candidate_path),
            encoder_key="mpnet_label_control",
            bootstrap_resamples=100,
            seed=7,
        )
        baseline = run_label_experiment(
            baseline_config,
            query_embeddings=self.query_embeddings,
            label_embeddings=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            encoder_metadata={"backend": "fixture"},
        )
        qwen = run_label_experiment(
            self.config,
            query_embeddings=self.query_embeddings,
            label_embeddings=self.label_embeddings,
            encoder_metadata={"backend": "fixture"},
        )

        summary = build_sweep_summary(
            baseline=baseline,
            model_results={"qwen3_embedding_8b": qwen},
            repeat_results={"qwen3_embedding_8b": qwen},
            dense_reference=baseline.evaluation,
            maximum_zero_ndcg=0,
            n_resamples=100,
            seed=7,
            issue_examples={"q1": "Alpha_topic"},
        )

        self.assertTrue(summary["models"]["qwen3_embedding_8b"]["gate"]["passed"])
        self.assertTrue(
            summary["models"]["qwen3_embedding_8b"]["repeat_validation"]["passed"]
        )
        self.assertIn(
            "ndcg_delta",
            summary["models"]["qwen3_embedding_8b"]["dense_reference_comparison"],
        )
        self.assertIn(
            "exactly_1_positive",
            summary["models"]["qwen3_embedding_8b"]["density_slices"],
        )
        self.assertIn(
            "other",
            summary["models"]["qwen3_embedding_8b"]["question_type_slices"],
        )
        self.assertEqual(
            [
                "qwen3_embedding_4b",
                "qwen3_embedding_0_6b",
                "jina_v5_small_retrieval",
            ],
            summary["stage_two"]["encoders_to_run"],
        )
        self.assertEqual(2, summary["issue_11_examples"]["q1"]["baseline_rank"])
        self.assertEqual(
            1,
            summary["issue_11_examples"]["q1"]["model_ranks"]["qwen3_embedding_8b"],
        )

        output_dir = Path(self.directory.name) / "sweep"
        paths = write_sweep_summary(summary, output_dir)
        self.assertEqual({"sweep_summary.json", "decision.md"}, set(paths))
        receipt = Path(paths["decision.md"]).read_text(encoding="utf-8")
        self.assertIn("Stage 2", receipt)
        self.assertIn("Paired MRR 95% CI", receipt)
        self.assertIn("Zero-query transitions", receipt)
        self.assertIn("Issue #11 examples", receipt)
        self.assertIn("Frozen dense reference", receipt)

        missing_repeat = build_sweep_summary(
            baseline=baseline,
            model_results={"qwen3_embedding_8b": qwen},
            maximum_zero_ndcg=0,
            n_resamples=100,
            seed=7,
            issue_examples={},
        )
        self.assertFalse(
            missing_repeat["models"]["qwen3_embedding_8b"]["eligible_for_advancement"]
        )
        self.assertEqual([], missing_repeat["stage_two"]["encoders_to_run"])

    def test_diversity_summary_compares_baselines_context_and_repeats(self) -> None:
        baseline_config = replace(
            self.config,
            encoder_key="mpnet_label_control",
            label_representation="bare",
        )
        baseline = run_label_experiment(
            baseline_config,
            query_embeddings=self.query_embeddings,
            label_embeddings=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            encoder_metadata={"backend": "fixture"},
        )
        candidate = run_label_experiment(
            self.config,
            query_embeddings=self.query_embeddings,
            label_embeddings=self.label_embeddings,
            encoder_metadata={
                "backend": "fixture",
                "encoding_seconds": 1.5,
                "peak_memory_allocated_bytes": 1024,
                "parameter_count": 10,
            },
        )

        summary = build_diversity_summary(
            baseline=baseline,
            e5_reference=candidate,
            model_results={
                "qwen3_embedding_8b/bare": candidate,
                "qwen3_embedding_8b/expert_capability": candidate,
            },
            repeat_results={
                "qwen3_embedding_8b/bare": candidate,
                "qwen3_embedding_8b/expert_capability": candidate,
            },
            n_resamples=100,
            seed=7,
            issue_examples={"q1": "Alpha_topic"},
        )

        condition = summary["conditions"]["qwen3_embedding_8b/bare"]
        self.assertIn("ndcg_delta", condition["mpnet_comparison"])
        self.assertIn("ndcg_delta", condition["e5_comparison"])
        self.assertTrue(condition["repeat_validation"]["passed"])
        self.assertEqual(10, condition["resources"]["parameter_count"])
        self.assertIn(
            "qwen3_embedding_8b",
            summary["within_model_contextualization"],
        )
        self.assertEqual(
            "post_gate_exploratory_evidence",
            summary["decision"]["status"],
        )
        self.assertIsNone(summary["decision"]["selected_deployment_model"])
        self.assertEqual(
            1,
            summary["issue_11_examples"]["q1"]["condition_ranks"][
                "qwen3_embedding_8b/bare"
            ],
        )
        paths = write_diversity_summary(
            summary,
            Path(self.directory.name) / "diversity_summary",
        )
        self.assertEqual({"diversity_summary.json", "decision.md"}, set(paths))
        receipt = Path(paths["decision.md"]).read_text(encoding="utf-8")
        self.assertIn("post-gate exploratory", receipt)
        self.assertIn("Context effect", receipt)

    def test_repeat_comparison_requires_identical_rankings_and_close_scores(
        self,
    ) -> None:
        first = run_label_experiment(
            self.config,
            query_embeddings=self.query_embeddings,
            label_embeddings=self.label_embeddings,
            encoder_metadata={"backend": "fixture"},
        )
        close = run_label_experiment(
            self.config,
            query_embeddings=self.query_embeddings + 1e-7,
            label_embeddings=self.label_embeddings,
            encoder_metadata={"backend": "fixture"},
        )

        comparison = compare_repeated_results(first, close, score_tolerance=1e-5)

        self.assertTrue(comparison["passed"])
        self.assertTrue(comparison["rankings_identical"])
        self.assertTrue(comparison["metrics_identical"])
        self.assertLessEqual(comparison["maximum_absolute_score_difference"], 1e-5)

        changed = run_label_experiment(
            self.config,
            query_embeddings=self.query_embeddings,
            label_embeddings=np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            encoder_metadata={"backend": "fixture"},
        )
        self.assertFalse(compare_repeated_results(first, changed)["passed"])

    def test_cli_exposes_label_run_and_summary_commands(self) -> None:
        parser = build_parser()

        run = parser.parse_args(
            [
                "label-retrieval",
                "--encoder",
                "qwen3_embedding_8b",
                "--label-representation",
                "expert_capability",
                "--output-dir",
                "artifacts/qwen",
            ]
        )
        summary = parser.parse_args(
            [
                "label-sweep-summary",
                "--baseline-dir",
                "artifacts/mpnet",
                "--model-result",
                "qwen3_embedding_8b=artifacts/qwen",
                "--repeat-result",
                "qwen3_embedding_8b=artifacts/qwen-repeat",
                "--output-dir",
                "artifacts/sweep",
            ]
        )
        diversity = parser.parse_args(
            [
                "label-diversity-summary",
                "--baseline-dir",
                "artifacts/mpnet",
                "--e5-reference-dir",
                "artifacts/e5",
                "--condition-result",
                "qwen3_embedding_4b/bare=artifacts/qwen4",
                "--repeat-result",
                "qwen3_embedding_4b/bare=artifacts/qwen4-repeat",
                "--output-dir",
                "artifacts/diversity",
            ]
        )

        self.assertEqual("qwen3_embedding_8b", run.encoder)
        self.assertEqual("expert_capability", run.label_representation)
        self.assertEqual(
            ["qwen3_embedding_8b=artifacts/qwen"],
            summary.model_result,
        )
        self.assertEqual(
            ["qwen3_embedding_8b=artifacts/qwen-repeat"],
            summary.repeat_result,
        )
        self.assertEqual(
            ["qwen3_embedding_4b/bare=artifacts/qwen4"],
            diversity.condition_result,
        )


if __name__ == "__main__":
    unittest.main()
