from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from trec_mlm import cli, llm_ranker_pilot as pilot


ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "artifacts/llm_label_ranker/pilot/stage2/selection.json"
SMOKE_PATH = ROOT / "artifacts/llm_label_ranker/smoke/smoke.json"
POINTWISE_DIR = ROOT / "artifacts/llm_label_ranker/pilot/stage2/pointwise"
QWEN_CONTEXT_DIR = (
    ROOT
    / "artifacts/label_sweep_diverse/qwen3_embedding_4b"
    / "expert_capability/run1"
)
E5_CONTEXT_DIR = (
    ROOT
    / "artifacts/label_sweep_corrected/e5_mistral_7b_instruct"
    / "expert_capability/run1"
)
QRELS_PATH = ROOT / "data/dev/llm_dev_qrels.txt"
CONTAINER = "trec-llm-ranker-stage2-listwise-5717e09"
CODE_REVISION = "5717e091e4a19a663ad1ecae56ab661e5152484f"
IMAGE = (
    "docker.io/vllm/vllm-openai-rocm@sha256:"
    "3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28"
)
RANKING_DIR = "artifacts/llm_label_ranker/pilot/stage2/listwise"
COMMAND = (
    "python -m trec_mlm.cli llm-label-ranker-pilot --stage 2 "
    "--selection artifacts/llm_label_ranker/pilot/stage2/selection.json "
    "--method listwise --smoke-receipt "
    "artifacts/llm_label_ranker/smoke/smoke.json --code-revision "
    f"{CODE_REVISION} --output-dir {RANKING_DIR}"
)
ERROR_MESSAGE = (
    "pilot listwise parse failure for 57326d2f0fdd8d15006c6ad5/0: "
    "pilot ranking has duplicates, omissions, or unknown handles"
)


def _inspect_bytes(*, environment: list[str] | None = None) -> bytes:
    payload = [
        {
            "Id": "9743eb98851f01e7199d6bc94cb476c0deb9b1573768402cd50717033d9a4073",
            "Name": CONTAINER,
            "ImageName": IMAGE,
            "Path": "/bin/bash",
            "Args": ["-lc", COMMAND],
            "State": {
                "Status": "exited",
                "ExitCode": 1,
                "OOMKilled": False,
                "StartedAt": "2026-07-13T12:33:17.505273876-04:00",
                "FinishedAt": "2026-07-13T12:37:44.863994486-04:00",
            },
            "Config": {
                "Image": IMAGE,
                "Env": environment or ["TOKENIZERS_PARALLELISM=false"],
            },
        }
    ]
    return (json.dumps(payload, separators=(",", ":")) + "\n").encode()


def _log_bytes() -> bytes:
    return (
        "2026-07-13T12:37:38.000000000-04:00 Processed prompts: 100% "
        "36/36 [00:10<00:00]\n"
        "2026-07-13T12:37:41.535935000-04:00 ValueError: "
        f"{ERROR_MESSAGE}\n"
    ).encode()


def _capture(output_dir: Path, *, inspect: bytes | None = None) -> dict:
    responses = (
        SimpleNamespace(stdout=inspect or _inspect_bytes(), stderr=b""),
        SimpleNamespace(stdout=_log_bytes(), stderr=None),
    )
    with patch("trec_mlm.llm_ranker_failure.subprocess.run", side_effect=responses):
        return pilot.capture_terminal_listwise_failure(
            container_name=CONTAINER,
            output_dir=output_dir,
            selection_path=SELECTION_PATH,
            smoke_receipt_path=SMOKE_PATH,
            expected_ranking_dir=RANKING_DIR,
        )


class TerminalListwiseFailureTest(unittest.TestCase):
    def test_capture_preserves_exact_bytes_and_validates_qrel_free_expected_prompts(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure"
            receipt = _capture(output)

            self.assertEqual(_inspect_bytes(), (output / "inspect.json").read_bytes())
            self.assertEqual(_log_bytes(), (output / "container.log").read_bytes())
            prompts = [
                json.loads(line)
                for line in (output / "expected_prompts.jsonl").read_text().splitlines()
            ]
            self.assertEqual(36, len(prompts))
            self.assertTrue(
                all(row["evidence_status"] == "expected_not_observed" for row in prompts)
            )
            self.assertTrue(
                all(row["provenance"] == "regenerated_qrel_free" for row in prompts)
            )
            self.assertEqual("unscorable_final_attempt", receipt["disposition"])
            self.assertFalse(receipt["rerun_permitted"])
            self.assertEqual(36, receipt["execution"]["planned_request_count"])
            self.assertEqual(36, receipt["execution"]["log_observed_request_count"])
            self.assertEqual(0, receipt["execution"]["persisted_raw_output_count"])
            self.assertFalse(receipt["execution"]["rankings_available"])
            self.assertEqual(1, receipt["container"]["exit_code"])
            self.assertFalse(receipt["container"]["oom_killed"])
            self.assertEqual(
                "57326d2f0fdd8d15006c6ad5", receipt["failure"]["query_id"]
            )
            self.assertEqual(0, receipt["failure"]["rotation"])
            self.assertEqual(receipt, pilot.validate_terminal_listwise_failure(output))

    def test_validator_rejects_mutation_inventory_and_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure"
            _capture(output)
            (output / "container.log").write_bytes(_log_bytes() + b"mutated\n")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                pilot.validate_terminal_listwise_failure(output)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure"
            _capture(output)
            (output / "unexpected.txt").write_text("surprise")
            with self.assertRaisesRegex(ValueError, "inventory"):
                pilot.validate_terminal_listwise_failure(output)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure"
            _capture(output)
            receipt_path = output / "failure_receipt.json"
            receipt = json.loads(receipt_path.read_text())
            receipt["container"]["id"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "container identity"):
                pilot.validate_terminal_listwise_failure(output)

    def test_secret_detection_fails_closed_without_writing_observed_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure"
            with self.assertRaisesRegex(ValueError, "secret"):
                _capture(output, inspect=_inspect_bytes(environment=["HF_TOKEN=hf_12345678901234567890"]))
            self.assertFalse(output.exists())

    def test_partial_progress_cannot_masquerade_as_all_requests_completed(self) -> None:
        responses = (
            SimpleNamespace(stdout=_inspect_bytes(), stderr=b""),
            SimpleNamespace(
                stdout=_log_bytes().replace(b"36/36", b"18/36"), stderr=None
            ),
        )
        with tempfile.TemporaryDirectory() as directory, patch(
            "trec_mlm.llm_ranker_failure.subprocess.run", side_effect=responses
        ):
            output = Path(directory) / "failure"
            with self.assertRaisesRegex(ValueError, "request count"):
                pilot.capture_terminal_listwise_failure(
                    container_name=CONTAINER,
                    output_dir=output,
                    selection_path=SELECTION_PATH,
                    smoke_receipt_path=SMOKE_PATH,
                    expected_ranking_dir=RANKING_DIR,
                )
            self.assertFalse(output.exists())

    def test_failure_evidence_cannot_load_as_a_ranking(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failure"
            _capture(output)
            with self.assertRaises(FileNotFoundError):
                pilot.load_frozen_pilot(output)
            self.assertFalse((output / "ranking_receipt.json").exists())
            self.assertFalse((output / "rankings.json").exists())
            self.assertFalse((output / "scores.tsv").exists())

    def test_future_cli_persists_all_raw_generations_before_nonzero_exit(self) -> None:
        llm = object()
        tokenizer = object()
        valid = json.dumps([f"C{index:02d}" for index in range(12)])
        invalid = '["C00","C00"]'

        def engine(**_kwargs):
            return (
                llm,
                tokenizer,
                {
                    "model_id": "Qwen/Qwen3-4B-Instruct-2507",
                    "revision": "cdbee75f17c01a7cc42f958dc650907174af0554",
                    "dtype": "float32",
                    "seed": 2026,
                    "gpu_memory_utilization": 0.30,
                    "max_model_len": 8192,
                    "attention_backend": "TRITON_ATTN",
                    "vllm_version": "0.24.0",
                    "torch_version": "2.11.0",
                    "hip_version": "7.2",
                    "device_properties": "fixture gfx1151",
                },
            )

        def infer(
            observed_llm,
            messages,
            *,
            handles,
            seed,
            use_tqdm,
            stats,
        ):
            self.assertIs(llm, observed_llm)
            self.assertEqual(18, len(messages))
            stats.update(
                {
                    "request_count": 18,
                    "prompt_token_count": 3600,
                    "output_token_count": 216,
                }
            )
            return [invalid, *([valid] * 17)]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "listwise_failure"
            with (
                patch(
                    "trec_mlm.llm_ranker_inference.load_ranker_engine",
                    side_effect=engine,
                ),
                patch(
                    "trec_mlm.artifact_manifest.validate_pilot_smoke_receipt",
                    return_value={"passed": True},
                ),
                patch(
                    "trec_mlm.llm_ranker_pilot.validate_pilot_smoke_receipt",
                    side_effect=lambda path: {"receipt_sha256": pilot._sha256(path)},
                ),
                patch(
                    "trec_mlm.cli._verify_git_launch_provenance",
                    return_value="clean",
                ),
                patch(
                    "trec_mlm.llm_ranker_pilot.run_pilot_listwise_messages",
                    side_effect=infer,
                ),
                patch.object(
                    pilot,
                    "load_complete_qrels",
                    side_effect=AssertionError("qrels must remain unopened"),
                ) as qrels,
            ):
                with self.assertRaisesRegex(ValueError, "parse failure"):
                    cli.main(
                        [
                            "llm-label-ranker-pilot",
                            "--stage",
                            "1",
                            "--selection",
                            "artifacts/llm_label_ranker/pilot/stage1/selection.json",
                            "--method",
                            "listwise",
                            "--smoke-receipt",
                            "artifacts/llm_label_ranker/smoke/smoke.json",
                            "--code-revision",
                            "a" * 40,
                            "--output-dir",
                            str(output),
                        ]
                    )
            qrels.assert_not_called()
            self.assertEqual(
                {
                    "config.json",
                    "failure_receipt.json",
                    "prompts.jsonl",
                    "raw_outputs.jsonl",
                    "runtime.json",
                    "selection.json",
                },
                {path.name for path in output.iterdir()},
            )
            rows = [json.loads(line) for line in (output / "raw_outputs.jsonl").read_text().splitlines()]
            self.assertEqual(18, len(rows))
            self.assertEqual(invalid, rows[0]["raw_output"])
            self.assertIsNotNone(rows[0]["validation_error"])
            self.assertTrue(
                all("selected_handles" not in row and "selected_candidate_ids" not in row for row in rows)
            )
            receipt = json.loads((output / "failure_receipt.json").read_text())
            self.assertEqual("unscorable_parse_failure", receipt["disposition"])
            self.assertEqual(18, receipt["execution"]["persisted_raw_output_count"])
            self.assertFalse(receipt["execution"]["rankings_available"])
            self.assertFalse((output / "ranking_receipt.json").exists())
            self.assertFalse((output / "rankings.json").exists())
            self.assertFalse((output / "scores.tsv").exists())

    def test_current_incident_remains_an_honest_zero_raw_output_capture(self) -> None:
        source = ROOT / "artifacts/llm_label_ranker/pilot/stage2/listwise_failure"
        self.assertEqual(
            "d83ebfb60a7a5ea6b1d209ec99f2720b324b6872c751ab482bd2c3ce8493a2f8",
            pilot._sha256(source / "failure_receipt.json"),
        )
        receipt = pilot.validate_terminal_listwise_failure(source)
        self.assertEqual(0, receipt["execution"]["persisted_raw_output_count"])


class Stage2PointwiseReportTest(unittest.TestCase):
    def test_downstream_inputs_stay_unopened_until_both_terminals_validate(self) -> None:
        with (
            patch.object(pilot, "load_frozen_pilot", return_value=object()),
            patch.object(
                pilot,
                "validate_terminal_listwise_failure",
                side_effect=ValueError("terminal failure invalid"),
            ),
            patch.object(
                pilot,
                "load_complete_qrels",
                side_effect=AssertionError("qrels opened too early"),
            ) as qrels,
            patch(
                "trec_mlm.label_experiment.load_label_experiment_artifacts",
                side_effect=AssertionError("baseline opened too early"),
            ) as baselines,
        ):
            with self.assertRaisesRegex(ValueError, "terminal failure invalid"):
                pilot.build_stage2_pointwise_report(
                    pointwise_dir=POINTWISE_DIR,
                    listwise_failure_dir="failure",
                    qwen_dir=QWEN_CONTEXT_DIR,
                    e5_dir=E5_CONTEXT_DIR,
                    qrels_path=QRELS_PATH,
                )
        qrels.assert_not_called()
        baselines.assert_not_called()

    def test_valid_failure_permits_pointwise_only_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failure = Path(directory) / "failure"
            _capture(failure)
            report = pilot.build_stage2_pointwise_report(
                pointwise_dir=POINTWISE_DIR,
                listwise_failure_dir=failure,
                qwen_dir=QWEN_CONTEXT_DIR,
                e5_dir=E5_CONTEXT_DIR,
                qrels_path=QRELS_PATH,
            )

            self.assertEqual(2, report["stage"])
            self.assertEqual("qrel-assisted exploratory", report["evidence_scope"])
            self.assertEqual("complete", report["method_status"]["pointwise"])
            self.assertEqual("unscorable", report["method_status"]["listwise"])
            self.assertIsNone(report["metrics"]["listwise"]["top3_relevant_count"])
            self.assertEqual(12, len(report["questions"]))
            self.assertEqual(
                {"issue_like_clear", "contextual_qwen_success", "contextual_qwen_failure"},
                set(report["strata"]),
            )
            self.assertEqual(
                ERROR_MESSAGE, report["listwise_failure"]["error_message"]
            )
            self.assertEqual(
                {
                    "code_revision": CODE_REVISION,
                    "smoke_receipt_sha256": "a88525a22ff51ec67600fd1e3e5ac3d7553feb48d3c01bab831a7232729b36b3",
                    "matches": True,
                },
                report["matching_provenance"],
            )
            gate = report["scaled_feasibility_gate"]
            self.assertFalse(gate["frozen_before_stage2_execution"])
            self.assertFalse(gate["passes"])
            self.assertFalse(gate["conditions"]["top3_not_lower"]["passes"])
            self.assertFalse(gate["conditions"]["ndcg_not_lower"]["passes"])
            self.assertTrue(
                all(
                    set(row["pointwise_vs_baseline_best_rank_delta"])
                    == {"qwen4b_contextual", "e5_contextual"}
                    and row["pointwise_outcome"]["classification"]
                    in {"top3_hit", "near_miss_rank_4_or_5", "deep_miss_rank_6_plus"}
                    for row in report["questions"]
                )
            )
            self.assertEqual(5, report["metrics"]["pointwise"]["top3_relevant_count"])
            self.assertEqual(
                8, report["metrics"]["qwen4b_contextual"]["top3_relevant_count"]
            )
            self.assertEqual(
                8, report["metrics"]["e5_contextual"]["top3_relevant_count"]
            )
            json_path = Path(directory) / "report.json"
            markdown_path = Path(directory) / "report.md"
            first = pilot.write_stage2_pointwise_report(
                report, report_path=json_path, markdown_path=markdown_path
            )
            second = pilot.write_stage2_pointwise_report(
                report, report_path=json_path, markdown_path=markdown_path
            )
            self.assertEqual(first, second)
            markdown = markdown_path.read_text()
            self.assertIn("Listwise | unscorable | unavailable", markdown)
            self.assertIn("not that 35 usable outputs existed", markdown)
            self.assertIn("## Scaled feasibility gate", markdown)
            self.assertIn("## Question-level comparison", markdown)
            self.assertEqual(1, markdown.count("| Query | Stratum | Relevant labels |"))
            self.assertTrue(all(f"`{query_id}`" in markdown for query_id in pilot.STAGE2_QUERY_IDS))

    def test_provenance_mismatch_blocks_qrels_and_baselines(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failure = Path(directory) / "failure"
            receipt = _capture(failure)
            cases = (
                ("code_revision", "b" * 40, "code revision"),
                ("smoke", "0" * 64, "smoke receipt"),
            )
            for field, value, message in cases:
                with self.subTest(field=field):
                    mutated = json.loads(json.dumps(receipt))
                    if field == "code_revision":
                        mutated["code_revision"] = value
                    else:
                        mutated["smoke_receipt"]["sha256"] = value
                    with (
                        patch.object(
                            pilot,
                            "validate_terminal_listwise_failure",
                            return_value=mutated,
                        ),
                        patch.object(
                            pilot,
                            "load_complete_qrels",
                            side_effect=AssertionError("qrels opened too early"),
                        ) as qrels,
                        patch(
                            "trec_mlm.label_experiment.load_label_experiment_artifacts",
                            side_effect=AssertionError("baseline opened too early"),
                        ) as baselines,
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            pilot.build_stage2_pointwise_report(
                                pointwise_dir=POINTWISE_DIR,
                                listwise_failure_dir=failure,
                                qwen_dir=QWEN_CONTEXT_DIR,
                                e5_dir=E5_CONTEXT_DIR,
                                qrels_path=QRELS_PATH,
                            )
                    qrels.assert_not_called()
                    baselines.assert_not_called()

    def test_stage1_report_behavior_is_unchanged(self) -> None:
        self.assertFalse(
            "validate_terminal_listwise_failure"
            in pilot.build_pilot_report.__code__.co_names
        )

    def test_host_capture_cli_dispatches_without_loading_a_model(self) -> None:
        with patch(
            "trec_mlm.llm_ranker_failure.capture_terminal_listwise_failure",
            return_value={"disposition": "unscorable_final_attempt"},
        ) as capture, patch("builtins.print"):
            result = cli.main(
                [
                    "llm-label-ranker-failure-capture",
                    "--container",
                    CONTAINER,
                    "--selection",
                    str(SELECTION_PATH),
                    "--smoke-receipt",
                    str(SMOKE_PATH),
                    "--expected-ranking-dir",
                    RANKING_DIR,
                    "--output-dir",
                    "failure",
                ]
            )
        self.assertEqual(0, result)
        capture.assert_called_once_with(
            container_name=CONTAINER,
            output_dir="failure",
            selection_path=str(SELECTION_PATH),
            smoke_receipt_path=str(SMOKE_PATH),
            expected_ranking_dir=RANKING_DIR,
        )


if __name__ == "__main__":
    unittest.main()
