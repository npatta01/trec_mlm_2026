from __future__ import annotations

import hashlib
import importlib
import json
from math import fsum, log2
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from trec_mlm.cli import build_parser, main
from trec_mlm.grouprank import GROUPING_SEEDS, build_group_requests
from trec_mlm.grouprank_pilot import GroupRankPhaseAResult


ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "artifacts/llm_label_ranker/pilot/stage2/selection.json"
JUDGMENTS_PATH = (
    ROOT / "artifacts/llm_label_ranker/pilot/stage2/selection_judgments.json"
)
BASELINE_REPORT_PATH = ROOT / "artifacts/llm_label_ranker/pilot/stage2/report.json"
CODE_REVISION = "a" * 40


def _report_module():
    try:
        return importlib.import_module("trec_mlm.grouprank_report")
    except ImportError as error:  # Intentional RED assertion, not collection failure.
        raise AssertionError("GroupRank report module is missing") from error


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _phase_a_fixture() -> tuple[GroupRankPhaseAResult, tuple[int, ...]]:
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    judgments = json.loads(JUDGMENTS_PATH.read_text(encoding="utf-8"))
    relevant_by_query = {
        entry["query_id"]: entry["roles"]["relevant"][0]
        for entry in judgments["questions"]
    }
    relevant_ranks = tuple(range(1, 13))
    rankings: dict[str, list[str]] = {}
    candidate_scores: dict[str, dict[str, float]] = {}
    target_scores: dict[str, dict[str, float]] = {}
    for entry, relevant_rank in zip(
        selection["questions"], relevant_ranks, strict=True
    ):
        query_id = entry["query_id"]
        relevant = relevant_by_query[query_id]
        ranking = [
            candidate for candidate in entry["candidate_ids"] if candidate != relevant
        ]
        ranking.insert(relevant_rank - 1, relevant)
        rankings[query_id] = ranking
        target_scores[query_id] = {
            candidate: 10.0 - (rank - 1) * 0.5
            for rank, candidate in enumerate(ranking, start=1)
        }
        candidate_scores[query_id] = dict(target_scores[query_id])

    raw_outputs = []
    for request in build_group_requests(selection):
        scores = {}
        for handle, candidate in request.handle_to_candidate.items():
            target = target_scores[request.query_id][candidate]
            scores[handle] = (
                int(target // 1)
                if request.repetition == GROUPING_SEEDS[0]
                else int(-(-target // 1))
            )
        raw_outputs.append(
            {
                "query_id": request.query_id,
                "repetition": request.repetition,
                "group": request.group,
                "ordered_candidate_ids": list(request.ordered_candidate_ids),
                "handle_to_candidate": dict(request.handle_to_candidate),
                "raw_text": (
                    "<reason>Compared all experts.</reason>\n<answer>"
                    + json.dumps(scores)
                    + "</answer>"
                ),
                "prompt_token_count": 300,
                "output_token_count": 80,
                "finish_reason": "stop",
                "sampling_seed": request.sampling_seed,
            }
        )

    return (
        GroupRankPhaseAResult(
            selection=selection,
            prompts=(),
            raw_outputs=tuple(raw_outputs),
            candidate_scores=candidate_scores,
            rankings=rankings,
            config={
                "selection_sha256": _sha256(SELECTION_PATH),
                "code_revision": CODE_REVISION,
            },
            runtime={},
        ),
        relevant_ranks,
    )


class GroupRankMetricAndGateTest(unittest.TestCase):
    def test_report_metrics_match_known_relevant_ranks(self) -> None:
        report_module = _report_module()
        phase_a, relevant_ranks = _phase_a_fixture()

        with patch.object(report_module, "load_frozen_grouprank", return_value=phase_a):
            result = report_module.build_grouprank_report(
                self.id(),
                judgments_path=JUDGMENTS_PATH,
                baseline_report_path=BASELINE_REPORT_PATH,
                code_revision=CODE_REVISION,
            )

        expected_ndcg = fsum(
            0.0 if rank > 10 else 1.0 / log2(rank + 1) for rank in relevant_ranks
        ) / len(relevant_ranks)
        expected_mrr = fsum(1.0 / rank for rank in relevant_ranks) / len(relevant_ranks)
        self.assertEqual("grouprank", result["method"])
        metrics = result["metrics"]["grouprank"]
        self.assertAlmostEqual(expected_ndcg, metrics["ndcg_at_10"])
        self.assertAlmostEqual(expected_mrr, metrics["mrr"])
        self.assertEqual(3, metrics["top3_relevant_count"])
        self.assertEqual(48, metrics["validated_output_count"])
        self.assertEqual(12, len(result["questions"]))

        for row, expected_rank in zip(result["questions"], relevant_ranks, strict=True):
            grouprank = row["methods"]["grouprank"]
            self.assertEqual(expected_rank, grouprank["best_relevant_rank"])
            self.assertEqual(3, len(grouprank["top_three"]))
            self.assertEqual(20, len(row["repetition_score_differences"]))
            self.assertEqual(
                {"candidate_pairs", "candidates", "score_groups"},
                set(row["tie_counts"]),
            )
            self.assertEqual(
                {"e5_contextual", "pointwise", "qwen4b_contextual"},
                set(row["grouprank_vs_baselines"]),
            )

    def test_gate_passes_only_when_all_three_frozen_conditions_pass(self) -> None:
        report_module = _report_module()
        passing = {
            "validated_output_count": 48,
            "top3_relevant_count": 8,
            "ndcg_at_10": 0.7232,
        }

        gate = report_module.build_grouprank_gate(passing)

        self.assertTrue(gate["passes"])
        for field, value in (
            ("validated_output_count", 47),
            ("top3_relevant_count", 7),
            ("ndcg_at_10", 0.723199999),
        ):
            with self.subTest(field=field):
                failing = {**passing, field: value}
                self.assertFalse(report_module.build_grouprank_gate(failing)["passes"])


class GroupRankSealedEvaluationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.phase_a, _relevant_ranks = _phase_a_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_artifact_validation_completes_before_any_judgment_open(self) -> None:
        report_module = _report_module()
        judgments = self.root / "must-not-open.json"
        judgments.write_text("{}", encoding="utf-8")

        with (
            patch.object(
                report_module,
                "load_frozen_grouprank",
                side_effect=ValueError("invalid Phase A artifact"),
            ) as load,
            patch.object(
                report_module,
                "_read_json_bytes",
                side_effect=AssertionError("judgments opened too early"),
            ) as read,
        ):
            with self.assertRaisesRegex(ValueError, "invalid Phase A artifact"):
                report_module.build_grouprank_report(
                    self.root / "invalid-artifact",
                    judgments_path=judgments,
                    baseline_report_path=BASELINE_REPORT_PATH,
                )

        load.assert_called_once_with(self.root / "invalid-artifact")
        read.assert_not_called()

    def test_judgment_query_and_candidate_inventories_must_match_phase_a(self) -> None:
        report_module = _report_module()
        judgments = json.loads(JUDGMENTS_PATH.read_text(encoding="utf-8"))
        entry = judgments["questions"][0]
        replaced = entry["roles"]["negative"][0]
        entry["roles"]["negative"][0] = "candidate_outside_phase_a"
        entry["grades"]["candidate_outside_phase_a"] = entry["grades"].pop(replaced)
        path = self.root / "mismatched-judgments.json"
        path.write_text(json.dumps(judgments), encoding="utf-8")

        with patch.object(
            report_module, "load_frozen_grouprank", return_value=self.phase_a
        ):
            with self.assertRaisesRegex(ValueError, "candidate inventory"):
                report_module.build_grouprank_report(
                    self.root / "artifact",
                    judgments_path=path,
                    baseline_report_path=BASELINE_REPORT_PATH,
                )

    def test_frozen_baseline_headlines_reject_any_mismatch(self) -> None:
        report_module = _report_module()
        baseline = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
        baseline["metrics"]["e5_contextual"]["ndcg_at_10"] = 0.7000
        path = self.root / "changed-baseline.json"
        path.write_text(json.dumps(baseline), encoding="utf-8")

        with (
            patch.object(
                report_module, "load_frozen_grouprank", return_value=self.phase_a
            ),
            patch.object(report_module, "FROZEN_STAGE2_REPORT_SHA256", _sha256(path)),
        ):
            with self.assertRaisesRegex(ValueError, "frozen Stage 2 e5_contextual"):
                report_module.build_grouprank_report(
                    self.root / "artifact",
                    judgments_path=JUDGMENTS_PATH,
                    baseline_report_path=path,
                )

    def test_frozen_baseline_sha_rejects_valid_looking_top_three_mutation(self) -> None:
        report_module = _report_module()
        self.assertEqual(
            _sha256(BASELINE_REPORT_PATH),
            report_module.FROZEN_STAGE2_REPORT_SHA256,
        )
        baseline = json.loads(BASELINE_REPORT_PATH.read_text(encoding="utf-8"))
        selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        judgments = json.loads(JUDGMENTS_PATH.read_text(encoding="utf-8"))
        relevant = set(judgments["questions"][0]["roles"]["relevant"])
        nonrelevant = [
            candidate_id
            for candidate_id in selection["questions"][0]["candidate_ids"]
            if candidate_id not in relevant
        ][:3]
        self.assertEqual(3, len(nonrelevant))
        baseline["questions"][0]["methods"]["e5_contextual"]["top_three"] = nonrelevant
        path = self.root / "changed-baseline-top-three.json"
        path.write_text(json.dumps(baseline), encoding="utf-8")

        with patch.object(
            report_module, "load_frozen_grouprank", return_value=self.phase_a
        ):
            with self.assertRaisesRegex(ValueError, "frozen Stage 2 report SHA256"):
                report_module.build_grouprank_report(
                    self.root / "artifact",
                    judgments_path=JUDGMENTS_PATH,
                    baseline_report_path=path,
                )


class GroupRankWriterAndCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.phase_a, _relevant_ranks = _phase_a_fixture()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _report(self) -> dict[str, object]:
        report_module = _report_module()
        with patch.object(
            report_module, "load_frozen_grouprank", return_value=self.phase_a
        ):
            return report_module.build_grouprank_report(
                self.root / "artifact",
                judgments_path=JUDGMENTS_PATH,
                baseline_report_path=BASELINE_REPORT_PATH,
                selection_path=SELECTION_PATH,
                code_revision=CODE_REVISION,
            )

    def test_report_writers_are_deterministic_and_label_the_exploratory_result(
        self,
    ) -> None:
        report_module = _report_module()
        result = self._report()
        json_path = self.root / "result.json"
        markdown_path = self.root / "result.md"

        first = report_module.write_grouprank_report(
            result,
            report_path=json_path,
            markdown_path=markdown_path,
        )
        first_json = json_path.read_bytes()
        first_markdown = markdown_path.read_bytes()
        second = report_module.write_grouprank_report(
            result,
            report_path=json_path,
            markdown_path=markdown_path,
        )

        self.assertEqual(first, second)
        self.assertEqual(first_json, json_path.read_bytes())
        self.assertEqual(first_markdown, markdown_path.read_bytes())
        self.assertTrue(first_json.endswith(b"\n"))
        self.assertTrue(first_markdown.endswith(b"\n"))
        self.assertEqual(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            first_json.decode("utf-8"),
        )
        markdown = first_markdown.decode("utf-8")
        self.assertIn("Qrel-assisted exploratory", markdown)
        self.assertIn("No expansion was started", markdown)
        self.assertIn("Contextual E5", markdown)
        self.assertIn("Pointwise Qwen", markdown)

    def test_evaluate_cli_requires_sealed_paths_and_avoids_inference_imports(
        self,
    ) -> None:
        report_module = _report_module()
        parser = build_parser()
        report_json = self.root / "report.json"
        report_md = self.root / "report.md"
        argv = [
            "grouprank-pilot",
            "--phase",
            "evaluate",
            "--selection",
            str(SELECTION_PATH),
            "--code-revision",
            CODE_REVISION,
            "--output-dir",
            str(self.root / "phase-a"),
            "--judgments",
            str(JUDGMENTS_PATH),
            "--report-json",
            str(report_json),
            "--report-md",
            str(report_md),
        ]

        args = parser.parse_args(argv)
        self.assertEqual("evaluate", args.phase)
        build = Mock(return_value={"version": 1})
        write = Mock(return_value={"report_path": str(report_json)})
        with (
            patch.object(report_module, "build_grouprank_report", build),
            patch.object(report_module, "write_grouprank_report", write),
            patch.dict(sys.modules, {"trec_mlm.grouprank_inference": None}),
            patch("builtins.print"),
        ):
            status = main(argv)

        self.assertEqual(0, status)
        build.assert_called_once_with(
            self.root / "phase-a",
            judgments_path=JUDGMENTS_PATH,
            selection_path=SELECTION_PATH,
            code_revision=CODE_REVISION,
        )
        write.assert_called_once_with(
            {"version": 1},
            report_path=report_json,
            markdown_path=report_md,
        )

        for missing in ("--judgments", "--report-json", "--report-md"):
            with self.subTest(missing=missing):
                index = argv.index(missing)
                with self.assertRaisesRegex(SystemExit, "evaluate phase requires"):
                    main([*argv[:index], *argv[index + 2 :]])


if __name__ == "__main__":
    unittest.main()
