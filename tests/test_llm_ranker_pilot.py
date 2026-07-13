from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from trec_mlm.label_experiment import load_candidate_manifest
from trec_mlm.llm_ranker_pilot import (
    build_stage1_selection,
    validate_pilot_selection,
)
from trec_mlm.query_knn import load_complete_qrels, load_queries_tsv
from trec_mlm.runio import rankings_from_run


ROOT = Path(__file__).resolve().parents[1]
QUERY_PATH = ROOT / "data/dev/llm_dev_data.tsv"
QRELS_PATH = ROOT / "data/dev/llm_dev_qrels.txt"
CANDIDATE_PATH = ROOT / "reports/trec_2026_discovery_profile_audit.json"
QWEN_RUN_PATH = (
    ROOT
    / "artifacts/label_sweep_diverse/qwen3_embedding_4b"
    / "expert_capability/run1/run.trec"
)

STAGE1_QUERY_IDS = (
    "56fc3b342603e7140040a037",  # issue #11, linguistics
    "57265eeaf1498d1400e8dd76",  # issue #11, landscape
    "570a34484103511400d595c8",  # embedding success, food/drink
    "57298ef11d04691400779530",  # embedding success, mathematics
    "5726eb785951b619008f8276",  # embedding failure, bacterial immunity
    "5729068e1d04691400778f8f",  # embedding failure, Parc de la Villette
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PilotSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.queries = load_queries_tsv(QUERY_PATH)
        cls.qrels = load_complete_qrels(
            QRELS_PATH,
            expected_query_ids=cls.queries,
        )
        cls.candidate_ids = load_candidate_manifest(CANDIDATE_PATH)
        cls.qwen_rankings = rankings_from_run(QWEN_RUN_PATH)

    def _build(self) -> tuple[dict[str, object], dict[str, object]]:
        return build_stage1_selection(
            query_path=QUERY_PATH,
            qrels_path=QRELS_PATH,
            candidate_path=CANDIDATE_PATH,
            qwen_run_path=QWEN_RUN_PATH,
        )

    def test_stage1_selection_is_deterministic_and_qrel_free(self) -> None:
        selection, judgments = self._build()
        repeated_selection, repeated_judgments = self._build()

        self.assertEqual(selection, repeated_selection)
        self.assertEqual(judgments, repeated_judgments)
        self.assertEqual(1, selection["stage"])
        self.assertEqual(2026, selection["seed"])
        self.assertEqual(
            {"queries": _sha256(QUERY_PATH), "candidates": _sha256(CANDIDATE_PATH)},
            selection["input_sha256"],
        )
        self.assertEqual(
            {"qrels": _sha256(QRELS_PATH), "qwen_run": _sha256(QWEN_RUN_PATH)},
            judgments["input_sha256"],
        )
        self.assertNotIn("roles", json.dumps(selection, sort_keys=True))
        self.assertNotIn("grades", json.dumps(selection, sort_keys=True))
        self.assertNotIn("qrels", json.dumps(selection, sort_keys=True))
        self.assertNotIn("qwen_run", json.dumps(selection, sort_keys=True))

        selection_entries = selection["questions"]
        judgment_entries = judgments["questions"]
        self.assertEqual(
            STAGE1_QUERY_IDS,
            tuple(entry["query_id"] for entry in selection_entries),
        )
        self.assertEqual(
            STAGE1_QUERY_IDS,
            tuple(entry["query_id"] for entry in judgment_entries),
        )

        for entry, judgment in zip(selection_entries, judgment_entries, strict=True):
            query_id = entry["query_id"]
            grades = self.qrels[query_id]
            expected_relevant = [
                candidate_id
                for candidate_id, grade in sorted(
                    grades.items(), key=lambda item: (-item[1], item[0])
                )
                if grade > 0
            ][:2]
            expected_confounders = [
                candidate_id
                for candidate_id in self.qwen_rankings[query_id]
                if grades[candidate_id] == 0
                and candidate_id not in expected_relevant
            ][:4]
            excluded = set(expected_relevant) | set(expected_confounders)
            expected_negatives = sorted(
                (
                    candidate_id
                    for candidate_id in self.candidate_ids
                    if grades[candidate_id] == 0 and candidate_id not in excluded
                ),
                key=lambda candidate_id: hashlib.sha256(
                    f"2026|{query_id}|{candidate_id}".encode()
                ).digest(),
            )[: 12 - len(excluded)]

            candidate_ids = entry["candidate_ids"]
            self.assertEqual({"query_id", "question", "candidate_ids"}, set(entry))
            self.assertEqual(self.queries[query_id], entry["question"])
            self.assertEqual(sorted(candidate_ids), candidate_ids)
            self.assertEqual(12, len(candidate_ids))
            self.assertEqual(12, len(set(candidate_ids)))
            self.assertEqual(
                {
                    "relevant": expected_relevant,
                    "confounder": expected_confounders,
                    "negative": expected_negatives,
                },
                judgment["roles"],
            )
            self.assertIn(len(expected_relevant), (1, 2))
            self.assertEqual(4, len(expected_confounders))
            self.assertEqual(
                {candidate_id: grades[candidate_id] for candidate_id in candidate_ids},
                judgment["grades"],
            )

        self.assertEqual(
            {
                "valid": True,
                "stage": 1,
                "question_count": 6,
                "candidate_count": 12,
                "pair_count": 72,
                "manifest_kind": "selection",
            },
            validate_pilot_selection(
                selection,
                expected_stage=1,
                expected_candidate_count=12,
            ),
        )
        self.assertEqual(
            "judgments",
            validate_pilot_selection(
                judgments,
                expected_stage=1,
                expected_candidate_count=12,
            )["manifest_kind"],
        )

    def test_stage1_selection_freezes_issue_11_relevant_labels(self) -> None:
        _selection, judgments = self._build()
        by_query = {entry["query_id"]: entry for entry in judgments["questions"]}

        self.assertEqual(
            ["Linguistics"],
            by_query["56fc3b342603e7140040a037"]["roles"]["relevant"],
        )
        self.assertEqual(
            ["Landscape"],
            by_query["57265eeaf1498d1400e8dd76"]["roles"]["relevant"],
        )

    def test_stage1_selection_module_entrypoint_writes_sorted_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "stage1"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "trec_mlm.llm_ranker_pilot",
                    "--output-dir",
                    str(destination),
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )

            selection_path = destination / "selection.json"
            judgments_path = destination / "selection_judgments.json"
            self.assertEqual(0, completed.returncode, completed.stderr)
            self.assertTrue(selection_path.is_file())
            self.assertTrue(judgments_path.is_file())
            selection, judgments = self._build()
            self.assertEqual(
                json.dumps(selection, indent=2, sort_keys=True) + "\n",
                selection_path.read_text(encoding="utf-8"),
            )
            self.assertEqual(
                json.dumps(judgments, indent=2, sort_keys=True) + "\n",
                judgments_path.read_text(encoding="utf-8"),
            )
            summary = json.loads(completed.stdout)
            self.assertEqual(6, summary["question_count"])
            self.assertEqual(72, summary["pair_count"])

    def test_stage1_selection_validator_rejects_role_overlap_and_bad_size(self) -> None:
        selection, judgments = self._build()

        bad_size = deepcopy(selection)
        bad_size["questions"][0]["candidate_ids"].pop()
        with self.assertRaisesRegex(ValueError, "exactly 12 unique candidates"):
            validate_pilot_selection(
                bad_size,
                expected_stage=1,
                expected_candidate_count=12,
            )

        overlapping = deepcopy(judgments)
        relevant = overlapping["questions"][0]["roles"]["relevant"][0]
        overlapping["questions"][0]["roles"]["negative"][0] = relevant
        with self.assertRaisesRegex(ValueError, "roles overlap"):
            validate_pilot_selection(
                overlapping,
                expected_stage=1,
                expected_candidate_count=12,
            )

    def test_stage1_selection_validator_rejects_changed_input_hash(self) -> None:
        selection, _judgments = self._build()
        changed = deepcopy(selection)
        changed["input_sha256"]["queries"] = "0" * 64

        with self.assertRaisesRegex(ValueError, "input SHA256 changed"):
            validate_pilot_selection(
                changed,
                expected_stage=1,
                expected_candidate_count=12,
            )

    def test_stage1_selection_builder_rejects_missing_queries_and_bad_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            missing_queries = root / "missing_queries.tsv"
            missing_queries.write_text(
                "".join(
                    line
                    for line in QUERY_PATH.read_text(encoding="utf-8").splitlines(
                        keepends=True
                    )
                    if not line.startswith(f"{STAGE1_QUERY_IDS[0]}\t")
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "missing fixed query IDs"):
                build_stage1_selection(
                    query_path=missing_queries,
                    qrels_path=QRELS_PATH,
                    candidate_path=CANDIDATE_PATH,
                    qwen_run_path=QWEN_RUN_PATH,
                )

            wrong_candidates = root / "wrong_candidates.json"
            candidate_payload = json.loads(CANDIDATE_PATH.read_text(encoding="utf-8"))
            candidate_ids = candidate_payload.get("candidate_ids")
            if candidate_ids is None:
                candidate_ids = candidate_payload["plain"]["candidate_ids"]
            candidate_ids.pop()
            wrong_candidates.write_text(
                json.dumps(candidate_payload),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "candidate universe"):
                build_stage1_selection(
                    query_path=QUERY_PATH,
                    qrels_path=QRELS_PATH,
                    candidate_path=wrong_candidates,
                    qwen_run_path=QWEN_RUN_PATH,
                )

            no_relevant_qrels = root / "no_relevant_qrels.txt"
            no_relevant_qrels.write_text(
                QRELS_PATH.read_text(encoding="utf-8").replace(
                    f"{STAGE1_QUERY_IDS[0]} 0 Linguistics 2\n",
                    f"{STAGE1_QUERY_IDS[0]} 0 Linguistics 0\n",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "no relevant candidates"):
                build_stage1_selection(
                    query_path=QUERY_PATH,
                    qrels_path=no_relevant_qrels,
                    candidate_path=CANDIDATE_PATH,
                    qwen_run_path=QWEN_RUN_PATH,
                )

            changed_qwen_run = root / "run.trec"
            shutil.copyfile(QWEN_RUN_PATH, changed_qwen_run)
            with changed_qwen_run.open("a", encoding="utf-8") as handle:
                handle.write("\n")
            with self.assertRaisesRegex(ValueError, "qwen_run input SHA256 changed"):
                build_stage1_selection(
                    query_path=QUERY_PATH,
                    qrels_path=QRELS_PATH,
                    candidate_path=CANDIDATE_PATH,
                    qwen_run_path=changed_qwen_run,
                )


if __name__ == "__main__":
    unittest.main()
