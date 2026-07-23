from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = (
    ROOT
    / ".agents/skills/inspect-query-results/scripts/build_query_explorer.py"
)
SPEC = importlib.util.spec_from_file_location("build_query_explorer", SCRIPT)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import system failure
    raise RuntimeError("could not load query explorer builder")
EXPLORER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPLORER)


class QueryExplorerSchemaTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_predictions(
        self,
        header: tuple[str, ...],
        rows: list[tuple[object, ...]],
    ) -> Path:
        path = self.root / "predictions.tsv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)
        return path

    def load(self, path: Path) -> tuple[list[dict[str, object]], dict[str, str]]:
        return EXPLORER.load_predictions_tsv(
            path,
            query_column=None,
            candidate_column=None,
            rank_column=None,
            score_column=None,
        )

    def test_detects_current_and_legacy_prediction_schemas(self) -> None:
        variants = (
            (
                ("target_query_id", "model_id", "raw_score", "final_rank"),
                ("q1", "model-a", "0.75", 1),
                "raw_score",
            ),
            (
                (
                    "target_query_id",
                    "configuration",
                    "model_id",
                    "aggregation_score",
                    "global_prior",
                    "final_rank",
                ),
                ("q1", "cfg", "model-a", "2.5e-3", "0.1", 1),
                "aggregation_score",
            ),
            (
                (
                    "target_query_id",
                    "configuration",
                    "model_id",
                    "sum_score",
                    "global_prior",
                    "final_rank",
                ),
                ("q1", "cfg", "model-a", "4.25", "0.1", 1),
                "sum_score",
            ),
            (
                (
                    "query_id",
                    "model_id",
                    "primary_score",
                    "diagnostic_score",
                    "primary_rank",
                    "diagnostic_rank",
                ),
                ("q1", "model-a", "0.4", "999", 1, 12),
                "primary_score",
            ),
            (
                ("query_id", "candidate_id", "score", "rank"),
                ("q1", "model-a", "-0.125", 1),
                "score",
            ),
        )
        for header, row, expected_score in variants:
            with self.subTest(expected_score=expected_score):
                path = self.write_predictions(header, [row])
                values, schema = self.load(path)
                self.assertEqual(expected_score, schema["score_column"])
                self.assertEqual("model-a", values[0]["candidate_id"])
                self.assertEqual(1, values[0]["rank"])
                self.assertEqual(str(row[header.index(expected_score)]), values[0]["score_text"])

    def test_explicit_rank_is_authoritative_when_scores_tie(self) -> None:
        path = self.write_predictions(
            ("target_query_id", "model_id", "raw_score", "final_rank"),
            [
                ("q1", "model-b", "0.75", 2),
                ("q1", "model-a", "0.75", 1),
                ("q1", "model-c", "-1.25e-4", 3),
            ],
        )
        rows, _schema = self.load(path)
        grouped, _warnings = EXPLORER.validate_and_group_rows(
            rows, top_candidates=3
        )
        self.assertEqual(
            ["model-a", "model-b", "model-c"],
            [row["candidate_id"] for row in grouped["q1"]],
        )
        self.assertEqual("-1.25e-4", grouped["q1"][2]["score_text"])

    def test_rejects_duplicate_rank(self) -> None:
        path = self.write_predictions(
            ("query_id", "candidate_id", "score", "rank"),
            [("q1", "a", 1, 1), ("q1", "b", 0, 1)],
        )
        rows, _schema = self.load(path)
        with self.assertRaisesRegex(EXPLORER.ExplorerInputError, "duplicate rank"):
            EXPLORER.validate_and_group_rows(rows, top_candidates=10)

    def test_trec_fallback_is_labeled_as_exported_score(self) -> None:
        path = self.root / "run.trec"
        path.write_text(
            "q1 Q0 model-a 1 3 demo\nq1 Q0 model-b 2 2 demo\n",
            encoding="utf-8",
        )
        rows, schema = EXPLORER.load_trec_run(path)
        self.assertEqual("trec_run", schema["input_kind"])
        self.assertIn("exported run score", schema["score_column"])
        self.assertEqual("3", rows[0]["score_text"])

    def test_run_root_with_multiple_prediction_children_is_ambiguous(self) -> None:
        run_root = self.root / "artifacts/normalization"
        for child in ("normalized", "sumscore"):
            directory = run_root / child
            directory.mkdir(parents=True)
            (directory / "predictions.tsv").write_text(
                "query_id\tcandidate_id\tscore\trank\nq1\ta\t1\t1\n",
                encoding="utf-8",
            )
        (run_root / "normalized_run.trec").write_text(
            "q1 Q0 a 1 1 demo\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(EXPLORER.ExplorerInputError, "multiple prediction"):
            EXPLORER.choose_input(
                self.root,
                run_dir=run_root,
                explicit_input=None,
            )

    def test_discovery_parser_accepts_compact_json_and_reversed_fields(self) -> None:
        path = self.root / "discovery.json"
        path.write_text(
            json.dumps(
                {
                    "source-1": {
                        "responses": {
                            "model-a": "A sufficiently detailed historical response.",
                            "model-b": None,
                        },
                        "query": "A historical question",
                    }
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        self.assertEqual(
            [
                (
                    "source-1",
                    "A historical question",
                    {
                        "model-a": "A sufficiently detailed historical response.",
                        "model-b": None,
                    },
                )
            ],
            list(EXPLORER.iter_discovery_queries(path)),
        )

    def test_query_path_remaps_a_stale_absolute_worktree_path(self) -> None:
        run_dir = self.root / "artifacts/demo"
        run_dir.mkdir(parents=True)
        expected = self.root / "data/hidden_queries.tsv"
        expected.parent.mkdir()
        expected.write_text("q1\tQuestion\n", encoding="utf-8")
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "query_path": (
                        "/old/worktree/data/hidden_queries.tsv"
                    )
                }
            ),
            encoding="utf-8",
        )
        self.assertEqual(
            expected,
            EXPLORER.infer_query_path(self.root, run_dir, None),
        )

    def test_evaluation_keeps_all_relevant_llms_beyond_ndcg_cutoff(self) -> None:
        rows = [
            {"query_id": "q1", "candidate_id": f"model-{index}", "rank": index}
            for index in range(1, 13)
        ]
        qrels = {"q1": {f"model-{index}": 1.0 for index in range(1, 13)}}
        evaluation = EXPLORER.build_evaluation(rows, qrels)
        self.assertEqual(
            12,
            len(evaluation["per_query"]["q1"]["ideal_candidates"]),
        )


class QueryExplorerEndToEndTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        (self.root / "data/dev").mkdir(parents=True)
        (self.root / "reports").mkdir()
        self.run_dir = self.root / "artifacts/demo"
        self.run_dir.mkdir(parents=True)

    def tearDown(self) -> None:
        self.directory.cleanup()

    def write_fixture(self) -> None:
        (self.root / "data/dev/llm_dev_data.tsv").write_text(
            "q1\tWho wrote </script><script>alert('query')</script> & why?\n"
            "q2\tWhat is café science?\n",
            encoding="utf-8",
        )
        (self.run_dir / "predictions.tsv").write_text(
            "target_query_id\tmodel_id\traw_score\tfinal_rank\n"
            "q1\tmodel-b\t0.75\t2\n"
            "q1\tmodel-a\t0.75\t1\n"
            "q1\tmodel-c\t-1.25e-4\t3\n"
            "q2\tmodel-b\t0\t1\n"
            "q2\tmodel-c\t-0.5\t2\n"
            "q2\tmodel-a\t-4\t3\n",
            encoding="utf-8",
        )
        profiles = {
            "profile_version": "test-v1",
            "profiles": [
                {
                    "candidate_id": "model-a",
                    "profile_summary": "Observed <b>history</b>, not expertise.",
                    "history": {"record_count": 3},
                    "coverage": {
                        "attempt_share": 2 / 3,
                        "exact_refusal_share": 1 / 3,
                    },
                    "attempt_style": {"word_count_median": 42},
                    "metadata_quality": {"token_logprob_usable_share": 0.5},
                    "attempted_query_terms": [
                        {"term": "science", "support_queries": 2}
                    ],
                    "warnings": ["Do not treat </script><script> as truth."],
                },
                {
                    "candidate_id": "model-b",
                    "profile_summary": "No successful attempts yet.",
                    "history": {"record_count": 3},
                    "coverage": {"attempt_share": 0, "exact_refusal_share": 1},
                    "attempt_style": {"word_count_median": None},
                    "metadata_quality": {"token_logprob_usable_share": 0},
                    "attempted_query_terms": [],
                    "warnings": [],
                },
            ],
        }
        (self.root / "reports/trec_2026_llm_profiles.json").write_text(
            json.dumps(profiles, indent=2), encoding="utf-8"
        )
        discovery = {
            "s1": {
                "query": "who wrote the script query explanation",
                "responses": {
                    "model-a": "Useful </script><script>alert('response')</script> answer with evidence.",
                    "model-b": "I don't know",
                    "model-c": "A separate sufficiently long historical response.",
                },
            },
            "s2": {
                "query": "café science history",
                "responses": {
                    "model-a": "Another sufficiently detailed café science response.",
                    "model-b": "A model-b café science response with enough detail.",
                    "model-c": None,
                },
            },
            "s3": {
                "query": "unrelated sporting event",
                "responses": {
                    "model-a": "An unrelated response that should not win overlap.",
                    "model-b": "An unrelated response that should not win overlap.",
                    "model-c": "An unrelated response that should not win overlap.",
                },
            },
        }
        (self.root / "data/dev/llm_discovery_data.json").write_text(
            json.dumps(discovery, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def payload_from_html(self, html: str) -> dict[str, object]:
        match = re.search(
            r'<script id="explorerData" type="application/json">(.*?)</script>',
            html,
            flags=re.DOTALL,
        )
        self.assertIsNotNone(match)
        return json.loads(match.group(1))

    def run_builder(
        self, *extra_args: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(self.root),
                "--run-dir",
                str(self.run_dir),
                *extra_args,
            ],
            check=False,
            capture_output=True,
            text=True,
        )

    def payload_from_output(self) -> dict[str, Any]:
        return self.payload_from_html(
            (self.run_dir / "query_explorer.html").read_text(encoding="utf-8")
        )

    def write_complete_qrel_fixture(
        self,
        *,
        candidate_count: int,
        positive_candidate: str,
    ) -> None:
        (self.root / "data/dev/llm_dev_data.tsv").write_text(
            "q1\tA grounded test query\n", encoding="utf-8"
        )
        candidate_ids = [
            f"model-{chr(ord('a') + index)}" for index in range(candidate_count)
        ]
        (self.run_dir / "predictions.tsv").write_text(
            "query_id\tcandidate_id\tscore\trank\n"
            + "".join(
                f"q1\t{candidate_id}\t{candidate_count - rank + 1}\t{rank}\n"
                for rank, candidate_id in enumerate(candidate_ids, start=1)
            ),
            encoding="utf-8",
        )
        qrels = self.root / "data/dev/labels.txt"
        qrels.write_text(
            "".join(
                f"q1 0 {candidate_id} {2 if candidate_id == positive_candidate else 0}\n"
                for candidate_id in candidate_ids
            ),
            encoding="utf-8",
        )
        (self.run_dir / "config.json").write_text(
            json.dumps(
                {
                    "qrels_path": "data/dev/labels.txt",
                }
            ),
            encoding="utf-8",
        )

    def write_evidence_bundle(
        self,
        *,
        query_ids: list[str] | None = None,
        display_ids: list[str] | None = None,
        evidence_ids: list[str] | None = None,
        override: dict[str, Any] | None = None,
    ) -> Path:
        with (self.run_dir / "predictions.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            prediction_rows = list(
                csv.DictReader(
                    handle,
                    delimiter="\t",
                )
            )
        candidates_by_query: dict[str, list[str]] = {}
        ranks_by_query: dict[str, dict[str, int]] = {}
        for row in prediction_rows:
            query_id = row.get("query_id") or row.get("target_query_id")
            candidate_id = row.get("candidate_id") or row.get("model_id")
            rank_text = row.get("rank") or row.get("final_rank")
            assert query_id and candidate_id and rank_text
            candidates_by_query.setdefault(query_id, []).append(candidate_id)
            ranks_by_query.setdefault(query_id, {})[candidate_id] = int(rank_text)
        for query_id, candidate_ids in candidates_by_query.items():
            candidate_ids.sort(key=ranks_by_query[query_id].__getitem__)

        with (self.root / "data/dev/llm_dev_data.tsv").open(
            encoding="utf-8", newline=""
        ) as handle:
            query_texts = {
                row[0]: row[1]
                for row in csv.reader(handle, delimiter="\t")
            }
        qrels: dict[str, dict[str, float]] = {}
        qrels_path = self.root / "data/dev/labels.txt"
        if qrels_path.is_file():
            for line in qrels_path.read_text(encoding="utf-8").splitlines():
                query_id, _iteration, candidate_id, grade = line.split()
                qrels.setdefault(query_id, {})[candidate_id] = float(grade)

        selected_query_ids = query_ids or list(candidates_by_query)
        query_payload: dict[str, dict[str, Any]] = {}
        for query_id in selected_query_ids:
            source_query_id = (
                query_id if query_id in candidates_by_query else next(iter(candidates_by_query))
            )
            candidate_ids = candidates_by_query[source_query_id]
            final_ranks = ranks_by_query[source_query_id]
            selected_display_ids = display_ids or candidate_ids
            positive_ids = {
                candidate_id
                for candidate_id in candidate_ids
                if qrels.get(source_query_id, {}).get(candidate_id, 0.0) > 0.0
            }
            selected_evidence_ids = evidence_ids or [
                candidate_id
                for candidate_id in candidate_ids
                if final_ranks[candidate_id] <= 10
                or candidate_id in positive_ids
            ]
            candidate_payload: dict[str, dict[str, Any]] = {}
            for candidate_id in candidate_ids:
                rank = final_ranks[candidate_id]
                grade = qrels.get(source_query_id, {}).get(candidate_id, 0.0)
                candidate_payload[candidate_id] = {
                    "candidate_id": candidate_id,
                    "qrel_grade": grade,
                    "is_positive": grade > 0.0,
                    "ranks": {
                        "dense_reference": rank,
                        "profile_only": rank,
                        "dense_profile_equal": rank,
                        "raw_hierarchical": rank,
                        "raw_hierarchical_dense_equal": rank,
                        "best_record_cosine": rank,
                    },
                    "best_record_cosine": 1.0 / rank,
                    "mean_top_10_cosine": 0.5 / rank,
                    "evidence": (
                        [
                            {
                                "record_id": f"{candidate_id}-record-{index}",
                                "source_query_id": f"source-{index}",
                                "question": f"Historical question {index}",
                                "response": f"Historical response {index}",
                                "similarity": 1.0 / (rank + index),
                                "behavior": "substantive response",
                            }
                            for index in range(1, 4)
                        ]
                        if candidate_id in selected_evidence_ids
                        else []
                    ),
                }
            positive_ranks = [
                final_ranks[candidate_id]
                for candidate_id in candidate_ids
                if qrels.get(source_query_id, {}).get(candidate_id, 0.0) > 0.0
            ]
            best_positive_rank = min(positive_ranks, default=None)
            query_payload[query_id] = {
                "query_id": query_id,
                "query_text": query_texts.get(source_query_id, "Wrong query"),
                "display_candidate_ids": selected_display_ids,
                "evidence_candidate_ids": selected_evidence_ids,
                "component_metrics": {
                    system: {"ndcg_at_10": 0.0, "reciprocal_rank": 0.0}
                    for system in (
                        "dense_reference",
                        "profile_only",
                        "dense_profile_equal",
                        "raw_hierarchical",
                        "raw_hierarchical_dense_equal",
                    )
                },
                "cohorts": ["zero fused nDCG@10"],
                "context_risk": False,
                "retrieval_outcome": (
                    "top-1 success"
                    if best_positive_rank == 1
                    else "top-10 but not top-1"
                    if best_positive_rank is not None and best_positive_rank <= 10
                    else "no relevant candidate in top 10"
                ),
                "mechanism": (
                    "relevant candidate ranked first"
                    if best_positive_rank == 1
                    else "top-10 retrieval, top-1 miss"
                    if best_positive_rank is not None and best_positive_rank <= 10
                    else "evidence/routing gap"
                    if best_positive_rank is not None
                    else "no development-positive candidate"
                ),
                "candidates": candidate_payload,
            }
        candidate_count = len(next(iter(candidates_by_query.values())))
        payload: dict[str, Any] = {
            "schema_version": 1,
            "meta": {
                "query_count": len(selected_query_ids),
                "candidate_count": candidate_count,
                "evidence_per_candidate": 3,
                "qrels_usage": "Post-freeze evaluation and failure inspection only",
            },
            "queries": query_payload,
        }
        if override:
            payload.update(override)
        path = self.root / "evidence.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path

    def test_no_bundle_preserves_existing_payload_shape(self) -> None:
        self.write_fixture()
        completed = self.run_builder(
            "--top-candidates", "2", "--no-responses"
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        payload = self.payload_from_output()
        self.assertNotIn("grounded_evidence", payload["meta"])
        self.assertNotIn("grounded", payload["queries"][0])
        self.assertEqual(
            ["model-a", "model-b"],
            [
                result["candidate_id"]
                for result in payload["queries"][0]["results"]
            ],
        )

    def test_bundle_adds_positive_below_limit_without_reordering(self) -> None:
        self.write_complete_qrel_fixture(
            candidate_count=12, positive_candidate="model-l"
        )
        display_ids = [
            f"model-{chr(ord('a') + index)}" for index in range(10)
        ] + ["model-l"]
        bundle = self.write_evidence_bundle(
            display_ids=display_ids
        )
        completed = self.run_builder(
            "--top-candidates",
            "10",
            "--evidence-bundle",
            str(bundle),
            "--no-responses",
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        payload = self.payload_from_output()
        query = payload["queries"][0]
        self.assertEqual(
            display_ids,
            [result["candidate_id"] for result in query["results"]],
        )
        self.assertEqual(12, query["results"][-1]["rank"])
        self.assertEqual(2.0, query["results"][-1]["grade"])
        self.assertEqual(
            payload["meta"]["grounded_evidence"],
            {
                "query_count": 1,
                "candidate_count": 12,
                "evidence_per_candidate": 3,
                "qrels_usage": "Post-freeze evaluation and failure inspection only",
            },
        )
        grounded = query["grounded"]
        self.assertEqual("q1", grounded["query_id"])
        self.assertEqual(
            "no relevant candidate in top 10",
            grounded["retrieval_outcome"],
        )
        self.assertEqual("evidence/routing gap", grounded["mechanism"])
        self.assertEqual(display_ids, grounded["evidence_candidate_ids"])
        self.assertTrue(
            all(
                len(grounded["candidates"][candidate_id]["evidence"]) == 3
                for candidate_id in display_ids
            )
        )

    def test_bundle_inventory_mismatch_leaves_existing_html_unchanged(self) -> None:
        self.write_fixture()
        output = self.run_dir / "query_explorer.html"
        output.write_text("known-good", encoding="utf-8")
        bundle = self.write_evidence_bundle(query_ids=["wrong-query"])
        completed = self.run_builder(
            "--evidence-bundle",
            str(bundle),
            "--output",
            str(output),
            "--no-responses",
        )
        self.assertEqual(2, completed.returncode)
        self.assertIn("evidence bundle query inventory differs", completed.stderr)
        self.assertEqual("known-good", output.read_text(encoding="utf-8"))

    def test_bundle_requires_complete_qrels(self) -> None:
        self.write_complete_qrel_fixture(
            candidate_count=4, positive_candidate="model-d"
        )
        (self.root / "data/dev/labels.txt").write_text(
            "q1 0 model-a 0\nq1 0 model-d 2\n", encoding="utf-8"
        )
        bundle = self.write_evidence_bundle(
            display_ids=["model-a", "model-b", "model-c", "model-d"]
        )
        completed = self.run_builder(
            "--top-candidates",
            "4",
            "--evidence-bundle",
            str(bundle),
            "--no-responses",
        )
        self.assertEqual(2, completed.returncode)
        self.assertIn("evidence bundle requires complete qrels", completed.stderr)

    def test_bundle_rejects_extra_qrel_queries(self) -> None:
        self.write_complete_qrel_fixture(
            candidate_count=4, positive_candidate="model-d"
        )
        bundle = self.write_evidence_bundle(
            display_ids=["model-a", "model-b", "model-c", "model-d"]
        )
        qrels_path = self.root / "data/dev/labels.txt"
        with qrels_path.open("a", encoding="utf-8") as handle:
            for candidate_id in ("model-a", "model-b", "model-c", "model-d"):
                handle.write(f"q2 0 {candidate_id} 0\n")

        completed = self.run_builder(
            "--top-candidates",
            "4",
            "--evidence-bundle",
            str(bundle),
            "--no-responses",
        )

        self.assertEqual(2, completed.returncode)
        self.assertIn("evidence bundle requires complete qrels", completed.stderr)

    def test_bundle_query_contract_mismatch_fails_closed(self) -> None:
        self.write_complete_qrel_fixture(
            candidate_count=4, positive_candidate="model-d"
        )
        bundle = self.write_evidence_bundle(
            display_ids=["model-a", "model-b", "model-c", "model-d"]
        )
        payload = json.loads(bundle.read_text(encoding="utf-8"))
        payload["queries"]["q1"]["query_text"] = "Different text"
        bundle.write_text(json.dumps(payload), encoding="utf-8")
        output = self.run_dir / "query_explorer.html"
        output.write_text("known-good", encoding="utf-8")

        completed = self.run_builder(
            "--top-candidates",
            "4",
            "--evidence-bundle",
            str(bundle),
            "--output",
            str(output),
            "--no-responses",
        )

        self.assertEqual(2, completed.returncode)
        self.assertIn("evidence query text differs for q1", completed.stderr)
        self.assertEqual("known-good", output.read_text(encoding="utf-8"))

    def test_bundle_cannot_reorder_qrel_selected_rows(self) -> None:
        self.write_complete_qrel_fixture(
            candidate_count=4, positive_candidate="model-d"
        )
        bundle = self.write_evidence_bundle(
            display_ids=["model-d", "model-a", "model-b", "model-c"]
        )

        completed = self.run_builder(
            "--top-candidates",
            "4",
            "--evidence-bundle",
            str(bundle),
            "--no-responses",
        )

        self.assertEqual(2, completed.returncode)
        self.assertIn("evidence display ordering differs for q1", completed.stderr)

    def test_bundle_rejects_nested_schema_differences(self) -> None:
        self.write_complete_qrel_fixture(
            candidate_count=4, positive_candidate="model-d"
        )
        bundle = self.write_evidence_bundle(
            display_ids=["model-a", "model-b", "model-c", "model-d"]
        )
        payload = json.loads(bundle.read_text(encoding="utf-8"))
        del payload["queries"]["q1"]["candidates"]["model-a"]["ranks"][
            "dense_reference"
        ]
        bundle.write_text(json.dumps(payload), encoding="utf-8")

        completed = self.run_builder(
            "--top-candidates",
            "4",
            "--evidence-bundle",
            str(bundle),
            "--no-responses",
        )

        self.assertEqual(2, completed.returncode)
        self.assertIn("evidence bundle candidate schema differs", completed.stderr)

    def test_bundle_reports_malformed_nested_types_as_input_errors(self) -> None:
        self.write_complete_qrel_fixture(
            candidate_count=4, positive_candidate="model-d"
        )
        bundle = self.write_evidence_bundle(
            display_ids=["model-a", "model-b", "model-c", "model-d"]
        )
        payload = json.loads(bundle.read_text(encoding="utf-8"))
        payload["queries"]["q1"]["retrieval_outcome"] = []
        bundle.write_text(json.dumps(payload), encoding="utf-8")

        completed = self.run_builder(
            "--top-candidates",
            "4",
            "--evidence-bundle",
            str(bundle),
            "--no-responses",
        )

        self.assertEqual(2, completed.returncode)
        self.assertIn("evidence bundle query schema differs", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_grounded_bundle_disables_legacy_discovery_matching(self) -> None:
        self.write_complete_qrel_fixture(
            candidate_count=12, positive_candidate="model-l"
        )
        display_ids = [
            f"model-{chr(ord('a') + index)}" for index in range(10)
        ] + ["model-l"]
        bundle = self.write_evidence_bundle(display_ids=display_ids)
        (self.root / "data/dev/llm_discovery_data.json").write_text(
            json.dumps(
                {
                    "source-1": {
                        "query": "A grounded test query with historical context",
                        "responses": {
                            "model-l": (
                                "A long legacy response that would otherwise be "
                                "selected by lexical overlap for the qrel-added row."
                            )
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        completed = self.run_builder(
            "--top-candidates",
            "10",
            "--evidence-bundle",
            str(bundle),
        )

        self.assertEqual(0, completed.returncode, completed.stderr)
        payload = self.payload_from_output()
        self.assertEqual({}, payload["responses"])
        self.assertIsNone(payload["meta"]["discovery_source"])
        self.assertEqual(0, payload["meta"]["discovery_query_count"])
        self.assertEqual(0, payload["meta"]["embedded_response_count"])

    def test_grounded_bundle_rejects_explicit_legacy_discovery(self) -> None:
        self.write_complete_qrel_fixture(
            candidate_count=4, positive_candidate="model-d"
        )
        bundle = self.write_evidence_bundle(
            display_ids=["model-a", "model-b", "model-c", "model-d"]
        )
        discovery = self.root / "discovery.json"
        discovery.write_text("{}", encoding="utf-8")

        completed = self.run_builder(
            "--top-candidates",
            "4",
            "--evidence-bundle",
            str(bundle),
            "--discovery",
            str(discovery),
        )

        self.assertEqual(2, completed.returncode)
        self.assertIn(
            "--discovery cannot be combined with --evidence-bundle",
            completed.stderr,
        )

    def test_bundle_rejects_false_semantic_classifications(self) -> None:
        for field, false_value in (
            ("retrieval_outcome", "top-1 success"),
            ("mechanism", "relevant candidate ranked first"),
        ):
            with self.subTest(field=field):
                self.write_complete_qrel_fixture(
                    candidate_count=4, positive_candidate="model-b"
                )
                bundle = self.write_evidence_bundle(
                    display_ids=[
                        "model-a",
                        "model-b",
                        "model-c",
                        "model-d",
                    ]
                )
                payload = json.loads(bundle.read_text(encoding="utf-8"))
                payload["queries"]["q1"][field] = false_value
                bundle.write_text(json.dumps(payload), encoding="utf-8")

                completed = self.run_builder(
                    "--top-candidates",
                    "4",
                    "--evidence-bundle",
                    str(bundle),
                    "--no-responses",
                )

                self.assertEqual(2, completed.returncode)
                self.assertIn(
                    "evidence classification differs for q1",
                    completed.stderr,
                )

    def test_bundle_derives_required_top_ten_and_positive_evidence_ids(
        self,
    ) -> None:
        self.write_complete_qrel_fixture(
            candidate_count=12, positive_candidate="model-l"
        )
        top_ten = [
            f"model-{chr(ord('a') + index)}" for index in range(10)
        ]
        bundle = self.write_evidence_bundle(
            display_ids=[*top_ten, "model-l"],
            evidence_ids=top_ten,
        )

        completed = self.run_builder(
            "--top-candidates",
            "10",
            "--evidence-bundle",
            str(bundle),
            "--no-responses",
        )

        self.assertEqual(2, completed.returncode)
        self.assertIn(
            "evidence candidate ordering differs for q1",
            completed.stderr,
        )

    def test_grounded_bundle_requires_top_ten_display_depth(self) -> None:
        self.write_complete_qrel_fixture(
            candidate_count=12, positive_candidate="model-l"
        )
        display_ids = [
            f"model-{chr(ord('a') + index)}" for index in range(10)
        ] + ["model-l"]
        bundle = self.write_evidence_bundle(display_ids=display_ids)

        completed = self.run_builder(
            "--top-candidates",
            "9",
            "--evidence-bundle",
            str(bundle),
            "--no-responses",
        )

        self.assertEqual(2, completed.returncode)
        self.assertIn(
            "grounded evidence requires --top-candidates of at least 10",
            completed.stderr,
        )

    def test_skill_command_builds_safe_self_contained_explorer(self) -> None:
        self.write_fixture()
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(self.root),
                "--run-dir",
                str(self.run_dir),
                "--top-candidates",
                "2",
                "--responses-per-candidate",
                "2",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        output = self.run_dir / "query_explorer.html"
        self.assertTrue(output.is_file())
        html = output.read_text(encoding="utf-8")
        payload = self.payload_from_html(html)

        self.assertNotIn("</script><script>", html)
        self.assertNotIn('src="http', html)
        self.assertNotIn('href="http', html)
        self.assertIn("\\u003c/script\\u003e", html)
        self.assertIn("value !== null && value !== undefined", html)
        self.assertIn('id="queryToggle"', html)
        self.assertIn("const formatScore", html)
        self.assertIn('button.setAttribute("aria-pressed"', html)
        self.assertIn("Exact system score", html)
        self.assertEqual(2, payload["meta"]["query_count"])
        self.assertEqual(2, payload["meta"]["candidate_limit"])
        self.assertEqual(4, payload["meta"]["displayed_result_count"])
        self.assertEqual("raw_score", payload["meta"]["score_field"])
        self.assertEqual(
            ["model-a", "model-b"],
            [row["candidate_id"] for row in payload["queries"][0]["results"]],
        )
        self.assertEqual("test-v1", payload["meta"]["profile_version"])
        self.assertIn("model-a", payload["profiles"])
        self.assertTrue(payload["responses"]["q1"]["model-a"])
        self.assertNotIn("configuration", payload["queries"][0]["results"][0])
        self.assertNotIn("global_prior", payload["queries"][0]["results"][0])

    def test_succeeds_without_profiles_or_discovery(self) -> None:
        (self.root / "data/dev/llm_dev_data.tsv").write_text(
            "q1\tA test query\n", encoding="utf-8"
        )
        (self.run_dir / "predictions.tsv").write_text(
            "query_id\tcandidate_id\tscore\trank\nq1\tmodel-a\t0.2\t1\n",
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(self.root),
                "--run-dir",
                str(self.run_dir),
                "--queries",
                str(self.root / "data/dev/llm_dev_data.tsv"),
                "--no-responses",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        payload = self.payload_from_html(
            (self.run_dir / "query_explorer.html").read_text(encoding="utf-8")
        )
        self.assertEqual({}, payload["profiles"])
        self.assertEqual({}, payload["responses"])
        self.assertFalse(payload["meta"]["labels_included"])
        self.assertIsNone(payload["queries"][0]["evaluation"])

    def test_completed_dev_run_adds_optional_labels_and_failure_metrics(self) -> None:
        (self.root / "data/dev/llm_dev_data.tsv").write_text(
            "q1\tFirst evaluation query\nq2\tSecond evaluation query\n",
            encoding="utf-8",
        )
        qrels = self.root / "data/dev/labels.txt"
        qrels.write_text(
            "q1 0 model-a 2\n"
            "q1 0 model-b 0\n"
            "q1 0 model-c 0\n"
            "q2 0 model-a 0\n"
            "q2 0 model-b 0\n"
            "q2 0 model-c 1\n",
            encoding="utf-8",
        )
        (self.run_dir / "predictions.tsv").write_text(
            "query_id\tcandidate_id\tscore\trank\n"
            "q1\tmodel-b\t0.9\t1\n"
            "q1\tmodel-a\t0.8\t2\n"
            "q1\tmodel-c\t0.7\t3\n"
            "q2\tmodel-c\t0.9\t1\n"
            "q2\tmodel-b\t0.8\t2\n"
            "q2\tmodel-a\t0.7\t3\n",
            encoding="utf-8",
        )
        (self.run_dir / "config.json").write_text(
            json.dumps(
                {
                    "query_path": "data/dev/llm_dev_data.tsv",
                    "qrels_path": "data/dev/labels.txt",
                }
            ),
            encoding="utf-8",
        )
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(self.root),
                "--run-dir",
                str(self.run_dir),
                "--queries",
                str(self.root / "data/dev/llm_dev_data.tsv"),
                "--no-responses",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        html = (self.run_dir / "query_explorer.html").read_text(encoding="utf-8")
        payload = self.payload_from_html(html)
        self.assertTrue(payload["meta"]["labels_included"])
        self.assertEqual("data/dev/labels.txt", payload["meta"]["qrels_source"])
        self.assertAlmostEqual(0.75, payload["meta"]["mean_reciprocal_rank"])
        self.assertAlmostEqual(
            (0.6309297535714575 + 1) / 2,
            payload["meta"]["macro_ndcg_at_10"],
        )
        first = payload["queries"][0]
        self.assertEqual(2, first["evaluation"]["first_relevant_rank"])
        self.assertEqual(1.0, first["evaluation"]["ideal_ndcg_at_10"])
        self.assertEqual(
            [{"candidate_id": "model-a", "grade": 2.0, "system_rank": 2}],
            first["evaluation"]["ideal_candidates"],
        )
        self.assertEqual(2, first["results"][1]["grade"])
        self.assertTrue(first["results"][1]["judged"])
        self.assertIn("Zero nDCG@10 only", html)
        self.assertIn("G0/G1/G2 badges", html)
        self.assertIn("All relevant LLMs from development labels", html)

        blind_output = self.run_dir / "blind.html"
        blind = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(self.root),
                "--run-dir",
                str(self.run_dir),
                "--queries",
                str(self.root / "data/dev/llm_dev_data.tsv"),
                "--no-responses",
                "--no-labels",
                "--output",
                str(blind_output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, blind.returncode, blind.stderr)
        blind_payload = self.payload_from_html(blind_output.read_text(encoding="utf-8"))
        self.assertFalse(blind_payload["meta"]["labels_included"])

    def test_failure_does_not_replace_existing_output(self) -> None:
        (self.root / "data/dev/llm_dev_data.tsv").write_text(
            "q1\tA test query\n", encoding="utf-8"
        )
        (self.run_dir / "predictions.tsv").write_text(
            "query_id\tcandidate_id\tscore\trank\n"
            "q1\ta\tNaN\t1\n",
            encoding="utf-8",
        )
        output = self.run_dir / "query_explorer.html"
        output.write_text("existing", encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(self.root),
                "--run-dir",
                str(self.run_dir),
                "--no-responses",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, completed.returncode)
        self.assertEqual("existing", output.read_text(encoding="utf-8"))

    def test_custom_title_markers_are_data_and_sources_cannot_be_overwritten(self) -> None:
        (self.root / "data/dev/llm_dev_data.tsv").write_text(
            "q1\tA test query\n", encoding="utf-8"
        )
        predictions = self.run_dir / "predictions.tsv"
        original_predictions = (
            "query_id\tcandidate_id\tscore\trank\nq1\tmodel-a\t0.2\t1\n"
        )
        predictions.write_text(original_predictions, encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(self.root),
                "--run-dir",
                str(self.run_dir),
                "--no-responses",
                "--title",
                "__EXPLORER_PAYLOAD__ and __EXPLORER_TITLE__",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        html = (self.run_dir / "query_explorer.html").read_text(encoding="utf-8")
        payload = self.payload_from_html(html)
        self.assertEqual(
            "__EXPLORER_PAYLOAD__ and __EXPLORER_TITLE__",
            payload["meta"]["title"],
        )

        overwrite = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(self.root),
                "--run-dir",
                str(self.run_dir),
                "--no-responses",
                "--output",
                str(predictions),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, overwrite.returncode)
        self.assertIn("must end in .html", overwrite.stderr)
        self.assertEqual(original_predictions, predictions.read_text(encoding="utf-8"))

    def test_atomic_writer_preserves_existing_temp_file_and_rejects_output_symlink(self) -> None:
        (self.root / "data/dev/llm_dev_data.tsv").write_text(
            "q1\tA test query\n", encoding="utf-8"
        )
        (self.run_dir / "predictions.tsv").write_text(
            "query_id\tcandidate_id\tscore\trank\nq1\tmodel-a\t0.2\t1\n",
            encoding="utf-8",
        )
        legacy_temp = self.run_dir / "query_explorer.html.tmp"
        legacy_temp.write_text("unrelated artifact", encoding="utf-8")
        completed = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(self.root),
                "--run-dir",
                str(self.run_dir),
                "--no-responses",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        self.assertEqual("unrelated artifact", legacy_temp.read_text(encoding="utf-8"))
        self.assertTrue((self.run_dir / "query_explorer.html").is_file())
        self.assertEqual([], list(self.run_dir.glob(".query_explorer.html.*.tmp")))

        protected = self.run_dir / "protected.html"
        protected.write_text("protected artifact", encoding="utf-8")
        output = self.run_dir / "linked-output.html"
        output.symlink_to(protected.name)
        rejected = subprocess.run(
            [
                sys.executable,
                str(SCRIPT),
                "--repo-root",
                str(self.root),
                "--run-dir",
                str(self.run_dir),
                "--no-responses",
                "--output",
                str(output),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(2, rejected.returncode)
        self.assertIn("must not be a symlink", rejected.stderr)
        self.assertEqual("protected artifact", protected.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
