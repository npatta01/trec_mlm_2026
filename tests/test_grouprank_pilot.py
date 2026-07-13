from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from trec_mlm.grouprank import GroupRankRequest, parse_grouprank_scores
from trec_mlm.grouprank_inference import (
    ATTENTION_BACKEND,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "artifacts/llm_label_ranker/pilot/stage2/selection.json"
JUDGMENTS_PATH = (
    ROOT / "artifacts/llm_label_ranker/pilot/stage2/selection_judgments.json"
)
QUERY_PATH = ROOT / "data/dev/llm_dev_data.tsv"
CANDIDATE_PATH = ROOT / "reports/trec_2026_discovery_profile_audit.json"


def _pilot_module():
    try:
        return importlib.import_module("trec_mlm.grouprank_pilot")
    except (
        ImportError
    ) as error:  # Produces an intentional RED assertion, not collection error.
        raise AssertionError("GroupRank Phase A module is missing") from error


def _runtime() -> dict[str, object]:
    return {
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "resolved_revision": MODEL_REVISION,
        "dtype": MODEL_DTYPE,
        "license": MODEL_LICENSE,
        "parameter_count": 7_615_616_512,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "attention_backend": ATTENTION_BACKEND,
        "vllm_version": "0.24.0",
        "torch_version": "2.11.0",
        "hip_version": "7.2.0",
        "device_properties": "fixture Radeon 8060S",
        "code_revision": "a" * 40,
        "worktree_status": "clean",
        "query_path": str(QUERY_PATH),
        "candidate_path": str(CANDIDATE_PATH),
    }


def _answer(request: GroupRankRequest) -> str:
    scores = {
        handle: sum(candidate.encode("utf-8")) % 11
        for handle, candidate in request.handle_to_candidate.items()
    }
    return (
        "<reason>Compared every expert.</reason>\n<answer>"
        + json.dumps(scores)
        + "</answer>"
    )


def _raw_record(
    request: GroupRankRequest, *, raw_text: str | None = None
) -> dict[str, object]:
    return {
        "query_id": request.query_id,
        "repetition": request.repetition,
        "group": request.group,
        "ordered_candidate_ids": list(request.ordered_candidate_ids),
        "handle_to_candidate": dict(request.handle_to_candidate),
        "raw_text": _answer(request) if raw_text is None else raw_text,
        "prompt_token_count": 300,
        "output_token_count": 80,
        "finish_reason": "stop",
        "sampling_seed": request.sampling_seed,
    }


def _valid_infer(requests: list[GroupRankRequest] | tuple[GroupRankRequest, ...]):
    return [_raw_record(request) for request in requests]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GroupRankPhaseAContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_phase_a_signature_and_inputs_are_qrel_free(self) -> None:
        pilot = _pilot_module()
        signature = inspect.signature(pilot.execute_grouprank_phase_a)
        self.assertEqual(
            ("selection", "infer_batch", "runtime_metadata", "output_dir"),
            tuple(signature.parameters),
        )
        self.assertFalse(
            any(
                "qrel" in name.lower() or "judgment" in name.lower()
                for name in signature.parameters
            )
        )

        first = json.loads(JUDGMENTS_PATH.read_text(encoding="utf-8"))
        second = deepcopy(first)
        entry = second["questions"][0]
        relevant = entry["roles"]["relevant"][0]
        replacement = entry["roles"]["negative"][0]
        entry["grades"][relevant], entry["grades"][replacement] = (
            entry["grades"][replacement],
            entry["grades"][relevant],
        )
        entry["roles"]["relevant"][0] = replacement
        entry["roles"]["negative"][0] = relevant

        for index, judgments in enumerate((first, second)):
            with self.subTest(index=index):
                infer = Mock(side_effect=AssertionError("inference must not run"))
                with self.assertRaisesRegex(ValueError, "qrel-free|Stage 2"):
                    pilot.execute_grouprank_phase_a(
                        judgments,
                        infer_batch=infer,
                        runtime_metadata=_runtime(),
                        output_dir=self.root / f"judgments-{index}",
                    )
                infer.assert_not_called()

    def test_phase_a_persists_exact_complete_success_inventory(self) -> None:
        pilot = _pilot_module()
        destination = self.root / "success"

        result = pilot.execute_grouprank_phase_a(
            self.selection,
            infer_batch=_valid_infer,
            runtime_metadata=_runtime(),
            output_dir=destination,
        )

        self.assertEqual(48, len(result.prompts))
        self.assertEqual(48, len(result.raw_outputs))
        self.assertEqual(12, len(result.candidate_scores))
        self.assertEqual(12, len(result.rankings))
        observations = Counter(
            (row["query_id"], candidate)
            for row in result.raw_outputs
            for candidate in row["handle_to_candidate"].values()
        )
        self.assertEqual({2}, set(observations.values()))
        self.assertTrue(
            all(
                len(scores) == 20 and len(result.rankings[query_id]) == 20
                for query_id, scores in result.candidate_scores.items()
            )
        )
        self.assertEqual(
            {
                "config.json",
                "selection.json",
                "prompts.jsonl",
                "raw_outputs.jsonl",
                "scores.tsv",
                "rankings.json",
                "runtime.json",
                "ranking_receipt.json",
            },
            {path.name for path in destination.iterdir()},
        )
        for name in ("config.json", "selection.json", "rankings.json", "runtime.json"):
            self.assertTrue((destination / name).read_bytes().endswith(b"\n"))
        for name in ("prompts.jsonl", "raw_outputs.jsonl", "scores.tsv"):
            payload = (destination / name).read_bytes()
            self.assertTrue(payload.endswith(b"\n"))
            self.assertNotIn(b"\n\n", payload)

        receipt = pilot.validate_grouprank_artifact(destination)
        loaded = pilot.load_frozen_grouprank(destination)
        self.assertTrue(receipt["qrel_free_phase"])
        self.assertEqual(result.rankings, loaded.rankings)
        self.assertEqual(result.candidate_scores, loaded.candidate_scores)

    def test_phase_a_refuses_an_existing_destination_before_inference(self) -> None:
        pilot = _pilot_module()
        destination = self.root / "exists"
        destination.mkdir()
        infer = Mock(side_effect=AssertionError("inference must not run"))

        with self.assertRaisesRegex(ValueError, "destination already exists"):
            pilot.execute_grouprank_phase_a(
                self.selection,
                infer_batch=infer,
                runtime_metadata=_runtime(),
                output_dir=destination,
            )

        infer.assert_not_called()


class GroupRankFailurePersistenceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_failure_raw_outputs_are_fsynced_before_parse_and_atomically_retained(
        self,
    ) -> None:
        pilot = _pilot_module()
        for kind in ("duplicate", "length"):
            with self.subTest(kind=kind):
                destination = self.root / kind

                def infer(requests):
                    records = [_raw_record(request) for request in requests]
                    if kind == "duplicate":
                        request = requests[0]
                        pairs = [
                            f'"{handle}":5' for handle in request.handle_to_candidate
                        ]
                        pairs.append(f'"{next(iter(request.handle_to_candidate))}":6')
                        records[0]["raw_text"] = (
                            "<reason>Compared every expert.</reason>\n<answer>{"
                            + ",".join(pairs)
                            + "}</answer>"
                        )
                    else:
                        records[0]["finish_reason"] = "length"
                    return records

                fsynced: list[str] = []
                observed_parse_calls = 0
                real_fsync = os.fsync

                def tracking_fsync(file_descriptor: int) -> None:
                    target = Path(os.readlink(f"/proc/self/fd/{file_descriptor}"))
                    fsynced.append(target.name)
                    real_fsync(file_descriptor)

                def observing_parser(raw: str, *, expected_handles):
                    nonlocal observed_parse_calls
                    observed_parse_calls += 1
                    staging = tuple(self.root.glob(f".{kind}.*"))
                    self.assertEqual(1, len(staging))
                    raw_path = staging[0] / "raw_outputs.jsonl"
                    self.assertEqual(
                        48,
                        len(raw_path.read_text(encoding="utf-8").splitlines()),
                    )
                    self.assertIn("raw_outputs.jsonl", fsynced)
                    return parse_grouprank_scores(
                        raw,
                        expected_handles=expected_handles,
                    )

                with (
                    patch.object(pilot.os, "fsync", side_effect=tracking_fsync),
                    patch.object(
                        pilot,
                        "parse_grouprank_scores",
                        side_effect=observing_parser,
                    ),
                ):
                    with self.assertRaises(pilot.GroupRankParseFailure) as raised:
                        pilot.execute_grouprank_phase_a(
                            self.selection,
                            infer_batch=infer,
                            runtime_metadata=_runtime(),
                            output_dir=destination,
                        )

                self.assertGreater(observed_parse_calls, 0)
                self.assertEqual(48, len(raised.exception.result.raw_outputs))
                self.assertEqual(
                    {
                        "config.json",
                        "selection.json",
                        "prompts.jsonl",
                        "raw_outputs.jsonl",
                        "runtime.json",
                        "failure_receipt.json",
                    },
                    {path.name for path in destination.iterdir()},
                )
                self.assertFalse((destination / "scores.tsv").exists())
                self.assertFalse((destination / "rankings.json").exists())
                receipt = json.loads(
                    (destination / "failure_receipt.json").read_text(encoding="utf-8")
                )
                self.assertEqual(48, receipt["execution"]["planned_request_count"])
                self.assertEqual(48, receipt["execution"]["persisted_raw_output_count"])
                self.assertGreaterEqual(receipt["execution"]["parse_failure_count"], 1)
                self.assertFalse(receipt["rankings_available"])
                self.assertFalse(receipt["scores_available"])

    def test_exception_before_complete_raw_inventory_retains_and_reports_staging(
        self,
    ) -> None:
        pilot = _pilot_module()
        destination = self.root / "inference-error"

        def fail_inference(_requests):
            raise RuntimeError("fixture inference failed")

        with self.assertRaisesRegex(RuntimeError, "staging retained at"):
            pilot.execute_grouprank_phase_a(
                self.selection,
                infer_batch=fail_inference,
                runtime_metadata=_runtime(),
                output_dir=destination,
            )

        self.assertFalse(destination.exists())
        staging = tuple(self.root.glob(".inference-error.*"))
        self.assertEqual(1, len(staging))
        self.assertEqual(
            {"config.json", "selection.json", "prompts.jsonl"},
            {path.name for path in staging[0].iterdir()},
        )
        self.assertFalse((staging[0] / "raw_outputs.jsonl").exists())


class GroupRankArtifactValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
        self.pilot = _pilot_module()
        self.artifact = self.root / "valid"
        self.pilot.execute_grouprank_phase_a(
            self.selection,
            infer_batch=_valid_infer,
            runtime_metadata=_runtime(),
            output_dir=self.artifact,
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    @staticmethod
    def _write_json(path: Path, payload: object) -> None:
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def _rehash_core(cls, artifact: Path, name: str) -> None:
        receipt_path = artifact / "ranking_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["artifact_sha256"][name] = _sha256(artifact / name)
        cls._write_json(receipt_path, receipt)

    def _copy(self, name: str) -> Path:
        import shutil

        destination = self.root / name
        shutil.copytree(self.artifact, destination)
        return destination

    def test_validation_rejects_each_corrupted_core_file(self) -> None:
        core_files = (
            "config.json",
            "selection.json",
            "prompts.jsonl",
            "raw_outputs.jsonl",
            "scores.tsv",
            "rankings.json",
            "runtime.json",
        )
        for name in core_files:
            with self.subTest(name=name):
                changed = self._copy(f"corrupt-{name.replace('.', '-')}")
                with (changed / name).open("ab") as handle:
                    handle.write(b"corruption")
                with self.assertRaisesRegex(
                    ValueError,
                    f"artifact hash mismatch: {name}",
                ):
                    self.pilot.validate_grouprank_artifact(changed)

    def test_validation_rejects_an_extra_directory_entry(self) -> None:
        changed = self._copy("extra-entry")
        (changed / "unexpected.txt").write_text("not in inventory\n", encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError,
            "GroupRank directory inventory changed",
        ):
            self.pilot.validate_grouprank_artifact(changed)

    def test_validation_rejects_a_falsified_receipt_inventory(self) -> None:
        changed = self._copy("receipt-inventory")
        receipt_path = changed / "ranking_receipt.json"
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["inventory"] = [*receipt["inventory"], "unexpected.txt"]
        self._write_json(receipt_path, receipt)

        with self.assertRaisesRegex(
            ValueError,
            "GroupRank receipt inventory changed",
        ):
            self.pilot.validate_grouprank_artifact(changed)

    def test_validation_rejects_rehashed_semantic_corruption(self) -> None:
        cases = (
            ("selection-hash", "selection SHA256 changed"),
            ("model-revision", "model revision changed"),
            ("prompt-hash", "prompt SHA256 changed"),
            ("runtime-field", "runtime vLLM version changed"),
            ("ranking-entry", "ranking disagrees with scores"),
        )
        for kind, expected in cases:
            with self.subTest(kind=kind):
                changed = self._copy(kind)
                if kind == "selection-hash":
                    path = changed / "config.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["selection_sha256"] = "0" * 64
                    self._write_json(path, payload)
                    self._rehash_core(changed, path.name)
                elif kind == "model-revision":
                    path = changed / "config.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["model_revision"] = "changed-revision"
                    self._write_json(path, payload)
                    self._rehash_core(changed, path.name)
                elif kind == "prompt-hash":
                    path = changed / "ranking_receipt.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["prompt_sha256"] = "0" * 64
                    self._write_json(path, payload)
                elif kind == "runtime-field":
                    path = changed / "runtime.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    payload["vllm_version"] = "changed-version"
                    self._write_json(path, payload)
                    self._rehash_core(changed, path.name)
                else:
                    path = changed / "rankings.json"
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    query_id = sorted(payload)[0]
                    payload[query_id][0], payload[query_id][1] = (
                        payload[query_id][1],
                        payload[query_id][0],
                    )
                    self._write_json(path, payload)
                    self._rehash_core(changed, path.name)

                with self.assertRaisesRegex(ValueError, expected):
                    self.pilot.validate_grouprank_artifact(changed)

    def test_validation_rejects_rehashed_sub_tolerance_score_corruption(self) -> None:
        changed = self._copy("sub-tolerance-score")
        scores_path = changed / "scores.tsv"
        lines = scores_path.read_text(encoding="utf-8").splitlines()
        fields = lines[1].split("\t")
        fields[2] = repr(float(fields[2]) + 1e-13)
        lines[1] = "\t".join(fields)
        scores_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self._rehash_core(changed, scores_path.name)

        with self.assertRaisesRegex(
            ValueError,
            "GroupRank score disagrees with raw output",
        ):
            self.pilot.validate_grouprank_artifact(changed)

    def test_valid_artifact_reloads_deterministically(self) -> None:
        first = self.pilot.load_frozen_grouprank(self.artifact)
        first_receipt = self.pilot.validate_grouprank_artifact(self.artifact)
        second = self.pilot.load_frozen_grouprank(self.artifact)
        second_receipt = self.pilot.validate_grouprank_artifact(self.artifact)

        self.assertEqual(first.rankings, second.rankings)
        self.assertEqual(first.candidate_scores, second.candidate_scores)
        self.assertEqual(first_receipt, second_receipt)
        self.assertEqual(
            first_receipt["receipt_sha256"],
            _sha256(self.artifact / "ranking_receipt.json"),
        )


if __name__ == "__main__":
    unittest.main()
