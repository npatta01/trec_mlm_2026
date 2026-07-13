from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from trec_mlm.cli import build_parser, main
from trec_mlm.grouprank import (
    GroupRankRequest,
    build_group_requests,
    parse_grouprank_scores,
)
from trec_mlm.grouprank_inference import (
    ATTENTION_BACKEND,
    GPU_MEMORY_UTILIZATION,
    MAX_MODEL_LEN,
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_LICENSE,
    MODEL_REVISION,
    ROCM_CONTAINER_IMAGE,
)


ROOT = Path(__file__).resolve().parents[1]
SELECTION_PATH = ROOT / "artifacts/llm_label_ranker/pilot/stage2/selection.json"
QUERY_PATH = ROOT / "data/dev/llm_dev_data.tsv"
CANDIDATE_PATH = ROOT / "reports/trec_2026_discovery_profile_audit.json"
CODE_REVISION = "a" * 40


def _smoke_module():
    try:
        return importlib.import_module("trec_mlm.grouprank_smoke")
    except ImportError as error:  # Intentional RED assertion, not collection failure.
        raise AssertionError("GroupRank smoke module is missing") from error


def _selection() -> dict[str, object]:
    return json.loads(SELECTION_PATH.read_text(encoding="utf-8"))


def _engine_metadata() -> dict[str, object]:
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
        "container_image": ROCM_CONTAINER_IMAGE,
    }


def _runtime_metadata() -> dict[str, object]:
    return {
        **_engine_metadata(),
        "code_revision": CODE_REVISION,
        "worktree_status": "clean",
        "query_path": str(QUERY_PATH),
        "candidate_path": str(CANDIDATE_PATH),
        "elapsed_seconds": 1.25,
        "inference_stats": {
            "request_count": 1,
            "prompt_token_count": 300,
            "output_token_count": 80,
        },
    }


def _answer(request: GroupRankRequest) -> str:
    scores = {
        handle: sum(candidate.encode("utf-8")) % 11
        for handle, candidate in request.handle_to_candidate.items()
    }
    return (
        "<reason>Compared all experts.</reason>\n<answer>"
        + json.dumps(scores)
        + "</answer>"
    )


def _raw_record(
    request: GroupRankRequest,
    *,
    raw_text: str | None = None,
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class GroupRankSmokeReceiptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.selection = _selection()
        self.request = build_group_requests(self.selection)[0]
        self.path = self.root / "smoke.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_valid_smoke(self) -> dict[str, object]:
        return _smoke_module().write_grouprank_smoke(
            self.selection,
            raw_output=_raw_record(self.request),
            runtime_metadata=_runtime_metadata(),
            output_path=self.path,
        )

    def test_smoke_uses_first_frozen_group_and_records_exact_evidence(self) -> None:
        smoke = _smoke_module()

        receipt = self._write_valid_smoke()

        persisted = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertTrue(receipt["passed"])
        self.assertEqual(_sha256(self.path), receipt["receipt_sha256"])
        self.assertEqual(
            persisted, {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        )
        self.assertEqual(smoke.request_to_record(self.request), persisted["prompt"])
        self.assertEqual(_raw_record(self.request), persisted["raw_output"])
        self.assertEqual(_runtime_metadata(), persisted["runtime"])
        self.assertEqual(
            {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "resolved_revision": MODEL_REVISION,
                "dtype": MODEL_DTYPE,
                "license": MODEL_LICENSE,
                "parameter_count": 7_615_616_512,
                "max_model_len": MAX_MODEL_LEN,
                "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
                "attention_backend": ATTENTION_BACKEND,
            },
            persisted["model"],
        )
        self.assertEqual(
            parse_grouprank_scores(
                persisted["raw_output"]["raw_text"],
                expected_handles=tuple(self.request.handle_to_candidate),
            ),
            persisted["validation"]["scores"],
        )

    def test_smoke_fsyncs_raw_output_before_parsing(self) -> None:
        smoke = _smoke_module()
        fsynced_raw = False
        parsed = False
        real_fsync = os.fsync

        def tracking_fsync(file_descriptor: int) -> None:
            nonlocal fsynced_raw
            target = Path(os.readlink(f"/proc/self/fd/{file_descriptor}"))
            if target == self.path:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                fsynced_raw = payload["raw_output"] == _raw_record(self.request)
            real_fsync(file_descriptor)

        def observing_parse(raw: str, *, expected_handles):
            nonlocal parsed
            self.assertTrue(fsynced_raw, "raw response must be fsynced before parse")
            parsed = True
            return parse_grouprank_scores(raw, expected_handles=expected_handles)

        with (
            patch.object(smoke.os, "fsync", side_effect=tracking_fsync),
            patch.object(smoke, "parse_grouprank_scores", side_effect=observing_parse),
        ):
            self._write_valid_smoke()

        self.assertTrue(parsed)

    def test_smoke_requires_one_valid_ten_key_result_and_persists_failure(self) -> None:
        smoke = _smoke_module()
        invalid = (
            '<reason>Compared all experts.</reason>\n<answer>{"C00": 5}</answer>'
        )

        with self.assertRaisesRegex(ValueError, "missing or extra handles"):
            smoke.write_grouprank_smoke(
                self.selection,
                raw_output=_raw_record(self.request, raw_text=invalid),
                runtime_metadata=_runtime_metadata(),
                output_path=self.path,
            )

        receipt = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertFalse(receipt["passed"])
        self.assertEqual("failed", receipt["validation"]["state"])
        self.assertIn(
            "missing or extra handles", receipt["validation"]["error_message"]
        )
        self.assertEqual(invalid, receipt["raw_output"]["raw_text"])
        with self.assertRaisesRegex(ValueError, "did not pass"):
            smoke.validate_grouprank_smoke(
                self.path,
                selection=self.selection,
                code_revision=CODE_REVISION,
            )

    def test_smoke_validation_rejects_failed_wrong_model_revision_and_corruption(
        self,
    ) -> None:
        smoke = _smoke_module()
        self._write_valid_smoke()
        original = json.loads(self.path.read_text(encoding="utf-8"))
        cases = (
            ("failed", "did not pass"),
            ("model", "model ID changed"),
            ("revision", "model revision changed"),
            ("raw", "raw output SHA256 changed"),
        )
        for kind, expected in cases:
            with self.subTest(kind=kind):
                changed = deepcopy(original)
                if kind == "failed":
                    changed["passed"] = False
                    changed["validation"] = {
                        "state": "failed",
                        "error_type": "ValueError",
                        "error_message": "fixture failure",
                    }
                elif kind == "model":
                    changed["model"]["model_id"] = "wrong/model"
                elif kind == "revision":
                    changed["model"]["revision"] = "wrong-revision"
                else:
                    changed["raw_output"]["raw_text"] += "corruption"
                self.path.write_text(json.dumps(changed), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, expected):
                    smoke.validate_grouprank_smoke(
                        self.path,
                        selection=self.selection,
                        code_revision=CODE_REVISION,
                    )

        self.path.write_text("{not-json", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "valid JSON"):
            smoke.validate_grouprank_smoke(
                self.path,
                selection=self.selection,
                code_revision=CODE_REVISION,
            )

    def test_smoke_validation_rejects_forged_runtime_totals(self) -> None:
        smoke = _smoke_module()
        self._write_valid_smoke()
        forged = json.loads(self.path.read_text(encoding="utf-8"))
        forged["runtime"]["inference_stats"]["output_token_count"] += 1
        self.path.write_text(json.dumps(forged), encoding="utf-8")

        with self.assertRaisesRegex(
            ValueError,
            "inference stats disagree with raw output",
        ):
            smoke.validate_grouprank_smoke(
                self.path,
                selection=self.selection,
                code_revision=CODE_REVISION,
            )


class GroupRankCliTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.selection = _selection()
        self.request = build_group_requests(self.selection)[0]
        self.smoke_path = self.root / "smoke.json"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def _write_valid_smoke(self) -> None:
        _smoke_module().write_grouprank_smoke(
            self.selection,
            raw_output=_raw_record(self.request),
            runtime_metadata=_runtime_metadata(),
            output_path=self.smoke_path,
        )

    def _argv(self, phase: str, output: Path) -> list[str]:
        argv = [
            "grouprank-pilot",
            "--phase",
            phase,
            "--selection",
            str(SELECTION_PATH),
            "--code-revision",
            CODE_REVISION,
            "--output-dir",
            str(output),
        ]
        if phase == "rank":
            argv.extend(("--smoke-receipt", str(self.smoke_path)))
        return argv

    def test_cli_surface_requires_frozen_phase_arguments(self) -> None:
        parser = build_parser()
        argv = self._argv("smoke", self.smoke_path)

        args = parser.parse_args(argv)

        self.assertEqual("grouprank-pilot", args.command)
        self.assertEqual("smoke", args.phase)
        self.assertIsNone(args.smoke_receipt)
        for option in (
            "--phase",
            "--selection",
            "--code-revision",
            "--output-dir",
        ):
            with self.subTest(option=option):
                index = argv.index(option)
                with self.assertRaises(SystemExit):
                    parser.parse_args([*argv[:index], *argv[index + 2 :]])
        with self.assertRaises(SystemExit):
            parser.parse_args([*argv, "--qrels", "must-not-open"])

    def test_smoke_helper_finalizes_runtime_after_generation(self) -> None:
        from trec_mlm import cli

        llm = object()
        metadata = {
            **_engine_metadata(),
            "code_revision": CODE_REVISION,
            "worktree_status": "clean",
            "query_path": str(QUERY_PATH),
            "candidate_path": str(CANDIDATE_PATH),
        }

        def run_requests(observed_llm, requests, *, stats):
            self.assertIs(llm, observed_llm)
            self.assertEqual((self.request,), tuple(requests))
            stats.update(
                request_count=999,
                prompt_token_count=999,
                output_token_count=999,
            )
            return [_raw_record(self.request)]

        with (
            patch(
                "trec_mlm.grouprank_inference.run_grouprank_requests",
                side_effect=run_requests,
            ),
            patch.object(cli.time, "perf_counter", side_effect=(20.0, 22.5)),
        ):
            receipt = cli._run_grouprank_smoke_generation(
                llm=llm,
                selection=self.selection,
                runtime_metadata=metadata,
                output_path=self.smoke_path,
            )

        self.assertEqual(2.5, receipt["runtime"]["elapsed_seconds"])
        self.assertEqual(
            ROCM_CONTAINER_IMAGE,
            receipt["runtime"]["container_image"],
        )
        self.assertEqual(
            {
                "request_count": 1,
                "prompt_token_count": 300,
                "output_token_count": 80,
            },
            receipt["runtime"]["inference_stats"],
        )
        self.assertTrue(receipt["passed"])

    def test_terminal_cli_rejects_arbitrary_future_smoke_and_rank_before_engine(
        self,
    ) -> None:
        for phase in ("smoke", "rank"):
            with self.subTest(phase=phase):
                argv = self._argv(phase, self.root / f"arbitrary-{phase}")
                if phase == "rank":
                    argv[argv.index("--smoke-receipt") + 1] = str(
                        self.root / "arbitrary-third-smoke.json"
                    )
                with (
                    patch(
                        "trec_mlm.llm_ranker_pilot.preflight_pilot_inputs",
                        side_effect=AssertionError("selection must not be opened"),
                    ) as preflight,
                    patch(
                        "trec_mlm.grouprank_inference.load_grouprank_engine",
                        side_effect=AssertionError("engine must not load"),
                    ) as load,
                    patch(
                        "trec_mlm.cli._verify_git_launch_provenance",
                        side_effect=AssertionError("Git must not be inspected"),
                    ) as provenance,
                ):
                    with self.assertRaisesRegex(
                        SystemExit,
                        "terminal.*separately named experiment",
                    ):
                        main(argv)
                preflight.assert_not_called()
                load.assert_not_called()
                provenance.assert_not_called()


if __name__ == "__main__":
    unittest.main()
