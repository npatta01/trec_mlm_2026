from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
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
    }


def _runtime_metadata() -> dict[str, object]:
    return {
        **_engine_metadata(),
        "code_revision": CODE_REVISION,
        "worktree_status": "clean",
        "query_path": str(QUERY_PATH),
        "candidate_path": str(CANDIDATE_PATH),
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
    return "Compared all experts.\n<answer>" + json.dumps(scores) + "</answer>\n"


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
        invalid = '<answer>{"C00": 5}</answer>'

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

    def test_smoke_cli_preflights_and_uses_only_first_group(self) -> None:
        events: list[str] = []
        llm = object()
        tokenizer = object()

        def preflight(*args, **kwargs):
            events.append("preflight")
            return self.selection, dict(self.selection["input_sha256"])

        def verify(_revision):
            events.append("provenance")
            return "clean"

        def load_engine():
            events.append("engine")
            return llm, tokenizer, _engine_metadata()

        def run_requests(observed_llm, requests, *, stats):
            events.append("inference")
            self.assertIs(llm, observed_llm)
            self.assertEqual((self.request,), tuple(requests))
            stats.update(
                request_count=1,
                prompt_token_count=300,
                output_token_count=80,
            )
            return [_raw_record(self.request)]

        with (
            patch(
                "trec_mlm.llm_ranker_pilot.preflight_pilot_inputs",
                side_effect=preflight,
            ),
            patch("trec_mlm.cli._verify_git_launch_provenance", side_effect=verify),
            patch(
                "trec_mlm.grouprank_inference.load_grouprank_engine",
                side_effect=load_engine,
            ) as load,
            patch(
                "trec_mlm.grouprank_inference.run_grouprank_requests",
                side_effect=run_requests,
            ),
            patch("builtins.print"),
        ):
            status = main(self._argv("smoke", self.smoke_path))

        self.assertEqual(0, status)
        self.assertEqual(["preflight", "provenance", "engine", "inference"], events)
        load.assert_called_once_with()
        receipt = _smoke_module().validate_grouprank_smoke(
            self.smoke_path,
            selection=self.selection,
            code_revision=CODE_REVISION,
        )
        self.assertTrue(receipt["passed"])

    def test_rank_cli_validates_matching_smoke_before_one_engine_load(self) -> None:
        self._write_valid_smoke()
        events: list[str] = []
        llm = object()

        def preflight(*args, **kwargs):
            events.append("preflight")
            return self.selection, dict(self.selection["input_sha256"])

        def verify(_revision):
            events.append("provenance")
            return "clean"

        real_validate = _smoke_module().validate_grouprank_smoke

        def validate(*args, **kwargs):
            events.append("smoke")
            return real_validate(*args, **kwargs)

        def load_engine():
            events.append("engine")
            return llm, object(), _engine_metadata()

        def execute(selection, *, infer_batch, runtime_metadata, output_dir):
            events.append("execute")
            requests = build_group_requests(selection)
            self.assertEqual(48, len(requests))
            outputs = infer_batch(requests)
            self.assertEqual(48, len(outputs))
            self.assertEqual(self.root / "rank", Path(output_dir))
            return SimpleNamespace(runtime=runtime_metadata)

        def run_requests(observed_llm, requests, *, stats):
            self.assertIs(llm, observed_llm)
            stats.update(
                request_count=len(requests),
                prompt_token_count=14_400,
                output_token_count=3_840,
            )
            return [_raw_record(request) for request in requests]

        with (
            patch(
                "trec_mlm.llm_ranker_pilot.preflight_pilot_inputs",
                side_effect=preflight,
            ),
            patch("trec_mlm.cli._verify_git_launch_provenance", side_effect=verify),
            patch(
                "trec_mlm.grouprank_smoke.validate_grouprank_smoke",
                side_effect=validate,
            ) as validate_smoke,
            patch(
                "trec_mlm.grouprank_inference.load_grouprank_engine",
                side_effect=load_engine,
            ) as load,
            patch(
                "trec_mlm.grouprank_inference.run_grouprank_requests",
                side_effect=run_requests,
            ) as run,
            patch(
                "trec_mlm.grouprank_pilot.execute_grouprank_phase_a",
                side_effect=execute,
            ) as execute_phase_a,
            patch("builtins.print"),
        ):
            status = main(self._argv("rank", self.root / "rank"))

        self.assertEqual(0, status)
        self.assertEqual(
            ["preflight", "provenance", "smoke", "engine", "execute"],
            events,
        )
        validate_smoke.assert_called_once()
        load.assert_called_once_with()
        execute_phase_a.assert_called_once()
        run.assert_called_once()

    def test_rank_cli_rejects_missing_failed_wrong_or_corrupted_smoke_before_engine(
        self,
    ) -> None:
        smoke = _smoke_module()
        self._write_valid_smoke()
        valid = json.loads(self.smoke_path.read_text(encoding="utf-8"))
        cases = ("missing", "failed", "model", "revision", "corrupt")
        for kind in cases:
            with self.subTest(kind=kind):
                path = self.root / f"{kind}.json"
                if kind != "missing":
                    changed = deepcopy(valid)
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
                    path.write_text(json.dumps(changed), encoding="utf-8")
                argv = self._argv("rank", self.root / f"rank-{kind}")
                argv[argv.index("--smoke-receipt") + 1] = str(path)
                with (
                    patch(
                        "trec_mlm.cli._verify_git_launch_provenance",
                        return_value="clean",
                    ),
                    patch(
                        "trec_mlm.grouprank_inference.load_grouprank_engine",
                        side_effect=AssertionError("engine must not load"),
                    ) as load,
                ):
                    with self.assertRaises(ValueError):
                        main(argv)
                load.assert_not_called()

        with self.assertRaisesRegex(ValueError, "code revision changed"):
            smoke.validate_grouprank_smoke(
                self.smoke_path,
                selection=self.selection,
                code_revision="b" * 40,
            )

    def test_cli_provenance_rejects_dirty_or_mismatched_revision_before_engine(
        self,
    ) -> None:
        self._write_valid_smoke()
        for phase in ("smoke", "rank"):
            for message in (
                "pilot code revision does not match HEAD",
                "pilot tracked worktree is not clean",
            ):
                with self.subTest(phase=phase, message=message):
                    output = self.smoke_path if phase == "smoke" else self.root / "rank"
                    with (
                        patch(
                            "trec_mlm.cli._verify_git_launch_provenance",
                            side_effect=ValueError(message),
                        ),
                        patch(
                            "trec_mlm.grouprank_inference.load_grouprank_engine",
                            side_effect=AssertionError("engine must not load"),
                        ) as load,
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            main(self._argv(phase, output))
                    load.assert_not_called()

    def test_rank_cli_returns_nonzero_after_persisted_parse_failure(self) -> None:
        self._write_valid_smoke()
        destination = self.root / "failed-rank"

        def invalid_outputs(_llm, requests, *, stats):
            records = [_raw_record(request) for request in requests]
            records[0]["raw_text"] = '<answer>{"C00":5}</answer>'
            stats.update(
                request_count=len(requests),
                prompt_token_count=14_400,
                output_token_count=3_840,
            )
            return records

        with (
            patch("trec_mlm.cli._verify_git_launch_provenance", return_value="clean"),
            patch(
                "trec_mlm.grouprank_inference.load_grouprank_engine",
                return_value=(object(), object(), _engine_metadata()),
            ),
            patch(
                "trec_mlm.grouprank_inference.run_grouprank_requests",
                side_effect=invalid_outputs,
            ),
            patch("builtins.print"),
        ):
            status = main(self._argv("rank", destination))

        self.assertNotEqual(0, status)
        self.assertTrue((destination / "failure_receipt.json").is_file())
        self.assertFalse((destination / "ranking_receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
