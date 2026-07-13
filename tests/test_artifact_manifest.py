from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from trec_mlm.artifact_manifest import (
    build_artifact_manifest,
    validate_pilot_smoke_receipt,
    validate_smoke_receipt,
)
from trec_mlm.llm_ranker_inference import MODEL_DTYPE, MODEL_ID, MODEL_REVISION


class ArtifactManifestTest(unittest.TestCase):
    def _write_pilot_smoke(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "model_id": MODEL_ID,
                    "revision": MODEL_REVISION,
                    "dtype": MODEL_DTYPE,
                    "grade_token_ids": {"0": 15, "1": 16, "2": 17},
                    "maximum_logprob_delta": 0.00004,
                    "transformers_logprobs": {
                        "0": -3.0,
                        "1": -2.0,
                        "2": -1.0,
                    },
                    "vllm_logprobs": {
                        "0": -2.99996,
                        "1": -2.0,
                        "2": -1.0,
                    },
                    "listwise_valid_count": 20,
                    "listwise_output": json.dumps(
                        [f"C{index:03d}" for index in range(20)]
                    ),
                    "metadata": {
                        "model_id": MODEL_ID,
                        "revision": MODEL_REVISION,
                        "dtype": MODEL_DTYPE,
                        "seed": 2026,
                        "vllm_version": "0.24.0",
                        "torch_version": "2.11.0",
                        "hip_version": "7.2",
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_manifest_is_deterministic_and_records_explicit_revision(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "run"
            artifact.mkdir()
            (artifact / "b.txt").write_text("beta\n", encoding="utf-8")
            (artifact / "a.txt").write_text("alpha\n", encoding="utf-8")

            first = build_artifact_manifest(
                [artifact],
                code_revision="6275c43",
                worktree_status="clean",
            )
            second = build_artifact_manifest(
                [artifact],
                code_revision="6275c43",
                worktree_status="clean",
            )

            self.assertEqual(first, second)
            self.assertEqual("6275c43", first["code_revision"])
            self.assertEqual(["a.txt", "b.txt"], list(first["artifacts"][0]["files"]))

    def test_smoke_receipt_must_pass_and_match_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "model_id": "org/model",
                        "revision": "abc",
                        "dtype": "float32",
                    }
                ),
                encoding="utf-8",
            )

            receipt = validate_smoke_receipt(
                path,
                model_id="org/model",
                revision="abc",
                dtype="float32",
            )

            self.assertTrue(receipt["passed"])
            self.assertRegex(receipt["receipt_sha256"], r"^[0-9a-f]{64}$")
            with self.assertRaisesRegex(ValueError, "model"):
                validate_smoke_receipt(
                    path,
                    model_id="other/model",
                    revision="abc",
                    dtype="float32",
                )

    def test_pilot_smoke_receipt_requires_the_complete_real_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.json"
            self._write_pilot_smoke(path)

            receipt = validate_pilot_smoke_receipt(path)

            self.assertEqual({0: 15, 1: 16, 2: 17}, receipt["grade_token_ids"])
            self.assertRegex(receipt["receipt_sha256"], r"^[0-9a-f]{64}$")

            mutations = {
                "grade token IDs": lambda payload: payload["grade_token_ids"].update(
                    {"2": 18}
                ),
                "parity delta": lambda payload: payload.update(
                    {"maximum_logprob_delta": 0.001}
                ),
                "semantic handles": lambda payload: payload.update(
                    {"listwise_output": json.dumps(["C000"] * 20)}
                ),
                "vLLM version": lambda payload: payload["metadata"].update(
                    {"vllm_version": ""}
                ),
                "Torch version": lambda payload: payload["metadata"].update(
                    {"torch_version": ""}
                ),
                "HIP version": lambda payload: payload["metadata"].update(
                    {"hip_version": ""}
                ),
            }
            original = json.loads(path.read_text(encoding="utf-8"))
            for expected, mutate in mutations.items():
                with self.subTest(expected=expected):
                    changed = json.loads(json.dumps(original))
                    mutate(changed)
                    path.write_text(json.dumps(changed), encoding="utf-8")
                    with self.assertRaisesRegex(ValueError, expected):
                        validate_pilot_smoke_receipt(path)


if __name__ == "__main__":
    unittest.main()
