from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from trec_mlm.artifact_manifest import (
    build_artifact_manifest,
    validate_smoke_receipt,
)


class ArtifactManifestTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
