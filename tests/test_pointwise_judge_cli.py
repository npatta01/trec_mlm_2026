import json
from pathlib import Path
import stat

import pytest

from scripts.build_pointwise_judge_smoke import main


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    grounded = tmp_path / "grounded.json"
    records = {}
    discovery_rows = {}
    for index in range(10):
        question_id = f"q{index}"
        records[question_id] = {
            "question_id": question_id,
            "question": f"What happened in event {index}?",
            "answer": f"Reference answer {index}.",
            "finish_reason": "stop",
        }
        discovery_rows[question_id] = {
            "query": f"What happened in event {index}?",
            "responses": {f"PRIVATE-CANDIDATE-{index}": f"Candidate answer {index}."},
        }
    grounded.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps(discovery_rows, indent=2), encoding="utf-8")
    return discovery, grounded


def test_cli_writes_blind_manifest_and_private_seal_atomically(tmp_path: Path) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    output = tmp_path / "smoke"
    args = [
        "--discovery", str(discovery), "--grounded", str(grounded),
        "--output-dir", str(output), "--seed", "2026",
    ]

    assert main(args) == 0
    manifest_text = (output / "smoke_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    sealed = json.loads((output / "sealed_candidates.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 2026
    assert len(manifest["cases"]) == 10
    assert set(manifest["selection_reasons"]) == {case["selection_reason"] for case in manifest["cases"]}
    assert set(manifest["sources"]) == {"discovery", "grounded"}
    assert all(source["sha256"] for source in manifest["sources"].values())
    assert all(candidate not in manifest_text for candidate in sealed.values())
    assert stat.S_IMODE((output / "sealed_candidates.json").stat().st_mode) == 0o600
    assert not list(output.glob("*.tmp"))

    assert main(args) == 0


def test_cli_rejects_existing_output_with_different_manifest(tmp_path: Path) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    output = tmp_path / "smoke"
    args = ["--discovery", str(discovery), "--grounded", str(grounded), "--output-dir", str(output)]
    assert main(args) == 0
    discovery.write_text(discovery.read_text().replace("Candidate answer 0.", "Changed answer."), encoding="utf-8")

    with pytest.raises(ValueError, match="different manifest hash"):
        main(args)
