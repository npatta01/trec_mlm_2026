import json
import stat
from pathlib import Path

from scripts.build_pointwise_production_inputs import main


def _write_fixture(directory: Path) -> tuple[Path, Path, Path]:
    discovery = directory / "discovery.json"
    grounded = directory / "grounded.json"
    refinement_records = directory / "records.json"
    discovery.write_text(
        json.dumps(
            {"q1": {"query": "Question?", "responses": {"candidate-a": "Answer."}}},
            indent=2,
        ),
        encoding="utf-8",
    )
    grounded.write_text(
        json.dumps(
            {
                "records": {
                    "q1": {
                        "question_id": "q1",
                        "question": "Question?",
                        "answer": "Reference.",
                        "finish_reason": "stop",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    refinement_records.write_text(
        json.dumps(
            [
                {
                    "record_id": "profile-record-v2:one",
                    "source_query_id": "q1",
                    "candidate_id": "candidate-a",
                    "question": "Question?",
                    "response": "Answer.",
                    "formatted_sha256": "0" * 64,
                    "raw_membership_sha256": "1" * 64,
                }
            ]
        ),
        encoding="utf-8",
    )
    return discovery, grounded, refinement_records


def test_cli_builds_private_inventory_without_network(tmp_path: Path) -> None:
    discovery, grounded, refinement_records = _write_fixture(tmp_path)
    output_dir = tmp_path / "sealed"

    assert (
        main(
            [
                "--discovery",
                str(discovery),
                "--grounded",
                str(grounded),
                "--refinement-records",
                str(refinement_records),
                "--output-dir",
                str(output_dir),
                "--expected-count",
                "1",
            ]
        )
        == 0
    )

    manifest = json.loads((output_dir / "production-manifest.json").read_text())
    assert manifest["record_count"] == 1
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert (
        stat.S_IMODE((output_dir / "production-records.jsonl").stat().st_mode) == 0o600
    )
