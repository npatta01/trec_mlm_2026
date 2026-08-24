import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_gemini_grounding_retry_sample.py"


def test_cli_builds_a_deterministic_subset_of_non_search_questions(
    tmp_path: Path,
) -> None:
    sample = tmp_path / "sample.json"
    audit = tmp_path / "audit.json"
    output = tmp_path / "retry.json"
    sample.write_text(
        json.dumps(
            {
                "schema_version": "reference-validation-sample-v1",
                "sample_size": 4,
                "questions": [
                    {"question_id": "q1", "question": "one"},
                    {"question_id": "q2", "question": "two"},
                    {"question_id": "q3", "question": "three"},
                    {"question_id": "q4", "question": "four"},
                ],
            }
        ),
        encoding="utf-8",
    )
    audit.write_text(
        json.dumps({"non_search_ids": ["q1", "q2", "q4"]}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--sample",
            str(sample),
            "--audit",
            str(audit),
            "--output",
            str(output),
            "--limit",
            "2",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    retry = json.loads(output.read_text(encoding="utf-8"))
    assert retry == {
        "schema_version": "reference-validation-sample-v1",
        "sample_size": 2,
        "questions": [
            {"question_id": "q4", "question": "four", "search_expected": True},
            {"question_id": "q1", "question": "one", "search_expected": True},
        ],
    }
