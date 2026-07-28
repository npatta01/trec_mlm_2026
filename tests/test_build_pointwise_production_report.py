from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.build_pointwise_production_report import (
    build_report,
    render_sanitized_markdown,
    write_report,
)


def _private(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    _private(
        run / "run-manifest.json",
        {"version": 1, "record_count": 1000, "model_id": "deepseek/deepseek-v4-flash"},
    )
    _private(
        run / "spend-ledger.json",
        {
            "version": 1,
            "cap_usd": 60.0,
            "spent_usd": 1.25,
            "unreconciled_reservation_usd": 0.75,
        },
    )
    for ordinal, decision in enumerate(
        [
            {"label": "good", "confidence": "high"},
            {"label": "partial", "confidence": "low"},
            None,
        ]
    ):
        _private(
            run / "results" / f"{ordinal:08d}.json",
            {
                "version": 1,
                "record_id": f"record-{ordinal}",
                "source_ordinal": ordinal,
                "status": "valid" if decision else "invalid",
                "attempt": ordinal + 1,
                "decision": (
                    None
                    if decision is None
                    else {
                        **decision,
                        "correctness": 4,
                        "completeness": 2,
                        "relevance": 2,
                        "error_flags": ["reference_conflict"] if ordinal == 1 else [],
                        "reference_status": "sound",
                        "rationale": "private candidate answer API_KEY=sk-do-not-emit",
                    }
                ),
                "reason": "private malformed provider response",
            },
        )
        _private(
            run / "attempts" / f"{ordinal + 1:09d}.json",
            {
                "version": 1,
                "attempt": ordinal + 1,
                "source_ordinal": ordinal,
                "repair": ordinal == 1,
                "state": "reconciled",
            },
        )
        _private(
            run / "raw" / f"{ordinal:08d}-{ordinal + 1:09d}.json",
            {
                "response": {
                    "content": "raw question and candidate answer sk-do-not-emit",
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                    "cost_usd": 0.25,
                    "latency_seconds": 0.5,
                }
            },
        )
    _private(
        run / "blocks" / "000000-receipt.json",
        {
            "version": 1,
            "block_index": 0,
            "expected_count": 3,
            "record_ids": ["record-0", "record-1", "record-2"],
        },
    )
    _private(
        run / "anomalies" / "000000.json",
        {
            "version": 1,
            "block_index": 0,
            "diagnostics": ["repair or retry rate exceeds 25 percent"],
            "metrics": {},
        },
    )
    _private(
        run / "opus-findings.json",
        {
            "version": 1,
            "finding_counts": {"accepted": 1, "rejected": 2},
            "private_note": "raw question sk-do-not-emit",
        },
    )
    return run


def test_report_projects_cost_and_preserves_claim_boundaries(tmp_path: Path) -> None:
    summary = build_report(_run(tmp_path))

    assert summary["coverage"] == {
        "completed_records": 3,
        "planned_records": 1000,
        "block_receipts": 1,
    }
    assert summary["cost"]["spent_usd"] == 1.25
    assert summary["cost"]["unreconciled_reservation_usd"] == 0.75
    assert summary["cost"]["committed_usd"] == 2.0
    assert summary["cost"]["projected_total_usd"] == 2000.0 / 3
    assert summary["claims"]["population_quality"] is False
    assert summary["opus_findings"] == {"accepted": 1, "rejected": 2}


def test_markdown_is_sanitized_but_contains_operational_sections(
    tmp_path: Path,
) -> None:
    markdown = render_sanitized_markdown(build_report(_run(tmp_path)))

    for heading in (
        "Coverage",
        "Parse and invariant failures",
        "Retries and repairs",
        "Cost and latency",
        "Distribution checks",
        "Opus findings",
        "Claim boundary",
    ):
        assert heading in markdown
    assert "raw question" not in markdown
    assert "candidate answer" not in markdown
    assert "sk-do-not-emit" not in markdown
    assert "private malformed" not in markdown
    assert "unreconciled reservation" in markdown
    assert "does not establish population quality" in markdown


def test_public_writer_rejects_arbitrary_markdown_and_secrets(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="summary"):
        write_report(tmp_path / "unsafe.md", "# raw question sk-do-not-emit-secret")
