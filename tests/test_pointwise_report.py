from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_pointwise_judge_smoke_report import main
from trec_mlm.pointwise_judge import CandidateCase
from trec_mlm.pointwise_report import (
    build_smoke_summary,
    render_private_report,
    sanitize_summary,
)
from trec_mlm.pointwise_runner import FROZEN_DECISION_SCHEMA_SHA256
from trec_mlm.pointwise_sampling import DIAGNOSTIC_REASONS


CASES = [
    CandidateCase(
        question_id=f"q-private-{index}",
        response_id=f"r-private-{index}",
        question=f"Private question {index}?",
        reference_answer=f"Private reference answer {index}.",
        candidate_answer=f"Private candidate answer {index}.",
        strata=(reason,),
        selection_reason=reason,
    )
    for index, reason in enumerate(DIAGNOSTIC_REASONS, start=1)
]


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()


def _decision(
    label: str, *, confidence: str = "high", reference_conflict: bool = False
) -> dict[str, object]:
    return {
        "label": label,
        "correctness": 4 if label == "good" else 2,
        "completeness": 2,
        "relevance": 2,
        "error_flags": ["reference_conflict"] if reference_conflict else [],
        "reference_status": "suspect" if reference_conflict else "sound",
        "confidence": confidence,
        "rationale": "Private reviewer rationale must never be public.",
    }


def _record(
    case: CandidateCase,
    alias: str,
    model_id: str,
    decision: dict[str, object] | None,
    *,
    repaired: bool = False,
) -> dict[str, object]:
    attempts: list[dict[str, object]] = []
    if repaired:
        attempts.append(
            {
                "raw": "private malformed body",
                "response_body": {
                    "model": model_id,
                    "choices": [{"message": {"content": "private malformed body"}}],
                    "usage": {"total_tokens": 2},
                },
                "parsed": None,
                "error": "decision is not valid JSON",
            }
        )
    if decision is None:
        for raw in ("private malformed body", "private invalid body"):
            attempts.append(
                {
                    "raw": raw,
                    "response_body": {
                        "model": model_id,
                        "choices": [{"message": {"content": raw}}],
                        "usage": {"total_tokens": 3},
                    },
                    "parsed": None,
                    "error": "decision is not valid JSON",
                }
            )
    else:
        raw = json.dumps(decision)
        attempts.append(
            {
                "raw": raw,
                "response_body": {
                    "model": model_id,
                    "choices": [{"message": {"content": raw}}],
                    "usage": {"total_tokens": 7},
                },
                "parsed": decision,
            }
        )
    return {
        "input_sha256": _hash(asdict(case)),
        "prompt_sha256": "1" * 64,
        "schema_sha256": FROZEN_DECISION_SCHEMA_SHA256,
        "model_alias": alias,
        "model_id": model_id,
        "runtime_receipt_sha256": hashlib.sha256(alias.encode()).hexdigest(),
        "attempts": attempts,
        "decision": decision,
        "failure": (
            {
                "type": "invalid_decision_after_repair",
                "message": "decision is not valid JSON",
            }
            if decision is None
            else None
        ),
        "elapsed_seconds": 2.0,
        "usage": {
            "total_tokens": sum(
                a["response_body"]["usage"]["total_tokens"] for a in attempts
            )
        },
    }


def _run(
    alias: str, model_id: str, decisions: list[dict[str, object] | None]
) -> dict[str, object]:
    return {
        "version": 1,
        "model_alias": alias,
        "model_id": model_id,
        "prompt_sha256": "1" * 64,
        "schema_sha256": FROZEN_DECISION_SCHEMA_SHA256,
        "runtime_receipt_sha256": hashlib.sha256(alias.encode()).hexdigest(),
        "records": {
            case.response_id: _record(
                case, alias, model_id, decision, repaired=index == 0
            )
            for index, (case, decision) in enumerate(zip(CASES, decisions, strict=True))
        },
    }


def _review_row(
    case: CandidateCase, reviewer: str, phase: str, label: str
) -> dict[str, object]:
    return {
        **_decision(label),
        "reviewer": reviewer,
        "response_id": case.response_id,
        "review_phase": phase,
        "decisive_disagreements": [],
        "prompt_findings": [],
    }


def _review(reviewer: str, labels: list[str]) -> dict[str, object]:
    return {
        "version": 1,
        "reviewer": reviewer,
        "pass1": [
            _review_row(case, reviewer, "pass1", label)
            for case, label in zip(CASES, labels, strict=True)
        ],
        "pass2": [
            _review_row(case, reviewer, "pass2", label)
            for case, label in zip(CASES, labels, strict=True)
        ],
    }


def _inputs() -> dict[str, object]:
    return {
        "cases": CASES,
        "runs": {
            "local-secret-a": _run(
                "local-secret-a",
                "org/private-secret-a",
                [_decision("good") for _ in CASES],
            ),
            "local-secret-b": _run(
                "local-secret-b",
                "org/private-secret-b",
                [
                    _decision("incorrect"),
                    *[_decision("good") for _ in CASES[1:-1]],
                    None,
                ],
            ),
        },
        "advisor_review": _review("advisor", ["good" for _ in CASES]),
        "claude_review": _review("claude", ["good" for _ in CASES]),
        "sealed_judge_aliases": {
            "judge_a": "local-secret-a",
            "judge_b": "local-secret-b",
        },
        "runtime_receipts": {
            "local-secret-a": {
                "startup_seconds": 3.5,
                "hardware": {"visible_memory_bytes": 100},
            },
            "local-secret-b": {
                "startup_seconds": 4.5,
                "hardware": {"visible_memory_bytes": 200},
            },
        },
        "manifest_hash": "a" * 64,
    }


def test_summary_reports_schema_failures_false_confidence_and_consensus_only() -> None:
    summary = build_smoke_summary(**_inputs())

    judge_a = summary["judges"]["judge_a"]
    judge_b = summary["judges"]["judge_b"]
    assert judge_a["schema_success"] == 1.0
    assert judge_a["repair_rate"] == 0.1
    assert judge_b["schema_success"] == 0.9
    assert judge_b["failure_rate"] == 0.1
    assert judge_b["false_confident_errors"] == 1
    assert judge_a["model_reviewer_consensus_exact_agreement"] == 1.0
    assert summary["reviewers"]["pass2_disagreement_count"] == 0
    assert "gold" not in json.dumps(summary).casefold()
    assert "accuracy" not in json.dumps(summary).casefold()


def test_sanitized_summary_is_a_whitelist_without_private_text_or_identities() -> None:
    summary = build_smoke_summary(**_inputs())
    sanitized = sanitize_summary(summary)
    text = json.dumps(sanitized, sort_keys=True)

    for secret in [
        "question",
        "candidate",
        "response_id",
        "q-private",
        "r-private",
        "local-secret",
        "org/private",
        "rationale",
        "raw",
        "attempt",
        "path",
    ]:
        assert secret not in text.casefold()
    assert sanitized["case_count"] == 10
    assert set(sanitized["judges"]) == {"judge_a", "judge_b"}


def test_unknown_diagnostic_reason_cannot_become_a_public_key() -> None:
    inputs = _inputs()
    secret = "candidate answer: PRIVATE SECRET"
    inputs["cases"] = [replace(CASES[0], selection_reason=secret), *CASES[1:]]

    with pytest.raises(ValueError, match="diagnostic reasons"):
        build_smoke_summary(**inputs)

    summary = build_smoke_summary(**_inputs())
    summary["diagnostic_proxies"][secret] = summary["diagnostic_proxies"][
        DIAGNOSTIC_REASONS[0]
    ]
    public = json.dumps(sanitize_summary(summary))
    assert "PRIVATE SECRET" not in public
    assert set(sanitize_summary(summary)["diagnostic_proxies"]) == set(
        DIAGNOSTIC_REASONS
    )


@pytest.mark.parametrize(
    "cases",
    [
        CASES[:-1],
        [*CASES[:-1], replace(CASES[-1], selection_reason=DIAGNOSTIC_REASONS[0])],
    ],
    ids=["missing", "duplicate"],
)
def test_summary_requires_each_frozen_diagnostic_reason_once(
    cases: list[CandidateCase],
) -> None:
    inputs = _inputs()
    inputs["cases"] = cases

    with pytest.raises(ValueError, match="diagnostic reasons exactly once"):
        build_smoke_summary(**inputs)


def test_private_report_contains_case_data_and_both_reviewer_passes() -> None:
    report = render_private_report(build_smoke_summary(**_inputs()))

    assert "Private question 1?" in report
    assert "Private candidate answer 1." in report
    assert "Advisor pass 1" in report
    assert "Claude pass 2" in report
    assert "judge_a" in report


def test_report_cli_writes_private_and_sanitized_outputs_from_fabricated_artifacts(
    tmp_path: Path,
) -> None:
    inputs = _inputs()
    artifact_dir = tmp_path / "private"
    artifact_dir.mkdir(mode=0o700)
    (artifact_dir / "runs").mkdir(mode=0o700)
    (artifact_dir / "reviews").mkdir(mode=0o700)
    manifest = {"cases": [asdict(case) for case in CASES], "manifest_hash": "a" * 64}
    files = {
        artifact_dir / "smoke_manifest.json": manifest,
        artifact_dir / "runs" / "local-secret-a.json": inputs["runs"]["local-secret-a"],
        artifact_dir / "runs" / "local-secret-b.json": inputs["runs"]["local-secret-b"],
        artifact_dir / "reviews" / "advisor.json": inputs["advisor_review"],
        artifact_dir / "reviews" / "claude.json": inputs["claude_review"],
        artifact_dir / "reviews" / "sealed-judge-aliases.json": {
            "version": 1,
            "seed": 2026,
            "judge_aliases": inputs["sealed_judge_aliases"],
        },
        artifact_dir / "runtime-local-secret-a.json": inputs["runtime_receipts"][
            "local-secret-a"
        ],
        artifact_dir / "runtime-local-secret-b.json": inputs["runtime_receipts"][
            "local-secret-b"
        ],
    }
    for path, payload in files.items():
        path.write_text(json.dumps(payload), encoding="utf-8")
        path.chmod(0o600)
    output = tmp_path / "public" / "summary.json"

    assert (
        main(["--artifact-dir", str(artifact_dir), "--sanitized-output", str(output)])
        == 0
    )
    assert (artifact_dir / "smoke-review.md").exists()
    assert (artifact_dir / "smoke-summary.json").exists()
    assert output.exists()
    assert (
        "Private candidate answer 1." in (artifact_dir / "smoke-review.md").read_text()
    )
    public = output.read_text()
    assert "Private candidate answer 1." not in public
    assert "local-secret-a" not in public
