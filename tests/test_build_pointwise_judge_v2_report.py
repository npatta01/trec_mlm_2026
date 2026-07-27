from __future__ import annotations

import copy

import pytest

from scripts.build_pointwise_judge_v2_report import (
    audit_rationale_quotes,
    compare_runs,
    permitted_claim,
    render_sanitized_markdown,
)


CONFIGS = (
    "deepseek-v4-flash-direct",
    "gpt-5.4-nano-direct",
    "qwen3.6-35b-a3b-direct",
    "qwen3.6-35b-a3b-thinking",
)
PROMPT_SHA256 = "408a66ab4461d08973b645a4aa466a08b7f533beb324b902a5c3dbbe6e99a5c0"
SCHEMA_SHA256 = "bf647c4f3ae85148774c9196f5dea6dc92c95e1a77c5a1288761c3697f4e0615"


def _decision(
    label: str = "good", *, flags: list[str] | None = None, rationale: str | None = None
) -> dict[str, object]:
    flags = [] if flags is None else flags
    return {
        "label": label,
        "correctness": 4 if label == "good" else 1,
        "completeness": 2 if label == "good" else 0,
        "relevance": 2,
        "error_flags": flags,
        "reference_status": "sound",
        "confidence": "high",
        "rationale": rationale
        or "The decision is grounded in the available candidate text and its stated answer.",
    }


def _run(config_id: str, decisions: list[dict[str, object]]) -> dict[str, object]:
    model_id = (
        "qwen/qwen3.6-35b-a3b"
        if config_id.startswith("qwen")
        else "deepseek/deepseek-v4-flash"
    )
    return {
        "version": 1,
        "model_alias": config_id,
        "model_id": model_id,
        "prompt_sha256": PROMPT_SHA256,
        "schema_sha256": SCHEMA_SHA256,
        "runtime_receipt_sha256": "3" * 64,
        "records": {
            f"r{index:02d}": {
                "decision": decision,
                "failure": None,
                "attempts": [{"parsed": decision}],
                "elapsed_seconds": float(index),
                "usage": {"cost": 0.001},
                "model_alias": config_id,
                "model_id": model_id,
                "prompt_sha256": PROMPT_SHA256,
                "schema_sha256": SCHEMA_SHA256,
                "runtime_receipt_sha256": "3" * 64,
            }
            for index, decision in enumerate(decisions, start=1)
        },
    }


def _receipt(config_id: str) -> dict[str, object]:
    model_id = (
        "qwen/qwen3.6-35b-a3b"
        if config_id.startswith("qwen")
        else "deepseek/deepseek-v4-flash"
    )
    return {
        "config_id": config_id,
        "model_id": model_id,
        "provider": "AkashML" if config_id.startswith("qwen") else "DeepInfra",
        "prompt_sha256": PROMPT_SHA256,
        "schema_sha256": SCHEMA_SHA256,
        "transport_schema_sha256": "4" * 64,
        "reasoning": {"enabled": config_id.endswith("thinking")}
        if config_id.startswith("qwen")
        else {"enabled": False},
    }


def _review() -> dict[str, object]:
    return {
        "version": 1,
        "pass2": [
            {"response_id": f"r{index:02d}", **_decision()} for index in range(1, 11)
        ],
    }


def _inputs() -> dict[str, object]:
    decisions = [_decision() for _ in range(10)]
    decisions[8] = _decision("incorrect")
    decisions[8]["error_flags"] = ["entity_error"]
    decisions[8]["rationale"] = (
        'The candidate quote "evidence phrase" establishes the material entity error.'
    )
    decisions[9]["error_flags"] = ["reference_conflict"]
    decisions[9]["reference_status"] = "incomplete"
    decisions[9]["rationale"] = (
        'The candidate quote "evidence phrase" establishes the reference conflict.'
    )
    return {
        "manifest": {
            "manifest_hash": "a" * 64,
            "cases": [
                {
                    "response_id": f"r{index:02d}",
                    "candidate_answer": f"Private candidate answer {index}: evidence phrase.",
                }
                for index in range(1, 11)
            ],
        },
        "runs": {config_id: _run(config_id, decisions) for config_id in CONFIGS},
        "runtime_receipts": {config_id: _receipt(config_id) for config_id in CONFIGS},
        "advisor_review": _review(),
        "claude_review": _review(),
        "spend_ledger": {"version": 1, "cap_usd": 0.1, "spent_usd": 0.04},
        "attempt_ledger": {"version": 2, "cap_calls": 100, "used_calls": 40},
        "v1_runs": {"legacy": _run("legacy", [_decision() for _ in range(10)])},
    }


def test_metrics_report_exact_agreement_jaccard_cost_latency() -> None:
    """This fails if field agreement or recorded run costs are calculated wrongly."""
    result = compare_runs(_inputs())

    metrics = result["configs"]["deepseek-v4-flash-direct"]
    assert metrics["label_agreement"] == 0.9
    assert metrics["error_flag_mean_jaccard"] == pytest.approx(0.8)
    assert metrics["cost_usd"] == pytest.approx(0.01)
    assert metrics["latency_seconds"] == pytest.approx(55.0)
    assert metrics["completion"] == "10/10"
    assert metrics["hard_gate_failures"] == []


def test_claim_is_exploratory_for_ten_cases() -> None:
    """This fails if a canary result is phrased as a population-quality claim."""
    claim = permitted_claim(10)
    assert claim.startswith("On this 10-case canary")
    assert "generally" not in claim.casefold()
    assert "too small for a population-quality claim" in claim


def test_quote_audit_reports_absent_and_mismatched_quotes_without_dropping_them() -> (
    None
):
    """This fails if flagged decisions without a faithful quote are silently accepted."""
    candidate = "The candidate says exactly this evidence phrase."
    absent = audit_rationale_quotes(
        candidate,
        _decision(
            "incorrect", flags=["entity_error"], rationale="A material error appears."
        ),
    )
    mismatched = audit_rationale_quotes(
        candidate,
        _decision(
            "incorrect",
            flags=["entity_error"],
            rationale='The candidate says "invented phrase" and is wrong.',
        ),
    )

    assert absent == {"passed": False, "reason": "missing_quote", "quote": None}
    assert mismatched == {
        "passed": False,
        "reason": "quote_not_in_candidate",
        "quote": "invented phrase",
    }


def test_markdown_is_sanitized_while_private_comparison_keeps_audit_findings() -> None:
    """This fails if private candidates, rationales, or record identifiers leak to Markdown."""
    result = compare_runs(_inputs())
    markdown = render_sanitized_markdown(result)

    assert "Private candidate answer" not in markdown
    assert "evidence phrase" not in markdown
    assert "r09" not in markdown
    assert (
        result["configs"]["deepseek-v4-flash-direct"]["rationale_audit"]["failed"] == 0
    )
    assert "On this 10-case canary" in markdown


def test_provider_identity_gate_fails_closed_on_receipt_mismatch() -> None:
    """This fails if a provider mismatch can be reported as a successful canary."""
    inputs = _inputs()
    inputs["runtime_receipts"] = copy.deepcopy(inputs["runtime_receipts"])
    inputs["runtime_receipts"]["deepseek-v4-flash-direct"]["provider"] = (
        "wrong-provider"
    )

    result = compare_runs(inputs)

    assert (
        "provider_identity"
        in result["configs"]["deepseek-v4-flash-direct"]["hard_gate_failures"]
    )
