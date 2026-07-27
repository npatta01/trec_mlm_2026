from __future__ import annotations

import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import stat

import pytest

from scripts.build_pointwise_judge_v2_report import (
    audit_rationale_quotes,
    compare_runs,
    permitted_claim,
    render_sanitized_markdown,
)
from scripts.run_hosted_pointwise_smoke import (
    V2_CONFIG_SPECS,
    _runtime_receipt,
    _v2_canary_identity,
)
from trec_mlm.pointwise_judge import CandidateCase, load_prompt_contract
from trec_mlm.pointwise_runner import FROZEN_DECISION_SCHEMA_SHA256


CONFIGS = (
    "deepseek-v4-flash-direct",
    "gpt-5.4-nano-direct",
    "qwen3.6-35b-a3b-direct",
    "qwen3.6-35b-a3b-thinking",
)
PROMPT_SHA256 = "408a66ab4461d08973b645a4aa466a08b7f533beb324b902a5c3dbbe6e99a5c0"
SCHEMA_SHA256 = "bf647c4f3ae85148774c9196f5dea6dc92c95e1a77c5a1288761c3697f4e0615"
PROTOCOL_DIR = (
    Path(__file__).resolve().parents[1] / "protocols" / "pointwise_judge_bakeoff"
)
V2_CONTRACT = load_prompt_contract(PROTOCOL_DIR, version="v2")
MANIFEST_SHA256 = "a" * 64
CANARY_IDENTITY = _v2_canary_identity(
    manifest_sha256=MANIFEST_SHA256, contract=V2_CONTRACT
)


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


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _cases() -> list[CandidateCase]:
    return [
        CandidateCase(
            question_id=f"q{index:02d}",
            response_id=f"r{index:02d}",
            question=f"Private question {index}?",
            reference_answer=f"Private reference {index}.",
            candidate_answer=f"Private candidate answer {index}: evidence phrase.",
            strata=("other",),
            selection_reason="other",
        )
        for index in range(1, 11)
    ]


def _run(
    config_id: str,
    decisions: list[dict[str, object]],
    *,
    runtime_receipt_sha256: str = "3" * 64,
) -> dict[str, object]:
    model_id = {
        "deepseek-v4-flash-direct": "deepseek/deepseek-v4-flash",
        "gpt-5.4-nano-direct": "openai/gpt-5.4-nano",
        "qwen3.6-35b-a3b-direct": "qwen/qwen3.6-35b-a3b",
        "qwen3.6-35b-a3b-thinking": "qwen/qwen3.6-35b-a3b",
    }.get(config_id, "deepseek/deepseek-v4-flash")
    return {
        "version": 1,
        "model_alias": config_id,
        "model_id": model_id,
        "prompt_sha256": PROMPT_SHA256,
        "schema_sha256": SCHEMA_SHA256,
        "runtime_receipt_sha256": runtime_receipt_sha256,
        "records": {
            f"r{index:02d}": {
                "input_sha256": _hash(asdict(_cases()[index - 1])),
                "decision": decision,
                "failure": None,
                "attempts": [
                    {
                        "raw": json.dumps(decision, ensure_ascii=False),
                        "response_body": {
                            "model": model_id,
                            "choices": [
                                {
                                    "message": {
                                        "content": json.dumps(
                                            decision, ensure_ascii=False
                                        )
                                    }
                                }
                            ],
                            "usage": {"cost": 0.001},
                        },
                        "logical_model_id": model_id,
                        "backend_model_id": model_id,
                        "parsed": decision,
                    }
                ],
                "elapsed_seconds": float(index),
                "usage": {"cost": 0.001},
                "model_alias": config_id,
                "model_id": model_id,
                "prompt_sha256": PROMPT_SHA256,
                "schema_sha256": SCHEMA_SHA256,
                "runtime_receipt_sha256": runtime_receipt_sha256,
            }
            for index, decision in enumerate(decisions, start=1)
        },
    }


def _receipt(config_id: str) -> dict[str, object]:
    return _runtime_receipt(
        V2_CONFIG_SPECS[config_id],
        V2_CONTRACT,
        version="v2",
        canary_identity_sha256=CANARY_IDENTITY,
    )


def _v1_run() -> dict[str, object]:
    run = _run("legacy", [_decision() for _ in range(10)])
    prompt_sha256 = hashlib.sha256(
        load_prompt_contract(PROTOCOL_DIR, version="v1").prompt.encode("utf-8")
    ).hexdigest()
    run["prompt_sha256"] = prompt_sha256
    run["schema_sha256"] = FROZEN_DECISION_SCHEMA_SHA256
    for record in run["records"].values():
        record["prompt_sha256"] = prompt_sha256
        record["schema_sha256"] = FROZEN_DECISION_SCHEMA_SHA256
    return run


def _review(reviewer: str) -> dict[str, object]:
    def row(index: int, phase: str) -> dict[str, object]:
        return {
            "response_id": f"r{index:02d}",
            "reviewer": reviewer,
            "review_phase": phase,
            **_decision(),
            "decisive_disagreements": [],
            "prompt_findings": [],
        }

    return {
        "version": 1,
        "reviewer": reviewer,
        "pass1": [row(index, "pass1") for index in range(1, 11)],
        "pass2": [row(index, "pass2") for index in range(1, 11)],
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
    receipts = {config_id: _receipt(config_id) for config_id in CONFIGS}
    runs = {
        config_id: _run(
            config_id,
            decisions,
            runtime_receipt_sha256=_hash(receipts[config_id]),
        )
        for config_id in CONFIGS
    }
    advisor_review = _review("advisor")
    advisor_review["pass2"][0]["decisive_disagreements"] = [
        {
            "judge_aliases": ["judge_a"],
            "finding": "The anonymous judge disagreed on the material conclusion.",
        }
    ]
    return {
        "manifest_sha256": MANIFEST_SHA256,
        "manifest": {
            "manifest_hash": "a" * 64,
            "cases": [asdict(case) for case in _cases()],
        },
        "runs": runs,
        "runtime_receipts": receipts,
        "advisor_review": advisor_review,
        "claude_review": _review("claude"),
        "sealed_judge_aliases": {
            "version": 1,
            "seed": 2026,
            "judge_aliases": {
                "judge_a": "sealed-model-alpha",
                "judge_b": "sealed-model-beta",
            },
        },
        "spend_ledger": {"version": 1, "cap_usd": 0.1, "spent_usd": 0.04},
        "attempt_ledger": {
            "version": 2,
            "cap_calls": 100,
            "used_calls": 40,
            "canary_identity_sha256": CANARY_IDENTITY,
        },
        "attempt_markers": [
            {"version": 1, "attempt": index, "canary_identity_sha256": CANARY_IDENTITY}
            for index in range(1, 41)
        ],
        "spend_receipt": {
            "version": 1,
            "cap_usd": 0.1,
            "spent_usd": 0.04,
            "call_cap": 100,
            "used_calls": 40,
            "api_key_serialized": False,
            "models": [
                {
                    "config_id": config_id,
                    "record_count": 10,
                    "run_sha256": _hash(runs[config_id]),
                }
                for config_id in CONFIGS
            ],
        },
        "v1_runs": {"v1_canary": _v1_run()},
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


def test_locked_reviewer_ledgers_reject_wrong_identity_or_pass_phase() -> None:
    """This fails if untrusted or misfiled reviewer rows can become comparison truth."""
    for patch in (
        {"reviewer": "claude"},
        {"row_phase": "pass1"},
    ):
        inputs = _inputs()
        review = inputs["advisor_review"]
        if "reviewer" in patch:
            review["reviewer"] = patch["reviewer"]
        else:
            review["pass2"][0]["review_phase"] = patch["row_phase"]
        with pytest.raises(ValueError, match="reviewer"):
            compare_runs(inputs)


def test_allowed_sealed_judge_aliases_validate_actual_style_disagreement() -> None:
    """This fails if locked critique rows cannot use their sealed judge aliases."""
    result = compare_runs(_inputs())

    assert result["inputs"]["sealed_judge_alias_count"] == 2


def test_unknown_judge_alias_in_locked_reviewer_ledger_fails_closed() -> None:
    """This fails if reviewer critique can name a judge absent from sealed evidence."""
    inputs = _inputs()
    inputs["advisor_review"]["pass2"][0]["decisive_disagreements"][0][
        "judge_aliases"
    ] = ["judge_unknown"]

    with pytest.raises(ValueError, match="unknown judge alias"):
        compare_runs(inputs)


@pytest.mark.parametrize(
    "sealed",
    [
        {},
        {"version": 1, "seed": "2026", "judge_aliases": {"judge_a": "model-a"}},
        {"version": 1, "seed": 2026, "judge_aliases": {}},
        {
            "version": 1,
            "seed": 2026,
            "judge_aliases": {"": "model-a", "judge_b": "model-b"},
        },
        {
            "version": 1,
            "seed": 2026,
            "judge_aliases": {"judge_a": "model-a", "judge_b": "model-a"},
        },
        {
            "version": 1,
            "seed": 2026,
            "judge_aliases": {"judge_a": "model-a"},
            "unexpected": True,
        },
    ],
    ids=[
        "missing",
        "bad_seed",
        "empty",
        "empty_alias",
        "duplicate_model",
        "extra",
    ],
)
def test_sealed_judge_alias_evidence_rejects_malformed_or_duplicate_mapping(
    sealed: object,
) -> None:
    """This fails if malformed or duplicate sealed alias evidence is trusted."""
    inputs = _inputs()
    inputs["sealed_judge_aliases"] = sealed

    with pytest.raises(ValueError, match="sealed judge aliases"):
        compare_runs(inputs)


def test_swapped_sealed_judge_alias_mapping_cannot_validate_reviewer_rows() -> None:
    """This fails if a reversed alias map can authorize the locked critique rows."""
    inputs = _inputs()
    inputs["sealed_judge_aliases"] = {
        "version": 1,
        "seed": 2026,
        "judge_aliases": {"sealed-model-alpha": "judge_a"},
    }

    with pytest.raises(ValueError, match="unknown judge alias"):
        compare_runs(inputs)


def test_run_receipt_and_manifest_identity_breaks_are_hard_gates() -> None:
    """This fails if fabricated records or receipts can retain valid aggregate metrics."""
    inputs = _inputs()
    inputs["runs"]["deepseek-v4-flash-direct"]["records"]["r01"]["input_sha256"] = (
        "0" * 64
    )
    inputs["runtime_receipts"]["deepseek-v4-flash-direct"]["endpoint"] = "wrong"

    result = compare_runs(inputs)

    gates = result["configs"]["deepseek-v4-flash-direct"]["hard_gate_failures"]
    assert "runner_record_validation" in gates
    assert "runtime_receipt_binding" in gates


def test_reviewer_disagreement_preserves_field_denominators() -> None:
    """This fails if an unrelated reviewer disagreement removes a label from its rate."""
    inputs = _inputs()
    inputs["claude_review"]["pass2"][0]["reference_status"] = "incomplete"

    result = compare_runs(inputs)

    metrics = result["configs"]["deepseek-v4-flash-direct"]
    assert metrics["label_agreement_detail"] == {
        "numerator": 9,
        "denominator": 10,
        "excluded_count": 0,
        "excluded_response_ids": [],
    }
    assert metrics["reference_status_agreement_detail"]["denominator"] == 9
    assert metrics["reference_status_agreement_detail"]["excluded_count"] == 1
    assert metrics["reference_status_agreement_detail"]["excluded_response_ids"] == [
        "r01"
    ]


def test_unpaired_qwen_metrics_are_explicitly_unavailable() -> None:
    """This fails if an invalid paired record makes Qwen changes look like zero."""
    inputs = _inputs()
    inputs["runs"]["qwen3.6-35b-a3b-thinking"]["records"]["r01"]["decision"] = None

    result = compare_runs(inputs)

    assert result["qwen_direct_thinking"]["available"] is False
    assert result["qwen_direct_thinking"]["excluded_count"] == 1
    assert result["qwen_direct_thinking"]["excluded_response_ids"] == ["r01"]
    assert result["qwen_direct_thinking"]["changed_case_count"] is None


def test_accounting_and_required_v1_evidence_fail_closed() -> None:
    """This fails if unreconciled ledgers or omitted v1 evidence can be reported."""
    inputs = _inputs()
    inputs["v1_runs"] = {}
    with pytest.raises(ValueError, match="v1"):
        compare_runs(inputs)

    inputs = _inputs()
    inputs["spend_ledger"]["spent_usd"] = 0.03
    inputs["spend_receipt"]["spent_usd"] = 0.03
    result = compare_runs(inputs)
    assert "accounting_reconciliation" in result["global_hard_gate_failures"]


def test_sanitized_markdown_contains_all_safe_aggregate_sections() -> None:
    """This fails if a standalone Markdown handoff omits required aggregate evidence."""
    markdown = render_sanitized_markdown(compare_runs(_inputs()))

    for text in (
        "Provider",
        "Correctness agreement",
        "Repair attempts",
        "Cross-model label agreement",
        "Rationale audit",
        "Hard-gate failures",
        "v1 differences",
    ):
        assert text in markdown


def test_cli_writes_private_json_and_sanitized_markdown(tmp_path: Path) -> None:
    """This fails if the command-line boundary leaks private fixture text or modes."""
    from scripts.build_pointwise_judge_v2_report import main

    inputs = _inputs()
    artifact_dir = tmp_path / "private"
    artifact_dir.mkdir(mode=0o700)
    paths: dict[str, Path] = {}
    for name in (
        "manifest",
        "advisor_review",
        "claude_review",
        "sealed_judge_aliases",
        "spend_ledger",
        "attempt_ledger",
        "spend_receipt",
    ):
        path = artifact_dir / f"{name}.json"
        path.write_text(json.dumps(inputs[name]), encoding="utf-8")
        path.chmod(0o600)
        paths[name] = path
    runs: list[str] = []
    receipts: list[str] = []
    for config_id in CONFIGS:
        run_path = artifact_dir / f"{config_id}.run.json"
        receipt_path = artifact_dir / f"{config_id}.receipt.json"
        run_path.write_text(json.dumps(inputs["runs"][config_id]), encoding="utf-8")
        receipt_path.write_text(
            json.dumps(inputs["runtime_receipts"][config_id]), encoding="utf-8"
        )
        run_path.chmod(0o600)
        receipt_path.chmod(0o600)
        runs.extend(["--v2-run", f"{config_id}={run_path}"])
        receipts.extend(["--runtime-receipt", f"{config_id}={receipt_path}"])
    v1_path = artifact_dir / "legacy.json"
    v1_path.write_text(json.dumps(inputs["v1_runs"]["v1_canary"]), encoding="utf-8")
    v1_path.chmod(0o600)
    private_json = artifact_dir / "comparison.json"
    markdown = tmp_path / "report.md"
    marker_dir = artifact_dir / "attempt-markers"
    marker_dir.mkdir(mode=0o700)
    for marker in inputs["attempt_markers"]:
        path = marker_dir / f"{marker['attempt']:03d}.json"
        path.write_text(json.dumps(marker), encoding="utf-8")
        path.chmod(0o600)

    argv = [
        "--manifest",
        str(paths["manifest"]),
        "--advisor-review",
        str(paths["advisor_review"]),
        "--claude-review",
        str(paths["claude_review"]),
        "--sealed-judge-aliases",
        str(paths["sealed_judge_aliases"]),
        *runs,
        *receipts,
        "--spend-ledger",
        str(paths["spend_ledger"]),
        "--attempt-ledger",
        str(paths["attempt_ledger"]),
        "--attempt-marker-dir",
        str(marker_dir),
        "--spend-receipt",
        str(paths["spend_receipt"]),
        "--v1-run",
        f"v1_canary={v1_path}",
        "--output-json",
        str(private_json),
        "--output-markdown",
        str(markdown),
    ]
    assert main(argv) == 0
    assert stat.S_IMODE(private_json.stat().st_mode) == 0o600
    rendered = markdown.read_text(encoding="utf-8")
    for private_identity in (
        "Private candidate answer",
        "judge_a",
        "sealed-model-alpha",
        "sealed-model-beta",
    ):
        assert private_identity not in rendered
    sealed_index = argv.index("--sealed-judge-aliases")
    with pytest.raises(SystemExit):
        main(argv[:sealed_index] + argv[sealed_index + 2 :])
    paths["sealed_judge_aliases"].chmod(0o644)
    with pytest.raises(ValueError, match="private report input"):
        main(argv)


def test_synthetic_v2_attempt_summary_cannot_pass_runner_validation() -> None:
    """This fails if hand-written decisions and usage can masquerade as runner records."""
    inputs = _inputs()
    inputs["runs"]["deepseek-v4-flash-direct"]["records"]["r01"]["attempts"] = []
    result = compare_runs(inputs)

    assert (
        "runner_record_validation"
        in result["configs"]["deepseek-v4-flash-direct"]["hard_gate_failures"]
    )


def test_cross_model_pair_reports_its_exact_unavailable_coverage() -> None:
    """This fails if incomplete cross-model pairs are averaged over a reduced subset."""
    inputs = _inputs()
    inputs["runs"]["gpt-5.4-nano-direct"]["records"]["r01"]["decision"] = None

    result = compare_runs(inputs)

    pair = result["configs"]["deepseek-v4-flash-direct"][
        "cross_model_label_agreement_detail"
    ]["gpt-5.4-nano-direct"]
    assert pair == {
        "available": False,
        "numerator": None,
        "denominator": 9,
        "excluded_response_ids": ["r01"],
    }


def test_v1_terminal_record_must_use_real_attempts_and_frozen_parser() -> None:
    """This fails if a v1 decision without a terminal runner attempt becomes evidence."""
    inputs = _inputs()
    inputs["v1_runs"]["v1_canary"]["records"]["r01"]["attempts"] = []

    with pytest.raises(ValueError, match="v1 evidence"):
        compare_runs(inputs)


def test_v1_cli_name_is_not_rendered_into_sanitized_markdown() -> None:
    """This fails if caller-controlled v1 names can inject private text into Markdown."""
    inputs = _inputs()
    injected = "private candidate | injected"
    inputs["v1_runs"] = {injected: inputs["v1_runs"]["v1_canary"]}

    with pytest.raises(ValueError, match="fixed v1_canary"):
        compare_runs(inputs)
