#!/usr/bin/env python3
"""Build a claim-safe private comparison for the four v2 judge canary runs."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import stat
import tempfile

from scripts.run_hosted_pointwise_smoke import (
    V2_CONFIG_SPECS,
    _runtime_receipt,
    _v2_canary_identity,
)
from trec_mlm.pointwise_judge import (
    CandidateCase,
    load_prompt_contract,
    parse_judge_decision,
)
from trec_mlm.pointwise_review import validate_reviewer_record
from trec_mlm.pointwise_runner import (
    FROZEN_DECISION_SCHEMA_SHA256,
    FROZEN_V2_DECISION_SCHEMA_SHA256,
    FROZEN_V2_PROMPT_SHA256,
    _record_identity,
    _validate_run_container,
    _validated_same_identity_record,
    trusted_run_backend_model,
)


CONFIGS = (
    "deepseek-v4-flash-direct",
    "gpt-5.4-nano-direct",
    "qwen3.6-35b-a3b-direct",
    "qwen3.6-35b-a3b-thinking",
)
_DECISION_FIELDS = (
    "label",
    "correctness",
    "completeness",
    "relevance",
    "error_flags",
    "reference_status",
)
_EXPECTED_CONFIGS = {
    "deepseek-v4-flash-direct": {
        "model_id": "deepseek/deepseek-v4-flash",
        "provider": "DeepInfra",
        "reasoning": {"enabled": False},
    },
    "gpt-5.4-nano-direct": {
        "model_id": "openai/gpt-5.4-nano",
        "provider": "OpenAI",
        "reasoning": None,
    },
    "qwen3.6-35b-a3b-direct": {
        "model_id": "qwen/qwen3.6-35b-a3b",
        "provider": "AkashML",
        "reasoning": {"enabled": False},
    },
    "qwen3.6-35b-a3b-thinking": {
        "model_id": "qwen/qwen3.6-35b-a3b",
        "provider": "AkashML",
        "reasoning": {"enabled": True},
    },
}
_QUOTED_SPAN = re.compile(r'"([^"\n]{1,240})"|“([^”\n]{1,240})”|`([^`\n]{1,240})`')
_PROTOCOL_DIR = (
    Path(__file__).resolve().parents[1] / "protocols" / "pointwise_judge_bakeoff"
)
_V1_EVIDENCE_IDS = ("v1_canary",)


def permitted_claim(case_count: int) -> str:
    """Return the only allowed interpretation language for this small canary."""
    if type(case_count) is not int or case_count < 1:
        raise ValueError("case count must be a positive integer")
    return (
        f"On this {case_count}-case canary, these results are diagnostic evidence "
        "about these frozen cases and configurations only; the sample is too small "
        "for a population-quality claim."
    )


def audit_rationale_quotes(
    candidate_answer: str, decision: Mapping[str, object]
) -> dict[str, object]:
    """Require a verbatim candidate quote whenever a v2 decision sets a flag."""
    if type(candidate_answer) is not str:
        raise ValueError("candidate answer must be a string")
    flags = decision.get("error_flags")
    rationale = decision.get("rationale")
    if type(flags) is not list or any(type(flag) is not str for flag in flags):
        raise ValueError("decision error_flags must be a string list")
    if type(rationale) is not str:
        raise ValueError("decision rationale must be a string")
    if not flags:
        return {"passed": True, "reason": None, "quote": None}
    quotes = [
        next(part for part in match.groups() if part is not None)
        for match in _QUOTED_SPAN.finditer(rationale)
    ]
    usable = [quote for quote in quotes if len(quote.strip()) >= 3]
    if not usable:
        return {"passed": False, "reason": "missing_quote", "quote": None}
    for quote in usable:
        if quote in candidate_answer:
            return {"passed": True, "reason": None, "quote": quote}
    return {
        "passed": False,
        "reason": "quote_not_in_candidate",
        "quote": usable[0],
    }


def compare_runs(inputs: Mapping[str, object]) -> dict[str, object]:
    """Validate deterministic v2 evidence and return a private comparison object."""
    if type(inputs) is not dict:
        raise ValueError("comparison inputs must be an object")
    manifest = _mapping(inputs.get("manifest"), "manifest")
    cases = _cases(manifest)
    manifest_sha256 = inputs.get("manifest_sha256")
    if not _sha256(manifest_sha256):
        raise ValueError("manifest_sha256 must be a SHA-256 digest")
    case_ids = [case.response_id for case in cases]
    candidates = {case.response_id: case.candidate_answer for case in cases}
    reviewer = _reviewer_consensus(
        _mapping(inputs.get("advisor_review"), "advisor review"),
        _mapping(inputs.get("claude_review"), "claude review"),
        case_ids,
    )
    runs = _config_mapping(inputs.get("runs"), "runs")
    receipts = _config_mapping(inputs.get("runtime_receipts"), "runtime receipts")
    spend = _spend_ledger(_mapping(inputs.get("spend_ledger"), "spend ledger"))
    attempts = _attempt_ledger(_mapping(inputs.get("attempt_ledger"), "attempt ledger"))
    markers = _attempt_markers(inputs.get("attempt_markers"), attempts)
    spend_receipt = _mapping(inputs.get("spend_receipt"), "spend receipt")
    v1_runs = _mapping(inputs.get("v1_runs"), "v1 runs")
    if not v1_runs:
        raise ValueError("required v1 evidence is absent")
    contract = load_prompt_contract(_PROTOCOL_DIR, version="v2")
    expected_canary_identity = _v2_canary_identity(
        manifest_sha256=manifest_sha256, contract=contract
    )

    config_results: dict[str, dict[str, object]] = {}
    parsed_by_config: dict[str, dict[str, dict[str, object] | None]] = {}
    for config_id in CONFIGS:
        result, parsed = _compare_config(
            config_id,
            _mapping(runs[config_id], f"{config_id} run"),
            _mapping(receipts[config_id], f"{config_id} receipt"),
            cases,
            candidates,
            reviewer,
            expected_canary_identity,
        )
        config_results[config_id] = result
        parsed_by_config[config_id] = parsed

    _add_cross_model_agreement(config_results, parsed_by_config, case_ids)
    qwen_changes = _qwen_changes(parsed_by_config, case_ids)
    v1_differences = _v1_differences(v1_runs, parsed_by_config, cases)
    global_gates = _accounting_gates(
        config_results,
        runs,
        spend,
        attempts,
        markers,
        spend_receipt,
        expected_canary_identity,
    )
    return {
        "version": 1,
        "comparison_kind": "pointwise_judge_v2_diagnostic_canary",
        "case_count": len(cases),
        "claim": permitted_claim(len(cases)),
        "inputs": {
            "manifest_sha256": manifest_sha256,
            "spend_ledger": spend,
            "attempt_ledger": attempts,
            "attempt_markers": {"count": len(markers)},
            "spend_receipt": {"spent_usd": spend_receipt.get("spent_usd")},
            "reviewer_pass2": reviewer["summary"],
        },
        "configs": config_results,
        "qwen_direct_thinking": qwen_changes,
        "v1_differences": v1_differences,
        "global_hard_gate_failures": global_gates,
    }


def _compare_config(
    config_id: str,
    run: Mapping[str, object],
    receipt: Mapping[str, object],
    cases: Sequence[CandidateCase],
    candidates: Mapping[str, str],
    reviewer: Mapping[str, object],
    expected_canary_identity: str,
) -> tuple[dict[str, object], dict[str, dict[str, object] | None]]:
    case_ids = [case.response_id for case in cases]
    gates = _identity_gates(config_id, run, receipt, expected_canary_identity)
    records = run.get("records")
    try:
        _validate_run_container(dict(run))
    except (TypeError, ValueError):
        gates.append("runner_record_validation")
    if type(records) is not dict:
        raise ValueError(f"{config_id} run records must be an object")
    if set(records) != set(case_ids):
        gates.append("record_coverage")
    decisions: dict[str, dict[str, object] | None] = {}
    audit_records: dict[str, dict[str, object]] = {}
    elapsed = 0.0
    cost: float | None = 0.0
    repair_attempts = 0
    retry_attempts = 0
    for response_id in case_ids:
        record = records.get(response_id)
        if type(record) is not dict:
            gates.append("record_coverage")
            decisions[response_id] = None
            continue
        identity = _record_identity(
            cases[case_ids.index(response_id)],
            model_alias=run.get("model_alias"),
            model_id=run.get("model_id"),
            prompt_sha256=run.get("prompt_sha256"),
            schema_sha256=run.get("schema_sha256"),
            runtime_receipt_sha256=run.get("runtime_receipt_sha256"),
        )
        try:
            reusable = _validated_same_identity_record(
                record,
                identity,
                backend_response_model=receipt["backend_response_model"],
                version="v2",
            )
        except (KeyError, TypeError, ValueError):
            gates.append("runner_record_validation")
            decisions[response_id] = None
            continue
        if not reusable.terminal or record.get("decision") is None:
            gates.append("terminal_decision")
            decisions[response_id] = None
            continue
        retry_attempts += max(0, len(reusable.attempts) - 1)
        repair_attempts += int(len(reusable.attempts) > 1)
        elapsed += reusable.elapsed_seconds
        attempt_cost = reusable.usage.get("cost")
        if not _finite_nonnegative(attempt_cost):
            gates.append("usage")
            cost = None
        elif cost is not None:
            cost += float(attempt_cost)
        decision = record["decision"]
        try:
            parsed = _validated_v2_decision(decision)
        except ValueError:
            gates.append("parse_schema_invariant")
            decisions[response_id] = None
            continue
        decisions[response_id] = parsed
        audit = audit_rationale_quotes(candidates[response_id], parsed)
        audit_records[response_id] = audit
        if not audit["passed"]:
            gates.append("rationale_quote_fidelity")

    consensus = _mapping(reviewer["consensus"], "review consensus")
    agreement = _agreement_metrics(decisions, consensus, case_ids)
    audit_failed = sum(not item["passed"] for item in audit_records.values())
    result: dict[str, object] = {
        "completion": f"{sum(value is not None for value in decisions.values())}/{len(case_ids)}",
        "parse_schema_invariant_valid": all(
            value is not None for value in decisions.values()
        ),
        "provider_identity": receipt.get("provider"),
        "repair_attempts": repair_attempts,
        "retry_attempts": retry_attempts,
        "cost_usd": cost,
        "latency_seconds": elapsed,
        "rationale_audit": {
            "checked": len(audit_records),
            "failed": audit_failed,
            "records": audit_records,
        },
        **agreement,
        "hard_gate_failures": sorted(set(gates)),
    }
    return result, decisions


def _identity_gates(
    config_id: str,
    run: Mapping[str, object],
    receipt: Mapping[str, object],
    expected_canary_identity: str,
) -> list[str]:
    expected = _EXPECTED_CONFIGS[config_id]
    gates: list[str] = []
    if run.get("model_alias") != config_id or receipt.get("config_id") != config_id:
        gates.append("configuration_identity")
    if (
        run.get("model_id") != expected["model_id"]
        or receipt.get("model_id") != expected["model_id"]
    ):
        gates.append("model_identity")
    if receipt.get("provider") != expected["provider"]:
        gates.append("provider_identity")
    if receipt.get("reasoning") != expected["reasoning"]:
        gates.append("reasoning_identity")
    if (
        run.get("prompt_sha256") != FROZEN_V2_PROMPT_SHA256
        or receipt.get("prompt_sha256") != FROZEN_V2_PROMPT_SHA256
    ):
        gates.append("prompt_contract")
    if (
        run.get("schema_sha256") != FROZEN_V2_DECISION_SCHEMA_SHA256
        or receipt.get("schema_sha256") != FROZEN_V2_DECISION_SCHEMA_SHA256
    ):
        gates.append("schema_contract")
    expected_receipt = _runtime_receipt(
        V2_CONFIG_SPECS[config_id],
        load_prompt_contract(_PROTOCOL_DIR, version="v2"),
        version="v2",
        canary_identity_sha256=expected_canary_identity,
    )
    if receipt != expected_receipt:
        gates.append("runtime_receipt_contract")
    if run.get("runtime_receipt_sha256") != _json_hash(receipt):
        gates.append("runtime_receipt_binding")
    return gates


def _validated_v2_decision(value: Mapping[str, object]) -> dict[str, object]:
    try:
        raw = json.dumps(value, ensure_ascii=False, allow_nan=False)
        decision = parse_judge_decision(raw, version="v2")
    except (TypeError, ValueError) as error:
        raise ValueError("terminal decision is not a valid v2 decision") from error
    return {
        "label": decision.label,
        "correctness": decision.correctness,
        "completeness": decision.completeness,
        "relevance": decision.relevance,
        "error_flags": list(decision.error_flags),
        "reference_status": decision.reference_status,
        "confidence": decision.confidence,
        "rationale": decision.rationale,
    }


def _agreement_metrics(
    decisions: Mapping[str, dict[str, object] | None],
    reviewer: Mapping[str, object],
    case_ids: Sequence[str],
) -> dict[str, object]:
    result: dict[str, object] = {}
    consensus_by_field = _mapping(reviewer, "review consensus")
    for field in _DECISION_FIELDS:
        field_consensus = _mapping(consensus_by_field[field], f"{field} consensus")
        comparable = [
            response_id
            for response_id in case_ids
            if response_id in field_consensus and decisions[response_id] is not None
        ]
        excluded = [
            response_id for response_id in case_ids if response_id not in comparable
        ]
        excluded_count = len(excluded)
        if field == "error_flags":
            scores = [
                _jaccard(
                    set(decisions[response_id][field]),
                    set(field_consensus[response_id][field]),
                )
                for response_id in comparable
            ]
            value = sum(scores) / len(scores) if scores else None
            result["error_flag_mean_jaccard"] = value
            result["error_flag_mean_jaccard_detail"] = {
                "numerator": sum(scores) if scores else None,
                "denominator": len(comparable),
                "excluded_count": excluded_count,
                "excluded_response_ids": excluded,
            }
            continue
        exact = sum(
            decisions[response_id][field] == field_consensus[response_id][field]
            for response_id in comparable
        )
        result[f"{field}_agreement"] = exact / len(comparable) if comparable else None
        result[f"{field}_agreement_detail"] = {
            "numerator": exact if comparable else None,
            "denominator": len(comparable),
            "excluded_count": excluded_count,
            "excluded_response_ids": excluded,
        }
    return result


def _jaccard(left: set[object], right: set[object]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _reviewer_consensus(
    advisor: Mapping[str, object], claude: Mapping[str, object], case_ids: Sequence[str]
) -> dict[str, object]:
    advisor_rows = _review_rows(advisor, "advisor", case_ids)
    claude_rows = _review_rows(claude, "claude", case_ids)
    consensus: dict[str, dict[str, dict[str, object]]] = {
        field: {} for field in _DECISION_FIELDS
    }
    disagreement_by_field = {field: 0 for field in _DECISION_FIELDS}
    for response_id in case_ids:
        left = advisor_rows[response_id]
        right = claude_rows[response_id]
        for field in _DECISION_FIELDS:
            if left[field] != right[field]:
                disagreement_by_field[field] += 1
            else:
                consensus[field][response_id] = left
    return {
        "consensus": consensus,
        "summary": {
            "case_count": len(case_ids),
            "fully_agreed_cases": len(
                set.intersection(*(set(rows) for rows in consensus.values()))
            ),
            "field_disagreements": disagreement_by_field,
        },
    }


def _review_rows(
    review: Mapping[str, object], name: str, case_ids: Sequence[str]
) -> dict[str, dict[str, object]]:
    try:
        validate_reviewer_record(
            review,
            case_ids,
            expected_reviewer=name,
            allowed_reviewers=("advisor", "claude"),
        )
    except ValueError as error:
        raise ValueError(f"{name} reviewer ledger is invalid: {error}") from error
    rows = review["pass2"]
    result: dict[str, dict[str, object]] = {}
    for row in rows:
        if type(row) is not dict or type(row.get("response_id")) is not str:
            raise ValueError(f"{name} pass2 row is invalid")
        response_id = row["response_id"]
        if response_id in result:
            raise ValueError(f"{name} pass2 contains a duplicate response ID")
        result[response_id] = _review_decision(row)
    return result


def _review_decision(row: Mapping[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in _DECISION_FIELDS:
        value = row.get(field)
        if field == "error_flags":
            if type(value) is not list or any(type(item) is not str for item in value):
                raise ValueError("review error_flags are invalid")
        elif field in {"correctness", "completeness", "relevance"}:
            if type(value) is not int:
                raise ValueError(f"review {field} is invalid")
        elif type(value) is not str:
            raise ValueError(f"review {field} is invalid")
        result[field] = value
    return result


def _add_cross_model_agreement(
    config_results: Mapping[str, dict[str, object]],
    decisions: Mapping[str, Mapping[str, dict[str, object] | None]],
    case_ids: Sequence[str],
) -> None:
    for config_id in CONFIGS:
        scores: list[float] = []
        pair_details: dict[str, dict[str, object]] = {}
        for other in CONFIGS:
            if other == config_id:
                continue
            comparable = [
                response_id
                for response_id in case_ids
                if decisions[config_id][response_id] is not None
                and decisions[other][response_id] is not None
            ]
            excluded = [
                response_id for response_id in case_ids if response_id not in comparable
            ]
            if excluded:
                pair_details[other] = {
                    "available": False,
                    "numerator": None,
                    "denominator": len(comparable),
                    "excluded_response_ids": excluded,
                }
                continue
            numerator = sum(
                decisions[config_id][response_id]["label"]
                == decisions[other][response_id]["label"]
                for response_id in comparable
            )
            score = numerator / len(comparable)
            scores.append(score)
            pair_details[other] = {
                "available": True,
                "numerator": numerator,
                "denominator": len(comparable),
                "excluded_response_ids": [],
            }
        available = all(detail["available"] for detail in pair_details.values())
        config_results[config_id]["cross_model_label_agreement"] = (
            sum(scores) / len(scores) if available and scores else None
        )
        config_results[config_id]["cross_model_label_agreement_detail"] = pair_details


def _qwen_changes(
    decisions: Mapping[str, Mapping[str, dict[str, object] | None]],
    case_ids: Sequence[str],
) -> dict[str, object]:
    direct = decisions["qwen3.6-35b-a3b-direct"]
    thinking = decisions["qwen3.6-35b-a3b-thinking"]
    excluded = [
        response_id
        for response_id in case_ids
        if direct[response_id] is None or thinking[response_id] is None
    ]
    if excluded:
        return {
            "available": False,
            "excluded_count": len(excluded),
            "excluded_response_ids": excluded,
            "changed_case_count": None,
            "field_change_counts": None,
            "changes": [],
        }
    changes: list[dict[str, object]] = []
    field_counts = {field: 0 for field in _DECISION_FIELDS}
    for response_id in case_ids:
        if direct[response_id] is None or thinking[response_id] is None:
            continue
        changed = [
            field
            for field in _DECISION_FIELDS
            if direct[response_id][field] != thinking[response_id][field]
        ]
        for field in changed:
            field_counts[field] += 1
        if changed:
            changes.append({"response_id": response_id, "changed_fields": changed})
    return {
        "available": True,
        "excluded_count": 0,
        "excluded_response_ids": [],
        "changed_case_count": len(changes),
        "field_change_counts": field_counts,
        "changes": changes,
    }


def _v1_differences(
    v1_runs: Mapping[str, object],
    v2_decisions: Mapping[str, Mapping[str, dict[str, object] | None]],
    cases: Sequence[CandidateCase],
) -> dict[str, object]:
    case_ids = [case.response_id for case in cases]
    if set(v1_runs) != set(_V1_EVIDENCE_IDS):
        raise ValueError("v1 evidence must use the fixed v1_canary identifier")
    result: dict[str, object] = {}
    for name, raw_run in v1_runs.items():
        if type(raw_run) is not dict:
            raise ValueError("v1 runs must map names to run objects")
        try:
            _validate_run_container(raw_run)
        except ValueError as error:
            raise ValueError("v1 evidence run container is invalid") from error
        records = raw_run.get("records")
        if type(records) is not dict:
            raise ValueError("v1 run records must be an object")
        if raw_run.get("schema_sha256") != FROZEN_DECISION_SCHEMA_SHA256:
            raise ValueError("v1 evidence does not use the frozen v1 schema")
        expected_v1_prompt = hashlib.sha256(
            load_prompt_contract(_PROTOCOL_DIR, version="v1").prompt.encode("utf-8")
        ).hexdigest()
        if raw_run.get("prompt_sha256") != expected_v1_prompt:
            raise ValueError("v1 evidence does not use the frozen v1 prompt")
        if set(records) != set(case_ids):
            raise ValueError("v1 evidence coverage does not match the manifest")
        try:
            backend_response_model = trusted_run_backend_model(raw_run, case_ids)
        except ValueError as error:
            raise ValueError("v1 evidence backend identity is invalid") from error
        for case in cases:
            record = records[case.response_id]
            identity = _record_identity(
                case,
                model_alias=raw_run["model_alias"],
                model_id=raw_run["model_id"],
                prompt_sha256=raw_run["prompt_sha256"],
                schema_sha256=raw_run["schema_sha256"],
                runtime_receipt_sha256=raw_run["runtime_receipt_sha256"],
            )
            try:
                reusable = _validated_same_identity_record(
                    record,
                    identity,
                    backend_response_model=backend_response_model,
                    version="v1",
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("v1 evidence terminal record is invalid") from error
            if not reusable.terminal:
                raise ValueError("v1 evidence terminal record is incomplete")
        for config_id, decisions in v2_decisions.items():
            differences: list[dict[str, object]] = []
            for response_id in case_ids:
                record = records.get(response_id)
                old = record.get("decision") if type(record) is dict else None
                current = decisions[response_id]
                if type(old) is not dict or current is None:
                    differences.append(
                        {"response_id": response_id, "changed_fields": ["terminal"]}
                    )
                    continue
                changed = [
                    field
                    for field in _DECISION_FIELDS
                    if old.get(field) != current[field]
                ]
                if changed:
                    differences.append(
                        {"response_id": response_id, "changed_fields": changed}
                    )
            result[f"{name}__to__{config_id}"] = {
                "changed_case_count": len(differences),
                "differences": differences,
            }
    return result


def render_sanitized_markdown(comparison: Mapping[str, object]) -> str:
    """Render only aggregate, non-identifying results suitable for Markdown handoff."""
    configs = _mapping(comparison.get("configs"), "comparison configs")
    case_count = comparison.get("case_count")
    claim = comparison.get("claim")
    if type(case_count) is not int or type(claim) is not str:
        raise ValueError("comparison is missing its diagnostic claim")
    lines = [
        "# Pointwise Judge v2 Diagnostic Canary",
        "",
        claim,
        "",
        "## Per-configuration summary",
        "",
        "| Configuration | Provider | Completion | Label agreement | Correctness agreement | Completeness agreement | Relevance agreement | Reference-status agreement | Flag Jaccard | Cross-model label agreement | Cost (USD) | Latency (s) | Repair attempts | Retry attempts | Rationale audit | Hard-gate failures |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for config_id in CONFIGS:
        metrics = _mapping(configs.get(config_id), f"{config_id} metrics")
        lines.append(
            "| {config} | {provider} | {completion} | {label} | {correctness} | {completeness} | {relevance} | {reference_status} | {jaccard} | {cross_model} | {cost:.6f} | {latency:.3f} | {repairs} | {retries} | {audit} | {gates} |".format(
                config=html.escape(config_id),
                provider=html.escape(str(metrics.get("provider_identity"))),
                completion=metrics.get("completion"),
                label=_format_metric(metrics.get("label_agreement")),
                correctness=_format_metric(metrics.get("correctness_agreement")),
                completeness=_format_metric(metrics.get("completeness_agreement")),
                relevance=_format_metric(metrics.get("relevance_agreement")),
                reference_status=_format_metric(
                    metrics.get("reference_status_agreement")
                ),
                jaccard=_format_metric(metrics.get("error_flag_mean_jaccard")),
                cross_model=_format_metric(metrics.get("cross_model_label_agreement")),
                cost=_number_for_render(metrics.get("cost_usd")),
                latency=_number_for_render(metrics.get("latency_seconds")),
                repairs=metrics.get("repair_attempts"),
                retries=metrics.get("retry_attempts"),
                audit="{}/{}".format(
                    metrics["rationale_audit"]["failed"],
                    metrics["rationale_audit"]["checked"],
                ),
                gates=", ".join(metrics.get("hard_gate_failures", [])) or "none",
            )
        )
    qwen = _mapping(comparison.get("qwen_direct_thinking"), "Qwen comparison")
    lines.extend(
        [
            "",
            "## Paired Qwen direct/thinking changes",
            "",
            "Paired comparison available: {available}; changed cases: {changed}; excluded pairs: {excluded}.".format(
                available=qwen.get("available"),
                changed=qwen.get("changed_case_count"),
                excluded=qwen.get("excluded_count"),
            ),
            "",
            "## v1 differences",
            "",
            "| Comparison | Changed cases |",
            "| --- | ---: |",
            *[
                f"| {html.escape(name)} | {value.get('changed_case_count')} |"
                for name, value in _mapping(
                    comparison.get("v1_differences"), "v1 differences"
                ).items()
            ],
            "",
            "## Hard-gate failures",
            "",
            ", ".join(comparison.get("global_hard_gate_failures", [])) or "none",
            "",
            "## Claim boundary",
            "",
            "This report does not make a broad model-quality claim or rank models beyond this diagnostic sample.",
        ]
    )
    return "\n".join(lines) + "\n"


def _format_metric(value: object) -> str:
    return "n/a" if value is None else f"{_number_for_render(value):.3f}"


def _number_for_render(value: object) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)):
        raise ValueError("report metric must be finite")
    return float(value)


def _cases(manifest: Mapping[str, object]) -> list[CandidateCase]:
    rows = manifest.get("cases")
    if type(rows) is not list or len(rows) != 10:
        raise ValueError("v2 canary manifest must contain exactly ten cases")
    result: list[CandidateCase] = []
    for row in rows:
        if type(row) is not dict:
            raise ValueError("manifest case must be an object")
        try:
            result.append(
                CandidateCase(
                    question_id=row["question_id"],
                    response_id=row["response_id"],
                    question=row["question"],
                    reference_answer=row["reference_answer"],
                    candidate_answer=row["candidate_answer"],
                    strata=tuple(row["strata"]),
                    selection_reason=row["selection_reason"],
                )
            )
        except (KeyError, TypeError) as error:
            raise ValueError("manifest case is invalid") from error
    if any(
        type(getattr(case, field)) is not str or not getattr(case, field)
        for case in result
        for field in (
            "question_id",
            "response_id",
            "question",
            "reference_answer",
            "candidate_answer",
            "selection_reason",
        )
    ) or any(
        not case.strata
        or any(type(item) is not str or not item for item in case.strata)
        for case in result
    ):
        raise ValueError("manifest case fields are invalid")
    if len({case.response_id for case in result}) != len(result):
        raise ValueError("manifest response IDs must be unique")
    if not _sha256(manifest.get("manifest_hash")):
        raise ValueError("manifest hash must be a SHA-256 digest")
    return result


def _spend_ledger(value: Mapping[str, object]) -> dict[str, object]:
    if (
        value.get("version") != 1
        or not _finite_nonnegative(value.get("spent_usd"))
        or not _finite_nonnegative(value.get("cap_usd"))
    ):
        raise ValueError("spend ledger is invalid")
    if value["spent_usd"] > value["cap_usd"]:
        raise ValueError("spend ledger exceeds its cap")
    return {"cap_usd": value["cap_usd"], "spent_usd": value["spent_usd"]}


def _attempt_ledger(value: Mapping[str, object]) -> dict[str, object]:
    if (
        value.get("version") != 2
        or type(value.get("cap_calls")) is not int
        or type(value.get("used_calls")) is not int
        or not _sha256(value.get("canary_identity_sha256"))
    ):
        raise ValueError("attempt ledger is invalid")
    if (
        not 1 <= value["cap_calls"] <= 100
        or not 40 <= value["used_calls"] <= value["cap_calls"]
    ):
        raise ValueError("attempt ledger is outside canary bounds")
    return {
        "cap_calls": value["cap_calls"],
        "used_calls": value["used_calls"],
        "canary_identity_sha256": value["canary_identity_sha256"],
    }


def _attempt_markers(
    value: object, attempts: Mapping[str, object]
) -> list[dict[str, object]]:
    if type(value) is not list:
        raise ValueError("attempt markers must be a list")
    expected_count = attempts["used_calls"]
    if len(value) != expected_count:
        raise ValueError("attempt markers do not reconcile with the attempt ledger")
    markers: list[dict[str, object]] = []
    for count, marker in enumerate(value, start=1):
        if marker != {
            "version": 1,
            "attempt": count,
            "canary_identity_sha256": attempts["canary_identity_sha256"],
        }:
            raise ValueError("attempt marker identity is invalid")
        markers.append(marker)
    return markers


def _accounting_gates(
    config_results: Mapping[str, Mapping[str, object]],
    runs: Mapping[str, object],
    spend: Mapping[str, object],
    attempts: Mapping[str, object],
    markers: Sequence[Mapping[str, object]],
    spend_receipt: Mapping[str, object],
    expected_canary_identity: str,
) -> list[str]:
    gates: list[str] = []
    if attempts["canary_identity_sha256"] != expected_canary_identity:
        gates.append("canary_identity")
    semantic_attempts = sum(
        len(run.get("records", {}).get(response_id, {}).get("attempts", []))
        for run in runs.values()
        if type(run) is dict
        for response_id in run.get("records", {})
    )
    costs = [result["cost_usd"] for result in config_results.values()]
    if any(not _finite_nonnegative(cost) for cost in costs):
        gates.append("accounting_reconciliation")
        recorded_cost = None
    else:
        recorded_cost = sum(float(cost) for cost in costs)
    if (
        attempts["used_calls"] < semantic_attempts
        or len(markers) != attempts["used_calls"]
    ):
        gates.append("accounting_reconciliation")
    if not _valid_spend_receipt(spend_receipt, runs, spend, attempts):
        gates.append("accounting_reconciliation")
    if recorded_cost is None or not math.isclose(
        recorded_cost, float(spend["spent_usd"]), abs_tol=1e-12
    ):
        gates.append("accounting_reconciliation")
    return sorted(set(gates))


def _valid_spend_receipt(
    receipt: Mapping[str, object],
    runs: Mapping[str, object],
    spend: Mapping[str, object],
    attempts: Mapping[str, object],
) -> bool:
    if (
        receipt.get("version") != 1
        or receipt.get("api_key_serialized") is not False
        or receipt.get("cap_usd") != spend["cap_usd"]
        or receipt.get("spent_usd") != spend["spent_usd"]
        or receipt.get("call_cap") != attempts["cap_calls"]
        or receipt.get("used_calls") != attempts["used_calls"]
    ):
        return False
    models = receipt.get("models")
    if type(models) is not list or len(models) != len(CONFIGS):
        return False
    by_config = {
        model.get("config_id"): model for model in models if type(model) is dict
    }
    if set(by_config) != set(CONFIGS):
        return False
    for config_id in CONFIGS:
        model = by_config[config_id]
        run = runs[config_id]
        if model.get("record_count") != 10 or model.get("run_sha256") != _json_hash(
            run
        ):
            return False
    return True


def _config_mapping(value: object, name: str) -> Mapping[str, object]:
    mapping = _mapping(value, name)
    if set(mapping) != set(CONFIGS):
        raise ValueError(f"{name} must contain exactly the four v2 configurations")
    return mapping


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if type(value) is not dict:
        raise ValueError(f"{name} must be an object")
    return value


def _sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _finite_nonnegative(value: object) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and value >= 0


def _nonnegative_number(value: object, name: str) -> float:
    if not _finite_nonnegative(value):
        raise ValueError(f"{name} must be finite and nonnegative")
    return float(value)


def _read_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"could not read JSON input: {path}") from error


def _read_private_json(path: Path) -> object:
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise ValueError("private report input must be a mode-0600 regular file")
    return _read_json(path)


def _parse_keyed_paths(values: Sequence[str], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        config_id, separator, raw_path = value.partition("=")
        if (
            not separator
            or config_id not in CONFIGS
            or not raw_path
            or config_id in result
        ):
            raise ValueError(f"{label} entries must be unique CONFIG_ID=PATH values")
        result[config_id] = Path(raw_path)
    if set(result) != set(CONFIGS):
        raise ValueError(f"{label} must supply all four configurations")
    return result


def _write_private_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _write_markdown(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--advisor-review", type=Path, required=True)
    parser.add_argument("--claude-review", type=Path, required=True)
    parser.add_argument(
        "--v2-run", action="append", required=True, metavar="CONFIG=PATH"
    )
    parser.add_argument(
        "--runtime-receipt", action="append", required=True, metavar="CONFIG=PATH"
    )
    parser.add_argument("--spend-ledger", type=Path, required=True)
    parser.add_argument("--attempt-ledger", type=Path, required=True)
    parser.add_argument("--attempt-marker-dir", type=Path, required=True)
    parser.add_argument("--spend-receipt", type=Path, required=True)
    parser.add_argument("--v1-run", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    runs = _parse_keyed_paths(args.v2_run, "v2 runs")
    receipts = _parse_keyed_paths(args.runtime_receipt, "runtime receipts")
    v1_paths: dict[str, Path] = {}
    for value in args.v1_run:
        name, separator, raw_path = value.partition("=")
        if (
            not separator
            or name not in _V1_EVIDENCE_IDS
            or not raw_path
            or name in v1_paths
        ):
            raise ValueError("v1 runs must use the fixed v1_canary=PATH value")
        v1_paths[name] = Path(raw_path)
    if set(v1_paths) != set(_V1_EVIDENCE_IDS):
        raise ValueError("v1 evidence must supply v1_canary=PATH")
    marker_dir = args.attempt_marker_dir
    if not marker_dir.is_dir() or stat.S_IMODE(marker_dir.stat().st_mode) != 0o700:
        raise ValueError("attempt marker directory must be mode 0700")
    marker_paths = sorted(marker_dir.iterdir())
    comparison = compare_runs(
        {
            "manifest": _read_private_json(args.manifest),
            "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
            "runs": {
                config_id: _read_private_json(path) for config_id, path in runs.items()
            },
            "runtime_receipts": {
                config_id: _read_private_json(path)
                for config_id, path in receipts.items()
            },
            "advisor_review": _read_private_json(args.advisor_review),
            "claude_review": _read_private_json(args.claude_review),
            "spend_ledger": _read_private_json(args.spend_ledger),
            "attempt_ledger": _read_private_json(args.attempt_ledger),
            "attempt_markers": [_read_private_json(path) for path in marker_paths],
            "spend_receipt": _read_private_json(args.spend_receipt),
            "v1_runs": {
                name: _read_private_json(path) for name, path in v1_paths.items()
            },
        }
    )
    _write_private_json(args.output_json, comparison)
    _write_markdown(args.output_markdown, render_sanitized_markdown(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
