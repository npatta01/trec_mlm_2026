"""Aggregate private pointwise-smoke evidence without manufacturing a gold label."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict

from trec_mlm.pointwise_judge import CandidateCase, LABELS
from trec_mlm.pointwise_review import validate_reviewer_record
from trec_mlm.pointwise_runner import (
    FROZEN_DECISION_SCHEMA_SHA256,
    _validate_run_container,
    _validated_same_identity_record,
)
from trec_mlm.pointwise_sampling import DIAGNOSTIC_REASONS


_REVIEWERS = ("advisor", "claude")
_ADJACENT = frozenset({"good", "partial"})
_JUDGE_PREFIX = "judge_"


def build_smoke_summary(
    *,
    cases: Sequence[CandidateCase],
    runs: Mapping[str, Mapping[str, object]],
    advisor_review: Mapping[str, object],
    claude_review: Mapping[str, object],
    sealed_judge_aliases: Mapping[str, str],
    runtime_receipts: Mapping[str, Mapping[str, object]] | None = None,
    manifest_hash: str | None = None,
) -> dict[str, object]:
    """Build deterministic, reviewer-consensus-only smoke metrics.

    This function intentionally reports agreement with two fallible reviewers;
    it does not produce a gold label or an accuracy estimate.
    """
    validated_cases = _validate_cases(cases)
    case_ids = [case.response_id for case in validated_cases]
    aliases = _validate_aliases(sealed_judge_aliases, runs)
    reviews = {"advisor": dict(advisor_review), "claude": dict(claude_review)}
    for reviewer, ledger in reviews.items():
        validate_reviewer_record(
            ledger,
            case_ids,
            expected_reviewer=reviewer,
            judge_aliases=aliases,
            forbidden_identities=set(runs)
            | {
                str(run.get("model_id"))
                for run in runs.values()
                if type(run) is dict and type(run.get("model_id")) is str
            },
        )
    reviewer_rows = {
        reviewer: {
            phase: {row["response_id"]: dict(row) for row in ledger[phase]}
            for phase in ("pass1", "pass2")
        }
        for reviewer, ledger in reviews.items()
    }
    consensus = {
        response_id: _reviewer_consensus(
            reviewer_rows["advisor"]["pass2"][response_id],
            reviewer_rows["claude"]["pass2"][response_id],
        )
        for response_id in case_ids
    }
    receipts = {} if runtime_receipts is None else runtime_receipts
    judges: dict[str, object] = {}
    private_case_rows: list[dict[str, object]] = []
    records_by_judge: dict[str, dict[str, dict[str, object]]] = {}
    for judge, run_alias in sorted(aliases.items()):
        run = runs[run_alias]
        records = _validated_run(run, run_alias, validated_cases)
        records_by_judge[judge] = records
        judges[judge] = _judge_metrics(
            records,
            reviewer_rows,
            consensus,
            receipts.get(run_alias),
        )

    for case in validated_cases:
        response_id = case.response_id
        private_case_rows.append(
            {
                "case": asdict(case),
                "reviewers": {
                    reviewer: {
                        "pass1": reviewer_rows[reviewer]["pass1"][response_id],
                        "pass2": reviewer_rows[reviewer]["pass2"][response_id],
                    }
                    for reviewer in _REVIEWERS
                },
                "model_reviewer_consensus": consensus[response_id],
                "judges": {
                    judge: _private_judge_record(records[response_id])
                    for judge, records in sorted(records_by_judge.items())
                },
            }
        )

    review_disagreements = sum(value is None for value in consensus.values())
    return {
        "version": 1,
        "case_count": len(validated_cases),
        "manifest_hash": _hash_or_none(manifest_hash),
        "reviewers": {
            "pass2_disagreement_count": review_disagreements,
            "pass2_disagreement_rate": _rate(
                review_disagreements, len(validated_cases)
            ),
            "model_reviewer_consensus_count": len(validated_cases)
            - review_disagreements,
            "unresolved_case_count": review_disagreements,
        },
        "judges": judges,
        "diagnostic_proxies": _diagnostic_rollups(
            validated_cases, records_by_judge, consensus
        ),
        "cases": private_case_rows,
    }


def sanitize_summary(summary: Mapping[str, object]) -> dict[str, object]:
    """Return a public whitelist: counts, rates, runtime facts, and hashes only."""
    if type(summary) is not dict:
        raise ValueError("summary must be an object")
    judges = summary.get("judges")
    reviewers = summary.get("reviewers")
    diagnostics = summary.get("diagnostic_proxies")
    if (
        type(judges) is not dict
        or type(reviewers) is not dict
        or type(diagnostics) is not dict
        or type(summary.get("case_count")) is not int
    ):
        raise ValueError("summary has invalid report fields")
    return {
        "version": summary.get("version"),
        "case_count": summary["case_count"],
        "manifest_hash": summary.get("manifest_hash"),
        "reviewers": {
            key: reviewers[key]
            for key in (
                "pass2_disagreement_count",
                "pass2_disagreement_rate",
                "model_reviewer_consensus_count",
                "unresolved_case_count",
            )
        },
        "judges": {
            judge: _public_judge_metrics(metrics)
            for judge, metrics in sorted(judges.items())
            if type(judge) is str and judge.startswith(_JUDGE_PREFIX)
        },
        "diagnostic_proxies": {
            reason: _public_proxy_metrics(diagnostics[reason])
            for reason in DIAGNOSTIC_REASONS
        },
    }


def render_private_report(summary: Mapping[str, object]) -> str:
    """Render the private case table, including reviewer passes and prompt findings."""
    if type(summary) is not dict or type(summary.get("cases")) is not list:
        raise ValueError("summary must include private case rows")
    lines = ["# Local pointwise judge smoke review", ""]
    lines.extend(
        [
            "## Aggregate evidence",
            "",
            "All comparisons are against fallible reviewer records; no gold label or accuracy estimate is asserted.",
            "",
            f"Cases: {summary.get('case_count')}",
            f"Unresolved reviewer cases: {summary['reviewers']['unresolved_case_count']}",
            "",
            "## Private case table",
        ]
    )
    for index, row in enumerate(summary["cases"], start=1):
        if type(row) is not dict:
            raise ValueError("private case row must be an object")
        case = row["case"]
        lines.extend(
            [
                "",
                f"### Case {index}: {case['response_id']}",
                "",
                "**Question**",
                str(case["question"]),
                "",
                "**Provisional reference**",
                str(case["reference_answer"]),
                "",
                "**Candidate response**",
                str(case["candidate_answer"]),
                "",
            ]
        )
        for reviewer in _REVIEWERS:
            ledgers = row["reviewers"][reviewer]
            lines.extend(
                [
                    f"**{reviewer.title()} pass 1:** {_decision_line(ledgers['pass1'])}",
                    f"**{reviewer.title()} pass 2:** {_decision_line(ledgers['pass2'])}",
                ]
            )
        consensus = row["model_reviewer_consensus"]
        lines.append(
            "**Model reviewer consensus:** "
            + (str(consensus) if consensus is not None else "unresolved")
        )
        for judge, record in sorted(row["judges"].items()):
            lines.append(f"**{judge}:** {_decision_line(record.get('decision'))}")
        findings = [
            finding
            for reviewer in _REVIEWERS
            for finding in row["reviewers"][reviewer]["pass2"]["prompt_findings"]
        ]
        if findings:
            lines.append("**Proposed prompt changes:** " + " | ".join(findings))
    return "\n".join(lines) + "\n"


def _validate_cases(cases: Sequence[CandidateCase]) -> list[CandidateCase]:
    if type(cases) not in {list, tuple} or not cases:
        raise ValueError("cases must be a nonempty sequence")
    if any(type(case) is not CandidateCase for case in cases):
        raise ValueError("cases must contain CandidateCase values")
    if len({case.response_id for case in cases}) != len(cases):
        raise ValueError("case response IDs must be unique")
    reasons = [case.selection_reason for case in cases]
    if (
        len(reasons) != len(DIAGNOSTIC_REASONS)
        or len(set(reasons)) != len(reasons)
        or set(reasons) != set(DIAGNOSTIC_REASONS)
    ):
        raise ValueError(
            "cases must contain each frozen Task 2 diagnostic reasons exactly once"
        )
    return list(cases)


def _validate_aliases(
    sealed: Mapping[str, str], runs: Mapping[str, Mapping[str, object]]
) -> dict[str, str]:
    if type(sealed) is not dict or type(runs) is not dict or not sealed:
        raise ValueError("sealed judge aliases and runs must be nonempty objects")
    result = dict(sealed)
    if (
        any(
            type(judge) is not str
            or not judge.startswith(_JUDGE_PREFIX)
            or type(alias) is not str
            or not alias
            for judge, alias in result.items()
        )
        or len(set(result.values())) != len(result)
        or set(result.values()) != set(runs)
    ):
        raise ValueError("sealed judge aliases must exactly cover runs")
    return result


def _validated_run(
    run: Mapping[str, object], alias: str, cases: Sequence[CandidateCase]
) -> dict[str, dict[str, object]]:
    if type(run) is not dict or run.get("model_alias") != alias:
        raise ValueError("run does not match sealed judge alias")
    try:
        _validate_run_container(run)
    except ValueError as error:
        raise ValueError(f"invalid run {alias}: {error}") from error
    if run["schema_sha256"] != FROZEN_DECISION_SCHEMA_SHA256:
        raise ValueError("run does not use the frozen decision schema")
    records = run.get("records")
    case_ids = [case.response_id for case in cases]
    if type(records) is not dict or set(records) != set(case_ids):
        raise ValueError("run response coverage does not match cases")
    result: dict[str, dict[str, object]] = {}
    for case in cases:
        response_id = case.response_id
        record = records[response_id]
        if type(record) is not dict:
            raise ValueError("run record must be an object")
        if record.get("input_sha256") != _canonical_hash(asdict(case)):
            raise ValueError("run record input identity does not match case")
        try:
            reusable = _validated_same_identity_record(record, record)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid run record {response_id}: {error}") from error
        if not reusable.terminal:
            raise ValueError("run record is not terminal")
        decision = record.get("decision")
        if decision is not None:
            if type(decision) is not dict or decision.get("label") not in LABELS:
                raise ValueError("run record has invalid decision")
        result[response_id] = dict(record)
    return result


def _judge_metrics(
    records: Mapping[str, Mapping[str, object]],
    reviewer_rows: Mapping[str, Mapping[str, Mapping[str, Mapping[str, object]]]],
    consensus: Mapping[str, str | None],
    receipt: Mapping[str, object] | None,
) -> dict[str, object]:
    count = len(records)
    decisions = {
        response_id: record["decision"]
        for response_id, record in records.items()
        if record.get("decision") is not None
    }
    success = len(decisions)
    repair_count = sum(len(record["attempts"]) > 1 for record in records.values())
    failures = count - success
    exact = {
        reviewer: sum(
            decision["label"] == reviewer_rows[reviewer]["pass2"][response_id]["label"]
            for response_id, decision in decisions.items()
        )
        for reviewer in _REVIEWERS
    }
    adjacent = {
        reviewer: sum(
            _adjacent_or_exact(
                decision["label"],
                reviewer_rows[reviewer]["pass2"][response_id]["label"],
            )
            for response_id, decision in decisions.items()
        )
        for reviewer in _REVIEWERS
    }
    consensus_decisions = {
        response_id: label
        for response_id, label in consensus.items()
        if label is not None and response_id in decisions
    }
    consensus_exact = sum(
        decisions[response_id]["label"] == label
        for response_id, label in consensus_decisions.items()
    )
    consensus_adjacent = sum(
        _adjacent_or_exact(decisions[response_id]["label"], label)
        for response_id, label in consensus_decisions.items()
    )
    false_confident = sum(
        decisions[response_id].get("confidence") == "high"
        and decisions[response_id]["label"] != label
        for response_id, label in consensus_decisions.items()
    )
    labels = Counter(str(decision["label"]) for decision in decisions.values())
    reference_conflicts = sum(
        "reference_conflict" in decision.get("error_flags", [])
        for decision in decisions.values()
    )
    elapsed = sum(float(record["elapsed_seconds"]) for record in records.values())
    tokens = sum(_total_tokens(record.get("usage")) for record in records.values())
    return {
        "record_count": count,
        "schema_success": _rate(success, count),
        "schema_failure_count": failures,
        "failure_rate": _rate(failures, count),
        "repair_count": repair_count,
        "repair_rate": _rate(repair_count, count),
        "label_distribution": {label: labels.get(label, 0) for label in sorted(LABELS)},
        "reference_conflict_count": reference_conflicts,
        "reference_conflict_rate": _rate(reference_conflicts, success),
        **{
            f"{reviewer}_exact_agreement": _rate(exact[reviewer], success)
            for reviewer in _REVIEWERS
        },
        **{
            f"{reviewer}_adjacent_good_partial_agreement": _rate(
                adjacent[reviewer], success
            )
            for reviewer in _REVIEWERS
        },
        "model_reviewer_consensus_count": len(consensus_decisions),
        "model_reviewer_consensus_exact_agreement": _rate(
            consensus_exact, len(consensus_decisions)
        ),
        "model_reviewer_consensus_adjacent_good_partial_agreement": _rate(
            consensus_adjacent, len(consensus_decisions)
        ),
        "false_confident_errors": false_confident,
        "false_confident_error_rate": _rate(false_confident, len(consensus_decisions)),
        "latency_seconds": {
            "total": elapsed,
            "mean": _rate(elapsed, count),
        },
        "throughput": {
            "records_per_second": _rate(count, elapsed),
            "tokens_per_second": _rate(tokens, elapsed),
            "total_tokens": tokens,
        },
        "runtime": _runtime_facts(receipt),
    }


def _diagnostic_rollups(
    cases: Sequence[CandidateCase],
    records_by_judge: Mapping[str, Mapping[str, Mapping[str, object]]],
    consensus: Mapping[str, str | None],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for proxy in DIAGNOSTIC_REASONS:
        case_ids = [
            case.response_id for case in cases if case.selection_reason == proxy
        ]
        result[proxy] = {
            "case_count": len(case_ids),
            "model_reviewer_consensus_count": sum(
                consensus[response_id] is not None for response_id in case_ids
            ),
            "unresolved_case_count": sum(
                consensus[response_id] is None for response_id in case_ids
            ),
            "judges": {
                judge: {
                    "schema_success_count": sum(
                        records[response_id].get("decision") is not None
                        for response_id in case_ids
                    ),
                    "reference_conflict_count": sum(
                        records[response_id].get("decision") is not None
                        and "reference_conflict"
                        in records[response_id]["decision"].get("error_flags", [])
                        for response_id in case_ids
                    ),
                }
                for judge, records in sorted(records_by_judge.items())
            },
        }
    return result


def _public_judge_metrics(metrics: object) -> dict[str, object]:
    if type(metrics) is not dict:
        raise ValueError("judge metrics must be an object")
    allowed = {
        "record_count",
        "schema_success",
        "schema_failure_count",
        "failure_rate",
        "repair_count",
        "repair_rate",
        "label_distribution",
        "reference_conflict_count",
        "reference_conflict_rate",
        "advisor_exact_agreement",
        "claude_exact_agreement",
        "advisor_adjacent_good_partial_agreement",
        "claude_adjacent_good_partial_agreement",
        "model_reviewer_consensus_count",
        "model_reviewer_consensus_exact_agreement",
        "model_reviewer_consensus_adjacent_good_partial_agreement",
        "false_confident_errors",
        "false_confident_error_rate",
        "latency_seconds",
        "throughput",
        "runtime",
    }
    return {key: metrics[key] for key in sorted(allowed)}


def _public_proxy_metrics(metrics: object) -> dict[str, object]:
    if type(metrics) is not dict or type(metrics.get("judges")) is not dict:
        raise ValueError("diagnostic proxy metrics must be an object")
    return {
        "case_count": metrics["case_count"],
        "model_reviewer_consensus_count": metrics["model_reviewer_consensus_count"],
        "unresolved_case_count": metrics["unresolved_case_count"],
        "judges": {
            judge: {
                "schema_success_count": values["schema_success_count"],
                "reference_conflict_count": values["reference_conflict_count"],
            }
            for judge, values in sorted(metrics["judges"].items())
            if type(judge) is str and judge.startswith(_JUDGE_PREFIX)
        },
    }


def _private_judge_record(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "decision": record.get("decision"),
        "failure": record.get("failure"),
        "elapsed_seconds": record.get("elapsed_seconds"),
        "usage": record.get("usage"),
        "attempts": record.get("attempts"),
    }


def _reviewer_consensus(
    advisor: Mapping[str, object], claude: Mapping[str, object]
) -> str | None:
    left, right = advisor["label"], claude["label"]
    return left if left == right else None


def _adjacent_or_exact(left: object, right: object) -> bool:
    return left == right or (left in _ADJACENT and right in _ADJACENT)


def _decision_line(value: object) -> str:
    if value is None:
        return "terminal schema failure"
    if type(value) is not dict:
        raise ValueError("decision must be an object")
    return f"{value.get('label')} (confidence: {value.get('confidence')}); {value.get('rationale')}"


def _total_tokens(value: object) -> float:
    if type(value) is not dict:
        return 0.0
    total = value.get("total_tokens", 0)
    return float(total) if type(total) in {int, float} and total >= 0 else 0.0


def _runtime_facts(receipt: Mapping[str, object] | None) -> dict[str, object]:
    if receipt is None or type(receipt) is not dict:
        return {"startup_seconds": None, "visible_memory_bytes": None}
    hardware = receipt.get("hardware")
    memory = hardware.get("visible_memory_bytes") if type(hardware) is dict else None
    return {
        "startup_seconds": receipt.get("startup_seconds"),
        "visible_memory_bytes": memory,
    }


def _rate(numerator: int | float, denominator: int | float) -> float:
    return 0.0 if denominator == 0 else float(numerator) / float(denominator)


def _hash_or_none(value: object) -> str | None:
    return value if type(value) is str and len(value) == 64 else None


def _canonical_hash(value: object) -> str:
    import hashlib
    import json

    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
