#!/usr/bin/env python3
"""Audit grounded Gemini caches and enforce the pre-spend canary gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


VALID_STATUSES = {
    "answered",
    "ambiguous",
    "insufficient_evidence",
    "conflicting_evidence",
}
REFUSAL_RE = re.compile(
    r"\b(?:i (?:can(?:not|'t)|am unable to)|cannot (?:answer|help)|unable to answer)\b",
    re.IGNORECASE,
)


def _number(mapping: Mapping[str, Any], field: str) -> float:
    value = mapping.get(field, 0)
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool)
        else 0.0
    )


def audit_generation(
    questions: list[Mapping[str, Any]],
    public_cache: Mapping[str, Any],
    private_cache: Mapping[str, Any],
    *,
    projected_question_count: int,
    cost_cap_usd: float = 150.0,
    free_grounded_prompts: int = 1_500,
    search_reviews: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a deterministic integrity, provenance, and cost audit."""

    expected_ids = [str(row["question_id"]) for row in questions]
    duplicate_expected_ids = sorted(
        {
            question_id
            for question_id in expected_ids
            if expected_ids.count(question_id) > 1
        }
    )
    expected = set(expected_ids)
    records = public_cache.get("records", {})
    failures = public_cache.get("failures", {})
    evidence = private_cache.get("records", {})
    if (
        not isinstance(records, Mapping)
        or not isinstance(failures, Mapping)
        or not isinstance(evidence, Mapping)
    ):
        raise ValueError("audit caches must contain records/failures mappings")

    observed = set(str(value) for value in records)
    missing_ids = sorted(expected - observed)
    extra_ids = sorted(observed - expected)
    failure_ids = sorted(expected & set(str(value) for value in failures))
    invalid_status_ids: list[str] = []
    empty_answer_ids: list[str] = []
    truncated_ids: list[str] = []
    abnormal_finish_ids: list[str] = []
    missing_evidence_ids: list[str] = []
    missing_source_url_ids: list[str] = []
    non_search_ids: list[str] = []
    uncertain_ids: list[str] = []
    refusal_ids: list[str] = []
    expected_search_not_used_ids: list[str] = []
    search_reviewed_not_required_ids: list[str] = []
    search_reviews = search_reviews or {}
    expected_search = {
        str(row["question_id"])
        for row in questions
        if row.get("search_expected") is True
    }
    input_tokens = output_tokens = search_queries = 0.0
    search_used_count = 0
    status_counts = {status: 0 for status in sorted(VALID_STATUSES)}

    for question_id in sorted(expected & observed):
        record = records[question_id]
        if not isinstance(record, Mapping):
            missing_ids.append(question_id)
            continue
        answer = record.get("answer")
        status = record.get("uncertainty_status")
        reason = record.get("uncertainty_reason")
        if (
            status not in VALID_STATUSES
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            invalid_status_ids.append(question_id)
        else:
            status_counts[status] += 1
        if not isinstance(answer, str) or not answer.strip():
            empty_answer_ids.append(question_id)
        elif REFUSAL_RE.search(answer):
            refusal_ids.append(question_id)
        if status in VALID_STATUSES - {"answered"}:
            uncertain_ids.append(question_id)
        finish_reason = record.get("finish_reason")
        if finish_reason == "length":
            truncated_ids.append(question_id)
        if finish_reason != "stop":
            abnormal_finish_ids.append(question_id)

        private_record = evidence.get(question_id)
        if not isinstance(private_record, Mapping):
            missing_evidence_ids.append(question_id)
            search_invoked = False
        else:
            search_invoked = private_record.get("search_invoked") is True
            sources = private_record.get("sources")
            has_url = isinstance(sources, list) and any(
                isinstance(source, Mapping)
                and isinstance(source.get("url"), str)
                and bool(source["url"].strip())
                for source in sources
            )
            if search_invoked and not has_url:
                missing_source_url_ids.append(question_id)
        if not search_invoked:
            non_search_ids.append(question_id)
            if question_id in expected_search:
                review = search_reviews.get(question_id)
                if (
                    isinstance(review, Mapping)
                    and review.get("decision") == "not_required"
                    and isinstance(review.get("reason"), str)
                    and review["reason"].strip()
                ):
                    search_reviewed_not_required_ids.append(question_id)
                else:
                    expected_search_not_used_ids.append(question_id)
        else:
            search_used_count += 1

        usage = record.get("usage")
        if isinstance(usage, Mapping):
            prompt = _number(usage, "promptTokenCount") + _number(
                usage, "toolUsePromptTokenCount"
            )
            reported_output = _number(usage, "candidatesTokenCount") + _number(
                usage, "thoughtsTokenCount"
            )
            residual_output = max(0.0, _number(usage, "totalTokenCount") - prompt)
            input_tokens += prompt
            output_tokens += max(reported_output, residual_output)
        request_metadata = record.get("request_metadata")
        if isinstance(request_metadata, Mapping):
            search_queries += _number(request_metadata, "grounding_query_count")

    completed_count = len(expected & observed)
    scale = (
        projected_question_count / completed_count if completed_count else float("inf")
    )
    projected_input = input_tokens * scale
    projected_output = output_tokens * scale
    average_queries = (
        search_queries / completed_count if completed_count else float("inf")
    )
    # Deliberately use twice the published $0.50/$3.00 standard token prices.
    model_cost = projected_input / 1_000_000 + projected_output * 6 / 1_000_000
    standard_model_cost = (
        projected_input * 0.5 / 1_000_000 + projected_output * 3 / 1_000_000
    )
    if free_grounded_prompts < 0:
        raise ValueError("free_grounded_prompts must be non-negative")
    billed_search_queries = max(0, projected_question_count - free_grounded_prompts) * average_queries
    search_cost = billed_search_queries * 14 / 1_000
    conservative_total = model_cost + search_cost

    blockers = {
        "duplicate_expected_ids": duplicate_expected_ids,
        "missing_ids": sorted(set(missing_ids)),
        "extra_ids": extra_ids,
        "failure_ids": failure_ids,
        "invalid_status_ids": invalid_status_ids,
        "empty_answer_ids": empty_answer_ids,
        "truncated_ids": truncated_ids,
        "abnormal_finish_ids": abnormal_finish_ids,
        "missing_evidence_ids": missing_evidence_ids,
        "missing_source_url_ids": missing_source_url_ids,
        "expected_search_not_used_ids": expected_search_not_used_ids,
    }
    gate_passed = not any(blockers.values()) and conservative_total < cost_cap_usd
    return {
        "gate_passed": gate_passed,
        "cost_cap_usd": cost_cap_usd,
        "counts": {
            "expected": len(expected_ids),
            "unique_question_ids": len(expected),
            "records": len(records),
            "failures": len(failures),
            "refusals": len(refusal_ids),
            "non_search": len(non_search_ids),
            "uncertain": len(uncertain_ids),
            "search_used": search_used_count,
        },
        "status_counts": status_counts,
        **blockers,
        "refusal_ids": refusal_ids,
        "non_search_ids": non_search_ids,
        "search_reviewed_not_required_ids": search_reviewed_not_required_ids,
        "uncertain_ids": uncertain_ids,
        "projected_cost_usd": {
            "estimated_standard_model": round(standard_model_cost, 4),
            "estimated_standard_total": round(standard_model_cost + search_cost, 4),
            "conservative_model": round(model_cost, 4),
            "search": round(search_cost, 4),
            "conservative_total": round(conservative_total, 4),
            "average_search_queries_per_request": round(average_queries, 4),
            "assumptions": {
                "standard_input_per_million_tokens": 0.5,
                "standard_output_per_million_tokens": 3.0,
                "conservative_input_per_million_tokens": 1.0,
                "conservative_output_per_million_tokens": 6.0,
                "free_grounded_prompts": free_grounded_prompts,
                "search_per_thousand_queries_after_free_allowance": 14.0,
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--evidence-cache", type=Path, required=True)
    parser.add_argument("--projected-question-count", type=int, default=7_759)
    parser.add_argument("--cost-cap-usd", type=float, default=150.0)
    parser.add_argument(
        "--free-grounded-prompts",
        type=int,
        default=1_500,
        help="Free grounded prompts used for projection (Gemini 3 project default: 1,500).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--search-review",
        type=Path,
        help="Optional manual review ledger for expected-but-unused Search.",
    )
    args = parser.parse_args(argv)
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    audit = audit_generation(
        sample["questions"],
        json.loads(args.cache.read_text(encoding="utf-8")),
        json.loads(args.evidence_cache.read_text(encoding="utf-8")),
        projected_question_count=args.projected_question_count,
        cost_cap_usd=args.cost_cap_usd,
        free_grounded_prompts=args.free_grounded_prompts,
        search_reviews=(
            json.loads(args.search_review.read_text(encoding="utf-8"))["reviews"]
            if args.search_review is not None
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, sort_keys=True))
    return 0 if audit["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
