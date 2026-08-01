#!/usr/bin/env python3
"""Compare two v8 validation repetitions and expose every disagreement."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_deepinfra_pointwise_batch import _write_private  # noqa: E402


_IDENTITY_KEYS = (
    "contract_version",
    "inventory_manifest_sha256",
    "inventory_sha256",
    "grounding_sha256",
    "prompt_sha256",
    "schema_sha256",
    "model",
    "provider",
    "transport",
    "fallbacks",
    "reasoning_effort",
    "temperature",
    "max_tokens",
    "prompt_cache_key",
    "selected_ordinals",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--semantic-review", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def _read_private_object(path: Path) -> dict[str, Any]:
    details = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(details.st_mode) or stat.S_IMODE(details.st_mode) != 0o600:
        raise ValueError(f"input must be a regular mode-0600 file: {path.name}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"input must contain an object: {path.name}")
    return value


def _write_private_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _decision(record: dict[str, Any]) -> dict[str, Any]:
    normalized = record.get("normalized")
    decision = normalized.get("decision") if type(normalized) is dict else None
    if type(decision) is not dict:
        raise ValueError(f"record {record.get('ordinal')} has no normalized decision")
    return decision


def _rate(count: int, total: int) -> dict[str, float | int]:
    return {"count": count, "rate": count / total if total else 0.0}


def _recomputed_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    value = {
        "request_count": len(records),
        "clean": 0,
        "coerced": 0,
        "unscorable": 0,
        "provider_errors": 0,
        "retried_requests": 0,
        "estimated_cost_usd": 0.0,
    }
    for record in records:
        if "provider_error" in record:
            value["provider_errors"] += 1
            continue
        normalized = record.get("normalized")
        status = normalized.get("parse_status") if type(normalized) is dict else None
        if status not in {"clean", "coerced", "unscorable"}:
            raise ValueError(f"record {record.get('ordinal')} has invalid parse status")
        value[status] += 1
        value["retried_requests"] += record.get("attempt_count") == 2
        value["estimated_cost_usd"] += float(record.get("estimated_cost_usd", 0.0))
    return value


def _validated_summary(
    supplied: dict[str, Any], records: list[dict[str, Any]], repetition: str
) -> dict[str, Any]:
    recomputed = _recomputed_summary(records)
    for key, value in recomputed.items():
        supplied_value = supplied.get(key)
        if key == "estimated_cost_usd":
            matches = abs(float(supplied_value or 0.0) - value) < 1e-12
        else:
            matches = supplied_value == value
        if not matches:
            raise ValueError(f"repetition {repetition} summary does not match records")
    return recomputed


def build_audit(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Build deterministic agreement metrics without claiming factual accuracy."""
    for key in _IDENTITY_KEYS:
        if a.get(key) != b.get(key):
            raise ValueError(f"repetition identity mismatch: {key}")
    if a.get("repetition") != "a" or b.get("repetition") != "b":
        raise ValueError("inputs must be repetitions a and b")
    records_a = a.get("records")
    records_b = b.get("records")
    if type(records_a) is not list or type(records_b) is not list:
        raise ValueError("both repetitions must contain records")
    if len(records_a) != len(records_b) or not records_a:
        raise ValueError("repetitions must contain the same nonzero record count")

    by_ordinal_a = {record.get("ordinal"): record for record in records_a}
    by_ordinal_b = {record.get("ordinal"): record for record in records_b}
    ordinals = a["selected_ordinals"]
    if (
        len(by_ordinal_a) != len(records_a)
        or len(by_ordinal_b) != len(records_b)
        or list(by_ordinal_a) != ordinals
        or list(by_ordinal_b) != ordinals
    ):
        raise ValueError("records do not exactly match selected ordinal order")

    correctness_agree = coverage_agree = exact_agree = 0
    correctness_a: Counter[str] = Counter()
    correctness_b: Counter[str] = Counter()
    coverage_a: Counter[str] = Counter()
    coverage_b: Counter[str] = Counter()
    tiers: Counter[str] = Counter()
    stable_correctness_ordinals: dict[str, list[int]] = {
        "correct": [],
        "mixed": [],
        "wrong": [],
        "unknown": [],
    }
    disagreements: list[dict[str, Any]] = []
    for ordinal in ordinals:
        record_a = by_ordinal_a[ordinal]
        record_b = by_ordinal_b[ordinal]
        if record_a.get("record_id") != record_b.get("record_id"):
            raise ValueError(f"record identity mismatch at ordinal {ordinal}")
        decision_a = _decision(record_a)
        decision_b = _decision(record_b)
        correctness_a[decision_a["correctness"]] += 1
        correctness_b[decision_b["correctness"]] += 1
        coverage_a[decision_a["coverage"]] += 1
        coverage_b[decision_b["coverage"]] += 1
        tiers[str(record_a.get("reference_tier"))] += 1
        same_correctness = decision_a["correctness"] == decision_b["correctness"]
        same_coverage = decision_a["coverage"] == decision_b["coverage"]
        correctness_agree += same_correctness
        if same_correctness:
            stable_correctness_ordinals[decision_a["correctness"]].append(ordinal)
        coverage_agree += same_coverage
        exact_agree += same_correctness and same_coverage
        if not (same_correctness and same_coverage):
            disagreements.append(
                {
                    "ordinal": ordinal,
                    "record_id": record_a["record_id"],
                    "question_id": record_a["question_id"],
                    "candidate_id": record_a["candidate_id"],
                    "reference_tier": record_a["reference_tier"],
                    "question": record_a["question"],
                    "provisional_reference": record_a["provisional_reference"],
                    "candidate_response": record_a["candidate_response"],
                    "a": {
                        key: decision_a[key]
                        for key in ("correctness", "coverage", "rationale")
                    },
                    "b": {
                        key: decision_b[key]
                        for key in ("correctness", "coverage", "rationale")
                    },
                }
            )

    total = len(ordinals)
    summary_a = a.get("summary")
    summary_b = b.get("summary")
    if type(summary_a) is not dict or type(summary_b) is not dict:
        raise ValueError("both repetitions must contain summaries")
    summary_a = _validated_summary(summary_a, records_a, "a")
    summary_b = _validated_summary(summary_b, records_b, "b")
    return {
        "contract_version": a["contract_version"],
        "prompt_sha256": a["prompt_sha256"],
        "schema_sha256": a["schema_sha256"],
        "inventory_sha256": a["inventory_sha256"],
        "grounding_sha256": a["grounding_sha256"],
        "model": a["model"],
        "reasoning_effort": a["reasoning_effort"],
        "case_count": total,
        "correctness_agreement": _rate(correctness_agree, total),
        "coverage_agreement": _rate(coverage_agree, total),
        "exact_agreement": _rate(exact_agree, total),
        "correctness_distribution": {
            "a": dict(correctness_a),
            "b": dict(correctness_b),
        },
        "coverage_distribution": {
            "a": dict(coverage_a),
            "b": dict(coverage_b),
        },
        "reference_tier_distribution": dict(tiers),
        "stable_correctness_ordinals": stable_correctness_ordinals,
        "transport_quality": {"a": summary_a, "b": summary_b},
        "total_estimated_cost_usd": round(
            float(summary_a.get("estimated_cost_usd", 0.0))
            + float(summary_b.get("estimated_cost_usd", 0.0)),
            12,
        ),
        "disagreements": disagreements,
        "interpretation": (
            "Repeat agreement measures consistency, not factual accuracy. "
            "Correctness means alignment with the provisional grounded reference."
        ),
    }


def attach_semantic_review(
    audit: dict[str, Any], review: dict[str, Any]
) -> dict[str, Any]:
    """Validate and attach a manual review without changing deterministic metrics."""
    required = {
        "reviewer",
        "disagreement_assessments",
        "stable_examples",
        "v7_correctness_baseline",
        "scale_recommendation",
    }
    if set(review) != required:
        raise ValueError("semantic review shape is invalid")
    expected = {str(item["ordinal"]) for item in audit["disagreements"]}
    assessments = review["disagreement_assessments"]
    if type(assessments) is not dict or set(assessments) != expected:
        raise ValueError("semantic assessments must exactly cover disagreements")
    for ordinal, assessment in assessments.items():
        if (
            type(assessment) is not dict
            or set(assessment) != {"assessment", "preferred"}
            or type(assessment["assessment"]) is not str
            or not assessment["assessment"].strip()
            or assessment["preferred"] not in {"a", "b", "neither", "both"}
        ):
            raise ValueError(f"semantic assessment is invalid for {ordinal}")
    examples = review["stable_examples"]
    if type(examples) is not dict or set(examples) != {
        "correct",
        "mixed",
        "wrong",
        "unknown",
    }:
        raise ValueError("stable examples must cover all correctness labels")
    if any(
        type(values) is not list
        or not values
        or any(type(value) is not int for value in values)
        for values in examples.values()
    ):
        raise ValueError("stable example ordinals are invalid")
    stable = audit.get("stable_correctness_ordinals")
    if type(stable) is not dict:
        raise ValueError("audit stable-label index is missing")
    for label, ordinals in examples.items():
        actual = stable.get(label)
        if type(actual) is not list or not set(ordinals).issubset(actual):
            raise ValueError(f"semantic examples for {label} are not stable")
    baseline = review["v7_correctness_baseline"]
    if (
        type(baseline) is not dict
        or set(baseline) != {"count", "total"}
        or type(baseline["count"]) is not int
        or type(baseline["total"]) is not int
        or baseline != {"count": 41, "total": 50}
    ):
        raise ValueError("v7 correctness baseline must be the verified 41/50")
    if (
        type(review["reviewer"]) is not str
        or not review["reviewer"].strip()
        or type(review["scale_recommendation"]) is not str
        or not review["scale_recommendation"].strip()
    ):
        raise ValueError("semantic review text fields are invalid")
    return {**audit, "semantic_review": review}


def _pct(metric: dict[str, Any], total: int) -> str:
    return f"{metric['count']}/{total} ({metric['rate']:.1%})"


def render_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# DeepSeek v8 reference-grounded validation audit",
        "",
        "## Run summary",
        "",
        "| Check | Result |",
        "|---|---:|",
        f"| Cases per repetition | {audit['case_count']} |",
        f"| Correctness agreement | {_pct(audit['correctness_agreement'], audit['case_count'])} |",
        f"| Coverage agreement | {_pct(audit['coverage_agreement'], audit['case_count'])} |",
        f"| Exact agreement | {_pct(audit['exact_agreement'], audit['case_count'])} |",
        f"| Total estimated inference cost | ${audit['total_estimated_cost_usd']:.6f} |",
        "",
        audit["interpretation"],
        "",
        "## Distributions",
        "",
        f"- Correctness A: `{json.dumps(audit['correctness_distribution']['a'], sort_keys=True)}`",
        f"- Correctness B: `{json.dumps(audit['correctness_distribution']['b'], sort_keys=True)}`",
        f"- Coverage A: `{json.dumps(audit['coverage_distribution']['a'], sort_keys=True)}`",
        f"- Coverage B: `{json.dumps(audit['coverage_distribution']['b'], sort_keys=True)}`",
        f"- Reference tiers: `{json.dumps(audit['reference_tier_distribution'], sort_keys=True)}`",
        "",
        "## Disagreements",
    ]
    if not audit["disagreements"]:
        lines.extend(["", "None."])
    for index, item in enumerate(audit["disagreements"], start=1):
        lines.extend(
            [
                "",
                f"## Disagreement {index}: ordinal {item['ordinal']}",
                "",
                f"Question ID: `{item['question_id']}` · candidate: `{item['candidate_id']}` · reference tier: `{item['reference_tier']}`",
                "",
                "### Question",
                "",
                item["question"],
                "",
                "### Provisional reference",
                "",
                item["provisional_reference"],
                "",
                "### Candidate response",
                "",
                item["candidate_response"],
                "",
                "### Judgments",
                "",
                f"A: `{item['a']['correctness']} / {item['a']['coverage']}` — {item['a']['rationale']}",
                "",
                f"B: `{item['b']['correctness']} / {item['b']['coverage']}` — {item['b']['rationale']}",
            ]
        )
    review = audit.get("semantic_review")
    if type(review) is dict:
        baseline = review["v7_correctness_baseline"]
        lines.extend(
            [
                "",
                "## Manual semantic review",
                "",
                f"Reviewer: {review['reviewer']}",
                "",
                f"- v7 correctness baseline | {baseline['count']}/{baseline['total']} ({baseline['count'] / baseline['total']:.1%})",
            ]
        )
        for label, ordinals in review["stable_examples"].items():
            lines.append(
                f"- Stable `{label}` examples: {', '.join(str(value) for value in ordinals)}"
            )
        lines.extend(["", "### Disagreement assessments"])
        for ordinal, assessment in review["disagreement_assessments"].items():
            lines.extend(
                [
                    "",
                    f"- Ordinal {ordinal} — preferred `{assessment['preferred']}`: {assessment['assessment']}",
                ]
            )
        lines.extend(
            [
                "",
                "### Scale recommendation",
                "",
                review["scale_recommendation"],
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    audit = build_audit(_read_private_object(args.a), _read_private_object(args.b))
    if args.semantic_review is not None:
        audit = attach_semantic_review(
            audit, _read_private_object(args.semantic_review)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.chmod(0o700)
    _write_private(args.output_dir / "audit.json", audit)
    _write_private_text(args.output_dir / "audit.md", render_markdown(audit))
    print(
        json.dumps(
            {
                "case_count": audit["case_count"],
                "correctness_agreement": audit["correctness_agreement"],
                "coverage_agreement": audit["coverage_agreement"],
                "exact_agreement": audit["exact_agreement"],
                "disagreement_count": len(audit["disagreements"]),
                "total_estimated_cost_usd": audit["total_estimated_cost_usd"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
