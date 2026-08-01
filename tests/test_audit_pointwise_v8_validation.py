import pytest

from scripts.audit_pointwise_v8_validation import (
    attach_semantic_review,
    build_audit,
    render_markdown,
)


def _record(
    ordinal: int, correctness: str, coverage: str, rationale: str
) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "record_id": f"record-{ordinal}",
        "question_id": "question-1",
        "candidate_id": f"candidate-{ordinal}",
        "reference_tier": "search_grounded",
        "question": "Who wrote Hamlet?",
        "provisional_reference": "William Shakespeare wrote Hamlet.",
        "candidate_response": "Shakespeare wrote it.",
        "normalized": {
            "decision": {
                "correctness": correctness,
                "coverage": coverage,
                "rationale": rationale,
                "acceptable": correctness == "correct",
                "needs_review": correctness in {"mixed", "unknown"},
            },
            "parse_status": "clean",
            "repair_codes": [],
        },
    }


def _run(repetition: str, records: list[dict[str, object]], cost: float) -> dict[str, object]:
    records = [dict(record) for record in records]
    if records:
        records[0]["estimated_cost_usd"] = cost
    return {
        "contract_version": "v8",
        "inventory_sha256": "a" * 64,
        "grounding_sha256": "b" * 64,
        "prompt_sha256": "c" * 64,
        "schema_sha256": "d" * 64,
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "provider": "DeepInfra",
        "transport": "deepinfra-direct",
        "fallbacks": False,
        "reasoning_effort": None,
        "temperature": 0,
        "max_tokens": 8192,
        "prompt_cache_key": "trec-mlm-pointwise-v8-reference-grounded",
        "selected_ordinals": [record["ordinal"] for record in records],
        "repetition": repetition,
        "records": records,
        "summary": {
            "request_count": len(records),
            "clean": len(records),
            "coerced": 0,
            "unscorable": 0,
            "provider_errors": 0,
            "retried_requests": 0,
            "estimated_cost_usd": cost,
        },
    }


def test_audit_counts_literal_agreements_distributions_cost_and_disagreements() -> None:
    a = _run(
        "a",
        [
            _record(1, "correct", "complete", "Supported."),
            _record(2, "unknown", "partial", "Reference is silent."),
        ],
        0.01,
    )
    b = _run(
        "b",
        [
            _record(1, "correct", "complete", "Supported again."),
            _record(2, "wrong", "partial", "Contradicted."),
        ],
        0.02,
    )

    audit = build_audit(a, b)

    assert audit["case_count"] == 2
    assert audit["correctness_agreement"] == {"count": 1, "rate": 0.5}
    assert audit["coverage_agreement"] == {"count": 2, "rate": 1.0}
    assert audit["exact_agreement"] == {"count": 1, "rate": 0.5}
    assert audit["correctness_distribution"] == {
        "a": {"correct": 1, "unknown": 1},
        "b": {"correct": 1, "wrong": 1},
    }
    assert audit["total_estimated_cost_usd"] == 0.03
    assert audit["transport_quality"]["a"]["clean"] == 2
    assert len(audit["disagreements"]) == 1
    disagreement = audit["disagreements"][0]
    assert disagreement["ordinal"] == 2
    assert disagreement["question"] == "Who wrote Hamlet?"
    assert disagreement["provisional_reference"] == "William Shakespeare wrote Hamlet."
    assert disagreement["candidate_response"] == "Shakespeare wrote it."
    assert disagreement["a"]["correctness"] == "unknown"
    assert disagreement["b"]["correctness"] == "wrong"


def test_audit_markdown_exposes_metrics_and_full_disagreement_evidence() -> None:
    a = _run("a", [_record(1, "unknown", "partial", "Silent.")], 0.01)
    b = _run("b", [_record(1, "wrong", "missing", "Wrong.")], 0.02)

    markdown = render_markdown(build_audit(a, b))

    assert "Correctness agreement | 0/1 (0.0%)" in markdown
    assert "## Disagreement 1: ordinal 1" in markdown
    assert "### Question\n\nWho wrote Hamlet?" in markdown
    assert "### Provisional reference" in markdown
    assert "### Candidate response" in markdown
    assert "A: `unknown / partial`" in markdown
    assert "B: `wrong / missing`" in markdown


def test_semantic_review_requires_every_disagreement_and_all_stable_labels() -> None:
    a = _run(
        "a",
        [
            _record(1, "unknown", "partial", "Silent."),
            _record(2, "correct", "complete", "Supported."),
            _record(3, "mixed", "partial", "Both."),
            _record(4, "wrong", "missing", "Wrong."),
            _record(5, "unknown", "partial", "Silent."),
        ],
        0.01,
    )
    b = _run(
        "b",
        [
            _record(1, "wrong", "missing", "Wrong."),
            _record(2, "correct", "complete", "Supported."),
            _record(3, "mixed", "partial", "Both."),
            _record(4, "wrong", "missing", "Wrong."),
            _record(5, "unknown", "partial", "Silent."),
        ],
        0.02,
    )
    audit = build_audit(a, b)
    review = {
        "reviewer": "Codex manual audit",
        "disagreement_assessments": {
            "1": {
                "assessment": "A is more defensible because the reference is silent.",
                "preferred": "a",
            }
        },
        "stable_examples": {
            "correct": [2],
            "mixed": [3],
            "wrong": [4],
            "unknown": [5],
        },
        "v7_correctness_baseline": {"count": 41, "total": 50},
        "scale_recommendation": "Do not scale until a balanced human key passes.",
    }

    reviewed = attach_semantic_review(audit, review)
    markdown = render_markdown(reviewed)

    assert reviewed["semantic_review"] == review
    assert "## Manual semantic review" in markdown
    assert "A is more defensible" in markdown
    assert "v7 correctness baseline | 41/50 (82.0%)" in markdown
    assert "Stable `correct` examples: 2" in markdown

    incomplete = dict(review)
    incomplete["disagreement_assessments"] = {}
    with pytest.raises(ValueError, match="exactly cover"):
        attach_semantic_review(audit, incomplete)

    fabricated = dict(review)
    fabricated["stable_examples"] = review["stable_examples"] | {"correct": [99]}
    with pytest.raises(ValueError, match="not stable"):
        attach_semantic_review(audit, fabricated)

    wrong_baseline = dict(review)
    wrong_baseline["v7_correctness_baseline"] = {"count": 0, "total": 0}
    with pytest.raises(ValueError, match="41/50"):
        attach_semantic_review(audit, wrong_baseline)


def test_audit_rejects_summary_that_disagrees_with_records() -> None:
    a = _run("a", [_record(1, "correct", "complete", "Supported.")], 0.01)
    b = _run("b", [_record(1, "correct", "complete", "Supported.")], 0.01)
    a["summary"]["clean"] = 0

    with pytest.raises(ValueError, match="summary does not match records"):
        build_audit(a, b)
