from __future__ import annotations

from dataclasses import asdict
import hashlib
import json

import pytest

from scripts.build_pointwise_judge_100_comparison import build_comparison
from trec_mlm.pointwise_judge import CandidateCase, parse_judge_decision
from trec_mlm.pointwise_runner import FROZEN_DECISION_SCHEMA_SHA256


CASES = [
    CandidateCase(
        question_id=f"question-{index}",
        response_id=f"response-{index}",
        question=f"Question {index}?",
        reference_answer=f"Reference {index}.",
        candidate_answer=f"Answer {index}.",
        strata=(reason, "shared") if index < 3 else (reason,),
        selection_reason=reason,
    )
    for index, reason in enumerate(("reason-a", "reason-b", "reason-c", "reason-d"), 1)
]


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _decision(label: str) -> dict[str, object]:
    score = {"incorrect": 0, "partial": 2, "good": 4}[label]
    return {
        "label": label,
        "correctness": score,
        "completeness": 2,
        "relevance": 2,
        "error_flags": [],
        "reference_status": "sound",
        "confidence": "medium",
        "rationale": f"Private rationale for the {label} judgment.",
    }


def _attempt(
    model_id: str,
    backend_model_id: str,
    raw: str,
    parsed: dict[str, object] | None,
    total_tokens: int,
) -> dict[str, object]:
    value: dict[str, object] = {
        "raw": raw,
        "response_body": {
            "model": backend_model_id,
            "choices": [{"message": {"content": raw}}],
            "usage": {
                "prompt_tokens": total_tokens - 1,
                "completion_tokens": 1,
                "total_tokens": total_tokens,
            },
        },
        "logical_model_id": model_id,
        "backend_model_id": backend_model_id,
        "parsed": parsed,
    }
    if parsed is None:
        with pytest.raises(ValueError) as error:
            parse_judge_decision(raw)
        value["error"] = str(error.value)
    return value


def _record(
    case: CandidateCase,
    *,
    alias: str,
    model_id: str,
    receipt: str,
    label: str | None,
    elapsed: float,
    total_tokens: int,
) -> dict[str, object]:
    backend_model_id = f"/private/{alias}.gguf"
    identity = {
        "input_sha256": _json_hash(asdict(case)),
        "prompt_sha256": "1" * 64,
        "schema_sha256": FROZEN_DECISION_SCHEMA_SHA256,
        "model_alias": alias,
        "model_id": model_id,
        "runtime_receipt_sha256": receipt,
    }
    if label is not None:
        decision = _decision(label)
        attempts = [
            _attempt(
                model_id,
                backend_model_id,
                json.dumps(decision),
                decision,
                total_tokens,
            )
        ]
        failure = None
    else:
        first = _attempt(model_id, backend_model_id, "not json one", None, total_tokens)
        second = _attempt(
            model_id, backend_model_id, "not json two", None, total_tokens
        )
        attempts = [first, second]
        decision = None
        failure = {
            "type": "invalid_decision_after_repair",
            "message": second["error"],
        }
    return {
        **identity,
        "attempts": attempts,
        "decision": decision,
        "failure": failure,
        "elapsed_seconds": elapsed,
        "usage": {
            "prompt_tokens": (total_tokens - 1) * len(attempts),
            "completion_tokens": len(attempts),
            "total_tokens": total_tokens * len(attempts),
        },
    }


def _run(
    alias: str,
    model_id: str,
    labels: list[str | None],
    elapsed: list[float],
    tokens: list[int],
) -> dict[str, object]:
    receipt = hashlib.sha256(alias.encode()).hexdigest()
    return {
        "version": 1,
        "model_alias": alias,
        "model_id": model_id,
        "prompt_sha256": "1" * 64,
        "schema_sha256": FROZEN_DECISION_SCHEMA_SHA256,
        "runtime_receipt_sha256": receipt,
        "records": {
            case.response_id: _record(
                case,
                alias=alias,
                model_id=model_id,
                receipt=receipt,
                label=label,
                elapsed=seconds,
                total_tokens=token_count,
            )
            for case, label, seconds, token_count in zip(
                CASES, labels, elapsed, tokens, strict=True
            )
        },
    }


def _inputs() -> tuple[dict[str, object], dict[str, object]]:
    direct = _run(
        "private-direct-alias",
        "org/private-direct-model",
        ["incorrect", "partial", "good", None],
        [1.0, 2.0, 3.0, 4.0],
        [10, 20, 30, 40],
    )
    thinking = _run(
        "private-thinking-alias",
        "org/private-thinking-model",
        ["partial", "good", "incorrect", "good"],
        [10.0, 20.0, 30.0, 40.0],
        [11, 21, 31, 41],
    )
    return direct, thinking


def test_comparison_reports_direction_strata_reused_slice_costs_and_failures() -> None:
    direct, thinking = _inputs()

    artifacts = build_comparison(
        CASES,
        direct,
        thinking,
        reused_response_ids=[CASES[0].response_id],
        input_hashes={
            "manifest": "a" * 64,
            "direct_run": "b" * 64,
            "thinking_run": "c" * 64,
        },
        seed=2026,
    )
    comparison = artifacts["comparison"]

    assert comparison["agreement"] == {
        "paired_valid_decisions": 3,
        "exact_count": 0,
        "exact_rate": 0.0,
        "within_one_label_count": 2,
        "within_one_label_rate": 2 / 3,
        "nonadjacent_count": 1,
    }
    assert comparison["confusion_matrix"]["counts"] == [
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
    ]
    assert comparison["transitions"] == [
        {"direct_label": "incorrect", "thinking_label": "partial", "count": 1},
        {"direct_label": "partial", "thinking_label": "good", "count": 1},
        {"direct_label": "good", "thinking_label": "incorrect", "count": 1},
    ]
    assert comparison["per_reason"]["reason-a"]["transitions"] == {
        "incorrect_to_partial": 1
    }
    assert comparison["per_stratum"]["shared"]["case_count"] == 2
    assert comparison["per_stratum"]["shared"]["paired_valid_decisions"] == 2
    assert comparison["reused_slice"]["case_count"] == 1
    assert comparison["reused_slice"]["within_one_label_count"] == 1
    assert comparison["runs"]["direct"]["failed_records"] == 1
    assert comparison["runs"]["direct"]["valid_decisions"] == 3
    assert comparison["runs"]["direct"]["elapsed_seconds_total"] == 10.0
    assert comparison["runs"]["direct"]["usage"]["total_tokens"] == 140
    assert comparison["runs"]["thinking"]["elapsed_seconds_total"] == 100.0
    assert comparison["runs"]["thinking"]["usage"]["total_tokens"] == 104
    assert comparison["case_differences"][-1]["direct_status"] == "failure"
    assert comparison["case_differences"][-1]["thinking_status"] == "decision"
    assert "correct" not in comparison["headline"].casefold()


def test_review_packets_hide_mode_and_model_identity() -> None:
    direct, thinking = _inputs()

    artifacts = build_comparison(
        CASES,
        direct,
        thinking,
        reused_response_ids=[CASES[0].response_id],
        input_hashes={
            "manifest": "a" * 64,
            "direct_run": "b" * 64,
            "thinking_run": "c" * 64,
        },
        seed=2026,
    )
    packet_text = json.dumps(
        {
            "independent": artifacts["independent_review_packet"],
            "critique": artifacts["critique_review_packet"],
        },
        sort_keys=True,
    ).casefold()

    for forbidden in (
        "private-direct",
        "private-thinking",
        "org/private",
        "direct_run",
        "thinking_run",
        "response-",
        "question-",
        "model_alias",
        "model_id",
    ):
        assert forbidden not in packet_text
    assert set(artifacts["sealed_mode_aliases"]["judge_aliases"].values()) == {
        "direct",
        "thinking",
    }


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_coverage", "coverage"),
        ("wrong_input", "input"),
        ("wrong_prompt", "prompt"),
        ("wrong_schema", "schema"),
    ],
)
def test_comparison_fails_closed_on_coverage_or_binding_mismatch(
    mutation: str, message: str
) -> None:
    direct, thinking = _inputs()
    if mutation == "missing_coverage":
        del thinking["records"][CASES[-1].response_id]
    elif mutation == "wrong_input":
        thinking["records"][CASES[0].response_id]["input_sha256"] = "f" * 64
    elif mutation == "wrong_prompt":
        thinking["prompt_sha256"] = "2" * 64
    else:
        thinking["schema_sha256"] = "3" * 64

    with pytest.raises(ValueError, match=message):
        build_comparison(
            CASES,
            direct,
            thinking,
            reused_response_ids=[CASES[0].response_id],
            input_hashes={
                "manifest": "a" * 64,
                "direct_run": "b" * 64,
                "thinking_run": "c" * 64,
            },
            seed=2026,
        )
