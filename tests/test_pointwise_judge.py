import json
from pathlib import Path

import pytest

from trec_mlm.pointwise_judge import (
    CandidateCase,
    JudgeDecision,
    decision_to_dict,
    load_prompt_contract,
    parse_judge_decision,
    render_judge_messages,
)


PROTOCOL_DIR = Path(__file__).parents[1] / "protocols" / "pointwise_judge_bakeoff"
CONTRACT = load_prompt_contract(PROTOCOL_DIR)
CASE = CandidateCase(
    question_id="q1",
    response_id="r1",
    question="Who wrote Hamlet?",
    reference_answer="William Shakespeare wrote Hamlet.",
    candidate_answer="Hamlet was written by William Shakespeare.",
    strata=("high_overlap_control",),
    selection_reason="high_overlap_control",
)
VALID_DECISION = {
    "label": "good",
    "correctness": 4,
    "completeness": 2,
    "relevance": 2,
    "error_flags": [],
    "reference_status": "sound",
    "confidence": "high",
    "rationale": "The candidate directly gives the correct author without a material error.",
}


def test_parse_judge_decision_rejects_extra_keys() -> None:
    raw = json.dumps({**VALID_DECISION, "extra": 1})

    with pytest.raises(ValueError, match="exact keys"):
        parse_judge_decision(raw)


def test_parse_judge_decision_rejects_duplicate_object_keys() -> None:
    raw = json.dumps(VALID_DECISION).replace(
        '"label": "good"', '"label": "incorrect", "label": "good"', 1
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_judge_decision(raw)


def test_render_messages_keeps_reference_provisional() -> None:
    messages = render_judge_messages(CASE, CONTRACT)
    content = messages[-1]["content"]

    assert "PROVISIONAL REFERENCE" in content
    assert "may be incomplete or wrong" in content
    assert "CANDIDATE RESPONSE" in content


def test_render_messages_hides_diagnostic_selection_metadata() -> None:
    case = CandidateCase(
        question_id="q2",
        response_id="r2",
        question="Question?",
        reference_answer="Reference.",
        candidate_answer="Candidate.",
        strata=("low_overlap_error_proxy", "entity_mismatch_proxy"),
        selection_reason="entity_mismatch_proxy",
    )
    content = render_judge_messages(case, CONTRACT)[-1]["content"]

    assert "DIAGNOSTIC STRATA" not in content
    assert "SELECTION REASON" not in content
    assert "low_overlap_error_proxy" not in content
    assert "entity_mismatch_proxy" not in content


def test_load_prompt_contract_loads_frozen_prompt_and_schema() -> None:
    assert CONTRACT.prompt_path == PROTOCOL_DIR / "pointwise-v1.txt"
    assert CONTRACT.schema_path == PROTOCOL_DIR / "decision-v1.schema.json"
    assert CONTRACT.schema["additionalProperties"] is False
    assert CONTRACT.schema["properties"]["label"]["enum"] == [
        "good",
        "partial",
        "incorrect",
        "uncertain",
    ]


@pytest.mark.parametrize(
    "schema_text",
    [
        '{"title":"first","title":"second"}',
        '{"title":"Pointwise Judge Decision v1","maximum":NaN}',
    ],
)
def test_load_prompt_contract_rejects_ambiguous_or_nonfinite_schema_json(
    tmp_path: Path, schema_text: str
) -> None:
    (tmp_path / "pointwise-v1.txt").write_text("fixture", encoding="utf-8")
    (tmp_path / "decision-v1.schema.json").write_text(schema_text, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate|non-finite"):
        load_prompt_contract(tmp_path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("label", "excellent", "label"),
        ("correctness", True, "correctness"),
        ("correctness", 5, "correctness"),
        ("completeness", False, "completeness"),
        ("completeness", -1, "completeness"),
        ("relevance", True, "relevance"),
        ("relevance", 3, "relevance"),
        ("reference_status", "verified", "reference_status"),
        ("confidence", "certain", "confidence"),
    ],
)
def test_parse_judge_decision_rejects_invalid_scalars(
    field: str, value: object, message: str
) -> None:
    raw = {**VALID_DECISION, field: value}

    with pytest.raises(ValueError, match=message):
        parse_judge_decision(json.dumps(raw))


def test_parse_judge_decision_rejects_unknown_or_duplicate_flags() -> None:
    unknown = {**VALID_DECISION, "error_flags": ["made_up_error"]}
    duplicate = {**VALID_DECISION, "error_flags": ["entity_error", "entity_error"]}

    with pytest.raises(ValueError, match="error_flags"):
        parse_judge_decision(json.dumps(unknown))
    with pytest.raises(ValueError, match="duplicate"):
        parse_judge_decision(json.dumps(duplicate))


@pytest.mark.parametrize("rationale", ["", "word " * 81])
def test_parse_judge_decision_enforces_rationale_word_bound(rationale: str) -> None:
    with pytest.raises(ValueError, match="rationale"):
        parse_judge_decision(json.dumps({**VALID_DECISION, "rationale": rationale}))


@pytest.mark.parametrize(("correctness", "relevance"), [(3, 2), (4, 1)])
def test_parse_judge_decision_gates_good_label(
    correctness: int, relevance: int
) -> None:
    raw = {**VALID_DECISION, "correctness": correctness, "relevance": relevance}

    with pytest.raises(ValueError, match="good"):
        parse_judge_decision(json.dumps(raw))


def test_parse_judge_decision_returns_immutable_domain_record() -> None:
    decision = parse_judge_decision(
        json.dumps({**VALID_DECISION, "error_flags": ["entity_error"]})
    )

    assert decision == JudgeDecision(
        label="good",
        correctness=4,
        completeness=2,
        relevance=2,
        error_flags=("entity_error",),
        reference_status="sound",
        confidence="high",
        rationale=VALID_DECISION["rationale"],
    )
    assert decision_to_dict(decision) == {
        **VALID_DECISION,
        "error_flags": ["entity_error"],
    }


def test_render_repair_message_includes_raw_output_and_unchanged_schema() -> None:
    messages = render_judge_messages(CASE, CONTRACT, repair_raw="not valid json")

    assert len(messages) == 3
    assert messages[-1]["role"] == "user"
    assert "not valid json" in messages[-1]["content"]
    assert "only a corrected JSON object" in messages[-1]["content"]
    assert "unchanged schema" in messages[-1]["content"]
