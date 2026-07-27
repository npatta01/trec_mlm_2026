import json
from pathlib import Path

import pytest

from trec_mlm.pointwise_judge import (
    CandidateCase,
    DecisionV2,
    JudgeDecision,
    decision_to_dict,
    load_prompt_contract,
    normalized_decision_v2_schema,
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
VALID_V2_DECISION = {
    "label": "good",
    "correctness": 4,
    "completeness": 2,
    "relevance": 2,
    "error_flags": [],
    "reference_status": "sound",
    "confidence": "high",
    "rationale": (
        "The candidate directly gives the correct author, fully answers the "
        "single-fact question, and introduces no material error."
    ),
}


def test_v1_remains_default_and_v2_is_explicit() -> None:
    assert load_prompt_contract(PROTOCOL_DIR).prompt_path.name == "pointwise-v1.txt"
    assert (
        load_prompt_contract(PROTOCOL_DIR, version="v2").schema_path.name
        == "decision-v2.schema.json"
    )


def test_v2_rejects_coercion_and_extra_keys() -> None:
    coerced = dict(VALID_V2_DECISION, correctness="4")
    extra_key = dict(VALID_V2_DECISION, extra="forbidden")

    with pytest.raises(ValueError):
        parse_judge_decision(json.dumps(coerced), version="v2")
    with pytest.raises(ValueError):
        parse_judge_decision(json.dumps(extra_key), version="v2")


def test_v2_schema_snapshot_matches_model() -> None:
    committed = load_prompt_contract(PROTOCOL_DIR, version="v2").schema

    assert committed == normalized_decision_v2_schema()


def test_v2_schema_requires_unique_error_flags() -> None:
    schema = load_prompt_contract(PROTOCOL_DIR, version="v2").schema

    assert schema["properties"]["error_flags"]["uniqueItems"] is True


@pytest.mark.parametrize(
    "patch",
    [
        {"label": "good", "completeness": 1},
        {"label": "incorrect", "error_flags": []},
        {"label": "uncertain", "correctness": 1},
        {"error_flags": ["reference_conflict"], "reference_status": "sound"},
    ],
)
def test_v2_cross_field_invariants_reject(patch: dict[str, object]) -> None:
    value = VALID_V2_DECISION | patch

    with pytest.raises(ValueError):
        parse_judge_decision(json.dumps(value), version="v2")


def test_v2_invariant_error_is_concise_and_does_not_echo_input() -> None:
    private_rationale = "PRIVATE-RATIONALE-DO-NOT-ECHO " + "x" * 80
    value = {
        **VALID_V2_DECISION,
        "error_flags": ["reference_conflict"],
        "reference_status": "sound",
        "rationale": private_rationale,
    }

    with pytest.raises(ValueError) as caught:
        parse_judge_decision(json.dumps(value), version="v2")

    message = str(caught.value)
    assert message == (
        "v2 decision invariant violated: reference_conflict requires incomplete or "
        "suspect reference"
    )
    assert private_rationale not in message
    assert "input_value" not in message


def test_v2_extra_field_error_does_not_echo_arbitrary_key() -> None:
    private_key = "PRIVATE-FIELD-NAME-DO-NOT-ECHO"
    value = {**VALID_V2_DECISION, private_key: "private-value"}

    with pytest.raises(ValueError) as caught:
        parse_judge_decision(json.dumps(value), version="v2")

    assert str(caught.value) == (
        "v2 decision invalid at unknown field: Extra inputs are not permitted"
    )
    assert private_key not in str(caught.value)


def test_v2_duplicate_key_error_does_not_echo_arbitrary_key() -> None:
    private_key = "PRIVATE-DUPLICATE-KEY-DO-NOT-ECHO"
    raw = f'{{"{private_key}": 1, "{private_key}": 2}}'

    with pytest.raises(ValueError) as caught:
        parse_judge_decision(raw, version="v2")

    assert str(caught.value) == "decision contains a duplicate JSON key"
    assert private_key not in str(caught.value)


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        (
            {"correctness": "4"},
            "v2 decision invalid at correctness: Input should be a valid integer",
        ),
        (
            {"error_flags": [1]},
            (
                "v2 decision invalid at error_flags.0: Input should be "
                "'entity_error', 'number_error', 'date_error', 'negation_error', "
                "'unsupported_specificity', 'refusal', 'ambiguity' or "
                "'reference_conflict'"
            ),
        ),
    ],
)
def test_v2_error_location_allows_only_schema_fields_and_list_indices(
    patch: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=".*") as caught:
        parse_judge_decision(json.dumps(VALID_V2_DECISION | patch), version="v2")

    assert str(caught.value) == message


@pytest.mark.parametrize(
    ("patch", "invariant"),
    [
        ({"label": "partial"}, "good iff its three scores and flags match"),
        (
            {"label": "partial", "error_flags": ["entity_error"]},
            "correctness 4 excludes factual-error flags",
        ),
        (
            {"label": "incorrect", "correctness": 3, "error_flags": ["entity_error"]},
            "correctness 3 requires partial",
        ),
        (
            {"label": "partial", "correctness": 1, "error_flags": ["entity_error"]},
            "correctness at most 1 requires incorrect or uncertain",
        ),
        (
            {
                "label": "incorrect",
                "correctness": 0,
                "completeness": 1,
                "relevance": 0,
                "error_flags": ["refusal"],
            },
            "refusal fixes the non-uncertain scores and label",
        ),
        (
            {
                "label": "uncertain",
                "correctness": 0,
                "completeness": 0,
                "error_flags": [],
                "reference_status": "sound",
            },
            "uncertain needs ambiguity or an unsettled reference",
        ),
        (
            {
                "label": "uncertain",
                "correctness": 0,
                "completeness": 0,
                "relevance": 1,
                "error_flags": ["refusal", "ambiguity"],
                "reference_status": "uncertain",
            },
            "refusal pins every score to zero even when the item is uncertain",
        ),
        (
            {
                "label": "incorrect",
                "correctness": 0,
                "completeness": 0,
                "relevance": 1,
                "error_flags": ["ambiguity"],
            },
            "ambiguity is reserved for an unjudgeable item",
        ),
    ],
)
def test_v2_rejects_each_remaining_cross_field_invariant(
    patch: dict[str, object], invariant: str
) -> None:
    value = VALID_V2_DECISION | patch

    with pytest.raises(ValueError):
        parse_judge_decision(json.dumps(value), version="v2")


@pytest.mark.parametrize(
    "patch",
    [
        {"correctness": True},
        {"correctness": 4.0},
        {"error_flags": ["entity_error", "entity_error"]},
        {
            "error_flags": [
                "entity_error",
                "number_error",
                "date_error",
                "negation_error",
            ]
        },
        {"rationale": "short"},
        {"rationale": "x" * 601},
    ],
)
def test_v2_enforces_strict_field_constraints(patch: dict[str, object]) -> None:
    value = VALID_V2_DECISION | patch

    with pytest.raises(ValueError):
        parse_judge_decision(json.dumps(value), version="v2")


def test_v2_validates_to_the_existing_runner_decision_type() -> None:
    decision = parse_judge_decision(json.dumps(VALID_V2_DECISION), version="v2")

    assert isinstance(decision, JudgeDecision)
    assert decision_to_dict(decision) == VALID_V2_DECISION
    assert DecisionV2.model_validate_json(json.dumps(VALID_V2_DECISION)).label == "good"


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
