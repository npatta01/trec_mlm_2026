import json
from pathlib import Path

import pytest

from trec_mlm.pointwise_judge import (
    CandidateCase,
    DecisionV10,
    DecisionV11,
    DecisionV8,
    DecisionV9,
    DecisionV4,
    DecisionV5,
    DecisionV3,
    DecisionV2,
    JudgeDecision,
    decision_to_dict,
    load_prompt_contract,
    normalized_decision_v2_schema,
    normalized_decision_v3_schema,
    normalized_decision_v4_schema,
    normalized_decision_v5_schema,
    normalized_decision_v8_schema,
    normalized_decision_v9_schema,
    normalized_decision_v10_schema,
    normalized_decision_v11_schema,
    normalize_judge_decision_v3,
    normalize_judge_decision_v4,
    normalize_judge_decision_v5,
    normalize_judge_decision_v8,
    normalize_judge_decision_v9,
    normalize_judge_decision_v10,
    normalize_judge_decision_v11,
    derive_v11_label,
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


def test_v3_contract_is_compact_and_explains_score_anchors() -> None:
    contract = load_prompt_contract(PROTOCOL_DIR, version="v3")
    schema = contract.schema

    assert contract.prompt_path.name == "pointwise-v3.txt"
    assert contract.schema_path.name == "decision-v3.schema.json"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "correctness",
        "completeness",
        "relevance",
        "error_flags",
        "reference_status",
        "uncertain",
        "rationale",
    }
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["properties"]["rationale"]["maxLength"] == 240
    assert schema["properties"]["uncertain"]["type"] == "boolean"
    for anchor in (
        "correctness: 0 means",
        "correctness: 1 means",
        "correctness: 2 means",
        "correctness: 3 means",
        "correctness: 4 means",
        "completeness: 0 means",
        "completeness: 1 means",
        "completeness: 2 means",
        "relevance: 0 means",
        "relevance: 1 means",
        "relevance: 2 means",
    ):
        assert anchor in contract.prompt
    assert "GENERIC EXAMPLES" in contract.prompt
    assert "incomplete is not refusal" in contract.prompt
    assert "uncertain is not irrelevant" in contract.prompt
    assert "mixed claims cannot be correctness 0" in contract.prompt


def test_v3_schema_has_unique_error_flags_and_no_model_label() -> None:
    schema = load_prompt_contract(PROTOCOL_DIR, version="v3").schema

    assert "label" not in schema["properties"]
    assert "confidence" not in schema["properties"]
    assert schema["properties"]["error_flags"]["uniqueItems"] is True


def test_v3_schema_snapshot_matches_pydantic_model() -> None:
    committed = load_prompt_contract(PROTOCOL_DIR, version="v3").schema

    assert committed == normalized_decision_v3_schema()


def test_v3_clean_decision_derives_label_and_does_not_retain_raw() -> None:
    value = {
        "correctness": 4,
        "completeness": 2,
        "relevance": 2,
        "error_flags": [],
        "reference_status": "sound",
        "uncertain": False,
        "rationale": "The answer directly satisfies the question without a material error.",
    }

    result = normalize_judge_decision_v3(json.dumps(value))

    assert result == {
        "decision": value,
        "label": "good",
        "parse_status": "clean",
        "repair_codes": [],
    }


def test_v3_repairs_common_shape_errors_once() -> None:
    value = {
        "correctness": "9",
        "completeness": 2.0,
        "relevance": "2",
        "error_flags": ["entity_error", "entity_error", "new_flag", "refusal"],
        "reference_status": "bogus",
        "uncertain": "false",
        "rationale": "x" * 300,
        "extra": "drop me",
    }

    result = normalize_judge_decision_v3("prefix " + json.dumps(value))

    assert result["parse_status"] == "coerced"
    assert result["label"] == "incorrect"
    assert result["decision"] == {
        "correctness": 0,
        "completeness": 0,
        "relevance": 2,
        "error_flags": ["entity_error", "refusal"],
        "reference_status": "uncertain",
        "uncertain": False,
        "rationale": "x" * 240,
    }
    assert "raw" in result
    assert "clamp_correctness" in result["repair_codes"]
    assert "enforce_refusal_scores" in result["repair_codes"]


def test_v3_missing_score_is_unscorable_and_retains_raw() -> None:
    raw = json.dumps(
        {
            "correctness": 4,
            "completeness": 2,
            "error_flags": [],
            "reference_status": "sound",
            "uncertain": False,
            "rationale": "",
        }
    )

    result = normalize_judge_decision_v3(raw)

    assert result["parse_status"] == "unscorable"
    assert result["label"] == "uncertain"
    assert result["decision"]["correctness"] is None
    assert result["raw"] == raw


def test_v3_pydantic_model_remains_strict() -> None:
    value = {
        "correctness": 4,
        "completeness": 2,
        "relevance": 2,
        "error_flags": [],
        "reference_status": "sound",
        "uncertain": False,
        "rationale": "",
    }

    assert DecisionV3.model_validate(value).correctness == 4
    with pytest.raises(ValueError):
        DecisionV3.model_validate(value | {"correctness": "4"})


def test_v4_contract_is_small_and_categorical() -> None:
    contract = load_prompt_contract(PROTOCOL_DIR, version="v4")
    schema = contract.schema

    assert contract.prompt_path.name == "pointwise-v4.txt"
    assert contract.schema_path.name == "decision-v4.schema.json"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {
        "verdict",
        "correctness",
        "completeness",
        "relevance",
        "reference_status",
        "rationale",
    }
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["properties"]["rationale"]["maxLength"] == 240
    assert "correctness 0" not in contract.prompt
    assert "A partial," in contract.prompt
    assert "hedged" in contract.prompt


def test_v4_schema_snapshot_matches_pydantic_model() -> None:
    committed = load_prompt_contract(PROTOCOL_DIR, version="v4").schema

    assert committed == normalized_decision_v4_schema()


def test_v4_clean_decision_is_compact_and_strict() -> None:
    value = {
        "verdict": "yes",
        "correctness": "correct",
        "completeness": "complete",
        "relevance": "relevant",
        "reference_status": "sound",
        "rationale": "The response is correct, complete, and directly answers the question.",
    }

    result = normalize_judge_decision_v4(json.dumps(value))

    assert result == {
        "decision": value,
        "parse_status": "clean",
        "repair_codes": [],
    }
    assert DecisionV4.model_validate(value).verdict == "yes"


def test_v4_uncertain_dimensions_are_explicit() -> None:
    value = {
        "verdict": "uncertain",
        "correctness": "unknown",
        "completeness": "unknown",
        "relevance": "relevant",
        "reference_status": "uncertain",
        "rationale": "The question lacks the information needed to verify the answer.",
    }

    assert normalize_judge_decision_v4(json.dumps(value))["decision"] == value


def test_v4_repairs_shape_errors_without_retry() -> None:
    value = {
        "verdict": "YES",
        "correctness": "CORRECT",
        "completeness": "COMPLETE",
        "relevance": "RELEVANT",
        "reference_status": "SOUND",
        "rationale": "x" * 300,
        "extra": "drop me",
    }

    result = normalize_judge_decision_v4("prefix " + json.dumps(value))

    assert result["parse_status"] == "coerced"
    assert result["decision"] == {
        "verdict": "yes",
        "correctness": "correct",
        "completeness": "complete",
        "relevance": "relevant",
        "reference_status": "sound",
        "rationale": "x" * 240,
    }
    assert "raw" in result


def test_v5_contract_makes_correctness_primary_and_derives_acceptance() -> None:
    contract = load_prompt_contract(PROTOCOL_DIR, version="v5")
    schema = contract.schema

    assert set(schema["required"]) == {
        "correctness",
        "completeness",
        "relevance",
        "reference_status",
        "rationale",
    }
    assert "verdict" not in schema["properties"]
    assert "Omission lowers completeness, not correctness" in contract.prompt

    result = normalize_judge_decision_v5(
        json.dumps(
            {
                "correctness": "correct",
                "completeness": "partial",
                "relevance": "relevant",
                "reference_status": "sound",
                "rationale": "The stated claim is correct, but requested details are omitted.",
            }
        )
    )

    assert result["parse_status"] == "clean"
    assert result["decision"]["verdict"] == "partial"
    assert result["decision"]["acceptable"] is True
    assert result["decision"]["needs_review"] is False


def test_v5_mixed_claims_are_partial_but_not_acceptable() -> None:
    result = normalize_judge_decision_v5(
        json.dumps(
            {
                "correctness": "mixed",
                "completeness": "partial",
                "relevance": "relevant",
                "reference_status": "sound",
                "rationale": "The response includes useful true content but also a material false claim.",
            }
        )
    )

    assert result["decision"]["verdict"] == "partial"
    assert result["decision"]["acceptable"] is False
    assert result["decision"]["needs_review"] is True


def test_v5_schema_snapshot_and_model_are_strict() -> None:
    committed = load_prompt_contract(PROTOCOL_DIR, version="v5").schema

    assert committed == normalized_decision_v5_schema()
    value = {
        "correctness": "correct",
        "completeness": "complete",
        "relevance": "relevant",
        "reference_status": "sound",
        "rationale": "The answer is fully supported and covers the requested information.",
    }
    assert DecisionV5.model_validate(value).correctness == "correct"
    with pytest.raises(ValueError):
        DecisionV5.model_validate(value | {"verdict": "yes"})


def test_v6_prompt_makes_claim_check_and_partial_attempt_explicit() -> None:
    prompt = load_prompt_contract(PROTOCOL_DIR, version="v6").prompt

    assert "Check each material claim or list item independently" in prompt
    assert "is not a refusal" in prompt
    assert "Do not let the number of false claims erase useful true claims" in prompt


def test_v8_contract_has_only_reference_grounded_decision_fields() -> None:
    contract = load_prompt_contract(PROTOCOL_DIR, version="v8")
    schema = contract.schema

    assert contract.prompt_path.name == "pointwise-v8.txt"
    assert contract.schema_path.name == "decision-v8.schema.json"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"correctness", "coverage", "rationale"}
    assert set(schema["properties"]) == set(schema["required"])
    assert schema["properties"]["correctness"]["enum"] == [
        "correct",
        "mixed",
        "wrong",
        "unknown",
    ]
    assert schema["properties"]["coverage"]["enum"] == [
        "complete",
        "partial",
        "missing",
        "unknown",
    ]
    assert schema["properties"]["rationale"]["maxLength"] == 240
    assert "Omission does not lower correctness" in contract.prompt
    assert "Reference silence is unknown, not wrong" in contract.prompt
    assert "A short direct answer can be complete" in contract.prompt


def test_v8_schema_snapshot_matches_strict_pydantic_model() -> None:
    committed = load_prompt_contract(PROTOCOL_DIR, version="v8").schema
    value = {
        "correctness": "correct",
        "coverage": "partial",
        "rationale": "Every stated claim is supported, although some requested parts are omitted.",
    }

    assert committed == normalized_decision_v8_schema()
    assert DecisionV8.model_validate(value).coverage == "partial"
    with pytest.raises(ValueError):
        DecisionV8.model_validate(value | {"reference_status": "sound"})


def test_v8_partial_but_correct_is_acceptable() -> None:
    value = {
        "correctness": "correct",
        "coverage": "partial",
        "rationale": "Every stated claim is supported, although some requested parts are omitted.",
    }

    result = normalize_judge_decision_v8(json.dumps(value))

    assert result == {
        "decision": value | {"acceptable": True, "needs_review": False},
        "parse_status": "clean",
        "repair_codes": [],
    }


@pytest.mark.parametrize("correctness", ["mixed", "unknown"])
def test_v8_mixed_or_reference_silent_results_need_review(correctness: str) -> None:
    result = normalize_judge_decision_v8(
        json.dumps(
            {
                "correctness": correctness,
                "coverage": "partial",
                "rationale": "The reference does not settle every material claim.",
            }
        )
    )

    assert result["decision"]["acceptable"] is False
    assert result["decision"]["needs_review"] is True


def test_v8_repairs_shape_only_and_records_every_mutation() -> None:
    raw = "result: " + json.dumps(
        {
            "correctness": " CORRECT ",
            "coverage": "Partial",
            "rationale": "x" * 300,
            "reference_status": "sound",
        }
    )

    result = normalize_judge_decision_v8(raw)

    assert result["decision"] == {
        "correctness": "correct",
        "coverage": "partial",
        "rationale": "x" * 240,
        "acceptable": True,
        "needs_review": False,
    }
    assert result["parse_status"] == "coerced"
    assert result["repair_codes"] == [
        "extract_json_object",
        "drop_unknown_keys",
        "normalize_correctness",
        "normalize_coverage",
        "truncate_rationale",
    ]
    assert result["raw"] == raw


def test_v8_missing_required_enum_is_unscorable_not_a_judgment() -> None:
    raw = json.dumps({"coverage": "partial", "rationale": "No correctness label."})

    result = normalize_judge_decision_v8(raw)

    assert result == {
        "decision": {
            "correctness": None,
            "coverage": None,
            "rationale": "",
            "acceptable": False,
            "needs_review": True,
        },
        "parse_status": "unscorable",
        "repair_codes": ["missing_or_invalid_correctness"],
        "raw": raw,
    }


@pytest.mark.parametrize("rationale", [None, 7, [], {}])
def test_v8_non_string_rationale_is_unscorable(rationale: object) -> None:
    raw = json.dumps(
        {
            "correctness": "correct",
            "coverage": "partial",
            "rationale": rationale,
        }
    )

    result = normalize_judge_decision_v8(raw)

    assert result["parse_status"] == "unscorable"
    assert result["repair_codes"] == ["missing_or_invalid_rationale"]
    assert result["decision"]["acceptable"] is False


def test_v8_missing_rationale_is_unscorable() -> None:
    raw = json.dumps({"correctness": "correct", "coverage": "partial"})

    result = normalize_judge_decision_v8(raw)

    assert result["parse_status"] == "unscorable"
    assert result["repair_codes"] == ["missing_or_invalid_rationale"]


def test_v9_contract_separates_axes_and_removes_unknown_coverage() -> None:
    contract = load_prompt_contract(PROTOCOL_DIR, version="v9")
    schema = contract.schema

    assert contract.prompt_path.name == "pointwise-v9.txt"
    assert contract.schema_path.name == "decision-v9.schema.json"
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"correctness", "coverage", "rationale"}
    assert schema["properties"]["correctness"]["enum"] == [
        "correct",
        "mixed",
        "wrong",
        "unknown",
    ]
    assert schema["properties"]["coverage"]["enum"] == [
        "complete",
        "partial",
        "missing",
    ]
    assert "Coverage uses only the QUESTION and CANDIDATE RESPONSE" in contract.prompt
    assert "Correctness uses the CANDIDATE RESPONSE and REFERENCE" in contract.prompt
    assert "Reference silence is not contradiction" in contract.prompt
    assert contract.prompt.count("EXAMPLE ") == 8


def test_v9_schema_snapshot_matches_strict_pydantic_model() -> None:
    committed = load_prompt_contract(PROTOCOL_DIR, version="v9").schema
    value = {
        "correctness": "correct",
        "coverage": "partial",
        "rationale": "The stated claim is supported, but another requested part is omitted.",
    }

    assert committed == normalized_decision_v9_schema()
    assert DecisionV9.model_validate(value).coverage == "partial"
    with pytest.raises(ValueError):
        DecisionV9.model_validate(value | {"coverage": "unknown"})


def test_v9_keeps_partial_supported_answers_acceptable() -> None:
    value = {
        "correctness": "correct",
        "coverage": "partial",
        "rationale": "The stated claim is supported, but another requested part is omitted.",
    }

    result = normalize_judge_decision_v9(json.dumps(value))

    assert result == {
        "decision": value | {"acceptable": True, "needs_review": False},
        "parse_status": "clean",
        "repair_codes": [],
    }


def test_v9_unknown_coverage_is_unscorable_not_silently_accepted() -> None:
    raw = json.dumps(
        {
            "correctness": "unknown",
            "coverage": "unknown",
            "rationale": "The reference does not settle the candidate claim.",
        }
    )

    result = normalize_judge_decision_v9(raw)

    assert result["parse_status"] == "unscorable"
    assert result["repair_codes"] == ["missing_or_invalid_coverage"]
    assert result["decision"]["acceptable"] is False


def test_v10_contract_has_exact_grade_reason_schema_and_generic_examples() -> None:
    """Catches a contract that adds fields or omits the approved rubric anchors."""
    contract = load_prompt_contract(PROTOCOL_DIR, version="v10")
    schema = contract.schema

    assert contract.prompt_path.name == "pointwise-v10.txt"
    assert contract.schema_path.name == "decision-v10.schema.json"
    assert schema == {
        "additionalProperties": False,
        "properties": {
            "grade": {"enum": [0, 1, 2], "type": "integer"},
            "reason": {
                "enum": [
                    "supported",
                    "mixed",
                    "reference_silent",
                    "contradicted",
                    "nonanswer",
                ],
                "type": "string",
            },
        },
        "required": ["grade", "reason"],
        "type": "object",
    }
    assert "written independently without the later evaluator evidence" in contract.prompt
    assert contract.prompt.count("EXAMPLE ") == 5


@pytest.mark.parametrize(
    ("value", "valid"),
    [
        ({"grade": 2, "reason": "supported"}, True),
        ({"grade": 1, "reason": "mixed"}, True),
        ({"grade": 1, "reason": "reference_silent"}, True),
        ({"grade": 0, "reason": "contradicted"}, True),
        ({"grade": 0, "reason": "nonanswer"}, True),
        ({"grade": 2, "reason": "mixed"}, False),
        ({"grade": 2, "reason": "reference_silent"}, False),
        ({"grade": 2, "reason": "contradicted"}, False),
        ({"grade": 2, "reason": "nonanswer"}, False),
        ({"grade": 1, "reason": "supported"}, False),
        ({"grade": 1, "reason": "contradicted"}, False),
        ({"grade": 1, "reason": "nonanswer"}, False),
        ({"grade": 0, "reason": "supported"}, False),
        ({"grade": 0, "reason": "mixed"}, False),
        ({"grade": 0, "reason": "reference_silent"}, False),
    ],
)
def test_v10_grade_reason_matrix_is_strict(
    value: dict[str, object], valid: bool
) -> None:
    """Catches an impossible grade/reason pairing being accepted."""
    if valid:
        assert DecisionV10.model_validate(value).model_dump(mode="json") == value
    else:
        with pytest.raises(ValueError):
            DecisionV10.model_validate(value)


def test_v10_schema_snapshot_matches_strict_pydantic_model() -> None:
    """Catches divergence between the committed API schema and model boundary."""
    assert load_prompt_contract(PROTOCOL_DIR, version="v10").schema == (
        normalized_decision_v10_schema()
    )
    with pytest.raises(ValueError):
        DecisionV10.model_validate({"grade": "2", "reason": "supported"})
    with pytest.raises(ValueError):
        DecisionV10.model_validate(
            {"grade": 2, "reason": "supported", "extra": "forbidden"}
        )


def test_v10_clean_parsing_preserves_a_partial_supported_grade_two() -> None:
    """Catches correct-but-incomplete historical answers being downgraded."""
    value = {"grade": 2, "reason": "supported"}

    assert normalize_judge_decision_v10(json.dumps(value)) == {
        "decision": value,
        "parse_status": "clean",
        "repair_codes": [],
    }


def test_v10_repairs_only_reason_casing_and_whitespace() -> None:
    """Catches avoidable enum presentation drift from making a valid decision fail."""
    raw = json.dumps({"grade": 1, "reason": " Reference_Silent "})

    assert normalize_judge_decision_v10(raw) == {
        "decision": {"grade": 1, "reason": "reference_silent"},
        "parse_status": "coerced",
        "repair_codes": ["normalize_reason"],
        "raw": raw,
    }


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("not json", "invalid_json"),
        (json.dumps({"grade": 2, "reason": "mixed"}), "invalid_decision"),
        (json.dumps({"grade": "2", "reason": "supported"}), "invalid_decision"),
        (json.dumps({"grade": 1, "reason": "mixed", "extra": True}), "invalid_decision"),
    ],
)
def test_v10_malformed_output_is_unscorable(raw: str, code: str) -> None:
    """Catches malformed model output being silently reinterpreted as a judgment."""
    assert normalize_judge_decision_v10(raw) == {
        "decision": {"grade": None, "reason": None},
        "parse_status": "unscorable",
        "repair_codes": [code],
        "raw": raw,
    }


@pytest.mark.parametrize(
    ("value", "expected_grade"),
    [
        ({"grade": 1, "reason": "mixed"}, 1),
        ({"grade": 1, "reason": "reference_silent"}, 1),
        ({"grade": 0, "reason": "contradicted"}, 0),
        ({"grade": 0, "reason": "nonanswer"}, 0),
    ],
)
def test_v10_preserves_non_supported_rubric_outcomes(
    value: dict[str, object], expected_grade: int
) -> None:
    """Catches the distinct grade-one and grade-zero rubric outcomes collapsing."""
    result = normalize_judge_decision_v10(json.dumps(value))

    assert result["decision"] == value
    assert result["decision"]["grade"] == expected_grade
    assert result["parse_status"] == "clean"


def test_v11_contract_has_exact_four_field_schema_and_seven_triggers() -> None:
    """Catches a model-facing contract that grows beyond central-claim judgment."""
    contract = load_prompt_contract(PROTOCOL_DIR, version="v11")

    assert contract.prompt_path.name == "pointwise-v11.txt"
    assert contract.schema_path.name == "decision-v11.schema.json"
    assert contract.schema == {
        "additionalProperties": False,
        "properties": {
            "central_claim": {"maxLength": 240, "type": "string"},
            "answer_present": {"type": "boolean"},
            "supported_core": {"type": "boolean"},
            "material_conflict": {"type": "boolean"},
        },
        "required": [
            "central_claim",
            "answer_present",
            "supported_core",
            "material_conflict",
        ],
        "type": "object",
    }
    assert contract.prompt.count("EXAMPLE ") == 7
    assert "1. Identify whether" in contract.prompt
    assert "2. Extract its central claim" in contract.prompt
    assert "3. Test explicit evidence-backed contradiction" in contract.prompt
    assert "4. Test evidence support" in contract.prompt
    assert "5. Emit the four fields only" in contract.prompt


def test_v11_central_claim_is_limited_to_twenty_whitespace_delimited_words() -> None:
    """Catches a central-claim extractor that admits an unbounded response summary."""
    valid = {
        "central_claim": " ".join(["claim"] * 20),
        "answer_present": True,
        "supported_core": True,
        "material_conflict": False,
    }

    assert DecisionV11.model_validate(valid).central_claim == valid["central_claim"]
    with pytest.raises(ValueError):
        DecisionV11.model_validate(valid | {"central_claim": " ".join(["claim"] * 21)})


@pytest.mark.parametrize(
    ("central_claim", "answer_present"),
    [("", False), ("The answer is tidal power.", True)],
)
def test_v11_empty_claim_matches_answer_presence(
    central_claim: str, answer_present: bool
) -> None:
    """Catches a nonanswer that retains a claim, or an answer that loses one."""
    value = {
        "central_claim": central_claim,
        "answer_present": answer_present,
        "supported_core": answer_present,
        "material_conflict": False,
    }

    assert DecisionV11.model_validate(value).answer_present is answer_present


@pytest.mark.parametrize(
    ("central_claim", "answer_present"),
    [("", True), ("The answer is tidal power.", False)],
)
def test_v11_rejects_claims_inconsistent_with_answer_presence(
    central_claim: str, answer_present: bool
) -> None:
    """Catches an invalid nonanswer/claim pairing being accepted locally."""
    value = {
        "central_claim": central_claim,
        "answer_present": answer_present,
        "supported_core": False,
        "material_conflict": False,
    }

    with pytest.raises(ValueError):
        DecisionV11.model_validate(value)


@pytest.mark.parametrize(
    ("value", "label"),
    [
        (
            {
                "central_claim": "",
                "answer_present": False,
                "supported_core": False,
                "material_conflict": False,
            },
            "nonanswer",
        ),
        (
            {
                "central_claim": "The dam produces electricity.",
                "answer_present": True,
                "supported_core": True,
                "material_conflict": False,
            },
            "correct",
        ),
        (
            {
                "central_claim": "The dam produces electricity.",
                "answer_present": True,
                "supported_core": True,
                "material_conflict": True,
            },
            "partial",
        ),
        (
            {
                "central_claim": "The dam was built in 1901.",
                "answer_present": True,
                "supported_core": False,
                "material_conflict": True,
            },
            "incorrect",
        ),
        (
            {
                "central_claim": "The dam was built in 1901.",
                "answer_present": True,
                "supported_core": False,
                "material_conflict": False,
            },
            "not_established",
        ),
    ],
)
def test_v11_valid_boolean_combinations_derive_fixed_labels(
    value: dict[str, object], label: str
) -> None:
    """Catches an approved boolean state deriving the wrong downstream label."""
    decision = DecisionV11.model_validate(value)

    assert derive_v11_label(decision) == label
    assert derive_v11_label(value) == label


def test_v11_label_derivation_revalidates_mutated_and_constructed_instances() -> None:
    """Catches Pydantic instance bypasses producing labels for invalid states."""
    valid = {
        "central_claim": "The dam produces electricity.",
        "answer_present": True,
        "supported_core": True,
        "material_conflict": False,
    }
    mutated = DecisionV11.model_validate(valid)
    mutated.answer_present = False
    copied = DecisionV11.model_validate(valid).model_copy(
        update={"answer_present": False}
    )
    constructed = DecisionV11.model_construct(
        **(valid | {"answer_present": False})
    )

    for invalid in (mutated, copied, constructed):
        with pytest.raises(ValueError):
            derive_v11_label(invalid)


@pytest.mark.parametrize(
    "value",
    [
        {
            "central_claim": "",
            "answer_present": False,
            "supported_core": True,
            "material_conflict": False,
        },
        {
            "central_claim": "",
            "answer_present": False,
            "supported_core": False,
            "material_conflict": True,
        },
        {
            "central_claim": "",
            "answer_present": False,
            "supported_core": True,
            "material_conflict": True,
        },
    ],
)
def test_v11_rejects_three_impossible_boolean_combinations(
    value: dict[str, object]
) -> None:
    """Catches a nonanswer being allowed to carry support or contradiction."""
    with pytest.raises(ValueError):
        DecisionV11.model_validate(value)


def test_v11_clean_parsing_derives_the_label_without_retaining_raw() -> None:
    """Catches clean v11 output losing its deterministic derived label."""
    value = {
        "central_claim": "The dam produces electricity.",
        "answer_present": True,
        "supported_core": True,
        "material_conflict": False,
    }

    assert normalize_judge_decision_v11(json.dumps(value)) == {
        "decision": value,
        "label": "correct",
        "parse_status": "clean",
        "repair_codes": [],
    }


def test_v11_normalization_only_strips_central_claim_outer_whitespace() -> None:
    """Catches v11 normalization changing keys, booleans, or inner claim text."""
    raw = json.dumps(
        {
            "central_claim": "  The dam  produces electricity.  ",
            "answer_present": True,
            "supported_core": True,
            "material_conflict": False,
        }
    )

    assert normalize_judge_decision_v11(raw) == {
        "decision": {
            "central_claim": "The dam  produces electricity.",
            "answer_present": True,
            "supported_core": True,
            "material_conflict": False,
        },
        "label": "correct",
        "parse_status": "coerced",
        "repair_codes": ["strip_central_claim"],
        "raw": raw,
    }


@pytest.mark.parametrize(
    ("raw", "repair_code"),
    [
        (
            json.dumps(
                {
                    "central_claim": "The dam produces electricity.",
                    "answer_present": True,
                    "supported_core": True,
                    "material_conflict": False,
                    "extra": "forbidden",
                }
            ),
            "invalid_decision",
        ),
        (
            json.dumps(
                {
                    "central_claim": "The dam produces electricity.",
                    "answer_present": True,
                    "supported_core": True,
                }
            ),
            "invalid_decision",
        ),
        (
            json.dumps(
                {
                    "central_claim": "The dam produces electricity.",
                    "answer_present": "true",
                    "supported_core": 1,
                    "material_conflict": False,
                }
            ),
            "invalid_decision",
        ),
        (
            '{"central_claim":"A claim.","central_claim":"Another claim.",'
            '"answer_present":true,"supported_core":true,'
            '"material_conflict":false}',
            "duplicate_json_key",
        ),
        ("not json", "invalid_json"),
        ("[]", "json_not_object"),
        (
            json.dumps(
                {
                    "central_claim": "  The dam produces electricity.  ",
                    "answer_present": "true",
                    "supported_core": True,
                    "material_conflict": False,
                }
            ),
            "invalid_decision",
        ),
    ],
)
def test_v11_normalization_rejects_every_non_whitespace_repair(
    raw: str, repair_code: str
) -> None:
    """Catches v11 normalizing keys, booleans, or malformed JSON beyond trimming."""
    assert normalize_judge_decision_v11(raw) == {
        "decision": {
            "central_claim": None,
            "answer_present": None,
            "supported_core": None,
            "material_conflict": None,
        },
        "label": None,
        "parse_status": "unscorable",
        "repair_codes": [repair_code],
        "raw": raw,
    }


def test_v11_schema_snapshot_matches_the_provider_safe_schema() -> None:
    """Catches drift between the committed provider schema and the v11 boundary."""
    assert load_prompt_contract(PROTOCOL_DIR, version="v11").schema == (
        normalized_decision_v11_schema()
    )


def test_v2_accepts_other_error_catchall() -> None:
    value = VALID_V2_DECISION | {
        "label": "partial",
        "correctness": 3,
        "error_flags": ["other_error"],
    }

    decision = parse_judge_decision(json.dumps(value), version="v2")

    assert decision.error_flags == ("other_error",)


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
                "'unsupported_specificity', 'refusal', 'ambiguity', "
                "'reference_conflict' or 'other_error'"
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


def test_v2_accepts_parseable_scores_when_refusal_flag_is_inconsistent() -> None:
    value = VALID_V2_DECISION | {
        "label": "incorrect",
        "correctness": 0,
        "completeness": 0,
        "relevance": 2,
        "error_flags": ["refusal"],
    }

    decision = parse_judge_decision(json.dumps(value), version="v2")

    assert decision_to_dict(decision) == value


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
