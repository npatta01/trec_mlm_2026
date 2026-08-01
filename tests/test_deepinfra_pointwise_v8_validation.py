import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_deepinfra_pointwise_v8_validation import (
    MAX_TOKENS,
    MODEL,
    PROMPT_CACHE_KEY,
    _build_body,
    _error_record,
    _grounding_tier,
    _record_cache_matches,
    _record_identity,
    _record_from_body,
    _retryable_status,
    _validate_ordinal_count,
)


runner = sys.modules["scripts.run_deepinfra_pointwise_v8_validation"]


CASE = SimpleNamespace(
    ordinal=7,
    source_ordinal=70,
    record_id="record-7",
    response_id="record-7",
    question_id="question-1",
    candidate_id="candidate-1",
    selection_reason="targeted_v8_validation",
    question="Who wrote Hamlet?",
    reference_answer="William Shakespeare wrote Hamlet.",
    candidate_answer="Shakespeare wrote it.",
)

SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "correctness": {"type": "string"},
        "coverage": {"type": "string"},
        "rationale": {"type": "string"},
    },
    "required": ["correctness", "coverage", "rationale"],
}


V10_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "grade": {"type": "integer", "enum": [0, 1, 2]},
        "reason": {
            "type": "string",
            "enum": [
                "supported",
                "mixed",
                "reference_silent",
                "contradicted",
                "nonanswer",
            ],
        },
    },
    "required": ["grade", "reason"],
}


def _v10_interface(name: str):
    value = getattr(runner, name, None)
    assert callable(value), f"v10 runner must provide {name}"
    return value


def test_v10_bare_idk_filter_rejects_only_the_frozen_full_string_variants() -> None:
    is_bare_idk = _v10_interface("_is_bare_idk_response")

    rejected = (
        "I don't know",
        "I don’t know",
        "I don\u2019t know",
        "I don't t know",
        "I don 2017 t know",
        "I don imgur quote right know",
    )
    retained = (
        "I don't know the answer, but it may be Shakespeare.",
        "I don't know, so I cannot determine it from the question.",
        "The answer is not known.",
    )

    assert all(is_bare_idk(response) for response in rejected)
    assert not any(is_bare_idk(response) for response in retained)


def test_v10_evaluator_evidence_preserves_only_verified_bracketed_years_byte_for_byte() -> None:
    evaluator_evidence = _v10_interface("_evaluator_evidence")
    first = "The campaign began in [1971]."
    second = "The agreement was signed in [1978]."

    assert evaluator_evidence("5a7286cf5542990c210a4129", first) == first
    assert evaluator_evidence("5ab29e0b554299194fa9346b", second) == second

    with pytest.raises(ValueError, match="numeric square-bracket marker"):
        evaluator_evidence("other-question", "The campaign began in [1971].")
    with pytest.raises(ValueError, match="numeric square-bracket marker"):
        evaluator_evidence("5a7286cf5542990c210a4129", "The campaign began in [1978].")


@pytest.mark.parametrize(
    "answer",
    (
        "Evidence: https://example.test/source",
        "Evidence: http://example.test/source",
        "Evidence: www.example.test/source",
        "The answer is A.\nSources:\n- archive",
        "The answer is A.\nReferences:\n- archive",
        "The answer is A.\r\nSources:\r\n- archive",
        "The answer is A.\r\nReferences:\r\n- archive",
    ),
)
def test_v10_evaluator_evidence_fails_closed_for_urls_and_explicit_source_sections(
    answer: str,
) -> None:
    evaluator_evidence = _v10_interface("_evaluator_evidence")

    with pytest.raises(ValueError):
        evaluator_evidence("question-1", answer)


def test_v10_evaluator_evidence_does_not_reject_or_rewrite_non_section_source_words() -> None:
    evaluator_evidence = _v10_interface("_evaluator_evidence")
    answer = "The historian explains the term.\nSources: plural records can disagree."

    assert evaluator_evidence("question-1", answer) == answer


def test_v10_request_uses_evaluator_only_labels_thinking_and_strict_schema() -> None:
    contract = SimpleNamespace(prompt="v10 judge prompt", schema=V10_SCHEMA)
    build_body = _v10_interface("_build_body")

    body = build_body(
        contract,
        CASE,
        contract_version="v10",
        evaluator_evidence="Shakespeare wrote Hamlet.",
    )

    assert body["model"] == MODEL
    assert body["temperature"] == 0
    assert body["max_tokens"] == 8192
    assert body["reasoning_effort"] == "high"
    assert body["prompt_cache_key"] == "trec-mlm-pointwise-v10-evaluator-evidence-thinking"
    assert "provider" not in body
    assert "fallback" not in body
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "pointwise_judge_v10_decision",
            "strict": True,
            "schema": V10_SCHEMA,
        },
    }
    message = body["messages"][1]["content"]
    assert "ORIGINAL QUESTION:" in message
    assert "GEMINI GROUNDED ANSWER — EVALUATOR EVIDENCE ONLY:" in message
    assert "HISTORICAL MODEL RESPONSE TO SCORE:" in message
    assert "QUESTION ID" not in message
    assert "RESPONSE ID" not in message
    assert CASE.question_id not in message
    assert CASE.response_id not in message
    assert CASE.candidate_id not in message
    assert "PROVISIONAL GROUNDED REFERENCE" not in message
    assert "CANDIDATE RESPONSE" not in message


def test_v10_selection_requires_exactly_100_unique_non_bare_idk_cases() -> None:
    validate_selection = _v10_interface("_validate_v10_selection")

    _validate_ordinal_count("v10", list(range(100)))
    with pytest.raises(ValueError, match="v10 validation requires exactly 100"):
        _validate_ordinal_count("v10", list(range(99)))
    with pytest.raises(ValueError, match="bare IDK"):
        validate_selection(
            [
                SimpleNamespace(
                    ordinal=7,
                    candidate_answer="I don't know",
                )
            ]
        )


def test_v10_records_store_only_derived_evaluator_evidence_and_bind_its_transform() -> None:
    record_from_body = _v10_interface("_record_from_body")
    evaluator_evidence = "Shakespeare wrote Hamlet."
    body = {
        "model": MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": json.dumps({"grade": 2, "reason": "supported"}),
                },
            }
        ],
    }

    record = record_from_body(
        body,
        case=CASE,
        repetition="a",
        reference_tier="search_grounded",
        attempt_count=1,
        contract_version="v10",
        evaluator_evidence=evaluator_evidence,
    )

    assert record["evaluator_evidence"] == evaluator_evidence
    assert "provisional_reference" not in record
    assert record["normalized"]["decision"] == {"grade": 2, "reason": "supported"}

    metadata = {
        "contract_version": "v10",
        "evaluator_evidence_transform_version": "v10-evaluator-evidence-1",
        "selected_ordinals": list(range(100)),
    }
    assert _record_identity(metadata, CASE) != _record_identity(
        {**metadata, "evaluator_evidence_transform_version": "v10-evaluator-evidence-2"},
        CASE,
    )
    assert _record_identity(metadata, CASE) != _record_identity(
        {**metadata, "selected_ordinals": list(range(1, 101))}, CASE
    )


def test_v8_request_body_is_cached_direct_strict_and_has_no_fallback_controls() -> None:
    contract = SimpleNamespace(prompt="judge prompt", schema=SCHEMA)

    body = _build_body(contract, CASE)

    assert body["model"] == MODEL == "deepseek-ai/DeepSeek-V4-Flash"
    assert body["temperature"] == 0
    assert body["max_tokens"] == MAX_TOKENS == 8192
    assert body["prompt_cache_key"] == PROMPT_CACHE_KEY
    assert "reasoning" not in body
    assert "reasoning_effort" not in body
    assert "provider" not in body
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "pointwise_judge_v8_decision",
            "strict": True,
            "schema": SCHEMA,
        },
    }
    assert body["messages"][0] == {"role": "system", "content": "judge prompt"}
    assert "QUESTION:\nWho wrote Hamlet?" in body["messages"][1]["content"]
    assert "PROVISIONAL GROUNDED REFERENCE" in body["messages"][1]["content"]
    assert "CANDIDATE RESPONSE:\nShakespeare wrote it." in body["messages"][1]["content"]


def test_v9_request_body_uses_additive_contract_identity() -> None:
    contract = SimpleNamespace(prompt="v9 judge prompt", schema=SCHEMA)

    body = _build_body(contract, CASE, contract_version="v9")

    assert body["model"] == MODEL
    assert body["temperature"] == 0
    assert body["max_tokens"] == 8192
    assert body["prompt_cache_key"] == "trec-mlm-pointwise-v9-few-shot"
    assert body["response_format"]["json_schema"]["name"] == (
        "pointwise_judge_v9_decision"
    )
    assert "reasoning" not in body
    assert "reasoning_effort" not in body


def test_v9_thinking_request_is_explicit_and_has_distinct_cache_identity() -> None:
    contract = SimpleNamespace(prompt="v9 judge prompt", schema=SCHEMA)

    body = _build_body(
        contract, CASE, contract_version="v9", reasoning_enabled=True
    )

    assert body["reasoning_effort"] == "high"
    assert body["max_tokens"] == 8192
    assert body["prompt_cache_key"] == "trec-mlm-pointwise-v9-few-shot-thinking"
    assert "provider" not in body


def test_validation_size_allows_v9_smoke_or_holdout_without_weakening_v8() -> None:
    _validate_ordinal_count("v8", list(range(50)))
    _validate_ordinal_count("v9", list(range(10)))
    _validate_ordinal_count("v9", list(range(100)))

    with pytest.raises(ValueError, match="v8 validation requires exactly 50"):
        _validate_ordinal_count("v8", list(range(100)))
    with pytest.raises(ValueError, match="v9 validation requires exactly 10 or 100"):
        _validate_ordinal_count("v9", list(range(11)))
    with pytest.raises(ValueError, match="unique ordinals"):
        _validate_ordinal_count("v9", [0] * 10)


def test_grounding_tier_uses_recorded_search_invocation_not_request_intent() -> None:
    search = {
        "uncertainty_status": "answered",
        "request_metadata": {"grounding": True, "grounding_query_count": 2},
    }
    no_search = {
        "uncertainty_status": "answered",
        "request_metadata": {"grounding": True},
    }
    failed = {
        "uncertainty_status": "unknown",
        "request_metadata": {"grounding": True, "grounding_query_count": 1},
    }

    assert _grounding_tier(search) == "search_grounded"
    assert _grounding_tier(no_search) == "generated_without_search"
    assert _grounding_tier(failed) == "unanswered_or_failed"


def test_provider_response_becomes_private_auditable_v8_record() -> None:
    content = json.dumps(
        {
            "correctness": "correct",
            "coverage": "partial",
            "rationale": "The stated claim is supported but other requested parts are omitted.",
        }
    )
    body = {
        "model": MODEL,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": content},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "estimated_cost": 0.0000123,
        },
    }

    record = _record_from_body(
        body,
        case=CASE,
        repetition="a",
        reference_tier="search_grounded",
        attempt_count=1,
    )

    assert record["ordinal"] == 7
    assert record["record_id"] == "record-7"
    assert record["repetition"] == "a"
    assert record["reference_tier"] == "search_grounded"
    assert record["question"] == CASE.question
    assert record["provisional_reference"] == CASE.reference_answer
    assert record["candidate_response"] == CASE.candidate_answer
    assert record["attempt_count"] == 1
    assert record["normalized"]["parse_status"] == "clean"
    assert record["normalized"]["decision"]["acceptable"] is True
    assert record["estimated_cost_usd"] == 0.0000123
    assert record["finish_reason"] == "stop"


def test_only_transient_http_statuses_are_retried() -> None:
    assert _retryable_status(408)
    assert _retryable_status(429)
    assert _retryable_status(500)
    assert _retryable_status(504)
    assert not _retryable_status(400)
    assert not _retryable_status(401)
    assert not _retryable_status(404)


def test_record_cache_identity_binds_manifest_contract_and_case() -> None:
    metadata = {
        "inventory_manifest_sha256": "a" * 64,
        "inventory_sha256": "b" * 64,
        "grounding_sha256": "c" * 64,
        "prompt_sha256": "d" * 64,
        "schema_sha256": "e" * 64,
        "model": MODEL,
        "repetition": "a",
    }
    identity = _record_identity(metadata, CASE)
    record = {
        "ordinal": CASE.ordinal,
        "record_id": CASE.record_id,
        "repetition": "a",
        "cache_identity_sha256": identity,
    }

    assert _record_cache_matches(
        record, case=CASE, repetition="a", cache_identity_sha256=identity
    )
    assert not _record_cache_matches(
        record,
        case=CASE,
        repetition="a",
        cache_identity_sha256="f" * 64,
    )


def test_provider_error_artifact_never_persists_untrusted_exception_text() -> None:
    secret = "Bearer private-token-must-not-be-written"

    record = _error_record(
        RuntimeError(secret),
        case=CASE,
        repetition="a",
        reference_tier="search_grounded",
    )

    assert record["provider_error"] == "unexpected_error"
    assert secret not in json.dumps(record)
