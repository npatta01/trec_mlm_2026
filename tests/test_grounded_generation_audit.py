from pathlib import Path

from scripts.audit_grounded_generation import audit_generation


def test_canary_audit_passes_and_projects_conservative_cost(tmp_path: Path) -> None:
    questions = [
        {
            "question_id": "q1",
            "question": "Who is president in 2026?",
            "search_expected": True,
        },
        {"question_id": "q2", "question": "What is 2+2?", "search_expected": False},
    ]
    public = {
        "records": {
            question_id: {
                "question_id": question_id,
                "answer": "A complete answer.",
                "uncertainty_status": "answered",
                "uncertainty_reason": "none",
                "finish_reason": "stop",
                "usage": {"promptTokenCount": 100, "candidatesTokenCount": 400},
                "request_metadata": {"grounding_query_count": searches},
            }
            for question_id, searches in (("q1", 1), ("q2", 0))
        },
        "failures": {},
    }
    private = {
        "records": {
            "q1": {
                "search_invoked": True,
                "web_search_queries": ["query"],
                "sources": [{"url": "https://example.com", "title": "Example"}],
            },
            "q2": {"search_invoked": False, "web_search_queries": [], "sources": []},
        },
        "failures": {},
    }

    audit = audit_generation(questions, public, private, projected_question_count=7_759)

    assert audit["gate_passed"] is True
    assert audit["counts"]["unique_question_ids"] == 2
    assert audit["counts"]["non_search"] == 1
    assert audit["cost_audit"]["current_repricing_estimate_usd"][
        "total_with_fresh_monthly_allowance"
    ] < 150
    assert audit["cost_audit"]["historical_recorded_spend"]["status"] == (
        "not_supplied"
    )
    assert audit["cost_audit"]["historical_recorded_spend"][
        "provider_billed_actual_usd"
    ] is None


def test_canary_audit_fails_closed_on_integrity_errors() -> None:
    questions = [
        {"question_id": "q1", "question": "latest news", "search_expected": True}
    ]
    public = {
        "records": {
            "q1": {
                "answer": "",
                "uncertainty_status": "invalid",
                "uncertainty_reason": "",
                "finish_reason": "length",
                "usage": {},
                "request_metadata": {},
            }
        },
        "failures": {},
    }
    private = {
        "records": {"q1": {"search_invoked": False, "sources": []}},
        "failures": {},
    }

    audit = audit_generation(questions, public, private, projected_question_count=7_759)

    assert audit["gate_passed"] is False
    assert audit["invalid_status_ids"] == ["q1"]
    assert audit["empty_answer_ids"] == ["q1"]
    assert audit["truncated_ids"] == ["q1"]
    assert audit["expected_search_not_used_ids"] == ["q1"]


def test_audit_uses_current_gemini_36_pricing_and_configured_grounding_allowance() -> None:
    questions = [{"question_id": "q1", "question": "latest news", "search_expected": True}]
    record = {
        "question_id": "q1",
        "answer": "A complete answer.",
        "uncertainty_status": "answered",
        "uncertainty_reason": "none",
        "finish_reason": "stop",
        "usage": {"promptTokenCount": 100, "candidatesTokenCount": 400},
        "request_metadata": {"grounding_query_count": 1},
    }
    private = {
        "records": {
            "q1": {
                "search_invoked": True,
                "web_search_queries": ["query"],
                "sources": [{"url": "https://example.com", "title": "Example"}],
            }
        },
        "failures": {},
    }

    audit = audit_generation(
        questions,
        {"records": {"q1": record}, "failures": {}},
        private,
        projected_question_count=7_759,
        monthly_free_grounded_prompts=0,
    )

    estimate = audit["cost_audit"]["current_repricing_estimate_usd"]
    assert estimate["search_with_fresh_monthly_allowance"] > 0
    assert estimate["assumptions"] == {
        "model": "gemini-3.6-flash",
        "pricing_as_of": "2026-07-21",
        "standard_input_per_million_tokens": 1.5,
        "standard_output_per_million_tokens_including_thinking": 7.5,
        "monthly_free_grounded_prompts_shared_across_gemini_3": 0,
        "search_per_thousand_queries_after_free_allowance": 14.0,
        "allowance_scope": "per project, shared across Gemini 3",
        "source": "https://ai.google.dev/gemini-api/docs/pricing",
    }
    assert estimate["total_if_allowance_already_exhausted"] >= estimate[
        "total_with_fresh_monthly_allowance"
    ]


def test_audit_separates_recorded_spend_ceiling_from_provider_billed_actual() -> None:
    questions = [{"question_id": "q1", "question": "latest news"}]
    record = {
        "question_id": "q1",
        "answer": "A complete answer.",
        "uncertainty_status": "answered",
        "uncertainty_reason": "none",
        "finish_reason": "stop",
        "usage": {"promptTokenCount": 100, "candidatesTokenCount": 400},
        "request_metadata": {"grounding_query_count": 1},
    }
    private = {
        "records": {
            "q1": {
                "search_invoked": True,
                "sources": [{"url": "https://example.com", "title": "Example"}],
            }
        },
        "failures": {},
    }
    receipts = [
        {
            "overall_maximum_tracked_spend_usd": 87.872091,
            "pricing_assumptions": {
                "input_per_million_tokens_usd": 1.5,
                "output_per_million_tokens_usd": 7.5,
            },
        }
    ]

    audit = audit_generation(
        questions,
        {"records": {"q1": record}, "failures": {}},
        private,
        projected_question_count=1,
        historical_spend_receipts=receipts,
    )

    historical = audit["cost_audit"]["historical_recorded_spend"]
    assert historical["status"] == "run_time_upper_bound_recorded"
    assert historical["maximum_tracked_spend_usd"] == 87.872091
    assert historical["provider_billed_actual_usd"] is None
    assert historical["provider_billed_actual_status"] == (
        "not_verified_from_google_cloud_billing"
    )
