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
    assert audit["projected_cost_usd"]["conservative_total"] < 150


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


def test_audit_uses_configured_grounding_allowance() -> None:
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
        free_grounded_prompts=0,
    )

    assert audit["projected_cost_usd"]["search"] > 0
    assert "free_grounded_prompts" in audit["projected_cost_usd"]["assumptions"]
