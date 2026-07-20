from __future__ import annotations

import json
from pathlib import Path

from scripts.fetch_perplexity_sonar import (
    citation_state,
    compact_response,
    load_questions,
    promote_retry,
    strip_citations,
)


def test_strip_citations_removes_inline_markers_without_collapsing_lines() -> None:
    answer = "First claim [1].\n\nSecond claim [2][3]."

    assert strip_citations(answer) == "First claim.\n\nSecond claim."


def test_citation_state_rejects_missing_sources_and_out_of_range_markers() -> None:
    assert citation_state({"answer": "No evidence", "citations": []}) == (False, True)
    assert citation_state({"answer": "Claim [2]", "citations": ["https://a.test"]}) == (
        True,
        False,
    )
    assert citation_state({"answer": "Claim [1]", "citations": ["https://a.test"]}) == (
        True,
        True,
    )


def test_compact_response_keeps_answer_citations_search_results_and_usage() -> None:
    body = {
        "id": "req-1",
        "model": "sonar",
        "created": 123,
        "choices": [{"message": {"content": "Answer [1]"}, "finish_reason": "stop"}],
        "citations": ["https://example.test/source"],
        "search_results": [{"title": "Source"}],
        "usage": {"prompt_tokens": 5},
    }

    result = compact_response(body)

    assert result["answer"] == "Answer [1]"
    assert result["answer_no_citations"] == "Answer"
    assert result["citations"] == ["https://example.test/source"]
    assert result["search_results"] == [{"title": "Source"}]
    assert result["usage"] == {"prompt_tokens": 5}


def test_load_questions_reads_query_mapping(tmp_path: Path) -> None:
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps({"q-2": {"query": "second"}, "q-1": {"query": "first"}}),
        encoding="utf-8",
    )

    assert load_questions(path) == [("q-2", "second"), ("q-1", "first")]


def test_promote_retry_only_replaces_response_when_citations_are_valid() -> None:
    record = {"response": {"answer": "Original", "citations": []}}
    valid_retry = {"answer": "Replacement [1]", "citations": ["https://a.test"]}

    assert promote_retry(record, valid_retry) is True
    assert record["response"] == valid_retry

    record = {"response": {"answer": "Original", "citations": []}}
    invalid_retry = {"answer": "Uncited", "citations": []}

    assert promote_retry(record, invalid_retry) is False
    assert record["response"]["answer"] == "Original"
