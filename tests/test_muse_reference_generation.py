import json
import os
from pathlib import Path

import pytest

from trec_mlm.muse_reference_generation import (
    MUSE_ENDPOINT,
    MUSE_MODEL,
    MuseRunError,
    MUSE_REFERENCE_FOLLOWUP_PROMPT,
    MUSE_REFERENCE_GENERIC_FOLLOWUP_PROMPT,
    MUSE_REFERENCE_GENERIC_INITIAL_PROMPT,
    MUSE_REFERENCE_GENERIC_BRANCHES_FOLLOWUP_PROMPT,
    MUSE_REFERENCE_GENERIC_BRANCHES_INITIAL_PROMPT,
    MUSE_REFERENCE_GENERIC_BRANCHES_V2_FOLLOWUP_PROMPT,
    MUSE_REFERENCE_GENERIC_BRANCHES_V2_INITIAL_PROMPT,
    MUSE_REFERENCE_GENERIC_BRANCHES_V3_FOLLOWUP_PROMPT,
    MUSE_REFERENCE_GENERIC_BRANCHES_V3_INITIAL_PROMPT,
    MUSE_REFERENCE_GENERIC_CITATIONS_FOLLOWUP_PROMPT,
    MUSE_REFERENCE_GENERIC_CITATIONS_INITIAL_PROMPT,
    MUSE_REFERENCE_GENERIC_UNCERTAINTY_FOLLOWUP_PROMPT,
    MUSE_REFERENCE_GENERIC_UNCERTAINTY_INITIAL_PROMPT,
    MUSE_REFERENCE_INITIAL_PROMPT,
    ProviderRequestError,
    build_muse_followup_request,
    build_muse_initial_request,
    is_complete_muse_record,
    load_muse_cache,
    parse_muse_response,
    run_muse_two_call,
    save_muse_cache,
    select_canary_questions,
)


def _response(
    response_id: str,
    text: str,
    *,
    query: str = "search query",
    opened_url: str = "https://example.com/source",
    status: str = "completed",
) -> dict:
    return {
        "id": response_id,
        "model": MUSE_MODEL,
        "status": status,
        "output": [
            {
                "type": "reasoning",
                "id": f"rs_{response_id}",
                "status": "completed",
            },
            {
                "type": "message",
                "role": "assistant",
                "phase": "commentary",
                "content": [{"type": "output_text", "text": "Searching."}],
                "status": "completed",
            },
            {
                "type": "web_search_call",
                "id": f"ws_{response_id}",
                "status": "completed",
                "action": {"type": "search", "query": query},
            },
            {
                "type": "web_search_call",
                "id": f"wp_{response_id}",
                "status": "completed",
                "action": {"type": "open_page", "url": opened_url},
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [
                            {
                                "type": "url_citation",
                                "url": opened_url,
                                "title": "Example source",
                                "start_index": 0,
                                "end_index": len(text),
                            }
                        ],
                    }
                ],
                "status": "completed",
            },
        ],
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 3},
            "output_tokens": 7,
            "output_tokens_details": {"reasoning_tokens": 2},
            "total_tokens": 17,
        },
    }


def test_muse_requests_enable_search_without_arbitrary_limits() -> None:
    initial = build_muse_initial_request("Who wrote Hamlet?")
    followup = build_muse_followup_request("resp_previous")

    assert initial["model"] == MUSE_MODEL
    assert initial["tools"] == [{"type": "web_search"}]
    assert "Who wrote Hamlet?" in initial["input"]
    assert followup["previous_response_id"] == "resp_previous"
    assert followup["tools"] == [{"type": "web_search"}]
    encoded = json.dumps((initial, followup))
    assert "max_output_tokens" not in encoded
    assert "max_tokens" not in encoded
    assert "max_uses" not in encoded


def test_reference_prompt_builds_an_evaluation_record_without_source_blacklist() -> None:
    initial = build_muse_initial_request("Who wrote Hamlet?", prompt_variant="reference")
    followup = build_muse_followup_request(
        "resp_previous", prompt_variant="reference"
    )

    assert MUSE_REFERENCE_INITIAL_PROMPT in initial["input"]
    assert initial["tools"] == [{"type": "web_search"}]
    assert followup["input"] == MUSE_REFERENCE_FOLLOWUP_PROMPT
    assert "wikipedia" in MUSE_REFERENCE_INITIAL_PROMPT.lower()
    encoded = json.dumps((initial, followup)).lower()
    assert "max_output_tokens" not in encoded
    assert "max_tokens" not in encoded
    assert "max_uses" not in encoded


def test_generic_reference_prompt_checks_question_semantics_without_domain_rules() -> None:
    initial = build_muse_initial_request(
        "Which answer is correct?", prompt_variant="reference_generic"
    )
    followup = build_muse_followup_request(
        "resp_previous", prompt_variant="reference_generic"
    )

    assert MUSE_REFERENCE_GENERIC_INITIAL_PROMPT in initial["input"]
    assert followup["input"] == MUSE_REFERENCE_GENERIC_FOLLOWUP_PROMPT
    prompt = MUSE_REFERENCE_GENERIC_INITIAL_PROMPT.lower()
    assert "exactly what the question asks" in prompt
    assert "do not substitute a related fact" in prompt
    assert "do not generalize from one example" in prompt
    assert "vegetation" not in prompt
    encoded = json.dumps((initial, followup)).lower()
    assert "max_output_tokens" not in encoded
    assert "max_tokens" not in encoded
    assert "max_uses" not in encoded


def test_generic_reference_citation_variant_requires_inline_provenance() -> None:
    initial = build_muse_initial_request(
        "Which answer is correct?", prompt_variant="reference_generic_citations"
    )
    followup = build_muse_followup_request(
        "resp_previous", prompt_variant="reference_generic_citations"
    )

    assert MUSE_REFERENCE_GENERIC_CITATIONS_INITIAL_PROMPT in initial["input"]
    assert followup["input"] == MUSE_REFERENCE_GENERIC_CITATIONS_FOLLOWUP_PROMPT
    encoded = json.dumps((initial, followup)).lower()
    assert "attach inline url citations" in encoded
    assert "max_output_tokens" not in encoded
    assert "max_tokens" not in encoded
    assert "max_uses" not in encoded


def test_generic_reference_uncertainty_variant_uses_one_simple_status() -> None:
    initial = build_muse_initial_request(
        "Which answer is correct?", prompt_variant="reference_generic_uncertainty"
    )
    followup = build_muse_followup_request(
        "resp_previous", prompt_variant="reference_generic_uncertainty"
    )

    assert MUSE_REFERENCE_GENERIC_UNCERTAINTY_INITIAL_PROMPT in initial["input"]
    assert followup["input"] == MUSE_REFERENCE_GENERIC_UNCERTAINTY_FOLLOWUP_PROMPT
    prompt = MUSE_REFERENCE_GENERIC_UNCERTAINTY_INITIAL_PROMPT.lower()
    assert "reference status: answered or uncertain" in prompt
    assert "uncertainty reason" in prompt
    assert "do not silently choose" in prompt
    assert "do not impose an arbitrary search or output-length limit" in prompt
    encoded = json.dumps((initial, followup)).lower()
    assert "max_output_tokens" not in encoded
    assert "max_tokens" not in encoded
    assert "max_uses" not in encoded


def test_generic_reference_branches_prompt_is_conditional_and_domain_neutral() -> None:
    initial = build_muse_initial_request(
        "Which answer is correct?", prompt_variant="reference_generic_branches"
    )
    followup = build_muse_followup_request(
        "resp_previous", prompt_variant="reference_generic_branches"
    )

    assert MUSE_REFERENCE_GENERIC_BRANCHES_INITIAL_PROMPT in initial["input"]
    assert followup["input"] == MUSE_REFERENCE_GENERIC_BRANCHES_FOLLOWUP_PROMPT
    prompt = MUSE_REFERENCE_GENERIC_BRANCHES_INITIAL_PROMPT.lower()
    assert "only when each reading honors all explicit constraints" in prompt
    assert "conflicting evidence" in prompt
    assert "scope or granularity" in prompt
    assert "acceptance set" in prompt
    assert "vegetation" not in prompt
    encoded = json.dumps((initial, followup)).lower()
    assert "max_output_tokens" not in encoded
    assert "max_tokens" not in encoded
    assert "max_uses" not in encoded


def test_generic_reference_branches_v2_requires_stable_decision_and_branches() -> None:
    initial = build_muse_initial_request(
        "Which answer is correct?", prompt_variant="reference_generic_branches_v2"
    )
    followup = build_muse_followup_request(
        "resp_previous", prompt_variant="reference_generic_branches_v2"
    )

    assert MUSE_REFERENCE_GENERIC_BRANCHES_V2_INITIAL_PROMPT in initial["input"]
    assert followup["input"] == MUSE_REFERENCE_GENERIC_BRANCHES_V2_FOLLOWUP_PROMPT
    prompt = MUSE_REFERENCE_GENERIC_BRANCHES_V2_INITIAL_PROMPT.lower()
    assert "interpretation decision: single" in prompt
    assert "interpretation decision: multiple" in prompt
    assert "interpretation 1" in prompt
    assert "interpretation 2" in prompt
    assert "do not relabel that situation as scope or granularity" in prompt
    assert "silently replacing the draft" in prompt
    assert "vegetation" not in prompt
    encoded = json.dumps((initial, followup)).lower()
    assert "max_output_tokens" not in encoded
    assert "max_tokens" not in encoded
    assert "max_uses" not in encoded


def test_generic_reference_branches_v3_checks_each_dimension_before_single() -> None:
    initial = build_muse_initial_request(
        "Which answer is correct?", prompt_variant="reference_generic_branches_v3"
    )
    followup = build_muse_followup_request(
        "resp_previous", prompt_variant="reference_generic_branches_v3"
    )

    assert MUSE_REFERENCE_GENERIC_BRANCHES_V3_INITIAL_PROMPT in initial["input"]
    assert followup["input"] == MUSE_REFERENCE_GENERIC_BRANCHES_V3_FOLLOWUP_PROMPT
    prompt = MUSE_REFERENCE_GENERIC_BRANCHES_V3_INITIAL_PROMPT.lower()
    assert "every plausible semantic dimension" in prompt
    assert "a dimension that was never tested is not ruled out" in prompt
    assert "if two tested dimensions remain evidence-supported" in prompt
    encoded = json.dumps((initial, followup)).lower()
    assert "max_output_tokens" not in encoded
    assert "max_tokens" not in encoded
    assert "max_uses" not in encoded


def test_parse_muse_response_extracts_final_text_and_search_provenance() -> None:
    parsed = parse_muse_response(
        _response("resp_1", "ANSWERED\n\nWilliam Shakespeare.")
    )

    assert parsed["response_id"] == "resp_1"
    assert parsed["answer"] == "ANSWERED\n\nWilliam Shakespeare."
    assert parsed["status"] == "completed"
    assert parsed["usage"]["total_tokens"] == 17
    assert parsed["search_queries"] == ["search query"]
    assert parsed["opened_urls"] == ["https://example.com/source"]
    assert parsed["citations"] == [
        {
            "url": "https://example.com/source",
            "title": "Example source",
            "start_index": 0,
            "end_index": 30,
        }
    ]


def test_parse_muse_response_preserves_find_in_page_provenance() -> None:
    response = _response("resp_1", "Grounded answer.")
    response["output"].insert(
        -1,
        {
            "type": "web_search_call",
            "id": "find_1",
            "status": "completed",
            "action": {
                "type": "find_in_page",
                "url": "https://example.com/find",
                "pattern": "relevant phrase",
            },
        },
    )

    parsed = parse_muse_response(response)

    assert parsed["search_actions"][-1] == {
        "type": "find_in_page",
        "id": "find_1",
        "url": "https://example.com/find",
        "pattern": "relevant phrase",
    }
    assert "https://example.com/find" in parsed["opened_urls"]


def test_two_call_runner_uses_previous_response_and_keeps_both_raw_responses() -> None:
    requests: list[dict] = []
    responses = [
        _response("resp_1", "First grounded answer."),
        _response("resp_2", "Final grounded answer.", query="second pass"),
    ]

    def request(url: str, body: dict, headers: dict[str, str]):
        assert url == MUSE_ENDPOINT
        assert headers == {"Authorization": "Bearer test-key"}
        requests.append(body)
        return responses.pop(0)

    record = run_muse_two_call(
        "question",
        api_key="test-key",
        request_fn=request,
        sleep_fn=lambda _: None,
    )

    assert record["final_stage"] == 2
    assert record["answer"] == "Final grounded answer."
    assert record["call_1"]["raw_response"]["id"] == "resp_1"
    assert record["call_2"]["raw_response"]["id"] == "resp_2"
    assert requests[1]["previous_response_id"] == "resp_1"
    assert record["search_queries"] == ["search query", "second pass"]
    assert "test-key" not in json.dumps(record)


def test_two_call_runner_records_prompt_variant() -> None:
    responses = [
        _response("resp_1", "First grounded reference."),
        _response("resp_2", "Final grounded reference.", query="second pass"),
    ]

    def request(url: str, body: dict, headers: dict[str, str]):
        return responses.pop(0)

    record = run_muse_two_call(
        "question",
        api_key="test-key",
        prompt_variant="reference",
        request_fn=request,
        sleep_fn=lambda _: None,
    )

    assert record["prompt_variant"] == "reference"


def test_two_call_runner_records_generic_reference_prompt_variant() -> None:
    responses = [
        _response("resp_1", "First grounded reference."),
        _response("resp_2", "Final grounded reference.", query="second pass"),
    ]

    def request(url: str, body: dict, headers: dict[str, str]):
        return responses.pop(0)

    record = run_muse_two_call(
        "question",
        api_key="test-key",
        prompt_variant="reference_generic",
        request_fn=request,
        sleep_fn=lambda _: None,
    )

    assert record["prompt_variant"] == "reference_generic"


def test_two_call_runner_records_generic_citation_prompt_variant() -> None:
    responses = [
        _response("resp_1", "First grounded reference."),
        _response("resp_2", "Final grounded reference.", query="second pass"),
    ]

    def request(url: str, body: dict, headers: dict[str, str]):
        return responses.pop(0)

    record = run_muse_two_call(
        "question",
        api_key="test-key",
        prompt_variant="reference_generic_citations",
        request_fn=request,
        sleep_fn=lambda _: None,
    )

    assert record["prompt_variant"] == "reference_generic_citations"


def test_two_call_runner_falls_back_to_stage_one_when_followup_fails() -> None:
    calls = 0

    def request(url: str, body: dict, headers: dict[str, str]):
        nonlocal calls
        calls += 1
        if calls == 1:
            return _response("resp_1", "First grounded answer.")
        raise ProviderRequestError(category="http", http_status=503, retryable=True)

    record = run_muse_two_call(
        "question",
        api_key="test-key",
        request_fn=request,
        sleep_fn=lambda _: None,
    )

    assert calls == 2
    assert record["final_stage"] == 1
    assert record["answer"] == "First grounded answer."
    assert record["followup_failure"] == {
        "category": "http",
        "error_type": "ProviderRequestError",
        "http_status": 503,
        "retryable": True,
    }


def test_first_stage_parse_failure_preserves_received_raw_response() -> None:
    response = {
        "id": "resp_1",
        "model": MUSE_MODEL,
        "status": "completed",
        "output": [],
        "usage": {"total_tokens": 3},
    }

    def request(url: str, body: dict, headers: dict[str, str]):
        return response

    with pytest.raises(MuseRunError) as raised:
        run_muse_two_call(
            "question",
            api_key="test-key",
            request_fn=request,
            sleep_fn=lambda _: None,
        )

    assert raised.value.payload["call_1"]["raw_response"] == response
    assert raised.value.payload["call_1"]["error"]["error_type"] == "ValueError"


def test_canary_selection_is_stable_and_cache_is_atomic(tmp_path: Path) -> None:
    questions = [(f"q{i}", f"question {i}") for i in range(10)]
    assert select_canary_questions(questions, size=4, seed=20260822) == (
        select_canary_questions(questions, size=4, seed=20260822)
    )

    path = tmp_path / "private" / "cache.json"
    record = {
        "question_id": "q1",
        "question": "question",
        "provider": "meta-muse",
        "model": MUSE_MODEL,
        "answer": "answer",
        "final_stage": 2,
        "call_1": {},
        "call_2": {},
    }
    save_muse_cache(
        path,
        {"schema_version": 1, "model": MUSE_MODEL, "records": {"q1": record}, "failures": {}},
    )
    cache = load_muse_cache(path)
    assert cache["records"]["q1"] == record
    assert is_complete_muse_record(cache["records"]["q1"], model=MUSE_MODEL)
    assert path.stat().st_mode & 0o777 == 0o600
    assert not path.with_suffix(".json.tmp").exists()


def test_muse_cache_temporary_is_private_before_any_content_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed_modes: list[int] = []
    real_fdopen = os.fdopen

    def checked_fdopen(fd: int, *args, **kwargs):
        observed_modes.append(os.fstat(fd).st_mode & 0o777)
        return real_fdopen(fd, *args, **kwargs)

    monkeypatch.setattr(os, "fdopen", checked_fdopen)
    save_muse_cache(
        tmp_path / "private" / "cache.json",
        {"schema_version": 1, "records": {}, "failures": {}},
    )

    assert observed_modes == [0o600]


def test_canary_selection_rejects_oversized_request() -> None:
    with pytest.raises(ValueError, match="size"):
        select_canary_questions([("q1", "one")], size=2, seed=1)


def test_muse_resume_rejects_record_from_stale_prompt_revision() -> None:
    record = {
        "provider": "meta-muse",
        "model": MUSE_MODEL,
        "prompt_variant": "answer",
        "prompt_sha256": "stale-prompt-hash",
        "answer": "answer",
        "final_stage": 2,
        "call_1": {},
        "call_2": {},
    }

    assert not is_complete_muse_record(
        record, model=MUSE_MODEL, prompt_variant="answer"
    )
