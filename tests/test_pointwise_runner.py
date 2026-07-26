from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import stat
from typing import Any
import urllib.request

import pytest

import trec_mlm.pointwise_runner as runner
from trec_mlm.pointwise_judge import CandidateCase, load_prompt_contract
from trec_mlm.pointwise_runner import (
    call_openai_judge,
    run_cases,
    validate_loopback_endpoint,
)


TEST_MAX_RESPONSE_BYTES = 1024 * 1024
CASE = CandidateCase(
    question_id="q1",
    response_id="r1",
    question="Who wrote Hamlet?",
    reference_answer="William Shakespeare wrote Hamlet.",
    candidate_answer="Shakespeare wrote Hamlet.",
    strata=("high_overlap_control",),
    selection_reason="high_overlap_control",
)
SECOND_CASE = replace(
    CASE,
    question_id="q2",
    response_id="r2",
    question="What is water?",
    reference_answer="Water is H2O.",
    candidate_answer="Water is H2O.",
)
CONTRACT = replace(
    load_prompt_contract(Path("protocols/pointwise_judge_bakeoff")),
    prompt="Judge the candidate under the frozen contract.",
)
RECEIPT = {
    "runtime": "local-test",
    "gpu": "fake",
    "passed": True,
    "schema_sha256": runner.FROZEN_DECISION_SCHEMA_SHA256,
}
VALID_DECISION = {
    "label": "good",
    "correctness": 4,
    "completeness": 2,
    "relevance": 2,
    "error_flags": [],
    "reference_status": "sound",
    "confidence": "high",
    "rationale": "The response gives the correct author directly.",
}
VALID_DECISION_JSON = json.dumps(VALID_DECISION)
DUPLICATE_KEY_DECISION_JSON = VALID_DECISION_JSON.replace(
    '"label": "good"', '"label": "incorrect", "label": "good"', 1
)


def _response_body(
    raw: str,
    *,
    model: str = "model-a",
    usage: object = None,
) -> dict[str, object]:
    if usage is None:
        usage = {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20}
    return {
        "id": "completion-1",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": raw}}],
        "usage": usage,
    }


class ScriptedCall:
    def __init__(self, responses: list[str | BaseException]) -> None:
        self.responses = iter(responses)
        self.requests: list[dict[str, object]] = []

    def __call__(self, model: str, messages: object, schema: object):
        self.requests.append({"model": model, "messages": messages, "schema": schema})
        response = next(self.responses)
        if isinstance(response, BaseException):
            raise response
        return response, _response_body(response, model=model)


class FakeHTTPResponse:
    def __init__(
        self,
        body: object,
        *,
        status: int = 200,
        final_url: str = "http://127.0.0.1:13305/v1/chat/completions",
        content_length: str | None = None,
    ) -> None:
        self.body = (
            body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        )
        self.status = status
        self.final_url = final_url
        self.headers = (
            {} if content_length is None else {"Content-Length": content_length}
        )
        self.read_sizes: list[int] = []

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def geturl(self) -> str:
        return self.final_url

    def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        return self.body if size < 0 else self.body[:size]


@pytest.mark.parametrize("boundary", ["contract", "receipt"])
def test_run_cases_rejects_nonfrozen_schema_before_cache_or_call(
    tmp_path: Path, boundary: str
) -> None:
    output = tmp_path / "private" / "run.json"
    contract = replace(CONTRACT, schema={}) if boundary == "contract" else CONTRACT
    receipt = (
        RECEIPT if boundary == "contract" else {**RECEIPT, "schema_sha256": "0" * 64}
    )
    call = ScriptedCall([])

    with pytest.raises(ValueError, match="schema"):
        run_cases(
            [CASE],
            model_alias="judge-a",
            model_id="model-a",
            contract=contract,
            output_path=output,
            runtime_receipt=receipt,
            call=call,
        )

    assert call.requests == []
    assert not output.exists()


def test_loopback_transport_disables_ambient_http_proxies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers: tuple[object, ...] = ()

    class FakeOpener:
        def open(self, request: object, *, timeout: float) -> FakeHTTPResponse:
            return FakeHTTPResponse(_response_body(VALID_DECISION_JSON))

    def fake_build_opener(*values: object) -> FakeOpener:
        nonlocal handlers
        handlers = values
        return FakeOpener()

    monkeypatch.setattr(urllib.request, "build_opener", fake_build_opener)
    request = urllib.request.Request(
        "http://127.0.0.1:13305/v1/chat/completions", data=b"{}"
    )

    runner._open_no_redirect(request, 2.0)

    proxy_handlers = [
        handler
        for handler in handlers
        if isinstance(handler, urllib.request.ProxyHandler)
    ]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}


@pytest.mark.parametrize(
    "url",
    [
        "http://0.0.0.0:13305/v1",
        "http://192.168.1.2:8000/v1",
        "https://api.example.com/v1",
        "http://localhost.example.com/v1",
        "http://localhost:13305/v1",
        "http://127.0.0.1:13305/not-v1",
        "http://user@127.0.0.1:13305/v1",
        "http://127.0.0.1:13305/v1?forward=https://example.com",
    ],
)
def test_endpoint_must_be_literal_loopback_v1_without_ambient_authority(
    url: str,
) -> None:
    with pytest.raises(ValueError, match="loopback|/v1|userinfo|query"):
        validate_loopback_endpoint(url)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://127.0.0.1:13305/v1", "http://127.0.0.1:13305/v1"),
        ("http://127.0.0.1:13305/v1/", "http://127.0.0.1:13305/v1"),
        ("http://[::1]:8000/v1", "http://[::1]:8000/v1"),
    ],
)
def test_endpoint_accepts_only_literal_loopback_v1(url: str, expected: str) -> None:
    assert validate_loopback_endpoint(url) == expected


def test_openai_call_sends_strict_schema_and_returns_complete_private_body(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = _response_body(VALID_DECISION_JSON)
    seen: dict[str, object] = {}

    def fake_open(request: Any, timeout: float):
        seen["url"] = request.full_url
        seen["headers"] = dict(request.header_items())
        seen["payload"] = json.loads(request.data)
        seen["timeout"] = timeout
        return FakeHTTPResponse(body)

    monkeypatch.setattr(runner, "_open_no_redirect", fake_open)

    raw, returned_body = call_openai_judge(
        "http://127.0.0.1:13305/v1",
        "model-a",
        [{"role": "user", "content": "judge this"}],
        CONTRACT.schema,
        timeout_seconds=17.5,
    )

    assert raw == VALID_DECISION_JSON
    assert returned_body == body
    assert seen["url"] == "http://127.0.0.1:13305/v1/chat/completions"
    assert seen["timeout"] == 17.5
    assert seen["headers"] == {
        "Accept": "application/json",
        "Content-type": "application/json",
    }
    assert seen["payload"] == {
        "model": "model-a",
        "messages": [{"role": "user", "content": "judge this"}],
        "temperature": 0,
        "max_tokens": 384,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pointwise_judge_decision",
                "strict": True,
                "schema": CONTRACT.schema,
            },
        },
    }


def test_openai_call_rejects_duplicate_response_body_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = (
        b'{"model":"wrong","model":"model-a","choices":'
        b'[{"message":{"content":"{}"}}],"usage":{}}'
    )
    monkeypatch.setattr(
        runner,
        "_open_no_redirect",
        lambda request, timeout: FakeHTTPResponse(raw),
    )

    with pytest.raises(ValueError, match="duplicate"):
        call_openai_judge(
            "http://127.0.0.1:13305/v1",
            "model-a",
            [],
            CONTRACT.schema,
            timeout_seconds=1,
        )


@pytest.mark.parametrize(
    ("body", "message"),
    [
        ({"model": "model-a", "choices": [], "usage": {}}, "choices"),
        ({"model": "model-a", "usage": {}}, "choices"),
        (_response_body(""), "empty"),
        (_response_body("   "), "empty"),
        (_response_body(VALID_DECISION_JSON, model="model-b"), "model"),
        (_response_body(VALID_DECISION_JSON, usage={"total_tokens": "20"}), "usage"),
        (_response_body(VALID_DECISION_JSON, usage={"total_tokens": True}), "usage"),
        (
            _response_body(VALID_DECISION_JSON, usage={"total_tokens": math.nan}),
            "non-finite",
        ),
    ],
)
def test_openai_call_rejects_untrusted_response_shapes(
    monkeypatch: pytest.MonkeyPatch, body: object, message: str
) -> None:
    monkeypatch.setattr(
        runner,
        "_open_no_redirect",
        lambda request, timeout: FakeHTTPResponse(body),
    )

    with pytest.raises(ValueError, match=message):
        call_openai_judge(
            "http://127.0.0.1:13305/v1",
            "model-a",
            [{"role": "user", "content": "judge this"}],
            CONTRACT.schema,
            timeout_seconds=10,
        )


def test_openai_call_rejects_a_redirected_final_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        runner,
        "_open_no_redirect",
        lambda request, timeout: FakeHTTPResponse(
            _response_body(VALID_DECISION_JSON),
            final_url="http://127.0.0.1:9999/v1/chat/completions",
        ),
    )

    with pytest.raises(ValueError, match="redirect"):
        call_openai_judge(
            "http://127.0.0.1:13305/v1",
            "model-a",
            [{"role": "user", "content": "judge this"}],
            CONTRACT.schema,
            timeout_seconds=10,
        )


@pytest.mark.parametrize("content_length", [str(1024 * 1024 + 1), "invalid", "-1"])
def test_openai_call_rejects_invalid_or_oversized_declared_content_length(
    monkeypatch: pytest.MonkeyPatch, content_length: str
) -> None:
    response = FakeHTTPResponse(
        _response_body(VALID_DECISION_JSON), content_length=content_length
    )
    monkeypatch.setattr(runner, "_open_no_redirect", lambda request, timeout: response)

    with pytest.raises(ValueError, match="Content-Length|too large"):
        call_openai_judge(
            "http://127.0.0.1:13305/v1",
            "model-a",
            [{"role": "user", "content": "judge this"}],
            CONTRACT.schema,
            timeout_seconds=10,
        )
    assert response.read_sizes == []


def test_openai_call_bounds_undeclared_response_read_to_max_plus_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeHTTPResponse(_response_body("x" * TEST_MAX_RESPONSE_BYTES))
    monkeypatch.setattr(runner, "_open_no_redirect", lambda request, timeout: response)

    with pytest.raises(ValueError, match="too large"):
        call_openai_judge(
            "http://127.0.0.1:13305/v1",
            "model-a",
            [{"role": "user", "content": "judge this"}],
            CONTRACT.schema,
            timeout_seconds=10,
        )
    assert response.read_sizes == [TEST_MAX_RESPONSE_BYTES + 1]


def test_openai_call_rejects_body_shorter_than_declared_content_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = FakeHTTPResponse(_response_body(VALID_DECISION_JSON))
    response.headers["Content-Length"] = str(len(response.body) + 1)
    monkeypatch.setattr(runner, "_open_no_redirect", lambda request, timeout: response)

    with pytest.raises(ValueError, match="Content-Length"):
        call_openai_judge(
            "http://127.0.0.1:13305/v1",
            "model-a",
            [{"role": "user", "content": "judge this"}],
            CONTRACT.schema,
            timeout_seconds=10,
        )
    assert response.read_sizes == [TEST_MAX_RESPONSE_BYTES + 1]


def test_run_cases_repairs_malformed_json_once_and_preserves_raw_body(
    tmp_path: Path,
) -> None:
    call = ScriptedCall(["not-json", VALID_DECISION_JSON])
    output = tmp_path / "private" / "run.json"

    result = run_cases(
        [CASE],
        model_alias="judge-a",
        model_id="model-a",
        contract=CONTRACT,
        output_path=output,
        runtime_receipt=RECEIPT,
        call=call,
    )

    record = result["records"][CASE.response_id]
    assert record["attempts"][0]["raw"] == "not-json"
    assert record["attempts"][0]["response_body"] == _response_body("not-json")
    assert record["attempts"][1]["parsed"]["label"] == "good"
    assert record["decision"] == VALID_DECISION
    assert record["failure"] is None
    assert len(call.requests) == 2
    repair_messages = call.requests[1]["messages"]
    assert repair_messages[-1]["role"] == "user"
    assert "not-json" in repair_messages[-1]["content"]
    assert "unchanged schema" in repair_messages[-1]["content"]
    assert call.requests[0]["schema"] == CONTRACT.schema
    assert call.requests[1]["schema"] == CONTRACT.schema
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    assert not list(output.parent.glob("*.tmp"))


def test_run_cases_repairs_a_live_decision_with_duplicate_object_keys(
    tmp_path: Path,
) -> None:
    call = ScriptedCall([DUPLICATE_KEY_DECISION_JSON, VALID_DECISION_JSON])

    result = run_cases(
        [CASE],
        model_alias="judge-a",
        model_id="model-a",
        contract=CONTRACT,
        output_path=tmp_path / "run.json",
        runtime_receipt=RECEIPT,
        call=call,
    )

    record = result["records"][CASE.response_id]
    assert record["attempts"][0]["raw"] == DUPLICATE_KEY_DECISION_JSON
    assert "duplicate JSON key" in record["attempts"][0]["error"]
    assert record["decision"] == VALID_DECISION
    assert len(call.requests) == 2


def test_run_cases_rejects_nonprivate_existing_parent_before_cache_read(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    output = parent / "run.json"
    output.write_text("not-json", encoding="utf-8")
    output.chmod(0o600)

    with pytest.raises(ValueError, match="parent.*0700"):
        run_cases(
            [CASE],
            model_alias="judge-a",
            model_id="model-a",
            contract=CONTRACT,
            output_path=output,
            runtime_receipt=RECEIPT,
            call=ScriptedCall([]),
        )
    assert output.read_text(encoding="utf-8") == "not-json"


def test_run_cases_records_explicit_failure_after_exactly_one_repair(
    tmp_path: Path,
) -> None:
    call = ScriptedCall(["not-json", '{"still": "invalid"}'])

    result = run_cases(
        [CASE],
        model_alias="judge-a",
        model_id="model-a",
        contract=CONTRACT,
        output_path=tmp_path / "run.json",
        runtime_receipt=RECEIPT,
        call=call,
    )

    record = result["records"][CASE.response_id]
    assert len(call.requests) == 2
    assert len(record["attempts"]) == 2
    assert record["decision"] is None
    assert record["failure"]["type"] == "invalid_decision_after_repair"
    assert record["failure"]["message"]


def test_interruption_checkpoints_completed_records_and_resume_skips_only_exact_match(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run.json"
    interrupted = ScriptedCall([VALID_DECISION_JSON, KeyboardInterrupt()])

    with pytest.raises(KeyboardInterrupt):
        run_cases(
            [CASE, SECOND_CASE],
            model_alias="judge-a",
            model_id="model-a",
            contract=CONTRACT,
            output_path=output,
            runtime_receipt=RECEIPT,
            call=interrupted,
        )

    checkpoint = json.loads(output.read_text(encoding="utf-8"))
    assert set(checkpoint["records"]) == {CASE.response_id}
    resumed = ScriptedCall([VALID_DECISION_JSON])

    result = run_cases(
        [CASE, SECOND_CASE],
        model_alias="judge-a",
        model_id="model-a",
        contract=CONTRACT,
        output_path=output,
        runtime_receipt=RECEIPT,
        call=resumed,
    )

    assert len(resumed.requests) == 1
    assert SECOND_CASE.question in resumed.requests[0]["messages"][1]["content"]
    assert set(result["records"]) == {CASE.response_id, SECOND_CASE.response_id}


def test_interruption_during_repair_preserves_raw_attempt_and_resumes_with_repair(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run.json"

    with pytest.raises(KeyboardInterrupt):
        run_cases(
            [CASE],
            model_alias="judge-a",
            model_id="model-a",
            contract=CONTRACT,
            output_path=output,
            runtime_receipt=RECEIPT,
            call=ScriptedCall(["not-json", KeyboardInterrupt()]),
        )

    checkpoint = json.loads(output.read_text(encoding="utf-8"))
    partial = checkpoint["records"][CASE.response_id]
    assert partial["attempts"][0]["raw"] == "not-json"
    assert partial["decision"] is None
    assert partial["failure"] is None
    resumed = ScriptedCall([VALID_DECISION_JSON])

    result = run_cases(
        [CASE],
        model_alias="judge-a",
        model_id="model-a",
        contract=CONTRACT,
        output_path=output,
        runtime_receipt=RECEIPT,
        call=resumed,
    )

    assert len(resumed.requests) == 1
    assert "not-json" in resumed.requests[0]["messages"][-1]["content"]
    assert len(result["records"][CASE.response_id]["attempts"]) == 2
    assert result["records"][CASE.response_id]["decision"] == VALID_DECISION


def test_resume_revalidates_cached_duplicate_key_output_and_issues_only_repair(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run.json"
    with pytest.raises(KeyboardInterrupt):
        run_cases(
            [CASE],
            model_alias="judge-a",
            model_id="model-a",
            contract=CONTRACT,
            output_path=output,
            runtime_receipt=RECEIPT,
            call=ScriptedCall([DUPLICATE_KEY_DECISION_JSON, KeyboardInterrupt()]),
        )
    resumed = ScriptedCall([VALID_DECISION_JSON])

    result = run_cases(
        [CASE],
        model_alias="judge-a",
        model_id="model-a",
        contract=CONTRACT,
        output_path=output,
        runtime_receipt=RECEIPT,
        call=resumed,
    )

    assert len(resumed.requests) == 1
    assert DUPLICATE_KEY_DECISION_JSON in resumed.requests[0]["messages"][-1]["content"]
    assert result["records"][CASE.response_id]["decision"] == VALID_DECISION


@pytest.mark.parametrize(
    "changed",
    ["input", "prompt", "model_alias", "model_id", "runtime_receipt"],
)
def test_resume_rejects_record_when_any_content_address_changes(
    tmp_path: Path, changed: str
) -> None:
    output = tmp_path / "run.json"
    run_cases(
        [CASE],
        model_alias="judge-a",
        model_id="model-a",
        contract=CONTRACT,
        output_path=output,
        runtime_receipt=RECEIPT,
        call=ScriptedCall([VALID_DECISION_JSON]),
    )
    cases = [CASE]
    model_alias = "judge-a"
    model_id = "model-a"
    contract = CONTRACT
    receipt = RECEIPT
    if changed == "input":
        cases = [replace(CASE, candidate_answer="A changed answer.")]
    elif changed == "prompt":
        contract = replace(CONTRACT, prompt="A changed frozen prompt.")
    elif changed == "model_alias":
        model_alias = "judge-b"
    elif changed == "model_id":
        model_id = "model-b"
    elif changed == "runtime_receipt":
        receipt = {**RECEIPT, "runtime": "changed"}
    call = ScriptedCall([VALID_DECISION_JSON])

    result = run_cases(
        cases,
        model_alias=model_alias,
        model_id=model_id,
        contract=contract,
        output_path=output,
        runtime_receipt=receipt,
        call=call,
    )

    record = result["records"][CASE.response_id]
    assert len(call.requests) == 1
    assert record["input_sha256"]
    assert record["prompt_sha256"]
    assert record["schema_sha256"]
    assert record["runtime_receipt_sha256"]
    assert record["model_alias"] == model_alias
    assert record["model_id"] == model_id
    assert record["elapsed_seconds"] >= 0
    assert record["usage"]["total_tokens"] == 20


@pytest.mark.parametrize(
    "corruption",
    [
        "decision",
        "usage",
        "raw",
        "body",
        "parsed",
        "attempt_count",
        "unknown_key",
        "invalid_failure",
        "both_outcomes",
    ],
)
def test_resume_fails_closed_on_same_identity_completed_record_corruption(
    tmp_path: Path, corruption: str
) -> None:
    output = tmp_path / "run.json"
    run_cases(
        [CASE],
        model_alias="judge-a",
        model_id="model-a",
        contract=CONTRACT,
        output_path=output,
        runtime_receipt=RECEIPT,
        call=ScriptedCall([VALID_DECISION_JSON]),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    record = payload["records"][CASE.response_id]
    attempt = record["attempts"][0]
    if corruption == "decision":
        record["decision"] = {"tampered": True}
    elif corruption == "usage":
        record["usage"]["total_tokens"] += 1
    elif corruption == "raw":
        attempt["raw"] = json.dumps({**VALID_DECISION, "label": "partial"})
    elif corruption == "body":
        attempt["response_body"]["model"] = "foreign-model"
    elif corruption == "parsed":
        attempt["parsed"] = {**VALID_DECISION, "confidence": "low"}
    elif corruption == "attempt_count":
        record["attempts"].append(attempt)
    elif corruption == "unknown_key":
        record["untrusted"] = True
    elif corruption == "invalid_failure":
        record["decision"] = None
        record["failure"] = {"type": "retry_later", "message": "not terminal"}
    elif corruption == "both_outcomes":
        record["failure"] = {
            "type": "invalid_decision_after_repair",
            "message": "fabricated",
        }
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt reusable record"):
        run_cases(
            [CASE],
            model_alias="judge-a",
            model_id="model-a",
            contract=CONTRACT,
            output_path=output,
            runtime_receipt=RECEIPT,
            call=ScriptedCall([]),
        )


@pytest.mark.parametrize(
    ("location", "field", "tampered"),
    [
        ("attempt", "correctness", True),
        ("attempt", "completeness", False),
        ("decision", "correctness", True),
        ("decision", "completeness", False),
        ("attempt", "correctness", 1.0),
        ("decision", "correctness", math.nan),
    ],
)
def test_resume_rejects_type_confused_or_nonfinite_cached_scores(
    tmp_path: Path, location: str, field: str, tampered: object
) -> None:
    numeric_decision = {
        **VALID_DECISION,
        "label": "incorrect",
        "correctness": 1,
        "completeness": 0,
    }
    output = tmp_path / "run.json"
    run_cases(
        [CASE],
        model_alias="judge-a",
        model_id="model-a",
        contract=CONTRACT,
        output_path=output,
        runtime_receipt=RECEIPT,
        call=ScriptedCall([json.dumps(numeric_decision)]),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    record = payload["records"][CASE.response_id]
    target = (
        record["attempts"][0]["parsed"] if location == "attempt" else record[location]
    )
    target[field] = tampered
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt reusable record"):
        run_cases(
            [CASE],
            model_alias="judge-a",
            model_id="model-a",
            contract=CONTRACT,
            output_path=output,
            runtime_receipt=RECEIPT,
            call=ScriptedCall([]),
        )


def test_resume_reuses_type_correct_cached_integer_scores_without_call(
    tmp_path: Path,
) -> None:
    numeric_decision = {
        **VALID_DECISION,
        "label": "incorrect",
        "correctness": 1,
        "completeness": 0,
    }
    output = tmp_path / "run.json"
    expected = run_cases(
        [CASE],
        model_alias="judge-a",
        model_id="model-a",
        contract=CONTRACT,
        output_path=output,
        runtime_receipt=RECEIPT,
        call=ScriptedCall([json.dumps(numeric_decision)]),
    )

    actual = run_cases(
        [CASE],
        model_alias="judge-a",
        model_id="model-a",
        contract=CONTRACT,
        output_path=output,
        runtime_receipt=RECEIPT,
        call=ScriptedCall([]),
    )

    assert actual["records"][CASE.response_id] == expected["records"][CASE.response_id]


def test_resume_fails_closed_on_same_identity_partial_record_corruption(
    tmp_path: Path,
) -> None:
    output = tmp_path / "run.json"
    with pytest.raises(KeyboardInterrupt):
        run_cases(
            [CASE],
            model_alias="judge-a",
            model_id="model-a",
            contract=CONTRACT,
            output_path=output,
            runtime_receipt=RECEIPT,
            call=ScriptedCall(["not-json", KeyboardInterrupt()]),
        )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["records"][CASE.response_id]["attempts"][0]["parsed"] = VALID_DECISION
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt reusable record"):
        run_cases(
            [CASE],
            model_alias="judge-a",
            model_id="model-a",
            contract=CONTRACT,
            output_path=output,
            runtime_receipt=RECEIPT,
            call=ScriptedCall([]),
        )


@pytest.mark.parametrize(
    "corruption", ["unknown_key", "wrong_version", "record_conflict"]
)
def test_resume_fails_closed_on_corrupt_run_container(
    tmp_path: Path, corruption: str
) -> None:
    output = tmp_path / "run.json"
    run_cases(
        [CASE],
        model_alias="judge-a",
        model_id="model-a",
        contract=CONTRACT,
        output_path=output,
        runtime_receipt=RECEIPT,
        call=ScriptedCall([VALID_DECISION_JSON]),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    if corruption == "unknown_key":
        payload["untrusted"] = True
    elif corruption == "wrong_version":
        payload["version"] = 2
    else:
        payload["model_id"] = "foreign-model"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="corrupt existing run"):
        run_cases(
            [CASE],
            model_alias="judge-a",
            model_id="model-a",
            contract=CONTRACT,
            output_path=output,
            runtime_receipt=RECEIPT,
            call=ScriptedCall([]),
        )
