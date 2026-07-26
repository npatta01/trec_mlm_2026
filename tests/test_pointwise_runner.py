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
from trec_mlm.pointwise_judge import CandidateCase, PromptContract
from trec_mlm.pointwise_runner import (
    call_openai_judge,
    run_cases,
    validate_loopback_endpoint,
)


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
CONTRACT = PromptContract(
    prompt_path=Path("pointwise-v1.txt"),
    schema_path=Path("decision-v1.schema.json"),
    prompt="Judge the candidate under the frozen contract.",
    schema={
        "type": "object",
        "additionalProperties": False,
        "required": ["label"],
        "properties": {"label": {"enum": ["good", "partial"]}},
    },
)
RECEIPT = {"runtime": "local-test", "gpu": "fake", "passed": True}
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
    ) -> None:
        self.body = json.dumps(body).encode("utf-8")
        self.status = status
        self.final_url = final_url

    def __enter__(self) -> FakeHTTPResponse:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def geturl(self) -> str:
        return self.final_url

    def read(self) -> bytes:
        return self.body


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
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pointwise_judge_decision",
                "strict": True,
                "schema": CONTRACT.schema,
            },
        },
    }


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
            "usage",
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


@pytest.mark.parametrize(
    "changed",
    ["input", "prompt", "schema", "model_alias", "model_id", "runtime_receipt"],
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
    elif changed == "schema":
        contract = replace(CONTRACT, schema={"type": "object"})
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
