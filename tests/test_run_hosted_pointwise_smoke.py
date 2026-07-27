from __future__ import annotations

import json
import os
from pathlib import Path
import site
import stat
import subprocess
import sys
import urllib.error

import pytest

import scripts.run_hosted_pointwise_smoke as hosted


VALID_DECISION = {
    "label": "good",
    "correctness": 4,
    "completeness": 2,
    "relevance": 2,
    "error_flags": [],
    "reference_status": "sound",
    "confidence": "high",
    "rationale": "The candidate directly and correctly answers the question.",
}


def _router_metadata(
    spec: hosted.ModelSpec,
    *,
    provider: str | None = None,
    available: list[dict[str, object]] | None = None,
    attempt: int = 1,
) -> dict[str, object]:
    return {
        "requested": spec.model_id,
        "attempt": attempt,
        "endpoints": {
            "total": 1 if available is None else len(available),
            "available": available
            if available is not None
            else [
                {
                    "provider": provider or spec.provider,
                    "model": spec.model_id,
                    "selected": True,
                }
            ],
        },
    }


def test_model_registry_pins_the_four_authorized_routes() -> None:
    assert set(hosted.MODEL_SPECS) == {
        "deepseek/deepseek-v4-flash",
        "nvidia/nemotron-3-super-120b-a12b",
        "qwen/qwen3-235b-a22b-2507",
        "openai/gpt-5.4-nano",
    }
    assert hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"].provider == "DeepInfra"
    assert (
        hosted.MODEL_SPECS["nvidia/nemotron-3-super-120b-a12b"].provider == "DeepInfra"
    )
    assert hosted.MODEL_SPECS["qwen/qwen3-235b-a22b-2507"].provider == "DeepInfra"
    assert hosted.MODEL_SPECS["openai/gpt-5.4-nano"].provider == "OpenAI"


@pytest.mark.parametrize(
    "model",
    list(
        {
            "deepseek/deepseek-v4-flash",
            "nvidia/nemotron-3-super-120b-a12b",
        }
    ),
)
def test_payload_disables_reasoning_for_reasoning_models(model: str) -> None:
    payload = hosted._build_payload(
        hosted.MODEL_SPECS[model],
        [{"role": "user", "content": "judge"}],
        {"type": "object"},
    )
    assert payload["reasoning"] == {"enabled": False}


def test_gpt_payload_selects_no_reasoning() -> None:
    spec = hosted.MODEL_SPECS["openai/gpt-5.4-nano"]
    payload = hosted._build_payload(
        spec, [{"role": "user", "content": "judge"}], {"type": "object"}
    )
    assert payload["reasoning"] == {"effort": "none"}


def test_qwen_payload_omits_reasoning_and_every_route_is_strict_and_pinned() -> None:
    spec = hosted.MODEL_SPECS["qwen/qwen3-235b-a22b-2507"]
    payload = hosted._build_payload(
        spec, [{"role": "user", "content": "judge"}], {"type": "object"}
    )
    assert "reasoning" not in payload
    assert payload["max_tokens"] == 512
    assert payload["temperature"] == 0
    assert payload["stream"] is False
    assert payload["usage"] == {"include": True}
    assert payload["provider"] == {
        "only": ["DeepInfra"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "max_price": {"prompt": 0.09, "completion": 0.55},
    }
    assert payload["response_format"]["json_schema"]["strict"] is True


def test_api_key_prefers_environment_without_reading_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "missing"
    monkeypatch.setenv("OPENROUTER_API_KEY", "environment-secret")
    assert hosted._load_api_key(missing) == "environment-secret"


def test_api_key_file_must_be_regular_mode_0600(tmp_path: Path) -> None:
    key_file = tmp_path / "openrouter.key"
    key_file.write_text("file-secret\n", encoding="utf-8")
    key_file.chmod(0o600)
    assert hosted._load_api_key(key_file, environ={}) == "file-secret"
    key_file.chmod(0o644)
    with pytest.raises(ValueError, match="0600"):
        hosted._load_api_key(key_file, environ={})


def test_budget_refuses_before_a_request_that_could_cross_cap() -> None:
    budget = hosted.SpendBudget(cap_usd=0.10)
    budget.charge(0.099)
    with pytest.raises(RuntimeError, match="budget"):
        budget.reserve(0.002)
    assert budget.spent_usd == pytest.approx(0.099)


def test_budget_cannot_be_raised_above_the_authorized_hard_cap() -> None:
    with pytest.raises(ValueError, match="authorized"):
        hosted.SpendBudget(0.100001)


def test_spend_ledger_is_private_and_restores_interrupted_run_budget(
    tmp_path: Path,
) -> None:
    path = tmp_path / "state" / "spend-ledger.json"
    hosted._write_spend_ledger(path, hosted.SpendBudget(0.10, spent_usd=0.023))
    restored = hosted._load_spend_ledger(path, cap_usd=0.10)
    assert restored.spent_usd == pytest.approx(0.023)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_call_checkpoints_spend_immediately_after_provider_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    checkpoints: list[float] = []
    request_headers: list[dict[str, str]] = []
    body = {
        "model": spec.model_id,
        "openrouter_metadata": _router_metadata(spec),
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(VALID_DECISION)},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cost": 0.0001,
        },
    }

    def successful_request(request: object, **_kwargs: object) -> object:
        request_headers.append(
            {key.lower(): value for key, value in request.header_items()}
        )
        return body

    monkeypatch.setattr(hosted, "_post_json", successful_request)
    budget = hosted.SpendBudget(0.10)
    call = hosted.OpenRouterJudgeCall(
        spec=spec,
        api_key="test-only-secret",
        budget=budget,
        timeout_seconds=1,
        checkpoint_spend=lambda: checkpoints.append(budget.spent_usd),
    )
    call(spec.model_id, [{"role": "user", "content": "judge"}], {"type": "object"})
    assert len(checkpoints) == 2
    assert checkpoints[0] > 0.0001
    assert checkpoints[1] == pytest.approx(0.0001)
    assert request_headers[0]["x-openrouter-metadata"] == "enabled"


def test_invalid_response_keeps_durable_worst_case_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    checkpoints: list[float] = []
    body = {
        "model": spec.model_id,
        "openrouter_metadata": _router_metadata(spec, provider="WrongProvider"),
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(VALID_DECISION)},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cost": 0.0001,
        },
    }
    monkeypatch.setattr(hosted, "_post_json", lambda *_args, **_kwargs: body)
    budget = hosted.SpendBudget(0.10)
    call = hosted.OpenRouterJudgeCall(
        spec=spec,
        api_key="test-only-secret",
        budget=budget,
        timeout_seconds=1,
        checkpoint_spend=lambda: checkpoints.append(budget.spent_usd),
    )
    with pytest.raises(ValueError, match="provider"):
        call(
            spec.model_id,
            [{"role": "user", "content": "judge"}],
            {"type": "object"},
        )
    assert checkpoints == [pytest.approx(budget.spent_usd)]
    assert budget.spent_usd > 0.0001


def test_retry_durably_reserves_each_attempt_before_reconciling_valid_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    checkpoints: list[float] = []
    calls = 0
    body = {
        "model": spec.model_id,
        "openrouter_metadata": _router_metadata(spec),
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(VALID_DECISION)},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cost": 0.0001,
        },
    }

    def retry_once(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError("url", 429, "busy", {}, None)
        return body

    monkeypatch.setattr(hosted, "_post_json", retry_once)
    budget = hosted.SpendBudget(0.10)
    call = hosted.OpenRouterJudgeCall(
        spec=spec,
        api_key="test-only-secret",
        budget=budget,
        timeout_seconds=1,
        checkpoint_spend=lambda: checkpoints.append(budget.spent_usd),
    )
    call(
        spec.model_id,
        [{"role": "user", "content": "judge"}],
        {"type": "object"},
    )
    assert len(checkpoints) == 3
    one_reservation = checkpoints[0]
    assert checkpoints[1] == pytest.approx(2 * one_reservation)
    assert checkpoints[2] == pytest.approx(one_reservation + 0.0001)


def test_oversized_serialized_input_is_rejected_before_network_or_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    network_calls: list[object] = []
    checkpoints: list[float] = []
    monkeypatch.setattr(
        hosted,
        "_post_json",
        lambda *args, **kwargs: network_calls.append((args, kwargs)),
    )
    budget = hosted.SpendBudget(0.10)
    call = hosted.OpenRouterJudgeCall(
        spec=spec,
        api_key="test-only-secret",
        budget=budget,
        timeout_seconds=1,
        checkpoint_spend=lambda: checkpoints.append(budget.spent_usd),
    )
    with pytest.raises(ValueError, match="serialized input"):
        call(
            spec.model_id,
            [{"role": "user", "content": "x" * 100_000}],
            {"type": "object"},
        )
    assert network_calls == []
    assert checkpoints == []
    assert budget.spent_usd == 0


def test_response_rejects_provider_or_model_route_changes() -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    body = {
        "model": spec.model_id,
        "openrouter_metadata": _router_metadata(spec),
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(VALID_DECISION)},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cost": 0.001,
        },
    }
    assert hosted._parse_response(body, spec)[0] == json.dumps(VALID_DECISION)
    with pytest.raises(ValueError, match="provider"):
        hosted._parse_response(
            {
                **body,
                "openrouter_metadata": _router_metadata(spec, provider="Other"),
            },
            spec,
        )
    with pytest.raises(ValueError, match="model"):
        hosted._parse_response({**body, "model": "other/model"}, spec)


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {
            "requested": "wrong/model",
            "attempt": 1,
            "endpoints": {"total": 0, "available": []},
        },
        {
            "requested": "deepseek/deepseek-v4-flash",
            "attempt": 2,
            "endpoints": {
                "total": 1,
                "available": [
                    {
                        "provider": "DeepInfra",
                        "model": "deepseek/deepseek-v4-flash",
                        "selected": True,
                    }
                ],
            },
        },
        {
            "requested": "deepseek/deepseek-v4-flash",
            "attempt": 1,
            "endpoints": {
                "total": 2,
                "available": [
                    {
                        "provider": "DeepInfra",
                        "model": "deepseek/deepseek-v4-flash",
                        "selected": True,
                    },
                    {
                        "provider": "Other",
                        "model": "deepseek/deepseek-v4-flash",
                        "selected": True,
                    },
                ],
            },
        },
    ],
)
def test_response_rejects_missing_wrong_or_ambiguous_router_metadata(
    metadata: object,
) -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    body = {
        "model": spec.model_id,
        "openrouter_metadata": metadata,
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps(VALID_DECISION)},
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cost": 0.001,
        },
    }
    with pytest.raises(ValueError, match="metadata|requested|attempt|selected"):
        hosted._parse_response(body, spec)


def test_response_rejects_length_stop_and_missing_numeric_cost() -> None:
    spec = hosted.MODEL_SPECS["openai/gpt-5.4-nano"]
    body = {
        "model": spec.model_id,
        "openrouter_metadata": _router_metadata(spec),
        "choices": [{"finish_reason": "length", "message": {"content": "{}"}}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 512,
            "total_tokens": 612,
            "cost": 0.01,
        },
    }
    with pytest.raises(ValueError, match="finish_reason"):
        hosted._parse_response(body, spec)
    body["choices"][0]["finish_reason"] = "stop"
    body["usage"].pop("cost")
    with pytest.raises(ValueError, match="cost"):
        hosted._parse_response(body, spec)


def test_transport_retries_once_only_for_retryable_http_statuses() -> None:
    calls: list[int] = []

    def retry_then_succeed(*_args: object, **_kwargs: object):
        calls.append(1)
        if len(calls) == 1:
            raise urllib.error.HTTPError("url", 429, "busy", {}, None)
        return {"ok": True}

    assert hosted._request_with_retry(
        retry_then_succeed, object(), sleep=lambda _: None
    ) == {"ok": True}
    assert len(calls) == 2

    def forbidden(*_args: object, **_kwargs: object):
        raise urllib.error.HTTPError("url", 403, "forbidden", {}, None)

    with pytest.raises(urllib.error.HTTPError):
        hosted._request_with_retry(forbidden, object(), sleep=lambda _: None)


def test_private_json_is_atomic_mode_0600_and_does_not_contain_auth(
    tmp_path: Path,
) -> None:
    path = tmp_path / "private" / "receipt.json"
    hosted._write_private_json(path, {"model": "example", "secret": None})
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert "Bearer" not in path.read_text(encoding="utf-8")


def test_canary_failure_stops_before_the_remaining_nine_cases(
    tmp_path: Path,
) -> None:
    cases = [object() for _ in range(10)]
    seen: list[int] = []

    def failed_canary(selected: list[object]) -> dict[str, object]:
        seen.append(len(selected))
        return {"records": {"first": {"decision": None, "failure": {"type": "bad"}}}}

    with pytest.raises(RuntimeError, match="canary"):
        hosted._run_canary_then_remaining(cases, failed_canary)
    assert seen == [1]


def test_canary_success_resumes_with_all_ten_cases() -> None:
    cases = [object() for _ in range(10)]
    seen: list[int] = []

    def successful(selected: list[object]) -> dict[str, object]:
        seen.append(len(selected))
        return {"records": {"first": {"decision": VALID_DECISION, "failure": None}}}

    hosted._run_canary_then_remaining(cases, successful)
    assert seen == [1, 10]


def test_default_key_path_is_private_user_config() -> None:
    assert hosted.DEFAULT_API_KEY_FILE == Path(
        "/home/npatta01/.config/trec-mlm/openrouter.key"
    )
    assert hosted.OPENROUTER_ENDPOINT == "https://openrouter.ai/api/v1/chat/completions"
    assert hosted.DEFAULT_CAP_USD == pytest.approx(0.10)


def test_script_entry_point_can_show_help_from_repository_root() -> None:
    result = subprocess.run(
        [sys.executable, "-S", "scripts/run_hosted_pointwise_smoke.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": os.environ["PATH"],
            "PYTHONPATH": site.getsitepackages()[0],
        },
    )
    assert result.returncode == 0, result.stderr
    assert "--manifest-sha256" in result.stdout
