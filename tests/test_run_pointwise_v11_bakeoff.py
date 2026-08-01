from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import stat
import pytest
import requests

from scripts import run_pointwise_v11_bakeoff as runner


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _record(ordinal: int = 7, *, idk: bool = False) -> dict[str, object]:
    return {
        "ordinal": ordinal,
        "question": "Who wrote Hamlet?",
        "evaluator_evidence": "William Shakespeare wrote Hamlet.",
        "historical_response": "I don't know" if idk else "Shakespeare wrote it.",
        "expected_v11_derived_label": "nonanswer" if idk else "correct",
        "evidence_tier": "generated_without_search",
        "deterministic_no_provider_call": idk,
    }


def _manifest(records: list[dict[str, object]]) -> dict[str, object]:
    body: dict[str, object] = {
        "version": 1,
        "record_count": len(records),
        "historical_response_transform": {
            "version": "v11-historical-response-citation-markers-1",
            "sha256": "a" * 64,
        },
        "records": records,
    }
    return body | {"manifest_sha256": _sha256_json(body)}


def _write_private_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, mode=0o700)
    path.parent.chmod(0o700)
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)


def _contract() -> object:
    return runner.load_prompt_contract(Path("protocols/pointwise_judge_bakeoff"), version="v11")


def _provider_response() -> dict[str, object]:
    return {
        "model": "fixture-model",
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "content": json.dumps(
                        {
                            "central_claim": "Shakespeare wrote Hamlet",
                            "answer_present": True,
                            "supported_core": True,
                            "material_conflict": False,
                        }
                    )
                },
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": 4, "cost": 0.001},
    }


def test_arms_pin_exact_endpoints_models_routes_and_request_parameters() -> None:
    assert {arm.name for arm in runner.ARMS.values()} == {
        "deepseek-v4-flash-thinking",
        "gpt-5.4-nano-direct",
        "qwen3.6-35b-a3b-thinking",
    }
    assert runner.ARMS["deepseek-v4-flash-thinking"].as_identity() == {
        "endpoint": "https://api.deepinfra.com/v1/openai/chat/completions",
        "model": "deepseek-ai/DeepSeek-V4-Flash",
        "provider": "DeepInfra",
        "provider_order": None,
        "reasoning_effort": "high",
        "reasoning_enabled": False,
        "max_tokens": 8192,
        "workers": 8,
        "temperature": 0,
    }
    assert runner.ARMS["gpt-5.4-nano-direct"].as_identity() == {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "model": "openai/gpt-5.4-nano",
        "provider": "OpenRouter",
        "provider_order": "OpenAI",
        "reasoning_effort": None,
        "reasoning_enabled": False,
        "max_tokens": 1024,
        "workers": 4,
        "temperature": None,
    }
    assert runner.ARMS["qwen3.6-35b-a3b-thinking"].as_identity() == {
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
        "model": "qwen/qwen3.6-35b-a3b",
        "provider": "OpenRouter",
        "provider_order": "AkashML",
        "reasoning_effort": None,
        "reasoning_enabled": True,
        "max_tokens": 4096,
        "workers": 4,
        "temperature": 0,
    }


def test_provider_message_is_a_three_field_citation_free_fixture() -> None:
    message = runner._user_message(_record())

    assert message == (
        "ORIGINAL QUESTION:\nWho wrote Hamlet?\n\n"
        "GEMINI EVALUATOR EVIDENCE — RETROSPECTIVE JUDGE ONLY:\n"
        "William Shakespeare wrote Hamlet.\n\n"
        "HISTORICAL RESPONSE:\nShakespeare wrote it."
    )
    lowered = message.casefold()
    for forbidden in ("ordinal", "record_id", "response_id", "model", "source", "http"):
        assert forbidden not in lowered


@pytest.mark.parametrize(
    ("arm_name", "temperature", "provider", "reasoning"),
    [
        ("deepseek-v4-flash-thinking", 0, None, None),
        (
            "gpt-5.4-nano-direct",
            None,
            {"only": ["OpenAI"], "allow_fallbacks": False, "require_parameters": True},
            None,
        ),
        (
            "qwen3.6-35b-a3b-thinking",
            0,
            {"only": ["AkashML"], "allow_fallbacks": False, "require_parameters": True},
            {"enabled": True},
        ),
    ],
)
def test_request_body_is_strict_and_uses_only_supported_temperature(
    arm_name: str, temperature: object, provider: object, reasoning: object
) -> None:
    body = runner._build_body(runner.ARMS[arm_name], _contract(), _record())

    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "pointwise_judge_v11_decision",
            "strict": True,
            "schema": _contract().schema,
        },
    }
    assert body["stream"] is False
    assert body["model"] == runner.ARMS[arm_name].model
    assert body.get("temperature") == temperature
    assert body.get("provider") == provider
    assert body.get("reasoning") == reasoning
    assert body.get("reasoning_effort") == runner.ARMS[arm_name].reasoning_effort


def test_retry_replays_one_identical_request_only_for_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    body = {"model": "fixed", "messages": []}
    calls: list[dict[str, object]] = []

    def transient(value: dict[str, object], **_: object) -> dict[str, object]:
        calls.append(value)
        if len(calls) == 1:
            raise runner.ProviderHTTPError(429)
        return {"ok": True}

    monkeypatch.setattr(runner, "_post_once", transient)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    assert runner._post_with_one_retry(body, token="secret", timeout_seconds=2) == ({"ok": True}, 2)
    assert calls == [body, body]


@pytest.mark.parametrize(
    "failure",
    [runner.ProviderHTTPError(400), requests.Timeout("fixture"), requests.ConnectionError("fixture")],
)
def test_retry_never_changes_route_and_stops_after_one_retry(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    calls: list[dict[str, object]] = []

    def fail(body: dict[str, object], **_: object) -> dict[str, object]:
        calls.append(body)
        raise failure

    monkeypatch.setattr(runner, "_post_once", fail)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)

    with pytest.raises(type(failure)):
        runner._post_with_one_retry({"model": "fixed"}, token="secret", timeout_seconds=2)
    assert len(calls) == (1 if isinstance(failure, runner.ProviderHTTPError) else 2)


def test_cache_identity_binds_manifest_contract_transform_route_parameters_and_record() -> None:
    metadata = runner._metadata(
        arm=runner.ARMS["qwen3.6-35b-a3b-thinking"],
        manifest=_manifest([_record()]),
        contract=_contract(),
        selected_ordinals=[7],
        max_calls=1,
        max_cost_usd=1.0,
        timeout_seconds=20,
    )
    identity = runner._record_cache_identity(metadata, _record())
    assert identity != runner._record_cache_identity({**metadata, "manifest_sha256": "b" * 64}, _record())
    assert identity != runner._record_cache_identity({**metadata, "prompt_sha256": "c" * 64}, _record())
    assert identity != runner._record_cache_identity({**metadata, "schema_sha256": "d" * 64}, _record())
    assert identity != runner._record_cache_identity({**metadata, "historical_response_transform_sha256": "e" * 64}, _record())
    assert identity != runner._record_cache_identity({**metadata, "route": {"model": "other"}}, _record())
    assert identity != runner._record_cache_identity(metadata, _record(ordinal=8))


def test_runner_preflights_caps_and_writes_private_resumable_checkpoints(tmp_path: Path) -> None:
    manifest = _manifest([_record(), _record(ordinal=8, idk=True)])
    output = tmp_path / "run"
    calls: list[dict[str, object]] = []

    def post(body: dict[str, object], **_: object) -> dict[str, object]:
        calls.append(body)
        return _provider_response()

    result = runner.run_bakeoff(
        arm=runner.ARMS["gpt-5.4-nano-direct"],
        manifest=manifest,
        contract=_contract(),
        selected_ordinals=[7, 8],
        output_dir=output,
        max_calls=1,
        max_cost_usd=1.0,
        token="secret",
        timeout_seconds=20,
        post=post,
    )
    assert len(calls) == 1
    assert result["summary"] == {
        "provider_calls": 1,
        "deterministic_nonanswers": 1,
        "provider_errors": 0,
        "schema_errors": 0,
        "estimated_cost_usd": 0.001,
    }
    idk = json.loads((output / "records" / "000008.json").read_text())
    assert idk["label"] == "nonanswer"
    assert idk["provider_call"] is False
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "records").stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "metadata.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "records" / "000007.json").stat().st_mode) == 0o600

    runner.run_bakeoff(
        arm=runner.ARMS["gpt-5.4-nano-direct"],
        manifest=manifest,
        contract=_contract(),
        selected_ordinals=[7, 8],
        output_dir=output,
        max_calls=1,
        max_cost_usd=1.0,
        token="secret",
        timeout_seconds=20,
        post=post,
    )
    assert len(calls) == 1


def test_runner_fails_before_provider_call_for_bad_budget_or_bad_evidence(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def post(body: dict[str, object], **_: object) -> dict[str, object]:
        calls.append(body)
        return _provider_response()

    with pytest.raises(ValueError, match="max-calls"):
        runner.run_bakeoff(
            arm=runner.ARMS["deepseek-v4-flash-thinking"],
            manifest=_manifest([_record(), _record(ordinal=8)]),
            contract=_contract(),
            selected_ordinals=[7, 8],
            output_dir=tmp_path / "budget",
            max_calls=1,
            max_cost_usd=1.0,
            token="secret",
            timeout_seconds=20,
            post=post,
        )
    invalid = _record()
    invalid["evaluator_evidence"] = "Evidence at https://example.test"
    with pytest.raises(ValueError, match="evidence"):
        runner.run_bakeoff(
            arm=runner.ARMS["deepseek-v4-flash-thinking"],
            manifest=_manifest([invalid]),
            contract=_contract(),
            selected_ordinals=[7],
            output_dir=tmp_path / "evidence",
            max_calls=1,
            max_cost_usd=1.0,
            token="secret",
            timeout_seconds=20,
            post=post,
        )
    assert calls == []


def test_frozen_private_manifest_passes_runner_preflight_without_a_provider_call() -> None:
    manifest = json.loads(
        Path("artifacts/pointwise_judge_bakeoff/v11/adjudication/manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert len(runner._validate_manifest(manifest)) == 282


def test_runner_rejects_tampered_schema_or_unpinned_route_before_provider_call(
    tmp_path: Path,
) -> None:
    calls: list[dict[str, object]] = []

    def post(body: dict[str, object], **_: object) -> dict[str, object]:
        calls.append(body)
        return _provider_response()

    common = {
        "manifest": _manifest([_record()]),
        "selected_ordinals": [7],
        "output_dir": tmp_path / "run",
        "max_calls": 1,
        "max_cost_usd": 1.0,
        "token": "secret",
        "timeout_seconds": 20,
        "post": post,
    }
    with pytest.raises(ValueError, match="schema"):
        runner.run_bakeoff(
            arm=runner.ARMS["deepseek-v4-flash-thinking"],
            contract=replace(_contract(), schema={}),
            **common,
        )
    with pytest.raises(ValueError, match="pinned"):
        runner.run_bakeoff(
            arm=replace(runner.ARMS["deepseek-v4-flash-thinking"], model="other"),
            contract=_contract(),
            **common,
        )
    assert calls == []
