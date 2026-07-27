from __future__ import annotations

import copy
from dataclasses import replace
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
from trec_mlm.pointwise_judge import CandidateCase, load_prompt_contract
from trec_mlm.pointwise_runner import run_cases, validate_frozen_judge_contract


FROZEN_CONTRACT = load_prompt_contract(Path("protocols/pointwise_judge_bakeoff"))
FROZEN_SCHEMA = FROZEN_CONTRACT.schema
V2_CONTRACT = load_prompt_contract(
    Path("protocols/pointwise_judge_bakeoff"), version="v2"
)
V2_SCHEMA = V2_CONTRACT.schema


VALID_DECISION = {
    "label": "good",
    "correctness": 4,
    "completeness": 2,
    "relevance": 2,
    "error_flags": [],
    "reference_status": "sound",
    "confidence": "high",
    "rationale": "The candidate directly and correctly answers the question in full.",
}


def test_v2_registry_has_four_distinct_configurations() -> None:
    specs = hosted.V2_CONFIG_SPECS
    assert set(specs) == {
        "deepseek-v4-flash-direct",
        "gpt-5.4-nano-direct",
        "qwen3.6-35b-a3b-direct",
        "qwen3.6-35b-a3b-thinking",
    }
    assert (
        specs["qwen3.6-35b-a3b-direct"].model_id
        == specs["qwen3.6-35b-a3b-thinking"].model_id
        == "qwen/qwen3.6-35b-a3b"
    )
    assert (
        specs["qwen3.6-35b-a3b-direct"].provider
        == specs["qwen3.6-35b-a3b-thinking"].provider
        == "AkashML"
    )


def test_qwen_pair_differs_only_by_reasoning() -> None:
    messages = [{"role": "user", "content": "judge"}]
    direct = hosted._build_payload(
        hosted.V2_CONFIG_SPECS["qwen3.6-35b-a3b-direct"], messages, V2_SCHEMA
    )
    thinking = hosted._build_payload(
        hosted.V2_CONFIG_SPECS["qwen3.6-35b-a3b-thinking"], messages, V2_SCHEMA
    )
    assert direct["reasoning"] == {"enabled": False}
    assert thinking["reasoning"] == {"enabled": True}
    assert {key: value for key, value in direct.items() if key != "reasoning"} == {
        key: value for key, value in thinking.items() if key != "reasoning"
    }


def test_v2_configuration_rejects_a_v1_source_schema() -> None:
    with pytest.raises(ValueError, match="v2.*schema|schema.*v2"):
        hosted._build_payload(
            hosted.V2_CONFIG_SPECS["deepseek-v4-flash-direct"],
            [{"role": "user", "content": "judge"}],
            FROZEN_SCHEMA,
        )


def test_attempt_budget_stops_before_a_physical_request_over_the_ceiling() -> None:
    budget = hosted.AttemptBudget(cap_calls=2)
    budget.reserve()
    budget.reserve()
    with pytest.raises(RuntimeError, match="call ceiling"):
        budget.reserve()
    assert budget.used_calls == 2


def test_attempt_ledger_restores_interrupted_count_privately(tmp_path: Path) -> None:
    path = tmp_path / "attempt-ledger.json"
    hosted._write_attempt_ledger(path, hosted.AttemptBudget(100, 17))
    restored = hosted._load_attempt_ledger(path, cap_calls=100)
    assert restored.used_calls == 17
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_retry_persists_each_attempt_before_the_physical_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = hosted.V2_CONFIG_SPECS["deepseek-v4-flash-direct"]
    ledger_path = tmp_path / "attempt-ledger.json"
    budget = hosted.AttemptBudget(2)
    counts_seen_by_network: list[int] = []
    body = {
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
        counts_seen_by_network.append(
            json.loads(ledger_path.read_text(encoding="utf-8"))["used_calls"]
        )
        if len(counts_seen_by_network) == 1:
            raise urllib.error.HTTPError("url", 429, "busy", {}, None)
        return body

    monkeypatch.setattr(hosted, "_post_json", retry_once)
    call = hosted.OpenRouterJudgeCall(
        spec=spec,
        api_key="test-only-secret",
        budget=hosted.SpendBudget(0.10),
        attempt_budget=budget,
        timeout_seconds=1,
        checkpoint_attempt=lambda: hosted._write_attempt_ledger(ledger_path, budget),
    )
    call(spec.model_id, [{"role": "user", "content": "judge"}], V2_SCHEMA)

    assert counts_seen_by_network == [1, 2]
    assert budget.used_calls == 2


def test_call_ceiling_prevents_the_next_physical_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = hosted.V2_CONFIG_SPECS["deepseek-v4-flash-direct"]
    ledger_path = tmp_path / "attempt-ledger.json"
    budget = hosted.AttemptBudget(1)
    physical_sends = 0
    spend_checkpoints: list[float] = []
    spend_budget = hosted.SpendBudget(0.10)

    def always_retry(*_args: object, **_kwargs: object) -> object:
        nonlocal physical_sends
        physical_sends += 1
        raise urllib.error.HTTPError("url", 429, "busy", {}, None)

    monkeypatch.setattr(hosted, "_post_json", always_retry)
    call = hosted.OpenRouterJudgeCall(
        spec=spec,
        api_key="test-only-secret",
        budget=spend_budget,
        attempt_budget=budget,
        timeout_seconds=1,
        checkpoint_spend=lambda: spend_checkpoints.append(spend_budget.spent_usd),
        checkpoint_attempt=lambda: hosted._write_attempt_ledger(ledger_path, budget),
    )
    with pytest.raises(RuntimeError, match="call ceiling"):
        call(spec.model_id, [{"role": "user", "content": "judge"}], V2_SCHEMA)

    assert physical_sends == 1
    assert budget.used_calls == 1
    assert len(spend_checkpoints) == 1


def test_v2_call_requires_the_durable_attempt_budget() -> None:
    with pytest.raises(ValueError, match="attempt budget"):
        hosted.OpenRouterJudgeCall(
            spec=hosted.V2_CONFIG_SPECS["deepseek-v4-flash-direct"],
            api_key="test-only-secret",
            budget=hosted.SpendBudget(0.10),
            timeout_seconds=1,
        )


def test_v2_receipt_binds_config_prompt_schema_and_transport_identity() -> None:
    spec = hosted.V2_CONFIG_SPECS["qwen3.6-35b-a3b-thinking"]
    receipt = hosted._runtime_receipt(spec, V2_CONTRACT, version="v2")
    assert receipt["config_id"] == "qwen3.6-35b-a3b-thinking"
    assert receipt["model_id"] == "qwen/qwen3.6-35b-a3b"
    assert receipt["provider"] == "AkashML"
    assert receipt["reasoning"] == {"enabled": True}
    assert (
        receipt["prompt_sha256"]
        == hosted.hashlib.sha256(V2_CONTRACT.prompt.encode("utf-8")).hexdigest()
    )
    assert receipt["schema_sha256"] == hosted._sha256_json(V2_SCHEMA)
    assert receipt["transport_schema_omitted_json_paths"] == [
        "$.properties.error_flags.uniqueItems"
    ]
    assert receipt["transport_schema_sha256"] == hosted._sha256_json(
        hosted._openrouter_transport_schema(V2_SCHEMA)
    )


def test_v2_config_filter_uses_distinct_ids_in_registry_order(tmp_path: Path) -> None:
    requested = ["qwen3.6-35b-a3b-thinking", "qwen3.6-35b-a3b-direct"]
    args = hosted._parser().parse_args(
        [
            *_required_cli_args(tmp_path),
            "--config",
            requested[0],
            "--config",
            requested[1],
        ]
    )
    assert args.config == requested
    assert [spec.config_id for spec in hosted._select_config_specs(args.config)] == [
        "qwen3.6-35b-a3b-direct",
        "qwen3.6-35b-a3b-thinking",
    ]


def test_v2_diagnostic_paths_are_keyed_by_config_id(tmp_path: Path) -> None:
    direct = hosted._diagnostic_response_path(
        tmp_path, hosted.V2_CONFIG_SPECS["qwen3.6-35b-a3b-direct"]
    )
    thinking = hosted._diagnostic_response_path(
        tmp_path, hosted.V2_CONFIG_SPECS["qwen3.6-35b-a3b-thinking"]
    )
    assert direct.name == "qwen3.6-35b-a3b-direct-last-response.json"
    assert thinking.name == "qwen3.6-35b-a3b-thinking-last-response.json"
    assert direct != thinking


def test_v2_main_keys_artifacts_by_config_and_counts_schema_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = Path("artifacts/pointwise_judge_bakeoff/smoke-v2/smoke_manifest.json")
    output_dir = tmp_path / "hosted-v2"
    physical_sends = 0

    def hosted_response(request: object, **_kwargs: object) -> object:
        nonlocal physical_sends
        physical_sends += 1
        payload = json.loads(request.data.decode("utf-8"))
        content = "{}" if physical_sends == 1 else json.dumps(VALID_DECISION)
        return {
            "openrouter_metadata": {
                "requested": payload["model"],
                "strategy": "direct",
                "attempt": 1,
                "endpoints": {
                    "total": 1,
                    "available": [
                        {
                            "provider": payload["provider"]["only"][0],
                            "model": payload["model"],
                            "selected": True,
                        }
                    ],
                },
            },
            "choices": [{"finish_reason": "stop", "message": {"content": content}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
                "cost": 0.0001,
            },
        }

    monkeypatch.setattr(hosted, "_post_json", hosted_response)
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-only-secret")
    assert (
        hosted.main(
            [
                "--manifest",
                str(manifest),
                "--manifest-sha256",
                hosted.hashlib.sha256(manifest.read_bytes()).hexdigest(),
                "--protocol-dir",
                "protocols/pointwise_judge_bakeoff",
                "--output-dir",
                str(output_dir),
                "--config",
                "qwen3.6-35b-a3b-thinking",
                "--call-cap",
                "11",
            ]
        )
        == 0
    )

    run_path = output_dir / "runs" / "qwen3.6-35b-a3b-thinking.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    receipt = json.loads(
        (output_dir / "runtime" / "qwen3.6-35b-a3b-thinking.json").read_text(
            encoding="utf-8"
        )
    )
    attempt_ledger = json.loads(
        (output_dir / "attempt-ledger.json").read_text(encoding="utf-8")
    )
    assert run["model_alias"] == "qwen3.6-35b-a3b-thinking"
    assert run["model_id"] == "qwen/qwen3.6-35b-a3b"
    assert len(run["records"]) == 10
    assert sorted(len(record["attempts"]) for record in run["records"].values()) == [
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
    ]
    assert receipt["config_id"] == "qwen3.6-35b-a3b-thinking"
    assert attempt_ledger == {"version": 1, "cap_calls": 11, "used_calls": 11}
    assert physical_sends == 11
    assert (
        output_dir / "diagnostics" / "qwen3.6-35b-a3b-thinking-last-response.json"
    ).exists()


def _router_metadata(
    spec: hosted.ModelSpec,
    *,
    provider: str | None = None,
    available: list[dict[str, object]] | None = None,
    attempt: int = 1,
) -> dict[str, object]:
    return {
        "requested": spec.model_id,
        "strategy": "direct",
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
        hosted.MODEL_SPECS["nvidia/nemotron-3-super-120b-a12b"].provider
        == "DigitalOcean"
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
        FROZEN_SCHEMA,
    )
    assert payload["reasoning"] == {"enabled": False}


def test_gpt_payload_omits_reasoning_for_documented_nonthinking_default() -> None:
    spec = hosted.MODEL_SPECS["openai/gpt-5.4-nano"]
    payload = hosted._build_payload(
        spec, [{"role": "user", "content": "judge"}], FROZEN_SCHEMA
    )
    assert "reasoning" not in payload
    assert "temperature" not in payload


def test_nemotron_payload_pins_healthy_digitalocean_route_and_prices() -> None:
    spec = hosted.MODEL_SPECS["nvidia/nemotron-3-super-120b-a12b"]
    payload = hosted._build_payload(
        spec, [{"role": "user", "content": "judge"}], FROZEN_SCHEMA
    )
    assert spec.provider == "DigitalOcean"
    assert spec.input_usd_per_million == pytest.approx(0.21)
    assert spec.output_usd_per_million == pytest.approx(0.455)
    assert payload["reasoning"] == {"enabled": False}
    assert payload["provider"] == {
        "only": ["DigitalOcean"],
        "allow_fallbacks": False,
        "require_parameters": True,
        "max_price": {"prompt": 0.21, "completion": 0.455},
    }
    input_ceiling = 5_000
    assert spec.worst_case_usd(input_ceiling) == pytest.approx(
        (input_ceiling * 0.21 + 512 * 0.455) / 1_000_000
    )


def test_corrected_routes_are_bound_in_runtime_receipts() -> None:
    nemotron = hosted._runtime_receipt(
        hosted.MODEL_SPECS["nvidia/nemotron-3-super-120b-a12b"],
        FROZEN_CONTRACT,
    )
    gpt = hosted._runtime_receipt(
        hosted.MODEL_SPECS["openai/gpt-5.4-nano"], FROZEN_CONTRACT
    )
    assert nemotron["provider"] == "DigitalOcean"
    assert nemotron["input_usd_per_million"] == pytest.approx(0.21)
    assert nemotron["output_usd_per_million"] == pytest.approx(0.455)
    assert nemotron["reasoning"] == {"enabled": False}
    assert gpt["reasoning"] is None
    assert gpt["temperature"] is None


def test_qwen_payload_omits_reasoning_and_every_route_is_strict_and_pinned() -> None:
    spec = hosted.MODEL_SPECS["qwen/qwen3-235b-a22b-2507"]
    payload = hosted._build_payload(
        spec, [{"role": "user", "content": "judge"}], FROZEN_SCHEMA
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


def test_hosted_transport_schema_omits_only_unique_items_without_mutating_frozen() -> (
    None
):
    original = copy.deepcopy(FROZEN_SCHEMA)
    expected = copy.deepcopy(FROZEN_SCHEMA)
    assert expected["properties"]["error_flags"].pop("uniqueItems") is True

    transport = hosted._openrouter_transport_schema(FROZEN_SCHEMA)

    assert transport == expected
    assert FROZEN_SCHEMA == original
    assert transport["properties"]["rationale"]["maxLength"] == 4000


def test_payload_uses_adjusted_transport_schema_but_parser_contract_stays_frozen() -> (
    None
):
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    payload = hosted._build_payload(
        spec, [{"role": "user", "content": "judge"}], FROZEN_SCHEMA
    )
    transmitted = payload["response_format"]["json_schema"]["schema"]
    assert "uniqueItems" not in transmitted["properties"]["error_flags"]
    assert transmitted["properties"]["rationale"]["maxLength"] == 4000
    validate_frozen_judge_contract(FROZEN_CONTRACT)
    with pytest.raises(ValueError, match="frozen"):
        validate_frozen_judge_contract(replace(FROZEN_CONTRACT, schema=transmitted))


def test_runtime_receipt_binds_exact_hosted_schema_adjustment_and_hash() -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    transport = hosted._openrouter_transport_schema(FROZEN_SCHEMA)
    receipt = hosted._runtime_receipt(spec, FROZEN_CONTRACT)
    assert receipt["schema_transport_adjustment"] == (
        "error_flags.uniqueItems omitted; frozen parser retained"
    )
    assert receipt["transport_schema_sha256"] == hosted._sha256_json(transport)
    assert receipt["schema_sha256"] != receipt["transport_schema_sha256"]


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
    call(spec.model_id, [{"role": "user", "content": "judge"}], FROZEN_SCHEMA)
    assert len(checkpoints) == 2
    assert checkpoints[0] > 0.0001
    assert checkpoints[1] == pytest.approx(0.0001)
    assert request_headers[0]["x-openrouter-metadata"] == "enabled"


def test_invalid_response_is_privately_checkpointed_before_parse_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    invalid_body = {"error": {"code": 200, "message": "provider returned no choices"}}
    monkeypatch.setattr(hosted, "_post_json", lambda *_args, **_kwargs: invalid_body)
    diagnostic_path = hosted._diagnostic_response_path(tmp_path, spec)
    call = hosted.OpenRouterJudgeCall(
        spec=spec,
        api_key="test-only-secret",
        budget=hosted.SpendBudget(0.10),
        timeout_seconds=1,
        checkpoint_response=lambda body: hosted._write_response_checkpoint(
            diagnostic_path, body
        ),
    )
    with pytest.raises(ValueError, match="choice"):
        call(
            spec.model_id,
            [{"role": "user", "content": "judge"}],
            FROZEN_SCHEMA,
        )
    stored = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    assert stored == {"version": 1, "response_body": invalid_body}
    assert stat.S_IMODE(diagnostic_path.stat().st_mode) == 0o600
    assert diagnostic_path == (
        tmp_path / "diagnostics" / "deepseek--deepseek-v4-flash-last-response.json"
    )
    assert "Authorization" not in diagnostic_path.read_text(encoding="utf-8")


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
            FROZEN_SCHEMA,
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
        FROZEN_SCHEMA,
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
            FROZEN_SCHEMA,
        )
    assert network_calls == []
    assert checkpoints == []
    assert budget.spent_usd == 0


def test_response_accepts_provider_versioned_top_level_model_when_metadata_is_exact() -> (
    None
):
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
    versioned = {**body, "model": "deepseek-ai/DeepSeek-V4-Flash-2026-04-23"}
    assert hosted._parse_response(versioned, spec)[0] == json.dumps(VALID_DECISION)


def test_call_preserves_provider_versioned_raw_body_while_binding_logical_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    raw_body = {
        "model": "deepseek-ai/DeepSeek-V4-Flash-2026-04-23",
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
    monkeypatch.setattr(hosted, "_post_json", lambda *_args, **_kwargs: raw_body)
    call = hosted.OpenRouterJudgeCall(
        spec=spec,
        api_key="test-only-secret",
        budget=hosted.SpendBudget(0.10),
        timeout_seconds=1,
    )
    _, runner_body = call(
        spec.model_id,
        [{"role": "user", "content": "judge"}],
        FROZEN_SCHEMA,
    )
    assert runner_body["model"] == spec.model_id
    assert runner_body["openrouter_raw_response_body"] == raw_body


def test_response_accepts_missing_or_opaque_top_level_model_provenance() -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    body = {
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
    assert hosted._parse_response(body, spec)[0] == json.dumps(VALID_DECISION)
    for provenance in (None, "", 1, {"provider_model": "opaque"}):
        assert hosted._parse_response({**body, "model": provenance}, spec)[
            0
        ] == json.dumps(VALID_DECISION)


def test_response_rejects_conflicting_selected_route_metadata() -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    body = {
        "model": "deepseek-ai/DeepSeek-V4-Flash-2026-04-23",
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
    with pytest.raises(ValueError, match="provider"):
        hosted._parse_response(
            {
                **body,
                "openrouter_metadata": _router_metadata(spec, provider="Other"),
            },
            spec,
        )


def test_metadata_accepts_exact_eight_digit_provider_version_alias() -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    alias = f"{spec.model_id}-20260423"
    body = {
        "openrouter_metadata": _router_metadata(
            spec,
            available=[
                {
                    "provider": spec.provider,
                    "model": alias,
                    "selected": True,
                }
            ],
        ),
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
    assert hosted._parse_response(body, spec)[0] == json.dumps(VALID_DECISION)


@pytest.mark.parametrize(
    "selected_model",
    [
        "deepseek/deepseek-v4-flash-latest",
        "deepseek/deepseek-v4-flash-2026042",
        "deepseek/deepseek-v4-flash-202604230",
        "prefix-deepseek/deepseek-v4-flash-20260423",
        "deepseek/other-model-20260423",
        "deepseek/deepseek-v4-flash-２０２６０４２３",
    ],
)
def test_metadata_rejects_noncanonical_selected_model_aliases(
    selected_model: str,
) -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    metadata = _router_metadata(
        spec,
        available=[
            {
                "provider": spec.provider,
                "model": selected_model,
                "selected": True,
            }
        ],
    )
    with pytest.raises(ValueError, match="model"):
        hosted._validate_router_metadata(metadata, spec)


def test_metadata_rejects_non_direct_strategy() -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    metadata = {**_router_metadata(spec), "strategy": "fallback"}
    with pytest.raises(ValueError, match="strategy"):
        hosted._validate_router_metadata(metadata, spec)


def test_qwen_metadata_accepts_only_explicit_deepinfra_endpoint_alias() -> None:
    spec = hosted.MODEL_SPECS["qwen/qwen3-235b-a22b-2507"]
    accepted = _router_metadata(
        spec,
        available=[
            {
                "provider": spec.provider,
                "model": "qwen/qwen3-235b-a22b-07-25",
                "selected": True,
            }
        ],
    )
    hosted._validate_router_metadata(accepted, spec)

    for nearby_alias in (
        "qwen/qwen3-235b-a22b-07-24",
        "qwen/qwen3-235b-a22b-07-250",
        "qwen/qwen3-235b-a22b-07-25-latest",
    ):
        changed = copy.deepcopy(accepted)
        changed["endpoints"]["available"][0]["model"] = nearby_alias
        with pytest.raises(ValueError, match="model"):
            hosted._validate_router_metadata(changed, spec)


def test_qwen_receipt_binds_explicit_alias_without_changing_deepseek_shape() -> None:
    qwen = hosted.MODEL_SPECS["qwen/qwen3-235b-a22b-2507"]
    deepseek = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    qwen_receipt = hosted._runtime_receipt(qwen, FROZEN_CONTRACT)
    deepseek_receipt = hosted._runtime_receipt(deepseek, FROZEN_CONTRACT)

    assert qwen_receipt["allowed_selected_models"] == ["qwen/qwen3-235b-a22b-07-25"]
    assert "allowed_selected_models" not in deepseek_receipt
    assert set(deepseek_receipt) == {
        "schema_sha256",
        "transport_schema_sha256",
        "schema_transport_adjustment",
        "backend_response_model",
        "transport",
        "endpoint",
        "provider",
        "model_id",
        "reasoning",
        "temperature",
        "max_output_tokens",
        "max_serialized_request_bytes",
        "chat_template_overhead_tokens",
        "input_usd_per_million",
        "output_usd_per_million",
        "require_parameters",
        "allow_fallbacks",
    }


def test_response_accepts_absent_optional_router_metadata() -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    body = {
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
    assert hosted._parse_response(body, spec)[0] == json.dumps(VALID_DECISION)
    with pytest.raises(ValueError, match="model"):
        hosted._parse_response(
            {
                **body,
                "openrouter_metadata": _router_metadata(
                    spec,
                    available=[
                        {
                            "provider": spec.provider,
                            "model": "other/model",
                            "selected": True,
                        }
                    ],
                ),
            },
            spec,
        )


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


@pytest.mark.parametrize(
    ("retry_after", "expected_delay"),
    [
        ("7", 7.0),
        (None, 1.0),
        ("malformed", 1.0),
        ("-2", 1.0),
        ("nan", 1.0),
        ("inf", 1.0),
        ("0", 1.0),
        ("999999", 60.0),
    ],
)
def test_retry_after_delta_seconds_is_validated_and_bounded(
    retry_after: str | None, expected_delay: float
) -> None:
    calls = 0
    sleeps: list[float] = []
    headers = {} if retry_after is None else {"Retry-After": retry_after}

    def retry_then_succeed(*_args: object, **_kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError("url", 429, "busy", headers, None)
        return {"ok": True}

    assert hosted._request_with_retry(
        retry_then_succeed,
        object(),
        sleep=sleeps.append,
    ) == {"ok": True}
    assert sleeps == [expected_delay]
    assert calls == 2


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


def _resume_cases() -> list[CandidateCase]:
    return [
        CandidateCase(
            question_id=f"q{index}",
            response_id=f"r{index}",
            question=f"Question {index}?",
            reference_answer=f"Reference {index}.",
            candidate_answer=f"Candidate {index}.",
            strata=("resume_test",),
            selection_reason="resume_test",
        )
        for index in range(10)
    ]


def _build_partial_trusted_run(
    path: Path,
) -> tuple[list[CandidateCase], dict[str, object]]:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    cases = _resume_cases()
    receipt = hosted._runtime_receipt(spec, FROZEN_CONTRACT)

    def valid_call(model: str, _messages: object, _schema: object):
        raw = json.dumps(VALID_DECISION)
        return raw, {
            "model": model,
            "choices": [{"message": {"content": raw}}],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
        }

    run_cases(
        cases,
        model_alias=hosted._safe_name(spec.model_id),
        model_id=spec.model_id,
        contract=FROZEN_CONTRACT,
        output_path=path,
        runtime_receipt=receipt,
        call=valid_call,
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["records"].pop(cases[-1].response_id)
    hosted._write_private_json(path, payload)
    return cases, receipt


def test_trusted_partial_resume_skips_shrinking_canary_and_runs_all_cases(
    tmp_path: Path,
) -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    output = tmp_path / "runs" / "partial.json"
    cases, receipt = _build_partial_trusted_run(output)
    terminal_count = hosted._validated_existing_terminal_count(
        output,
        cases=cases,
        model_alias=hosted._safe_name(spec.model_id),
        model_id=spec.model_id,
        contract=FROZEN_CONTRACT,
        runtime_receipt=receipt,
    )
    seen: list[int] = []

    def resume(selected: list[object]) -> dict[str, object]:
        seen.append(len(selected))
        return {"records": {case.response_id: {} for case in cases}}

    hosted._run_canary_then_remaining(
        cases,
        resume,
        existing_terminal_records=terminal_count,
    )
    assert terminal_count == 9
    assert seen == [10]


@pytest.mark.parametrize("corruption", ["container", "record"])
def test_corrupted_or_mismatched_partial_resume_fails_closed(
    tmp_path: Path, corruption: str
) -> None:
    spec = hosted.MODEL_SPECS["deepseek/deepseek-v4-flash"]
    output = tmp_path / "runs" / "partial.json"
    cases, receipt = _build_partial_trusted_run(output)
    payload = json.loads(output.read_text(encoding="utf-8"))
    if corruption == "container":
        payload["model_id"] = "wrong/model"
    else:
        first = payload["records"][cases[0].response_id]
        first["input_sha256"] = "0" * 64
    hosted._write_private_json(output, payload)

    with pytest.raises(ValueError, match="existing|identity|container|mismatch"):
        hosted._validated_existing_terminal_count(
            output,
            cases=cases,
            model_alias=hosted._safe_name(spec.model_id),
            model_id=spec.model_id,
            contract=FROZEN_CONTRACT,
            runtime_receipt=receipt,
        )


def test_default_key_path_is_private_user_config() -> None:
    assert hosted.DEFAULT_API_KEY_FILE == Path(
        "/home/npatta01/.config/trec-mlm/openrouter.key"
    )
    assert hosted.OPENROUTER_ENDPOINT == "https://openrouter.ai/api/v1/chat/completions"
    assert hosted.DEFAULT_CAP_USD == pytest.approx(0.10)


def _required_cli_args(tmp_path: Path) -> list[str]:
    return [
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--manifest-sha256",
        "0" * 64,
        "--protocol-dir",
        str(tmp_path / "protocol"),
        "--output-dir",
        str(tmp_path / "output"),
    ]


def test_model_filter_defaults_to_all_models_in_fixed_registry_order(
    tmp_path: Path,
) -> None:
    args = hosted._parser().parse_args(_required_cli_args(tmp_path))
    assert args.model is None
    assert [spec.model_id for spec in hosted._select_model_specs(args.model)] == list(
        hosted.MODEL_SPECS
    )


def test_model_filter_selects_multiple_exact_ids_in_fixed_registry_order(
    tmp_path: Path,
) -> None:
    requested = [
        "openai/gpt-5.4-nano",
        "deepseek/deepseek-v4-flash",
    ]
    args = hosted._parser().parse_args(
        [
            *_required_cli_args(tmp_path),
            "--model",
            requested[0],
            "--model",
            requested[1],
        ]
    )
    assert args.model == requested
    assert [spec.model_id for spec in hosted._select_model_specs(args.model)] == [
        "deepseek/deepseek-v4-flash",
        "openai/gpt-5.4-nano",
    ]


@pytest.mark.parametrize(
    "requested",
    [
        ["unknown/model"],
        ["deepseek/deepseek-v4-flash", "deepseek/deepseek-v4-flash"],
        [""],
        [],
    ],
)
def test_model_filter_rejects_unknown_duplicate_or_empty_ids(
    requested: list[str],
) -> None:
    with pytest.raises(ValueError, match="model|duplicate|empty"):
        hosted._select_model_specs(requested)


def test_invalid_model_filter_fails_before_api_key_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        hosted,
        "_load_api_key",
        lambda *_args, **_kwargs: pytest.fail("API key must not be loaded"),
    )
    with pytest.raises(ValueError, match="model"):
        hosted.main(
            [
                *_required_cli_args(tmp_path),
                "--model",
                "unknown/model",
            ]
        )


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
