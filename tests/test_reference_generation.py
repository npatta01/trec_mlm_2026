import hashlib
import json
from pathlib import Path
import threading
import urllib.error

import pytest

from scripts.preflight_reference_models import MODELS, build_preflight
from trec_mlm.reference_generation import (
    GEMINI_ENDPOINT_TEMPLATE,
    SYSTEM_PROMPT,
    build_gemini_request,
    build_openai_request,
    call_gemini,
    call_openai_compatible,
    generate_missing,
    parse_gemini_response,
    validate_openai_compatible_url,
)
from scripts.generate_reference_challengers import main


def _commands(*, rocminfo: str = "GPU agent gfx1151") -> dict[str, str]:
    return {
        "rocminfo": rocminfo,
        "podman_version": "podman version 5.4.0",
        "podman_images": "vllm/vllm-openai-rocm:latest sha256:image",
        "meminfo": "MemAvailable: 120000000 kB\nSwapFree: 8000000 kB\n",
    }


def test_preflight_records_presence_not_secret_value() -> None:
    receipt = build_preflight(
        environ={"GEMINI_API_KEY": "super-secret-value"},
        command_results=_commands(),
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    assert receipt["credentials"]["gemini_api_key"] == "configured"
    assert "super-secret-value" not in json.dumps(receipt)


def test_preflight_rejects_missing_rocm_device_for_local_models() -> None:
    receipt = build_preflight(
        environ={},
        command_results=_commands(rocminfo=""),
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    assert receipt["gates"]["local_model_smoke_ready"] is False
    assert "gfx1151" in receipt["gates"]["local_model_reason"]


def test_preflight_pins_verified_qwen_revisions() -> None:
    assert MODELS["qwen35"]["base_revision"] == (
        "995ad96eacd98c81ed38be0c5b274b04031597b0"
    )
    assert MODELS["qwen35"]["fp8_revision"] == (
        "95a723d08a9490559dae23d0cff1d9466213d989"
    )
    assert MODELS["qwen27"]["base_revision"] == (
        "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"
    )
    assert MODELS["qwen27"]["fp8_revision"] == (
        "e89b16ebf1988b3d6befa7de50abc2d76f26eb09"
    )


def test_preflight_sums_hf_dry_run_json_sizes_in_bytes() -> None:
    commands = _commands()
    commands["qwen35_fp8_dry_run"] = json.dumps(
        [
            {"file": "weights.safetensors", "size": "1.5G"},
            {"file": "config.json", "size": "2.0K"},
            {"file": "empty", "size": "58.0"},
        ]
    )
    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    assert receipt["models"]["qwen35"]["fp8"]["dry_run_download_bytes"] == (
        1_500_002_058
    )


def test_preflight_requires_metadata_for_every_model_pin() -> None:
    commands = _commands()
    commands["qwen35_base_info"] = json.dumps(
        {
            "sha": MODELS["qwen35"]["base_revision"],
            "tags": ["license:apache-2.0"],
        }
    )
    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    assert receipt["gates"]["model_pins_verified"] is False


def test_preflight_records_exact_hub_repository_bytes() -> None:
    commands = _commands()
    commands["qwen35_base_info"] = json.dumps(
        {
            "sha": MODELS["qwen35"]["base_revision"],
            "used_storage": 71_900_000_123,
        }
    )
    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    assert receipt["models"]["qwen35"]["base"]["exact_repository_bytes"] == (
        71_900_000_123
    )


def test_preflight_rejects_insufficient_disk_for_selected_models() -> None:
    commands = _commands()
    commands["qwen35_fp8_dry_run"] = '[{"size": "2.0G"}]'
    commands["qwen27_fp8_dry_run"] = '[{"size": "1.0G"}]'
    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=2_999_999_999,
        commit="abc123",
        dirty=False,
    )
    assert receipt["gates"]["local_model_smoke_ready"] is False
    assert "disk" in receipt["gates"]["local_model_reason"]


def test_preflight_requires_explicit_rootless_podman_evidence() -> None:
    receipt = build_preflight(
        environ={},
        command_results=_commands(),
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    assert receipt["gates"]["local_model_smoke_ready"] is False
    assert "rootless" in receipt["gates"]["local_model_reason"]


def test_prompt_forbids_guessing_and_requests_direct_answer() -> None:
    assert "Do not guess" in SYSTEM_PROMPT
    assert "underspecified" in SYSTEM_PROMPT


def test_gemini_request_contains_question_but_no_other_answers() -> None:
    body = build_gemini_request("Who wrote Hamlet?", max_output_tokens=768)
    encoded = json.dumps(body)
    assert "Who wrote Hamlet?" in encoded
    assert "Perplexity" not in encoded
    assert body["generationConfig"]["temperature"] == 0.0


def test_openai_request_uses_the_same_prompt() -> None:
    body = build_openai_request(
        "Who wrote Hamlet?", model="Qwen/test", max_output_tokens=768
    )
    assert body["messages"][0]["content"] == SYSTEM_PROMPT
    assert body["messages"][1]["content"] == "Who wrote Hamlet?"


def test_parse_gemini_response_extracts_answer_usage_and_finish_reason() -> None:
    response = {
        "candidates": [
            {
                "content": {"parts": [{"text": "Shakespeare."}]},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {"totalTokenCount": 9},
    }
    parsed = parse_gemini_response(response)
    assert parsed == ("Shakespeare.", {"totalTokenCount": 9}, "STOP")


def test_generate_missing_resumes_and_never_serializes_secrets(tmp_path: Path) -> None:
    cache_path = tmp_path / "generations" / "gemini.json"
    calls: list[str] = []
    secret = "test-api-key-value"

    def first_call(question: str):
        calls.append(question)
        if question == "second":
            raise RuntimeError("temporary failure")
        return (
            "answer one",
            {"totalTokenCount": 2},
            "STOP",
            {"endpoint": "https://example.invalid", "x-goog-api-key": secret},
        )

    first = generate_missing(
        [("q1", "first"), ("q2", "second")],
        cache_path=cache_path,
        provider="gemini",
        model="gemini-3.5-flash",
        call=first_call,
        workers=1,
        checkpoint_every=1,
    )
    assert set(first["records"]) == {"q1"}
    assert set(first["failures"]) == {"q2"}

    def resumed_call(question: str):
        calls.append(question)
        return "answer two", {}, "STOP", {"authorization": f"Bearer {secret}"}

    second = generate_missing(
        [("q1", "first"), ("q2", "second")],
        cache_path=cache_path,
        provider="gemini",
        model="gemini-3.5-flash",
        call=resumed_call,
        workers=1,
        checkpoint_every=1,
    )
    assert calls == ["first", "second", "second"]
    assert set(second["records"]) == {"q1", "q2"}
    assert second["failures"] == {}
    cache_text = cache_path.read_text(encoding="utf-8")
    assert secret not in cache_text
    assert not cache_path.with_suffix(".json.tmp").exists()


def test_resume_does_not_skip_a_record_from_a_different_model(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    calls: list[str] = []

    def call(question: str):
        calls.append(question)
        return question, {}, None, {}

    generate_missing(
        [("q1", "question")],
        cache_path=cache_path,
        provider="gemini",
        model="old-model",
        call=call,
        workers=1,
    )
    generate_missing(
        [("q1", "question")],
        cache_path=cache_path,
        provider="gemini",
        model="new-model",
        call=call,
        workers=1,
    )
    assert calls == ["question", "question"]


def test_gemini_retries_retriable_status_with_injected_sleep() -> None:
    attempts = 0
    sleeps: list[float] = []

    def request(url: str, body: dict, headers: dict[str, str]):
        nonlocal attempts
        attempts += 1
        assert url == GEMINI_ENDPOINT_TEMPLATE.format(model="gemini-3.5-flash")
        assert headers["x-goog-api-key"] == "secret"
        assert body["generationConfig"]["maxOutputTokens"] == 768
        if attempts < 3:
            raise urllib.error.HTTPError(url, 429, "limited", {}, None)
        return {
            "candidates": [
                {"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"totalTokenCount": 3},
        }

    answer, usage, finish_reason, metadata = call_gemini(
        "question",
        api_key="secret",
        model="gemini-3.5-flash",
        request_fn=request,
        sleep_fn=sleeps.append,
    )
    assert (answer, usage, finish_reason) == (
        "answer",
        {"totalTokenCount": 3},
        "STOP",
    )
    assert metadata["attempts"] == 3
    assert sleeps == [1.0, 2.0]


@pytest.mark.parametrize(
    "url",
    [
        "https://example.com/v1/chat/completions",
        "http://127.0.0.1:8000/v1/models",
        "http://localhost.evil.test/v1/chat/completions",
    ],
)
def test_openai_compatible_url_fails_closed(url: str) -> None:
    with pytest.raises(ValueError, match="loopback|/v1/chat/completions"):
        validate_openai_compatible_url(url)


def test_openai_compatible_call_parses_response_without_serializing_auth() -> None:
    def request(url: str, body: dict, headers: dict[str, str]):
        assert url == "http://[::1]:8000/v1/chat/completions"
        assert "Authorization" not in headers
        assert body["model"] == "Qwen/test"
        return {
            "choices": [
                {"message": {"content": "local answer"}, "finish_reason": "stop"}
            ],
            "usage": {"total_tokens": 4},
        }

    result = call_openai_compatible(
        "question",
        model="Qwen/test",
        base_url="http://[::1]:8000/v1/chat/completions",
        request_fn=request,
        sleep_fn=lambda _: None,
    )
    assert result[:3] == ("local answer", {"total_tokens": 4}, "stop")


def test_cli_missing_gemini_key_fails_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "sample_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "reference-validation-sample-v1",
                "sample_size": 1,
                "questions": [{"question_id": "q1", "question": "Question?"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="GEMINI_API_KEY"):
        main(
            [
                "gemini",
                "--sample",
                str(manifest),
                "--cache",
                str(tmp_path / "cache.json"),
            ]
        )


def test_cache_drops_unapproved_metadata_and_raw_exception_text(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    secret = "AIza-fake-review-secret"

    def call(question: str):
        if question == "failure":
            raise RuntimeError(f"request exploded with {secret}")
        return (
            "safe answer",
            {},
            "STOP",
            {
                "endpoint": GEMINI_ENDPOINT_TEMPLATE.format(model="gemini-3.5-flash"),
                "attempts": 1,
                "diagnostic_note": f"innocuous-looking field containing {secret}",
            },
        )

    cache = generate_missing(
        [("q1", "success"), ("q2", "failure")],
        cache_path=cache_path,
        provider="gemini",
        model="gemini-3.5-flash",
        call=call,
        workers=1,
        checkpoint_every=1,
    )

    assert cache["records"]["q1"]["request_metadata"] == {
        "attempts": 1,
        "endpoint": GEMINI_ENDPOINT_TEMPLATE.format(model="gemini-3.5-flash"),
    }
    assert cache["failures"]["q2"] == {
        "category": "provider",
        "error_type": "RuntimeError",
        "http_status": None,
        "model": "gemini-3.5-flash",
        "prompt_sha256": cache["prompt_sha256"],
        "provider": "gemini",
        "retryable": False,
    }
    assert secret not in cache_path.read_text(encoding="utf-8")


def test_gemini_cli_rejects_non_frozen_model_before_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = tmp_path / "sample_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "reference-validation-sample-v1",
                "sample_size": 1,
                "questions": [{"question_id": "q1", "question": "Question?"}],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("GEMINI_API_KEY", "unused-test-key")

    with pytest.raises(SystemExit):
        main(
            [
                "gemini",
                "--sample",
                str(manifest),
                "--cache",
                str(tmp_path / "cache.json"),
                "--model",
                "gemini-other",
            ]
        )
    assert not (tmp_path / "cache.json").exists()


def test_failed_replacement_removes_incompatible_cached_success(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    generate_missing(
        [("q1", "question")],
        cache_path=cache_path,
        provider="gemini",
        model="old-model",
        call=lambda _: ("stale answer", {}, "STOP", {}),
        workers=1,
    )

    def fail(_: str):
        raise RuntimeError("replacement unavailable")

    cache = generate_missing(
        [("q1", "question")],
        cache_path=cache_path,
        provider="gemini",
        model="new-model",
        call=fail,
        workers=1,
    )

    assert "q1" not in cache["records"]
    assert cache["failures"]["q1"]["model"] == "new-model"
    persisted = json.loads(cache_path.read_text(encoding="utf-8"))
    assert "q1" not in persisted["records"]


def test_keyboard_interrupt_checkpoints_completed_ids_and_resume_skips_them(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache.json"
    calls: list[str] = []
    interrupt_once = threading.Event()

    def interrupted_call(question: str):
        calls.append(question)
        if question == "second" and not interrupt_once.is_set():
            interrupt_once.set()
            raise KeyboardInterrupt
        return f"answer {question}", {}, "STOP", {"attempts": 1}

    with pytest.raises(KeyboardInterrupt):
        generate_missing(
            [("q1", "first"), ("q2", "second"), ("q3", "third")],
            cache_path=cache_path,
            provider="gemini",
            model="gemini-3.5-flash",
            call=interrupted_call,
            workers=1,
            checkpoint_every=10,
        )

    interrupted_cache = json.loads(cache_path.read_text(encoding="utf-8"))
    assert set(interrupted_cache["records"]) == {"q1"}
    assert calls == ["first", "second"]

    resumed = generate_missing(
        [("q1", "first"), ("q2", "second"), ("q3", "third")],
        cache_path=cache_path,
        provider="gemini",
        model="gemini-3.5-flash",
        call=interrupted_call,
        workers=1,
        checkpoint_every=10,
    )
    assert calls == ["first", "second", "second", "third"]
    assert set(resumed["records"]) == {"q1", "q2", "q3"}


def test_gemini_adapter_rejects_model_override_before_transport() -> None:
    transport_called = False

    def request(url: str, body: dict, headers: dict[str, str]):
        nonlocal transport_called
        transport_called = True
        raise AssertionError("transport must not run")

    with pytest.raises(ValueError, match="frozen"):
        call_gemini(
            "question",
            api_key="unused-test-key",
            model="gemini-other",
            request_fn=request,
            sleep_fn=lambda _: None,
        )
    assert transport_called is False


def test_gemini_request_id_is_hashed_before_adapter_metadata() -> None:
    secret = "AIza-fake-provider-request-id"

    def request(url: str, body: dict, headers: dict[str, str]):
        return {
            "responseId": secret,
            "candidates": [
                {"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"totalTokenCount": 3},
        }

    result = call_gemini(
        "question",
        api_key="unused-test-key",
        request_fn=request,
        sleep_fn=lambda _: None,
    )

    assert (
        result[3]["request_id_sha256"]
        == hashlib.sha256(secret.encode("utf-8")).hexdigest()
    )
    assert secret not in json.dumps(result)


@pytest.mark.parametrize(
    ("provider", "model", "usage", "expected_usage"),
    [
        (
            "gemini",
            "gemini-3.5-flash",
            {
                "totalTokenCount": 7,
                "promptTokenCount": 3,
                "providerNote": "AIza-fake-cache-channel",
                "unknownNumeric": 999,
            },
            {"promptTokenCount": 3, "totalTokenCount": 7},
        ),
        (
            "openai-compatible",
            "Qwen/test",
            {
                "total_tokens": 7,
                "prompt_tokens": 3,
                "provider_note": "AIza-fake-cache-channel",
                "unknown_numeric": 999,
                "prompt_tokens_details": {
                    "cached_tokens": 1,
                    "provider_note": "AIza-fake-cache-channel",
                },
            },
            {
                "prompt_tokens": 3,
                "prompt_tokens_details": {"cached_tokens": 1},
                "total_tokens": 7,
            },
        ),
    ],
)
def test_cache_allowlists_usage_and_finish_reason_and_metadata_strings(
    tmp_path: Path,
    provider: str,
    model: str,
    usage: dict,
    expected_usage: dict,
) -> None:
    cache_path = tmp_path / "cache.json"
    secret = "AIza-fake-cache-channel"

    cache = generate_missing(
        [("q1", "question")],
        cache_path=cache_path,
        provider=provider,
        model=model,
        call=lambda _: (
            "safe answer",
            usage,
            secret,
            {
                "endpoint": f"http://127.0.0.1:8000/{secret}",
                "request_id_sha256": hashlib.sha256(secret.encode("utf-8")).hexdigest(),
                "attempts": 1,
            },
        ),
        workers=1,
    )

    record = cache["records"]["q1"]
    assert record["usage"] == expected_usage
    assert record["finish_reason"] == "unknown"
    assert record["request_metadata"] == {
        "attempts": 1,
        "request_id_sha256": hashlib.sha256(secret.encode("utf-8")).hexdigest(),
    }
    assert secret not in cache_path.read_text(encoding="utf-8")
