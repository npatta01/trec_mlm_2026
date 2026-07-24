import hashlib
import json
from pathlib import Path
import threading
import urllib.error

import pytest

from scripts.preflight_reference_models import (
    MODELS,
    build_preflight,
    main as preflight_main,
    preflight_exit_code,
)
from trec_mlm.reference_generation import (
    DEEPINFRA_GEMMA_MODEL,
    GEMMA_MODEL,
    GEMINI_ENDPOINT_TEMPLATE,
    GEMINI_MODEL,
    GOOGLE_MAX_OUTPUT_TOKENS,
    SYSTEM_PROMPT,
    build_gemini_request,
    build_openai_request,
    call_deepinfra,
    call_gemini,
    call_openai_compatible,
    generate_missing,
    parse_gemini_response,
    parse_structured_answer,
    validate_openai_compatible_url,
)
from scripts.generate_reference_challengers import main


def test_gemini_model_is_pinned_to_stable_3_6_flash() -> None:
    assert GEMINI_MODEL == "gemini-3.6-flash"


def test_gemma_model_is_pinned_to_stable_4_31b() -> None:
    assert GEMMA_MODEL == "gemma-4-31b-it"


def test_google_generation_ceiling_allows_long_reasoning() -> None:
    assert GOOGLE_MAX_OUTPUT_TOKENS == 16_384


def _commands(*, rocminfo: str = "GPU agent gfx1151") -> dict[str, str]:
    commands = {
        "rocminfo": rocminfo,
        "podman_version": "podman version 5.4.0",
        "podman_images": "vllm/vllm-openai-rocm:latest sha256:image",
        "podman_rootless": "true\n",
        "meminfo": "MemAvailable: 120000000 kB\nSwapFree: 8000000 kB\n",
    }
    for model_key, variants in MODELS.items():
        for representation in ("base", "fp8"):
            commands[f"{model_key}_{representation}_info"] = json.dumps(
                {
                    "sha": variants[f"{representation}_revision"],
                    "tags": ["license:apache-2.0"],
                    "safetensors": {"total": 1_000_000},
                    "used_storage": 2_000_000_000,
                }
            )
            commands[f"{model_key}_{representation}_dry_run"] = (
                '[{"file": "weights", "size": "1.9G"}]'
            )
    return commands


def test_preflight_complete_receipt_is_ready_without_gemini_key() -> None:
    receipt = build_preflight(
        environ={},
        command_results=_commands(),
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    assert receipt["gates"]["receipt_complete"] is True
    assert receipt["gates"]["local_download_ready"] is True
    assert receipt["execution_gates"]["gemini_pilot_ready"] is False
    assert "Task 5" in receipt["execution_gates"]["gemini_pilot_reason"]
    assert preflight_exit_code(receipt) == 0


def test_preflight_key_presence_does_not_authorize_gemini_pilot() -> None:
    receipt = build_preflight(
        environ={"GEMINI_API_KEY": "secret-is-presence-only"},
        command_results=_commands(),
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    execution = receipt["execution_gates"]
    assert execution["gemini_credential_configured"] is True
    assert execution["task5_authorization_required"] is True
    assert execution["gemini_pilot_ready"] is False
    assert "authorization" in execution["gemini_pilot_reason"]
    assert "secret-is-presence-only" not in json.dumps(receipt)


@pytest.mark.parametrize(
    ("output", "expected_error"),
    [
        ("not JSON", "valid JSON"),
        ('[{"file": "weights", "size": NaN}]', "finite"),
        ('[{"file": "weights", "size": Infinity}]', "finite"),
        ('[{"file": "weights", "size": "twelveG"}]', "numeric"),
        ('[{"file": "weights", "size": "1e999"}]', "overflow"),
        ('[{"file": "weights", "size": "-1.0G"}]', "nonnegative"),
        ('[{"file": "weights"}]', "size field"),
        ('["not-a-file-row"]', "object"),
        (
            '[{"file": "valid", "size": "1.0K"}, {"bogus": true}]',
            "file field",
        ),
    ],
)
def test_preflight_dry_run_observation_errors_fail_closed(
    output: str, expected_error: str
) -> None:
    commands = _commands()
    commands["qwen35_base_dry_run"] = output
    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    model = receipt["models"]["qwen35"]["base"]
    assert model["dry_run_download_bytes"] is None
    assert any(expected_error in error for error in model["dry_run_observation_errors"])
    assert model["evidence_complete"] is False
    assert receipt["gates"]["receipt_complete"] is False
    assert preflight_exit_code(receipt) == 1
    json.dumps(receipt, allow_nan=False)


def test_preflight_oversized_json_integer_becomes_observation_error() -> None:
    commands = _commands()
    oversized_integer = "9" * 5_000
    commands["qwen35_base_dry_run"] = (
        f'[{{"file": "weights", "size": {oversized_integer}}}]'
    )
    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    model = receipt["models"]["qwen35"]["base"]
    assert model["dry_run_download_bytes"] is None
    assert any(
        "oversized numeric token" in error
        for error in model["dry_run_observation_errors"]
    )
    assert receipt["gates"]["receipt_complete"] is False
    assert preflight_exit_code(receipt) == 1
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value", "expected_reason"),
    [
        ("parameter_count", float("nan"), "parameter count"),
        ("exact_repository_bytes", float("inf"), "repository bytes"),
    ],
)
def test_preflight_normalizes_nonfinite_hub_metadata(
    field: str, value: float, expected_reason: str
) -> None:
    commands = _commands()
    metadata = json.loads(commands["qwen35_base_info"])
    if field == "parameter_count":
        metadata["safetensors"]["total"] = value
    else:
        metadata["used_storage"] = value
    commands["qwen35_base_info"] = json.dumps(metadata)

    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    model = receipt["models"]["qwen35"]["base"]
    assert model[field] is None
    assert any(expected_reason in reason for reason in model["evidence_reasons"])
    assert receipt["gates"]["receipt_complete"] is False
    assert preflight_exit_code(receipt) == 1
    json.dumps(receipt, allow_nan=False)


def test_preflight_oversized_hub_integer_becomes_observation_error() -> None:
    commands = _commands()
    oversized_integer = "9" * 5_000
    commands["qwen35_base_info"] = (
        f'{{"sha": "{MODELS["qwen35"]["base_revision"]}", '
        f'"used_storage": {oversized_integer}}}'
    )
    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    model = receipt["models"]["qwen35"]["base"]
    assert any(
        "oversized numeric token" in error for error in model["hub_observation_errors"]
    )
    assert model["evidence_complete"] is False
    assert receipt["gates"]["receipt_complete"] is False
    assert preflight_exit_code(receipt) == 1
    json.dumps(receipt, allow_nan=False)


def test_preflight_normalizes_nonstring_sha_to_null() -> None:
    commands = _commands()
    metadata = json.loads(commands["qwen35_base_info"])
    metadata["sha"] = float("nan")
    metadata["tags"].append({"unsafe": float("inf")})
    commands["qwen35_base_info"] = json.dumps(metadata)
    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    model = receipt["models"]["qwen35"]["base"]
    assert model["observed_sha"] is None
    assert model["license"] == "apache-2.0"
    assert any(
        "SHA must be a string" in error for error in model["hub_observation_errors"]
    )
    assert any(
        "tag 1 must be a string" in error for error in model["hub_observation_errors"]
    )
    assert model["evidence_complete"] is False
    assert receipt["gates"]["receipt_complete"] is False
    assert preflight_exit_code(receipt) == 1
    json.dumps(receipt, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "expected_reason"),
    [
        ("license", "license"),
        ("parameter_count", "parameter count"),
        ("dry_run_download_bytes", "dry-run bytes"),
        ("exact_repository_bytes", "repository bytes"),
    ],
)
def test_preflight_required_model_evidence_fails_closed(
    field: str, expected_reason: str
) -> None:
    commands = _commands()
    metadata = json.loads(commands["qwen35_base_info"])
    if field == "license":
        metadata["tags"] = []
    elif field == "parameter_count":
        metadata["safetensors"] = {"total": 0}
    elif field == "exact_repository_bytes":
        metadata.pop("used_storage")
    else:
        commands["qwen35_base_dry_run"] = "[]"
    commands["qwen35_base_info"] = json.dumps(metadata)

    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    model = receipt["models"]["qwen35"]["base"]
    assert model["evidence_complete"] is False
    assert any(expected_reason in reason for reason in model["evidence_reasons"])
    assert receipt["gates"]["receipt_complete"] is False
    assert receipt["gates"]["local_download_ready"] is False
    assert preflight_exit_code(receipt) == 1


def test_preflight_rejects_image_reference_with_malicious_suffix() -> None:
    commands = _commands()
    commands["podman_images"] = "vllm/vllm-openai-rocm:latest-malicious sha256:image"
    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    assert receipt["gates"]["local_download_ready"] is False
    assert receipt["runtime"]["vllm_image_present"] is False


def test_preflight_accepts_podman_canonical_docker_io_image_reference() -> None:
    commands = _commands()
    commands["podman_images"] = "docker.io/vllm/vllm-openai-rocm:latest sha256:image"
    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    assert receipt["runtime"]["vllm_image_present"] is True
    assert receipt["gates"]["local_download_ready"] is True


@pytest.mark.parametrize("cap", ["-1", "nan", "inf"])
def test_preflight_cli_rejects_nonfinite_or_negative_cost_cap(cap: str) -> None:
    with pytest.raises(SystemExit, match="finite and nonnegative"):
        preflight_main(["--gemini-cost-cap-usd", cap])


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
    assert receipt["gates"]["receipt_complete"] is False


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
    commands = _commands()
    commands.pop("podman_rootless")
    receipt = build_preflight(
        environ={},
        command_results=commands,
        disk_free_bytes=500_000_000_000,
        commit="abc123",
        dirty=False,
    )
    assert receipt["gates"]["local_model_smoke_ready"] is False
    assert "rootless" in receipt["gates"]["local_model_reason"]


def test_prompt_forbids_guessing_and_requests_direct_answer() -> None:
    assert "Do not guess" in SYSTEM_PROMPT
    assert "underspecified" in SYSTEM_PROMPT
    assert "Begin with exactly one status" in SYSTEM_PROMPT
    assert "CONFLICTING_EVIDENCE" in SYSTEM_PROMPT
    assert "For time-sensitive claims, state the relevant date" in SYSTEM_PROMPT


@pytest.mark.parametrize(
    ("raw_status", "expected_status"),
    [
        ("ANSWERED", "answered"),
        ("AMBIGUOUS", "ambiguous"),
        ("INSUFFICIENT_EVIDENCE", "insufficient_evidence"),
        ("CONFLICTING_EVIDENCE", "conflicting_evidence"),
    ],
)
def test_parse_structured_answer_separates_status_reason_and_clean_answer(
    raw_status: str, expected_status: str
) -> None:
    answer, status, reason = parse_structured_answer(
        f"{raw_status}\nUNCERTAINTY_REASON: Source scope is limited.\n\nClean answer."
    )

    assert answer == "Clean answer."
    assert status == expected_status
    assert reason == "Source scope is limited."


@pytest.mark.parametrize(
    "text",
    [
        "MAYBE\nUNCERTAINTY_REASON: unclear\nanswer",
        "ANSWERED\nanswer without reason field",
        "ANSWERED\nUNCERTAINTY_REASON: none",
    ],
)
def test_parse_structured_answer_rejects_invalid_contract(text: str) -> None:
    with pytest.raises(ValueError, match="structured answer"):
        parse_structured_answer(text)


def test_gemini_request_contains_question_but_no_other_answers() -> None:
    body = build_gemini_request("Who wrote Hamlet?", max_output_tokens=768)
    encoded = json.dumps(body)
    assert "Who wrote Hamlet?" in encoded
    assert "Perplexity" not in encoded
    assert "tools" not in body
    assert body["generationConfig"]["temperature"] == 1.0
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "low"}


def test_grounded_gemini_request_enables_google_search_only_when_requested() -> None:
    grounded = build_gemini_request("Who wrote Hamlet?", grounding=True)
    ungrounded = build_gemini_request("Who wrote Hamlet?")

    assert grounded["tools"] == [{"google_search": {}}]
    assert "tools" not in ungrounded


def test_gemma_request_uses_high_thinking_and_large_default_ceiling() -> None:
    body = build_gemini_request("Who wrote Hamlet?", model=GEMMA_MODEL)
    assert body["generationConfig"]["maxOutputTokens"] == 16_384
    assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}


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
        model="gemini-3.6-flash",
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
        model="gemini-3.6-flash",
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


def test_resume_regenerates_blank_or_unknown_finish_record(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    calls: list[str] = []

    def call(question: str):
        calls.append(question)
        return "valid answer", {}, "STOP", {}

    generate_missing(
        [("q1", "question")],
        cache_path=cache_path,
        provider="gemini",
        model="gemini-3.6-flash",
        call=call,
        workers=1,
    )
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache["records"]["q1"]["answer"] = ""
    cache["records"]["q1"]["finish_reason"] = "unknown"
    cache_path.write_text(json.dumps(cache), encoding="utf-8")

    generate_missing(
        [("q1", "question")],
        cache_path=cache_path,
        provider="gemini",
        model="gemini-3.6-flash",
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
        assert url == GEMINI_ENDPOINT_TEMPLATE.format(model="gemini-3.6-flash")
        assert headers["x-goog-api-key"] == "secret"
        assert body["generationConfig"]["maxOutputTokens"] == 16_384
        if attempts < 3:
            raise urllib.error.HTTPError(url, 503, "unavailable", {}, None)
        return {
            "candidates": [
                {"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"totalTokenCount": 3},
        }

    answer, usage, finish_reason, metadata = call_gemini(
        "question",
        api_key="secret",
        model="gemini-3.6-flash",
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


def test_gemini_does_not_amplify_429_with_immediate_retries() -> None:
    attempts = 0

    def request(url: str, body: dict, headers: dict[str, str]):
        nonlocal attempts
        attempts += 1
        raise urllib.error.HTTPError(url, 429, "limited", {}, None)

    with pytest.raises(Exception) as caught:
        call_gemini(
            "question",
            api_key="secret",
            request_fn=request,
            sleep_fn=lambda _: None,
        )

    assert type(caught.value).__name__ == "ProviderRequestError"
    assert attempts == 1


def test_grounded_gemini_call_records_private_structured_provenance() -> None:
    def request(url: str, body: dict, headers: dict[str, str]):
        assert body["tools"] == [{"google_search": {}}]
        return {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "text": (
                                    "ANSWERED\nUNCERTAINTY_REASON: none\n\n"
                                    "grounded answer"
                                )
                            }
                        ]
                    },
                    "finishReason": "STOP",
                    "groundingMetadata": {
                        "webSearchQueries": ["query one", "query two"],
                        "groundingChunks": [
                            {"web": {"uri": "https://example.com/secret", "title": "A"}}
                        ],
                        "groundingSupports": [
                            {
                                "segment": {
                                    "startIndex": 0,
                                    "endIndex": 15,
                                    "text": "grounded answer",
                                },
                                "groundingChunkIndices": [0],
                            }
                        ],
                    },
                }
            ],
            "usageMetadata": {"totalTokenCount": 9},
        }

    answer, _, _, metadata = call_gemini(
        "question",
        api_key="secret",
        grounding=True,
        request_fn=request,
        sleep_fn=lambda _: None,
    )

    assert answer == "grounded answer"
    assert metadata["grounding"] is True
    assert metadata["grounding_query_count"] == 2
    assert metadata["grounding_chunk_count"] == 1
    assert metadata["grounding_support_count"] == 1
    assert "groundingMetadata" not in metadata
    assert metadata["uncertainty_status"] == "answered"
    assert metadata["uncertainty_reason"] == "none"
    evidence = metadata["_private_evidence"]
    assert evidence["search_invoked"] is True
    assert evidence["web_search_queries"] == ["query one", "query two"]
    assert evidence["sources"] == [
        {"index": 0, "url": "https://example.com/secret", "title": "A"}
    ]
    assert evidence["supports"] == [
        {
            "answer_passage": "grounded answer",
            "start_index": 0,
            "end_index": 15,
            "source_indices": [0],
        }
    ]


def test_grounded_generation_cache_preserves_only_safe_grounding_receipt(
    tmp_path: Path,
) -> None:
    cache = generate_missing(
        [("q1", "question")],
        cache_path=tmp_path / "grounded.json",
        provider="gemini-grounded",
        model=GEMINI_MODEL,
        call=lambda _: (
            "answer",
            {"totalTokenCount": 9},
            "STOP",
            {
                "endpoint": GEMINI_ENDPOINT_TEMPLATE.format(model=GEMINI_MODEL),
                "attempts": 1,
                "grounding": True,
                "grounding_query_count": 2,
                "grounding_chunk_count": 3,
                "grounding_support_count": 4,
                "unsafe_grounding_uri": "https://example.com/private",
            },
        ),
        workers=1,
    )

    assert cache["records"]["q1"]["request_metadata"] == {
        "attempts": 1,
        "endpoint": GEMINI_ENDPOINT_TEMPLATE.format(model=GEMINI_MODEL),
        "grounding": True,
        "grounding_query_count": 2,
        "grounding_chunk_count": 3,
        "grounding_support_count": 4,
    }


def test_grounded_generation_writes_private_evidence_ledger_only(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "public" / "grounded.json"
    evidence_path = tmp_path / "private" / "grounded.evidence.json"
    private_evidence = {
        "search_invoked": True,
        "web_search_queries": ["query one"],
        "sources": [
            {"index": 0, "url": "https://example.com/private", "title": "Source"}
        ],
        "supports": [
            {
                "answer_passage": "answer",
                "start_index": 0,
                "end_index": 6,
                "source_indices": [0],
            }
        ],
        "generated_at": "2026-07-21T12:00:00Z",
    }

    cache = generate_missing(
        [("q1", "question")],
        cache_path=cache_path,
        evidence_cache_path=evidence_path,
        provider="gemini-grounded",
        model=GEMINI_MODEL,
        call=lambda _: (
            "answer",
            {"promptTokenCount": 2, "totalTokenCount": 5},
            "STOP",
            {
                "grounding": True,
                "grounding_query_count": 1,
                "grounding_chunk_count": 1,
                "grounding_support_count": 1,
                "uncertainty_status": "answered",
                "uncertainty_reason": "none",
                "_private_evidence": private_evidence,
            },
        ),
        workers=1,
    )

    record = cache["records"]["q1"]
    assert record["uncertainty_status"] == "answered"
    assert record["uncertainty_reason"] == "none"
    assert "example.com" not in cache_path.read_text(encoding="utf-8")
    ledger = json.loads(evidence_path.read_text(encoding="utf-8"))
    private_record = ledger["records"]["q1"]
    assert private_record["model"] == GEMINI_MODEL
    assert private_record["prompt_sha256"] == cache["prompt_sha256"]
    assert private_record["usage"] == {
        "promptTokenCount": 2,
        "totalTokenCount": 5,
    }
    assert private_record["finish_reason"] == "stop"
    assert private_record["sources"][0]["url"] == "https://example.com/private"
    assert evidence_path.stat().st_mode & 0o777 == 0o600


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


def test_deepinfra_call_is_pinned_and_keeps_token_out_of_metadata() -> None:
    secret = "deepinfra-test-token"

    def request(url: str, body: dict, headers: dict[str, str]):
        assert url == "https://api.deepinfra.com/v1/openai/chat/completions"
        assert headers == {"Authorization": f"Bearer {secret}"}
        assert body["model"] == "Qwen/Qwen3.6-27B"
        assert body["max_tokens"] == 16_384
        return {
            "id": "request-id",
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
        }

    result = call_deepinfra(
        "question", api_token=secret, request_fn=request, sleep_fn=lambda _: None
    )
    assert result[0] == "answer"
    assert secret not in json.dumps(result)


def test_deepinfra_call_accepts_pinned_gemma_model() -> None:
    def request(url: str, body: dict, headers: dict[str, str]):
        assert body["model"] == DEEPINFRA_GEMMA_MODEL
        return {
            "choices": [{"message": {"content": "answer"}, "finish_reason": "stop"}],
            "usage": {"total_tokens": 7},
        }

    result = call_deepinfra(
        "question",
        api_token="test-token",
        model=DEEPINFRA_GEMMA_MODEL,
        request_fn=request,
        sleep_fn=lambda _: None,
    )
    assert result[0] == "answer"


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
                "endpoint": GEMINI_ENDPOINT_TEMPLATE.format(model="gemini-3.6-flash"),
                "attempts": 1,
                "diagnostic_note": f"innocuous-looking field containing {secret}",
            },
        )

    cache = generate_missing(
        [("q1", "success"), ("q2", "failure")],
        cache_path=cache_path,
        provider="gemini",
        model="gemini-3.6-flash",
        call=call,
        workers=1,
        checkpoint_every=1,
    )

    assert cache["records"]["q1"]["request_metadata"] == {
        "attempts": 1,
        "endpoint": GEMINI_ENDPOINT_TEMPLATE.format(model="gemini-3.6-flash"),
    }
    assert cache["failures"]["q2"] == {
        "category": "provider",
        "error_type": "RuntimeError",
        "http_status": None,
        "model": "gemini-3.6-flash",
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
            model="gemini-3.6-flash",
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
        model="gemini-3.6-flash",
        call=interrupted_call,
        workers=1,
        checkpoint_every=10,
    )
    assert calls == ["first", "second", "second", "third"]
    assert set(resumed["records"]) == {"q1", "q2", "q3"}


def test_google_adapter_accepts_pinned_gemma_model() -> None:
    def request(url: str, body: dict, headers: dict[str, str]):
        assert url == GEMINI_ENDPOINT_TEMPLATE.format(model=GEMMA_MODEL)
        assert body["generationConfig"]["thinkingConfig"] == {"thinkingLevel": "high"}
        return {
            "candidates": [
                {"content": {"parts": [{"text": "answer"}]}, "finishReason": "STOP"}
            ],
            "usageMetadata": {"totalTokenCount": 3},
        }

    answer, _, finish_reason, _ = call_gemini(
        "question",
        api_key="unused-test-key",
        model=GEMMA_MODEL,
        request_fn=request,
        sleep_fn=lambda _: None,
    )
    assert (answer, finish_reason) == ("answer", "STOP")


def test_google_adapter_rejects_unapproved_model_before_transport() -> None:
    transport_called = False

    def request(url: str, body: dict, headers: dict[str, str]):
        nonlocal transport_called
        transport_called = True
        raise AssertionError("transport must not run")

    with pytest.raises(ValueError, match="approved"):
        call_gemini(
            "question",
            api_key="unused-test-key",
            model="gemini-other",
            request_fn=request,
            sleep_fn=lambda _: None,
        )
    assert transport_called is False


def test_resume_retries_length_truncated_record(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    answers = iter(
        [
            ("cut off", {}, "MAX_TOKENS", {}),
            ("complete answer", {}, "STOP", {}),
        ]
    )
    calls = 0

    def call(question: str):
        nonlocal calls
        calls += 1
        return next(answers)

    questions = [("q1", "question")]
    first = generate_missing(
        questions,
        cache_path=cache_path,
        provider="gemini",
        model=GEMINI_MODEL,
        call=call,
        workers=1,
    )
    assert first["records"]["q1"]["finish_reason"] == "length"

    second = generate_missing(
        questions,
        cache_path=cache_path,
        provider="gemini",
        model=GEMINI_MODEL,
        call=call,
        workers=1,
    )
    assert calls == 2
    assert second["records"]["q1"]["answer"] == "complete answer"
    assert second["records"]["q1"]["finish_reason"] == "stop"


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
            "gemini-3.6-flash",
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
