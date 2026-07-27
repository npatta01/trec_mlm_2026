from __future__ import annotations

import hashlib
import json
import math
import stat
from pathlib import Path
import subprocess
import urllib.error

import pytest

from scripts.run_opus_pointwise_prompt_review import main
from trec_mlm.pointwise_opus_review import (
    ClaudeCliReviewCall,
    OpusPromptReview,
    OpusReviewCall,
    OPUS_MODEL_ALIAS,
    SpendBudget,
    adjudicate_opus_review,
    build_review_receipt,
    parse_opus_review,
    validate_opus_review_packet,
    write_private_json,
)
from trec_mlm.pointwise_production import build_opus_prompt_review_packet


def _packet(tmp_path: Path) -> dict[str, object]:
    protocol_dir = Path(__file__).parents[1] / "protocols" / "pointwise_judge_bakeoff"
    return build_opus_prompt_review_packet(
        protocol_dir=protocol_dir,
        output_path=tmp_path / "packet.json",
    )


def _protocol_dir() -> Path:
    return Path(__file__).parents[1] / "protocols" / "pointwise_judge_bakeoff"


def _review(packet: dict[str, object]) -> dict[str, object]:
    prompt = packet["contract"]["prompt"]
    assert isinstance(prompt, str)
    return {
        "version": 1,
        "packet_sha256": packet["packet_sha256"],
        "summary": "The contract is clear, but the opening instruction can be more precise.",
        "recommendations": [
            {
                "recommendation_id": "prompt-opening",
                "target": "prompt",
                "expected": "You are a careful pointwise judge of a candidate response.",
                "replacement": "You are a careful, deterministic pointwise judge of a candidate response.",
                "reason": "This makes the deterministic-output objective explicit.",
            }
        ],
    }


def test_parse_opus_review_binds_packet_and_rejects_malformed_or_identity_text(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    raw = json.dumps(_review(packet))

    review = parse_opus_review(raw, packet=packet)

    assert isinstance(review, OpusPromptReview)
    assert review.packet_sha256 == packet["packet_sha256"]
    with pytest.raises(ValueError, match="packet identity"):
        parse_opus_review(
            raw.replace(str(packet["packet_sha256"]), "0" * 64), packet=packet
        )
    with pytest.raises(ValueError, match="valid JSON|exact keys"):
        parse_opus_review('{"version": 1, "version": 1}', packet=packet)
    identity_bearing = _review(packet)
    identity_bearing["summary"] = "This names DeepSeek and must not be accepted."
    with pytest.raises(ValueError, match="identity"):
        parse_opus_review(json.dumps(identity_bearing), packet=packet)


def test_packet_validation_rebinds_to_the_exact_current_v2_contract(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    altered = json.loads(json.dumps(packet))
    altered["contract"]["prompt"] = "Altered prompt."
    altered["contract"]["prompt_sha256"] = hashlib.sha256(
        altered["contract"]["prompt"].encode()
    ).hexdigest()
    body = {key: altered[key] for key in altered if key != "packet_sha256"}
    altered["packet_sha256"] = hashlib.sha256(
        json.dumps(
            body, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
    ).hexdigest()

    validate_opus_review_packet(packet, protocol_dir=_protocol_dir())
    with pytest.raises(ValueError, match="exact frozen v2 contract"):
        validate_opus_review_packet(altered, protocol_dir=_protocol_dir())

    altered["contract"]["prompt"] = "Forged prompt."
    altered["contract"]["prompt_sha256"] = hashlib.sha256(
        altered["contract"]["prompt"].encode()
    ).hexdigest()
    altered["packet_sha256"] = hashlib.sha256(
        json.dumps(
            {key: altered[key] for key in altered if key != "packet_sha256"},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    with pytest.raises(ValueError, match="frozen|pinned"):
        validate_opus_review_packet(altered)


def test_adjudication_records_rejections_without_mutating_the_contract(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    review = parse_opus_review(json.dumps(_review(packet)), packet=packet)

    receipt = adjudicate_opus_review(
        packet=packet,
        review=review,
        decisions={
            "prompt-opening": {
                "decision": "rejected",
                "reason": "Keep the existing wording for the frozen v2 baseline.",
            }
        },
    )

    assert receipt["contract"]["version"] == "v2"
    assert receipt["contract"]["prompt"] == packet["contract"]["prompt"]
    assert receipt["recommendation_ledger"] == [
        {
            "recommendation_id": "prompt-opening",
            "target": "prompt",
            "decision": "rejected",
            "reason": "Keep the existing wording for the frozen v2 baseline.",
        }
    ]


def test_adjudication_applies_only_explicitly_accepted_recommendations(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    review_body = _review(packet)
    review_body["recommendations"].append(
        {
            "recommendation_id": "prompt-ending",
            "target": "prompt",
            "expected": "Return exactly one JSON object",
            "replacement": "Return precisely one JSON object",
            "reason": "Use a single consistent imperative.",
        }
    )
    review = parse_opus_review(json.dumps(review_body), packet=packet)

    receipt = adjudicate_opus_review(
        packet=packet,
        review=review,
        decisions={
            "prompt-opening": {"decision": "accepted", "reason": "Approved."},
            "prompt-ending": {"decision": "rejected", "reason": "No need."},
        },
    )

    assert receipt["contract"]["version"] == "v2-opus-1"
    assert "deterministic pointwise judge" in receipt["contract"]["prompt"]
    assert "Return exactly one JSON object" in receipt["contract"]["prompt"]
    assert [row["decision"] for row in receipt["recommendation_ledger"]] == [
        "accepted",
        "rejected",
    ]


def test_credential_free_dry_run_writes_a_private_receipt_without_a_key(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    packet_path = tmp_path / "packet.json"
    output = tmp_path / "nested" / "preflight.json"

    assert (
        main(
            [
                "--packet",
                str(packet_path),
                "--output",
                str(output),
                "--protocol-dir",
                str(_protocol_dir()),
                "--dry-run",
            ]
        )
        == 0
    )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt == {
        "version": 1,
        "mode": "dry-run",
        "packet_sha256": packet["packet_sha256"],
        "model_alias": "anthropic/claude-opus-4.6",
    }
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert "key" not in output.read_text(encoding="utf-8").casefold()


def test_opus_call_retries_once_and_keeps_a_conservative_failed_reservation(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    attempts = 0

    def post(_request: object, _timeout: float) -> object:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise urllib.error.HTTPError("https://example.test", 429, "busy", {}, None)
        return {
            "model": OPUS_MODEL_ALIAS,
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": json.dumps(_review(packet))},
                }
            ],
            "usage": {"cost": 0.01, "prompt_tokens": 100, "completion_tokens": 50},
        }

    budget = SpendBudget(2.0)
    review, provider = OpusReviewCall(
        api_key="not-serialized",
        budget=budget,
        timeout_seconds=1.0,
        post=post,
        sleep=lambda _seconds: None,
    )(packet, protocol_dir=_protocol_dir())

    assert review.packet_sha256 == packet["packet_sha256"]
    assert attempts == 2
    assert provider["cost_usd"] == 0.01
    assert provider["unreconciled_reservation_usd"] > 0
    assert budget.spent_usd > provider["cost_usd"]


def test_cli_adjudication_mode_writes_the_sealed_contract_and_ledger(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    packet_path = tmp_path / "packet.json"
    write_private_json(packet_path, packet)
    review = parse_opus_review(json.dumps(_review(packet)), packet=packet)
    review_receipt = build_review_receipt(
        packet=packet,
        review=review,
        provider={
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "cost_usd": 0.01,
            "model_alias": OPUS_MODEL_ALIAS,
            "unreconciled_reservation_usd": 0.0,
        },
    )
    review_path = tmp_path / "review.json"
    decisions_path = tmp_path / "decisions.json"
    adjudication_path = tmp_path / "adjudication.json"
    write_private_json(review_path, review_receipt)
    write_private_json(
        decisions_path,
        {
            "prompt-opening": {
                "decision": "rejected",
                "reason": "Retain the frozen wording.",
            }
        },
    )

    assert (
        main(
            [
                "--packet",
                str(packet_path),
                "--protocol-dir",
                str(_protocol_dir()),
                "--review-input",
                str(review_path),
                "--decisions",
                str(decisions_path),
                "--adjudication-output",
                str(adjudication_path),
                "--output",
                str(tmp_path / "unused.json"),
                "--adjudicate-only",
            ]
        )
        == 0
    )
    adjudication = json.loads(adjudication_path.read_text(encoding="utf-8"))
    assert adjudication["contract"]["version"] == "v2"
    assert adjudication["recommendation_ledger"][0]["decision"] == "rejected"
    assert stat.S_IMODE(adjudication_path.stat().st_mode) == 0o600


def test_review_content_rejects_all_identity_terms(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    for term in (
        "Anthropic",
        "Claude",
        "Opus",
        "provider",
        "model",
        "run",
        "credential",
        "OpenRouter",
        "API key",
        "DeepSeek",
    ):
        body = _review(packet)
        body["summary"] = f"The review mentions {term}."
        with pytest.raises(ValueError, match="identity"):
            parse_opus_review(json.dumps(body), packet=packet)


def test_provider_receipt_allowlists_numeric_usage_and_costs(tmp_path: Path) -> None:
    packet = _packet(tmp_path)
    review = parse_opus_review(json.dumps(_review(packet)), packet=packet)
    provider = {
        "usage": {"input_tokens": 1, "output_tokens": 2},
        "cost_usd": 0.01,
        "model_alias": OPUS_MODEL_ALIAS,
        "unreconciled_reservation_usd": 0.0,
    }
    assert build_review_receipt(packet=packet, review=review, provider=provider)
    for bad_usage in ({"secret": "do-not-copy"}, {"input_tokens": -1}):
        bad = dict(provider)
        bad["usage"] = bad_usage
        with pytest.raises(ValueError, match="usage"):
            build_review_receipt(packet=packet, review=review, provider=bad)
    for bad_cost in (-0.01, math.nan, math.inf):
        bad = dict(provider)
        bad["cost_usd"] = bad_cost
        with pytest.raises(ValueError, match="cost"):
            build_review_receipt(packet=packet, review=review, provider=bad)


def test_private_json_does_not_change_an_existing_parent_mode(tmp_path: Path) -> None:
    parent = tmp_path / "caller-owned"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    path = parent / "receipt.json"
    with pytest.raises(ValueError, match="mode-0700"):
        write_private_json(path, {"safe": True})
    assert stat.S_IMODE(parent.stat().st_mode) == 0o755


def test_claude_cli_transport_uses_bounded_json_mode_without_credentials(
    tmp_path: Path,
) -> None:
    packet = _packet(tmp_path)
    calls: list[tuple[list[str], str]] = []

    def runner(argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((argv, kwargs["input"]))
        envelope = {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "result": json.dumps(_review(packet)),
            "model": "opus",
            "usage": {"input_tokens": 10, "output_tokens": 20},
            "total_cost_usd": 0.01,
        }
        return subprocess.CompletedProcess(argv, 0, json.dumps(envelope), "")

    review, provider = ClaudeCliReviewCall(
        budget=SpendBudget(2.0), timeout_seconds=3.0, runner=runner
    )(packet, protocol_dir=_protocol_dir())
    assert review.packet_sha256 == packet["packet_sha256"]
    assert provider["usage"] == {"input_tokens": 10, "output_tokens": 20}
    assert provider["cost_usd"] == 0.01
    argv, input_text = calls[0]
    assert argv[:5] == ["claude", "-p", "--model", "opus", "--output-format"]
    assert "--json-schema" in argv
    assert "--max-budget-usd" in argv
    assert "--no-session-persistence" in argv
    assert argv[-2:] == ["--tools", ""]
    assert "not-serialized" not in input_text
