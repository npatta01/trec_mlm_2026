"""Private, bounded Claude Opus review and explicit prompt adjudication."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import time
from typing import Callable, Literal
import urllib.error
import urllib.request

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from trec_mlm.pointwise_judge import load_prompt_contract


OPENROUTER_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_API_KEY_FILE = Path("/home/npatta01/.config/trec-mlm/openrouter.key")
OPUS_MODEL_ALIAS = "anthropic/claude-opus-4.6"
OPUS_PROVIDER = "Anthropic"
MAX_SERIALIZED_REQUEST_BYTES = 256 * 1024
MAX_RESPONSE_BYTES = 128 * 1024
MAX_OUTPUT_TOKENS = 4_096
DEFAULT_CAP_USD = 10.0
INPUT_USD_PER_MILLION = 15.0
OUTPUT_USD_PER_MILLION = 75.0
_RETRYABLE_STATUS = {408, 429} | set(range(500, 600))
_FROZEN_V2_PROMPT_SHA256 = (
    "408a66ab4461d08973b645a4aa466a08b7f533beb324b902a5c3dbbe6e99a5c0"
)
_FROZEN_V2_SCHEMA_SHA256 = (
    "20bd1e2c9bc7ad51e03a5afbf3f8f6a5600fe86ad27c2b77bf8404d900decb32"
)
_PACKET_FORBIDDEN_TEXT = ("deepseek", "openrouter", "api key", "api_key")
_REVIEW_FORBIDDEN_TEXT = (
    "anthropic",
    "claude",
    "opus",
    "provider",
    "model",
    "run",
    "credential",
    "openrouter",
    "api key",
    "api_key",
    "deepseek",
)
_USAGE_NUMERIC_FIELDS = frozenset(
    {
        "input_tokens",
        "output_tokens",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
        "cache_creation_input_tokens_5m",
        "cache_creation_input_tokens_1h",
        "reasoning_tokens",
    }
)


class OpusRecommendation(BaseModel):
    """One exact, machine-applicable recommendation from the reviewer."""

    model_config = ConfigDict(extra="forbid", strict=True)

    recommendation_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    target: Literal["prompt", "schema"]
    expected: str = Field(min_length=1, max_length=100_000)
    replacement: str = Field(min_length=1, max_length=100_000)
    reason: str = Field(min_length=1, max_length=4_000)


class OpusPromptReview(BaseModel):
    """Strictly parsed advisory review bound to one sealed packet."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1]
    packet_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    summary: str = Field(min_length=1, max_length=4_000)
    recommendations: list[OpusRecommendation] = Field(max_length=24)

    @field_validator("recommendations")
    @classmethod
    def _recommendation_ids_are_unique(
        cls, recommendations: list[OpusRecommendation]
    ) -> list[OpusRecommendation]:
        ids = [recommendation.recommendation_id for recommendation in recommendations]
        if len(ids) != len(set(ids)):
            raise ValueError("recommendation IDs must be unique")
        return recommendations


class AdjudicationDecision(BaseModel):
    """An explicit human/controller decision for a single recommendation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    decision: Literal["accepted", "rejected"]
    reason: str = Field(min_length=1, max_length=4_000)


@dataclass(slots=True)
class SpendBudget:
    """One-review spend bound with conservative reservation and reconciliation."""

    cap_usd: float = DEFAULT_CAP_USD
    spent_usd: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.cap_usd) or not 0 < self.cap_usd <= DEFAULT_CAP_USD:
            raise ValueError("Opus review budget must be within the authorized cap")
        if not math.isfinite(self.spent_usd) or not 0 <= self.spent_usd <= self.cap_usd:
            raise ValueError("Opus review spend is invalid")

    def reserve(self, amount: float) -> None:
        if (
            not math.isfinite(amount)
            or amount < 0
            or self.spent_usd + amount > self.cap_usd
        ):
            raise RuntimeError("Opus review request would exceed its spend cap")
        self.spent_usd += amount

    def reconcile(self, reservation: float, actual: float) -> None:
        if not math.isfinite(actual) or actual < 0 or actual > reservation:
            raise ValueError("provider-reported Opus cost exceeds its reservation")
        self.spent_usd -= reservation - actual


def validate_opus_review_packet(
    packet: Mapping[str, object], *, protocol_dir: Path | None = None
) -> dict[str, object]:
    """Validate the sealed Task 1 packet before it is transmitted or parsed."""
    if type(packet) is not dict or set(packet) != {
        "version",
        "contract",
        "adjudication_questions",
        "reviewer_instruction",
        "packet_sha256",
    }:
        raise ValueError("Opus review packet must contain exact keys")
    body = {key: packet[key] for key in packet if key != "packet_sha256"}
    if packet["version"] != 1 or packet["packet_sha256"] != _sha256_json(body):
        raise ValueError("Opus review packet identity is invalid")
    contract = packet["contract"]
    if type(contract) is not dict or set(contract) != {
        "version",
        "prompt",
        "schema",
        "prompt_sha256",
        "schema_sha256",
    }:
        raise ValueError("Opus review contract must contain exact keys")
    if (
        contract["version"] != "v2"
        or type(contract["prompt"]) is not str
        or not contract["prompt"].strip()
        or type(contract["schema"]) is not dict
        or contract["prompt_sha256"]
        != hashlib.sha256(contract["prompt"].encode("utf-8")).hexdigest()
        or not _is_sha256(contract["schema_sha256"])
    ):
        raise ValueError("Opus review packet contract binding is invalid")
    if (
        type(packet["adjudication_questions"]) is not list
        or not packet["adjudication_questions"]
        or any(
            type(question) is not str or not question.strip()
            for question in packet["adjudication_questions"]
        )
        or type(packet["reviewer_instruction"]) is not str
        or not packet["reviewer_instruction"].strip()
    ):
        raise ValueError("Opus review packet content is invalid")
    if protocol_dir is not None:
        frozen = load_prompt_contract(Path(protocol_dir), version="v2")
        if (
            contract["prompt"] != frozen.prompt
            or contract["schema"] != frozen.schema
            or contract["prompt_sha256"]
            != hashlib.sha256(frozen.prompt_path.read_bytes()).hexdigest()
            or contract["schema_sha256"]
            != hashlib.sha256(frozen.schema_path.read_bytes()).hexdigest()
        ):
            raise ValueError(
                "Opus review packet does not bind the exact frozen v2 contract"
            )
    elif (
        contract["prompt_sha256"] != _FROZEN_V2_PROMPT_SHA256
        or contract["schema_sha256"] != _FROZEN_V2_SCHEMA_SHA256
    ):
        raise ValueError(
            "Opus review packet does not bind the pinned frozen v2 contract"
        )
    _reject_forbidden_text(packet, label="packet", tokens=_PACKET_FORBIDDEN_TEXT)
    return dict(packet)


def parse_opus_review(
    raw: str, *, packet: Mapping[str, object], protocol_dir: Path | None = None
) -> OpusPromptReview:
    """Parse only a strict review for the exact sealed input packet."""
    validated_packet = validate_opus_review_packet(packet, protocol_dir=protocol_dir)
    if type(raw) is not str or len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
        raise ValueError("Opus review response is too large or not text")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (_DuplicateJsonKeyError, json.JSONDecodeError) as error:
        raise ValueError("Opus review is not valid JSON") from error
    try:
        review = OpusPromptReview.model_validate(value)
    except ValidationError as error:
        raise ValueError("Opus review has invalid exact keys or fields") from error
    if review.packet_sha256 != validated_packet["packet_sha256"]:
        raise ValueError("Opus review packet identity does not match")
    _reject_forbidden_text(
        review.model_dump(mode="json"), label="review", tokens=_REVIEW_FORBIDDEN_TEXT
    )
    _validate_recommendations_against_packet(review, validated_packet)
    return review


def adjudicate_opus_review(
    *,
    packet: Mapping[str, object],
    review: OpusPromptReview,
    decisions: Mapping[str, object],
) -> dict[str, object]:
    """Apply exactly the accepted review replacements and seal the receipt."""
    validated_packet = validate_opus_review_packet(packet)
    if review.packet_sha256 != validated_packet["packet_sha256"]:
        raise ValueError("review is not bound to this packet")
    expected_ids = {item.recommendation_id for item in review.recommendations}
    if type(decisions) is not dict or set(decisions) != expected_ids:
        raise ValueError("adjudication decisions must exactly cover recommendations")
    contract = validated_packet["contract"]
    assert isinstance(contract, dict)
    prompt = contract["prompt"]
    schema = contract["schema"]
    assert isinstance(prompt, str) and isinstance(schema, dict)
    ledger: list[dict[str, str]] = []
    accepted = 0
    for recommendation in review.recommendations:
        try:
            decision = AdjudicationDecision.model_validate(
                decisions[recommendation.recommendation_id]
            )
        except ValidationError as error:
            raise ValueError("adjudication decision is invalid") from error
        ledger.append(
            {
                "recommendation_id": recommendation.recommendation_id,
                "target": recommendation.target,
                "decision": decision.decision,
                "reason": decision.reason,
            }
        )
        if decision.decision == "accepted":
            accepted += 1
            if recommendation.target == "prompt":
                if prompt.count(recommendation.expected) != 1:
                    raise ValueError(
                        "accepted prompt replacement no longer binds exactly"
                    )
                prompt = prompt.replace(
                    recommendation.expected, recommendation.replacement, 1
                )
            else:
                schema = _apply_schema_replacement(schema, recommendation)
    contract_version = "v2-opus-1" if accepted else "v2"
    revised_contract = {
        "version": contract_version,
        "source_version": "v2",
        "prompt": prompt,
        "schema": schema,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "schema_sha256": _sha256_json(schema),
    }
    body = {
        "version": 1,
        "packet_sha256": validated_packet["packet_sha256"],
        "review_sha256": _sha256_json(review.model_dump(mode="json")),
        "contract": revised_contract,
        "recommendation_ledger": ledger,
    }
    receipt = {**body, "adjudication_sha256": _sha256_json(body)}
    _reject_forbidden_text(receipt, label="adjudication receipt")
    return receipt


class OpusReviewCall:
    """A single Opus review request with one retry for transient transport errors."""

    def __init__(
        self,
        *,
        api_key: str,
        budget: SpendBudget,
        timeout_seconds: float,
        post: Callable[[urllib.request.Request, float], object] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if type(api_key) is not str or not api_key.strip():
            raise ValueError("Opus review API credential is unavailable")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Opus review timeout must be positive")
        self._api_key = api_key.strip()
        self._budget = budget
        self._timeout_seconds = timeout_seconds
        self._post = post or _post_json
        self._sleep = sleep
        self._unreconciled_reservation_usd = 0.0

    def __call__(
        self, packet: Mapping[str, object], *, protocol_dir: Path | None = None
    ) -> tuple[OpusPromptReview, dict[str, object]]:
        validated_packet = validate_opus_review_packet(
            packet, protocol_dir=protocol_dir
        )
        payload = _build_request_payload(validated_packet)
        encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(
            "utf-8"
        )
        if len(encoded) > MAX_SERIALIZED_REQUEST_BYTES:
            raise ValueError("Opus review request exceeds its response-safe size limit")
        reservation = _worst_case_cost(len(encoded))
        request = urllib.request.Request(
            OPENROUTER_ENDPOINT,
            data=encoded,
            headers={
                "Accept": "application/json",
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        body = self._send_with_one_retry(request, reservation)
        raw, usage, cost = _parse_provider_response(body)
        self._budget.reconcile(reservation, cost)
        review = parse_opus_review(
            raw, packet=validated_packet, protocol_dir=protocol_dir
        )
        return review, {
            "usage": usage,
            "cost_usd": cost,
            "model_alias": OPUS_MODEL_ALIAS,
            "unreconciled_reservation_usd": self._unreconciled_reservation_usd,
        }

    def _send_with_one_retry(
        self, request: urllib.request.Request, reservation: float
    ) -> object:
        for attempt in range(2):
            self._budget.reserve(reservation)
            try:
                return self._post(request, self._timeout_seconds)
            except urllib.error.HTTPError as error:
                self._unreconciled_reservation_usd += reservation
                if error.code not in _RETRYABLE_STATUS or attempt == 1:
                    raise
                self._sleep(_retry_delay_seconds(error))
            except (urllib.error.URLError, TimeoutError):
                self._unreconciled_reservation_usd += reservation
                if attempt == 1:
                    raise
                self._sleep(1.0)
        raise RuntimeError("unreachable Opus retry state")


class ClaudeCliReviewCall:
    """Non-interactive Claude Code transport for the authorized Opus review."""

    def __init__(
        self,
        *,
        budget: SpendBudget,
        timeout_seconds: float,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Claude review timeout must be positive")
        self._budget = budget
        self._timeout_seconds = timeout_seconds
        self._runner = runner
        self._unreconciled_reservation_usd = 0.0

    def __call__(
        self, packet: Mapping[str, object], *, protocol_dir: Path | None = None
    ) -> tuple[OpusPromptReview, dict[str, object]]:
        validated_packet = validate_opus_review_packet(
            packet, protocol_dir=protocol_dir
        )
        schema_text = json.dumps(
            _transport_review_schema(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        prompt_text = json.dumps(validated_packet, ensure_ascii=False)
        argv = [
            "claude",
            "-p",
            "--model",
            "opus",
            "--output-format",
            "json",
            "--json-schema",
            schema_text,
            "--max-budget-usd",
            f"{self._budget.cap_usd - self._budget.spent_usd:.6f}",
            "--no-session-persistence",
            "--tools",
            "",
        ]
        encoded_size = len(prompt_text.encode("utf-8")) + len(
            schema_text.encode("utf-8")
        )
        reservation = _worst_case_cost(encoded_size)
        for attempt in range(2):
            self._budget.reserve(reservation)
            try:
                completed = self._runner(
                    argv,
                    input=prompt_text,
                    text=True,
                    capture_output=True,
                    timeout=self._timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                self._unreconciled_reservation_usd += reservation
                if attempt == 1:
                    raise ValueError("Claude CLI review transport failed") from error
                continue
            if completed.returncode != 0:
                self._unreconciled_reservation_usd += reservation
                if attempt == 1:
                    raise ValueError("Claude CLI review failed")
                continue
            if (
                type(completed.stdout) is not str
                or len(completed.stdout.encode("utf-8")) > MAX_RESPONSE_BYTES
            ):
                raise ValueError("Claude CLI review response exceeds its size limit")
            raw, usage, cost = _parse_claude_cli_envelope(completed.stdout)
            self._budget.reconcile(reservation, cost)
            # Parse semantic content only after accounting for the successful
            # transport; malformed/schema-invalid content is never retried.
            review = parse_opus_review(
                raw, packet=validated_packet, protocol_dir=protocol_dir
            )
            return review, {
                "usage": usage,
                "cost_usd": cost,
                "model_alias": OPUS_MODEL_ALIAS,
                "unreconciled_reservation_usd": self._unreconciled_reservation_usd,
            }
        raise RuntimeError("unreachable Claude CLI retry state")


def build_review_receipt(
    *,
    packet: Mapping[str, object],
    review: OpusPromptReview,
    provider: Mapping[str, object],
) -> dict[str, object]:
    """Create a private review receipt without recording credentials or headers."""
    validated_packet = validate_opus_review_packet(packet)
    if review.packet_sha256 != validated_packet["packet_sha256"]:
        raise ValueError("review is not bound to the packet")
    if type(provider) is not dict or set(provider) != {
        "usage",
        "cost_usd",
        "model_alias",
        "unreconciled_reservation_usd",
    }:
        raise ValueError("provider review receipt has invalid keys")
    if (
        provider["model_alias"] != OPUS_MODEL_ALIAS
        or type(provider["cost_usd"]) not in {int, float}
        or type(provider["unreconciled_reservation_usd"]) not in {int, float}
        or not math.isfinite(float(provider["cost_usd"]))
        or float(provider["cost_usd"]) < 0
        or not math.isfinite(float(provider["unreconciled_reservation_usd"]))
        or float(provider["unreconciled_reservation_usd"]) < 0
    ):
        raise ValueError("provider review receipt has invalid identity or cost")
    usage = _sanitize_usage(provider["usage"])
    body = {
        "version": 1,
        "packet_sha256": validated_packet["packet_sha256"],
        "model_alias": OPUS_MODEL_ALIAS,
        "provider": {
            "usage": usage,
            "cost_usd": float(provider["cost_usd"]),
            "unreconciled_reservation_usd": float(
                provider["unreconciled_reservation_usd"]
            ),
        },
        "review": review.model_dump(mode="json"),
    }
    receipt = {**body, "review_sha256": _sha256_json(body)}
    _reject_forbidden_text(receipt, label="review receipt")
    return receipt


def load_api_key(path: Path, *, environ: Mapping[str, str] | None = None) -> str:
    """Read an authorized OpenRouter credential without placing it in an artifact."""
    environment = os.environ if environ is None else environ
    from_environment = environment.get("OPENROUTER_API_KEY")
    if from_environment is not None:
        if not from_environment.strip():
            raise ValueError("OPENROUTER_API_KEY must not be empty")
        return from_environment.strip()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(Path(path), flags)
    except OSError as error:
        raise ValueError("Opus review API credential is unavailable") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode) != 0o600:
            raise ValueError(
                "Opus review API credential must be a regular mode-0600 file"
            )
        if opened.st_size > 16_384:
            raise ValueError("Opus review API credential file is too large")
        raw = os.read(descriptor, 16_385)
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if _file_identity(opened) != _file_identity(after) or _file_identity(
            after
        ) != _file_identity(current):
            raise ValueError("Opus review API credential changed during read")
    finally:
        os.close(descriptor)
    try:
        key = raw.decode("utf-8").strip()
    except UnicodeDecodeError as error:
        raise ValueError("Opus review API credential must be UTF-8") from error
    if not key:
        raise ValueError("Opus review API credential must not be empty")
    return key


def read_private_json(path: Path) -> dict[str, object]:
    """Read a small private JSON artifact without following symlinks."""
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(Path(path), flags)
    except OSError as error:
        raise ValueError("private Opus artifact could not be opened safely") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode) != 0o600:
            raise ValueError("private Opus artifact must be a regular mode-0600 file")
        raw = os.read(descriptor, MAX_RESPONSE_BYTES + 1)
        if len(raw) > MAX_RESPONSE_BYTES:
            raise ValueError("private Opus artifact is too large")
        after = os.fstat(descriptor)
        current = os.stat(path, follow_symlinks=False)
        if _file_identity(opened) != _file_identity(after) or _file_identity(
            after
        ) != _file_identity(current):
            raise ValueError("private Opus artifact changed during read")
    finally:
        os.close(descriptor)
    try:
        value = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_object_without_duplicate_keys
        )
    except (UnicodeDecodeError, _DuplicateJsonKeyError, json.JSONDecodeError) as error:
        raise ValueError("private Opus artifact is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError("private Opus artifact must be a JSON object")
    return value


def write_private_json(path: Path, value: object) -> None:
    """Atomically write one JSON artifact with private file and parent modes."""
    path = Path(path)
    if path.parent.exists():
        if (
            not path.parent.is_dir()
            or stat.S_IMODE(path.parent.stat().st_mode) != 0o700
        ):
            raise ValueError(
                "private artifact parent must be an existing mode-0700 directory"
            )
    else:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(
                value,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        Path(temporary).unlink(missing_ok=True)


def _build_request_payload(packet: Mapping[str, object]) -> dict[str, object]:
    return {
        "model": OPUS_MODEL_ALIAS,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return only one JSON object matching the supplied review schema. "
                    "Do not identify any target model, provider, credential, or run."
                ),
            },
            {
                "role": "user",
                "content": "Review this sealed packet:\n"
                + json.dumps(packet, ensure_ascii=False),
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "opus_pointwise_prompt_review",
                "strict": True,
                "schema": _transport_review_schema(),
            },
        },
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "provider": {
            "only": [OPUS_PROVIDER],
            "allow_fallbacks": False,
            "require_parameters": True,
            "max_price": {
                "prompt": INPUT_USD_PER_MILLION,
                "completion": OUTPUT_USD_PER_MILLION,
            },
        },
    }


def _transport_review_schema() -> dict[str, object]:
    """Adapt only the provider transport schema; local Pydantic stays strict."""
    return _remove_schema_keyword(OpusPromptReview.model_json_schema(), "maxItems")


def _remove_schema_keyword(value: object, keyword: str) -> object:
    if isinstance(value, dict):
        return {
            key: _remove_schema_keyword(item, keyword)
            for key, item in value.items()
            if key != keyword
        }
    if isinstance(value, list):
        return [_remove_schema_keyword(item, keyword) for item in value]
    return value


def _parse_provider_response(body: object) -> tuple[str, dict[str, object], float]:
    if type(body) is not dict:
        raise ValueError("Opus provider response must be a JSON object")
    if body.get("model") != OPUS_MODEL_ALIAS:
        raise ValueError("Opus provider returned an unexpected model identity")
    choices = body.get("choices")
    if type(choices) is not list or len(choices) != 1 or type(choices[0]) is not dict:
        raise ValueError("Opus provider response must contain one choice")
    if choices[0].get("finish_reason") != "stop":
        raise ValueError("Opus provider response must finish normally")
    message = choices[0].get("message")
    usage = body.get("usage")
    if type(message) is not dict or type(message.get("content")) is not str:
        raise ValueError("Opus provider response content is invalid")
    if type(usage) is not dict or type(usage.get("cost")) not in {int, float}:
        raise ValueError("Opus provider response must report numeric cost")
    cost = float(usage["cost"])
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("Opus provider cost is invalid")
    return message["content"], _sanitize_usage(usage), cost


def _parse_claude_cli_envelope(raw: str) -> tuple[str, dict[str, int | float], float]:
    try:
        envelope = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except (_DuplicateJsonKeyError, json.JSONDecodeError) as error:
        raise ValueError("Claude CLI envelope is not valid JSON") from error
    allowed = {
        "type",
        "subtype",
        "is_error",
        "result",
        "model",
        "usage",
        "total_cost_usd",
        "duration_ms",
        "duration_api_ms",
        "num_turns",
        "session_id",
    }
    if (
        type(envelope) is not dict
        or not {"type", "result", "model"} <= set(envelope)
        or not set(envelope) <= allowed
        or envelope["type"] != "result"
        or type(envelope["result"]) is not str
        or type(envelope["model"]) is not str
        or "opus" not in envelope["model"].casefold()
        or envelope.get("is_error") is True
    ):
        raise ValueError("Claude CLI envelope has invalid exact result/model fields")
    usage = _sanitize_usage(envelope.get("usage", {}))
    cost_value = envelope.get("total_cost_usd", 0.0)
    if type(cost_value) not in {int, float} or isinstance(cost_value, bool):
        raise ValueError("Claude CLI envelope cost is invalid")
    cost = float(cost_value)
    if not math.isfinite(cost) or cost < 0:
        raise ValueError("Claude CLI envelope cost is invalid")
    return envelope["result"], usage, cost


def _post_json(request: urllib.request.Request, timeout_seconds: float) -> object:
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _RejectRedirects()
    )
    with opener.open(request, timeout=timeout_seconds) as response:
        if response.geturl() != OPENROUTER_ENDPOINT:
            raise ValueError("Opus review redirects are not allowed")
        raw = response.read(MAX_RESPONSE_BYTES + 1)
    if len(raw) > MAX_RESPONSE_BYTES:
        raise ValueError("Opus provider response exceeds its size limit")
    try:
        return json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Opus provider response is not valid UTF-8 JSON") from error


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: object, **kwargs: object) -> None:
        return None


class _DuplicateJsonKeyError(ValueError):
    pass


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonKeyError(key)
        value[key] = item
    return value


def _apply_schema_replacement(
    schema: dict[str, object], recommendation: OpusRecommendation
) -> dict[str, object]:
    try:
        expected = json.loads(
            recommendation.expected, object_pairs_hook=_object_without_duplicate_keys
        )
        replacement = json.loads(
            recommendation.replacement, object_pairs_hook=_object_without_duplicate_keys
        )
    except (_DuplicateJsonKeyError, json.JSONDecodeError) as error:
        raise ValueError(
            "schema recommendation must contain valid JSON objects"
        ) from error
    if (
        type(expected) is not dict
        or type(replacement) is not dict
        or expected != schema
    ):
        raise ValueError("accepted schema replacement no longer binds exactly")
    return replacement


def _validate_recommendations_against_packet(
    review: OpusPromptReview, packet: Mapping[str, object]
) -> None:
    contract = packet["contract"]
    assert isinstance(contract, dict)
    prompt = contract["prompt"]
    schema = contract["schema"]
    assert isinstance(prompt, str) and isinstance(schema, dict)
    for recommendation in review.recommendations:
        if recommendation.target == "prompt":
            if prompt.count(recommendation.expected) != 1:
                raise ValueError(
                    "prompt recommendation must bind exactly one current substring"
                )
        else:
            _apply_schema_replacement(schema, recommendation)


def _sanitize_usage(value: object) -> dict[str, int | float]:
    if type(value) is not dict:
        raise ValueError("provider usage must be an object")
    sanitized: dict[str, int | float] = {}
    for key, item in value.items():
        if key == "cost":
            continue
        if key not in _USAGE_NUMERIC_FIELDS:
            raise ValueError("provider usage contains an unexpected field")
        if type(item) not in {int, float} or isinstance(item, bool):
            raise ValueError("provider usage fields must be numeric")
        numeric = float(item)
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("provider usage fields must be finite and nonnegative")
        sanitized[key] = item
    return sanitized


def _reject_forbidden_text(
    value: object,
    *,
    label: str,
    tokens: tuple[str, ...] | frozenset[str] = _PACKET_FORBIDDEN_TEXT,
) -> None:
    serialized = json.dumps(value, ensure_ascii=False, allow_nan=False).casefold()
    if any(token in serialized for token in tokens):
        raise ValueError(f"{label} contains forbidden identity or credential text")


def _worst_case_cost(request_bytes: int) -> float:
    return (
        (request_bytes + 2_048) * INPUT_USD_PER_MILLION
        + MAX_OUTPUT_TOKENS * OUTPUT_USD_PER_MILLION
    ) / 1_000_000


def _retry_delay_seconds(error: urllib.error.HTTPError) -> float:
    raw = error.headers.get("Retry-After") if hasattr(error.headers, "get") else None
    try:
        delay = float(raw) if type(raw) is str else 1.0
    except ValueError:
        delay = 1.0
    return min(60.0, max(1.0, delay)) if math.isfinite(delay) else 1.0


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON is forbidden: {value}")
