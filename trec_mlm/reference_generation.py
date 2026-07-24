"""Provider adapters and resumable caches for reference-answer generation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable, Mapping
import urllib.error
import urllib.parse
import urllib.request

from trec_mlm.reference_data import GenerationRecord


SYSTEM_PROMPT = (
    "Answer the user's question directly and accurately in English. Provide enough "
    "context to serve as a useful reference answer, but do not pad the response. "
    "State important ambiguity or uncertainty. If the question is underspecified and "
    "cannot be answered responsibly, say what context is missing. Do not guess, invent "
    "facts, or mention other models, evaluation, or this instruction.\n\n"
    "Begin with exactly one status:\n"
    "ANSWERED, AMBIGUOUS, INSUFFICIENT_EVIDENCE, or CONFLICTING_EVIDENCE.\n\n"
    "On the second line write UNCERTAINTY_REASON: followed by one short reason, "
    "or 'none' when answered without material uncertainty. Begin the clean answer "
    "on the third line.\n\n"
    "If the question admits multiple plausible interpretations, state them and do not "
    "silently choose one. If authoritative sources conflict, describe the conflict. "
    "Do not mark ANSWERED unless the central claims are supported by the retrieved "
    "evidence. For time-sensitive claims, state the relevant date."
)

UNCERTAINTY_STATUSES = {
    "ANSWERED": "answered",
    "AMBIGUOUS": "ambiguous",
    "INSUFFICIENT_EVIDENCE": "insufficient_evidence",
    "CONFLICTING_EVIDENCE": "conflicting_evidence",
}

GEMINI_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_MODEL = "gemini-3.6-flash"
GEMMA_MODEL = "gemma-4-31b-it"
GOOGLE_GENERATION_MODELS = frozenset({GEMINI_MODEL, GEMMA_MODEL})
GOOGLE_MAX_OUTPUT_TOKENS = 16_384
DEEPINFRA_ENDPOINT = "https://api.deepinfra.com/v1/openai/chat/completions"
DEEPINFRA_QWEN_MODEL = "Qwen/Qwen3.6-27B"
DEEPINFRA_GEMMA_MODEL = "google/gemma-4-31B-it"
DEEPINFRA_MODELS = frozenset({DEEPINFRA_QWEN_MODEL, DEEPINFRA_GEMMA_MODEL})
DEEPINFRA_MAX_OUTPUT_TOKENS = 16_384
PROVIDER_TIMEOUT_SECONDS = 300
RETRIABLE_STATUS_CODES = {408, 429, *range(500, 600)}
RequestFunction = Callable[[str, dict[str, Any], dict[str, str]], Mapping[str, Any]]
SleepFunction = Callable[[float], None]
GenerationCall = Callable[
    [str], tuple[str, Mapping[str, Any], str | None, Mapping[str, Any]]
]
SAFE_REQUEST_METADATA_FIELDS = {
    "attempts": int,
    "grounding_chunk_count": int,
    "grounding_query_count": int,
    "grounding_support_count": int,
    "max_output_tokens": int,
    "temperature": (int, float),
}
GEMINI_USAGE_SCHEMA: Mapping[str, Any] = {
    "cachedContentTokenCount": "number",
    "candidatesTokenCount": "number",
    "promptTokenCount": "number",
    "thoughtsTokenCount": "number",
    "toolUsePromptTokenCount": "number",
    "totalTokenCount": "number",
}
OPENAI_USAGE_SCHEMA: Mapping[str, Any] = {
    "completion_tokens": "number",
    "completion_tokens_details": {
        "accepted_prediction_tokens": "number",
        "audio_tokens": "number",
        "reasoning_tokens": "number",
        "rejected_prediction_tokens": "number",
    },
    "prompt_tokens": "number",
    "prompt_tokens_details": {
        "audio_tokens": "number",
        "cached_tokens": "number",
    },
    "total_tokens": "number",
}
FINISH_REASON_MAP = {
    "STOP": "stop",
    "stop": "stop",
    "MAX_TOKENS": "length",
    "length": "length",
    "SAFETY": "safety",
    "RECITATION": "recitation",
    "LANGUAGE": "language",
    "OTHER": "other",
    "content_filter": "content_filter",
    "tool_calls": "tool_calls",
    "function_call": "function_call",
}


class ProviderRequestError(RuntimeError):
    """A provider failure carrying only non-secret structured diagnostics."""

    def __init__(
        self,
        *,
        category: str,
        http_status: int | None,
        retryable: bool,
    ) -> None:
        super().__init__("provider request failed")
        self.category = category
        self.http_status = http_status
        self.retryable = retryable


def build_gemini_request(
    question: str,
    *,
    model: str = GEMINI_MODEL,
    max_output_tokens: int = GOOGLE_MAX_OUTPUT_TOKENS,
    grounding: bool = False,
) -> dict[str, Any]:
    """Build a Google generateContent body without challenger-answer context."""

    if model not in GOOGLE_GENERATION_MODELS:
        raise ValueError(f"model is not an approved Google challenger: {model}")
    thinking_level = "high" if model == GEMMA_MODEL else "low"

    body: dict[str, Any] = {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {
            "temperature": 1.0,
            "maxOutputTokens": max_output_tokens,
            "thinkingConfig": {"thinkingLevel": thinking_level},
        },
    }
    if grounding:
        body["tools"] = [{"google_search": {}}]
    return body


def build_openai_request(
    question: str, *, model: str, max_output_tokens: int = 768
) -> dict[str, Any]:
    """Build an OpenAI-compatible chat-completions body."""

    return {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        "temperature": 0.0,
        "max_tokens": max_output_tokens,
    }


def parse_gemini_response(
    response: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any], str | None]:
    """Extract the answer and audit fields from a Gemini response."""

    candidates = response.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("Gemini response contains no candidates")
    candidate = candidates[0]
    if not isinstance(candidate, Mapping):
        raise ValueError("Gemini response candidate is not an object")
    content = candidate.get("content")
    parts = content.get("parts") if isinstance(content, Mapping) else None
    if not isinstance(parts, list):
        raise ValueError("Gemini response contains no content parts")
    answer = "".join(
        str(part["text"])
        for part in parts
        if isinstance(part, Mapping) and isinstance(part.get("text"), str)
    ).strip()
    if not answer:
        raise ValueError("Gemini response contains no answer text")
    usage = response.get("usageMetadata")
    return (
        answer,
        dict(usage) if isinstance(usage, Mapping) else {},
        str(candidate["finishReason"]) if candidate.get("finishReason") else None,
    )


def parse_structured_answer(text: str) -> tuple[str, str, str]:
    """Separate the required status envelope from a clean reference answer."""

    lines = text.strip().splitlines()
    if len(lines) < 3 or lines[0] not in UNCERTAINTY_STATUSES:
        raise ValueError("structured answer has an invalid or missing status")
    reason_prefix = "UNCERTAINTY_REASON:"
    if not lines[1].startswith(reason_prefix):
        raise ValueError("structured answer has no uncertainty reason")
    reason = lines[1][len(reason_prefix) :].strip()
    answer = "\n".join(lines[2:]).strip()
    if not reason or not answer:
        raise ValueError("structured answer has an empty reason or answer")
    return answer, UNCERTAINTY_STATUSES[lines[0]], reason


def _grounding_evidence(grounding_metadata: object) -> dict[str, Any]:
    """Extract only structured Search provenance returned by Gemini."""

    metadata = grounding_metadata if isinstance(grounding_metadata, Mapping) else {}
    raw_queries = metadata.get("webSearchQueries")
    queries = (
        [value for value in raw_queries if isinstance(value, str)]
        if isinstance(raw_queries, list)
        else []
    )
    sources: list[dict[str, Any]] = []
    raw_chunks = metadata.get("groundingChunks")
    if isinstance(raw_chunks, list):
        for index, chunk in enumerate(raw_chunks):
            web = chunk.get("web") if isinstance(chunk, Mapping) else None
            if not isinstance(web, Mapping):
                continue
            uri, title = web.get("uri"), web.get("title")
            if isinstance(uri, str) or isinstance(title, str):
                sources.append(
                    {
                        "index": index,
                        "url": uri if isinstance(uri, str) else "",
                        "title": title if isinstance(title, str) else "",
                    }
                )
    supports: list[dict[str, Any]] = []
    raw_supports = metadata.get("groundingSupports")
    if isinstance(raw_supports, list):
        for support in raw_supports:
            if not isinstance(support, Mapping):
                continue
            segment = support.get("segment")
            indices = support.get("groundingChunkIndices")
            if not isinstance(segment, Mapping):
                continue
            supports.append(
                {
                    "answer_passage": str(segment.get("text") or ""),
                    "start_index": segment.get("startIndex")
                    if isinstance(segment.get("startIndex"), int)
                    else None,
                    "end_index": segment.get("endIndex")
                    if isinstance(segment.get("endIndex"), int)
                    else None,
                    "source_indices": [
                        value for value in indices if isinstance(value, int)
                    ]
                    if isinstance(indices, list)
                    else [],
                }
            )
    return {
        "search_invoked": bool(queries or sources or supports),
        "web_search_queries": queries,
        "sources": sources,
        "supports": supports,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def parse_openai_response(
    response: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any], str | None]:
    """Extract the answer and audit fields from a chat-completions response."""

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("OpenAI-compatible response contains no choices")
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, Mapping) else None
    answer = message.get("content") if isinstance(message, Mapping) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("OpenAI-compatible response contains no answer text")
    usage = response.get("usage")
    return (
        answer.strip(),
        dict(usage) if isinstance(usage, Mapping) else {},
        str(choice["finish_reason"]) if choice.get("finish_reason") else None,
    )


def _post_json(
    url: str, body: dict[str, Any], headers: dict[str, str]
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT_SECONDS) as response:
        payload = json.load(response)
    if not isinstance(payload, Mapping):
        raise ValueError("provider response must be a JSON object")
    return payload


def _request_with_retries(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    *,
    request_fn: RequestFunction,
    sleep_fn: SleepFunction,
) -> tuple[Mapping[str, Any], int]:
    for attempt in range(1, 5):
        try:
            return request_fn(url, body, headers), attempt
        except urllib.error.HTTPError as exc:
            # A 429 is already a request-rate signal. Immediate per-thread retries
            # bypass the outer rate limiter and amplify the overload; checkpoint it
            # once and let the resumable runner retry in a later controlled pass.
            if (
                exc.code == 429
                or exc.code not in RETRIABLE_STATUS_CODES
                or attempt == 4
            ):
                raise ProviderRequestError(
                    category="http",
                    http_status=exc.code,
                    retryable=exc.code in RETRIABLE_STATUS_CODES,
                ) from exc
        except (TimeoutError, urllib.error.URLError) as exc:
            if attempt == 4:
                raise ProviderRequestError(
                    category="transport",
                    http_status=None,
                    retryable=True,
                ) from exc
        sleep_fn(float(2 ** (attempt - 1)))
    raise AssertionError("unreachable")


def call_gemini(
    question: str,
    *,
    api_key: str,
    model: str = GEMINI_MODEL,
    max_output_tokens: int = GOOGLE_MAX_OUTPUT_TOKENS,
    grounding: bool = False,
    request_fn: RequestFunction = _post_json,
    sleep_fn: SleepFunction = time.sleep,
) -> tuple[str, Mapping[str, Any], str | None, Mapping[str, Any]]:
    """Call an approved Gemini-API model using injected I/O hooks."""

    if not api_key:
        raise ValueError("GEMINI_API_KEY is required")
    if model not in GOOGLE_GENERATION_MODELS:
        raise ValueError(f"model is not an approved Google challenger: {model}")
    endpoint = GEMINI_ENDPOINT_TEMPLATE.format(model=model)
    body = build_gemini_request(
        question,
        model=model,
        max_output_tokens=max_output_tokens,
        grounding=grounding,
    )
    response, attempts = _request_with_retries(
        endpoint,
        body,
        {"x-goog-api-key": api_key},
        request_fn=request_fn,
        sleep_fn=sleep_fn,
    )
    answer, usage, finish_reason = parse_gemini_response(response)
    metadata: dict[str, Any] = {
        "endpoint": endpoint,
        "temperature": 1.0,
        "max_output_tokens": max_output_tokens,
        "attempts": attempts,
        "grounding": grounding,
    }
    if grounding:
        candidate = response.get("candidates", [{}])[0]
        grounding_metadata = (
            candidate.get("groundingMetadata")
            if isinstance(candidate, Mapping)
            else None
        )
        if isinstance(grounding_metadata, Mapping):
            for source_field, target_field in (
                ("webSearchQueries", "grounding_query_count"),
                ("groundingChunks", "grounding_chunk_count"),
                ("groundingSupports", "grounding_support_count"),
            ):
                values = grounding_metadata.get(source_field)
                if isinstance(values, list):
                    metadata[target_field] = len(values)
        answer, status, uncertainty_reason = parse_structured_answer(answer)
        metadata["uncertainty_status"] = status
        metadata["uncertainty_reason"] = uncertainty_reason
        metadata["_private_evidence"] = _grounding_evidence(grounding_metadata)
    request_id = response.get("responseId")
    if isinstance(request_id, str):
        metadata["request_id_sha256"] = hashlib.sha256(
            request_id.encode("utf-8")
        ).hexdigest()
    return answer, usage, finish_reason, metadata


def validate_openai_compatible_url(base_url: str) -> str:
    """Return a local chat-completions URL, rejecting every non-loopback host."""

    parsed = urllib.parse.urlsplit(base_url)
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or parsed.path != "/v1/chat/completions"
    ):
        raise ValueError("base URL must end at /v1/chat/completions")
    hostname = parsed.hostname
    if hostname == "localhost":
        return base_url
    try:
        is_loopback = (
            hostname is not None and ipaddress.ip_address(hostname).is_loopback
        )
    except ValueError:
        is_loopback = False
    if not is_loopback:
        raise ValueError("OpenAI-compatible base URL must use a loopback host")
    return base_url


def call_openai_compatible(
    question: str,
    *,
    model: str,
    base_url: str,
    max_output_tokens: int = 768,
    request_fn: RequestFunction = _post_json,
    sleep_fn: SleepFunction = time.sleep,
) -> tuple[str, Mapping[str, Any], str | None, Mapping[str, Any]]:
    """Call a loopback-only OpenAI-compatible chat-completions endpoint."""

    endpoint = validate_openai_compatible_url(base_url)
    body = build_openai_request(
        question, model=model, max_output_tokens=max_output_tokens
    )
    response, attempts = _request_with_retries(
        endpoint, body, {}, request_fn=request_fn, sleep_fn=sleep_fn
    )
    answer, usage, finish_reason = parse_openai_response(response)
    metadata: dict[str, Any] = {
        "endpoint": endpoint,
        "temperature": 0.0,
        "max_output_tokens": max_output_tokens,
        "attempts": attempts,
    }
    request_id = response.get("id")
    if isinstance(request_id, str):
        metadata["request_id_sha256"] = hashlib.sha256(
            request_id.encode("utf-8")
        ).hexdigest()
    return answer, usage, finish_reason, metadata


def call_deepinfra(
    question: str,
    *,
    api_token: str,
    model: str = DEEPINFRA_QWEN_MODEL,
    max_output_tokens: int = DEEPINFRA_MAX_OUTPUT_TOKENS,
    request_fn: RequestFunction = _post_json,
    sleep_fn: SleepFunction = time.sleep,
) -> tuple[str, Mapping[str, Any], str | None, Mapping[str, Any]]:
    """Call the pinned Qwen challenger through DeepInfra's OpenAI endpoint."""

    if not api_token:
        raise ValueError("DEEPINFRA_API_TOKEN is required")
    if model not in DEEPINFRA_MODELS:
        raise ValueError(f"model is not an approved DeepInfra challenger: {model}")
    body = build_openai_request(
        question, model=model, max_output_tokens=max_output_tokens
    )
    response, attempts = _request_with_retries(
        DEEPINFRA_ENDPOINT,
        body,
        {"Authorization": f"Bearer {api_token}"},
        request_fn=request_fn,
        sleep_fn=sleep_fn,
    )
    answer, usage, finish_reason = parse_openai_response(response)
    metadata: dict[str, Any] = {
        "endpoint": DEEPINFRA_ENDPOINT,
        "temperature": 0.0,
        "max_output_tokens": max_output_tokens,
        "attempts": attempts,
    }
    request_id = response.get("id")
    if isinstance(request_id, str):
        metadata["request_id_sha256"] = hashlib.sha256(
            request_id.encode("utf-8")
        ).hexdigest()
    return answer, usage, finish_reason, metadata


def _safe_request_metadata(
    metadata: Mapping[str, Any], *, provider: str
) -> dict[str, Any]:
    """Keep only adapter-authored scalar audit fields; discard everything else."""

    safe: dict[str, Any] = {}
    for field, expected_type in SAFE_REQUEST_METADATA_FIELDS.items():
        value = metadata.get(field)
        if isinstance(value, expected_type) and not isinstance(value, bool):
            safe[field] = value
    endpoint = metadata.get("endpoint")
    if isinstance(endpoint, str):
        approved_google_endpoints = {
            GEMINI_ENDPOINT_TEMPLATE.format(model=model)
            for model in GOOGLE_GENERATION_MODELS
        }
        if (
            provider in {"gemini", "gemini-grounded"}
            and endpoint in approved_google_endpoints
        ):
            safe["endpoint"] = endpoint
        elif provider == "deepinfra" and endpoint == DEEPINFRA_ENDPOINT:
            safe["endpoint"] = endpoint
        elif provider == "openai-compatible":
            try:
                safe["endpoint"] = validate_openai_compatible_url(endpoint)
            except ValueError:
                pass
    request_id_sha256 = metadata.get("request_id_sha256")
    if (
        isinstance(request_id_sha256, str)
        and len(request_id_sha256) == 64
        and all(character in "0123456789abcdef" for character in request_id_sha256)
    ):
        safe["request_id_sha256"] = request_id_sha256
    if metadata.get("grounding") is True:
        safe["grounding"] = True
    return safe


def _safe_usage(usage: Mapping[str, Any], *, provider: str) -> dict[str, Any]:
    schema = (
        GEMINI_USAGE_SCHEMA
        if provider in {"gemini", "gemini-grounded"}
        else OPENAI_USAGE_SCHEMA
        if provider in {"openai-compatible", "deepinfra"}
        else {}
    )

    def filter_mapping(
        value: Mapping[str, Any], allowed: Mapping[str, Any]
    ) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for field, rule in allowed.items():
            item = value.get(field)
            if rule == "number":
                if isinstance(item, (int, float)) and not isinstance(item, bool):
                    result[field] = item
            elif rule == "bool":
                if isinstance(item, bool):
                    result[field] = item
            elif isinstance(rule, Mapping) and isinstance(item, Mapping):
                nested = filter_mapping(item, rule)
                if nested:
                    result[field] = nested
        return result

    return filter_mapping(usage, schema)


def _safe_finish_reason(finish_reason: str | None) -> str:
    if finish_reason is None:
        return "unknown"
    return FINISH_REASON_MAP.get(finish_reason, "unknown")


def _failure_payload(
    exc: Exception,
    *,
    provider: str,
    model: str,
    prompt_sha256: str,
) -> dict[str, Any]:
    """Describe a failure without retaining its potentially secret message."""

    status: int | None = None
    category = "provider"
    retryable = False
    if isinstance(exc, ProviderRequestError):
        status = exc.http_status
        category = exc.category
        retryable = exc.retryable
    elif isinstance(exc, urllib.error.HTTPError):
        status = exc.code
        category = "http"
        retryable = status in RETRIABLE_STATUS_CODES
    elif isinstance(exc, (TimeoutError, urllib.error.URLError)):
        category = "transport"
        retryable = True
    return {
        "provider": provider,
        "model": model,
        "prompt_sha256": prompt_sha256,
        "error_type": type(exc).__name__,
        "category": category,
        "http_status": status,
        "retryable": retryable,
    }


def _load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"cache_version": 1, "records": {}, "failures": {}}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("records"), dict)
        or not isinstance(payload.get("failures"), dict)
    ):
        raise ValueError("generation cache must contain records and failures objects")
    return payload


def _write_cache(
    path: Path, payload: Mapping[str, Any], *, private: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    if private:
        path.chmod(0o600)


class GenerationRunner:
    """Generate missing answers while atomically checkpointing successes/failures."""

    def __init__(
        self,
        *,
        cache_path: Path,
        provider: str,
        model: str,
        prompt_sha256: str,
        call: GenerationCall,
        workers: int,
        checkpoint_every: int = 10,
        evidence_cache_path: Path | None = None,
    ) -> None:
        if workers < 1:
            raise ValueError("workers must be at least one")
        if checkpoint_every < 1:
            raise ValueError("checkpoint_every must be at least one")
        self.cache_path = cache_path
        self.provider = provider
        self.model = model
        self.prompt_sha256 = prompt_sha256
        self.call = call
        self.workers = workers
        self.checkpoint_every = checkpoint_every
        self.evidence_cache_path = evidence_cache_path

    def _is_complete(self, record: object, *, has_evidence: bool = True) -> bool:
        return (
            isinstance(record, Mapping)
            and record.get("provider") == self.provider
            and record.get("model") == self.model
            and record.get("prompt_sha256") == self.prompt_sha256
            and isinstance(record.get("answer"), str)
            and record.get("finish_reason") != "length"
            and has_evidence
        )

    def run(self, questions: list[tuple[str, str]]) -> dict[str, Any]:
        cache = _load_cache(self.cache_path)
        cache.update(
            {
                "cache_version": 1,
                "provider": self.provider,
                "model": self.model,
                "prompt_sha256": self.prompt_sha256,
            }
        )
        records = cache["records"]
        failures = cache["failures"]
        evidence_cache = (
            _load_cache(self.evidence_cache_path)
            if self.evidence_cache_path is not None
            else None
        )
        evidence_records: dict[str, Any] = (
            evidence_cache["records"] if evidence_cache is not None else {}
        )
        if evidence_cache is not None:
            evidence_cache.update(
                {
                    "cache_version": 1,
                    "provider": self.provider,
                    "model": self.model,
                    "prompt_sha256": self.prompt_sha256,
                }
            )

        def has_current_evidence(question_id: str) -> bool:
            if evidence_cache is None:
                return True
            evidence = evidence_records.get(question_id)
            return (
                isinstance(evidence, Mapping)
                and evidence.get("provider") == self.provider
                and evidence.get("model") == self.model
                and evidence.get("prompt_sha256") == self.prompt_sha256
            )

        def write_all() -> None:
            _write_cache(self.cache_path, cache)
            if self.evidence_cache_path is not None and evidence_cache is not None:
                _write_cache(self.evidence_cache_path, evidence_cache, private=True)

        cache_changed = False
        for question_id, _ in questions:
            if question_id in records and not self._is_complete(
                records[question_id], has_evidence=has_current_evidence(question_id)
            ):
                del records[question_id]
                cache_changed = True
            failure = failures.get(question_id)
            if isinstance(failure, Mapping) and (
                failure.get("provider") != self.provider
                or failure.get("model") != self.model
                or failure.get("prompt_sha256") != self.prompt_sha256
            ):
                del failures[question_id]
                cache_changed = True
        if cache_changed:
            write_all()
        pending = [
            (question_id, question)
            for question_id, question in questions
            if not self._is_complete(
                records.get(question_id), has_evidence=has_current_evidence(question_id)
            )
        ]
        if not pending:
            return cache

        def invoke(question: str):
            started = time.monotonic()
            result = self.call(question)
            return result, time.monotonic() - started

        completed_since_checkpoint = 0

        def store_future(question_id: str, future: Any) -> None:
            nonlocal completed_since_checkpoint
            try:
                (answer, usage, finish_reason, metadata), elapsed = future.result()
                base_record = GenerationRecord(
                    question_id=question_id,
                    provider=self.provider,
                    model=self.model,
                    answer=answer,
                    elapsed_seconds=elapsed,
                    usage=_safe_usage(usage, provider=self.provider),
                    finish_reason=_safe_finish_reason(finish_reason),
                )
                records[question_id] = {
                    **asdict(base_record),
                    "prompt_sha256": self.prompt_sha256,
                    "request_metadata": _safe_request_metadata(
                        metadata, provider=self.provider
                    ),
                }
                status = metadata.get("uncertainty_status")
                reason = metadata.get("uncertainty_reason")
                if status in UNCERTAINTY_STATUSES.values() and isinstance(reason, str):
                    records[question_id]["uncertainty_status"] = status
                    records[question_id]["uncertainty_reason"] = reason
                if evidence_cache is not None:
                    private_evidence = metadata.get("_private_evidence")
                    if not isinstance(private_evidence, Mapping):
                        raise ValueError(
                            "grounded response contains no private evidence"
                        )
                    evidence_records[question_id] = {
                        "question_id": question_id,
                        "provider": self.provider,
                        "model": self.model,
                        "prompt_sha256": self.prompt_sha256,
                        "uncertainty_status": status,
                        "uncertainty_reason": reason,
                        "usage": _safe_usage(usage, provider=self.provider),
                        "finish_reason": _safe_finish_reason(finish_reason),
                        **dict(private_evidence),
                    }
                failures.pop(question_id, None)
            except Exception as exc:
                records.pop(question_id, None)
                evidence_records.pop(question_id, None)
                failures[question_id] = _failure_payload(
                    exc,
                    provider=self.provider,
                    model=self.model,
                    prompt_sha256=self.prompt_sha256,
                )
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= self.checkpoint_every:
                write_all()
                completed_since_checkpoint = 0

        executor = ThreadPoolExecutor(max_workers=self.workers)
        pending_iterator = iter(pending)
        futures: dict[Any, str] = {}

        def submit_next() -> bool:
            try:
                question_id, question = next(pending_iterator)
            except StopIteration:
                return False
            futures[executor.submit(invoke, question)] = question_id
            return True

        for _ in range(min(self.workers, len(pending))):
            submit_next()
        try:
            while futures:
                future = next(as_completed(tuple(futures)))
                question_id = futures.pop(future)
                store_future(question_id, future)
                submit_next()
        except BaseException:
            for future in futures:
                future.cancel()
            executor.shutdown(wait=True, cancel_futures=True)
            for future, question_id in list(futures.items()):
                if future.done() and not future.cancelled():
                    try:
                        store_future(question_id, future)
                    except BaseException:
                        pass
            write_all()
            raise
        else:
            executor.shutdown(wait=True)
        finally:
            write_all()
        return cache


def generate_missing(
    questions: list[tuple[str, str]],
    *,
    cache_path: Path,
    provider: str,
    model: str,
    call: GenerationCall,
    workers: int,
    checkpoint_every: int = 10,
    evidence_cache_path: Path | None = None,
) -> dict[str, Any]:
    runner = GenerationRunner(
        cache_path=cache_path,
        provider=provider,
        model=model,
        prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        call=call,
        workers=workers,
        checkpoint_every=checkpoint_every,
        evidence_cache_path=evidence_cache_path,
    )
    return runner.run(questions)


class RateLimiter:
    """Thread-safe fixed-spacing request limiter."""

    def __init__(self, requests_per_minute: int) -> None:
        if requests_per_minute < 1:
            raise ValueError("requests_per_minute must be at least one")
        self.interval = 60.0 / requests_per_minute
        self._next_request = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_request - now)
            self._next_request = max(now, self._next_request) + self.interval
        if delay:
            time.sleep(delay)
