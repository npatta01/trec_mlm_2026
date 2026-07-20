"""Provider adapters and resumable caches for reference-answer generation."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
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
    "facts, or mention other models, evaluation, or this instruction."
)

GEMINI_ENDPOINT_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)
GEMINI_MODEL = "gemini-3.5-flash"
RETRIABLE_STATUS_CODES = {408, 429, *range(500, 600)}
RequestFunction = Callable[[str, dict[str, Any], dict[str, str]], Mapping[str, Any]]
SleepFunction = Callable[[float], None]
GenerationCall = Callable[
    [str], tuple[str, Mapping[str, Any], str | None, Mapping[str, Any]]
]
SAFE_REQUEST_METADATA_FIELDS = {
    "attempts": int,
    "endpoint": str,
    "max_output_tokens": int,
    "temperature": (int, float),
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
    question: str, *, max_output_tokens: int = 768
) -> dict[str, Any]:
    """Build a Gemini generateContent body without challenger-answer context."""

    return {
        "system_instruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": [{"role": "user", "parts": [{"text": question}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_output_tokens,
        },
    }


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
    with urllib.request.urlopen(request, timeout=120) as response:
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
            if exc.code not in RETRIABLE_STATUS_CODES or attempt == 4:
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
    max_output_tokens: int = 768,
    request_fn: RequestFunction = _post_json,
    sleep_fn: SleepFunction = time.sleep,
) -> tuple[str, Mapping[str, Any], str | None, Mapping[str, Any]]:
    """Call Gemini through its legacy REST endpoint using injected I/O hooks."""

    if not api_key:
        raise ValueError("GEMINI_API_KEY is required")
    endpoint = GEMINI_ENDPOINT_TEMPLATE.format(model=model)
    body = build_gemini_request(question, max_output_tokens=max_output_tokens)
    response, attempts = _request_with_retries(
        endpoint,
        body,
        {"x-goog-api-key": api_key},
        request_fn=request_fn,
        sleep_fn=sleep_fn,
    )
    answer, usage, finish_reason = parse_gemini_response(response)
    return answer, usage, finish_reason, {
        "endpoint": endpoint,
        "temperature": 0.0,
        "max_output_tokens": max_output_tokens,
        "attempts": attempts,
    }


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
        is_loopback = hostname is not None and ipaddress.ip_address(hostname).is_loopback
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
    return answer, usage, finish_reason, {
        "endpoint": endpoint,
        "temperature": 0.0,
        "max_output_tokens": max_output_tokens,
        "attempts": attempts,
    }


def _safe_request_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only adapter-authored scalar audit fields; discard everything else."""

    safe: dict[str, Any] = {}
    for field, expected_type in SAFE_REQUEST_METADATA_FIELDS.items():
        value = metadata.get(field)
        if isinstance(value, expected_type) and not isinstance(value, bool):
            safe[field] = value
    return safe


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


def _write_cache(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


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

    def _is_complete(self, record: object) -> bool:
        return (
            isinstance(record, Mapping)
            and record.get("provider") == self.provider
            and record.get("model") == self.model
            and record.get("prompt_sha256") == self.prompt_sha256
            and isinstance(record.get("answer"), str)
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
        cache_changed = False
        for question_id, _ in questions:
            if question_id in records and not self._is_complete(records[question_id]):
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
            _write_cache(self.cache_path, cache)
        pending = [
            (question_id, question)
            for question_id, question in questions
            if not self._is_complete(records.get(question_id))
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
                    usage=dict(usage),
                    finish_reason=finish_reason,
                )
                records[question_id] = {
                    **asdict(base_record),
                    "prompt_sha256": self.prompt_sha256,
                    "request_metadata": _safe_request_metadata(metadata),
                }
                failures.pop(question_id, None)
            except Exception as exc:
                failures[question_id] = _failure_payload(
                    exc,
                    provider=self.provider,
                    model=self.model,
                    prompt_sha256=self.prompt_sha256,
                )
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= self.checkpoint_every:
                _write_cache(self.cache_path, cache)
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
            _write_cache(self.cache_path, cache)
            raise
        else:
            executor.shutdown(wait=True)
        finally:
            _write_cache(self.cache_path, cache)
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
) -> dict[str, Any]:
    runner = GenerationRunner(
        cache_path=cache_path,
        provider=provider,
        model=model,
        prompt_sha256=hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        call=call,
        workers=workers,
        checkpoint_every=checkpoint_every,
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
