#!/usr/bin/env python3
"""Generate and repair Perplexity Sonar answers for an LLM-discovery file.

The API key is read from ``PPLX_API_KEY``. The cache is resumable and keeps
answer text, citation URLs, search results, usage metadata, and retry history.
Use ``generate`` for the initial run and ``retry`` to revisit records without
citations or with citation markers outside the returned source list.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


ENDPOINT = "https://api.perplexity.ai/v1/sonar"
MODEL = "sonar"
MARKER_RE = re.compile(r"\[(\d+)\]")
SOURCE_FILTER = [
    "-reddit.com",
    "-youtube.com",
    "-facebook.com",
    "-quora.com",
    "-pinterest.com",
    "-medium.com",
]

SYSTEM_PROMPT = (
    "Write a clear, source-grounded reference answer in English, using only the "
    "retrieved web evidence; do not fill gaps from memory. Answer fully and "
    "directly. Length is flexible: be concise without omitting important context, "
    "ambiguity, comparisons, or caveats. Cite factual claims inline with the "
    "citation markers supplied by the API. Prefer primary, official, academic, "
    "medical, legal, institutional, or reputable news sources. Wikipedia may be "
    "used as a fallback for basic background, but do not rely on social media, "
    "user-generated forums, or generic blogs. Do not invent citations, quotes, "
    "numbers, or exact dates. If evidence is insufficient, stale, conflicting, "
    "or ambiguous, say 'Insufficient or conflicting evidence' instead of guessing."
)

RETRY_PROMPT = (
    "Write a concise, source-grounded reference answer using web search evidence. "
    "Prefer primary, official, academic, medical, legal, institutional, or reputable "
    "news sources. Cite factual claims inline. Every citation marker must be a valid "
    "numeric marker corresponding to a source returned for this answer; never emit a "
    "marker whose number is not present in the returned citation list. Do not invent "
    "citations, quotes, numbers, or dates. If evidence is insufficient, conflicting, "
    "or too obscure, say 'Insufficient or conflicting evidence' instead of guessing."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def strip_citations(text: str | None) -> str:
    """Return answer text without numeric inline citation markers."""

    if not text:
        return ""
    text = re.sub(r"[ \t]*\[\d+\]", "", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text.strip()


def citation_state(response: dict[str, Any]) -> tuple[bool, bool]:
    """Return ``(has_citations, markers_are_in_range)`` for one response."""

    if not isinstance(response, dict):
        return False, True
    answer = response.get("answer") or ""
    citations = response.get("citations") or []
    markers = [int(value) for value in MARKER_RE.findall(answer)]
    return bool(citations), not markers or max(markers) <= len(citations)


def compact_response(body: dict[str, Any], *, retry: bool = False) -> dict[str, Any]:
    """Keep the response fields needed for audit and later evaluation."""

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    answer = message.get("content") or ""
    compact = {
        "model": body.get("model"),
        "request_id": body.get("id"),
        "created": body.get("created"),
        "answer": answer,
        "answer_no_citations": strip_citations(answer),
        "citations": body.get("citations") or [],
        "search_results": body.get("search_results") or [],
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage") or {},
    }
    if retry:
        compact["retried_at"] = utc_now()
        compact["retry_search_context_size"] = "medium"
    return compact


def load_questions(path: Path) -> list[tuple[str, str]]:
    """Load ``{record_id: {"query": ...}}`` questions in file order."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("questions file must contain a JSON object")
    questions = []
    for question_id, item in data.items():
        if not isinstance(item, dict) or not isinstance(item.get("query"), str):
            raise ValueError(f"question {question_id!r} has no string 'query'")
        questions.append((str(question_id), item["query"]))
    return questions


class RateLimiter:
    """Simple process-local sliding-window request limiter."""

    def __init__(self, maximum: int, window_seconds: float = 60.0):
        self.maximum = maximum
        self.window_seconds = window_seconds
        self.timestamps: deque[float] = deque()
        self.lock = threading.Lock()

    def acquire(self) -> None:
        while True:
            with self.lock:
                current = time.monotonic()
                while self.timestamps and current - self.timestamps[0] >= self.window_seconds:
                    self.timestamps.popleft()
                if len(self.timestamps) < self.maximum:
                    self.timestamps.append(current)
                    return
                delay = self.window_seconds - (current - self.timestamps[0]) + 0.05
            time.sleep(min(delay, 30.0))


def request_sonar(
    question: str,
    api_key: str,
    limiter: RateLimiter,
    *,
    prompt: str,
    search_context_size: str,
    source_filter: list[str] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": question},
        ],
        "search_context_size": search_context_size,
    }
    if source_filter:
        payload["search_domain_filter"] = source_filter

    last_error: Exception | None = None
    for attempt in range(4):
        limiter.acquire()
        request = urllib.request.Request(
            ENDPOINT,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {detail[:600]}")
            if exc.code not in {408, 429, 500, 502, 503, 504}:
                raise last_error
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
        time.sleep(min(2**attempt, 20))
    raise RuntimeError(f"request failed after retries: {last_error}")


def save_cache(path: Path, cache: dict[str, Any]) -> None:
    """Write a cache atomically, preserving a usable previous checkpoint."""

    path.parent.mkdir(parents=True, exist_ok=True)
    cache["updated_at"] = utc_now()
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_cache(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"cache_version": 1, "provider": "perplexity", "records": {}, "failures": {}}
    cache = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cache, dict) or not isinstance(cache.get("records"), dict):
        raise ValueError("cache must be an object containing a records object")
    return cache


def run_parallel(
    items: list[tuple[str, str]],
    call: Callable[[str], dict[str, Any]],
    *,
    workers: int,
    progress: Callable[[dict[str, dict[str, Any]], dict[str, str]], None] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    results: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call, question): question_id for question_id, question in items}
        for future in as_completed(futures):
            question_id = futures[future]
            try:
                results[question_id] = future.result()
            except Exception as exc:
                failures[question_id] = repr(exc)
            completed = len(results) + len(failures)
            if progress is not None and (completed % 25 == 0 or completed == len(items)):
                progress(results, failures)
            if completed % 10 == 0 or completed == len(items):
                print(
                    json.dumps(
                        {"completed": completed, "total": len(items), "failures": len(failures)}
                    ),
                    flush=True,
                )
    return results, failures


def promote_retry(record: dict[str, Any], response: dict[str, Any]) -> bool:
    """Promote only a response with sources and valid inline markers."""

    has_citations, valid_markers = citation_state(response)
    if has_citations and valid_markers:
        record["response"] = response
        return True
    return False


def generate(args: argparse.Namespace) -> None:
    api_key = os.environ.get("PPLX_API_KEY")
    if not api_key:
        raise SystemExit("PPLX_API_KEY is required")
    if args.workers < 1 or args.requests_per_minute < 1:
        raise SystemExit("workers and requests-per-minute must be positive")

    questions = load_questions(args.questions_path)
    cache = load_cache(args.cache_path)
    records = {} if args.refresh else cache.get("records", {})
    failures = {} if args.refresh else cache.get("failures", {})
    pending = [(question_id, question) for question_id, question in questions if question_id not in records]
    cache.update(
        {
            "cache_version": 1,
            "provider": "perplexity",
            "endpoint": ENDPOINT,
            "model": MODEL,
            "search_context_size": "low",
            "source_filter": SOURCE_FILTER,
            "prompt_version": "quality_first_v2_source_filter",
            "total_questions": len(questions),
            "records": records,
            "failures": failures,
        }
    )
    save_cache(args.cache_path, cache)
    print(
        json.dumps(
            {
                "cache_path": str(args.cache_path),
                "total": len(questions),
                "already_cached": len(records),
                "pending": len(pending),
            }
        ),
        flush=True,
    )
    if not pending:
        return

    limiter = RateLimiter(args.requests_per_minute)
    question_by_id = dict(pending)

    def call(question: str) -> dict[str, Any]:
        return request_sonar(
            question,
            api_key,
            limiter,
            prompt=SYSTEM_PROMPT,
            search_context_size="low",
            source_filter=SOURCE_FILTER,
        )

    def checkpoint(
        results: dict[str, dict[str, Any]],
        current_failures: dict[str, str],
    ) -> None:
        for question_id, body in results.items():
            existing = records.get(question_id) or {}
            records[question_id] = {
                "question": question_by_id[question_id],
                "cached_at": existing.get("cached_at") or utc_now(),
                "response": compact_response(body),
            }
        failures.update(current_failures)
        cache["records"] = records
        cache["failures"] = failures
        save_cache(args.cache_path, cache)

    raw_results, new_failures = run_parallel(
        pending,
        call,
        workers=args.workers,
        progress=checkpoint,
    )
    for question_id, body in raw_results.items():
        question = next(question for current_id, question in pending if current_id == question_id)
        records[question_id] = {
            "question": question,
            "cached_at": utc_now(),
            "response": compact_response(body),
        }
        failures.pop(question_id, None)
    failures.update(new_failures)
    cache["records"] = records
    cache["failures"] = failures
    save_cache(args.cache_path, cache)
    print(
        json.dumps(
            {"finished": True, "records": len(records), "failures": len(failures), "cache_path": str(args.cache_path)}
        ),
        flush=True,
    )


def retry(args: argparse.Namespace) -> None:
    api_key = os.environ.get("PPLX_API_KEY")
    if not api_key:
        raise SystemExit("PPLX_API_KEY is required")
    if args.workers < 1 or args.requests_per_minute < 1:
        raise SystemExit("workers and requests-per-minute must be positive")

    cache = load_cache(args.cache_path)
    targets = []
    for question_id, record in cache["records"].items():
        if citation_state(record.get("response") or {}) != (True, True):
            targets.append((question_id, record["question"]))
    print(json.dumps({"targets": len(targets)}), flush=True)
    if not targets:
        return

    limiter = RateLimiter(args.requests_per_minute)

    def call(question: str) -> dict[str, Any]:
        return request_sonar(
            question,
            api_key,
            limiter,
            prompt=RETRY_PROMPT,
            search_context_size="medium",
            source_filter=None,
        )

    raw_results, failures = run_parallel(targets, call, workers=args.workers)
    accepted = 0
    unresolved = list(failures)
    for question_id, body in raw_results.items():
        record = cache["records"][question_id]
        response = compact_response(body, retry=True)
        record.setdefault("retry_history", []).append(
            {
                "original_response": record.get("response"),
                "retried_response": response,
                "retried_at": utc_now(),
            }
        )
        if promote_retry(record, response):
            accepted += 1
        else:
            unresolved.append(question_id)
    cache["retry_summary"] = {
        "retried": len(targets),
        "accepted": accepted,
        "unresolved": sorted(unresolved),
        "failures": failures,
        "completed_at": utc_now(),
    }
    save_cache(args.cache_path, cache)
    print(json.dumps(cache["retry_summary"]), flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate_parser = subparsers.add_parser("generate", help="generate missing Sonar answers")
    generate_parser.add_argument("--questions-path", type=Path, required=True)
    generate_parser.add_argument("--cache-path", type=Path, required=True)
    generate_parser.add_argument("--workers", type=int, default=8)
    generate_parser.add_argument("--requests-per-minute", type=int, default=45)
    generate_parser.add_argument("--refresh", action="store_true")
    generate_parser.set_defaults(handler=generate)

    retry_parser = subparsers.add_parser("retry", help="retry uncited or malformed responses")
    retry_parser.add_argument("--cache-path", type=Path, required=True)
    retry_parser.add_argument("--workers", type=int, default=8)
    retry_parser.add_argument("--requests-per-minute", type=int, default=45)
    retry_parser.set_defaults(handler=retry)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
