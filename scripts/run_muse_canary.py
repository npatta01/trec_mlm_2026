#!/usr/bin/env python3
"""Run a bounded, resumable two-call Muse grounding canary."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
from statistics import median
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from trec_mlm.muse_reference_generation import (
    MUSE_MODEL,
    MuseRunError,
    is_complete_muse_record,
    load_muse_cache,
    muse_prompt_sha256,
    run_muse_two_call,
    save_muse_cache,
    select_canary_questions,
)


def load_questions(path: Path) -> list[tuple[str, str]]:
    questions: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t", maxsplit=1)
            if len(fields) != 2 or not fields[0].strip() or not fields[1].strip():
                raise ValueError(f"{path}:{line_number}: expected id<TAB>question")
            question_id, question = fields[0].strip(), fields[1].strip()
            if question_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate question ID")
            seen.add(question_id)
            questions.append((question_id, question))
    if not questions:
        raise ValueError(f"{path}: question file is empty")
    return questions


def _safe_failure(exc: Exception) -> dict[str, Any]:
    if isinstance(exc, MuseRunError):
        return dict(exc.payload)
    return {
        "category": "runner",
        "error_type": type(exc).__name__,
        "http_status": None,
        "retryable": False,
    }


def _public_failure_summary(failure: Mapping[str, Any]) -> dict[str, Any]:
    return {
        field: failure.get(field)
        for field in ("category", "error_type", "http_status", "retryable")
    }


def run_canary(
    selected_questions: list[tuple[str, str]],
    *,
    cache_path: Path,
    model: str,
    prompt_variant: str = "answer",
    call: Callable[[str], dict[str, Any]],
    workers: int = 2,
) -> tuple[dict[str, Any], int]:
    """Run pending selected questions and checkpoint each result."""

    if workers < 1:
        raise ValueError("workers must be at least one")
    cache = load_muse_cache(cache_path)
    expected_identity = {
        "provider": "meta-muse",
        "model": model,
        "prompt_variant": prompt_variant,
        "prompt_sha256": muse_prompt_sha256(prompt_variant),
    }
    for field, expected in expected_identity.items():
        existing = cache.get(field)
        if existing is not None and existing != expected:
            raise ValueError(
                f"canary cache has incompatible {field} {existing!r}; "
                "use a separate cache"
            )
    records = cache["records"]
    failures = cache["failures"]
    selected_by_id = dict(selected_questions)
    for question_id, record in records.items():
        if not isinstance(record, Mapping):
            raise ValueError("canary cache contains a malformed record")
        for field, expected in expected_identity.items():
            if record.get(field) != expected:
                raise ValueError(
                    f"canary cache record {question_id!r} has incompatible {field}; "
                    "use a separate cache"
                )
        if (
            question_id in selected_by_id
            and record.get("question") != selected_by_id[question_id]
        ):
            raise ValueError(
                f"canary cache record {question_id!r} has different question text; "
                "use a separate cache"
            )
    pending: list[tuple[str, str]] = []
    skipped = 0
    for question_id, question in selected_questions:
        record = records.get(question_id)
        if (
            is_complete_muse_record(
                record, model=model, prompt_variant=prompt_variant
            )
            and record.get("question") == question
        ):
            skipped += 1
        else:
            pending.append((question_id, question))
    cache.update(
        {
            "schema_version": 1,
            "provider": "meta-muse",
            "model": model,
            "prompt_variant": prompt_variant,
            "prompt_sha256": expected_identity["prompt_sha256"],
            "selected_question_ids": [question_id for question_id, _ in selected_questions],
            "selected_questions": [
                {"question_id": question_id, "question": question}
                for question_id, question in selected_questions
            ],
        }
    )
    save_muse_cache(cache_path, cache)

    def store(question_id: str, question: str, future: Any) -> None:
        try:
            record = future.result()
            record["question_id"] = question_id
            records[question_id] = record
            failures.pop(question_id, None)
            outcome = {
                "question_id": question_id,
                "final_stage": record.get("final_stage"),
                "searches": len(record.get("search_queries", [])),
            }
        except Exception as exc:
            records.pop(question_id, None)
            failure = _safe_failure(exc)
            failures[question_id] = {
                "question_id": question_id,
                "question": question,
                "failure": failure,
            }
            outcome = {
                "question_id": question_id,
                "failure": _public_failure_summary(failure),
            }
        save_muse_cache(cache_path, cache)
        print(json.dumps({"event": "question_finished", **outcome}, sort_keys=True), flush=True)

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(call, question): (question_id, question)
                for question_id, question in pending
            }
            for future in as_completed(futures):
                question_id, question = futures[future]
                store(question_id, question, future)
    save_muse_cache(cache_path, cache)
    return cache, skipped


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = min(len(values) - 1, int(round((len(values) - 1) * percentile)))
    return values[index]


def build_report(
    cache: dict[str, Any],
    *,
    selected_questions: list[tuple[str, str]],
    cache_path: Path,
    report_path: Path,
    skipped: int,
) -> str:
    records = [
        cache["records"][question_id]
        for question_id, _ in selected_questions
        if question_id in cache["records"]
    ]
    failures = [
        cache["failures"][question_id]
        for question_id, _ in selected_questions
        if question_id in cache["failures"]
    ]
    stage2 = [record for record in records if record.get("final_stage") == 2]
    fallback = [record for record in records if record.get("final_stage") == 1]
    search_calls = [action for record in records for action in record.get("search_actions", [])]
    search_types = Counter(str(action.get("type")) for action in search_calls)
    host_counts = Counter(
        urlsplit(url).netloc.lower()
        for record in records
        for url in record.get("opened_urls", [])
        if urlsplit(url).netloc
    )
    latencies = [float(record["elapsed_seconds"]) for record in records]
    answer_lengths = [float(record["answer_characters"]) for record in records]
    usage_totals = Counter()
    for record in records:
        usage_totals.update(record.get("usage", {}).get("combined", {}))
    incomplete_followups = sum(
        1
        for record in records
        if record.get("call_2", {}).get("error", {}).get("category") == "incomplete"
    )
    lines = [
        "# Muse Spark 1.2 Contributor — 50-question grounding canary",
        "",
        f"- Model: `{cache.get('model', MUSE_MODEL)}`",
        f"- Prompt variant: `{cache.get('prompt_variant', 'answer')}`",
        "- Endpoint: Meta native Responses API with `web_search`",
        "- Policy: two calls per question; no application output-token or Search-use cap",
        f"- Selected questions: {len(selected_questions)}",
        f"- Completed records: {len(records)}",
        f"- Hard failures: {len(failures)}",
        f"- Stage-2 finals: {len(stage2)}",
        f"- Stage-1 fallbacks: {len(fallback)}",
        f"- Resumed without API calls: {skipped}",
        "",
        "## Grounding and provenance",
        "",
        f"- Search actions: {len(search_calls)} ({search_types.get('search', 0)} searches, {search_types.get('open_page', 0)} page opens)",
        f"- Questions with at least one Search query: {sum(bool(record.get('search_queries')) for record in records)}",
        f"- URL citations in final answers: {sum(len(record.get('citations', [])) for record in records)}",
        f"- Follow-up responses reported incomplete: {incomplete_followups}",
        "- Opened-source host counts are a quality signal, not an adjudicated source-quality score:",
    ]
    for host, count in host_counts.most_common(12):
        lines.append(f"  - `{host}`: {count}")
    lines.extend(
        [
            "",
            "## Usage and length",
            "",
            f"- Input tokens: {int(usage_totals.get('input_tokens', 0))}",
            f"- Output tokens: {int(usage_totals.get('output_tokens', 0))}",
            f"- Total tokens: {int(usage_totals.get('total_tokens', 0))}",
            f"- Answer characters (min / median / p95 / max): {int(min(answer_lengths)) if answer_lengths else 'n/a'} / {int(median(answer_lengths)) if answer_lengths else 'n/a'} / {int(_percentile(answer_lengths, 0.95)) if answer_lengths else 'n/a'} / {int(max(answer_lengths)) if answer_lengths else 'n/a'}",
            f"- Latency seconds (median / p95): {median(latencies):.1f} / {_percentile(latencies, 0.95):.1f}" if latencies else "- Latency: n/a",
            "- The API responses provide usage counts, but this report does not infer a dollar cost without the account's applicable Meta billing rate.",
            "",
            "## Failures",
            "",
        ]
    )
    if failures:
        for item in failures:
            summary = _public_failure_summary(item["failure"])
            lines.append(
                f"- `{item['question_id']}`: {json.dumps(summary, sort_keys=True)}"
            )
    else:
        lines.append("- None.")
    lines.extend(
        [
            "",
            "## Canary boundary",
            "",
            "This run is bounded to the selected 50 development questions. The private cache retains both raw Responses payloads, request payloads, response IDs, Search actions, URL annotations, usage, and retry/fallback diagnostics for audit. No full-dataset run was started.",
            "",
            f"Private cache: `{cache_path}`",
            f"Selected IDs: {', '.join(question_id for question_id, _ in selected_questions)}",
        ]
    )
    report = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=Path("data/dev/llm_dev_data.tsv"))
    parser.add_argument("--cache", type=Path, default=Path("artifacts/muse_canary_50.json"))
    parser.add_argument("--report", type=Path, default=Path("reports/muse_canary_50_2026-08-22.md"))
    parser.add_argument("--model", default=MUSE_MODEL)
    parser.add_argument(
        "--prompt-variant",
        choices=(
            "answer",
            "reference",
            "reference_generic",
            "reference_generic_citations",
        ),
        default="answer",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--workers", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        raise SystemExit("MODEL_API_KEY is required")
    questions = load_questions(args.queries)
    selected = select_canary_questions(questions, size=args.limit, seed=args.seed)
    def call(question: str) -> dict[str, Any]:
        return run_muse_two_call(
            question,
            api_key=api_key,
            model=args.model,
            prompt_variant=args.prompt_variant,
        )
    cache, skipped = run_canary(
        selected,
        cache_path=args.cache,
        model=args.model,
        prompt_variant=args.prompt_variant,
        call=call,
        workers=args.workers,
    )
    build_report(
        cache,
        selected_questions=selected,
        cache_path=args.cache,
        report_path=args.report,
        skipped=skipped,
    )
    print(
        json.dumps(
            {
                "cache": str(args.cache),
                "report": str(args.report),
                "records": len(cache["records"]),
                "failures": len(cache["failures"]),
                "resumed_without_api_calls": skipped,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
