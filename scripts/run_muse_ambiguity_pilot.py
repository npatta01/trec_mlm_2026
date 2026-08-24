#!/usr/bin/env python3
"""Run a small repeated pilot for generic ambiguity-aware Muse references."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

from scripts.run_muse_canary import load_questions
from trec_mlm.muse_reference_generation import (
    MUSE_MODEL,
    MuseRunError,
    is_complete_muse_record,
    load_muse_cache,
    muse_prompt_sha256,
    run_muse_two_call,
    save_muse_cache,
)


PILOT_PROMPT_VARIANT = "reference_generic_branches"
PILOT_PROMPT_VARIANTS = (
    "reference_generic_uncertainty",
    "reference_generic_branches",
    "reference_generic_branches_v2",
    "reference_generic_branches_v3",
)


@dataclass(frozen=True)
class PilotQuestion:
    category: str
    question_id: str
    question: str


PILOT_QUESTIONS: tuple[PilotQuestion, ...] = (
    PilotQuestion(
        "clear",
        "5ae6168f5542996de7b71b47",
        "Adrian the Barbarian (later Olaff the Madlander) was a comic strip in the comic The Beezer, and later the merged Beezer and Topper, first introduced in which year?",
    ),
    PilotQuestion(
        "clear",
        "5ac4e4595542996feb3fe968",
        "Which cola drink is the leader in India, Thums Up or Ricqlès?",
    ),
    PilotQuestion(
        "clear",
        "5adce8c15542992c1e3a249d",
        "Are Thyme and Cortaderia both aromatic plants?",
    ),
    PilotQuestion(
        "ambiguous",
        "5ab9e3d25542994dbf019889",
        "What type of vegetation does Lythrum and Erythrina have in common?",
    ),
    PilotQuestion(
        "ambiguous",
        "57302972b2c2fd140056899e",
        "What could be considered a trait of Rome's military campaigns?",
    ),
    PilotQuestion(
        "ambiguous",
        "57260b4389a1e219009ac188",
        "What is an example of something that serves a function other than protection?",
    ),
    PilotQuestion(
        "broad-but-single",
        "56e2419acd28a01900c67c2a",
        "What structures can be viewed as both works of art and symbols of certain cultures?",
    ),
    PilotQuestion(
        "broad-but-single",
        "56f71740711bf01900a44932",
        "Bilateral treaties are concluded between how many states or entities?",
    ),
    PilotQuestion(
        "broad-but-single",
        "5ae336ef5542991a06ce995d",
        "What profession do both Phil Alden Robinson and David Nixon share?",
    ),
)


def load_pilot_questions(
    path: Path, *, manifest: Iterable[PilotQuestion] = PILOT_QUESTIONS
) -> list[PilotQuestion]:
    """Load and validate the fixed pilot questions from the development TSV."""

    available = dict(load_questions(path))
    selected = list(manifest)
    seen: set[str] = set()
    for item in selected:
        if item.question_id in seen:
            raise ValueError(f"duplicate pilot question ID: {item.question_id}")
        seen.add(item.question_id)
        actual = available.get(item.question_id)
        if actual is None:
            raise ValueError(f"pilot question ID not found in {path}: {item.question_id}")
        if actual != item.question:
            raise ValueError(
                f"pilot question text changed for {item.question_id}: {actual!r}"
            )
    if not selected:
        raise ValueError("pilot manifest is empty")
    return selected


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


def _record_key(repeat: int, question_id: str) -> str:
    return f"{repeat}:{question_id}"


def run_pilot(
    questions: list[PilotQuestion],
    *,
    repeats: int,
    cache_path: Path,
    model: str,
    call: Callable[[str], dict[str, Any]],
    workers: int = 2,
    prompt_variant: str = PILOT_PROMPT_VARIANT,
) -> tuple[dict[str, Any], int]:
    """Run each pilot question the requested number of times with checkpoints."""

    if repeats < 1:
        raise ValueError("repeats must be at least one")
    if workers < 1:
        raise ValueError("workers must be at least one")
    if prompt_variant not in PILOT_PROMPT_VARIANTS:
        raise ValueError(f"unsupported pilot prompt variant: {prompt_variant}")
    question_ids = [item.question_id for item in questions]
    if len(set(question_ids)) != len(question_ids):
        raise ValueError("pilot question IDs must be unique")

    cache = load_muse_cache(cache_path)
    expected_identity = {
        "provider": "meta-muse",
        "model": model,
        "prompt_variant": prompt_variant,
        "prompt_sha256": muse_prompt_sha256(prompt_variant),
    }
    for field, expected_value in expected_identity.items():
        existing = cache.get(field)
        if existing is not None and existing != expected_value:
            raise ValueError(
                f"pilot cache has incompatible {field} {existing!r}; "
                "use a separate cache"
            )
    records = cache["records"]
    failures = cache["failures"]
    for key, record in records.items():
        if not isinstance(record, Mapping):
            raise ValueError("pilot cache contains a malformed record")
        for field, expected_value in expected_identity.items():
            if record.get(field) != expected_value:
                raise ValueError(
                    f"pilot cache record {key!r} has incompatible {field}; "
                    "use a separate cache"
                )
    expected: dict[str, PilotQuestion] = {
        _record_key(repeat, item.question_id): item
        for repeat in range(1, repeats + 1)
        for item in questions
    }
    for key, item in expected.items():
        record = records.get(key)
        if isinstance(record, Mapping) and record.get("question") != item.question:
            raise ValueError(
                f"pilot cache record {key!r} has different question text; "
                "use a separate cache"
            )
    pending: list[tuple[str, PilotQuestion, int]] = []
    skipped = 0
    for key, item in expected.items():
        repeat = int(key.split(":", maxsplit=1)[0])
        record = records.get(key)
        if (
            is_complete_muse_record(
                record, model=model, prompt_variant=prompt_variant
            )
            and record.get("question") == item.question
            and record.get("question_id") == item.question_id
            and record.get("repeat") == repeat
        ):
            skipped += 1
        else:
            pending.append((key, item, repeat))
    cache.update(
        {
            "schema_version": 1,
            "provider": "meta-muse",
            "model": model,
            "prompt_variant": prompt_variant,
            "prompt_sha256": expected_identity["prompt_sha256"],
            "repeats": repeats,
            "questions": [item.__dict__ for item in questions],
        }
    )
    save_muse_cache(cache_path, cache)

    def store(key: str, item: PilotQuestion, repeat: int, future: Any) -> None:
        try:
            record = future.result()
            record.update(
                {
                    "question_id": item.question_id,
                    "repeat": repeat,
                    "category": item.category,
                }
            )
            records[key] = record
            failures.pop(key, None)
            outcome = {
                "event": "pilot_item_finished",
                "key": key,
                "category": item.category,
                "final_stage": record.get("final_stage"),
                "searches": len(record.get("search_queries", [])),
            }
        except Exception as exc:
            records.pop(key, None)
            failure = _safe_failure(exc)
            failures[key] = {
                "question_id": item.question_id,
                "repeat": repeat,
                "category": item.category,
                "question": item.question,
                "failure": failure,
            }
            outcome = {
                "event": "pilot_item_failed",
                "key": key,
                "category": item.category,
                "failure": _public_failure_summary(failure),
            }
        save_muse_cache(cache_path, cache)
        print(json.dumps(outcome, sort_keys=True), flush=True)

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(call, item.question): (key, item, repeat)
                for key, item, repeat in pending
            }
            for future in as_completed(futures):
                key, item, repeat = futures[future]
                store(key, item, repeat, future)
    save_muse_cache(cache_path, cache)
    return cache, skipped


def _answer_signals(answer: str) -> dict[str, Any]:
    normalized = answer.lower()
    status = "other"
    interpretation_decision = re.search(
        r"interpretation decision\s*:\s*(single|multiple)", normalized
    )
    if interpretation_decision:
        status = interpretation_decision.group(1).upper()
    elif re.search(r"reference status\s*:\s*uncertain", normalized):
        status = "UNCERTAIN"
    elif re.search(r"reference status\s*:\s*answered", normalized):
        status = "ANSWERED"
    elif re.search(r"status\s*:\s*ambiguous", normalized):
        status = "AMBIGUOUS"
    elif re.search(
        r"status\s*:\s*(?:answered|clear|unambiguous|canonical|direct)", normalized
    ):
        status = "SINGLE"
    branch_count = len(
        re.findall(
            r"(?im)^\s*(?:interpretation|reading)\s*(?:\d+|[a-z])\s*"
            r"(?::|[.)-]|–|—)",
            answer,
        )
    )
    return {
        "status_signal": status,
        "branch_heading_count": branch_count,
        "acceptance_set_signal": "acceptance set" in normalized,
        "grading_guidance_signal": "grading guidance" in normalized,
    }


def build_pilot_report(
    cache: dict[str, Any],
    *,
    questions: list[PilotQuestion],
    repeats: int,
    cache_path: Path,
    report_path: Path,
    skipped: int,
) -> str:
    question_by_id = {item.question_id: item for item in questions}
    active_keys = {
        _record_key(repeat, item.question_id)
        for repeat in range(1, repeats + 1)
        for item in questions
    }
    records = sorted(
        (
            record
            for key, record in cache.get("records", {}).items()
            if key in active_keys
        ),
        key=lambda record: (
            int(record.get("repeat", 0)),
            str(record.get("question_id", "")),
        ),
    )
    failures = {
        key: failure
        for key, failure in cache.get("failures", {}).items()
        if key in active_keys
    }
    stage2 = sum(record.get("final_stage") == 2 for record in records)
    searches = sum(len(record.get("search_queries", [])) for record in records)
    citations = sum(len(record.get("citations", [])) for record in records)
    category_counts = Counter(item.category for item in questions)
    lines = [
        "# Muse ambiguity-handling pilot",
        "",
        f"- Model: `{cache.get('model', MUSE_MODEL)}`",
        f"- Prompt variant: `{cache.get('prompt_variant', PILOT_PROMPT_VARIANT)}`",
        "- Endpoint: Meta native Responses API with `web_search`",
        "- Policy tested: two calls per item; no application Search or output-length cap",
        f"- Questions: {len(questions)} ({', '.join(f'{count} {category}' for category, count in sorted(category_counts.items()))})",
        f"- Repeats per question: {repeats}",
        f"- Expected records: {len(questions) * repeats}",
        f"- Completed records: {len(records)}",
        f"- Hard failures: {len(failures)}",
        f"- Stage-2 finals: {stage2}",
        f"- Cache-only rerun items skipped without API calls: {skipped}",
        f"- Search queries recorded: {searches}",
        f"- URL citations recorded: {citations}",
        "",
        "The signals below are mechanical indicators for review; they are not a semantic quality judgment.",
        "",
        "## Output signals",
        "",
        "| Repeat | ID | Category | Status signal | Branch headings | Acceptance set | Citations |",
        "| ---: | --- | --- | --- | ---: | --- | ---: |",
    ]
    for record in records:
        item = question_by_id[record["question_id"]]
        signals = _answer_signals(str(record.get("answer", "")))
        lines.append(
            f"| {record.get('repeat')} | `{record['question_id']}` | {item.category} | "
            f"{signals['status_signal']} | {signals['branch_heading_count']} | "
            f"{'yes' if signals['acceptance_set_signal'] else 'no'} | "
            f"{len(record.get('citations', []))} |"
        )
    lines.extend(["", "## Answers for manual review", ""])
    for record in records:
        item = question_by_id[record["question_id"]]
        signals = _answer_signals(str(record.get("answer", "")))
        lines.extend(
            [
                f"### Repeat {record.get('repeat')} — {item.category} — `{item.question_id}`",
                "",
                f"Question: {item.question}",
                "",
                f"Signals: `{json.dumps(signals, sort_keys=True)}`",
                "",
                str(record.get("answer", "")).rstrip(),
                "",
            ]
        )
    lines.extend(
        [
            "## Pilot boundary",
            "",
            f"This is a {len(questions)}-question, {repeats}-repeat pilot for prompt behavior. No full-dataset run was started.",
            "",
            f"Private cache: `{cache_path}`",
        ]
    )
    if failures:
        lines.extend(["", "## Failures", ""])
        for key, item in sorted(failures.items()):
            summary = _public_failure_summary(item.get("failure", {}))
            lines.append(f"- `{key}`: {json.dumps(summary, sort_keys=True)}")
    report = "\n".join(lines) + "\n"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=Path("data/dev/llm_dev_data.tsv"))
    parser.add_argument(
        "--cache", type=Path, default=Path("artifacts/muse_ambiguity_pilot_v3_2026-08-22.json")
    )
    parser.add_argument(
        "--report", type=Path, default=Path("reports/muse_ambiguity_pilot_v3_2026-08-22.md")
    )
    parser.add_argument("--model", default=MUSE_MODEL)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--prompt-variant",
        choices=PILOT_PROMPT_VARIANTS,
        default="reference_generic_branches_v3",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    api_key = os.environ.get("MODEL_API_KEY")
    if not api_key:
        raise SystemExit("MODEL_API_KEY is required")
    questions = load_pilot_questions(args.queries)

    def call(question: str) -> dict[str, Any]:
        return run_muse_two_call(
            question,
            api_key=api_key,
            model=args.model,
            prompt_variant=args.prompt_variant,
        )

    cache, skipped = run_pilot(
        questions,
        repeats=args.repeats,
        cache_path=args.cache,
        model=args.model,
        call=call,
        workers=args.workers,
        prompt_variant=args.prompt_variant,
    )
    build_pilot_report(
        cache,
        questions=questions,
        repeats=args.repeats,
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
