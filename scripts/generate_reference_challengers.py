#!/usr/bin/env python3
"""Generate resumable Gemini or local-Qwen reference challengers."""

from __future__ import annotations

import argparse
from functools import partial
import json
import os
from pathlib import Path
from typing import Sequence

from trec_mlm.reference_generation import (
    DEEPINFRA_MAX_OUTPUT_TOKENS,
    DEEPINFRA_MODELS,
    DEEPINFRA_QWEN_MODEL,
    GEMINI_MODEL,
    GOOGLE_GENERATION_MODELS,
    GOOGLE_MAX_OUTPUT_TOKENS,
    RateLimiter,
    call_gemini,
    call_deepinfra,
    call_openai_compatible,
    generate_missing,
    validate_openai_compatible_url,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="provider", required=True)

    gemini = subparsers.add_parser("gemini")
    _common_arguments(gemini, default_model=GEMINI_MODEL, default_workers=8)
    gemini.add_argument("--requests-per-minute", type=int, default=120)
    gemini.add_argument(
        "--max-output-tokens", type=int, default=GOOGLE_MAX_OUTPUT_TOKENS
    )
    gemini.add_argument(
        "--grounding",
        action="store_true",
        help="Enable Google Search grounding and write a distinct grounded cache.",
    )
    gemini.add_argument(
        "--evidence-cache",
        type=Path,
        help="Private mode-0600 Search provenance cache (required with --grounding).",
    )

    openai = subparsers.add_parser("openai-compatible")
    _common_arguments(openai, default_model="Qwen/Qwen3.6-35B-A3B", default_workers=4)
    openai.add_argument("--base-url", required=True)

    deepinfra = subparsers.add_parser("deepinfra")
    _common_arguments(deepinfra, default_model=DEEPINFRA_QWEN_MODEL, default_workers=8)
    deepinfra.add_argument(
        "--max-output-tokens", type=int, default=DEEPINFRA_MAX_OUTPUT_TOKENS
    )
    deepinfra.add_argument("--requests-per-minute", type=int, default=120)
    return parser


def _common_arguments(
    parser: argparse.ArgumentParser, *, default_model: str, default_workers: int
) -> None:
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--model", default=default_model)
    parser.add_argument("--workers", type=int, default=default_workers)
    parser.add_argument("--checkpoint-every", type=int, default=10)


def _load_sample(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, dict) else None
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != "reference-validation-sample-v1"
        or not isinstance(questions, list)
        or payload.get("sample_size") != len(questions)
    ):
        raise ValueError("sample must be a reference-validation-sample-v1 manifest")
    result: list[tuple[str, str]] = []
    for row in questions:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("question_id"), str)
            or not isinstance(row.get("question"), str)
        ):
            raise ValueError(
                "sample question rows require string question_id and question"
            )
        result.append((row["question_id"], row["question"]))
    if len({question_id for question_id, _ in result}) != len(result):
        raise ValueError("sample question IDs must be unique")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.provider == "gemini":
        if args.model not in GOOGLE_GENERATION_MODELS:
            approved = ", ".join(sorted(GOOGLE_GENERATION_MODELS))
            raise SystemExit(f"Google model must be one of: {approved}")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise SystemExit("GEMINI_API_KEY is required")
        if args.grounding and args.evidence_cache is None:
            raise SystemExit("--evidence-cache is required with --grounding")
        if not args.grounding and args.evidence_cache is not None:
            raise SystemExit("--evidence-cache requires --grounding")
        limiter = RateLimiter(args.requests_per_minute)

        def call(question: str):
            limiter.acquire()
            return call_gemini(
                question,
                api_key=api_key,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
                grounding=args.grounding,
            )

        provider = "gemini-grounded" if args.grounding else "gemini"
    elif args.provider == "deepinfra":
        if args.model not in DEEPINFRA_MODELS:
            approved = ", ".join(sorted(DEEPINFRA_MODELS))
            raise SystemExit(f"DeepInfra model must be one of: {approved}")
        api_token = os.environ.get("DEEPINFRA_API_TOKEN")
        if not api_token:
            raise SystemExit("DEEPINFRA_API_TOKEN is required")
        limiter = RateLimiter(args.requests_per_minute)

        def call(question: str):
            limiter.acquire()
            return call_deepinfra(
                question,
                api_token=api_token,
                model=args.model,
                max_output_tokens=args.max_output_tokens,
            )

        provider = "deepinfra"
    else:
        base_url = validate_openai_compatible_url(args.base_url)
        call = partial(call_openai_compatible, model=args.model, base_url=base_url)
        provider = "openai-compatible"

    cache = generate_missing(
        _load_sample(args.sample),
        cache_path=args.cache,
        provider=provider,
        model=args.model,
        call=call,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every,
        evidence_cache_path=(
            args.evidence_cache if args.provider == "gemini" else None
        ),
    )
    print(
        json.dumps(
            {
                "records": len(cache["records"]),
                "failures": len(cache["failures"]),
                "cache": str(args.cache),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
