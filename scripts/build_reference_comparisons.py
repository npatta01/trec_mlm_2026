#!/usr/bin/env python3
"""Build a deterministic four-way blind reference-answer review packet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping, Sequence

from trec_mlm.reference_adjudication import SYSTEMS, build_blind_packet
from trec_mlm.reference_data import (
    load_perplexity_questions,
    validate_exact_ids,
    write_jsonl_atomic,
)


def _load_generation_cache(
    path: Path, *, expected_model: str
) -> dict[str, str | None]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("model") != expected_model:
        raise ValueError(f"{path}: expected model {expected_model}")
    failures = payload.get("failures")
    records = payload.get("records")
    if not isinstance(failures, Mapping):
        raise ValueError(f"{path}: generation failures must be an object")
    if not isinstance(records, Mapping):
        raise ValueError(f"{path}: records must be an object")
    answers: dict[str, str | None] = {str(question_id): None for question_id in failures}
    for question_id, record in records.items():
        if not isinstance(record, Mapping):
            raise ValueError(f"{path}: invalid record {question_id}")
        answer = record.get("answer")
        if not isinstance(answer, str) or not answer.strip():
            raise ValueError(f"{path}: empty answer {question_id}")
        answers[str(question_id)] = (
            None if record.get("finish_reason") == "length" else answer
        )
    return answers


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--perplexity-cache", type=Path, required=True)
    for system in SYSTEMS[1:]:
        parser.add_argument(f"--{system}-cache", type=Path, required=True)
        parser.add_argument(f"--{system}-model", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--sealed-map", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def write_sealed_map(path: Path, sealed: Mapping[str, object]) -> None:
    """Atomically write sealed metadata without a permissive temp-file window."""

    content = json.dumps(sealed, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL,
        0o600,
    )
    try:
        handle = os.fdopen(descriptor, "w", encoding="utf-8")
    except Exception:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    try:
        with handle:
            handle.write(content)
        temporary.replace(path)
        os.chmod(path, 0o600)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    questions = sample.get("questions") if isinstance(sample, Mapping) else None
    if not isinstance(questions, list):
        raise ValueError("sample questions must be a list")
    expected_ids = [str(row["question_id"]) for row in questions]
    perplexity = {
        record.question_id: record.perplexity_answer_no_citations
        for record in load_perplexity_questions(args.perplexity_cache)
        if record.question_id in set(expected_ids)
    }
    answers: dict[str, dict[str, str | None]] = {"perplexity": perplexity}
    for system in SYSTEMS[1:]:
        answers[system] = _load_generation_cache(
            getattr(args, f"{system}_cache"),
            expected_model=getattr(args, f"{system}_model"),
        )
    for system, system_answers in answers.items():
        validate_exact_ids(expected_ids, system_answers, label=system)

    packet, sealed = build_blind_packet(sample, answers, seed=args.seed)
    write_jsonl_atomic(args.packet, packet)
    write_sealed_map(args.sealed_map, sealed)
    print(json.dumps({"questions": len(packet), "packet": str(args.packet)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
