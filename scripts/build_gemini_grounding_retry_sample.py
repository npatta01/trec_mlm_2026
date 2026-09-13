#!/usr/bin/env python3
"""Build a deterministic retry manifest from Gemini's no-Search audit IDs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


SELECTION_SEED = "gemini-search-retry-v1"


def _load_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def build_retry_sample(
    sample: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    limit: int | None,
) -> dict[str, Any]:
    questions = sample.get("questions")
    non_search_ids = audit.get("non_search_ids")
    if (
        sample.get("schema_version") != "reference-validation-sample-v1"
        or not isinstance(questions, list)
        or sample.get("sample_size") != len(questions)
    ):
        raise ValueError("sample must be a reference-validation-sample-v1 manifest")
    if not isinstance(non_search_ids, list) or not all(
        isinstance(question_id, str) for question_id in non_search_ids
    ):
        raise ValueError("audit must contain a string non_search_ids list")
    if len(set(non_search_ids)) != len(non_search_ids):
        raise ValueError("audit non_search_ids must be unique")
    if limit is not None and limit < 1:
        raise ValueError("limit must be at least one")

    by_id: dict[str, Mapping[str, Any]] = {}
    for row in questions:
        if (
            not isinstance(row, Mapping)
            or not isinstance(row.get("question_id"), str)
            or not isinstance(row.get("question"), str)
        ):
            raise ValueError("sample rows require string question_id and question")
        question_id = row["question_id"]
        if question_id in by_id:
            raise ValueError("sample question IDs must be unique")
        by_id[question_id] = row

    missing = sorted(set(non_search_ids) - set(by_id))
    if missing:
        raise ValueError(f"audit IDs are absent from the sample: {missing[:5]}")

    ordered_ids = sorted(
        non_search_ids,
        key=lambda question_id: hashlib.sha256(
            f"{SELECTION_SEED}\0{question_id}".encode("utf-8")
        ).hexdigest(),
    )
    if limit is not None:
        ordered_ids = ordered_ids[:limit]
    retry_questions = [
        {
            "question_id": question_id,
            "question": by_id[question_id]["question"],
            "search_expected": True,
        }
        for question_id in ordered_ids
    ]
    return {
        "schema_version": "reference-validation-sample-v1",
        "sample_size": len(retry_questions),
        "questions": retry_questions,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)

    result = build_retry_sample(
        _load_object(args.sample),
        _load_object(args.audit),
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "sample_size": result["sample_size"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
