#!/usr/bin/env python3
"""Build deterministic full-corpus and 50-question grounded canary manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence


SEARCH_EXPECTED_RE = re.compile(
    r"\b(?:current|currently|latest|today|now|recent|recently|202[4-9]|"
    r"president|prime minister|ceo|price|cost|weather|score|schedule|"
    r"stock|exchange rate|law|regulation|available|release date)\b",
    re.IGNORECASE,
)


def build_manifests(cache: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    records = cache.get("records")
    if not isinstance(records, Mapping) or len(records) != 7_759:
        raise ValueError("Perplexity source must contain exactly 7,759 records")
    questions = []
    for question_id, record in records.items():
        if not isinstance(record, Mapping) or not isinstance(
            record.get("question"), str
        ):
            raise ValueError("every source record must contain a string question")
        question = record["question"]
        questions.append(
            {
                "question_id": str(question_id),
                "question": question,
                "search_expected": bool(SEARCH_EXPECTED_RE.search(question)),
            }
        )
    questions.sort(key=lambda row: row["question_id"])
    ranked = sorted(
        questions,
        key=lambda row: hashlib.sha256(
            f"grounded-canary-2026:{row['question_id']}".encode()
        ).hexdigest(),
    )
    expected = [row for row in ranked if row["search_expected"]][:25]
    controls = [row for row in ranked if not row["search_expected"]][:25]
    canary_questions = sorted(expected + controls, key=lambda row: row["question_id"])
    if len(canary_questions) != 50:
        raise ValueError("could not construct a balanced 50-question canary")
    common = {"schema_version": "reference-validation-sample-v1"}
    full_questions = [
        {"question_id": row["question_id"], "question": row["question"]}
        for row in questions
    ]
    full = {
        **common,
        "sample_size": len(full_questions),
        "questions": full_questions,
    }
    canary = {
        **common,
        "sample_size": len(canary_questions),
        "selection": {
            "method": "sha256 deterministic balanced canary",
            "search_expected": 25,
            "controls": 25,
        },
        "questions": canary_questions,
    }
    return full, canary


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perplexity-cache", type=Path, required=True)
    parser.add_argument("--full-output", type=Path, required=True)
    parser.add_argument("--canary-output", type=Path, required=True)
    args = parser.parse_args(argv)
    full, canary = build_manifests(
        json.loads(args.perplexity_cache.read_text(encoding="utf-8"))
    )
    _write(args.full_output, full)
    _write(args.canary_output, canary)
    print(json.dumps({"full": full["sample_size"], "canary": canary["sample_size"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
