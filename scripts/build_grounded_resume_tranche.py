#!/usr/bin/env python3
"""Build a bounded pending-question tranche from a resumable generation cache."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def build_tranche(
    manifest: Mapping[str, Any], cache: Mapping[str, Any], *, limit: int
) -> dict[str, Any]:
    if limit < 1:
        raise ValueError("limit must be positive")
    questions = manifest.get("questions")
    records = cache.get("records")
    if not isinstance(questions, list) or not isinstance(records, Mapping):
        raise ValueError("manifest questions and cache records are required")
    pending = [
        row
        for row in questions
        if isinstance(row, Mapping) and str(row.get("question_id")) not in records
    ][:limit]
    return {
        "schema_version": "reference-validation-sample-v1",
        "sample_size": len(pending),
        "selection": {
            "method": "first pending IDs in frozen full-manifest order",
            "limit": limit,
        },
        "questions": pending,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    tranche = build_tranche(
        json.loads(args.manifest.read_text(encoding="utf-8")),
        json.loads(args.cache.read_text(encoding="utf-8")),
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(tranche, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"questions": tranche["sample_size"], "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
