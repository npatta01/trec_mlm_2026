"""Validated contracts for reference-answer sampling, generation, and review."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from trec_mlm.artifact_io import sha256_file

EXPECTED_PERPLEXITY_SHA256 = (
    "cde4ca41c555e29533961bc41496361b00f73b70264899597573362e355a4ee0"
)


@dataclass(frozen=True, slots=True)
class QuestionRecord:
    question_id: str
    question: str
    perplexity_answer: str
    perplexity_answer_no_citations: str
    perplexity_citations: tuple[str, ...]
    perplexity_unresolved: bool


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    question_id: str
    provider: str
    model: str
    answer: str
    elapsed_seconds: float
    usage: Mapping[str, Any]
    finish_reason: str | None


@dataclass(frozen=True, slots=True)
class ReviewDecision:
    question_id: str
    outcome: str
    selected_systems: tuple[str, ...]
    selected_answer: str | None
    rationale: str
    uncertainty: str
    deep_adjudication: bool
    evidence_urls: tuple[str, ...]


def load_perplexity_questions(
    path: Path,
    *,
    expected_sha256: str | None = EXPECTED_PERPLEXITY_SHA256,
) -> list[QuestionRecord]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError("Perplexity cache SHA-256 does not match the frozen manifest")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, dict) or payload.get("total_questions") != len(records):
        raise ValueError("Perplexity cache record count is inconsistent")
    unresolved = set(payload.get("retry_summary", {}).get("unresolved", []))
    result = []
    for question_id in sorted(records):
        item = records[question_id]
        response = item.get("response") or {}
        result.append(
            QuestionRecord(
                question_id=str(question_id),
                question=str(item["question"]),
                perplexity_answer=str(response.get("answer") or ""),
                perplexity_answer_no_citations=str(
                    response.get("answer_no_citations")
                    or response.get("answer")
                    or ""
                ),
                perplexity_citations=tuple(
                    str(url) for url in response.get("citations") or []
                ),
                perplexity_unresolved=str(question_id) in unresolved,
            )
        )
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        (dict(row) for row in rows), key=lambda row: str(row["question_id"])
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in ordered
        ),
        encoding="utf-8",
    )
    temporary.replace(path)


def validate_exact_ids(
    expected: Iterable[str], observed: Iterable[str], *, label: str
) -> None:
    expected_set, observed_set = set(expected), set(observed)
    if expected_set != observed_set:
        raise ValueError(
            f"{label}: missing={sorted(expected_set-observed_set)!r} "
            f"extra={sorted(observed_set-expected_set)!r}"
        )
