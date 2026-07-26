"""Frozen prompt and strict decision contract for pointwise judging."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


LABELS = frozenset({"good", "partial", "incorrect", "uncertain"})
ERROR_FLAGS = frozenset(
    {
        "entity_error",
        "number_error",
        "date_error",
        "negation_error",
        "unsupported_specificity",
        "refusal",
        "ambiguity",
        "reference_conflict",
    }
)
REFERENCE_STATUSES = frozenset({"sound", "incomplete", "suspect", "uncertain"})
CONFIDENCES = frozenset({"high", "medium", "low"})
DECISION_KEYS = frozenset(
    {
        "label",
        "correctness",
        "completeness",
        "relevance",
        "error_flags",
        "reference_status",
        "confidence",
        "rationale",
    }
)


@dataclass(frozen=True, slots=True)
class CandidateCase:
    question_id: str
    response_id: str
    question: str
    reference_answer: str
    candidate_answer: str
    strata: tuple[str, ...]
    selection_reason: str


@dataclass(frozen=True, slots=True)
class JudgeDecision:
    label: str
    correctness: int
    completeness: int
    relevance: int
    error_flags: tuple[str, ...]
    reference_status: str
    confidence: str
    rationale: str


@dataclass(frozen=True, slots=True)
class PromptContract:
    prompt_path: Path
    schema_path: Path
    prompt: str
    schema: dict[str, object]


def load_prompt_contract(protocol_dir: Path) -> PromptContract:
    """Load the versioned prompt and JSON schema without altering either."""
    prompt_path = protocol_dir / "pointwise-v1.txt"
    schema_path = protocol_dir / "decision-v1.schema.json"
    prompt = prompt_path.read_text(encoding="utf-8")
    try:
        schema = json.loads(
            schema_path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as error:
        raise ValueError("decision schema is not valid JSON") from error
    if type(schema) is not dict:
        raise ValueError("decision schema must be a JSON object")
    return PromptContract(
        prompt_path=prompt_path,
        schema_path=schema_path,
        prompt=prompt,
        schema=schema,
    )


def render_judge_messages(
    case: CandidateCase,
    contract: PromptContract,
    *,
    repair_raw: str | None = None,
) -> list[dict[str, str]]:
    """Render the invariant judge request, optionally followed by one repair request."""
    messages = [
        {"role": "system", "content": contract.prompt},
        {
            "role": "user",
            "content": (
                f"QUESTION ID: {case.question_id}\n"
                f"RESPONSE ID: {case.response_id}\n\n"
                f"QUESTION:\n{case.question}\n\n"
                "PROVISIONAL REFERENCE (may be incomplete or wrong):\n"
                f"{case.reference_answer}\n\n"
                f"CANDIDATE RESPONSE:\n{case.candidate_answer}"
            ),
        },
    ]
    if repair_raw is not None:
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your preceding output did not satisfy the unchanged schema. "
                    "Return only a corrected JSON object under the unchanged schema.\n\n"
                    f"MALFORMED OUTPUT:\n{repair_raw}"
                ),
            }
        )
    return messages


def parse_judge_decision(raw: str) -> JudgeDecision:
    """Parse a decision only when it satisfies the frozen domain contract."""
    try:
        value = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except json.JSONDecodeError as error:
        raise ValueError("decision is not valid JSON") from error
    if type(value) is not dict or set(value) != DECISION_KEYS:
        raise ValueError("decision must contain exact keys")

    label = _closed_string(value, "label", LABELS)
    correctness = _bounded_integer(value, "correctness", 0, 4)
    completeness = _bounded_integer(value, "completeness", 0, 2)
    relevance = _bounded_integer(value, "relevance", 0, 2)
    error_flags = _error_flags(value["error_flags"])
    reference_status = _closed_string(value, "reference_status", REFERENCE_STATUSES)
    confidence = _closed_string(value, "confidence", CONFIDENCES)
    rationale = _rationale(value["rationale"])

    if label == "good" and (correctness < 4 or relevance < 2):
        raise ValueError("good label requires correctness 4 and relevance 2")
    return JudgeDecision(
        label=label,
        correctness=correctness,
        completeness=completeness,
        relevance=relevance,
        error_flags=error_flags,
        reference_status=reference_status,
        confidence=confidence,
        rationale=rationale,
    )


def decision_to_dict(decision: JudgeDecision) -> dict[str, object]:
    """Convert a validated decision to JSON-ready primitive values."""
    return {
        "label": decision.label,
        "correctness": decision.correctness,
        "completeness": decision.completeness,
        "relevance": decision.relevance,
        "error_flags": list(decision.error_flags),
        "reference_status": decision.reference_status,
        "confidence": decision.confidence,
        "rationale": decision.rationale,
    }


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")


def _closed_string(value: dict[str, Any], field: str, allowed: frozenset[str]) -> str:
    item = value[field]
    if type(item) is not str or item not in allowed:
        raise ValueError(f"invalid {field}")
    return item


def _bounded_integer(
    value: dict[str, Any], field: str, minimum: int, maximum: int
) -> int:
    item = value[field]
    if type(item) is not int or not minimum <= item <= maximum:
        raise ValueError(f"invalid {field}")
    return item


def _error_flags(value: object) -> tuple[str, ...]:
    if type(value) is not list:
        raise ValueError("invalid error_flags")
    if any(type(flag) is not str or flag not in ERROR_FLAGS for flag in value):
        raise ValueError("invalid error_flags")
    if len(value) != len(set(value)):
        raise ValueError("duplicate error_flags")
    return tuple(value)


def _rationale(value: object) -> str:
    if type(value) is not str:
        raise ValueError("invalid rationale")
    words = value.split()
    if not 1 <= len(words) <= 80:
        raise ValueError("rationale must contain 1 to 80 words")
    return value


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key in decision: {key}")
        value[key] = item
    return value
