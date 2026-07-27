"""Frozen prompt and strict decision contract for pointwise judging."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    conint,
    conlist,
    constr,
    field_validator,
    model_validator,
    ValidationError,
)


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


class DecisionV2(BaseModel):
    """Strict, self-consistent v2 judge output before runner conversion."""

    model_config = ConfigDict(extra="forbid", strict=True)

    label: Literal["good", "partial", "incorrect", "uncertain"]
    correctness: conint(strict=True, ge=0, le=4)
    completeness: conint(strict=True, ge=0, le=2)
    relevance: conint(strict=True, ge=0, le=2)
    error_flags: conlist(
        Literal[
            "entity_error",
            "number_error",
            "date_error",
            "negation_error",
            "unsupported_specificity",
            "refusal",
            "ambiguity",
            "reference_conflict",
        ],
        max_length=3,
    ) = Field(json_schema_extra={"uniqueItems": True})
    reference_status: Literal["sound", "incomplete", "suspect", "uncertain"]
    confidence: Literal["high", "medium", "low"]
    rationale: constr(strict=True, min_length=60, max_length=600)

    @field_validator("error_flags")
    @classmethod
    def _error_flags_are_unique(cls, flags: list[str]) -> list[str]:
        if len(flags) != len(set(flags)):
            raise ValueError("error_flags must be unique")
        return flags

    @model_validator(mode="after")
    def _enforce_cross_field_invariants(self) -> Self:
        flags = set(self.error_flags)
        good_values = (
            self.correctness == 4
            and self.completeness == 2
            and self.relevance == 2
            and flags in (set(), {"reference_conflict"})
        )
        if (self.label == "good") != good_values:
            raise ValueError("good must exactly match its required scores and flags")

        factual_error_flags = {
            "entity_error",
            "number_error",
            "date_error",
            "negation_error",
            "unsupported_specificity",
            "refusal",
        }
        if self.correctness == 4 and flags & factual_error_flags:
            raise ValueError("correctness 4 excludes factual-error flags")
        if self.correctness == 3 and self.label != "partial":
            raise ValueError("correctness 3 requires label partial")
        if self.correctness <= 1 and self.label not in {"incorrect", "uncertain"}:
            raise ValueError("correctness at most 1 requires incorrect or uncertain")
        if self.label == "incorrect" and not flags:
            raise ValueError("incorrect requires an error flag")

        if "refusal" in flags:
            if (
                self.correctness != 0
                or self.completeness != 0
                or self.relevance != 0
                or self.label not in {"incorrect", "uncertain"}
            ):
                raise ValueError(
                    "refusal requires zero scores and label incorrect or uncertain"
                )
        if self.label == "uncertain":
            if self.correctness != 0 or self.completeness != 0:
                raise ValueError("uncertain requires correctness and completeness 0")
            if "ambiguity" not in flags and self.reference_status not in {
                "suspect",
                "uncertain",
            }:
                raise ValueError(
                    "uncertain requires ambiguity or an unsettled reference"
                )
        if "reference_conflict" in flags and self.reference_status not in {
            "incomplete",
            "suspect",
        }:
            raise ValueError(
                "reference_conflict requires incomplete or suspect reference"
            )
        if "ambiguity" in flags and self.label != "uncertain":
            raise ValueError("ambiguity is reserved for an unjudgeable item")
        return self


def normalized_decision_v2_schema() -> dict[str, object]:
    """Return the stable, committed representation of the v2 model schema."""
    return _normalize_schema_presentation(DecisionV2.model_json_schema())


def _normalize_schema_presentation(value: Any) -> Any:
    """Remove recursively generated presentation metadata from a JSON schema."""
    if isinstance(value, dict):
        return {
            key: _normalize_schema_presentation(item)
            for key, item in value.items()
            if key not in {"title", "description"}
        }
    if isinstance(value, list):
        return [_normalize_schema_presentation(item) for item in value]
    return value


def load_prompt_contract(
    protocol_dir: Path, version: Literal["v1", "v2"] = "v1"
) -> PromptContract:
    """Load the versioned prompt and JSON schema without altering either."""
    versioned_paths = {
        "v1": ("pointwise-v1.txt", "decision-v1.schema.json"),
        "v2": ("pointwise-v2.txt", "decision-v2.schema.json"),
    }
    try:
        prompt_name, schema_name = versioned_paths[version]
    except KeyError as error:
        raise ValueError(f"unsupported pointwise judge version: {version}") from error
    prompt_path = protocol_dir / prompt_name
    schema_path = protocol_dir / schema_name
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
    repair_error: str | None = None,
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
        validation = (
            f"VALIDATION ERROR:\n{repair_error}\n\n" if repair_error is not None else ""
        )
        messages.append(
            {
                "role": "user",
                "content": (
                    "Your preceding output did not satisfy the unchanged schema. "
                    "Return only a corrected JSON object under the unchanged schema.\n\n"
                    + validation
                    + f"MALFORMED OUTPUT:\n{repair_raw}"
                ),
            }
        )
    return messages


def parse_judge_decision(
    raw: str, version: Literal["v1", "v2"] = "v1"
) -> JudgeDecision:
    """Parse a decision only when it satisfies the frozen domain contract."""
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as error:
        raise ValueError("decision is not valid JSON") from error
    if version == "v2":
        return _parse_v2_judge_decision(value)
    if version != "v1":
        raise ValueError(f"unsupported pointwise judge version: {version}")
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


def _parse_v2_judge_decision(value: object) -> JudgeDecision:
    try:
        decision = DecisionV2.model_validate(value)
    except ValidationError as error:
        details = error.errors(
            include_url=False,
            include_context=False,
            include_input=False,
        )
        detail = details[0]
        message = str(detail.get("msg", "invalid value"))
        if message.startswith("Value error, "):
            message = message.removeprefix("Value error, ")
        location = ".".join(str(item) for item in detail.get("loc", ()))
        if location:
            summary = f"v2 decision invalid at {location}: {message}"
        else:
            summary = f"v2 decision invariant violated: {message}"
        raise ValueError(summary) from error
    return JudgeDecision(
        label=decision.label,
        correctness=decision.correctness,
        completeness=decision.completeness,
        relevance=decision.relevance,
        error_flags=tuple(decision.error_flags),
        reference_status=decision.reference_status,
        confidence=decision.confidence,
        rationale=decision.rationale,
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
