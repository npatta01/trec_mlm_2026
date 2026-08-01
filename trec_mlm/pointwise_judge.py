"""Frozen prompt and strict decision contract for pointwise judging."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Self

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
        "other_error",
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


class _DuplicateJsonKeyError(ValueError):
    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__("duplicate JSON key in decision")


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
            "other_error",
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
            "other_error",
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
                or self.label not in {"incorrect", "uncertain"}
            ):
                raise ValueError(
                    "refusal requires zero correctness/completeness and label "
                    "incorrect or uncertain"
                )
            # A model can mark a wrong attempted answer as a refusal while
            # still assigning on-topic relevance. Preserve that parseable
            # score; downstream consumers can inspect the flag and rationale.
            if self.label == "uncertain" and self.relevance != 0:
                raise ValueError("uncertain refusal requires relevance 0")
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


class DecisionV3(BaseModel):
    """Compact model-facing v3 output before deterministic runner conversion."""

    model_config = ConfigDict(extra="forbid", strict=True)

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
            "other_error",
        ],
        max_length=3,
    ) = Field(json_schema_extra={"uniqueItems": True})
    reference_status: Literal["sound", "incomplete", "suspect", "uncertain"]
    uncertain: bool
    rationale: constr(strict=True, max_length=240)

    @field_validator("error_flags")
    @classmethod
    def _error_flags_are_unique(cls, flags: list[str]) -> list[str]:
        if len(flags) != len(set(flags)):
            raise ValueError("error_flags must be unique")
        return flags


class DecisionV4(BaseModel):
    """Compact categorical v4 output for the model-facing judge contract."""

    model_config = ConfigDict(extra="forbid", strict=True)

    verdict: Literal["yes", "partial", "no", "uncertain"]
    correctness: Literal["correct", "mixed", "wrong", "unknown"]
    completeness: Literal["complete", "partial", "missing", "unknown"]
    relevance: Literal["relevant", "off_topic", "unknown"]
    reference_status: Literal["sound", "incomplete", "suspect", "uncertain"]
    rationale: constr(strict=True, max_length=240)


class DecisionV5(BaseModel):
    """Correctness-first model output; overall outcomes are derived locally."""

    model_config = ConfigDict(extra="forbid", strict=True)

    correctness: Literal["correct", "mixed", "wrong", "unknown"]
    completeness: Literal["complete", "partial", "missing", "unknown"]
    relevance: Literal["relevant", "off_topic", "unknown"]
    reference_status: Literal["sound", "incomplete", "suspect", "uncertain"]
    rationale: constr(strict=True, max_length=240)


class DecisionV8(BaseModel):
    """Reference-grounded model output with correctness-first acceptance."""

    model_config = ConfigDict(extra="forbid", strict=True)

    correctness: Literal["correct", "mixed", "wrong", "unknown"]
    coverage: Literal["complete", "partial", "missing", "unknown"]
    rationale: constr(strict=True, max_length=240)


class DecisionV9(BaseModel):
    """Few-shot reference-grounded output with truth-independent coverage."""

    model_config = ConfigDict(extra="forbid", strict=True)

    correctness: Literal["correct", "mixed", "wrong", "unknown"]
    coverage: Literal["complete", "partial", "missing"]
    rationale: constr(strict=True, max_length=240)


class DecisionV10(BaseModel):
    """Strict two-field agreement decision for the simplified v10 pilot."""

    model_config = ConfigDict(extra="forbid", strict=True)

    grade: conint(strict=True, ge=0, le=2)
    reason: Literal[
        "supported", "mixed", "reference_silent", "contradicted", "nonanswer"
    ]

    @model_validator(mode="after")
    def _enforce_grade_reason_matrix(self) -> Self:
        allowed_reasons = {
            2: {"supported"},
            1: {"mixed", "reference_silent"},
            0: {"contradicted", "nonanswer"},
        }
        if self.reason not in allowed_reasons[self.grade]:
            raise ValueError("grade and reason are an invalid v10 combination")
        return self


class DecisionV11(BaseModel):
    """Central-claim output for the single-pass pointwise v11 bakeoff."""

    model_config = ConfigDict(extra="forbid", strict=True)

    central_claim: constr(strict=True, max_length=240)
    answer_present: bool
    supported_core: bool
    material_conflict: bool

    @field_validator("central_claim")
    @classmethod
    def _central_claim_has_at_most_twenty_words(cls, claim: str) -> str:
        if claim != claim.strip():
            raise ValueError("central_claim must not have outer whitespace")
        if len(claim.split()) > 20:
            raise ValueError("central_claim must contain at most 20 words")
        return claim

    @model_validator(mode="after")
    def _enforce_v11_invariants(self) -> Self:
        if (self.central_claim == "") != (not self.answer_present):
            raise ValueError("central_claim must be empty iff answer_present is false")
        if not self.answer_present and (
            self.supported_core or self.material_conflict
        ):
            raise ValueError(
                "a nonanswer cannot have supported_core or material_conflict"
            )
        return self


def normalized_decision_v2_schema() -> dict[str, object]:
    """Return the stable, committed representation of the v2 model schema."""
    return _normalize_schema_presentation(DecisionV2.model_json_schema())


def normalized_decision_v3_schema() -> dict[str, object]:
    """Return the stable, committed representation of the v3 model schema."""
    return _normalize_schema_presentation(DecisionV3.model_json_schema())


def normalized_decision_v4_schema() -> dict[str, object]:
    """Return the stable, committed representation of the v4 model schema."""
    return _normalize_schema_presentation(DecisionV4.model_json_schema())


def normalized_decision_v5_schema() -> dict[str, object]:
    """Return the stable, committed representation of the v5 model schema."""
    return _normalize_schema_presentation(DecisionV5.model_json_schema())


def normalized_decision_v8_schema() -> dict[str, object]:
    """Return the stable, committed representation of the v8 model schema."""
    return _normalize_schema_presentation(DecisionV8.model_json_schema())


def normalized_decision_v9_schema() -> dict[str, object]:
    """Return the stable, committed representation of the v9 model schema."""
    return _normalize_schema_presentation(DecisionV9.model_json_schema())


def normalized_decision_v10_schema() -> dict[str, object]:
    """Return the provider-safe v10 schema, keeping pair validation local."""
    return {
        "additionalProperties": False,
        "properties": {
            "grade": {"enum": [0, 1, 2], "type": "integer"},
            "reason": {
                "enum": [
                    "supported",
                    "mixed",
                    "reference_silent",
                    "contradicted",
                    "nonanswer",
                ],
                "type": "string",
            },
        },
        "required": ["grade", "reason"],
        "type": "object",
    }


def normalized_decision_v11_schema() -> dict[str, object]:
    """Return the provider-safe v11 schema with local invariant validation."""
    return {
        "additionalProperties": False,
        "properties": {
            "central_claim": {"maxLength": 240, "type": "string"},
            "answer_present": {"type": "boolean"},
            "supported_core": {"type": "boolean"},
            "material_conflict": {"type": "boolean"},
        },
        "required": [
            "central_claim",
            "answer_present",
            "supported_core",
            "material_conflict",
        ],
        "type": "object",
    }


_V3_FIELDS = frozenset(
    {
        "correctness",
        "completeness",
        "relevance",
        "error_flags",
        "reference_status",
        "uncertain",
        "rationale",
    }
)
_V3_ERROR_FLAG_ORDER = (
    "entity_error",
    "number_error",
    "date_error",
    "negation_error",
    "unsupported_specificity",
    "refusal",
    "ambiguity",
    "reference_conflict",
    "other_error",
)
_V3_STATUS_VALUES = frozenset({"sound", "incomplete", "suspect", "uncertain"})
_V3_INTEGER_RE = re.compile(r"^[+-]?\d+$")


def normalize_judge_decision_v3(raw: str) -> dict[str, object]:
    """Parse a v3 response once and return a stable, automation-safe record.

    v3 intentionally does not retry ordinary model formatting mistakes. Clean
    output is validated strictly with :class:`DecisionV3`; recoverable mistakes
    are coerced locally and recorded in ``repair_codes``. If the three scores
    cannot be recovered, the result is ``unscorable`` and retains the raw text
    for later inspection.
    """
    if type(raw) is not str:
        raise TypeError("v3 raw decision must be a string")

    value, extraction_error = _decode_v3_object(raw)
    if extraction_error is not None and value is None:
        return _v3_unscorable(raw, extraction_error)
    if type(value) is not dict:
        return _v3_unscorable(raw, "json_not_object")

    try:
        clean = DecisionV3.model_validate(value)
    except ValidationError:
        clean = None
    else:
        result = {
            "decision": clean.model_dump(mode="json"),
            "label": _derive_v3_label(clean.correctness, clean.completeness, clean.relevance, clean.error_flags, clean.uncertain),
            "parse_status": "coerced" if extraction_error else "clean",
            "repair_codes": [extraction_error] if extraction_error else [],
        }
        if extraction_error:
            result["raw"] = raw
        return result

    repaired, repair_codes, scores_ok = _coerce_v3_value(value)
    if extraction_error:
        repair_codes.insert(0, extraction_error)
    if not scores_ok:
        return _v3_unscorable(raw, "missing_or_invalid_scores", repair_codes)
    try:
        decision = DecisionV3.model_validate(repaired)
    except ValidationError:
        # The coercion path is deliberately conservative. A failed model
        # validation here is still represented, rather than entering a retry
        # loop or leaking an exception into the production batch.
        return _v3_unscorable(raw, "coercion_failed", repair_codes)

    result: dict[str, object] = {
        "decision": decision.model_dump(mode="json"),
        "label": _derive_v3_label(
            decision.correctness,
            decision.completeness,
            decision.relevance,
            decision.error_flags,
            decision.uncertain,
        ),
        "parse_status": "coerced",
        "repair_codes": repair_codes,
        "raw": raw,
    }
    return result


def _decode_v3_object(raw: str) -> tuple[object | None, str | None]:
    try:
        return (
            json.loads(
                raw,
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
            ),
            None,
        )
    except _DuplicateJsonKeyError:
        return None, "duplicate_json_key"
    except (json.JSONDecodeError, ValueError):
        # Some providers wrap otherwise useful JSON in a short preamble or
        # code fence. Recover only the outermost object; never guess fields.
        start = raw.find("{")
        end = raw.rfind("}")
        if start < 0 or end <= start:
            return None, "invalid_json"
        try:
            value = json.loads(
                raw[start : end + 1],
                object_pairs_hook=_object_without_duplicate_keys,
                parse_constant=_reject_nonfinite_json,
            )
        except _DuplicateJsonKeyError:
            return None, "duplicate_json_key"
        except (json.JSONDecodeError, ValueError):
            return None, "invalid_json"
        if type(value) is not dict:
            return None, "json_not_object"
        return value, "extract_json_object"


def _coerce_v3_value(value: dict[str, Any]) -> tuple[dict[str, object], list[str], bool]:
    repaired: dict[str, object] = {}
    repair_codes: list[str] = []
    if set(value) - _V3_FIELDS:
        repair_codes.append("drop_unknown_keys")

    scores_ok = True
    for field, minimum, maximum in (
        ("correctness", 0, 4),
        ("completeness", 0, 2),
        ("relevance", 0, 2),
    ):
        if field not in value:
            scores_ok = False
            continue
        parsed = _coerce_v3_integer(value[field])
        if parsed is None:
            scores_ok = False
            continue
        if type(value[field]) is not int:
            repair_codes.append(f"coerce_{field}")
        if parsed < minimum or parsed > maximum:
            parsed = min(maximum, max(minimum, parsed))
            repair_codes.append(f"clamp_{field}")
        repaired[field] = parsed

    flags = value.get("error_flags", [])
    if "error_flags" not in value:
        repair_codes.append("default_error_flags")
    elif type(flags) is str:
        flags = [flags]
        repair_codes.append("coerce_error_flags")
    if type(flags) is not list:
        flags = []
        repair_codes.append("default_error_flags")
    normalized_flags: list[str] = []
    for flag in flags:
        if type(flag) is str:
            normalized_flag = flag.strip().lower()
            if normalized_flag != flag:
                repair_codes.append("normalize_error_flags")
            flag = normalized_flag
        if type(flag) is not str or flag not in ERROR_FLAGS:
            repair_codes.append("drop_unknown_error_flags")
            continue
        if flag in normalized_flags:
            repair_codes.append("dedupe_error_flags")
            continue
        normalized_flags.append(flag)
    if len(normalized_flags) > 3:
        normalized_flags = normalized_flags[:3]
        repair_codes.append("cap_error_flags")
    repaired["error_flags"] = normalized_flags

    reference_status = value.get("reference_status", "uncertain")
    if type(reference_status) is str:
        normalized_status = reference_status.strip().lower()
    else:
        normalized_status = ""
    if normalized_status not in _V3_STATUS_VALUES:
        reference_status = "uncertain"
        repair_codes.append("default_reference_status")
    else:
        reference_status = normalized_status
        if reference_status != value.get("reference_status"):
            repair_codes.append("normalize_reference_status")
        if "reference_status" not in value:
            repair_codes.append("default_reference_status")
    repaired["reference_status"] = reference_status

    uncertain = value.get("uncertain", True)
    if type(uncertain) is not bool:
        if type(uncertain) is str and uncertain.strip().lower() in {"true", "false"}:
            uncertain = uncertain.strip().lower() == "true"
            repair_codes.append("coerce_uncertain")
        else:
            uncertain = True
            repair_codes.append("default_uncertain")
    elif "uncertain" not in value:
        repair_codes.append("default_uncertain_true")
    repaired["uncertain"] = uncertain

    rationale = value.get("rationale", "")
    if type(rationale) is not str:
        rationale = "" if rationale is None else str(rationale)
        repair_codes.append("coerce_rationale")
    if len(rationale) > 240:
        rationale = rationale[:240]
        repair_codes.append("truncate_rationale")
    repaired["rationale"] = rationale

    if "refusal" in normalized_flags:
        if repaired.get("correctness") != 0 or repaired.get("completeness") != 0:
            repaired["correctness"] = 0
            repaired["completeness"] = 0
            repair_codes.append("enforce_refusal_scores")
    return repaired, _dedupe_preserving_order(repair_codes), scores_ok


def _coerce_v3_integer(value: object) -> int | None:
    if type(value) is bool:
        return None
    if type(value) is int:
        return value
    if type(value) is float and value.is_integer():
        return int(value)
    if type(value) is str and _V3_INTEGER_RE.fullmatch(value.strip()):
        return int(value.strip())
    return None


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def _derive_v3_label(
    correctness: int,
    completeness: int,
    relevance: int,
    error_flags: list[str],
    uncertain: bool,
) -> str:
    if uncertain:
        return "uncertain"
    if correctness <= 1:
        return "incorrect"
    if correctness == 4 and completeness == 2 and relevance == 2 and not error_flags:
        return "good"
    return "partial"


def _v3_unscorable(
    raw: str,
    code: str,
    repair_codes: list[str] | None = None,
) -> dict[str, object]:
    codes = _dedupe_preserving_order([*(repair_codes or []), code])
    return {
        "decision": {
            "correctness": None,
            "completeness": None,
            "relevance": None,
            "error_flags": [],
            "reference_status": "uncertain",
            "uncertain": True,
            "rationale": "",
        },
        "label": "uncertain",
        "parse_status": "unscorable",
        "repair_codes": codes,
        "raw": raw,
    }


_V4_FIELDS = frozenset(
    {
        "verdict",
        "correctness",
        "completeness",
        "relevance",
        "reference_status",
        "rationale",
    }
)
_V4_ENUMS: dict[str, frozenset[str]] = {
    "verdict": frozenset({"yes", "partial", "no", "uncertain"}),
    "correctness": frozenset({"correct", "mixed", "wrong", "unknown"}),
    "completeness": frozenset({"complete", "partial", "missing", "unknown"}),
    "relevance": frozenset({"relevant", "off_topic", "unknown"}),
    "reference_status": frozenset({"sound", "incomplete", "suspect", "uncertain"}),
}

_V5_FIELDS = frozenset(
    {"correctness", "completeness", "relevance", "reference_status", "rationale"}
)
_V5_ENUMS: dict[str, frozenset[str]] = {
    "correctness": frozenset({"correct", "mixed", "wrong", "unknown"}),
    "completeness": frozenset({"complete", "partial", "missing", "unknown"}),
    "relevance": frozenset({"relevant", "off_topic", "unknown"}),
    "reference_status": frozenset({"sound", "incomplete", "suspect", "uncertain"}),
}


def normalize_judge_decision_v4(raw: str) -> dict[str, object]:
    """Validate one compact v4 response without entering a retry loop."""
    if type(raw) is not str:
        raise TypeError("v4 raw decision must be a string")
    value, extraction_code = _decode_v3_object(raw)
    if value is None:
        return _v4_unscorable(raw, extraction_code or "invalid_json")
    if type(value) is not dict:
        return _v4_unscorable(raw, "json_not_object")
    try:
        decision = DecisionV4.model_validate(value)
    except ValidationError:
        decision = None
    else:
        result: dict[str, object] = {
            "decision": decision.model_dump(mode="json"),
            "parse_status": "coerced" if extraction_code else "clean",
            "repair_codes": [extraction_code] if extraction_code else [],
        }
        if extraction_code:
            result["raw"] = raw
        return result

    repaired: dict[str, object] = {}
    repair_codes: list[str] = []
    if set(value) - _V4_FIELDS:
        repair_codes.append("drop_unknown_keys")
    defaults = {
        "verdict": "uncertain",
        "correctness": "unknown",
        "completeness": "unknown",
        "relevance": "unknown",
        "reference_status": "uncertain",
    }
    for field, allowed in _V4_ENUMS.items():
        item = value.get(field)
        normalized = item.strip().lower() if type(item) is str else ""
        if normalized not in allowed:
            repaired[field] = defaults[field]
            repair_codes.append(f"default_{field}")
        else:
            repaired[field] = normalized
            if normalized != item:
                repair_codes.append(f"normalize_{field}")
    rationale = value.get("rationale", "")
    if type(rationale) is not str:
        rationale = "" if rationale is None else str(rationale)
        repair_codes.append("coerce_rationale")
    if len(rationale) > 240:
        rationale = rationale[:240]
        repair_codes.append("truncate_rationale")
    repaired["rationale"] = rationale
    if extraction_code:
        repair_codes.insert(0, extraction_code)
    try:
        decision = DecisionV4.model_validate(repaired)
    except ValidationError:
        return _v4_unscorable(raw, "coercion_failed", repair_codes)
    return {
        "decision": decision.model_dump(mode="json"),
        "parse_status": "coerced",
        "repair_codes": _dedupe_preserving_order(repair_codes),
        "raw": raw,
    }


def _v4_unscorable(raw: str, code: str) -> dict[str, object]:
    return {
        "decision": {
            "verdict": "uncertain",
            "correctness": "unknown",
            "completeness": "unknown",
            "relevance": "unknown",
            "reference_status": "uncertain",
            "rationale": "",
        },
        "parse_status": "unscorable",
        "repair_codes": [code],
        "raw": raw,
    }


def _derive_v5_outcome(decision: dict[str, str]) -> tuple[str, bool, bool]:
    """Derive the user-facing verdict and correctness-first acceptance gate."""
    correctness = decision["correctness"]
    completeness = decision["completeness"]
    relevance = decision["relevance"]
    reference_status = decision["reference_status"]
    if "unknown" in {correctness, completeness, relevance}:
        verdict = "uncertain"
    elif relevance == "off_topic" or correctness == "wrong":
        verdict = "no"
    elif correctness == "mixed" or completeness != "complete":
        verdict = "partial"
    else:
        verdict = "yes"
    acceptable = (
        correctness == "correct"
        and completeness in {"complete", "partial"}
        and relevance == "relevant"
        and reference_status == "sound"
    )
    needs_review = (
        correctness in {"mixed", "unknown"}
        or relevance == "unknown"
        or reference_status != "sound"
    )
    return verdict, acceptable, needs_review


def _v5_result(decision: DecisionV5, *, parse_status: str, repair_codes: list[str], raw: str | None = None) -> dict[str, object]:
    value = decision.model_dump(mode="json")
    verdict, acceptable, needs_review = _derive_v5_outcome(value)
    value.update(
        {"verdict": verdict, "acceptable": acceptable, "needs_review": needs_review}
    )
    result: dict[str, object] = {
        "decision": value,
        "parse_status": parse_status,
        "repair_codes": repair_codes,
    }
    if raw is not None:
        result["raw"] = raw
    return result


def normalize_judge_decision_v5(raw: str) -> dict[str, object]:
    """Validate correctness-first v5 output and derive its overall outcome."""
    if type(raw) is not str:
        raise TypeError("v5 raw decision must be a string")
    value, extraction_code = _decode_v3_object(raw)
    if value is None:
        return _v5_unscorable(raw, extraction_code or "invalid_json")
    if type(value) is not dict:
        return _v5_unscorable(raw, "json_not_object")
    try:
        decision = DecisionV5.model_validate(value)
    except ValidationError:
        decision = None
    else:
        return _v5_result(
            decision,
            parse_status="coerced" if extraction_code else "clean",
            repair_codes=[extraction_code] if extraction_code else [],
            raw=raw if extraction_code else None,
        )

    repaired: dict[str, object] = {}
    repair_codes: list[str] = []
    if set(value) - _V5_FIELDS:
        repair_codes.append("drop_unknown_keys")
    defaults = {
        "correctness": "unknown",
        "completeness": "unknown",
        "relevance": "unknown",
        "reference_status": "uncertain",
    }
    for field, allowed in _V5_ENUMS.items():
        item = value.get(field)
        normalized = item.strip().lower() if type(item) is str else ""
        if normalized not in allowed:
            repaired[field] = defaults[field]
            repair_codes.append(f"default_{field}")
        else:
            repaired[field] = normalized
            if normalized != item:
                repair_codes.append(f"normalize_{field}")
    rationale = value.get("rationale", "")
    if type(rationale) is not str:
        rationale = "" if rationale is None else str(rationale)
        repair_codes.append("coerce_rationale")
    if len(rationale) > 240:
        rationale = rationale[:240]
        repair_codes.append("truncate_rationale")
    repaired["rationale"] = rationale
    if extraction_code:
        repair_codes.insert(0, extraction_code)
    try:
        decision = DecisionV5.model_validate(repaired)
    except ValidationError:
        return _v5_unscorable(raw, "coercion_failed", repair_codes)
    return _v5_result(
        decision,
        parse_status="coerced",
        repair_codes=_dedupe_preserving_order(repair_codes),
        raw=raw,
    )


def _v5_unscorable(
    raw: str, code: str, repair_codes: list[str] | None = None
) -> dict[str, object]:
    decision = DecisionV5(
        correctness="unknown",
        completeness="unknown",
        relevance="unknown",
        reference_status="uncertain",
        rationale="",
    )
    result = _v5_result(
        decision,
        parse_status="unscorable",
        repair_codes=_dedupe_preserving_order([*(repair_codes or []), code]),
        raw=raw,
    )
    return result


_V8_FIELDS = frozenset({"correctness", "coverage", "rationale"})
_V8_ENUMS = {
    "correctness": frozenset({"correct", "mixed", "wrong", "unknown"}),
    "coverage": frozenset({"complete", "partial", "missing", "unknown"}),
}
_V9_ENUMS = {
    "correctness": frozenset({"correct", "mixed", "wrong", "unknown"}),
    "coverage": frozenset({"complete", "partial", "missing"}),
}


def _v8_result(
    decision: DecisionV8 | DecisionV9,
    *,
    parse_status: str,
    repair_codes: list[str],
    raw: str | None = None,
) -> dict[str, object]:
    value = decision.model_dump(mode="json")
    value.update(
        {
            "acceptable": decision.correctness == "correct",
            "needs_review": decision.correctness in {"mixed", "unknown"},
        }
    )
    result: dict[str, object] = {
        "decision": value,
        "parse_status": parse_status,
        "repair_codes": repair_codes,
    }
    if raw is not None:
        result["raw"] = raw
    return result


def _v8_unscorable(raw: str, repair_codes: list[str]) -> dict[str, object]:
    return {
        "decision": {
            "correctness": None,
            "coverage": None,
            "rationale": "",
            "acceptable": False,
            "needs_review": True,
        },
        "parse_status": "unscorable",
        "repair_codes": _dedupe_preserving_order(repair_codes),
        "raw": raw,
    }


def normalize_judge_decision_v8(raw: str) -> dict[str, object]:
    """Validate a v8 response and transparently repair shape-only mistakes."""
    return _normalize_reference_grounded_decision(
        raw, version="v8", decision_type=DecisionV8, enums=_V8_ENUMS
    )


def normalize_judge_decision_v9(raw: str) -> dict[str, object]:
    """Validate a v9 response and transparently repair shape-only mistakes."""
    return _normalize_reference_grounded_decision(
        raw, version="v9", decision_type=DecisionV9, enums=_V9_ENUMS
    )


def normalize_judge_decision_v10(raw: str) -> dict[str, object]:
    """Validate v10 output, repairing only reason casing and surrounding space."""
    if type(raw) is not str:
        raise TypeError("v10 raw decision must be a string")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except _DuplicateJsonKeyError:
        return _v10_unscorable(raw, "duplicate_json_key")
    except (json.JSONDecodeError, ValueError):
        return _v10_unscorable(raw, "invalid_json")
    if type(value) is not dict:
        return _v10_unscorable(raw, "json_not_object")

    try:
        decision = DecisionV10.model_validate(value)
    except ValidationError:
        decision = None
    else:
        return {
            "decision": decision.model_dump(mode="json"),
            "parse_status": "clean",
            "repair_codes": [],
        }

    if set(value) == {"grade", "reason"} and type(value["reason"]) is str:
        normalized_reason = value["reason"].strip().lower()
        if normalized_reason != value["reason"]:
            repaired = {"grade": value["grade"], "reason": normalized_reason}
            try:
                decision = DecisionV10.model_validate(repaired)
            except ValidationError:
                pass
            else:
                return {
                    "decision": decision.model_dump(mode="json"),
                    "parse_status": "coerced",
                    "repair_codes": ["normalize_reason"],
                    "raw": raw,
                }
    return _v10_unscorable(raw, "invalid_decision")


def _v10_unscorable(raw: str, code: str) -> dict[str, object]:
    return {
        "decision": {"grade": None, "reason": None},
        "parse_status": "unscorable",
        "repair_codes": [code],
        "raw": raw,
    }


def derive_v11_label(decision: DecisionV11 | Mapping[str, object]) -> str:
    """Map one valid v11 decision to its single deterministic label."""
    value: Mapping[str, object]
    if isinstance(decision, DecisionV11):
        value = decision.model_dump(mode="json")
    else:
        value = decision
    validated = DecisionV11.model_validate(value)
    if not validated.answer_present:
        return "nonanswer"
    if validated.supported_core:
        return "partial" if validated.material_conflict else "correct"
    return "incorrect" if validated.material_conflict else "not_established"


def normalize_judge_decision_v11(raw: str) -> dict[str, object]:
    """Validate v11 output, stripping only a central claim's outer whitespace."""
    if type(raw) is not str:
        raise TypeError("v11 raw decision must be a string")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except _DuplicateJsonKeyError:
        return _v11_unscorable(raw, "duplicate_json_key")
    except (json.JSONDecodeError, ValueError):
        return _v11_unscorable(raw, "invalid_json")
    if type(value) is not dict:
        return _v11_unscorable(raw, "json_not_object")

    try:
        decision = DecisionV11.model_validate(value)
    except ValidationError:
        decision = None
    else:
        return {
            "decision": decision.model_dump(mode="json"),
            "label": derive_v11_label(decision),
            "parse_status": "clean",
            "repair_codes": [],
        }

    if set(value) == {
        "central_claim",
        "answer_present",
        "supported_core",
        "material_conflict",
    } and type(value["central_claim"]) is str:
        central_claim = value["central_claim"].strip()
        if central_claim != value["central_claim"]:
            repaired = {**value, "central_claim": central_claim}
            try:
                decision = DecisionV11.model_validate(repaired)
            except ValidationError:
                pass
            else:
                return {
                    "decision": decision.model_dump(mode="json"),
                    "label": derive_v11_label(decision),
                    "parse_status": "coerced",
                    "repair_codes": ["strip_central_claim"],
                    "raw": raw,
                }
    return _v11_unscorable(raw, "invalid_decision")


def _v11_unscorable(raw: str, code: str) -> dict[str, object]:
    return {
        "decision": {
            "central_claim": None,
            "answer_present": None,
            "supported_core": None,
            "material_conflict": None,
        },
        "label": None,
        "parse_status": "unscorable",
        "repair_codes": [code],
        "raw": raw,
    }


def _normalize_reference_grounded_decision(
    raw: str,
    *,
    version: str,
    decision_type: type[DecisionV8] | type[DecisionV9],
    enums: dict[str, frozenset[str]],
) -> dict[str, object]:
    """Normalize one compact reference-grounded decision version."""
    if type(raw) is not str:
        raise TypeError(f"{version} raw decision must be a string")
    value, extraction_code = _decode_v3_object(raw)
    if value is None:
        return _v8_unscorable(raw, [extraction_code or "invalid_json"])
    if type(value) is not dict:
        return _v8_unscorable(raw, ["json_not_object"])

    try:
        decision = decision_type.model_validate(value)
    except ValidationError:
        decision = None
    else:
        return _v8_result(
            decision,
            parse_status="coerced" if extraction_code else "clean",
            repair_codes=[extraction_code] if extraction_code else [],
            raw=raw if extraction_code else None,
        )

    repaired: dict[str, object] = {}
    repair_codes: list[str] = []
    if extraction_code:
        repair_codes.append(extraction_code)
    if set(value) - _V8_FIELDS:
        repair_codes.append("drop_unknown_keys")
    for field, allowed in enums.items():
        item = value.get(field)
        normalized = item.strip().lower() if type(item) is str else ""
        if normalized not in allowed:
            repair_codes.append(f"missing_or_invalid_{field}")
            return _v8_unscorable(raw, repair_codes)
        repaired[field] = normalized
        if normalized != item:
            repair_codes.append(f"normalize_{field}")

    rationale = value.get("rationale")
    if type(rationale) is not str:
        repair_codes.append("missing_or_invalid_rationale")
        return _v8_unscorable(raw, repair_codes)
    if len(rationale) > 240:
        rationale = rationale[:240]
        repair_codes.append("truncate_rationale")
    repaired["rationale"] = rationale

    try:
        decision = decision_type.model_validate(repaired)
    except ValidationError:
        return _v8_unscorable(raw, [*repair_codes, "coercion_failed"])
    return _v8_result(
        decision,
        parse_status="coerced",
        repair_codes=_dedupe_preserving_order(repair_codes),
        raw=raw,
    )


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
    protocol_dir: Path,
    version: Literal[
        "v1", "v2", "v3", "v4", "v5", "v6", "v8", "v9", "v10", "v11"
    ] = "v1",
) -> PromptContract:
    """Load the versioned prompt and JSON schema without altering either."""
    versioned_paths = {
        "v1": ("pointwise-v1.txt", "decision-v1.schema.json"),
        "v2": ("pointwise-v2.txt", "decision-v2.schema.json"),
        "v3": ("pointwise-v3.txt", "decision-v3.schema.json"),
        "v4": ("pointwise-v4.txt", "decision-v4.schema.json"),
        "v5": ("pointwise-v5.txt", "decision-v5.schema.json"),
        "v6": ("pointwise-v6.txt", "decision-v6.schema.json"),
        "v8": ("pointwise-v8.txt", "decision-v8.schema.json"),
        "v9": ("pointwise-v9.txt", "decision-v9.schema.json"),
        "v10": ("pointwise-v10.txt", "decision-v10.schema.json"),
        "v11": ("pointwise-v11.txt", "decision-v11.schema.json"),
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
    except _DuplicateJsonKeyError as error:
        if version == "v2":
            raise ValueError("decision contains a duplicate JSON key") from error
        raise ValueError(f"duplicate JSON key in decision: {error.key}") from error
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
        location = _sanitized_v2_error_location(detail.get("loc"))
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


def _sanitized_v2_error_location(value: object) -> str | None:
    if type(value) not in {list, tuple} or not value:
        return None
    first, *rest = value
    if type(first) is not str or first not in DECISION_KEYS:
        return "unknown field"
    if any(type(item) is not int or item < 0 for item in rest):
        return "unknown field"
    return ".".join([first, *(str(item) for item in rest)])


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
            raise _DuplicateJsonKeyError(key)
        value[key] = item
    return value
