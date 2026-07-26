"""Validated private inputs and identity-blind diagnostic smoke selection."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
import bisect
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from trec_mlm.pointwise_judge import CandidateCase
from trec_mlm.reference_sampling import question_strata


DIAGNOSTIC_REASONS = (
    "high_overlap_control",
    "short_incomplete_proxy",
    "low_overlap_error_proxy",
    "entity_mismatch_proxy",
    "number_or_date_mismatch_proxy",
    "negation_mismatch_proxy",
    "verbose_error_proxy",
    "refusal_proxy",
    "underspecified_proxy",
    "possible_reference_conflict_proxy",
)

_QUERY_RE = re.compile(r'^  "([^"]+)": \{$')
_QUERY_TEXT_RE = re.compile(r'^    "query": (.+)$')
_RESPONSE_RE = re.compile(r'^      "([^"]+)": (.+)$')
_TOKEN_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_NUMBER_OR_DATE_RE = re.compile(r"\b\d{1,4}(?:[.,]\d+)?\b")
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|neither|nor|without|cannot|can't|isn't|wasn't|didn't)\b", re.I
)
_REFUSAL_RE = re.compile(
    r"\b(?:i (?:do not|don't) know|cannot answer|can't answer|unable to answer)\b", re.I
)
_CONFLICT_RE = re.compile(
    r"\b(?:premise|reference|actually|however|instead|contrary|wrong|incorrect)\b", re.I
)
_ENTITY_RE = re.compile(r"\b[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)*\b")
_ELIGIBLE_POOL_LIMIT = 64


@dataclass(frozen=True, slots=True)
class GroundedReference:
    question_id: str
    question: str
    answer: str


@dataclass(frozen=True, slots=True)
class CandidateResponse:
    question_id: str
    candidate_id: str
    answer: str


@dataclass(frozen=True, slots=True)
class _CandidateFeatures:
    response: CandidateResponse
    overlap: float
    word_count: int
    reference_word_count: int
    entity_difference: int
    has_reference_entities: bool
    has_candidate_entities: bool
    number_difference: int
    has_reference_numbers: bool
    has_candidate_numbers: bool
    negation_mismatch: bool
    refusal: bool
    underspecified: bool
    possible_reference_conflict: bool


_RankedCandidate = tuple[tuple[float, ...], str, str, str, CandidateResponse]


def load_grounded_references(
    path: Path, *, questions: Mapping[str, str] | None = None
) -> dict[str, GroundedReference]:
    """Load complete successful records from a preserved generation cache."""
    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except json.JSONDecodeError as error:
        raise ValueError("grounded cache is not valid JSON") from error
    if type(payload) is not dict or type(payload.get("records")) is not dict:
        raise ValueError("grounded cache must contain a records object")

    raw_records: dict[str, Any] = payload["records"]
    if not raw_records:
        raise ValueError("grounded cache records must not be empty")
    if ("provider" in payload or "model" in payload) and len(raw_records) != 7_759:
        raise ValueError("real grounded cache must contain exactly 7,759 records")

    result: dict[str, GroundedReference] = {}
    for record_key, raw_record in raw_records.items():
        if type(record_key) is not str or not record_key.strip():
            raise ValueError("grounded record ID must be nonempty")
        if type(raw_record) is not dict:
            raise ValueError(f"grounded record {record_key} must be an object")
        question_id = _nonempty_string(raw_record, "question_id", record_key)
        if question_id != record_key:
            raise ValueError(f"grounded record ID mismatch for {record_key}")
        raw_question = raw_record.get("question")
        mapped_question = questions.get(question_id) if questions is not None else None
        if raw_question is None:
            question = _validated_question(mapped_question, record_key)
        else:
            question = _validated_question(raw_question, record_key)
            if mapped_question is not None and question != _validated_question(
                mapped_question, record_key
            ):
                raise ValueError(
                    f"grounded/discovery question mismatch for {record_key}"
                )
        answer = _nonempty_string(raw_record, "answer", record_key)
        if raw_record.get("finish_reason") != "stop":
            raise ValueError(
                f"grounded record {record_key} has unsuccessful finish_reason"
            )
        if question_id in result:
            raise ValueError(f"duplicate grounded question ID: {question_id}")
        result[question_id] = GroundedReference(question_id, question, answer)
    return result


def load_discovery_questions(path: Path) -> dict[str, str]:
    """Validate and load only question IDs/text from the discovery source."""
    questions: dict[str, str] = {}
    query_id: str | None = None
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\r\n")
            match = _QUERY_RE.match(line)
            if match:
                if query_id is not None and query_id not in questions:
                    raise ValueError(f"discovery question {query_id} has no query text")
                query_id = match.group(1)
                if not query_id.strip():
                    raise ValueError(
                        f"empty discovery question ID at line {line_number}"
                    )
                if query_id in questions:
                    raise ValueError(f"duplicate discovery question ID: {query_id}")
                continue
            match = _QUERY_TEXT_RE.match(line)
            if match and query_id is not None:
                if query_id in questions:
                    raise ValueError(f"duplicate discovery query text for {query_id}")
                questions[query_id] = _decode_nonempty_json_string(
                    match.group(1), line_number
                )
    if query_id is not None and query_id not in questions:
        raise ValueError(f"discovery question {query_id} has no query text")
    if not questions:
        raise ValueError("no discovery questions found")
    return questions


def iter_discovery_responses(path: Path) -> Iterator[CandidateResponse]:
    """Stream candidates from the official pretty-printed discovery JSON."""
    query_id: str | None = None
    query_seen = False
    found = False
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\r\n")
            match = _QUERY_RE.match(line)
            if match:
                query_id = match.group(1)
                query_seen = False
                if not query_id.strip():
                    raise ValueError(
                        f"empty discovery question ID at line {line_number}"
                    )
                continue
            match = _QUERY_TEXT_RE.match(line)
            if match and query_id is not None:
                query = _decode_nonempty_json_string(match.group(1), line_number)
                query_seen = bool(query)
                continue
            match = _RESPONSE_RE.match(line)
            if not match or query_id is None:
                continue
            if not query_seen:
                raise ValueError(
                    f"discovery response before query text at line {line_number}"
                )
            candidate_id = match.group(1)
            if not candidate_id.strip():
                raise ValueError(f"empty discovery candidate ID at line {line_number}")
            answer = _decode_nonempty_json_string(match.group(2), line_number)
            found = True
            yield CandidateResponse(query_id, candidate_id, answer)
    if not found:
        raise ValueError("no discovery responses found")


def select_diagnostic_cases(
    references: Mapping[str, GroundedReference],
    responses: Iterable[CandidateResponse],
    *,
    seed: int = 2026,
) -> tuple[list[CandidateCase], dict[str, str]]:
    """Select one identity-blind case for each stable diagnostic proxy."""
    pools: dict[str, list[_RankedCandidate]] = {
        reason: [] for reason in DIAGNOSTIC_REASONS
    }
    seen: set[tuple[str, str]] = set()
    for response in responses:
        if response.question_id not in references:
            raise ValueError(f"missing grounded reference for {response.question_id}")
        key = (response.question_id, response.candidate_id)
        if key in seen:
            continue
        seen.add(key)
        features = _candidate_features(references[response.question_id], response)
        for reason, pool in pools.items():
            if not _eligible(reason, features):
                continue
            ranked = (
                _proxy_rank(reason, features),
                _stable_tiebreak(seed, response.question_id, response.candidate_id),
                response.question_id,
                response.candidate_id,
                response,
            )
            _insert_bounded(pool, ranked)
    if len(seen) < len(DIAGNOSTIC_REASONS):
        raise ValueError("at least 10 unique question/candidate pairs are required")

    for reason, pool in pools.items():
        if not pool:
            raise ValueError(f"no eligible candidate for {reason}")
    assignments = _distinct_assignments(pools)
    if assignments is None:
        raise ValueError("eligible pools do not contain 10 distinct candidates")

    cases: list[CandidateCase] = []
    sealed: dict[str, str] = {}
    for reason in DIAGNOSTIC_REASONS:
        response = assignments[reason]
        reference = references[response.question_id]
        response_id = _response_id(response)
        sealed[response_id] = response.candidate_id
        cases.append(
            CandidateCase(
                question_id=response.question_id,
                response_id=response_id,
                question=reference.question,
                reference_answer=reference.answer,
                candidate_answer=response.answer,
                strata=tuple(sorted({*question_strata(reference.question), reason})),
                selection_reason=reason,
            )
        )
    return cases, sealed


def _insert_bounded(pool: list[_RankedCandidate], item: _RankedCandidate) -> None:
    bisect.insort(pool, item)
    del pool[_ELIGIBLE_POOL_LIMIT:]


def _distinct_assignments(
    pools: Mapping[str, list[_RankedCandidate]],
) -> dict[str, CandidateResponse] | None:
    candidate_to_reason: dict[tuple[str, str], str] = {}
    assigned: dict[str, CandidateResponse] = {}

    def assign(reason: str, visited: set[tuple[str, str]]) -> bool:
        for *_, response in pools[reason]:
            key = (response.question_id, response.candidate_id)
            if key in visited:
                continue
            visited.add(key)
            previous_reason = candidate_to_reason.get(key)
            if previous_reason is None or assign(previous_reason, visited):
                candidate_to_reason[key] = reason
                assigned[reason] = response
                return True
        return False

    for reason in DIAGNOSTIC_REASONS:
        if not assign(reason, set()):
            return None
    return assigned


def _nonempty_string(record: Mapping[str, object], field: str, record_id: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value.strip():
        raise ValueError(f"grounded record {record_id} requires nonempty {field}")
    return value


def _validated_question(value: object, record_id: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"grounded record {record_id} requires nonempty question")
    return value


def _object_without_duplicate_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _decode_nonempty_json_string(payload: str, line_number: int) -> str:
    try:
        value = json.loads(payload.rstrip().removesuffix(","))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"invalid discovery JSON string at line {line_number}"
        ) from error
    if type(value) is not str or not value.strip():
        raise ValueError(f"empty discovery string at line {line_number}")
    return value


def _tokens(value: str) -> set[str]:
    return {token.casefold() for token in _TOKEN_RE.findall(value)}


def _overlap(reference: str, candidate: str) -> float:
    reference_tokens = _tokens(reference)
    candidate_tokens = _tokens(candidate)
    union = reference_tokens | candidate_tokens
    return len(reference_tokens & candidate_tokens) / len(union) if union else 0.0


def _candidate_features(
    reference: GroundedReference, response: CandidateResponse
) -> _CandidateFeatures:
    candidate = response.answer
    overlap = _overlap(reference.answer, candidate)
    word_count = len(_TOKEN_RE.findall(candidate))
    reference_entities = {
        item.casefold() for item in _ENTITY_RE.findall(reference.answer)
    }
    candidate_entities = {item.casefold() for item in _ENTITY_RE.findall(candidate)}
    reference_numbers = set(_NUMBER_OR_DATE_RE.findall(reference.answer))
    candidate_numbers = set(_NUMBER_OR_DATE_RE.findall(candidate))
    return _CandidateFeatures(
        response=response,
        overlap=overlap,
        word_count=word_count,
        reference_word_count=len(_TOKEN_RE.findall(reference.answer)),
        entity_difference=len(reference_entities ^ candidate_entities),
        has_reference_entities=bool(reference_entities),
        has_candidate_entities=bool(candidate_entities),
        number_difference=len(reference_numbers ^ candidate_numbers),
        has_reference_numbers=bool(reference_numbers),
        has_candidate_numbers=bool(candidate_numbers),
        negation_mismatch=(
            bool(_NEGATION_RE.search(reference.answer))
            != bool(_NEGATION_RE.search(candidate))
        ),
        refusal=bool(_REFUSAL_RE.search(candidate)),
        underspecified="underspecified" in question_strata(reference.question),
        possible_reference_conflict=bool(_CONFLICT_RE.search(candidate)),
    )


def _eligible(reason: str, features: _CandidateFeatures) -> bool:
    checks = {
        "high_overlap_control": features.overlap >= 0.8
        and not features.refusal
        and not features.negation_mismatch
        and features.number_difference == 0
        and features.entity_difference == 0,
        "short_incomplete_proxy": features.word_count < features.reference_word_count
        and features.overlap >= 0.2
        and not features.refusal,
        "low_overlap_error_proxy": features.overlap <= 0.2 and not features.refusal,
        "entity_mismatch_proxy": features.has_reference_entities
        and features.has_candidate_entities
        and features.entity_difference > 0
        and features.overlap >= 0.2,
        "number_or_date_mismatch_proxy": features.has_reference_numbers
        and features.has_candidate_numbers
        and features.number_difference > 0,
        "negation_mismatch_proxy": features.negation_mismatch,
        "verbose_error_proxy": features.word_count
        >= max(30, features.reference_word_count * 3),
        "refusal_proxy": features.refusal,
        "underspecified_proxy": features.underspecified,
        "possible_reference_conflict_proxy": features.possible_reference_conflict,
    }
    return checks[reason]


def _proxy_rank(reason: str, features: _CandidateFeatures) -> tuple[float, ...]:
    ranks: dict[str, tuple[float, ...]] = {
        "high_overlap_control": (-features.overlap,),
        "short_incomplete_proxy": (features.word_count, -features.overlap),
        "low_overlap_error_proxy": (features.overlap,),
        "entity_mismatch_proxy": (-float(features.entity_difference), features.overlap),
        "number_or_date_mismatch_proxy": (
            -float(features.number_difference),
            features.overlap,
        ),
        "negation_mismatch_proxy": (features.overlap,),
        "verbose_error_proxy": (-float(features.word_count), features.overlap),
        "refusal_proxy": (features.word_count,),
        "underspecified_proxy": (features.word_count,),
        "possible_reference_conflict_proxy": (features.overlap,),
    }
    return ranks[reason]


def _stable_tiebreak(seed: int, question_id: str, candidate_id: str) -> str:
    payload = json.dumps(
        [seed, question_id, candidate_id], separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _response_id(response: CandidateResponse) -> str:
    payload = json.dumps(
        [response.question_id, response.answer, response.candidate_id],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
