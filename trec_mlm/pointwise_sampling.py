"""Validated private inputs and identity-blind diagnostic smoke selection."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
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
_NUMBER_OR_DATE_RE = re.compile(r"\b(?:\d{1,4}(?:[.,]\d+)?|(?:19|20)\d{2})\b")
_NEGATION_RE = re.compile(r"\b(?:no|not|never|neither|nor|without|cannot|can't|isn't|wasn't|didn't)\b", re.I)
_REFUSAL_RE = re.compile(r"\b(?:i (?:do not|don't) know|cannot answer|can't answer|unable to answer)\b", re.I)
_CONFLICT_RE = re.compile(r"\b(?:premise|reference|actually|however|instead|contrary|wrong|incorrect)\b", re.I)
_UNDERSPECIFIED_RE = re.compile(r"\b(?:it|they|this|that|something|someone|somewhere)\b", re.I)
_ENTITY_RE = re.compile(r"\b[A-Z][\w'-]+(?:\s+[A-Z][\w'-]+)*\b")


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


def load_grounded_references(path: Path) -> dict[str, GroundedReference]:
    """Load complete successful records from a preserved generation cache."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
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
        question = _nonempty_string(raw_record, "question", record_key)
        answer = _nonempty_string(raw_record, "answer", record_key)
        if raw_record.get("finish_reason") != "stop":
            raise ValueError(f"grounded record {record_key} has unsuccessful finish_reason")
        if question_id in result:
            raise ValueError(f"duplicate grounded question ID: {question_id}")
        result[question_id] = GroundedReference(question_id, question, answer)
    return result


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
                    raise ValueError(f"empty discovery question ID at line {line_number}")
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
                raise ValueError(f"discovery response before query text at line {line_number}")
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
    # Ten entries per reason guarantee an unused entry for the tenth greedy pick,
    # while avoiding retention of the million-cell answer corpus.
    pools: dict[str, list[CandidateResponse]] = {
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
        for reason, pool in pools.items():
            pool.append(response)
            pool.sort(key=lambda row: _candidate_rank(reason, row, references, seed))
            del pool[len(DIAGNOSTIC_REASONS) :]
    if len(seen) < len(DIAGNOSTIC_REASONS):
        raise ValueError("at least 10 unique question/candidate pairs are required")

    chosen: set[tuple[str, str]] = set()
    cases: list[CandidateCase] = []
    sealed: dict[str, str] = {}
    for reason in DIAGNOSTIC_REASONS:
        response = next(
            row
            for row in pools[reason]
            if (row.question_id, row.candidate_id) not in chosen
        )
        reference = references[response.question_id]
        response_id = _response_id(response)
        chosen.add((response.question_id, response.candidate_id))
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


def _candidate_rank(
    reason: str,
    response: CandidateResponse,
    references: Mapping[str, GroundedReference],
    seed: int,
) -> tuple[tuple[float, ...], str]:
    return (
        _proxy_rank(reason, references[response.question_id], response),
        _stable_tiebreak(seed, response.question_id, response.candidate_id),
    )


def _nonempty_string(record: Mapping[str, object], field: str, record_id: str) -> str:
    value = record.get(field)
    if type(value) is not str or not value.strip():
        raise ValueError(f"grounded record {record_id} requires nonempty {field}")
    return value


def _decode_nonempty_json_string(payload: str, line_number: int) -> str:
    try:
        value = json.loads(payload.rstrip().removesuffix(","))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid discovery JSON string at line {line_number}") from error
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


def _proxy_rank(
    reason: str, reference: GroundedReference, response: CandidateResponse
) -> tuple[float, ...]:
    candidate = response.answer
    overlap = _overlap(reference.answer, candidate)
    word_count = len(_TOKEN_RE.findall(candidate))
    entity_difference = len(
        {item.casefold() for item in _ENTITY_RE.findall(reference.answer)}
        ^ {item.casefold() for item in _ENTITY_RE.findall(candidate)}
    )
    number_difference = len(
        set(_NUMBER_OR_DATE_RE.findall(reference.answer))
        ^ set(_NUMBER_OR_DATE_RE.findall(candidate))
    )
    ranks: dict[str, tuple[float, ...]] = {
        "high_overlap_control": (-overlap,),
        "short_incomplete_proxy": (word_count, -overlap),
        "low_overlap_error_proxy": (overlap,),
        "entity_mismatch_proxy": (-float(entity_difference), overlap),
        "number_or_date_mismatch_proxy": (-float(number_difference), overlap),
        "negation_mismatch_proxy": (
            -float(bool(_NEGATION_RE.search(reference.answer)) != bool(_NEGATION_RE.search(candidate))),
            overlap,
        ),
        "verbose_error_proxy": (-float(word_count), overlap),
        "refusal_proxy": (-float(bool(_REFUSAL_RE.search(candidate))), word_count),
        "underspecified_proxy": (
            -float(bool(_UNDERSPECIFIED_RE.search(reference.question)) or bool(_UNDERSPECIFIED_RE.search(candidate))),
            word_count,
        ),
        "possible_reference_conflict_proxy": (
            -float(bool(_CONFLICT_RE.search(candidate))),
            overlap,
        ),
    }
    return ranks[reason]


def _stable_tiebreak(seed: int, question_id: str, candidate_id: str) -> str:
    payload = json.dumps([seed, question_id, candidate_id], separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _response_id(response: CandidateResponse) -> str:
    payload = json.dumps(
        [response.question_id, response.answer, response.candidate_id],
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
