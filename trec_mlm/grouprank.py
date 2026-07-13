"""Pure prompt and grouping contract for the Stage 2 GroupRank pilot."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any

from trec_mlm.llm_ranker import SYSTEM_PROMPT, _capability


GROUPING_SEEDS = (666, 667)
_QUESTION_COUNT = 12
_CANDIDATE_COUNT = 20
_GROUP_SIZE = 10


@dataclass(frozen=True)
class GroupRankRequest:
    """One deterministic ten-expert GroupRank comparison request."""

    query_id: str
    repetition: int
    group: int
    ordered_candidate_ids: tuple[str, ...]
    handle_to_candidate: Mapping[str, str]
    messages: tuple[Mapping[str, str], ...]
    sampling_seed: int


def request_seed(query_id: str, repetition: int, group: int) -> int:
    """Derive the frozen model-sampling seed for one request."""

    digest = hashlib.sha256(
        f"2026|{query_id}|{repetition}|{group}".encode()
    ).digest()
    return int.from_bytes(digest[:4], "big")


def format_grouprank_messages(
    question: str,
    *,
    ordered_candidate_ids: Sequence[str],
    handle_to_candidate: Mapping[str, str],
) -> tuple[Mapping[str, str], ...]:
    """Format one expert-oriented GroupRank prompt without evaluation data."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be non-empty")
    candidates = tuple(ordered_candidate_ids)
    if (
        len(candidates) != _GROUP_SIZE
        or len(set(candidates)) != _GROUP_SIZE
        or not all(isinstance(candidate, str) and candidate for candidate in candidates)
    ):
        raise ValueError("GroupRank prompts require 10 unique candidate IDs")
    if (
        len(handle_to_candidate) != _GROUP_SIZE
        or len(set(handle_to_candidate)) != _GROUP_SIZE
        or set(handle_to_candidate.values()) != set(candidates)
    ):
        raise ValueError("GroupRank handles and candidate order must agree")
    candidate_to_handle = {
        candidate: handle for handle, candidate in handle_to_candidate.items()
    }
    rows = [
        f"{candidate_to_handle[candidate]} | {candidate} | {_capability(candidate)}"
        for candidate in candidates
    ]
    user = (
        "Compare all described LLM experts for the user question. Assign every "
        "handle one integer score from 0 through 10; higher scores mean the expert "
        "is more likely to answer the question correctly. You may explain the "
        "comparison first. Finish with exactly one JSON object containing every "
        "handle inside one final <answer> block.\n\n"
        f"<question>\n{question.strip()}\n</question>\n"
        "<experts>\n"
        + "\n".join(rows)
        + "\n</experts>"
    )
    return (
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    )


def _selection_questions(selection: Mapping[str, Any]) -> Sequence[Mapping[str, Any]]:
    if not isinstance(selection, Mapping) or selection.get("stage") != 2:
        raise ValueError("GroupRank requires a Stage 2 selection")
    questions = selection.get("questions")
    if (
        not isinstance(questions, Sequence)
        or isinstance(questions, (str, bytes))
        or len(questions) != _QUESTION_COUNT
        or not all(isinstance(entry, Mapping) for entry in questions)
    ):
        raise ValueError("Stage 2 GroupRank requires exactly 12 questions")
    return questions


def _question_inventory(
    entry: Mapping[str, Any],
) -> tuple[str, str, tuple[str, ...]]:
    query_id = entry.get("query_id")
    question = entry.get("question")
    candidate_ids = entry.get("candidate_ids")
    if not isinstance(query_id, str) or not query_id:
        raise ValueError("Stage 2 questions require non-empty query IDs")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Stage 2 questions require non-empty text")
    if (
        not isinstance(candidate_ids, Sequence)
        or isinstance(candidate_ids, (str, bytes))
        or len(candidate_ids) != _CANDIDATE_COUNT
        or not all(
            isinstance(candidate_id, str) and candidate_id
            for candidate_id in candidate_ids
        )
    ):
        raise ValueError("Stage 2 questions require exactly 20 candidate IDs")
    candidates = tuple(candidate_ids)
    if len(set(candidates)) != _CANDIDATE_COUNT:
        raise ValueError("Stage 2 questions require 20 unique candidate IDs")
    return query_id, question, candidates


def build_group_requests(
    selection: Mapping[str, Any],
) -> tuple[GroupRankRequest, ...]:
    """Build the frozen 48-request Stage 2 GroupRank inventory."""

    inventories = tuple(
        _question_inventory(entry) for entry in _selection_questions(selection)
    )
    query_ids = tuple(query_id for query_id, _, _ in inventories)
    if len(set(query_ids)) != _QUESTION_COUNT:
        raise ValueError("Stage 2 GroupRank requires 12 unique query IDs")

    requests: list[GroupRankRequest] = []
    for query_id, question, candidate_ids in inventories:
        handles = {
            f"C{index:02d}": candidate
            for index, candidate in enumerate(sorted(candidate_ids))
        }
        for grouping_seed in GROUPING_SEEDS:
            ordered = tuple(
                sorted(
                    candidate_ids,
                    key=lambda candidate: hashlib.sha256(
                        f"{grouping_seed}|{query_id}|{candidate}".encode()
                    ).digest(),
                )
            )
            for group, offset in enumerate((0, _GROUP_SIZE)):
                group_candidates = ordered[offset : offset + _GROUP_SIZE]
                group_inventory = set(group_candidates)
                group_handles = {
                    handle: candidate
                    for handle, candidate in handles.items()
                    if candidate in group_inventory
                }
                requests.append(
                    GroupRankRequest(
                        query_id=query_id,
                        repetition=grouping_seed,
                        group=group,
                        ordered_candidate_ids=group_candidates,
                        handle_to_candidate=group_handles,
                        messages=format_grouprank_messages(
                            question,
                            ordered_candidate_ids=group_candidates,
                            handle_to_candidate=group_handles,
                        ),
                        sampling_seed=request_seed(query_id, grouping_seed, group),
                    )
                )
    return tuple(requests)


def _validate_expected_handles(expected_handles: Sequence[str]) -> tuple[str, ...]:
    handles = tuple(expected_handles)
    if (
        len(handles) != _GROUP_SIZE
        or len(set(handles)) != _GROUP_SIZE
        or not all(isinstance(handle, str) and handle for handle in handles)
    ):
        raise ValueError("GroupRank requires exactly 10 unique expected handles")
    return handles


def grouprank_score_schema(handles: Sequence[str]) -> dict[str, Any]:
    """Return the deterministic fixed-key schema for one GroupRank answer."""

    expected = _validate_expected_handles(handles)
    return {
        "type": "object",
        "properties": {
            handle: {"type": "integer", "minimum": 0, "maximum": 10}
            for handle in expected
        },
        "required": list(expected),
        "additionalProperties": False,
    }


def parse_grouprank_scores(
    raw: str,
    *,
    expected_handles: Sequence[str],
) -> dict[str, int]:
    """Parse one strict ten-key score object from the final answer block."""

    expected = _validate_expected_handles(expected_handles)
    if (
        not isinstance(raw, str)
        or raw.count("<answer>") != 1
        or raw.count("</answer>") != 1
    ):
        raise ValueError("GroupRank output must contain one final <answer> block")
    match = re.fullmatch(
        r".*<answer>\s*(.*?)\s*</answer>\s*",
        raw,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError("GroupRank output must contain one final <answer> block")
    try:
        pairs = json.loads(match.group(1), object_pairs_hook=lambda values: values)
    except json.JSONDecodeError as error:
        raise ValueError("GroupRank answer must be valid JSON") from error
    if not isinstance(pairs, list) or any(
        not isinstance(pair, tuple) for pair in pairs
    ):
        raise ValueError("GroupRank answer must be a JSON object")
    keys = [key for key, _ in pairs]
    if len(keys) != len(set(keys)):
        raise ValueError("GroupRank answer contains duplicate handles")
    if set(keys) != set(expected):
        raise ValueError("GroupRank answer has missing or extra handles")
    result: dict[str, int] = {}
    for key, value in pairs:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value <= 10
        ):
            raise ValueError("GroupRank scores must be integers from 0 through 10")
        result[key] = value
    return result


def aggregate_grouprank_scores(
    requests: Sequence[GroupRankRequest],
    parsed_scores: Sequence[Mapping[str, int]],
) -> tuple[dict[str, dict[str, float]], dict[str, list[str]]]:
    """Average two groupwise observations and rank candidates deterministically."""

    if len(requests) != len(parsed_scores):
        raise ValueError("GroupRank aggregation requires one score set per request")
    if len(requests) != _QUESTION_COUNT * len(GROUPING_SEEDS) * 2:
        raise ValueError("GroupRank aggregation requires exactly 48 requests")

    request_keys = {
        (request.query_id, request.repetition, request.group) for request in requests
    }
    if len(request_keys) != len(requests):
        raise ValueError("GroupRank aggregation requires unique requests")

    observations: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )
    query_requests: dict[str, set[tuple[int, int]]] = defaultdict(set)
    repetition_candidates: dict[tuple[str, int], list[str]] = defaultdict(list)
    for request, scores in zip(requests, parsed_scores, strict=True):
        if set(scores) != set(request.handle_to_candidate):
            raise ValueError("GroupRank score handles disagree with the request")
        query_requests[request.query_id].add((request.repetition, request.group))
        for handle, candidate in request.handle_to_candidate.items():
            repetition_candidates[(request.query_id, request.repetition)].append(
                candidate
            )
            value = scores[handle]
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 10
            ):
                raise ValueError(
                    "GroupRank scores must be integers from 0 through 10"
                )
            observations[request.query_id][candidate].append(value)

    expected_request_keys = {
        (repetition, group)
        for repetition in GROUPING_SEEDS
        for group in (0, 1)
    }
    if len(query_requests) != _QUESTION_COUNT or any(
        keys != expected_request_keys for keys in query_requests.values()
    ):
        raise ValueError("GroupRank aggregation requires four requests per query")
    for query_id in query_requests:
        candidate_sets: list[set[str]] = []
        for repetition in GROUPING_SEEDS:
            candidates = repetition_candidates[(query_id, repetition)]
            candidate_set = set(candidates)
            if (
                len(candidates) != _CANDIDATE_COUNT
                or len(candidate_set) != _CANDIDATE_COUNT
            ):
                raise ValueError(
                    "GroupRank aggregation requires each candidate exactly once "
                    "per repetition"
                )
            candidate_sets.append(candidate_set)
        if candidate_sets[0] != candidate_sets[1]:
            raise ValueError(
                "GroupRank aggregation requires each candidate exactly once per "
                "repetition"
            )

    candidate_scores: dict[str, dict[str, float]] = {}
    rankings: dict[str, list[str]] = {}
    for query_id in query_requests:
        candidate_observations = observations[query_id]
        if len(candidate_observations) != _CANDIDATE_COUNT or any(
            len(values) != len(GROUPING_SEEDS)
            for values in candidate_observations.values()
        ):
            raise ValueError(
                "GroupRank aggregation requires two observations per candidate"
            )
        scores = {
            candidate: sum(values) / len(values)
            for candidate, values in candidate_observations.items()
        }
        candidate_scores[query_id] = scores
        rankings[query_id] = sorted(
            scores,
            key=lambda candidate: (-scores[candidate], candidate),
        )
    return candidate_scores, rankings
