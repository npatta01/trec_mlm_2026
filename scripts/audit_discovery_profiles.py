#!/usr/bin/env python3
"""Stream the official discovery JSON files into a small profile-quality audit.

The official files are stable, pretty-printed JSON. This parser deliberately keeps
only counters and two worked candidate summaries in memory; it does not load either
large document as one Python object. If the organizers reserialize the files with a
different indentation layout, use a general streaming parser instead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median, quantiles
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAIN = ROOT / "data/dev/llm_discovery_data.json"
DEFAULT_METADATA = ROOT / "data/dev/llm_discovery_metadata.json"
DEFAULT_OUTPUT = ROOT / "reports/trec_2026_discovery_profile_audit.json"
TARGET_CANDIDATES = ("Anthropology", "Applied_mathematics")

QUERY_RE = re.compile(r'^  "([^"]+)": \{$')
PLAIN_RESPONSE_RE = re.compile(r'^      "([^"]+)": (.+)$')
METADATA_CANDIDATE_RE = re.compile(r'^      "([^"]+)": \{$')
METADATA_RESPONSE_RE = re.compile(r'^        "response": (.+)$')
WORD_RE = re.compile(r"\b\w+\b")
CONTEXT_TEMPLATE_RE = re.compile(
    r"\b(?:provided context|context provided|provided information|"
    r"information provided|context passages?)\b",
    re.IGNORECASE,
)
INSUFFICIENCY_RE = re.compile(
    r"(?:i don['’]?t know|i do not know|not enough information|"
    r"insufficient information|does not provide|do not provide|"
    r"cannot determine|can['’]?t determine|unable to determine|"
    r"no information|not mentioned|not specified|"
    r"not directly (?:addressed|stated)|cannot be definitively)",
    re.IGNORECASE,
)


def normalize_response(response: str) -> str:
    return re.sub(r"[^a-z]+", " ", response.lower()).strip()


def is_exact_refusal(response: str) -> bool:
    return normalize_response(response) in {"i don t know", "i do not know"}


def decode_json_scalar(payload: str) -> str:
    payload = payload.rstrip()
    if payload.endswith(","):
        payload = payload[:-1]
    value = json.loads(payload)
    if not isinstance(value, str):
        raise ValueError("expected a JSON string scalar")
    return value


def update_tuple_digest(digest: Any, values: Iterable[str]) -> None:
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def empty_candidate_stats() -> dict[str, Any]:
    return {
        "records": 0,
        "exact_refusals": 0,
        "attempt_word_counts": [],
        "attempts_with_insufficiency_phrase": 0,
        "attempts_with_context_template": 0,
    }


def close_plain_query(
    query_id: str | None,
    seen_candidates: set[str],
    non_refusal_count: int,
    reference_candidates: set[str] | None,
    non_refusals_per_query: list[int],
) -> set[str] | None:
    if query_id is None:
        return reference_candidates
    if len(seen_candidates) != 150:
        raise ValueError(f"query {query_id} has {len(seen_candidates)} candidates, expected 150")
    if reference_candidates is None:
        reference_candidates = set(seen_candidates)
    elif seen_candidates != reference_candidates:
        raise ValueError(f"query {query_id} has a different candidate set")
    non_refusals_per_query.append(non_refusal_count)
    return reference_candidates


def audit_plain(path: Path) -> dict[str, Any]:
    query_id: str | None = None
    seen_candidates: set[str] = set()
    reference_candidates: set[str] | None = None
    non_refusal_count = 0
    non_refusals_per_query: list[int] = []
    response_count = 0
    exact_refusals = 0
    tuple_digest = hashlib.sha256()
    candidate_stats = {candidate: empty_candidate_stats() for candidate in TARGET_CANDIDATES}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.rstrip("\n")
            query_match = QUERY_RE.match(raw)
            if query_match:
                reference_candidates = close_plain_query(
                    query_id,
                    seen_candidates,
                    non_refusal_count,
                    reference_candidates,
                    non_refusals_per_query,
                )
                query_id = query_match.group(1)
                seen_candidates = set()
                non_refusal_count = 0
                continue

            response_match = PLAIN_RESPONSE_RE.match(raw)
            if not response_match or query_id is None:
                continue
            candidate = response_match.group(1)
            if candidate in seen_candidates:
                raise ValueError(f"duplicate candidate {candidate} at line {line_number}")
            response = decode_json_scalar(response_match.group(2))
            seen_candidates.add(candidate)
            response_count += 1
            update_tuple_digest(tuple_digest, (query_id, candidate, response))

            exact = is_exact_refusal(response)
            if exact:
                exact_refusals += 1
            else:
                non_refusal_count += 1

            if candidate in candidate_stats:
                stats = candidate_stats[candidate]
                stats["records"] += 1
                if exact:
                    stats["exact_refusals"] += 1
                else:
                    stats["attempt_word_counts"].append(len(WORD_RE.findall(response)))
                    if INSUFFICIENCY_RE.search(response):
                        stats["attempts_with_insufficiency_phrase"] += 1
                    if CONTEXT_TEMPLATE_RE.search(response):
                        stats["attempts_with_context_template"] += 1

    reference_candidates = close_plain_query(
        query_id,
        seen_candidates,
        non_refusal_count,
        reference_candidates,
        non_refusals_per_query,
    )
    if reference_candidates is None:
        raise ValueError("no discovery queries found")

    query_count = len(non_refusals_per_query)
    expected_responses = query_count * len(reference_candidates)
    if response_count != expected_responses:
        raise ValueError(f"found {response_count} responses, expected {expected_responses}")

    summarized_candidates = []
    for candidate in TARGET_CANDIDATES:
        stats = candidate_stats[candidate]
        words = sorted(stats.pop("attempt_word_counts"))
        if not words:
            raise ValueError(f"candidate {candidate} has no non-refusal attempts")
        q1, _, q3 = quantiles(words, n=4, method="inclusive")
        attempts = len(words)
        summarized_candidates.append(
            {
                "candidate": candidate,
                "records": stats["records"],
                "exact_refusals": stats["exact_refusals"],
                "exact_refusal_share": stats["exact_refusals"] / stats["records"],
                "attempts": attempts,
                "attempt_share": attempts / stats["records"],
                "attempt_word_median": median(words),
                "attempt_word_q1": q1,
                "attempt_word_q3": q3,
                "attempts_with_insufficiency_phrase": stats[
                    "attempts_with_insufficiency_phrase"
                ],
                "attempt_insufficiency_share": stats[
                    "attempts_with_insufficiency_phrase"
                ]
                / attempts,
                "attempts_with_context_template": stats["attempts_with_context_template"],
                "attempt_context_template_share": stats["attempts_with_context_template"]
                / attempts,
            }
        )

    return {
        "query_count": query_count,
        "candidate_count": len(reference_candidates),
        "response_count": response_count,
        "exact_refusals": exact_refusals,
        "exact_refusal_share": exact_refusals / response_count,
        "non_refusal_attempts": response_count - exact_refusals,
        "non_refusal_attempt_share": (response_count - exact_refusals) / response_count,
        "median_non_refusal_attempts_per_query": median(non_refusals_per_query),
        "queries_with_zero_non_refusal_attempts": sum(
            count == 0 for count in non_refusals_per_query
        ),
        "queries_with_zero_non_refusal_attempts_share": sum(
            count == 0 for count in non_refusals_per_query
        )
        / query_count,
        "candidate_ids": sorted(reference_candidates),
        "candidate_examples": summarized_candidates,
        "tuple_sha256": tuple_digest.hexdigest(),
    }


def audit_metadata(path: Path, expected_candidates: set[str]) -> dict[str, Any]:
    query_id: str | None = None
    candidate: str | None = None
    response: str | None = None
    token_count: int | None = None
    logprob_count: int | None = None
    array_mode: str | None = None
    tuple_digest = hashlib.sha256()
    record_count = 0
    aligned = 0
    mismatched = 0
    missing_arrays = 0
    query_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    candidate_alignment: Counter[tuple[str, str]] = Counter()

    def close_candidate() -> None:
        nonlocal candidate, response, token_count, logprob_count
        nonlocal record_count, aligned, mismatched, missing_arrays
        if candidate is None:
            return
        if query_id is None or response is None:
            raise ValueError(f"incomplete metadata record for candidate {candidate}")
        record_count += 1
        query_counts[query_id] += 1
        candidate_counts[candidate] += 1
        update_tuple_digest(tuple_digest, (query_id, candidate, response))
        if token_count is None or logprob_count is None:
            missing_arrays += 1
            candidate_alignment[(candidate, "missing")] += 1
        elif token_count == logprob_count and token_count > 0:
            aligned += 1
            candidate_alignment[(candidate, "aligned")] += 1
        else:
            mismatched += 1
            candidate_alignment[(candidate, "mismatched")] += 1
        candidate = None
        response = None
        token_count = None
        logprob_count = None

    with path.open("r", encoding="utf-8") as handle:
        for raw in handle:
            raw = raw.rstrip("\n")
            query_match = QUERY_RE.match(raw)
            if query_match:
                close_candidate()
                query_id = query_match.group(1)
                array_mode = None
                continue

            candidate_match = METADATA_CANDIDATE_RE.match(raw)
            if candidate_match and query_id is not None:
                close_candidate()
                candidate = candidate_match.group(1)
                array_mode = None
                continue

            if candidate is None:
                continue
            response_match = METADATA_RESPONSE_RE.match(raw)
            if response_match:
                response = decode_json_scalar(response_match.group(1))
                continue
            if raw == '        "tokens": [':
                token_count = 0
                array_mode = "tokens"
                continue
            if raw == '        "logprobs": [':
                logprob_count = 0
                array_mode = "logprobs"
                continue
            if array_mode and raw.strip() in {"]", "],"}:
                array_mode = None
                continue
            if array_mode == "tokens" and raw.startswith("          "):
                token_count = (token_count or 0) + 1
            elif array_mode == "logprobs" and raw.startswith("          "):
                logprob_count = (logprob_count or 0) + 1

    close_candidate()
    if any(count != 150 for count in query_counts.values()):
        bad = {query: count for query, count in query_counts.items() if count != 150}
        raise ValueError(f"metadata queries with unexpected candidate count: {bad}")
    if set(candidate_counts) != expected_candidates:
        raise ValueError("metadata candidate set differs from plain discovery data")
    if any(count != len(query_counts) for count in candidate_counts.values()):
        raise ValueError("metadata candidates do not appear once in every query")

    examples = []
    for selected in TARGET_CANDIDATES:
        examples.append(
            {
                "candidate": selected,
                "aligned": candidate_alignment[(selected, "aligned")],
                "mismatched": candidate_alignment[(selected, "mismatched")],
                "missing": candidate_alignment[(selected, "missing")],
                "aligned_share": candidate_alignment[(selected, "aligned")]
                / candidate_counts[selected],
            }
        )
    return {
        "query_count": len(query_counts),
        "response_count": record_count,
        "token_logprob_aligned": aligned,
        "token_logprob_mismatched": mismatched,
        "token_logprob_missing": missing_arrays,
        "token_logprob_mismatch_share": mismatched / record_count,
        "candidate_examples": examples,
        "tuple_sha256": tuple_digest.hexdigest(),
    }


def build_audit(plain: Path, metadata: Path) -> dict[str, Any]:
    plain_audit = audit_plain(plain)
    metadata_audit = audit_metadata(metadata, set(plain_audit["candidate_ids"]))
    if plain_audit["tuple_sha256"] != metadata_audit["tuple_sha256"]:
        raise ValueError("plain and metadata files disagree on ordered response tuples")
    if plain_audit["response_count"] != metadata_audit["response_count"]:
        raise ValueError("plain and metadata response counts differ")

    return {
        "version": 1,
        "sources": {
            "plain": {
                "path": "data/dev/llm_discovery_data.json",
                "sha256": file_sha256(plain),
            },
            "metadata": {
                "path": "data/dev/llm_discovery_metadata.json",
                "sha256": file_sha256(metadata),
            },
            "ordered_response_tuple_sha256": plain_audit["tuple_sha256"],
        },
        "definitions": {
            "exact_refusal": (
                "After lowercasing and replacing non-letters with spaces, the complete response "
                "is exactly 'i don t know' or 'i do not know'."
            ),
            "attempt": (
                "Any response that is not an exact_refusal. Attempt is a coverage proxy, not a "
                "claim of relevance, correctness, or completeness."
            ),
            "logprob_usable": "len(tokens) == len(logprobs) and len(tokens) > 0",
        },
        "plain": plain_audit,
        "metadata": metadata_audit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain", type=Path, default=DEFAULT_PLAIN)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    audit = build_audit(args.plain, args.metadata)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
