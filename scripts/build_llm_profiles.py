#!/usr/bin/env python3
"""Build one auditable discovery profile for each 2026 candidate.

Version 1 intentionally stays simple.  It summarizes observed response behavior,
metadata usability, and words enriched in the discovery queries a candidate
attempted.  Those are routing clues, not verified expertise or correctness.

The official JSON files are large, pretty-printed documents.  This builder streams
them line by line and keeps only candidate counters and small summary lists in
memory.  If the organizers change the serialization layout, replace these parsers
with a general streaming JSON reader before rebuilding the profiles.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from statistics import median, quantiles
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLAIN = ROOT / "data/dev/llm_discovery_data.json"
DEFAULT_METADATA = ROOT / "data/dev/llm_discovery_metadata.json"
DEFAULT_QRELS = ROOT / "data/dev/llm_dev_qrels.txt"
DEFAULT_AUDIT = ROOT / "reports/trec_2026_discovery_profile_audit.json"
DEFAULT_OUTPUT = ROOT / "reports/trec_2026_llm_profiles.json"
PROFILE_VERSION = "discovery-behavior-topic-v1"
GENERATED_AT = "2026-07-11T00:00:00Z"

QUERY_RE = re.compile(r'^  "([^"]+)": \{$')
QUERY_TEXT_RE = re.compile(r'^    "query": (.+)$')
PLAIN_RESPONSE_RE = re.compile(r'^      "([^"]+)": (.+)$')
METADATA_CANDIDATE_RE = re.compile(r'^      "([^"]+)": \{$')
METADATA_RESPONSE_RE = re.compile(r'^        "response": (.+)$')
WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)
QUERY_TERM_RE = re.compile(r"[a-z][a-z0-9'-]{2,}")
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

# Question scaffolding is removed so the short lexical summary emphasizes subject
# words.  No stemming, model, or external vocabulary is used.
STOPWORDS = {
    "about", "above", "after", "again", "against", "affect", "affected", "all", "also", "among",
    "and", "any", "are", "because", "been", "before", "being", "below",
    "between", "both", "but", "can", "cause", "caused", "could", "describe", "did", "different",
    "does", "doing", "during", "each", "explain", "few", "first", "for",
    "from", "further", "give", "had", "has", "have", "having", "her", "here",
    "hers", "herself", "him", "himself", "his", "how", "into", "its", "itself",
    "just", "known", "list", "main", "mean", "means", "more", "most", "name", "nor", "not",
    "now", "off", "once", "only", "other", "our", "ours", "ourselves", "out",
    "over", "own", "same", "should", "some", "such", "tell", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "too", "under", "until", "used", "uses",
    "using", "very", "was", "were", "what", "when", "where", "which", "while",
    "who", "whom", "why", "will", "with", "would", "you", "your", "yours",
    "happen", "happened", "according", "following", "role", "work",
}


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


def query_terms(query: str) -> set[str]:
    return {
        term
        for term in QUERY_TERM_RE.findall(query.lower())
        if term not in STOPWORDS
    }


def empty_plain_stats() -> dict[str, Any]:
    return {
        "record_count": 0,
        "exact_refusal_count": 0,
        "attempt_word_counts": [],
        "attempts_with_insufficiency_phrase": 0,
        "attempts_with_context_template": 0,
        "attempted_query_terms": Counter(),
    }


def empty_metadata_stats() -> dict[str, Any]:
    return {
        "record_count": 0,
        "token_logprob_aligned_count": 0,
        "token_logprob_mismatched_count": 0,
        "token_logprob_missing_count": 0,
        "attempt_token_logprob_usable_count": 0,
    }


def summarize_distribution(values: list[float | int]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "q1": None, "q3": None}
    ordered = sorted(values)
    if len(ordered) == 1:
        value = float(ordered[0])
        return {"median": value, "q1": value, "q3": value}
    q1, _, q3 = quantiles(ordered, n=4, method="inclusive")
    return {
        "median": float(median(ordered)),
        "q1": float(q1),
        "q3": float(q3),
    }


def close_plain_query(
    query_id: str | None,
    query_text: str | None,
    seen_candidates: set[str],
    reference_candidates: set[str] | None,
    query_document_frequency: Counter[str],
) -> set[str] | None:
    if query_id is None:
        return reference_candidates
    if query_text is None:
        raise ValueError(f"query {query_id} has no query text")
    if len(seen_candidates) != 150:
        raise ValueError(
            f"query {query_id} has {len(seen_candidates)} candidates, expected 150"
        )
    if reference_candidates is None:
        reference_candidates = set(seen_candidates)
    elif seen_candidates != reference_candidates:
        raise ValueError(f"query {query_id} has a different candidate set")
    query_document_frequency.update(query_terms(query_text))
    return reference_candidates


def build_plain_profiles(path: Path) -> dict[str, Any]:
    query_id: str | None = None
    query_text: str | None = None
    terms_for_query: set[str] = set()
    seen_candidates: set[str] = set()
    reference_candidates: set[str] | None = None
    query_count = 0
    response_count = 0
    tuple_digest = hashlib.sha256()
    query_document_frequency: Counter[str] = Counter()
    candidate_stats: dict[str, dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            raw = raw.rstrip("\n")
            query_match = QUERY_RE.match(raw)
            if query_match:
                reference_candidates = close_plain_query(
                    query_id,
                    query_text,
                    seen_candidates,
                    reference_candidates,
                    query_document_frequency,
                )
                query_id = query_match.group(1)
                query_text = None
                terms_for_query = set()
                seen_candidates = set()
                query_count += 1
                continue

            query_text_match = QUERY_TEXT_RE.match(raw)
            if query_text_match and query_id is not None:
                query_text = decode_json_scalar(query_text_match.group(1))
                terms_for_query = query_terms(query_text)
                continue

            response_match = PLAIN_RESPONSE_RE.match(raw)
            if not response_match or query_id is None:
                continue
            if query_text is None:
                raise ValueError(f"response before query text at line {line_number}")
            candidate = response_match.group(1)
            if candidate in seen_candidates:
                raise ValueError(f"duplicate candidate {candidate} at line {line_number}")
            response = decode_json_scalar(response_match.group(2))
            seen_candidates.add(candidate)
            response_count += 1
            update_tuple_digest(tuple_digest, (query_id, candidate, response))

            stats = candidate_stats.setdefault(candidate, empty_plain_stats())
            stats["record_count"] += 1
            if is_exact_refusal(response):
                stats["exact_refusal_count"] += 1
                continue

            stats["attempt_word_counts"].append(len(WORD_RE.findall(response)))
            if INSUFFICIENCY_RE.search(response):
                stats["attempts_with_insufficiency_phrase"] += 1
            if CONTEXT_TEMPLATE_RE.search(response):
                stats["attempts_with_context_template"] += 1
            stats["attempted_query_terms"].update(terms_for_query)

    reference_candidates = close_plain_query(
        query_id,
        query_text,
        seen_candidates,
        reference_candidates,
        query_document_frequency,
    )
    if reference_candidates is None:
        raise ValueError("no discovery queries found")
    if response_count != query_count * len(reference_candidates):
        raise ValueError("plain discovery response count does not form a complete matrix")
    if set(candidate_stats) != reference_candidates:
        raise ValueError("plain discovery candidate set is inconsistent")

    return {
        "query_count": query_count,
        "response_count": response_count,
        "candidate_ids": sorted(reference_candidates),
        "query_document_frequency": query_document_frequency,
        "candidate_stats": candidate_stats,
        "tuple_sha256": tuple_digest.hexdigest(),
    }


def build_metadata_profiles(path: Path, expected_candidates: set[str]) -> dict[str, Any]:
    query_id: str | None = None
    candidate: str | None = None
    response: str | None = None
    token_count: int | None = None
    logprob_count: int | None = None
    array_mode: str | None = None
    tuple_digest = hashlib.sha256()
    query_counts: Counter[str] = Counter()
    candidate_stats: dict[str, dict[str, Any]] = {}

    def close_candidate() -> None:
        nonlocal candidate, response, token_count, logprob_count
        if candidate is None:
            return
        if query_id is None or response is None:
            raise ValueError(f"incomplete metadata record for candidate {candidate}")
        stats = candidate_stats.setdefault(candidate, empty_metadata_stats())
        stats["record_count"] += 1
        query_counts[query_id] += 1
        update_tuple_digest(tuple_digest, (query_id, candidate, response))
        if token_count is None or logprob_count is None:
            stats["token_logprob_missing_count"] += 1
        elif token_count == logprob_count and token_count > 0:
            stats["token_logprob_aligned_count"] += 1
            if not is_exact_refusal(response):
                stats["attempt_token_logprob_usable_count"] += 1
        else:
            stats["token_logprob_mismatched_count"] += 1
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
    if set(candidate_stats) != expected_candidates:
        raise ValueError("metadata candidate set differs from plain discovery data")
    if any(
        stats["record_count"] != len(query_counts)
        for stats in candidate_stats.values()
    ):
        raise ValueError("metadata candidates do not appear once in every query")

    return {
        "query_count": len(query_counts),
        "response_count": sum(
            stats["record_count"] for stats in candidate_stats.values()
        ),
        "candidate_stats": candidate_stats,
        "tuple_sha256": tuple_digest.hexdigest(),
    }


def enriched_terms(
    term_counts: Counter[str],
    attempt_count: int,
    query_document_frequency: Counter[str],
    query_count: int,
    limit: int = 8,
) -> list[dict[str, Any]]:
    if attempt_count == 0:
        return []
    candidates: list[dict[str, Any]] = []
    for term, support in term_counts.items():
        global_support = query_document_frequency[term]
        if support < 5 or global_support < 10:
            continue
        candidate_rate = (support + 0.5) / (attempt_count + 1.0)
        all_query_rate = (global_support + 0.5) / (query_count + 1.0)
        lift = candidate_rate / all_query_rate
        if lift < 1.25:
            continue
        score = math.log2(lift) * math.log1p(support)
        candidates.append(
            {
                "term": term,
                "support_queries": support,
                "candidate_attempt_share": support / attempt_count,
                "all_query_share": global_support / query_count,
                "lift": lift,
                "ranking_score": score,
            }
        )
    candidates.sort(
        key=lambda row: (
            -row["ranking_score"],
            -row["support_queries"],
            row["term"],
        )
    )
    return candidates[:limit]


def qrel_candidate_ids(path: Path) -> set[str]:
    candidates: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            parts = raw.split()
            if len(parts) != 4:
                raise ValueError(f"invalid qrel row {line_number}")
            candidates.add(parts[2])
    return candidates


def build_profiles(
    plain: Path,
    metadata: Path,
    audit_path: Path,
    qrels: Path = DEFAULT_QRELS,
) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    plain_result = build_plain_profiles(plain)
    metadata_result = build_metadata_profiles(
        metadata, set(plain_result["candidate_ids"])
    )
    if set(plain_result["candidate_ids"]) != qrel_candidate_ids(qrels):
        raise ValueError("discovery candidate IDs differ from development qrels")

    expected_digest = audit["sources"]["ordered_response_tuple_sha256"]
    for label, result in (("plain", plain_result), ("metadata", metadata_result)):
        if result["tuple_sha256"] != expected_digest:
            raise ValueError(f"{label} ordered response tuples differ from saved audit")
    if plain_result["query_count"] != metadata_result["query_count"]:
        raise ValueError("plain and metadata query counts differ")
    if plain_result["response_count"] != metadata_result["response_count"]:
        raise ValueError("plain and metadata response counts differ")

    profiles = []
    for candidate_id in plain_result["candidate_ids"]:
        plain_stats = plain_result["candidate_stats"][candidate_id]
        metadata_stats = metadata_result["candidate_stats"][candidate_id]
        record_count = plain_stats["record_count"]
        exact_refusal_count = plain_stats["exact_refusal_count"]
        attempt_count = record_count - exact_refusal_count
        word_summary = summarize_distribution(plain_stats["attempt_word_counts"])
        top_terms = enriched_terms(
            plain_stats["attempted_query_terms"],
            attempt_count,
            plain_result["query_document_frequency"],
            plain_result["query_count"],
        )
        term_text = ", ".join(row["term"] for row in top_terms[:5]) or "none"
        word_text = (
            f"{word_summary['median']:.0f} words"
            if word_summary["median"] is not None
            else "unavailable"
        )
        profiles.append(
            {
                "candidate_id": candidate_id,
                "profile_version": PROFILE_VERSION,
                "profile_kind": "observed discovery behavior and attempted-query associations",
                "history": {
                    "discovery_query_count": plain_result["query_count"],
                    "record_count": record_count,
                },
                "coverage": {
                    "exact_refusal_count": exact_refusal_count,
                    "exact_refusal_share": exact_refusal_count / record_count,
                    "attempt_count": attempt_count,
                    "attempt_share": attempt_count / record_count,
                },
                "attempt_style": {
                    "word_count_median": word_summary["median"],
                    "word_count_q1": word_summary["q1"],
                    "word_count_q3": word_summary["q3"],
                    "insufficiency_phrase_count": plain_stats[
                        "attempts_with_insufficiency_phrase"
                    ],
                    "insufficiency_phrase_share": (
                        plain_stats["attempts_with_insufficiency_phrase"] / attempt_count
                        if attempt_count
                        else None
                    ),
                    "context_template_count": plain_stats[
                        "attempts_with_context_template"
                    ],
                    "context_template_share": (
                        plain_stats["attempts_with_context_template"] / attempt_count
                        if attempt_count
                        else None
                    ),
                },
                "metadata_quality": {
                    "token_logprob_aligned_count": metadata_stats[
                        "token_logprob_aligned_count"
                    ],
                    "token_logprob_mismatched_count": metadata_stats[
                        "token_logprob_mismatched_count"
                    ],
                    "token_logprob_missing_count": metadata_stats[
                        "token_logprob_missing_count"
                    ],
                    "token_logprob_usable_share": metadata_stats[
                        "token_logprob_aligned_count"
                    ]
                    / record_count,
                    "attempt_token_logprob_usable_count": metadata_stats[
                        "attempt_token_logprob_usable_count"
                    ],
                    "attempt_token_logprob_usable_share": (
                        metadata_stats["attempt_token_logprob_usable_count"]
                        / attempt_count
                        if attempt_count
                        else None
                    ),
                },
                "attempted_query_terms": top_terms,
                "profile_summary": (
                    f"Attempted {attempt_count / record_count:.1%} of discovery queries; "
                    f"median attempted response: {word_text}; enriched attempted-query "
                    f"terms: {term_text}."
                ),
                "warnings": [
                    "An attempt is not evidence that the response was relevant or correct.",
                    "Enriched terms describe queries this candidate attempted in this release, not verified expertise.",
                    "Token confidence is a behavior clue, not a correctness label.",
                ],
            }
        )

    validate_profiles(profiles, plain_result, metadata_result, audit)
    return {
        "version": 1,
        "profile_version": PROFILE_VERSION,
        "generated_at": GENERATED_AT,
        "profile_kind": "observed discovery behavior and attempted-query associations",
        "scope": {
            "query_count": plain_result["query_count"],
            "candidate_count": len(profiles),
            "response_count": plain_result["response_count"],
            "one_profile_row_per_candidate": True,
        },
        "definitions": {
            "exact_refusal": audit["definitions"]["exact_refusal"],
            "attempt": audit["definitions"]["attempt"],
            "token_logprob_usable": audit["definitions"]["logprob_usable"],
            "enriched_attempted_query_term": (
                "A non-stopword occurring in at least five queries this candidate attempted "
                "and at least ten discovery queries overall, with smoothed occurrence lift "
                "of at least 1.25 versus all discovery queries. "
                "Terms are ranked by log2(lift) * log(1 + support)."
            ),
            "attempt_token_logprob_usable": (
                "A non-exact-refusal response whose token and logprob arrays have the same "
                "positive length. This only says confidence features can be computed."
            ),
        },
        "field_guide": {
            "coverage": "How often the candidate emitted an exact default versus anything else.",
            "attempt_style": "Length and template-language summaries over non-exact attempts.",
            "metadata_quality": "Whether token/logprob arrays are structurally usable for later confidence features.",
            "attempted_query_terms": "Words overrepresented in queries the candidate attempted, with support and lift.",
        },
        "topic_term_method": {
            "tokenizer": "lowercase regular expression [a-z][a-z0-9'-]{2,}",
            "counting_unit": "each cleaned term counts at most once per discovery query",
            "stop_terms": sorted(STOPWORDS),
            "minimum_candidate_attempted_query_support": 5,
            "minimum_all_query_support": 10,
            "minimum_smoothed_lift": 1.25,
            "smoothing": "Jeffreys-style add 0.5 to numerator and 1.0 to denominator",
            "ranking_score": "log2(smoothed lift) * log(1 + candidate support)",
            "retained_terms_per_candidate": 8,
        },
        "what_this_is": (
            "A real first-pass profile table built from the official discovery history. It can "
            "supply candidate-only behavior features and transparent lexical topic clues."
        ),
        "what_this_is_not": (
            "A correctness score, gold expertise label, semantic centroid, or final router score."
        ),
        "next_profile_versions": [
            "response-level lexical index, kept as source evidence rather than compressed away",
            "several semantic coverage centroids, evaluated as a separate likely-external experiment",
            "topic-local judged-quality rows after a calibrated LLM-judge pilot",
        ],
        "sources": {
            "plain": audit["sources"]["plain"],
            "metadata": audit["sources"]["metadata"],
            "development_qrels": {"path": "data/dev/llm_dev_qrels.txt"},
            "ordered_response_tuple_sha256": expected_digest,
            "audit_path": "reports/trec_2026_discovery_profile_audit.json",
            "builder_path": "scripts/build_llm_profiles.py",
        },
        "profiles": profiles,
    }


def validate_profiles(
    profiles: list[dict[str, Any]],
    plain_result: dict[str, Any],
    metadata_result: dict[str, Any],
    audit: dict[str, Any],
) -> None:
    if len(profiles) != 150:
        raise ValueError(f"expected 150 profiles, found {len(profiles)}")
    candidate_ids = [profile["candidate_id"] for profile in profiles]
    if candidate_ids != sorted(set(candidate_ids)):
        raise ValueError("profile candidate IDs are duplicated or not sorted")

    total_attempts = 0
    total_refusals = 0
    total_aligned = 0
    total_mismatched = 0
    total_missing = 0
    for profile in profiles:
        record_count = profile["history"]["record_count"]
        coverage = profile["coverage"]
        quality = profile["metadata_quality"]
        if record_count != plain_result["query_count"]:
            raise ValueError(f"{profile['candidate_id']} has an incomplete history")
        if coverage["attempt_count"] + coverage["exact_refusal_count"] != record_count:
            raise ValueError(f"{profile['candidate_id']} coverage counts do not reconcile")
        if (
            quality["token_logprob_aligned_count"]
            + quality["token_logprob_mismatched_count"]
            + quality["token_logprob_missing_count"]
            != record_count
        ):
            raise ValueError(f"{profile['candidate_id']} metadata counts do not reconcile")
        for rate in (
            coverage["attempt_share"],
            coverage["exact_refusal_share"],
            quality["token_logprob_usable_share"],
        ):
            if not 0.0 <= rate <= 1.0:
                raise ValueError(f"{profile['candidate_id']} has an invalid rate")
        total_attempts += coverage["attempt_count"]
        total_refusals += coverage["exact_refusal_count"]
        total_aligned += quality["token_logprob_aligned_count"]
        total_mismatched += quality["token_logprob_mismatched_count"]
        total_missing += quality["token_logprob_missing_count"]

    if total_attempts != audit["plain"]["non_refusal_attempts"]:
        raise ValueError("profile attempt total differs from saved audit")
    if total_refusals != audit["plain"]["exact_refusals"]:
        raise ValueError("profile exact-refusal total differs from saved audit")
    if total_aligned != audit["metadata"]["token_logprob_aligned"]:
        raise ValueError("profile aligned-metadata total differs from saved audit")
    if total_mismatched != audit["metadata"]["token_logprob_mismatched"]:
        raise ValueError("profile mismatched-metadata total differs from saved audit")
    if total_missing != audit["metadata"]["token_logprob_missing"]:
        raise ValueError("profile missing-metadata total differs from saved audit")
    if metadata_result["response_count"] != plain_result["response_count"]:
        raise ValueError("plain and metadata profile totals differ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plain", type=Path, default=DEFAULT_PLAIN)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    profiles = build_profiles(args.plain, args.metadata, args.audit, args.qrels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
