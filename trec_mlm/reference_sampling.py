"""Deterministic clustered sampling for reference-answer validation."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from trec_mlm.reference_data import QuestionRecord


QUESTION_PREFIX = {
    "why": "explanation",
    "how": "explanation",
    "what": "factoid",
    "who": "factoid",
    "when": "factoid",
    "where": "factoid",
    "which": "factoid",
}
TIME_RE = re.compile(r"\b(latest|current|currently|today|now|recent|as of)\b", re.I)
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?\b")
UNDERSPECIFIED_RE = re.compile(
    r"\b(the company|the city|the three|that determination|what happened next)\b",
    re.I,
)
DISCOVERY_QUERY_RE = re.compile(r'^  "([^"]+)": \{$')
DISCOVERY_QUERY_TEXT_RE = re.compile(r'^    "query": (.+)$')
DISCOVERY_RESPONSE_RE = re.compile(r'^      "([^"]+)": (.+)$')
TOKEN_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
TITLE_SPAN_RE = re.compile(r"\b[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*\b")

DIFFICULTY_INDICATOR_FIELDS = {
    "high_refusal": "exact_refusal_share",
    "high_disagreement": "normalized_answer_disagreement_entropy",
    "high_entity_density": "entity_density",
}


class HistoricalDifficultyFeatures(dict[str, dict[str, Any]]):
    """Question aggregates plus the empirical thresholds used for their flags."""

    def __init__(
        self,
        values: Mapping[str, dict[str, Any]],
        thresholds: Mapping[str, float],
    ) -> None:
        super().__init__(values)
        self.thresholds = dict(thresholds)


def question_strata(text: str) -> tuple[str, ...]:
    lowered = text.strip().lower()
    first = lowered.split(maxsplit=1)[0] if lowered else ""
    labels = {QUESTION_PREFIX.get(first, "other")}
    if NUMBER_RE.search(text):
        labels.add("numeric")
    if TIME_RE.search(text):
        labels.add("time_sensitive")
    if UNDERSPECIFIED_RE.search(text):
        labels.add("underspecified")
    if len(text.split()) >= 25:
        labels.add("long_or_multihop")
    return tuple(sorted(labels))


def _decode_json_string(payload: str) -> str:
    value = json.loads(payload.rstrip().removesuffix(","))
    if not isinstance(value, str):
        raise ValueError("expected a JSON string")
    return value


def _normalize_response(response: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", response.casefold()).strip()


def _entity_density(query: str) -> float:
    tokens = TOKEN_RE.findall(query)
    if not tokens:
        return 0.0
    after_first = query[query.find(tokens[0]) + len(tokens[0]) :]
    return len(TITLE_SPAN_RE.findall(after_first)) / len(tokens)


def historical_difficulty_features(path: Path) -> HistoricalDifficultyFeatures:
    """Stream discovery responses into per-question aggregate difficulty proxies.

    The official input is pretty-printed with one response JSON string per line.
    Raw response strings exist only for the duration of processing that line.
    """
    result: dict[str, dict[str, Any]] = {}
    query_id: str | None = None
    query_text: str | None = None
    response_count = 0
    refusal_count = 0
    non_refusal_counts: Counter[str] = Counter()

    def finish() -> None:
        nonlocal query_id, query_text, response_count, refusal_count, non_refusal_counts
        if query_id is None:
            return
        if query_text is None:
            raise ValueError(f"discovery question {query_id} has no query text")
        non_refusal_count = response_count - refusal_count
        if non_refusal_count <= 1 or len(non_refusal_counts) <= 1:
            entropy = 0.0
        else:
            raw_entropy = -sum(
                (count / non_refusal_count) * math.log(count / non_refusal_count)
                for count in non_refusal_counts.values()
            )
            entropy = raw_entropy / math.log(non_refusal_count)
        refusal_share = refusal_count / response_count if response_count else 0.0
        non_refusal_share = non_refusal_count / response_count if response_count else 0.0
        entity_density = _entity_density(query_text)
        result[query_id] = {
            "response_count": response_count,
            "exact_refusal_share": refusal_share,
            "non_refusal_share": non_refusal_share,
            "distinct_normalized_non_refusal_answers": len(non_refusal_counts),
            "normalized_answer_disagreement_entropy": entropy,
            "entity_density": entity_density,
        }

    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, 1):
            line = raw_line.rstrip("\n")
            match = DISCOVERY_QUERY_RE.match(line)
            if match:
                finish()
                query_id = match.group(1)
                query_text = None
                response_count = 0
                refusal_count = 0
                non_refusal_counts = Counter()
                continue
            match = DISCOVERY_QUERY_TEXT_RE.match(line)
            if match and query_id is not None:
                query_text = _decode_json_string(match.group(1))
                continue
            match = DISCOVERY_RESPONSE_RE.match(line)
            if not match or query_id is None:
                continue
            if query_text is None:
                raise ValueError(f"response before query text at line {line_number}")
            response = _decode_json_string(match.group(2))
            normalized = _normalize_response(response)
            response_count += 1
            if normalized in {"i don t know", "i do not know"}:
                refusal_count += 1
            else:
                non_refusal_counts[normalized] += 1
    finish()
    if not result:
        raise ValueError("no discovery questions found")
    thresholds = {
        field: float(
            np.quantile(
                [row[field] for row in result.values()],
                0.75,
                method="linear",
            )
        )
        for field in DIFFICULTY_INDICATOR_FIELDS.values()
    }
    for row in result.values():
        for indicator, field in DIFFICULTY_INDICATOR_FIELDS.items():
            # Inclusive comparison makes ties at the empirical threshold deterministic.
            row[indicator] = row[field] >= thresholds[field]
    return HistoricalDifficultyFeatures(result, thresholds)


def _stable_tiebreak(question_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{question_id}".encode()).hexdigest()


def build_clustered_sample(
    records: Iterable[QuestionRecord],
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    sample_size: int,
    seed: int,
    required_ids: set[str] | None = None,
    known_failure_ids: set[str] | None = None,
    historical_features: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Select medoids, required questions, then challenging cluster-balanced rows."""
    rows = list(records)
    vectors = np.asarray(embeddings)
    cluster_labels = np.asarray(labels)
    if vectors.ndim != 2 or len(vectors) != len(rows):
        raise ValueError("embeddings must be a 2D array aligned with records")
    if not np.isfinite(vectors).all():
        raise ValueError("embeddings must contain only finite values")
    if cluster_labels.ndim != 1 or len(cluster_labels) != len(rows):
        raise ValueError("labels must be a 1D array aligned with records")
    if not np.issubdtype(cluster_labels.dtype, np.number) or not np.isfinite(
        cluster_labels
    ).all():
        raise ValueError("labels must contain finite numeric cluster IDs")
    if not np.equal(cluster_labels, np.floor(cluster_labels)).all():
        raise ValueError("labels must contain integer cluster IDs")
    cluster_labels = cluster_labels.astype(np.int64, copy=False)
    clusters = sorted(int(value) for value in np.unique(cluster_labels))
    if sample_size < len(clusters):
        raise ValueError("sample_size must be at least the cluster count")
    if sample_size > len(rows):
        raise ValueError("sample_size cannot exceed the record count")

    id_to_index = {record.question_id: index for index, record in enumerate(rows)}
    if len(id_to_index) != len(rows):
        raise ValueError("question IDs must be unique")
    required = set(required_ids or ())
    missing = required - id_to_index.keys()
    if missing:
        raise ValueError(f"required question IDs are missing: {sorted(missing)!r}")

    distances: dict[int, float] = {}
    medoids: dict[int, int] = {}
    for cluster in clusters:
        indices = np.flatnonzero(cluster_labels == cluster)
        centroid = vectors[indices].mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        for index in indices:
            vector_norm = float(np.linalg.norm(vectors[index]))
            similarity = (
                float(np.dot(vectors[index], centroid)) / (vector_norm * centroid_norm)
                if vector_norm and centroid_norm
                else 0.0
            )
            distances[int(index)] = 1.0 - similarity
        medoids[cluster] = min(
            (int(index) for index in indices),
            key=lambda index: (distances[index], _stable_tiebreak(rows[index].question_id, seed)),
        )

    if len(set(medoids.values()) | {id_to_index[item] for item in required}) > sample_size:
        raise ValueError("sample_size cannot contain all medoids and required IDs")

    failures = set(known_failure_ids or ())
    history = historical_features or {}
    details: dict[int, dict[str, Any]] = {}
    for index, record in enumerate(rows):
        strata = question_strata(record.question)
        historical = history.get(record.question_id, {})
        indicators = {
            "unresolved": record.perplexity_unresolved,
            "known_failure_analogue": record.question_id in failures,
            "time_sensitive": "time_sensitive" in strata,
            "numeric": "numeric" in strata,
            "underspecified": "underspecified" in strata,
            "long_or_multihop": "long_or_multihop" in strata,
            "high_refusal": bool(historical.get("high_refusal", False)),
            "high_disagreement": bool(historical.get("high_disagreement", False)),
            "high_entity_density": bool(historical.get("high_entity_density", False)),
        }
        details[index] = {
            "question_id": record.question_id,
            "question": record.question,
            "cluster_id": int(cluster_labels[index]),
            "centroid_distance": distances[index],
            "strata": list(strata),
            "challenge_score": sum(indicators.values()),
            "challenge_indicators": indicators,
        }

    selected: set[int] = set()
    reasons: dict[int, list[str]] = {}
    for index in medoids.values():
        selected.add(index)
        reasons.setdefault(index, []).append("cluster_medoid")
    for question_id in sorted(required):
        index = id_to_index[question_id]
        selected.add(index)
        reasons.setdefault(index, []).append("required_id")

    candidates: dict[int, list[int]] = {}
    for cluster in clusters:
        candidates[cluster] = sorted(
            (
                index
                for index in range(len(rows))
                if int(cluster_labels[index]) == cluster and index not in selected
            ),
            key=lambda index: (
                details[index]["challenge_score"],
                details[index]["centroid_distance"],
                _stable_tiebreak(rows[index].question_id, seed),
            ),
            reverse=True,
        )
    while len(selected) < sample_size:
        progressed = False
        for cluster in clusters:
            if candidates[cluster] and len(selected) < sample_size:
                index = candidates[cluster].pop(0)
                selected.add(index)
                reasons.setdefault(index, []).append("cluster_round_robin")
                progressed = True
        if not progressed:
            raise RuntimeError("unable to fill requested sample")

    result = []
    for index in sorted(selected, key=lambda item: rows[item].question_id):
        row = dict(details[index])
        row["selection_reasons"] = reasons[index]
        result.append(row)
    return result
