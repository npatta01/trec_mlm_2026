"""Qrel-free pointwise evidence construction for the Phase 0 ranking gate."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.artifact_io import (
    manifest_aggregate_sha256,
    manifest_entries,
    sha256_file,
)


FROZEN_LABELS_SHA256 = (
    "7cde37d0d913b97ede43c4791b2f808e6d1b0c863bdcac729efde78480113c48"
)
FROZEN_RECORDS_SHA256 = (
    "e25237be81ffd86726fb4b3ccd1a2a8bdea7213197f84341b6bf50c23274ae92"
)
OFFICIAL_RECORD_COUNT = 182_764
OFFICIAL_QUESTION_COUNT = 6_816
OFFICIAL_CANDIDATE_COUNT = 150
SELECTOR_VERSION = "pointwise-phase0-pair-selector-v1"
SELECTOR_INITIAL_CAP = 20
SELECTOR_TARGET_PAIRS = 500
RRF_K = 60
FROZEN_REFINEMENT_AGGREGATE_SHA256 = (
    "113fa6391df361bd161c2c9ad6a1d3ccb7f6e6f59a917b66135c6a87b0a33f83"
)
FROZEN_SELECTED_RECORD_IDS_SHA256 = (
    "79c2d61cda800739eb36a1cbe7eb52bb81fda36d057e19bc6ee43991c673a371"
)
FROZEN_TOP_RECORD_INDICES_SHA256 = (
    "567d2b0aa7d44508976f857a0188cbe3013f6e2c44935ae099edc39116c69047"
)
FROZEN_TOP_RECORD_SIMILARITIES_SHA256 = (
    "de6fb07d5221d439bb2d9e31c74737cc913cb111fb4705f90f7ad90ad724a44b"
)
SUPPORTED_VERDICTS = frozenset({"correct", "partial"})
VERDICTS = frozenset({"correct", "partial", "incorrect", "nonanswer", "not_established"})
LABEL_KEYS = frozenset(
    {
        "record_id",
        "question_id",
        "candidate_id",
        "source_ordinal",
        "supported",
        "verdict",
    }
)
RECORD_IDENTITY_KEYS = frozenset({"record_id", "source_query_id", "candidate_id"})


@dataclass(frozen=True, slots=True)
class PointwiseLabel:
    """Compact audited identity and support result for one response."""

    record_id: str
    question_id: str
    candidate_id: str
    source_ordinal: int
    supported: bool
    verdict: str


@dataclass(frozen=True, slots=True)
class PointwiseEvidence:
    """Strictly validated pointwise evidence joined to construction identities."""

    labels_sha256: str
    records_sha256: str
    labels_by_record_id: dict[str, PointwiseLabel]
    record_ids: tuple[str, ...]
    question_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    audited_counts: dict[str, int]
    supported_counts: dict[str, int]
    audited_question_coverage: dict[str, int]
    candidate_priors: dict[str, float]


@dataclass(frozen=True, slots=True)
class PointwisePredictions:
    """Frozen prior, routing, and equal-RRF pointwise predictions."""

    query_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    candidate_priors: NDArray[np.float64]
    local_scores: NDArray[np.float64]
    residual_scores: NDArray[np.float64]
    pointwise_rrf_scores: NDArray[np.float64]
    prior_rankings: dict[str, list[str]]
    routing_rankings: dict[str, list[str]]
    pointwise_rankings: dict[str, list[str]]
    query_similarity_means: dict[str, float]
    query_similarity_quartiles: dict[str, int]
    candidate_audited_quartiles: dict[str, int]
    candidate_prior_quartiles: dict[str, int]


@dataclass(frozen=True, slots=True)
class SelectedPair:
    """Sanitized selected-pair identity; no private text or outcome is retained."""

    question_id: str
    candidate_ids: tuple[str, str]
    record_ids: tuple[str, str]
    candidate_use_cap: int


@dataclass(frozen=True, slots=True)
class PairSelectionReceipt:
    """IDs-and-counts-only receipt for the frozen zero-call selector."""

    selector_version: str
    target_pair_count: int
    eligible_pair_count: int
    eligible_source_question_count: int
    selected_pair_count: int
    source_question_count: int
    initial_candidate_use_cap: int
    final_candidate_use_cap: int
    candidate_use_counts: dict[str, int]
    pairs: tuple[SelectedPair, ...]


@dataclass(frozen=True, slots=True)
class _EligiblePair:
    question_id: str
    candidate_ids: tuple[str, str]
    record_ids: tuple[str, str]
    tier: int
    prior_distance: float
    selector_hash: str


def _require_non_empty_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _stream_json_array(path: Path, *, chunk_size: int = 64 * 1024) -> Iterator[Any]:
    """Yield a top-level JSON array without retaining its private payloads."""

    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def fill() -> None:
            nonlocal buffer, eof
            chunk = handle.read(chunk_size)
            if chunk:
                buffer += chunk
            else:
                eof = True

        def skip_whitespace() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                fill()

        fill()
        skip_whitespace()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError("construction records must be a top-level JSON array")
        position += 1
        first = True
        while True:
            skip_whitespace()
            if position < len(buffer) and buffer[position] == "]":
                position += 1
                break
            if not first:
                if position >= len(buffer) or buffer[position] != ",":
                    raise ValueError("construction records JSON array is malformed")
                position += 1
                skip_whitespace()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError as error:
                    if eof:
                        raise ValueError("construction records JSON is malformed") from error
                    fill()
            yield value
            position = end
            first = False
            if position > chunk_size:
                buffer = buffer[position:]
                position = 0
        skip_whitespace()
        if position != len(buffer):
            raise ValueError("construction records contain trailing JSON content")


def _validate_expected_count(actual: int, expected: int | None, label: str) -> None:
    if expected is not None and actual != expected:
        raise ValueError(f"{label} count changed: expected {expected}, observed {actual}")


def _validate_hash(path: Path, expected: str | None, label: str) -> str:
    observed = sha256_file(path)
    if expected is not None and observed != expected:
        raise ValueError(
            f"{label} SHA-256 changed: expected {expected}, observed {observed}"
        )
    return observed


def load_pointwise_evidence(
    labels_path: str | Path,
    records_path: str | Path,
    *,
    expected_labels_sha256: str | None = FROZEN_LABELS_SHA256,
    expected_records_sha256: str | None = FROZEN_RECORDS_SHA256,
    expected_record_count: int | None = OFFICIAL_RECORD_COUNT,
    expected_question_count: int | None = OFFICIAL_QUESTION_COUNT,
    expected_candidate_count: int | None = OFFICIAL_CANDIDATE_COUNT,
) -> PointwiseEvidence:
    """Stream, validate, and join frozen labels to records only by ``record_id``."""

    labels_file = Path(labels_path)
    records_file = Path(records_path)
    labels_sha256 = _validate_hash(labels_file, expected_labels_sha256, "label file")
    records_sha256 = _validate_hash(records_file, expected_records_sha256, "record file")

    labels: dict[str, PointwiseLabel] = {}
    ordinals: set[int] = set()
    with labels_file.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"label JSONL line {line_number} is malformed") from error
            if not isinstance(row, dict) or set(row) != LABEL_KEYS:
                raise ValueError(f"label JSONL line {line_number} schema changed")
            record_id = _require_non_empty_string(row["record_id"], "record ID")
            question_id = _require_non_empty_string(row["question_id"], "question ID")
            candidate_id = _require_non_empty_string(row["candidate_id"], "candidate ID")
            source_ordinal = row["source_ordinal"]
            supported = row["supported"]
            verdict = row["verdict"]
            if (
                isinstance(source_ordinal, bool)
                or not isinstance(source_ordinal, int)
                or source_ordinal < 0
            ):
                raise ValueError("source ordinal must be a non-negative integer")
            if not isinstance(supported, bool):
                raise ValueError("supported must be boolean")
            if not isinstance(verdict, str) or verdict not in VERDICTS:
                raise ValueError(f"unsupported pointwise verdict: {verdict!r}")
            if supported != (verdict in SUPPORTED_VERDICTS):
                raise ValueError("supported is inconsistent with verdict")
            if record_id in labels:
                raise ValueError(f"duplicate label record ID: {record_id!r}")
            if source_ordinal in ordinals:
                raise ValueError(f"duplicate source ordinal: {source_ordinal}")
            labels[record_id] = PointwiseLabel(
                record_id,
                question_id,
                candidate_id,
                source_ordinal,
                supported,
                verdict,
            )
            ordinals.add(source_ordinal)

    record_ids: list[str] = []
    seen_records: set[str] = set()
    for index, row in enumerate(_stream_json_array(records_file)):
        if not isinstance(row, dict) or not RECORD_IDENTITY_KEYS <= set(row):
            raise ValueError(f"construction record {index} identity schema changed")
        record_id = _require_non_empty_string(row["record_id"], "record ID")
        question_id = _require_non_empty_string(
            row["source_query_id"], "source query ID"
        )
        candidate_id = _require_non_empty_string(row["candidate_id"], "candidate ID")
        if record_id in seen_records:
            raise ValueError(f"duplicate construction record ID: {record_id!r}")
        label = labels.get(record_id)
        if label is None:
            raise ValueError(f"construction record has no label: {record_id!r}")
        if (label.question_id, label.candidate_id) != (question_id, candidate_id):
            raise ValueError(f"record-ID identity mismatch: {record_id!r}")
        seen_records.add(record_id)
        record_ids.append(record_id)

    if set(labels) != seen_records:
        missing = min(set(labels) - seen_records)
        raise ValueError(f"label record ID is absent from construction records: {missing!r}")
    _validate_expected_count(len(labels), expected_record_count, "pointwise record")
    question_ids = tuple(sorted({label.question_id for label in labels.values()}))
    candidate_ids = tuple(sorted({label.candidate_id for label in labels.values()}))
    _validate_expected_count(len(question_ids), expected_question_count, "source question")
    _validate_expected_count(len(candidate_ids), expected_candidate_count, "candidate")

    audited_counts = {candidate_id: 0 for candidate_id in candidate_ids}
    supported_counts = {candidate_id: 0 for candidate_id in candidate_ids}
    covered_questions = {candidate_id: set() for candidate_id in candidate_ids}
    for label in labels.values():
        audited_counts[label.candidate_id] += 1
        supported_counts[label.candidate_id] += int(label.supported)
        covered_questions[label.candidate_id].add(label.question_id)
    coverage = {
        candidate_id: len(covered_questions[candidate_id])
        for candidate_id in candidate_ids
    }
    priors = {
        candidate_id: (supported_counts[candidate_id] + 1)
        / (audited_counts[candidate_id] + 2)
        for candidate_id in candidate_ids
    }
    return PointwiseEvidence(
        labels_sha256,
        records_sha256,
        labels,
        tuple(record_ids),
        question_ids,
        candidate_ids,
        audited_counts,
        supported_counts,
        coverage,
        priors,
    )


def stable_quartile_labels(features: Mapping[str, float]) -> dict[str, int]:
    """Assign exact low-to-high quartiles by stable ``(value, identifier)`` rank."""

    pairs: list[tuple[float, str]] = []
    for identifier, value in features.items():
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("quartile identifiers must be non-empty strings")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("quartile feature values must be finite")
        pairs.append((numeric, identifier))
    pairs.sort()
    count = len(pairs)
    return {
        identifier: 1 + (4 * position) // count
        for position, (_, identifier) in enumerate(pairs)
    }


def _rank_scores(
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    scores: NDArray[np.floating[Any]],
) -> dict[str, list[str]]:
    values = np.asarray(scores, dtype=np.float64)
    shape = (len(query_ids), len(candidate_ids))
    if values.shape != shape or not np.isfinite(values).all():
        raise ValueError("ranking scores must be a finite query-by-candidate matrix")
    return {
        query_id: sorted(
            candidate_ids,
            key=lambda candidate_id: (
                -float(values[query_index, candidate_ids.index(candidate_id)]),
                candidate_id,
            ),
        )
        for query_index, query_id in enumerate(query_ids)
    }


def _equal_rrf(
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    prior_rankings: Mapping[str, Sequence[str]],
    routing_rankings: Mapping[str, Sequence[str]],
) -> tuple[NDArray[np.float64], dict[str, list[str]]]:
    scores = np.empty((len(query_ids), len(candidate_ids)), dtype=np.float64)
    rankings: dict[str, list[str]] = {}
    candidate_index = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    expected = set(candidate_ids)
    for query_index, query_id in enumerate(query_ids):
        prior = tuple(prior_rankings[query_id])
        routing = tuple(routing_rankings[query_id])
        if set(prior) != expected or set(routing) != expected:
            raise ValueError("RRF ranking candidate inventory changed")
        prior_ranks = {candidate_id: rank for rank, candidate_id in enumerate(prior, 1)}
        routing_ranks = {
            candidate_id: rank for rank, candidate_id in enumerate(routing, 1)
        }
        for candidate_id in candidate_ids:
            scores[query_index, candidate_index[candidate_id]] = (
                1.0 / (RRF_K + prior_ranks[candidate_id])
                + 1.0 / (RRF_K + routing_ranks[candidate_id])
            )
        rankings[query_id] = sorted(
            candidate_ids,
            key=lambda candidate_id: (
                -float(scores[query_index, candidate_index[candidate_id]]),
                prior_ranks[candidate_id],
                routing_ranks[candidate_id],
                candidate_id,
            ),
        )
    return scores, rankings


def _load_refinement_inputs(root: Path) -> tuple[tuple[str, ...], tuple[str, ...]]:
    payload = json.loads((root / "inputs.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"query_ids", "candidate_ids"}:
        raise ValueError("refinement input schema changed")
    query_ids = tuple(payload["query_ids"])
    candidate_ids = tuple(payload["candidate_ids"])
    if (
        not query_ids
        or not candidate_ids
        or len(set(query_ids)) != len(query_ids)
        or len(set(candidate_ids)) != len(candidate_ids)
        or any(
            not isinstance(identifier, str) or not identifier
            for identifier in query_ids + candidate_ids
        )
    ):
        raise ValueError("refinement input IDs must be non-empty and unique")
    return query_ids, candidate_ids


def construct_pointwise_rankings(
    evidence: PointwiseEvidence,
    refinement_root: str | Path,
    *,
    expected_refinement_aggregate_sha256: str | None = (
        FROZEN_REFINEMENT_AGGREGATE_SHA256
    ),
    expected_selected_record_ids_sha256: str | None = (
        FROZEN_SELECTED_RECORD_IDS_SHA256
    ),
    expected_top_record_indices_sha256: str | None = (
        FROZEN_TOP_RECORD_INDICES_SHA256
    ),
    expected_top_record_similarities_sha256: str | None = (
        FROZEN_TOP_RECORD_SIMILARITIES_SHA256
    ),
) -> PointwisePredictions:
    """Construct fixed prior, routing residual, and pointwise equal-RRF ranks."""

    root = Path(refinement_root)
    if expected_refinement_aggregate_sha256 is not None:
        manifest_path = root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        entries = manifest.get("files") if isinstance(manifest, dict) else None
        if not isinstance(entries, dict):
            raise ValueError("refinement manifest schema changed")
        actual_entries = manifest_entries(root)
        observed_aggregate = manifest_aggregate_sha256(actual_entries)
        if (
            entries != actual_entries
            or manifest.get("aggregate_sha256") != observed_aggregate
            or observed_aggregate != expected_refinement_aggregate_sha256
        ):
            raise ValueError("refinement aggregate SHA-256 changed")
    selected_path = root / "selected_record_ids.json"
    indices_path = root / "top_record_indices.npy"
    similarities_path = root / "top_record_similarities.npy"
    _validate_hash(
        selected_path, expected_selected_record_ids_sha256, "selected record IDs"
    )
    _validate_hash(indices_path, expected_top_record_indices_sha256, "top-record indices")
    _validate_hash(
        similarities_path,
        expected_top_record_similarities_sha256,
        "top-record similarities",
    )
    query_ids, candidate_ids = _load_refinement_inputs(root)
    if set(candidate_ids) != set(evidence.candidate_ids):
        raise ValueError("refinement and pointwise candidate inventories differ")
    shape = (len(query_ids), len(candidate_ids), 10)
    indices = np.load(indices_path, allow_pickle=False)
    similarities = np.load(similarities_path, allow_pickle=False)
    if (
        indices.dtype != np.dtype("<i4")
        or indices.shape != shape
        or not indices.flags.c_contiguous
    ):
        raise ValueError("top-record index tensor contract changed")
    if (
        similarities.dtype != np.dtype("<f8")
        or similarities.shape != shape
        or not similarities.flags.c_contiguous
        or not np.isfinite(similarities).all()
    ):
        raise ValueError("top-record similarities must be exact, complete, and finite")
    if np.any(indices < 0) or np.any(indices >= len(evidence.record_ids)):
        raise ValueError("top-record index is out of construction-record range")
    selected_payload = json.loads(selected_path.read_text(encoding="utf-8"))
    if (
        not isinstance(selected_payload, list)
        or len(selected_payload) != len(query_ids)
    ):
        raise ValueError("selected record ID query shape changed")

    local_scores = np.empty((len(query_ids), len(candidate_ids)), dtype=np.float64)
    for query_index, selected_query in enumerate(selected_payload):
        if not isinstance(selected_query, list) or len(selected_query) != len(candidate_ids):
            raise ValueError("selected record ID candidate shape changed")
        for candidate_index, selected_candidate in enumerate(selected_query):
            if (
                not isinstance(selected_candidate, list)
                or len(selected_candidate) != 10
                or len(set(selected_candidate)) != 10
            ):
                raise ValueError("every candidate slot must name exactly 10 record IDs")
            supports: list[float] = []
            for neighbor_index, record_id in enumerate(selected_candidate):
                if not isinstance(record_id, str) or not record_id:
                    raise ValueError("selected record IDs must be non-empty strings")
                label = evidence.labels_by_record_id.get(record_id)
                if label is None:
                    raise ValueError(f"selected record ID has no pointwise label: {record_id!r}")
                if label.candidate_id != candidate_ids[candidate_index]:
                    raise ValueError("selected record ID belongs to the wrong candidate")
                record_index = int(indices[query_index, candidate_index, neighbor_index])
                if evidence.record_ids[record_index] != record_id:
                    raise ValueError(
                        "selected record ID disagrees with frozen top-record index"
                    )
                supports.append(float(label.supported))
            local_scores[query_index, candidate_index] = math.fsum(supports) / 10.0

    priors = np.asarray(
        [evidence.candidate_priors[candidate_id] for candidate_id in candidate_ids],
        dtype=np.float64,
    )
    residuals = np.asarray(local_scores - priors[None, :], dtype=np.float64)
    prior_matrix = np.broadcast_to(priors, local_scores.shape)
    prior_rankings = _rank_scores(query_ids, candidate_ids, prior_matrix)
    routing_rankings = _rank_scores(query_ids, candidate_ids, residuals)
    rrf_scores, pointwise_rankings = _equal_rrf(
        query_ids, candidate_ids, prior_rankings, routing_rankings
    )
    query_similarity_means = {
        query_id: float(np.mean(similarities[index], dtype=np.float64))
        for index, query_id in enumerate(query_ids)
    }
    return PointwisePredictions(
        query_ids,
        candidate_ids,
        priors,
        local_scores,
        residuals,
        rrf_scores,
        prior_rankings,
        routing_rankings,
        pointwise_rankings,
        query_similarity_means,
        stable_quartile_labels(query_similarity_means),
        stable_quartile_labels(
            {
                candidate_id: float(evidence.audited_counts[candidate_id])
                for candidate_id in candidate_ids
            }
        ),
        stable_quartile_labels(
            {candidate_id: evidence.candidate_priors[candidate_id] for candidate_id in candidate_ids}
        ),
    )


def _selector_hash(question_id: str, candidate_ids: tuple[str, str]) -> str:
    payload = {
        "candidate_ids": list(candidate_ids),
        "question_id": question_id,
        "selector_version": SELECTOR_VERSION,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _eligible_pairs(evidence: PointwiseEvidence) -> tuple[_EligiblePair, ...]:
    grouped: dict[tuple[str, str], list[PointwiseLabel]] = defaultdict(list)
    for label in evidence.labels_by_record_id.values():
        if label.supported:
            grouped[(label.question_id, label.verdict)].append(label)
    output: list[_EligiblePair] = []
    for (question_id, verdict), labels in grouped.items():
        by_candidate: dict[str, PointwiseLabel] = {}
        for label in labels:
            if label.candidate_id in by_candidate:
                raise ValueError(
                    "selector requires one supported record per question and candidate"
                )
            by_candidate[label.candidate_id] = label
        for left_id, right_id in combinations(sorted(by_candidate), 2):
            left = by_candidate[left_id]
            right = by_candidate[right_id]
            candidate_ids = (left_id, right_id)
            output.append(
                _EligiblePair(
                    question_id,
                    candidate_ids,
                    (left.record_id, right.record_id),
                    0 if verdict == "partial" else 1,
                    abs(
                        evidence.candidate_priors[left_id]
                        - evidence.candidate_priors[right_id]
                    ),
                    _selector_hash(question_id, candidate_ids),
                )
            )
    return tuple(output)


def select_pairwise_candidates(
    evidence: PointwiseEvidence,
    *,
    target_pair_count: int = SELECTOR_TARGET_PAIRS,
) -> PairSelectionReceipt:
    """Run the frozen dynamic, diversity-first, cap-escalating zero-call selector."""

    if (
        isinstance(target_pair_count, bool)
        or not isinstance(target_pair_count, int)
        or target_pair_count < 1
    ):
        raise ValueError("target pair count must be a positive integer")
    eligible = _eligible_pairs(evidence)
    priority_groups: dict[tuple[int, float], list[int]] = defaultdict(list)
    for index, pair in enumerate(eligible):
        priority_groups[(pair.tier, pair.prior_distance)].append(index)
    ordered_priority_groups = tuple(
        (key, tuple(indices)) for key, indices in sorted(priority_groups.items())
    )
    selected_indices: set[int] = set()
    selected: list[SelectedPair] = []
    uses = {candidate_id: 0 for candidate_id in evidence.candidate_ids}
    cap = SELECTOR_INITIAL_CAP

    def dynamic_priority(index: int) -> tuple[Any, ...]:
        pair = eligible[index]
        left_id, right_id = pair.candidate_ids
        left_use = uses[left_id]
        right_use = uses[right_id]
        use_key = (
            max(left_use, right_use),
            left_use + right_use,
            left_use,
            right_use,
        )
        return use_key, pair.selector_hash

    def next_available(*, excluded_questions: set[str] | None = None) -> int | None:
        for _, group in ordered_priority_groups:
            choices = (
                index
                for index in group
                if index not in selected_indices
                and (
                    excluded_questions is None
                    or eligible[index].question_id not in excluded_questions
                )
                and all(
                    uses[candidate_id] < cap
                    for candidate_id in eligible[index].candidate_ids
                )
            )
            chosen = min(choices, key=dynamic_priority, default=None)
            if chosen is not None:
                return chosen
        return None

    while len(selected) < target_pair_count:
        pass_questions: set[str] = set()
        while len(selected) < target_pair_count:
            chosen = next_available(excluded_questions=pass_questions)
            if chosen is None:
                break
            pair = eligible[chosen]
            selected_indices.add(chosen)
            pass_questions.add(pair.question_id)
            for candidate_id in pair.candidate_ids:
                uses[candidate_id] += 1
            selected.append(
                SelectedPair(
                    pair.question_id,
                    pair.candidate_ids,
                    pair.record_ids,
                    cap,
                )
            )
        while len(selected) < target_pair_count:
            chosen = next_available()
            if chosen is None:
                break
            pair = eligible[chosen]
            selected_indices.add(chosen)
            for candidate_id in pair.candidate_ids:
                uses[candidate_id] += 1
            selected.append(
                SelectedPair(
                    pair.question_id,
                    pair.candidate_ids,
                    pair.record_ids,
                    cap,
                )
            )
        if len(selected) >= target_pair_count or len(selected_indices) == len(eligible):
            break
        cap += 1

    return PairSelectionReceipt(
        SELECTOR_VERSION,
        target_pair_count,
        len(eligible),
        len({pair.question_id for pair in eligible}),
        len(selected),
        len({pair.question_id for pair in selected}),
        SELECTOR_INITIAL_CAP,
        cap,
        uses,
        tuple(selected),
    )


__all__ = [
    "SELECTOR_VERSION",
    "PointwiseEvidence",
    "PointwiseLabel",
    "PointwisePredictions",
    "PairSelectionReceipt",
    "SelectedPair",
    "construct_pointwise_rankings",
    "load_pointwise_evidence",
    "select_pairwise_candidates",
    "stable_quartile_labels",
]
