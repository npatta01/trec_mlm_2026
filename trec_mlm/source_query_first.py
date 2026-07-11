"""Deterministic source-query-first lexical response scoring.

This module implements the qrel-free scoring layer locked in
``protocols/trec_2026_source_query_first.md``.  It retrieves unique historical
source queries first, gives every candidate model the same response slots, and
keeps source relevance, answer alignment, and direct target-response relevance
as separate tensors.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
import hashlib
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.response_retrieval import (
    AGGRESSIVE_PREFIXES,
    BM25_B,
    BM25_K1,
    OFFICIAL_DISCOVERY_SHA256,
    OFFICIAL_RAW_DOCUMENT_COUNT,
    AnalyzerDiagnostics,
    DiscoveryCorpus,
    DiscoveryRecord,
    FrozenAnalyzer,
    _hash_arrays,
    _hash_json,
    _hash_records,
    _validate_model_ids,
    _validate_records_for_scoring,
    _validate_targets,
    normalize_filter_text,
)


SOURCE_QUERY_K_VALUES = (10, 50, 100)
EXPECTED_SOURCE_QUERY_COUNT = 7_759
EXPECTED_CANONICAL_QUERY_COUNT = 7_693
EXPECTED_DUPLICATE_KEY_GROUP_COUNT = 58
EXPECTED_NONCANONICAL_SOURCE_ID_COUNT = 66
EXPECTED_BASIC_RETAINED_COUNT = 182_800
EXPECTED_BASIC_PLUS_PREFIX_RETAINED_COUNT = 182_798

VALIDITY_VALID = np.uint8(0)
VALIDITY_NULL = np.uint8(1)
VALIDITY_BLANK = np.uint8(2)
VALIDITY_EXACT_IDK = np.uint8(3)
VALIDITY_PREFIX_REFUSAL = np.uint8(4)
VALIDITY_REASON_NAMES = (
    "valid",
    "null",
    "blank",
    "exact_idk",
    "prefix_refusal",
)


@dataclass(frozen=True, slots=True)
class SourceQueryCandidate:
    """One preregistered source-query depth."""

    k: int
    # Compatibility with the existing response nested-diagnostic helpers.  The
    # frozen gates continue to use the PR #8 BASIC inventory vector.
    filter_name: str = field(init=False, default="basic", repr=False)

    def __post_init__(self) -> None:
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError("source-query K must be a positive integer")

    @property
    def key(self) -> str:
        return f"source_alignment__k{self.k}"


DEFAULT_SOURCE_QUERY_CANDIDATES = tuple(
    SourceQueryCandidate(value) for value in SOURCE_QUERY_K_VALUES
)


@dataclass(frozen=True, slots=True)
class CanonicalQueryAudit:
    source_query_id_count: int
    nonempty_analyzer_key_count: int
    canonical_source_query_count: int
    duplicate_key_group_count: int
    noncanonical_source_id_count: int
    empty_analyzer_key_source_id_count: int


@dataclass(frozen=True, slots=True)
class OpportunityFilterAudit:
    input_response_cells: int
    eligible_response_cells: int
    zero_null: int
    zero_blank: int
    zero_exact_idk: int
    zero_prefix_refusal: int
    rule_name: str = "basic_plus_prefix"

    @property
    def zero_response_cells(self) -> int:
        return self.input_response_cells - self.eligible_response_cells


@dataclass(frozen=True, slots=True)
class SourceIndexDiagnostics:
    document_count: int
    average_document_length: float
    analyzed_token_count: int
    empty_document_count: int
    vocabulary_size: int
    vocabulary_sha256: str
    index_sha256: str


@dataclass(frozen=True, slots=True)
class LocalResponseIndexDiagnostics:
    built_source_query_count: int
    slot_count_per_source_query: int
    all_empty_source_query_count: int
    combined_index_sha256: str


@dataclass(slots=True)
class SourceQueryFirstScoreResult:
    """All qrel-free signals, primary scores, diagnostics, and hashes."""

    candidates: tuple[SourceQueryCandidate, ...]
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    selected_source_ids: tuple[tuple[str, ...], ...]
    selected_source_keys: tuple[tuple[str, ...], ...]
    source_relevance_raw: NDArray[np.float64]
    source_weights: NDArray[np.float64]
    answer_alignment_raw: NDArray[np.float64]
    answer_alignment_normalized: NDArray[np.float64]
    target_response_raw: NDArray[np.float64]
    target_response_normalized: NDArray[np.float64]
    model_scores: NDArray[np.float64]
    diagnostic_model_scores: NDArray[np.float64]
    opportunity_counts: NDArray[np.int32]
    valid_opportunity_counts: NDArray[np.int32]
    validity_codes: NDArray[np.uint8]
    validity_reason_names: tuple[str, ...]
    exact_mask_canonical_counts: NDArray[np.int32]
    exact_mask_source_id_counts: NDArray[np.int32]
    positive_retrieval_counts: NDArray[np.int32]
    source_weight_sums: NDArray[np.float64]
    basic_retained_document_counts: NDArray[np.int64]
    basic_retained_character_counts: NDArray[np.int64]
    basic_retained_token_counts: NDArray[np.int64]
    basic_plus_prefix_retained_document_counts: NDArray[np.int64]
    basic_plus_prefix_retained_character_counts: NDArray[np.int64]
    basic_plus_prefix_retained_token_counts: NDArray[np.int64]
    inventory_filter_names: tuple[str, ...]
    retained_document_counts: NDArray[np.int64]
    retained_character_counts: NDArray[np.int64]
    retained_token_counts: NDArray[np.int64]
    canonical_audit: CanonicalQueryAudit
    filter_audit: OpportunityFilterAudit
    filter_audits: tuple[OpportunityFilterAudit, ...]
    index_diagnostics: tuple[
        SourceIndexDiagnostics | LocalResponseIndexDiagnostics, ...
    ]
    analyzer_diagnostics: AnalyzerDiagnostics
    source_sha256: str
    corpus_sha256: str
    canonical_query_sha256: str
    response_cell_sha256: str
    configuration_sha256: str
    signal_tensor_sha256: str
    score_tensor_sha256: str

    @property
    def hit_counts(self) -> NDArray[np.int32]:
        """Compatibility alias used by nested response diagnostics."""

        return self.valid_opportunity_counts

    @property
    def mean_hit_scores(self) -> None:
        return None

    @property
    def overlap_document_counts(self) -> NDArray[np.int32]:
        return np.broadcast_to(
            self.exact_mask_canonical_counts,
            (len(self.candidates), len(self.query_ids)),
        )


@dataclass(frozen=True, slots=True)
class _FrozenBM25Index:
    """A small deterministic Lucene-BM25 posting index."""

    document_ids: tuple[str, ...]
    postings: dict[str, tuple[NDArray[np.int32], NDArray[np.float64]]]
    average_document_length: float
    analyzed_token_count: int
    empty_document_count: int
    vocabulary: tuple[str, ...]
    vocabulary_sha256: str
    index_sha256: str

    @classmethod
    def build(
        cls,
        document_ids: Sequence[str],
        document_tokens: Sequence[Sequence[str]],
    ) -> _FrozenBM25Index:
        ids = tuple(document_ids)
        tokens = tuple(tuple(value) for value in document_tokens)
        if not ids or len(ids) != len(tokens):
            raise ValueError(
                "BM25 document IDs and tokens must have equal nonzero length"
            )
        if len(set(ids)) != len(ids):
            raise ValueError("BM25 document IDs must be unique")
        if any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("BM25 document IDs must be non-empty strings")

        lengths = np.asarray([len(value) for value in tokens], dtype=np.float64)
        average_length = float(lengths.mean())
        term_rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for document_index, values in enumerate(tokens):
            for term, frequency in Counter(values).items():
                term_rows[term].append((document_index, frequency))

        postings: dict[str, tuple[NDArray[np.int32], NDArray[np.float64]]] = {}
        document_count = len(ids)
        if average_length > 0.0:
            for term in sorted(term_rows):
                rows = term_rows[term]
                document_frequency = len(rows)
                idf = math.log(
                    1.0
                    + (document_count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                indices = np.asarray([row[0] for row in rows], dtype=np.int32)
                weights = np.empty(len(rows), dtype=np.float64)
                for position, (document_index, frequency) in enumerate(rows):
                    length_ratio = lengths[document_index] / average_length
                    denominator = float(frequency) + BM25_K1 * (
                        1.0 - BM25_B + BM25_B * length_ratio
                    )
                    # Lucene's implementation omits the query-constant
                    # ``k1 + 1`` multiplier.  This matches bm25s's pinned
                    # ``method='lucene'`` scores exactly.
                    weights[position] = idf * (float(frequency) / denominator)
                indices.setflags(write=False)
                weights.setflags(write=False)
                postings[term] = (indices, weights)

        vocabulary = tuple(sorted(term_rows))
        vocabulary_sha256 = _hash_json(list(vocabulary))
        digest = hashlib.sha256()
        digest.update(_hash_json(list(ids)).encode("ascii"))
        digest.update(vocabulary_sha256.encode("ascii"))
        digest.update(np.asarray(lengths, dtype=np.float64).tobytes())
        for term in vocabulary:
            digest.update(term.encode("utf-8"))
            if term in postings:
                indices, weights = postings[term]
                digest.update(indices.tobytes())
                digest.update(weights.tobytes())
        return cls(
            document_ids=ids,
            postings=postings,
            average_document_length=average_length,
            analyzed_token_count=int(lengths.sum()),
            empty_document_count=int(np.count_nonzero(lengths == 0.0)),
            vocabulary=vocabulary,
            vocabulary_sha256=vocabulary_sha256,
            index_sha256=digest.hexdigest(),
        )

    def score(self, query_tokens: Sequence[str]) -> NDArray[np.float64]:
        """Score one analyzed query, preserving analyzed-term accumulation order."""

        scores = np.zeros(len(self.document_ids), dtype=np.float64)
        if self.average_document_length == 0.0 or not query_tokens:
            return scores
        contributions: dict[int, list[float]] = defaultdict(list)
        for term in query_tokens:
            posting = self.postings.get(term)
            if posting is None:
                continue
            indices, values = posting
            for document_index, value in zip(indices, values, strict=True):
                contributions[int(document_index)].append(float(value))
        for document_index, values in contributions.items():
            score = math.fsum(values)
            if math.isfinite(score) and score > 0.0:
                scores[document_index] = score
        return scores


def response_validity_code(response: str | None) -> np.uint8:
    """Return the frozen ``basic_plus_prefix`` validity code."""

    if response is None:
        return VALIDITY_NULL
    if not response.strip():
        return VALIDITY_BLANK
    normalized = normalize_filter_text(response)
    if normalized == "i don't know":
        return VALIDITY_EXACT_IDK
    if normalized.startswith(AGGRESSIVE_PREFIXES):
        return VALIDITY_PREFIX_REFUSAL
    return VALIDITY_VALID


def score_source_query_first(
    records: DiscoveryCorpus | Sequence[DiscoveryRecord],
    *,
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    expected_model_ids: Sequence[str],
    candidates: Sequence[SourceQueryCandidate] = DEFAULT_SOURCE_QUERY_CANDIDATES,
    show_progress: bool = False,
) -> SourceQueryFirstScoreResult:
    """Compute all frozen source-query-first signals without reading qrels."""

    del show_progress  # Kept as a CLI-compatible, behavior-neutral option.
    target_ids, target_texts = _validate_targets(query_ids, query_texts)
    model_ids = _validate_model_ids(expected_model_ids)
    record_tuple, source_sha256 = _validate_records_for_scoring(records, model_ids)
    candidate_tuple = _validate_candidates(candidates)
    analyzer = FrozenAnalyzer()

    records_by_source, source_texts = _group_records(record_tuple, model_ids)
    (
        canonical_source_ids,
        canonical_source_keys,
        key_to_source_ids,
        canonical_audit,
    ) = _canonical_sources(source_texts, analyzer)

    source_tokens = tuple(
        analyzer.analyze(source_texts[source_id]) for source_id in canonical_source_ids
    )
    source_index = _FrozenBM25Index.build(canonical_source_ids, source_tokens)
    source_diagnostics = SourceIndexDiagnostics(
        document_count=len(canonical_source_ids),
        average_document_length=source_index.average_document_length,
        analyzed_token_count=source_index.analyzed_token_count,
        empty_document_count=source_index.empty_document_count,
        vocabulary_size=len(source_index.vocabulary),
        vocabulary_sha256=source_index.vocabulary_sha256,
        index_sha256=source_index.index_sha256,
    )

    maximum_k = max(candidate.k for candidate in candidate_tuple)
    query_count = len(target_ids)
    model_count = len(model_ids)
    candidate_count = len(candidate_tuple)
    selected_indices = np.full((query_count, maximum_k), -1, dtype=np.int32)
    source_relevance_raw = np.zeros((query_count, maximum_k), dtype=np.float64)
    source_weights = np.zeros(
        (candidate_count, query_count, maximum_k), dtype=np.float64
    )
    exact_mask_canonical_counts = np.zeros(query_count, dtype=np.int32)
    exact_mask_source_id_counts = np.zeros(query_count, dtype=np.int32)
    positive_retrieval_counts = np.zeros(query_count, dtype=np.int32)
    canonical_id_to_index = {
        source_id: index for index, source_id in enumerate(canonical_source_ids)
    }

    for query_index, query_text in enumerate(target_texts):
        query_tokens = analyzer.analyze(query_text)
        target_key = analyzer.overlap_key(query_text)
        masked_ids = key_to_source_ids.get(target_key, ())
        exact_mask_source_id_counts[query_index] = len(masked_ids)
        canonical_mask = (
            {canonical_id_to_index[min(masked_ids)]} if masked_ids else set()
        )
        exact_mask_canonical_counts[query_index] = len(canonical_mask)
        scores = source_index.score(query_tokens)
        if canonical_mask:
            scores[np.asarray(sorted(canonical_mask), dtype=np.int32)] = 0.0
        positive_indices = np.flatnonzero(scores > 0.0)
        positive_retrieval_counts[query_index] = len(positive_indices)
        if len(positive_indices) < maximum_k:
            raise ValueError(
                f"target query {target_ids[query_index]!r} has only "
                f"{len(positive_indices)} positive source queries; needs {maximum_k}"
            )
        ordered = sorted(
            (int(value) for value in positive_indices),
            key=lambda value: (-float(scores[value]), canonical_source_ids[value]),
        )[:maximum_k]
        selected_indices[query_index] = np.asarray(ordered, dtype=np.int32)
        source_relevance_raw[query_index] = scores[selected_indices[query_index]]
        for candidate_index, candidate in enumerate(candidate_tuple):
            prefix = source_relevance_raw[query_index, : candidate.k]
            source_weights[candidate_index, query_index, : candidate.k] = _l1(prefix)

    selected_source_ids = tuple(
        tuple(canonical_source_ids[int(index)] for index in row)
        for row in selected_indices
    )
    selected_source_keys = tuple(
        tuple(canonical_source_keys[int(index)] for index in row)
        for row in selected_indices
    )

    inventory = _build_inventories(record_tuple, model_ids, analyzer)
    filter_audit = inventory["filter_audit"]
    if source_sha256 == OFFICIAL_DISCOVERY_SHA256:
        _validate_official_structural_counts(canonical_audit, filter_audit)

    shape = (query_count, maximum_k, model_count)
    answer_alignment_raw = np.zeros(shape, dtype=np.float64)
    answer_alignment_normalized = np.zeros(shape, dtype=np.float64)
    target_response_raw = np.zeros(shape, dtype=np.float64)
    target_response_normalized = np.zeros(shape, dtype=np.float64)
    validity_codes = np.zeros(shape, dtype=np.uint8)
    target_tokens = tuple(analyzer.analyze(text) for text in target_texts)

    occurrences: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for query_index, row in enumerate(selected_source_ids):
        for rank_index, source_id in enumerate(row):
            occurrences[source_id].append((query_index, rank_index))

    local_digest = hashlib.sha256()
    all_empty_local_count = 0
    for source_id in sorted(occurrences):
        rows = records_by_source[source_id]
        codes = np.asarray(
            [response_validity_code(row.response) for row in rows], dtype=np.uint8
        )
        response_tokens = tuple(
            analyzer.analyze(row.response or "")
            if codes[index] == VALIDITY_VALID
            else ()
            for index, row in enumerate(rows)
        )
        local_index = _FrozenBM25Index.build(model_ids, response_tokens)
        local_digest.update(source_id.encode("utf-8"))
        local_digest.update(local_index.index_sha256.encode("ascii"))
        if local_index.average_document_length == 0.0:
            all_empty_local_count += 1
        alignment = local_index.score(analyzer.analyze(source_texts[source_id]))
        normalized_alignment = _l1(alignment)
        for query_index, rank_index in occurrences[source_id]:
            direct = local_index.score(target_tokens[query_index])
            answer_alignment_raw[query_index, rank_index] = alignment
            answer_alignment_normalized[query_index, rank_index] = normalized_alignment
            target_response_raw[query_index, rank_index] = direct
            target_response_normalized[query_index, rank_index] = _l1(direct)
            validity_codes[query_index, rank_index] = codes

    local_diagnostics = LocalResponseIndexDiagnostics(
        built_source_query_count=len(occurrences),
        slot_count_per_source_query=model_count,
        all_empty_source_query_count=all_empty_local_count,
        combined_index_sha256=local_digest.hexdigest(),
    )

    model_scores = np.zeros(
        (candidate_count, query_count, model_count), dtype=np.float64
    )
    diagnostic_model_scores = np.zeros_like(model_scores)
    opportunity_counts = np.zeros(
        (candidate_count, query_count, model_count), dtype=np.int32
    )
    valid_opportunity_counts = np.zeros_like(opportunity_counts)
    source_weight_sums = np.zeros((candidate_count, query_count), dtype=np.float64)
    valid_cells = validity_codes == VALIDITY_VALID
    for candidate_index, candidate in enumerate(candidate_tuple):
        for query_index in range(query_count):
            weights = source_weights[candidate_index, query_index, : candidate.k]
            source_weight_sums[candidate_index, query_index] = math.fsum(
                float(value) for value in weights
            )
            opportunity_counts[candidate_index, query_index] = candidate.k
            valid_opportunity_counts[candidate_index, query_index] = np.sum(
                valid_cells[query_index, : candidate.k], axis=0, dtype=np.int32
            )
            for model_index in range(model_count):
                model_scores[candidate_index, query_index, model_index] = math.fsum(
                    float(weights[source_index])
                    * float(
                        answer_alignment_normalized[
                            query_index, source_index, model_index
                        ]
                    )
                    for source_index in range(candidate.k)
                )
                diagnostic_model_scores[candidate_index, query_index, model_index] = (
                    math.fsum(
                        float(weights[source_index])
                        * float(
                            target_response_normalized[
                                query_index, source_index, model_index
                            ]
                        )
                        for source_index in range(candidate.k)
                    )
                )

    canonical_query_sha256 = _hash_json(
        [
            {
                "source_query_id": source_id,
                "overlap_key": key,
                "query_text": source_texts[source_id],
            }
            for source_id, key in zip(
                canonical_source_ids, canonical_source_keys, strict=True
            )
        ]
    )
    response_cell_sha256 = _hash_arrays(validity_codes)
    configuration_sha256 = _hash_json(
        {
            "candidate_keys": [candidate.key for candidate in candidate_tuple],
            "analyzer_sha256": analyzer.diagnostics.analyzer_sha256,
            "bm25": {
                "formula": "lucene",
                "idf": "log(1+(N-df+0.5)/(df+0.5))",
                "k1": BM25_K1,
                "b": BM25_B,
                "dtype": "float64",
            },
            "canonicalization": (
                "one_nonempty_analyzer_key_lexicographically_smallest_source_id"
            ),
            "response_corpus": "local_150_slots_invalid_cells_empty",
            "invalid_rule": "basic_plus_frozen_21_prefixes_no_short_cutoff",
            "normalization": "separate_l1_no_epsilon",
            "primary_formula": "math.fsum_j(w_j*a_jm)",
            "diagnostic_formula": "math.fsum_j(w_j*t_jm)",
        }
    )
    signal_tensor_sha256 = _hash_arrays(
        source_relevance_raw,
        source_weights,
        answer_alignment_raw,
        answer_alignment_normalized,
        target_response_raw,
        target_response_normalized,
        validity_codes,
    )
    score_tensor_sha256 = _hash_arrays(
        model_scores,
        diagnostic_model_scores,
        opportunity_counts,
        valid_opportunity_counts,
    )

    arrays: tuple[NDArray[Any], ...] = (
        source_relevance_raw,
        source_weights,
        answer_alignment_raw,
        answer_alignment_normalized,
        target_response_raw,
        target_response_normalized,
        model_scores,
        diagnostic_model_scores,
        opportunity_counts,
        valid_opportunity_counts,
        validity_codes,
        exact_mask_canonical_counts,
        exact_mask_source_id_counts,
        positive_retrieval_counts,
        source_weight_sums,
        *inventory["arrays"],
    )
    for array in arrays:
        array.setflags(write=False)

    return SourceQueryFirstScoreResult(
        candidates=candidate_tuple,
        query_ids=target_ids,
        model_ids=model_ids,
        selected_source_ids=selected_source_ids,
        selected_source_keys=selected_source_keys,
        source_relevance_raw=source_relevance_raw,
        source_weights=source_weights,
        answer_alignment_raw=answer_alignment_raw,
        answer_alignment_normalized=answer_alignment_normalized,
        target_response_raw=target_response_raw,
        target_response_normalized=target_response_normalized,
        model_scores=model_scores,
        diagnostic_model_scores=diagnostic_model_scores,
        opportunity_counts=opportunity_counts,
        valid_opportunity_counts=valid_opportunity_counts,
        validity_codes=validity_codes,
        validity_reason_names=VALIDITY_REASON_NAMES,
        exact_mask_canonical_counts=exact_mask_canonical_counts,
        exact_mask_source_id_counts=exact_mask_source_id_counts,
        positive_retrieval_counts=positive_retrieval_counts,
        source_weight_sums=source_weight_sums,
        basic_retained_document_counts=inventory["basic_documents"],
        basic_retained_character_counts=inventory["basic_characters"],
        basic_retained_token_counts=inventory["basic_tokens"],
        basic_plus_prefix_retained_document_counts=inventory["primary_documents"],
        basic_plus_prefix_retained_character_counts=inventory["primary_characters"],
        basic_plus_prefix_retained_token_counts=inventory["primary_tokens"],
        inventory_filter_names=("basic", "basic_plus_prefix"),
        retained_document_counts=inventory["document_stack"],
        retained_character_counts=inventory["character_stack"],
        retained_token_counts=inventory["token_stack"],
        canonical_audit=canonical_audit,
        filter_audit=filter_audit,
        filter_audits=(filter_audit,),
        index_diagnostics=(source_diagnostics, local_diagnostics),
        analyzer_diagnostics=analyzer.diagnostics,
        source_sha256=source_sha256,
        corpus_sha256=_hash_records(record_tuple),
        canonical_query_sha256=canonical_query_sha256,
        response_cell_sha256=response_cell_sha256,
        configuration_sha256=configuration_sha256,
        signal_tensor_sha256=signal_tensor_sha256,
        score_tensor_sha256=score_tensor_sha256,
    )


def _validate_candidates(
    candidates: Sequence[SourceQueryCandidate],
) -> tuple[SourceQueryCandidate, ...]:
    values = tuple(candidates)
    if not values or any(
        not isinstance(value, SourceQueryCandidate) for value in values
    ):
        raise TypeError("candidates must contain SourceQueryCandidate values")
    if len({value.key for value in values}) != len(values):
        raise ValueError("source-query candidates must be unique")
    if tuple(sorted(value.k for value in values)) != tuple(value.k for value in values):
        raise ValueError("source-query candidates must be ordered by increasing K")
    return values


def _group_records(
    records: Sequence[DiscoveryRecord],
    model_ids: tuple[str, ...],
) -> tuple[dict[str, tuple[DiscoveryRecord, ...]], dict[str, str]]:
    grouped: dict[str, list[DiscoveryRecord]] = defaultdict(list)
    source_texts: dict[str, str] = {}
    for record in records:
        grouped[record.source_query_id].append(record)
        source_texts.setdefault(record.source_query_id, record.query_text)
    output: dict[str, tuple[DiscoveryRecord, ...]] = {}
    for source_id in sorted(grouped):
        rows = tuple(grouped[source_id])
        if tuple(row.model_id for row in rows) != model_ids:
            raise ValueError(f"source query {source_id!r} is not in model-ID order")
        output[source_id] = rows
    return output, source_texts


def _canonical_sources(
    source_texts: dict[str, str],
    analyzer: FrozenAnalyzer,
) -> tuple[
    tuple[str, ...], tuple[str, ...], dict[str, tuple[str, ...]], CanonicalQueryAudit
]:
    key_lists: dict[str, list[str]] = defaultdict(list)
    empty_count = 0
    for source_id in sorted(source_texts):
        key = analyzer.overlap_key(source_texts[source_id])
        if not key:
            empty_count += 1
            continue
        key_lists[key].append(source_id)
    key_to_ids = {
        key: tuple(sorted(values)) for key, values in sorted(key_lists.items())
    }
    canonical_pairs = sorted(
        ((values[0], key) for key, values in key_to_ids.items()),
        key=lambda item: item[0],
    )
    canonical_ids = tuple(item[0] for item in canonical_pairs)
    canonical_keys = tuple(item[1] for item in canonical_pairs)
    duplicate_groups = sum(len(values) > 1 for values in key_to_ids.values())
    noncanonical = sum(len(values) - 1 for values in key_to_ids.values())
    audit = CanonicalQueryAudit(
        source_query_id_count=len(source_texts),
        nonempty_analyzer_key_count=len(key_to_ids),
        canonical_source_query_count=len(canonical_ids),
        duplicate_key_group_count=duplicate_groups,
        noncanonical_source_id_count=noncanonical,
        empty_analyzer_key_source_id_count=empty_count,
    )
    return canonical_ids, canonical_keys, key_to_ids, audit


def _l1(values: Sequence[float] | NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not np.isfinite(array).all() or np.any(array < 0.0):
        raise ValueError("L1 inputs must be a finite nonnegative vector")
    denominator = math.fsum(float(value) for value in array)
    if denominator <= 0.0:
        return np.zeros_like(array)
    return array / denominator


def _build_inventories(
    records: Sequence[DiscoveryRecord],
    model_ids: tuple[str, ...],
    analyzer: FrozenAnalyzer,
) -> dict[str, Any]:
    model_position = {model_id: index for index, model_id in enumerate(model_ids)}
    shape = (len(model_ids),)
    basic_documents = np.zeros(shape, dtype=np.int64)
    basic_characters = np.zeros(shape, dtype=np.int64)
    basic_tokens = np.zeros(shape, dtype=np.int64)
    primary_documents = np.zeros(shape, dtype=np.int64)
    primary_characters = np.zeros(shape, dtype=np.int64)
    primary_tokens = np.zeros(shape, dtype=np.int64)
    validity_counts: Counter[int] = Counter()
    for record in records:
        code = int(response_validity_code(record.response))
        validity_counts[code] += 1
        basic_valid = code not in {
            int(VALIDITY_NULL),
            int(VALIDITY_BLANK),
            int(VALIDITY_EXACT_IDK),
        }
        primary_valid = code == int(VALIDITY_VALID)
        if not basic_valid:
            continue
        assert record.response is not None
        position = model_position[record.model_id]
        tokens = analyzer.analyze(record.response)
        characters = len(record.response.strip())
        basic_documents[position] += 1
        basic_characters[position] += characters
        basic_tokens[position] += len(tokens)
        if primary_valid:
            primary_documents[position] += 1
            primary_characters[position] += characters
            primary_tokens[position] += len(tokens)

    input_count = len(records)
    filter_audit = OpportunityFilterAudit(
        input_response_cells=input_count,
        eligible_response_cells=validity_counts[int(VALIDITY_VALID)],
        zero_null=validity_counts[int(VALIDITY_NULL)],
        zero_blank=validity_counts[int(VALIDITY_BLANK)],
        zero_exact_idk=validity_counts[int(VALIDITY_EXACT_IDK)],
        zero_prefix_refusal=validity_counts[int(VALIDITY_PREFIX_REFUSAL)],
    )
    document_stack = np.stack((basic_documents, primary_documents))
    character_stack = np.stack((basic_characters, primary_characters))
    token_stack = np.stack((basic_tokens, primary_tokens))
    return {
        "basic_documents": basic_documents,
        "basic_characters": basic_characters,
        "basic_tokens": basic_tokens,
        "primary_documents": primary_documents,
        "primary_characters": primary_characters,
        "primary_tokens": primary_tokens,
        "document_stack": document_stack,
        "character_stack": character_stack,
        "token_stack": token_stack,
        "filter_audit": filter_audit,
        "arrays": (
            basic_documents,
            basic_characters,
            basic_tokens,
            primary_documents,
            primary_characters,
            primary_tokens,
            document_stack,
            character_stack,
            token_stack,
        ),
    }


def _validate_official_structural_counts(
    canonical: CanonicalQueryAudit,
    filter_audit: OpportunityFilterAudit,
) -> None:
    expected_canonical = {
        "source_query_id_count": EXPECTED_SOURCE_QUERY_COUNT,
        "canonical_source_query_count": EXPECTED_CANONICAL_QUERY_COUNT,
        "duplicate_key_group_count": EXPECTED_DUPLICATE_KEY_GROUP_COUNT,
        "noncanonical_source_id_count": EXPECTED_NONCANONICAL_SOURCE_ID_COUNT,
        "empty_analyzer_key_source_id_count": 0,
    }
    for name, expected in expected_canonical.items():
        if getattr(canonical, name) != expected:
            raise RuntimeError(
                f"official source-query audit changed for {name}: "
                f"{getattr(canonical, name)} != {expected}"
            )
    if filter_audit.input_response_cells != OFFICIAL_RAW_DOCUMENT_COUNT:
        raise RuntimeError("official response-cell count changed")
    if filter_audit.zero_exact_idk != 981_050:
        raise RuntimeError("official exact-IDK count changed")
    if filter_audit.zero_prefix_refusal != 2:
        raise RuntimeError("official frozen-prefix refusal count changed")
    if (
        filter_audit.eligible_response_cells
        != EXPECTED_BASIC_PLUS_PREFIX_RETAINED_COUNT
    ):
        raise RuntimeError("official basic_plus_prefix eligible count changed")
    if filter_audit.zero_null or filter_audit.zero_blank:
        raise RuntimeError("official null/blank response counts changed")
    basic_count = (
        filter_audit.eligible_response_cells + filter_audit.zero_prefix_refusal
    )
    if basic_count != EXPECTED_BASIC_RETAINED_COUNT:
        raise RuntimeError("official BASIC retained-response count changed")
