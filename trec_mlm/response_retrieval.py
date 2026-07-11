"""Frozen response-level BM25/RM3 retrieval for TREC Million LLMs 2026.

The discovery responses are organizer-provided, qrel-free evidence.  This module
builds deterministic response indexes and returns complete model-score vectors;
fold-local qrel priors and effectiveness evaluation deliberately live in the
experiment layer.

``bm25s`` and ``PyStemmer`` are optional dependencies.  They are imported only
when the analyzer or scorer is constructed, so the rest of :mod:`trec_mlm`
continues to work without the ``response`` extra.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import hashlib
from importlib import metadata as importlib_metadata
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, overload
import unicodedata

import numpy as np
from numpy.typing import NDArray


Method = Literal["bm25", "bm25_rm3"]
Field = Literal["response_only", "query_response"]
FilterName = Literal["basic", "aggressive"]
AggregationName = Literal[
    "sum_all",
    "best_hit",
    "top3_sum",
    "top10_sum",
    "top3_mean",
    "top10_mean",
    "mean_all",
]

METHODS: tuple[Method, ...] = ("bm25", "bm25_rm3")
FIELDS: tuple[Field, ...] = ("response_only", "query_response")
FILTER_NAMES: tuple[FilterName, ...] = ("basic", "aggressive")
DEPTHS = (100, 500, 1000)

SUM_ALL_AGGREGATION: AggregationName = "sum_all"
NORMALIZED_RESPONSE_AGGREGATIONS: tuple[AggregationName, ...] = (
    "best_hit",
    "top3_sum",
    "top10_sum",
    "top3_mean",
    "top10_mean",
    "mean_all",
)
ALL_RESPONSE_AGGREGATIONS: tuple[AggregationName, ...] = (
    SUM_ALL_AGGREGATION,
    *NORMALIZED_RESPONSE_AGGREGATIONS,
)

# Immutable, executable-spec-adjacent descriptions included in every
# configuration hash.  The implementation below follows these definitions
# with float64 values and ``math.fsum`` exactly.
RESPONSE_AGGREGATION_DEFINITIONS: tuple[tuple[AggregationName, str], ...] = (
    ("sum_all", "math.fsum(H) if n > 0 else 0.0"),
    ("best_hit", "s1 if n > 0 else 0.0"),
    ("top3_sum", "math.fsum(H[:min(3,n)]) if n > 0 else 0.0"),
    ("top10_sum", "math.fsum(H[:min(10,n)]) if n > 0 else 0.0"),
    ("top3_mean", "top3_sum / min(3,n) if n > 0 else 0.0"),
    ("top10_mean", "top10_sum / min(10,n) if n > 0 else 0.0"),
    ("mean_all", "sum_all / n if n > 0 else 0.0"),
)

ANALYZER_REGEX = r"(?u)\b\w\w+\b"
ANALYZER_STOPWORDS_SHA256 = (
    "fa4d8d14ba2c243e3f1b72e6124bd46141e6ea1bfadef22c5b0bb70bd602c9c8"
)
BM25_K1 = 0.9
BM25_B = 0.4
RM3_FEEDBACK_DOCUMENTS = 10
RM3_FEEDBACK_TERMS = 10
RM3_ORIGINAL_QUERY_WEIGHT = 0.5
OFFICIAL_DISCOVERY_SHA256 = (
    "b33c28aa2571cdf5b2867dcc00f0dc55e22198b90022dd23f748a03395301b25"
)
OFFICIAL_RAW_DOCUMENT_COUNT = 1_163_850
OFFICIAL_BASIC_DOCUMENT_COUNT = 182_800
OFFICIAL_AGGRESSIVE_DOCUMENT_COUNT = 173_162

AGGRESSIVE_PREFIXES = (
    "there is not enough information",
    "there is insufficient information",
    "not enough information",
    "insufficient information",
    "the provided context does not contain enough information",
    "the provided context does not provide enough information",
    "the context does not contain enough information",
    "the context does not provide enough information",
    "the given context does not contain enough information",
    "the given context does not provide enough information",
    "the information provided does not contain enough information",
    "the information provided does not provide enough information",
    "the passage does not contain enough information",
    "the passage does not provide enough information",
    "the passages do not contain enough information",
    "the passages do not provide enough information",
    "the question cannot be answered",
    "cannot be determined from the provided",
    "cannot be answered from the provided",
    "i cannot answer",
    "i can't answer",
)


@dataclass(frozen=True, slots=True)
class ResponseCandidate:
    """One preregistered response-retrieval configuration."""

    method: Method
    field: Field
    filter_name: FilterName
    depth: int
    aggregation: AggregationName = SUM_ALL_AGGREGATION

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unsupported retrieval method: {self.method!r}")
        if self.field not in FIELDS:
            raise ValueError(f"unsupported indexed field: {self.field!r}")
        if self.filter_name not in FILTER_NAMES:
            raise ValueError(f"unsupported response filter: {self.filter_name!r}")
        if (
            isinstance(self.depth, bool)
            or not isinstance(self.depth, int)
            or self.depth < 1
        ):
            raise ValueError("retrieval depth must be a positive integer")
        if self.aggregation not in ALL_RESPONSE_AGGREGATIONS:
            raise ValueError(f"unsupported response aggregation: {self.aggregation!r}")

    @property
    def key(self) -> str:
        """Return a stable, filesystem-safe configuration key."""

        retrieval_key = (
            f"{self.method}__{self.field}__{self.filter_name}__d{self.depth}"
        )
        if self.aggregation == SUM_ALL_AGGREGATION:
            return retrieval_key
        return f"{retrieval_key}__agg_{self.aggregation}"


DEFAULT_RESPONSE_CANDIDATES = tuple(
    ResponseCandidate(method, field, filter_name, depth)
    for method in METHODS
    for field in FIELDS
    for filter_name in FILTER_NAMES
    for depth in DEPTHS
)

NORMALIZED_RESPONSE_CANDIDATES = tuple(
    ResponseCandidate(method, field, filter_name, depth, aggregation)
    for aggregation in NORMALIZED_RESPONSE_AGGREGATIONS
    for method in METHODS
    for field in FIELDS
    for filter_name in FILTER_NAMES
    for depth in DEPTHS
)

ALL_RESPONSE_CANDIDATES = (
    *DEFAULT_RESPONSE_CANDIDATES,
    *NORMALIZED_RESPONSE_CANDIDATES,
)


@dataclass(frozen=True, slots=True)
class DiscoveryRecord:
    """One discovery-query/model response before filtering."""

    source_query_id: str
    query_text: str
    model_id: str
    response: str | None

    @property
    def source_query(self) -> str:
        """Alias used by report code for the discovery query text."""

        return self.query_text

    @property
    def discovery_query(self) -> str:
        """Alias matching the terminology in the frozen protocol."""

        return self.query_text


@dataclass(frozen=True, slots=True)
class DiscoveryCorpus(Sequence[DiscoveryRecord]):
    """Strictly validated discovery records plus source-file provenance."""

    records: tuple[DiscoveryRecord, ...]
    source_sha256: str
    source_query_count: int
    raw_document_count: int
    model_ids: tuple[str, ...]

    @overload
    def __getitem__(self, index: int) -> DiscoveryRecord: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[DiscoveryRecord, ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> DiscoveryRecord | tuple[DiscoveryRecord, ...]:
        return self.records[index]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[DiscoveryRecord]:
        return iter(self.records)


@dataclass(frozen=True, slots=True)
class FilterAuditStats:
    """Mutually exclusive outcomes from one frozen response filter."""

    filter_name: FilterName
    input_documents: int
    retained_documents: int
    removed_null: int
    removed_blank: int
    removed_exact_idk: int
    removed_short: int
    removed_prefix: int

    @property
    def removed_documents(self) -> int:
        return self.input_documents - self.retained_documents


@dataclass(frozen=True, slots=True)
class AnalyzerDiagnostics:
    """Pinned analyzer behavior and its canonical hash."""

    unicode_normalization: str
    lowercase: bool
    token_regex: str
    stopwords: tuple[str, ...]
    stopwords_sha256: str
    stemmer: str
    bm25s_version: str
    pystemmer_version: str
    analyzer_sha256: str


@dataclass(frozen=True, slots=True)
class IndexDiagnostics:
    """Provenance and volume for one field/filter index."""

    field: Field
    filter_name: FilterName
    document_count: int
    character_count: int
    analyzed_token_count: int
    empty_document_count: int
    vocabulary_size: int
    vocabulary_sha256: str
    index_sha256: str


@dataclass(frozen=True, slots=True)
class ResponseScoreResult:
    """All candidate/query/model scores and qrel-free retrieval diagnostics."""

    candidates: tuple[ResponseCandidate, ...]
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    model_scores: NDArray[np.float64]
    hit_counts: NDArray[np.int32]
    mean_hit_scores: NDArray[np.float64]
    overlap_document_counts: NDArray[np.int32]
    filter_audits: tuple[FilterAuditStats, ...]
    inventory_filter_names: tuple[FilterName, ...]
    retained_document_counts: NDArray[np.int64]
    retained_character_counts: NDArray[np.int64]
    retained_token_counts: NDArray[np.int64]
    index_diagnostics: tuple[IndexDiagnostics, ...]
    analyzer_diagnostics: AnalyzerDiagnostics
    source_sha256: str
    corpus_sha256: str
    configuration_sha256: str
    score_tensor_sha256: str


class FrozenAnalyzer:
    """The exact NFKC/lowercase/regex/stopword/PyStemmer analyzer."""

    __slots__ = ("_stemmer", "_stopwords", "diagnostics")

    def __init__(self) -> None:
        bm25s, stemmer_module = _load_optional_dependencies()
        stopwords = tuple(bm25s.stopwords.STOPWORDS_EN)
        if len(stopwords) != 33:
            raise RuntimeError(
                "bm25s.stopwords.STOPWORDS_EN changed from the pinned 33-word list"
            )
        stopwords_sha256 = _hash_json(sorted(stopwords))
        if stopwords_sha256 != ANALYZER_STOPWORDS_SHA256:
            raise RuntimeError(
                "bm25s.stopwords.STOPWORDS_EN does not match the preregistered list"
            )
        self._stopwords = frozenset(stopwords)
        self._stemmer = stemmer_module.Stemmer("english")
        payload = {
            "unicode_normalization": "NFKC",
            "lowercase": True,
            "token_regex": ANALYZER_REGEX,
            "stopwords": list(stopwords),
            "stopwords_sha256": stopwords_sha256,
            "stemmer": "PyStemmer English",
            "bm25s_version": "0.3.9",
            "pystemmer_version": "3.1.0",
        }
        self.diagnostics = AnalyzerDiagnostics(
            unicode_normalization="NFKC",
            lowercase=True,
            token_regex=ANALYZER_REGEX,
            stopwords=stopwords,
            stopwords_sha256=stopwords_sha256,
            stemmer="PyStemmer English",
            bm25s_version="0.3.9",
            pystemmer_version="3.1.0",
            analyzer_sha256=_hash_json(payload),
        )

    def analyze(self, text: str) -> tuple[str, ...]:
        """Analyze one string without learning or mutating a vocabulary."""

        if not isinstance(text, str):
            raise TypeError("analyzer input must be a string")
        normalized = unicodedata.normalize("NFKC", text).lower()
        words = _TOKEN_RE.findall(normalized)
        content_words = [word for word in words if word not in self._stopwords]
        if not content_words:
            return ()
        return tuple(self._stemmer.stemWords(content_words))

    def overlap_key(self, text: str) -> str:
        """Return the exact-query guard key required by the protocol."""

        casefolded = unicodedata.normalize("NFKC", text).casefold()
        return " ".join(self.analyze(casefolded))


_TOKEN_RE = re.compile(ANALYZER_REGEX)


def load_discovery_records(
    path: str | Path,
    expected_model_ids: Sequence[str],
) -> DiscoveryCorpus:
    """Load the discovery JSON with rectangularity and type validation.

    The returned records are ordered lexicographically by
    ``(source_query_id, model_id)`` regardless of JSON insertion order.
    Duplicate JSON object keys are rejected rather than silently overwritten.
    """

    model_ids = _validate_model_ids(expected_model_ids)
    source_path = Path(path)
    raw_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()

    try:
        payload = json.loads(raw_bytes, object_pairs_hook=_strict_json_object)
    except UnicodeDecodeError as error:
        raise ValueError(f"{source_path}: discovery JSON is not valid UTF-8") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"{source_path}: invalid discovery JSON: {error}") from error
    if not isinstance(payload, dict) or not payload:
        raise ValueError(f"{source_path}: expected a non-empty top-level object")

    expected_models = set(model_ids)
    records: list[DiscoveryRecord] = []
    for source_query_id in sorted(payload):
        if not isinstance(source_query_id, str) or not source_query_id:
            raise ValueError("discovery query IDs must be non-empty strings")
        item = payload[source_query_id]
        if not isinstance(item, dict):
            raise ValueError(
                f"discovery query {source_query_id!r} must map to an object"
            )
        if set(item) != {"query", "responses"}:
            raise ValueError(
                f"discovery query {source_query_id!r} must contain exactly "
                "'query' and 'responses'"
            )
        query_text = item["query"]
        responses = item["responses"]
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(
                f"discovery query {source_query_id!r} has an invalid query string"
            )
        if not isinstance(responses, dict):
            raise ValueError(
                f"discovery query {source_query_id!r} responses must be an object"
            )
        actual_models = set(responses)
        if actual_models != expected_models:
            missing = sorted(expected_models - actual_models)
            extra = sorted(actual_models - expected_models)
            raise ValueError(
                f"discovery query {source_query_id!r} model mismatch: "
                f"missing={missing!r}, extra={extra!r}"
            )
        for model_id in model_ids:
            response = responses[model_id]
            if response is not None and not isinstance(response, str):
                raise ValueError(
                    f"response {source_query_id!r}/{model_id!r} must be a "
                    "string or null"
                )
            records.append(
                DiscoveryRecord(
                    source_query_id=source_query_id,
                    query_text=query_text,
                    model_id=model_id,
                    response=response,
                )
            )

    return DiscoveryCorpus(
        records=tuple(records),
        source_sha256=source_sha256,
        source_query_count=len(payload),
        raw_document_count=len(records),
        model_ids=model_ids,
    )


def normalize_filter_text(text: str) -> str:
    """Return the frozen NFKC/casefold/whitespace-collapsed filter text."""

    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def response_filter_reason(
    response: str | None,
    filter_name: FilterName,
) -> str | None:
    """Return a mutually exclusive removal reason, or ``None`` if retained."""

    if filter_name not in FILTER_NAMES:
        raise ValueError(f"unsupported response filter: {filter_name!r}")
    if response is None:
        return "null"
    stripped = response.strip()
    if not stripped:
        return "blank"
    normalized = normalize_filter_text(response)
    if normalized == "i don't know":
        return "exact_idk"
    if filter_name == "aggressive":
        if len(stripped) < 50:
            return "short"
        if normalized.startswith(AGGRESSIVE_PREFIXES):
            return "prefix"
    return None


def response_passes_filter(
    response: str | None,
    filter_name: FilterName,
) -> bool:
    """Return whether a response survives one frozen filter."""

    return response_filter_reason(response, filter_name) is None


def build_index_text(record: DiscoveryRecord, field: Field) -> str:
    """Construct one indexed document field exactly as preregistered."""

    if record.response is None:
        raise ValueError("a null response cannot be indexed")
    if field == "response_only":
        return record.response
    if field == "query_response":
        return record.query_text + "\n" + record.response
    raise ValueError(f"unsupported indexed field: {field!r}")


def audit_response_filters(
    records: DiscoveryCorpus | Sequence[DiscoveryRecord],
) -> tuple[FilterAuditStats, ...]:
    """Audit both filters without inspecting qrels or effectiveness scores."""

    record_tuple = tuple(records)
    audits: list[FilterAuditStats] = []
    for filter_name in FILTER_NAMES:
        counts: Counter[str] = Counter()
        for record in record_tuple:
            reason = response_filter_reason(record.response, filter_name)
            counts["retained" if reason is None else reason] += 1
        audit = FilterAuditStats(
            filter_name=filter_name,
            input_documents=len(record_tuple),
            retained_documents=counts["retained"],
            removed_null=counts["null"],
            removed_blank=counts["blank"],
            removed_exact_idk=counts["exact_idk"],
            removed_short=counts["short"],
            removed_prefix=counts["prefix"],
        )
        if audit.retained_documents + audit.removed_documents != len(record_tuple):
            raise AssertionError("filter audit outcomes are not exhaustive")
        audits.append(audit)
    return tuple(audits)


def score_response_candidates(
    records: DiscoveryCorpus | Sequence[DiscoveryRecord],
    *,
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    expected_model_ids: Sequence[str],
    candidates: Sequence[ResponseCandidate] = DEFAULT_RESPONSE_CANDIDATES,
    show_progress: bool = False,
) -> ResponseScoreResult:
    """Score every requested candidate on every target query and model.

    Output tensor axes follow the exact input candidate order, target-query
    order, and lexicographic model order.  Retrieval is performed once at the
    maximum requested depth for each method/field/filter and shallower
    candidates use prefixes of that deterministic ranking.
    """

    candidate_tuple = _validate_candidates(candidates)
    target_query_ids, target_query_texts = _validate_targets(query_ids, query_texts)
    model_ids = _validate_model_ids(expected_model_ids)
    record_tuple, source_sha256 = _validate_records_for_scoring(records, model_ids)

    analyzer = FrozenAnalyzer()
    audits = audit_response_filters(record_tuple)
    if source_sha256 == OFFICIAL_DISCOVERY_SHA256:
        observed = {audit.filter_name: audit.retained_documents for audit in audits}
        if len(record_tuple) != OFFICIAL_RAW_DOCUMENT_COUNT or observed != {
            "basic": OFFICIAL_BASIC_DOCUMENT_COUNT,
            "aggressive": OFFICIAL_AGGRESSIVE_DOCUMENT_COUNT,
        }:
            raise RuntimeError(
                "official discovery source hash matched but frozen corpus counts did not"
            )
    model_to_index = {model_id: index for index, model_id in enumerate(model_ids)}
    inventory = _build_inventory(record_tuple, model_to_index, analyzer)

    candidate_count = len(candidate_tuple)
    query_count = len(target_query_ids)
    model_count = len(model_ids)
    model_scores = np.zeros(
        (candidate_count, query_count, model_count), dtype=np.float64
    )
    hit_counts = np.zeros((candidate_count, query_count, model_count), dtype=np.int32)
    mean_hit_scores = np.zeros_like(model_scores)
    overlap_document_counts = np.zeros((candidate_count, query_count), dtype=np.int32)

    candidates_by_index: dict[tuple[Field, FilterName], list[int]] = defaultdict(list)
    for candidate_index, candidate in enumerate(candidate_tuple):
        candidates_by_index[(candidate.field, candidate.filter_name)].append(
            candidate_index
        )

    index_diagnostics: list[IndexDiagnostics] = []
    for field in FIELDS:
        for filter_name in FILTER_NAMES:
            relevant_candidate_indices = candidates_by_index.get((field, filter_name))
            if not relevant_candidate_indices:
                continue
            built = _build_index(
                records=record_tuple,
                field=field,
                filter_name=filter_name,
                analyzer=analyzer,
                show_progress=show_progress,
            )
            index_diagnostics.append(built.diagnostics)
            max_depth_by_method = {
                method: max(
                    candidate_tuple[index].depth
                    for index in relevant_candidate_indices
                    if candidate_tuple[index].method == method
                )
                for method in METHODS
                if any(
                    candidate_tuple[index].method == method
                    for index in relevant_candidate_indices
                )
            }

            for query_index, query_text in enumerate(target_query_texts):
                query_tokens = analyzer.analyze(query_text)
                overlap_key = analyzer.overlap_key(query_text)
                masked_documents = built.overlap_documents.get(overlap_key, ())
                initial_scores = _score_unweighted_query(built, query_tokens)

                rankings: dict[
                    Method, tuple[NDArray[np.int32], NDArray[np.float64]]
                ] = {}
                if "bm25" in max_depth_by_method:
                    rankings["bm25"] = _top_positive_documents(
                        initial_scores,
                        max_depth_by_method["bm25"],
                        masked_documents,
                    )
                if "bm25_rm3" in max_depth_by_method:
                    rm3_scores = _score_rm3_query(
                        built,
                        query_tokens=query_tokens,
                        initial_scores=initial_scores,
                        masked_documents=masked_documents,
                    )
                    # The guard is deliberately applied again here, independently
                    # of feedback selection.
                    rankings["bm25_rm3"] = _top_positive_documents(
                        rm3_scores,
                        max_depth_by_method["bm25_rm3"],
                        masked_documents,
                    )

                aggregation_cache: dict[
                    tuple[Method, int],
                    tuple[
                        dict[AggregationName, NDArray[np.float64]],
                        NDArray[np.int32],
                        NDArray[np.float64],
                    ],
                ] = {}
                for candidate_index in relevant_candidate_indices:
                    candidate = candidate_tuple[candidate_index]
                    retrieval_key = (candidate.method, candidate.depth)
                    cached = aggregation_cache.get(retrieval_key)
                    if cached is None:
                        document_indices, document_scores = rankings[candidate.method]
                        retained = min(candidate.depth, len(document_indices))
                        cached = _aggregate_scores_by_model(
                            document_indices[:retained],
                            document_scores[:retained],
                            built.document_model_indices,
                            model_count,
                        )
                        aggregation_cache[retrieval_key] = cached
                    aggregate_scores, counts, means = cached
                    model_scores[candidate_index, query_index] = aggregate_scores[
                        candidate.aggregation
                    ]
                    hit_counts[candidate_index, query_index] = counts
                    mean_hit_scores[candidate_index, query_index] = means
                    overlap_document_counts[candidate_index, query_index] = len(
                        masked_documents
                    )

    corpus_sha256 = _hash_records(record_tuple)
    configuration_sha256 = _hash_json(
        {
            "candidates": [candidate.key for candidate in candidate_tuple],
            "analyzer_sha256": analyzer.diagnostics.analyzer_sha256,
            "bm25": {
                "method": "lucene",
                "idf_method": "lucene",
                "k1": BM25_K1,
                "b": BM25_B,
                "dtype": "float64",
                "int_dtype": "int32",
                "backend": "numpy",
                "csc_backend": "numpy",
                "auto_compile": False,
            },
            "rm3": {
                "feedback_documents": RM3_FEEDBACK_DOCUMENTS,
                "feedback_terms": RM3_FEEDBACK_TERMS,
                "original_query_weight": RM3_ORIGINAL_QUERY_WEIGHT,
            },
            "filters": {
                "basic_exact_normalized_failures": ["", "i don't know"],
                "aggressive_minimum_stripped_characters": 50,
                "aggressive_prefixes": list(AGGRESSIVE_PREFIXES),
            },
            "field_construction": {
                "response_only": "response",
                "query_response": 'discovery_query + "\\n" + response',
            },
            "retrieval_rules": {
                "positive_score_only": True,
                "document_tie_break": "stable_document_ordinal",
                "overlap_mask_before_feedback_and_final": True,
                "aggregation_order": list(ALL_RESPONSE_AGGREGATIONS),
                "aggregation_definitions": [
                    {"name": name, "formula": formula}
                    for name, formula in RESPONSE_AGGREGATION_DEFINITIONS
                ],
                "shared_retrieval_for_aggregation_duplicates": True,
            },
        }
    )
    score_tensor_sha256 = _hash_arrays(
        model_scores,
        hit_counts,
        mean_hit_scores,
        overlap_document_counts,
    )

    for array in (
        model_scores,
        hit_counts,
        mean_hit_scores,
        overlap_document_counts,
        inventory.document_counts,
        inventory.character_counts,
        inventory.token_counts,
    ):
        array.setflags(write=False)

    return ResponseScoreResult(
        candidates=candidate_tuple,
        query_ids=target_query_ids,
        model_ids=model_ids,
        model_scores=model_scores,
        hit_counts=hit_counts,
        mean_hit_scores=mean_hit_scores,
        overlap_document_counts=overlap_document_counts,
        filter_audits=audits,
        inventory_filter_names=FILTER_NAMES,
        retained_document_counts=inventory.document_counts,
        retained_character_counts=inventory.character_counts,
        retained_token_counts=inventory.token_counts,
        index_diagnostics=tuple(index_diagnostics),
        analyzer_diagnostics=analyzer.diagnostics,
        source_sha256=source_sha256,
        corpus_sha256=corpus_sha256,
        configuration_sha256=configuration_sha256,
        score_tensor_sha256=score_tensor_sha256,
    )


@dataclass(frozen=True, slots=True)
class _Inventory:
    document_counts: NDArray[np.int64]
    character_counts: NDArray[np.int64]
    token_counts: NDArray[np.int64]


@dataclass(frozen=True, slots=True)
class _BuiltIndex:
    retriever: Any | None
    document_tokens: tuple[tuple[str, ...], ...]
    document_token_ids: tuple[tuple[int, ...], ...]
    document_model_indices: NDArray[np.int32]
    vocabulary: tuple[str, ...]
    vocabulary_map: dict[str, int]
    overlap_documents: dict[str, tuple[int, ...]]
    diagnostics: IndexDiagnostics


def _load_optional_dependencies() -> tuple[Any, Any]:
    try:
        import bm25s
        import Stemmer
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "response retrieval requires optional dependencies; run "
            "`uv sync --extra response` or prefix the command with "
            "`uv run --extra response`"
        ) from error

    bm25s_version = importlib_metadata.version("bm25s")
    stemmer_version = importlib_metadata.version("PyStemmer")
    if bm25s_version != "0.3.9" or stemmer_version != "3.1.0":
        raise RuntimeError(
            "response retrieval requires exactly bm25s==0.3.9 and "
            f"PyStemmer==3.1.0; found {bm25s_version} and {stemmer_version}"
        )
    return bm25s, Stemmer


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _validate_model_ids(model_ids: Sequence[str]) -> tuple[str, ...]:
    values = tuple(model_ids)
    if not values:
        raise ValueError("expected_model_ids must not be empty")
    if any(not isinstance(value, str) or not value for value in values):
        raise ValueError("model IDs must be non-empty strings")
    if len(set(values)) != len(values):
        raise ValueError("model IDs must be unique")
    return tuple(sorted(values))


def _validate_targets(
    query_ids: Sequence[str], query_texts: Sequence[str]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    ids = tuple(query_ids)
    texts = tuple(query_texts)
    if not ids or len(ids) != len(texts):
        raise ValueError("query_ids and query_texts must have equal non-zero length")
    if len(set(ids)) != len(ids):
        raise ValueError("target query IDs must be unique")
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("target query IDs must be non-empty strings")
    if any(not isinstance(value, str) or not value.strip() for value in texts):
        raise ValueError("target query texts must be non-empty strings")
    return ids, texts


def _validate_candidates(
    candidates: Sequence[ResponseCandidate],
) -> tuple[ResponseCandidate, ...]:
    values = tuple(candidates)
    if not values:
        raise ValueError("at least one response candidate is required")
    if any(not isinstance(value, ResponseCandidate) for value in values):
        raise TypeError("candidates must contain ResponseCandidate instances")
    keys = [candidate.key for candidate in values]
    if len(set(keys)) != len(keys):
        raise ValueError("response candidates must be unique")
    return values


def _validate_records_for_scoring(
    records: DiscoveryCorpus | Sequence[DiscoveryRecord],
    model_ids: tuple[str, ...],
) -> tuple[tuple[DiscoveryRecord, ...], str]:
    if isinstance(records, DiscoveryCorpus):
        record_tuple = records.records
        source_sha256 = records.source_sha256
        if records.model_ids != model_ids:
            raise ValueError("discovery corpus model IDs do not match expected models")
    else:
        record_tuple = tuple(records)
        source_sha256 = _hash_records(record_tuple)
    if not record_tuple:
        raise ValueError("discovery records must not be empty")
    if any(not isinstance(record, DiscoveryRecord) for record in record_tuple):
        raise TypeError("records must contain DiscoveryRecord instances")
    expected_order = tuple(
        sorted(record_tuple, key=lambda row: (row.source_query_id, row.model_id))
    )
    if record_tuple != expected_order:
        raise ValueError(
            "discovery records must be ordered by (source_query_id, model_id)"
        )
    expected_models = set(model_ids)
    seen_pairs: set[tuple[str, str]] = set()
    query_models: dict[str, set[str]] = defaultdict(set)
    query_texts: dict[str, str] = {}
    for record in record_tuple:
        pair = (record.source_query_id, record.model_id)
        if pair in seen_pairs:
            raise ValueError(f"duplicate discovery record: {pair!r}")
        seen_pairs.add(pair)
        if record.model_id not in expected_models:
            raise ValueError(f"unexpected discovery model: {record.model_id!r}")
        query_models[record.source_query_id].add(record.model_id)
        previous_text = query_texts.setdefault(
            record.source_query_id, record.query_text
        )
        if record.query_text != previous_text:
            raise ValueError(
                f"inconsistent text for source query {record.source_query_id!r}"
            )
        if record.response is not None and not isinstance(record.response, str):
            raise ValueError("discovery responses must be strings or null")
    for source_query_id, actual_models in query_models.items():
        if actual_models != expected_models:
            raise ValueError(
                f"source query {source_query_id!r} does not contain every model"
            )
    return record_tuple, source_sha256


def _build_inventory(
    records: tuple[DiscoveryRecord, ...],
    model_to_index: dict[str, int],
    analyzer: FrozenAnalyzer,
) -> _Inventory:
    shape = (len(FILTER_NAMES), len(model_to_index))
    document_counts = np.zeros(shape, dtype=np.int64)
    character_counts = np.zeros(shape, dtype=np.int64)
    token_counts = np.zeros(shape, dtype=np.int64)
    analyzed_responses: dict[int, tuple[str, ...]] = {}
    for record_index, record in enumerate(records):
        model_index = model_to_index[record.model_id]
        for filter_index, filter_name in enumerate(FILTER_NAMES):
            if not response_passes_filter(record.response, filter_name):
                continue
            assert record.response is not None
            tokens = analyzed_responses.get(record_index)
            if tokens is None:
                tokens = analyzer.analyze(record.response)
                analyzed_responses[record_index] = tokens
            document_counts[filter_index, model_index] += 1
            character_counts[filter_index, model_index] += len(record.response.strip())
            token_counts[filter_index, model_index] += len(tokens)
    return _Inventory(document_counts, character_counts, token_counts)


def _build_index(
    *,
    records: tuple[DiscoveryRecord, ...],
    field: Field,
    filter_name: FilterName,
    analyzer: FrozenAnalyzer,
    show_progress: bool,
) -> _BuiltIndex:
    retained = tuple(
        record
        for record in records
        if response_passes_filter(record.response, filter_name)
    )
    document_texts = tuple(build_index_text(record, field) for record in retained)
    document_tokens = tuple(analyzer.analyze(text) for text in document_texts)
    vocabulary = tuple(
        sorted({token for tokens in document_tokens for token in tokens})
    )
    vocabulary_map = {token: index for index, token in enumerate(vocabulary)}
    document_token_ids = tuple(
        tuple(vocabulary_map[token] for token in tokens) for tokens in document_tokens
    )
    model_order = tuple(sorted({record.model_id for record in records}))
    model_to_index = {model_id: index for index, model_id in enumerate(model_order)}
    document_model_indices = np.asarray(
        [model_to_index[record.model_id] for record in retained], dtype=np.int32
    )

    overlap_lists: dict[str, list[int]] = defaultdict(list)
    for ordinal, record in enumerate(retained):
        overlap_lists[analyzer.overlap_key(record.query_text)].append(ordinal)
    overlap_documents = {
        key: tuple(values) for key, values in sorted(overlap_lists.items())
    }

    retriever: Any | None = None
    if retained and vocabulary:
        bm25s, _stemmer = _load_optional_dependencies()
        retriever = bm25s.BM25(
            k1=BM25_K1,
            b=BM25_B,
            method="lucene",
            idf_method="lucene",
            dtype="float64",
            int_dtype="int32",
            backend="numpy",
            csc_backend="numpy",
            auto_compile=False,
        )
        retriever.index(
            (list(map(list, document_token_ids)), vocabulary_map),
            create_empty_token=False,
            show_progress=show_progress,
            leave_progress=False,
        )
        if np.asarray(retriever.scores["data"]).dtype != np.dtype(np.float64):
            raise AssertionError("bm25s index did not preserve float64 scores")
        if np.asarray(retriever.scores["indices"]).dtype != np.dtype(np.int32):
            raise AssertionError("bm25s index did not preserve int32 document indices")

    vocabulary_sha256 = _hash_json(list(vocabulary))
    index_hasher = hashlib.sha256()
    index_hasher.update(field.encode())
    index_hasher.update(b"\0")
    index_hasher.update(filter_name.encode())
    index_hasher.update(b"\0")
    index_hasher.update(vocabulary_sha256.encode())
    index_hasher.update(document_model_indices.tobytes())
    index_hasher.update(_hash_json(overlap_documents).encode())
    if retriever is not None:
        for key in ("data", "indices", "indptr"):
            array = np.ascontiguousarray(retriever.scores[key])
            index_hasher.update(str(array.dtype).encode())
            index_hasher.update(np.asarray(array.shape, dtype=np.int64).tobytes())
            index_hasher.update(array.tobytes())
    document_model_indices.setflags(write=False)
    diagnostics = IndexDiagnostics(
        field=field,
        filter_name=filter_name,
        document_count=len(retained),
        character_count=sum(len(text) for text in document_texts),
        analyzed_token_count=sum(len(tokens) for tokens in document_tokens),
        empty_document_count=sum(not tokens for tokens in document_tokens),
        vocabulary_size=len(vocabulary),
        vocabulary_sha256=vocabulary_sha256,
        index_sha256=index_hasher.hexdigest(),
    )
    return _BuiltIndex(
        retriever=retriever,
        document_tokens=document_tokens,
        document_token_ids=document_token_ids,
        document_model_indices=document_model_indices,
        vocabulary=vocabulary,
        vocabulary_map=vocabulary_map,
        overlap_documents=overlap_documents,
        diagnostics=diagnostics,
    )


def _score_unweighted_query(
    built: _BuiltIndex, query_tokens: tuple[str, ...]
) -> NDArray[np.float64]:
    document_count = len(built.document_tokens)
    if built.retriever is None or not query_tokens:
        return np.zeros(document_count, dtype=np.float64)
    token_ids = [
        built.vocabulary_map[token]
        for token in query_tokens
        if token in built.vocabulary_map
    ]
    if not token_ids:
        return np.zeros(document_count, dtype=np.float64)
    scores = built.retriever.get_scores_from_ids(token_ids)
    return np.asarray(scores, dtype=np.float64)


def _score_rm3_query(
    built: _BuiltIndex,
    *,
    query_tokens: tuple[str, ...],
    initial_scores: NDArray[np.float64],
    masked_documents: Sequence[int],
) -> NDArray[np.float64]:
    feedback_documents, feedback_scores = _top_positive_documents(
        initial_scores,
        RM3_FEEDBACK_DOCUMENTS,
        masked_documents,
    )
    if not len(feedback_documents):
        return initial_scores.copy()

    contributions: dict[str, list[float]] = defaultdict(list)
    for document_index, document_score in zip(
        feedback_documents, feedback_scores, strict=True
    ):
        tokens = built.document_tokens[int(document_index)]
        if not tokens:
            continue
        counts = Counter(tokens)
        length = float(len(tokens))
        for term, frequency in counts.items():
            contributions[term].append(
                (float(frequency) / length) * float(document_score)
            )
    feedback_weights = {
        term: math.fsum(values) for term, values in contributions.items()
    }
    feedback_weights = {
        term: weight for term, weight in feedback_weights.items() if weight > 0.0
    }
    if not feedback_weights:
        return initial_scores.copy()
    selected_terms = sorted(
        feedback_weights,
        key=lambda term: (-feedback_weights[term], term),
    )[:RM3_FEEDBACK_TERMS]
    feedback_mass = math.fsum(feedback_weights[term] for term in selected_terms)
    if feedback_mass <= 0.0:
        return initial_scores.copy()

    query_counts = Counter(query_tokens)
    query_mass = math.fsum(float(value) for value in query_counts.values())
    mixed_weights: dict[str, float] = defaultdict(float)
    if query_mass > 0.0:
        for term, count in query_counts.items():
            mixed_weights[term] += RM3_ORIGINAL_QUERY_WEIGHT * (
                float(count) / query_mass
            )
    for term in selected_terms:
        mixed_weights[term] += (1.0 - RM3_ORIGINAL_QUERY_WEIGHT) * (
            feedback_weights[term] / feedback_mass
        )
    return _score_weighted_terms(built, mixed_weights)


def _score_weighted_terms(
    built: _BuiltIndex,
    term_weights: dict[str, float],
) -> NDArray[np.float64]:
    scores = np.zeros(len(built.document_tokens), dtype=np.float64)
    if built.retriever is None:
        return scores
    data = np.asarray(built.retriever.scores["data"], dtype=np.float64)
    indices = np.asarray(built.retriever.scores["indices"], dtype=np.int32)
    indptr = np.asarray(built.retriever.scores["indptr"], dtype=np.int64)
    # Sorted term order makes float64 accumulation reproducible even when two
    # terms contribute to the same document.
    for term in sorted(term_weights):
        weight = float(term_weights[term])
        token_id = built.vocabulary_map.get(term)
        if token_id is None or weight == 0.0:
            continue
        start, end = int(indptr[token_id]), int(indptr[token_id + 1])
        document_indices = indices[start:end]
        scores[document_indices] += data[start:end] * weight
    return scores


def _top_positive_documents(
    scores: NDArray[np.float64],
    depth: int,
    masked_documents: Sequence[int],
) -> tuple[NDArray[np.int32], NDArray[np.float64]]:
    if depth < 1:
        raise ValueError("retrieval depth must be positive")
    eligible = np.asarray(scores, dtype=np.float64).copy()
    if masked_documents:
        eligible[np.asarray(masked_documents, dtype=np.int32)] = 0.0
    positive_indices = np.flatnonzero(eligible > 0.0)
    if not len(positive_indices):
        return np.asarray([], dtype=np.int32), np.asarray([], dtype=np.float64)

    retained = min(depth, len(positive_indices))
    if retained < len(positive_indices):
        positive_scores = eligible[positive_indices]
        threshold = float(
            np.partition(positive_scores, len(positive_scores) - retained)[
                len(positive_scores) - retained
            ]
        )
        above = positive_indices[positive_scores > threshold]
        tied = positive_indices[positive_scores == threshold]
        needed = retained - len(above)
        selected = np.concatenate((above, tied[:needed]))
    else:
        selected = positive_indices
    order = np.lexsort((selected, -eligible[selected]))
    selected = selected[order]
    return (
        np.asarray(selected, dtype=np.int32),
        np.asarray(eligible[selected], dtype=np.float64),
    )


def _aggregate_scores_by_model(
    document_indices: NDArray[np.int32],
    document_scores: NDArray[np.float64],
    document_model_indices: NDArray[np.int32],
    model_count: int,
) -> tuple[
    dict[AggregationName, NDArray[np.float64]],
    NDArray[np.int32],
    NDArray[np.float64],
]:
    """Aggregate one shared ranked hit list by model in float64.

    Each model's score list preserves the input document order, which is the
    frozen descending-score/stable-ordinal retrieval order.  All seven frozen
    aggregations are derived together so aggregation candidates cannot trigger
    independent retrieval passes.
    """

    score_lists: list[list[float]] = [[] for _ in range(model_count)]
    counts = np.zeros(model_count, dtype=np.int32)
    for document_index, score in zip(document_indices, document_scores, strict=True):
        model_index = int(document_model_indices[int(document_index)])
        score_lists[model_index].append(float(score))
        counts[model_index] += 1

    aggregate_scores = {
        aggregation: np.zeros(model_count, dtype=np.float64)
        for aggregation in ALL_RESPONSE_AGGREGATIONS
    }
    for model_index, values in enumerate(score_lists):
        hit_count = len(values)
        if hit_count == 0:
            continue
        sum_all = math.fsum(values)
        top3_count = min(3, hit_count)
        top10_count = min(10, hit_count)
        top3_sum = math.fsum(values[:top3_count])
        top10_sum = math.fsum(values[:top10_count])
        aggregate_scores["sum_all"][model_index] = sum_all
        aggregate_scores["best_hit"][model_index] = values[0]
        aggregate_scores["top3_sum"][model_index] = top3_sum
        aggregate_scores["top10_sum"][model_index] = top10_sum
        aggregate_scores["top3_mean"][model_index] = top3_sum / top3_count
        aggregate_scores["top10_mean"][model_index] = top10_sum / top10_count
        aggregate_scores["mean_all"][model_index] = sum_all / hit_count

    means = aggregate_scores["mean_all"].copy()
    return aggregate_scores, counts, means


def _sumscore_by_model(
    document_indices: NDArray[np.int32],
    document_scores: NDArray[np.float64],
    document_model_indices: NDArray[np.int32],
    model_count: int,
) -> tuple[NDArray[np.float64], NDArray[np.int32], NDArray[np.float64]]:
    """Return the legacy SumScore outputs via the shared aggregator."""

    scores, counts, means = _aggregate_scores_by_model(
        document_indices,
        document_scores,
        document_model_indices,
        model_count,
    )
    return scores[SUM_ALL_AGGREGATION], counts, means


def _hash_records(records: Sequence[DiscoveryRecord]) -> str:
    hasher = hashlib.sha256()
    for record in records:
        row = [
            record.source_query_id,
            record.query_text,
            record.model_id,
            record.response,
        ]
        encoded = json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        hasher.update(len(encoded).to_bytes(8, "big"))
        hasher.update(encoded)
    return hasher.hexdigest()


def _hash_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _hash_arrays(*arrays: NDArray[Any]) -> str:
    hasher = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        hasher.update(str(contiguous.dtype).encode())
        hasher.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        hasher.update(contiguous.tobytes())
    return hasher.hexdigest()


__all__ = [
    "AGGRESSIVE_PREFIXES",
    "ALL_RESPONSE_AGGREGATIONS",
    "ALL_RESPONSE_CANDIDATES",
    "ANALYZER_REGEX",
    "ANALYZER_STOPWORDS_SHA256",
    "AggregationName",
    "AnalyzerDiagnostics",
    "DEFAULT_RESPONSE_CANDIDATES",
    "DiscoveryCorpus",
    "DiscoveryRecord",
    "FilterAuditStats",
    "FrozenAnalyzer",
    "IndexDiagnostics",
    "OFFICIAL_AGGRESSIVE_DOCUMENT_COUNT",
    "OFFICIAL_BASIC_DOCUMENT_COUNT",
    "OFFICIAL_DISCOVERY_SHA256",
    "OFFICIAL_RAW_DOCUMENT_COUNT",
    "NORMALIZED_RESPONSE_AGGREGATIONS",
    "NORMALIZED_RESPONSE_CANDIDATES",
    "RESPONSE_AGGREGATION_DEFINITIONS",
    "ResponseCandidate",
    "ResponseScoreResult",
    "SUM_ALL_AGGREGATION",
    "audit_response_filters",
    "build_index_text",
    "load_discovery_records",
    "normalize_filter_text",
    "response_filter_reason",
    "response_passes_filter",
    "score_response_candidates",
]
