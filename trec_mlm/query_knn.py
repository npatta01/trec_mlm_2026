"""Leakage-safe query-neighbor baseline for TREC Million LLMs.

The query embedding in this baseline is a word TF-IDF vector.  It is used only
to find similar *training* queries.  The expertise signal comes from the
training qrels: the grades of each candidate model are aggregated over the
nearest queries and converted into a deterministic, complete model ranking.

No third-party machine-learning package is required.  Keeping the TF-IDF
implementation here also makes it explicit that document frequencies and the
vocabulary are learned from training text only.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import TypeAlias


QueryMap: TypeAlias = dict[str, str]
Qrels: TypeAlias = dict[str, dict[str, float]]
SparseVector: TypeAlias = dict[str, float]

_TOKEN_RE = re.compile(r"(?u)\b\w+\b")
_VALID_AGGREGATIONS = frozenset({"weighted_mean", "weighted_sum", "max"})


@dataclass(frozen=True)
class DevelopmentData:
    """Validated development queries and their complete qrel matrix."""

    queries: QueryMap
    qrels: Qrels
    models: tuple[str, ...]


@dataclass(frozen=True)
class Neighbor:
    """A training query and its cosine similarity to the target query."""

    query_id: str
    similarity: float


@dataclass(frozen=True)
class RankedModel:
    """One candidate in a complete model ranking."""

    model_id: str
    score: float
    rank: int
    global_prior: float


def load_queries_tsv(path: str | Path) -> QueryMap:
    """Load an unheaded ``query_id<TAB>query_text`` file.

    A split at the first tab is intentional: a tab occurring in the query text
    remains part of the text instead of being mistaken for another column.
    """

    queries: QueryMap = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            if not line:
                continue
            fields = line.split("\t", maxsplit=1)
            if len(fields) != 2:
                raise ValueError(
                    f"{path}:{line_number}: expected query_id<TAB>query_text"
                )
            query_id, query_text = fields[0].strip(), fields[1].strip()
            if not query_id or not query_text:
                raise ValueError(
                    f"{path}:{line_number}: query ID and text must be non-empty"
                )
            if query_id in queries:
                raise ValueError(
                    f"{path}:{line_number}: duplicate query ID {query_id!r}"
                )
            queries[query_id] = query_text

    if not queries:
        raise ValueError(f"{path}: query file is empty")
    return queries


def load_complete_qrels(
    path: str | Path,
    *,
    expected_query_ids: Iterable[str] | None = None,
) -> Qrels:
    """Load and validate a complete four-column TREC qrel matrix.

    Completeness means that every expected query occurs exactly once with every
    candidate model found in the file.  Duplicate query/model pairs, non-finite
    grades, missing queries, extra queries, and ragged model sets are rejected.
    """

    qrels: Qrels = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            fields = line.split()
            if len(fields) != 4:
                raise ValueError(
                    f"{path}:{line_number}: expected four qrel columns"
                )
            query_id, _iteration, model_id, grade_text = fields
            try:
                grade = float(grade_text)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: invalid grade {grade_text!r}"
                ) from error
            if not math.isfinite(grade):
                raise ValueError(
                    f"{path}:{line_number}: grade must be finite, got {grade_text!r}"
                )

            query_grades = qrels.setdefault(query_id, {})
            if model_id in query_grades:
                raise ValueError(
                    f"{path}:{line_number}: duplicate qrel for "
                    f"{query_id!r}/{model_id!r}"
                )
            query_grades[model_id] = grade

    validate_complete_qrels(qrels, expected_query_ids=expected_query_ids)
    return qrels


def validate_complete_qrels(
    qrels: Mapping[str, Mapping[str, float]],
    *,
    expected_query_ids: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Validate a rectangular qrel matrix and return sorted model IDs."""

    if not qrels:
        raise ValueError("qrels are empty")

    actual_query_ids = set(qrels)
    if expected_query_ids is not None:
        expected = set(expected_query_ids)
        missing = sorted(expected - actual_query_ids)
        extra = sorted(actual_query_ids - expected)
        if missing or extra:
            details: list[str] = []
            if missing:
                details.append(f"missing query IDs: {missing!r}")
            if extra:
                details.append(f"unexpected query IDs: {extra!r}")
            raise ValueError("qrels/query mismatch (" + "; ".join(details) + ")")

    first_query_id = min(qrels)
    expected_models = set(qrels[first_query_id])
    if not expected_models:
        raise ValueError(f"query {first_query_id!r} has no model grades")

    for query_id in sorted(qrels):
        grades = qrels[query_id]
        model_ids = set(grades)
        if model_ids != expected_models:
            missing_models = sorted(expected_models - model_ids)
            extra_models = sorted(model_ids - expected_models)
            raise ValueError(
                f"incomplete qrels for query {query_id!r}: "
                f"missing={missing_models!r}, extra={extra_models!r}"
            )
        for model_id, grade in grades.items():
            if not math.isfinite(float(grade)):
                raise ValueError(
                    f"non-finite qrel for {query_id!r}/{model_id!r}: {grade!r}"
                )

    return tuple(sorted(expected_models))


def load_development_data(
    query_path: str | Path,
    qrels_path: str | Path,
) -> DevelopmentData:
    """Load development queries and their complete query-by-model qrels."""

    queries = load_queries_tsv(query_path)
    qrels = load_complete_qrels(qrels_path, expected_query_ids=queries)
    models = validate_complete_qrels(qrels, expected_query_ids=queries)
    return DevelopmentData(queries=queries, qrels=qrels, models=models)


class WordTfidfVectorizer:
    """Small L2-normalized word TF-IDF vectorizer fitted on training text.

    IDF uses ``log((1 + n_documents) / (1 + document_frequency)) + 1`` and
    term frequency uses ``1 + log(count)``.  The learned vocabulary and IDF do
    not change when target queries are transformed.
    """

    def __init__(self, *, ngram_range: tuple[int, int] = (1, 2)) -> None:
        minimum, maximum = ngram_range
        if minimum < 1 or maximum < minimum:
            raise ValueError("ngram_range must satisfy 1 <= min_n <= max_n")
        self.ngram_range = ngram_range
        self.vocabulary_: dict[str, int] = {}
        self.idf_: dict[str, float] = {}
        self.n_documents_: int = 0
        self._is_fitted = False

    def fit(self, documents: Iterable[str]) -> WordTfidfVectorizer:
        """Learn vocabulary and IDF from the supplied training documents."""

        document_frequencies: Counter[str] = Counter()
        document_count = 0
        for document in documents:
            if not isinstance(document, str):
                raise TypeError("TF-IDF documents must be strings")
            document_count += 1
            document_frequencies.update(set(self._terms(document)))

        if document_count == 0:
            raise ValueError("cannot fit TF-IDF on zero documents")

        vocabulary = sorted(document_frequencies)
        self.vocabulary_ = {term: index for index, term in enumerate(vocabulary)}
        self.idf_ = {
            term: math.log(
                (1.0 + document_count) / (1.0 + document_frequencies[term])
            )
            + 1.0
            for term in vocabulary
        }
        self.n_documents_ = document_count
        self._is_fitted = True
        return self

    def transform_one(self, document: str) -> SparseVector:
        """Transform one document without changing the fitted state."""

        self._check_is_fitted()
        if not isinstance(document, str):
            raise TypeError("TF-IDF documents must be strings")

        counts = Counter(
            term for term in self._terms(document) if term in self.vocabulary_
        )
        weighted = {
            term: (1.0 + math.log(count)) * self.idf_[term]
            for term, count in counts.items()
        }
        norm = math.sqrt(math.fsum(value * value for value in weighted.values()))
        if norm == 0.0:
            return {}
        return {term: value / norm for term, value in weighted.items()}

    def transform(self, documents: Iterable[str]) -> list[SparseVector]:
        """Transform documents using the training-fitted vocabulary and IDF."""

        return [self.transform_one(document) for document in documents]

    def fit_transform(self, documents: Iterable[str]) -> list[SparseVector]:
        """Fit on and transform the same training documents."""

        materialized = list(documents)
        self.fit(materialized)
        return self.transform(materialized)

    def _terms(self, document: str) -> Iterable[str]:
        tokens = [token.casefold() for token in _TOKEN_RE.findall(document)]
        minimum, maximum = self.ngram_range
        for ngram_size in range(minimum, maximum + 1):
            for start in range(0, len(tokens) - ngram_size + 1):
                yield " ".join(tokens[start : start + ngram_size])

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("TF-IDF vectorizer has not been fitted")


def cosine_similarity(left: Mapping[str, float], right: Mapping[str, float]) -> float:
    """Return the dot product of two L2-normalized sparse vectors."""

    if len(left) > len(right):
        left, right = right, left
    return math.fsum(value * right.get(term, 0.0) for term, value in left.items())


class QueryKNNRanker:
    """Rank candidate models from qrels of cosine-nearest training queries."""

    def __init__(
        self,
        *,
        k: int = 10,
        aggregation: str = "weighted_mean",
        ngram_range: tuple[int, int] = (1, 2),
    ) -> None:
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer")
        self.k = k
        self.aggregation = _validate_aggregation(aggregation)
        self.ngram_range = ngram_range
        self.vectorizer = WordTfidfVectorizer(ngram_range=ngram_range)
        self._query_ids: tuple[str, ...] = ()
        self._vectors: dict[str, SparseVector] = {}
        self._qrels: Qrels = {}
        self._models: tuple[str, ...] = ()
        self._global_priors: dict[str, float] = {}
        self._is_fitted = False

    @property
    def models(self) -> tuple[str, ...]:
        """The deterministic candidate model universe."""

        self._check_is_fitted()
        return self._models

    @property
    def global_priors(self) -> dict[str, float]:
        """Mean training grade per model, returned as a defensive copy."""

        self._check_is_fitted()
        return dict(self._global_priors)

    def fit(
        self,
        queries: Mapping[str, str],
        qrels: Mapping[str, Mapping[str, float]],
    ) -> QueryKNNRanker:
        """Fit TF-IDF and model priors strictly on the training queries.

        For leakage-safe cross-validation, pass only the training fold here.
        :func:`leave_one_out_rankings` performs that exclusion and refitting
        automatically for each held-out development query.
        """

        if not queries:
            raise ValueError("training queries are empty")
        if any(not isinstance(text, str) for text in queries.values()):
            raise TypeError("query texts must be strings")

        models = validate_complete_qrels(qrels, expected_query_ids=queries)
        query_ids = tuple(sorted(queries))

        # A fresh vectorizer prevents state from an earlier fit leaking into a
        # later fold when callers reuse a QueryKNNRanker instance.
        self.vectorizer = WordTfidfVectorizer(ngram_range=self.ngram_range)
        training_text = [queries[query_id] for query_id in query_ids]
        vectors = self.vectorizer.fit_transform(training_text)

        copied_qrels: Qrels = {
            query_id: {
                model_id: float(qrels[query_id][model_id]) for model_id in models
            }
            for query_id in query_ids
        }
        self._query_ids = query_ids
        self._vectors = dict(zip(query_ids, vectors, strict=True))
        self._qrels = copied_qrels
        self._models = models
        self._global_priors = {
            model_id: math.fsum(
                copied_qrels[query_id][model_id] for query_id in query_ids
            )
            / len(query_ids)
            for model_id in models
        }
        self._is_fitted = True
        return self

    def nearest_queries(
        self,
        query_text: str,
        *,
        k: int | None = None,
        exclude_query_id: str | None = None,
    ) -> list[Neighbor]:
        """Find positive-cosine training neighbors, excluding a target ID.

        Zero-cosine queries carry no evidence and are omitted.  If none remain,
        model scoring falls back to the global training prior.
        """

        self._check_is_fitted()
        requested_k = self.k if k is None else k
        if (
            isinstance(requested_k, bool)
            or not isinstance(requested_k, int)
            or requested_k < 1
        ):
            raise ValueError("k must be a positive integer")

        target_vector = self.vectorizer.transform_one(query_text)
        neighbors = [
            Neighbor(
                query_id=query_id,
                similarity=cosine_similarity(target_vector, self._vectors[query_id]),
            )
            for query_id in self._query_ids
            if query_id != exclude_query_id
        ]
        positive_neighbors = [
            neighbor for neighbor in neighbors if neighbor.similarity > 0.0
        ]
        positive_neighbors.sort(key=lambda item: (-item.similarity, item.query_id))
        return positive_neighbors[:requested_k]

    def score_models(
        self,
        query_text: str,
        *,
        k: int | None = None,
        aggregation: str | None = None,
        exclude_query_id: str | None = None,
    ) -> dict[str, float]:
        """Score every candidate model from its nearest-query evidence."""

        self._check_is_fitted()
        method = self.aggregation if aggregation is None else _validate_aggregation(
            aggregation
        )
        neighbors = self.nearest_queries(
            query_text,
            k=k,
            exclude_query_id=exclude_query_id,
        )
        if not neighbors:
            return dict(self._global_priors)

        weight_sum = math.fsum(neighbor.similarity for neighbor in neighbors)
        if weight_sum <= 0.0:
            return dict(self._global_priors)

        scores: dict[str, float] = {}
        for model_id in self._models:
            evidence = [
                neighbor.similarity
                * self._qrels[neighbor.query_id][model_id]
                for neighbor in neighbors
            ]
            if method == "weighted_sum":
                score = math.fsum(evidence)
            elif method == "weighted_mean":
                score = math.fsum(evidence) / weight_sum
            else:  # method == "max"
                score = max(evidence)
            scores[model_id] = score
        return scores

    def rank(
        self,
        query_text: str,
        *,
        k: int | None = None,
        aggregation: str | None = None,
        exclude_query_id: str | None = None,
    ) -> list[RankedModel]:
        """Return a deterministic full model ranking.

        The primary key is the selected neighbor-aggregation score.  Exact
        score ties use the mean training grade (global prior), then model ID.
        If a target has no in-vocabulary overlap, ``score_models`` returns the
        priors directly, providing a deterministic fallback rather than an
        arbitrary ranking.
        """

        scores = self.score_models(
            query_text,
            k=k,
            aggregation=aggregation,
            exclude_query_id=exclude_query_id,
        )
        ordered_models = sorted(
            self._models,
            key=lambda model_id: (
                -scores[model_id],
                -self._global_priors[model_id],
                model_id,
            ),
        )
        return [
            RankedModel(
                model_id=model_id,
                score=scores[model_id],
                rank=rank,
                global_prior=self._global_priors[model_id],
            )
            for rank, model_id in enumerate(ordered_models, start=1)
        ]

    def rank_many(
        self,
        queries: Mapping[str, str],
        *,
        k: int | None = None,
        aggregation: str | None = None,
        exclude_matching_query_ids: bool = False,
    ) -> dict[str, list[RankedModel]]:
        """Rank a mapping of target queries in sorted query-ID order."""

        return {
            query_id: self.rank(
                queries[query_id],
                k=k,
                aggregation=aggregation,
                exclude_query_id=query_id if exclude_matching_query_ids else None,
            )
            for query_id in sorted(queries)
        }

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("QueryKNNRanker has not been fitted")


def leave_one_out_rankings(
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    k: int = 10,
    aggregation: str = "weighted_mean",
    ngram_range: tuple[int, int] = (1, 2),
) -> dict[str, list[RankedModel]]:
    """Create leakage-safe leave-one-query-out development rankings.

    Each fold fits a fresh vocabulary, IDF values, and global prior after the
    held-out query and its qrels have been removed.
    """

    validate_complete_qrels(qrels, expected_query_ids=queries)
    if len(queries) < 2:
        raise ValueError("leave-one-out ranking needs at least two queries")

    rankings: dict[str, list[RankedModel]] = {}
    for held_out_id in sorted(queries):
        training_queries = {
            query_id: queries[query_id]
            for query_id in queries
            if query_id != held_out_id
        }
        training_qrels = {
            query_id: qrels[query_id]
            for query_id in queries
            if query_id != held_out_id
        }
        ranker = QueryKNNRanker(
            k=k,
            aggregation=aggregation,
            ngram_range=ngram_range,
        ).fit(training_queries, training_qrels)
        rankings[held_out_id] = ranker.rank(queries[held_out_id])
    return rankings


def ranking_model_ids(ranking: Sequence[RankedModel]) -> list[str]:
    """Extract model IDs from a ranking; useful for metric adapters and tests."""

    return [item.model_id for item in ranking]


def _validate_aggregation(aggregation: str) -> str:
    if aggregation not in _VALID_AGGREGATIONS:
        choices = ", ".join(sorted(_VALID_AGGREGATIONS))
        raise ValueError(f"aggregation must be one of: {choices}")
    return aggregation
