"""Evaluation metrics for ranked TREC Million LLMs runs.

The nDCG implementation follows the default ``trec_eval`` ``ndcg_cut``
semantics: a judgment's gain is its positive relevance grade (not
``2**grade - 1``), and the item at one-based rank ``r`` is discounted by
``log2(r + 1)``. Unjudged items and non-positive judgments have zero gain.

Macro metrics are evaluated over every query in ``qrels``. A query with no
ranking therefore contributes zero, matching complete-query evaluation rather
than silently inflating a score by dropping missing queries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import floor, fsum, isfinite, log2
from random import Random


@dataclass(frozen=True, slots=True)
class QueryMetrics:
    """Metrics for one query."""

    ndcg_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Per-query and macro results for a set of rankings."""

    cutoff: int
    per_query: dict[str, QueryMetrics]
    macro_ndcg_at_k: float
    mean_reciprocal_rank: float


@dataclass(frozen=True, slots=True)
class PairedBootstrapCI:
    """Percentile confidence interval for a paired mean difference.

    ``mean_difference`` and the bounds are candidate minus baseline.
    """

    mean_difference: float
    lower: float
    upper: float
    confidence_level: float
    n_resamples: int


def _validate_cutoff(k: int) -> None:
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")


def _gain(value: int | float) -> float:
    """Return the raw positive TREC gain for a relevance judgment."""

    try:
        gain = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"relevance judgments must be numeric, got {value!r}") from exc
    if not isfinite(gain):
        raise ValueError(f"relevance judgments must be finite, got {value!r}")
    return gain if gain > 0.0 else 0.0


def _ranking_tuple(ranking: Sequence[str]) -> tuple[str, ...]:
    if isinstance(ranking, (str, bytes)):
        raise TypeError("ranking must be a sequence of document IDs, not a string")
    ranked = tuple(ranking)
    try:
        if len(set(ranked)) != len(ranked):
            raise ValueError("ranking contains duplicate document IDs")
    except TypeError as exc:
        raise TypeError("document IDs in a ranking must be hashable") from exc
    return ranked


def ndcg_at_k(
    ranking: Sequence[str],
    qrels: Mapping[str, int | float],
    k: int = 10,
) -> float:
    """Return nDCG at ``k`` using raw graded gains.

    The ranking is already assumed to be in descending system-score order.
    Missing judgments have zero gain. Duplicate document IDs are rejected, as
    they are invalid in a TREC run.
    """

    _validate_cutoff(k)
    ranked = _ranking_tuple(ranking)

    dcg = fsum(
        _gain(qrels.get(document_id, 0.0)) / log2(rank + 1)
        for rank, document_id in enumerate(ranked[:k], start=1)
    )
    ideal_gains = sorted((_gain(value) for value in qrels.values()), reverse=True)[:k]
    ideal_dcg = fsum(
        gain / log2(rank + 1)
        for rank, gain in enumerate(ideal_gains, start=1)
        if gain > 0.0
    )
    return dcg / ideal_dcg if ideal_dcg > 0.0 else 0.0


def reciprocal_rank(
    ranking: Sequence[str],
    qrels: Mapping[str, int | float],
) -> float:
    """Return reciprocal rank, treating every judgment greater than zero as relevant."""

    ranked = _ranking_tuple(ranking)
    for rank, document_id in enumerate(ranked, start=1):
        if _gain(qrels.get(document_id, 0.0)) > 0.0:
            return 1.0 / rank
    return 0.0


def per_query_metrics(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int | float]],
    k: int = 10,
) -> dict[str, QueryMetrics]:
    """Evaluate each qrels query, assigning zero to a missing ranking."""

    _validate_cutoff(k)
    output: dict[str, QueryMetrics] = {}
    for query_id, query_qrels in qrels.items():
        ranking = rankings.get(query_id, ())
        output[query_id] = QueryMetrics(
            ndcg_at_k=ndcg_at_k(ranking, query_qrels, k),
            reciprocal_rank=reciprocal_rank(ranking, query_qrels),
        )
    return output


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("at least one query is required for a macro metric")
    return fsum(values) / len(values)


def macro_ndcg_at_k(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int | float]],
    k: int = 10,
) -> float:
    """Return macro-average nDCG at ``k`` over every qrels query."""

    values = [metric.ndcg_at_k for metric in per_query_metrics(rankings, qrels, k).values()]
    return _mean(values)


def mean_reciprocal_rank(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int | float]],
) -> float:
    """Return mean reciprocal rank over every qrels query."""

    values = [
        reciprocal_rank(rankings.get(query_id, ()), query_qrels)
        for query_id, query_qrels in qrels.items()
    ]
    return _mean(values)


def evaluate_rankings(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, int | float]],
    k: int = 10,
) -> EvaluationResult:
    """Return per-query nDCG/RR and their macro averages."""

    per_query = per_query_metrics(rankings, qrels, k)
    ndcg_values = [metric.ndcg_at_k for metric in per_query.values()]
    rr_values = [metric.reciprocal_rank for metric in per_query.values()]
    return EvaluationResult(
        cutoff=k,
        per_query=per_query,
        macro_ndcg_at_k=_mean(ndcg_values),
        mean_reciprocal_rank=_mean(rr_values),
    )


def _finite_score(value: float, *, query_id: str, system_name: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{system_name} score for query {query_id!r} must be numeric"
        ) from exc
    if not isfinite(score):
        raise ValueError(f"{system_name} score for query {query_id!r} must be finite")
    return score


def _linear_quantile(sorted_values: Sequence[float], probability: float) -> float:
    """Type-7 linear quantile, also used by common numerical packages."""

    position = (len(sorted_values) - 1) * probability
    lower_index = floor(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return (
        sorted_values[lower_index] * (1.0 - fraction)
        + sorted_values[upper_index] * fraction
    )


def paired_bootstrap_ci(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    confidence_level: float = 0.95,
    n_resamples: int = 10_000,
    seed: int | None = 0,
) -> PairedBootstrapCI:
    """Estimate a paired percentile CI for candidate-minus-baseline.

    Each resample draws query IDs with replacement, preserving the baseline and
    candidate score pair for a query. Inputs must contain exactly the same,
    non-empty set of query IDs.
    """

    baseline_ids = set(baseline)
    candidate_ids = set(candidate)
    if baseline_ids != candidate_ids:
        missing_candidate = baseline_ids - candidate_ids
        missing_baseline = candidate_ids - baseline_ids
        raise ValueError(
            "paired inputs must have identical query IDs; "
            f"missing from candidate={sorted(missing_candidate)!r}, "
            f"missing from baseline={sorted(missing_baseline)!r}"
        )
    if not baseline:
        raise ValueError("paired bootstrap requires at least one query")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")
    if (
        isinstance(n_resamples, bool)
        or not isinstance(n_resamples, int)
        or n_resamples <= 0
    ):
        raise ValueError("n_resamples must be a positive integer")

    differences = [
        _finite_score(candidate[query_id], query_id=query_id, system_name="candidate")
        - _finite_score(baseline[query_id], query_id=query_id, system_name="baseline")
        for query_id in baseline
    ]
    sample_size = len(differences)
    mean_difference = fsum(differences) / sample_size
    random = Random(seed)
    bootstrap_means = [
        fsum(differences[random.randrange(sample_size)] for _ in range(sample_size))
        / sample_size
        for _ in range(n_resamples)
    ]
    bootstrap_means.sort()
    alpha = 1.0 - confidence_level
    return PairedBootstrapCI(
        mean_difference=mean_difference,
        lower=_linear_quantile(bootstrap_means, alpha / 2.0),
        upper=_linear_quantile(bootstrap_means, 1.0 - alpha / 2.0),
        confidence_level=confidence_level,
        n_resamples=n_resamples,
    )
