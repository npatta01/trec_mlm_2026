"""Build a renderer-neutral bundle from frozen E5 diagnostic artifacts."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np

from trec_mlm.metrics import ndcg_at_k, reciprocal_rank
from trec_mlm.profile_evaluation import validate_qrel_inventories
from trec_mlm.query_knn import load_complete_qrels, load_queries_tsv


class EvidenceBundleError(ValueError):
    """Frozen E5 inputs cannot safely produce diagnostic evidence."""


EXACT_REFUSALS = frozenset({"i don t know", "i do not know"})
SYSTEMS = (
    "dense_reference",
    "profile_only",
    "dense_profile_equal",
    "raw_hierarchical",
    "raw_hierarchical_dense_equal",
)
RANK_SYSTEMS = (*SYSTEMS, "best_record_cosine")
EVIDENCE_PER_CANDIDATE = 3
QRELS_USAGE = "Post-freeze evaluation and failure inspection only"
METRIC_TOLERANCE = 1e-12
CONTEXT_RISK_RE = re.compile(
    r"\b(this|that|these|those|it|they|them|he|she|there|the former|the latter|what happened next)\b",
    re.IGNORECASE,
)


def derive_cohorts(metrics: Mapping[str, float]) -> list[str]:
    """Return ordered descriptive tags for frozen per-system metrics."""

    tags: list[str] = []
    new = metrics["raw_hierarchical_dense_equal"]
    prior = metrics["dense_profile_equal"]
    raw = metrics["raw_hierarchical"]
    if new == 0.0:
        tags.append("zero fused nDCG@10")
    if all(metrics[name] == 0.0 for name in SYSTEMS):
        tags.append("persistent all-system zero")
    if new == 0.0 and raw > 0.0:
        tags.append("raw-success/fusion-suppressed")
    if new == 0.0 and prior > 0.0:
        tags.append("new zero regression")
    if new > 0.0 and prior == 0.0:
        tags.append("rescued prior zero")
    return tags


def derive_mechanism(
    positive_ids: Collection[str], positions: Mapping[str, Mapping[str, int]]
) -> str:
    """Describe the relevant-positive rank pattern without asserting a cause."""

    if not positive_ids:
        return "no development-positive candidate"

    def best(system: str) -> int:
        return min(positions[system][candidate_id] for candidate_id in positive_ids)

    if best("raw_hierarchical") <= 10 and best("raw_hierarchical_dense_equal") > 10:
        return "fusion suppression"
    if best("raw_hierarchical_dense_equal") == 1:
        return "relevant candidate ranked first"
    if best("raw_hierarchical_dense_equal") <= 10:
        return "top-10 retrieval, top-1 miss"
    if best("best_record_cosine") <= 10 and best("raw_hierarchical") > 10:
        return "aggregation suppression"
    if best("best_record_cosine") > 10:
        return "evidence/routing gap"
    raise EvidenceBundleError("mechanism rank classification is incomplete")


def derive_retrieval_outcome(
    positive_ids: Collection[str], positions: Mapping[str, Mapping[str, int]]
) -> str:
    """Classify the best positive final-fusion position by rank-one success."""

    if not positive_ids:
        return "no relevant candidate in top 10"
    final_rank = min(
        positions["raw_hierarchical_dense_equal"][candidate_id]
        for candidate_id in positive_ids
    )
    if final_rank == 1:
        return "top-1 success"
    if final_rank <= 10:
        return "top-10 but not top-1"
    return "no relevant candidate in top 10"


def has_context_risk(text: str) -> bool:
    """Flag surface context dependence without diagnosing a failure cause."""

    return bool(CONTEXT_RISK_RE.search(text))


def classify_behavior(response: str) -> str:
    """Classify an exact persisted response without judging its correctness."""

    normalized = re.sub(r"[^a-z0-9]+", " ", response.casefold()).strip()
    if not normalized:
        return "empty"
    if normalized in EXACT_REFUSALS:
        return "exact refusal"
    return "substantive response"


def _is_finite_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _unit_interval_metric(value: Any, *, label: str) -> float:
    if not _is_finite_number(value) or not 0.0 <= float(value) <= 1.0:
        raise EvidenceBundleError(
            f"{label} must be finite and in [0, 1]"
        )
    return float(value)


def _valid_identifier_list(value: Any) -> bool:
    return (
        isinstance(value, list)
        and all(isinstance(item, str) and item for item in value)
        and len(value) == len(set(value))
    )


def _validate_metric_pair(
    metric: Any,
    *,
    label: str,
    schema_error: str,
) -> dict[str, float]:
    if (
        not isinstance(metric, Mapping)
        or set(metric) != {"ndcg_at_10", "reciprocal_rank"}
    ):
        raise EvidenceBundleError(schema_error)
    return {
        name: _unit_interval_metric(
            metric[name],
            label=f"{label} {name}",
        )
        for name in ("ndcg_at_10", "reciprocal_rank")
    }


def _validate_evidence_record_schema(record: Any) -> None:
    if (
        not isinstance(record, Mapping)
        or set(record)
        != {
            "record_id",
            "source_query_id",
            "question",
            "response",
            "similarity",
            "behavior",
        }
        or not isinstance(record["record_id"], str)
        or not record["record_id"]
        or not isinstance(record["source_query_id"], str)
        or not record["source_query_id"]
        or not isinstance(record["question"], str)
        or not isinstance(record["response"], str)
        or not _is_finite_number(record["similarity"])
        or not isinstance(record["behavior"], str)
    ):
        raise EvidenceBundleError(
            "evidence bundle candidate schema differs"
        )


def _validate_evidence_candidate_schema(
    candidate: Any,
    *,
    candidate_id: str,
    candidate_count: int,
    expected_evidence_count: int,
) -> None:
    if (
        not isinstance(candidate, Mapping)
        or set(candidate)
        != {
            "candidate_id",
            "qrel_grade",
            "is_positive",
            "ranks",
            "best_record_cosine",
            "mean_top_10_cosine",
            "evidence",
        }
    ):
        raise EvidenceBundleError(
            "evidence bundle candidate schema differs"
        )
    ranks = candidate["ranks"]
    evidence = candidate["evidence"]
    if (
        candidate["candidate_id"] != candidate_id
        or not _is_finite_number(candidate["qrel_grade"])
        or not isinstance(candidate["is_positive"], bool)
        or candidate["is_positive"]
        != (float(candidate["qrel_grade"]) > 0.0)
        or not isinstance(ranks, Mapping)
        or set(ranks) != set(RANK_SYSTEMS)
        or any(
            type(rank) is not int or not 1 <= rank <= candidate_count
            for rank in ranks.values()
        )
        or not _is_finite_number(candidate["best_record_cosine"])
        or not _is_finite_number(candidate["mean_top_10_cosine"])
        or not isinstance(evidence, list)
        or len(evidence) != expected_evidence_count
    ):
        raise EvidenceBundleError(
            "evidence bundle candidate schema differs"
        )
    for record in evidence:
        _validate_evidence_record_schema(record)


def _derive_component_metric(
    candidates: Mapping[str, Mapping[str, Any]],
    *,
    system: str,
) -> dict[str, float]:
    ranking = sorted(
        candidates,
        key=lambda candidate_id: int(
            candidates[candidate_id]["ranks"][system]
        ),
    )
    qrels = {
        candidate_id: float(candidate["qrel_grade"])
        for candidate_id, candidate in candidates.items()
    }
    return {
        "ndcg_at_10": ndcg_at_k(ranking, qrels, 10),
        "reciprocal_rank": reciprocal_rank(ranking, qrels),
    }


def _validate_evidence_query(
    query: Any,
    *,
    query_id: str,
    candidate_count: int,
    evidence_per_candidate: int,
) -> frozenset[str]:
    query_schema_error = (
        f"evidence bundle query schema differs for {query_id}"
    )
    if (
        not isinstance(query, Mapping)
        or set(query)
        != {
            "query_id",
            "query_text",
            "display_candidate_ids",
            "evidence_candidate_ids",
            "component_metrics",
            "cohorts",
            "context_risk",
            "retrieval_outcome",
            "mechanism",
            "candidates",
        }
    ):
        raise EvidenceBundleError(query_schema_error)
    display_ids = query["display_candidate_ids"]
    evidence_ids = query["evidence_candidate_ids"]
    metrics = query["component_metrics"]
    cohorts = query["cohorts"]
    candidates = query["candidates"]
    if (
        query["query_id"] != query_id
        or not isinstance(query["query_text"], str)
        or not _valid_identifier_list(display_ids)
        or not _valid_identifier_list(evidence_ids)
        or not isinstance(metrics, Mapping)
        or set(metrics) != set(SYSTEMS)
        or not isinstance(cohorts, list)
        or any(not isinstance(cohort, str) or not cohort for cohort in cohorts)
        or len(cohorts) != len(set(cohorts))
        or not isinstance(query["context_risk"], bool)
        or not isinstance(query["retrieval_outcome"], str)
        or not isinstance(query["mechanism"], str)
        or not isinstance(candidates, Mapping)
        or len(candidates) != candidate_count
        or any(
            not isinstance(candidate_id, str) or not candidate_id
            for candidate_id in candidates
        )
        or not set(display_ids) <= set(candidates)
        or not set(evidence_ids) <= set(display_ids)
    ):
        raise EvidenceBundleError(query_schema_error)

    validated_metrics = {
        system: _validate_metric_pair(
            metrics[system],
            label=f"evidence {query_id} {system}",
            schema_error=query_schema_error,
        )
        for system in SYSTEMS
    }
    if query["context_risk"] is not has_context_risk(
        query["query_text"]
    ):
        raise EvidenceBundleError(
            f"evidence context-risk flag differs for {query_id}"
        )

    evidence_id_set = set(evidence_ids)
    record_ids: set[str] = set()
    for candidate_id, candidate in candidates.items():
        _validate_evidence_candidate_schema(
            candidate,
            candidate_id=candidate_id,
            candidate_count=candidate_count,
            expected_evidence_count=(
                evidence_per_candidate
                if candidate_id in evidence_id_set
                else 0
            ),
        )
        evidence = candidate["evidence"]
        similarities = [
            float(record["similarity"]) for record in evidence
        ]
        if similarities != sorted(similarities, reverse=True):
            raise EvidenceBundleError(
                f"evidence similarity order differs for {query_id}"
            )
        for record in evidence:
            record_id = record["record_id"]
            if record_id in record_ids:
                raise EvidenceBundleError(
                    f"evidence record IDs differ for {query_id}"
                )
            record_ids.add(record_id)
            if record["behavior"] != classify_behavior(
                record["response"]
            ):
                raise EvidenceBundleError(
                    f"evidence behavior differs for {query_id}"
                )
        if evidence and (
            float(evidence[0]["similarity"])
            != float(candidate["best_record_cosine"])
        ):
            raise EvidenceBundleError(
                f"evidence best-record cosine differs for {query_id}"
            )

    for system in RANK_SYSTEMS:
        ranks = {
            int(candidate["ranks"][system])
            for candidate in candidates.values()
        }
        if ranks != set(range(1, candidate_count + 1)):
            raise EvidenceBundleError(
                f"evidence bundle candidate schema differs for {query_id}"
            )

    for system in SYSTEMS:
        expected_metric = _derive_component_metric(
            candidates,
            system=system,
        )
        if any(
            not math.isclose(
                validated_metrics[system][name],
                expected_metric[name],
                rel_tol=0.0,
                abs_tol=METRIC_TOLERANCE,
            )
            for name in ("ndcg_at_10", "reciprocal_rank")
        ):
            raise EvidenceBundleError(
                "evidence component metrics differ for "
                f"{query_id} {system}"
            )

    expected_cohorts = derive_cohorts(
        {
            system: validated_metrics[system]["ndcg_at_10"]
            for system in SYSTEMS
        }
    )
    if list(cohorts) != expected_cohorts:
        raise EvidenceBundleError(
            f"evidence cohorts differ for {query_id}"
        )

    expected_best_record_order = sorted(
        candidates,
        key=lambda candidate_id: (
            -float(candidates[candidate_id]["best_record_cosine"]),
            candidate_id,
        ),
    )
    actual_best_record_order = sorted(
        candidates,
        key=lambda candidate_id: int(
            candidates[candidate_id]["ranks"]["best_record_cosine"]
        ),
    )
    if actual_best_record_order != expected_best_record_order:
        raise EvidenceBundleError(
            f"evidence best-record cosine ranks differ for {query_id}"
        )

    positive_ids = {
        candidate_id
        for candidate_id, candidate in candidates.items()
        if candidate["is_positive"]
    }
    positions = {
        system: {
            candidate_id: int(candidate["ranks"][system])
            for candidate_id, candidate in candidates.items()
        }
        for system in RANK_SYSTEMS
    }
    if (
        query["retrieval_outcome"]
        != derive_retrieval_outcome(positive_ids, positions)
        or query["mechanism"] != derive_mechanism(positive_ids, positions)
    ):
        raise EvidenceBundleError(
            f"evidence classification differs for {query_id}"
        )
    return frozenset(candidates)


def validate_evidence_bundle(
    payload: Any,
    *,
    expected_query_ids: Collection[str] | None = None,
    final_component_metrics: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
) -> frozenset[str]:
    """Validate all bundle-intrinsic semantics and optional final metrics.

    Non-final component ranks are source-bound by the task-specific builder.
    A generic bundle consumer can validate their complete permutations, but it
    cannot independently prove those ranks without the external artifacts.
    """

    if (
        not isinstance(payload, Mapping)
        or set(payload) != {"schema_version", "meta", "queries"}
        or type(payload.get("schema_version")) is not int
        or payload.get("schema_version") != 1
        or not isinstance(payload.get("queries"), Mapping)
        or any(
            not isinstance(query_id, str) or not query_id
            for query_id in payload.get("queries", {})
        )
    ):
        raise EvidenceBundleError(
            "evidence bundle query inventory differs"
        )
    queries = payload["queries"]
    if (
        expected_query_ids is not None
        and set(queries) != set(expected_query_ids)
    ):
        raise EvidenceBundleError(
            "evidence bundle query inventory differs"
        )
    meta = payload.get("meta")
    if (
        not isinstance(meta, Mapping)
        or set(meta)
        != {
            "query_count",
            "candidate_count",
            "evidence_per_candidate",
            "qrels_usage",
        }
        or type(meta["query_count"]) is not int
        or meta["query_count"] != len(queries)
        or type(meta["candidate_count"]) is not int
        or meta["candidate_count"] < 1
        or type(meta["evidence_per_candidate"]) is not int
        or meta["evidence_per_candidate"] != EVIDENCE_PER_CANDIDATE
        or meta["qrels_usage"] != QRELS_USAGE
    ):
        raise EvidenceBundleError(
            "evidence bundle meta schema differs"
        )

    candidate_inventory: frozenset[str] | None = None
    for query_id, query in queries.items():
        current_inventory = _validate_evidence_query(
            query,
            query_id=query_id,
            candidate_count=meta["candidate_count"],
            evidence_per_candidate=meta["evidence_per_candidate"],
        )
        if candidate_inventory is None:
            candidate_inventory = current_inventory
        elif current_inventory != candidate_inventory:
            raise EvidenceBundleError(
                "evidence bundle candidate inventory differs"
            )

    if final_component_metrics is not None:
        if set(final_component_metrics) != set(queries):
            raise EvidenceBundleError(
                "evidence final component metric inventory differs"
            )
        for query_id, query in queries.items():
            expected = _validate_metric_pair(
                final_component_metrics[query_id],
                label=f"generic evaluation {query_id}",
                schema_error=(
                    "generic final component metric schema differs"
                ),
            )
            actual = query["component_metrics"][
                "raw_hierarchical_dense_equal"
            ]
            if any(
                not math.isclose(
                    float(actual[name]),
                    expected[name],
                    rel_tol=0.0,
                    abs_tol=METRIC_TOLERANCE,
                )
                for name in ("ndcg_at_10", "reciprocal_rank")
            ):
                raise EvidenceBundleError(
                    "evidence final component metrics differ for "
                    f"{query_id}"
                )
    return candidate_inventory or frozenset()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise EvidenceBundleError(f"cannot load frozen JSON: {path}") from error


def _frozen_ids(inputs: Any) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(inputs, Mapping) or set(inputs) != {"query_ids", "candidate_ids"}:
        raise EvidenceBundleError("frozen input schema differs")
    query_ids = tuple(inputs["query_ids"])
    candidate_ids = tuple(inputs["candidate_ids"])
    for label, identifiers in (("query", query_ids), ("candidate", candidate_ids)):
        if (
            not identifiers
            or any(not isinstance(item, str) or not item for item in identifiers)
            or len(identifiers) != len(set(identifiers))
        ):
            raise EvidenceBundleError(f"{label} inventory differs")
    return query_ids, candidate_ids


def _rank_positions(
    ranking: Sequence[str], candidates: tuple[str, ...], label: str
) -> dict[str, int]:
    if len(ranking) != len(candidates) or set(ranking) != set(candidates):
        raise EvidenceBundleError(f"{label} candidate inventory is incomplete")
    return {candidate_id: rank for rank, candidate_id in enumerate(ranking, 1)}


def _load_ranking(
    path: Path, query_ids: tuple[str, ...], candidate_ids: tuple[str, ...], label: str
) -> dict[str, list[str]]:
    payload = _load_json(path)
    if not isinstance(payload, Mapping) or set(payload) != set(query_ids):
        raise EvidenceBundleError(f"{label} query inventory differs")
    rankings: dict[str, list[str]] = {}
    for query_id in query_ids:
        ranking = payload[query_id]
        if not isinstance(ranking, list) or any(not isinstance(item, str) for item in ranking):
            raise EvidenceBundleError(f"{label} ranking schema differs")
        _rank_positions(ranking, candidate_ids, label)
        rankings[query_id] = ranking
    return rankings


def _load_all_rankings(
    refinement_dir: Path,
    construction_dir: Path,
    query_ids: tuple[str, ...],
    candidate_ids: tuple[str, ...],
) -> dict[str, dict[str, list[str]]]:
    roots = {
        "dense_reference": refinement_dir,
        "profile_only": construction_dir,
        "dense_profile_equal": construction_dir,
        "raw_hierarchical": refinement_dir,
        "raw_hierarchical_dense_equal": refinement_dir,
    }
    return {
        name: _load_ranking(
            roots[name] / "rankings" / f"{name}.json",
            query_ids,
            candidate_ids,
            name,
        )
        for name in SYSTEMS
    }


def _load_selected_ids(
    path: Path, query_ids: tuple[str, ...], candidate_ids: tuple[str, ...]
) -> list[list[list[str]]]:
    payload = _load_json(path)
    if not isinstance(payload, list) or len(payload) != len(query_ids):
        raise EvidenceBundleError("selected evidence query inventory differs")
    selected: list[list[list[str]]] = []
    for query_records in payload:
        if not isinstance(query_records, list) or len(query_records) != len(candidate_ids):
            raise EvidenceBundleError("selected evidence candidate inventory differs")
        candidates: list[list[str]] = []
        for record_ids in query_records:
            if (
                not isinstance(record_ids, list)
                or len(record_ids) != 10
                or any(not isinstance(record_id, str) or not record_id for record_id in record_ids)
                or len(record_ids) != len(set(record_ids))
            ):
                raise EvidenceBundleError("selected evidence must contain persisted top 10")
            candidates.append(record_ids)
        selected.append(candidates)
    return selected


def _load_similarities(
    path: Path, expected_shape: tuple[int, int, int]
) -> np.ndarray:
    try:
        similarities = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise EvidenceBundleError("cannot load top-record similarities") from error
    if similarities.shape != expected_shape or similarities.dtype != np.dtype("<f8"):
        raise EvidenceBundleError(
            f"top-record similarity shape or dtype differs: {similarities.shape}"
        )
    return similarities


def _load_records(path: Path, candidate_ids: tuple[str, ...]) -> dict[str, Mapping[str, Any]]:
    payload = _load_json(path)
    if not isinstance(payload, list):
        raise EvidenceBundleError("record inventory differs")
    records: dict[str, Mapping[str, Any]] = {}
    candidates = set(candidate_ids)
    required = {"record_id", "candidate_id", "source_query_id", "question", "response"}
    for record in payload:
        if not isinstance(record, Mapping) or not required <= set(record):
            raise EvidenceBundleError("record schema differs")
        record_id = record["record_id"]
        candidate_id = record["candidate_id"]
        if (
            not isinstance(record_id, str)
            or not record_id
            or record_id in records
            or not isinstance(candidate_id, str)
            or candidate_id not in candidates
            or any(not isinstance(record[field], str) for field in required - {"record_id", "candidate_id"})
        ):
            raise EvidenceBundleError("record inventory differs")
        records[record_id] = record
    if not records:
        raise EvidenceBundleError("record inventory differs")
    return records


def _selected_evidence(
    record_ids: Sequence[str],
    similarities: np.ndarray,
    records: Mapping[str, Mapping[str, Any]],
    candidate_id: str,
) -> list[dict[str, Any]]:
    if len(record_ids) != 10 or similarities.shape != (10,):
        raise EvidenceBundleError("selected evidence must contain persisted top 10")
    if not np.isfinite(similarities).all() or np.any(similarities[:-1] < similarities[1:]):
        raise EvidenceBundleError("selected similarities must be finite and non-increasing")
    selected_records: list[Mapping[str, Any]] = []
    for record_id in record_ids:
        record = records.get(record_id)
        if record is None or record.get("candidate_id") != candidate_id:
            raise EvidenceBundleError(
                f"selected record membership differs for {candidate_id}"
            )
        selected_records.append(record)
    return [
        {
            "record_id": record_id,
            "source_query_id": str(record["source_query_id"]),
            "question": str(record["question"]),
            "response": str(record["response"]),
            "similarity": float(similarity),
            "behavior": classify_behavior(str(record["response"])),
        }
        for record_id, similarity, record in zip(
            record_ids[:3], similarities[:3], selected_records[:3], strict=True
        )
    ]


def _evaluation_rows(
    path: Path, query_ids: tuple[str, ...]
) -> dict[str, dict[str, dict[str, float]]]:
    payload = _load_json(path)
    if isinstance(payload, Mapping) and isinstance(payload.get("evaluation"), Mapping):
        payload = payload["evaluation"]
    if not isinstance(payload, Mapping) or not isinstance(payload.get("per_query_rows"), list):
        raise EvidenceBundleError("evaluation row schema differs")
    rows: dict[str, dict[str, dict[str, float]]] = {query_id: {} for query_id in query_ids}
    for row in payload["per_query_rows"]:
        if not isinstance(row, Mapping):
            raise EvidenceBundleError("evaluation row schema differs")
        query_id, system = row.get("query_id"), row.get("system")
        if query_id not in rows or system not in SYSTEMS or system in rows[query_id]:
            raise EvidenceBundleError("evaluation row inventory differs")
        try:
            ndcg_value = row["ndcg_at_10"]
            reciprocal_rank_value = row["reciprocal_rank"]
        except KeyError as error:
            raise EvidenceBundleError("evaluation row schema differs") from error
        ndcg = _unit_interval_metric(
            ndcg_value,
            label="evaluation ndcg_at_10",
        )
        reciprocal_rank = _unit_interval_metric(
            reciprocal_rank_value,
            label="evaluation reciprocal_rank",
        )
        rows[query_id][str(system)] = {
            "ndcg_at_10": ndcg,
            "reciprocal_rank": reciprocal_rank,
        }
    if any(set(metrics) != set(SYSTEMS) for metrics in rows.values()):
        raise EvidenceBundleError("evaluation row inventory differs")
    return rows


def _build_query_entry(
    *,
    query_id: str,
    query_text: str,
    candidate_ids: tuple[str, ...],
    display_ids: list[str],
    evidence_ids: list[str],
    positives: set[str],
    qrels: Mapping[str, float],
    positions: Mapping[str, Mapping[str, int]],
    evaluation: Mapping[str, Mapping[str, float]],
    selected_ids: Sequence[Sequence[str]],
    similarities: np.ndarray,
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    evidence_set = set(evidence_ids)
    candidates: dict[str, dict[str, Any]] = {}
    for candidate_index, candidate_id in enumerate(candidate_ids):
        candidate_similarities = similarities[candidate_index]
        selected_evidence = _selected_evidence(
            selected_ids[candidate_index],
            candidate_similarities,
            records,
            candidate_id,
        )
        if not np.isfinite(candidate_similarities).all() or np.any(
            candidate_similarities[:-1] < candidate_similarities[1:]
        ):
            raise EvidenceBundleError("selected similarities must be finite and non-increasing")
        candidates[candidate_id] = {
            "candidate_id": candidate_id,
            "qrel_grade": float(qrels[candidate_id]),
            "is_positive": candidate_id in positives,
            "ranks": {name: int(positions[name][candidate_id]) for name in positions},
            "best_record_cosine": float(candidate_similarities[0]),
            "mean_top_10_cosine": float(np.mean(candidate_similarities, dtype=np.float64)),
            "evidence": selected_evidence if candidate_id in evidence_set else [],
        }
    return {
        "query_id": query_id,
        "query_text": query_text,
        "display_candidate_ids": display_ids,
        "evidence_candidate_ids": evidence_ids,
        "component_metrics": {name: dict(evaluation[name]) for name in SYSTEMS},
        "cohorts": derive_cohorts(
            {name: evaluation[name]["ndcg_at_10"] for name in SYSTEMS}
        ),
        "context_risk": has_context_risk(query_text),
        "retrieval_outcome": derive_retrieval_outcome(positives, positions),
        "mechanism": derive_mechanism(positives, positions),
        "candidates": candidates,
    }


def build_evidence_bundle(
    *,
    refinement_dir: Path,
    construction_dir: Path,
    evaluation_path: Path,
    queries_path: Path,
    qrels_path: Path,
) -> dict[str, Any]:
    """Join validated persisted artifacts without recomputing any neighbors."""

    try:
        inputs = _load_json(refinement_dir / "inputs.json")
        query_ids, candidate_ids = _frozen_ids(inputs)
        queries = load_queries_tsv(queries_path)
        if tuple(queries) != query_ids:
            raise EvidenceBundleError("query inventory or ordering differs")
        try:
            qrels = load_complete_qrels(qrels_path, expected_query_ids=query_ids)
            validate_qrel_inventories(
                qrels, query_ids=query_ids, candidate_ids=candidate_ids
            )
        except ValueError as error:
            raise EvidenceBundleError(f"qrel inventory differs: {error}") from error

        rankings = _load_all_rankings(
            refinement_dir, construction_dir, query_ids, candidate_ids
        )
        selected_ids = _load_selected_ids(
            refinement_dir / "selected_record_ids.json", query_ids, candidate_ids
        )
        similarities = _load_similarities(
            refinement_dir / "top_record_similarities.npy",
            (len(query_ids), len(candidate_ids), 10),
        )
        records = _load_records(construction_dir / "records.json", candidate_ids)
        evaluation = _evaluation_rows(evaluation_path, query_ids)
    except EvidenceBundleError:
        raise
    except (KeyError, OSError, TypeError, ValueError) as error:
        raise EvidenceBundleError(str(error)) from error

    query_payload: dict[str, dict[str, Any]] = {}
    candidate_indices = {candidate_id: index for index, candidate_id in enumerate(candidate_ids)}
    for query_index, query_id in enumerate(query_ids):
        positions = {
            name: _rank_positions(rankings[name][query_id], candidate_ids, name)
            for name in SYSTEMS
        }
        best_record_order = sorted(
            candidate_ids,
            key=lambda candidate_id: (
                -float(similarities[query_index, candidate_indices[candidate_id], 0]),
                candidate_id,
            ),
        )
        positions["best_record_cosine"] = {
            candidate_id: rank
            for rank, candidate_id in enumerate(best_record_order, 1)
        }
        positives = {
            candidate_id
            for candidate_id, grade in qrels[query_id].items()
            if grade > 0.0
        }
        final = rankings["raw_hierarchical_dense_equal"][query_id]
        display_set = set(final[:20]) | positives
        evidence_set = set(final[:10]) | positives
        display_ids = [candidate_id for candidate_id in final if candidate_id in display_set]
        evidence_ids = [candidate_id for candidate_id in final if candidate_id in evidence_set]
        query_payload[query_id] = _build_query_entry(
            query_id=query_id,
            query_text=queries[query_id],
            candidate_ids=candidate_ids,
            display_ids=display_ids,
            evidence_ids=evidence_ids,
            positives=positives,
            qrels=qrels[query_id],
            positions=positions,
            evaluation=evaluation[query_id],
            selected_ids=selected_ids[query_index],
            similarities=similarities[query_index],
            records=records,
        )
    bundle = {
        "schema_version": 1,
        "meta": {
            "query_count": len(query_ids),
            "candidate_count": len(candidate_ids),
            "evidence_per_candidate": EVIDENCE_PER_CANDIDATE,
            "qrels_usage": QRELS_USAGE,
        },
        "queries": query_payload,
    }
    validate_evidence_bundle(
        bundle,
        expected_query_ids=query_ids,
    )
    return bundle
