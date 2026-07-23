"""Build a renderer-neutral bundle from frozen E5 diagnostic artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

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


def classify_behavior(response: str) -> str:
    """Classify an exact persisted response without judging its correctness."""

    normalized = re.sub(r"[^a-z0-9]+", " ", response.casefold()).strip()
    if not normalized:
        return "empty"
    if normalized in EXACT_REFUSALS:
        return "exact refusal"
    return "substantive response"


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
            ndcg = float(row["ndcg_at_10"])
            reciprocal_rank = float(row["reciprocal_rank"])
        except (KeyError, TypeError, ValueError) as error:
            raise EvidenceBundleError("evaluation row schema differs") from error
        if not np.isfinite((ndcg, reciprocal_rank)).all():
            raise EvidenceBundleError("evaluation metrics must be finite")
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
    return {
        "schema_version": 1,
        "meta": {
            "query_count": len(query_ids),
            "candidate_count": len(candidate_ids),
            "evidence_per_candidate": 3,
            "qrels_usage": "Post-freeze evaluation and failure inspection only",
        },
        "queries": query_payload,
    }
