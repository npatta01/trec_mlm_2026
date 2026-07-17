#!/usr/bin/env python3
"""Run the exploratory Issue 13 matched-support clustering preflight."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.cluster import KMeans

from trec_mlm.artifact_io import sha256_file, write_pretty_json
from trec_mlm.dense_query_knn import array_sha256, cosine_similarity_matrix
from trec_mlm.metrics import evaluate_rankings
from trec_mlm.query_knn import load_development_data


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class FrozenInputContract:
    """Exact frozen identity and layout expected by the real preflight."""

    query_file_sha256: str
    qrel_file_sha256: str
    embedding_file_sha256: str
    encoder_metadata_sha256: str
    query_input_sha256: str
    embedding_logical_sha256: str
    similarity_logical_sha256: str
    embedding_shape: tuple[int, int]
    embedding_dtype: str


FROZEN_INPUT_CONTRACT = FrozenInputContract(
    query_file_sha256="862ff85381b8055cd927ada25c56666594ab1ec3c21e182beb20c58f6c637f3c",
    qrel_file_sha256="3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35",
    embedding_file_sha256="d96ef7a1e45f40621c303c855673efa1de7c02d7dacd16cf0cee93c562a09559",
    encoder_metadata_sha256="5bdfe3f5297467f91bd992e47c6555f582b536c15b97d853dd904fdcda871091",
    query_input_sha256="98faa04f979e428933c3d8e67dac43cf4dded112bd2c16969b3482a1d52886d9",
    embedding_logical_sha256="6d197055eb62ab9809ae321b0f60072e20f17524861e6a989b49cd35c389055b",
    similarity_logical_sha256="80c69a8dca29cd5e9a4de08dc4e44553fabf813428dd4956f7ecbeb2852968ed",
    embedding_shape=(322, 768),
    embedding_dtype="float32",
)


def query_input_sha256(
    query_ids: Sequence[str], query_texts: Sequence[str]
) -> str:
    """Hash canonical MPNet query rows, including their exact order."""

    digest = hashlib.sha256()
    for query_id, query_text in zip(query_ids, query_texts, strict=True):
        digest.update(query_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(query_text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _require_equal(observed: object, expected: object, *, name: str) -> None:
    if observed != expected:
        raise ValueError(f"{name} mismatch: expected {expected!r}, observed {observed!r}")


def validate_frozen_inputs(
    *,
    query_path: Path,
    qrel_path: Path,
    embedding_path: Path,
    metadata_path: Path,
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    embeddings: ArrayLike,
    contract: FrozenInputContract = FROZEN_INPUT_CONTRACT,
) -> dict[str, str]:
    """Fail closed unless source bytes, row order, and arrays match the lock."""

    array = np.asarray(embeddings)
    similarity = cosine_similarity_matrix(array)
    observed = {
        "queries_sha256": sha256_file(query_path),
        "qrels_sha256": sha256_file(qrel_path),
        "embedding_file_sha256": sha256_file(embedding_path),
        "query_input_sha256": query_input_sha256(query_ids, query_texts),
        "embedding_logical_sha256": array_sha256(array),
        "similarity_logical_sha256": array_sha256(similarity),
        "encoder_metadata_sha256": sha256_file(metadata_path),
    }
    _require_equal(
        observed["queries_sha256"], contract.query_file_sha256, name="query file SHA-256"
    )
    _require_equal(
        observed["qrels_sha256"], contract.qrel_file_sha256, name="qrel file SHA-256"
    )
    _require_equal(
        observed["embedding_file_sha256"],
        contract.embedding_file_sha256,
        name="embedding file SHA-256",
    )
    _require_equal(
        observed["encoder_metadata_sha256"],
        contract.encoder_metadata_sha256,
        name="encoder metadata SHA-256",
    )
    _require_equal(
        observed["query_input_sha256"],
        contract.query_input_sha256,
        name="query input SHA-256",
    )
    _require_equal(
        observed["embedding_logical_sha256"],
        contract.embedding_logical_sha256,
        name="embedding logical SHA-256",
    )
    _require_equal(
        observed["similarity_logical_sha256"],
        contract.similarity_logical_sha256,
        name="similarity logical SHA-256",
    )
    _require_equal(array.shape, contract.embedding_shape, name="embedding shape")
    _require_equal(str(array.dtype), contract.embedding_dtype, name="embedding dtype")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))["mpnet_base_v2"]
    for field, expected in {
        "embedding_dimension": contract.embedding_shape[1],
        "embedding_sha256": contract.embedding_logical_sha256,
        "input_sha256": contract.query_input_sha256,
        "normalized": True,
        "query_count": contract.embedding_shape[0],
        "similarity_sha256": contract.similarity_logical_sha256,
    }.items():
        _require_equal(metadata.get(field), expected, name=f"encoder metadata {field}")
    return observed


def _finite_vector(values: ArrayLike, *, name: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def select_positive_neighbors(
    similarity_row: ArrayLike,
    support_indices: Sequence[int],
    query_ids: Sequence[str],
    count: int,
) -> tuple[int, ...]:
    """Select exactly ``count`` positive-cosine support rows deterministically."""

    similarities = _finite_vector(similarity_row, name="similarity_row")
    identifiers = tuple(query_ids)
    support = tuple(support_indices)
    if len(identifiers) != len(similarities):
        raise ValueError("query_ids must match the similarity inventory")
    if type(count) is not int or count < 1:
        raise ValueError("count must be a positive integer")
    if len(set(support)) != len(support):
        raise ValueError("support indices must be unique")
    if any(type(index) is not int or index < 0 or index >= len(similarities) for index in support):
        raise ValueError("support index lies outside the similarity inventory")
    positive = [index for index in support if float(similarities[index]) > 0.0]
    positive.sort(key=lambda index: (-float(similarities[index]), identifiers[index]))
    if len(positive) < count:
        raise ValueError("insufficient positive-cosine support for requested count")
    return tuple(positive[:count])


def mean_grades(grades: ArrayLike, indices: Sequence[int]) -> NDArray[np.float64]:
    """Return a uniform candidate-grade mean over nonempty support rows."""

    matrix = np.asarray(grades, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("grades must be two-dimensional")
    if not np.isfinite(matrix).all():
        raise ValueError("grades must contain only finite values")
    support = tuple(indices)
    if not support:
        raise ValueError("grade support must be nonempty")
    if len(set(support)) != len(support):
        raise ValueError("grade support indices must be unique")
    if any(type(index) is not int or index < 0 or index >= matrix.shape[0] for index in support):
        raise ValueError("grade support index lies outside the matrix")
    return np.mean(matrix[np.asarray(support, dtype=np.intp)], axis=0, dtype=np.float64)


def rank_candidates(
    scores: ArrayLike,
    prior: ArrayLike,
    candidate_ids: Sequence[str],
) -> tuple[str, ...]:
    """Rank by score, then fold prior, then candidate ID."""

    score_vector = _finite_vector(scores, name="scores")
    prior_vector = _finite_vector(prior, name="prior")
    candidates = tuple(candidate_ids)
    if not (len(score_vector) == len(prior_vector) == len(candidates)):
        raise ValueError("scores, prior, and candidate_ids must have equal length")
    if len(set(candidates)) != len(candidates) or any(
        type(candidate) is not str or not candidate for candidate in candidates
    ):
        raise ValueError("candidate_ids must be nonempty and unique")
    order = sorted(
        range(len(candidates)),
        key=lambda index: (
            -float(score_vector[index]),
            -float(prior_vector[index]),
            candidates[index],
        ),
    )
    return tuple(candidates[index] for index in order)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("summary values must be a nonempty finite vector")
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
    }


def _metric_summary(evaluation: Any) -> dict[str, float | int]:
    return {
        "ndcg_at_10": float(evaluation.macro_ndcg_at_k),
        "mrr": float(evaluation.mean_reciprocal_rank),
        "zero_ndcg_count": sum(
            metric.ndcg_at_k == 0.0 for metric in evaluation.per_query.values()
        ),
    }


def _paired_summary(cluster_values: Sequence[float], matched_values: Sequence[float]) -> dict[str, float | int]:
    cluster = np.asarray(cluster_values, dtype=np.float64)
    matched = np.asarray(matched_values, dtype=np.float64)
    if cluster.shape != matched.shape or cluster.ndim != 1 or not len(cluster):
        raise ValueError("paired metric vectors must be equal nonempty vectors")
    delta = cluster - matched
    return {
        "mean_ndcg_at_10_delta": float(np.mean(delta)),
        "wins": int(np.count_nonzero(delta > 0.0)),
        "ties": int(np.count_nonzero(delta == 0.0)),
        "losses": int(np.count_nonzero(delta < 0.0)),
    }


def run_preflight(
    *,
    embeddings: ArrayLike,
    grades: ArrayLike,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
    cluster_counts: Sequence[int] = (20, 40),
) -> dict[str, object]:
    """Compare hard clusters with equal-support unweighted neighbors."""

    embedding_matrix = np.asarray(embeddings, dtype=np.float32)
    grade_matrix = np.asarray(grades, dtype=np.float64)
    queries = tuple(query_ids)
    candidates = tuple(candidate_ids)
    counts = tuple(cluster_counts)
    if embedding_matrix.ndim != 2 or not np.isfinite(embedding_matrix).all():
        raise ValueError("embeddings must be a finite two-dimensional matrix")
    if grade_matrix.ndim != 2 or not np.isfinite(grade_matrix).all():
        raise ValueError("grades must be a finite two-dimensional matrix")
    if embedding_matrix.shape[0] != len(queries) or grade_matrix.shape != (
        len(queries),
        len(candidates),
    ):
        raise ValueError("input arrays do not match query/candidate inventories")
    if (
        len(set(queries)) != len(queries)
        or len(set(candidates)) != len(candidates)
        or any(type(value) is not str or not value for value in queries + candidates)
    ):
        raise ValueError("query and candidate inventories must be nonempty and unique")
    if any(type(value) is not int or value < 2 or value >= len(queries) for value in counts):
        raise ValueError("cluster counts must be integers in [2, query_count)")
    if len(set(counts)) != len(counts):
        raise ValueError("cluster counts must be unique")
    if np.any(grade_matrix < 0.0) or np.any(grade_matrix > 2.0):
        raise ValueError("grades must lie in [0,2]")
    norms = np.linalg.norm(embedding_matrix, axis=1)
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("embeddings must be L2-normalized")

    similarities = np.asarray(embedding_matrix @ embedding_matrix.T, dtype=np.float32)
    qrels = {
        query_id: {
            candidate_id: float(grade_matrix[q_index, c_index])
            for c_index, candidate_id in enumerate(candidates)
        }
        for q_index, query_id in enumerate(queries)
    }
    output_by_count: dict[str, object] = {}
    all_indices = tuple(range(len(queries)))
    for cluster_count in counts:
        cluster_rankings: dict[str, tuple[str, ...]] = {}
        matched_rankings: dict[str, tuple[str, ...]] = {}
        supports: list[float] = []
        cluster_unique: list[float] = []
        matched_unique: list[float] = []
        cluster_zeros: list[float] = []
        matched_zeros: list[float] = []
        centroid_margins: list[float] = []
        overlaps: list[float] = []
        for target in all_indices:
            support = tuple(index for index in all_indices if index != target)
            estimator = KMeans(
                n_clusters=cluster_count,
                init="k-means++",
                random_state=2026,
                n_init=10,
                algorithm="lloyd",
                max_iter=300,
                tol=1e-4,
            )
            support_array = np.asarray(support, dtype=np.intp)
            labels = estimator.fit_predict(embedding_matrix[support_array])
            centers = np.asarray(estimator.cluster_centers_, dtype=np.float64)
            target_row = np.asarray(embedding_matrix[target], dtype=np.float64)
            distances = np.sum((centers - target_row) ** 2, axis=1, dtype=np.float64)
            nearest = np.argsort(distances, kind="stable")
            selected_label = int(nearest[0])
            member_positions = np.flatnonzero(labels == selected_label)
            members = tuple(int(support_array[position]) for position in member_positions)
            matched = select_positive_neighbors(
                similarities[target], support, queries, len(members)
            )
            prior = mean_grades(grade_matrix, support)
            cluster_scores = mean_grades(grade_matrix, members)
            matched_scores = mean_grades(grade_matrix, matched)
            query_id = queries[target]
            cluster_rankings[query_id] = rank_candidates(cluster_scores, prior, candidates)
            matched_rankings[query_id] = rank_candidates(matched_scores, prior, candidates)
            supports.append(float(len(members)))
            cluster_unique.append(float(len(np.unique(cluster_scores))))
            matched_unique.append(float(len(np.unique(matched_scores))))
            cluster_zeros.append(float(np.count_nonzero(cluster_scores == 0.0)))
            matched_zeros.append(float(np.count_nonzero(matched_scores == 0.0)))
            centroid_margins.append(float(distances[nearest[1]] - distances[nearest[0]]))
            overlaps.append(float(len(set(members) & set(matched)) / len(members)))

        cluster_evaluation = evaluate_rankings(cluster_rankings, qrels, k=10)
        matched_evaluation = evaluate_rankings(matched_rankings, qrels, k=10)
        cluster_ndcg = [
            cluster_evaluation.per_query[query_id].ndcg_at_k for query_id in queries
        ]
        matched_ndcg = [
            matched_evaluation.per_query[query_id].ndcg_at_k for query_id in queries
        ]
        output_by_count[str(cluster_count)] = {
            "cluster_mean": _metric_summary(cluster_evaluation),
            "matched_neighbor_mean": _metric_summary(matched_evaluation),
            "paired_delta_cluster_minus_matched": _paired_summary(
                cluster_ndcg, matched_ndcg
            ),
            "support": _summary(supports),
            "score_granularity": {
                "cluster_distinct_scores": _summary(cluster_unique),
                "matched_distinct_scores": _summary(matched_unique),
                "cluster_zero_scores": _summary(cluster_zeros),
                "matched_zero_scores": _summary(matched_zeros),
            },
            "target_assignment": {
                "centroid_squared_distance_margin": _summary(centroid_margins),
                "cluster_neighbor_support_overlap_rate": _summary(overlaps),
            },
        }

    result: dict[str, object] = {
        "schema": "trec-mlm-issue13-matched-support-preflight-v1",
        "status": "exploratory_not_promotable",
        "question": "Does a hard query cluster beat equal-support nearest neighbors?",
        "completed_issue13_result": {
            "outcome": "reject",
            "baseline": {
                "ndcg_at_10": 0.35985474409504775,
                "mrr": 0.40406330811168817,
                "zero_ndcg_count": 116,
            },
            "candidate": {
                "ndcg_at_10": 0.34868084786664477,
                "mrr": 0.380158644416514,
                "zero_ndcg_count": 123,
            },
            "candidate_minus_baseline": {
                "ndcg_at_10": -0.011173896228364245,
                "mrr": -0.023904663695149868,
            },
            "ndcg_at_10_delta_95pct_ci": [
                -0.02543034280251127,
                0.002637356577463938,
            ],
            "integrity_checks_passed": True,
            "effectiveness_gates_passed": False,
            "query_ndcg_win_tie_loss": {"wins": 66, "ties": 168, "losses": 88},
            "conditional_component_diagnostics": {
                "frozen_baseline": {
                    "ndcg_at_10": 0.35985474409504775,
                    "mrr": 0.40406330811168817,
                    "zero_ndcg_count": 116,
                },
                "selected_local_only": {
                    "ndcg_at_10": 0.35882195388143606,
                    "mrr": 0.40225940880779676,
                    "zero_ndcg_count": 117,
                },
                "selected_local_plus_prior": {
                    "ndcg_at_10": 0.35804282537688814,
                    "mrr": 0.4024982237307636,
                    "zero_ndcg_count": 118,
                },
                "selected_domain_only": {
                    "ndcg_at_10": 0.2808510162816271,
                    "mrr": 0.31801218695694844,
                    "zero_ndcg_count": 151,
                },
                "selected_local_plus_domain_no_prior": {
                    "ndcg_at_10": 0.3486392536103764,
                    "mrr": 0.38009715829891116,
                    "zero_ndcg_count": 123,
                },
                "domain_effect_final_minus_local_plus_prior": -0.009361977510243338,
                "prior_effect_final_minus_no_prior_blend": 0.00004159425626837854,
            },
            "selection": {
                "local_weight_0_75": 322,
                "local_weight_0_50": 0,
                "neighbors_50": 287,
                "neighbors_20": 35,
                "clusters_20": 196,
                "clusters_40": 126,
                "prior_0": 249,
                "prior_0_1": 73,
            },
            "outer_signal_granularity": {
                "local_median_distinct_scores": 95.0,
                "domain_median_distinct_scores": 8.0,
                "domain_mean_zero_scores": 57.00310559006211,
                "queries_with_domain_top_score_tie": 96,
            },
            "provenance": {
                "decision_sha256": "cac8599227f170aa8123155b2690d7fc9510261042f54d6cc9e56eb77b2ed7a7",
                "result_manifest_sha256": "4ae0da8f0f34967258a44d2f99d76f5e31ffa6c750fb2caa02f06f7a9d63c805",
                "fit_semantic_sha256": "a48f7e1d6087a26cc372f479c284c310f7d6265a34fb6c3b233e05a9de85f313",
            },
            "reproduced_checks": {
                "baseline": True,
                "component_recomposition": True,
                "final_ranks": True,
                "result_validation": True,
            },
        },
        "query_count": len(queries),
        "candidate_count": len(candidates),
        "fit_count": len(queries) * len(counts),
        "cluster_counts": output_by_count,
    }
    result["method"] = {
        "cluster_counts": list(counts),
        "clusterer": {
            "algorithm": "lloyd",
            "init": "k-means++",
            "max_iter": 300,
            "n_init": 10,
            "random_state": 2026,
            "tol": 1e-4,
        },
        "matched_control": "uniform mean over equal-count positive-cosine neighbors",
        "tie_break": "fold global prior descending, then candidate ID ascending",
    }
    result["decision"] = {
        "rule": "continue clustering only if any hard cluster beats its matched control on mean nDCG@10",
        "continue_clustering": any(
            row["paired_delta_cluster_minus_matched"]["mean_ndcg_at_10_delta"] > 0.0
            for row in output_by_count.values()
        ),
    }
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=ROOT / "data/dev/llm_dev_data.tsv")
    parser.add_argument("--qrels", type=Path, default=ROOT / "data/dev/llm_dev_qrels.txt")
    parser.add_argument(
        "--embeddings",
        type=Path,
        default=ROOT / "artifacts/frozen_dense_input/mpnet_base_v2.npy",
    )
    parser.add_argument(
        "--encoder-metadata",
        type=Path,
        default=ROOT / "artifacts/frozen_dense_input/encoder_metadata.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "reports/issue_13_matched_support_preflight.json",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data = load_development_data(args.queries, args.qrels)
    query_ids = tuple(sorted(data.queries))
    candidate_ids = tuple(sorted(data.models))
    embeddings = np.load(args.embeddings, allow_pickle=False)
    sources = validate_frozen_inputs(
        query_path=args.queries,
        qrel_path=args.qrels,
        embedding_path=args.embeddings,
        metadata_path=args.encoder_metadata,
        query_ids=query_ids,
        query_texts=tuple(data.queries[query_id] for query_id in query_ids),
        embeddings=embeddings,
    )
    grades = np.asarray(
        [
            [data.qrels[query_id][candidate_id] for candidate_id in candidate_ids]
            for query_id in query_ids
        ],
        dtype=np.float64,
    )
    result = run_preflight(
        embeddings=embeddings,
        grades=grades,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
    )
    result["sources"] = sources
    write_pretty_json(args.output, result)


if __name__ == "__main__":
    main()
