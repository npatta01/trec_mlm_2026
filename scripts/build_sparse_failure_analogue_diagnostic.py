#!/usr/bin/env python3
"""Diagnose whether similar queries identify sole-positive failure labels.

The diagnostic is post-qrel and development-only.  Development-query transfer
always excludes the target query.  Discovery responses are reduced to refusal
and insufficiency flags; a non-refusal is an attempt proxy, not correctness.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
from statistics import median
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from trec_mlm.dense_query_knn import (
    DEFAULT_ENCODERS,
    encode_frozen_queries,
)
from trec_mlm.query_knn import load_development_data


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUERIES = ROOT / "data/dev/llm_dev_data.tsv"
DEFAULT_QRELS = ROOT / "data/dev/llm_dev_qrels.txt"
DEFAULT_DISCOVERY = ROOT / "data/dev/llm_discovery_data.json"
DEFAULT_EXPLORER = (
    ROOT / "reports/trec_2026_candidate_semantic_profiles_query_explorer.html"
)
DEFAULT_PROFILES = ROOT / "reports/trec_2026_llm_profiles.json"
DEFAULT_JSON = ROOT / "reports/trec_2026_sparse_failure_analogue_diagnostic.json"
DEFAULT_TSV = ROOT / "reports/trec_2026_sparse_failure_analogue_cases.tsv"

EXPLORER_DATA_RE = re.compile(
    r'<script id="explorer-data" type="application/json">(.*?)</script>',
    re.DOTALL,
)
NON_ALPHA_RE = re.compile(r"[^a-z]+")
WORD_RE = re.compile(r"\b\w+\b")
INSUFFICIENCY_RE = re.compile(
    r"(?:i don['’]?t know|i do not know|not enough information|"
    r"insufficient information|does not provide|do not provide|"
    r"cannot determine|can['’]?t determine|unable to determine|"
    r"no information|not mentioned|not specified|"
    r"not directly (?:addressed|stated)|cannot be definitively)",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--explorer", type=Path, default=DEFAULT_EXPLORER)
    parser.add_argument("--profiles", type=Path, default=DEFAULT_PROFILES)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-tsv", type=Path, default=DEFAULT_TSV)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--torch-threads", type=int, default=16)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_response(response: str) -> str:
    return NON_ALPHA_RE.sub(" ", response.lower()).strip()


def is_exact_refusal(response: str) -> bool:
    return normalize_response(response) in {"i don t know", "i do not know"}


def response_status(response: str) -> str:
    if is_exact_refusal(response):
        return "exact_refusal"
    if INSUFFICIENCY_RE.search(response):
        return "attempt_with_insufficiency"
    return "non_refusal_attempt"


def is_blinded_screen_attempt(response: str) -> bool:
    """Reproduce the deliberately narrow blinded-screen attempt predicate."""

    return response.strip().lower() != "i don't know"


def load_explorer(path: Path) -> dict[str, Any]:
    payload = path.read_text(encoding="utf-8")
    match = EXPLORER_DATA_RE.search(payload)
    if match is None:
        raise ValueError(f"{path}: explorer-data payload not found")
    return json.loads(html.unescape(match.group(1)))


def rank_position(
    scores: np.ndarray,
    *,
    target_index: int,
    candidate_ids: list[str],
) -> int:
    names = np.asarray(candidate_ids)
    order = np.lexsort((names, -scores))
    return int(np.where(order == target_index)[0][0]) + 1


def json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    args = parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    development = load_development_data(args.queries, args.qrels)
    query_ids = list(development.queries)
    query_texts = [development.queries[query_id] for query_id in query_ids]
    candidate_ids = list(development.models)
    candidate_index = {candidate_id: i for i, candidate_id in enumerate(candidate_ids)}
    grades = np.asarray(
        [
            [development.qrels[query_id][candidate_id] for candidate_id in candidate_ids]
            for query_id in query_ids
        ],
        dtype=np.float64,
    )

    explorer = load_explorer(args.explorer)
    explorer_queries = {row["query_id"]: row for row in explorer["queries"]}
    if set(explorer_queries) != set(query_ids):
        raise ValueError("explorer and development query inventories differ")

    failures: list[str] = []
    for query_id in query_ids:
        positive_count = sum(
            grade > 0 for grade in development.qrels[query_id].values()
        )
        if (
            positive_count == 1
            and explorer_queries[query_id]["primary_ndcg_at_10"] == 0
        ):
            failures.append(query_id)
    if len(failures) != 75:
        raise ValueError(f"expected 75 sole-positive failures, found {len(failures)}")

    discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
    discovery_ids = list(discovery)
    discovery_texts = [discovery[query_id]["query"] for query_id in discovery_ids]
    if len(discovery_ids) != 7_759:
        raise ValueError(
            f"expected 7,759 discovery queries, found {len(discovery_ids)}"
        )

    profiles_payload = json.loads(args.profiles.read_text(encoding="utf-8"))
    profiles = {
        row["candidate_id"]: row for row in profiles_payload["profiles"]
    }
    if set(profiles) != set(candidate_ids):
        raise ValueError("profile and qrel candidate inventories differ")

    # Reproduce the all-75 blinded lexical screen.  The joint fit is qrel-free;
    # qrels are used only to select and evaluate the already-frozen failure set.
    lexical_vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    lexical_matrix = lexical_vectorizer.fit_transform(
        discovery_texts + [development.queries[query_id] for query_id in failures]
    )
    discovery_lexical = lexical_matrix[: len(discovery_ids)]
    target_lexical = lexical_matrix[len(discovery_ids) :]
    lexical_similarities = linear_kernel(target_lexical, discovery_lexical)
    attempt_matrix = np.asarray(
        [
            [
                is_blinded_screen_attempt(
                    discovery[discovery_id]["responses"][candidate_id]
                )
                for candidate_id in candidate_ids
            ]
            for discovery_id in discovery_ids
        ],
        dtype=bool,
    )

    spec = next(spec for spec in DEFAULT_ENCODERS if spec.key == "mpnet_base_v2")
    embeddings_by_encoder, encoder_metadata = encode_frozen_queries(
        query_ids=query_ids + discovery_ids,
        query_texts=query_texts + discovery_texts,
        specs=(spec,),
        device="cpu",
        batch_size=args.batch_size,
        torch_threads=args.torch_threads,
        show_progress_bar=False,
    )
    embeddings = embeddings_by_encoder[spec.key]
    development_embeddings = embeddings[: len(query_ids)]
    discovery_embeddings = embeddings[len(query_ids) :]
    development_similarity = development_embeddings @ development_embeddings.T
    np.fill_diagonal(development_similarity, -2.0)
    discovery_similarity = development_embeddings @ discovery_embeddings.T

    query_name_array = np.asarray(query_ids)
    discovery_name_array = np.asarray(discovery_ids)
    top_k_overlap = {1: 0, 3: 0, 5: 0, 10: 0}
    nearest_agreement_by_query: dict[str, bool] = {}
    nearest_similarity_by_query: dict[str, float] = {}

    for query_id in query_ids:
        target_index = query_ids.index(query_id)
        order = np.lexsort(
            (query_name_array, -development_similarity[target_index])
        )
        target_positives = set(
            np.where(grades[target_index] > 0)[0].tolist()
        )
        nearest_positive = set(np.where(grades[order[0]] > 0)[0].tolist())
        nearest_agreement_by_query[query_id] = bool(
            target_positives & nearest_positive
        )
        nearest_similarity_by_query[query_id] = float(
            development_similarity[target_index, order[0]]
        )

    case_rows: list[dict[str, Any]] = []
    group_attempts: dict[str, list[dict[str, Any]]] = {
        key: []
        for key in (
            "official_positive",
            "primary_top1",
            "profile_top1",
            "dense_top1",
        )
    }
    component_counts = {
        "profile_top10_only": 0,
        "dense_top10_only": 0,
        "both_top10": 0,
        "neither_top10": 0,
    }

    for failure_index, query_id in enumerate(failures):
        target_index = query_ids.index(query_id)
        explorer_row = explorer_queries[query_id]
        pairs = explorer_row["pairs"]
        positive_pair = next(pair for pair in pairs if pair["grade"] > 0)
        official_positive = positive_pair["candidate_id"]
        official_index = candidate_index[official_positive]

        lexical_similarity = lexical_similarities[failure_index]
        attempted_similarity = np.where(
            attempt_matrix,
            lexical_similarity[:, np.newaxis],
            0.0,
        )
        best_similarity = attempted_similarity.max(axis=0)
        top5_similarity_sum = np.partition(
            attempted_similarity,
            -5,
            axis=0,
        )[-5:].sum(axis=0)
        best_rank = 1 + int(
            np.sum(best_similarity > best_similarity[official_index] + 1e-12)
        )
        top5_rank = 1 + int(
            np.sum(
                top5_similarity_sum
                > top5_similarity_sum[official_index] + 1e-12
            )
        )
        primary_top3_indices = [
            candidate_index[pair["candidate_id"]]
            for pair in pairs
            if pair["primary_rank"] <= 3
        ]
        margin_top3 = float(
            top5_similarity_sum[official_index]
            - max(top5_similarity_sum[index] for index in primary_top3_indices)
        )
        component_rank = min(
            positive_pair["profile_rank"], positive_pair["dense_rank"]
        )
        corroborated = (
            component_rank <= 10
            and best_similarity[official_index] >= 0.10
            and (best_rank <= 25 or top5_rank <= 25)
            and margin_top3 >= -0.02
        ) or (
            margin_top3 >= 0.05
            and best_similarity[official_index] >= 0.10
            and component_rank <= 50
        )
        strong = positive_pair["profile_rank"] <= 3 or corroborated
        weak_signal = (
            component_rank <= 10
            or margin_top3 > 0
            or (
                best_similarity[official_index] >= 0.10
                and (best_rank <= 25 or top5_rank <= 25)
            )
            or (best_similarity[official_index] >= 0.15 and best_rank <= 50)
            or (best_similarity[official_index] >= 0.10 and top5_rank <= 50)
        )
        evidence_bucket = "strong" if strong else "weak" if weak_signal else "not"

        profile_top10 = positive_pair["profile_rank"] <= 10
        dense_top10 = positive_pair["dense_rank"] <= 10
        if profile_top10 and dense_top10:
            component_state = "both_top10"
        elif profile_top10:
            component_state = "profile_top10_only"
        elif dense_top10:
            component_state = "dense_top10_only"
        else:
            component_state = "neither_top10"
        component_counts[component_state] += 1

        development_order = np.lexsort(
            (query_name_array, -development_similarity[target_index])
        )[:10]
        for k in top_k_overlap:
            if any(grades[index, official_index] > 0 for index in development_order[:k]):
                top_k_overlap[k] += 1
        nearest_index = int(development_order[0])
        nearest_positive_labels = [
            candidate_ids[index]
            for index in np.where(grades[nearest_index] > 0)[0]
        ]

        discovery_order = np.lexsort(
            (discovery_name_array, -discovery_similarity[target_index])
        )[:3]
        candidate_groups = {
            "official_positive": official_positive,
            "primary_top1": min(pairs, key=lambda row: row["primary_rank"])[
                "candidate_id"
            ],
            "profile_top1": min(pairs, key=lambda row: row["profile_rank"])[
                "candidate_id"
            ],
            "dense_top1": min(pairs, key=lambda row: row["dense_rank"])[
                "candidate_id"
            ],
        }
        group_values: dict[str, dict[str, Any]] = {}
        for group_name, candidate_id in candidate_groups.items():
            responses = [
                discovery[discovery_ids[index]]["responses"][candidate_id]
                for index in discovery_order
            ]
            statuses = [response_status(response) for response in responses]
            attempts = sum(status != "exact_refusal" for status in statuses)
            word_counts = [
                len(WORD_RE.findall(response))
                for response, status in zip(responses, statuses, strict=True)
                if status != "exact_refusal"
            ]
            values = {
                "candidate_id": candidate_id,
                "attempt_count": attempts,
                "exact_refusal_count": 3 - attempts,
                "insufficiency_count": sum(
                    status != "non_refusal_attempt" for status in statuses
                ),
                "median_attempt_word_count": median(word_counts)
                if word_counts
                else 0,
                "global_attempt_share": profiles[candidate_id]["coverage"][
                    "attempt_share"
                ],
                "local_attempt_lift": attempts / 3
                - profiles[candidate_id]["coverage"]["attempt_share"],
                "statuses": statuses,
            }
            group_values[group_name] = values
            group_attempts[group_name].append(values)

        case_rows.append(
            {
                "query_id": query_id,
                "query_text": development.queries[query_id],
                "official_positive": official_positive,
                "official_grade": development.qrels[query_id][official_positive],
                "primary_rank": positive_pair["primary_rank"],
                "profile_rank": positive_pair["profile_rank"],
                "dense_rank": positive_pair["dense_rank"],
                "component_state": component_state,
                "evidence_bucket": evidence_bucket,
                "lexical_evidence": {
                    "best_attempted_similarity": float(
                        best_similarity[official_index]
                    ),
                    "best_attempted_rank": best_rank,
                    "top5_similarity_sum_rank": top5_rank,
                    "margin_vs_primary_top3": margin_top3,
                    "component_rank": component_rank,
                    "corroborated": bool(corroborated),
                },
                "nearest_dev_query_id": query_ids[nearest_index],
                "nearest_dev_query_text": development.queries[query_ids[nearest_index]],
                "nearest_dev_similarity": float(
                    development_similarity[target_index, nearest_index]
                ),
                "nearest_dev_positive_labels": nearest_positive_labels,
                "nearest_dev_has_official_positive": official_positive
                in nearest_positive_labels,
                "nearest_discovery_query_ids": [
                    discovery_ids[index] for index in discovery_order
                ],
                "nearest_discovery_query_texts": [
                    discovery[discovery_ids[index]]["query"]
                    for index in discovery_order
                ],
                "nearest_discovery_similarities": [
                    float(discovery_similarity[target_index, index])
                    for index in discovery_order
                ],
                "response_evidence": group_values,
            }
        )

    sole_positive_ids = [
        query_ids[index]
        for index in range(len(query_ids))
        if int(np.sum(grades[index] > 0)) == 1
    ]
    other_sole_positive_ids = [
        query_id for query_id in sole_positive_ids if query_id not in set(failures)
    ]

    group_summary: dict[str, dict[str, Any]] = {}
    for group_name, rows in group_attempts.items():
        group_summary[group_name] = {
            "response_cells": len(rows) * 3,
            "non_exact_refusal_attempts": sum(
                row["attempt_count"] for row in rows
            ),
            "queries_with_any_attempt": sum(
                row["attempt_count"] > 0 for row in rows
            ),
            "queries_with_zero_attempts": sum(
                row["attempt_count"] == 0 for row in rows
            ),
            "median_local_attempt_lift": float(
                np.median([row["local_attempt_lift"] for row in rows])
            ),
        }

    evidence_bucket_counts = {
        bucket: sum(row["evidence_bucket"] == bucket for row in case_rows)
        for bucket in ("strong", "weak", "not")
    }
    if evidence_bucket_counts != {"strong": 11, "weak": 34, "not": 30}:
        raise ValueError(
            "blinded evidence-screen counts changed: "
            f"{evidence_bucket_counts!r}"
        )

    rank_bands = (
        ("11-20", 11, 20),
        ("21-50", 21, 50),
        ("51-100", 51, 100),
        ("101-150", 101, 150),
    )
    evidence_by_primary_rank = []
    for label, lower, upper in rank_bands:
        band_rows = [
            row for row in case_rows if lower <= row["primary_rank"] <= upper
        ]
        evidence_by_primary_rank.append(
            {
                "primary_rank_band": label,
                "query_count": len(band_rows),
                **{
                    bucket: sum(row["evidence_bucket"] == bucket for row in band_rows)
                    for bucket in ("strong", "weak", "not")
                },
            }
        )

    lexical_global_nearest = lexical_similarities.max(axis=1)
    lexical_screen = {
        "method": {
            "vectorizer": "scikit-learn TfidfVectorizer",
            "stop_words": "english",
            "ngram_range": [1, 2],
            "sublinear_tf": True,
            "fit_population": "7,759 discovery query texts plus 75 target texts",
            "similarity": "cosine via linear kernel",
            "attempt_predicate": "response.strip().lower() != 'i don\'t know'",
        },
        "evidence_bucket_counts": evidence_bucket_counts,
        "evidence_by_primary_rank": evidence_by_primary_rank,
        "global_nearest_similarity_threshold_counts": {
            str(threshold): int(np.sum(lexical_global_nearest >= threshold))
            for threshold in (0.10, 0.15, 0.20, 0.30)
        },
        "official_best_attempted_similarity_below_0_10": sum(
            row["lexical_evidence"]["best_attempted_similarity"] < 0.10
            for row in case_rows
        ),
        "official_best_attempt_rank_top25": sum(
            row["lexical_evidence"]["best_attempted_rank"] <= 25
            for row in case_rows
        ),
        "official_top5_sum_rank": {
            str(k): sum(
                row["lexical_evidence"]["top5_similarity_sum_rank"] <= k
                for row in case_rows
            )
            for k in (1, 10, 25)
        },
        "official_beats_all_primary_top3": sum(
            row["lexical_evidence"]["margin_vs_primary_top3"] > 0
            for row in case_rows
        ),
        "official_beats_primary_top3_by_0_05": sum(
            row["lexical_evidence"]["margin_vs_primary_top3"] >= 0.05
            for row in case_rows
        ),
    }

    summary = {
        "artifact_type": "trec_2026_sparse_failure_analogue_diagnostic",
        "version": 1,
        "status": "diagnostic_only_post_qrel",
        "population": {
            "development_queries": len(query_ids),
            "candidate_count": len(candidate_ids),
            "qrel_pairs": int(grades.size),
            "sole_positive_queries": len(sole_positive_ids),
            "promoted_zero_ndcg_queries": sum(
                explorer_queries[query_id]["primary_ndcg_at_10"] == 0
                for query_id in query_ids
            ),
            "sole_positive_zero_ndcg_failures": len(failures),
            "discovery_queries": len(discovery_ids),
        },
        "component_states": component_counts,
        "blinded_lexical_evidence_screen": lexical_screen,
        "mpnet_leave_one_out": {
            "nearest_label_agreement": {
                "failure_75": sum(
                    nearest_agreement_by_query[query_id] for query_id in failures
                ),
                "other_sole_positive_105": sum(
                    nearest_agreement_by_query[query_id]
                    for query_id in other_sole_positive_ids
                ),
                "all_sole_positive_180": sum(
                    nearest_agreement_by_query[query_id]
                    for query_id in sole_positive_ids
                ),
            },
            "official_positive_seen_in_any_top_k_neighbor_labels": {
                str(k): count for k, count in top_k_overlap.items()
            },
            "official_positive_dense_rank_recall": {
                str(k): sum(row["dense_rank"] <= k for row in case_rows)
                for k in (1, 5, 10, 20, 50, 100, 150)
            },
            "failure_nearest_similarity_quantiles": {
                label: float(value)
                for label, value in zip(
                    ("min", "q1", "median", "q3", "max"),
                    np.quantile(
                        [nearest_similarity_by_query[query_id] for query_id in failures],
                        (0, 0.25, 0.5, 0.75, 1),
                    ),
                    strict=True,
                )
            },
        },
        "discovery_response_evidence_top3_mpnet": group_summary,
        "cross_group_response_checks": {
            "official_zero_while_any_system_top1_attempts": sum(
                row["response_evidence"]["official_positive"]["attempt_count"] == 0
                and any(
                    row["response_evidence"][group]["attempt_count"] > 0
                    for group in ("primary_top1", "profile_top1", "dense_top1")
                )
                for row in case_rows
            ),
            "official_fewer_attempts_than_profile_top1": sum(
                row["response_evidence"]["official_positive"]["attempt_count"]
                < row["response_evidence"]["profile_top1"]["attempt_count"]
                for row in case_rows
            ),
            "official_more_attempts_than_profile_top1": sum(
                row["response_evidence"]["official_positive"]["attempt_count"]
                > row["response_evidence"]["profile_top1"]["attempt_count"]
                for row in case_rows
            ),
        },
        "encoder_metadata": encoder_metadata[spec.key],
        "source_sha256": {
            "queries": sha256(args.queries),
            "qrels": sha256(args.qrels),
            "discovery": sha256(args.discovery),
            "explorer": sha256(args.explorer),
            "profiles": sha256(args.profiles),
        },
        "definitions": {
            "official_positive": (
                "The sole candidate with official qrel grade greater than zero."
            ),
            "nearest_label_agreement": (
                "The target official positive is also positive for the closest other "
                "development query."
            ),
            "attempt": (
                "A historical discovery response that is not exactly 'I don't know' "
                "or 'I do not know'; this is not correctness evidence."
            ),
            "component_state": (
                "Whether the official positive is top ten in the frozen profile and/or "
                "dense component ranking before 1:1 reciprocal-rank fusion."
            ),
            "evidence_bucket": (
                "A deterministic upper-bound screen over component ranks and lexical "
                "attempt evidence. Strong is not a correctness judgment."
            ),
        },
        "limitations": [
            "Actual responses to the development target queries are unavailable.",
            "Qrel assessor, rubric, reference answers, and 2026 judging pipeline are undocumented.",
            "Candidate identifiers are not canonical topic labels.",
            "All development-label analyses are post-hoc and not unbiased test-time estimates.",
            "Historical non-refusal attempts can be irrelevant or factually wrong.",
        ],
        "cases": case_rows,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )

    args.output_tsv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tsv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "query_id",
            "query_text",
            "official_positive",
            "official_grade",
            "primary_rank",
            "profile_rank",
            "dense_rank",
            "component_state",
            "evidence_bucket",
            "best_attempted_similarity",
            "best_attempted_rank",
            "top5_similarity_sum_rank",
            "margin_vs_primary_top3",
            "nearest_dev_query_id",
            "nearest_dev_query_text",
            "nearest_dev_similarity",
            "nearest_dev_positive_labels",
            "nearest_dev_has_official_positive",
            "nearest_discovery_query_text",
            "nearest_discovery_similarity",
            "official_top3_attempts",
            "profile_top1_candidate",
            "profile_top1_top3_attempts",
        ]
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for row in case_rows:
            writer.writerow(
                {
                    "query_id": row["query_id"],
                    "query_text": row["query_text"],
                    "official_positive": row["official_positive"],
                    "official_grade": row["official_grade"],
                    "primary_rank": row["primary_rank"],
                    "profile_rank": row["profile_rank"],
                    "dense_rank": row["dense_rank"],
                    "component_state": row["component_state"],
                    "evidence_bucket": row["evidence_bucket"],
                    "best_attempted_similarity": (
                        f"{row['lexical_evidence']['best_attempted_similarity']:.9f}"
                    ),
                    "best_attempted_rank": row["lexical_evidence"][
                        "best_attempted_rank"
                    ],
                    "top5_similarity_sum_rank": row["lexical_evidence"][
                        "top5_similarity_sum_rank"
                    ],
                    "margin_vs_primary_top3": (
                        f"{row['lexical_evidence']['margin_vs_primary_top3']:.9f}"
                    ),
                    "nearest_dev_query_id": row["nearest_dev_query_id"],
                    "nearest_dev_query_text": row["nearest_dev_query_text"],
                    "nearest_dev_similarity": f"{row['nearest_dev_similarity']:.9f}",
                    "nearest_dev_positive_labels": json.dumps(
                        row["nearest_dev_positive_labels"], separators=(",", ":")
                    ),
                    "nearest_dev_has_official_positive": row[
                        "nearest_dev_has_official_positive"
                    ],
                    "nearest_discovery_query_text": row[
                        "nearest_discovery_query_texts"
                    ][0],
                    "nearest_discovery_similarity": f"{row['nearest_discovery_similarities'][0]:.9f}",
                    "official_top3_attempts": row["response_evidence"][
                        "official_positive"
                    ]["attempt_count"],
                    "profile_top1_candidate": row["response_evidence"][
                        "profile_top1"
                    ]["candidate_id"],
                    "profile_top1_top3_attempts": row["response_evidence"][
                        "profile_top1"
                    ]["attempt_count"],
                }
            )

    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_tsv": str(args.output_tsv),
                "failure_count": len(case_rows),
                "nearest_label_agreement": summary["mpnet_leave_one_out"][
                    "nearest_label_agreement"
                ],
                "component_states": component_counts,
                "evidence_bucket_counts": evidence_bucket_counts,
                "response_evidence": group_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
