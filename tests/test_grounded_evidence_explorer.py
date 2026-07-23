from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from trec_mlm.grounded_evidence_explorer import (
    EvidenceBundleError,
    build_evidence_bundle,
    classify_behavior,
    derive_cohorts,
    derive_mechanism,
    derive_retrieval_outcome,
    has_context_risk,
)


QUERY_IDS = ("q1", "q2")
CANDIDATE_IDS = tuple(f"c{index}" for index in range(1, 23))
SYSTEMS = (
    "dense_reference",
    "profile_only",
    "dense_profile_equal",
    "raw_hierarchical",
    "raw_hierarchical_dense_equal",
)


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (
            {
                "dense_reference": 0,
                "profile_only": 0,
                "dense_profile_equal": 0,
                "raw_hierarchical": 0,
                "raw_hierarchical_dense_equal": 0,
            },
            ["zero fused nDCG@10", "persistent all-system zero"],
        ),
        (
            {
                "dense_reference": 0.2,
                "profile_only": 0,
                "dense_profile_equal": 0.3,
                "raw_hierarchical": 0.4,
                "raw_hierarchical_dense_equal": 0,
            },
            [
                "zero fused nDCG@10",
                "raw-success/fusion-suppressed",
                "new zero regression",
            ],
        ),
        (
            {
                "dense_reference": 0,
                "profile_only": 0.1,
                "dense_profile_equal": 0,
                "raw_hierarchical": 0.1,
                "raw_hierarchical_dense_equal": 0.2,
            },
            ["rescued prior zero"],
        ),
    ],
)
def test_derive_cohorts(metrics: dict[str, float], expected: list[str]) -> None:
    assert derive_cohorts(metrics) == expected


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        (
            {"raw_hierarchical_dense_equal": {"p": 1}},
            "top-1 success",
        ),
        (
            {"raw_hierarchical_dense_equal": {"p": 7}},
            "top-10 but not top-1",
        ),
        (
            {"raw_hierarchical_dense_equal": {"p": 11}},
            "no relevant candidate in top 10",
        ),
    ],
)
def test_derive_retrieval_outcome(
    positions: dict[str, dict[str, int]], expected: str
) -> None:
    assert derive_retrieval_outcome({"p"}, positions) == expected


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        (
            {
                "raw_hierarchical": {"p": 1},
                "raw_hierarchical_dense_equal": {"p": 1},
                "best_record_cosine": {"p": 1},
            },
            "relevant candidate ranked first",
        ),
        (
            {
                "raw_hierarchical": {"p": 4},
                "raw_hierarchical_dense_equal": {"p": 20},
                "best_record_cosine": {"p": 3},
            },
            "fusion suppression",
        ),
        (
            {
                "raw_hierarchical": {"p": 14},
                "raw_hierarchical_dense_equal": {"p": 20},
                "best_record_cosine": {"p": 3},
            },
            "aggregation suppression",
        ),
        (
            {
                "raw_hierarchical": {"p": 14},
                "raw_hierarchical_dense_equal": {"p": 20},
                "best_record_cosine": {"p": 13},
            },
            "evidence/routing gap",
        ),
        (
            {
                "raw_hierarchical": {"p": 14},
                "raw_hierarchical_dense_equal": {"p": 7},
                "best_record_cosine": {"p": 13},
            },
            "top-10 retrieval, top-1 miss",
        ),
    ],
)
def test_derive_mechanism(
    positions: dict[str, dict[str, int]], expected: str
) -> None:
    assert derive_mechanism({"p"}, positions) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What happened next?", True),
        ("Which law did they invoke?", True),
        ("When was USB Type-C Specification 1.0 finalized?", False),
    ],
)
def test_context_risk_is_only_a_surface_flag(text: str, expected: bool) -> None:
    assert has_context_risk(text) is expected


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_e5_fixture(
    tmp_path: Path, mutation: str | None = None
) -> dict[str, Path]:
    refinement_dir = tmp_path / "refinement"
    construction_dir = tmp_path / "construction"
    refinement_rankings = refinement_dir / "rankings"
    construction_rankings = construction_dir / "rankings"
    refinement_rankings.mkdir(parents=True)
    construction_rankings.mkdir(parents=True)

    _write_json(
        refinement_dir / "inputs.json",
        {"query_ids": list(QUERY_IDS), "candidate_ids": list(CANDIDATE_IDS)},
    )
    queries_path = tmp_path / "queries.tsv"
    query_rows = ["q1\tWhat is the answer?", "q2\tWhich law did they invoke?"]
    if mutation == "query_inventory":
        query_rows[1] = "other\tWhich law did they invoke?"
    queries_path.write_text("\n".join(query_rows) + "\n", encoding="utf-8")

    rankings = {
        "dense_reference": {
            "q1": ["c1", "c3", "c2", *CANDIDATE_IDS[3:]],
            "q2": list(CANDIDATE_IDS),
        },
        "profile_only": {
            "q1": ["c1", "c2", "c3", *CANDIDATE_IDS[3:]],
            "q2": list(CANDIDATE_IDS),
        },
        "dense_profile_equal": {
            "q1": ["c1", "c2", "c3", *CANDIDATE_IDS[3:]],
            "q2": list(CANDIDATE_IDS),
        },
        "raw_hierarchical": {
            "q1": ["c2", "c1", "c3", *CANDIDATE_IDS[3:]],
            "q2": list(CANDIDATE_IDS),
        },
        "raw_hierarchical_dense_equal": {
            "q1": ["c1", "c2", "c3", *CANDIDATE_IDS[3:]],
            "q2": list(CANDIDATE_IDS),
        },
    }
    if mutation == "candidate_inventory":
        rankings["profile_only"]["q1"] = ["c1", "c2", *CANDIDATE_IDS[3:-1], "other"]
    if mutation == "ranking_duplicate":
        rankings["raw_hierarchical"]["q1"] = ["c1", "c1", *CANDIDATE_IDS[2:]]
    for system, payload in rankings.items():
        root = refinement_rankings if system.startswith("raw_") or system == "dense_reference" else construction_rankings
        _write_json(root / f"{system}.json", payload)

    selected_ids: list[list[list[str]]] = []
    records: list[dict[str, str]] = []
    similarities = np.zeros((2, len(CANDIDATE_IDS), 10), dtype="<f8")
    for query_index, query_id in enumerate(QUERY_IDS):
        selected_for_query: list[list[str]] = []
        for candidate_index, candidate_id in enumerate(CANDIDATE_IDS):
            record_ids = [f"{query_id}-{candidate_id}-r{rank}" for rank in range(1, 11)]
            if query_id == "q1" and candidate_id == "c1":
                record_ids[:3] = ["r1", "r2", "r3"]
            selected_for_query.append(record_ids)
            for rank, record_id in enumerate(record_ids, start=1):
                response = f"Answer {rank} for {candidate_id}."
                if record_id == "r2":
                    response = "I don't know"
                elif record_id == "r3":
                    response = "   "
                owner = candidate_id
                if mutation == "record_owner" and record_id == "r1":
                    owner = "c2"
                if mutation == "record_owner_outside_evidence" and record_id == "q1-c11-r1":
                    owner = "c12"
                records.append(
                    {
                        "record_id": record_id,
                        "candidate_id": owner,
                        "source_query_id": f"source-{query_id}-{rank}",
                        "question": f"Historical question {rank}",
                        "response": response,
                    }
                )
            start = 0.91 if (query_id, candidate_id) == ("q1", "c1") else 0.8 - candidate_index / 100
            similarities[query_index, candidate_index] = np.asarray(
                [start - rank * 0.01 for rank in range(10)], dtype="<f8"
            )
        selected_ids.append(selected_for_query)
    similarities[0, 0] = np.asarray(
        [0.91, 0.83, 0.72, 0.60, 0.55, 0.50, 0.45, 0.40, 0.35, 0.34],
        dtype="<f8",
    )
    if mutation == "similarity_nan":
        similarities[0, 0, 1] = np.nan
    if mutation == "similarity_order":
        similarities[0, 0, 1] = 0.92
    if mutation == "array_shape":
        similarities = similarities[:, :, :9]
    _write_json(refinement_dir / "selected_record_ids.json", selected_ids)
    np.save(refinement_dir / "top_record_similarities.npy", similarities)
    _write_json(construction_dir / "records.json", records)

    qrels_path = tmp_path / "qrels.txt"
    qrel_rows = [
        f"{query_id} 0 {candidate_id} {1 if query_id == 'q1' and candidate_id in {'c3', 'c22'} else 0}"
        for query_id in QUERY_IDS
        for candidate_id in CANDIDATE_IDS
    ]
    if mutation == "qrel_pair_missing":
        qrel_rows.pop()
    qrels_path.write_text("\n".join(qrel_rows) + "\n", encoding="utf-8")

    evaluation_path = tmp_path / "evaluation.json"
    _write_json(
        evaluation_path,
        {
            "per_query_rows": [
                {
                    "query_id": query_id,
                    "system": system,
                    "ndcg_at_10": 0.0,
                    "reciprocal_rank": 0.0,
                }
                for query_id in QUERY_IDS
                for system in SYSTEMS
            ]
        },
    )
    return {
        "refinement_dir": refinement_dir,
        "construction_dir": construction_dir,
        "evaluation_path": evaluation_path,
        "queries_path": queries_path,
        "qrels_path": qrels_path,
    }


def test_build_bundle_uses_first_three_persisted_records(tmp_path: Path) -> None:
    paths = write_e5_fixture(tmp_path)

    bundle = build_evidence_bundle(**paths)

    candidate = bundle["queries"]["q1"]["candidates"]["c1"]
    assert [item["record_id"] for item in candidate["evidence"]] == ["r1", "r2", "r3"]
    assert [item["similarity"] for item in candidate["evidence"]] == [0.91, 0.83, 0.72]
    assert [item["behavior"] for item in candidate["evidence"]] == [
        "substantive response",
        "exact refusal",
        "empty",
    ]
    assert candidate["best_record_cosine"] == 0.91
    assert candidate["mean_top_10_cosine"] == pytest.approx(0.565)


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("A supported historical answer.", "substantive response"),
        ("I don't know", "exact refusal"),
        ("I do not know.", "exact refusal"),
        ("   ", "empty"),
    ],
)
def test_classify_behavior(response: str, expected: str) -> None:
    assert classify_behavior(response) == expected


def test_bundle_has_complete_component_ranks_and_needed_evidence(tmp_path: Path) -> None:
    paths = write_e5_fixture(tmp_path)

    bundle = build_evidence_bundle(**paths)

    query = bundle["queries"]["q1"]
    assert len(CANDIDATE_IDS) > 20
    assert query["display_candidate_ids"] == [*CANDIDATE_IDS[:20], "c22"]
    assert query["evidence_candidate_ids"] == [*CANDIDATE_IDS[:10], "c22"]
    assert query["candidates"]["c22"]["is_positive"] is True
    assert query["candidates"]["c22"]["ranks"]["raw_hierarchical_dense_equal"] == 22
    late_positive_evidence = query["candidates"]["c22"]["evidence"]
    assert [record["record_id"] for record in late_positive_evidence] == [
        "q1-c22-r1",
        "q1-c22-r2",
        "q1-c22-r3",
    ]
    assert len(late_positive_evidence) == 3
    assert all(
        set(record) == {
            "record_id",
            "source_query_id",
            "question",
            "response",
            "similarity",
            "behavior",
        }
        for record in late_positive_evidence
    )
    assert query["candidates"]["c2"]["ranks"] == {
        "dense_reference": 3,
        "profile_only": 2,
        "dense_profile_equal": 2,
        "raw_hierarchical": 1,
        "raw_hierarchical_dense_equal": 2,
        "best_record_cosine": 2,
    }
    assert query["candidates"]["c11"]["evidence"] == []


def test_bundle_adds_rank_one_retrieval_outcome(tmp_path: Path) -> None:
    paths = write_e5_fixture(tmp_path)

    bundle = build_evidence_bundle(**paths)

    query = bundle["queries"]["q1"]
    assert query["retrieval_outcome"] == "top-10 but not top-1"
    assert query["mechanism"] == "top-10 retrieval, top-1 miss"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("query_inventory", "query inventory"),
        ("candidate_inventory", "candidate inventory"),
        ("ranking_duplicate", "candidate inventory is incomplete"),
        ("record_owner", "record membership differs"),
        ("record_owner_outside_evidence", "record membership differs"),
        ("similarity_nan", "finite and non-increasing"),
        ("similarity_order", "finite and non-increasing"),
        ("array_shape", "shape"),
        ("qrel_pair_missing", "qrel inventory"),
    ],
)
def test_bundle_fails_closed(tmp_path: Path, mutation: str, message: str) -> None:
    paths = write_e5_fixture(tmp_path, mutation=mutation)

    with pytest.raises(EvidenceBundleError, match=message):
        build_evidence_bundle(**paths)
