from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from trec_mlm.profile_retrieval import (
    CandidateProfile,
    ProfileRecord,
    aggregate_cluster_labels,
    canonical_profile_record_id,
    canonical_kmeans,
    filter_and_deduplicate_records,
    reciprocal_rank_fusion,
    score_profiles,
    validate_clustering_runtime_identity,
)
from trec_mlm.response_retrieval import DiscoveryRecord


def _record(query_id: str, candidate: str, question: str | None = None) -> ProfileRecord:
    return ProfileRecord(
        record_id=f"{query_id}:record",
        source_query_id=query_id,
        candidate_id=candidate,
        question=question or f"Question {query_id}?",
        response=f"Answer {query_id}.",
    )


def _profile(candidate: str, centroids: list[list[float]]) -> CandidateProfile:
    records = tuple(_record(f"q{i}", candidate) for i in range(len(centroids)))
    return canonical_kmeans(
        candidate,
        records,
        np.asarray(centroids, dtype=np.float32),
        requested_clusters=len(centroids),
        enforce_thread_contract=False,
    )


def test_clustering_runtime_identity_is_exact() -> None:
    validate_clustering_runtime_identity(
        python_version="3.13.9",
        numpy_version="2.5.1",
        scipy_version="1.18.0",
        sklearn_version="1.9.0",
    )
    with pytest.raises(RuntimeError, match="clustering runtime"):
        validate_clustering_runtime_identity(
            python_version="3.12.13",
            numpy_version="2.5.1",
            scipy_version="1.18.0",
            sklearn_version="1.9.0",
        )


def test_basic_filter_and_raw_exact_dedup_are_candidate_local() -> None:
    records = (
        DiscoveryRecord("q1", " Question ", "A", " Answer "),
        DiscoveryRecord("q2", " Question ", "A", " Answer "),
        DiscoveryRecord("q3", "Question", "A", "Answer"),
        DiscoveryRecord("q4", "Question", "A", "I don't know"),
        DiscoveryRecord("q1", " Question ", "B", " Answer "),
    )

    result = filter_and_deduplicate_records(records, ("A", "B"))

    assert [record.source_query_id for record in result.records_by_candidate["A"]] == [
        "q1",
        "q3",
    ]
    assert result.audit_by_candidate["A"].removed_exact_idk == 1
    assert result.audit_by_candidate["A"].duplicates == 1
    assert len(result.records_by_candidate["B"]) == 1
    assert result.records_by_candidate["A"][0].question == " Question "


def test_record_ids_are_full_candidate_aware_canonical_digests() -> None:
    records = (
        DiscoveryRecord("q1", "Same question", "A", "Same answer"),
        DiscoveryRecord("q1", "Same question", "B", "Same answer"),
    )

    result = filter_and_deduplicate_records(records, ("A", "B"))
    left = result.records_by_candidate["A"][0]
    right = result.records_by_candidate["B"][0]

    assert left.record_id == canonical_profile_record_id(
        "A", "q1", "Same question", "Same answer"
    )
    assert right.record_id == canonical_profile_record_id(
        "B", "q1", "Same question", "Same answer"
    )
    assert left.record_id != right.record_id
    assert left.record_id.startswith("profile-record-v2:")
    assert len(left.record_id.removeprefix("profile-record-v2:")) == 64


def test_identical_embeddings_collapse_and_centroids_are_canonical() -> None:
    records = tuple(_record(f"q{i}", "A") for i in range(3))
    embeddings = np.asarray([[1.0, 0.0]] * 3, dtype=np.float32)

    profile = canonical_kmeans(
        "A", records, embeddings, requested_clusters=3, enforce_thread_contract=False
    )

    assert profile.requested_clusters == 3
    assert len(profile.clusters) == 1
    assert profile.clusters[0].cluster_id == 0
    assert profile.clusters[0].member_record_ids == tuple(
        record.record_id for record in records
    )
    np.testing.assert_array_equal(profile.centroids, [[1.0, 0.0]])
    assert profile.clusters[0].dispersion == pytest.approx(0.0)
    assert profile.clusters[0].medoid_record_id == records[0].record_id


def test_clusters_use_stable_member_ids_and_representative_questions() -> None:
    records = (
        _record("z", "A", "same"),
        _record("a", "A", "same"),
        _record("m", "A", "different"),
    )
    embeddings = np.asarray([[1, 0], [1, 0], [0, 1]], dtype=np.float32)

    profile = canonical_kmeans(
        "A", records, embeddings, requested_clusters=2, enforce_thread_contract=False
    )

    assert [cluster.cluster_id for cluster in profile.clusters] == [0, 1]
    assert profile.clusters[0].member_record_ids[0] == "a:record"
    assert profile.clusters[0].representative_questions == ("same",)


def test_profile_scoring_uses_max_cosine_and_deterministic_ties() -> None:
    profiles = {
        "A": _profile("A", [[1.0, 0.0], [0.0, 1.0]]),
        "B": _profile("B", [[1.0, 0.0]]),
    }

    result = score_profiles(
        query_ids=("q",),
        candidate_ids=("A", "B"),
        query_embeddings=np.asarray([[0.0, 1.0]], dtype=np.float32),
        profiles=profiles,
    )

    np.testing.assert_allclose(result.scores, [[1.0, 0.0]])
    assert result.winning_centroid_ids.tolist() == [[1, 0]]
    assert result.rankings["q"] == ["A", "B"]


def test_unavailable_profile_has_tail_sentinel_and_zero_fusion_term() -> None:
    scored = score_profiles(
        query_ids=("q",),
        candidate_ids=("A", "B"),
        query_embeddings=np.asarray([[1.0, 0.0]], np.float32),
        profiles={"A": _profile("A", [[1.0, 0.0]]), "B": None},
    )
    assert scored.rankings["q"] == ["A", "B"]
    assert scored.scores[0, 1] == -2.0

    fused = reciprocal_rank_fusion(
        query_ids=("q",),
        candidate_ids=("A", "B"),
        rankings_by_signal={
            "dense": {"q": ["B", "A"]},
            "profile": scored.rankings,
        },
        weights={"dense": 1.0, "profile": 1.0},
        unavailable_profile_ids=frozenset({"B"}),
        k=60,
    )

    assert fused.scores[0, 1] == pytest.approx(1 / 61)
    assert fused.scores[0, 0] == pytest.approx(1 / 62 + 1 / 61)
    assert fused.rankings["q"] == ["A", "B"]


@pytest.mark.parametrize(
    ("scores", "primary", "secondary", "status"),
    [
        ([0.70, 0.67, 0.10], "A", None, "primary"),
        ([0.70, 0.69, 0.10], "A", "B", "primary_secondary"),
        ([0.59, 0.20, 0.10], None, None, "mixed_or_unclear"),
    ],
)
def test_cluster_label_thresholds(
    scores: list[float],
    primary: str | None,
    secondary: str | None,
    status: str,
) -> None:
    annotation = aggregate_cluster_labels(
        member_question_indices=(0,),
        question_label_scores=np.asarray([scores], dtype=np.float32),
        label_ids=("A", "B", "C"),
    )
    assert annotation.primary_label == primary
    assert annotation.secondary_label == secondary
    assert annotation.status == status
    assert annotation.affects_ranking is False


def test_annotations_have_no_path_into_profile_scoring() -> None:
    profile = _profile("A", [[1.0, 0.0]])
    mutated = replace(
        profile,
        clusters=(replace(profile.clusters[0], annotation={"malicious": 1}),),
    )
    kwargs = dict(
        query_ids=("q",),
        candidate_ids=("A",),
        query_embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
    )
    first = score_profiles(profiles={"A": profile}, **kwargs)
    second = score_profiles(profiles={"A": mutated}, **kwargs)
    np.testing.assert_array_equal(first.scores, second.scores)
    assert first.rankings == second.rankings
