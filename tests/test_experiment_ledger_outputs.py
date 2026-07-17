from __future__ import annotations

import csv
from io import StringIO
import json
from pathlib import Path

import pytest

import trec_mlm.experiment_ledger as module
from trec_mlm.experiment_ledger import (
    LedgerBuild,
    PerQueryRecord,
    SliceRecord,
    SystemRecord,
    assert_outputs_current,
    render_density_slices_tsv,
    render_ledger_json,
    render_ledger_markdown,
    render_per_query_tsv,
    render_systems_tsv,
    render_topic_slices_tsv,
    write_ledger_outputs,
)


@pytest.fixture
def sample_build() -> LedgerBuild:
    def system(
        system_id: str,
        *,
        evaluation_scope: str = "dev_322x150",
        evidence_status: str = "aggregate_only",
        promotion_status: str = "stable_control",
        ndcg_at_10: float | None = 0.4,
        query_count: int | None = 322,
        candidate_count: int | None = 150,
        per_query_available: bool = False,
        artifact_state: str = "tracked_complete",
        blocker: str | None = "Only aggregate evidence is available.",
        caveat_ids: tuple[str, ...] = ("development_only",),
        aggregate_payload_json: str = '{"ndcg_at_10":0.4}',
    ) -> SystemRecord:
        return SystemRecord(
            system_id=system_id,
            source_system_id=f"source_{system_id}",
            experiment_id=f"experiment_{system_id}",
            experiment_family="sample_family",
            evaluation_scope=evaluation_scope,
            evidence_status=evidence_status,
            promotion_status=promotion_status,
            query_count=query_count,
            candidate_count=candidate_count,
            ndcg_at_10=ndcg_at_10,
            mrr=None if ndcg_at_10 is None else ndcg_at_10 + 0.1,
            zero_ndcg_queries=None,
            recomputed_ndcg_at_10=ndcg_at_10 if per_query_available else None,
            recomputed_mrr=(
                None
                if ndcg_at_10 is None or not per_query_available
                else ndcg_at_10 + 0.1
            ),
            recomputed_zero_ndcg_queries=7 if per_query_available else None,
            aggregate_payload_json=aggregate_payload_json,
            per_query_available=per_query_available,
            ranking_coverage=(
                "per_query_metrics_only" if per_query_available else "none"
            ),
            artifact_state=artifact_state,
            source_path=f"reports/{system_id}.json",
            source_sha256=system_id[0] * 64,
            blocker=blocker,
            caveat_ids=caveat_ids,
        )

    systems = (
        system(
            "promoted_system",
            evidence_status="stored_per_query",
            promotion_status="promoted_development",
            ndcg_at_10=0.42,
            per_query_available=True,
            blocker=None,
            caveat_ids=("development_only", "qrel_transferred_training"),
            aggregate_payload_json=('{"ndcg_at_10":0.42,"relevant_in_top_3":"5/6"}'),
        ),
        system(
            "diagnostic_system",
            promotion_status="diagnostic_only",
            ndcg_at_10=0.44,
        ),
        system("stable_system", ndcg_at_10=0.41),
        system(
            "aggregate_only_system",
            evidence_status="aggregate_only",
            promotion_status="valid_negative",
            ndcg_at_10=0.3,
            per_query_available=False,
            artifact_state="local_artifact_missing",
            blocker="Original per-query artifact is absent.",
            caveat_ids=("development_only", "missing_raw_artifacts"),
        ),
        system(
            "restricted_system",
            evaluation_scope="restricted_6",
            evidence_status="restricted_diagnostic",
            promotion_status="restricted_pilot",
            ndcg_at_10=None,
            query_count=6,
            candidate_count=None,
            per_query_available=False,
            artifact_state="tracked_report_snapshot",
            blocker="Only an aggregate restricted snapshot was retained.",
            caveat_ids=("development_only", "restricted_not_comparable"),
            aggregate_payload_json='{"relevant_in_top_3":"5/6"}',
        ),
        system(
            "invalid_system",
            evaluation_scope="invalid_unscored",
            evidence_status="invalid",
            promotion_status="invalid",
            ndcg_at_10=None,
            query_count=None,
            candidate_count=None,
            per_query_available=False,
            artifact_state="tracked_report_snapshot",
            blocker=None,
            caveat_ids=("development_only",),
            aggregate_payload_json='{"result":"invalid"}',
        ),
    )
    per_query = (
        PerQueryRecord(
            system_id="promoted_system",
            source_system_id="source_promoted_system",
            experiment_family="sample_family",
            evaluation_scope="dev_322x150",
            query_id="q2",
            query_text="Question with\ta tab?",
            ndcg_at_10=0.0,
            reciprocal_rank=0.25,
            grade_1_count=1,
            grade_2_count=0,
            positive_count=1,
            relevant_candidates=(("B", 1),),
            density_bucket="at_most_5_positives",
            zero_ndcg=True,
            metric_provenance="stored",
            source_path="reports/per_query.tsv",
            source_sha256="e" * 64,
        ),
        PerQueryRecord(
            system_id="promoted_system",
            source_system_id="source_promoted_system",
            experiment_family="sample_family",
            evaluation_scope="dev_322x150",
            query_id="q1",
            query_text="Question one?",
            ndcg_at_10=0.5,
            reciprocal_rank=1.0,
            grade_1_count=1,
            grade_2_count=1,
            positive_count=2,
            relevant_candidates=(("A", 2), ("B", 1)),
            density_bucket="at_most_5_positives",
            zero_ndcg=False,
            metric_provenance="stored",
            source_path="reports/per_query.tsv",
            source_sha256="e" * 64,
        ),
    )

    def slice_record(
        *, slice_type: str, slice_key: str, overlapping: bool
    ) -> SliceRecord:
        return SliceRecord(
            system_id="promoted_system",
            slice_type=slice_type,
            slice_key=slice_key,
            query_count=2,
            ndcg_at_10=0.25,
            mrr=0.625,
            zero_ndcg_count=1,
            zero_ndcg_rate=0.5,
            grade_1_memberships=2,
            grade_2_memberships=1,
            overlapping=overlapping,
            membership_policy=(
                "qrel_grade_gt_zero" if overlapping else "positive_candidate_count"
            ),
            metric_source_path="reports/per_query.tsv",
            metric_source_sha256="e" * 64,
            query_source_path="data/dev/queries.tsv",
            query_source_sha256="f" * 64,
            qrel_source_path="data/dev/qrels.txt",
            qrel_source_sha256="a" * 64,
        )

    caveats = (
        {
            "caveat_id": "development_only",
            "scope": "program",
            "summary": "All scores are development-only evidence.",
        },
        {
            "caveat_id": "missing_raw_artifacts",
            "scope": "program",
            "summary": "Some original artifacts are missing.",
        },
        {
            "caveat_id": "qrel_transferred_training",
            "scope": "method",
            "summary": "This method transfers qrel evidence.",
        },
        {
            "caveat_id": "restricted_not_comparable",
            "scope": "program",
            "summary": "Restricted pilots are not comparable to full development.",
        },
        {
            "caveat_id": "topic_slices_overlap",
            "scope": "program",
            "summary": "Relevant-candidate slices overlap and are not official topics.",
        },
    )
    return LedgerBuild(
        systems=systems,
        per_query=per_query,
        topic_slices=(
            slice_record(
                slice_type="relevant_candidate_slice",
                slice_key="Candidate B",
                overlapping=True,
            ),
        ),
        density_slices=(
            slice_record(
                slice_type="positive_density_slice",
                slice_key="at_most_5_positives",
                overlapping=False,
            ),
        ),
        sources=(
            {
                "source_id": "queries",
                "path": "data/dev/queries.tsv",
                "sha256": "f" * 64,
            },
            {
                "source_id": "metrics",
                "path": "reports/per_query.tsv",
                "sha256": "e" * 64,
            },
        ),
        caveats=caveats,
        validation={"system_count": len(systems), "query_count": 322},
    )


def _tsv_rows(payload: str) -> list[dict[str, str]]:
    return list(csv.DictReader(StringIO(payload), delimiter="\t"))


def test_renderers_are_deterministic_and_repository_relative(
    sample_build: LedgerBuild,
) -> None:
    first = render_ledger_json(sample_build)
    second = render_ledger_json(sample_build)
    assert first == second
    assert "/home/" not in first
    assert "generated_at" not in first


def test_json_preserves_receipt_payload_and_separates_recomputed_metrics(
    sample_build: LedgerBuild,
) -> None:
    payload = json.loads(render_ledger_json(sample_build))
    promoted = next(
        row for row in payload["systems"] if row["system_id"] == "promoted_system"
    )
    assert promoted["aggregate_payload"]["relevant_in_top_3"] == "5/6"
    assert promoted["receipt_metrics"]["ndcg_at_10"] == 0.42
    assert promoted["recomputed_metrics"]["zero_ndcg_queries"] == 7
    assert promoted["receipt_provenance"]["source_path"].endswith(
        "promoted_system.json"
    )
    assert promoted["recomputed_provenance"] == {
        "derivation": "macro_from_per_query",
        "per_query_metric_provenance": "stored",
        "source_path": "reports/per_query.tsv",
        "source_sha256": "e" * 64,
    }
    assert payload["topic_slices"][0]["aggregation_warning"] == (
        "overlapping_relevant_candidate_slices_are_not_official_topics_and_"
        "means_cannot_be_summed_or_averaged_as_disjoint"
    )
    assert payload["density_slices"][0]["aggregation_warning"] is None


def test_systems_tsv_preserves_payload_and_nullable_counts(
    sample_build: LedgerBuild,
) -> None:
    rows = {
        row["system_id"]: row for row in _tsv_rows(render_systems_tsv(sample_build))
    }
    assert (
        json.loads(rows["promoted_system"]["aggregate_payload_json"])[
            "relevant_in_top_3"
        ]
        == "5/6"
    )
    assert rows["promoted_system"]["recomputed_ndcg_at_10"] == "0.42"
    assert rows["promoted_system"]["recomputed_metric_derivation"] == (
        "macro_from_per_query"
    )
    assert rows["promoted_system"]["per_query_metric_provenance"] == "stored"
    assert rows["promoted_system"]["recomputed_metric_source_path"] == (
        "reports/per_query.tsv"
    )
    assert rows["restricted_system"]["candidate_count"] == ""
    assert rows["invalid_system"]["query_count"] == ""


def test_per_query_tsv_is_sorted_and_uses_compact_relevance_json(
    sample_build: LedgerBuild,
) -> None:
    rows = _tsv_rows(render_per_query_tsv(sample_build))
    assert [row["query_id"] for row in rows] == ["q1", "q2"]
    assert rows[0]["relevant_candidates"] == '[["A",2],["B",1]]'
    assert rows[0]["zero_ndcg"] == "false"


@pytest.mark.parametrize(
    ("renderer", "expected_type"),
    [
        (render_topic_slices_tsv, "relevant_candidate_slice"),
        (render_density_slices_tsv, "positive_density_slice"),
    ],
)
def test_slice_tsv_exposes_metric_query_and_qrel_receipts(
    sample_build: LedgerBuild,
    renderer: object,
    expected_type: str,
) -> None:
    row = _tsv_rows(renderer(sample_build))[0]  # type: ignore[operator]
    assert row["slice_type"] == expected_type
    if expected_type == "relevant_candidate_slice":
        assert row["aggregation_warning"] == (
            "overlapping_relevant_candidate_slices_are_not_official_topics_and_"
            "means_cannot_be_summed_or_averaged_as_disjoint"
        )
    else:
        assert row["aggregation_warning"] == ""
    assert row["metric_source_sha256"] == "e" * 64
    assert row["query_source_sha256"] == "f" * 64
    assert row["qrel_source_sha256"] == "a" * 64


def test_markdown_separates_scopes_and_makes_caveats_visible(
    sample_build: LedgerBuild,
) -> None:
    markdown = render_ledger_markdown(sample_build)
    full_table = markdown.split("## Full development results", 1)[1].split("##", 1)[0]
    assert "promoted_system" in full_table
    assert "diagnostic_system" in full_table
    assert "restricted_system" not in full_table
    assert "invalid_system" not in full_table
    assert "Restricted pilots and invalid/unscored work" in markdown
    assert "diagnostic_only" in markdown
    assert "aggregate_only" in markdown
    assert "Original per-query artifact is absent" in markdown
    assert "development-only" in markdown
    assert "overlap" in markdown
    assert "not official topics" in markdown


def test_markdown_headlines_distinguish_promoted_diagnostic_and_qrel_free_control(
    sample_build: LedgerBuild,
) -> None:
    headline = render_ledger_markdown(sample_build).split(
        "## Full development results", 1
    )[0]
    assert "promoted_system" in headline
    assert "diagnostic_system" in headline
    assert "diagnostic_only" in headline
    assert "stable_system" in headline
    assert "aggregate-only systems: 3" in headline


def test_write_outputs_is_atomic_on_render_failure(
    tmp_path: Path,
    sample_build: LedgerBuild,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    existing = tmp_path / "trec_2026_experiment_ledger.json"
    existing.write_text("old", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "render_per_query_tsv",
        lambda build: (_ for _ in ()).throw(ValueError("boom")),
    )
    with pytest.raises(ValueError, match="boom"):
        write_ledger_outputs(sample_build, tmp_path)
    assert existing.read_text(encoding="utf-8") == "old"
    assert list(tmp_path.iterdir()) == [existing]


def test_write_outputs_rolls_back_if_an_atomic_replace_fails(
    tmp_path: Path,
    sample_build: LedgerBuild,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = write_ledger_outputs(sample_build, tmp_path)
    for path in paths:
        path.write_text(f"old:{path.name}", encoding="utf-8")

    original_replace = Path.replace

    def failing_replace(source: Path, destination: Path) -> Path:
        destination = Path(destination)
        if (
            source.parent.name.startswith(".experiment-ledger-")
            and destination.parent == tmp_path
            and destination.name == "trec_2026_experiment_ledger.md"
        ):
            raise OSError("simulated replace failure")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", failing_replace)
    with pytest.raises(OSError, match="simulated replace failure"):
        write_ledger_outputs(sample_build, tmp_path)

    for path in paths:
        assert path.read_text(encoding="utf-8") == f"old:{path.name}"


def test_write_and_check_outputs_detects_every_missing_or_stale_file(
    tmp_path: Path,
    sample_build: LedgerBuild,
) -> None:
    paths = write_ledger_outputs(sample_build, tmp_path)
    assert [path.name for path in paths] == sorted(path.name for path in paths)
    assert len(paths) == 6
    assert_outputs_current(sample_build, tmp_path)

    paths[0].write_text("stale", encoding="utf-8")
    paths[1].unlink()
    with pytest.raises(ValueError) as error:
        assert_outputs_current(sample_build, tmp_path)
    assert paths[0].name in str(error.value)
    assert paths[1].name in str(error.value)
    assert "stale" in str(error.value)
    assert "missing" in str(error.value)
