"""Shared fixtures for research hub tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from trec_mlm.experiment_ledger import LedgerBuild, build_ledger
from trec_mlm.research_hub.model import HubRecord, HubSources


@pytest.fixture(scope="session")
def ledger_build() -> LedgerBuild:
    return build_ledger(Path("."))


@pytest.fixture
def minimal_sources(tmp_path: Path) -> HubSources:
    evidence_path = tmp_path / "evidence" / "receipt.md"
    evidence_path.parent.mkdir()
    evidence_path.write_text("tracked evidence\n", encoding="utf-8")
    evidence_ref = "evidence/receipt.md"

    def record(kind: str, record_id: str, **metadata: object) -> HubRecord:
        return HubRecord(
            kind=kind,
            record_id=record_id,
            title=f"{record_id} title",
            source_path=f"research/hub/{kind}s/{record_id}.qmd",
            metadata={"title": f"{record_id} title", **metadata},
            body="A concise record body.",
        )

    return HubSources(
        themes=(
            record(
                "theme",
                "candidate-evidence",
                plain_english="Candidate-side semantic evidence for ranking.",
                current_conclusion="Candidate evidence improves development ranking.",
                decision="adopt",
                confidence="moderate",
                confidence_boundary="Development-only evidence.",
                next_test_ids=["test-candidate-evidence"],
                updated_on="2026-07-24",
                evidence_refs=[evidence_ref],
            ),
        ),
        questions=(
            record(
                "question",
                "candidate-evidence",
                primary_theme_id="candidate-evidence",
                theme_ids=["candidate-evidence"],
                plain_english="Does candidate evidence improve ranking?",
                status="open",
                current_answer="Evidence is insufficient.",
                confidence="low",
                confidence_boundary="No held-out evaluation supports this answer.",
                study_ids=["dense-baseline"],
                open_subquestions=["Which signals matter?"],
                next_test_ids=["test-candidate-evidence"],
                updated_on="2026-07-24",
                evidence_refs=[evidence_ref],
            ),
        ),
        studies=(
            record(
                "study",
                "dense-baseline",
                primary_theme_id="candidate-evidence",
                theme_ids=["candidate-evidence"],
                study_type="ranking_experiment",
                question_ids=["candidate-evidence"],
                approach_ids=["dense-retrieval"],
                status="concluded",
                outcome="positive",
                objective="Measure the dense baseline.",
                hypothesis="Dense retrieval is a viable control.",
                rationale="The existing baseline provides a control.",
                rationale_provenance="contemporaneous",
                protocol_summary="Evaluate on development data.",
                conclusion="The control remains useful.",
                decision="retain_control",
                decision_summary="Keep the established baseline.",
                system_ids=["dense_nested"],
                representative_system_ids=["dense_nested"],
                comparison_system_ids=["dense_nested"],
                evidence_refs=[evidence_ref],
                dataset_ids=["development-set"],
                started_on="2026-07-01",
                concluded_on="2026-07-02",
                supersedes=[],
                superseded_by=[],
                limitations="Development-only evidence.",
            ),
        ),
        tests=(
            record(
                "test",
                "test-candidate-evidence",
                primary_theme_id="candidate-evidence",
                theme_ids=["candidate-evidence"],
                stage="idea",
                summary="Test candidate evidence.",
                linked_question_ids=["candidate-evidence"],
                missing_readiness_fields=["decision_rule"],
            ),
        ),
        approaches=(
            record(
                "approach",
                "dense-retrieval",
                summary="Rank candidates using dense query retrieval.",
                question_ids=["candidate-evidence"],
                study_ids=["dense-baseline"],
            ),
        ),
        datasets=(
            record(
                "dataset",
                "development-set",
                plain_english="Development queries and candidate labels.",
                category="provided",
                provenance="TREC development package.",
                method="Provided by the organizers.",
                version="unversioned",
                repository_path=evidence_ref,
                sensitivity="public",
                sharing="repository",
                leakage_constraints="Do not use held-out labels.",
                permitted_use="Development evaluation only.",
                study_ids=["dense-baseline"],
                availability="available",
                reproducibility="reproducible",
            ),
        ),
        background=(
            record(
                "background",
                "trec-overview",
                resource_type="benchmark",
                summary="Background for the benchmark.",
                canonical_source_path=evidence_ref,
                citations=["TREC 2026 overview"],
                tags=["trec"],
                question_ids=["candidate-evidence"],
                reviewed_on="2026-07-24",
            ),
        ),
    )
