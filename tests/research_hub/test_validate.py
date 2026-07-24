from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from trec_mlm.experiment_ledger import LedgerBuild
from trec_mlm.research_hub.model import HubSources
from trec_mlm.research_hub.validate import validate_hub_sources


def test_validation_returns_record_and_linked_system_counts(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    summary = validate_hub_sources(minimal_sources, ledger_build, tmp_path)

    assert summary.record_counts == {
        "question": 1,
        "study": 1,
        "test": 1,
        "approach": 1,
        "dataset": 1,
        "background": 1,
    }
    assert summary.linked_system_count == 1


def test_validation_requires_plain_english_question(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    question = replace(
        minimal_sources.questions[0],
        metadata={
            **minimal_sources.questions[0].metadata,
            "plain_english": "",
        },
    )
    with pytest.raises(ValueError, match="plain_english"):
        validate_hub_sources(
            replace(minimal_sources, questions=(question,)),
            ledger_build,
            tmp_path,
        )


def test_validation_rejects_ready_test_without_decision_rule(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    ready = replace(
        minimal_sources.tests[0],
        metadata={
            **minimal_sources.tests[0].metadata,
            "stage": "ready",
            "hypothesis": "Candidate evidence will improve ranking.",
            "rationale": "The control leaves room for signal.",
            "expected_signal": "A positive nDCG delta.",
            "decision_rule": "",
            "dependencies": "Development evidence.",
            "priority": "high",
            "evaluation_boundary": "Development only.",
            "artifact_plan": "Store a receipt.",
            "next_action": "Run the evaluation.",
            "missing_readiness_fields": [],
        },
    )
    with pytest.raises(ValueError, match="decision_rule"):
        validate_hub_sources(
            replace(minimal_sources, tests=(ready,)),
            ledger_build,
            tmp_path,
        )


def test_validation_rejects_unknown_study_system(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    study = replace(
        minimal_sources.studies[0],
        metadata={
            **minimal_sources.studies[0].metadata,
            "system_ids": ["not-a-real-system"],
        },
    )
    with pytest.raises(ValueError, match="not-a-real-system"):
        validate_hub_sources(
            replace(minimal_sources, studies=(study,)),
            ledger_build,
            tmp_path,
        )


def test_validation_requires_reconstructed_rationale_evidence(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    study = replace(
        minimal_sources.studies[0],
        metadata={
            **minimal_sources.studies[0].metadata,
            "rationale_provenance": "reconstructed",
            "evidence_refs": [],
        },
    )
    with pytest.raises(ValueError, match="reconstructed.*evidence_refs"):
        validate_hub_sources(
            replace(minimal_sources, studies=(study,)),
            ledger_build,
            tmp_path,
        )


def test_validation_rejects_cross_scope_comparison(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    study = replace(
        minimal_sources.studies[0],
        metadata={
            **minimal_sources.studies[0].metadata,
            "comparison_system_ids": ["dense_nested", "pointwise_qwen_feasibility_6"],
        },
    )
    with pytest.raises(ValueError, match="evaluation scope"):
        validate_hub_sources(
            replace(minimal_sources, studies=(study,)),
            ledger_build,
            tmp_path,
        )


def test_validation_accepts_yaml_date_values(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    question = replace(
        minimal_sources.questions[0],
        metadata={
            **minimal_sources.questions[0].metadata,
            "updated_on": date(2026, 7, 24),
        },
    )
    study = replace(
        minimal_sources.studies[0],
        metadata={
            **minimal_sources.studies[0].metadata,
            "started_on": date(2026, 7, 1),
            "concluded_on": date(2026, 7, 2),
        },
    )
    background = replace(
        minimal_sources.background[0],
        metadata={
            **minimal_sources.background[0].metadata,
            "reviewed_on": date(2026, 7, 24),
        },
    )

    validate_hub_sources(
        replace(
            minimal_sources,
            questions=(question,),
            studies=(study,),
            background=(background,),
        ),
        ledger_build,
        tmp_path,
    )


def test_validation_rejects_concluded_study_without_concluded_on(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    study = replace(
        minimal_sources.studies[0],
        metadata={
            **minimal_sources.studies[0].metadata,
            "concluded_on": None,
        },
    )

    with pytest.raises(ValueError, match="concluded_on"):
        validate_hub_sources(
            replace(minimal_sources, studies=(study,)), ledger_build, tmp_path
        )


def test_validation_rejects_unknown_missing_readiness_field(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    test = replace(
        minimal_sources.tests[0],
        metadata={
            **minimal_sources.tests[0].metadata,
            "missing_readiness_fields": ["decision_rul"],
        },
    )

    with pytest.raises(ValueError, match="missing_readiness_fields"):
        validate_hub_sources(
            replace(minimal_sources, tests=(test,)), ledger_build, tmp_path
        )
