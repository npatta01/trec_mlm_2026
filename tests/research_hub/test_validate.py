from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from trec_mlm.experiment_ledger import LedgerBuild
from trec_mlm.research_hub.model import HubSources
from trec_mlm.research_hub.validate import validate_hub_sources

from conftest import CAUSAL_VISUAL

def test_validation_returns_record_and_linked_system_counts(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    summary = validate_hub_sources(minimal_sources, ledger_build, tmp_path)

    assert summary.record_counts == {
        "theme": 1,
        "question": 1,
        "study": 1,
        "test": 1,
        "approach": 1,
        "dataset": 1,
        "background": 1,
    }
    assert summary.linked_system_count == len(ledger_build.systems)


@pytest.mark.parametrize(
    "field",
    [
        "plain_english",
        "current_conclusion",
        "decision",
        "confidence",
        "confidence_boundary",
        "next_test_ids",
        "updated_on",
        "evidence_refs",
    ],
)
def test_validation_requires_theme_fields(
    field: str,
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    theme = minimal_sources.themes[0]
    metadata = dict(theme.metadata)
    metadata.pop(field)

    with pytest.raises(ValueError, match=field):
        validate_hub_sources(
            replace(minimal_sources, themes=(replace(theme, metadata=metadata),)),
            ledger_build,
            tmp_path,
        )


def test_validation_requires_theme_evidence_refs(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    theme = minimal_sources.themes[0]
    theme = replace(
        theme,
        metadata={**theme.metadata, "evidence_refs": []},
    )

    with pytest.raises(ValueError, match="evidence_refs"):
        validate_hub_sources(
            replace(minimal_sources, themes=(theme,)),
            ledger_build,
            tmp_path,
        )


def test_validation_rejects_unknown_primary_theme(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    question = minimal_sources.questions[0]
    question = replace(
        question,
        metadata={
            **question.metadata,
            "primary_theme_id": "unknown-theme",
            "theme_ids": ["unknown-theme"],
        },
    )

    with pytest.raises(ValueError, match="primary_theme_id references unknown"):
        validate_hub_sources(
            replace(minimal_sources, questions=(question,)),
            ledger_build,
            tmp_path,
        )


def test_validation_requires_primary_theme_in_theme_ids(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    question = minimal_sources.questions[0]
    question = replace(
        question,
        metadata={
            **question.metadata,
            "theme_ids": [],
        },
    )

    with pytest.raises(ValueError, match="primary_theme_id must occur in theme_ids"):
        validate_hub_sources(
            replace(minimal_sources, questions=(question,)),
            ledger_build,
            tmp_path,
        )


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


def test_validation_requires_question_confidence_boundary(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    question = replace(
        minimal_sources.questions[0],
        metadata={
            **minimal_sources.questions[0].metadata,
            "confidence_boundary": "",
        },
    )

    with pytest.raises(ValueError, match="confidence_boundary"):
        validate_hub_sources(
            replace(minimal_sources, questions=(question,)), ledger_build, tmp_path
        )


def test_validation_requires_question_evidence_refs(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    question = replace(
        minimal_sources.questions[0],
        metadata={
            **minimal_sources.questions[0].metadata,
            "evidence_refs": [],
        },
    )

    with pytest.raises(ValueError, match="evidence_refs"):
        validate_hub_sources(
            replace(minimal_sources, questions=(question,)), ledger_build, tmp_path
        )


def test_validation_requires_evidence_for_study_decision(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    study = replace(
        minimal_sources.studies[0],
        metadata={
            **minimal_sources.studies[0].metadata,
            "evidence_refs": [],
        },
    )

    with pytest.raises(ValueError, match="decision.*evidence_refs"):
        validate_hub_sources(
            replace(minimal_sources, studies=(study,)), ledger_build, tmp_path
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


@pytest.mark.parametrize("field", ["plain_english", "why_it_matters", "next_action"])
def test_validation_requires_test_explainer_fields(
    field: str,
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    test = minimal_sources.tests[0]
    test = replace(test, metadata={**test.metadata, field: ""})

    with pytest.raises(ValueError, match=field):
        validate_hub_sources(
            replace(minimal_sources, tests=(test,)), ledger_build, tmp_path
        )


def test_validation_requires_nonempty_test_mechanism_steps(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    test = minimal_sources.tests[0]
    test = replace(test, metadata={**test.metadata, "mechanism_steps": []})

    with pytest.raises(ValueError, match="mechanism_steps must not be empty"):
        validate_hub_sources(
            replace(minimal_sources, tests=(test,)), ledger_build, tmp_path
        )


def test_validation_rejects_incomplete_causal_comparison(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    idea = replace(
        minimal_sources.tests[0],
        metadata={
            **minimal_sources.tests[0].metadata,
            "visual": {
                "type": "causal_comparison",
                "current_label": "Today",
            },
        },
    )

    with pytest.raises(ValueError, match="current_summary"):
        validate_hub_sources(
            replace(minimal_sources, tests=(idea,)), ledger_build, tmp_path
        )


def test_validation_rejects_unsupported_visual_type(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    idea = replace(
        minimal_sources.tests[0],
        metadata={
            **minimal_sources.tests[0].metadata,
            "visual": {"type": "placeholder"},
        },
    )

    with pytest.raises(ValueError, match="visual.type"):
        validate_hub_sources(
            replace(minimal_sources, tests=(idea,)), ledger_build, tmp_path
        )


def test_validation_accepts_record_without_visual(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    validate_hub_sources(minimal_sources, ledger_build, tmp_path)


def test_validation_accepts_complete_causal_comparison_and_rejects_unknown_keys(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    idea = replace(
        minimal_sources.tests[0],
        metadata={**minimal_sources.tests[0].metadata, "visual": CAUSAL_VISUAL},
    )
    sources = replace(minimal_sources, tests=(idea,))

    validate_hub_sources(sources, ledger_build, tmp_path)

    unknown_key = replace(
        idea,
        metadata={
            **idea.metadata,
            "visual": {**CAUSAL_VISUAL, "current_summry": "Typo."},
        },
    )
    with pytest.raises(ValueError, match="unknown field"):
        validate_hub_sources(
            replace(minimal_sources, tests=(unknown_key,)), ledger_build, tmp_path
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


def test_validation_rejects_orphan_ledger_run(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    study = minimal_sources.studies[0]
    orphan_id = ledger_build.systems[-1].system_id
    study = replace(
        study,
        metadata={
            **study.metadata,
            "system_ids": [
                system_id
                for system_id in study.metadata["system_ids"]
                if system_id != orphan_id
            ],
        },
    )

    with pytest.raises(ValueError, match=rf"orphan ledger run.*{orphan_id}"):
        validate_hub_sources(
            replace(minimal_sources, studies=(study,)), ledger_build, tmp_path
        )


def test_validation_rejects_study_missing_from_linked_question(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    question = minimal_sources.questions[0]
    question = replace(
        question,
        metadata={**question.metadata, "study_ids": []},
    )

    with pytest.raises(ValueError, match="question.*study relationships.*dense-baseline"):
        validate_hub_sources(
            replace(minimal_sources, questions=(question,)), ledger_build, tmp_path
        )


def test_validation_rejects_question_missing_from_linked_study(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    question = minimal_sources.questions[0]
    secondary = replace(
        question,
        record_id="secondary-question",
        source_path="research/hub/questions/secondary-question.qmd",
        metadata={
            **question.metadata,
            "question_id": "secondary-question",
            "study_ids": [],
        },
    )
    study = minimal_sources.studies[0]
    study = replace(
        study,
        metadata={**study.metadata, "question_ids": ["secondary-question"]},
    )

    with pytest.raises(
        ValueError, match="question.*study relationships.*candidate-evidence"
    ):
        validate_hub_sources(
            replace(
                minimal_sources,
                questions=(question, secondary),
                studies=(study,),
            ),
            ledger_build,
            tmp_path,
        )


def test_validation_rejects_study_missing_from_linked_approach(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    approach = minimal_sources.approaches[0]
    approach = replace(
        approach,
        metadata={**approach.metadata, "study_ids": []},
    )

    with pytest.raises(ValueError, match="approach-study relationships.*dense-baseline"):
        validate_hub_sources(
            replace(minimal_sources, approaches=(approach,)),
            ledger_build,
            tmp_path,
        )


def test_validation_rejects_approach_missing_from_linked_study(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    study = minimal_sources.studies[0]
    study = replace(
        study,
        metadata={**study.metadata, "approach_ids": []},
    )

    with pytest.raises(
        ValueError, match="approach-study relationships.*dense-retrieval"
    ):
        validate_hub_sources(
            replace(minimal_sources, studies=(study,)),
            ledger_build,
            tmp_path,
        )


def test_validation_rejects_study_missing_from_linked_dataset(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    dataset = minimal_sources.datasets[0]
    dataset = replace(
        dataset,
        metadata={**dataset.metadata, "study_ids": []},
    )

    with pytest.raises(ValueError, match="dataset-study relationships.*dense-baseline"):
        validate_hub_sources(
            replace(minimal_sources, datasets=(dataset,)),
            ledger_build,
            tmp_path,
        )


def test_validation_rejects_dataset_missing_from_linked_study(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    tmp_path: Path,
) -> None:
    study = minimal_sources.studies[0]
    study = replace(
        study,
        metadata={**study.metadata, "dataset_ids": []},
    )

    with pytest.raises(
        ValueError, match="dataset-study relationships.*development-set"
    ):
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
