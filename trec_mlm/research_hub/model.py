from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

RECORD_KINDS = frozenset(
    {"theme", "question", "study", "test", "approach", "dataset", "background"}
)


@dataclass(frozen=True)
class HubRecord:
    kind: str
    record_id: str
    title: str
    source_path: str
    metadata: Mapping[str, Any]
    body: str


@dataclass(frozen=True)
class HubSources:
    themes: tuple[HubRecord, ...] = ()
    questions: tuple[HubRecord, ...] = ()
    studies: tuple[HubRecord, ...] = ()
    tests: tuple[HubRecord, ...] = ()
    approaches: tuple[HubRecord, ...] = ()
    datasets: tuple[HubRecord, ...] = ()
    background: tuple[HubRecord, ...] = ()

    def all_records(self) -> tuple[HubRecord, ...]:
        return (
            self.themes
            + self.questions
            + self.studies
            + self.tests
            + self.approaches
            + self.datasets
            + self.background
        )


@dataclass(frozen=True, slots=True)
class CompletedFollowUpEvidence:
    """Validated aggregate receipt for the off-ledger completed E5 follow-up."""

    study_id: str
    source_path: str
    baseline_id: str
    candidate_id: str
    decision: str
    ndcg_at_10: float
    mrr: float
    zero_ndcg_queries: int
    ndcg_delta: float
    ndcg_ci_lower: float
    ndcg_ci_upper: float
    mrr_delta: float
    mrr_ci_lower: float
    mrr_ci_upper: float
