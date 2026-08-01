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


@dataclass(frozen=True, slots=True)
class Phase0Metrics:
    """One complete effectiveness summary from authoritative Phase 0 evidence."""

    ndcg_at_10: float
    mrr: float
    zero_ndcg_queries: int


@dataclass(frozen=True, slots=True)
class Phase0SystemEvidence:
    """One main or diagnostic Phase 0 system and its ledger-owned state."""

    source_system_id: str
    ledger_system_id: str | None
    role: str
    metrics: Phase0Metrics
    query_count: int
    candidate_count: int
    evaluation_scope: str
    evidence_status: str
    promotion_status: str
    artifact_state: str
    qrel_usage: str


@dataclass(frozen=True, slots=True)
class Phase0PairedRow:
    """One paired metric comparison against the frozen dense control."""

    source_system_id: str
    metric: str
    difference: float
    ci_lower: float
    ci_upper: float
    wins: int
    ties: int
    losses: int


@dataclass(frozen=True, slots=True)
class Phase0ZeroTransitions:
    """Zero/nonzero nDCG transitions for one candidate ranking."""

    source_system_id: str
    zero_both: int
    dense_zero_candidate_nonzero: int
    dense_nonzero_candidate_zero: int
    nonzero_both: int


@dataclass(frozen=True, slots=True)
class Phase0Slice:
    """One complete query or candidate slice across the five frozen systems."""

    family: str
    label: str
    member_count: int
    member_ids: tuple[str, ...]
    systems: tuple[tuple[str, Phase0Metrics], ...]


@dataclass(frozen=True, slots=True)
class Phase0GateRow:
    """One frozen decision-gate check."""

    key: str
    label: str
    required: str
    observed: str
    passed: bool


@dataclass(frozen=True, slots=True)
class PointwisePhase0Evidence:
    """Validated, typed evidence used to render the Phase 0 study."""

    study_id: str
    protocol_path: str
    results_path: str
    per_query_path: str
    candidate_slices_path: str
    systems: tuple[Phase0SystemEvidence, ...]
    paired_rows: tuple[Phase0PairedRow, ...]
    zero_transitions: tuple[Phase0ZeroTransitions, ...]
    query_slices: tuple[Phase0Slice, ...]
    candidate_slices: tuple[Phase0Slice, ...]
    gate_rows: tuple[Phase0GateRow, ...]
    passed_gate_count: int
    decision: str
    recommendation: str
    selector_selected_pairs: int
    selector_source_questions: int
    selector_candidate_use_cap: int
    audited_label_count: int
    validation_metric_cells: int
    validation_candidate_rows: int
    validation_run_rows: int
    validation_source_questions: int
