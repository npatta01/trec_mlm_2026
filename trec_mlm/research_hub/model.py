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
