"""Validated build input for deterministic research-hub views."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from trec_mlm.experiment_ledger import LedgerBuild, build_ledger

from .load import load_hub_sources
from .model import HubSources
from .validate import ValidationSummary, validate_hub_sources


@dataclass(frozen=True)
class HubBuild:
    sources: HubSources
    ledger: LedgerBuild
    validation: ValidationSummary


def build_research_hub(root: Path) -> HubBuild:
    """Load and validate every canonical input used by generated hub views."""

    sources = load_hub_sources(root)
    ledger = build_ledger(root)
    validation = validate_hub_sources(sources, ledger, root)
    return HubBuild(sources=sources, ledger=ledger, validation=validation)
