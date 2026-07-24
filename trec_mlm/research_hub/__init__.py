"""Canonical records and loaders for the TREC research hub."""

from .load import load_hub_sources, parse_qmd_record
from .model import HubRecord, HubSources

__all__ = [
    "HubRecord",
    "HubSources",
    "load_hub_sources",
    "parse_qmd_record",
]
