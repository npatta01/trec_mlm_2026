from __future__ import annotations

import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class FitWorkItem:
    exclusion_kind: Literal["pair", "single"]
    inventory_index: int
    excluded_indices: tuple[int, ...]
    support_indices: tuple[int, ...]
    cluster_count: Literal[20, 40]


def iter_pair_fit_work(query_ids: Sequence[str]) -> Iterator[FitWorkItem]:
    query_count = len(query_ids)
    for inventory_index, excluded_indices in enumerate(
        itertools.combinations(range(query_count), 2)
    ):
        support_indices = tuple(
            index for index in range(query_count) if index not in excluded_indices
        )
        for cluster_count in (20, 40):
            yield FitWorkItem(
                exclusion_kind="pair",
                inventory_index=inventory_index,
                excluded_indices=excluded_indices,
                support_indices=support_indices,
                cluster_count=cluster_count,
            )


def iter_single_fit_work(query_ids: Sequence[str]) -> Iterator[FitWorkItem]:
    query_count = len(query_ids)
    for inventory_index in range(query_count):
        support_indices = tuple(
            index for index in range(query_count) if index != inventory_index
        )
        for cluster_count in (20, 40):
            yield FitWorkItem(
                exclusion_kind="single",
                inventory_index=inventory_index,
                excluded_indices=(inventory_index,),
                support_indices=support_indices,
                cluster_count=cluster_count,
            )


def retained_inner_row(outer_index: int, inner_index: int) -> int:
    if outer_index == inner_index:
        raise ValueError("inner target cannot equal outer target")
    return inner_index if inner_index < outer_index else inner_index - 1
