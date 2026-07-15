from __future__ import annotations

import itertools
import math
from dataclasses import FrozenInstanceError, fields

import pytest

from trec_mlm.dense_aggregation_experiment import (
    FitWorkItem,
    iter_pair_fit_work,
    iter_single_fit_work,
    retained_inner_row,
)


def _query_ids(count: int) -> tuple[str, ...]:
    return tuple(f"q{index:03d}" for index in range(count))


@pytest.mark.parametrize("query_count", (3, 4, 42, 322))
def test_work_scheduler_inventory_order_and_support(query_count: int) -> None:
    query_ids = _query_ids(query_count)
    seen_inventory_keys: set[tuple[str, int, int]] = set()
    pair_count = 0

    pair_work = iter_pair_fit_work(query_ids)
    for inventory_index, excluded_indices in enumerate(
        itertools.combinations(range(query_count), 2)
    ):
        expected_support = tuple(
            index for index in range(query_count) if index not in excluded_indices
        )
        for cluster_count in (20, 40):
            item = next(pair_work)
            assert item.exclusion_kind == "pair"
            assert item.inventory_index == inventory_index
            assert item.excluded_indices == excluded_indices
            assert item.support_indices == expected_support
            assert item.cluster_count == cluster_count
            inventory_key = (
                item.exclusion_kind,
                item.inventory_index,
                item.cluster_count,
            )
            assert inventory_key not in seen_inventory_keys
            seen_inventory_keys.add(inventory_key)
            pair_count += 1
    with pytest.raises(StopIteration):
        next(pair_work)

    single_count = 0
    single_work = iter_single_fit_work(query_ids)
    for inventory_index in range(query_count):
        expected_support = tuple(
            index for index in range(query_count) if index != inventory_index
        )
        for cluster_count in (20, 40):
            item = next(single_work)
            assert item.exclusion_kind == "single"
            assert item.inventory_index == inventory_index
            assert item.excluded_indices == (inventory_index,)
            assert item.support_indices == expected_support
            assert item.cluster_count == cluster_count
            inventory_key = (
                item.exclusion_kind,
                item.inventory_index,
                item.cluster_count,
            )
            assert inventory_key not in seen_inventory_keys
            seen_inventory_keys.add(inventory_key)
            single_count += 1
    with pytest.raises(StopIteration):
        next(single_work)

    assert pair_count == 2 * math.comb(query_count, 2)
    assert single_count == 2 * query_count
    assert len(seen_inventory_keys) == 2 * math.comb(query_count, 2) + 2 * query_count


def test_fit_work_item_is_exact_frozen_slotted_metadata() -> None:
    assert tuple(field.name for field in fields(FitWorkItem)) == (
        "exclusion_kind",
        "inventory_index",
        "excluded_indices",
        "support_indices",
        "cluster_count",
    )
    item = next(iter_pair_fit_work(("q0", "q1", "q2")))
    assert not hasattr(item, "__dict__")
    assert all(
        isinstance(value, (str, int, tuple))
        for value in (
            item.exclusion_kind,
            item.inventory_index,
            item.excluded_indices,
            item.support_indices,
            item.cluster_count,
        )
    )
    with pytest.raises(FrozenInstanceError):
        item.inventory_index = 1  # type: ignore[misc]


def test_official_fit_inventory_and_membership() -> None:
    query_ids = _query_ids(322)
    pair_work = tuple(iter_pair_fit_work(query_ids))
    single_work = tuple(iter_single_fit_work(query_ids))
    assert len(pair_work) == 103_362
    assert len(single_work) == 644
    assert len(pair_work) + len(single_work) == 104_006
    assert 10 * (len(pair_work) + len(single_work)) == 1_040_060
    for item in pair_work:
        assert len(item.excluded_indices) == 2
        assert item.support_indices == tuple(
            index for index in range(322) if index not in item.excluded_indices
        )
    for item in single_work:
        assert len(item.excluded_indices) == 1
        assert item.support_indices == tuple(
            index for index in range(322) if index not in item.excluded_indices
        )


def test_retained_inner_row_order() -> None:
    assert [retained_inner_row(2, inner) for inner in (0, 1, 3, 4)] == [0, 1, 2, 3]
    with pytest.raises(ValueError, match="cannot equal"):
        retained_inner_row(2, 2)
