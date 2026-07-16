from __future__ import annotations

import copy
import hashlib
import itertools
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import stat
from types import MappingProxyType
from typing import Literal, TypeVar

from trec_mlm.dense_aggregation import (
    DENSE_AGGREGATION_GRID,
    DenseAggregationCandidate,
)
from trec_mlm.dense_aggregation_contract import FrozenDenseInputs
from trec_mlm.dense_aggregation_hashing import canonical_json, framed_sha256


OFFICIAL_CONFIG_SCHEMA = "trec-mlm-dense-aggregation-experiment-config-v1"
CONSTRUCTION_SCHEMA = "trec-mlm-dense-aggregation-construction-v1"
CONSTRUCTION_PROVENANCE_SCHEMA = "trec-mlm-dense-aggregation-construction-provenance-v1"
PREDICTION_FREEZE_SCHEMA = "trec-mlm-dense-aggregation-prediction-freeze-v1"
PRIMARY_TREC_FILENAME = "run.trec"
PRIMARY_TREC_RUN_ID = "qknn_dense_aggregation_nested_loo"
CONSTRUCTION_VOLATILE_JSON_PATHS = (
    ("construction_receipt.json", "/started_at_utc"),
    ("construction_receipt.json", "/finished_at_utc"),
    ("construction_receipt.json", "/wall_elapsed_ns"),
)
FROZEN_FILE_PATHS = (
    "cluster_assignments.tsv",
    "construction_provenance.json",
    "environment.json",
    "fits.h5",
    "inner_metrics.tsv",
    "inner_ranks.npy",
    "inner_scores.npy",
    "neighbors.tsv",
    "predictions.tsv",
    "primary_domain_contributions.npy",
    "primary_domain_scores.npy",
    "primary_final_scores.npy",
    "primary_local_contributions.npy",
    "primary_local_scores.npy",
    "primary_prior_contributions.npy",
    "primary_prior_scores.npy",
    "primary_ranks.npy",
    "run.trec",
    "selections.tsv",
)
OFFICIAL_CONFIG_KEYS = (
    "schema",
    "query_path",
    "qrel_path",
    "embedding_path",
    "encoder_metadata_path",
    "protocol_path",
    "feasibility_receipt_path",
    "feasibility_receipt_sha256",
    "legacy_control_lock_path",
    "legacy_control_lock_sha256",
    "legacy_control_tsv_path",
    "legacy_control_tsv_sha256",
    "dense_snapshot_path",
    "dense_snapshot_sha256",
    "dense_per_query_path",
    "dense_per_query_sha256",
    "grid",
)

_OFFICIAL_PATH_KEYS = (
    "query_path",
    "qrel_path",
    "embedding_path",
    "encoder_metadata_path",
    "protocol_path",
    "feasibility_receipt_path",
    "legacy_control_lock_path",
    "legacy_control_tsv_path",
    "dense_snapshot_path",
    "dense_per_query_path",
)
_OFFICIAL_HASH_KEYS = (
    "feasibility_receipt_sha256",
    "legacy_control_lock_sha256",
    "legacy_control_tsv_sha256",
    "dense_snapshot_sha256",
    "dense_per_query_sha256",
)
_GRID_KEYS = {
    "clusters",
    "neighbors",
    "local_weight",
    "prior_shrinkage",
}
_LOWERCASE_HEX = frozenset("0123456789abcdef")
_OFFICIAL_CONFIG_BYTE_LIMIT = 64 * 1024
_T = TypeVar("_T")

INNER_METRIC_COLUMNS = (
    "outer_query_id",
    "config_index",
    "clusters",
    "neighbors",
    "local_weight",
    "beta",
    "inner_macro_ndcg_at_10",
    "inner_macro_mrr",
    "selected",
    "selection_rank",
    "gap_to_winner",
    "winner_margin",
    "score_sha256",
    "rank_sha256",
)
SELECTION_COLUMNS = (
    "outer_query_id",
    "selected_config_index",
    "clusters",
    "neighbors",
    "local_weight",
    "beta",
    "winner_margin",
)
PREDICTION_COLUMNS = (
    "outer_query_id",
    "candidate_id",
    "selected_config_index",
    "clusters",
    "neighbors",
    "local_weight",
    "beta",
    "local_score",
    "domain_score",
    "prior_score",
    "local_contribution",
    "domain_contribution",
    "prior_contribution",
    "final_score",
    "aggregation_score",
    "prior_tie_value",
    "final_rank",
)
NEIGHBOR_COLUMNS = (
    "outer_query_id",
    "realized_neighbor_count",
    "neighbor_mass",
    "neighbor_ids",
    "neighbor_similarities",
    "neighbor_weights",
)
CLUSTER_ASSIGNMENT_COLUMNS = (
    "outer_query_id",
    "target_cluster_id",
    "cluster_support",
    "squared_distance_hex",
    "membership_sha256",
    "center_sha256",
    "cluster_inertia_hex",
    "n_iter",
    "sklearn_inertia_hex",
    "recomputed_total_inertia_hex",
    "fit_membership_sha256",
)


def _freeze_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise TypeError("clustering contract keys must be strings")
        return MappingProxyType(
            {key: _freeze_json_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json_value(item) for item in value)
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise TypeError("clustering contract contains a non-JSON value")


def _thaw_json_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(item) for item in value]
    return value


class _ImmutableJsonMapping(Mapping[str, object]):
    __slots__ = ("_frozen",)

    def __init__(self, value: Mapping[str, object]) -> None:
        frozen = _freeze_json_value(value)
        if not isinstance(frozen, Mapping):  # pragma: no cover - structural guard
            raise TypeError("clustering contract must be a mapping")
        self._frozen = frozen

    def __getitem__(self, key: str) -> object:
        return _thaw_json_value(self._frozen[key])

    def __iter__(self) -> Iterator[str]:
        return iter(self._frozen)

    def __len__(self) -> int:
        return len(self._frozen)

    def __deepcopy__(self, memo: dict[int, object]) -> dict[str, object]:
        return {key: _thaw_json_value(value) for key, value in self._frozen.items()}


class _ImmutableFlatMapping(Mapping[str, _T]):
    __slots__ = ("_values",)

    def __init__(self, value: Mapping[str, _T]) -> None:
        self._values = MappingProxyType(dict(value))

    def __getitem__(self, key: str) -> _T:
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __deepcopy__(self, memo: dict[int, object]) -> dict[str, _T]:
        return {
            copy.deepcopy(key, memo): copy.deepcopy(value, memo)
            for key, value in self._values.items()
        }


@dataclass(frozen=True, slots=True)
class DenseAggregationExperimentConfig:
    repo_root: Path
    source_path: Path
    source_sha256: str
    query_path: Path
    qrel_path: Path
    embedding_path: Path
    encoder_metadata_path: Path
    protocol_path: Path
    feasibility_receipt_path: Path
    feasibility_receipt_sha256: str
    legacy_control_lock_path: Path
    legacy_control_lock_sha256: str
    legacy_control_tsv_path: Path
    legacy_control_tsv_sha256: str
    dense_snapshot_path: Path
    dense_snapshot_sha256: str
    dense_per_query_path: Path
    dense_per_query_sha256: str
    grid: tuple[DenseAggregationCandidate, ...]


@dataclass(frozen=True, slots=True)
class ConstructionReceipt:
    clustering_contract_sha256: str
    construction_provenance_sha256: str
    prediction_freeze_sha256: str
    fit_count: int
    start_count: int
    selected_config_indices: tuple[int, ...]
    inner_score_hashes: tuple[tuple[str, ...], ...]
    inner_rank_hashes: tuple[tuple[str, ...], ...]
    started_at_utc: str
    finished_at_utc: str
    wall_elapsed_ns: int
    schema: str = CONSTRUCTION_SCHEMA


@dataclass(frozen=True, slots=True)
class ConstructionProvenance:
    query_file_sha256: str
    qrel_file_sha256: str
    query_input_sha256: str
    candidate_inventory_sha256: str
    embedding_sha256: str
    similarity_sha256: str
    protocol_sha256: str
    clustering_contract_sha256: str
    implementation_inventory_sha256: str
    dependency_inventory_sha256: str
    runtime_inventory_sha256: str
    numerical_environment_sha256: str
    feasibility_manifest_sha256: str
    legacy_control_lock_sha256: str
    legacy_control_tsv_sha256: str
    official_config_sha256: str
    schema: str = CONSTRUCTION_PROVENANCE_SCHEMA


@dataclass(frozen=True, slots=True)
class PredictionFreezeReceipt:
    construction_provenance_sha256: str
    environment_sha256: str
    query_count: int
    candidate_count: int
    config_count: int
    fit_count: int
    start_count: int
    selected_config_indices: tuple[int, ...]
    inner_score_hashes: tuple[tuple[str, ...], ...]
    inner_rank_hashes: tuple[tuple[str, ...], ...]
    frozen_file_sha256: Mapping[str, str]
    row_counts: Mapping[str, int]
    schema: str = PREDICTION_FREEZE_SCHEMA

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "frozen_file_sha256",
            _ImmutableFlatMapping(self.frozen_file_sha256),
        )
        object.__setattr__(
            self,
            "row_counts",
            _ImmutableFlatMapping(self.row_counts),
        )


@dataclass(frozen=True, slots=True)
class OfficialPreflight:
    inputs: FrozenDenseInputs
    clustering_contract: Mapping[str, object]
    clustering_contract_sha256: str
    feasibility_manifest_sha256: str
    legacy_control_lock_sha256: str
    legacy_control_tsv_sha256: str
    legacy_control_qrel_file_sha256: str
    legacy_control_candidate_inventory_sha256: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "clustering_contract",
            _ImmutableJsonMapping(self.clustering_contract),
        )


def candidate_inventory_sha256(model_ids: Sequence[str]) -> str:
    return framed_sha256(
        "candidate-inventory",
        canonical_json(list(model_ids)),
    )


def _stat_snapshot(status: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        status.st_dev,
        status.st_ino,
        status.st_mode,
        status.st_size,
        status.st_mtime_ns,
        status.st_ctime_ns,
    )


def _read_official_config_bytes(source: Path) -> tuple[bytes, Path]:
    try:
        before = source.lstat()
    except OSError as error:
        raise ValueError(f"official config is missing: {source}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("official config must be a regular file")
    if before.st_size > _OFFICIAL_CONFIG_BYTE_LIMIT:
        raise ValueError(
            f"official config exceeds its {_OFFICIAL_CONFIG_BYTE_LIMIT}-byte limit"
        )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError("official config changed while it was read") from error
    try:
        opened_before = os.fstat(descriptor)
        if (opened_before.st_dev, opened_before.st_ino) != (
            before.st_dev,
            before.st_ino,
        ):
            raise ValueError(
                "official config opened-file identity differs from its path"
            )
        if _stat_snapshot(opened_before) != _stat_snapshot(before):
            raise ValueError("official config changed while it was opened")

        chunks: list[bytes] = []
        remaining = before.st_size + 1
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        opened_after = os.fstat(descriptor)
        resolved_source = source.resolve(strict=True)
        resolved_status = resolved_source.lstat()
        after = source.lstat()
    except (OSError, RuntimeError) as error:
        raise ValueError("official config changed while it was read") from error
    finally:
        os.close(descriptor)
    expected_snapshot = _stat_snapshot(before)
    if (
        len(payload) != before.st_size
        or _stat_snapshot(opened_after) != expected_snapshot
        or _stat_snapshot(resolved_status) != expected_snapshot
        or _stat_snapshot(after) != expected_snapshot
    ):
        raise ValueError("official config changed while it was read")
    return payload, resolved_source


def _parse_official_config(source_bytes: bytes) -> dict[str, object]:
    try:
        payload_value = json.loads(source_bytes)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("official config is not valid JSON") from error
    try:
        canonical_bytes = canonical_json(payload_value) + b"\n"
    except (TypeError, ValueError) as error:
        raise ValueError("official config is not canonical JSON") from error
    if source_bytes != canonical_bytes:
        raise ValueError("official config is not canonical JSON")
    if type(payload_value) is not dict:
        raise ValueError("official config must be a JSON object")
    payload: dict[str, object] = payload_value
    expected_keys = set(OFFICIAL_CONFIG_KEYS)
    actual_keys = set(payload)
    if actual_keys != expected_keys:
        raise ValueError(
            "official config keys differ: "
            f"missing={sorted(expected_keys - actual_keys)!r}, "
            f"extra={sorted(actual_keys - expected_keys)!r}"
        )
    if (
        type(payload["schema"]) is not str
        or payload["schema"] != OFFICIAL_CONFIG_SCHEMA
    ):
        raise ValueError("unsupported official config schema")
    return payload


def _require_sha256(value: object, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWERCASE_HEX for character in value)
    ):
        raise ValueError(f"official config {name} must be a lowercase SHA-256")
    return value


def _require_relative_posix_path(value: object, *, name: str) -> PurePosixPath:
    if type(value) is not str or not value:
        raise ValueError(f"official config {name} must be a nonempty string")
    if "\\" in value or any(ord(character) < 32 for character in value):
        raise ValueError(f"official config {name} is not a canonical POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or value == "."
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"official config {name} is not a canonical relative path")
    return path


def _parse_grid(value: object) -> tuple[DenseAggregationCandidate, ...]:
    if type(value) is not list or len(value) != len(DENSE_AGGREGATION_GRID):
        raise ValueError("official config grid has the wrong shape")
    parsed: list[DenseAggregationCandidate] = []
    for index, (row, expected) in enumerate(
        zip(value, DENSE_AGGREGATION_GRID, strict=True)
    ):
        if type(row) is not dict or set(row) != _GRID_KEYS:
            raise ValueError(f"official config grid row {index} has invalid keys")
        clusters = row["clusters"]
        neighbors = row["neighbors"]
        local_weight = row["local_weight"]
        prior_shrinkage = row["prior_shrinkage"]
        if type(clusters) is not int or type(neighbors) is not int:
            raise ValueError(f"official config grid row {index} integer type mismatch")
        if type(local_weight) is not float or type(prior_shrinkage) is not float:
            raise ValueError(f"official config grid row {index} float type mismatch")
        candidate = DenseAggregationCandidate(
            clusters=clusters,
            neighbors=neighbors,
            local_weight=local_weight,
            prior_shrinkage=prior_shrinkage,
        )
        if (
            candidate.clusters != expected.clusters
            or candidate.neighbors != expected.neighbors
            or candidate.local_weight.hex() != expected.local_weight.hex()
            or candidate.prior_shrinkage.hex() != expected.prior_shrinkage.hex()
        ):
            raise ValueError(
                f"official config grid row {index} differs from frozen grid"
            )
        parsed.append(candidate)
    return tuple(parsed)


def load_official_config(path: str | Path) -> DenseAggregationExperimentConfig:
    source = Path(path)
    source_bytes, source_path = _read_official_config_bytes(source)
    payload = _parse_official_config(source_bytes)
    repo_root = source_path.parent.parent

    resolved_paths = {
        name: repo_root.joinpath(
            *_require_relative_posix_path(payload[name], name=name).parts
        )
        for name in _OFFICIAL_PATH_KEYS
    }
    digests = {
        name: _require_sha256(payload[name], name=name) for name in _OFFICIAL_HASH_KEYS
    }
    return DenseAggregationExperimentConfig(
        repo_root=repo_root,
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        query_path=resolved_paths["query_path"],
        qrel_path=resolved_paths["qrel_path"],
        embedding_path=resolved_paths["embedding_path"],
        encoder_metadata_path=resolved_paths["encoder_metadata_path"],
        protocol_path=resolved_paths["protocol_path"],
        feasibility_receipt_path=resolved_paths["feasibility_receipt_path"],
        feasibility_receipt_sha256=digests["feasibility_receipt_sha256"],
        legacy_control_lock_path=resolved_paths["legacy_control_lock_path"],
        legacy_control_lock_sha256=digests["legacy_control_lock_sha256"],
        legacy_control_tsv_path=resolved_paths["legacy_control_tsv_path"],
        legacy_control_tsv_sha256=digests["legacy_control_tsv_sha256"],
        dense_snapshot_path=resolved_paths["dense_snapshot_path"],
        dense_snapshot_sha256=digests["dense_snapshot_sha256"],
        dense_per_query_path=resolved_paths["dense_per_query_path"],
        dense_per_query_sha256=digests["dense_per_query_sha256"],
        grid=_parse_grid(payload["grid"]),
    )


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
