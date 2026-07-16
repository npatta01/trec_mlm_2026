from __future__ import annotations

from dataclasses import asdict
import builtins
import hashlib
import io
import itertools
import json
import math
from dataclasses import MISSING, FrozenInstanceError, fields
import os
from pathlib import Path
from typing import Any, get_type_hints

import pytest

import trec_mlm.dense_aggregation_experiment as experiment_module
from trec_mlm.dense_aggregation import DENSE_AGGREGATION_GRID
from trec_mlm.dense_aggregation_experiment import (
    CLUSTER_ASSIGNMENT_COLUMNS,
    CONSTRUCTION_PROVENANCE_SCHEMA,
    CONSTRUCTION_SCHEMA,
    CONSTRUCTION_VOLATILE_JSON_PATHS,
    INNER_METRIC_COLUMNS,
    NEIGHBOR_COLUMNS,
    OFFICIAL_CONFIG_SCHEMA,
    PREDICTION_COLUMNS,
    PREDICTION_FREEZE_SCHEMA,
    PRIMARY_TREC_FILENAME,
    PRIMARY_TREC_RUN_ID,
    SELECTION_COLUMNS,
    ConstructionProvenance,
    ConstructionReceipt,
    DenseAggregationExperimentConfig,
    FitWorkItem,
    OfficialPreflight,
    PredictionFreezeReceipt,
    candidate_inventory_sha256,
    iter_pair_fit_work,
    iter_single_fit_work,
    load_official_config,
    retained_inner_row,
)
from trec_mlm.dense_aggregation_hashing import canonical_json, framed_sha256
from trec_mlm.dense_aggregation_contract import FrozenDenseInputs


def _query_ids(count: int) -> tuple[str, ...]:
    return tuple(f"q{index:03d}" for index in range(count))


def _official_config_payload() -> dict[str, object]:
    return {
        "schema": OFFICIAL_CONFIG_SCHEMA,
        "query_path": "data/dev/queries.tsv",
        "qrel_path": "sealed/dev/qrels.txt",
        "embedding_path": "artifacts/frozen/embeddings.npy",
        "encoder_metadata_path": "artifacts/frozen/metadata.json",
        "protocol_path": "protocols/trec_2026_dense_aggregation.md",
        "feasibility_receipt_path": "artifacts/feasibility/manifest.json",
        "feasibility_receipt_sha256": "1" * 64,
        "legacy_control_lock_path": "artifacts/legacy/control.lock.json",
        "legacy_control_lock_sha256": "2" * 64,
        "legacy_control_tsv_path": "artifacts/legacy/control.tsv",
        "legacy_control_tsv_sha256": "3" * 64,
        "dense_snapshot_path": "reports/dense/results.json",
        "dense_snapshot_sha256": "4" * 64,
        "dense_per_query_path": "reports/dense/per_query.tsv",
        "dense_per_query_sha256": "5" * 64,
        "grid": [asdict(candidate) for candidate in DENSE_AGGREGATION_GRID],
    }


def _write_official_config(
    tmp_path: Path,
    payload: object | None = None,
    *,
    raw_bytes: bytes | None = None,
) -> tuple[Path, Path, bytes]:
    repo_root = tmp_path / "repo"
    config_path = repo_root / "configs" / "official.json"
    config_path.parent.mkdir(parents=True)
    if raw_bytes is None:
        value = _official_config_payload() if payload is None else payload
        raw_bytes = canonical_json(value) + b"\n"
    config_path.write_bytes(raw_bytes)
    return repo_root.resolve(), config_path.resolve(), raw_bytes


def test_task9b_schema_constants_and_records_are_exact() -> None:
    assert OFFICIAL_CONFIG_SCHEMA == "trec-mlm-dense-aggregation-experiment-config-v1"
    assert CONSTRUCTION_SCHEMA == "trec-mlm-dense-aggregation-construction-v1"
    assert CONSTRUCTION_PROVENANCE_SCHEMA == (
        "trec-mlm-dense-aggregation-construction-provenance-v1"
    )
    assert PREDICTION_FREEZE_SCHEMA == (
        "trec-mlm-dense-aggregation-prediction-freeze-v1"
    )
    assert PRIMARY_TREC_FILENAME == "run.trec"
    assert PRIMARY_TREC_RUN_ID == "qknn_dense_aggregation_nested_loo"
    assert CONSTRUCTION_VOLATILE_JSON_PATHS == (
        ("construction_receipt.json", "/started_at_utc"),
        ("construction_receipt.json", "/finished_at_utc"),
        ("construction_receipt.json", "/wall_elapsed_ns"),
    )

    assert INNER_METRIC_COLUMNS == (
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
    assert SELECTION_COLUMNS == (
        "outer_query_id",
        "selected_config_index",
        "clusters",
        "neighbors",
        "local_weight",
        "beta",
        "winner_margin",
    )
    assert PREDICTION_COLUMNS == (
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
    assert NEIGHBOR_COLUMNS == (
        "outer_query_id",
        "realized_neighbor_count",
        "neighbor_mass",
        "neighbor_ids",
        "neighbor_similarities",
        "neighbor_weights",
    )
    assert CLUSTER_ASSIGNMENT_COLUMNS == (
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

    expected_fields = {
        DenseAggregationExperimentConfig: (
            "repo_root",
            "source_path",
            "source_sha256",
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
        ),
        ConstructionReceipt: (
            "clustering_contract_sha256",
            "construction_provenance_sha256",
            "prediction_freeze_sha256",
            "fit_count",
            "start_count",
            "selected_config_indices",
            "inner_score_hashes",
            "inner_rank_hashes",
            "started_at_utc",
            "finished_at_utc",
            "wall_elapsed_ns",
            "schema",
        ),
        ConstructionProvenance: (
            "query_file_sha256",
            "qrel_file_sha256",
            "query_input_sha256",
            "candidate_inventory_sha256",
            "embedding_sha256",
            "similarity_sha256",
            "protocol_sha256",
            "clustering_contract_sha256",
            "implementation_inventory_sha256",
            "dependency_inventory_sha256",
            "runtime_inventory_sha256",
            "numerical_environment_sha256",
            "feasibility_manifest_sha256",
            "legacy_control_lock_sha256",
            "legacy_control_tsv_sha256",
            "official_config_sha256",
            "schema",
        ),
        PredictionFreezeReceipt: (
            "construction_provenance_sha256",
            "environment_sha256",
            "query_count",
            "candidate_count",
            "config_count",
            "fit_count",
            "start_count",
            "selected_config_indices",
            "inner_score_hashes",
            "inner_rank_hashes",
            "frozen_file_sha256",
            "row_counts",
            "schema",
        ),
        OfficialPreflight: (
            "inputs",
            "clustering_contract",
            "clustering_contract_sha256",
            "feasibility_manifest_sha256",
            "legacy_control_lock_sha256",
            "legacy_control_tsv_sha256",
            "legacy_control_qrel_file_sha256",
            "legacy_control_candidate_inventory_sha256",
        ),
    }
    expected_schemas = {
        ConstructionReceipt: CONSTRUCTION_SCHEMA,
        ConstructionProvenance: CONSTRUCTION_PROVENANCE_SCHEMA,
        PredictionFreezeReceipt: PREDICTION_FREEZE_SCHEMA,
    }
    for record_type, expected in expected_fields.items():
        record_fields = fields(record_type)
        assert tuple(field.name for field in record_fields) == expected
        assert record_type.__dataclass_params__.frozen is True
        assert "__dict__" not in record_type.__slots__
        if record_type in expected_schemas:
            assert record_fields[-1].default == expected_schemas[record_type]
        else:
            assert all(field.default is MISSING for field in record_fields)


def test_official_preflight_public_annotations_resolve_at_runtime() -> None:
    assert get_type_hints(OfficialPreflight)["inputs"] is FrozenDenseInputs


def test_prediction_freeze_receipt_copies_and_freezes_its_maps() -> None:
    frozen_files = {"run.trec": "1" * 64}
    row_counts = {"run.trec": 1}
    receipt = PredictionFreezeReceipt(
        construction_provenance_sha256="2" * 64,
        environment_sha256="3" * 64,
        query_count=1,
        candidate_count=1,
        config_count=1,
        fit_count=1,
        start_count=10,
        selected_config_indices=(0,),
        inner_score_hashes=(("4" * 64,),),
        inner_rank_hashes=(("5" * 64,),),
        frozen_file_sha256=frozen_files,
        row_counts=row_counts,
    )

    frozen_files["run.trec"] = "6" * 64
    row_counts["run.trec"] = 2
    assert receipt.frozen_file_sha256 == {"run.trec": "1" * 64}
    assert receipt.row_counts == {"run.trec": 1}
    with pytest.raises(TypeError):
        receipt.frozen_file_sha256["run.trec"] = "7" * 64  # type: ignore[index]
    with pytest.raises(TypeError):
        receipt.row_counts["run.trec"] = 3  # type: ignore[index]
    serialized = asdict(receipt)
    assert serialized["frozen_file_sha256"] == {"run.trec": "1" * 64}
    assert serialized["row_counts"] == {"run.trec": 1}


def test_official_preflight_recursively_copies_and_freezes_contract() -> None:
    nested = {"rows": [["source.py", "1" * 64]]}
    contract: dict[str, object] = {"implementation": nested}
    preflight = OfficialPreflight(
        inputs=object(),  # type: ignore[arg-type]
        clustering_contract=contract,
        clustering_contract_sha256="2" * 64,
        feasibility_manifest_sha256="3" * 64,
        legacy_control_lock_sha256="4" * 64,
        legacy_control_tsv_sha256="5" * 64,
        legacy_control_qrel_file_sha256="6" * 64,
        legacy_control_candidate_inventory_sha256="7" * 64,
    )

    nested["rows"][0].append("caller mutation")
    assert preflight.clustering_contract == {
        "implementation": {"rows": [["source.py", "1" * 64]]}
    }
    with pytest.raises(TypeError):
        preflight.clustering_contract["new"] = {}  # type: ignore[index]
    contract_view = preflight.clustering_contract["implementation"]
    assert type(contract_view) is dict
    rows_view = contract_view["rows"]
    assert type(rows_view) is list
    assert type(rows_view[0]) is list
    rows_view[0].append("view mutation")
    assert preflight.clustering_contract == {
        "implementation": {"rows": [["source.py", "1" * 64]]}
    }

    serialized = asdict(preflight)
    serialized_contract = serialized["clustering_contract"]
    assert type(serialized_contract) is dict
    serialized_implementation = serialized_contract["implementation"]
    assert type(serialized_implementation) is dict
    serialized_rows = serialized_implementation["rows"]
    assert type(serialized_rows) is list
    assert type(serialized_rows[0]) is list
    serialized_rows[0].append("serialized mutation")
    assert preflight.clustering_contract == {
        "implementation": {"rows": [["source.py", "1" * 64]]}
    }


def test_frozen_file_paths_are_the_exact_sorted_prediction_inventory() -> None:
    expected = (
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
    assert experiment_module.FROZEN_FILE_PATHS == expected
    assert experiment_module.FROZEN_FILE_PATHS == tuple(sorted(expected))


def test_candidate_inventory_hash_binds_exact_order() -> None:
    model_ids = ("model-001", "model-Ω", "model-003")
    expected = framed_sha256(
        "candidate-inventory",
        canonical_json(list(model_ids)),
    )
    assert candidate_inventory_sha256(model_ids) == expected
    assert candidate_inventory_sha256(tuple(reversed(model_ids))) != expected


def test_load_official_config_captures_exact_canonical_source(tmp_path: Path) -> None:
    repo_root, config_path, source_bytes = _write_official_config(tmp_path)

    config = load_official_config(config_path)

    assert config.repo_root == repo_root
    assert config.source_path == config_path
    assert config.source_sha256 == hashlib.sha256(source_bytes).hexdigest()
    assert config.query_path == repo_root / "data/dev/queries.tsv"
    assert config.qrel_path == repo_root / "sealed/dev/qrels.txt"
    assert config.embedding_path == repo_root / "artifacts/frozen/embeddings.npy"
    assert config.encoder_metadata_path == repo_root / "artifacts/frozen/metadata.json"
    assert config.protocol_path == (
        repo_root / "protocols/trec_2026_dense_aggregation.md"
    )
    assert config.feasibility_receipt_path == (
        repo_root / "artifacts/feasibility/manifest.json"
    )
    assert config.legacy_control_lock_path == (
        repo_root / "artifacts/legacy/control.lock.json"
    )
    assert config.legacy_control_tsv_path == repo_root / "artifacts/legacy/control.tsv"
    assert config.dense_snapshot_path == repo_root / "reports/dense/results.json"
    assert config.dense_per_query_path == repo_root / "reports/dense/per_query.tsv"
    assert config.feasibility_receipt_sha256 == "1" * 64
    assert config.legacy_control_lock_sha256 == "2" * 64
    assert config.legacy_control_tsv_sha256 == "3" * 64
    assert config.dense_snapshot_sha256 == "4" * 64
    assert config.dense_per_query_sha256 == "5" * 64
    assert config.grid == DENSE_AGGREGATION_GRID
    assert not hasattr(config, "__dict__")
    with pytest.raises(FrozenInstanceError):
        config.source_sha256 = "0" * 64  # type: ignore[misc]


def test_load_official_config_rejects_opened_inode_swap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_payload = _official_config_payload()
    original_payload["query_path"] = "data/dev/one.tsv"
    _, config_path, original_bytes = _write_official_config(
        tmp_path,
        original_payload,
    )
    alternate_payload = _official_config_payload()
    alternate_payload["query_path"] = "data/dev/two.tsv"
    alternate_path = config_path.with_name("alternate.json")
    alternate_bytes = canonical_json(alternate_payload) + b"\n"
    assert len(alternate_bytes) == len(original_bytes)
    alternate_path.write_bytes(alternate_bytes)

    original_open = os.open
    observed_flags: list[int] = []

    def swapped_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        observed_flags.append(flags)
        opened_path = alternate_path if Path(path) == config_path else path
        return original_open(opened_path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", swapped_open)

    with pytest.raises(ValueError, match="identity"):
        load_official_config(config_path)
    assert len(observed_flags) == 1
    assert observed_flags[0] & os.O_NOFOLLOW
    assert observed_flags[0] & os.O_NONBLOCK
    assert observed_flags[0] & os.O_ACCMODE == os.O_RDONLY


def test_load_official_config_never_touches_qrel_leaf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, config_path, _ = _write_official_config(tmp_path)
    qrel_path = repo_root / "sealed/dev/qrels.txt"

    def is_qrel_path(value: object) -> bool:
        if isinstance(value, int):
            return False
        try:
            return Path(value) == qrel_path  # type: ignore[arg-type]
        except TypeError:
            return False

    for method_name in ("resolve", "stat", "lstat", "open"):
        original = getattr(Path, method_name)

        def guarded_path_method(
            self: Path,
            *args: object,
            _method_name: str = method_name,
            _original: Any = original,
            **kwargs: object,
        ) -> Any:
            if self == qrel_path:
                raise AssertionError(f"qrel leaf {_method_name} before preflight")
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(Path, method_name, guarded_path_method)

    for module, function_name in (
        (os, "stat"),
        (os, "lstat"),
        (os, "open"),
        (builtins, "open"),
        (io, "open"),
    ):
        original = getattr(module, function_name)

        def guarded_file_operation(
            path: object,
            *args: object,
            _function_name: str = function_name,
            _original: Any = original,
            **kwargs: object,
        ) -> Any:
            if is_qrel_path(path):
                raise AssertionError(f"qrel leaf {_function_name} before preflight")
            return _original(path, *args, **kwargs)

        monkeypatch.setattr(module, function_name, guarded_file_operation)

    config = load_official_config(config_path)
    assert config.qrel_path == qrel_path


@pytest.mark.parametrize("missing", tuple(_official_config_payload()))
def test_load_official_config_rejects_every_missing_key(
    tmp_path: Path,
    missing: str,
) -> None:
    payload = _official_config_payload()
    del payload[missing]
    _, config_path, _ = _write_official_config(tmp_path, payload)
    with pytest.raises(ValueError, match="keys differ"):
        load_official_config(config_path)


def test_load_official_config_rejects_extra_key(tmp_path: Path) -> None:
    payload = _official_config_payload()
    payload["unexpected"] = "value"
    _, config_path, _ = _write_official_config(tmp_path, payload)
    with pytest.raises(ValueError, match="keys differ"):
        load_official_config(config_path)


@pytest.mark.parametrize("schema", ("v2", 1, None, True))
def test_load_official_config_rejects_unknown_or_wrong_type_schema(
    tmp_path: Path,
    schema: object,
) -> None:
    payload = _official_config_payload()
    payload["schema"] = schema
    _, config_path, _ = _write_official_config(tmp_path, payload)
    with pytest.raises(ValueError, match="schema"):
        load_official_config(config_path)


@pytest.mark.parametrize(
    ("key", "invalid"),
    (
        ("query_path", 7),
        ("qrel_path", True),
        ("legacy_control_lock_path", None),
        ("feasibility_receipt_sha256", True),
        ("legacy_control_lock_sha256", "A" * 64),
        ("legacy_control_tsv_sha256", "0" * 63),
        ("dense_snapshot_sha256", 5),
        ("dense_per_query_sha256", None),
    ),
)
def test_load_official_config_rejects_wrong_field_types_or_digests(
    tmp_path: Path,
    key: str,
    invalid: object,
) -> None:
    payload = _official_config_payload()
    payload[key] = invalid
    _, config_path, _ = _write_official_config(tmp_path, payload)
    with pytest.raises(ValueError, match=key):
        load_official_config(config_path)


@pytest.mark.parametrize(
    "invalid_path",
    (
        "",
        ".",
        "../outside",
        "data/../outside",
        "/absolute/path",
        "data/./queries.tsv",
        "data//queries.tsv",
        "data/queries.tsv/",
        "data\\queries.tsv",
        "data/queries\x00.tsv",
    ),
)
def test_load_official_config_rejects_noncanonical_repository_paths(
    tmp_path: Path,
    invalid_path: str,
) -> None:
    payload = _official_config_payload()
    payload["qrel_path"] = invalid_path
    _, config_path, _ = _write_official_config(tmp_path, payload)
    with pytest.raises(ValueError, match="qrel_path"):
        load_official_config(config_path)


@pytest.mark.parametrize(
    "mutation",
    ("not_list", "missing_row", "reordered", "extra_key", "bool_int", "int_float"),
)
def test_load_official_config_rejects_grid_shape_types_and_order(
    tmp_path: Path,
    mutation: str,
) -> None:
    payload = _official_config_payload()
    grid = payload["grid"]
    assert type(grid) is list
    if mutation == "not_list":
        payload["grid"] = {"invalid": True}
    elif mutation == "missing_row":
        payload["grid"] = grid[:-1]
    elif mutation == "reordered":
        payload["grid"] = list(reversed(grid))
    else:
        rows = [dict(row) for row in grid]
        if mutation == "extra_key":
            rows[0]["extra"] = 1
        elif mutation == "bool_int":
            rows[0]["clusters"] = True
        elif mutation == "int_float":
            rows[0]["local_weight"] = 1
        payload["grid"] = rows
    _, config_path, _ = _write_official_config(tmp_path, payload)
    with pytest.raises(ValueError, match="grid"):
        load_official_config(config_path)


@pytest.mark.parametrize(
    "raw_builder",
    (
        lambda value: canonical_json(value),
        lambda value: json.dumps(value, indent=2).encode("utf-8") + b"\n",
        lambda value: canonical_json(value) + b"\n\n",
    ),
)
def test_load_official_config_rejects_noncanonical_file_bytes(
    tmp_path: Path,
    raw_builder: Any,
) -> None:
    raw_bytes = raw_builder(_official_config_payload())
    _, config_path, _ = _write_official_config(tmp_path, raw_bytes=raw_bytes)
    with pytest.raises(ValueError, match="canonical JSON"):
        load_official_config(config_path)


def test_load_official_config_rejects_nonobject_and_oversized_input(
    tmp_path: Path,
) -> None:
    _, list_path, _ = _write_official_config(tmp_path, payload=[])
    with pytest.raises(ValueError, match="JSON object"):
        load_official_config(list_path)

    list_path.unlink()
    list_path.write_bytes(b" " * (64 * 1024 + 1))
    with pytest.raises(ValueError, match="byte limit"):
        load_official_config(list_path)


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
