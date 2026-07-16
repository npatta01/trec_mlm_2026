from __future__ import annotations

import builtins
import dataclasses
from dataclasses import replace
import hashlib
import io
import itertools
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import get_type_hints

import h5py
import numpy as np
import pytest

import trec_mlm.dense_aggregation_feasibility as feasibility
from trec_mlm import dense_aggregation_artifacts as artifacts
from trec_mlm.dense_aggregation_artifacts import raw_file_sha256, validate_manifest
from trec_mlm.dense_aggregation_clustering import (
    CanonicalClusterFit,
    FitKey,
    NumericalEnvironmentReceipt,
    TargetAssignment,
    numerical_environment_file_bytes,
)
from trec_mlm.dense_aggregation_contract import FeasibilityConfig, FrozenDenseInputs
from trec_mlm.dense_aggregation_feasibility import (
    FEASIBILITY_PROVENANCE_KEYS,
    FEASIBILITY_REPEAT_CHECKS,
    FeasibilityGate,
    FeasibilityReceipt,
    TimingCategorySummary,
    _linear_p95,
    _projection_seconds,
    _receipt_from_dict,
    _receipt_to_dict,
    _repeat_integrity_failures,
    _summarize_timing_rows,
    evaluate_feasibility_repeats,
    official_sample_indices,
    parse_gnu_time_verbose,
)
from trec_mlm.dense_aggregation_hashing import canonical_json, framed_sha256
from trec_mlm.dense_aggregation_hashing import (
    canonical_membership_sha256,
    logical_array_sha256,
)


FIXTURE_ROOT = Path(__file__).parent / "fixtures/dense_aggregation"
SAMPLE_FIXTURE = FIXTURE_ROOT / "sampling_numpy_2_5_1.json"
TIME_FIXTURE = FIXTURE_ROOT / "time_v.txt"
SAMPLE_FIXTURE_SHA256 = (
    "6b74301b9edb16438739a2c46b455ff37985698a6795bd7a4b9d1f5530cf9d39"
)
CATEGORIES = ("pair_c20", "pair_c40", "single_c20", "single_c40")


def _summary(mean: float = 0.01, *, fit_count: int = 1) -> TimingCategorySummary:
    return TimingCategorySummary(
        fit_count=fit_count,
        mean_seconds=mean,
        median_seconds=mean,
        p95_seconds=mean,
        maximum_seconds=mean,
        p95_method="linear_index_(n-1)*0.95",
        n_iter_min=1,
        n_iter_mean=1.0,
        n_iter_max=1,
        n_iter_values_sha256=framed_sha256(
            "feasibility-n-iter-values", canonical_json([1] * fit_count)
        ),
    )


def _provenance(root: Path, worker: str) -> dict[str, object]:
    values: dict[str, object] = {name: "1" * 64 for name in FEASIBILITY_PROVENANCE_KEYS}
    values.update(
        {
            "python_version": "3.13.9",
            "numpy_version": "2.5.1",
            "scipy_version": "1.18.0",
            "scikit_learn_version": "1.9.0",
            "thread_environment": {
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "BLIS_NUM_THREADS": "1",
                "VECLIB_MAXIMUM_THREADS": "1",
                "NUMEXPR_NUM_THREADS": "1",
            },
            "fit_work_file": str((root / f"{worker}-fits.partial.h5").resolve()),
            "output_root": str((root / f"{worker}-output").resolve()),
            "worker_pid": 101 if worker == "a" else 202,
        }
    )
    return values


def _receipt(root: Path, worker: str = "a") -> FeasibilityReceipt:
    return FeasibilityReceipt(
        fit_count=576,
        start_count=5_760,
        pair_indices=tuple(range(256)),
        single_indices=tuple(range(32)),
        timing_summary={name: _summary() for name in CATEGORIES},
        fit_file_sha256="2" * 64,
        fit_evidence_sha256="3" * 64,
        provenance=_provenance(root, worker),
        exclusive_resource_attested=True,
    )


def test_public_dataclasses_are_exact_frozen_slotted_contracts() -> None:
    expected = {
        TimingCategorySummary: (
            "fit_count",
            "mean_seconds",
            "median_seconds",
            "p95_seconds",
            "maximum_seconds",
            "p95_method",
            "n_iter_min",
            "n_iter_mean",
            "n_iter_max",
            "n_iter_values_sha256",
        ),
        FeasibilityReceipt: (
            "fit_count",
            "start_count",
            "pair_indices",
            "single_indices",
            "timing_summary",
            "fit_file_sha256",
            "fit_evidence_sha256",
            "provenance",
            "exclusive_resource_attested",
            "schema",
        ),
        FeasibilityGate: (
            "status",
            "failures",
            "projection_seconds",
            "maximum_rss_bytes",
            "schema",
        ),
    }
    for data_type, fields in expected.items():
        assert dataclasses.is_dataclass(data_type)
        assert data_type.__dataclass_params__.frozen is True
        assert data_type.__slots__ == fields
        assert tuple(field.name for field in dataclasses.fields(data_type)) == fields
        assert get_type_hints(data_type)


def test_repeat_constants_are_exact_and_ordered() -> None:
    assert FEASIBILITY_PROVENANCE_KEYS == {
        "config_sha256",
        "protocol_sha256",
        "clustering_contract_sha256",
        "implementation_inventory_sha256",
        "dependency_inventory_sha256",
        "runtime_inventory_sha256",
        "query_file_sha256",
        "query_input_sha256",
        "embedding_sha256",
        "similarity_sha256",
        "sample_fixture_sha256",
        "numerical_environment_sha256",
        "python_version",
        "numpy_version",
        "scipy_version",
        "scikit_learn_version",
        "thread_environment",
        "fit_work_file",
        "output_root",
        "worker_pid",
    }
    assert FEASIBILITY_REPEAT_CHECKS == (
        "receipt_schema",
        "config",
        "protocol",
        "clustering_contract",
        "implementation_inventory",
        "dependency_inventory",
        "runtime_inventory",
        "query_input",
        "embedding",
        "similarity",
        "sample_fixture",
        "sample_indices",
        "category_counts",
        "fit_count",
        "start_count",
        "fit_supports",
        "fit_memberships",
        "fit_centers",
        "target_assignments",
        "manifests",
        "fresh_distinct_roots",
        "finite_timings",
        "exclusive_resource_attestation",
    )


def test_official_sample_indices_match_independent_numpy_fixture() -> None:
    fixture = json.loads(SAMPLE_FIXTURE.read_text(encoding="utf-8"))
    assert hashlib.sha256(canonical_json(fixture)).hexdigest() == SAMPLE_FIXTURE_SHA256
    pair_indices, single_indices = official_sample_indices()
    assert pair_indices.dtype == np.dtype("int64")
    assert single_indices.dtype == np.dtype("int64")
    assert pair_indices.tolist() == fixture["pair_indices"]
    assert single_indices.tolist() == fixture["single_indices"]
    assert len(pair_indices) == 256 == len(set(map(int, pair_indices)))
    assert len(single_indices) == 32 == len(set(map(int, single_indices)))
    assert np.all(pair_indices[:-1] < pair_indices[1:])
    assert np.all(single_indices[:-1] < single_indices[1:])
    pairs = tuple(itertools.combinations(range(322), 2))
    assert len(pairs) == 51_681
    assert pairs[0] == (0, 1)
    assert pairs[-1] == (320, 321)
    assert tuple(pairs[int(index)] for index in pair_indices[:3]) == (
        (0, 93),
        (0, 214),
        (2, 22),
    )


def test_linear_p95_uses_locked_interpolation() -> None:
    assert _linear_p95([4.0]) == 4.0
    assert _linear_p95([4.0, 1.0, 3.0, 2.0]) == pytest.approx(3.85)
    with pytest.raises(ValueError, match="at least one"):
        _linear_p95([])


def test_timing_summaries_are_separate_and_exact() -> None:
    rows: list[dict[str, object]] = []
    values = {
        "pair_c20": ([1.0, 2.0, 7.0, 10.0], [1, 2, 3, 4]),
        "pair_c40": ([2.0, 4.0], [4, 6]),
        "single_c20": ([3.0, 9.0], [2, 8]),
        "single_c40": ([5.0, 6.0, 8.0], [7, 8, 9]),
    }
    for category, (seconds, iterations) in values.items():
        kind, cluster = category.split("_c")
        for index, (elapsed, n_iter) in enumerate(
            zip(seconds, iterations, strict=True)
        ):
            rows.append(
                {
                    "exclusion_kind": kind,
                    "inventory_index": index,
                    "cluster_count": int(cluster),
                    "elapsed_ns": int(elapsed * 1_000_000_000),
                    "n_iter": n_iter,
                }
            )
    summaries = _summarize_timing_rows(rows)
    assert tuple(summaries) == CATEGORIES
    for category, (seconds, iterations) in values.items():
        summary = summaries[category]
        assert summary.fit_count == len(seconds)
        assert summary.mean_seconds == pytest.approx(math.fsum(seconds) / len(seconds))
        assert summary.median_seconds == pytest.approx(float(np.median(seconds)))
        assert summary.p95_seconds == pytest.approx(_linear_p95(seconds))
        assert summary.maximum_seconds == max(seconds)
        assert summary.n_iter_min == min(iterations)
        assert summary.n_iter_mean == pytest.approx(
            math.fsum(iterations) / len(iterations)
        )
        assert summary.n_iter_max == max(iterations)
        assert summary.n_iter_values_sha256 == framed_sha256(
            "feasibility-n-iter-values", canonical_json(iterations)
        )


def test_receipt_round_trip_requires_exact_keys_and_provenance(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path)
    payload = _receipt_to_dict(receipt)
    assert _receipt_from_dict(payload) == receipt
    for mutation in (
        {**payload, "unknown": 1},
        {key: value for key, value in payload.items() if key != "fit_count"},
        {
            **payload,
            "provenance": {**payload["provenance"], "unknown": "x"},
        },
        {
            **payload,
            "provenance": {
                key: value
                for key, value in payload["provenance"].items()
                if key != "config_sha256"
            },
        },
        {
            **payload,
            "provenance": {**payload["provenance"], "config_sha256": "bad"},
        },
        {
            **payload,
            "provenance": {**payload["provenance"], "worker_pid": True},
        },
        {
            **payload,
            "provenance": {**payload["provenance"], "thread_environment": []},
        },
    ):
        with pytest.raises(ValueError):
            _receipt_from_dict(mutation)


def test_repeat_integrity_executes_checks_once_in_declared_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repeat_a = _receipt(tmp_path, "a")
    repeat_b = _receipt(tmp_path, "b")
    observed: list[str] = []

    def check(name: str, _a: FeasibilityReceipt, _b: FeasibilityReceipt) -> bool:
        observed.append(name)
        return name not in {"protocol", "fit_centers"}

    monkeypatch.setattr(feasibility, "_repeat_check", check)
    assert _repeat_integrity_failures(repeat_a, repeat_b) == (
        "feasibility repeat mismatch: protocol",
        "feasibility repeat mismatch: fit_centers",
    )
    assert observed == list(FEASIBILITY_REPEAT_CHECKS)


def test_projection_formula_and_larger_repeat_are_exact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(feasibility, "_repeat_integrity_failures", lambda *_: ())
    repeat_a = _receipt(tmp_path, "a")
    repeat_b = replace(
        _receipt(tmp_path, "b"),
        timing_summary={name: _summary(0.02) for name in CATEGORIES},
    )
    expected_a = 51_681 * 0.02 + 322 * 0.02
    expected_b = 51_681 * 0.04 + 322 * 0.04
    assert _projection_seconds(repeat_a.timing_summary) == pytest.approx(expected_a)
    gate = evaluate_feasibility_repeats(
        repeat_a,
        repeat_b,
        repeat_a_rss_bytes=100,
        repeat_b_rss_bytes=200,
        exclusive_resource_attested=True,
    )
    assert gate.projection_seconds == pytest.approx(expected_b)
    assert gate.maximum_rss_bytes == 200


def test_gate_thresholds_are_inclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(feasibility, "_repeat_integrity_failures", lambda *_: ())
    valid_repeat = _receipt(tmp_path)
    summaries = valid_repeat.timing_summary
    exact = replace(
        valid_repeat,
        timing_summary={
            "pair_c20": replace(
                summaries["pair_c20"], mean_seconds=86_400 / (2 * 51_681)
            ),
            "pair_c40": replace(
                summaries["pair_c40"], mean_seconds=86_400 / (2 * 51_681)
            ),
            "single_c20": replace(summaries["single_c20"], mean_seconds=0.0),
            "single_c40": replace(summaries["single_c40"], mean_seconds=0.0),
        },
    )
    gate = evaluate_feasibility_repeats(
        exact,
        exact,
        repeat_a_rss_bytes=16 * 1024**3,
        repeat_b_rss_bytes=16 * 1024**3,
        exclusive_resource_attested=True,
    )
    assert gate.status == "pass"
    too_large = replace(
        exact,
        timing_summary={
            **exact.timing_summary,
            "single_c40": replace(
                exact.timing_summary["single_c40"],
                mean_seconds=float.fromhex("0x1p-30"),
            ),
        },
    )
    assert (
        evaluate_feasibility_repeats(
            too_large,
            exact,
            repeat_a_rss_bytes=1,
            repeat_b_rss_bytes=1,
            exclusive_resource_attested=True,
        ).status
        == "stop"
    )
    assert (
        evaluate_feasibility_repeats(
            exact,
            exact,
            repeat_a_rss_bytes=16 * 1024**3 + 1,
            repeat_b_rss_bytes=1,
            exclusive_resource_attested=True,
        ).status
        == "stop"
    )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("fit_count", 575, "exactly 576 fits"),
        ("start_count", 5_759, "exactly 5760 starts"),
        ("exclusive_resource_attested", False, "resources were not attested"),
    ),
)
def test_every_gate_failure_stops_without_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    monkeypatch.setattr(feasibility, "_repeat_integrity_failures", lambda *_: ())
    receipt = replace(_receipt(tmp_path), **{field: value})
    gate = evaluate_feasibility_repeats(
        receipt,
        receipt,
        repeat_a_rss_bytes=1,
        repeat_b_rss_bytes=1,
        exclusive_resource_attested=True,
    )
    assert gate.status == "stop"
    assert any(message in failure for failure in gate.failures)
    assert not hasattr(gate, "fallback")


def test_gnu_time_verbose_parses_one_c_locale_kib_field() -> None:
    assert parse_gnu_time_verbose(TIME_FIXTURE) == 123_456 * 1024


@pytest.mark.parametrize(
    "payload",
    (
        "Maximum resident set size (KiB): 1\n",
        "Maximum resident set size (kbytes): nope\n",
        "Maximum resident set size (kbytes): 1\n"
        "Maximum resident set size (kbytes): 2\n",
    ),
)
def test_gnu_time_verbose_rejects_missing_ambiguous_or_noninteger(
    tmp_path: Path, payload: str
) -> None:
    source = tmp_path / "time.txt"
    source.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="unique RSS KiB"):
        parse_gnu_time_verbose(source)


def test_bounded_reader_uses_nonblocking_descriptor_when_regular_file_becomes_fifo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    replacement = tmp_path / "replacement.fifo"
    source.write_bytes(b"{}\n")
    os.mkfifo(replacement)
    original_open = os.open
    observed_flags: list[int] = []

    def swapped_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if Path(path) == source:  # type: ignore[arg-type]
            observed_flags.append(flags)
            os.replace(replacement, source)
        return original_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", swapped_open)

    with pytest.raises(ValueError, match="changed|regular"):
        feasibility._read_bounded_regular_file(
            source,
            byte_limit=128,
            label="synthetic bounded input",
        )
    assert observed_flags
    assert observed_flags[0] & os.O_NONBLOCK
    assert observed_flags[0] & os.O_NOFOLLOW


def test_bounded_reader_rejects_same_size_inode_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.json"
    replacement = tmp_path / "replacement.json"
    source.write_bytes(b'{"a":1}\n')
    replacement.write_bytes(b'{"b":2}\n')
    original_open = os.open

    def swapped_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if Path(path) == source:  # type: ignore[arg-type]
            os.replace(replacement, source)
        return original_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", swapped_open)

    with pytest.raises(ValueError, match="changed while opening"):
        feasibility._read_bounded_regular_file(
            source,
            byte_limit=128,
            label="synthetic bounded input",
        )


def test_bounded_canonical_json_reader_rejects_same_size_inode_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "evidence.json"
    replacement = tmp_path / "replacement.json"
    source.write_bytes(canonical_json({"value": "a"}) + b"\n")
    replacement.write_bytes(canonical_json({"value": "b"}) + b"\n")
    assert source.stat().st_size == replacement.stat().st_size
    original_open = os.open

    def swapped_open(
        path: object,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        if Path(path) == source:  # type: ignore[arg-type]
            os.replace(replacement, source)
        return original_open(path, flags, mode, dir_fd=dir_fd)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "open", swapped_open)

    with pytest.raises(ValueError, match="changed while opening"):
        feasibility._read_bounded_canonical_json(
            source,
            label="synthetic evidence",
        )


def _synthetic_worker_config(tmp_path: Path) -> FeasibilityConfig:
    repo_root = tmp_path / "repo"
    config_path = repo_root / "configs/feasibility.json"
    query_path = repo_root / "data/queries.tsv"
    embedding_path = repo_root / "artifacts/frozen/embedding.npy"
    metadata_path = repo_root / "artifacts/frozen/metadata.json"
    protocol_path = repo_root / "protocols/protocol.md"
    for parent in {
        config_path.parent,
        query_path.parent,
        embedding_path.parent,
        protocol_path.parent,
    }:
        parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(b'{"schema":"synthetic"}\n')
    query_path.write_text("q000\tSynthetic\n", encoding="utf-8")
    embedding_path.write_bytes(b"synthetic embedding")
    metadata_path.write_text("{}\n", encoding="utf-8")
    protocol_path.write_text("# Synthetic locked protocol\n", encoding="utf-8")
    return FeasibilityConfig(
        repo_root=repo_root.resolve(),
        source_path=config_path.resolve(),
        source_sha256=raw_file_sha256(config_path),
        query_path=query_path.resolve(),
        embedding_path=embedding_path.resolve(),
        encoder_metadata_path=metadata_path.resolve(),
        protocol_path=protocol_path.resolve(),
    )


def _valid_config_value() -> dict[str, object]:
    return {
        "schema": "trec-mlm-dense-aggregation-feasibility-config-v1",
        "query_path": "data/dev/llm_dev_data.tsv",
        "embedding_path": "artifacts/frozen_dense_input/mpnet_base_v2.npy",
        "encoder_metadata_path": (
            "artifacts/frozen_dense_input/encoder_metadata.json"
        ),
        "protocol_path": "protocols/trec_2026_dense_aggregation.md",
    }


def test_load_feasibility_config_rejects_noncanonical_json(tmp_path: Path) -> None:
    path = tmp_path / "configs/feasibility.json"
    path.parent.mkdir()
    path.write_text(json.dumps(_valid_config_value(), indent=2) + "\n")
    with pytest.raises(ValueError, match="canonical JSON bytes"):
        feasibility.load_feasibility_config(path)


def test_worker_preflight_rejection_precedes_every_other_worker_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_worker_config(tmp_path)
    observed: list[str] = []

    def reject_preflight(*, fit_work_file: Path, output_dir: Path) -> None:
        del fit_work_file, output_dir
        observed.append("preflight")
        raise OSError("fit work file and output must share one filesystem")

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        observed.append("forbidden")
        raise AssertionError("worker passed preflight failure")

    monkeypatch.setattr(
        feasibility, "preflight_atomic_paths", reject_preflight, raising=False
    )
    for name in (
        "raw_file_sha256",
        "load_and_validate_frozen_dense_inputs",
        "build_clustering_contract",
        "capture_numerical_environment",
    ):
        monkeypatch.setattr(feasibility, name, forbidden, raising=False)

    with pytest.raises(OSError, match="share one filesystem"):
        feasibility.run_feasibility_worker(
            config,
            fit_work_file=tmp_path / "fits.partial.h5",
            output_dir=tmp_path / "output",
            exclusive_resource_attested=True,
        )
    assert observed == ["preflight"]


def _synthetic_fit_and_assignments(
    *,
    query_ids: tuple[str, ...],
    excluded: tuple[int, ...],
    cluster_count: int,
    embedding_sha256: str,
    clustering_contract_sha256: str,
    center_offset: float = 0.0,
) -> tuple[CanonicalClusterFit, tuple[TargetAssignment, ...]]:
    support_indices = tuple(
        index for index in range(len(query_ids)) if index not in excluded
    )
    support_ids = tuple(query_ids[index] for index in support_indices)
    labels = np.asarray(
        [index % cluster_count for index in range(len(support_ids))], dtype="<u2"
    )
    centers = np.asarray(
        [
            [cluster_id + 0.25 + center_offset, cluster_id + 0.75]
            for cluster_id in range(cluster_count)
        ],
        dtype="<f4",
    )
    memberships = tuple(
        tuple(
            query_id
            for query_id, label in zip(support_ids, labels, strict=True)
            if int(label) == cluster_id
        )
        for cluster_id in range(cluster_count)
    )
    cluster_inertia = tuple(float(index + 1) for index in range(cluster_count))
    total = math.fsum(cluster_inertia)
    fit = CanonicalClusterFit(
        key=FitKey(
            support_ids,
            cluster_count,
            embedding_sha256,
            clustering_contract_sha256,
        ),
        centers=centers,
        labels=labels,
        memberships=memberships,
        fit_membership_sha256=canonical_membership_sha256(
            tuple(enumerate(memberships))
        ),
        cluster_membership_sha256=tuple(
            canonical_membership_sha256(((cluster_id, membership),))
            for cluster_id, membership in enumerate(memberships)
        ),
        center_sha256=logical_array_sha256(
            centers,
            tag="canonical-centers",
            dtype="<f4",
            row_inventory=tuple(range(cluster_count)),
            column_inventory=(),
        ),
        cluster_inertia=cluster_inertia,
        sklearn_inertia=total,
        recomputed_total_inertia=total,
        n_iter=2 if cluster_count == 20 else 4,
    )
    support_index_by_id = dict(zip(support_ids, support_indices, strict=True))
    assignments = tuple(
        TargetAssignment(
            query_id=query_ids[target],
            canonical_cluster_id=target % cluster_count,
            squared_distance=float(target) + 0.5,
            member_indices=tuple(
                support_index_by_id[query_id]
                for query_id in memberships[target % cluster_count]
            ),
            cluster_support=len(memberships[target % cluster_count]),
        )
        for target in excluded
    )
    return fit, assignments


def test_worker_runs_exact_inventory_timing_window_and_publishes_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_worker_config(tmp_path)
    query_ids = tuple(f"q{index:03d}" for index in range(322))
    query_texts = tuple(f"Synthetic query {index}" for index in range(322))
    inputs = FrozenDenseInputs(
        query_ids=query_ids,
        query_texts=query_texts,
        embeddings=np.zeros((322, 2), dtype=np.float32),
        similarities=np.zeros((322, 322), dtype=np.float32),
        query_input_sha256="a" * 64,
        embedding_sha256="b" * 64,
        similarity_sha256="c" * 64,
    )
    thread_environment = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "BLIS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    numerical_receipt = NumericalEnvironmentReceipt(
        thread_environment=thread_environment,
        threadpool_info=(),
        threadpool_info_sha256="d" * 64,
    )
    environment_bytes = numerical_environment_file_bytes(numerical_receipt)
    environment_sha256 = hashlib.sha256(environment_bytes).hexdigest()
    clustering_contract = {
        "versions": {
            "python": "3.13.9",
            "numpy": "2.5.1",
            "scipy": "1.18.0",
            "scikit_learn": "1.9.0",
        },
        "threads": {
            **thread_environment,
            "threadpool_limit": 1,
            "process_count": 1,
        },
    }
    clustering_contract_sha256 = "e" * 64
    implementation_sha256 = "f" * 64
    dependency_sha256 = "0" * 64
    runtime_sha256 = "1" * 64
    events: list[tuple[str, object]] = []
    environment_file_checks: list[tuple[Path, str]] = []
    clock_calls = 0
    full_completion_reads = {20: 0, 40: 0}
    write_completion_read_deltas: list[int] = []

    real_preflight = feasibility.preflight_atomic_paths
    real_initialize_fit_datasets = feasibility.initialize_fit_datasets
    real_write_fit_row = feasibility.write_fit_row
    real_dataset_getitem = h5py.Dataset.__getitem__

    def preflight(*, fit_work_file: Path, output_dir: Path) -> None:
        events.append(("preflight", (fit_work_file, output_dir)))
        real_preflight(fit_work_file=fit_work_file, output_dir=output_dir)

    def load_inputs(candidate: FeasibilityConfig) -> FrozenDenseInputs:
        assert candidate is config
        events.append(("load_inputs", None))
        return inputs

    def build_contract(
        candidate: FrozenDenseInputs, *, repo_root: Path, protocol_path: Path
    ) -> tuple[dict[str, object], str]:
        assert candidate is inputs
        assert repo_root == config.repo_root
        assert protocol_path == config.protocol_path
        events.append(("build_contract", None))
        return clustering_contract, clustering_contract_sha256

    def capture_environment() -> NumericalEnvironmentReceipt:
        events.append(("capture_environment", None))
        return numerical_receipt

    def check_environment(expected: NumericalEnvironmentReceipt) -> None:
        assert expected is numerical_receipt
        events.append(("environment_check", None))

    def check_environment_file(path: Path, expected_sha256: str) -> None:
        assert raw_file_sha256(path) == expected_sha256
        events.append(("environment_file_check", None))
        environment_file_checks.append((path, expected_sha256))

    def initialize_fit_datasets(*args: object, **kwargs: object) -> None:
        events.append(("hdf5_initialize", None))
        real_initialize_fit_datasets(*args, **kwargs)

    def track_completion_reads(dataset: h5py.Dataset, index: object) -> object:
        if dataset.name in ("/fits/c20/complete", "/fits/c40/complete") and (
            index is Ellipsis
        ):
            cluster_count = int(dataset.parent.name.removeprefix("/fits/c"))
            full_completion_reads[cluster_count] += 1
        return real_dataset_getitem(dataset, index)

    def tracked_write_fit_row(*args: object, **kwargs: object) -> None:
        before = sum(full_completion_reads.values())
        real_write_fit_row(*args, **kwargs)
        write_completion_read_deltas.append(
            sum(full_completion_reads.values()) - before
        )

    def perf_counter_ns() -> int:
        nonlocal clock_calls
        call_index = clock_calls
        clock_calls += 1
        events.append(("clock", call_index))
        return call_index * 1_000_000

    def fit_clusters(
        *,
        embeddings: np.ndarray[object, object],
        query_ids: tuple[str, ...],
        support_indices: tuple[int, ...],
        cluster_count: int,
        embedding_sha256: str,
        clustering_contract_sha256: str,
    ) -> CanonicalClusterFit:
        del embeddings
        assert query_ids == inputs.query_ids
        assert embedding_sha256 == inputs.embedding_sha256
        assert clustering_contract_sha256 == "e" * 64
        excluded = tuple(index for index in range(322) if index not in support_indices)
        fit, _assignments = _synthetic_fit_and_assignments(
            query_ids=inputs.query_ids,
            excluded=excluded,
            cluster_count=cluster_count,
            embedding_sha256=inputs.embedding_sha256,
            clustering_contract_sha256="e" * 64,
        )
        events.append(("fit", cluster_count))
        return fit

    def assign(
        *,
        fit: object,
        embeddings: np.ndarray[object, object],
        query_ids: tuple[str, ...],
        target_indices: tuple[int, ...],
        support_index_by_id: dict[str, int],
    ) -> tuple[TargetAssignment, ...]:
        del embeddings
        assert query_ids == inputs.query_ids
        assert set(support_index_by_id) == set(fit.key.support_query_ids)
        assert all(
            inputs.query_ids[index] == query_id
            for query_id, index in support_index_by_id.items()
        )
        assert set(target_indices).isdisjoint(support_index_by_id.values())
        excluded = tuple(target_indices)
        _synthetic_fit, assignments = _synthetic_fit_and_assignments(
            query_ids=inputs.query_ids,
            excluded=excluded,
            cluster_count=fit.key.cluster_count,
            embedding_sha256=inputs.embedding_sha256,
            clustering_contract_sha256="e" * 64,
        )
        events.append(("assign", target_indices))
        return assignments

    monkeypatch.setattr(feasibility, "preflight_atomic_paths", preflight)
    monkeypatch.setattr(
        feasibility, "load_and_validate_frozen_dense_inputs", load_inputs, raising=False
    )
    monkeypatch.setattr(
        feasibility, "build_clustering_contract", build_contract, raising=False
    )
    monkeypatch.setattr(
        feasibility, "capture_numerical_environment", capture_environment, raising=False
    )
    monkeypatch.setattr(
        feasibility,
        "require_same_numerical_environment",
        check_environment,
        raising=False,
    )
    monkeypatch.setattr(
        feasibility,
        "_verify_environment_file",
        check_environment_file,
        raising=False,
    )
    monkeypatch.setattr(
        feasibility,
        "initialize_fit_datasets",
        initialize_fit_datasets,
    )
    monkeypatch.setattr(feasibility, "write_fit_row", tracked_write_fit_row)
    monkeypatch.setattr(h5py.Dataset, "__getitem__", track_completion_reads)
    monkeypatch.setattr(
        feasibility, "fit_canonical_clusters", fit_clusters, raising=False
    )
    monkeypatch.setattr(feasibility, "assign_targets", assign, raising=False)
    monkeypatch.setattr(
        feasibility.time, "perf_counter_ns", perf_counter_ns, raising=False
    )
    monkeypatch.setattr(
        feasibility,
        "contract_source_inventory_sha256",
        lambda contract, *, category: (
            implementation_sha256 if category == "implementation" else dependency_sha256
        ),
        raising=False,
    )
    monkeypatch.setattr(
        feasibility,
        "runtime_source_inventory_sha256",
        lambda repo_root: runtime_sha256,
        raising=False,
    )

    fit_work_file = tmp_path / "worker/fits.partial.h5"
    output_dir = tmp_path / "worker/output"
    receipt = feasibility.run_feasibility_worker(
        config,
        fit_work_file=fit_work_file,
        output_dir=output_dir,
        exclusive_resource_attested=True,
    )

    assert events[0][0] == "preflight"
    assert [event[0] for event in events[:4]] == [
        "preflight",
        "load_inputs",
        "build_contract",
        "capture_environment",
    ]
    first_fit = next(
        index for index, (name, _value) in enumerate(events) if name == "fit"
    )
    pre_fit_attestation = next(
        index
        for index, (name, _value) in enumerate(events)
        if name == "environment_check"
    )
    assert [name for name, _value in events[pre_fit_attestation - 1 : first_fit]] == [
        "hdf5_initialize",
        "environment_check",
        "environment_file_check",
        "clock",
    ]
    assert all(
        name != "hdf5_initialize"
        for name, _value in events[pre_fit_attestation + 1 : first_fit]
    )
    assert clock_calls == 1_728
    assert receipt.fit_count == 576
    assert receipt.start_count == 5_760
    assert write_completion_read_deltas == [0] * 576
    assert full_completion_reads == {20: 1, 40: 1}
    fixture = json.loads(SAMPLE_FIXTURE.read_text(encoding="utf-8"))
    assert receipt.pair_indices == tuple(fixture["pair_indices"])
    assert receipt.single_indices == tuple(fixture["single_indices"])
    for category, count in (
        ("pair_c20", 256),
        ("pair_c40", 256),
        ("single_c20", 32),
        ("single_c40", 32),
    ):
        assert receipt.timing_summary[category].fit_count == count
    assert receipt.fit_file_sha256 == raw_file_sha256(output_dir / "fits.h5")
    assert len(receipt.fit_evidence_sha256) == 64
    assert receipt.provenance["fit_work_file"] == str(fit_work_file.resolve())
    assert receipt.provenance["output_root"] == str(output_dir.resolve())
    assert receipt.provenance["numerical_environment_sha256"] == environment_sha256
    assert receipt.provenance["sample_fixture_sha256"] == SAMPLE_FIXTURE_SHA256
    assert receipt.provenance["worker_pid"] == os.getpid()
    assert (output_dir / "environment.json").read_bytes() == environment_bytes
    assert (
        _receipt_from_dict(
            json.loads((output_dir / "feasibility_receipt.json").read_bytes())
        )
        == receipt
    )
    validate_manifest(output_dir)
    assert not fit_work_file.exists()
    with h5py.File(output_dir / "fits.h5", "r") as h5:
        assert int(h5.attrs["complete"]) == 1
        first = feasibility.read_fit_row(h5, 0, 20)
        last = feasibility.read_fit_row(h5, 287, 40)
    assert first.excluded_query_ids == (query_ids[0], query_ids[93])
    assert last.excluded_query_ids == (query_ids[321],)
    assert [name for name, _ in events].count("environment_check") == 2
    assert len(environment_file_checks) == 2
    assert {expected for _path, expected in environment_file_checks} == {
        environment_sha256
    }


_REPEAT_FIT_TEMPLATE: tuple[
    bytes,
    dict[str, TimingCategorySummary],
    str,
] | None = None
_REPEAT_EVIDENCE_TEMPLATE: object | None = None


def _publish_repeat_fixture(
    tmp_path: Path,
    worker: str,
    *,
    changed_center: bool = False,
) -> FeasibilityReceipt:
    global _REPEAT_FIT_TEMPLATE
    output_root = (tmp_path / f"repeat-{worker}").resolve()
    output_root.mkdir(parents=True)
    pair_indices, single_indices = official_sample_indices()
    provenance = _provenance(tmp_path, worker)
    environment_bytes = canonical_json({"synthetic": "environment"}) + b"\n"
    (output_root / "environment.json").write_bytes(environment_bytes)
    provenance.update(
        {
            "output_root": str(output_root),
            "sample_fixture_sha256": SAMPLE_FIXTURE_SHA256,
            "numerical_environment_sha256": hashlib.sha256(
                environment_bytes
            ).hexdigest(),
        }
    )
    query_ids = tuple(f"q{index:03d}" for index in range(322))
    inventory = feasibility._worker_inventory(pair_indices, single_indices)
    metadata = feasibility._fit_metadata(
        query_ids=query_ids,
        query_input_sha256=provenance["query_input_sha256"],
        embedding_sha256=provenance["embedding_sha256"],
        similarity_sha256=provenance["similarity_sha256"],
        clustering_contract_sha256=provenance["clustering_contract_sha256"],
        implementation_inventory_sha256=provenance[
            "implementation_inventory_sha256"
        ],
        dependency_inventory_sha256=provenance["dependency_inventory_sha256"],
        runtime_inventory_sha256=provenance["runtime_inventory_sha256"],
        numerical_environment_sha256=provenance["numerical_environment_sha256"],
        protocol_sha256=provenance["protocol_sha256"],
    )
    timing_rows: list[dict[str, object]] = []
    fits_path = output_root / "fits.h5"
    if _REPEAT_FIT_TEMPLATE is not None and not changed_center:
        fit_bytes, timing_summary, fit_evidence_sha256 = _REPEAT_FIT_TEMPLATE
        fits_path.write_bytes(fit_bytes)
    else:
        with h5py.File(fits_path, "x") as h5:
            feasibility.initialize_fit_datasets(
                h5, metadata=metadata, inventory=inventory, embedding_width=2
            )
            for row_index, row in enumerate(inventory):
                for cluster_count in (20, 40):
                    fit, assignments = _synthetic_fit_and_assignments(
                        query_ids=query_ids,
                        excluded=row.excluded_query_indices,
                        cluster_count=cluster_count,
                        embedding_sha256=provenance["embedding_sha256"],
                        clustering_contract_sha256=provenance[
                            "clustering_contract_sha256"
                        ],
                        center_offset=(
                            1.0
                            if changed_center and row_index == 0 and cluster_count == 40
                            else 0.0
                        ),
                    )
                    feasibility.write_fit_row(
                        h5,
                        row_index,
                        fit,
                        assignments,
                        started_ns=time.perf_counter_ns(),
                    )
                    timing_rows.append(
                        {
                            "exclusion_kind": row.exclusion_kind,
                            "cluster_count": cluster_count,
                            "elapsed_ns": int(
                                h5[f"fits/c{cluster_count}/elapsed_ns"][row_index]
                            ),
                            "n_iter": fit.n_iter,
                        }
                    )
            h5.attrs.modify("complete", np.uint8(1))
        timing_summary = _summarize_timing_rows(timing_rows)
        fit_evidence_sha256 = feasibility._fit_evidence_sha256(
            fits_path, row_count=len(inventory)
        )
        if not changed_center:
            _REPEAT_FIT_TEMPLATE = (
                fits_path.read_bytes(),
                dict(timing_summary),
                fit_evidence_sha256,
            )
    receipt = FeasibilityReceipt(
        fit_count=576,
        start_count=5_760,
        pair_indices=tuple(map(int, pair_indices)),
        single_indices=tuple(map(int, single_indices)),
        timing_summary=timing_summary,
        fit_file_sha256=raw_file_sha256(fits_path),
        fit_evidence_sha256=fit_evidence_sha256,
        provenance=provenance,
        exclusive_resource_attested=True,
    )
    (output_root / "feasibility_receipt.json").write_bytes(
        canonical_json(_receipt_to_dict(receipt)) + b"\n"
    )
    artifacts.write_manifest(output_root)
    return receipt


def _rewrite_repeat_receipt(receipt: FeasibilityReceipt) -> None:
    output_root = Path(receipt.provenance["output_root"])
    (output_root / "feasibility_receipt.json").write_bytes(
        canonical_json(_receipt_to_dict(receipt)) + b"\n"
    )
    (output_root / "manifest.json").unlink()
    artifacts.write_manifest(output_root)


def _gate_value(gate: FeasibilityGate) -> dict[str, object]:
    return {
        "schema": gate.schema,
        "status": gate.status,
        "failures": list(gate.failures),
        "projection_seconds": gate.projection_seconds,
        "maximum_rss_bytes": gate.maximum_rss_bytes,
    }


def _task7_config_sha256(
    *,
    repo_root: Path,
    query_path: Path,
    embedding_path: Path,
    encoder_metadata_path: Path,
    protocol_path: Path,
) -> str:
    payload = {
        "schema": "trec-mlm-dense-aggregation-feasibility-config-v1",
        "query_path": query_path.relative_to(repo_root).as_posix(),
        "embedding_path": embedding_path.relative_to(repo_root).as_posix(),
        "encoder_metadata_path": encoder_metadata_path.relative_to(
            repo_root
        ).as_posix(),
        "protocol_path": protocol_path.relative_to(repo_root).as_posix(),
    }
    return hashlib.sha256(canonical_json(payload) + b"\n").hexdigest()


def _publish_validated_feasibility_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    global _REPEAT_EVIDENCE_TEMPLATE
    repo_root = (tmp_path / "repo").resolve()
    query_path = repo_root / "data/dev/llm_dev_data.tsv"
    embedding_path = repo_root / "artifacts/frozen_dense_input/mpnet_base_v2.npy"
    encoder_metadata_path = (
        repo_root / "artifacts/frozen_dense_input/encoder_metadata.json"
    )
    protocol_path = repo_root / "protocols/trec_2026_dense_aggregation.md"
    for path, payload in (
        (query_path, b"synthetic queries\n"),
        (embedding_path, b"synthetic embeddings\n"),
        (encoder_metadata_path, b"{}\n"),
        (protocol_path, b"synthetic protocol\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    repeat_a = _publish_repeat_fixture(tmp_path, "a")
    repeat_b = _publish_repeat_fixture(tmp_path, "b")
    config_sha256 = _task7_config_sha256(
        repo_root=repo_root,
        query_path=query_path,
        embedding_path=embedding_path,
        encoder_metadata_path=encoder_metadata_path,
        protocol_path=protocol_path,
    )
    repeat_a = replace(
        repeat_a,
        provenance={**repeat_a.provenance, "config_sha256": config_sha256},
    )
    repeat_b = replace(
        repeat_b,
        provenance={**repeat_b.provenance, "config_sha256": config_sha256},
    )
    _rewrite_repeat_receipt(repeat_a)
    _rewrite_repeat_receipt(repeat_b)

    environment_bytes = (
        Path(repeat_a.provenance["output_root"]) / "environment.json"
    ).read_bytes()
    state: dict[str, object] = {
        "protocol_sha256": "1" * 64,
        "clustering_contract_sha256": "1" * 64,
        "implementation_inventory_sha256": "1" * 64,
        "dependency_inventory_sha256": "1" * 64,
        "runtime_inventory_sha256": "1" * 64,
        "query_file_sha256": "1" * 64,
        "query_input_sha256": "1" * 64,
        "embedding_sha256": "1" * 64,
        "similarity_sha256": "1" * 64,
        "environment_bytes": environment_bytes,
        "versions": {
            "python": "3.13.9",
            "numpy": "2.5.1",
            "scipy": "1.18.0",
            "scikit_learn": "1.9.0",
        },
        "thread_environment": dict(repeat_a.provenance["thread_environment"]),
    }
    real_raw_file_sha256 = feasibility.raw_file_sha256
    real_bounded_file_sha256 = feasibility._bounded_regular_file_sha256

    def current_raw_file_sha256(path: str | Path) -> str:
        source = Path(path)
        if source.name == "llm_dev_qrels.txt":
            raise AssertionError("qrel leaf was accessed by qrel-free validation")
        resolved = source.resolve()
        if resolved == query_path:
            return str(state["query_file_sha256"])
        if resolved == protocol_path:
            return str(state["protocol_sha256"])
        return real_raw_file_sha256(source)

    def current_bounded_file_sha256(path: Path, *, byte_limit: int, label: str) -> str:
        source = Path(path)
        if source.name == "llm_dev_qrels.txt":
            raise AssertionError("qrel leaf was accessed by qrel-free validation")
        resolved = source.resolve()
        if resolved == query_path:
            return str(state["query_file_sha256"])
        if resolved == protocol_path:
            return str(state["protocol_sha256"])
        return real_bounded_file_sha256(
            source,
            byte_limit=byte_limit,
            label=label,
        )

    def load_current_inputs(config: object) -> FrozenDenseInputs:
        assert not hasattr(config, "source_path")
        assert config.query_path == query_path
        assert config.embedding_path == embedding_path
        assert config.encoder_metadata_path.is_relative_to(repo_root)
        return FrozenDenseInputs(
            query_ids=(),
            query_texts=(),
            embeddings=np.empty((0, 0), dtype=np.float32),
            similarities=np.empty((0, 0), dtype=np.float32),
            query_input_sha256=str(state["query_input_sha256"]),
            embedding_sha256=str(state["embedding_sha256"]),
            similarity_sha256=str(state["similarity_sha256"]),
        )

    def build_current_contract(
        _inputs: FrozenDenseInputs, *, repo_root: Path, protocol_path: Path
    ) -> tuple[dict[str, object], str]:
        assert repo_root == fixture.repo_root
        assert protocol_path == fixture.protocol_path
        return (
            {"versions": dict(state["versions"])},
            str(state["clustering_contract_sha256"]),
        )

    def source_inventory(_contract: object, *, category: str) -> str:
        return str(state[f"{category}_inventory_sha256"])

    fixture = SimpleNamespace(
        repo_root=repo_root,
        query_path=query_path,
        embedding_path=embedding_path,
        encoder_metadata_path=encoder_metadata_path,
        protocol_path=protocol_path,
        repeat_a=repeat_a,
        repeat_b=repeat_b,
        state=state,
    )
    monkeypatch.setattr(feasibility, "raw_file_sha256", current_raw_file_sha256)
    monkeypatch.setattr(
        feasibility,
        "_bounded_regular_file_sha256",
        current_bounded_file_sha256,
    )
    monkeypatch.setattr(
        feasibility, "load_and_validate_frozen_dense_inputs", load_current_inputs
    )
    monkeypatch.setattr(
        feasibility, "build_clustering_contract", build_current_contract
    )
    monkeypatch.setattr(
        feasibility, "contract_source_inventory_sha256", source_inventory
    )
    monkeypatch.setattr(
        feasibility,
        "runtime_source_inventory_sha256",
        lambda _root: str(state["runtime_inventory_sha256"]),
    )
    monkeypatch.setattr(
        feasibility,
        "capture_numerical_environment",
        lambda: SimpleNamespace(thread_environment=dict(state["thread_environment"])),
    )
    monkeypatch.setattr(
        feasibility,
        "numerical_environment_file_bytes",
        lambda _receipt: bytes(state["environment_bytes"]),
    )
    if _REPEAT_EVIDENCE_TEMPLATE is None:
        _REPEAT_EVIDENCE_TEMPLATE = feasibility._read_repeat_output_evidence(
            Path(repeat_a.provenance["output_root"]), repeat_a
        )
    monkeypatch.setattr(
        feasibility,
        "_read_repeat_output_evidence",
        lambda _output_root, _receipt: _REPEAT_EVIDENCE_TEMPLATE,
    )
    gate = evaluate_feasibility_repeats(
        repeat_a,
        repeat_b,
        repeat_a_rss_bytes=1,
        repeat_b_rss_bytes=2,
        exclusive_resource_attested=True,
    )
    assert gate.status == "pass"
    aggregate_root = (tmp_path / "feasibility-retry-2").resolve()
    aggregate_root.mkdir()
    (aggregate_root / "feasibility_gate.json").write_bytes(
        canonical_json(_gate_value(gate)) + b"\n"
    )
    (aggregate_root / "repeat_a_receipt.json").write_bytes(
        canonical_json(_receipt_to_dict(repeat_a)) + b"\n"
    )
    (aggregate_root / "repeat_b_receipt.json").write_bytes(
        canonical_json(_receipt_to_dict(repeat_b)) + b"\n"
    )
    (aggregate_root / "repeat_measurements.json").write_bytes(
        canonical_json(
            {
                "schema": (
                    "trec-mlm-dense-aggregation-feasibility-repeat-measurements-v1"
                ),
                "repeat_a_manifest_sha256": current_raw_file_sha256(
                    Path(repeat_a.provenance["output_root"]) / "manifest.json"
                ),
                "repeat_b_manifest_sha256": current_raw_file_sha256(
                    Path(repeat_b.provenance["output_root"]) / "manifest.json"
                ),
                "repeat_a_rss_bytes": 1,
                "repeat_b_rss_bytes": 2,
            }
        )
        + b"\n"
    )
    artifacts.write_manifest(aggregate_root)
    fixture.aggregate_root = aggregate_root
    fixture.manifest_path = aggregate_root / "manifest.json"
    fixture.expected_sha256 = current_raw_file_sha256(fixture.manifest_path)
    return fixture


def _validate_fixture(fixture: SimpleNamespace, **overrides: object) -> object:
    arguments = {
        "expected_sha256": fixture.expected_sha256,
        "repo_root": fixture.repo_root,
        "query_path": fixture.query_path,
        "embedding_path": fixture.embedding_path,
        "encoder_metadata_path": fixture.encoder_metadata_path,
        "protocol_path": fixture.protocol_path,
        **overrides,
    }
    return feasibility.validate_feasibility_manifest(
        fixture.manifest_path,
        **arguments,
    )


def _forbid_official_qrel_leaf(monkeypatch: pytest.MonkeyPatch) -> None:
    guarded_name = "llm_dev_qrels.txt"

    def guard(method: object) -> object:
        def guarded(path: Path, *args: object, **kwargs: object) -> object:
            if guarded_name in path.parts:
                raise AssertionError("qrel leaf was accessed by qrel-free validation")
            return method(path, *args, **kwargs)

        return guarded

    for method_name in ("open", "lstat", "stat", "resolve"):
        monkeypatch.setattr(Path, method_name, guard(getattr(Path, method_name)))

    def guards_file_operation(method: object) -> object:
        def guarded(path: object, *args: object, **kwargs: object) -> object:
            if (
                not isinstance(path, int) and guarded_name in Path(path).parts  # type: ignore[arg-type]
            ):
                raise AssertionError("qrel leaf was accessed by qrel-free validation")
            return method(path, *args, **kwargs)  # type: ignore[operator]

        return guarded

    for module, function_name in (
        (os, "open"),
        (os, "lstat"),
        (os, "stat"),
        (builtins, "open"),
        (io, "open"),
    ):
        monkeypatch.setattr(
            module,
            function_name,
            guards_file_operation(getattr(module, function_name)),
        )


def test_repeat_integrity_reopens_hdf5_and_accepts_distinct_manifested_roots(
    tmp_path: Path,
) -> None:
    repeat_a = _publish_repeat_fixture(tmp_path, "a")
    repeat_b = _publish_repeat_fixture(tmp_path, "b")
    assert _repeat_integrity_failures(repeat_a, repeat_b) == ()
    assert repeat_a.fit_evidence_sha256 == repeat_b.fit_evidence_sha256
    assert len(repeat_a.fit_file_sha256) == len(repeat_b.fit_file_sha256) == 64
    paths = tuple(
        Path(receipt.provenance["output_root"]) / "fits.h5"
        for receipt in (repeat_a, repeat_b)
    )
    assert paths[0] != paths[1]
    assert paths[0].stat().st_ino != paths[1].stat().st_ino
    assert all((path.parent / "manifest.json").is_file() for path in paths)


def test_repeat_integrity_reports_center_manifest_runtime_and_root_failures(
    tmp_path: Path,
) -> None:
    repeat_a = _publish_repeat_fixture(tmp_path, "a")
    repeat_b = _publish_repeat_fixture(tmp_path, "b", changed_center=True)
    failures = _repeat_integrity_failures(repeat_a, repeat_b)
    assert "feasibility repeat mismatch: fit_centers" in failures

    drifted = replace(
        repeat_b,
        provenance={**repeat_b.provenance, "numpy_version": "unexpected"},
    )
    failures = _repeat_integrity_failures(repeat_a, drifted)
    assert "feasibility repeat mismatch: runtime_inventory" in failures
    assert "feasibility repeat mismatch: manifests" in failures

    reused = replace(
        repeat_b,
        provenance={
            **repeat_b.provenance,
            "fit_work_file": repeat_a.provenance["fit_work_file"],
        },
    )
    assert (
        "feasibility repeat mismatch: fresh_distinct_roots"
        in _repeat_integrity_failures(repeat_a, reused)
    )

    manifest_path = Path(repeat_b.provenance["output_root"]) / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    assert "feasibility repeat mismatch: manifests" in _repeat_integrity_failures(
        repeat_a, repeat_b
    )


def test_repeat_integrity_rederives_manifested_timing_summaries(
    tmp_path: Path,
) -> None:
    repeat_a = _publish_repeat_fixture(tmp_path, "a")
    repeat_b = _publish_repeat_fixture(tmp_path, "b")
    falsified: list[FeasibilityReceipt] = []
    for receipt in (repeat_a, repeat_b):
        falsified_receipt = replace(
            receipt,
            timing_summary={
                category: replace(summary, mean_seconds=0.0)
                for category, summary in receipt.timing_summary.items()
            },
        )
        _rewrite_repeat_receipt(falsified_receipt)
        falsified.append(falsified_receipt)

    failures = _repeat_integrity_failures(falsified[0], falsified[1])

    assert "feasibility repeat mismatch: finite_timings" in failures


def test_repeat_integrity_binds_hdf5_inventory_to_receipt_sample(
    tmp_path: Path,
) -> None:
    repeat_a = _publish_repeat_fixture(tmp_path, "a")
    repeat_b = _publish_repeat_fixture(tmp_path, "b")
    for receipt in (repeat_a, repeat_b):
        falsified = replace(receipt, pair_indices=tuple(range(256)))
        _rewrite_repeat_receipt(falsified)
        if receipt is repeat_a:
            repeat_a = falsified
        else:
            repeat_b = falsified

    failures = _repeat_integrity_failures(repeat_a, repeat_b)

    assert "feasibility repeat mismatch: sample_indices" in failures


def test_repeat_integrity_binds_hdf5_metadata_hashes_to_receipt_provenance(
    tmp_path: Path,
) -> None:
    repeat_a = _publish_repeat_fixture(tmp_path, "a")
    repeat_b = _publish_repeat_fixture(tmp_path, "b")
    for receipt in (repeat_a, repeat_b):
        falsified = replace(
            receipt,
            provenance={**receipt.provenance, "embedding_sha256": "9" * 64},
        )
        _rewrite_repeat_receipt(falsified)
        if receipt is repeat_a:
            repeat_a = falsified
        else:
            repeat_b = falsified

    failures = _repeat_integrity_failures(repeat_a, repeat_b)

    assert "feasibility repeat mismatch: embedding" in failures


def test_receipt_and_repeat_roots_reject_ancestor_overlap(tmp_path: Path) -> None:
    receipt = _receipt(tmp_path, "a")
    nested = replace(
        receipt,
        provenance={
            **receipt.provenance,
            "fit_work_file": str(
                (tmp_path / "worker/output/fits.partial.h5").resolve()
            ),
            "output_root": str((tmp_path / "worker/output").resolve()),
        },
    )
    with pytest.raises(ValueError, match="overlap"):
        _receipt_from_dict(_receipt_to_dict(nested))

    repeat_a = _receipt(tmp_path, "a")
    repeat_b = replace(
        _receipt(tmp_path, "b"),
        provenance={
            **_receipt(tmp_path, "b").provenance,
            "fit_work_file": str(
                (Path(repeat_a.provenance["output_root"]) / "nested").resolve()
            ),
        },
    )
    assert (
        "feasibility repeat mismatch: fresh_distinct_roots"
        in _repeat_integrity_failures(repeat_a, repeat_b)
    )


def test_validated_feasibility_gate_is_an_exact_frozen_slotted_contract() -> None:
    data_type = feasibility.ValidatedFeasibilityGate
    expected_fields = (
        "manifest_sha256",
        "status",
        "failures",
        "freshness_passed",
    )
    assert dataclasses.is_dataclass(data_type)
    assert data_type.__dataclass_params__.frozen is True
    assert data_type.__slots__ == expected_fields
    assert (
        tuple(field.name for field in dataclasses.fields(data_type)) == expected_fields
    )
    assert get_type_hints(data_type)


def test_validate_feasibility_manifest_accepts_only_a_fresh_passing_retry_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _publish_validated_feasibility_fixture(tmp_path, monkeypatch)

    validated = _validate_fixture(fixture)

    assert validated == feasibility.ValidatedFeasibilityGate(
        manifest_sha256=fixture.expected_sha256,
        status="pass",
        failures=(),
        freshness_passed=True,
    )
    assert fixture.aggregate_root.name == "feasibility-retry-2"
    assert not (fixture.repo_root / "configs/feasibility-retry-2.json").exists()


def test_repeat_integrity_rejects_different_raw_task7_config_hashes(
    tmp_path: Path,
) -> None:
    repeat_a = _receipt(tmp_path, "a")
    repeat_b = replace(
        _receipt(tmp_path, "b"),
        provenance={
            **_receipt(tmp_path, "b").provenance,
            "config_sha256": "4" * 64,
        },
    )

    assert "feasibility repeat mismatch: config" in _repeat_integrity_failures(
        repeat_a,
        repeat_b,
    )


def test_validate_feasibility_manifest_never_touches_the_official_qrel_leaf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _publish_validated_feasibility_fixture(tmp_path, monkeypatch)
    _forbid_official_qrel_leaf(monkeypatch)

    assert _validate_fixture(fixture).freshness_passed is True


@pytest.mark.parametrize(
    "path_field",
    ("query_path", "embedding_path", "encoder_metadata_path", "protocol_path"),
)
def test_validate_feasibility_manifest_rejects_qrel_as_any_frozen_input_without_leaf_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path_field: str,
) -> None:
    fixture = _publish_validated_feasibility_fixture(tmp_path, monkeypatch)
    qrel_path = fixture.repo_root / "data/dev/llm_dev_qrels.txt"
    _forbid_official_qrel_leaf(monkeypatch)

    with pytest.raises(ValueError, match="official .* path"):
        _validate_fixture(fixture, **{path_field: qrel_path})


@pytest.mark.parametrize("argument", ("manifest_path", "repo_root"))
def test_validate_feasibility_manifest_rejects_qrel_as_root_argument_without_leaf_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    argument: str,
) -> None:
    fixture = _publish_validated_feasibility_fixture(tmp_path, monkeypatch)
    qrel_path = fixture.repo_root / "data/dev/llm_dev_qrels.txt"
    _forbid_official_qrel_leaf(monkeypatch)
    arguments = {
        "expected_sha256": fixture.expected_sha256,
        "repo_root": fixture.repo_root,
        "query_path": fixture.query_path,
        "embedding_path": fixture.embedding_path,
        "encoder_metadata_path": fixture.encoder_metadata_path,
        "protocol_path": fixture.protocol_path,
    }
    manifest_path = fixture.manifest_path
    if argument == "manifest_path":
        manifest_path = qrel_path
    else:
        arguments["repo_root"] = qrel_path

    with pytest.raises(ValueError, match="manifest path|repository root"):
        feasibility.validate_feasibility_manifest(manifest_path, **arguments)


@pytest.mark.parametrize("root_field", ("fit_work_file", "output_root"))
def test_validate_feasibility_manifest_rejects_qrel_in_receipt_root_without_leaf_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    root_field: str,
) -> None:
    fixture = _publish_validated_feasibility_fixture(tmp_path, monkeypatch)
    receipt_path = fixture.aggregate_root / "repeat_a_receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    receipt["provenance"][root_field] = str(
        fixture.repo_root / "data/dev/llm_dev_qrels.txt"
    )
    receipt_path.write_bytes(canonical_json(receipt) + b"\n")
    fixture.manifest_path.unlink()
    artifacts.write_manifest(fixture.aggregate_root)
    fixture.expected_sha256 = raw_file_sha256(fixture.manifest_path)
    _forbid_official_qrel_leaf(monkeypatch)

    with pytest.raises(ValueError, match="protected|embargoed"):
        _validate_fixture(fixture)


def test_validate_feasibility_manifest_rejects_expected_digest_gate_and_layout_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _publish_validated_feasibility_fixture(tmp_path, monkeypatch)

    with pytest.raises(ValueError, match="manifest digest"):
        _validate_fixture(fixture, expected_sha256="0" * 64)

    gate_path = fixture.aggregate_root / "feasibility_gate.json"
    gate_value = json.loads(gate_path.read_bytes())
    gate_value["status"] = "stop"
    gate_value["failures"] = ["synthetic stopped gate"]
    gate_path.write_bytes(canonical_json(gate_value) + b"\n")
    (fixture.aggregate_root / "manifest.json").unlink()
    artifacts.write_manifest(fixture.aggregate_root)
    fixture.expected_sha256 = feasibility.raw_file_sha256(fixture.manifest_path)
    with pytest.raises(ValueError, match="stale or stopped"):
        _validate_fixture(fixture)

    gate_value["status"] = "pass"
    gate_value["failures"] = []
    gate_path.write_bytes(canonical_json(gate_value) + b"\n")
    (fixture.aggregate_root / "manifest.json").unlink()
    artifacts.write_manifest(fixture.aggregate_root)
    fixture.expected_sha256 = feasibility.raw_file_sha256(fixture.manifest_path)
    (fixture.aggregate_root / "unexpected.txt").write_bytes(b"unexpected\n")
    with pytest.raises(ValueError, match="physical layout"):
        _validate_fixture(fixture)


def test_validate_feasibility_manifest_rejects_worker_root_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _publish_validated_feasibility_fixture(tmp_path, monkeypatch)
    worker_root = Path(fixture.repeat_b.provenance["output_root"])
    (worker_root / "unexpected.txt").write_bytes(b"unexpected\n")

    with pytest.raises(ValueError, match="physical layout"):
        _validate_fixture(fixture)


def test_validate_feasibility_manifest_rejects_each_current_provenance_class(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _publish_validated_feasibility_fixture(tmp_path, monkeypatch)
    hash_keys = (
        "protocol_sha256",
        "clustering_contract_sha256",
        "implementation_inventory_sha256",
        "dependency_inventory_sha256",
        "runtime_inventory_sha256",
        "query_file_sha256",
        "query_input_sha256",
        "embedding_sha256",
        "similarity_sha256",
    )
    for key in hash_keys:
        original = fixture.state[key]
        fixture.state[key] = "4" * 64
        with pytest.raises(ValueError, match="current inputs"):
            _validate_fixture(fixture)
        fixture.state[key] = original

    original_environment = fixture.state["environment_bytes"]
    fixture.state["environment_bytes"] = b'{"changed":true}\n'
    with pytest.raises(ValueError, match="current inputs"):
        _validate_fixture(fixture)
    fixture.state["environment_bytes"] = original_environment

    original_versions = fixture.state["versions"]
    fixture.state["versions"] = {**original_versions, "python": "unexpected"}
    with pytest.raises(ValueError, match="current inputs"):
        _validate_fixture(fixture)
    fixture.state["versions"] = original_versions

    original_threads = fixture.state["thread_environment"]
    fixture.state["thread_environment"] = {
        **original_threads,
        "OMP_NUM_THREADS": "2",
    }
    with pytest.raises(ValueError, match="current inputs"):
        _validate_fixture(fixture)
    fixture.state["thread_environment"] = original_threads

    alternate_metadata = fixture.repo_root / "artifacts/frozen/retry_metadata.json"
    alternate_metadata.parent.mkdir(parents=True)
    alternate_metadata.write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="official encoder metadata path"):
        _validate_fixture(fixture, encoder_metadata_path=alternate_metadata)


def test_validate_feasibility_manifest_rejects_consistently_wrong_semantic_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _publish_validated_feasibility_fixture(tmp_path, monkeypatch)
    real_task7_config_payload = feasibility._task7_config_payload

    def consistently_wrong_payload(**kwargs: object) -> dict[str, str]:
        payload = real_task7_config_payload(**kwargs)
        return {**payload, "embedding_path": "artifacts/frozen/retry.npy"}

    monkeypatch.setattr(
        feasibility,
        "_task7_config_payload",
        consistently_wrong_payload,
    )

    with pytest.raises(ValueError, match="frozen semantic payload"):
        _validate_fixture(fixture)


@pytest.mark.parametrize(
    "provenance_field",
    (
        "semantic_config_payload",
        "protocol_sha256",
        "clustering_contract_sha256",
        "implementation_inventory_sha256",
        "dependency_inventory_sha256",
        "runtime_inventory_sha256",
        "query_file_sha256",
        "query_input_sha256",
        "embedding_sha256",
        "similarity_sha256",
        "sample_fixture_sha256",
        "numerical_environment_sha256",
        "python_version",
        "numpy_version",
        "scipy_version",
        "scikit_learn_version",
        "thread_environment",
    ),
)
def test_validate_feasibility_manifest_rejects_each_current_provenance_drift_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provenance_field: str,
) -> None:
    fixture = _publish_validated_feasibility_fixture(tmp_path, monkeypatch)
    _forbid_official_qrel_leaf(monkeypatch)
    drifted = False
    capture_count = 0

    real_task7_config_payload = feasibility._task7_config_payload

    def task7_config_payload(**kwargs: object) -> dict[str, str]:
        observed = real_task7_config_payload(**kwargs)
        if drifted and provenance_field == "semantic_config_payload":
            return {**observed, "query_path": "data/dev/drifted.tsv"}
        return observed

    real_official_sample_indices = feasibility.official_sample_indices

    def official_sample_indices() -> tuple[np.ndarray, np.ndarray]:
        pair_indices, single_indices = real_official_sample_indices()
        if drifted and provenance_field == "sample_fixture_sha256":
            pair_indices = pair_indices.copy()
            pair_indices[0] = pair_indices[0] + 1
        return pair_indices, single_indices

    real_current_provenance = feasibility._current_feasibility_provenance

    def current_provenance(**kwargs: object) -> dict[str, object]:
        nonlocal capture_count, drifted
        observed = real_current_provenance(**kwargs)
        capture_count += 1
        if capture_count == 1:
            if provenance_field in {
                "protocol_sha256",
                "clustering_contract_sha256",
                "implementation_inventory_sha256",
                "dependency_inventory_sha256",
                "runtime_inventory_sha256",
                "query_file_sha256",
                "query_input_sha256",
                "embedding_sha256",
                "similarity_sha256",
            }:
                fixture.state[provenance_field] = "4" * 64
            elif provenance_field == "numerical_environment_sha256":
                fixture.state["environment_bytes"] = b'{"changed":true}\n'
            elif provenance_field in {
                "python_version",
                "numpy_version",
                "scipy_version",
                "scikit_learn_version",
            }:
                version_key = {
                    "python_version": "python",
                    "numpy_version": "numpy",
                    "scipy_version": "scipy",
                    "scikit_learn_version": "scikit_learn",
                }[provenance_field]
                fixture.state["versions"] = {
                    **fixture.state["versions"],
                    version_key: "unexpected",
                }
            elif provenance_field == "thread_environment":
                fixture.state["thread_environment"] = {
                    **fixture.state["thread_environment"],
                    "OMP_NUM_THREADS": "2",
                }
            drifted = True
        return observed

    monkeypatch.setattr(feasibility, "_task7_config_payload", task7_config_payload)
    monkeypatch.setattr(feasibility, "official_sample_indices", official_sample_indices)
    monkeypatch.setattr(
        feasibility, "_current_feasibility_provenance", current_provenance
    )

    with pytest.raises(ValueError, match="changed during validation"):
        _validate_fixture(fixture)
    assert capture_count == 2


def test_validate_feasibility_manifest_rejects_aggregate_drift_during_second_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _publish_validated_feasibility_fixture(tmp_path, monkeypatch)
    real_current_provenance = feasibility._current_feasibility_provenance
    capture_count = 0

    def current_provenance(**kwargs: object) -> dict[str, object]:
        nonlocal capture_count
        observed = real_current_provenance(**kwargs)
        capture_count += 1
        if capture_count == 2:
            receipt_path = fixture.aggregate_root / "repeat_a_receipt.json"
            receipt_path.write_bytes(receipt_path.read_bytes() + b" ")
        return observed

    monkeypatch.setattr(
        feasibility,
        "_current_feasibility_provenance",
        current_provenance,
    )

    with pytest.raises(ValueError, match="manifest|changed"):
        _validate_fixture(fixture)
    assert capture_count == 2
