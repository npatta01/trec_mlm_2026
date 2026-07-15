from __future__ import annotations

import dataclasses
from dataclasses import replace
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from typing import get_type_hints
import weakref

import numpy as np
import pytest

import trec_mlm.dense_aggregation_feasibility as feasibility
from trec_mlm import dense_aggregation_artifacts as artifacts
from trec_mlm.dense_aggregation_artifacts import raw_file_sha256, validate_manifest
from trec_mlm.dense_aggregation_clustering import (
    FitKey,
    NumericalEnvironmentReceipt,
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
            "cache_root": str((root / f"{worker}-cache").resolve()),
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
        fit_receipt_sha256="2" * 64,
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
            "fit_receipt_sha256",
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
        "cache_root",
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


def test_worker_preflight_rejection_precedes_every_other_worker_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _synthetic_worker_config(tmp_path)
    observed: list[str] = []

    def reject_preflight(*, cache_dir: Path, output_dir: Path) -> None:
        del cache_dir, output_dir
        observed.append("preflight")
        raise OSError("cache and output must share one filesystem")

    def forbidden(*args: object, **kwargs: object) -> object:
        del args, kwargs
        observed.append("forbidden")
        raise AssertionError("worker passed preflight failure")

    monkeypatch.setattr(
        feasibility, "preflight_atomic_roots", reject_preflight, raising=False
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
            cache_dir=tmp_path / "cache",
            output_dir=tmp_path / "output",
            exclusive_resource_attested=True,
        )
    assert observed == ["preflight"]


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
    cache_headers: list[dict[str, object]] = []
    fit_rows: list[tuple[str, int, int, tuple[str, ...]]] = []
    timing_rows: list[tuple[str, int, int, int, int]] = []
    environment_file_checks: list[tuple[Path, str]] = []
    latest_fit_ref: list[weakref.ReferenceType[object] | None] = [None]
    clock_calls = 0

    real_preflight = artifacts.preflight_atomic_roots

    def preflight(*, cache_dir: Path, output_dir: Path) -> None:
        events.append(("preflight", (cache_dir, output_dir)))
        real_preflight(cache_dir=cache_dir, output_dir=output_dir)

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
        environment_file_checks.append((path, expected_sha256))

    def perf_counter_ns() -> int:
        nonlocal clock_calls
        call_index = clock_calls
        clock_calls += 1
        if call_index % 2 == 0:
            events.append(("start", call_index // 2))
            return call_index * 1_000_000
        assert latest_fit_ref[0] is not None
        assert latest_fit_ref[0]() is None, "fit reference survived end timestamp"
        events.append(("end", call_index // 2))
        return call_index * 1_000_000

    def fit_clusters(
        *,
        embeddings: np.ndarray[object, object],
        query_ids: tuple[str, ...],
        support_indices: tuple[int, ...],
        cluster_count: int,
        embedding_sha256: str,
        clustering_contract_sha256: str,
    ) -> object:
        del embeddings
        assert query_ids == inputs.query_ids
        assert embedding_sha256 == inputs.embedding_sha256
        assert clustering_contract_sha256 == "e" * 64
        support_ids = tuple(inputs.query_ids[index] for index in support_indices)

        class SyntheticFit:
            pass

        fit = SyntheticFit()
        fit.key = FitKey(
            support_query_ids=support_ids,
            cluster_count=cluster_count,
            embedding_sha256=inputs.embedding_sha256,
            clustering_contract_sha256="e" * 64,
        )
        fit.n_iter = 2 if cluster_count == 20 else 4
        latest_fit_ref[0] = weakref.ref(fit)
        events.append(("fit", cluster_count))
        return fit

    def assign(
        *,
        fit: object,
        embeddings: np.ndarray[object, object],
        query_ids: tuple[str, ...],
        target_indices: tuple[int, ...],
        support_index_by_id: dict[str, int],
    ) -> tuple[str, ...]:
        del embeddings
        assert query_ids == inputs.query_ids
        assert set(support_index_by_id) == set(fit.key.support_query_ids)
        assert all(
            inputs.query_ids[index] == query_id
            for query_id, index in support_index_by_id.items()
        )
        assert set(target_indices).isdisjoint(support_index_by_id.values())
        events.append(("assign", target_indices))
        return tuple(inputs.query_ids[index] for index in target_indices)

    class SyntheticCache:
        def __init__(self, root: Path, header: dict[str, object]) -> None:
            self.root = root
            self.header = header

        @classmethod
        def create(cls, root: Path, *, header: dict[str, object]) -> "SyntheticCache":
            assert not root.exists()
            root.mkdir()
            cache_headers.append(header)
            events.append(("cache_create", root))
            return cls(root, header)

        @classmethod
        def open_readonly(cls, root: Path) -> "SyntheticCache":
            assert root.is_dir()
            return cls(root, cache_headers[0])

        def append_fit(
            self,
            fit: object,
            assignments: tuple[str, ...],
            *,
            exclusion_kind: str,
            inventory_index: int,
            excluded_query_ids: tuple[str, ...],
        ) -> None:
            del assignments
            expected = self.header["expected_inventory"][len(fit_rows)]
            assert expected == {
                "exclusion_kind": exclusion_kind,
                "inventory_index": inventory_index,
                "cluster_count": fit.key.cluster_count,
            }
            fit_rows.append(
                (
                    exclusion_kind,
                    inventory_index,
                    fit.key.cluster_count,
                    excluded_query_ids,
                )
            )
            events.append(("append_fit", len(fit_rows) - 1))

        def release(self, key: FitKey) -> None:
            assert key.cluster_count in {20, 40}
            events.append(("release", len(fit_rows) - 1))

        def append_timing(
            self,
            *,
            exclusion_kind: str,
            inventory_index: int,
            cluster_count: int,
            elapsed_ns: int,
            n_iter: int,
        ) -> None:
            timing_rows.append(
                (
                    exclusion_kind,
                    inventory_index,
                    cluster_count,
                    elapsed_ns,
                    n_iter,
                )
            )
            events.append(("append_timing", len(timing_rows) - 1))

        def finalize(self) -> dict[str, object]:
            assert len(fit_rows) == len(timing_rows) == 576
            receipts = b"".join(
                canonical_json(
                    {
                        "sequence": index,
                        "kind": row[0],
                        "inventory_index": row[1],
                        "cluster_count": row[2],
                    }
                )
                + b"\n"
                for index, row in enumerate(fit_rows)
            )
            timings = b"".join(
                canonical_json(
                    {
                        "sequence": index,
                        "exclusion_kind": row[0],
                        "inventory_index": row[1],
                        "cluster_count": row[2],
                        "elapsed_ns": row[3],
                        "n_iter": row[4],
                    }
                )
                + b"\n"
                for index, row in enumerate(timing_rows)
            )
            (self.root / "fit_receipts.jsonl").write_bytes(receipts)
            (self.root / "timings.jsonl").write_bytes(timings)
            (self.root / "centers.f4").write_bytes(b"")
            (self.root / "labels.u2").write_bytes(b"")
            (self.root / "cluster_inertia.f8").write_bytes(b"")
            (self.root / "header.json").write_bytes(canonical_json(self.header) + b"\n")
            events.append(("finalize", None))
            return self.header

        def finalize_into(self, destination: Path) -> dict[str, object]:
            self.root.rename(destination)
            self.root = destination
            events.append(("finalize_into", destination))
            return self.header

    monkeypatch.setattr(feasibility, "preflight_atomic_roots", preflight, raising=False)
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
        feasibility, "fit_canonical_clusters", fit_clusters, raising=False
    )
    monkeypatch.setattr(feasibility, "assign_targets", assign, raising=False)
    monkeypatch.setattr(feasibility, "StreamingFitCache", SyntheticCache, raising=False)
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

    cache_dir = tmp_path / "worker/cache"
    output_dir = tmp_path / "worker/output"
    receipt = feasibility.run_feasibility_worker(
        config,
        cache_dir=cache_dir,
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
    assert events[4][0] == "cache_create" or events[4][0] == "environment_check"
    assert clock_calls == 1_152
    assert len(fit_rows) == 512 + 64 == receipt.fit_count == 576
    assert receipt.start_count == 5_760
    fixture = json.loads(SAMPLE_FIXTURE.read_text(encoding="utf-8"))
    expected_inventory = [
        (kind, inventory_index, cluster_count)
        for kind, indices in (
            ("pair", fixture["pair_indices"]),
            ("single", fixture["single_indices"]),
        )
        for inventory_index in indices
        for cluster_count in (20, 40)
    ]
    assert [row[:3] for row in fit_rows] == expected_inventory
    assert [row[:3] for row in timing_rows] == expected_inventory
    assert {row[3] for row in timing_rows} == {1_000_000}
    assert fit_rows[0][3] == (query_ids[0], query_ids[93])
    assert fit_rows[-1][3] == (query_ids[321],)
    assert cache_headers[0]["expected_counts"] == {
        "pair_fit_count": 512,
        "single_fit_count": 64,
        "fit_count": 576,
        "start_count": 5_760,
        "timing_record_count": 576,
    }
    assert cache_headers[0]["numerical_environment_sha256"] == environment_sha256
    for category, count in (
        ("pair_c20", 256),
        ("pair_c40", 256),
        ("single_c20", 32),
        ("single_c40", 32),
    ):
        assert receipt.timing_summary[category].fit_count == count
    assert receipt.fit_receipt_sha256 == raw_file_sha256(
        output_dir / "fits/fit_receipts.jsonl"
    )
    assert receipt.provenance["cache_root"] == str(cache_dir.resolve())
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
    assert not cache_dir.exists()
    per_fit_events = [
        name
        for name, _ in events
        if name
        in {"start", "fit", "assign", "append_fit", "release", "end", "append_timing"}
    ]
    assert per_fit_events == [
        name
        for _ in range(576)
        for name in (
            "start",
            "fit",
            "assign",
            "append_fit",
            "release",
            "end",
            "append_timing",
        )
    ]
    assert [name for name, _ in events].count("environment_check") == 2
    assert len(environment_file_checks) == 2
    assert {expected for _path, expected in environment_file_checks} == {
        environment_sha256
    }


def _publish_repeat_fixture(
    tmp_path: Path,
    worker: str,
    *,
    changed_center: bool = False,
) -> FeasibilityReceipt:
    output_root = (tmp_path / f"repeat-{worker}").resolve()
    fits_root = output_root / "fits"
    fits_root.mkdir(parents=True)
    fit_rows = []
    for sequence in range(576):
        kind = "pair" if sequence < 512 else "single"
        local_sequence = sequence if kind == "pair" else sequence - 512
        fit_rows.append(
            {
                "sequence": sequence,
                "exclusion_kind": kind,
                "inventory_index": local_sequence // 2,
                "cluster_count": 20 if sequence % 2 == 0 else 40,
                "fit_key_sha256": f"{sequence:064x}"[-64:],
                "support_query_ids_sha256": "3" * 64,
                "embedding_sha256": "4" * 64,
                "clustering_contract_sha256": "5" * 64,
                "fit_membership_sha256": "6" * 64,
                "cluster_membership_sha256": ["7" * 64],
                "center_sha256": (
                    "9" * 64 if changed_center and sequence == 1 else "8" * 64
                ),
                "assignments": [
                    {
                        "query_id": f"q{sequence % 322:03d}",
                        "canonical_cluster_id": 0,
                        "squared_distance_hex": "0x0.0p+0",
                        "cluster_support": 1,
                    }
                ],
            }
        )
    fit_bytes = b"".join(canonical_json(row) + b"\n" for row in fit_rows)
    (fits_root / "fit_receipts.jsonl").write_bytes(fit_bytes)
    environment_bytes = canonical_json({"synthetic": "environment"}) + b"\n"
    (output_root / "environment.json").write_bytes(environment_bytes)
    pair_indices, single_indices = official_sample_indices()
    provenance = _provenance(tmp_path, worker)
    provenance.update(
        {
            "output_root": str(output_root),
            "sample_fixture_sha256": SAMPLE_FIXTURE_SHA256,
            "numerical_environment_sha256": hashlib.sha256(
                environment_bytes
            ).hexdigest(),
        }
    )
    receipt = FeasibilityReceipt(
        fit_count=576,
        start_count=5_760,
        pair_indices=tuple(map(int, pair_indices)),
        single_indices=tuple(map(int, single_indices)),
        timing_summary={
            "pair_c20": _summary(fit_count=256),
            "pair_c40": _summary(fit_count=256),
            "single_c20": _summary(fit_count=32),
            "single_c40": _summary(fit_count=32),
        },
        fit_receipt_sha256=hashlib.sha256(fit_bytes).hexdigest(),
        provenance=provenance,
        exclusive_resource_attested=True,
    )
    (output_root / "feasibility_receipt.json").write_bytes(
        canonical_json(_receipt_to_dict(receipt)) + b"\n"
    )
    artifacts.write_manifest(output_root)
    return receipt


def test_repeat_integrity_reopens_caches_and_accepts_distinct_manifested_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repeat_a = _publish_repeat_fixture(tmp_path, "a")
    repeat_b = _publish_repeat_fixture(tmp_path, "b")
    opened: list[Path] = []

    def open_readonly(_cls: object, root: Path) -> object:
        opened.append(Path(root).resolve())
        return SimpleNamespace(root=Path(root).resolve())

    monkeypatch.setattr(
        feasibility.StreamingFitCache,
        "open_readonly",
        classmethod(open_readonly),
    )
    assert _repeat_integrity_failures(repeat_a, repeat_b) == ()
    assert opened == [
        Path(repeat_a.provenance["output_root"]) / "fits",
        Path(repeat_b.provenance["output_root"]) / "fits",
    ]


def test_repeat_integrity_reports_center_manifest_runtime_and_root_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repeat_a = _publish_repeat_fixture(tmp_path, "a")
    repeat_b = _publish_repeat_fixture(tmp_path, "b", changed_center=True)
    monkeypatch.setattr(
        feasibility.StreamingFitCache,
        "open_readonly",
        classmethod(lambda _cls, root: SimpleNamespace(root=Path(root))),
    )
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
            "cache_root": repeat_a.provenance["cache_root"],
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
