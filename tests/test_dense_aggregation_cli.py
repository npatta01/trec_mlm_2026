from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import trec_mlm.dense_aggregation_cli as cli
from trec_mlm import dense_aggregation_artifacts as artifacts
from trec_mlm.dense_aggregation_artifacts import raw_file_sha256, validate_manifest
import trec_mlm.dense_aggregation_feasibility as feasibility
from trec_mlm.dense_aggregation_feasibility import (
    FEASIBILITY_PROVENANCE_KEYS,
    FeasibilityGate,
    FeasibilityReceipt,
    TimingCategorySummary,
    _receipt_to_dict,
)
from trec_mlm.dense_aggregation_hashing import canonical_json


build_parser = cli.build_parser
main = cli.main


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_parser_exposes_only_task7_qrel_free_commands() -> None:
    parser = build_parser()
    assert parser.prog == "trec-mlm-dense-aggregation"
    subparsers = next(
        action
        for action in parser._actions
        if action.__class__.__name__ == "_SubParsersAction"
    )
    assert tuple(subparsers.choices) == (
        "build-feasibility-config",
        "prepare-frozen-input",
        "validate-frozen-input",
        "feasibility",
        "feasibility-worker",
        "validate-feasibility",
    )
    assert callable(main)


@pytest.mark.parametrize(
    "forbidden",
    (
        "--qrels",
        "--grades",
        "--cluster-count",
        "--sample-seed",
        "--pair-sample-count",
        "--single-sample-count",
        "--runtime-threshold",
        "--rss-threshold",
        "--formula",
        "--grid",
        "--folds",
        "--fallback",
    ),
)
def test_feasibility_parser_rejects_all_overrides(forbidden: str) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "feasibility",
                "--config",
                "config.json",
                "--output-root",
                "out",
                "--compact-receipt",
                "receipt.json",
                "--attest-exclusive-resources",
                forbidden,
                "forbidden",
            ]
        )


def test_parser_disables_abbreviations_and_requires_all_public_inputs() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["feasibility", "--conf", "config.json"])
    with pytest.raises(SystemExit):
        parser.parse_args(["build-feasibility-config", "--output", "x.json"])
    with pytest.raises(SystemExit):
        parser.parse_args(["prepare-frozen-input", "--output-dir", "out"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "feasibility-worker",
                "--config",
                "config.json",
                "--cache-dir",
                "cache",
                "--output-dir",
                "out",
                "--exclusive-resource-attested",
            ]
        )


def test_cli_and_worker_import_graphs_remain_qrel_free_and_lazy() -> None:
    code = r"""
import json
import sys
import trec_mlm.dense_aggregation_cli
cli_modules = sorted(name for name in sys.modules if name.startswith("trec_mlm."))
import trec_mlm.dense_aggregation_feasibility
worker_modules = sorted(name for name in sys.modules if name.startswith("trec_mlm."))
print(json.dumps({"cli": cli_modules, "worker": worker_modules}))
"""
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    imported = json.loads(completed.stdout)
    assert imported["cli"] == ["trec_mlm.dense_aggregation_cli"]
    forbidden_modules = {
        "trec_mlm.dense_query_knn",
        "trec_mlm.query_knn",
        "trec_mlm.metrics",
        "trec_mlm.runio",
        "trec_mlm.dense_aggregation",
        "trec_mlm.dense_aggregation_experiment",
        "trec_mlm.dense_aggregation_evaluation",
        "trec_mlm.legacy_dense_control",
    }
    assert forbidden_modules.isdisjoint(imported["worker"])


def test_worker_destination_preflight_precedes_all_numerical_imports(
    tmp_path: Path,
) -> None:
    code = r"""
import json
import sys
import trec_mlm.dense_aggregation_cli as cli
destination = sys.argv[1]
try:
    cli.main([
        "feasibility-worker",
        "--config", "missing-config.json",
        "--fit-work-file", destination + "/fits.partial.h5",
        "--output-dir", destination,
        "--exclusive-resource-attested",
    ])
except ValueError as error:
    message = str(error)
else:
    raise AssertionError("overlapping worker roots were accepted")
numerical = sorted(
    name for name in sys.modules
    if name.split(".", 1)[0] in {"numpy", "scipy", "sklearn", "threadpoolctl"}
)
print(json.dumps({
    "message": message,
    "numerical": numerical,
    "feasibility": "trec_mlm.dense_aggregation_feasibility" in sys.modules,
}))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "same-root")],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert "overlap" in result["message"]
    assert result["numerical"] == []
    assert result["feasibility"] is False


def test_build_config_publishes_canonical_create_exclusive_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "configs/feasibility.json"
    output.parent.mkdir()
    payload = {
        "schema": "trec-mlm-dense-aggregation-feasibility-config-v1",
        "query_path": "data/dev/queries.tsv",
        "embedding_path": "artifacts/frozen/mpnet_base_v2.npy",
        "encoder_metadata_path": "artifacts/frozen/encoder_metadata.json",
        "protocol_path": "protocols/trec_2026_dense_aggregation.md",
    }
    calls: list[object] = []

    def build_payload(args: object) -> tuple[Path, dict[str, object]]:
        calls.append(args)
        return output, payload

    monkeypatch.setattr(cli, "_build_config_payload", build_payload, raising=False)
    argv = [
        "build-feasibility-config",
        "--queries",
        "queries.tsv",
        "--embeddings",
        "embedding.npy",
        "--encoder-metadata",
        "metadata.json",
        "--protocol",
        "protocol.md",
        "--output",
        str(output),
    ]
    assert main(argv) == 0
    assert len(calls) == 1
    assert output.read_bytes() == canonical_json(payload) + b"\n"
    assert not list(output.parent.glob(f".{output.name}.*.tmp"))
    with pytest.raises(FileExistsError):
        main(argv)
    assert output.read_bytes() == canonical_json(payload) + b"\n"


def test_prepare_and_validate_frozen_input_are_manifested_and_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_embedding = tmp_path / "source.npy"
    source_metadata = tmp_path / "source.json"
    source_embedding.write_bytes(b"synthetic-numpy-array")
    source_metadata.write_bytes(b'{"synthetic":true}\n')
    output = tmp_path / "frozen"
    validations: list[tuple[Path, Path]] = []

    def validate_files(embedding: Path, metadata: Path) -> None:
        assert embedding.read_bytes()
        assert metadata.read_bytes()
        validations.append((embedding.resolve(), metadata.resolve()))

    monkeypatch.setattr(cli, "_validate_frozen_files", validate_files, raising=False)
    assert (
        main(
            [
                "prepare-frozen-input",
                "--source-embedding",
                str(source_embedding),
                "--source-metadata",
                str(source_metadata),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    validate_manifest(output)
    assert (output / "mpnet_base_v2.npy").read_bytes() == source_embedding.read_bytes()
    assert (
        output / "encoder_metadata.json"
    ).read_bytes() == source_metadata.read_bytes()
    assert validations[0] == (source_embedding.resolve(), source_metadata.resolve())
    assert len(validations) == 3
    assert validations[1][0].parent != output
    assert validations[2] == (
        (output / "mpnet_base_v2.npy").resolve(),
        (output / "encoder_metadata.json").resolve(),
    )
    snapshot = {
        path.relative_to(output).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    assert main(["validate-frozen-input", "--input-dir", str(output)]) == 0
    assert validations[-1] == (
        (output / "mpnet_base_v2.npy").resolve(),
        (output / "encoder_metadata.json").resolve(),
    )
    assert snapshot == {
        path.relative_to(output).as_posix(): (
            path.read_bytes(),
            path.stat().st_mtime_ns,
        )
        for path in output.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError):
        main(
            [
                "prepare-frozen-input",
                "--source-embedding",
                str(source_embedding),
                "--source-metadata",
                str(source_metadata),
                "--output-dir",
                str(output),
            ]
        )


def test_cli_frozen_validator_rejects_oversized_npy_before_numpy_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    embedding = tmp_path / "oversized.npy"
    metadata = tmp_path / "metadata.json"
    with embedding.open("wb") as handle:
        handle.truncate(8 * 1024 * 1024)
    metadata.write_bytes(b"{}\n")

    def forbidden_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("oversized NPY reached numpy.load")

    monkeypatch.setattr(np, "load", forbidden_load)
    with pytest.raises(ValueError, match="byte limit"):
        cli._validate_frozen_files(embedding, metadata)


def _cli_receipt(output_root: Path, fit_work_file: Path, pid: int) -> FeasibilityReceipt:
    provenance: dict[str, object] = {
        name: "1" * 64 for name in FEASIBILITY_PROVENANCE_KEYS
    }
    provenance.update(
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
            "fit_work_file": str(fit_work_file.resolve()),
            "output_root": str(output_root.resolve()),
            "worker_pid": pid,
        }
    )
    summary = TimingCategorySummary(
        fit_count=1,
        mean_seconds=0.001,
        median_seconds=0.001,
        p95_seconds=0.001,
        maximum_seconds=0.001,
        p95_method="linear_index_(n-1)*0.95",
        n_iter_min=1,
        n_iter_mean=1.0,
        n_iter_max=1,
        n_iter_values_sha256="2" * 64,
    )
    return FeasibilityReceipt(
        fit_count=576,
        start_count=5_760,
        pair_indices=tuple(range(256)),
        single_indices=tuple(range(32)),
        timing_summary={
            name: summary
            for name in ("pair_c20", "pair_c40", "single_c20", "single_c40")
        },
        fit_file_sha256="3" * 64,
        fit_evidence_sha256="4" * 64,
        provenance=provenance,
        exclusive_resource_attested=True,
    )


def _write_cli_worker_fixture(
    tmp_path: Path, *, canonical_receipt: bool = True
) -> tuple[Path, FeasibilityReceipt]:
    output_root = (tmp_path / "worker-output").resolve()
    output_root.mkdir(parents=True)
    (output_root / "fits.h5").write_bytes(b"synthetic hdf5 fixture")
    (output_root / "environment.json").write_bytes(b"{}\n")
    receipt = replace(
        _cli_receipt(output_root, tmp_path / "worker-fits.partial.h5", 101),
        fit_file_sha256=raw_file_sha256(output_root / "fits.h5"),
    )
    receipt_value = _receipt_to_dict(receipt)
    receipt_bytes = canonical_json(receipt_value) + b"\n"
    if not canonical_receipt:
        receipt_bytes = json.dumps(receipt_value, indent=2).encode("utf-8") + b"\n"
    (output_root / "feasibility_receipt.json").write_bytes(receipt_bytes)
    artifacts.write_manifest(output_root)
    return output_root, receipt


def _write_cli_aggregate_fixture(
    tmp_path: Path, *, canonical_repeat_a: bool = True
) -> tuple[Path, Path]:
    root = (tmp_path / "aggregate").resolve()
    root.mkdir(parents=True)
    repeat_a = _cli_receipt(
        (tmp_path / "worker-a-output").resolve(),
        (tmp_path / "worker-a-fits.partial.h5").resolve(),
        101,
    )
    repeat_b = _cli_receipt(
        (tmp_path / "worker-b-output").resolve(),
        (tmp_path / "worker-b-fits.partial.h5").resolve(),
        202,
    )
    gate = FeasibilityGate(
        status="pass",
        failures=(),
        projection_seconds=1.0,
        maximum_rss_bytes=1,
    )
    (root / "feasibility_gate.json").write_bytes(
        canonical_json(cli._gate_to_dict(gate)) + b"\n"
    )
    repeat_a_value = _receipt_to_dict(repeat_a)
    repeat_a_bytes = canonical_json(repeat_a_value) + b"\n"
    if not canonical_repeat_a:
        repeat_a_bytes = json.dumps(repeat_a_value, indent=2).encode("utf-8") + b"\n"
    (root / "repeat_a_receipt.json").write_bytes(repeat_a_bytes)
    (root / "repeat_b_receipt.json").write_bytes(
        canonical_json(_receipt_to_dict(repeat_b)) + b"\n"
    )
    (root / "repeat_measurements.json").write_bytes(
        canonical_json(
            {
                "schema": (
                    "trec-mlm-dense-aggregation-feasibility-repeat-measurements-v1"
                ),
                "repeat_a_manifest_sha256": "1" * 64,
                "repeat_b_manifest_sha256": "2" * 64,
                "repeat_a_rss_bytes": 1,
                "repeat_b_rss_bytes": 1,
            }
        )
        + b"\n"
    )
    artifacts.write_manifest(root)
    compact = tmp_path / "compact.json"
    compact.write_bytes(b"{}\n")
    return root, compact


def test_worker_output_requires_canonical_bounded_receipt_and_exact_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root, _receipt = _write_cli_worker_fixture(
        tmp_path / "noncanonical", canonical_receipt=False
    )
    with pytest.raises(ValueError, match="canonical"):
        cli._load_worker_output(output_root)

    output_root, _receipt = _write_cli_worker_fixture(tmp_path / "oversized")
    receipt_path = output_root / "feasibility_receipt.json"
    with receipt_path.open("wb") as handle:
        handle.truncate(2 * 1024 * 1024)
    (output_root / "manifest.json").unlink()
    artifacts.write_manifest(output_root)
    real_read_bytes = Path.read_bytes

    def reject_unbounded_receipt(path: Path) -> bytes:
        if path == receipt_path:
            raise AssertionError("oversized receipt reached Path.read_bytes")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", reject_unbounded_receipt)
    with pytest.raises(ValueError, match="byte limit"):
        cli._load_worker_output(output_root)


def test_worker_output_loader_defers_hdf5_scan_to_repeat_evaluation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root, expected = _write_cli_worker_fixture(tmp_path)

    monkeypatch.setattr(
        feasibility,
        "validate_fit_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("worker output loader scanned fits.h5")
        ),
    )

    assert cli._load_worker_output(output_root) == expected


def test_worker_manifest_preflight_rejects_extra_and_oversized_layouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_general_validation(root: Path) -> object:
        del root
        raise AssertionError("general manifest validation ran before Task 7 preflight")

    output_root, _receipt = _write_cli_worker_fixture(tmp_path / "extra")
    (output_root / "unexpected.txt").write_bytes(b"unexpected\n")
    (output_root / "manifest.json").unlink()
    artifacts.write_manifest(output_root)
    monkeypatch.setattr(artifacts, "validate_manifest", forbidden_general_validation)
    monkeypatch.setattr(feasibility, "validate_manifest", forbidden_general_validation)
    with pytest.raises(ValueError, match="unexpected layout"):
        cli._load_worker_output(output_root)

    output_root, _receipt = _write_cli_worker_fixture(tmp_path / "oversized")
    with (output_root / "manifest.json").open("wb") as handle:
        handle.truncate(2 * 1024 * 1024)
    with pytest.raises(ValueError, match="byte limit"):
        cli._load_worker_output(output_root)


def test_worker_physical_layout_preflight_rejects_unmanifested_extra_before_general(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root, _receipt = _write_cli_worker_fixture(tmp_path)
    (output_root / "unexpected.txt").write_bytes(b"unmanifested\n")

    def forbidden_general_validation(root: Path) -> object:
        del root
        raise AssertionError(
            "general manifest validation ran before physical preflight"
        )

    monkeypatch.setattr(feasibility, "validate_manifest", forbidden_general_validation)

    with pytest.raises(ValueError, match="unexpected physical layout"):
        cli._load_worker_output(output_root)


def test_worker_physical_layout_preflight_never_descends_unexpected_subtree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root, _receipt = _write_cli_worker_fixture(tmp_path)
    unexpected = output_root / "unexpected"
    (unexpected / "deep").mkdir(parents=True)
    (unexpected / "deep/sentinel").write_bytes(b"must not be traversed\n")
    real_scandir = os.scandir

    def guarded_scandir(path: os.PathLike[str] | str) -> os.ScandirIterator[str]:
        if Path(path).resolve() == unexpected.resolve():
            raise AssertionError("physical preflight descended unexpected subtree")
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", guarded_scandir)

    with pytest.raises(ValueError, match="unexpected physical layout"):
        cli._load_worker_output(output_root)


def test_aggregate_requires_canonical_receipt_copies_and_exact_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = type("Config", (), {})()
    monkeypatch.setattr(cli, "_load_config", lambda path: config, raising=False)
    root, compact = _write_cli_aggregate_fixture(
        tmp_path / "noncanonical", canonical_repeat_a=False
    )

    def forbidden_worker_load(root: Path) -> object:
        del root
        raise AssertionError("noncanonical aggregate receipt was accepted")

    monkeypatch.setattr(cli, "_load_worker_output", forbidden_worker_load)
    with pytest.raises(ValueError, match="canonical"):
        cli._validate_feasibility_artifact(Path("config"), root, compact)

    root, compact = _write_cli_aggregate_fixture(tmp_path / "extra")
    (root / "unexpected.txt").write_bytes(b"unexpected\n")
    (root / "manifest.json").unlink()
    artifacts.write_manifest(root)

    def forbidden_general_validation(candidate: Path) -> object:
        del candidate
        raise AssertionError("general manifest validation ran before layout preflight")

    monkeypatch.setattr(artifacts, "validate_manifest", forbidden_general_validation)
    monkeypatch.setattr(feasibility, "validate_manifest", forbidden_general_validation)
    with pytest.raises(ValueError, match="unexpected layout"):
        cli._validate_feasibility_artifact(Path("config"), root, compact)


@pytest.mark.parametrize("status", ("pass", "stop"))
def test_coordinator_runs_exact_sequential_workers_and_publishes_gate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    status: str,
) -> None:
    config_path = tmp_path / "configs/feasibility.json"
    config_path.parent.mkdir()
    config_path.write_text("{}\n", encoding="utf-8")
    output_root = tmp_path / "artifacts/feasibility"
    output_root.parent.mkdir()
    compact = tmp_path / "reports/feasibility.json"
    compact.parent.mkdir()
    config = type(
        "Config",
        (),
        {
            "repo_root": tmp_path.resolve(),
            "source_path": config_path.resolve(),
            "protocol_path": (
                tmp_path / "protocols/trec_2026_dense_aggregation.md"
            ).resolve(),
            "query_path": (tmp_path / "data/dev/queries.tsv").resolve(),
            "embedding_path": (
                tmp_path / "artifacts/frozen/mpnet_base_v2.npy"
            ).resolve(),
            "encoder_metadata_path": (
                tmp_path / "artifacts/frozen/encoder_metadata.json"
            ).resolve(),
        },
    )()
    subprocess_calls: list[tuple[list[str], Path, dict[str, str]]] = []
    worker_receipts: dict[Path, FeasibilityReceipt] = {}

    monkeypatch.setattr(cli, "_load_config", lambda path: config, raising=False)

    def run_worker(
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        assert check is False
        subprocess_calls.append((argv, Path(cwd), env))
        time_path = (Path(cwd) / argv[argv.index("-o") + 1]).resolve()
        output_path = (Path(cwd) / argv[argv.index("--output-dir") + 1]).resolve()
        fit_work_file = (
            Path(cwd) / argv[argv.index("--fit-work-file") + 1]
        ).resolve()
        assert not time_path.exists()
        assert not output_path.exists()
        assert not fit_work_file.exists()
        time_path.write_text(
            "Maximum resident set size (kbytes): 123456\n", encoding="utf-8"
        )
        output_path.mkdir()
        (output_path / "manifest.json").write_bytes(
            f"synthetic-manifest-{len(subprocess_calls)}\n".encode()
        )
        worker_receipts[output_path] = _cli_receipt(
            output_path, fit_work_file, 100 + len(subprocess_calls)
        )
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", run_worker)
    monkeypatch.setattr(
        cli,
        "_load_worker_output",
        lambda root: worker_receipts[Path(root).resolve()],
        raising=False,
    )
    gate = FeasibilityGate(
        status=status,
        failures=() if status == "pass" else ("synthetic stop",),
        projection_seconds=1.0,
        maximum_rss_bytes=123_456 * 1024,
    )
    monkeypatch.setattr(
        cli,
        "_evaluate_gate",
        lambda *args, **kwargs: gate,
        raising=False,
    )
    return_code = main(
        [
            "feasibility",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
            "--compact-receipt",
            str(compact),
            "--attest-exclusive-resources",
        ]
    )
    assert return_code == (0 if status == "pass" else 1)
    assert len(subprocess_calls) == 2
    for repeat_index, label in enumerate(("a", "b")):
        argv, cwd, environment = subprocess_calls[repeat_index]
        assert cwd == tmp_path.resolve()
        assert environment["LC_ALL"] == "C"
        assert argv[:2] == ["/usr/bin/time", "-v"]
        assert argv[4:12] == [
            "uv",
            "run",
            "--frozen",
            "--python",
            "3.13.9",
            "--extra",
            "dense-aggregation",
            "trec-mlm-dense-aggregation",
        ]
        assert argv[12] == "feasibility-worker"
        assert argv[-1] == "--exclusive-resource-attested"
        assert f"repeat-{label}" in " ".join(argv)
    validate_manifest(output_root)
    assert (output_root / "feasibility_gate.json").is_file()
    assert (output_root / "repeat_a_receipt.json").is_file()
    assert (output_root / "repeat_b_receipt.json").is_file()
    measurements = json.loads((output_root / "repeat_measurements.json").read_bytes())
    assert measurements["repeat_a_rss_bytes"] == 123_456 * 1024
    assert measurements["repeat_b_rss_bytes"] == 123_456 * 1024
    if status == "pass":
        compact_payload = json.loads(compact.read_bytes())
        assert compact_payload["gate"]["status"] == "pass"
        assert compact_payload["feasibility_manifest_sha256"] == raw_file_sha256(
            output_root / "manifest.json"
        )
    else:
        assert not compact.exists()


def test_coordinator_preserves_worker_failure_as_hard_error_without_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_root = tmp_path / "artifacts/feasibility"
    compact_receipt = tmp_path / "reports/feasibility.json"
    config = type(
        "Config",
        (),
        {
            "repo_root": tmp_path.resolve(),
            "source_path": (tmp_path / "configs/feasibility.json").resolve(),
            "protocol_path": (
                tmp_path / "protocols/trec_2026_dense_aggregation.md"
            ).resolve(),
            "query_path": (tmp_path / "data/dev/queries.tsv").resolve(),
            "embedding_path": (
                tmp_path / "artifacts/frozen/mpnet_base_v2.npy"
            ).resolve(),
            "encoder_metadata_path": (
                tmp_path / "artifacts/frozen/encoder_metadata.json"
            ).resolve(),
        },
    )()
    subprocess_calls: list[list[str]] = []
    monkeypatch.setattr(cli, "_load_config", lambda path: config, raising=False)

    def fail_worker(
        argv: list[str], **kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        subprocess_calls.append(argv)
        return subprocess.CompletedProcess(argv, 17)

    monkeypatch.setattr(subprocess, "run", fail_worker)

    with pytest.raises(RuntimeError, match="status 17"):
        main(
            [
                "feasibility",
                "--config",
                str(config.source_path),
                "--output-root",
                str(output_root),
                "--compact-receipt",
                str(compact_receipt),
                "--attest-exclusive-resources",
            ]
        )

    assert len(subprocess_calls) == 1
    assert not output_root.exists()
    assert not compact_receipt.exists()


@pytest.mark.parametrize("compact_location", ("inside-output", "repeat-time"))
def test_coordinator_rejects_overlapping_destinations_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    compact_location: str,
) -> None:
    output_root = tmp_path / "artifacts/feasibility"
    if compact_location == "inside-output":
        compact_receipt = output_root / "compact.json"
    else:
        compact_receipt = output_root.parent / ".feasibility-repeat-a.time.txt"
    config = type(
        "Config",
        (),
        {
            "repo_root": tmp_path.resolve(),
            "source_path": (tmp_path / "configs/feasibility.json").resolve(),
            "protocol_path": (
                tmp_path / "protocols/trec_2026_dense_aggregation.md"
            ).resolve(),
            "query_path": (tmp_path / "data/dev/queries.tsv").resolve(),
            "embedding_path": (
                tmp_path / "artifacts/frozen/mpnet_base_v2.npy"
            ).resolve(),
            "encoder_metadata_path": (
                tmp_path / "artifacts/frozen/encoder_metadata.json"
            ).resolve(),
        },
    )()
    subprocess_calls: list[list[str]] = []

    monkeypatch.setattr(cli, "_load_config", lambda path: config, raising=False)

    def run_worker(argv: list[str], **kwargs: object) -> None:
        subprocess_calls.append(argv)
        raise AssertionError("coordinator started a worker before path preflight")

    monkeypatch.setattr(subprocess, "run", run_worker)
    with pytest.raises(ValueError, match="overlap"):
        main(
            [
                "feasibility",
                "--config",
                str(tmp_path / "configs/feasibility.json"),
                "--output-root",
                str(output_root),
                "--compact-receipt",
                str(compact_receipt),
                "--attest-exclusive-resources",
            ]
        )
    assert subprocess_calls == []
    assert not output_root.parent.exists()


@pytest.mark.parametrize("destination_index", range(8))
def test_coordinator_rejects_configured_input_directory_destinations_before_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, destination_index: int
) -> None:
    frozen_root = tmp_path / "artifacts/frozen"
    output_root = tmp_path / "artifacts/feasibility"
    compact_receipt = tmp_path / "reports/feasibility.json"
    prefix = f".{output_root.name}-repeat"
    protected_destination = (
        output_root,
        compact_receipt,
        output_root.parent / f"{prefix}-a.time.txt",
        output_root.parent / f"{prefix}-a-fits.partial.h5",
        output_root.parent / f"{prefix}-a-output",
        output_root.parent / f"{prefix}-b.time.txt",
        output_root.parent / f"{prefix}-b-fits.partial.h5",
        output_root.parent / f"{prefix}-b-output",
    )[destination_index]
    config = type(
        "Config",
        (),
        {
            "repo_root": tmp_path.resolve(),
            "source_path": (tmp_path / "configs/feasibility.json").resolve(),
            "protocol_path": (
                tmp_path / "protocols/trec_2026_dense_aggregation.md"
            ).resolve(),
            "query_path": (protected_destination / "queries.tsv").resolve(),
            "embedding_path": (frozen_root / "mpnet_base_v2.npy").resolve(),
            "encoder_metadata_path": (frozen_root / "encoder_metadata.json").resolve(),
        },
    )()
    subprocess_calls: list[list[str]] = []

    monkeypatch.setattr(cli, "_load_config", lambda path: config, raising=False)

    def run_worker(argv: list[str], **kwargs: object) -> None:
        subprocess_calls.append(argv)
        raise AssertionError("coordinator started a worker before input path preflight")

    monkeypatch.setattr(subprocess, "run", run_worker)
    with pytest.raises(ValueError, match="configured input"):
        main(
            [
                "feasibility",
                "--config",
                str(config.source_path),
                "--output-root",
                str(output_root),
                "--compact-receipt",
                str(compact_receipt),
                "--attest-exclusive-resources",
            ]
        )
    assert subprocess_calls == []
    assert not output_root.parent.exists()
    assert not compact_receipt.parent.exists()


def test_validate_feasibility_command_is_strictly_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "config.json"
    root = tmp_path / "root"
    compact = tmp_path / "compact.json"
    config.write_text("config\n", encoding="utf-8")
    root.mkdir()
    (root / "sentinel").write_text("root\n", encoding="utf-8")
    compact.write_text("compact\n", encoding="utf-8")
    observed: list[tuple[Path, Path, Path]] = []

    def validate(config_path: Path, feasibility_root: Path, receipt: Path) -> None:
        observed.append((config_path, feasibility_root, receipt))

    monkeypatch.setattr(cli, "_validate_feasibility_artifact", validate, raising=False)
    before = {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (config, root / "sentinel", compact)
    }
    assert (
        main(
            [
                "validate-feasibility",
                "--config",
                str(config),
                "--feasibility-root",
                str(root),
                "--compact-receipt",
                str(compact),
            ]
        )
        == 0
    )
    assert observed == [(config, root, compact)]
    assert before == {
        path: (path.read_bytes(), path.stat().st_mtime_ns)
        for path in (config, root / "sentinel", compact)
    }
