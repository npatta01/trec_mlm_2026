from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import math
import os
from pathlib import Path
import shutil
import subprocess
import uuid


_PROTOCOL_SHA256 = "bbe6b50fd3d20efe3213a446f6665cbe0527114f34bf5c1e90236af4f998ad49"
_COMPACT_SCHEMA = "trec-mlm-dense-aggregation-feasibility-compact-v1"
_MEASUREMENTS_SCHEMA = "trec-mlm-dense-aggregation-feasibility-repeat-measurements-v1"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trec-mlm-dense-aggregation", allow_abbrev=False
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_config = subparsers.add_parser("build-feasibility-config", allow_abbrev=False)
    build_config.add_argument("--queries", required=True)
    build_config.add_argument("--embeddings", required=True)
    build_config.add_argument("--encoder-metadata", required=True)
    build_config.add_argument("--protocol", required=True)
    build_config.add_argument("--output", required=True)

    prepare = subparsers.add_parser("prepare-frozen-input", allow_abbrev=False)
    prepare.add_argument("--source-embedding", required=True)
    prepare.add_argument("--source-metadata", required=True)
    prepare.add_argument("--output-dir", required=True)

    validate_input = subparsers.add_parser("validate-frozen-input", allow_abbrev=False)
    validate_input.add_argument("--input-dir", required=True)

    coordinate = subparsers.add_parser("feasibility", allow_abbrev=False)
    coordinate.add_argument("--config", required=True)
    coordinate.add_argument("--output-root", required=True)
    coordinate.add_argument("--compact-receipt", required=True)
    coordinate.add_argument(
        "--attest-exclusive-resources", action="store_true", required=True
    )

    worker = subparsers.add_parser("feasibility-worker", allow_abbrev=False)
    worker.add_argument("--config", required=True)
    worker.add_argument("--cache-dir", required=True)
    worker.add_argument("--output-dir", required=True)
    worker.add_argument(
        "--exclusive-resource-attested", action="store_true", required=True
    )

    validate = subparsers.add_parser("validate-feasibility", allow_abbrev=False)
    validate.add_argument("--config", required=True)
    validate.add_argument("--feasibility-root", required=True)
    validate.add_argument("--compact-receipt", required=True)
    return parser


def _path_exists_lexically(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _paths_overlap(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _configured_input_directories(config: object) -> tuple[Path, ...]:
    directories = {
        Path(getattr(config, field)).resolve().parent
        for field in (
            "source_path",
            "protocol_path",
            "query_path",
            "embedding_path",
            "encoder_metadata_path",
        )
    }
    return tuple(sorted(directories, key=str))


def _preflight_worker_destinations(*, cache_dir: Path, output_dir: Path) -> None:
    cache_path = Path(cache_dir)
    output_path = Path(output_dir)
    cache_root = cache_path.resolve()
    output_root = output_path.resolve()
    if (
        cache_root == output_root
        or cache_root in output_root.parents
        or output_root in cache_root.parents
    ):
        raise ValueError("cache and output destinations overlap")
    for destination_kind, destination_path, destination_root in (
        ("cache", cache_path, cache_root),
        ("output", output_path, output_root),
    ):
        for candidate in (destination_path, destination_root):
            if _path_exists_lexically(candidate):
                raise FileExistsError(
                    f"{destination_kind} destination exists: {destination_path}"
                )
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    if cache_root.parent.stat().st_dev != output_root.parent.stat().st_dev:
        raise OSError("cache and output must share one filesystem")


def _atomic_publish_bytes(destination: Path, payload: bytes) -> None:
    from trec_mlm.dense_aggregation_artifacts import rename_noreplace

    final = Path(destination)
    if _path_exists_lexically(final):
        raise FileExistsError(f"artifact destination exists: {final}")
    parent = final.parent.resolve(strict=True)
    temporary = parent / f".{final.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        rename_noreplace(temporary, final)
        directory_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_publish_json(destination: Path, value: object) -> None:
    from trec_mlm.dense_aggregation_hashing import canonical_json

    _atomic_publish_bytes(Path(destination), canonical_json(value) + b"\n")


def _require_under_root(path: Path, root: Path, *, name: str) -> Path:
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{name} resolves outside repository root")
    return resolved


def _build_config_payload(args: argparse.Namespace) -> tuple[Path, dict[str, object]]:
    from trec_mlm.dense_aggregation_artifacts import raw_file_sha256
    from trec_mlm.dense_aggregation_contract import (
        FeasibilityConfig,
        build_clustering_contract,
        load_and_validate_frozen_dense_inputs,
    )
    from trec_mlm.dense_aggregation_hashing import canonical_json

    protocol_path = Path(args.protocol).resolve(strict=True)
    if protocol_path.name != "trec_2026_dense_aggregation.md" or (
        protocol_path.parent.name != "protocols"
    ):
        raise ValueError("protocol must be the locked repository protocol path")
    repo_root = protocol_path.parent.parent.resolve(strict=True)
    expected_protocol = (
        repo_root / "protocols/trec_2026_dense_aggregation.md"
    ).resolve(strict=True)
    if protocol_path != expected_protocol:
        raise ValueError("protocol path does not identify the repository root")
    if raw_file_sha256(protocol_path) != _PROTOCOL_SHA256:
        raise ValueError("locked dense aggregation protocol hash mismatch")

    configs_root = (repo_root / "configs").resolve(strict=True)
    output = Path(args.output)
    if output.parent.resolve(strict=True) != configs_root:
        raise ValueError("feasibility config output must be a direct child of configs")
    output = configs_root / output.name
    if _path_exists_lexically(output):
        raise FileExistsError(f"artifact destination exists: {output}")

    paths = {
        "query_path": _require_under_root(
            Path(args.queries), repo_root, name="queries"
        ),
        "embedding_path": _require_under_root(
            Path(args.embeddings), repo_root, name="embeddings"
        ),
        "encoder_metadata_path": _require_under_root(
            Path(args.encoder_metadata), repo_root, name="encoder metadata"
        ),
        "protocol_path": protocol_path,
    }
    payload: dict[str, object] = {
        "schema": "trec-mlm-dense-aggregation-feasibility-config-v1",
        **{
            name: path.relative_to(repo_root).as_posix() for name, path in paths.items()
        },
    }
    payload_bytes = canonical_json(payload) + b"\n"
    config = FeasibilityConfig(
        repo_root=repo_root,
        source_path=output,
        source_sha256=hashlib.sha256(payload_bytes).hexdigest(),
        query_path=paths["query_path"],
        embedding_path=paths["embedding_path"],
        encoder_metadata_path=paths["encoder_metadata_path"],
        protocol_path=paths["protocol_path"],
    )
    inputs = load_and_validate_frozen_dense_inputs(config)
    build_clustering_contract(inputs, repo_root=repo_root, protocol_path=protocol_path)
    return output, payload


def _validate_frozen_files(embedding_path: Path, metadata_path: Path) -> None:
    import numpy as np

    from trec_mlm.dense_aggregation_contract import (
        _EMBEDDING_SHA256,
        _SIMILARITY_SHA256,
        _cosine_similarity_matrix_qrel_free,
        _legacy_array_sha256_qrel_free,
        _load_frozen_embedding_qrel_free,
        _load_frozen_metadata_qrel_free,
        _require_exact_mpnet_metadata,
    )

    embeddings = _load_frozen_embedding_qrel_free(embedding_path)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.isfinite(embeddings).all() or np.any(norms == 0.0):
        raise ValueError("frozen MPNet rows must be finite and nonzero")
    if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("frozen MPNet rows are not normalized")
    if _legacy_array_sha256_qrel_free(embeddings) != _EMBEDDING_SHA256:
        raise ValueError("frozen MPNet embedding logical hash mismatch")
    similarities = _cosine_similarity_matrix_qrel_free(embeddings)
    if _legacy_array_sha256_qrel_free(similarities) != _SIMILARITY_SHA256:
        raise ValueError("frozen MPNet cosine logical hash mismatch")
    metadata_value = _load_frozen_metadata_qrel_free(metadata_path)
    metadata = metadata_value.get("mpnet_base_v2")
    if type(metadata) is not dict:
        raise ValueError("frozen encoder metadata is missing mpnet_base_v2")
    _require_exact_mpnet_metadata(metadata)


def _validate_frozen_root(input_dir: Path) -> None:
    from trec_mlm.dense_aggregation_artifacts import validate_manifest

    root = Path(input_dir).resolve(strict=True)
    manifest = validate_manifest(root)
    files = manifest.get("files")
    if type(files) is not list or [row[0] for row in files] != [
        "encoder_metadata.json",
        "mpnet_base_v2.npy",
    ]:
        raise ValueError("frozen input manifest has an unexpected inventory")
    _validate_frozen_files(root / "mpnet_base_v2.npy", root / "encoder_metadata.json")


def _load_config(path: Path) -> object:
    from trec_mlm.dense_aggregation_contract import load_feasibility_config

    return load_feasibility_config(path)


def _load_worker_output(root: Path) -> object:
    from trec_mlm.dense_aggregation_artifacts import raw_file_sha256
    from trec_mlm.dense_aggregation_feasibility import (
        _WORKER_OUTPUT_FILES,
        _read_bounded_canonical_json,
        _receipt_from_dict,
        _validate_task7_manifest,
    )

    output_root = Path(root).resolve(strict=True)
    _validate_task7_manifest(
        output_root,
        expected_files=_WORKER_OUTPUT_FILES,
        label="worker output",
    )
    receipt = _receipt_from_dict(
        _read_bounded_canonical_json(
            output_root / "feasibility_receipt.json",
            label="worker feasibility receipt",
        )
    )
    if receipt.provenance.get("output_root") != str(output_root):
        raise ValueError("worker receipt output root is not self-binding")
    if (
        raw_file_sha256(output_root / "fits/fit_receipts.jsonl")
        != receipt.fit_receipt_sha256
    ):
        raise ValueError("worker fit receipt stream digest mismatch")
    return receipt


def _evaluate_gate(*args: object, **kwargs: object) -> object:
    from trec_mlm.dense_aggregation_feasibility import (
        evaluate_feasibility_repeats,
    )

    return evaluate_feasibility_repeats(*args, **kwargs)


def _gate_to_dict(gate: object) -> dict[str, object]:
    return {
        "schema": gate.schema,
        "status": gate.status,
        "failures": list(gate.failures),
        "projection_seconds": gate.projection_seconds,
        "maximum_rss_bytes": gate.maximum_rss_bytes,
    }


def _gate_from_dict(value: object) -> object:
    from trec_mlm.dense_aggregation_feasibility import FeasibilityGate

    keys = {
        "schema",
        "status",
        "failures",
        "projection_seconds",
        "maximum_rss_bytes",
    }
    if type(value) is not dict or set(value) != keys:
        raise ValueError("feasibility gate has an invalid contract")
    if value["schema"] != "trec-mlm-dense-aggregation-feasibility-gate-v1":
        raise ValueError("unsupported feasibility gate schema")
    status = value["status"]
    failures = value["failures"]
    projection = value["projection_seconds"]
    rss = value["maximum_rss_bytes"]
    if status not in {"pass", "stop"} or type(status) is not str:
        raise ValueError("feasibility gate status is invalid")
    if type(failures) is not list or any(type(item) is not str for item in failures):
        raise ValueError("feasibility gate failures are invalid")
    if type(projection) not in {int, float} or not math.isfinite(float(projection)):
        raise ValueError("feasibility gate projection is invalid")
    if type(rss) is not int or rss < 0:
        raise ValueError("feasibility gate RSS is invalid")
    if (status == "pass") != (not failures):
        raise ValueError("feasibility gate status and failures disagree")
    return FeasibilityGate(
        status=status,
        failures=tuple(failures),
        projection_seconds=float(projection),
        maximum_rss_bytes=rss,
    )


def _repo_argument(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(repo_root):
        return resolved.relative_to(repo_root).as_posix()
    return str(resolved)


def _write_staging_json(staging: Path, name: str, value: object) -> None:
    from trec_mlm.dense_aggregation_hashing import canonical_json

    with (staging / name).open("xb") as handle:
        handle.write(canonical_json(value) + b"\n")


def _handle_build_config(args: argparse.Namespace) -> int:
    output, payload = _build_config_payload(args)
    _atomic_publish_json(output, payload)
    return 0


def _handle_prepare_frozen_input(args: argparse.Namespace) -> int:
    from trec_mlm.dense_aggregation_artifacts import atomic_artifact_directory

    source_embedding = Path(args.source_embedding).resolve(strict=True)
    source_metadata = Path(args.source_metadata).resolve(strict=True)
    _validate_frozen_files(source_embedding, source_metadata)
    output_dir = Path(args.output_dir).resolve()
    with atomic_artifact_directory(output_dir) as staging:
        shutil.copyfile(source_embedding, staging / "mpnet_base_v2.npy")
        shutil.copyfile(source_metadata, staging / "encoder_metadata.json")
        _validate_frozen_files(
            staging / "mpnet_base_v2.npy", staging / "encoder_metadata.json"
        )
    _validate_frozen_root(output_dir)
    return 0


def _handle_validate_frozen_input(args: argparse.Namespace) -> int:
    _validate_frozen_root(Path(args.input_dir))
    return 0


def _handle_worker(args: argparse.Namespace) -> int:
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output_dir)
    _preflight_worker_destinations(cache_dir=cache_dir, output_dir=output_dir)
    from trec_mlm.dense_aggregation_feasibility import run_feasibility_worker

    config = _load_config(Path(args.config))
    run_feasibility_worker(
        config,
        cache_dir=cache_dir,
        output_dir=output_dir,
        exclusive_resource_attested=args.exclusive_resource_attested,
    )
    return 0


def _handle_coordinate(args: argparse.Namespace) -> int:
    from trec_mlm.dense_aggregation_artifacts import (
        atomic_artifact_directory,
        raw_file_sha256,
    )
    from trec_mlm.dense_aggregation_feasibility import (
        _AGGREGATE_OUTPUT_FILES,
        _receipt_to_dict,
        _validate_task7_manifest,
        parse_gnu_time_verbose,
    )

    config = _load_config(Path(args.config))
    repo_root = Path(config.repo_root).resolve(strict=True)
    output_argument = Path(args.output_root)
    compact_argument = Path(args.compact_receipt)
    output_root = output_argument.resolve()
    compact_receipt = compact_argument.resolve()
    for name, path in (
        ("output root", output_root),
        ("compact receipt", compact_receipt),
    ):
        if not path.is_relative_to(repo_root):
            raise ValueError(f"{name} must remain under the repository root")

    repeat_paths: list[tuple[Path, Path, Path]] = []
    for label in ("a", "b"):
        prefix = f".{output_root.name}-repeat-{label}"
        time_path = output_root.parent / f"{prefix}.time.txt"
        cache_path = output_root.parent / f"{prefix}-cache"
        worker_output = output_root.parent / f"{prefix}-output"
        repeat_paths.append((time_path, cache_path, worker_output))

    destinations = [
        ("output root", output_argument, output_root),
        ("compact receipt", compact_argument, compact_receipt),
    ]
    for label, (time_path, cache_path, worker_output) in zip(
        ("a", "b"), repeat_paths, strict=True
    ):
        destinations.extend(
            (
                (f"repeat {label} time", time_path, time_path.resolve()),
                (f"repeat {label} cache", cache_path, cache_path.resolve()),
                (f"repeat {label} output", worker_output, worker_output.resolve()),
            )
        )
    for destination_name, _, destination in destinations:
        for input_directory in _configured_input_directories(config):
            if _paths_overlap(destination, input_directory):
                raise ValueError(
                    f"coordinator destination {destination_name} overlaps "
                    f"configured input directory: {input_directory}"
                )
    for index, (left_name, _, left) in enumerate(destinations):
        for right_name, _, right in destinations[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise ValueError(
                    f"coordinator destinations overlap: {left_name} and {right_name}"
                )
    for name, argument, resolved in destinations:
        for candidate in (argument, resolved):
            if _path_exists_lexically(candidate):
                raise FileExistsError(f"{name} exists: {resolved}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    compact_receipt.parent.mkdir(parents=True, exist_ok=True)

    environment = os.environ.copy()
    environment["LC_ALL"] = "C"
    receipts = []
    rss_values: list[int] = []
    worker_manifest_sha256: list[str] = []
    for time_path, cache_path, worker_output in repeat_paths:
        command = [
            "/usr/bin/time",
            "-v",
            "-o",
            _repo_argument(time_path, repo_root),
            "uv",
            "run",
            "--frozen",
            "--python",
            "3.13.9",
            "--extra",
            "dense-aggregation",
            "trec-mlm-dense-aggregation",
            "feasibility-worker",
            "--config",
            _repo_argument(Path(config.source_path), repo_root),
            "--cache-dir",
            _repo_argument(cache_path, repo_root),
            "--output-dir",
            _repo_argument(worker_output, repo_root),
            "--exclusive-resource-attested",
        ]
        completed = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"feasibility worker exited with status {completed.returncode}"
            )
        receipt = _load_worker_output(worker_output)
        receipts.append(receipt)
        rss_values.append(parse_gnu_time_verbose(time_path))
        worker_manifest_sha256.append(raw_file_sha256(worker_output / "manifest.json"))

    gate = _evaluate_gate(
        receipts[0],
        receipts[1],
        repeat_a_rss_bytes=rss_values[0],
        repeat_b_rss_bytes=rss_values[1],
        exclusive_resource_attested=args.attest_exclusive_resources,
    )
    with atomic_artifact_directory(output_root) as staging:
        _write_staging_json(staging, "feasibility_gate.json", _gate_to_dict(gate))
        _write_staging_json(
            staging, "repeat_a_receipt.json", _receipt_to_dict(receipts[0])
        )
        _write_staging_json(
            staging, "repeat_b_receipt.json", _receipt_to_dict(receipts[1])
        )
        _write_staging_json(
            staging,
            "repeat_measurements.json",
            {
                "schema": _MEASUREMENTS_SCHEMA,
                "repeat_a_manifest_sha256": worker_manifest_sha256[0],
                "repeat_b_manifest_sha256": worker_manifest_sha256[1],
                "repeat_a_rss_bytes": rss_values[0],
                "repeat_b_rss_bytes": rss_values[1],
            },
        )
    _validate_task7_manifest(
        output_root,
        expected_files=_AGGREGATE_OUTPUT_FILES,
        label="aggregate feasibility output",
    )
    if gate.status == "pass":
        _atomic_publish_json(
            compact_receipt,
            {
                "schema": _COMPACT_SCHEMA,
                "gate": _gate_to_dict(gate),
                "feasibility_manifest_sha256": raw_file_sha256(
                    output_root / "manifest.json"
                ),
            },
        )
        return 0
    return 1


def _validate_feasibility_artifact(
    config_path: Path, feasibility_root: Path, compact_receipt: Path
) -> None:
    from trec_mlm.dense_aggregation_artifacts import raw_file_sha256
    from trec_mlm.dense_aggregation_feasibility import (
        _AGGREGATE_OUTPUT_FILES,
        _read_bounded_canonical_json,
        _receipt_from_dict,
        _validate_task7_manifest,
    )

    config = _load_config(config_path)
    root = feasibility_root.resolve(strict=True)
    _validate_task7_manifest(
        root,
        expected_files=_AGGREGATE_OUTPUT_FILES,
        label="aggregate feasibility output",
    )
    gate_value = _read_bounded_canonical_json(
        root / "feasibility_gate.json",
        label="aggregate feasibility gate",
    )
    gate = _gate_from_dict(gate_value)
    repeat_a = _receipt_from_dict(
        _read_bounded_canonical_json(
            root / "repeat_a_receipt.json",
            label="aggregate repeat A receipt",
        )
    )
    repeat_b = _receipt_from_dict(
        _read_bounded_canonical_json(
            root / "repeat_b_receipt.json",
            label="aggregate repeat B receipt",
        )
    )
    measurements = _read_bounded_canonical_json(
        root / "repeat_measurements.json",
        label="aggregate repeat measurements",
    )
    measurement_keys = {
        "schema",
        "repeat_a_manifest_sha256",
        "repeat_b_manifest_sha256",
        "repeat_a_rss_bytes",
        "repeat_b_rss_bytes",
    }
    if (
        type(measurements) is not dict
        or set(measurements) != measurement_keys
        or measurements["schema"] != _MEASUREMENTS_SCHEMA
    ):
        raise ValueError("repeat measurements have an invalid contract")
    for label, receipt in (("a", repeat_a), ("b", repeat_b)):
        worker_root_value = receipt.provenance.get("output_root")
        if type(worker_root_value) is not str:
            raise ValueError("worker receipt output root is invalid")
        worker_root = Path(worker_root_value)
        persisted = _load_worker_output(worker_root)
        if persisted != receipt:
            raise ValueError("aggregate worker receipt copy differs from worker output")
        if (
            raw_file_sha256(worker_root / "manifest.json")
            != measurements[f"repeat_{label}_manifest_sha256"]
        ):
            raise ValueError("worker manifest digest differs from aggregate binding")
    rss_a = measurements["repeat_a_rss_bytes"]
    rss_b = measurements["repeat_b_rss_bytes"]
    if type(rss_a) is not int or rss_a < 0 or type(rss_b) is not int or rss_b < 0:
        raise ValueError("repeat RSS values are invalid")
    observed_gate = _evaluate_gate(
        repeat_a,
        repeat_b,
        repeat_a_rss_bytes=rss_a,
        repeat_b_rss_bytes=rss_b,
        exclusive_resource_attested=True,
    )
    if observed_gate != gate or gate.status != "pass":
        raise ValueError("feasibility gate is stale or stopped")

    compact = _read_bounded_canonical_json(
        compact_receipt.resolve(strict=True),
        label="compact feasibility receipt",
    )
    compact_keys = {"schema", "gate", "feasibility_manifest_sha256"}
    if (
        type(compact) is not dict
        or set(compact) != compact_keys
        or compact["schema"] != _COMPACT_SCHEMA
        or _gate_from_dict(compact["gate"]) != gate
        or compact["feasibility_manifest_sha256"]
        != raw_file_sha256(root / "manifest.json")
    ):
        raise ValueError("compact feasibility receipt is stale or invalid")

    from trec_mlm.dense_aggregation_clustering import (
        capture_numerical_environment,
        numerical_environment_file_bytes,
    )
    from trec_mlm.dense_aggregation_contract import (
        build_clustering_contract,
        contract_source_inventory_sha256,
        load_and_validate_frozen_dense_inputs,
        runtime_source_inventory_sha256,
    )

    inputs = load_and_validate_frozen_dense_inputs(config)
    contract, contract_sha256 = build_clustering_contract(
        inputs,
        repo_root=config.repo_root,
        protocol_path=config.protocol_path,
    )
    versions = contract["versions"]
    numerical_environment = capture_numerical_environment()
    numerical_environment_sha256 = hashlib.sha256(
        numerical_environment_file_bytes(numerical_environment)
    ).hexdigest()
    expected = {
        "config_sha256": config.source_sha256,
        "protocol_sha256": raw_file_sha256(config.protocol_path),
        "clustering_contract_sha256": contract_sha256,
        "implementation_inventory_sha256": contract_source_inventory_sha256(
            contract, category="implementation"
        ),
        "dependency_inventory_sha256": contract_source_inventory_sha256(
            contract, category="dependency"
        ),
        "runtime_inventory_sha256": runtime_source_inventory_sha256(config.repo_root),
        "query_file_sha256": raw_file_sha256(config.query_path),
        "query_input_sha256": inputs.query_input_sha256,
        "embedding_sha256": inputs.embedding_sha256,
        "similarity_sha256": inputs.similarity_sha256,
        "sample_fixture_sha256": (
            "6b74301b9edb16438739a2c46b455ff37985698a6795bd7a4b9d1f5530cf9d39"
        ),
        "numerical_environment_sha256": numerical_environment_sha256,
        "python_version": versions["python"],
        "numpy_version": versions["numpy"],
        "scipy_version": versions["scipy"],
        "scikit_learn_version": versions["scikit_learn"],
        "thread_environment": dict(numerical_environment.thread_environment),
    }
    for receipt in (repeat_a, repeat_b):
        if any(receipt.provenance.get(key) != value for key, value in expected.items()):
            raise ValueError("feasibility provenance differs from current inputs")


def _handle_validate_feasibility(args: argparse.Namespace) -> int:
    _validate_feasibility_artifact(
        Path(args.config),
        Path(args.feasibility_root),
        Path(args.compact_receipt),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {
        "build-feasibility-config": _handle_build_config,
        "prepare-frozen-input": _handle_prepare_frozen_input,
        "validate-frozen-input": _handle_validate_frozen_input,
        "feasibility": _handle_coordinate,
        "feasibility-worker": _handle_worker,
        "validate-feasibility": _handle_validate_feasibility,
    }
    return handlers[args.command](args)
