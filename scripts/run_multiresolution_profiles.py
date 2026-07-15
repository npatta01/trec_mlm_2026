#!/usr/bin/env python3
"""Construct the locked issue-18 qrel-free multi-resolution artifact."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path
import time

from trec_mlm.profile_multiresolution import (
    ISSUE18_FROZEN_ARTIFACT_PATH,
    MultiResolutionConfig,
    assess_multiresolution_viability,
    compare_multiresolution_artifacts,
    construct_multiresolution,
    load_multiresolution_artifact,
    validate_multiresolution_viability_receipt,
    write_multiresolution_artifact,
    write_multiresolution_receipt,
)


CANONICAL_VIABILITY_RECEIPT_PATH = Path(
    "reports/trec_2026_issue18_pr17head_viability.json"
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/trec_2026_multiresolution_profiles.json",
        help="locked qrel-free construction config",
    )
    parser.add_argument(
        "--viability-receipt",
        help="write the qrel-free viability and cost receipt to this new path",
    )
    parser.add_argument(
        "--first-artifact",
        help="comparison mode: already-validated pass-one artifact",
    )
    parser.add_argument(
        "--first-viability-receipt",
        help="comparison mode: pass-one viability and cost authorization receipt",
    )
    parser.add_argument(
        "--repeat-artifact",
        help="compare the new artifact with this independently constructed repeat",
    )
    parser.add_argument(
        "--repeatability-receipt",
        help="write the qrel-free repeatability receipt to this new path",
    )
    return parser


def _comparison_mode(args: argparse.Namespace) -> bool:
    values = (
        args.first_artifact,
        args.first_viability_receipt,
        args.repeat_artifact,
        args.repeatability_receipt,
    )
    if (args.repeat_artifact or args.repeatability_receipt) and not all(values):
        raise ValueError("repeat comparison mode requires all four repeat arguments")
    return bool(args.repeat_artifact and args.repeatability_receipt)


def _load_repeat_authorization(args: argparse.Namespace) -> dict[str, object]:
    if not args.first_artifact or not args.first_viability_receipt:
        raise ValueError(
            "repeat construction or comparison requires first artifact and viability receipt"
        )
    try:
        authorization = json.loads(
            Path(args.first_viability_receipt).read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid first viability receipt") from error
    try:
        validate_multiresolution_viability_receipt(
            authorization, args.first_artifact
        )
    except ValueError as error:
        raise ValueError("invalid first viability receipt") from error
    cost = authorization.get("cost_evidence")
    authorized = (
        authorization.get("artifact_type")
        == "issue18_multiresolution_viability_receipt"
        and authorization.get("version") == 1
        and authorization.get("qrel_free") is True
        and authorization.get("decision") == "continue"
        and authorization.get("repeat_authorized") is True
        and isinstance(cost, dict)
        and cost.get("storage_at_most_four_times_issue12") is True
        and cost.get("wall_time_at_most_four_times_issue12") is True
        and cost.get("repeat_cost_authorized") is True
    )
    if not authorized:
        raise ValueError("pass-one viability and cost receipt does not authorize repeat")
    first_artifact = load_multiresolution_artifact(args.first_artifact)
    if first_artifact.config != MultiResolutionConfig():
        raise ValueError("repeat authorization source is not canonical pass-one")
    try:
        first_manifest = json.loads(
            (Path(args.first_artifact) / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("invalid first artifact manifest") from error
    if first_manifest.get("aggregate_sha256") != authorization.get(
        "artifact_aggregate_sha256"
    ):
        raise ValueError("viability authorization does not bind the first artifact")
    return authorization


def _run_repeat_comparison(args: argparse.Namespace) -> int:
    authorization = _load_repeat_authorization(args)
    repeatability = compare_multiresolution_artifacts(
        args.first_artifact, args.repeat_artifact
    )
    if repeatability.get("first_aggregate_sha256") != authorization.get(
        "artifact_aggregate_sha256"
    ):
        raise ValueError("viability authorization does not bind the first artifact")
    write_multiresolution_receipt(repeatability, args.repeatability_receipt)
    print(json.dumps({"repeatability": repeatability}, indent=2, allow_nan=False))
    return 0


def _load_config(args: argparse.Namespace) -> MultiResolutionConfig:
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    expected = {field.name for field in fields(MultiResolutionConfig)}
    if set(payload) != expected:
        raise ValueError("config fields differ from MultiResolutionConfig")
    payload["resolutions"] = tuple(payload["resolutions"])
    return MultiResolutionConfig(**payload)


def _run_construction(
    args: argparse.Namespace, config: MultiResolutionConfig
) -> int:
    started = time.perf_counter()
    result = construct_multiresolution(config)
    paths = write_multiresolution_artifact(result, config.output_root)
    elapsed = time.perf_counter() - started
    viability = assess_multiresolution_viability(config.output_root)
    cost = viability["cost_evidence"]
    baseline = cost["issue12_construction_elapsed_seconds"]
    cost["multiresolution_construction_elapsed_seconds"] = elapsed
    cost["wall_time_ratio"] = elapsed / baseline if baseline else None
    cost["wall_time_at_most_four_times_issue12"] = (
        elapsed <= 4 * baseline if baseline else None
    )
    cost["repeat_cost_authorized"] = (
        cost["storage_at_most_four_times_issue12"] is True
        and cost["wall_time_at_most_four_times_issue12"] is True
    )
    viability["repeat_authorized"] = (
        viability["decision"] == "continue" and cost["repeat_cost_authorized"]
    )
    if args.viability_receipt:
        write_multiresolution_receipt(viability, args.viability_receipt)
    print(
        json.dumps(
            {
                "output_root": config.output_root,
                "files": paths,
                "viability": viability,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


def _run_viability_repair(args: argparse.Namespace, artifact_root: str) -> int:
    """Write a supplemental receipt from the immutable pass-one artifact only."""

    if not args.viability_receipt:
        raise ValueError("supplemental viability repair requires an output receipt")
    _validate_viability_repair_paths(artifact_root, args.viability_receipt)
    viability = assess_multiresolution_viability(artifact_root)
    write_multiresolution_receipt(viability, args.viability_receipt)
    print(json.dumps({"viability": viability}, indent=2, allow_nan=False))
    return 0


def _validate_viability_repair_paths(
    artifact_root: str | Path, receipt_path: str | Path
) -> None:
    artifact = Path(artifact_root)
    receipt = Path(receipt_path)
    if artifact.resolve() != Path(ISSUE18_FROZEN_ARTIFACT_PATH).resolve():
        raise ValueError("supplemental repair requires the canonical issue-18 artifact")
    if receipt.resolve() != CANONICAL_VIABILITY_RECEIPT_PATH.resolve():
        raise ValueError("supplemental repair requires the canonical viability receipt")
    if receipt.resolve() == artifact.resolve() or artifact.resolve() in receipt.resolve().parents:
        raise ValueError("canonical viability receipt must be outside the artifact")
    if receipt.exists():
        raise FileExistsError(f"receipt already exists: {receipt}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if _comparison_mode(args):
        return _run_repeat_comparison(args)
    config = _load_config(args)
    if config == MultiResolutionConfig():
        if args.first_artifact or args.first_viability_receipt:
            raise ValueError(
                "canonical pass-one construction cannot accept repeat authorization"
            )
        if Path(config.output_root).exists():
            if args.viability_receipt:
                return _run_viability_repair(args, config.output_root)
            raise FileExistsError(
                f"canonical pass-one artifact already exists: {config.output_root}"
            )
    else:
        _load_repeat_authorization(args)
    return _run_construction(args, config)


if __name__ == "__main__":
    raise SystemExit(main())
