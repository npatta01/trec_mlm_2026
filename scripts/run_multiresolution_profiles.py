#!/usr/bin/env python3
"""Construct the locked issue-18 qrel-free multi-resolution artifact."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
import math
from pathlib import Path
import time

from trec_mlm.profile_multiresolution import (
    MultiResolutionConfig,
    assess_multiresolution_viability,
    compare_multiresolution_artifacts,
    construct_multiresolution,
    write_multiresolution_artifact,
    write_multiresolution_receipt,
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
        "--repeat-construction",
        action="store_true",
        help="construct a fresh repeat only after validating pass-one authorization",
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
    if any(values) and not all(values):
        raise ValueError("repeat comparison mode requires all four repeat arguments")
    return all(values)


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
    cost = authorization.get("cost_evidence")
    storage_ratio = cost.get("storage_ratio") if isinstance(cost, dict) else None
    wall_time_ratio = cost.get("wall_time_ratio") if isinstance(cost, dict) else None
    ratios_valid = all(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0.0 <= value <= 4.0
        for value in (storage_ratio, wall_time_ratio)
    )
    authorized = (
        authorization.get("artifact_type")
        == "issue18_multiresolution_viability_receipt"
        and authorization.get("version") == 1
        and authorization.get("qrel_free") is True
        and authorization.get("decision") == "continue"
        and authorization.get("repeat_authorized") is True
        and isinstance(cost, dict)
        and ratios_valid
        and cost.get("storage_at_most_four_times_issue12") is True
        and cost.get("wall_time_at_most_four_times_issue12") is True
        and cost.get("repeat_cost_authorized") is True
    )
    if not authorized:
        raise ValueError("pass-one viability and cost receipt does not authorize repeat")
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


def _run_pass_one(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.config).read_text(encoding="utf-8"))
    expected = {field.name for field in fields(MultiResolutionConfig)}
    if set(payload) != expected:
        raise ValueError("config fields differ from MultiResolutionConfig")
    payload["resolutions"] = tuple(payload["resolutions"])
    config = MultiResolutionConfig(**payload)
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


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.repeat_construction:
        if args.repeat_artifact or args.repeatability_receipt:
            raise ValueError(
                "repeat construction and repeat comparison are separate CLI modes"
            )
        _load_repeat_authorization(args)
        return _run_pass_one(args)
    if _comparison_mode(args):
        return _run_repeat_comparison(args)
    return _run_pass_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
