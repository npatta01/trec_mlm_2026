#!/usr/bin/env python3
"""Construct the locked issue-18 qrel-free multi-resolution artifact."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
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
        "--repeat-artifact",
        help="compare the new artifact with this independently constructed repeat",
    )
    parser.add_argument(
        "--repeatability-receipt",
        help="write the qrel-free repeatability receipt to this new path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
    if bool(args.repeat_artifact) != bool(args.repeatability_receipt):
        raise ValueError(
            "--repeat-artifact and --repeatability-receipt must be supplied together"
        )
    repeatability = None
    if args.repeat_artifact:
        repeatability = compare_multiresolution_artifacts(
            config.output_root, args.repeat_artifact
        )
        write_multiresolution_receipt(repeatability, args.repeatability_receipt)
    print(
        json.dumps(
            {
                "output_root": config.output_root,
                "files": paths,
                "viability": viability,
                "repeatability": repeatability,
            },
            indent=2,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
