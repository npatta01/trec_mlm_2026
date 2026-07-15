#!/usr/bin/env python3
"""Construct the locked issue-18 qrel-free multi-resolution artifact."""

from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from trec_mlm.profile_multiresolution import (
    MultiResolutionConfig,
    construct_multiresolution,
    write_multiresolution_artifact,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/trec_2026_multiresolution_profiles.json",
        help="locked qrel-free construction config",
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
    result = construct_multiresolution(config)
    paths = write_multiresolution_artifact(result, config.output_root)
    print(json.dumps({"output_root": config.output_root, "files": paths}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
