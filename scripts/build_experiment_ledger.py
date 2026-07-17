"""Build or check the deterministic canonical TREC 2026 experiment ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

from trec_mlm.experiment_ledger import (
    assert_outputs_current,
    build_ledger,
    write_ledger_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the canonical TREC 2026 experiment ledger."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("experiments/ledger.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    build = build_ledger(args.root, args.registry)
    output_dir = args.root / args.output_dir
    if args.check:
        assert_outputs_current(build, output_dir)
    else:
        write_ledger_outputs(build, output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
