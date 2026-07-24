"""Build or check deterministic TREC 2026 Research Hub outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

from trec_mlm.research_hub import (
    assert_hub_outputs_current,
    build_research_hub,
    write_hub_outputs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the TREC 2026 Research Hub.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build = build_research_hub(args.root)
    if args.check:
        assert_hub_outputs_current(build, args.root)
    else:
        write_hub_outputs(build, args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
