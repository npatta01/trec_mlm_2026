from __future__ import annotations

from collections.abc import Sequence

from trec_mlm.dense_aggregation_runtime import (
    pin_single_thread_environment,
    validate_preimport_environment,
)


def main(argv: Sequence[str] | None = None) -> int:
    pin_single_thread_environment()
    validate_preimport_environment()
    from trec_mlm.dense_aggregation_cli import main as cli_main

    return cli_main(argv)
