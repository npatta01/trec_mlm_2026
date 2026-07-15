from __future__ import annotations

import os
import sys


THREAD_ENVIRONMENT: tuple[str, ...] = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
NUMERICAL_MODULE_PREFIXES = ("numpy", "scipy", "sklearn", "threadpoolctl")


def pin_single_thread_environment() -> None:
    for name in THREAD_ENVIRONMENT:
        current = os.environ.get(name)
        if current not in (None, "1"):
            raise RuntimeError(f"{name} must be unset or 1, got {current!r}")
        os.environ[name] = "1"


def validate_preimport_environment() -> dict[str, str]:
    imported = sorted(
        name
        for name in sys.modules
        if name.split(".", 1)[0] in NUMERICAL_MODULE_PREFIXES
    )
    if imported:
        raise RuntimeError(f"numerical modules imported before bootstrap: {imported!r}")
    values = {name: os.environ.get(name, "") for name in THREAD_ENVIRONMENT}
    if set(values.values()) != {"1"}:
        raise RuntimeError(f"single-thread environment is not pinned: {values!r}")
    return values
