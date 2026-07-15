from __future__ import annotations

import json
import subprocess
import sys

import pytest

from trec_mlm.dense_aggregation_runtime import pin_single_thread_environment


def test_entrypoint_pins_threads_before_cli_import() -> None:
    code = r'''
import json, os, sys, types
names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
         "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
for name in names:
    os.environ.pop(name, None)
fake = types.ModuleType("trec_mlm.dense_aggregation_cli")
fake.main = lambda argv=None: 0
sys.modules["trec_mlm.dense_aggregation_cli"] = fake
from trec_mlm.dense_aggregation_entrypoint import main
assert "numpy" not in sys.modules
assert main([]) == 0
print(json.dumps({name: os.environ[name] for name in names}, sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code], check=True, text=True, capture_output=True
    )
    assert set(json.loads(completed.stdout).values()) == {"1"}


def test_runtime_rejects_conflicting_thread_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS must be unset or 1"):
        pin_single_thread_environment()
