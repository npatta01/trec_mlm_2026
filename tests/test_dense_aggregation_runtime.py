from __future__ import annotations

import json
import subprocess
import sys
import types

import pytest

from trec_mlm.dense_aggregation_runtime import (
    THREAD_ENVIRONMENT,
    pin_single_thread_environment,
    validate_preimport_environment,
)


def test_entrypoint_pins_threads_before_cli_import() -> None:
    code = r'''
import importlib.abc, importlib.util, json, os, sys
names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
         "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")
numerical_prefixes = ("numpy", "scipy", "sklearn", "threadpoolctl")
for name in names:
    os.environ.pop(name, None)

class CliLoader(importlib.abc.Loader):
    def exec_module(self, module):
        values = {name: os.environ.get(name) for name in names}
        assert set(values.values()) == {"1"}, values
        imported = sorted(
            name for name in sys.modules
            if name.split(".", 1)[0] in numerical_prefixes
        )
        assert not imported, imported
        module.main = lambda argv=None: 0

class CliFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "trec_mlm.dense_aggregation_cli":
            return importlib.util.spec_from_loader(fullname, CliLoader())
        return None

sys.meta_path.insert(0, CliFinder())
from trec_mlm.dense_aggregation_entrypoint import main
assert main([]) == 0
print(json.dumps({name: os.environ[name] for name in names}, sort_keys=True))
'''
    completed = subprocess.run(
        [sys.executable, "-c", code], text=True, capture_output=True
    )
    assert completed.returncode == 0, completed.stderr
    assert set(json.loads(completed.stdout).values()) == {"1"}


def test_runtime_rejects_conflicting_thread_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    with pytest.raises(RuntimeError, match="OMP_NUM_THREADS must be unset or 1"):
        pin_single_thread_environment()


def test_runtime_rejects_preimported_numerical_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")
    monkeypatch.setitem(sys.modules, "numpy", types.ModuleType("numpy"))

    with pytest.raises(RuntimeError, match="numerical modules imported before bootstrap"):
        validate_preimport_environment()
