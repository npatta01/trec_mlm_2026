# Final review fixes evidence

Date: 2026-07-16
Branch: `codex/config-hardening`

## Findings resolved

- The multiresolution CLI now rejects a decoded JSON root that is not an object,
  requires `resolutions` to be a JSON list of integers, and validates every
  configuration field's expected JSON type before tuple conversion or `Path`
  usage. User configuration faults therefore become `ValueError` while the
  intentionally narrow `cli()` catch remains `(OSError, ValueError)`.
- The regression test covers JSON `null`, numeric/non-iterable `resolutions`,
  and numeric `output_root`. Every case asserts exit 2, no stdout, exactly one
  stderr error line, no traceback, and no construction.
- Both persisted configuration loader mutation tests now cover `version` and
  `qrel_free` sentinels in addition to `artifact_type` and an extra key.

## TDD evidence

RED:

```text
uv run --extra dev pytest -q tests/test_profile_multiresolution.py::test_cli_wrong_json_types_print_one_error_line_without_construction
3 failed in 0.15s
```

The failures were the expected uncaught `TypeError` paths at `set(payload)`,
`tuple(payload["resolutions"])`, and `Path(config.output_root)`.

GREEN:

```text
uv run --extra dev pytest -q tests/test_profile_multiresolution.py::test_cli_wrong_json_types_print_one_error_line_without_construction tests/test_profile_experiment.py::test_loader_rejects_nonexact_configuration_schema tests/test_profile_inference_artifact.py::test_inference_loader_rejects_nonexact_configuration_schema
11 passed in 0.12s
```

## Verification evidence

```text
uv run --extra dev pytest -q tests/test_profile_multiresolution.py tests/test_profile_experiment.py tests/test_profile_inference_artifact.py
exit 0

uv run --extra dev pytest -q
354 passed in 118.99s (0:01:58)

uv run --extra dev ruff check trec_mlm scripts tests
All checks passed!

uv lock --check
Resolved 142 packages in 0.72ms

uv run --extra dense python scripts/run_multiresolution_profiles.py --validate-config --config configs/trec_2026_multiresolution_profiles.json
exit 0; status=valid; qrel_free=true; source_artifact.status=verified

git diff --check
exit 0
```

No artifacts were constructed, published, pushed, or exposed during these
fixes.
