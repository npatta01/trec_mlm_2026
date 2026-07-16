# PR 24 review blocker fix report

## Outcome

Restored the restricted one-argument `load_label_experiment_artifacts()` public
boundary and added the explicitly named
`load_frozen_legacy_label_experiment_artifacts()` compatibility boundary for the
single pinned profile-inference use.

The private `_load_label_experiment_artifacts()` implementation and its
`allow_frozen_legacy_provenance` behavior were not changed. No artifact,
provenance-validation body, scientific code, or qrel behavior was changed.

## Root cause and scope

Commit `9365e7c` replaced the former one-argument public wrapper with a direct
alias to `_load_label_experiment_artifacts`. That made the private legacy opt-in
keyword part of the ordinary public API. The fix:

- restores the former public wrapper verbatim;
- adds and exports a one-argument compatibility wrapper that alone passes the
  private legacy opt-in;
- updates `profile_inference_artifact` to import and call that explicit wrapper;
- updates behavioral and AST boundary tests to seal both public APIs.

The nonblocking AGY `_manifest_entries` / `_sha256` monkeypatch seam was
assessed. All relevant production call sites still resolve through the
module-level aliases, so monkeypatches remain effective. No change was needed.

## TDD evidence

### RED

After changing tests only:

```bash
.venv/bin/pytest -q \
  tests/test_label_experiment.py::QrelFreeLabelExperimentTest::test_public_loader_signature_has_no_legacy_provenance_opt_in \
  tests/test_label_experiment.py::QrelFreeLabelExperimentTest::test_public_loader_rejects_legacy_provenance_opt_in_keyword \
  tests/test_label_experiment.py::QrelFreeLabelExperimentTest::test_frozen_legacy_provenance_requires_explicit_compatibility_loader \
  tests/test_public_cross_module_boundaries.py
```

Observed: `5 failed, 6 passed` (exit 1). The failures showed that the ordinary
signature included `allow_frozen_legacy_provenance`, the ordinary loader
accepted that keyword, the explicit compatibility loader/export was absent,
and the consumer AST still imported the ordinary loader.

### GREEN

After the minimal wrapper/import fix:

```text
$ .venv/bin/pytest -q tests/test_label_experiment.py \
    tests/test_profile_inference_artifact.py \
    tests/test_public_cross_module_boundaries.py
.............................                                            [100%]
29 passed in 0.31s
```

## Final verification

```text
$ .venv/bin/pytest -q
353 passed in 118.73s (0:01:58)

$ .venv/bin/ruff check trec_mlm/label_experiment.py \
    trec_mlm/profile_inference_artifact.py \
    tests/test_label_experiment.py \
    tests/test_public_cross_module_boundaries.py
All checks passed!

$ git diff --check
[no output; exit 0]
```

A repository-wide Ruff run additionally reports the pre-existing unused
`IPython.display.display` import in
`notebooks/trec_2026_file_manifest.ipynb` cell 4. That unrelated notebook was
not changed as part of this blocker fix.

## Self-review

- Ordinary callers cannot spell or introspect the legacy opt-in keyword.
- The compatibility API is deliberately conspicuous and accepts only
  `artifact_dir`.
- The sole frozen compatibility caller uses the explicit API without a boolean
  argument.
- Public exports and consumer imports are sealed by behavioral and AST tests.
- Private loader logic and all out-of-scope scientific/artifact behavior remain
  unchanged.
