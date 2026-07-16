# Configuration Hardening Design

## Objective

Make the merged candidate-profile configuration safer and easier to validate without constructing artifacts or changing any experiment result.

## Scope

- Require exact persisted configuration schemas for candidate-profile construction and inference artifacts, including the frozen `artifact_type`, `version`, and `qrel_free` sentinels.
- Add `--validate-config` to `scripts/run_multiresolution_profiles.py`. It validates the input JSON and frozen `MultiResolutionConfig` invariants, reports whether the bound source and output paths are present, and never constructs, ranks, evaluates, or loads qrels.
- If the source artifact is present, compare its manifest aggregate with the configured source aggregate. A missing source is reported as runtime-not-ready but is not a schema failure, so the tracked config remains inspectable in a fresh clone.
- Add a narrow command-line boundary that converts expected configuration, JSON, path, and overwrite failures into one-line `error:` messages and exit status 2. Keep `main()` exception behavior available to tests and programmatic callers.
- Add a `dev` optional dependency group containing pytest and Ruff, and document exact setup and validation commands.

## Non-goals

- No change to embeddings, clustering, centroid reconstruction, scoring, ranking, RRF, qrel sealing, artifact schemas, or frozen values.
- No reconstruction or expensive full-artifact validation during `--validate-config`.
- No loader/reconstruction separation and no absolute-path provenance redesign; both require a separate protocol-reviewed change.

## Interface

`python scripts/run_multiresolution_profiles.py --validate-config --config PATH` prints finite JSON with `status`, `qrel_free`, `config`, `source_artifact`, and `output` fields. It exits 0 when the config schema and frozen invariants are valid. `source_artifact.status` is `verified` when the source manifest exists and matches, otherwise `missing`; `output.exists` reports overwrite readiness.

The direct script entrypoint catches expected user-input errors and prints a concise message. Internal `main(argv)` continues to raise, preserving precise unit testing and existing call semantics.

## Verification

- TDD tests for extra-key and sentinel rejection in both artifact loaders.
- TDD tests proving `--validate-config` performs no construction, succeeds in a fresh-clone-shaped tree, verifies a present source manifest, and rejects a mismatched aggregate.
- TDD test for concise CLI error output.
- Lockfile consistency, Ruff, focused profile tests, full artifact-present suite, and a fresh-clone archive suite.

## Approval

This design instantiates the previously proposed low-risk config-hardening PR. The user approved execution with “do the prs as u planned” on 2026-07-16.
