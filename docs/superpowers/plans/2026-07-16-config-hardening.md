# Configuration Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Harden candidate-profile configuration loading and add a cheap qrel-free config validation command without changing experiment behavior.

**Architecture:** Keep strict schema checks at the artifact-loading boundaries and add a read-only validation path to the existing multi-resolution runner. Preserve `main()` for programmatic exceptions while a thin CLI entrypoint formats expected failures.

**Tech Stack:** Python 3.11+, dataclasses, argparse, pytest, Ruff, uv.

## Global Constraints

- Do not change embeddings, clustering, centroid arithmetic, scoring, ranking, RRF, qrel handling, frozen constants, or persisted bytes.
- `--validate-config` must not construct artifacts or load qrels.
- Missing ignored artifacts must be reported without making the tracked config invalid in a fresh clone.

---

### Task 1: Exact persisted configuration schemas

**Files:**
- Modify: `tests/test_profile_experiment.py`
- Modify: `tests/test_profile_inference_artifact.py`
- Modify: `trec_mlm/profile_experiment.py`
- Modify: `trec_mlm/profile_inference_artifact.py`

**Interfaces:**
- Consumes: persisted `config.json` dictionaries.
- Produces: exact-key validation before dataclass construction.

- [ ] Add tests that mutate valid fixture artifacts with an extra key or a changed identity sentinel, refresh the manifest, and assert the loaders reject them with a configuration-schema error.
- [ ] Run the focused tests and verify they fail because the loaders currently ignore extra fields.
- [ ] Add exact expected-key sets containing the three identity sentinels and every dataclass field; validate the set and sentinel values before selecting dataclass values.
- [ ] Run both focused test modules and verify they pass.

### Task 2: Read-only multi-resolution config validation

**Files:**
- Modify: `tests/test_profile_multiresolution.py`
- Modify: `scripts/run_multiresolution_profiles.py`
- Modify: `trec_mlm/profile_multiresolution.py`

**Interfaces:**
- Produces: `validate_multiresolution_config(config: MultiResolutionConfig) -> None`, `_validate_config_report(config: MultiResolutionConfig) -> dict[str, object]`, and `--validate-config`.

- [ ] Add tests proving fresh-clone validation succeeds without source/output artifacts and does not call construction.
- [ ] Add tests proving a present source manifest must expose the configured aggregate and that output presence is reported.
- [ ] Run those tests and verify failure because the flag/report do not exist.
- [ ] Promote the existing private config validator to the public `validate_multiresolution_config()` boundary, then add the parser flag and a finite JSON report path before construction/repeat logic.
- [ ] Run the focused tests and verify they pass.

### Task 3: Concise CLI failures and reproducible developer setup

**Files:**
- Modify: `tests/test_profile_multiresolution.py`
- Modify: `scripts/run_multiresolution_profiles.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `README.md`

**Interfaces:**
- Produces: `cli(argv: list[str] | None = None) -> int`; optional dependency group `dev`.

- [ ] Add a test asserting invalid JSON passed through `cli()` prints one `error:` line and returns 2.
- [ ] Run it and verify failure because the boundary does not exist.
- [ ] Add the narrow error boundary and route `__main__` through it.
- [ ] Add `dev = ["pytest>=8.4,<9", "ruff>=0.12,<1"]`, refresh `uv.lock`, and document `uv sync --extra dense --extra dev`, config validation, pytest, and Ruff commands.
- [ ] Run focused tests, `uv lock --check`, Ruff, and the full suite.

### Task 4: Portability and review gate

**Files:**
- Verify only.

**Interfaces:**
- Produces: evidence that the tracked tree works both with and without ignored artifacts.

- [ ] Export the committed tree to a temporary artifact-free directory and run the config validation tests and full suite there.
- [ ] Run the full suite in the artifact-present workspace.
- [ ] Obtain independent code review against this design and resolve every critical or important finding.
- [ ] Re-run fresh verification, commit intentionally, push `codex/config-hardening`, and open a draft PR against `main` without merging.
