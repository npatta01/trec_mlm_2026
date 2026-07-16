# Artifact I/O Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Consolidate duplicated candidate-profile artifact primitives behind byte-for-byte golden tests without changing scientific or persisted behavior.

**Architecture:** A dependency-free helper module owns only deterministic byte/hash primitives. Existing modules import those helpers and retain compatibility aliases where tests or sealing guards depend on private names.

**Tech Stack:** Python 3.11+, hashlib, json, pathlib, pytest, Ruff.

## Global Constraints

- Preserve every persisted byte, hash, schema, source-tree root, chunk size, dtype, ordering, scientific computation, ranking, RRF result, and qrel boundary.
- Do not alter atomic/no-overwrite receipt writers or JSON finiteness behavior.
- Do not regenerate frozen artifacts.

---

### Task 1: Golden artifact primitives

**Files:**
- Create: `trec_mlm/artifact_io.py`
- Create: `tests/test_artifact_io.py`

**Interfaces:**
- Produces: `sha256_file`, `write_pretty_json`, `length_framed_sha256`, `manifest_entries`, `manifest_aggregate_sha256`, `source_tree_sha256`.

- [ ] Add failing golden tests for: UTF-8 pretty JSON exact hash `7b494852f8424e42f70ebb61d358c329c048ceaacffed872df5e764c451d673e`; framed `("a", "β", "")` hash `0741cb3a91300dca6785fe5931bd1bfe28a1f620834ac49737d55205af638212`; manifest aggregate `5e33ce02b94bd9d959981775654cf52f7882221cef483246524e81a76eb6b0db`; standard `abc` SHA-256; a file larger than 1 MiB; exact inventory exclusion/sorting; and source-tree fixture hash `82cc50ade6b1554b9a32c98f61e9ee3eb5c9aad7df0b347f111bbc0d3d56f3c8`.
- [ ] Run the new module and confirm import/behavior failures.
- [ ] Implement only the frozen primitives with the serialization described in the design.
- [ ] Run the golden tests and Ruff.

### Task 2: Candidate-profile module migration

**Files:**
- Modify: `trec_mlm/profile_experiment.py`
- Modify: `trec_mlm/profile_inference_artifact.py`
- Modify: `trec_mlm/profile_multiresolution.py`
- Modify: `trec_mlm/profile_multiresolution_evaluation.py`
- Modify: `trec_mlm/profile_retrieval.py`
- Modify: focused tests only if compatibility assertions are needed.

**Interfaces:**
- Consumes: Task 1 public helpers.
- Produces: unchanged module behavior and compatibility aliases for monkeypatched private hash names.

- [ ] Add equivalence assertions comparing existing representative helper outputs to the shared helpers, then run them before migration.
- [ ] Replace only the duplicated implementations listed in the design; retain `_sha256 = sha256_file` where tests monkeypatch the module boundary.
- [ ] Run the golden tests plus profile experiment, inference artifact, multiresolution, evaluation, and retrieval test modules.
- [ ] Confirm representative written artifacts and manifests have unchanged exact bytes/hashes.

### Task 3: Public cross-module boundaries

**Files:**
- Modify: `trec_mlm/profile_experiment.py`
- Modify: `trec_mlm/profile_inference_artifact.py`
- Modify: `trec_mlm/profile_multiresolution.py`
- Modify: related tests.

**Interfaces:**
- Produces public aliases for currently cross-imported private functions; no body/arithmetic changes.

- [ ] Add failing public-export tests for the existing cross-module dependencies.
- [ ] Expose public names and update consumers without moving or rewriting their bodies.
- [ ] Keep private aliases if external or test compatibility requires them.
- [ ] Run all affected profile tests and Ruff.

### Task 4: Portability and publication gate

**Files:**
- Verify only.

**Interfaces:**
- Produces evidence that helper consolidation is byte-identical and portable.

- [ ] Run the full artifact-present suite, golden tests, scoped Ruff, and diff check.
- [ ] Run the full suite from a true artifact-free local clone; require only the three intentional local-artifact skips.
- [ ] Obtain independent task and whole-branch review and resolve every Critical or Important finding.
- [ ] Push `codex/artifact-io-cleanup` and open a draft PR against `main` without merging.
