# Artifact I/O Cleanup Design

## Objective

Remove duplicated artifact hashing and JSON bookkeeping from the candidate-profile modules while preserving every persisted byte, hash, schema, scientific computation, and qrel boundary.

## Scope

Add a dependency-free `trec_mlm/artifact_io.py` containing narrowly named public primitives:

- `sha256_file(path)` using the existing 1 MiB streaming chunks;
- `write_pretty_json(path, payload)` using UTF-8, `indent=2`, sorted keys, `ensure_ascii=False`, and one final newline;
- `length_framed_sha256(values)` using 8-byte big-endian UTF-8 lengths;
- `manifest_entries(root)` excluding files named `manifest.json` and returning sorted POSIX-relative keys;
- `manifest_aggregate_sha256(entries)` using the existing sorted compact JSON serialization;
- `source_tree_sha256(root)` preserving the existing relative-path and file-payload framing.

Migrate only `profile_experiment.py`, `profile_inference_artifact.py`, `profile_multiresolution.py`, `profile_multiresolution_evaluation.py`, and the shared framed-hash use in `profile_retrieval.py`. Keep compatibility aliases where tests or sealing guards monkeypatch a private name.

Replace cross-module private imports with public aliases whose bodies and arithmetic remain identical. This may expose, but must not relocate or rewrite, profile serialization, centroid reconstruction, complete-ranking validation, contextual-E5 encoding, and frozen label-artifact loading.

## Non-goals

- No artifact rewrite or regeneration.
- No change to JSON finiteness defaults, atomic/no-overwrite receipt writers, schemas, aggregate serialization, source-tree root, chunk size, dtype, ordering, record IDs, centroid math, clustering, scoring, rankings, RRF, evaluation, or qrel sealing.
- No broad sweep across older response, dense, source-query, or label experiment modules.
- No loader/reconstruction separation, path-provenance redesign, performance optimization, or module splitting.

## Compatibility contract

Golden tests freeze representative exact JSON bytes and SHA-256 values, length-framed hashes, manifest aggregate hashes (including escaped Unicode behavior), file inventory, chunked file hashes, and source-tree framing. Existing round-trip, tamper, repeatability, qrel-seal, and real-artifact tests remain authoritative.

## Verification

- Golden tests must fail before the shared helpers exist, then pass after minimal implementation.
- Each migration runs its focused module tests before moving on.
- Final checks: full artifact-present suite, true artifact-free clone suite, scoped Ruff, diff check, independent task reviews, and whole-branch review.

## Approval

This design instantiates the previously proposed second cleanup PR. The user approved execution with “do the prs as u planned” on 2026-07-16.
