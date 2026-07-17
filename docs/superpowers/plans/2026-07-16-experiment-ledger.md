# Experiment Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a deterministic, source-backed experiment ledger with aggregate, per-query, relevant-candidate, and density-slice metrics, while repairing the confined dense-helper grade-contract bug.

**Architecture:** A reviewed `experiments/ledger.json` registry declares experiment identity, evidence pointers, statuses, aliases, and caveats. Pure standard-library functions in `trec_mlm/experiment_ledger.py` validate and normalize tracked receipts, then a thin script writes JSON, TSV, and Markdown views atomically. Integration tests rebuild every checked-in output from tracked evidence and reject drift, mixed evaluation scopes, missing blockers, or unsupported topic claims.

**Tech Stack:** Python 3.13, standard-library `csv`, `dataclasses`, `hashlib`, `json`, `math`, `pathlib`, `tempfile`; pytest; Ruff; existing `trec_mlm.metrics` functions.

## Global Constraints

- Frozen experiment receipts remain scientific sources of truth; the ledger indexes and validates them without rewriting their claims.
- The builder reads only tracked repository evidence and must not depend on ignored `artifacts/` directories or discovery response JSON files.
- The repository has no organizer-provided semantic query-topic field; use `relevant_candidate_slice`, never `official_topic`.
- Relevant-candidate slices overlap and must carry support plus a double-counting warning.
- Positive-density slices are exactly `at_most_5_positives`, `6_to_20_positives`, and `more_than_20_positives`, with expected supports 248, 30, and 44.
- Only systems in the same evaluation scope may share a numerical comparison table.
- Per-query values are nDCG@10 and reciprocal rank, never probability or confidence.
- Aggregate-only systems require a concrete blocker; missing rows must never be inferred.
- Generated outputs must contain repository-relative paths only, use atomic replacement, omit wall-clock timestamps, and reproduce byte-for-byte.
- A `dev_322x150` stored per-query vector contains exactly 322 unique query IDs.
- Recomputed macro metrics must match their controlling receipt within `1e-12`; dense aliases may differ by at most `1e-10`.
- The dense helper fix must clip nonpositive judgments consistently with `trec_mlm.metrics.ndcg_at_k`; official historical metrics are not changed.
- Do not run new ranking, clustering, response-reliability, or judging experiments in this plan.

---

### Task 1: Repair the dense helper grade contract

**Files:**
- Modify: `tests/test_dense_experiment.py`
- Modify: `trec_mlm/dense_experiment.py:469`

**Interfaces:**
- Consumes: existing `exact_nested_loo_inner_scores(...)` and canonical `trec_mlm.metrics.ndcg_at_k` semantics.
- Produces: the same helper return type, with every computed nDCG in `[0, 1]` for arbitrary finite grade matrices.

- [ ] **Step 1: Add a failing negative-grade regression test**

Append a focused test that passes a finite matrix containing a negative judgment and asserts the helper agrees with the canonical clipped-gain metric. Use the existing helper construction patterns in the same test file; the essential assertion must be:

```python
def test_exact_nested_loo_inner_scores_clips_nonpositive_grades() -> None:
    grades = np.asarray(
        [
            [2.0, -100.0, 0.0],
            [0.0, 2.0, -100.0],
            [-100.0, 0.0, 2.0],
        ],
        dtype=np.float64,
    )
    similarities = np.asarray(
        [
            [1.0, 0.9, 0.8],
            [0.9, 1.0, 0.7],
            [0.8, 0.7, 1.0],
        ],
        dtype=np.float64,
    )

    result = exact_nested_loo_inner_scores(
        grade_matrix=grades,
        similarity_matrix=similarities,
        k_values=(1,),
        outer_indices=(0, 1, 2),
    )

    assert all(0.0 <= score <= 1.0 for score in result[1])
```

Adapt only the argument names and result indexing to the exact existing signature; preserve the negative value and range assertion.

- [ ] **Step 2: Run the regression test and verify RED**

Run:

```bash
uv run --extra dense --extra dev pytest tests/test_dense_experiment.py::test_exact_nested_loo_inner_scores_clips_nonpositive_grades -q
```

Expected: FAIL because at least one helper nDCG is negative or outside `[0, 1]`.

- [ ] **Step 3: Clip observed gains in the helper**

At the helper's observed-gain calculation, make the minimal production change:

```python
observed_gains = np.maximum(grade_matrix[validation_indices, ranking_indices], 0.0)
```

Do not alter the official runner's `{0, 1, 2}` validation or any selection logic.

- [ ] **Step 4: Verify GREEN and the dense test module**

Run:

```bash
uv run --extra dense --extra dev pytest tests/test_dense_experiment.py -q
```

Expected: all tests pass.

- [ ] **Step 5: Commit the isolated bug fix**

```bash
git add tests/test_dense_experiment.py trec_mlm/dense_experiment.py
git commit -m "fix: align dense helper metric gains"
```

---

### Task 2: Define and validate the reviewed registry

**Files:**
- Create: `experiments/ledger.json`
- Create: `trec_mlm/experiment_ledger.py`
- Create: `tests/test_experiment_ledger.py`

**Interfaces:**
- Consumes: the approved design at `docs/superpowers/specs/2026-07-16-experiment-ledger-design.md` and tracked result files named there.
- Produces:
  - `load_registry(root: Path, path: Path) -> dict[str, Any]`
  - `validate_registry(registry: Mapping[str, Any], root: Path) -> None`
  - `resolve_json_pointer(value: Any, pointer: str) -> Any`
  - constants `EVALUATION_SCOPES`, `EVIDENCE_STATUSES`, `PROMOTION_STATUSES`, `ARTIFACT_STATES`, and `QREL_USAGE_STATUSES`.

- [ ] **Step 1: Write failing registry-contract tests**

Create `tests/test_experiment_ledger.py` with temporary registries proving uniqueness, vocabulary, blockers, and source-path rules. Include these exact behavioral tests:

```python
from pathlib import Path

import pytest

from trec_mlm.experiment_ledger import resolve_json_pointer, validate_registry


def minimal_registry() -> dict[str, object]:
    return {
        "version": 1,
        "sources": {
            "receipt": {
                "path": "reports/result.json",
                "sha256": "0" * 64,
            }
        },
        "caveats": {
            "development_only": {
                "scope": "program",
                "summary": "Development evidence only.",
            }
        },
        "experiments": [
            {
                "experiment_id": "example",
                "experiment_family": "control",
                "systems": [
                    {
                        "system_id": "example_system",
                        "source_system_id": "example_source",
                        "aliases": [],
                        "method_summary": "Example.",
                        "evaluation_scope": "dev_322x150",
                        "evidence_status": "aggregate_only",
                        "promotion_status": "valid_negative",
                        "query_count": 322,
                        "candidate_count": 150,
                        "aggregate_source": {
                            "source_id": "receipt",
                            "pointer": "/metrics/example",
                        },
                        "per_query_source": None,
                        "per_query_blocker": "Tracked source stores aggregates only.",
                        "ranking_coverage": "none",
                        "native_scores_available": False,
                        "artifact_state": "tracked_receipt_only",
                        "qrel_usage": "evaluation_only",
                        "run_classification": "internal_automatic",
                        "caveat_ids": ["development_only"],
                    }
                ],
            }
        ],
    }


def test_resolve_json_pointer_handles_escaped_tokens() -> None:
    value = {"a/b": {"~key": 7}}
    assert resolve_json_pointer(value, "/a~1b/~0key") == 7


def test_validate_registry_rejects_duplicate_system_ids(tmp_path: Path) -> None:
    registry = minimal_registry()
    registry["experiments"].append(registry["experiments"][0])
    with pytest.raises(ValueError, match="duplicate experiment_id"):
        validate_registry(registry, tmp_path)


def test_validate_registry_requires_aggregate_only_blocker(tmp_path: Path) -> None:
    registry = minimal_registry()
    registry["experiments"][0]["systems"][0]["per_query_blocker"] = ""
    with pytest.raises(ValueError, match="per_query_blocker"):
        validate_registry(registry, tmp_path)


def test_validate_registry_rejects_absolute_source_path(tmp_path: Path) -> None:
    registry = minimal_registry()
    registry["sources"]["receipt"]["path"] = "/tmp/result.json"
    with pytest.raises(ValueError, match="repository-relative"):
        validate_registry(registry, tmp_path)
```

Tests may write the declared source file when they need to reach digest validation; keep each test focused on one rejection.

- [ ] **Step 2: Run the registry tests and verify RED**

Run:

```bash
uv run --extra dev pytest tests/test_experiment_ledger.py -q
```

Expected: collection fails because `trec_mlm.experiment_ledger` does not exist.

- [ ] **Step 3: Implement the minimal registry library**

Create `trec_mlm/experiment_ledger.py` with frozen vocabularies and strict helpers. The public skeleton is:

```python
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

EVALUATION_SCOPES = frozenset({
    "dev_322x150",
    "restricted_12x20",
    "restricted_6",
    "invalid_unscored",
    "constructed_not_evaluated",
})
EVIDENCE_STATUSES = frozenset({
    "stored_per_query",
    "recomputed_from_complete_ranking",
    "aggregate_only",
    "restricted_diagnostic",
    "invalid",
    "unscored",
})
PROMOTION_STATUSES = frozenset({
    "promoted_development",
    "stable_control",
    "valid_negative",
    "diagnostic_only",
    "restricted_pilot",
    "invalid",
    "stopped_unscored",
    "constructed_not_evaluated",
})
ARTIFACT_STATES = frozenset({
    "tracked_complete",
    "tracked_receipt_only",
    "tracked_report_snapshot",
    "local_artifact_missing",
    "construction_artifact_missing",
})
QREL_USAGE_STATUSES = frozenset({
    "training_and_evaluation",
    "evaluation_only",
    "restricted_selection_and_evaluation",
    "not_evaluated",
})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_json_pointer(value: Any, pointer: str) -> Any:
    if pointer == "":
        return value
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")
    current = value
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) else current[token]
    return current


def load_registry(root: Path, path: Path) -> dict[str, Any]:
    registry = json.loads((root / path).read_text(encoding="utf-8"))
    validate_registry(registry, root)
    return registry


def validate_registry(registry: Mapping[str, Any], root: Path) -> None:
    if registry.get("version") != 1:
        raise ValueError("ledger registry version must be 1")
    sources = registry.get("sources")
    caveats = registry.get("caveats")
    experiments = registry.get("experiments")
    if not isinstance(sources, Mapping) or not isinstance(caveats, Mapping) or not isinstance(experiments, list):
        raise ValueError("registry sources, caveats, and experiments have invalid shapes")

    experiment_ids: set[str] = set()
    system_ids: set[str] = set()
    referenced_sources: set[str] = set()
    for experiment in experiments:
        experiment_id = experiment.get("experiment_id")
        if not isinstance(experiment_id, str) or experiment_id in experiment_ids:
            raise ValueError(f"duplicate experiment_id: {experiment_id}")
        experiment_ids.add(experiment_id)
        for system in experiment.get("systems", []):
            system_id = system.get("system_id")
            if not isinstance(system_id, str) or system_id in system_ids:
                raise ValueError(f"duplicate system_id: {system_id}")
            system_ids.add(system_id)
            if system.get("evaluation_scope") not in EVALUATION_SCOPES:
                raise ValueError(f"unknown evaluation_scope for {system_id}")
            if system.get("evidence_status") not in EVIDENCE_STATUSES:
                raise ValueError(f"unknown evidence_status for {system_id}")
            if system.get("promotion_status") not in PROMOTION_STATUSES:
                raise ValueError(f"unknown promotion_status for {system_id}")
            if system.get("artifact_state") not in ARTIFACT_STATES:
                raise ValueError(f"unknown artifact_state for {system_id}")
            if system.get("qrel_usage") not in QREL_USAGE_STATUSES:
                raise ValueError(f"unknown qrel_usage for {system_id}")
            if system.get("evidence_status") == "aggregate_only" and not system.get("per_query_blocker"):
                raise ValueError(f"per_query_blocker required for {system_id}")
            for source_field in ("aggregate_source", "per_query_source"):
                source_ref = system.get(source_field)
                if source_ref is not None:
                    referenced_sources.add(source_ref["source_id"])
            unknown_caveats = set(system.get("caveat_ids", [])) - set(caveats)
            if unknown_caveats:
                raise ValueError(f"unknown caveat IDs for {system_id}: {sorted(unknown_caveats)}")

    unknown_sources = referenced_sources - set(sources)
    if unknown_sources:
        raise ValueError(f"unknown source IDs: {sorted(unknown_sources)}")
    for source_id, source in sources.items():
        relative = Path(source["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"source path must be repository-relative: {source_id}")
        expected = source["sha256"]
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"invalid source SHA-256: {source_id}")
        full_path = root / relative
        if not full_path.is_file():
            raise ValueError(f"missing controlling source: {relative}")
        if _sha256(full_path) != expected:
            raise ValueError(f"source digest changed: {relative}")
```

- [ ] **Step 4: Populate the reviewed registry**

Create `experiments/ledger.json` with source digests computed from current tracked files and program caveats from the spec. Include these experiment families and decision-relevant systems explicitly:

```text
query_controls: global_prior, tfidf_fixed_k10
dense_query_knn: dense_nested plus every Cartesian product of encoders
  {bge_small_en_v1_5, e5_small_v2, minilm_l6, mpnet_base_v2} and
  k {1, 3, 5, 10, 20, 50, 100}
candidate_label_matching: mpnet_raw_label, contextual_e5, qwen8b_contextual,
  qwen4b_contextual, qwen06b_contextual, jina_v5_contextual,
  bge_m3_contextual, nomic_v15_contextual
label_ranker_pilots: pointwise_qwen, generated_listwise, grouprank_7b,
  firstmistral_listwise plus restricted E5/Qwen controls
response_retrieval: response_sumscore_nested plus every Cartesian product of
  method {bm25, bm25_rm3}, field {query_response, response_only},
  filter {aggressive, basic}, and depth {100, 500, 1000}
response_volume_normalization: response_top10sum_nested; retain the exact tested
  aggregation order {best_hit, top3_sum, top10_sum, top3_mean, top10_mean,
  mean_all} as experiment metadata rather than inventing unreported metrics
source_query_first: source_alignment_nested, target_response_same_k plus k grid
behavior_profile: discovery_behavior_topic_v1
semantic_profiles: profile_only, dense_profile_equal, dense_profile_dense_heavy,
  dense_profile_profile_heavy, dense_e5_equal, dense_profile_e5_equal
multiresolution_profiles: a1_top2_k10, a1_top2_k10_dense_equal,
  r1_multires_max, r1_multires_max_dense_equal, x1_multires_top2,
  x1_multires_top2_dense_equal
query_cluster_aggregation: local_domain_prior
```

Use explicit `literal_snapshot` aggregate sources only for reviewed Markdown
tables (`expert-label-ranking-study.md` and `issue_13_dense_aggregation_result.md`),
and bind each literal snapshot to the declared file digest. JSON receipts use
JSON pointers. Grid expansions must declare an exact allowed-key list and inherit
only family-level `diagnostic_only` metadata; an unexpected new key is an error.

- [ ] **Step 5: Verify GREEN**

Run:

```bash
uv run --extra dev pytest tests/test_experiment_ledger.py -q
```

Expected: all registry-contract tests pass.

- [ ] **Step 6: Commit the reviewed registry foundation**

```bash
git add experiments/ledger.json trec_mlm/experiment_ledger.py tests/test_experiment_ledger.py
git commit -m "feat: add reviewed experiment registry"
```

---

### Task 3: Extract metrics and compute query/topic slices

**Files:**
- Modify: `trec_mlm/experiment_ledger.py`
- Modify: `tests/test_experiment_ledger.py`

**Interfaces:**
- Consumes: validated registry from Task 2, tracked query/qrel files, JSON receipts, dense per-query TSV.
- Produces:
  - `SystemRecord` frozen dataclass;
  - `PerQueryRecord` frozen dataclass;
  - `SliceRecord` frozen dataclass;
  - `LedgerBuild` frozen dataclass;
  - `build_ledger(root: Path, registry_path: Path = Path("experiments/ledger.json")) -> LedgerBuild`.

- [ ] **Step 1: Add failing extraction and slice tests**

Extend `tests/test_experiment_ledger.py` with small fixture receipts and qrels. The tests must assert:

```python
def test_build_ledger_recomputes_macro_and_overlapping_slices(tmp_path: Path) -> None:
    # q1 is relevant to A and B; q2 is relevant only to B.
    # System per-query values are q1=(0.5, 1.0), q2=(0.0, 0.25).
    build = build_fixture_ledger(tmp_path)
    system = next(row for row in build.systems if row.system_id == "example_system")
    assert system.ndcg_at_10 == pytest.approx(0.25)
    assert system.mrr == pytest.approx(0.625)
    slices = {(row.slice_key, row.query_count): row for row in build.topic_slices}
    assert slices[("A", 1)].ndcg_at_10 == pytest.approx(0.5)
    assert slices[("B", 2)].ndcg_at_10 == pytest.approx(0.25)
    assert all(row.overlapping for row in build.topic_slices)


def test_build_ledger_requires_322_rows_for_full_scope(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exactly 322"):
        build_fixture_ledger(tmp_path, per_query_count=321)


def test_build_ledger_rejects_macro_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="macro metric"):
        build_fixture_ledger(tmp_path, reported_ndcg=0.9)


def test_density_slices_are_disjoint() -> None:
    assert density_bucket(1) == "at_most_5_positives"
    assert density_bucket(5) == "at_most_5_positives"
    assert density_bucket(6) == "6_to_20_positives"
    assert density_bucket(20) == "6_to_20_positives"
    assert density_bucket(21) == "more_than_20_positives"
```

Add fixture helpers in the test module only; do not add test-only production APIs.

- [ ] **Step 2: Run targeted tests and verify RED**

Run:

```bash
uv run --extra dev pytest tests/test_experiment_ledger.py -q
```

Expected: FAIL because dataclasses, `build_ledger`, and `density_bucket` do not exist.

- [ ] **Step 3: Implement normalized records and source loaders**

Add these public shapes:

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SystemRecord:
    system_id: str
    source_system_id: str
    experiment_id: str
    experiment_family: str
    evaluation_scope: str
    evidence_status: str
    promotion_status: str
    query_count: int
    candidate_count: int
    ndcg_at_10: float | None
    mrr: float | None
    zero_ndcg_queries: int | None
    per_query_available: bool
    ranking_coverage: str
    artifact_state: str
    source_path: str
    source_sha256: str
    blocker: str | None
    caveat_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PerQueryRecord:
    system_id: str
    source_system_id: str
    experiment_family: str
    evaluation_scope: str
    query_id: str
    query_text: str
    ndcg_at_10: float
    reciprocal_rank: float
    grade_1_count: int
    grade_2_count: int
    positive_count: int
    relevant_candidates: tuple[tuple[str, int], ...]
    density_bucket: str
    zero_ndcg: bool
    metric_provenance: str
    source_path: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class SliceRecord:
    system_id: str
    slice_type: str
    slice_key: str
    query_count: int
    ndcg_at_10: float
    mrr: float
    zero_ndcg_count: int
    zero_ndcg_rate: float
    grade_1_memberships: int
    grade_2_memberships: int
    overlapping: bool
    membership_policy: str


@dataclass(frozen=True, slots=True)
class LedgerBuild:
    systems: tuple[SystemRecord, ...]
    per_query: tuple[PerQueryRecord, ...]
    topic_slices: tuple[SliceRecord, ...]
    density_slices: tuple[SliceRecord, ...]
    sources: tuple[dict[str, str], ...]
    caveats: tuple[dict[str, str], ...]
    validation: dict[str, object]
```

Implement JSON-pointer, literal-snapshot, grid, issue-18 per-query-map, and dense-TSV loaders. Normalize source field names `ndcg_at_k` to `ndcg_at_10` and `reciprocal_rank` to `mrr` only at the aggregate boundary; preserve `reciprocal_rank` in per-query rows.

- [ ] **Step 4: Implement qrel joins and slice calculations**

Implement:

```python
def density_bucket(positive_count: int) -> str:
    if positive_count <= 5:
        return "at_most_5_positives"
    if positive_count <= 20:
        return "6_to_20_positives"
    return "more_than_20_positives"
```

Load all 322 query texts and all 48,300 qrels with the repository's existing loaders where their contracts match. Require exactly 150 unique candidates per query and grades in `{0, 1, 2}` for ledger slices. Join every per-query vector to the complete qrel summary.

For relevant candidate `c`, include the complete query metric whenever `grade(q, c) > 0`. Set `membership_policy` exactly to `qrel_grade_gt_zero`; set `overlapping=True`.

For density slices, include each query once, set `membership_policy` exactly to `positive_candidate_count`, and `overlapping=False`. Assert support counts `(248, 30, 44)` for the real development data.

- [ ] **Step 5: Verify real-source extraction**

Add a repository test marked neither slow nor optional:

```python
def test_real_ledger_extracts_complete_recoverable_vectors() -> None:
    build = build_ledger(Path("."))
    counts = Counter(row.system_id for row in build.per_query)
    assert counts["dense_nested"] == 322
    assert counts["contextual_e5"] == 322
    assert counts["profile_only"] == 322
    assert counts["dense_profile_equal"] == 322
    assert counts["dense_profile_profile_heavy"] == 322
    assert sum(row.query_count for row in build.density_slices if row.system_id == "dense_nested") == 322
```

Also assert that `response_sumscore_nested`, `source_alignment_nested`, and `mpnet_raw_label` remain aggregate-only with nonempty blockers.

- [ ] **Step 6: Run tests and verify GREEN**

Run:

```bash
uv run --extra dev pytest tests/test_experiment_ledger.py -q
```

Expected: all tests pass against both fixtures and tracked evidence.

- [ ] **Step 7: Commit extraction and slicing**

```bash
git add trec_mlm/experiment_ledger.py tests/test_experiment_ledger.py
git commit -m "feat: derive query and topic ledger metrics"
```

---

### Task 4: Render deterministic outputs and add the builder CLI

**Files:**
- Modify: `trec_mlm/experiment_ledger.py`
- Create: `scripts/build_experiment_ledger.py`
- Create: `tests/test_experiment_ledger_outputs.py`

**Interfaces:**
- Consumes: `LedgerBuild` from Task 3.
- Produces:
  - `render_ledger_json(build: LedgerBuild) -> str`
  - `render_systems_tsv(build: LedgerBuild) -> str`
  - `render_per_query_tsv(build: LedgerBuild) -> str`
  - `render_topic_slices_tsv(build: LedgerBuild) -> str`
  - `render_density_slices_tsv(build: LedgerBuild) -> str`
  - `render_ledger_markdown(build: LedgerBuild) -> str`
  - `write_ledger_outputs(build: LedgerBuild, output_dir: Path) -> tuple[Path, ...]`
  - CLI flags `--root`, `--registry`, `--output-dir`, and `--check`.

- [ ] **Step 1: Write failing deterministic-render tests**

Create `tests/test_experiment_ledger_outputs.py` with a small `LedgerBuild` fixture and assertions that:

```python
def test_renderers_are_deterministic_and_repository_relative(sample_build: LedgerBuild) -> None:
    first = render_ledger_json(sample_build)
    second = render_ledger_json(sample_build)
    assert first == second
    assert "/home/" not in first
    assert "generated_at" not in first


def test_markdown_separates_full_and_restricted_scopes(sample_build: LedgerBuild) -> None:
    markdown = render_ledger_markdown(sample_build)
    full_table = markdown.split("## Full development results", 1)[1].split("##", 1)[0]
    assert "restricted_system" not in full_table
    assert "Restricted pilots" in markdown


def test_write_outputs_is_atomic_on_render_failure(tmp_path: Path, sample_build: LedgerBuild, monkeypatch: pytest.MonkeyPatch) -> None:
    existing = tmp_path / "trec_2026_experiment_ledger.json"
    existing.write_text("old", encoding="utf-8")
    monkeypatch.setattr(module, "render_per_query_tsv", lambda build: (_ for _ in ()).throw(ValueError("boom")))
    with pytest.raises(ValueError, match="boom"):
        write_ledger_outputs(sample_build, tmp_path)
    assert existing.read_text(encoding="utf-8") == "old"
```

- [ ] **Step 2: Run output tests and verify RED**

Run:

```bash
uv run --extra dev pytest tests/test_experiment_ledger_outputs.py -q
```

Expected: collection fails because render/write functions do not exist.

- [ ] **Step 3: Implement stable JSON and TSV rendering**

Use `json.dumps(..., indent=2, sort_keys=True, ensure_ascii=False) + "\n"` for JSON. Use `csv.DictWriter` with `lineterminator="\n"` and explicit field sequences for TSV.

Represent `relevant_candidates` as compact deterministic JSON inside the TSV cell. Sort systems by `(evaluation_scope, experiment_family, system_id)`, per-query rows by `(system_id, query_id)`, and slices by `(system_id, slice_type, slice_key)`.

The systems TSV includes:

```text
system_id, source_system_id, experiment_id, experiment_family,
evaluation_scope, evidence_status, promotion_status, query_count,
candidate_count, ndcg_at_10, mrr, zero_ndcg_queries,
per_query_available, ranking_coverage, artifact_state,
source_path, source_sha256, blocker, caveat_ids
```

- [ ] **Step 4: Implement concise Markdown rendering**

The Markdown must begin with the promoted development result, best diagnostic, best stable qrel-free control, count of systems with per-query evidence, count of aggregate-only systems, and unresolved artifact blockers.

Then render separate sections:

```text
## Full development results
## Restricted pilots and invalid/unscored work
## Per-query evidence coverage
## Topic and density slices
## Program-wide caveats
## Source receipts
## Rebuild command
```

Do not put restricted systems in the full-development table. Display `diagnostic_only`, `aggregate_only`, and missing artifacts in visible text rather than relying on footnotes.

- [ ] **Step 5: Implement all-or-nothing output writes and CLI**

Render all six payloads in memory first. Write them beneath one temporary directory inside `output_dir.parent`; only after all writes succeed, replace the six destinations with `Path.replace` in deterministic filename order. Clean the temporary directory on failure.

The CLI entry point must be:

```python
def main() -> int:
    parser = argparse.ArgumentParser(description="Build the canonical TREC 2026 experiment ledger.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--registry", type=Path, default=Path("experiments/ledger.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("reports"))
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    build = build_ledger(args.root, args.registry)
    if args.check:
        assert_outputs_current(build, args.root / args.output_dir)
    else:
        write_ledger_outputs(build, args.root / args.output_dir)
    return 0
```

`assert_outputs_current` renders expected payloads and raises `ValueError` naming every missing or stale file without modifying anything.

- [ ] **Step 6: Verify output tests and CLI help**

Run:

```bash
uv run --extra dev pytest tests/test_experiment_ledger_outputs.py -q
uv run python scripts/build_experiment_ledger.py --help
```

Expected: all tests pass and CLI help exits 0.

- [ ] **Step 7: Commit rendering and CLI**

```bash
git add trec_mlm/experiment_ledger.py scripts/build_experiment_ledger.py tests/test_experiment_ledger_outputs.py
git commit -m "feat: render deterministic experiment ledger"
```

---

### Task 5: Generate, document, and verify the canonical ledger

**Files:**
- Create: `reports/trec_2026_experiment_ledger.json`
- Create: `reports/trec_2026_experiment_systems.tsv`
- Create: `reports/trec_2026_experiment_per_query.tsv`
- Create: `reports/trec_2026_experiment_topic_slices.tsv`
- Create: `reports/trec_2026_experiment_density_slices.tsv`
- Create: `reports/trec_2026_experiment_ledger.md`
- Create: `tests/test_experiment_ledger_integration.py`
- Modify: `README.md:9-27`

**Interfaces:**
- Consumes: builder CLI and all tracked source receipts.
- Produces: checked-in canonical ledger views plus a repository drift test.

- [ ] **Step 1: Add a failing checked-output integration test**

Create `tests/test_experiment_ledger_integration.py`:

```python
from pathlib import Path

from trec_mlm.experiment_ledger import assert_outputs_current, build_ledger


def test_checked_in_experiment_ledger_is_current() -> None:
    root = Path(".")
    build = build_ledger(root)
    assert_outputs_current(build, root / "reports")


def test_real_ledger_has_required_program_caveats() -> None:
    build = build_ledger(Path("."))
    caveat_ids = {row["caveat_id"] for row in build.caveats}
    assert {
        "development_only",
        "adaptive_program_overfit",
        "restricted_not_comparable",
        "topic_slices_overlap",
        "missing_raw_artifacts",
        "issue18_not_restorable",
        "confidence_not_correctness",
        "bootstrap_protocols_differ",
    } <= caveat_ids
```

- [ ] **Step 2: Run the integration test and verify RED**

Run:

```bash
uv run --extra dev pytest tests/test_experiment_ledger_integration.py -q
```

Expected: FAIL because the six generated outputs do not exist.

- [ ] **Step 3: Generate the six tracked outputs**

Run:

```bash
uv run python scripts/build_experiment_ledger.py
```

Inspect the Markdown and the first five lines plus row counts of each TSV. Confirm no absolute `/home/` path exists:

```bash
rg -n '/home/' reports/trec_2026_experiment_ledger.* reports/trec_2026_experiment_*.tsv
```

Expected: no matches.

- [ ] **Step 4: Add README navigation**

Add a short section after the beginner approach guide introduction:

```markdown
## Canonical Experiment Ledger

For the current, source-backed inventory of every experiment, comparable score,
per-query evidence vector, relevant-candidate slice, caveat, and missing-artifact
status, see
[`reports/trec_2026_experiment_ledger.md`](reports/trec_2026_experiment_ledger.md).
The ledger is generated from reviewed receipts; detailed sections below explain
individual methods but are not the canonical cross-experiment leaderboard.
```

- [ ] **Step 5: Verify deterministic regeneration**

Record hashes, rebuild, and compare:

```bash
sha256sum reports/trec_2026_experiment_ledger.json reports/trec_2026_experiment_systems.tsv reports/trec_2026_experiment_per_query.tsv reports/trec_2026_experiment_topic_slices.tsv reports/trec_2026_experiment_density_slices.tsv reports/trec_2026_experiment_ledger.md > /tmp/ledger-hashes-before.txt
uv run python scripts/build_experiment_ledger.py
sha256sum -c /tmp/ledger-hashes-before.txt
uv run python scripts/build_experiment_ledger.py --check
```

Expected: all six hash checks report `OK`; `--check` exits 0.

- [ ] **Step 6: Run targeted integration tests**

Run:

```bash
uv run --extra dev pytest tests/test_experiment_ledger.py tests/test_experiment_ledger_outputs.py tests/test_experiment_ledger_integration.py tests/test_dense_experiment.py -q
```

Expected: all tests pass.

- [ ] **Step 7: Run full verification**

Run:

```bash
uv run --extra dense --extra response --extra dev pytest -q -rs
uv run --extra dev ruff check trec_mlm scripts tests
git diff --check
git status --short
```

Expected: 0 test failures; only the three known issue-18 artifact-dependent skips; Ruff and diff checks exit 0; status lists only intended task files.

- [ ] **Step 8: Commit the canonical generated ledger**

```bash
git add README.md reports/trec_2026_experiment_ledger.json reports/trec_2026_experiment_systems.tsv reports/trec_2026_experiment_per_query.tsv reports/trec_2026_experiment_topic_slices.tsv reports/trec_2026_experiment_density_slices.tsv reports/trec_2026_experiment_ledger.md tests/test_experiment_ledger_integration.py
git commit -m "docs: publish canonical experiment ledger"
```

---

## Final review gate

After Task 5:

1. generate one whole-branch review package from the branch merge base;
2. dispatch an independent reviewer against the approved specification and this plan;
3. fix and re-review every Critical or Important finding;
4. rerun the complete verification commands after the last fix; and
5. use the finishing-development-branch workflow to present integration options without pushing or opening a pull request unless the user explicitly authorizes it.
