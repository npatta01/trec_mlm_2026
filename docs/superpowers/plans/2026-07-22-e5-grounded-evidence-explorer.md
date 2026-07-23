# E5 Grounded-Evidence Explorer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and privately publish a self-contained explorer that diagnoses all 322 frozen E5 development queries using component ranks, qrels-derived cohorts, and the exact three nearest historical question-response records for each displayed evidence candidate.

**Architecture:** A new `trec_mlm.grounded_evidence_explorer` module validates the frozen E5 artifacts and produces a renderer-neutral JSON bundle without recomputing similarity. Its semantic validator is authoritative for both producer and generic consumer; the task-specific builder remains responsible for binding component ranks to source artifacts. The existing generic query explorer accepts that bundle optionally, unions the final top 20 with all positive candidates, and renders component ranks, filters, mechanism tags, and evidence drawers. A thin task-specific CLI stages and validates the private bundle and final HTML before promoting both outputs together.

**Tech Stack:** Python 3.11+, NumPy, stdlib JSON/HTML/JavaScript, pytest, Ruff, static tailnet-only Tailscale Serve.

## Global Constraints

- Development qrels may annotate only already-frozen rankings; they must never affect candidate ordering, evidence selection, or similarity.
- Consume `selected_record_ids.json` and `top_record_similarities.npy` exactly as persisted; do not recompute nearest neighbors.
- Display the final top 20 plus every development-positive candidate; preserve frozen final-rank order.
- Embed exactly three full historical records for every final top-ten candidate and every development-positive candidate.
- Label evidence as `Nearest historical behavior` and similarity as `E5 cosine similarity`; never describe either as correctness, confidence, probability, or expertise.
- Keep raw historical answers outside Git and only in the private build bundle and tailnet-only rendered HTML.
- Fail closed before replacing either output when inventories, shapes, record membership, ranking completeness, metric bounds, derivable semantics, or similarity validity differ.
- Stage bundle and HTML at unique sibling paths; validation or renderer failure must preserve both prior outputs byte-for-byte and remove all staging files.
- Preserve generic explorer behavior when no evidence bundle is supplied.
- Do not modify, stage, or commit the unrelated dirty files already present in the worktree.

---

## File map

- Create `trec_mlm/grounded_evidence_explorer.py`: artifact loading, validation, cohort/retrieval-outcome/mechanism derivation, evidence selection, and bundle serialization.
- Create `tests/test_grounded_evidence_explorer.py`: focused unit fixtures for bundle semantics and all fail-closed conditions.
- Modify `.agents/skills/inspect-query-results/scripts/build_query_explorer.py`: optional evidence-bundle loading, top-20-plus-positive union, bundle merge, and accessible UI rendering.
- Modify `tests/test_query_explorer_skill.py`: generic compatibility, bundle integration, escaping, UI labels, and atomic-failure coverage.
- Create `scripts/build_e5_grounded_evidence_explorer.py`: one reproducible CLI that builds the private bundle and invokes the generic renderer atomically.
- Create `tests/test_build_e5_grounded_evidence_explorer.py`: CLI orchestration and redaction/logging assertions.
- Modify `reports/trec_2026_e5_raw_response_failure_ledger.md`: replace the old “raw responses excluded” statement with the verified private explorer description.
- Modify `/home/npatta01/codex-rendered/index.html`: update only the existing E5 explorer entry after the final artifact passes local verification.
- Replace `/home/npatta01/codex-rendered/plans/e5-raw-response-failure-explorer.html`: sanitized derived artifact containing the approved evidence subset.

---

### Task 1: Build and validate the renderer-neutral evidence bundle

**Files:**
- Create: `trec_mlm/grounded_evidence_explorer.py`
- Test: `tests/test_grounded_evidence_explorer.py`

**Interfaces:**
- Consumes: `build_evidence_bundle(refinement_dir: Path, construction_dir: Path, evaluation_path: Path, queries_path: Path, qrels_path: Path) -> dict[str, Any]`.
- Produces: a JSON-safe mapping with `schema_version`, `meta`, and `queries`; each query contains `component_metrics` and candidate entries keyed by candidate ID. Task 2 adds `cohorts`, `context_risk`, `retrieval_outcome`, and `mechanism`.
- Raises: `EvidenceBundleError` for every validation failure.
- Test fixture: `write_e5_fixture(tmp_path: Path, mutation: str | None = None) -> dict[str, Path]` writes two queries, twelve candidates, complete qrels, all five complete rankings, a `(2, 12, 10)` little-endian float64 similarity array, matching selected IDs, records, and evaluation rows. A requested mutation changes only the named invariant.

- [ ] **Step 1: Write failing fixture tests for exact persisted evidence and behavior tags**

```python
def test_build_bundle_uses_first_three_persisted_records(tmp_path: Path) -> None:
    paths = write_e5_fixture(tmp_path)
    bundle = build_evidence_bundle(**paths)
    candidate = bundle["queries"]["q1"]["candidates"]["c1"]

    assert [item["record_id"] for item in candidate["evidence"]] == ["r1", "r2", "r3"]
    assert [item["similarity"] for item in candidate["evidence"]] == [0.91, 0.83, 0.72]
    assert [item["behavior"] for item in candidate["evidence"]] == [
        "substantive response",
        "exact refusal",
        "empty",
    ]
    assert candidate["best_record_cosine"] == 0.91
    assert candidate["mean_top_10_cosine"] == pytest.approx(0.565)


@pytest.mark.parametrize("response, expected", [
    ("A supported historical answer.", "substantive response"),
    ("I don't know", "exact refusal"),
    ("I do not know.", "exact refusal"),
    ("   ", "empty"),
])
def test_classify_behavior(response: str, expected: str) -> None:
    assert classify_behavior(response) == expected
```

- [ ] **Step 2: Run the focused tests and confirm the red state**

Run: `uv run pytest tests/test_grounded_evidence_explorer.py -q`

Expected: collection fails because `trec_mlm.grounded_evidence_explorer` does not exist.

- [ ] **Step 3: Implement strict loaders, rank positions, behavior classification, and evidence selection**

```python
class EvidenceBundleError(ValueError):
    """Frozen E5 inputs cannot safely produce diagnostic evidence."""


EXACT_REFUSALS = frozenset({"i don t know", "i do not know"})
SYSTEMS = (
    "dense_reference",
    "profile_only",
    "dense_profile_equal",
    "raw_hierarchical",
    "raw_hierarchical_dense_equal",
)


def classify_behavior(response: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", " ", response.casefold()).strip()
    if not normalized:
        return "empty"
    if normalized in EXACT_REFUSALS:
        return "exact refusal"
    return "substantive response"


def _rank_positions(ranking: Sequence[str], candidates: tuple[str, ...], label: str) -> dict[str, int]:
    if len(ranking) != len(candidates) or set(ranking) != set(candidates):
        raise EvidenceBundleError(f"{label} candidate inventory is incomplete")
    return {candidate_id: rank for rank, candidate_id in enumerate(ranking, 1)}


def _selected_evidence(record_ids: Sequence[str], similarities: np.ndarray,
                       records: Mapping[str, Mapping[str, Any]], candidate_id: str) -> list[dict[str, Any]]:
    if len(record_ids) != 10 or similarities.shape != (10,):
        raise EvidenceBundleError("selected evidence must contain persisted top 10")
    if not np.isfinite(similarities).all() or np.any(similarities[:-1] < similarities[1:]):
        raise EvidenceBundleError("selected similarities must be finite and non-increasing")
    output = []
    for record_id, similarity in zip(record_ids[:3], similarities[:3], strict=True):
        record = records.get(record_id)
        if record is None or record.get("candidate_id") != candidate_id:
            raise EvidenceBundleError(f"selected record membership differs for {candidate_id}")
        response = str(record["response"])
        output.append({
            "record_id": record_id,
            "source_query_id": str(record["source_query_id"]),
            "question": str(record["question"]),
            "response": response,
            "similarity": float(similarity),
            "behavior": classify_behavior(response),
        })
    return output
```

- [ ] **Step 4: Add failing tests for complete inventories, component ranks, and candidate inclusion**

```python
def test_bundle_has_complete_component_ranks_and_only_needed_evidence(tmp_path: Path) -> None:
    paths = write_e5_fixture(tmp_path)
    bundle = build_evidence_bundle(**paths)
    query = bundle["queries"]["q1"]

    assert query["display_candidate_ids"] == ["c1", "c2", "c3"]
    assert query["evidence_candidate_ids"] == ["c1", "c2"]
    assert query["candidates"]["c2"]["ranks"] == {
        "dense_reference": 3,
        "profile_only": 2,
        "dense_profile_equal": 2,
        "raw_hierarchical": 1,
        "raw_hierarchical_dense_equal": 2,
        "best_record_cosine": 2,
    }


@pytest.mark.parametrize("mutation, message", [
    ("query_inventory", "query inventory"),
    ("candidate_inventory", "candidate inventory"),
    ("ranking_duplicate", "candidate inventory is incomplete"),
    ("record_owner", "record membership differs"),
    ("similarity_nan", "finite and non-increasing"),
    ("similarity_order", "finite and non-increasing"),
    ("array_shape", "shape"),
    ("qrel_pair_missing", "qrel inventory"),
])
def test_bundle_fails_closed(tmp_path: Path, mutation: str, message: str) -> None:
    paths = write_e5_fixture(tmp_path, mutation=mutation)
    with pytest.raises(EvidenceBundleError, match=message):
        build_evidence_bundle(**paths)
```

- [ ] **Step 5: Implement complete artifact joining and bundle metadata**

```python
def build_evidence_bundle(*, refinement_dir: Path, construction_dir: Path,
                          evaluation_path: Path, queries_path: Path,
                          qrels_path: Path) -> dict[str, Any]:
    inputs = _load_json(refinement_dir / "inputs.json")
    query_ids = tuple(inputs["query_ids"])
    candidate_ids = tuple(inputs["candidate_ids"])
    queries = load_queries_tsv(queries_path)
    qrels = load_complete_qrels(qrels_path, expected_query_ids=queries)
    validate_qrel_inventories(qrels, query_ids=query_ids, candidate_ids=candidate_ids)
    if tuple(queries) != query_ids:
        raise EvidenceBundleError("query inventory or ordering differs")

    rankings = _load_all_rankings(refinement_dir, construction_dir, query_ids, candidate_ids)
    selected_ids = _load_json(refinement_dir / "selected_record_ids.json")
    similarities = np.load(refinement_dir / "top_record_similarities.npy", allow_pickle=False)
    expected_shape = (len(query_ids), len(candidate_ids), 10)
    if similarities.shape != expected_shape or similarities.dtype != np.dtype("<f8"):
        raise EvidenceBundleError(f"top-record similarity shape or dtype differs: {similarities.shape}")
    records = _load_records(construction_dir / "records.json", candidate_ids)
    evaluation = _evaluation_rows(evaluation_path, query_ids)

    query_payload = {}
    for query_index, query_id in enumerate(query_ids):
        positions = {name: _rank_positions(value[query_id], candidate_ids, name)
                     for name, value in rankings.items()}
        best_record_order = sorted(candidate_ids,
                                   key=lambda cid: (-float(similarities[query_index, candidate_ids.index(cid), 0]), cid))
        positions["best_record_cosine"] = {
            candidate_id: rank for rank, candidate_id in enumerate(best_record_order, 1)
        }
        positives = {candidate_id for candidate_id, grade in qrels[query_id].items() if grade > 0}
        final = rankings["raw_hierarchical_dense_equal"][query_id]
        display_ids = [candidate_id for candidate_id in final if candidate_id in set(final[:20]) | positives]
        evidence_ids = [candidate_id for candidate_id in final if candidate_id in set(final[:10]) | positives]
        query_payload[query_id] = _build_query_entry(
            query_id=query_id, query_text=queries[query_id], candidate_ids=candidate_ids,
            display_ids=display_ids, evidence_ids=evidence_ids, positives=positives,
            qrels=qrels[query_id], positions=positions, evaluation=evaluation[query_id],
            selected_ids=selected_ids[query_index], similarities=similarities[query_index],
            records=records,
        )
    return {
        "schema_version": 1,
        "meta": {
            "query_count": len(query_ids),
            "candidate_count": len(candidate_ids),
            "evidence_per_candidate": 3,
            "qrels_usage": "Post-freeze evaluation and failure inspection only",
        },
        "queries": query_payload,
    }
```

- [ ] **Step 6: Run focused tests and commit Task 1**

Run: `uv run pytest tests/test_grounded_evidence_explorer.py -q`

Expected: all tests pass.

Run: `uv run ruff check trec_mlm/grounded_evidence_explorer.py tests/test_grounded_evidence_explorer.py`

Expected: `All checks passed!`

```bash
git add trec_mlm/grounded_evidence_explorer.py tests/test_grounded_evidence_explorer.py
git commit -m "feat: build grounded E5 evidence bundle"
```

---

### Task 2: Derive deterministic cohorts, retrieval outcomes, mechanisms, and context-risk flags

**Files:**
- Modify: `trec_mlm/grounded_evidence_explorer.py`
- Modify: `tests/test_grounded_evidence_explorer.py`

**Interfaces:**
- Consumes: complete rank maps, per-system `ndcg_at_10`, positive candidate IDs, and frozen query text from Task 1.
- Produces: `derive_cohorts(metrics: Mapping[str, float]) -> list[str]`, `derive_retrieval_outcome(positive_ids: Collection[str], positions: Mapping[str, Mapping[str, int]]) -> str`, `derive_mechanism(positive_ids: Collection[str], positions: Mapping[str, Mapping[str, int]]) -> str`, and `has_context_risk(text: str) -> bool`. Each bundle query serializes all four classifications. Existing nDCG cohorts are unchanged; `retrieval_outcome` is an additional rank-1 diagnostic.

- [ ] **Step 1: Write failing table-driven tests for every cohort and mechanism**

```python
@pytest.mark.parametrize("metrics, expected", [
    ({"dense_reference": 0, "profile_only": 0, "dense_profile_equal": 0,
      "raw_hierarchical": 0, "raw_hierarchical_dense_equal": 0},
     ["zero fused nDCG@10", "persistent all-system zero"]),
    ({"dense_reference": .2, "profile_only": 0, "dense_profile_equal": .3,
      "raw_hierarchical": .4, "raw_hierarchical_dense_equal": 0},
     ["zero fused nDCG@10", "raw-success/fusion-suppressed", "new zero regression"]),
    ({"dense_reference": 0, "profile_only": .1, "dense_profile_equal": 0,
      "raw_hierarchical": .1, "raw_hierarchical_dense_equal": .2},
     ["rescued prior zero"]),
])
def test_derive_cohorts(metrics, expected):
    assert derive_cohorts(metrics) == expected


@pytest.mark.parametrize("positions, expected", [
    ({"raw_hierarchical": {"p": 4}, "raw_hierarchical_dense_equal": {"p": 20},
      "best_record_cosine": {"p": 3}}, "fusion suppression"),
    ({"raw_hierarchical": {"p": 14}, "raw_hierarchical_dense_equal": {"p": 20},
      "best_record_cosine": {"p": 3}}, "aggregation suppression"),
    ({"raw_hierarchical": {"p": 14}, "raw_hierarchical_dense_equal": {"p": 20},
      "best_record_cosine": {"p": 13}}, "evidence/routing gap"),
    ({"raw_hierarchical": {"p": 1}, "raw_hierarchical_dense_equal": {"p": 1},
      "best_record_cosine": {"p": 1}}, "relevant candidate ranked first"),
    ({"raw_hierarchical": {"p": 14}, "raw_hierarchical_dense_equal": {"p": 7},
      "best_record_cosine": {"p": 13}}, "top-10 retrieval, top-1 miss"),
])
def test_derive_mechanism(positions, expected):
    assert derive_mechanism({"p"}, positions) == expected


@pytest.mark.parametrize("final_rank, expected", [
    (1, "top-1 success"),
    (7, "top-10 but not top-1"),
    (11, "no relevant candidate in top 10"),
])
def test_derive_retrieval_outcome(final_rank, expected):
    assert derive_retrieval_outcome(
        {"p"}, {"raw_hierarchical_dense_equal": {"p": final_rank}}
    ) == expected
```

- [ ] **Step 2: Run these tests and confirm they fail**

Run: `uv run pytest tests/test_grounded_evidence_explorer.py -q`

Expected: failures name the missing derivation functions.

- [ ] **Step 3: Implement the exact ordered derivation rules**

```python
def derive_cohorts(metrics: Mapping[str, float]) -> list[str]:
    tags: list[str] = []
    new = metrics["raw_hierarchical_dense_equal"]
    prior = metrics["dense_profile_equal"]
    raw = metrics["raw_hierarchical"]
    if new == 0.0:
        tags.append("zero fused nDCG@10")
    if all(metrics[name] == 0.0 for name in SYSTEMS):
        tags.append("persistent all-system zero")
    if new == 0.0 and raw > 0.0:
        tags.append("raw-success/fusion-suppressed")
    if new == 0.0 and prior > 0.0:
        tags.append("new zero regression")
    if new > 0.0 and prior == 0.0:
        tags.append("rescued prior zero")
    return tags


def derive_mechanism(positive_ids: Collection[str], positions: Mapping[str, Mapping[str, int]]) -> str:
    if not positive_ids:
        return "no development-positive candidate"
    def best(system):
        return min(positions[system][candidate_id] for candidate_id in positive_ids)
    if best("raw_hierarchical") <= 10 and best("raw_hierarchical_dense_equal") > 10:
        return "fusion suppression"
    if best("raw_hierarchical_dense_equal") == 1:
        return "relevant candidate ranked first"
    if best("raw_hierarchical_dense_equal") <= 10:
        return "top-10 retrieval, top-1 miss"
    if best("best_record_cosine") <= 10 and best("raw_hierarchical") > 10:
        return "aggregation suppression"
    if best("best_record_cosine") > 10:
        return "evidence/routing gap"
    raise EvidenceBundleError("mechanism rank classification is incomplete")


def derive_retrieval_outcome(positive_ids: Collection[str], positions: Mapping[str, Mapping[str, int]]) -> str:
    if not positive_ids:
        return "no relevant candidate in top 10"
    final_rank = min(positions["raw_hierarchical_dense_equal"][candidate_id] for candidate_id in positive_ids)
    if final_rank == 1:
        return "top-1 success"
    if final_rank <= 10:
        return "top-10 but not top-1"
    return "no relevant candidate in top 10"


CONTEXT_RISK_RE = re.compile(
    r"\b(this|that|these|those|it|they|them|he|she|there|the former|the latter|what happened next)\b",
    re.IGNORECASE,
)


def has_context_risk(text: str) -> bool:
    return bool(CONTEXT_RISK_RE.search(text))
```

- [ ] **Step 4: Add and pass explicit non-causality/context wording tests**

```python
@pytest.mark.parametrize("text, expected", [
    ("What happened next?", True),
    ("Which law did they invoke?", True),
    ("When was USB Type-C Specification 1.0 finalized?", False),
])
def test_context_risk_is_only_a_surface_flag(text: str, expected: bool) -> None:
    assert has_context_risk(text) is expected
```

Run: `uv run pytest tests/test_grounded_evidence_explorer.py -q`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add trec_mlm/grounded_evidence_explorer.py tests/test_grounded_evidence_explorer.py
git commit -m "feat: classify grounded explorer failures"
```

---

### Task 3: Extend the generic explorer with an optional evidence-bundle contract

**Files:**
- Modify: `.agents/skills/inspect-query-results/scripts/build_query_explorer.py`
- Modify: `tests/test_query_explorer_skill.py`

**Interfaces:**
- Consumes: optional `--evidence-bundle PATH`; `load_evidence_bundle(path: Path, query_ids: Collection[str]) -> dict[str, Any]`.
- Produces: existing payload unchanged when absent; when present, each query gains `grounded`, results become final top 20 plus all positives, and `meta.grounded_evidence` describes the embedded evidence. The loader calls `validate_evidence_bundle(...)` from the producer module, rederives every bundle-intrinsic semantic relation, and binds final-component metrics to the generic run/qrels evaluation. Non-final component ranks are validated as complete permutations; their external source-truth binding remains the task-specific builder's responsibility.
- Test helpers: `self.run_builder(*extra_args: str) -> subprocess.CompletedProcess[str]` invokes the existing script against `self.root`; `self.payload_from_output() -> dict[str, Any]` parses `self.run_dir / "query_explorer.html"`; and `self.write_evidence_bundle(...) -> Path` writes a schema-version-1 bundle matching the existing three-candidate fixture unless an override is supplied.

- [ ] **Step 1: Write a failing generic-compatibility test before changing behavior**

```python
def test_no_bundle_preserves_existing_payload_shape(self) -> None:
    self.write_fixture()
    completed = self.run_builder("--top-candidates", "2", "--no-responses")
    self.assertEqual(0, completed.returncode, completed.stderr)
    payload = self.payload_from_output()
    self.assertNotIn("grounded_evidence", payload["meta"])
    self.assertNotIn("grounded", payload["queries"][0])
    self.assertEqual(["model-a", "model-b"], [r["candidate_id"] for r in payload["queries"][0]["results"]])
```

- [ ] **Step 2: Write failing integration tests for positive-row union and strict bundle schema**

```python
def test_bundle_adds_positive_below_limit_without_reordering(self) -> None:
    self.write_complete_qrel_fixture(candidate_count=4, positive_candidate="model-d")
    bundle = self.write_evidence_bundle(display_ids=["model-a", "model-b", "model-d"])
    completed = self.run_builder("--top-candidates", "2", "--evidence-bundle", str(bundle), "--no-responses")
    self.assertEqual(0, completed.returncode, completed.stderr)
    query = self.payload_from_output()["queries"][0]
    self.assertEqual(["model-a", "model-b", "model-d"], [r["candidate_id"] for r in query["results"]])
    self.assertEqual(4, query["results"][-1]["rank"])
    self.assertEqual(2.0, query["results"][-1]["grade"])


def test_bundle_inventory_mismatch_leaves_existing_html_unchanged(self) -> None:
    output = self.run_dir / "query_explorer.html"
    output.write_text("known-good", encoding="utf-8")
    bundle = self.write_evidence_bundle(query_ids=["wrong-query"])
    completed = self.run_builder("--evidence-bundle", str(bundle), "--output", str(output), "--no-responses")
    self.assertEqual(2, completed.returncode)
    self.assertIn("evidence bundle query inventory differs", completed.stderr)
    self.assertEqual("known-good", output.read_text(encoding="utf-8"))
```

- [ ] **Step 3: Retain validated full rows until display selection and add the optional loader**

```python
parser.add_argument(
    "--evidence-bundle",
    type=Path,
    help="Optional validated grounded-evidence JSON; requires complete qrels.",
)


def select_display_rows(values: Sequence[Mapping[str, Any]], *, top_candidates: int,
                        positive_ids: Collection[str] = ()) -> list[dict[str, Any]]:
    wanted = {str(item["candidate_id"]) for item in values[:top_candidates]} | set(positive_ids)
    return [dict(item) for item in values if str(item["candidate_id"]) in wanted]


def load_evidence_bundle(path: Path, *, query_ids: Collection[str]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExplorerInputError(f"could not read evidence bundle: {path}") from error
    if payload.get("schema_version") != 1 or set(payload.get("queries", {})) != set(query_ids):
        raise ExplorerInputError("evidence bundle query inventory differs")
    return payload
```

- [ ] **Step 4: Merge only validated grounded fields into the final payload**

```python
for query in query_rows:
    grounded = evidence_bundle["queries"][query["id"]] if evidence_bundle else None
    if grounded:
        expected_ids = grounded["display_candidate_ids"]
        actual_ids = [row["candidate_id"] for row in query["results"]]
        if actual_ids != expected_ids:
            raise ExplorerInputError(f"evidence display ordering differs for {query['id']}")
        query["grounded"] = grounded

if evidence_bundle:
    payload_meta["grounded_evidence"] = evidence_bundle["meta"]
```

- [ ] **Step 5: Run the generic test file and commit the data-contract change**

Run: `uv run pytest tests/test_query_explorer_skill.py -q`

Expected: all existing and new tests pass.

```bash
git add .agents/skills/inspect-query-results/scripts/build_query_explorer.py tests/test_query_explorer_skill.py
git commit -m "feat: accept grounded explorer evidence"
```

---

### Task 4: Render cohort filters, component ranks, and accessible evidence drawers

**Files:**
- Modify: `.agents/skills/inspect-query-results/scripts/build_query_explorer.py`
- Modify: `tests/test_query_explorer_skill.py`

**Interfaces:**
- Consumes: the merged `query.grounded` contract from Task 3.
- Produces: UI controls for all approved cohorts; a final-rank table with component ranks and cosine summaries; expandable evidence for final top-ten and positive candidates; secondary candidate profile content.

- [ ] **Step 1: Add failing HTML contract tests for approved labels and safe rendering**

```python
def test_grounded_ui_has_filters_ranks_and_neutral_evidence_copy(self) -> None:
    self.write_fixture_with_grounded_bundle()
    html = self.build_and_read_grounded_html()
    for expected in (
        "Persistent all-system zero",
        "Raw success / fusion suppressed",
        "New zero regression",
        "Rescued prior zero",
        "Context-risk flag",
        "Dense rank",
        "Profile rank",
        "Prior fusion rank",
        "Raw hierarchical rank",
        "New fusion rank",
        "E5 cosine similarity",
        "Nearest historical behavior",
        "Best development-positive candidate",
        "Similarity and substantive output do not establish correctness",
    ):
        self.assertIn(expected, html)
    self.assertNotIn('src="http', html)
    self.assertNotIn('href="http', html)


def test_grounded_response_is_json_escaped_and_dom_rendered_with_text_content(self) -> None:
    html = self.build_and_read_grounded_html(response="</script><img src=x onerror=alert(1)>")
    self.assertIn("\\u003c/script\\u003e", html)
    self.assertNotIn("</script><img", html)
    self.assertIn("item.response", html)
    self.assertIn("textContent", html)
```

- [ ] **Step 2: Run the focused UI tests and confirm they fail**

Run: `uv run pytest tests/test_query_explorer_skill.py -q`

Expected: failures identify the missing grounded controls and copy.

- [ ] **Step 3: Add grounded-aware filter predicates and weakest-first behavior**

```javascript
function matchesEvaluationFilter(query) {
  if (!query.grounded) return legacyEvaluationMatch(query);
  const tags = new Set(query.grounded.cohorts || []);
  if (state.evaluationFilter === "zero_fused") return tags.has("zero fused nDCG@10");
  if (state.evaluationFilter === "persistent_zero") return tags.has("persistent all-system zero");
  if (state.evaluationFilter === "fusion_suppressed") return tags.has("raw-success/fusion-suppressed");
  if (state.evaluationFilter === "new_zero_regression") return tags.has("new zero regression");
  if (state.evaluationFilter === "rescued") return tags.has("rescued prior zero");
  if (state.evaluationFilter === "context_risk") return query.grounded.context_risk === true;
  if (state.evaluationFilter === "top_1_success") return query.grounded.retrieval_outcome === "top-1 success";
  if (state.evaluationFilter === "top_10_miss") return query.grounded.retrieval_outcome === "top-10 but not top-1";
  if (state.evaluationFilter === "top_10_absent") return query.grounded.retrieval_outcome === "no relevant candidate in top 10";
  return true;
}
```

- [ ] **Step 4: Render the mechanism summary and component-rank table without cross-system score comparisons**

```javascript
const rankColumns = [
  ["dense_reference", "Dense rank"],
  ["profile_only", "Profile rank"],
  ["dense_profile_equal", "Prior fusion rank"],
  ["raw_hierarchical", "Raw hierarchical rank"],
  ["raw_hierarchical_dense_equal", "New fusion rank"],
];

function appendGroundedCells(row, groundedCandidate) {
  rankColumns.forEach(([key]) => row.appendChild(el("td", "score", groundedCandidate.ranks[key])));
  row.appendChild(el("td", "score", formatMetric(groundedCandidate.best_record_cosine)));
  row.appendChild(el("td", "score", formatMetric(groundedCandidate.mean_top_10_cosine)));
}

function renderBestPositiveSummary(query) {
  const best = query.grounded.best_positive;
  const summary = el("div", "mechanism-summary");
  summary.appendChild(el("strong", "", query.grounded.retrieval_outcome));
  summary.appendChild(el("strong", "", query.grounded.mechanism));
  summary.appendChild(el("p", "", "Best development-positive candidate: " + best.candidate_id));
  summary.appendChild(el("p", "", rankColumns.map(([key, label]) =>
    label + " " + best.ranks[key]).join(" · ")));
  if (query.grounded.context_risk) {
    summary.appendChild(el("span", "context-risk", "Context-risk flag · surface wording only"));
  }
  return summary;
}
```

- [ ] **Step 5: Render exactly three full records in a native accessible drawer**

```javascript
function renderGroundedEvidence(candidate) {
  const details = el("details", "evidence-drawer");
  details.appendChild(el("summary", "", "Nearest historical behavior"));
  details.appendChild(el("p", "section-note",
    "Similarity and substantive output do not establish correctness."));
  candidate.evidence.forEach((item) => {
    const card = el("article", "history-card");
    card.appendChild(el("div", "history-meta",
      "E5 cosine similarity " + formatMetric(item.similarity) +
      " · Historical query " + item.source_query_id + " · " + item.behavior));
    card.appendChild(el("p", "history-question", item.question));
    card.appendChild(el("p", "history-response", item.response));
    details.appendChild(card);
  });
  return details;
}
```

- [ ] **Step 6: Run accessibility-oriented and complete generic tests**

Run: `uv run pytest tests/test_query_explorer_skill.py -q`

Expected: all tests pass, including legacy no-bundle behavior.

Run: `uv run ruff check .agents/skills/inspect-query-results/scripts/build_query_explorer.py tests/test_query_explorer_skill.py`

Expected: `All checks passed!`

- [ ] **Step 7: Commit Task 4**

```bash
git add .agents/skills/inspect-query-results/scripts/build_query_explorer.py tests/test_query_explorer_skill.py
git commit -m "feat: render grounded ranking diagnostics"
```

---

### Task 5: Add the task-specific atomic build command

**Files:**
- Create: `scripts/build_e5_grounded_evidence_explorer.py`
- Create: `tests/test_build_e5_grounded_evidence_explorer.py`

**Interfaces:**
- Consumes CLI flags `--refinement`, `--construction`, `--evaluation`, `--queries`, `--qrels`, `--profiles`, `--bundle-output`, and `--html-output`.
- Produces staged private JSON and HTML siblings, validates the HTML against the staged bundle, and only then promotes both requested outputs. Validation or rendering failure preserves both prior outputs byte-for-byte; stdout contains paths and aggregate counts but no question or response text.
- Test helpers: `run_cli(e5_fixture: Mapping[str, Path], *, bundle: Path, html: Path) -> subprocess.CompletedProcess[str]` invokes the task-specific script; `e5_fixture` uses the complete fixture contract from Task 1 and exposes `corrupt_record_owner()` to change one selected record's candidate only.

- [ ] **Step 1: Write failing CLI orchestration and no-leak tests**

```python
def test_cli_builds_bundle_and_html_without_printing_raw_text(e5_fixture, tmp_path):
    bundle = tmp_path / "private" / "evidence.json"
    html = tmp_path / "rendered" / "explorer.html"
    completed = run_cli(e5_fixture, bundle=bundle, html=html)
    assert completed.returncode == 0, completed.stderr
    assert bundle.is_file() and html.is_file()
    assert "secret historical answer" not in completed.stdout
    assert "secret historical answer" not in completed.stderr
    assert json.loads(bundle.read_text())["meta"]["query_count"] == 2


def test_cli_renderer_failure_preserves_prior_outputs(e5_fixture, tmp_path):
    bundle = tmp_path / "evidence.json"
    html = tmp_path / "explorer.html"
    bundle.write_bytes(b"known-good-bundle")
    html.write_bytes(b"known-good-html")
    e5_fixture.corrupt_profiles()
    completed = run_cli(e5_fixture, bundle=bundle, html=html)
    assert completed.returncode == 2
    assert bundle.read_bytes() == b"known-good-bundle"
    assert html.read_bytes() == b"known-good-html"
    assert not list(tmp_path.rglob(".*.building.*"))
```

- [ ] **Step 2: Run the CLI tests and confirm the red state**

Run: `uv run pytest tests/test_build_e5_grounded_evidence_explorer.py -q`

Expected: collection or subprocess failure because the CLI does not exist.

- [ ] **Step 3: Implement one-command orchestration with two staged siblings**

```python
def main() -> int:
    args = parse_args()
    bundle = build_evidence_bundle(
        refinement_dir=args.refinement,
        construction_dir=args.construction,
        evaluation_path=args.evaluation,
        queries_path=args.queries,
        qrels_path=args.qrels,
    )
    temporary_bundle = _temporary_output_path(args.bundle_output, suffix=".json")
    temporary_html = _temporary_output_path(args.html_output, suffix=".html")
    try:
        write_json_atomically(temporary_bundle, bundle)
        run_generic_builder(
            input_path=args.refinement / "runs" / "raw_hierarchical_dense_equal.trec",
            queries=args.queries,
            qrels=args.qrels,
            profiles=args.profiles,
            evidence_bundle=temporary_bundle,
            output=temporary_html,
            top_candidates=20,
        )
        _promote_output_pair(
            staged_bundle=temporary_bundle,
            bundle=args.bundle_output,
            staged_html=temporary_html,
            html=args.html_output,
        )
    finally:
        temporary_bundle.unlink(missing_ok=True)
        temporary_html.unlink(missing_ok=True)
    print(json.dumps({
        "bundle": str(args.bundle_output.resolve()),
        "html": str(args.html_output.resolve()),
        "query_count": bundle["meta"]["query_count"],
        "evidence_per_candidate": 3,
    }, sort_keys=True))
    return 0
```

- [ ] **Step 4: Pass focused tests, lint, and commit Task 5**

Run: `uv run pytest tests/test_build_e5_grounded_evidence_explorer.py -q`

Expected: all tests pass.

Run: `uv run ruff check scripts/build_e5_grounded_evidence_explorer.py tests/test_build_e5_grounded_evidence_explorer.py`

Expected: `All checks passed!`

```bash
git add scripts/build_e5_grounded_evidence_explorer.py tests/test_build_e5_grounded_evidence_explorer.py
git commit -m "feat: build private E5 evidence explorer"
```

---

### Task 6: Run official-scale verification and update the private artifact

**Files:**
- Modify: `reports/trec_2026_e5_raw_response_failure_ledger.md`
- Modify: `/home/npatta01/codex-rendered/index.html`
- Replace: `/home/npatta01/codex-rendered/plans/e5-raw-response-failure-explorer.html`
- Create outside Git: `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/explorer/evidence_bundle.json`

**Interfaces:**
- Consumes: all code and tests from Tasks 1–5 plus the sealed official artifacts.
- Produces: the stable private portal URL and verification evidence matching the design success criteria.

- [ ] **Step 1: Run the complete relevant test and lint suite before touching the live artifact**

Run:

```bash
uv run pytest \
  tests/test_grounded_evidence_explorer.py \
  tests/test_build_e5_grounded_evidence_explorer.py \
  tests/test_query_explorer_skill.py \
  tests/test_raw_response_refinement_artifact.py \
  tests/test_raw_response_refinement_evaluation.py -q
```

Expected: all tests pass.

Run:

```bash
uv run ruff check \
  trec_mlm/grounded_evidence_explorer.py \
  scripts/build_e5_grounded_evidence_explorer.py \
  tests/test_grounded_evidence_explorer.py \
  tests/test_build_e5_grounded_evidence_explorer.py \
  tests/test_query_explorer_skill.py \
  .agents/skills/inspect-query-results/scripts/build_query_explorer.py
```

Expected: `All checks passed!`

- [ ] **Step 2: Build to temporary non-live paths using the official frozen inputs**

```bash
uv run python scripts/build_e5_grounded_evidence_explorer.py \
  --refinement /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/refinement_run1 \
  --construction /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/construction \
  --evaluation /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/evaluation/evaluation.json \
  --queries data/dev/llm_dev_data.tsv \
  --qrels data/dev/llm_dev_qrels.txt \
  --profiles reports/trec_2026_llm_profiles.json \
  --bundle-output /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/explorer/evidence_bundle.candidate.json \
  --html-output /home/npatta01/codex-rendered/plans/e5-raw-response-failure-explorer.candidate.html
```

Expected: JSON summary reports `query_count: 322` and `evidence_per_candidate: 3` without printing raw questions or answers.

- [ ] **Step 3: Run an independent artifact audit before replacement**

```bash
uv run python - <<'PY'
import json, re
from pathlib import Path

path = Path('/home/npatta01/codex-rendered/plans/e5-raw-response-failure-explorer.candidate.html')
html = path.read_text(encoding='utf-8')
match = re.search(r'<script id="explorerData" type="application/json">(.*?)</script>', html, re.S)
assert match
data = json.loads(match.group(1))
assert len(data['queries']) == 322
assert abs(data['meta']['macro_ndcg_at_10'] - 0.435625273018) < 1e-12
cohort_counts = {}
for query in data['queries']:
    grounded = query['grounded']
    for tag in grounded['cohorts']:
        cohort_counts[tag] = cohort_counts.get(tag, 0) + 1
    ids = [row['candidate_id'] for row in query['results']]
    assert ids == grounded['display_candidate_ids']
    assert len(ids) >= 20
    for candidate_id in grounded['evidence_candidate_ids']:
        evidence = grounded['candidates'][candidate_id]['evidence']
        assert len(evidence) == 3
        assert all(item['response'] is not None for item in evidence)
assert cohort_counts['zero fused nDCG@10'] == 92
assert cohort_counts['persistent all-system zero'] == 62
assert cohort_counts['raw-success/fusion-suppressed'] == 24
assert cohort_counts['new zero regression'] == 1
assert cohort_counts['rescued prior zero'] == 8
assert 'src="http' not in html and 'href="http' not in html
assert 'Nearest historical behavior' in html
assert 'E5 cosine similarity' in html
print({'queries': len(data['queries']), 'bytes': path.stat().st_size})
PY
```

Expected: assertions pass and only aggregate query/byte counts print.

- [ ] **Step 4: Rebuild directly to the stable pair after the candidate audit**

```bash
uv run python scripts/build_e5_grounded_evidence_explorer.py \
  --refinement /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/refinement_run1 \
  --construction /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/construction \
  --evaluation /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/evaluation/evaluation.json \
  --queries data/dev/llm_dev_data.tsv \
  --qrels data/dev/llm_dev_qrels.txt \
  --profiles reports/trec_2026_llm_profiles.json \
  --bundle-output /home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/explorer/evidence_bundle.json \
  --html-output /home/npatta01/codex-rendered/plans/e5-raw-response-failure-explorer.html
```

Expected: the CLI stages and validates both artifacts, then promotes the stable
pair together. A failure leaves both prior stable paths byte-for-byte unchanged.
Remove the `.candidate` files only after the stable outputs pass the same
artifact audit.

- [ ] **Step 5: Update the failure ledger and portal description with private-data disclosure**

```markdown
The complete private interactive explorer covers all 322 development queries,
defaults to weakest nDCG first, and supports the frozen failure cohorts. It
shows the final top 20 plus all development-positive candidates, component
ranks, and exactly three nearest historical question-response records for each
final top-ten or positive candidate. Historical answers are embedded in the
private HTML; similarity and substantive output do not establish correctness.
```

The `/home/npatta01/codex-rendered/index.html` entry must say: `Private: embeds selected full historical answers; tailnet access only.`

- [ ] **Step 6: Verify the live HTTPS artifact and tailnet-only mapping**

Run:

```bash
curl --fail --silent --show-error \
  https://npatta01-framework.tail481212.ts.net/plans/e5-raw-response-failure-explorer.html \
  | rg -q 'E5 grounded-evidence failure explorer|Nearest historical behavior'
tailscale serve status
```

Expected: `curl` exits 0; Serve status shows the existing HTTPS mapping and no Funnel/public exposure.

- [ ] **Step 7: Commit only the repository-owned report update**

```bash
git add reports/trec_2026_e5_raw_response_failure_ledger.md
git diff --cached --check
git commit -m "docs: link grounded E5 failure explorer"
```

- [ ] **Step 8: Run final repository and privacy checks**

Run:

```bash
git status --short
git show --stat --oneline HEAD
git ls-files | xargs rg -l 'Nearest historical behavior' || true
```

Expected: worktree still contains only the user's pre-existing unrelated changes beyond this task; the final commit contains only the failure ledger; raw historical answers and the private evidence bundle are not tracked by Git.
