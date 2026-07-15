# Candidate-Aware Semantic Profiles Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, repeat, and evaluate candidate-specific E5 semantic profiles and fixed qrel-free fusions against the frozen dense query-kNN reference.

**Architecture:** Keep qrel-free construction in three focused modules: pure profile mathematics and deterministic clustering, pinned ROCm inference, and atomic artifact orchestration. Put qrel loading and all effectiveness calculations in a separate evaluation module that consumes an immutable construction manifest read-only. Reuse the existing run I/O, metrics, contextual-E5 contracts, discovery loader, CLI style, report conventions, and post-run query-explorer skill.

**Approved execution amendment:** The pinned ROCm image uses Python 3.12/NumPy 2.1,
while clustering is locked to Python 3.13.9/NumPy 2.5.1. Therefore inference and
construction run as two qrel-free, hash-locked stages exactly as specified in the
protocol's “Locked two-stage execution boundary.” Both stages are repeated fresh;
construction consumes persisted inference arrays without transformation, and qrels
remain unavailable until both repeat gates pass.

**Tech Stack:** Python 3.13.9, NumPy 2.5.1, SciPy 1.18.0, scikit-learn 1.9.0, Transformers 5.12.1 in the pinned ROCm image, PyTorch HIP, pytest, Markdown, standalone accessible HTML.

## Global Constraints

- Authoritative protocol: `protocols/trec_2026_candidate_semantic_profiles.md` at commit `9f01d82`.
- Construction APIs and configs must contain no qrel input, relevance grade, effectiveness metric, or tunable fusion selector.
- Freeze all six complete rankings before qrels are opened; equal-weight dense+profile RRF at `k=60` is the only promotion candidate.
- Use only pinned E5 revision `d62cc4417be0e3289c03dd459d1ef2d137e2554e` and container digest `sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`.
- Profile inference is BF16, batch 8, seed 2026, 512 tokens including exactly one EOS, normalized last-token pooling, persisted C-contiguous float32.
- K-means uses one numerical thread, stable record order, `KMeans(init="k-means++", random_state=2026, n_init=10, algorithm="lloyd", max_iter=300, tol=1e-4)`.
- No support penalty, learned ranker, LLM judge, encoder sweep, nested fusion selection, adaptive batch fallback, or continuation of completed Qwen/Jina/pointwise/listwise/GroupRank work.
- Follow strict red-green-refactor TDD: every production behavior begins with a failing focused test.
- Complete local artifacts remain private; only compact reports and receipts enter git.

---

## File Structure

- `trec_mlm/profile_retrieval.py`: immutable contracts, filtering/deduplication, canonical K-means, profile scoring, label aggregation, and fixed RRF.
- `trec_mlm/profile_inference.py`: exact E5 profile formatting/tokenization and pinned ROCm embedding execution.
- `trec_mlm/profile_experiment.py`: qrel-free configuration, input loading, construction orchestration, atomic artifact writing/loading, hashes, validation, and repeat comparison.
- `trec_mlm/profile_evaluation.py`: read-only qrel evaluation, paired statistics, slices, Phase 0 gate, promotion receipt, and report-ready payload.
- `scripts/build_candidate_profile_report.py`: deterministic Markdown and accessible standalone HTML report from validated evaluation artifacts.
- `trec_mlm/cli.py`: thin construct, validate, repeat-check, evaluate, and report adapters.
- `tests/test_profile_retrieval.py`, `tests/test_profile_inference.py`, `tests/test_profile_experiment.py`, `tests/test_profile_evaluation.py`: focused unit and integration coverage.
- `reports/trec_2026_candidate_semantic_profiles.{md,html,json}` and `reports/trec_2026_candidate_semantic_profiles_receipt.json`: compact tracked final outputs.
- `README.md`: reproducible commands and qrel boundary.

### Task 1: Pure profile records, clustering, scoring, labels, and RRF

**Files:**
- Create: `trec_mlm/profile_retrieval.py`
- Create: `tests/test_profile_retrieval.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `DiscoveryRecord`, `response_filter_reason`, normalized float32 embeddings, complete source rankings.
- Produces: `ProfileRecord`, `CanonicalCluster`, `CandidateProfile`, `ProfileScoreResult`, `RRFResult`, `filter_and_deduplicate_records`, `canonical_kmeans`, `score_profiles`, `aggregate_cluster_labels`, and `reciprocal_rank_fusion`.

- [ ] **Step 1: Write failing record-contract tests**

```python
def test_basic_filter_and_raw_exact_dedup_are_candidate_local() -> None:
    records = (
        DiscoveryRecord("q1", " Question ", "A", " Answer "),
        DiscoveryRecord("q2", " Question ", "A", " Answer "),
        DiscoveryRecord("q3", "Question", "A", "Answer"),
        DiscoveryRecord("q4", "Question", "A", "I don't know"),
        DiscoveryRecord("q1", " Question ", "B", " Answer "),
    )
    result = filter_and_deduplicate_records(records, ("A", "B"))
    assert [record.source_query_id for record in result.records_by_candidate["A"]] == ["q1", "q3"]
    assert result.audit_by_candidate["A"].removed_exact_idk == 1
    assert result.audit_by_candidate["A"].duplicates == 1
    assert len(result.records_by_candidate["B"]) == 1
```

- [ ] **Step 2: Run the focused test and confirm RED**

Run: `PYTHONPATH=. uv run --with pytest --extra dense --extra response python -m pytest tests/test_profile_retrieval.py -q`

Expected: collection fails because `trec_mlm.profile_retrieval` does not exist.

- [ ] **Step 3: Implement immutable record and audit contracts**

Create frozen slotted dataclasses with these exact public fields:

```python
@dataclass(frozen=True, slots=True)
class ProfileRecord:
    record_id: str
    source_query_id: str
    candidate_id: str
    question: str
    response: str

@dataclass(frozen=True, slots=True)
class CandidateRecordAudit:
    raw: int
    removed_null: int
    removed_blank: int
    removed_exact_idk: int
    duplicates: int
    usable: int

@dataclass(frozen=True, slots=True)
class FilteredProfileRecords:
    candidate_ids: tuple[str, ...]
    records_by_candidate: dict[str, tuple[ProfileRecord, ...]]
    audit_by_candidate: dict[str, CandidateRecordAudit]
```

Use `response_filter_reason(response, "basic")`, exact raw `(question, response)` keys, and record IDs `f"{source_query_id}:{sha256(length-framed raw question and response)[:16]}"`. Reject duplicate candidate IDs and malformed candidate inventories.

- [ ] **Step 4: Verify the record tests pass**

Run the Step 2 command.

Expected: record filtering/deduplication tests pass.

- [ ] **Step 5: Write failing canonical clustering tests**

```python
def test_identical_embeddings_collapse_and_centroids_are_canonical() -> None:
    records = tuple(_profile_record(f"q{i}", "A") for i in range(3))
    embeddings = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    profile = canonical_kmeans("A", records, embeddings, requested_clusters=3)
    assert profile.requested_clusters == 3
    assert len(profile.clusters) == 1
    assert profile.clusters[0].cluster_id == 0
    assert profile.clusters[0].member_record_ids == tuple(record.record_id for record in records)
    np.testing.assert_array_equal(profile.centroids, [[1.0, 0.0]])
```

Also test empty input, `K=min(10,n)`, stable cluster numbering by minimum record ID, byte-identical centroid merging, normalized centroids, mean cosine dispersion, medoid and representative ties, and one-thread environment validation.

- [ ] **Step 6: Run clustering tests and confirm RED**

Run: `PYTHONPATH=. uv run --with pytest --extra dense python -m pytest tests/test_profile_retrieval.py -k 'cluster or centroid or identical' -q`

Expected: failures for missing `canonical_kmeans` and profile contracts.

- [ ] **Step 7: Implement canonical clustering**

Add direct `scikit-learn==1.9.0` to the `dense` optional dependency and refresh `uv.lock`. Implement `canonical_kmeans(candidate_id: str, records: Sequence[ProfileRecord], embeddings: FloatArray, *, requested_clusters: int | None = None) -> CandidateProfile`.

Validate C-contiguous finite normalized float32 inputs, call the frozen `KMeans` parameters, merge exact centroid-byte duplicates, recompute each occupied centroid through a float64 mean followed by float32 normalization, and sort clusters by minimum member record ID. Use scikit-learn `threadpool_limits(limits=1)` and reject non-one thread environment values before fitting.

- [ ] **Step 8: Verify clustering tests pass**

Run the Step 6 command.

Expected: all canonical clustering tests pass without convergence warnings escaping the explicit identical-vector fixture handling.

- [ ] **Step 9: Write failing scoring, annotation, and RRF tests**

```python
def test_unavailable_profile_has_tail_sentinel_and_zero_fusion_term() -> None:
    scored = score_profiles(
        query_ids=("q",), candidate_ids=("A", "B"),
        query_embeddings=np.asarray([[1.0, 0.0]], np.float32),
        profiles={"A": _profile([[1.0, 0.0]]), "B": None},
    )
    assert scored.rankings["q"] == ["A", "B"]
    assert scored.scores[0, 1] == -2.0
    fused = reciprocal_rank_fusion(
        query_ids=("q",), candidate_ids=("A", "B"),
        rankings_by_signal={"dense": {"q": ["B", "A"]}, "profile": scored.rankings},
        weights={"dense": 1.0, "profile": 1.0}, unavailable_profile_ids=frozenset({"B"}), k=60,
    )
    assert fused.scores[0, 1] == pytest.approx(1 / 61)
```

Add exact tests for maximum cosine, winning-centroid ties, candidate-ID rank ties, all five fixed RRF formulas, `0.60/0.02` label boundaries, and annotation mutation independence.

- [ ] **Step 10: Implement scoring, annotations, and fixed RRF**

Implement the exact public signatures `score_profiles(*, query_ids, candidate_ids, query_embeddings, profiles) -> ProfileScoreResult`, `aggregate_cluster_labels(*, member_question_indices, question_label_scores, label_ids) -> ClusterAnnotation`, and `reciprocal_rank_fusion(*, query_ids, candidate_ids, rankings_by_signal, weights, unavailable_profile_ids=frozenset(), k=60) -> RRFResult`.

All outputs are finite float32 score matrices plus complete deterministic rankings. Keep annotations out of every scoring signature.

- [ ] **Step 11: Run all pure-profile tests and the full suite**

Run:

```bash
PYTHONPATH=. uv run --with pytest --extra dense --extra response python -m pytest tests/test_profile_retrieval.py -q
PYTHONPATH=. uv run --with pytest --extra dense --extra response python -m pytest -q
```

Expected: new tests pass; existing 126 tests and 11 subtests remain green.

- [ ] **Step 12: Commit the pure profile core**

```bash
git add pyproject.toml uv.lock trec_mlm/profile_retrieval.py tests/test_profile_retrieval.py
git commit -m "feat: add deterministic candidate profile core"
```

### Task 2: Exact E5 profile inference

**Files:**
- Create: `trec_mlm/profile_inference.py`
- Create: `tests/test_profile_inference.py`
- Modify: `trec_mlm/label_inference.py`

**Interfaces:**
- Consumes: retained `validate_rocm_runtime`, `snapshot_revision_from_path`, and `last_token_pool` helpers.
- Produces: `PROFILE_TASK_INSTRUCTION`, `format_profile_query`, `format_profile_document`, `build_profile_token_ids`, `encode_profile_inputs`.

- [ ] **Step 1: Write failing exact formatting and token tests**

```python
def test_profile_formatting_is_exact() -> None:
    assert format_profile_query("  Why?  ") == (
        "Instruct: Given a user question, retrieve historical question-and-response "
        "records from LLMs that are likely to answer it correctly.\nQuery: Why?"
    )
    assert format_profile_document(" Q? ", " A. ") == "Question: Q?\nResponse: A."

def test_token_builder_right_truncates_and_appends_one_eos() -> None:
    tokenizer = _FakeTokenizer(rows=[[10, 11, 2, 2, 2], list(range(600))])
    rows, metadata = build_profile_token_ids(tokenizer, ("a", "b"), max_length=512)
    assert rows[0] == [10, 11, 2]
    assert len(rows[1]) == 512 and rows[1][-1] == 2
    assert metadata["truncated_input_count"] == 1
```

Cover left-padding metadata, question-prefix full-response truncation, max length less than two, missing/wrong EOS, and zero/invalid batches.

- [ ] **Step 2: Run inference tests and confirm RED**

Run: `PYTHONPATH=. uv run --with pytest --extra dense python -m pytest tests/test_profile_inference.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement formatting and tokenizer contract**

Implement the protocol text literally. `build_profile_token_ids` must tokenize once without truncation for diagnostics and once with right truncation, remove all terminal EOS IDs, take at most 511 remaining IDs, append ID 2, and return partial/full-response truncation counts derived from separately tokenizing the formatted question prefix.

- [ ] **Step 4: Verify formatting/token tests pass**

Run the Step 2 command.

- [ ] **Step 5: Write failing inference-boundary tests**

Test that `encode_profile_inputs` accepts only batch 8 and seed 2026, checks the exact resolved model and tokenizer revision, requires HIP/gfx, outputs C-contiguous normalized float32 of shape `(input_count, 4096)`, and returns all protocol metadata fields. Mock only model/tokenizer execution; exercise real normalization and metadata validation.

- [ ] **Step 6: Implement pinned ROCm execution**

Reuse the retained E5 helpers without changing label-ranking behavior. Load the exact model/tokenizer snapshot, BF16 model, left padding, batch 8, and normalized last-token pooling. Record peak allocated/reserved bytes, device properties, versions, token counts, hashes, and elapsed seconds. No CPU or batch fallback exists.

- [ ] **Step 7: Run inference and regression tests**

Run:

```bash
PYTHONPATH=. uv run --with pytest --extra dense python -m pytest tests/test_profile_inference.py tests/test_label_inference.py tests/test_label_experiment.py -q
```

Expected: all pass.

- [ ] **Step 8: Commit exact inference**

```bash
git add trec_mlm/profile_inference.py trec_mlm/label_inference.py tests/test_profile_inference.py
git commit -m "feat: add pinned E5 profile inference"
```

### Task 3: Qrel-free construction and immutable artifacts

**Files:**
- Create: `trec_mlm/profile_experiment.py`
- Create: `tests/test_profile_experiment.py`

**Interfaces:**
- Consumes: Tasks 1-2, discovery path, target queries, candidate manifest, frozen dense/E5 artifacts.
- Produces: `ProfileExperimentConfig`, `ProfileExperimentResult`, `run_profile_experiment`, `write_profile_artifacts`, `load_profile_artifacts`, `compare_profile_repeats`.

- [ ] **Step 1: Write failing public-boundary and construction tests**

```python
def test_public_construction_boundary_contains_no_qrels() -> None:
    assert not any("qrel" in field.name.lower() for field in fields(ProfileExperimentConfig))
    for function in (run_profile_experiment, write_profile_artifacts, load_profile_artifacts):
        assert "qrel" not in str(inspect.signature(function)).lower()

def test_construction_succeeds_when_qrels_are_absent(tmp_path: Path) -> None:
    result = _run_fixture_profile_experiment(tmp_path)
    assert result.summary["qrel_free"] is True
    assert set(result.rankings) == set(FROZEN_RANKING_KEYS)
```

- [ ] **Step 2: Run construction tests and confirm RED**

Run: `PYTHONPATH=. uv run --with pytest --extra dense --extra response python -m pytest tests/test_profile_experiment.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement frozen configuration and orchestration**

Define `ProfileExperimentConfig` with only paths and fixed operational values: query path, discovery path, candidate manifest, dense artifact, E5 artifact, output run IDs, batch 8, seed 2026, and thread count 1. Validation rejects every value that differs from the protocol. Load discovery through `load_discovery_records`; reuse retained E5 label embeddings only after validating their complete artifact.

- [ ] **Step 4: Verify fixture construction passes**

Run the Step 2 command using private embedding injection only in tests, mirroring the retained label experiment's private fixture boundary.

- [ ] **Step 5: Write failing atomic artifact and mutation tests**

Require these files in the construction manifest: config/provenance, inventories, audits, formatted inputs, record/query/question-label embeddings, memberships, centroids, annotations, profile scores, winning centroids, six prediction/ranking/run sets, validation, and manifest. Test atomic failure cleanup, mutation rejection, 322x150 completeness, score/rank recomputation, and that evaluator-like qrel mutations cannot alter any manifest hash.

- [ ] **Step 6: Implement artifact writer, loader, and canonical manifest**

Use a staging directory plus `os.replace`. Hash arrays with shape/dtype/bytes and text files with SHA-256. Exclude timestamps only from canonical repeat hashes, never from the file manifest. Loader validates source artifact hashes, inventory order, centroid/member invariants, all score arithmetic, winning centroids, predictions, rankings, TREC files, and aggregate manifest hash.

- [ ] **Step 7: Write and implement repeat comparison**

`compare_profile_repeats(first, second)` must require identical canonical input, membership, centroid, annotation, winning-centroid and ranking hashes; compute maximum absolute profile/fusion score difference; require `<=1e-5`; and emit a JSON receipt with per-check pass/fail values.

- [ ] **Step 8: Run construction tests and full regression**

Run:

```bash
PYTHONPATH=. uv run --with pytest --extra dense --extra response python -m pytest tests/test_profile_experiment.py -q
PYTHONPATH=. uv run --with pytest --extra dense --extra response python -m pytest -q
```

Expected: all pass.

- [ ] **Step 9: Commit construction artifacts**

```bash
git add trec_mlm/profile_experiment.py tests/test_profile_experiment.py
git commit -m "feat: build immutable semantic profile artifacts"
```

### Task 4: Separate evaluation, statistical gates, and receipts

**Files:**
- Create: `trec_mlm/profile_evaluation.py`
- Create: `tests/test_profile_evaluation.py`

**Interfaces:**
- Consumes: read-only validated construction directory and qrels path.
- Produces: `ProfileEvaluationConfig`, `evaluate_profile_artifacts`, `write_profile_evaluation`, `paired_interval`, `zero_transitions`, `phase0_decision`, `profile_decision`.

- [ ] **Step 1: Write failing statistics and zero-transition tests**

Use small exact fixtures to assert `Generator(PCG64(2026))`, 10,000 percentile samples, 1e-12 W/T/L equality, and the four transition cells summing to query count.

- [ ] **Step 2: Run evaluation tests and confirm RED**

Run: `PYTHONPATH=. uv run --with pytest --extra dense python -m pytest tests/test_profile_evaluation.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement read-only evaluation and slices**

Load and validate construction before qrels. Snapshot construction hashes before and after evaluation and fail on any difference. Reuse `trec_mlm.metrics` and `load_complete_qrels`. Implement density slices `<=5`, `6-20`, `>20` and first-word types `what/who/when/where/which/how/why/other` exactly.

- [ ] **Step 4: Write failing Phase 0 and final receipt boundary tests**

Test every threshold at one ULP above/below: Phase 0 `delta_ndcg>0`, `delta_mrr>=-0.005`, zeros `<116`; promotion lower CI `>0`, MRR `>=-0.005`, zeros `<116`, low-density delta `>0`, all integrity checks; diagnostic/reject/reject-invalid fallbacks.

- [ ] **Step 5: Implement decisions and separate atomic evaluation output**

Write evaluation JSON, per-query TSV, comparisons JSON, zero transitions, Phase 0 receipt, profile receipt, and a before/after construction hash receipt to a new directory. Never open construction files in write mode.

- [ ] **Step 6: Run evaluation and regression tests**

Run:

```bash
PYTHONPATH=. uv run --with pytest --extra dense python -m pytest tests/test_profile_evaluation.py tests/test_metrics.py -q
PYTHONPATH=. uv run --with pytest --extra dense --extra response python -m pytest -q
```

Expected: all pass.

- [ ] **Step 7: Commit evaluation**

```bash
git add trec_mlm/profile_evaluation.py tests/test_profile_evaluation.py
git commit -m "feat: evaluate frozen profile rankings"
```

### Task 5: CLI, accessible report, and documentation

**Files:**
- Modify: `trec_mlm/cli.py`
- Create: `scripts/build_candidate_profile_report.py`
- Modify: `README.md`
- Modify: `tests/test_profile_experiment.py`
- Modify: `tests/test_profile_evaluation.py`

**Interfaces:**
- Consumes: Tasks 3-4 public APIs.
- Produces: `profile-construct`, `profile-validate`, `profile-repeat-check`, `profile-evaluate`, and deterministic report builder commands.

- [ ] **Step 1: Write failing parser and adapter tests**

Assert construction commands expose no qrel option; only `profile-evaluate` accepts `--qrels`; adapters instantiate Python config objects and call one public function; fixed model, batch, RRF, and weights have no CLI switches.

- [ ] **Step 2: Run CLI tests and confirm RED**

Run: `PYTHONPATH=. uv run --with pytest --extra dense python -m pytest tests/test_profile_experiment.py tests/test_profile_evaluation.py -k cli -q`

Expected: missing subcommands.

- [ ] **Step 3: Implement thin CLI adapters**

Follow existing CLI structure. Construction takes only query/discovery/candidate/dense/E5/output paths. Evaluation takes construction/qrels/output paths. Repeat-check takes two construction directories and receipt path.

- [ ] **Step 4: Write failing report tests**

Build from a compact fixture and assert Markdown/HTML contain the headline metrics, Phase 0 status, promotion gates, zero transitions, density and question-type slices, coverage/truncation/runtime, limitations, source paths, receipt outcome, a skip link, semantic headings/tables, visible focus CSS, and no remote JavaScript dependency.

- [ ] **Step 5: Implement deterministic Markdown and standalone HTML report**

Render only validated evaluation JSON. Escape all user/source text, use semantic HTML, caption tables, include text equivalents for any visual, and write fixed-content outputs whose hashes do not depend on wall-clock time.

- [ ] **Step 6: Update README with exact workflow**

Document construct -> validate/repeat -> evaluate -> report ordering, emphasize qrels are absent until evaluation, show local artifact paths, and state that the dense input is an already-frozen qrel-trained signal.

- [ ] **Step 7: Run focused and full tests**

Run the complete suite with both optional extras. Expected: all pass.

- [ ] **Step 8: Commit CLI/report/docs**

```bash
git add trec_mlm/cli.py scripts/build_candidate_profile_report.py README.md tests/test_profile_experiment.py tests/test_profile_evaluation.py
git commit -m "feat: expose candidate profile experiment workflow"
```

### Task 6: Qrel-free smoke, Phase 0 lock, and full repeated construction

**Files:**
- Create locally under ignored `artifacts/candidate_semantic_profiles/`; do not commit large artifacts.

**Interfaces:**
- Consumes: official discovery data, retained E5 artifacts, frozen dense artifact, pinned ROCm image.
- Produces: smoke receipt, two complete construction directories, repeat receipt, and six frozen ranking hashes before qrels.

- [ ] **Step 1: Verify hardware, image, model cache, disk, and source hashes**

Run read-only ROCm, Podman, filesystem, and source-hash checks. Require `gfx1151`, the exact local image digest, pinned model snapshot, at least 40 GiB available RAM, and at least 20 GiB free disk.

- [ ] **Step 2: Run focused tests from a clean worktree**

Run the full suite and require green before container execution.

- [ ] **Step 3: Run the protocol smoke twice without qrels mounted**

Mount only code, discovery/query/candidate inputs, model cache read-only, and a new smoke output directory. Use batch 8 and the deterministic short/median/truncated record inventory. Compare fresh-process outputs and write the smoke repeat receipt. Stop for protocol amendment on failure.

- [ ] **Step 4: Freeze Phase 0 and all profile-dependent rankings in run 1**

Run `profile-construct` without mounting or passing qrels. Validate that six ranking sets are complete and manifest-locked.

- [ ] **Step 5: Repeat full construction in a fresh process**

Use an independent new output directory with identical inputs and no qrels. Do not reuse the first run's embeddings or memberships.

- [ ] **Step 6: Compare complete repeats**

Require identical inventories/memberships/centroids/annotations/winners/ranks and max score difference `<=1e-5`. Save the receipt. If it fails, diagnose with systematic-debugging before any qrels are opened.

### Task 6A: Repair candidate-aware record identity and re-gate pass one

**Files:**
- Modify: `trec_mlm/profile_retrieval.py`
- Modify: `trec_mlm/profile_inference_artifact.py`
- Modify: `trec_mlm/cli.py`
- Modify: `tests/test_profile_retrieval.py`
- Modify: `tests/test_profile_inference_artifact.py`
- Modify: `tests/test_profile_cli_report.py`
- Modify: `protocols/trec_2026_candidate_semantic_profiles.md`

**Interfaces:**
- Consumes: the immutable pass-one inference artifact with aggregate SHA-256 `20a72f0cbb04ef1c1e58a7caf2d10c9357f3529c1bcd3e66f4a7ec5f371657c5`.
- Produces: candidate-aware record IDs, a separately manifested repair artifact that reuses byte-identical inputs and arrays, a repaired construction, and a replacement qrel-free viability receipt.

- [ ] **Step 1: Obtain advisor approval for the protocol amendment**

Require an explicit approve/revise/reject review of the candidate-aware ID function, immutable repair artifact, byte-identity checks, repeat comparison, and continued qrel seal. Record the review before changing implementation.

- [ ] **Step 2: Write failing identity and repair-boundary tests**

Assert that the same source query/question/response under candidates `A` and `B` receives two distinct IDs, both equal to the canonical length-framed hash of `(candidate_id, source_query_id, question, response)`. Assert the ordinary loader rejects globally duplicated IDs. Assert repair accepts only the pinned pass-one aggregate, changes no record ordering/content except `record_id`, preserves every formatted input, array, and non-record source file byte-for-byte where applicable, and records source/destination hashes plus the protocol-amendment hash.

- [ ] **Step 3: Run focused tests and confirm RED**

Run:

```bash
PYTHONPATH=. uv run --with pytest --extra dense python -m pytest tests/test_profile_retrieval.py tests/test_profile_inference_artifact.py tests/test_profile_cli_report.py -q
```

Expected: failures for cross-candidate ID uniqueness, strict duplicate validation, and the missing repair command.

- [ ] **Step 4: Implement the minimal repair**

Add one canonical public record-ID helper used by discovery filtering and the repair path. Add a narrowly scoped atomic repair API/CLI that verifies the frozen source manifest before reading, rewrites only record IDs, copies arrays without numerical serialization, writes explicit provenance, and refuses any source other than the pinned pass-one artifact. Keep qrels absent from every signature.

- [ ] **Step 5: Run focused and full regression tests**

Run the focused command from Step 3, then:

```bash
PYTHONPATH=. uv run --with pytest --extra dense --extra response python -m pytest -q
```

Expected: all pass.

- [ ] **Step 6: Materialize and independently validate the repaired pass-one artifact**

Write to a new ignored directory. Verify the original artifact hash is unchanged; all four `.npy` SHA-256 values, all formatted inputs, record ordering, candidate/source/question/response fields, audits, E5 rankings, and encoder input hashes are unchanged; all 182,764 repaired IDs are globally unique and canonical. Save a compact tracked repair receipt.

- [ ] **Step 7: Reconstruct and rerun the qrel-free viability gate**

Build a new construction from the repaired artifact, freeze one provisional `profile_only` ranking, and rerun the unchanged gate thresholds with qrels absent. Replace neither the original artifacts nor the original stop receipt; write a superseding receipt linked to both.

- [ ] **Step 8: Obtain advisor code-and-result review**

Provide the diff, focused/full test evidence, repair receipt, construction manifest, and viability receipt. Continue only on explicit approval and a `continue` viability decision. If approved, run the second full inference fresh under the repaired code and compare semantic inputs, canonical IDs, arrays, construction, and rankings under the amended repeat contract before opening qrels.

### Task 7: Post-lock evaluation, report, query explorer, and receipt

**Files:**
- Create: `reports/trec_2026_candidate_semantic_profiles.md`
- Create: `reports/trec_2026_candidate_semantic_profiles.html`
- Create: `reports/trec_2026_candidate_semantic_profiles.json`
- Create: `reports/trec_2026_candidate_semantic_profiles_receipt.json`
- Create locally: ignored evaluation and query-explorer artifact directories.

**Interfaces:**
- Consumes: validated run-1 construction and qrels only after Task 6 passes.
- Produces: complete evaluation, accessible reports, query explorer, and final decision.

- [ ] **Step 1: Record the ranking lock and open qrels exactly once through evaluation**

Run `profile-evaluate` into a separate directory. Verify construction hashes before/after are identical.

- [ ] **Step 2: Build and validate report outputs**

Generate Markdown and standalone HTML from evaluation JSON. Run report tests, link checks, and HTML accessibility checks available in the repo environment.

- [ ] **Step 3: Build the query explorer using `$inspect-query-results`**

Follow the skill against the completed artifact directory. Extend the explorer input only through retained artifact fields so each query/candidate view shows winning centroid, support, dispersion, labels, medoid, and representative questions. Keep it local/private.

- [ ] **Step 4: Inspect required failure cases without changing rankings**

Use the frozen explorer for autosegmental phonology, Thuringia mountain, all dense-zero rescues/harms, largest wins/losses, low-density slices, truncation extremes, and diffuse labels. Record observations only in the report; do not rerank.

- [ ] **Step 5: Materialize the exact decision receipt**

Confirm Phase 0 and profile outcomes derive mechanically from full-precision JSON. The receipt must be exactly one of promote, diagnostic, reject, or reject-invalid with every gate value.

- [ ] **Step 6: Commit compact final evidence**

```bash
git add reports/trec_2026_candidate_semantic_profiles.md reports/trec_2026_candidate_semantic_profiles.html reports/trec_2026_candidate_semantic_profiles.json reports/trec_2026_candidate_semantic_profiles_receipt.json
git commit -m "docs: report candidate semantic profile results"
```

### Task 8: Final verification and PR scope audit

**Files:**
- Modify only files required by failures discovered here.

**Interfaces:**
- Consumes: complete branch and local artifacts.
- Produces: clean verification evidence and PR-ready branch.

- [ ] **Step 1: Run verification-before-completion checks**

Run full tests, focused official-artifact validators, repeat receipt validation, report regeneration/diff check, `git diff --check`, and `git status --short`.

- [ ] **Step 2: Audit scope and forbidden experiments**

Inspect `git diff origin/main...HEAD`. Confirm there is no Qwen, Jina, pointwise, listwise, GroupRank, FirstMistral, encoder sweep, learned ranker, qrel-bearing construction API, exploratory script, large embedding, or local path leakage.

- [ ] **Step 3: Review implementation against every protocol section**

Create a temporary checklist mapping each protocol invariant to a passing test, validator field, artifact receipt, or report section. Fix uncovered gaps through TDD and rerun verification.

- [ ] **Step 4: Request final code review and address findings**

Use the requesting-code-review skill, inspect every actionable finding, apply receiving-code-review rigor, and fix accepted issues test-first.

- [ ] **Step 5: Finish the development branch**

Use the finishing-a-development-branch skill. Do not push, publish, or open a PR without explicit user authorization.
