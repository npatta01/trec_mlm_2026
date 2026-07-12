# Small Diverse Label-Embedding Sweep Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add pinned BGE-M3 and Nomic embedding backends, validate all five small/diverse candidates against official preprocessing contracts, execute both label representations twice, and produce a combined exploratory receipt.

**Architecture:** Extend the existing backend-neutral registry and formatter layer, then add a Transformers encoder path with explicit pooling strategies beside the existing vLLM path. Keep experiment execution and artifact writing unchanged so every model produces the same validated score/ranking schema. Build a separate diversity receipt that can compare repeated conditions, MPNet, corrected E5, and bare/contextual variants without changing the original Stage 1 gate logic.

**Tech Stack:** Python 3.12, NumPy, PyTorch/ROCm, Transformers, vLLM 0.24 ROCm container, pytest/unittest, TREC run format.

## Global Constraints

- Use `vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28`.
- Pin BGE-M3 to `5617a9f61b028005a4858fdac845db406aefb181` and Nomic v1.5 to `e9b6763023c676ca8431644204f50c2b100d9aab`.
- Preserve the exact `bare` and `expert_capability` representations from the approved spec.
- Generate rankings and scores before loading qrels; exact ties use candidate ID.
- Run every passing condition twice in fresh processes and require identical rankings/metrics plus scores within `1e-5`.
- Treat results as post-gate exploratory evidence; do not select deployment, start fusion, or start the LLM ranker.

---

### Task 1: Register and format BGE-M3 and Nomic

**Files:**
- Modify: `trec_mlm/label_retrieval.py`
- Modify: `tests/test_label_retrieval.py`

**Interfaces:**
- Consumes: `LabelEncoderSpec`, `format_query()`, `format_document()`.
- Produces: registry keys `bge_m3` and `nomic_embed_text_v1_5`; formatter styles `plain_query` and `nomic_search`.

- [ ] **Step 1: Write failing registry and formatter tests**

Add assertions that the registry contains the two exact revisions, dimensions, licenses,
pooling labels, and that formatting returns raw BGE questions/documents plus
`search_query:`/`search_document:` Nomic inputs for both representations.

- [ ] **Step 2: Verify the tests fail**

Run:
`python -m pytest -q tests/test_label_retrieval.py::ModelRegistryTest`

Expected: failure because `bge_m3` and `nomic_embed_text_v1_5` are absent.

- [ ] **Step 3: Add the frozen specs and formatter branches**

Add `LabelEncoderSpec` entries with stage `2`, runtimes `transformers_rocm`, BGE
`dense_cls_normalized`, Nomic `mean_layernorm_normalized`, and the approved metadata.
Extend query/document formatting only for the two new style names.

- [ ] **Step 4: Verify the registry tests pass**

Run:
`python -m pytest -q tests/test_label_retrieval.py::ModelRegistryTest`

Expected: all tests pass.

### Task 2: Implement explicit encoder pooling

**Files:**
- Modify: `trec_mlm/label_inference.py`
- Modify: `tests/test_label_inference.py`

**Interfaces:**
- Consumes: hidden state tensor `[batch, tokens, dimension]`, attention mask `[batch, tokens]`, and `LabelEncoderSpec.pooling`.
- Produces: `cls_token_pool(last_hidden_state)`, `mean_token_pool(last_hidden_state, attention_mask)`, and normalized float32 matrices through `encode_transformers_encoder()`.

- [ ] **Step 1: Write failing pooling tests**

Create fixtures proving CLS returns token zero; masked mean excludes padding; Nomic applies
`layer_norm(..., normalized_shape=(dimension,))` before L2 normalization; invalid masks and
dimensions raise clear errors.

- [ ] **Step 2: Verify pooling tests fail**

Run:
`python -m pytest -q tests/test_label_inference.py -k 'cls or mean or nomic'`

Expected: import or assertion failures for missing pooling helpers.

- [ ] **Step 3: Implement pooling and encoder dispatch**

Load `AutoTokenizer` and `AutoModel` at the registered revision with
`trust_remote_code=False`, batch at a conservative fixed size, use BF16 on ROCm, apply
the registered pooler, convert to float32, normalize, and record versions, device,
lengths, elapsed time, and peak allocated memory. Route `transformers_rocm` specs through
this function while preserving the E5-specific last-token path.

- [ ] **Step 4: Verify inference tests pass**

Run:
`python -m pytest -q tests/test_label_inference.py`

Expected: all inference unit tests pass.

### Task 3: Freeze smoke parity and artifact metadata

**Files:**
- Modify: `trec_mlm/label_inference.py`
- Modify: `trec_mlm/label_experiment.py`
- Modify: `trec_mlm/cli.py`
- Modify: `tests/test_label_inference.py`
- Modify: `tests/test_label_experiment.py`

**Interfaces:**
- Consumes: `embed_texts(spec, texts, seed)` and two-query/two-document fixtures.
- Produces: `validate_embedding_smoke(...) -> dict[str, Any]` and metadata fields `smoke_reference`, `smoke_max_score_delta`, `smoke_passed`, `peak_memory_bytes`, and `parameter_count`.

- [ ] **Step 1: Write failing validation and CLI tests**

Test that a canonical score matrix within `1e-4` passes, a larger delta fails, non-finite
or wrong-dimensional embeddings fail, and new encoder keys parse through
`label-retrieval --encoder`.

- [ ] **Step 2: Verify the tests fail**

Run:
`python -m pytest -q tests/test_label_inference.py tests/test_label_experiment.py -k smoke`

Expected: failure because smoke validation metadata is absent.

- [ ] **Step 3: Implement smoke validation and metadata propagation**

Add a pure validator for shapes, finiteness, row norms, and reference score deltas. Store
the smoke receipt in every full artifact's encoder metadata. Refuse a full run when the
smoke receipt is absent or failed for BGE/Nomic; existing Qwen/Jina paths use their
already-frozen canonical checks.

- [ ] **Step 4: Run all label tests**

Run:
`python -m pytest -q tests/test_label_retrieval.py tests/test_label_inference.py tests/test_label_experiment.py`

Expected: all label tests pass.

### Task 4: Run canonical ROCm smoke tests

**Files:**
- Create: `artifacts/label_sweep_diverse/smoke/<encoder>/smoke.json`

**Interfaces:**
- Consumes: all five registered model specs and their official reference formatting/pooling.
- Produces: one immutable smoke receipt per model and a pass/block decision.

- [ ] **Step 1: Confirm GPU and cache capacity**

Run read-only ROCm device, container image, free disk, and Hugging Face cache checks.
Expected: `/dev/kfd` and `/dev/dri` accessible, pinned image present, and enough disk for
the selected checkpoints.

- [ ] **Step 2: Run two-query/two-document tests**

Use `China's capital is Beijing`, `The capital of China is Beijing`, `gravity`, and
`A force pulls objects toward Earth` through each official formatter/pooler. Compare
production-path scores to a direct official-reference implementation in the same
container and write `smoke.json` only after validation.

- [ ] **Step 3: Audit smoke receipts**

Verify exact revisions, dimensions, finite unit norms, score delta at most `1e-4`, and
formatter/token records. Block only the failing model and retain its diagnostic log.

### Task 5: Execute the 20-run sweep

**Files:**
- Create: `artifacts/label_sweep_diverse/<encoder>/<representation>/run1/`
- Create: `artifacts/label_sweep_diverse/<encoder>/<representation>/run2/`

**Interfaces:**
- Consumes: passed smoke receipts, `trec-mlm label-retrieval`, and the frozen query/candidate inputs.
- Produces: validated per-run embeddings, 48,300 scores, complete rankings, metrics, timing, memory, hashes, and repeat receipts.

- [ ] **Step 1: Execute Qwen 0.6B and 4B**

Run bare and expert-capability representations twice using explicit Qwen terminal token
IDs and fresh containers. Validate each artifact before continuing.

- [ ] **Step 2: Execute Jina small**

Run both representations twice using its official query/document prefixes and last-token
pooling. Record its CC-BY-NC-4.0 restriction in every artifact.

- [ ] **Step 3: Execute BGE-M3 and Nomic**

Run both representations twice through the validated Transformers encoder backend.
Preserve raw BGE inputs and Nomic retrieval prefixes exactly.

- [ ] **Step 4: Compare repeats**

For every condition, call `compare_repeated_results()` and record ranking equality,
metric equality, maximum score difference, and the `1e-5` decision.

### Task 6: Build the diversity comparison receipt

**Files:**
- Modify: `trec_mlm/label_experiment.py`
- Modify: `trec_mlm/cli.py`
- Modify: `tests/test_label_experiment.py`
- Create: `artifacts/label_sweep_diverse/combined/diversity_summary.json`
- Create: `artifacts/label_sweep_diverse/combined/decision.md`
- Create: `docs/results/2026-07-12-small-diverse-label-embedding-sweep.md`

**Interfaces:**
- Consumes: MPNet control, corrected contextual E5, and all validated diverse run pairs.
- Produces: `build_diversity_summary(...)` and `trec-mlm label-diversity-summary`.

- [ ] **Step 1: Write failing summary tests**

Use fixture results to require paired intervals against MPNet and E5, within-model
context comparisons, repeat eligibility, zero transitions, issue examples, and resource
fields. Verify exploratory status prevents an automatic deployment decision.

- [ ] **Step 2: Verify summary tests fail**

Run:
`python -m pytest -q tests/test_label_experiment.py -k diversity`

Expected: failure because the builder and CLI command do not exist.

- [ ] **Step 3: Implement and run the combined receipt**

Add the builder/writer/CLI following existing sweep-summary patterns, then generate the
machine-readable and plain-language receipts from run1 plus independent repeat paths.

- [ ] **Step 4: Verify final artifacts and code**

Run label tests in the pinned container, Ruff and format checks on changed Python files,
and an artifact audit asserting 322 unique rankings, 150 candidates per query, 48,300
finite scores, correct model revisions/licenses, complete comparisons, and valid repeat
decisions. Document unrelated repository-wide optional-dependency failures separately.
