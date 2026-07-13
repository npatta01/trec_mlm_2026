# Corrected Label Embeddings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce reference-compatible Qwen/E5 bare and contextual label runs and a replacement gate decision.

**Architecture:** Keep formatting, token construction, inference, scoring, and evaluation as separate qrel-free stages. Add one representation key to experiment configuration and pass explicit token IDs into decoder embedding runtimes.

**Tech Stack:** Python 3.13, NumPy, Transformers, vLLM 0.24 ROCm, pytest, TREC run artifacts.

## Global Constraints

- Exact checkpoint revisions and pinned ROCm image remain unchanged.
- Qrels are loaded only after scores and rankings are fixed.
- Context uses one fixed sentence and no generated descriptions or expert examples.
- Every executed condition is repeated and validated before advancement.

---

### Task 1: Freeze label representations

**Files:** Modify `trec_mlm/label_retrieval.py`, `trec_mlm/label_experiment.py`; test `tests/test_label_retrieval.py`, `tests/test_label_experiment.py`.

**Interfaces:** Produce `format_label_document(spec, model_id, representation)` and persist `label_representation` in configuration.

- [ ] Write tests for `bare` and `expert_capability` documents and invalid keys.
- [ ] Run focused tests and observe the missing-interface failures.
- [ ] Implement the formatter and configuration field.
- [ ] Run focused tests to green.

### Task 2: Freeze reference token IDs

**Files:** Modify `trec_mlm/label_inference.py`; test `tests/test_label_inference.py`.

**Interfaces:** Produce `build_qwen_token_ids(tokenizer, texts, max_length)` and `build_e5_token_ids(tokenizer, texts, max_length)` returning padded tensors plus truncation metadata.

- [ ] Write tests asserting Qwen `<|endoftext|>` and E5 `</s>` terminal IDs.
- [ ] Add a max-length test proving the terminal token is retained.
- [ ] Run tests and observe missing-interface failures.
- [ ] Implement minimal explicit token construction and validation.
- [ ] Run focused tests to green.

### Task 3: Use explicit token inputs

**Files:** Modify `trec_mlm/label_inference.py`; test `tests/test_label_inference.py`.

**Interfaces:** Qwen passes vLLM `TokensPrompt` objects; E5 passes explicit `input_ids` and `attention_mask` tensors.

- [ ] Write backend-boundary tests for token prompts and terminal-token metadata.
- [ ] Run tests and observe raw-string-path failures.
- [ ] Replace raw decoder inputs with explicit token inputs.
- [ ] Run focused and full tests to green.

### Task 4: Execute and summarize

**Files:** Modify `trec_mlm/cli.py`, `trec_mlm/label_experiment.py`, protocol and README; generate `artifacts/label_sweep_corrected/`.

**Interfaces:** CLI accepts `--label-representation`; combined summary keys results by encoder and representation.

- [ ] Validate the official Qwen canonical example and E5 reference token fixture.
- [ ] Execute Qwen/E5 × bare/contextual twice in the pinned container.
- [ ] Validate all 48,300-row runs, repeat comparisons, hashes, and metadata.
- [ ] Regenerate paired intervals, slices, issue examples, and the Stage 2 receipt.
- [ ] Run pytest, Ruff, formatting, and artifact integrity checks.
