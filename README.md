# TREC Million LLMs 2026

This repository collects the official TREC 2026 Million LLMs file inventory,
reproducible notebook tooling, leakage-safe sparse and dense query-neighbor baselines, and background
research from the 2025 track.

Official track page: https://trec-mllm.github.io/

## Beginner's Approach Guide

For a phone-friendly introduction to the task, fifteen accessible teaching visuals, a
plain-language primer, a source-audited review of every 2025 submission family, the
wider expert-search and LLM-routing literature, and a 2026 experiment ladder grounded
in the files currently available, open
[`reports/trec_2026_approaches_guide.html`](reports/trec_2026_approaches_guide.html).
The report is self-contained, works without a server, follows system light/dark
appearance, and retains a semantic no-script reading path and audio-friendly prose for
phone screen readers.

For the complete 2025 implementation playbooks and beginner diagrams explaining
profiles and embeddings, open the companion
[`reports/trec_2025_mllm_interactive_report.html`](reports/trec_2025_mllm_interactive_report.html).
Jump directly to [Features](reports/trec_2025_mllm_interactive_report.html#features),
[Approach Maps](reports/trec_2025_mllm_interactive_report.html#approach-maps), or
[What Each Team Submitted](reports/trec_2025_mllm_interactive_report.html#participants)
for the deeper recipes that the beginner guide now explains and cross-references.

## Quick Start

Install [`uv`](https://docs.astral.sh/uv/), then create the locked project environment:

```bash
uv sync
```

Open the notebooks interactively:

```bash
uv run jupyter lab
```

Or execute the manifest notebook from the repository root:

```bash
uv run jupyter nbconvert --execute --to notebook --inplace \
  notebooks/trec_2026_file_manifest.ipynb
```

Run the first ranking baseline and write a complete TREC run plus diagnostics:

```bash
uv run trec-mlm query-knn \
  --primary-selection-provenance predeclared_before_current_run \
  --output-dir artifacts/query_knn_baseline
```

The default experiment uses strict leave-one-query-out evaluation, word TF-IDF query
neighbors, a fixed primary setting of `k=10`, and a small exploratory sensitivity grid.
It ranks all 150 candidate model IDs for every query. Validate and independently score
the generated run with:

```bash
uv run trec-mlm validate-run artifacts/query_knn_baseline/run.trec \
  --queries data/dev/llm_dev_data.tsv \
  --qrels data/dev/llm_dev_qrels.txt
uv run trec-mlm score-run \
  --qrels data/dev/llm_dev_qrels.txt \
  --run artifacts/query_knn_baseline/run.trec
```

The executed, mobile-friendly analysis is in
[`notebooks/trec_2026_query_knn_baseline.ipynb`](notebooks/trec_2026_query_knn_baseline.ipynb).
For a phone-friendly, read-only view, open
[`reports/trec_2026_query_knn_baseline.html`](reports/trec_2026_query_knn_baseline.html).

## Inspect a Completed Run

The repository-local `$inspect-query-results` skill builds a simple, self-contained
query explorer after an experiment finishes. Ask Codex to use the skill with an artifact
directory, for example:

```text
Use $inspect-query-results for artifacts/dense_query_knn_baseline.
```

The equivalent direct command is:

```bash
python3 .agents/skills/inspect-query-results/scripts/build_query_explorer.py \
  --run-dir artifacts/dense_query_knn_baseline
```

The builder prefers `predictions.tsv`, joins the query text and saved candidate behavior
profiles, optionally samples related historical responses, and writes
`query_explorer.html` inside the selected artifact directory. Run it without
`--run-dir` to select the newest unambiguous prediction artifact automatically. Add
`--no-responses` for a faster profile-only explorer. The page shows the top 10
candidates per query by default; change that with `--top-candidates`. Completed
development runs automatically add ground-truth grades, per-query nDCG@10 versus its
ideal value, all positively labeled LLMs and their system ranks, first relevant rank, and
weakest/failure query filters when their config points to available qrels. Runs without
labels retain the same query, score, and profile workflow.

## Candidate-aware Semantic Profiles

Issue 12 is governed by the bounded, advisor-approved protocol in
[`protocols/trec_2026_candidate_semantic_profiles.md`](protocols/trec_2026_candidate_semantic_profiles.md).
Construction is config-driven and qrel-free. The approved two-stage boundary encodes
with the pinned E5-Mistral ROCm image, then consumes that immutable inference artifact
under the pinned Python 3.13 clustering environment to build deterministic multi-vector
profiles and freeze all six permitted rankings.

The full inference command refuses to start without the passing, qrel-free `gfx1151`
smoke-repeat receipt. Run it in rootless Podman without mounting qrels:

```bash
podman run --rm --device /dev/kfd --device /dev/dri \
  --security-opt label=disable --network none \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e PYTHONPATH=/code \
  -v "$PWD/trec_mlm:/code/trec_mlm:ro" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "$PWD/data/dev/llm_dev_data.tsv:/inputs/queries.tsv:ro" \
  -v "$PWD/reports/trec_2026_discovery_profile_audit.json:/inputs/candidates.json:ro" \
  -v "/path/to/llm_discovery_data.json:/inputs/llm_discovery_data.json:ro" \
  -v "/path/to/contextual_e5:/inputs/contextual_e5:ro" \
  -v "$PWD/artifacts/candidate_semantic_profiles:/outputs:rw" \
  docker.io/vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28 \
  python -m trec_mlm.cli profile-infer \
  --queries /inputs/queries.tsv \
  --discovery /inputs/llm_discovery_data.json \
  --candidates /inputs/candidates.json \
  --e5-artifact /inputs/contextual_e5 \
  --smoke-repeat-receipt /outputs/smoke_repeat_receipt.json \
  --output-dir /outputs/inference1

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
uv run --extra dense trec-mlm profile-construct \
  --queries data/dev/llm_dev_data.tsv \
  --discovery /path/to/llm_discovery_data.json \
  --candidates reports/trec_2026_discovery_profile_audit.json \
  --dense-artifact /path/to/dense_query_knn_baseline \
  --inference-artifact artifacts/candidate_semantic_profiles/inference1 \
  --output-dir artifacts/candidate_semantic_profiles/run1

uv run --extra dense trec-mlm profile-validate \
  --construction artifacts/candidate_semantic_profiles/run1
```

Rerun both stages independently as `inference2` and `run2`; run 2 may not reuse run 1
embeddings. Require the Stage 1 and construction repeat receipts before evaluation:

```bash
uv run --extra dense trec-mlm profile-infer-repeat-check \
  --first artifacts/candidate_semantic_profiles/inference1 \
  --second artifacts/candidate_semantic_profiles/inference2 \
  --receipt artifacts/candidate_semantic_profiles/inference_repeat_receipt.json

uv run --extra dense trec-mlm profile-repeat-check \
  --first artifacts/candidate_semantic_profiles/run1 \
  --second artifacts/candidate_semantic_profiles/run2 \
  --inference-repeat-receipt artifacts/candidate_semantic_profiles/inference_repeat_receipt.json \
  --receipt artifacts/candidate_semantic_profiles/repeat_receipt.json

# Qrels enter for the first time here, after ranking hashes are locked.
uv run --extra dense trec-mlm profile-evaluate \
  --construction artifacts/candidate_semantic_profiles/run1 \
  --repeat-receipt artifacts/candidate_semantic_profiles/repeat_receipt.json \
  --qrels data/dev/llm_dev_qrels.txt \
  --output-dir artifacts/candidate_semantic_profiles/evaluation

uv run trec-mlm profile-report \
  --evaluation artifacts/candidate_semantic_profiles/evaluation/evaluation.json \
  --markdown reports/trec_2026_candidate_semantic_profiles.md \
  --html reports/trec_2026_candidate_semantic_profiles.html
```

The sole promotion test is the frozen 1:1 RRF fusion of query-kNN and candidate
profiles against the frozen query-kNN baseline. The contextual E5 and alternative
fusion rankings are controls or preregistered ablations. The dense input was trained
with fold-local qrels before this study and is consumed as an already-frozen ranking;
profile construction, profile scoring, and the fixed RRF arithmetic do not load qrels.

## Baseline 2: Frozen Dense Query Retrieval

Install the optional CPU-only embedding stack and run the preregistered nested grid:

```bash
uv sync --extra dense
uv run --extra dense trec-mlm dense-query-knn \
  --output-dir artifacts/dense_query_knn_baseline
```

The default run evaluates four immutable encoder revisions—MiniLM-L6, E5-small-v2,
BGE-small-v1.5, and MPNet-base-v2—across `k={1,3,5,10,20,50,100}`. For every outer
leave-one-query-out prediction, a complete inner leave-one-query-out loop selects the
encoder and `k` using nDCG@10. The held-out outer qrel never enters configuration
selection, neighbor evidence, priors, or tie breaks. Frozen embeddings are qrel-free and
computed once; qrel aggregation and global priors remain fold-local.

On the 322-query development set, the stitched nested prediction achieved:

| System | nDCG@10 | MRR |
|---|---:|---:|
| Frozen dense, nested encoder + `k` | **0.3599** | **0.4041** |
| TF-IDF query-kNN, fixed `k=10` | 0.2177 | 0.2777 |
| Training-only global prior | 0.1385 | 0.2067 |

MPNet with `k=50` was selected in 320 of 322 outer folds; `k=20` was selected in the
other two. On a Ryzen AI MAX+ 395 CPU with 16 PyTorch threads, the cached model
inference, nested selection, predictions, and fixed-grid scoring take about 23 seconds;
the artifact records bootstrap-summary time separately. None of the 322 queries were
truncated. Because pretrained
encoders contain outside training data, this is an **external automatic run** under the
track taxonomy.

The self-contained, mobile-responsive technical report is
[`reports/trec_2026_dense_query_knn_baseline.html`](reports/trec_2026_dense_query_knn_baseline.html).
Its reviewed result snapshot and canonical report source are kept beside it for audit.

## Baseline 3: Discovery-Response Retrieval

Download the smaller official discovery-response file using the command in
[`data/dev/README.md`](data/dev/README.md#download-discovery-files), then install the
lightweight response-retrieval extra and run:

```bash
uv sync --extra response
uv run --extra response trec-mlm response-retrieval \
  --output-dir artifacts/response_retrieval_baseline
```

This CPU-only baseline treats each historical query/model response as a searchable
document. It evaluates BM25 and explicit RM3 feedback over response-only and
query-plus-response fields, two frozen filters, and retrieval depths 100, 500, and
1,000. SumScore aggregates retrieved evidence back to the 150 model IDs. The complete
24-configuration protocol—including analyzer hashes, refusal rules, exact-query
exclusion, nested selection, and tie handling—was locked before scoring in
[`protocols/trec_2026_response_retrieval.md`](protocols/trec_2026_response_retrieval.md).

Every outer query uses only the other 321 queries to select a configuration. Inner tie
priors exclude both the outer and inner qrels, and exact discovery-query matches are
masked before RM3 feedback and final retrieval. The experiment compares the stitched
response ranking directly with the tracked dense per-query reference, the rerun fixed
TF-IDF baseline, and the identical fold-local global prior. It does not use logprobs,
pretrained encoders, or external data, so it is an **internal automatic run**.

On the 322-query development set, the leakage-safe stitched prediction achieved:

| System | nDCG@10 | MRR |
|---|---:|---:|
| Frozen dense query-kNN | **0.3599** | **0.4041** |
| Nested response retrieval | 0.2774 | 0.3175 |
| TF-IDF query-kNN, fixed `k=10` | 0.2177 | 0.2777 |
| Training-only global prior | 0.1385 | 0.2067 |

Against TF-IDF, response retrieval improved nDCG@10 by 0.0597 (paired bootstrap 95%
CI 0.0193 to 0.1003). Its MRR point estimate improved by 0.0398, but that interval
crossed zero (-0.0030 to 0.0819). Dense query retrieval therefore remains the strongest
evaluated baseline. Nested selection chose response-only BM25+RM3 with the basic filter
and depth 500 in 314 of 322 folds. The deployment configuration showed strong response-
inventory dependence, so the report treats possible volume bias as a material caveat.

The full CPU run took about one minute and peaked at 1.49 GB RSS on a Ryzen AI MAX+
395. The self-contained, mobile-responsive plain-language report is
[`reports/trec_2026_response_retrieval_baseline.html`](reports/trec_2026_response_retrieval_baseline.html).

### Follow-up: response-volume normalization

The advisor-approved follow-up tests whether caps or averages can reduce SumScore's
response-inventory dependence without losing its ranking signal. It retrieves once and
derives six frozen alternatives from the same positive hits: strongest match, top-3 and
top-10 sums, top-3 and top-10 means, and the all-hit mean. The primary nested selector
contains 144 normalized candidates; the original 24 SumScore candidates are evaluated
by a separate nested control and cannot enter the primary selector. The exact protocol
was committed before effectiveness scoring in
[`protocols/trec_2026_response_volume_normalization.md`](protocols/trec_2026_response_volume_normalization.md).

Run the CPU-only experiment with:

```bash
uv run --extra response trec-mlm response-volume-normalization \
  --output-dir artifacts/response_volume_normalization
```

Nested validation selected Top10Sum in all 322 outer folds. Its held-out point estimates
were higher than SumScore's, but both preregistered deployment-volume diagnostics
worsened:

| System | nDCG@10 | MRR | Inventory rho | Largest-inventory top-10 share |
|---|---:|---:|---:|---:|
| Frozen dense query-kNN | **0.3599** | **0.4041** | — | — |
| Top10Sum response retrieval | 0.2892 | 0.3283 | 0.8894 | 48.3% |
| SumScore response control | 0.2774 | 0.3175 | 0.8388 | 46.6% |
| TF-IDF query-kNN, fixed `k=10` | 0.2177 | 0.2777 | — | — |
| Training-only global prior | 0.1385 | 0.2067 | — | — |

Top10Sum minus SumScore was +0.0119 nDCG@10, with paired 95% CI -0.0031 to
0.0265, so the point improvement is not decisive. The locked rule required all four
effectiveness and volume gates; only the two effectiveness gates passed. The resulting
decision is to retain SumScore as the response-search control and stop this aggregation
branch. This does not imply that SumScore is unbiased; it means the proposed cap did not
solve the diagnosed volume problem. Dense query retrieval remains the strongest overall
baseline.

The full experiment was repeated with identical tensors, selections, runs, diagnostics,
and decisions. It uses no GPU, log probabilities, or score fusion. The self-contained
plain-language report is
[`reports/trec_2026_response_volume_normalization.html`](reports/trec_2026_response_volume_normalization.html).

### Follow-up: source-query-first answer alignment

This follow-up asks a different question: after finding older queries related to the
new target, can each model be scored by how well its stored answer addresses its own
older source query? Every selected source query retains all 150 model slots. Missing,
blank, exact `I don't know`, and frozen-prefix refusal responses remain in the local
BM25 corpus and score zero, so models with more stored usable answers do not receive
extra source-query opportunities.

The leakage-safe, CPU-only protocol was advisor-approved and committed before
effectiveness scoring in
[`protocols/trec_2026_source_query_first.md`](protocols/trec_2026_source_query_first.md).
A qrel-free feasibility stop found that one target had only 85 positive source-query
matches, so the original `K={10,50,100}` grid could not be evaluated. No effectiveness
artifact was written. The advisor-approved amendment changed only the grid to
`K={10,20,40}` before scoring; target-to-response relevance remained a diagnostic and
could not select `K`, gate the result, or enter the primary score.

Run the official experiment twice and build the independently checked report source
with:

```bash
uv run --extra response trec-mlm source-query-first \
  --output-dir artifacts/source_query_first
uv run --extra response trec-mlm source-query-first \
  --output-dir artifacts/source_query_first_rerun
uv run python scripts/build_source_query_first_report.py
```

Nested validation selected `K=40` in all 322 held-out folds. The primary result was
weaker than SumScore, Top10Sum, TF-IDF, and dense query retrieval, but stronger than the
fold-local global prior:

| System | nDCG@10 | MRR |
|---|---:|---:|
| Frozen dense query-kNN | **0.3599** | **0.4041** |
| Top10Sum response retrieval | 0.2892 | 0.3283 |
| SumScore response control | 0.2774 | 0.3175 |
| TF-IDF query-kNN, fixed `k=10` | 0.2177 | 0.2777 |
| Source-query-first alignment | 0.2033 | 0.2482 |
| Training-only global prior | 0.1385 | 0.2067 |

Source-query-first minus SumScore was -0.0741 nDCG@10, with paired 95% CI -0.1040
to -0.0449. It passed none of the four locked gates: it was not clearly above TF-IDF,
was not within 0.010 nDCG@10 of SumScore, had absolute response-inventory correlation
0.8019 against a 0.7388 maximum, and gave the largest inventory quartile 50.4% of
top-10 placements against a 41.1% maximum. The locked recommendation is therefore to
retain SumScore and stop the source-query-first branch. The owner accepts that
method-selection result: conclude v1 and do not select it as the default or a submission
method.

Separately, the owner elected to merge the completed code and evidence as a frozen,
opt-in research archive. This is not a permanent rejection of the method family. A
future retest is explicitly allowed after a decision-relevant bug, new evidence, a
materially changed benchmark, or a newly preregistered qrel-free method. The scoped
decision, post-hoc case diagnostic, and required retest procedure are recorded in
[`reports/trec_2026_source_query_first_decision.md`](reports/trec_2026_source_query_first_decision.md).

Both official runs reproduced the decision receipt and the merged PR #8 SumScore and
Top10Sum run files byte-for-byte. The first run took about 142 CPU seconds and peaked at
1.63 GB RSS. The self-contained report explains the task, leakage controls, metrics,
uncertainty, and volume diagnostics in phone-friendly language:
[`reports/trec_2026_source_query_first.html`](reports/trec_2026_source_query_first.html).

The prespecified paired comparison recorded 57 nDCG@10 wins, 142 ties, and 123 losses
against SumScore. A post-hoc case diagnostic observed presently unrouteable
complementarity on this development set: source-query-first beat SumScore, Top10Sum,
dense, and TF-IDF together on 24 queries. Similar-looking questions also failed, and no
examined qrel-free signal reliably identified the winners before scoring. These
observations may motivate a separately preregistered experiment on genuinely unseen
judgments; they do not alter the locked 0-of-4-gates decision.

## Data

[`data/dev/file_manifest.tsv`](data/dev/file_manifest.tsv) records the official file
names, release status, Google Drive IDs, source URLs, expected local paths, and intended
uses. See [`data/dev/README.md`](data/dev/README.md) for the task definition, submission
format, metrics, and release timeline.

The released development query file and qrels are tracked in the repository. The two
large discovery JSON files remain ignored. The official source currently distributes
them through Google Drive; follow the resumable commands in
[`data/dev/README.md`](data/dev/README.md#download-discovery-files). The notebook reports
which expected files are present locally and whether their byte sizes match the manifest.

## Repository Layout

| Path | Purpose |
|---|---|
| `data/dev/` | Source notes, manifest, tracked development files, and ignored large files |
| `.agents/skills/` | Repository-local Codex skills, including the post-run query explorer |
| `notebooks/` | Executable exploration and validation notebooks |
| `reports/` | Literature reviews and reader-friendly baseline reports |
| `trec_mlm/` | Baseline ranking, evaluation metrics, and TREC run tooling |
| `tests/` | Unit and equivalence tests for the baseline pipeline |
| `pyproject.toml` | Default Python and notebook dependencies for `uv` |
| `uv.lock` | Reproducible dependency resolution |

## Important Version Note

The TREC 2026 candidate LLM set differs from the 2025 set. The reports in `reports/`
are useful for methods and feature ideas, but model-ID expertise should not be
transferred from 2025 to 2026.
