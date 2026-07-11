# TREC Million LLMs 2026

This repository collects the official TREC 2026 Million LLMs file inventory,
reproducible notebook tooling, leakage-safe sparse and dense query-neighbor baselines, and background
research from the 2025 track.

Official track page: https://trec-mllm.github.io/

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
