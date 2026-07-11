# TREC Million LLMs 2026

This repository collects the official TREC 2026 Million LLMs file inventory,
reproducible notebook tooling, a leakage-safe query-neighbor baseline, and background
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
