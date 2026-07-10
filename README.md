# TREC Million LLMs 2026

This repository collects the official TREC 2026 Million LLMs file inventory,
reproducible notebook tooling, and background research from the 2025 track. The
current codebase covers project setup and data discovery; it does not yet include a
ranking baseline or submission pipeline.

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

## Data

[`data/dev/file_manifest.tsv`](data/dev/file_manifest.tsv) records the official file
names, release status, Google Drive IDs, source URLs, expected local paths, and intended
uses. See [`data/dev/README.md`](data/dev/README.md) for the task definition, submission
format, metrics, and release timeline.

Raw TREC files are intentionally ignored because the discovery data may be large.
Download released files from the manifest into their `expected_local_path` locations
under `data/dev/`. The notebook reports which expected files are present locally.

## Repository Layout

| Path | Purpose |
|---|---|
| `data/dev/` | TREC 2026 source notes, file manifest, and ignored local data files |
| `notebooks/` | Executable exploration and validation notebooks |
| `reports/` | 2025 track literature review and interactive background report |
| `pyproject.toml` | Default Python and notebook dependencies for `uv` |
| `uv.lock` | Reproducible dependency resolution |

## Important Version Note

The TREC 2026 candidate LLM set differs from the 2025 set. The reports in `reports/`
are useful for methods and feature ideas, but model-ID expertise should not be
transferred from 2025 to 2026.
