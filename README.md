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
| `reports/` | 2025 track literature review and interactive background report |
| `pyproject.toml` | Default Python and notebook dependencies for `uv` |
| `uv.lock` | Reproducible dependency resolution |

## Important Version Note

The TREC 2026 candidate LLM set differs from the 2025 set. The reports in `reports/`
are useful for methods and feature ideas, but model-ID expertise should not be
transferred from 2025 to 2026.
