# Notebooks

Run the TREC 2026 manifest notebook with `uv`:

```bash
uv run jupyter nbconvert --execute --to notebook --inplace \
  notebooks/trec_2026_file_manifest.ipynb
```

`uv run` creates or updates the project environment from `pyproject.toml` and
`uv.lock` automatically.

The notebook reads `data/dev/file_manifest.tsv` and saves rendered outputs back into
`notebooks/trec_2026_file_manifest.ipynb`.

Run the query-neighbor baseline notebook the same way:

```bash
uv run jupyter nbconvert --execute --to notebook --inplace \
  notebooks/trec_2026_query_knn_baseline.ipynb
```

It refreshes the leakage-safe baseline artifacts and presents narrow tables plus
responsive metric cards for phone-sized screens.

Regenerate the read-only HTML report without exposing notebook code cells:

```bash
uv run jupyter nbconvert --to html \
  --TemplateExporter.exclude_input=True \
  --output trec_2026_query_knn_baseline.html \
  --output-dir reports \
  notebooks/trec_2026_query_knn_baseline.ipynb
```
