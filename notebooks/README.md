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
