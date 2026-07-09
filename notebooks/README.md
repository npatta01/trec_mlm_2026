# Notebooks

Run the TREC 2026 manifest notebook with `uv`:

```bash
uv run --with pandas --with jupyter --with nbconvert \
  python -m jupyter nbconvert --execute --to notebook --inplace \
  notebooks/trec_2026_file_manifest.ipynb
```

The notebook reads `data/dev/file_manifest.tsv` and saves rendered outputs back into
`notebooks/trec_2026_file_manifest.ipynb`.
