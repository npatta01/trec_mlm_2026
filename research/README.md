# TREC 2026 Research Hub

The canonical research narrative lives in `research/hub/` as Quarto Markdown.
The experiment ledger remains authoritative for run identities, metrics, scopes,
and artifact states. Build generated views with:

```bash
uv run python scripts/build_research_hub.py
quarto render research/hub
```

The private local site entry is `research/hub/_site/index.html`. Generated QMD,
the catalog, and HTML are derived views; edit the source records instead.
