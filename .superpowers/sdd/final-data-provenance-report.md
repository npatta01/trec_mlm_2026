# Data index automation and Background provenance

## Outcome

The Research Hub Data landing page is now a derived view of canonical dataset
records. Its editable `data/index.qmd` shell contains only page metadata and a
generated include. The generated page uses open sections, headings, summaries,
and the existing category/availability badges; it does not use the former card
or grid layout.

Background provenance is split by controlling source:

- `methods-landscape.qmd` contains only method-ladder and evaluation guidance
  controlled by `reports/trec_2026_approaches_guide.artifact.json`.
- `reference-model-planning.qmd` contains the dated Gemini, Qwen, Gemma, cost,
  and local-runtime planning controlled by
  `reports/trec_2026_reference_model_landscape_2026-07-21.md`.

The generated Background index places the methods landscape under wider methods
and the dated model review under current model/reference planning. Both records
retain their own deep-dive link and appear separately in the generated catalogs.

## Generated-output contract

`research/hub/generated/data-index.qmd` is part of the deterministic render,
write, stale-output, and freshness-check contract. Adding or changing a valid
dataset record changes that output automatically. The contributor guide names
the canonical dataset workflow and warns against manual index/catalog edits.

## TDD evidence

The first focused run failed in five expected places:

- the generated Data output path was absent;
- rendering the Data index raised `KeyError`;
- the dated reference-model Background record did not exist;
- the Background planning group still pointed at `methods-landscape`;
- `data/index.qmd` still contained duplicated dataset facts.

After the minimal implementation, the focused behavior checks passed. A second
red-green cycle added the contributor contract: its test first failed because
`Add a dataset` was absent, then passed after the workflow was documented.

The regression coverage now proves:

- the exact generated-output path contract includes the Data index;
- an added dataset and changed canonical title, description, category, and
  availability appear in the generated Data index;
- the Data wrapper is fact-free;
- every canonical dataset and Background record appears in generated navigation;
- the two Background records name distinct controlling sources and do not carry
  each other's claims or deep-dive link;
- the dated record alone occupies the current model/reference planning group.

## Verification

- Focused Data/provenance integration checks: 8 passed.
- Full Research Hub suite: 108 passed.
- Full Quarto site render: 162 pages rendered; pre-render generation and both
  post-render checks completed.
- Generated-output freshness check: passed.
- Scoped Ruff over the Research Hub package, tests, and build script: passed.
- Repository-wide Ruff was also run and reported one unrelated, pre-existing
  unused `IPython.display.display` import in
  `notebooks/trec_2026_file_manifest.ipynb`; that notebook is unchanged and
  outside this task.
- `git diff --check`: passed.

## Scope boundary

This change does not alter relationship policy, staging/privacy behavior,
stylesheets, portal delivery, or progress reporting. The rendered `_site`
remains an untracked local build artifact.
