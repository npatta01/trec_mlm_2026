# TREC 2026 Research Hub

The Research Hub is the canonical narrative for what the project has learned,
why decisions were made, and what should be tested next. Exact run identities,
metrics, scopes, promotion status, and artifact custody remain canonical in
[`../experiments/ledger.json`](../experiments/ledger.json).

## Where to start

For a newcomer:

1. Read the [current-state research map](hub/start-here/current-state.qmd) for
   the two-minute status, evidence boundary, and visual explanation.
2. Use the [task explanation](hub/start-here/problem.qmd) for the ranking setup,
   then the [deeper Hub summary](hub/index.qmd) for decision detail.
3. Follow the [study index](hub/studies/index.qmd) for completed evidence or the
   [ideas and tests index](hub/ideas/index.qmd) for prospective work.
4. Use the [evaluation guide](hub/start-here/evaluation.qmd) before comparing
   scores or promotion labels.

For an agent, start with [`catalog.json`](catalog.json). It is a generated,
machine-readable map of record IDs, relationships, source paths, and ledger
links. Follow each `canonical_path` to the authored QMD before changing narrative,
and read the experiment registry before stating exact run facts. Do not treat
the catalog as a second source of truth.

## Canonical boundaries

| Information | Canonical source | Editing rule |
|---|---|---|
| Run identity, metrics, scope, promotion, evidence, artifact custody | [`../experiments/ledger.json`](../experiments/ledger.json) | Update through the ledger workflow |
| Conclusions, decisions, questions, studies, and proposed tests | Authored QMD under [`hub/`](hub/) | Edit the matching record |
| Frozen construction, leakage, and evaluation contracts | [`../protocols/`](../protocols/) | Preserve amendments and provenance |
| Study evidence and receipts | Tracked paths named by the study and ledger | Do not move or rename hashed evidence |
| Catalogs, generated QMD, summaries, and HTML | Generator output | Never edit by hand |

Authored Hub records live in `hub/themes/`, `hub/questions/`, `hub/studies/`,
`hub/ideas/`, `hub/data/`, `hub/background/`, and `hub/approaches/`. The
following are derived views:

- `catalog.json` and `hub/catalog.json`;
- `hub/generated/`;
- `../reports/trec_2026_research_hub_summary.md`; and
- `hub/_site/`.

The rendered site is ignored, derived, and private by default. Rendering does
not authorize publishing or copying private artifacts into the site.

## Add or update a record

Use the matching example in [`hub/templates/`](hub/templates/) and the complete
[`hub/CONTRIBUTING.md`](hub/CONTRIBUTING.md) relationship rules.

- **Question:** state one bounded current answer, confidence boundary, linked
  studies, and tracked evidence.
- **Study:** record the objective, protocol, conclusion, decision, limitations,
  linked ledger systems, and canonical evidence. Set `rationale_provenance` to
  exactly `contemporaneous`, `reconstructed`, or `unknown`; never rewrite a
  reconstruction as preregistration.
- **Dataset:** record provenance, permitted use, leakage constraints,
  sensitivity, sharing, availability, and reproducibility without embedding
  sensitive source content.
- **Background resource:** name its controlling source, reviewed date, and the
  bounded contribution it makes to the research program.
- **Idea or test:** describe the mechanism and next action. A record may enter
  `ready`, `running`, or `completed` only when hypothesis, rationale, expected
  signal, decision rule, dependencies, priority, evaluation boundary, artifact
  plan, and next action are all present. Keep missing fields explicit while it
  remains an idea.

Use a stable lowercase hyphenated ID, repository-relative evidence paths, and
reciprocal record relationships. Optional `github_issue` metadata belongs only
on test records and stores the repository issue URL, not mutable issue state.

## Build, check, render, and audit

Run from the repository root:

```bash
# Write generated, tracked Hub views.
uv run python scripts/build_research_hub.py

# Fail if generated views drift from authored records or the ledger.
uv run python scripts/build_research_hub.py --check

# Render the complete private static site; pre/post hooks also build and audit.
quarto render research/hub

# Repeat the publication-boundary audit directly when needed.
uv run python scripts/audit_research_hub_site.py \
  --root . \
  --output-dir research/hub/_site
```

Before handing off a Hub change, also run:

```bash
uv run --extra dev pytest tests/research_hub -q
uv run --extra dev ruff check trec_mlm scripts tests
```

Generated files should change only when their canonical inputs change. The
audit prevents QMD sources from entering the rendered site; it does not make
private data safe to publish.
