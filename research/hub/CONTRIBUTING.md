# Contributing to the Research Hub

The editable source of truth is the QMD record layer under `research/hub/` plus
run facts in `experiments/ledger.json`. Generated QMD, JSON, Markdown, and HTML
are derived outputs; do not edit them by hand.

Start from the matching file in `research/hub/templates/`. Replace every example
value, keep repository paths relative to the repository root, and use a stable
lowercase ID with hyphens.

## Where facts belong

- Themes synthesize a durable research area, its current bounded conclusion,
  practical decision, confidence boundary, evidence, and next tests.
- Questions record one bounded answer and link to themes, studies, and next
  tests.
- Studies record why work was done, its protocol, result, decision, limitations,
  canonical evidence, and linked ledger runs.
- Tests in `research/hub/ideas/` are proposals or active tests, not results.
- Datasets describe provenance, permitted use, leakage constraints, and custody.
- Background records synthesize literature or benchmark context and name their
  canonical source.
- Run metrics, evidence state, evaluation scope, and artifact custody belong in
  `experiments/ledger.json`, not copied into QMD prose.

## Add a question

1. Copy `templates/question.qmd.example` to
   `questions/<question-id>.qmd`.
2. Set one `primary_theme_id` and include it in `theme_ids`.
3. State a bounded `current_answer`, `confidence`, and `confidence_boundary`.
4. Link existing studies and tests with `study_ids` and `next_test_ids`.
5. Attach at least one tracked, repository-relative canonical evidence path.

The next build automatically updates the question index, owning theme pages,
history snapshot, and machine-readable catalog.

## Add a study

1. Copy `templates/study.qmd.example` to `studies/<study-id>.qmd`.
2. Link its questions, themes, approaches, datasets, and canonical evidence.
3. Keep construction evidence separate from evaluation evidence. State qrel use
   and leakage limits in the relevant dataset and study boundary.
4. Preserve evaluation scope: only put runs from one scope in
   `comparison_system_ids`. Generated representative tables are partitioned by
   scope.
5. Keep the generated include at the end of the copied template and replace
   `STUDY_ID` with the final study ID.
6. Add the study ID to the linked question records and any linked approach or
   dataset records.

## Add a dataset

1. Copy `templates/dataset.qmd.example` to `data/<dataset-id>.qmd`.
2. Record its category, provenance, permitted use, leakage constraints,
   sharing boundary, availability, and reproducibility in front matter.
3. Keep sensitive source content out of the record body and static hub.
4. Link existing studies with `study_ids`; update those study records when the
   relationship is new.

The next build automatically updates the generated Data index and both copies
of the machine-readable catalog. Do not add the dataset manually to
`data/index.qmd` or either catalog.

## Link a run

Add or update the run through the experiment-ledger workflow, then use its exact
`system_id` in the study's `system_ids`. Put the small, reader-relevant subset in
`representative_system_ids`. Use `comparison_system_ids` only for a deliberate
within-scope comparison. Never infer a missing metric, candidate ID, artifact
state, or evaluation scope.

Construction and evaluation must remain distinguishable: a request's own qrels
may evaluate a frozen ranking afterward, but must not leak into construction
unless the method and dataset contract explicitly define a development-training
use. Hidden-test effectiveness must not be claimed from development evidence.

## Attach canonical evidence

Use tracked repository-relative paths in `evidence_refs`,
`canonical_source_path`, or `repository_path`. Prefer the structured result or
receipt plus the controlling authored report. Do not add local paths, secrets,
raw private responses, placeholder citations, or remote executable assets.

## Build generated outputs

From the repository root:

```bash
uv run python scripts/build_research_hub.py
```

## Render the site

From the repository root:

```bash
quarto render research/hub
```

The Quarto pre-render hook runs the generator with an explicit repository root,
so rendering the project by path also works from another working directory.

## Check generated outputs

After rendering:

```bash
uv run python scripts/build_research_hub.py --check
```

Before committing, run the Research Hub tests and Ruff. Commit canonical records,
generated tracked outputs, and relevant tests together.
