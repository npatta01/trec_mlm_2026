# TREC Million LLMs 2026

This repository builds systems that rank **150 candidate LLMs** for each new
request. A system returns a complete candidate ordering in TREC run format; it
does not answer the request itself. See the
[official track site](https://trec-mllm.github.io/) for organizer rules and
dates.

Development queries have relevance judgments (qrels), so they support method
development and evaluation. Test judgments are hidden. Every effectiveness
result currently recorded here is therefore **development evidence**, not
evidence about hidden-test performance.

## Choose a path

- **Get oriented:** start with the
  [current-state research map](research/hub/start-here/current-state.qmd) for
  the two-minute status and visual evidence path, then use the
  [plain-language task explanation](research/hub/start-here/problem.qmd) when
  the ranking setup is unfamiliar.
- **See what was tried and learned:** use the
  [study index](research/hub/studies/index.qmd) and
  [ideas and tests index](research/hub/ideas/index.qmd).
- **Audit an exact run:** inspect the canonical
  [experiment registry](experiments/ledger.json), including its scope,
  promotion status, evidence state, and artifact custody.
- **Add research or code:** start with the
  [research contribution guide](research/README.md) and the workflow below.

## Setup

Install [`uv`](https://docs.astral.sh/uv/), then create the default environment
with development tools:

```bash
uv sync --extra dev
```

Dense and response-based methods need their optional runtimes:

```bash
uv sync --extra dense --extra response --extra dev
```

## Run one cheap baseline

The word TF-IDF query-neighbor baseline is the quickest end-to-end development
example. It holds each query out while using the remaining development queries
as training evidence, ranks all candidates, writes a TREC run, and records
diagnostics.

```bash
uv run trec-mlm query-knn \
  --primary-selection-provenance predeclared_before_current_run \
  --output-dir artifacts/query_knn_baseline

uv run trec-mlm validate-run artifacts/query_knn_baseline/run.trec \
  --queries data/dev/llm_dev_data.tsv \
  --qrels data/dev/llm_dev_qrels.txt

uv run trec-mlm score-run \
  --qrels data/dev/llm_dev_qrels.txt \
  --run artifacts/query_knn_baseline/run.trec
```

For the hidden-test release, `trec-mlm submit-run` is the single qrel-free,
receipted entry point that turns the unseen-query file into a validated TREC
run; the frozen configuration, freeze-day invocation, and fail-closed rules are
documented in
[`docs/trec_2026_submission_pipeline.md`](docs/trec_2026_submission_pipeline.md).

The [executed notebook](notebooks/trec_2026_query_knn_baseline.ipynb) and
[read-only report](reports/trec_2026_query_knn_baseline.html) explain the
baseline in detail. To inspect any completed run query by query, follow the
repository-local
[`inspect-query-results` guide](.agents/skills/inspect-query-results/SKILL.md).

## Research Hub: conclusions and open work

The [Research Hub](research/README.md) is the living narrative layer. Its
authored records explain the task, evidence boundaries, studies, decisions,
unresolved questions, and proposed tests. Start with:

- [current status, evidence boundary, and next actions](research/hub/start-here/current-state.qmd);
- [deeper conclusions and decisions](research/hub/index.qmd);
- [completed and ongoing studies](research/hub/studies/index.qmd);
- [prospective ideas and tests](research/hub/ideas/index.qmd); and
- [frozen experiment protocols](research/hub/protocols/index.qmd).

Long method chronology belongs in those studies and protocols, not in this
README. The [background index](research/hub/background/index.qmd) covers prior
work, and the [reports index](reports/README.md) explains older reports and
generated snapshots.

## Exact experiment facts

[`experiments/ledger.json`](experiments/ledger.json) is the sole canonical
registry for run identity, metrics, evaluation scope, promotion, evidence
quality, and artifact state. The
[generated Markdown ledger](reports/trec_2026_experiment_ledger.md) is a more
readable view, but it is not an independent source.

A report, study, or artifact file existing in the repository does **not** mean
its system was promoted. Check the exact system record and comparison scope in
the registry before making a result claim.

## Contribution workflow

1. Start from the latest `origin/main` and read the relevant active
   [plan](plans/README.md), Hub study, or protocol.
2. Edit the canonical source for the information: the experiment registry for
   run facts, authored Hub QMD for conclusions and proposals, or a protocol for
   a frozen experiment contract.
3. Regenerate derived outputs instead of editing them by hand.
4. Keep construction and evaluation evidence separate. Never describe
   development results as hidden-test results.
5. Keep raw responses, large ignored artifacts, secrets, and private paths out
   of tracked documentation.

Run the relevant checks from the repository root:

```bash
uv run python scripts/build_experiment_ledger.py --check
uv run python scripts/build_research_hub.py --check
uv run --extra dev pytest
uv run --extra dev ruff check .
```

For Hub authoring, rendering, and audit commands, use
[`research/README.md`](research/README.md). Data inventory and download commands
live in [`data/dev/README.md`](data/dev/README.md).

## Repository map

| Path | Purpose |
|---|---|
| `experiments/ledger.json` | Canonical experiment and run facts |
| `research/hub/` | Authored research narrative plus generated Hub views |
| `protocols/` | Frozen experiment and leakage-boundary contracts |
| `plans/` | Active and historical working records |
| `data/dev/` | Official file manifest, tracked development data, and download notes |
| `data/test/` | Final test topics, official source, checksum, and deadline notes |
| `trec_mlm/` | Ranking, evaluation, validation, and reporting code |
| `scripts/` | Reproducible command-line entry points |
| `tests/` | Unit, integration, ledger, and Hub checks |
| `reports/` | Evidence reports, receipts, generated snapshots, and historical context |
| `artifacts/` | Ignored local run outputs; not a publishable evidence store |

The 2026 candidate set differs from 2025. Reuse methods and research ideas from
2025 material, not candidate-ID expertise or promotion claims.
