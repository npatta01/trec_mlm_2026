# Reports index

`reports/` contains durable study evidence, receipts, generated reading views,
and historical context. It is not a second experiment registry.

## How to read a result

1. Find the controlling study in the
   [Research Hub study index](../research/hub/studies/index.qmd).
2. Resolve every claimed system in the canonical
   [`experiments/ledger.json`](../experiments/ledger.json).
3. Check evaluation scope, promotion status, evidence quality, caveats, and
   artifact custody before comparing or quoting a metric.

File presence alone does **not** imply that a system was promoted, that two
scores are comparable, or that its full artifact is reproducible.

## Content types

- **Canonical study evidence:** authored reports and structured result receipts
  referenced by a Hub study or ledger record. They are authoritative only for
  their stated, bounded study.
- **Generated snapshots:**
  [`trec_2026_experiment_ledger.md`](trec_2026_experiment_ledger.md), its JSON
  and TSV companions, and
  [`trec_2026_research_hub_summary.md`](trec_2026_research_hub_summary.md) are
  derived from canonical records. Regenerate them; do not edit them directly.
- **Historical material:** 2025 reviews and the
  [approaches guide](trec_2026_approaches_guide.html) provide method context.
  They do not establish 2026 candidate identities, metrics, or promotion.
- **Artifact receipts:** `*.artifact.json` and other custody receipts describe
  a preserved output or recovery boundary. A receipt is not automatically a
  complete ranking artifact or an effectiveness result.

## Private and artifact boundary

Tracked reports must be safe, durable evidence or sanitized reading views. Raw
responses, provider caches, large model outputs, secrets, private URLs, and
unreviewed logs belong in ignored `artifacts/` or an approved private store, not
in this directory. A tracked report is not authorization to publish its source
artifact or change its sharing permissions.
