# Plans index

Plans are canonical working records and durable implementation history. A plan
existing in this directory does not by itself mean that its work is active,
complete, approved, or promoted into the experiment ledger.

## Active working records

- [Pointwise ranking Phase 0](trec-2026-pointwise-ranking-phase0.md) — active
  final review and PR handoff for the completed zero-call negative evaluation
  under GitHub issue #31; the evidence gate stops pairwise work.

## Completed or superseded

- [Newcomer research map and GitHub backlog](trec-2026-newcomer-research-map.md)
  — completed repository-clarity implementation, GitHub reconciliation,
  verification evidence, and dependency-aware next action.
- [Current-state tutorial restructure](trec-2026-current-state-tutorial-restructure.md)
  — completed tutorial-first narrative, evidence visuals, accessibility checks,
  and private rendered handoff.
- [Research Hub implementation plan](trec-2026-research-hub.md) — completed by
  PR #35 and superseded for ongoing navigation work; its checklist remains as
  historical implementation evidence.

## Scoped study records

These files preserve their own study status, decisions, and boundaries. Read
the status inside the file rather than inferring lifecycle from this index:

- [E5 raw-response refinement](e5-raw-response-refinement.md)
- [Reference-answer validation design](trec-2026-reference-answer-validation-design.md)
- [Reference-answer validation implementation](trec-2026-reference-answer-validation-implementation.md)
- [Sparse-failure analogue diagnosis](trec-2026-sparse-failure-analogue-diagnosis.md)

## Lifecycle expectations

- An **active** plan names its objective, scope, current status, decisions,
  verification evidence, and one next action.
- A **completed** plan records the delivered outcome and exact verification.
- A **superseded** plan points to its replacement while preserving historical
  checklists and decisions.
- Unknown or paused state stays explicit; do not silently mark incomplete work
  complete.
- Do not move or rename evidence referenced by a plan, receipt, or hash.

Plans may name private artifact locations to define custody, but must not embed
secrets, raw responses, private data, or publishable access links. Rendered HTML
is a derived viewing copy, never the canonical plan.
