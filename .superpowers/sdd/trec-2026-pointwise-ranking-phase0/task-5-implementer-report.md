# Task 5 implementer report

## Assessment

Task 5 is complete. The Research Hub now records the Phase 0 pointwise-ranking
result as a binding negative: audited pointwise support does not agree enough
with organizer qrels to improve the frozen dense control. The gate passed one
of five checks, so the recorded recommendation is `stop_pairwise_work`; no
pairwise, judge, inference, or API call was made in this task.

## Test-driven evidence

RED was established before implementation with three focused failures:

- `test_wide_per_query_tsv_selects_declared_metric_columns` rejected the new
  source kind.
- `test_pointwise_phase0_registers_exactly_two_complete_valid_negatives` found
  no Phase 0 ledger registrations.
- `test_pointwise_phase0_study_and_completed_idea_are_reciprocally_linked`
  found no completed study.

After implementation, the final focused command covering ledger unit,
integration, output, current-state, and reciprocal-link behavior passed `171`
tests. The broader ledger suites passed `148` tests and the non-browser Hub
suites passed `185` tests during implementation. Ruff passed on every changed
Python and test file, and `git diff --check` passed.

## Publication and ledger changes

- Added canonical study
  `research/hub/studies/pointwise-ranking-phase0.qmd`, with answer-first
  conclusion, terminology, exact aggregate and paired results, all required
  query and candidate slices, robustness evidence, gate, limitations, next
  action, open questions, and tracked evidence links.
- Marked `local-response-quality-judging` completed and linked the question,
  approach, pointwise-judgment data record, run-artifact data record, and study
  reciprocally. The current-state answer now identifies this study as the
  strongest direct evidence.
- Added exactly two `tracked_complete`, `evaluation_only`, `valid_negative`
  systems: `pointwise_phase0_pointwise_only` and
  `pointwise_phase0_combined`. Each has all 322 stored per-query vectors loaded
  from explicitly declared columns in the wide TSV.
- Reused the existing `dense_nested` control rather than registering a
  duplicate. The global promoted development system remains
  `dense_profile_equal`; neither Phase 0 system is promoted.
- Added generic `wide_per_query_tsv` loading while retaining the declared
  query, nDCG@10, and reciprocal-rank column contract.
- Recorded development-only, adaptive-overfit, external-encoder,
  failed-gate, and support-not-gold caveats. No hidden-test claim is made.

Generated outputs now contain 100 systems, 4,186 per-query rows, 1,950 topic
slice rows, 39 density-slice rows, 18 studies, and 13 systems with stored
per-query metrics.

## Result represented in the Hub

The combined system reaches nDCG@10 `0.2113240933733129` and MRR
`0.2691192482181426`, versus control `0.35985474409504775` and
`0.40406330811168817`. Its paired deltas are `-0.14853065072173485` nDCG@10
with 95% CI `[-0.18381098850790814, -0.1142669617790963]` and
`-0.1349440598935456` MRR with 95% CI
`[-0.17528312887959557, -0.09447628367917509]`. The lone positive candidate
slice, smoothed-prior quartile 2, is retained in context and does not overturn
the aggregate or gate result.

## Generation, render, and audit

- `scripts/build_experiment_ledger.py --check` passed.
- `scripts/build_research_hub.py --root . --check` passed.
- A full Research Hub Quarto render completed after moving a stale ignored
  Quarto cache to a recoverable temporary directory and rebuilding it.
- The study and current-state pages were rendered again after their final
  edits. The study HTML contains one visible Answer first section, eight
  compact tables, the required gate/slice/limitations/action sections, and
  valid internal links to both new systems and the existing dense control.
- `scripts/audit_research_hub_site.py --root . --output-dir
  research/hub/_site` passed: no source-format reader targets were exposed.
- A diff sanitization scan found no absolute home or temporary paths, private
  pointwise artifact paths, raw question/response text fields, or
  credential-like material.

The initial full render encountered only the stale ignored-cache condition;
the clean-cache full render, final page renders, static audit, and direct HTML
inspection all passed. The browser-backed full integration command did not
emit a capturable terminal summary in its PTY, so this report does not claim
that command as an additional pass.

## Commit and concerns

Implementation and generated outputs are committed in
`7793fcc8c6530271b355951dfc401d0a7cf10ab4` (`docs: publish pointwise Phase 0
result`). This report is intentionally a follow-up commit; resolve its commit
with `git log -1 --format=%H --` on this file to avoid a self-referential hash.

No load-bearing concern remains. The result is deliberately development-only,
pointwise support is not organizer relevance gold, and the failed gate is
binding. Future work should proceed with the non-pairwise analysis recorded in
the study rather than escalating this pointwise signal.
