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
  `research/hub/studies/pointwise-ranking-phase0.qmd`, with condition-bound
  terminology, method/leakage boundary, limitations, next action, and open
  questions. Exact quantitative evidence is generated from validated
  structured sources rather than copied into authored prose.
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

## Minor correction follow-up

Review found one display-rounding error in the candidate-slice table:
smoothed-prior Q4 prior-only nDCG receipt `0.11471287098429887` rounds to
`.114713`, not `.114712`. Commit
`895b482c1e66d29f584230da78dbc051b3020cd1` corrects only that value in the
canonical study and regenerated study copy; no other scientific value changed.

After the correction, the study-only render and full 175-page Hub render both
completed. The Hub build check, 23 focused current-state/study-link tests,
static site audit, rendered-value and three-run-link checks, whitespace check,
old-value absence check, and sanitization scan all passed. No Python changed,
so no additional Ruff run was applicable.

## Standards review follow-up

Whole-branch review found that the original source-only HTML comment duplicated
run metrics, slices, and gate facts that belong to structured evidence under
`research/hub/CONTRIBUTING.md`. Commit
`f648a794a5181c5d5aca56a3faaf257b26aa4b85` removes that duplication and adds a
typed, fail-closed Phase 0 evidence path to the Hub build.

- The canonical QMD now contains only definitions, frozen method and leakage
  boundary, limitations, next action, and further questions. It contains no
  copied metric literals, evidence hashes, or Markdown tables.
- The loader activates only when `pointwise-ranking-phase0` exists. It validates
  exact study/system/evidence identities; ledger aggregates, scope, status,
  promotion, qrel use, and custody; tracked report hashes; validated provenance;
  all unique per-query rows; aggregate and baseline reconciliation; every query
  and candidate partition; candidate membership; paired differences, W/T/L and
  transitions; selector state; and the binding gate.
- The generated study deterministically renders all five systems, four paired
  rows, two transition rows, nine query strata, eight candidate strata, the
  lone-positive context, independent validation, the one-of-five gate, and
  evidence links. Comparing old and new generated reports found no removed or
  added displayed numeric value.

TDD recorded RED for authored-QMD table duplication, the missing typed evidence
on `HubBuild`, candidate-membership drift that initially escaped validation, and
baseline-reproduction numeric drift that initially escaped reconciliation. The
final focused Phase 0 suite passed 9 tests; the full non-browser Hub
build/render/current-state/validation selection passed 195 tests. Hub build
`--check`, Ruff, whitespace and privacy scans, and the static audit passed.

For final rendering, only the ignored `research/hub/.quarto` cache was moved to
a recoverable temporary directory. A clean-cache render completed all 175 Hub
pages from the Hub project root, followed by a final study render. HTML
inspection confirmed one Answer-first copy, eight tables, the corrected
`.114713` value, and three of three internal run links resolving. The portal was
not modified.

Spec re-review clarified one provenance sentence without changing any number or
scientific conclusion. The generated study now attributes the full development
query/source-question ID and normalized-text overlap invariant to the qrel-free
construction preflight. It separately states that independent validation
reconciled the sanitized construction and selector receipts. The exact rendered
wording assertion failed before the change and passed afterward. The 9 focused
Phase 0 tests, Hub build check, Ruff, whitespace/privacy checks, clean-cache
study render, rendered-wording check, and static audit all passed.

## Commit and concerns

Implementation and generated outputs are committed in
`7793fcc8c6530271b355951dfc401d0a7cf10ab4` (`docs: publish pointwise Phase 0
result`), followed by the display-rounding and structured-evidence correction
commits above. This report update is intentionally a follow-up commit; resolve
its latest commit with `git log -1 --format=%H --` on this file to avoid a
self-referential hash.

No load-bearing concern remains. The result is deliberately development-only,
pointwise support is not organizer relevance gold, and the failed gate is
binding. Future work should proceed with the non-pairwise analysis recorded in
the study rather than escalating this pointwise signal.
