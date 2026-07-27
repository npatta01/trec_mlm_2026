# TREC 2026 newcomer research map and GitHub backlog

## Objective

Give a technically capable newcomer a trustworthy two-minute status, a
ten-minute visual explanation, and optional audit detail without creating a
second experiment ledger. Connect the living Research Hub to a scoped GitHub
backlog for the engineering and research work that remains before and after the
hidden-test release.

## Status

**Implemented and verified on 2026-07-25 from `origin/main` commit
`fd07842`.** The approved full tutorial-first restructure is complete, while
the original audit views remain available through progressive disclosure. Its
closed scoped record is
[`plans/trec-2026-current-state-tutorial-restructure.md`](trec-2026-current-state-tutorial-restructure.md).

The implementation is running in an isolated, externally managed worktree at a
detached HEAD. No branch, commit, push, public deployment, model run, evidence
move, or private-artifact upload is part of this work.

## Canonical boundaries

- `experiments/ledger.json` remains the sole canonical registry for run facts,
  metrics, scope, promotion status, evidence quality, and artifact custody.
- Canonical research narrative remains in authored QMD records under
  `research/hub/`; generated QMD, JSON, Markdown, and rendered HTML are views.
- `protocols/` preserves frozen experiment contracts; hashed evidence remains at
  its existing path.
- `reports/` contains durable evidence, receipts, and generated or historical
  reading views; ignored `artifacts/` and private stores remain outside the
  newcomer surface.
- The private Tailscale Serve portal is the only authorized rendered handoff.
  Funnel, public listeners, Codex Sites, and sharing-permission changes are out
  of scope.

## Verified starting state

- 11 experiment families and 98 registered systems.
- 16 authored studies and one promoted development system.
- 3 artifact-complete systems, 34 receipt-only systems, 18 report snapshots,
  and 43 portability blockers.
- 75 sole-positive promoted-system failures split into 21 fusion-dilution cases
  and 54 cases with no top-ten signal from either component.
- All registered effectiveness results are development-set evidence; hidden-test
  behavior is unknown.
- E5 raw-response refinement is the strongest completed follow-up candidate, but
  it is not a promoted or registered 98-system ledger result.

## Global constraints

- Keep the default reading path single-column, mobile-first, concise, and useful
  without JavaScript.
- Use semantic HTML, CSS, and SVG with adjacent explanations, complete text or
  table equivalents, source identity, explicit status labels, and non-color
  encodings that survive grayscale and forced-colors modes.
- Do not put the full 98-row table, raw artifact paths, dense implementation
  metadata, raw responses, secrets, or private data on the collapsed page.
- Generate quantitative current-state content from canonical records; do not
  manually duplicate ledger metrics or allow diagnostic scores to appear as
  promoted evidence.
- Preserve existing evidence paths and hashes.
- Audit live GitHub work for duplicates before creating or modifying issues.

## Task 1: Research Hub metadata and current-state source

- Add optional `github_issue` metadata to Hub test records.
- Validate it as an issue URL for this repository, render `Track in GitHub #N`,
  and never cache mutable GitHub state in generated files.
- Add canonical Hub test records for existing issues #19 and #31.
- Generate the quantitative include consumed by
  `research/hub/start-here/current-state.qmd` from the ledger and Hub catalog.
- Add tests first and preserve deterministic generated-output checks.

## Task 2: Accessible newcomer current-state report

- Create `research/hub/start-here/current-state.qmd` with the answer, evidence
  boundary, and three next actions visible before progressive detail.
- Add seven evidence visuals: ranking flow, canonical-source hierarchy,
  experiment-family decision map, representative full-development result bars,
  artifact-custody composition, the 21/54 remaining-failure split, and the
  pre-test engineering/research dependency map.
- Distinguish promoted, control, diagnostic, and rejected systems by labels,
  pattern/shape/tone, and text—not color alone.
- Give every visual an adjacent explanation, complete semantic equivalent,
  source identity, and explicit evidence status.
- Add the page to the Hub reading path without exposing a dense 98-row table.

## Task 3: Newcomer documentation and lifecycle indexes

- Refactor `README.md` into a concise path chooser: task overview, setup, one
  cheap baseline, Research Hub, exact ledger, contribution workflow, and a
  repository map. Link method chronology to existing studies and protocols.
- Expand `research/README.md` and add `reports/README.md` and `plans/README.md`
  explaining canonical, generated, historical, private, and active content.
- Mark `plans/trec-2026-research-hub.md` completed/superseded while preserving
  its historical checklist, with implementation PR/commit and verification
  references.
- Keep this working record current with decisions, evidence, and the next action.

## Task 4: GitHub contribution interfaces

- Add engineering and research issue forms under `.github/ISSUE_TEMPLATE/`.
- Engineering intake must require evidence, outcome, exclusions, acceptance
  criteria, privacy/artifact boundaries, dependencies, and verification.
- Research intake additionally requires hypothesis/null, leakage boundary,
  frozen variables, controls, primary metric, decision/stop rule, uncertainty,
  and interpretation limits.
- Add a conservative issue-template configuration that does not advertise an
  unsafe public support or contact channel.

## Task 5: Live GitHub backlog reconciliation

- Create the minimal label set: `kind/engineering`, `kind/research`,
  `kind/docs`; `priority/P0`, `priority/P1`, `priority/P2`;
  `needs-preregistration`, `blocked/external-release`, and
  `blocked/private-artifact`.
- Create milestones `Repository clarity & reproducibility`,
  `Pre-test freeze — 2026-09-04`,
  `Test execution & submission — 2026-09-05–21`, and
  `Post-evaluation analysis — 2026-10`.
- Update #11 with completed children #12/#13 and close it as completed.
- Classify and reciprocally cross-link #19, #31, #33, and #34.
- After a duplicate audit, create exactly the twelve approved P0/P1/P2 issues
  for unseen-query inference, artifact custody, portfolio freeze, E5
  registration, standard failure cohorts, manifest hashes, CI gates, newcomer
  Hub docs, component preservation, disagreement calibration, historical-plan
  lifecycle, and semantic failure clustering.
- Do not reopen rejected query-cluster/source-query-first variants or speculative
  entity-aware work without a frozen mechanism and evaluation boundary.

## Task 6: Verification and private delivery

- Run the full test suite and Ruff plus ledger and Hub generated-output checks.
- Render the complete Quarto site and run its security/link audit.
- Verify the current-state page at desktop and 390-pixel widths, light/dark,
  keyboard-only, reduced-motion, grayscale, forced colors, print/no-script, and
  screen-reader heading order.
- Confirm every quantitative value is generated from canonical records and no
  diagnostic score is presented as promoted.
- Audit titles, bodies, dependencies, milestones, labels, and reciprocal Hub
  links against live GitHub.
- Replace the sanitized private Hub snapshot, add a direct newcomer link to the
  portal, verify the live HTTPS page, and confirm Tailscale Serve remains
  tailnet-only.

## Baseline evidence

- Detached worktree fast-forwarded cleanly from `5445cd7` to `fd07842`.
- Full declared dependency environment installed with `dense`, `response`, and
  `dev` extras.
- Baseline test run: 932 passed, 3 skipped, and 3 browser/socket tests were
  blocked by the managed sandbox rather than assertion failures.
- Baseline Ruff found one pre-existing unused notebook import; it will be fixed
  and called out in the implementation diff.

## Decisions log

- 2026-07-25: Use the existing Quarto Research Hub as the sole report surface;
  no parallel standalone report source.
- 2026-07-25: Keep E5 raw-response refinement visibly promising but outside the
  registered development leaderboard until a new preregistered run exists.
- 2026-07-25: Treat browser/socket sandbox failures as environment evidence and
  repeat the affected checks with the narrowest required escalation.
- 2026-07-25: Created the approved GitHub taxonomy and four milestones, closed
  #11 as completed, classified/cross-linked #19, #31, #33, and #34, and created
  the duplicate-audited backlog as #39–#50. Authored Hub records, rather than
  generated views, hold reciprocal issue URLs.
- 2026-07-25: Keep `start-here/current-state.qmd` authored and its quantitative
  evidence in a generated include. The generator rejects missing result roles,
  prevents diagnostic rows from being represented as promoted, and verifies
  that the E5 follow-up remains outside the registered ledger.
- 2026-07-25: Disable the page-local Quarto table of contents so the first DOM
  heading is the report H1 and the page remains a true single column at desktop
  and mobile widths.
- 2026-07-25: Publish only the sanitized rendered Hub through the existing
  tailnet-only Tailscale Serve path. The live newcomer page and portal index
  were verified byte-for-byte against their local copies; the QMD source route
  returns 404 and no Funnel is active.
- 2026-07-25: Final review required print-only semantic equivalents outside
  closed disclosures, valid zero-value handling, and stronger canonical guards.
  Failure cohorts now come from authored study metadata and must match a
  completed receipt; the dependency snapshot's authored edges must match its
  visible lanes; and E5's completed-follow-up role requires a concluded study.
- 2026-07-25: Advisor and Claude Code Opus independently found the first page
  evidence-safe but conceptually too terse. The approved revision keeps the same
  report surface and canonical owners, but rebuilds the default reading path
  around a fabricated worked example, qualified terminology, six presentation
  themes covering all 16 studies, a fixed cross-family result ladder, and the
  existing verified audit views under progressive disclosure.
- 2026-07-25: “Past response” and “past LLM responses” are one historical
  response corpus used in different ways. The revision distinguishes direct
  lexical search, candidate-level semantic profiles, source-query-first use, and
  bounded raw-response refinement. It also distinguishes candidate labels,
  query clusters, and response-profile clusters rather than calling all three a
  “cluster name.”
- 2026-07-25: Final review tightened the dense-neighbor and source-query-first
  explanations, separated the three layers of E5 status, repaired audit heading
  ownership, and strengthened receipt-gate, zero-count, and metric-delta
  validation. Re-review found no remaining Critical, Important, or Minor issue.
- 2026-07-26: Adapted the answer-first guide pattern from `trec_video_2026`
  PR #22 without copying its authored-ledger architecture. Current State now
  leads with a one-minute answer and exact reader-action mapping; the six study
  groupings are named method families; the complete synthetic worked example,
  failure-mechanism library, branch-complete pre-test queue, and contributor
  lifecycle are separate, linked surfaces. Canonical ledger/Hub ownership and
  the off-ledger, unregistered, unpromoted E5 boundary remain enforced.

## Next action

Review the refreshed private newcomer page, then start with #39 (unseen-query
inference) and #40 (artifact custody). Use their outputs to close dependencies
on #41 (portfolio freeze) and #42 (E5 registration). Branch, commit, push, and
pull-request creation remain separate follow-up actions.

## Final verification evidence

- Full repository suite: **1,013 passed, 3 skipped**.
- Ruff: clean. Ledger generated-output check: clean. Research Hub drift check:
  clean. `git diff --check`: clean.
- Complete serial Quarto render: **169/169 pages** and 199 rendered files. The
  built-in audit and a strict **8,172-reference** local-link, anchor, and
  security audit both passed; the rendered tree contains no QMD sources, PDFs,
  symlinks, active remote loads, secrets, or private artifacts.
- Browser QA passed **33/33 checks** at 1,440-pixel desktop, 390-pixel and
  320-pixel mobile widths, and 400% reflow in light, dark, reduced-motion,
  grayscale, forced-colors, print, and no-script modes. Native disclosures are
  keyboard-operable with visible focus and 44-pixel targets; heading structure
  and semantic equivalents passed. The tagged 23-page PDF has no JavaScript.
- The generated current-state include reports 11 families, 98 systems, 16
  studies, one promoted development system, custody 3/34/18/43, and the 21/54
  failure split from canonical records. Its tests enforce the development-only
  boundary and role labels for promoted, control, diagnostic, and rejected
  rows.
- The private snapshot exactly matches `research/hub/_site/`. The live page is
  `https://npatta01-framework.tail481212.ts.net/plans/trec-2026-research-hub/start-here/current-state.html`;
  its local, copied, and served SHA-256 is
  `9171a89b4e5cd01f8eb6cfc7762aafd8e73f3ac7ca5c916a110eae4a5153a72a`;
  the source-QMD URL returned 404, and both Tailscale status checks showed only
  the existing `(tailnet only)` portal mapping.
- GitHub reconciliation completed after an exact-title duplicate audit: nine
  labels, four milestones, updated issues #11/#19/#31/#33/#34, and the approved
  twelve-issue backlog at #39–#50 with reciprocal Hub links where required.
- Browser QA evidence is retained under
  `/home/npatta01/.codex/visualizations/2026/07/25/019f9af0-8835-7c30-a565-1847c8587300/trec-current-state-qa-v2/`.
- Advisor and Claude Code Opus independently shaped the tutorial-first
  revision. Final broad review found five Important and three Minor issues; one
  fix wave addressed all findings, and focused re-review approved the result
  with no remaining Critical, Important, or Minor issue.

### 2026-07-26 answer-first adaptation verification

- Full repository suite: **1,022 passed, 3 skipped**; Ruff, Hub generated drift,
  and `git diff --check` passed.
- Complete serial Quarto render: **170/170 pages**; the built-in link/security
  audit passed after all newcomer links were routed to published surfaces.
- Desktop and 390-pixel screenshots confirm two-column action cards above
  768px and DOM-order single-column stacking on mobile.
- The refreshed local, private-copy, and live current-state HTML share SHA-256
  `2c3b8319003b6fe4672dbe9d19390021dd04420622dfa054e07bc16c65b15a82`.
  The live HTTPS route returned 200 and Tailscale Serve remained `tailnet only`.
