# PR #22 Guide-Pattern Adaptation — Implementation Plan

> **For Codex:** Execute this plan with test-first, task-scoped implementation and review. This is an externally managed detached worktree with substantial existing changes: do not create commits or overwrite unrelated work.

## Objective

Turn the Research Hub newcomer path into an answer-first guide that explains what the ranking system does, what was learned, what remains unknown, and what should happen next. Borrow the information architecture of `trec_video_2026` PR #22 while preserving this repository's canonical ledger, generated Hub, accessibility, privacy, and evidence boundaries.

## Global constraints

- Canonical experiment facts remain in `experiments/ledger.json`; canonical research claims remain in Hub records. Generated pages must not become competing sources.
- E5 raw-response refinement stays informational, off-ledger, unregistered, and unpromoted.
- Do not expose query text, raw responses, private IDs, artifact paths, datasets, or model rationale.
- Do not change ranking algorithms, experiment results, GitHub state, or publication state.
- Preserve `research-theme-map` as the visual ID while relabeling it “Studies by method family.”
- Cards stack in DOM order at mobile/400% zoom and use existing 44px targets, non-color status encoding, forced-colors, reduced-motion, print, and no-script fallbacks.
- Use `apply_patch` for edits. Run generators rather than hand-editing generated files.

## Task 1 — Generated answer-first research map

**Files:** `trec_mlm/research_hub/render.py`, Hub theme records/catalog schema as required, `tests/research_hub/test_current_state.py`, `tests/research_hub/test_render.py`.

1. Add behavior tests that fail when reader actions, method-family answers, or canonical-theme evidence/open-question fields are absent or inconsistent.
2. Verify the tests fail for the missing behavior.
3. Extend the six presentation-only method families with purpose, working answer, strongest canonical study/status, open question, next valid test/boundary, and complete ordered study membership. Validate exact partition and strongest-study membership.
4. Add required `strongest_evidence_summary` and `open_question` metadata to the four canonical themes and render answer-first canonical theme cards.
5. Generate the decision strip mapping canonical statuses to: Use now; Retain as controls; Promising follow-up; Still required before test; Do not reuse tested form; Unknown.
6. Run focused tests and Ruff; refactor only after green.
7. Record task-scoped diff and obtain spec/quality review.

## Task 2 — Action-first newcomer narrative and worked example

**Files:** `research/hub/start-here/current-state.qmd`, `research/hub/start-here/worked-example.qmd`, `research/hub/assets/theme.scss`, relevant site/integration tests.

1. Add failing rendered-site tests for the one-minute summary, explicit development/hidden-test boundary, action strip placement, worked-example stages, time-budgeted paths, and mobile-safe structure.
2. Verify failures represent missing reader behavior.
3. Reorder Current State: one-minute summary → evidence boundary → generated actions → worked example → failure preview → evidence routes/method families → 2/10/20-minute and contributor paths. Keep audit detail progressively disclosed.
4. Expand “How One Ranking Is Made” into ten numbered, synthetic basil-request stages: objective; allowed/forbidden evidence; similar requests; candidate labels; response archives/profiles; separate fabricated signal rankings; frozen equal-RRF ranking; legitimate development evaluation; post-hoc diagnosis/next falsifier; limits.
5. Keep the six evidence routes explicit and distinct from six method families and four canonical themes.
6. Add minimal responsive styling only where the existing system lacks the required component behavior.
7. Render focused pages; run focused tests and accessibility checks; obtain task review.

## Task 3 — Failure mechanisms, pre-test queue, and contributor lifecycle

**Files:** `research/hub/start-here/failure-library.qmd`, `research/hub/generated/failure-library.qmd`, `research/hub/background/pretest-readiness-plan.qmd`, `research/hub/CONTRIBUTING.md`, `trec_mlm/research_hub/render.py`, relevant tests.

1. Add failing generator/site tests for three privacy-safe mechanisms and complete #39–#42 stable work contracts.
2. Verify red.
3. Generate a failure library containing: fusion dilution (21), no component signal (54), and unreliable generated evidence (no prevalence count). Each record states failure, evidence, response/change, outcome, possible location, competing explanation, defect/gap/uncertainty, limit, next falsifier, source/status; state that cases explain mechanisms, not prevalence.
4. Expand the readiness queue with dependencies, outcome/hypothesis, smallest valid work item, controls, decision/stop rule, positive-result boundary, blocker, and stable links. Encode #39/#40 independent, #41 dependent on both, and #42 dependent on executable unseen-query inference.
5. Add the contributor lifecycle, ownership, update-location table, before/after checklists, reconsideration conditions, and copyable handoff while retaining technical commands and schemas.
6. Add companion-page links to newcomer paths without redesigning global navigation.
7. Run focused tests, generator drift checks, and task review.

## Task 4 — Integration, verification, and private delivery

1. Regenerate all Hub outputs and verify no generated drift.
2. Run the complete test suite, Ruff, ledger checks, Hub checks, full Quarto render, security/link audit, and quantitative/status integrity checks.
3. Review desktop and 390px layouts, light/dark, keyboard-only, reduced motion, grayscale, forced colors, print/no-script, screen-reader heading order, and 400% zoom behavior. Capture screenshots.
4. Obtain a broad independent review; resolve load-bearing findings and repeat affected checks.
5. Update both canonical plans with status, decisions, verification evidence, and next action.
6. Replace only the sanitized private Hub snapshot under `/home/npatta01/codex-rendered/plans/trec-2026-research-hub/`, update the portal newcomer link if needed, verify live HTTPS content, and confirm Tailscale Serve remains tailnet-only.
7. Hand off the live private page URL, canonical source links, screenshots, verified checks, and any remaining limitation.

## Current status

- Approved design: complete.
- Implementation: complete.
- Verification: 1,022 tests passed, 3 skipped; Ruff, generator drift, diff,
  full 170-page Quarto render, and its link/security audit passed.
- Review: Task 1 canonical/status review findings were resolved by exact
  mapping assertions, 44-pixel issue-link targets, and status-flexible study
  partition checks.
- Next action: refresh and visually verify the sanitized private portal copy.
