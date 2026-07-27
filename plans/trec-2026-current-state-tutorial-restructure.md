# Current-state tutorial restructure implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: use
> `superpowers:subagent-driven-development` task-by-task, with a fresh
> implementer and reviewer for each task. This externally managed worktree is
> detached and already contains the reviewed newcomer-map change, so do not
> create commits; use task-scoped before/after diffs for review instead.

**Goal:** Rebuild the existing current-state page as a ten-minute,
tutorial-first explanation of the research while preserving the verified audit
surface and its canonical evidence boundaries.

**Architecture:** Keep `research/hub/start-here/current-state.qmd` as the
human-authored frame and `research/hub/generated/current-state.qmd` as the
deterministic evidence view. Add one narrow validated loader for the off-ledger
E5 receipt, generate a six-theme reading map and a fixed cross-family result
ladder from canonical records, and move the existing audit views into a native
progressive-disclosure section without changing their evidence ownership.

**Tech stack:** Python dataclasses and JSON validation, Quarto QMD, semantic
HTML, SCSS, pytest, Ruff, headless Chrome, PDF accessibility tools.

## Status

**Completed and verified on 2026-07-25.** The tutorial-first page, generated
evidence layer, glossary, responsive presentation, tests, and sanitized private
portal snapshot are complete. The work remains in the externally managed
detached worktree; no branch, commit, push, or pull request was created.

## Global constraints

- All effectiveness claims remain development-only; hidden-test behavior is
  unknown.
- `experiments/ledger.json` remains the only run registry. An off-ledger receipt
  must never become a manufactured `SystemRecord` or promoted row.
- Every number, system status, cohort count, and study membership shown on the
  page comes from a validated canonical record or receipt.
- Use the existing fabricated balcony-basil request and the fictitious
  GardenGuide, QuickFacts, and CodeTutor candidates. Do not expose or paraphrase
  private discovery responses.
- Qualify every use of `cluster`, `response`, and `reference`; do not use those
  nouns bare when two meanings are possible.
- Preserve the seven existing audit visual IDs, their adjacent explanations,
  source and status labels, semantic screen equivalents, and print equivalents.
- The default page stays single-column, mobile-first, useful without JavaScript,
  keyboard operable, dark/light safe, reduced-motion safe, grayscale safe,
  forced-colors safe, and printable.
- Preserve the full 98-row ledger, raw paths, dense metadata, and private
  evidence outside the collapsed newcomer surface.
- Do not move evidence, run models, change GitHub state, create a branch or
  commit, push, publish publicly, enable Funnel, or expose a listener.

---

### Task 1: Normalize the completed E5 follow-up receipt

**Files:**

- Modify: `trec_mlm/research_hub/model.py`
- Modify: `trec_mlm/research_hub/build.py`
- Modify: `tests/research_hub/test_current_state_integrity.py`
- Modify: `tests/research_hub/test_build.py`

**Interfaces:**

- Produce `CompletedFollowUpEvidence`, an immutable record containing only:
  `study_id`, `source_path`, `baseline_id`, `candidate_id`, `decision`,
  `ndcg_at_10`, `mrr`, `zero_ndcg_queries`, `ndcg_delta`, `ndcg_ci_lower`,
  `ndcg_ci_upper`, `mrr_delta`, `mrr_ci_lower`, and `mrr_ci_upper`.
- Add `HubBuild.completed_follow_up: CompletedFollowUpEvidence | None = None`.
- Add `_load_completed_follow_up(root, sources, ledger)` and call it from
  `build_research_hub()` after source and ledger validation.
- Consume only the study carrying
  `current_state_role: strongest_completed_follow_up` and its canonical JSON
  evidence reference.

- [x] **Step 1: Write the failing receipt-loader tests.**

  Add literal fixtures that prove the loader accepts the prespecified primary
  comparison and rejects each realistic corruption:

  ```python
  assert evidence.baseline_id == "dense_profile_equal"
  assert evidence.candidate_id == "raw_hierarchical_dense_equal"
  assert evidence.decision == "advance"
  assert evidence.ndcg_at_10 == pytest.approx(0.4356252730181542)
  assert evidence.ndcg_delta == pytest.approx(0.015411280824014868)
  assert evidence.ndcg_ci_lower == pytest.approx(0.006964609514709062)
  ```

  Parameterize corruptions for: unfinished study, linked ledger system IDs,
  absent JSON evidence reference, wrong artifact type, wrong primary baseline,
  wrong primary candidate, missing/nonfinite/out-of-range metrics, reversed or
  invalid confidence bounds, decision other than `advance`, a false decision
  gate, and parent metrics that disagree with `dense_profile_equal` in the
  ledger.

- [x] **Step 2: Run the focused tests and verify RED.**

  Run:

  ```bash
  uv run pytest tests/research_hub/test_current_state_integrity.py tests/research_hub/test_build.py -q
  ```

  Expected: new tests fail because `CompletedFollowUpEvidence`,
  `_load_completed_follow_up`, and `HubBuild.completed_follow_up` do not exist.

- [x] **Step 3: Implement the narrow validated loader.**

  Parse `reports/trec_2026_e5_raw_response_refinement_results.json`; require
  `artifact_type == "raw_response_refinement_results"`; extract only
  `evaluation.comparisons.primary`,
  `evaluation.metrics.raw_hierarchical_dense_equal`, and the decision fields.
  Require every decision and integrity gate to be `true`. Compare the receipt's
  matched parent nDCG/MRR/zero count with canonical `dense_profile_equal`
  values. Do not retain `per_query_rows` in `HubBuild`.

- [x] **Step 4: Run focused tests and verify GREEN.**

  Run the command from Step 2; expected: all selected tests pass with pristine
  output.

- [x] **Step 5: Create a task-scoped diff and obtain spec/quality review.**

  The review must confirm receipt provenance, strict failure modes, finite
  metric validation, no raw response/per-query retention, and no ledger-row
  manufacture. Do not commit.

### Task 2: Generate the six-theme reading map and fixed result ladder

**Files:**

- Modify: `trec_mlm/research_hub/render.py`
- Modify: `tests/research_hub/test_current_state.py`
- Modify: `tests/research_hub/test_render.py`
- Modify: `tests/research_hub/test_current_state_integrity.py`

**Interfaces:**

- Replace score-driven fallback selection with fixed
  `_CURRENT_STATE_LADDER_SPECS` and `_current_state_ladder_rows(build)`.
- Registered ladder order is exactly:
  `dense_nested`, `contextual_e5`, `local_domain_prior`,
  `response_sumscore_nested`, `profile_only`, `dense_profile_equal`, and
  `dense_profile_profile_heavy`.
- Validate the exact expected evaluation scope and promotion status for every
  registered row; never sort or substitute by metric.
- Render `raw_hierarchical_dense_equal` from
  `build.completed_follow_up` in a separate off-ledger block outside the
  registered chart scale and registered count.
- Add `_CURRENT_STATE_THEME_SPECS` and render a presentation-only six-theme
  reading map that partitions all authored studies exactly once:

  1. Query baselines and neighbors: `query-controls`, `dense-query-knn`,
     `query-cluster-aggregation`.
  2. Candidate labels and learned scorers: `candidate-label-matching`,
     `label-ranker-pilots`.
  3. Candidate profiles and fusion: `behavior-profile`,
     `multiresolution-profiles`, `semantic-profiles`.
  4. Direct response evidence: `response-retrieval`,
     `response-volume-normalization`, `source-query-first`.
  5. References and bounded refinement: `four-model-reference-review`,
     `e5-raw-response-refinement`.
  6. Reliability, failures, and readiness: `gemini-grounding-failure-review`,
     `grounded-data-quality-audit`, `sparse-failure-analogue-diagnosis`.

- [x] **Step 1: Write failing behavior tests for the reading map.**

  Assert six unique presentation theme IDs, every current study ID exactly once,
  deterministic authored order, study links, explicit status vocabulary, and a
  visible statement that this is a reading map rather than a new canonical
  taxonomy.

- [x] **Step 2: Write failing behavior tests for the ladder.**

  Assert the exact fixed registered order; ledger-derived metrics/statuses;
  `aggregate_only` boundaries; non-color status labels; and a separate E5 block
  containing `outside the ledger`, `not registered`, and `not promoted`.
  Mutate the diagnostic score above and below the promoted score and prove its
  position/status never changes. Mutate an expected system's promotion status
  and require a validation error.

- [x] **Step 3: Run the focused tests and verify RED.**

  ```bash
  uv run pytest tests/research_hub/test_current_state.py tests/research_hub/test_current_state_integrity.py tests/research_hub/test_render.py -q
  ```

  Expected: failures name the missing theme map, fixed ladder, and normalized
  off-ledger renderer.

- [x] **Step 4: Implement the fixed presentation specs and renderers.**

  Keep the existing visual ID `representative-development-results` for the
  registered cross-family chart. Add `research-theme-map` as the eighth visual.
  Give both an adjacent plain-language explanation, screen and print table
  equivalents, source identity, and status. The E5 card follows the registered
  chart and is not included in its scale.

- [x] **Step 5: Recompose `render_current_state_qmd()`.**

  Generated order:

  1. canonical facts and development-only boundary;
  2. six-theme reading map;
  3. fixed registered result ladder;
  4. separate E5 completed-follow-up block;
  5. 75-case failure split;
  6. native `details.hub-audit-reference` containing ranking flow,
     source hierarchy, family decisions, and custody;
  7. visible pre-test dependency map.

  Preserve every one of the seven old visual IDs.

- [x] **Step 6: Run focused tests and verify GREEN.**

  Run the Step 3 command; expected: all selected tests pass.

- [x] **Step 7: Create a task-scoped diff and obtain spec/quality review.**

  The review must check fixed identity/status guards, exact study partition,
  E5 separation, no score-derived decisions, and complete equivalents. Do not
  commit.

### Task 3: Rewrite the authored tutorial, glossary, and responsive presentation

**Files:**

- Modify: `research/hub/start-here/current-state.qmd`
- Modify: `research/hub/start-here/glossary.qmd`
- Modify: `research/hub/assets/theme.scss`
- Modify: `tests/research_hub/test_current_state.py`
- Modify: `tests/research_hub/test_site_integration.py`

**Interfaces:**

- Keep `toc: false` and the single generated include.
- Authored content contains orientation and fabricated teaching material only;
  metrics, counts, system statuses, and receipt facts remain generated.
- Use scoped classes: `.hub-current-state-tutorial`, `.hub-example-thread`,
  `.hub-term-decoder`, `.hub-theme-map`, `.hub-theme-node`,
  `.hub-result-followup`, and `.hub-audit-reference`.

- [x] **Step 1: Add failing rendered-structure/accessibility tests.**

  Test consumer-visible HTML rather than exact prose. Require one H1, ordered
  H2/H3 levels, no duplicate IDs, native ordered-list example steps, semantic
  definition-list decoder, eight labelled figures, complete table captions or
  equivalents, and no image/canvas/script dependency. Verify the audit
  disclosure follows the tutorial/result/failure sections in DOM order.

- [x] **Step 2: Run the focused tests and verify RED.**

  ```bash
  uv run pytest tests/research_hub/test_current_state.py tests/research_hub/test_site_integration.py -q
  ```

  Expected: new tutorial structure/selectors are absent.

- [x] **Step 3: Rewrite the authored page in plain language.**

  Opening sequence:

  1. explain that one request produces a ranking of 150 candidate models;
  2. carry the exact fabricated basil request and three fabricated candidates
     through request → evidence → ranking → later evaluation;
  3. state that the example is illustrative and contains no private response;
  4. decode query/request, candidate, qrel, candidate label, query cluster,
     response-profile cluster, discovery response, target response, generated
     reference answer, control, promoted, diagnostic, and rejected;
  5. state the current development choice as “combine what worked for similar
     past requests with a summary of what each candidate's past answers suggest
     it is good at” before introducing dense/profile jargon;
  6. retain the development-only boundary and issues #39–#42.

  Link the example to `worked-example.qmd` and the short decoder to the expanded
  glossary. Do not duplicate generated metrics or counts.

- [x] **Step 4: Expand the glossary.**

  Define lexical versus semantic matching, embedding, nearest neighbor, query
  cluster, centroid, response-profile cluster, discovery/target response,
  generated reference answer, reciprocal-rank fusion, fusion dilution,
  registered/off-ledger, artifact custody, and generalization from first
  principles.

- [x] **Step 5: Implement responsive, non-color CSS.**

  Keep tutorial prose at 60–68ch and audit views at 76ch. Use one-column DOM
  order; allow two-column internal cards only above 768px. At mobile and 400%
  zoom use one column, no fixed heights, and `overflow-wrap: anywhere`. Extend
  forced-colors rules with Canvas/CanvasText/LinkText/Highlight, add 44px native
  disclosure targets and visible focus, show closed audit content in print,
  avoid page breaks inside short cards, and preserve no-script operation.

- [x] **Step 6: Run focused tests and verify GREEN.**

  Run the Step 2 command; expected: all selected tests pass.

- [x] **Step 7: Create a task-scoped diff and obtain spec/quality review.**

  Review conceptual accessibility, qualified terminology, synthetic-example
  safety, semantic order, mobile/print/forced-colors behavior, and unchanged
  canonical boundaries. Do not commit.

### Task 4: Regenerate, verify, and refresh the private artifact

**Files:**

- Regenerate: `research/hub/generated/current-state.qmd`
- Regenerate other Hub outputs only through `scripts/build_research_hub.py`
- Modify: `plans/trec-2026-newcomer-research-map.md`
- Modify: `plans/trec-2026-current-state-tutorial-restructure.md`
- Refresh the sanitized derived snapshot under
  `/home/npatta01/codex-rendered/plans/trec-2026-research-hub/`

- [x] **Step 1: Regenerate deterministic outputs and run drift checks.**

  ```bash
  uv run python scripts/build_research_hub.py
  uv run python scripts/build_research_hub.py --check
  uv run python scripts/build_experiment_ledger.py --check
  ```

- [x] **Step 2: Run code and repository verification.**

  ```bash
  uv run pytest -q
  uv run ruff check .
  git diff --check
  ```

- [x] **Step 3: Render and audit the complete Quarto site.**

  Render all pages; run the existing security/link audit; verify no QMD, raw
  response, secret, symlink, or private artifact entered `_site`.

- [x] **Step 4: Run the current-state browser/accessibility matrix.**

  Verify 1,440px, 390px, and 400% zoom/reflow; light/dark; keyboard-only;
  reduced motion; grayscale; forced colors; print/tagged PDF; no-script; one-H1
  heading order; native disclosure behavior; and complete text equivalents.

- [x] **Step 5: Obtain one broad final review and address its findings.**

  Review the complete narrative-restructure diff against this plan and the
  original newcomer-map constraints. Use one fix wave and one scoped re-review
  if needed. Do not commit.

- [x] **Step 6: Refresh and verify the authorized private portal snapshot.**

  Replace only the sanitized derived Hub copy, keep the mapping tailnet-only,
  verify the live HTTPS page and source-QMD 404, compare local/copied/served
  SHA-256 values, and ensure no Funnel is active.

- [x] **Step 7: Close the plan records.**

  Record final status, decisions, exact verification evidence, live private URL,
  and next action in both plan files. Leave branch/commit/push/PR creation as a
  separate explicitly authorized follow-up.

## Final decisions

- Use the full tutorial-first restructure while retaining the exact audit views
  under progressive disclosure.
- Teach the research as six evidence routes: similar-query neighborhoods,
  candidate labels, query clusters, source-query-first historical responses,
  candidate response archives/profiles with bounded E5 refinement, and
  generated reference answers.
- Keep the six-route map presentation-only. Authored study records and the
  experiment ledger remain the canonical owners of conclusions and run facts.
- Keep E5 visibly promising but off-ledger, unregistered, and unpromoted. Its
  receipt gate says `advance`; the concluded Hub study remains
  `informational` because no preregistered registered result exists yet.
- Preserve qualified distinctions among candidate labels, query clusters,
  response-profile clusters, discovery responses, target responses, and
  generated reference answers.

## Final verification evidence

- Full repository suite: **1,013 passed, 3 skipped**. Ruff, ledger drift, Hub
  drift, and `git diff --check` all passed on the final source tree.
- The complete serial Quarto build rendered **169/169 pages**. The rendered
  tree contains 199 files and 169 HTML pages, with no QMD or PDF sources.
- The strict site audit checked **8,172 references** across 169 HTML pages and
  194 scanned text files, with no broken links, active remote loads, secrets,
  source leakage, or symlinks.
- The browser/accessibility matrix passed **33/33 checks**, including desktop,
  390px and 320px reflow, light/dark, keyboard disclosures, reduced motion,
  grayscale, forced colors, print/no-script fallback, heading order, and 44px
  targets. Manual below-fold inspection found no clipping, overlap, or overflow.
- The tagged **23-page PDF** contains the semantic text/table equivalents and
  no JavaScript. Its SHA-256 is
  `43e4235097782727531c9b0e9ff714b89e7db7017d096f6d4ad3e6caf68c7fa9`.
- Advisor and Claude Code Opus independently recommended the tutorial-first
  structure. Broad review found five Important and three Minor issues; one fix
  wave addressed all of them, and scoped re-review approved the result with no
  remaining findings.
- The sanitized local, copied, and live current-state HTML are byte-identical:
  SHA-256
  `9171a89b4e5cd01f8eb6cfc7762aafd8e73f3ac7ca5c916a110eae4a5153a72a`.
  The live source-QMD route returns 404, and both Tailscale status checks show
  only the existing `(tailnet only)` mapping.
- Detailed browser, screenshot, PDF, and audit evidence is retained at
  `/home/npatta01/.codex/visualizations/2026/07/25/019f9af0-8835-7c30-a565-1847c8587300/trec-current-state-qa-v2/`.

## Next action

Start #39 (qrel-free unseen-query inference) and #40 (durable private artifact
custody). Use those outputs to unblock #41 (five-run portfolio freeze) and #42
(preregistered E5 inference). Branch, commit, push, and pull-request creation
remain separate follow-up actions.
