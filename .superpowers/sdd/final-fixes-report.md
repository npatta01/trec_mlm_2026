# Grounded Evidence Explorer Final-Fixes Report

Date: 2026-07-23

## Outcome

The initial final-review implementation is commit `2eec8a2`
(`fix: harden grounded evidence explorer contracts`) with its report in
`53361db`. This report also incorporates the subsequent re-review hardening;
that scoped commit is recorded in the handoff.

- Producer and generic consumer now call one authoritative semantic validator.
- nDCG@10 and reciprocal-rank values must be finite and within `[0, 1]`.
- For every query and all five systems, the validator independently recomputes
  nDCG@10 and reciprocal rank from the complete rank permutation and qrel
  grades, then compares them with the bundle metrics at `1e-12` tolerance.
- Cohorts are derived exclusively from the canonical recomputed nDCG map, so a
  submitted epsilon within metric-comparison tolerance cannot alter exact-zero
  cohort semantics. Context risk, response behavior, record-ID uniqueness,
  best-record scalar/rank consistency, retrieval outcome, and mechanism are
  also rederived and checked.
- The generic consumer independently binds the final component metric to its
  run/qrels evaluation.
- All rank fields must be full candidate permutations. The documented trust
  boundary is explicit: a bundle-only generic consumer cannot authenticate
  non-final ranks against absent source artifacts; the task-specific builder
  owns that source binding.
- Bundle and HTML are staged separately and promoted as a pair only after both
  succeed. Validation or rendering failure before promotion preserves both
  prior destinations and removes staging files. A partial-promotion failure
  normally restores both destinations; if restoration itself fails, its
  unrestored backup is retained and reported as recovery-required rather than
  deleted. A backup-cleanup failure after successful promotion leaves the
  valid live pair in place and reports the retained backup as a warning.
- Grounded-mode summary and cosine labels now state their scope unambiguously;
  the no-bundle UI remains unchanged.

## TDD Evidence

Baseline before the fixes:

```text
134 passed in 6.45s
```

The targeted regression selection was run before implementation:

```text
11 failed, 9 passed
```

The failures covered the intended missing behaviors: producer metric bounds;
false cohorts, context flag, behavior, record-ID, scalar, and tie-broken
best-cosine rank semantics; final-metric source binding; paired-output
preservation on renderer failure; and grounded UI wording. The consumer
already rejected an out-of-range metric through its prior local schema check,
but the regression intentionally required the new shared-validator error path.

Focused GREEN verification after implementation:

```text
106 passed in 6.06s
All checks passed!
```

Final re-review TDD wave:

```text
RED: 9 failed, 107 passed
GREEN: 116 passed in 6.26s
```

The new regressions cover within-range nDCG and reciprocal-rank mismatches,
stale metrics paired with false cohorts, ordinary rollback, rollback failure
with a recoverable old bundle, and backup-cleanup failure after both new
outputs are already valid and live.

Canonical-cohort epsilon follow-up:

```text
RED: 1 failed, 1 passed
Focused GREEN: 118 passed in 6.31s
```

The failing case submitted `5e-13` for a canonically zero nDCG value and
adjusted the submitted cohort list accordingly; the validator now rejects that
false cohort while accepting the canonical zero cohort list.

## Final Verification

Command:

```bash
uv run pytest \
  tests/test_grounded_evidence_explorer.py \
  tests/test_build_e5_grounded_evidence_explorer.py \
  tests/test_query_explorer_skill.py \
  tests/test_raw_response_refinement_artifact.py \
  tests/test_raw_response_refinement_evaluation.py -q
```

Result:

```text
155 passed in 6.49s
```

This run included the JavaScript/executed headless-Chrome DOM coverage; no
tests were skipped.

Command:

```bash
uv run ruff check \
  trec_mlm/grounded_evidence_explorer.py \
  scripts/build_e5_grounded_evidence_explorer.py \
  tests/test_grounded_evidence_explorer.py \
  tests/test_build_e5_grounded_evidence_explorer.py \
  tests/test_query_explorer_skill.py \
  .agents/skills/inspect-query-results/scripts/build_query_explorer.py
```

Result:

```text
All checks passed!
```

`git diff --check` also passed.

## Scope and Safety

Only the grounded-explorer source, tests, design, implementation plan, and this
sanitized report were included in the final-fixes commits; the re-review wave
touches only validator code, publication code, their tests, and this report.
Pre-existing unrelated worktree changes were preserved. No official/private
bundle, rendered explorer, portal entry, sharing setting, or live artifact was
created, replaced, or published during this final-fixes task.
