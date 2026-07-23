# Grounded Evidence Explorer Final-Fixes Report

Date: 2026-07-23

## Outcome

Final-review fixes are implemented in commit `2eec8a2`
(`fix: harden grounded evidence explorer contracts`).

- Producer and generic consumer now call one authoritative semantic validator.
- nDCG@10 and reciprocal-rank values must be finite and within `[0, 1]`.
- Bundle-intrinsic cohorts, context risk, response behavior, record-ID
  uniqueness, best-record scalar/rank consistency, retrieval outcome, and
  mechanism are rederived and checked.
- The generic consumer independently binds the final component metric to its
  run/qrels evaluation.
- All rank fields must be full candidate permutations. The documented trust
  boundary is explicit: a bundle-only generic consumer cannot authenticate
  non-final ranks against absent source artifacts; the task-specific builder
  owns that source binding.
- Bundle and HTML are staged separately and promoted as a pair only after both
  succeed. Validation or rendering failure preserves both prior destinations
  and removes staging files.
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
143 passed in 6.23s
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

Only the grounded-explorer source, tests, design, and implementation plan were
included in the implementation commit. Pre-existing unrelated worktree changes
were preserved. No official/private bundle, rendered explorer, portal entry,
sharing setting, or live artifact was created, replaced, or published during
this final-fixes task.
