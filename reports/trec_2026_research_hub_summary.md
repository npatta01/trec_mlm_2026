# TREC 2026 Research Hub

Development evidence only.

## Current position

- **Grounded reference data-quality audit** (`grounded-data-quality-audit`; `iterate`): Adjudicate uncertain records and a stratified non-Search sample before treating references as validated ground truth.
- **Four-model reference-answer review** (`four-model-reference-review`; `adopt`): Use Gemini as the default reference generator while retaining Perplexity evidence and escalation.
- **Gemini grounding failure review** (`gemini-grounding-failure-review`; `iterate`): Develop a two-pass disagreement trigger rather than grounding every request.

## Open questions

- **When do promising signals fail?** (`failure-conditions`; `open`; confidence `low`): Failures cluster around ambiguity, entity resolution, evidence custody, scope changes, and over-tuned fusion, but no complete predictor exists. **Confidence boundary:** Observed development and audit failures are diagnostic; they do not estimate hidden-test prevalence.
- **Can candidate-side semantic evidence improve query-only retrieval?** (`candidate-evidence`; `partially_answered`; confidence `moderate`): Yes on development evidence, subject to the evidence limits below. **Confidence boundary:** Development-only evidence; candidate artifacts have mixed custody and the hidden test is unavailable.
- **How far can query-only routing go?** (`query-only-routing`; `partially_answered`; confidence `moderate`): Dense query neighbors are a useful stable control, but candidate-side evidence improved development results. **Confidence boundary:** Development-only evidence; repeated tuning on the same queries may overfit and the hidden test is unavailable.

## Next tests

- **Test hierarchical candidate semantic profiles against the frozen centroid baseline** (`hierarchical-candidate-profiles`; `ready`; priority `high`): Compare one deterministic nested candidate hierarchy with frozen flat and multiresolution profile controls.
- **Failure-aware evidence fusion** (`failure-aware-fusion`; `idea`; priority `high`): Select evidence sources using a query-level reliability signal.
- **Pilot local LLM response-quality judging with selective pairwise ranking** (`local-response-quality-judging`; `idea`; priority `high`): Pilot a local pointwise judge and selective pairwise comparisons over candidate responses.

## Navigation

- [Generated summary](../research/hub/generated/summary.qmd)
- [Research question coverage](../research/hub/generated/coverage.qmd)
- [Decision history](../research/hub/generated/history.qmd)
- [Experiment runs](../research/hub/generated/runs.qmd)
- [Machine-readable catalog](../research/catalog.json)
