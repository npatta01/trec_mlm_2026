# TREC 2026 Research Hub

Development evidence only.

## Current position

- **OpenAI Luna pointwise production audit** (`openai-luna-pointwise-production-audit`; `adopt`): Use the compact supported label in downstream experiments; retain the five-way verdict and explanation for debugging and preserve the audited artifact unchanged. This is an artifact-use decision, not a ranking-system promotion.
- **Grounded reference data-quality audit** (`grounded-data-quality-audit`; `iterate`): Adjudicate uncertain records and a stratified non-Search sample before treating references as validated ground truth.
- **Four-model reference-answer review** (`four-model-reference-review`; `adopt`): Use Gemini as the default reference generator while retaining Perplexity evidence and escalation.

## Open questions

- **How reliable must a generated reference be?** (`reference-reliability`; `partially_answered`; confidence `moderate`): Gemini was the strongest reviewed default, and the production audit supports using its evidence for pointwise agreement labels; this still does not make Gemini absolute ground truth. **Confidence boundary:** The 400-question review was purposive and has no organizer gold; the grounding audit focused on known failures.
- **Can model responses provide useful ranking evidence?** (`response-evidence`; `partially_answered`; confidence `low`): A complete audited pointwise support-label artifact is available for downstream experiments, but its ranking value is not yet measured; direct lexical response retrieval underperformed. **Confidence boundary:** Most ranking evidence is development-only and some response corpora are private or locally held.
- **When do promising signals fail?** (`failure-conditions`; `open`; confidence `low`): Failures cluster around ambiguity, entity resolution, evidence custody, scope changes, and over-tuned fusion, but no complete predictor exists. **Confidence boundary:** Observed development and audit failures are diagnostic; they do not estimate hidden-test prevalence.

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
