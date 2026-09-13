# TREC 2026 Research Hub

Development evidence only.

## Current position

- **OpenAI Luna pointwise production audit** (`openai-luna-pointwise-production-audit`; `adopt`): Use the compact supported label in downstream experiments; retain the five-way verdict and explanation for debugging and preserve the audited artifact unchanged. This is an artifact-use decision, not a ranking-system promotion.
- **Pointwise response-quality ranking Phase 0** (`pointwise-ranking-phase0`; `stop`): Stop pairwise work because the frozen effectiveness gate failed despite sufficient zero-call selector capacity.
- **Grounded reference data-quality audit** (`grounded-data-quality-audit`; `iterate`): Adjudicate uncertain records and a stratified non-Search sample before treating references as validated ground truth.

## Open questions

- **How reliable must a generated reference be?** (`reference-reliability`; `partially_answered`; confidence `moderate`): Gemini was the strongest reviewed default, and the production audit supports using its evidence for pointwise agreement labels; this still does not make Gemini absolute ground truth. **Confidence boundary:** The 400-question review was purposive and has no organizer gold; the grounding audit focused on known failures.
- **Can model responses provide useful ranking evidence?** (`response-evidence`; `partially_answered`; confidence `moderate`): Audited pointwise support labels are usable as an artifact, but their frozen Phase 0 rankings disagreed too strongly with organizer qrels and reduced both nDCG@10 and MRR versus the dense control. **Confidence boundary:** Most ranking evidence is development-only and some response corpora are private or locally held.
- **When do promising signals fail?** (`failure-conditions`; `open`; confidence `low`): Failures cluster around ambiguity, entity resolution, evidence custody, scope changes, and over-tuned fusion, but no complete predictor exists. **Confidence boundary:** Observed development and audit failures are diagnostic; they do not estimate hidden-test prevalence.

## Next tests

- **Test hierarchical candidate semantic profiles against the frozen centroid baseline** (`hierarchical-candidate-profiles`; `ready`; priority `high`): Compare one deterministic nested candidate hierarchy with frozen flat and multiresolution profile controls.
- **Failure-aware evidence fusion** (`failure-aware-fusion`; `idea`; priority `high`): Select evidence sources using a query-level reliability signal.
- **Reference review trigger** (`reference-review-trigger`; `idea`; priority `high`): Predict which generated references need grounding or human review.

## Navigation

- [Generated summary](../research/hub/generated/summary.qmd)
- [Research question coverage](../research/hub/generated/coverage.qmd)
- [Decision history](../research/hub/generated/history.qmd)
- [Experiment runs](../research/hub/generated/runs.qmd)
- [Machine-readable catalog](../research/catalog.json)
