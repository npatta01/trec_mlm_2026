# TREC 2026 sparse-failure analogue diagnosis

## Preservation update — 2026-07-25

This diagnostic was originally executed from commit `20232f6`. Its source
inputs and derived evidence were rechecked while preserving it on updated
`origin/main` commit `45872ef`; the diagnostic findings remain unchanged.

The later team-method review narrows the research implication: explicit
relation/answer-contract extraction is an unvalidated hypothesis, not a method
shown successful by the reviewed TREC 2025 teams. The more defensible next
system experiment is strong neural reranking of retrieved historical response
evidence, with the semantic fields used only for bounded human diagnosis until
they demonstrate independent signal.

## Objective

Determine whether similar historical or development queries make the official
expert label predictable for the 75 sole-positive queries on which the promoted
`dense_profile_equal` system has zero nDCG@10.

## Scope

- The 75 promoted-run failures with exactly one positive development qrel.
- Leave-one-query-out label transfer from the other 321 development queries.
- Similar-query evidence from the 7,759 released discovery queries and their
  historical candidate responses.
- A progressive-disclosure update to the existing ranking-diagnosis report.

The actual candidate responses to the development targets are not available.
Historical discovery responses are treated as evidence of attempts, never as
verified correctness.

## Decisions

- Use the official complete 322 × 150 qrel matrix as evaluation labels, while
  describing a positive as “officially labeled for expected expertise.”
- Keep development-query transfer strictly leave-one-query-out.
- Report multiple frozen query representations as a diagnostic, not as an
  unbiased test-time model comparison.
- Keep the existing report’s product-stakeholder reading path and add the new
  failure analysis without deleting earlier evidence.
- Deliver sanitized derived HTML only through the existing private tailnet
  portal. Keep the canonical plan, report input, calculations, and notebook in
  this repository.

## Current status

- [x] Recorded the original analysis base, `origin/main` commit `20232f6`.
- [x] Rechecked source custody after syncing to `origin/main` commit `45872ef`.
- [x] Reproduced the 75-query sole-positive failure slice.
- [x] Audited positive-qrel meaning and development-response availability.
- [x] Measured component recall depth and fusion disagreement.
- [x] Ran blinded historical-analogue screening over all 75 cases.
- [x] Preserved the leave-one-out lexical/dense transfer table as an exploratory
  receipt. The original producer was not retained, so this sweep is not counted
  as reproducible evidence.
- [x] Saved the integrated calculations and full 75-case table.
- [x] Executed and validated the companion notebook.
- [x] Updated and packaged the progressive-disclosure HTML report.
- [x] Verified the portal URL, tailnet-only mapping, and final repository state.
- [x] Reproduced the exact reciprocal-rank arithmetic behind the 21 fusion misses.
- [x] Assigned deterministic released-evidence proxies to all 54 no-signal misses.
- [x] Added a structured manual archetype layer with explicit uncertainty.

## Verified findings so far

- Only 11/75 cases have a strong or corroborated pre-qrel analogue signal;
  34/75 are weak or non-unique, and 30/75 are not predictably identifiable.
- In 21/75 cases one component ranks the official positive in its top ten and
  the 1:1 fusion pushes it out. In the other 54/75, neither component has a
  top-ten positive signal.
- The exploratory eight-representation receipt reports that 37/75 targets have no
  nearest-neighbor label agreement under any representation.
- That unreproduced receipt's most optimistic inspected configuration, contextual E5-Mistral with
  three neighbors, places only 10/75 official positives in the top ten. This is
  a post-hoc grid maximum, not an unbiased estimate.
- Pure MPNet nearest-neighbor label agreement is 7/75 for the failures, versus
  32/105 for the other sole-positive development queries.
- The official positive produces a non-exact-refusal response on only 36/225
  top-three MPNet discovery analogues. The profile-top-one candidate produces
  50/225. Neither number establishes correctness.
- All 21 fusion misses are cross-component vetoes: 18 are profile-only and three
  are dense-only, while the opposing component ranks the positive 21–150.
- No static weight protects both directions. Profile-only recoveries require a
  profile/dense ratio above 1.30; dense-only recoveries require it below 0.78.
- Within the 54 no-signal cases, seven are near-component misses, 24 have no close
  attempted history, and 23 have released evidence whose nearby labels point
  elsewhere. These are deterministic proxies, not qrel-error judgments.
- Structured spot checks classify 30/54 as counter-domain or opaque under the
  released evidence, but target responses and assessor rationale remain missing.

## Next action

Run a bounded, query-conditional component-protection experiment on the 21
fusion-dilution cases. Do not use another static weight sweep: the profile-only
and dense-only recovery regions are incompatible. For the 54, prioritize history
gaps and surface collisions for entity-aware retrieval or response judging, and
treat opaque-profile cases as a provenance question.

## Verification evidence

- The diagnostic builder completed with the frozen 75-case denominator and
  exact 11/34/30 evidence buckets.
- The causal builder reproduced all 48,300 equal-RRF ranks, 0.4202139922
  nDCG@10, the 21/54 split, and exact 7/24/23 no-signal proxy counts.
- Ruff and Python compilation passed for the new analysis and report scripts.
- The companion notebook executed end to end with no cell errors.
- Portable report validation, packaging, and Chromium verification passed at
  1,440-pixel and 390-pixel viewports, including keyboard source interaction.
- The live HTTPS report and portal index returned the intended content through
  the existing Tailscale Serve mapping; Funnel is not enabled.
- The plan, scripts, notebook, canonical report input, and derived evidence are
  preserved together so the 75-case diagnosis can be reproduced and audited.
- The multi-representation transfer TSV is explicitly preserved as receipt-only:
  its original producer could not be recovered, so its headline numbers require
  a fresh preregistered rerun before they are used for a system decision.
