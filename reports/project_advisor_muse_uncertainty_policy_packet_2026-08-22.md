# Advisory question

For Muse-generated grounded reference answers used to judge other LLMs, how should we handle questions where the reference model may have multiple valid interpretations? Is this mainly a prompt problem, should we capture explicit uncertainty and ambiguity metadata, or is it acceptable to mark low-confidence questions and exclude them from expertise scoring?

## Desired outcome

Choose a defensible evaluation policy that avoids rewarding or penalizing other LLMs for disagreeing with an unstable reference answer, while preserving useful questions when uncertainty is limited. The advice should distinguish prompt improvements from measurement/design changes and recommend one bounded next action before any full-dataset run.

## User constraints and corrections

- The reference answer will be used to judge other LLM answers, so evaluation fairness matters more than forcing one canonical answer.
- The prompt must remain generic and not encode a Q10-specific answer.
- Muse uses two grounded calls with `previous_response_id`; Search use and output length are not application-capped. More than two Search calls are allowed when useful.
- Long reference presentations are acceptable. Wikipedia and secondary sources are acceptable when relevant and non-fabricated; better sources should be preferred when available.
- The full dataset has not been run under the ambiguity-aware variants.
- Review is read-only: do not implement, launch another run, or change evaluation data in this advisory turn.

## Current project state

- Repository and authoritative worktree: this repository root
- Branch and HEAD: detached worktree at `9bcbd6c01ad6002a9006112adb2090a5d8ae1183`
- Dirty state: task files, scripts, tests, reports, and private Muse caches are untracked; no unrelated tracked-file changes are in scope.
- Active workstreams: direct Meta Muse adapter, two-call grounded reference generation, ambiguity pilot and evaluation-policy review.

## Evidence

### E1 — Prior 50-question generic reference canary

- Status: Verified
- Source: `reports/muse_canary_50_generic_prompt_validation_2026-08-22.md`; `artifacts/muse_canary_50_reference_generic.json`
- Relevance: Shows the existing two-call grounded path is operational, but a successful API run does not guarantee semantic stability.
- Excerpt or compact result: 50/50 stage-2 finals, zero hard failures/fallbacks, 884 Search actions, 49/50 final records with URL citation annotations. Q10 (`What type of vegetation does Lythrum and Erythrina have in common?`) changed semantic dimension across grounded runs; the report explicitly records habitat/vegetation and plant life-form readings.

### E2 — Baseline ambiguity pilot, `reference_generic_branches`

- Status: Verified
- Source: `reports/muse_ambiguity_pilot_2026-08-22.md`; `artifacts/muse_ambiguity_pilot_2026-08-22.json`
- Relevance: Tests whether a generic conditional-branch prompt captures ambiguity without over-branching clear questions.
- Excerpt or compact result: 27/27 stage-2 finals, zero failures/fallbacks, 282 Search queries, 240 URL citations, 4/27 records with zero parsed citations. All 9 ambiguous-category records collapsed to one answer format. Q10 selected herbaceous/life-form in repeats 1–2 and wetland/hydric vegetation in repeat 3.

### E3 — v2 explicit decision contract

- Status: Verified
- Source: `reports/muse_ambiguity_pilot_v2_2026-08-22.md`; `artifacts/muse_ambiguity_pilot_v2_2026-08-22.json`
- Relevance: Tests an exact `Interpretation decision: SINGLE|MULTIPLE` contract and warns against silently replacing a semantic dimension.
- Excerpt or compact result: 27/27 stage-2 finals, zero failures/fallbacks, 337 Search queries, 9,733,554 combined tokens, 3/27 zero-citation records. Decision lines: 24 `SINGLE`, 3 `MULTIPLE`. Q10 was `MULTIPLE` in repeats 1 and 3 but `SINGLE` in repeat 2. Clear controls were 9/9 `SINGLE`; one broad control over-branched.

### E4 — v3 dimension-search contract

- Status: Verified
- Source: `reports/muse_ambiguity_pilot_v3_2026-08-22.md`; `artifacts/muse_ambiguity_pilot_v3_2026-08-22.json`
- Relevance: Tests whether requiring a focused Search for every plausible semantic dimension makes ambiguity capture stable.
- Excerpt or compact result: 27/27 stage-2 finals, zero failures/fallbacks, 364 Search queries, 9,580,771 combined tokens, 4/27 zero-citation records. Decision lines: 25 `SINGLE`, 2 `MULTIPLE`. Clear controls were 9/9 `SINGLE`; broad controls were 9/9 `SINGLE`; Q10 was still `MULTIPLE` in 2/3 repeats and `SINGLE` in 1/3.

### E5 — Q10 has genuinely competing evidence dimensions

- Status: Verified
- Source: independent source checks recorded during the pilot review: Kew Plants of the World Online pages for `Lythrum` and `Erythrina`; USGS NVCS Lythrum vegetation entry; PROSEA `Erythrina fusca` ecology.
- Relevance: Establishes that the instability is not merely spelling or citation noise. Different semantic dimensions have real source support.
- Excerpt or compact result: Kew supports life-form descriptions including perennial herbs for Lythrum and mostly trees/shrubs but rarely perennial herbs for Erythrina. USGS and PROSEA support wetland/swamp vegetation and habitat for Lythrum/Erythrina species. The question does not explicitly select life-form versus ecological vegetation/habitat.

### E6 — Existing reference-generation uncertainty vocabulary

- Status: Verified
- Source: `trec_mlm/reference_generation.py:24-36,200-260`
- Relevance: Shows the repository already has a conceptual place for ambiguity and evidence conflict, but the Muse path currently stores a free-text provisional reference for the downstream judge.
- Excerpt or compact result: The shared reference prompt recognizes `ANSWERED`, `AMBIGUOUS`, `INSUFFICIENT_EVIDENCE`, and `CONFLICTING_EVIDENCE`, with an `UNCERTAINTY_REASON`; it instructs the model not to silently choose among multiple plausible interpretations. Muse reference records are currently free-text sections, and the pointwise evaluator passes `provisional_reference` as text rather than consuming a structured branch schema.

## Candidate choices or proposed plan

- A — Keep improving the generic prompt until the model consistently emits branches.
- B — Add an explicit uncertainty/ambiguity record: decision type, interpretation branches, evidence per branch, confidence, and a flattened acceptance set for grading.
- C — Mark unstable questions as `UNRESOLVED_REFERENCE` and exclude or separately report them in expertise scoring; do not force a reference answer.
- D — Ignore the instability and use the model's single canonical answer.
- E — Hybrid: use prompt + structured ambiguity capture, then quarantine questions that fail a repeatability/branch-consistency gate.

## Known unknowns

- There is no independent gold label for which Q10 interpretation the original benchmark author intended.
- The current pilots measure prompt behavior but do not establish calibrated confidence or a validated threshold for excluding questions.
- The downstream scoring code has not yet been changed to consume structured branches or an exclusion status.
- It is unknown how many questions in the full dataset have the same ambiguity pattern; the pilot is intentionally small.

## Scope boundary

Read-only advisory review. No implementation, full-dataset generation, scoring change, purchase, publication, or external message is authorized by this packet.
