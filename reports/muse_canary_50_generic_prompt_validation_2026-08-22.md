# Generic reference-prompt validation

Date: 2026-08-22
Scope: same 50 development questions, two grounded Responses API calls per question

## Result

The generic semantic-discipline prompt completed all 50 questions successfully:

- 50/50 stage-2 finals
- 0 hard failures
- 0 stage-1 fallbacks
- 50/50 produced all five requested reference sections
- 49/50 final answers had inline URL citation annotations

The one missing citation annotation was Q10, the ambiguity case under investigation. It searched extensively but returned source discussion without attached final-answer URL annotations. This is a provenance-format regression, not an API or answer-generation failure.

## Usage comparison

| Metric | Original answer prompt | Earlier reference prompt | Generic reference prompt |
| --- | ---: | ---: | ---: |
| Search actions | 729 | 853 | 884 |
| Search queries | 411 | 478 | 528 |
| Page opens | 310 | 364 | 346 |
| Questions with citations | 50 | 50 | 49 |
| Citation annotations | 330 | 365 | 327 |
| Input tokens | 12,072,079 | 13,617,283 | 16,131,945 |
| Output tokens | 206,037 | 254,395 | 290,528 |
| Total tokens | 12,278,116 | 13,871,678 | 16,422,473 |

The generic prompt costs more because it asks for semantic checking, scope checking, and explicit reference sections. There is still no application-imposed Search or output-length limit.

## Answer-level review

I reviewed the canonical answers across all 50 records against the original run. The other 49 questions retained the same answer content; changes were mainly clearer scope, aliases, dates, or caveats. No clear collateral semantic regression was found.

For Q10, the generic prompt improved the behavior relative to the earlier revised prompt: it no longer treats “deciduous herbaceous perennial” as an invariant genus-level answer. It identifies “perennial flowering plants” as a broader commonality and explicitly preserves ambiguity around habitat, life form, and genus-level scope. Because this question has no formal gold answer in the canary, the prompt should preserve that uncertainty rather than pretend the intended interpretation is certain.

## Recommendation

The generic semantic rule is safe enough to keep as the candidate reference prompt for the other questions. Before using it for a full run, add one generic provenance requirement: whenever Search was used, attach URL citations to the central answer and evidence claims. Then rerun Q10 as a small verification, or rerun the canary if strict 50/50 citation coverage is required.

## Targeted Q10 citation pass

A targeted follow-up added the generic inline-citation requirement and reran Q10. It completed at stage 2 with 20 searches and 9 URL citation annotations. The answer was well-supported, but its canonical interpretation shifted again to **herbs / subshrubs / herbaceous vegetation**.

Across the grounded runs, Q10 produced several defensible interpretations: wetland/riparian vegetation, deciduous perennial plants, perennial flowering plants, and herbs/subshrubs. This is evidence that the wording does not identify one reliable gold dimension. The reference record should therefore preserve multiple supported interpretations and avoid marking alternatives as unacceptable unless the wording or evidence clearly rules them out.

Targeted cache: `artifacts/muse_q10_reference_generic_citations.json`

Artifacts:

- Generic cache: `artifacts/muse_canary_50_reference_generic.json`
- Generic report: `reports/muse_canary_50_reference_generic_2026-08-22.md`
- Earlier comparison: `reports/muse_canary_50_prompt_comparison_2026-08-22.md`
