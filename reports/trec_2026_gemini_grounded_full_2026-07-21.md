# Gemini 3.6 grounded references: full-corpus audit

## Technical summary

- **The grounded corpus is complete:** 7,759 records for 7,759 unique IDs, with zero missing IDs and 0 unresolved failures.
- **Provenance is retained privately:** Search ran on 6,274 responses; all Search-using records passed the source-URL capture check. The report contains no URLs or answer passages.
- **Uncertainty remains explicit:** 529 responses are non-`answered`; 1,485 did not invoke Search and 1 matched the refusal detector. These sets may overlap.
- **Spend stayed within the gate:** estimated standard total is $61.35; the deliberately doubled-token-price upper projection is $72.96, below $150.

## Scope and validation contract

The population is the frozen TREC Million LLMs discovery set: exactly 7,759 question IDs. A valid record has a non-empty clean answer, one of four structured uncertainty statuses, a short reason, a normal `stop` finish, and a matching private provenance record. Search-using responses must retain at least one returned source URL. The full-ID audit is an integrity and provenance check; it does not establish factual correctness for every answer.

## Semantic-quality interpretation

The full 7,759-question corpus has not been manually quality-reviewed question by question, so these references are usable as provisional pivots, not gold labels. In the blinded 400-question four-model stress test, Gemini was top-tier on 329/400 (82.3%) and had 28/400 (7.0%) major-or-fatal errors; Perplexity was top-tier on 249/400 (62.3%) with 86/400 (21.5%) serious errors. Those results support Gemini as the default reference generator, but do not establish corpus-wide accuracy. Retain Perplexity evidence and escalate uncertain, non-search, time-sensitive, and materially disagreeing cases.

`ambiguous` means multiple plausible interpretations; `insufficient_evidence` means the available evidence did not support the central claim; and `conflicting_evidence` means the model reported authoritative sources in conflict. These statuses are review-routing signals, not calibrated error probabilities. `answered` also does not guarantee truth.

## The 50-question canary unlocked the full run

The canary required 50/50 unique completions, valid statuses, non-empty and non-truncated answers, normal finishes, matching evidence records, URLs for Search-using responses, and a conservative projected cost below $150. It passed with 39 Search invocations, 11 manually inspected non-search responses, zero failures, and a $73.40 conservative projection. The full run was resumable and retried isolated format failures without duplicating valid records.

## Completeness is not correctness

Grounding provenance is insurance for later adjudication, not proof that each source entails each central claim. Organizer labels may also contain purposeful noise. The uncertain and conflicting subset should therefore be investigated first with the retained sources; a second blanket generation run is not justified by this audit. The refusal detector is deliberately narrow and may miss softer non-answers.

## Next actions

1. Review all uncertain and conflicting responses against their private support mappings.
2. Use the validated references as pivots for blinded, order-randomized candidate pairwise judgments with ties and reversal checks.
3. Analyze whether non-search responses are concentrated by question cluster or ambiguity type before changing the grounding prompt.
4. Rotate the Gemini key that appeared in chat after the run is archived.

## Further questions

- How many uncertain answers become usable after source-level adjudication?
- Do the retained support passages cover central claims or only peripheral facts?
- Does reference-guided pairwise ranking improve recall for the 94 sparse zero-nDCG queries without overfitting organizer noise?
