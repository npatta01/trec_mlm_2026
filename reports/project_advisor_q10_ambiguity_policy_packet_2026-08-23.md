# Advisory question

For grounded reference answers used to evaluate other LLMs, should we ask a separate ambiguity question, and are we demanding too much by expecting the answer generator to volunteer ambiguity? For Q10 — “What type of vegetation does Lythrum and Erythrina have in common?” — should the reference generator choose one likely lane or preserve multiple readings?

## Desired outcome

Choose a generic, low-cost policy that produces a useful canonical reference answer without unfairly penalizing candidate LLMs for following the likely interpretation of an underspecified question.

## User constraints and corrections

- The reference answer will be used to judge other LLM answers.
- The user prefers the existing two-call flow and does not want three independent generations for every question.
- Search and output length should not have arbitrary application caps.
- It is acceptable for a response to say it is unsure; the user is concerned that the current ambiguity policy may be too demanding.

## Current project state

- Repository and authoritative worktree: this repository root
- Current task artifacts are local, private, and exploratory; no full-dataset run has been started for the new uncertainty variant.
- The relevant Q10 probes use public botanical facts and no sensitive data.

## Evidence

### E1 — Muse direct reference output
- Status: Verified
- Source: `artifacts/muse_uncertainty_sample_2026-08-22.json`
- Relevance: Tests the generic two-line status format in the existing two-call flow.
- Excerpt or compact result: Q10 returned `Reference status: ANSWERED`, selected herbaceous/herbaceous perennial vegetation, and caveated that Erythrina is predominantly woody but includes herbaceous species. It did not flag material ambiguity.

### E2 — Muse ambiguity-branch pilot
- Status: Verified
- Source: `artifacts/muse_ambiguity_pilot_v2_2026-08-22.json`, `artifacts/muse_ambiguity_pilot_v3_2026-08-22.json`
- Relevance: Tests whether stronger branch instructions stabilize interpretation behavior.
- Excerpt or compact result: Q10 varied between SINGLE and MULTIPLE across repeats; v3 improved control behavior but did not make Q10 stable. This indicates prompt-only ambiguity detection is not reliable.

### E3 — Gemini 3.7 Flash via OpenRouter
- Status: Verified
- Source: `artifacts/q10_gemini_openrouter_2026-08-22.json`
- Relevance: Independent model comparison with web search.
- Excerpt or compact result: Gemini explicitly said the wording was materially ambiguous and listed growth-form, habitat, and taxonomic readings. It also included the herbaceous overlap, but the answer was too broad for direct grading.

### E4 — Luna via OpenRouter
- Status: Verified
- Source: `artifacts/q10_luna_openrouter_2026-08-22.json`
- Relevance: Independent model comparison with web search.
- Excerpt or compact result: Luna explicitly said the wording was materially ambiguous, but proposed “flowering plants” as the safest broad answer and mentioned shrubs as a possible narrower answer. That is factually broad but not a useful canonical reference answer.

### E5 — Botanical source evidence
- Status: Verified
- Source: Royal Botanic Gardens, Kew Plants of the World Online profiles for `Lythrum` and `Erythrina`.
- Relevance: Establishes why the question can support both a narrow growth-form answer and a caveated genus-level interpretation.
- Excerpt or compact result: Kew describes Lythrum as perennial or annual herbs, rarely subshrubby; Erythrina as trees or shrubs, rarely herbs, with other treatments describing deciduous trees/shrubs and occasional perennial herbs.

### E6 — Competition-model evidence boundary
- Status: Unknown
- Source: `reports/trec_2026_four_model_reference_review_2026-07-21.md` and `docs/results/2026-07-13-expert-label-ranking-study.md`
- Relevance: The repository contains aggregate competition-model results, but not a complete, directly comparable Q10 answer table for every model.
- Excerpt or compact result: Aggregate reports compare Gemini, Perplexity, Qwen, and Gemma overall. The local Q10-specific model outputs available for this decision are Muse, Gemini, and Luna; do not generalize the Q10 behavior to every competition model without a matched probe.

## Candidate choices or proposed plan

- Separate ambiguity pass: ask “Is the question materially ambiguous for grading? Identify at most two plausible interpretations and whether a reference should preserve them.” Use this as metadata, not as the canonical answer itself.
- One-pass answer only: require the model to choose the most likely interpretation and add a caveat only when needed.
- Hybrid: keep the normal two-call answer flow, add a lightweight separate ambiguity classification field or pass, and use a conservative policy only when ambiguity changes the acceptance set.

## Known unknowns

- We have not run the separate ambiguity question with the same exact wording across Muse, Gemini, Luna, Qwen, Gemma, and Perplexity.
- The intended gold answer for Q10 is not supplied as an external answer key; the likely canonical lane is inferred from botanical evidence and prior reference-generation outputs.
- It is unknown how often a separate ambiguity pass would identify genuine grading ambiguity versus harmless semantic alternatives.

## Scope boundary

Read-only review. No implementation, full-dataset run, or external publication is authorized by this packet.
