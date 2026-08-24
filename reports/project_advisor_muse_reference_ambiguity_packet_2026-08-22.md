# Advisory question

For a web-grounded reference-answer generator used to judge other LLM answers, what should happen when the question is materially ambiguous: should the model choose one interpretation, or return multiple supported interpretations? Assess the tradeoff between fair grading, usefulness, verbosity, and false ambiguity. Use Q10 as the stress case without designing a domain-specific rule.

## Desired outcome

Choose a generic reference-record policy and prompt behavior that gives fair acceptance guidance for ambiguous questions while keeping unambiguous questions direct and avoiding arbitrary output/search limits.

## User constraints and corrections

- The output is a reference answer used to judge other LLM responses, not merely a conversational answer.
- A single unsupported interpretation should not unfairly mark valid LLM answers wrong.
- Long answers and extra tokens are acceptable.
- Search grounding is required; multiple searches are allowed without an arbitrary cap.
- Wikipedia and other secondary sources are acceptable when relevant and non-fabricated.
- The prompt must remain generic and must not contain rules specific to vegetation, Q10, or another single domain.

## Current project state

- Repository and authoritative worktree: this repository root
- Branch and HEAD: detached at `9bcbd6c01ad6002a9006112adb2090a5d8ae1183`
- Dirty state: new local adapter, tests, canary caches, reports, and plan artifacts; no tracked files modified.
- Active workstreams: Muse two-call reference generation; original, structured-reference, generic-reference, and targeted citation-pass artifacts are kept separate.

## Evidence

### E1 — Generic 50-question canary mechanics
- Status: Verified
- Source: `reports/muse_canary_50_generic_prompt_validation_2026-08-22.md:8-16`
- Relevance: The generic prompt completed the same 50 questions with 50/50 stage-2 finals, zero failures/fallbacks, and all five reference sections. Citation annotations appeared on 49/50.
- Excerpt or compact result: `50/50 stage-2 finals; 0 hard failures; 0 stage-1 fallbacks; 50/50 five sections; 49/50 inline URL citations.`

### E2 — No clear collateral answer regression
- Status: Verified
- Source: `reports/muse_canary_50_generic_prompt_validation_2026-08-22.md:33-37`
- Relevance: A review of all 50 canonical answers found the other 49 retained the same substantive answer content; changes were mainly scope, aliases, dates, or caveats.
- Excerpt or compact result: `No clear collateral semantic regression was found` in the other 49 records.

### E3 — Q10 interpretation instability
- Status: Verified
- Source: `reports/muse_canary_50_generic_prompt_validation_2026-08-22.md:43-49`
- Relevance: The same question produced multiple grounded interpretations across runs, showing that forcing a single canonical answer may be less reliable than representing ambiguity.
- Excerpt or compact result: Q10 yielded wetland/riparian vegetation, deciduous perennial plants, perennial flowering plants, and herbs/subshrubs across grounded runs. The targeted citation pass completed with 20 searches and 9 URL citation annotations.

### E4 — Generic prompt already asks for semantic and scope checks
- Status: Verified
- Source: `trec_mlm/muse_reference_generation.py:86-119`
- Relevance: The current candidate prompt is generic and explicitly asks the model to identify the requested relationship/attribute, avoid substitution, check scope, and preserve material ambiguity.
- Excerpt or compact result: `Do not substitute a related fact`; `Do not generalize from one example to an entire group unless the evidence supports that scope`; `preserve [ambiguity] rather than forced into one answer.`

### E5 — Reference-record purpose
- Status: Verified
- Source: `trec_mlm/muse_reference_generation.py:48-68`
- Relevance: The earlier reference prompt explicitly defines canonical answer, acceptable variants, necessary caveats, evidence, and confidence for grading other LLM answers.
- Excerpt or compact result: The record is intended to distinguish acceptable variants from qualifications that affect fair marking.

### E6 — Evaluation risk
- Status: Inference
- Source: Derived from E3 and the user's stated use case.
- Relevance: A single answer chosen from an under-specified question can create false negatives by treating a defensible alternative as wrong. Returning every related fact can instead create false positives unless alternatives are tied to explicit interpretations and scope.
- Excerpt or compact result: The policy must separate `materially ambiguous` from merely `detailed` or `open-ended` questions.

## Candidate choices or proposed plan

- Choice A: Always select one canonical interpretation. Simple, but risks unfair grading when wording does not select one interpretation.
- Choice B: Always return multiple answers. Fairer in some cases, but noisy and can make the reference too permissive on clear questions.
- Choice C: Conditional policy: return one direct answer for unambiguous questions; for materially ambiguous questions, return interpretation–answer pairs with evidence, scope, confidence, and explicit grading guidance. This is the current leading proposal.

## Known unknowns

- The canary does not contain a formal gold answer for Q10, so it cannot prove which interpretation the original question author intended.
- The 50-question review is manual and semantic; citation quality was not independently adjudicated URL by URL.
- It is unknown whether the downstream grader can consume structured multiple-interpretation records directly or needs a flattened acceptance set.

## Scope boundary

Read-only review. No implementation, full-dataset run, external messages, or public sharing is authorized by this packet.
