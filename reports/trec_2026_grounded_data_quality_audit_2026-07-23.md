# Grounded reference data-quality audit — 2026-07-23

## Verdict

The completed corpus is mechanically usable as a reference-answer pivot: all 7,759 expected IDs are present exactly once, every record has a matching private evidence record, and the deterministic integrity gate passes. This is not a factual-accuracy certification; source entailment and answer quality remain unreviewed at corpus scale.

For practical use, the answer is **usable with safeguards, not gold**. The completed 7,759-question run has not been manually quality-reviewed question by question. The strongest available semantic audit is the blinded 400-question, four-model review: Gemini was top-tier on 329/400 (82.3%) and had 28/400 (7.0%) major-or-fatal errors, versus Perplexity at 62.3% top-tier and 21.5% major-or-fatal errors. That stress test supports Gemini as the best default reference among the tested systems, but it does not establish 92% corpus accuracy or eliminate hard-case failures.

Recommended use: treat Gemini as a provisional pivot for pairwise judging; retain Perplexity and its evidence beside it; escalate uncertain, non-search, time-sensitive, and material-disagreement cases. Do not silently treat the Gemini answer as the correct label when it conflicts with a candidate or when the answer is marked uncertain.

## What “uncertain” means

An uncertain response is not necessarily wrong. It means the model explicitly reported that the question or evidence did not support an unqualified answer:

- `ambiguous` (473): multiple plausible interpretations or an underspecified question.
- `insufficient_evidence` (56): the retrieved/available evidence was not enough to support the central claim.
- `conflicting_evidence` (0): the model reported authoritative evidence in conflict; none were labeled this way.

The status is a routing signal for review, not a calibrated probability of correctness. Conversely, `answered` means the model presented an answer; it is not a guarantee that the answer is true.

## Checks passed

- 7,759 expected questions → 7,759 cache records and 7,759 evidence records.
- Zero missing IDs, extra IDs, duplicate IDs, unresolved failures, empty answers, invalid statuses, abnormal finish reasons, or cache/evidence mismatches.
- All 7,759 generations ended with `finish_reason=stop`.
- Search was invoked for 6,274 responses (80.86%); 1,485 responses (19.14%) did not invoke Search.
- Every Search-using response has captured source metadata; 28,278 source URLs were retained privately.
- Search metadata is internally consistent: no Search-without-sources, no sources-without-Search, and no query-count contradictions.
- No duplicate request IDs remain in the final cache.
- Conservative projected generation cost is $72.96, below the $150 gate.

## Findings requiring attention

### 1. Exact duplicate question text (medium)

The frozen set contains one exact duplicate question text, `what triggered the civil war`, under IDs `850224` and `Q1462`. The cache is keyed by question ID, so both IDs were eligible to receive separate generations. This is not a record-integrity failure, but it is duplicated work and can create two nominally independent references for the same question. There are 18 additional normalized near-duplicate groups; these are semantically similar rather than byte-identical and need not be collapsed automatically.

### 2. Explicit uncertainty is concentrated but not exhaustive (medium)

There are 529 non-`answered` statuses (6.82%): 473 `ambiguous`, 56 `insufficient_evidence`, and 0 `conflicting_evidence`. The zero `conflicting_evidence` count should not be interpreted as proof that retrieved sources never disagree; it means the model did not label any response that way. The 529 cases should be the first adjudication set.

### 3. Non-search responses need stratified review (medium)

Grounding was enabled, but 1,485 questions did not trigger Search. Some are likely stable factoids or questions answerable from model knowledge; others may represent missed retrieval. The current data proves only that Search was not invoked, not that the answer is correct. Review a stratified sample by question type, uncertainty status, and difficulty before changing the prompt or forcing Search on every question.

### 4. Evidence URLs are provider redirect URLs (low/medium)

All retained URLs are Google grounding redirect URLs (`vertexaisearch.cloud.google.com`). This is useful private provenance, but it is less convenient for independent auditing than canonical destination URLs. Preserve these records; if later adjudication needs stable links, resolve and store canonical URLs in a separate private derivative with retrieval timestamps.

### 5. Short answers are not automatically failures (low)

Nineteen answers are under 40 characters. Spot inspection shows these correspond mostly to factoid questions whose requested answer is a name, place, or number. They should be sampled, not rejected by a blanket length rule.

## Runtime/code risks discovered separately

- The final cache cannot prove whether an overwritten record was generated more than once; request IDs only identify the surviving attempt.
- There is no process-level lock, so concurrent runners could race in a future run.
- Checkpointing every five records means an abrupt termination can repeat a small in-flight batch on resume.
- The cache does not store a question-text hash, so a changed question under an existing ID could reuse stale output in a future run.

These are future-run safeguards, not observed corruption in this completed corpus.

## Recommended next step

Use the 529 uncertain records plus a stratified sample of the 1,485 non-search records for source-level adjudication. Then use the resulting references for blinded pairwise candidate judging. Do not treat the full 7,759 answers as validated ground truth until that review establishes how often the central answer is actually supported.

## Evidence files

- Deterministic gate: `reports/trec_2026_gemini_grounded_full_audit.json`
- Sanitized full report: `reports/trec_2026_gemini_grounded_full_2026-07-21.md`
- Sanitized exception ledger: `reports/trec_2026_gemini_grounded_full_exceptions.tsv`
- Private cache and evidence: `artifacts/reference_answer_validation/grounding/` and `artifacts/reference_answer_validation/private/` (not included in public/sanitized reports)
