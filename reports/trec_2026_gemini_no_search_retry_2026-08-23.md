# Gemini Search retry of the 1,485 original no-Search references

## Outcome

The retry completed for the exact **1,485** question IDs on which the frozen
Gemini 3.6 Flash corpus had Search enabled but did not invoke it.

- **1,485/1,485** responses completed; no failures, missing records, empty
  answers, refusals, truncations, or abnormal finish reasons remain.
- **1,389/1,485 (93.5%)** invoked Google Search.
- **1,279/1,485 (86.1%)** are grounded `ANSWERED` references.
- **206/1,485 (13.9%)** were explicitly uncertain at generation time: 177
  `AMBIGUOUS` and 29 `INSUFFICIENT_EVIDENCE`. A subsequent review separated
  usable ambiguity from questions that genuinely lack enough context; see the
  uncertainty-routing section below.
- **96** uncertain responses did not invoke Search, generally because the
  question omitted the entity, event, country, timeframe, or antecedent needed
  to form a meaningful search. They remain skip records.
- **Zero** `ANSWERED` responses lack Search evidence. Three such edge cases were
  detected after the first full pass and automatically regenerated.

The old and retried answers are preserved separately. This run does not silently
replace the frozen reference corpus.

## Prompt and model

The run used the same `gemini-3.6-flash` model as the frozen corpus so the test
isolates the grounding behavior. The generic reference prompt now also requires
Google Search before answering every question, permits additional searches as
needed, forbids a fixed search count or arbitrary answer-length limit, prefers
authoritative sources when practical, and requires `INSUFFICIENT_EVIDENCE`
instead of a memory-only answer when Search cannot provide usable evidence.

The grounded prompt SHA-256 is
`b09dba18a3fb07e949d63bbd31ab826bbfd32ea4ae96c52478797076e3adfbe5`.

## Grounding and usage

- Search-using responses: **1,389**
- Searchless uncertain/skip responses: **96**
- Web Search queries recorded: **1,965**
- Average queries per Search-using response: **1.41**
- Maximum queries in one response: **7**
- Sources: **7,557**
- Grounded support mappings: **9,974**
- Input tokens: **566,831**
- Output tokens including thinking: **719,022**

At the current promotional Gemini 3.6 Flash rates of $0.75/M input tokens and
$3.75/M output tokens, successful recorded generations cost approximately
**$3.12** in model tokens. Six discarded attempts (three malformed envelopes and
three answered-without-Search responses) put the estimated actual token spend at
about **$3.13**. The run made **1,491** total grounded prompts, below the
`np-training` project's apparent 5,000-request monthly Search allowance, so the
estimated Search surcharge is **$0**. Provider billing can lag, so these are
usage-based estimates rather than a settled invoice.

The strict legacy audit reports `gate_passed: false` because its original rule
rejects every expected-but-unused Search call, including explicit uncertain
responses. This is not a generation failure: every final `ANSWERED` record has
Search evidence. Uncertain records require the separate routing policy below
rather than being treated automatically as either correct references or
failures.

## Subsequent uncertainty review and evaluation routing

All 206 uncertainty-labeled responses were reviewed for their intended use as
reference evidence when judging other LLM answers:

- **119** lack essential context and should be skipped for every candidate.
- **59** contain usable multiple interpretations. A candidate can receive full
  credit by answering any one supported interpretation coherently.
- **9** are genuinely debatable. Opposing conclusions can both receive full
  credit when their factual premises and reasoning are defensible.
- **5** are overcautious but answerable and can be retained as ordinary
  references.
- **14** need manual resolution before they enter a scoring denominator.

Together with the 1,279 grounded `ANSWERED` records, this yields **1,352
currently usable references**, **119 global skips**, and **14 manual holds**.
These counts supersede the earlier provisional rule to skip all 206 uncertain
records.

The reference is evidence, not an exact-answer string. Candidate answers do not
need to reproduce its wording, length, citations, or every interpretation. If a
candidate gives a plausible answer outside the recorded evidence, route the
case to `not_established` adjudication rather than scoring it wrong or changing
only that candidate's denominator.

## Fifty-question canary quality review

All 50 canary responses were compared with their frozen originals and reviewed
for factual correctness, ambiguity handling, grounding support, and regression.
Niche or changing claims were selectively checked on the web.

- Clean pass: **40**
- Minor issue: **10**
- Major issue: **0**
- Unverifiable: **0**
- Ungrounded `ANSWERED`: **0**

The ten minor flags were:

- `96727`: overstates Shakespeare's role in creating words and contains one
  malformed sentence.
- `975342`: imprecise phrasing about release after six years/in the seventh.
- `572f958ca23a5019007fc7b9`: loosely groups GaN HEMTs under MOSFETs.
- `932034`: gives a useful general mechanism but should retain an ambiguity
  caveat about which market crash is intended.
- `1000354`: several secondary historical claims are too broad or weakly
  supported.
- `57268be2dd62a815002e8902`: correct abstention but `AMBIGUOUS` is a better
  label than `INSUFFICIENT_EVIDENCE`.
- `5726eadbf1498d1400e8efd6`: some examples extend beyond the returned support.
- `Q2201`: treats range and codomain as equivalent rather than range as a subset
  of the codomain in general.
- `491082`: dates the NSDAP to the 1919 predecessor DAP rather than the 1920
  adoption of the NSDAP name.
- `5ae1f6cf5542994d89d5b2e4`: overcomplicates the likely intended
  `microcomputer` answer and generalizes beyond all Exidy products.

These are useful QA flags, not evidence of a central-answer failure rate: none
was rated major. Because this corpus will judge other models, downstream use
should preserve the old answer, the grounded retry, uncertainty status, and
source mappings rather than assuming every generated detail is gold.

## Artifacts

- Retry sample: `artifacts/reference_answer_validation/gemini_grounding_retry/gemini36_no_search_all_1485_sample.json`
- Public answer cache: `artifacts/reference_answer_validation/gemini_grounding_retry/gemini36_no_search_retry_all_1485.json`
- Private evidence ledger: `artifacts/reference_answer_validation/private/gemini36_no_search_retry_all_1485_evidence.json`
- Strict integrity audit: `artifacts/reference_answer_validation/gemini_grounding_retry/gemini36_no_search_retry_all_1485_audit.json`

The private evidence ledger is mode `0600`. No API key is serialized into the
artifacts.

The ignored answer and evidence artifacts are intentionally not committed as
raw provider output. Their hashes and custody metadata are recorded in
`reports/trec_2026_gemini_no_search_retry.artifact.json`.
