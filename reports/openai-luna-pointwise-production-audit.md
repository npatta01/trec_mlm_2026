# OpenAI Luna pointwise production audit

**Assessment: Ready**

## Coverage and identity

- 182,764 records; 182,764 unique record IDs.
- 182,764 unique question/candidate pairs across 6,816 questions and 150 candidates.
- Final artifact SHA-256: `033edf940d31299340ebb3283774fbf0f77e4fe75d834f67e7d36f9caca65436`.
- Inventory SHA-256: `16ca3ec4743c333015fe3c86dcf977f5447183be9e4bbcfdb075ee4f31d72ab2`.

## Execution integrity

- Controller state clean: True.
- Kinds: {"cached": 492, "deterministic": 734, "provider": 181538}.
- Provider invariant failures: 0; deterministic invariant failures: 0.
- Estimated Batch cost: $31.82.

## Verdict distribution

| Verdict | Count | Rate |
|---|---:|---:|
| correct | 114,337 | 62.56% |
| incorrect | 10,419 | 5.70% |
| nonanswer | 23,507 | 12.86% |
| not_established | 16,755 | 9.17% |
| partial | 17,746 | 9.71% |

### Curated downstream distribution

| Verdict | Count | Rate |
|---|---:|---:|
| correct | 114,319 | 62.55% |
| incorrect | 10,402 | 5.69% |
| nonanswer | 23,539 | 12.88% |
| not_established | 16,746 | 9.16% |
| partial | 17,758 | 9.72% |

## Exact-input consistency

- Non-deterministic duplicate groups/rows: 1,494/7,280.
- Supported/unsupported-inconsistent groups, raw → curated: 25 → 5.
- Pairwise supported/unsupported disagreements, raw → curated: 303 → 171 of 58,909 duplicate pairs.
- Twenty clearly adjudicated groups were canonicalized. Five ambiguous or broken-question groups remain explicitly unchanged.

## Semantic spot check

- 103 records were reviewed from a deterministic, verdict-stratified sample (20 per raw verdict plus explanation-integrity controls).
- Two clear supported/unsupported errors outside duplicate groups were corrected in the curated artifact.
- Other clear reviewer disagreements concerned the five-way debugging subtype while remaining in the same supported/unsupported bucket.

## Explanation quality

- Length p50/p95/p99/max: 179/249/282/400 characters.
- At 400-character schema maximum: 3.
- Control-character records in raw artifact: 3.
- Applied explanation repairs: 4; explicit audited verdict changes: 55; remaining control-character records: 0.

## Interpretation boundary

This audit establishes complete, schema-valid, identity-bound execution and a reviewed downstream label artifact. The primary automated label is `supported = verdict in {correct, partial}`; the three unsupported subtypes are debugging signals. The labels measure agreement with the supplied Gemini evaluator evidence, not absolute factual truth; the Gemini evidence can itself be incomplete or wrong.
