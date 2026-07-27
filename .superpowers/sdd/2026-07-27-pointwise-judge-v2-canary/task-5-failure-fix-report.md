# Task 5 canary failure fix report

## Scope

Applied the smallest local-only fixes for the preserved first v2 canary failure. No network or paid calls were made, and the original canary artifacts remain in place.

## Root-cause evidence

- `ModelSpec.worst_case_usd`, the OpenRouter payload, provider usage validation, and the runtime receipt all used the global 512-token output ceiling. The preserved Qwen direct response ended with `finish_reason=length` at that ceiling.
- `parse_judge_decision(..., version="v2")` replaced every Pydantic validation detail with `v2 decision violates the strict contract`. `run_cases` retained that generic text in the attempt record but passed only the malformed output to the repair prompt. Cached partial records likewise retained no derived repair error in `_ReusableRecord`.

## RED evidence

- Hosted budget tests failed with Qwen direct `512 != 1024` and worst-case reservation `0.001212 != 0.001724`.
- The v2 invariant test failed because it received only `v2 decision violates the strict contract`.
- The live runner repair test failed because the repair request had no `VALIDATION ERROR` section.

## Fix

- Added a per-configuration `max_output_tokens` field with a 512 default. Qwen direct uses 1,024 and Qwen thinking uses 4,096. Payloads, provider usage bounds, runtime receipts, and worst-case spend reservations now all consume the same field.
- Converted Pydantic validation failures to one concise first-error summary using `include_input=False`, `include_context=False`, and `include_url=False`. Model-level failures are identified as invariant violations without echoing the decision or rationale.
- Threaded the caught validation error into both live and cached repair requests. Existing run artifact keys and the v1 protocol remain unchanged.

## GREEN evidence

- Targeted regression tests: 5 passed.
- Pointwise/hosted suites: `279 passed in 4.79s`.
- Ruff: all checks passed on the six changed Python files.
- Ruff format: all six changed Python files are formatted.
- `git diff --check`: passed.
- Preserved artifact state re-read locally: 25 attempts, $0.008190882 spend, and 10 DeepSeek / 10 GPT / 2 Qwen-direct records.

## Concerns and next action

- The local tests prove configuration binding and repair-message behavior, not that 1,024/4,096 tokens are sufficient in live service. A fresh paid canary is still required.
- Use a new output directory for that canary. Reusing the failed directory would preserve the GPT terminal failure and conflicts with the Qwen runtime receipt that correctly recorded the old 512-token ceiling.

## Round 1 review remediation

### RED evidence

- Exact live and cached v1 repair-message assertions failed because the validation detail block changed the historical request bytes.
- An arbitrary extra key appeared in the v2 error as its Pydantic location, and an arbitrary duplicate key appeared in the duplicate-key error.
- The four Qwen provider-boundary tests were characterization coverage for an already-correct round-one consumer, so they were green on first execution: direct accepted 1,024 and rejected 1,025; thinking accepted 4,096 and rejected 4,097.

### Fixes

- `run_cases` forwards the derived validation detail only when `version == "v2"`; complete live and cached v1 repair messages are byte-for-byte locked by tests.
- V2 Pydantic locations allow only top-level schema field names followed by non-negative integer list indices. Every other location is rendered as `unknown field`.
- Duplicate JSON keys use a fixed v2 error without the key name. V1 retains its prior duplicate-key message.
- Live and cached v2 repair tests now cover both paid failures: `reference_conflict` with a sound reference and `ambiguity` on a judgeable non-uncertain item.
- Integration tests confirm private extra/duplicate field names do not enter the persisted `error` field or the repair prompt's `VALIDATION ERROR` section. The raw malformed output remains preserved separately by design.

### GREEN evidence

- Pointwise/hosted suites: `291 passed in 4.62s`.
- Ruff and Ruff format checks passed on all five changed Python files.
- `git diff --check` passed.
- No network calls were made. The preserved first canary remains at 25 calls and $0.008190882.

### Next paid command limits

The next fresh-directory invocation must explicitly include both remaining cumulative limits:

```text
--cap-usd 0.091809118 --call-cap 75
```

The spend reservation can stop the canary before all configurations finish; that is the intended fail-safe behavior, not authorization to raise either limit.
