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
