# Pointwise Judge v2 Design

Date: 2026-07-27

## Objective

Improve the stability of secondary pointwise-judge fields without reducing the existing final-label quality, then run a paid canary comparing DeepSeek V4 Flash, GPT-5.4 nano, and hosted Qwen3.6-35B-A3B in direct and thinking modes.

The first canary uses the same 10 frozen cases for all four configurations: 40 physical model calls before repair attempts. The experiment stops after this canary for user review. Across all later work, no more than 100 physical paid calls may be made without new authorization.

## Scope

This change includes:

- a versioned v2 prompt;
- a strict Pydantic v2 decision model;
- a committed JSON Schema snapshot generated from that model;
- cross-field invariant validation;
- transport-schema compatibility with the local Pydantic model remaining authoritative;
- an identity-separated v2 output directory and spend ledger;
- one 10-case run each for DeepSeek V4 Flash, GPT-5.4 nano, hosted Qwen3.6-35B-A3B direct, and hosted Qwen3.6-35B-A3B thinking;
- a comparison against the already locked advisor and Claude pass-2 labels.

This change does not include the 500-case bake-off, more than 100 paid calls, a new evidence field, chain-of-thought collection, new label or error-flag values, changed score ranges, or few-shot examples copied from the smoke cases.

## Decision Contract

Keep the eight existing output fields:

- `label`: `good | partial | incorrect | uncertain`;
- `correctness`: strict integer `0..4`;
- `completeness`: strict integer `0..2`;
- `relevance`: strict integer `0..2`;
- `error_flags`: unique list of at most three existing flags;
- `reference_status`: `sound | incomplete | suspect | uncertain`;
- `confidence`: `high | medium | low`;
- `rationale`: string of 60–600 characters.

No separate `evidence`, `reasoning`, or chain-of-thought field is added. When flags are non-empty, the prompt asks the rationale to ground its decisive defect in a short verbatim candidate quote. Quote fidelity is audited against the candidate text outside Pydantic because candidate context is not part of the output object.

## Pydantic Boundary

Add a strict Pydantic v2 `BaseModel` with `ConfigDict(extra="forbid", strict=True)`, `Literal` enums, constrained strict integers, a unique list limited to three flags, and a constrained rationale string.

The Pydantic model is the semantic source of truth. Its normalized `model_json_schema()` output is committed as the v2 schema snapshot. A test requires regenerated normalized schema to equal the committed snapshot, preventing model/schema drift. Runtime receipts bind the exact prompt and schema hashes.

The OpenRouter transport schema may omit only provider-incompatible validation keywords that are explicitly tested and receipt-bound. The complete Pydantic model validates every returned decision locally and remains authoritative.

## Cross-Field Invariants

Pydantic model-level validation enforces:

1. `good` requires and is required by correctness `4`, completeness `2`, relevance `2`, and no material-error flags. `reference_conflict` is allowed when the candidate correctly exposes a provisional-reference defect.
2. Correctness `4` excludes entity, number, date, negation, unsupported-specificity, and refusal flags.
3. Correctness `3` requires `partial`.
4. Correctness `0` or `1` requires `incorrect` or `uncertain`.
5. `incorrect` requires at least one error flag.
6. `refusal` requires correctness `0`, completeness `0`, relevance `0`, and `incorrect`, except that an unjudgeable item may be `uncertain`.
7. `uncertain` requires correctness `0`, completeness `0`, and either `ambiguity` or reference status `suspect/uncertain`.
8. `reference_conflict` requires reference status `incomplete` or `suspect`.
9. `reference_conflict` may not coexist with reference status `sound`.

Question-level semantics that cannot be derived from output alone—such as whether `ambiguity` actually belongs to the question—remain prompt rules and audit checks.

## Prompt v2

The v2 prompt follows the Opus recommendation with these sections:

1. Evidence surface: only question, provisional reference, and candidate are visible; unseen retrieval context is neither credited nor penalized by itself.
2. Truncation: judge only present text; truncation affects completeness, is not refusal, and is not independently a factual error.
3. Decision order: run an item-judgeability gate before judging the candidate.
4. Score anchors: define all correctness, completeness, and relevance values; relevance measures responsiveness independently of truth.
5. Flag boundaries: give positive and negative rules for all eight flags.
6. Reference status: assess the provisional reference only, never candidate completeness.
7. Confidence: confidence in the decision record, allowing high-confidence `uncertain` decisions.
8. Consistency checks: restate the machine-enforced invariants.
9. Rationale fidelity: do not invent, strengthen, reverse, or misquote claims.

The missing-ad case should resolve to `uncertain`; the leading “No,” private-university case should remain `good` with no negation flag. These expectations are regression assertions, not few-shot prompt examples.

## Canary Execution

Run the same frozen 10 cases through:

- DeepSeek V4 Flash, direct/non-thinking, pinned to DeepInfra;
- GPT-5.4 nano, default non-thinking, pinned to OpenAI.
- Qwen3.6-35B-A3B, direct/non-thinking, pinned to AkashML;
- Qwen3.6-35B-A3B, thinking enabled, pinned to the same AkashML endpoint.

The Qwen pair must differ only in the reasoning configuration. Their output identities, receipts, diagnostics, and resumable run paths are separate even though the provider and model ID are identical.

Use a fresh v2 output identity. Carry forward the existing conservative spend only if the same global cap ledger is reused; otherwise set a new explicit canary cap no higher than $0.10. Every physical attempt reserves worst-case cost before network access. Repairs and retries count toward the 100-call ceiling.

The canary stops immediately if schema compilation, provider routing, Pydantic validation, invariant validation, private checkpointing, or budget accounting fails. No larger run begins automatically.

## Acceptance Criteria

Hard gates for each model:

- 10/10 terminal records;
- zero invalid JSON after allowed repair;
- zero Pydantic or invariant violations in terminal decisions;
- zero provider/model identity violations;
- all private artifacts mode `0600` and directories mode `0700`;
- stable cases retain their previous `good` labels;
- missing-ad case is `uncertain`;
- leading “No,” case is `good` without `negation_error`;
- no rationale invents a candidate/reference claim during manual audit of non-good cases.

Comparison targets, reported rather than silently enforced:

- reviewer pass-2 label agreement at least 9/10 per model;
- relevance exact agreement at least 8/10;
- reference-status exact agreement at least 8/10;
- error-flag mean Jaccard at least 0.75;
- cross-model label agreement at least 9/10;
- zero repair attempts preferred;
- cost and latency recorded separately.

## Evidence and Handoff

Produce a private machine-readable comparison and a sanitized Markdown report containing schema validity, invariant validity, label/score/flag agreement, rationale audit, cost, latency, and exact differences from v1. Present the 40-call result to the user before changing the prompt again or expanding toward the 100-call ceiling.
