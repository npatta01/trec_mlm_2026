# Reference-Grounded Pointwise Judge v8

## Goal

Judge whether a candidate response is supported by the supplied Gemini grounded
reference. Correctness is the routing decision. A response may be incomplete
and still be accepted when everything it does say is supported.

This contract measures **agreement with the provisional grounded reference**,
not independently proven world truth. Reference quality is audited once per
question from the recorded Gemini grounding metadata, outside the candidate
judge.

## Considered designs

1. **Three fields: correctness, coverage, rationale (selected).** Keeps the
   automated decision explicit and preserves one useful debugging dimension.
2. Add candidate-level `reference_status`. Rejected because the same reference
   is reused across many candidates and the field previously leaked candidate
   content into the reference judgment.
3. Emit only `yes | no | unknown`. Rejected because it collapses partially
   supported answers into a single failure bucket and gives too little evidence
   for auditing surprising results.

## Model-facing contract

The model emits exactly one JSON object with three fields:

```json
{
  "correctness": "correct",
  "coverage": "partial",
  "rationale": "The claims made are supported, but the response omits other requested parts."
}
```

### Correctness

Judge only the material claims the candidate actually makes:

- `correct`: every material claim is supported by the provisional reference;
  no material contradiction is present. Omission does not lower correctness.
- `mixed`: the response contains at least one useful supported claim and at
  least one materially contradicted claim.
- `wrong`: the central answer is contradicted by the reference, or the response
  contains no useful supported answer content.
- `unknown`: the reference neither supports nor contradicts enough of the
  material candidate claims to judge them.

Reference silence is `unknown`, not `wrong`. Concision and omitted details do
not lower correctness.

### Coverage

Judge how much of the question the candidate attempts, independently of truth:

- `complete`: all material parts of the question are answered.
- `partial`: some material parts are answered and others are omitted.
- `missing`: almost none of the requested answer is present.
- `unknown`: coverage cannot be determined.

Answer length alone is never evidence of coverage. A short direct answer can be
complete, and a long answer can be partial.

### Rationale

One short sentence, at most 240 characters, explaining the decisive comparison
between candidate and reference. It must not introduce new facts.

## Deterministic local handling

Pydantic validates the three fields strictly. The normalizer may extract one
JSON object from surrounding text, normalize enum casing/whitespace, drop
unknown keys, and truncate an overlong rationale. Every mutation is recorded in
`repair_codes`; the raw output is retained only for repaired or unscorable
records.

Derived fields are local and are not requested from the model:

- `acceptable = (correctness == "correct")`
- `needs_review = (correctness in {"mixed", "unknown"})`
- malformed outputs become `parse_status = "unscorable"`; they are never
  silently converted into a semantic judgment.

There is no semantic retry and no provider fallback. A transport failure may be
retried once against the same provider with the identical request.

## Reference audit

Each unique Gemini reference is classified once using the frozen grounding
artifact, not once per candidate:

- `search_grounded`: Gemini recorded a Google Search grounding invocation.
- `generated_without_search`: Gemini answered without recorded Search use.
- `unanswered_or_failed`: the grounding generation did not produce an answer.

The audit tier travels beside candidate judgments for downstream filtering. It
does not change the model-facing correctness labels. `correct` therefore means
"supported by this provisional reference" in every tier.

## Validation run

Run the same targeted 50 candidates twice using DeepSeek V4 Flash direct on
DeepInfra, temperature 0, JSON mode, no provider fallback, and the same frozen
prompt/schema hashes. This is 100 calls total and is expected to cost roughly
$0.02–$0.03 based on the prior v7 run.

Report:

- transport, Pydantic validation, repair, and unscorable counts;
- label distributions and total cost;
- correctness, coverage, and exact repeat agreement;
- every disagreement with question, reference, candidate, both outputs, and a
  manual semantic assessment;
- representative stable `correct`, `mixed`, `wrong`, and `unknown` cases;
- comparison with v7's 41/50 correctness repeat agreement.

The repeat test measures consistency, not factual accuracy. The manual audit
checks obvious rubric violations and surprising cases, but it is not a human
gold set. Do not scale to the full corpus from schema success alone.

## Compatibility and privacy

v1 through v7 prompt and schema files remain byte-for-byte unchanged. v8 is
additive. Provider artifacts stay private (mode 0600), and no token or secret is
written into prompts, results, logs, or source control.
