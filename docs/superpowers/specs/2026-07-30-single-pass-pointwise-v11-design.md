# Single-Pass Pointwise v11 Bakeoff Design

## Objective

Select one judge model and one frozen contract that can be used for every production response without a second-judge cascade. The judge compares a historical candidate response with later Gemini evaluator evidence. Correctness of the central answer matters more than completeness.

## Diagnosis

The v10 transport and schema are reliable, but its five-way direct classification encourages conservative undergrading. DeepSeek frequently treats hedges, omissions, unrelated tails, and evidence-silent peripheral details as reasons to lower a centrally supported response. The v11 contract must force central-answer decomposition before any verdict is derived.

## Contract

The model returns exactly:

```json
{
  "central_claim": "short statement",
  "answer_present": true,
  "supported_core": true,
  "material_conflict": false
}
```

Definitions:

- `central_claim`: at most 20 words summarizing the response's main answer to the question; an empty string only when no substantive answer is present.
- `answer_present`: the response substantively addresses the question and the intended subject or referent.
- `supported_core`: Gemini evidence supports the central claim or a separable central answer to a requested part.
- `material_conflict`: Gemini evidence explicitly contradicts the central claim or a claim necessary to it.

Ordered decision procedure:

1. Identify whether the response actually answers the question about the intended subject.
2. Extract its central claim while ignoring retrieval disclaimers, hedges, and unrelated tails.
3. Test explicit evidence-backed contradiction. Evidence silence is never contradiction.
4. Test evidence support for the central claim.
5. Emit the four fields only.

Incompleteness, brevity, verbosity, hedging, and peripheral evidence-silent details do not lower a supported central claim. A wrong referent is not an answer. No outside knowledge is allowed.

## Deterministic Derived Label

| answer_present | supported_core | material_conflict | Derived label |
|---|---|---|---|
| false | false | false | `nonanswer` |
| true | true | false | `correct` |
| true | true | true | `partial` |
| true | false | true | `incorrect` |
| true | false | false | `not_established` |

All other combinations are invalid and fail closed. Numeric scores, if needed downstream, are derived in code: correct=2, partial/not_established=1, incorrect/nonanswer=0.

## Generic Few-Shot Coverage

Examples cover structural patterns rather than dataset topics:

- incomplete but supported central answer;
- hedged but supported central answer;
- supported answer with an unrelated tail;
- supported core plus a genuinely contradicted essential claim;
- plausible but evidence-silent central claim;
- wrong referent;
- explicit nonanswer.

## Bakeoff

Use identical evaluator inputs, prompt, schema, temperature, and deterministic mapping for:

1. DeepSeek V4 Flash thinking;
2. GPT Nano;
3. Qwen thinking.

Run a 10-case transport/schema smoke test before the audited bakeoff. No provider or model fallback is permitted. Cache identities bind the model, prompt, schema, evidence transform, and record identifiers.

The bakeoff set is reconstructed from the 282 records already semantically reviewed: all 100 original pilot records, all 148 newly low-scored records, 30 new grade-2 controls, and four deterministic bare-IDK records. Existing audit reports are converted into one machine-readable adjudication manifest with provenance and validation that every correction is accounted for.

## Acceptance Gate

A model is eligible for the full run only if it achieves on the audited set:

- supported/correct precision at least 98%;
- supported/correct recall at least 95%;
- exact derived-label agreement at least 90%;
- no schema or provider failures;
- long-versus-short correct-recall gap no greater than five percentage points.

Because the existing 282 records cover only 13 questions and are enriched for difficult low-score cases, the winning model must then pass a fresh question-diverse confirmation set of 200–500 responses before the 7,000-response run. The full production run uses the single winning model for all eligible responses.

## Evidence and Privacy

- The candidate model never receives Gemini material; only the retrospective judge does.
- Model identity, response IDs, citations, URLs, source lists, and grounding metadata remain absent from provider messages.
- Bare-IDK responses are labeled deterministically without an API call.
- Results are reported separately by Gemini evidence tier; the judge cannot repair incorrect or incomplete evaluator evidence.
- Private artifacts remain mode 0600 inside mode-0700 directories; secrets are never persisted.

## Deliverables

- frozen v11 prompt, schema, Pydantic normalizer, and deterministic mapper;
- machine-readable 282-case adjudication manifest;
- three-model 10-case smoke comparison;
- full audited-set bakeoff for models with clean smoke results;
- metrics and recommendation selecting exactly one production judge, or a clear no-model-passed result.
