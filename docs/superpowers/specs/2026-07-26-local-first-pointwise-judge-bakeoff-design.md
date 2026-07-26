# Local-first pointwise judge bake-off

**Date:** 2026-07-26

**Status:** Approved in conversation; awaiting written-spec review before implementation

**Related issues:** #33 candidate-answer calibration set, #34 local judge benchmark, #31 response-quality pilot

## Objective

Select one practical local model for pointwise candidate-response judging before
spending money on hosted judges. Use a ten-record smoke test to expose prompt and
runtime problems, review that checkpoint with the user, and then use a frozen
100-record diagnostic set to compare viable local models. Only after choosing a
local winner will hosted API models be run on the same records.

This bake-off evaluates a task-specific grading workflow. General model
benchmarks can justify a shortlist, but they cannot establish judge quality for
the discovery questions, grounded Gemini references, and historical candidate
responses in this repository.

## Non-goals

- Do not run 500 records, the 100-to-200-question pilot in #31, or the full
  response corpus.
- Do not call hosted candidate judges before the local winner is selected.
- Do not treat Gemini, Advisor, Claude, or cross-model consensus as human gold.
- Do not claim that 100 purposively selected records estimate corpus-wide
  accuracy.
- Do not expose a local inference service beyond loopback.
- Do not test GLM-4.7-Flash, Nemotron-3-Nano, or Qwen3.5-122B-A10B locally.

## Inputs and custody

Each evaluation record contains exactly:

- a stable question ID and response ID;
- the original discovery question;
- the grounded Gemini response as a provisional reference;
- one historical candidate response; and
- diagnostic strata and selection reason.

Candidate model identity is retained in a sealed mapping but hidden from every
judge and reviewer. Case-level question and response text remains in private
local artifacts. Reports committed to the repository must be sanitized unless
the repository's existing private-artifact policy explicitly permits otherwise.

The grounded Gemini response is a comparison pivot, not unquestionable truth.
The prompt must allow a candidate to add correct material missing from Gemini or
to expose a material Gemini error.

## Local candidate pool

The initial local pool is:

| Candidate | Planned representation | Current evidence boundary |
|---|---|---|
| Qwen3.6-35B-A3B | Lemonade/llama.cpp ROCm, UD-Q4_K_XL GGUF | Recommended host path; approximately 23.3 GB download explicitly authorized |
| Qwen3.6-27B | Cached official FP8 checkpoint through the available ROCm runtime | Dense model avoids the known MoE FP8 selector failure, but the vLLM path is experimental on gfx1151 |
| gpt-oss-20b | Cached official checkpoint through a compatible local runtime | Independent family; exact usable runtime and Harmony prompt rendering require preflight |
| Gemma 4 31B | Cached Q4 GGUF through llama.cpp ROCm | Independent family; runtime and chat-template behavior require preflight |

Before generation, the implementation must verify complete weights, exact
revision, license, selected runtime, GPU visibility, context configuration, and
one deterministic completion. A candidate that cannot produce a valid response
after one bounded repair attempt is recorded as a runtime failure and excluded
from the 100-record phase; it is not silently replaced by a different model or
quantization.

No additional unusually large model download is authorized by this design. If
the runtime requires a material new checkpoint beyond the already authorized
Qwen3.6-35B-A3B GGUF, execution pauses for approval.

## Evaluation schema

All judges receive the same semantic instructions and return the same strict
record. Provider-specific syntax may differ, but it must not change the rubric.

```json
{
  "label": "good | partial | incorrect | uncertain",
  "correctness": 0,
  "completeness": 0,
  "relevance": 0,
  "error_flags": [],
  "reference_status": "sound | incomplete | suspect | uncertain",
  "confidence": "high | medium | low",
  "rationale": "concise decisive reason"
}
```

Component scales are frozen in the prompt:

- `correctness`: 0 to 4, where a central factual error is at most 1 and no
  material error found is 4;
- `completeness`: 0 to 2, based only on material requirements of the question;
- `relevance`: 0 to 2, including correct handling of ambiguity; and
- `error_flags`: selected from a closed vocabulary that includes entity,
  number, date, negation, unsupported specificity, refusal, and reference
  conflict.

Correctness is a gate. Verbosity, overlap with Gemini, or extra harmless detail
cannot compensate for a major factual error. `uncertain` is a valid result when
the question, evidence, or reference cannot support a defensible verdict.

The exact prompt, schema, decoding settings, maximum output, and prompt hash are
frozen before any compared generation. Raw output is retained alongside parsed
output; parser repairs never overwrite the original.

## Ten-record smoke test

The first packet contains ten real question/candidate-response pairs selected
for diagnostic coverage rather than prevalence estimation:

1. clearly correct;
2. correct but incomplete;
3. wrong central fact;
4. wrong entity;
5. wrong number or date;
6. negation or relationship reversal;
7. verbose response with a material error;
8. refusal or evasion;
9. ambiguous or underspecified question; and
10. candidate that may expose a Gemini-reference error.

Selection is blind to candidate model identity. The manifest records the
deterministic selection procedure, seed where applicable, and the reason for
each record. The ten IDs, input hashes, and initial prompt are sealed before any
local judge output is generated.

Every viable local candidate judges all ten records. The smoke test measures:

- end-to-end runtime viability and GPU use;
- strict-schema success and repair frequency;
- obvious rubric inversions or label collapse;
- appropriate use of uncertainty and reference-conflict flags;
- latency, throughput, peak memory, and output length; and
- characteristic rationales and failures.

The smoke test does not select a winner. After independent review, execution
stops for user review. Prompt changes create a new version and rerun the same
ten records; old outputs remain immutable.

## Independent review protocol

Two reviewers are used after judge outputs are frozen:

1. **Advisor:** a fresh, narrowly scoped subagent with no access to the sealed
   model-identity mapping.
2. **Claude Code:** a non-interactive, read-only Claude invocation with a fixed
   model identifier, capped budget, no write tools, and no persistent session.

Each reviewer first labels every case independently from the question,
reference, and candidate response without seeing local-judge decisions. This
reduces anchoring. In a second pass, each reviewer sees anonymized local-judge
outputs and critiques the prompt, schema compliance, decisive disagreements,
false confidence, and reference handling. Reviewer outputs are immutable and
stored separately.

The comparison report preserves:

- each reviewer's independent label and rationale;
- agreement and disagreement with each anonymized local judge;
- cases with reviewer disagreement;
- cases where Gemini appears suspect; and
- proposed prompt changes, with no automatic adoption.

Advisor and Claude are independent model reviewers, not human annotators. Their
agreement is evidence for model selection, not ground truth. Material factual
conflicts may receive targeted primary-source checking; blanket web grounding
is out of scope.

## User checkpoint after ten

The user receives a case-by-case view containing inputs, anonymized judge
outputs, both independent reviews, runtime metrics, and proposed prompt edits.
No 100-record run begins until the user approves:

- the prompt version;
- the structured schema;
- the surviving local candidates; and
- the 100-record selection protocol.

## Frozen 100-record local bake-off

The 100-record set is response-level and purposively stratified across topic,
question form, candidate behavior, and known judging risks. It includes the ten
smoke records as a named diagnostic slice but reports them separately to avoid
concealing reuse. Selection remains blind to candidate model identity.

The strata include clear/easy controls plus partial, incorrect, uncertain,
entity-heavy, numeric/date, negation, verbose, refusal, ambiguous,
time-sensitive, and plausible-reference-conflict records. The manifest and
prompt are frozen before any candidate processes the set.

Each surviving local candidate judges the identical 100 records. Advisor and
Claude independently adjudicate the 100 inputs, then review anonymized judge
outputs using the same two-pass procedure. Targeted evidence checks are limited
to material reviewer disagreements and suspected reference errors.

## Local winner decision

No single metric determines the winner. Report at least:

- exact and adjacent agreement with each reviewer and reviewer consensus where
  one exists;
- confusion matrix for `good`, `partial`, `incorrect`, and `uncertain`;
- major-error and false-confident-error counts;
- entity, number/date, negation, ambiguity, refusal, and reference-conflict
  behavior;
- strict JSON success, repair, retry, and generation-failure rates;
- repeated-run stability on a fixed audit subset;
- latency, throughput, peak memory, startup time, and disk footprint; and
- results by prespecified stratum.

The preferred local judge is the smallest/fastest model whose serious-error,
false-confidence, stability, and schema behavior are not materially worse than
the strongest local alternative. If quality crosses by stratum, the result may
recommend a primary local judge plus a local escalation judge rather than force
one universal winner. If the evidence is unresolved, retain two finalists for
the hosted comparison.

## Hosted comparison phase

Only after the local decision is frozen will the hosted shortlist be called:

- DeepSeek V4 Flash;
- Nemotron 3 Super;
- Qwen3-235B Instruct; and
- GPT-5.4 nano.

Exact provider routes, model IDs, current pricing, schema support, and batch
discounts require a fresh preflight. The hosted models judge the same frozen
100 records with the same semantic prompt and schema. The selected local
winner, or two unresolved local finalists, remains in the comparison.

Execution stops again after the hosted comparison. Scaling to 500 records or
the substantive corpus requires a separate user decision based on the blinded
case review, quality metrics, runtime, and projected cost.

## Reproducibility and artifacts

The implementation uses content-addressed, resumable artifacts. Every run
records:

- source-record hashes and sample-manifest hash;
- prompt and schema hashes;
- exact checkpoint revision and quantization;
- runtime/backend versions and launch configuration;
- hardware and GPU visibility;
- decoding settings and random seed;
- raw and parsed outputs;
- errors, retries, latency, token counts, and memory; and
- blinded and sealed identity mappings kept separately.

Expected repository structure:

```text
artifacts/pointwise_judge_bakeoff/       # private case-level run state
reports/pointwise_judge_bakeoff/         # sanitized summaries
protocols/pointwise_judge_bakeoff/       # frozen prompts and schemas
```

Temporary inference services bind only to `127.0.0.1` and are stopped after
each model's run. No service is published, shared, or exposed through Funnel,
Sites, or a public listener.

## Failure and stopping rules

- Preserve initial failures and allow at most one predefined repair attempt.
- Do not silently change checkpoint, quantization, backend, prompt, or schema
  within a comparison.
- Stop if GPU/runtime behavior threatens host stability, available memory is
  insufficient, or output corruption suggests a backend problem.
- Stop for approval before any unapproved large download, system-package
  change, hosted call, 100-record run, or scale beyond 100.
- Never convert unresolved reviewer disagreement into a confident label merely
  to rank models.

## Success criteria for this phase

This design succeeds when it produces:

1. a reviewed, reproducible ten-record local smoke result;
2. an explicitly approved prompt and 100-record manifest;
3. a defensible local winner or documented unresolved finalists from 100
   records;
4. an apples-to-apples hosted comparison on the same records; and
5. a user-controlled go/no-go checkpoint before any larger run.
