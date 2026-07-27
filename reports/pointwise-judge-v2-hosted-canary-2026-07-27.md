# Pointwise Judge v2 Diagnostic Canary

On this 10-case canary, these results are diagnostic evidence about these frozen cases and configurations only; the sample is too small for a population-quality claim.

## Per-configuration summary

| Configuration | Provider | Completion | Label agreement | Correctness agreement | Completeness agreement | Relevance agreement | Reference-status agreement | Flag Jaccard | Cross-model label agreement | Cost (USD) | Latency (s) | Repair attempts | Retry attempts | Rationale audit | Hard-gate failures |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| deepseek-v4-flash-direct | DeepInfra | 10/10 | 1.000 | 0.800 | 1.000 | 0.750 | 1.000 | 0.929 | n/a | 0.001218 | 38.339 | 0 | 0 | 5/10 | rationale_quote_fidelity |
| gpt-5.4-nano-direct | OpenAI | 9/10 | 0.667 | 0.667 | 0.889 | 0.857 | 0.778 | 0.714 | n/a | 0.002946 | 18.869 | 0 | 0 | 2/9 | rationale_quote_fidelity, terminal_decision |
| qwen3.6-35b-a3b-direct | AkashML | 10/10 | 1.000 | 0.800 | 1.000 | 0.750 | 1.000 | 0.929 | n/a | 0.004439 | 16.295 | 0 | 0 | 5/10 | rationale_quote_fidelity |
| qwen3.6-35b-a3b-thinking | AkashML | 10/10 | 0.900 | 0.800 | 0.700 | 0.750 | 1.000 | 0.929 | n/a | 0.022724 | 116.240 | 1 | 1 | 4/10 | rationale_quote_fidelity |

## Paired Qwen direct/thinking changes

Paired comparison available: True; changed cases: 3; excluded pairs: 0.

DeepSeek direct and Qwen direct produced the same label, correctness, completeness,
relevance, and reference-status values on all 10 cases. They differed on one
error-flag set, so their complete structured decisions matched on 9/10 cases.

Qwen direct and Qwen thinking matched on 9/10 labels and 7/10 complete structured
decisions. Thinking changed correctness on three cases, completeness on three,
error flags on two, and relevance on one. These changes did not improve the
canary agreement metrics.

## Manual rationale audit

The manual audit covered 24 records: every terminal non-`good` decision, the GPT
terminal failure, and three `good` spot checks for the leading-“No,” control.

| Configuration | Audited | Passed | Failed |
| --- | ---: | ---: | ---: |
| DeepSeek direct | 6 | 3 | 3 |
| GPT nano direct | 6 | 0 | 6 |
| Qwen direct | 6 | 2 | 4 |
| Qwen thinking | 6 | 3 | 3 |
| **Total** | **24** | **8** | **16** |

The failures included unsupported factual assertions, an unsupported provisional-
reference conflict, a negation flag without a negation, and incorrect use of the
refusal and uncertainty categories. Three configurations handled the leading-“No,”
control correctly; GPT nano did not.

Ten flagged records also failed the required verbatim-candidate-quote rule. The
audit exposed a prompt-design conflict: when the defect is in the question or
provisional reference, requiring the rationale to quote the candidate is not
always appropriate. This rule should be revised before a larger run.

## Canary decision

- Advance DeepSeek direct and Qwen direct to the next prompt-validation round.
- Do not advance GPT nano: it had one terminal schema-invariant failure, weaker
  agreement, and no manual-audit passes in its six audited records.
- Do not advance Qwen thinking: it was slower and more expensive, required a
  repair, and did not improve agreement over direct mode.
- Revise the rationale evidence rule, then repeat these same 10 cases before
  expanding to 100 or 500 cases.

This is a configuration-screening decision, not a claim that either surviving
model is a generally reliable judge.

## v1 differences

| Comparison | Changed cases |
| --- | ---: |
| v1_canary__to__deepseek-v4-flash-direct | 4 |
| v1_canary__to__gpt-5.4-nano-direct | 6 |
| v1_canary__to__qwen3.6-35b-a3b-direct | 4 |
| v1_canary__to__qwen3.6-35b-a3b-thinking | 4 |

## Hard-gate failures

accounting_reconciliation

## Claim boundary

This report does not make a broad model-quality claim or rank models beyond this diagnostic sample.
