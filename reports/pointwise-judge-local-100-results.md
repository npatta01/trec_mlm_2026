# Local pointwise judge: 100-case results

## Technical summary

**Advance Qwen3.6-35B-A3B Q4_K_XL in thinking mode as the local finalist.** It produced valid structured decisions on all 100 cases and matched the two-reviewer independent pass-1 consensus on 84/87 resolved cases (96.6%), compared with 78/87 (89.7%) for direct mode. On those resolved cases, thinking matched consensus on six cases that direct missed; direct matched none that thinking missed.

The quality gain is expensive: thinking used 2.47× as many total tokens and 9.63× the summed request latency. Direct mode is therefore worth retaining as the fast local baseline or fallback, but it is not the best local quality candidate on this sample.

These are **agreement results, not accuracy estimates**. Advisor and Claude are fallible LLM reviewers, 13 cases lacked exact independent reviewer consensus, and the 100 cases were deliberately balanced across diagnostic proxies rather than sampled to represent production prevalence.

## Thinking improves reviewer agreement, especially on entity mismatches

Both inference modes evaluated the same 100 unique question-response pairs under the same frozen prompt and decision contract, 8,192-token context, 4,096-token output allowance, and temperature zero. In these historical canonical runs, Lemonade 11.5 received the same compatibility schema with both `rationale.maxLength` and `error_flags.uniqueItems` omitted. Each runtime observation records that adjustment but contains no separate transport-schema hash. The frozen strict parser remained authoritative, and both modes returned 100/100 valid terminal decisions under it. Each needed one successful repair after an initially malformed response; neither had a terminal failure. Commit `7e78bcb` improves future runs by omitting only the backend-incompatible `rationale.maxLength`, retaining `error_flags.uniqueItems`, and recording the adjusted transport schema hash; it does not retroactively change these artifacts.

| Evaluation view | Direct | Thinking |
| --- | ---: | ---: |
| Valid structured decisions | 100/100 | 100/100 |
| Independent Advisor agreement | 88/100 | 93/100 |
| Independent Claude agreement | 80/100 | 87/100 |
| Independent reviewer-consensus agreement | 78/87 | 84/87 |
| Critique-informed Advisor agreement | 87/100 | 93/100 |
| Critique-informed Claude agreement | 80/100 | 87/100 |
| Critique-informed reviewer-consensus agreement | 78/88 | 85/88 |

“Reviewer consensus” exists only where Advisor and Claude assigned the same label. Independent pass 1 was completed without model judgments. In pass 2, reviewers saw the two anonymous model judgments and their critiques, so pass-2 results are useful adjudication evidence but not independent confirmation.

The pass-1 result is the decisive comparison: among its 87 consensus cases, both modes matched 78, thinking alone matched six, direct alone matched zero, and neither matched three. Pass 2 preserved that direction: thinking alone matched seven of 88 consensus cases and direct alone again matched zero.

The modes themselves assigned the same label on 85/100 cases and were within one ordered label step on 96/100. Their 15 exact disagreements were concentrated: thinking moved nine `incorrect` decisions to `partial`, three `incorrect` decisions to `good`, one `partial` to `good`, one `partial` to `incorrect`, and one `good` to `incorrect`. Thinking was therefore generally less severe than direct, but not uniformly so.

## Diagnostic strata show where extra reasoning helped

Each proxy contains ten deliberately selected cases. “Consensus” below means exact Advisor-Claude pass-2 label agreement; its denominator varies because unresolved reviewer disagreements are excluded.

| Diagnostic proxy | Mode agreement | Consensus cases | Direct vs consensus | Thinking vs consensus |
| --- | ---: | ---: | ---: | ---: |
| Entity mismatch | 4/10 | 9 | 4/9 | **9/9** |
| High-overlap control | 10/10 | 10 | 10/10 | 10/10 |
| Low-overlap error | 9/10 | 9 | 8/9 | **9/9** |
| Negation mismatch | 10/10 | 10 | 10/10 | 10/10 |
| Number or date mismatch | 8/10 | 8 | 6/8 | **7/8** |
| Possible reference conflict | 7/10 | 7 | 5/7 | 5/7 |
| Refusal | 10/10 | 10 | 10/10 | 10/10 |
| Short or incomplete response | 10/10 | 10 | 10/10 | 10/10 |
| Underspecified question | 9/10 | 9 | 9/9 | 9/9 |
| Verbose error | 8/10 | 6 | 6/6 | 6/6 |

The clearest gain is the entity-mismatch stratum: thinking matched all nine resolved reviewer labels, versus four for direct. Thinking also gained one resolved match in each of the low-overlap and number/date strata. Neither mode had an exact-match advantage on the resolved possible-reference-conflict, underspecified, or verbose-error cases. The latter strata also had the most unresolved reviewer cases, so their apparently equal model performance is less conclusive.

The ten reused smoke cases remained stable enough to connect the experiments: the modes agreed exactly on 9/10 and were within one label step on all ten.

## Independent review and critique-informed review tell the same directional story

Advisor and Claude agreed exactly on 87/100 labels in independent pass 1 and 88/100 after critique-informed pass 2. Claude changed **zero pass-1 decision fields, labels, or rationales** in pass 2 while adding decisive-disagreement and prompt findings on 22 cases. Advisor changed two labels and at least one scored decision field on 24 cases, while recording decisive disagreements on 16 cases and prompt findings on 28.

The critique pass therefore improved the audit trail more than it changed the result. Importantly, thinking's advantage was already present in the identity-blind pass 1, before either reviewer saw the model judgments.

The two reviewers are not a gold standard. Their exact disagreement remained 12/100 after critique, including cases where the provisional reference or label boundary was genuinely debatable. No human adjudication was performed.

## Thinking's quality gain costs roughly tenfold latency

| Runtime measure | Direct | Thinking | Thinking / direct |
| --- | ---: | ---: | ---: |
| Summed request latency | 333.54 s | 3,212.05 s | 9.63× |
| Mean latency per case | 3.34 s | 32.12 s | 9.63× |
| Prompt tokens | 74,080 | 73,259 | 0.99× |
| Completion tokens | 13,220 | 141,981 | 10.74× |
| Total tokens | 87,300 | 215,240 | 2.47× |

Latency is the sum of per-request elapsed time, not an end-to-end parallel throughput benchmark. The local runs were sequential and used the same cached GGUF model on the same ROCm host.

## Other cached local candidates did not displace Qwen thinking

- **GPT-OSS-20B high reasoning:** increasing the output allowance from 4,096 to 8,192 did not fix reliability. Nine smoke cases again completed validly, using 6,391 completion tokens and 534.20 summed seconds. The tenth request returned HTTP 200 after roughly eleven minutes but consumed the full 8,192-token allowance without producing the required final JSON. This is captured model-generation behavior under the tested prompt/runtime, not merely a short HTTP timeout. It is not suitable as the primary local judge without a different reasoning/output control strategy.

- **Qwen3.6-27B FP8:** all 66 weight shards loaded in 77 seconds, after which the vLLM ROCm runtime remained in FP8 kernel initialization for more than 21 minutes and never opened the API. The log warned that no tuned gfx1151 W8A8 kernel configuration was available. The process remained CPU-active, did not show an out-of-memory failure, and ultimately required SIGKILL (exit 137) after SIGTERM did not stop it. This rules out the tested FP8/vLLM path on this host; it does not establish that a GGUF/llama.cpp or FP16 path would fail.

## Recommendation and next test

1. Carry **Qwen3.6-35B-A3B thinking** into the hosted-model bake-off as the local quality finalist.
2. Keep **the same model in direct mode** as the local speed and cost control.
3. Run the frozen ten-case API smoke before any 500-case spend. Compare the hosted shortlist against both local modes using independent review first; only the strongest two hosted candidates should advance to the 500-case stage.
4. If production throughput matters more than maximum local agreement, evaluate routing direct-mode cases to thinking only for strata resembling entity mismatches or other high-uncertainty cases. The current sample supports that hypothesis but is too deliberately stratified to set a routing threshold.

## Scope and limitations

- The sample has exactly ten cases from each of ten diagnostic proxies; aggregate rates are therefore diagnostic-set rates, not expected production rates.
- The provisional references can themselves be incomplete or conflicting. `uncertain` is allowed by the frozen local-judge schema, but neither canonical mode emitted it; both runs used only `good`, `partial`, and `incorrect`.
- Exact label agreement is strict. A secondary good/partial leniency check does not change the recommendation, but it is not used as the headline metric.
- Model-family and reviewer biases may be correlated. A small human-reviewed calibration slice would be needed before treating any agreement rate as operational accuracy.
- No paid hosted-model calls and no 500-case run are included in this report.
