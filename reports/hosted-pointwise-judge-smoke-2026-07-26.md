# Hosted pointwise judge smoke test

Date: 2026-07-26

## Decision

For the next 500-case bake-off, shortlist **DeepSeek V4 Flash** and **GPT-5.4 nano** after one prompt revision and a second 10-case confirmation. DeepSeek is the least expensive hosted candidate observed; GPT nano adds a different model family/provider and was the fastest hosted route. Qwen3-235B remains the reserve candidate because its label quality tied the leaders but it was slower and less provider-diverse.

Do not scale the current prompt yet if downstream logic uses detailed scores, flags, reference status, or rationales. Final labels are strong, but those secondary fields are not stable enough.

## Frozen sample and review protocol

- 10 identity-blind cases from the existing smoke manifest.
- Four hosted judges, direct/non-thinking configuration.
- One smart advisor and Anthropic Claude performed independent pass-1 review, then anonymous pass-2 critique.
- Model identities were unsealed only after both ledgers were locked.
- Both reviewer ledgers pass the repository's exact schema, coverage, alias, and identity-leak validators.
- Advisor and Claude agreed on 8/10 pass-1 labels and 10/10 pass-2 labels.

## Results

| Judge | Route | Reviewer label agreement | Exact structured agreement (rationale excluded) | Actual model cost / 10 | Model elapsed / 10 |
|---|---|---:|---:|---:|---:|
| DeepSeek V4 Flash | OpenRouter -> DeepInfra | 10/10 | 5/10 | $0.000609 | 40.2 s |
| Qwen3-235B Instruct 2507 | OpenRouter -> DeepInfra | 10/10 | 5/10 | $0.001286 | 208.9 s |
| GPT-5.4 nano | OpenRouter -> OpenAI, default non-thinking | 10/10 | 4/10 | $0.002867 | 23.2 s |
| Nemotron 3 Super | OpenRouter -> DigitalOcean, reasoning disabled | 9/10 | 5/10 | $0.001955 | 201.4 s |
| Local Qwen3.6-35B-A3B direct | local ROCm runtime | 9/10 | 5/10 | $0 API | 31.9 s |
| Local Qwen3.6-35B-A3B thinking | local ROCm runtime | 8/10 | 5/10 | $0 API | 242.8 s |

All hosted calls produced valid JSON without a repair attempt. The single disputed case omitted the advertisement needed to verify a proposed slogan. The final reviewers and three hosted leaders labeled it `uncertain`; Nemotron and local direct Qwen3.6 labeled it `incorrect`.

## Cost

The four hosted models' successful responses cost **$0.006717 total**. The conservative experiment ledger ended at **$0.019004** because failed routing/rate-limit attempts retain their worst-case reservation instead of being refunded.

Observed 10-case token use gives the following rough extrapolation. It is not a quote: a 10-case diagnostic sample is too small to estimate corpus-wide token length precisely.

| Judge | 182,300 substantive responses | All 1,163,850 responses without refusal filtering |
|---|---:|---:|
| DeepSeek V4 Flash | ~$11.10 | ~$70.88 |
| Qwen3-235B Instruct 2507 | ~$23.44 | ~$149.68 |
| GPT-5.4 nano | ~$52.26 | ~$333.67 |
| Nemotron 3 Super | ~$35.64 | ~$227.54 |

The planned 500-case top-two run using DeepSeek plus GPT nano extrapolates to only **about $0.17 in model inference** from this smoke. Review/adjudication labor or external reviewer calls are separate.

For local Qwen3.6 over the 182,300 substantive responses, the prior measured throughput estimate is about **7.04 days direct** versus **67.77 days all-thinking** when run sequentially. For all 1,163,850 responses without deterministic refusal filtering, the estimates are about **44.93 days direct** and **432.68 days thinking**.

## Prompt changes before confirmation

1. Define when missing question context should yield `uncertain` rather than `incorrect`.
2. Tighten `refusal`, `negation_error`, `ambiguity`, and `reference_conflict` definitions with one positive and one negative example each.
3. State that a rationale must not invent supporting errors. If the label is clear but a particular factual allegation cannot be verified, omit that allegation.
4. Clarify that error flags explain the candidate response, not defects imagined from the reference.
5. Keep thinking disabled for this task by default. On this sample, local thinking was 7.6x slower and reduced label agreement from 9/10 to 8/10.

After editing the prompt, rerun these same 10 cases with DeepSeek and GPT nano. If both retain at least 9/10 reviewer-label agreement and improve secondary-field agreement, freeze the prompt and run 500 cases for both.

## Audit notes

- OpenRouter transport schema omits only `error_flags.uniqueItems`; the local frozen parser still enforces uniqueness.
- Provider routing is exact, fallbacks are disabled, and runtime receipts bind provider, prices, reasoning mode, schema hashes, and known endpoint aliases.
- Nemotron moved from an unhealthy DeepInfra endpoint to the healthy DigitalOcean endpoint.
- GPT nano omits both `reasoning` and unsupported `temperature`; the catalog's default is non-thinking.
- Raw response, run, review, and spend artifacts remain private mode 0600 beneath `artifacts/pointwise_judge_bakeoff/`.
