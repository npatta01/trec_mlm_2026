# Hosted pointwise-judge provider preflight — 2026-07-26

**Scope.** A no-call, provider-catalog preflight for the requested shortlist: DeepSeek V4 Flash, NVIDIA Nemotron 3 Super, Qwen3-235B Instruct, and GPT-5.4 nano, plus same-model remote controls for the completed local Qwen3.6-35B and GPT-OSS-20B work.  Sources are provider-owned model catalogs, pricing pages, and API documentation, checked on 2026-07-26.  Prices are USD per one million tokens.  This is not a quality comparison and no paid model was invoked.

## Decision snapshot

- **Most deployment-ready strict-schema route:** direct **OpenAI `gpt-5.4-nano`**, including a documented 50% asynchronous Batch discount.  It is materially more expensive under the repository's final observed thinking profile.
- **Lowest transparent direct rate:** **DeepInfra `deepseek-ai/DeepSeek-V4-Flash`** ($0.09 input / $0.18 output; $0.018 cached input). OpenRouter's current headline for the same model is now also $0.09 / $0.18. DeepInfra's live catalog tags it for structured output and reasoning, but its public docs do not commit this specific model to a named reasoning-effort scale. Run a small canary before a batch.
- **Nemotron must be pinned, not treated as a single 1M/1M route.** `nvidia/nemotron-3-super-120b-a12b` is the paid OpenRouter ID; catalog-wide context is 1M, but the current default top endpoint is only **262,144 context / 16,384 completion**.  Pin a provider and re-read its endpoint record before committing a long-context run.
- **“Qwen3-235B Instruct” is ambiguous.** The currently identifiable requested instruct release is `qwen/qwen3-235b-a22b-2507` (OpenRouter), corresponding to `Qwen/Qwen3-235B-A22B-Instruct-2507` on DeepInfra.  Its model description explicitly says it has *no thinking mode*; do not substitute a Qwen3.5/3.6 model without a new decision.
- **OpenRouter itself has no public asynchronous Batch discount in the reviewed current docs.** It can route a job to different providers unless the request is constrained.  The only documented OpenRouter way to preserve schema capability is `provider.require_parameters: true`, preferably with a pinned provider/order and a max-price guard.

## What is verified, and what is not

### Verified model identity, availability, limits, and prices

The [OpenRouter live Models API](https://openrouter.ai/api/v1/models) is the primary catalog: it exposes model IDs, current token prices, context length, top-provider completion limit, supported parameters, and reasoning metadata.  Its [model-catalog guide](https://openrouter.ai/docs/guides/overview/models) describes those fields.  The [DeepInfra live catalog](https://deepinfra.com/models) and its OpenAI-compatible [chat API guide](https://docs.deepinfra.com/chat/overview) are the direct-provider sources.

| Requested label | Exact current model ID(s) | Provider / rate: input · cached input · output | Safe context / maximum output | Reasoning control | Strict JSON-schema status |
|---|---|---|---|---|---|
| DeepSeek V4 Flash | OpenRouter: `deepseek/deepseek-v4-flash`; direct DeepInfra: `deepseek-ai/DeepSeek-V4-Flash` | **OpenRouter headline:** $0.09 · cache rate endpoint-dependent · $0.18. **DeepInfra direct:** $0.09 · $0.018 · $0.18; Flex $0.072 · $0.0144 · $0.144. | 1,048,576-token model context; do not construe it as 1M generated tokens. | OR: optional `reasoning`, efforts `high`/`xhigh` (`xhigh` maps to max). DI tags reasoning/can-disable-reasoning but does not publish V4's exact effort tiers. | OR catalog exposes `response_format` and `structured_outputs`; direct DI catalog tags structured output/JSON. Both are usable only with the provider's schema contract below. |
| NVIDIA Nemotron 3 Super | OpenRouter: `nvidia/nemotron-3-super-120b-a12b`; free alias (not a fixed paid route): `nvidia/nemotron-3-super-120b-a12b:free`. Direct DeepInfra: `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B`. | **OR headline:** $0.085 · no published cached-input rate · $0.40. **DI direct:** $0.085 · not published · $0.40; Flex $0.068 · $0.32. | OR model context 1M, but the current DI card and safe default route are 262,144 context; check the pinned endpoint's output cap. | OR supports reasoning with max-token control; catalog lists `medium`/`low`, default `medium`. DI's generic reasoning API exists, but exact Super effort semantics are not published. | OR exposes `response_format`/`structured_outputs`; DI card advertises structured output/JSON. Pin endpoint and validate the frozen schema. |
| “Qwen3-235B Instruct” | **Not an exact ID.** Current match: OpenRouter `qwen/qwen3-235b-a22b-2507`; DeepInfra `Qwen/Qwen3-235B-A22B-Instruct-2507`. | **OR and DI standard:** $0.09 · no published cached-input rate · $0.55. **DI Flex:** $0.072 · $0.44. | 262,144 context; set an application output cap well below the window. | **No native thinking mode** for this `-Instruct-2507` release. | Both catalog entries advertise JSON/`response_format`; strict-schema endpoint support through OR must be filtered with `require_parameters: true`. |
| GPT-5.4 nano | Direct OpenAI: `gpt-5.4-nano`; OpenRouter: `openai/gpt-5.4-nano`. **Not a DeepInfra hosted open-model catalog entry.** | Direct OpenAI and OR headline: $0.20 · $0.02 · $1.25. Direct OpenAI Batch: $0.10 · $0.01 · $0.625. | 400,000 context / 128,000 output. | Direct OpenAI: `reasoning.effort` supports `none` (default), `low`, `medium`, `high`, `xhigh`. | Supported by OpenAI Structured Outputs; use strict schema and still handle refusals/length stops. |

DeepSeek's OpenRouter price is deliberately shown separately from DeepInfra's direct price: the router’s live catalog is the billing authority for a router request, even if a DeepInfra route is selected.  Likewise, do **not** use the free Nemotron alias as a reproducible paid judging route: its provider, context, output cap, retention, and availability can change.

### Provider behavior verified for all candidate integrations

- **OpenRouter structured outputs.** Its documented request form is `response_format: {type: "json_schema", json_schema: {...}}`.  Support is **endpoint-specific**, can change, and strict enforcement varies by provider.  OpenRouter specifically instructs clients to require parameters and send the schema: [`structured outputs`](https://openrouter.ai/docs/guides/features/structured-outputs).  For this workload send `provider: {require_parameters: true, order: [...]}` and fail closed on an unsupported endpoint rather than silently accepting JSON mode.  The router returns native-token usage, including `completion_tokens_details.reasoning_tokens`, and a per-request cost field: [API reference](https://openrouter.ai/docs/api_reference/overview).
- **OpenRouter reasoning.** The unified `reasoning` object can set `enabled`, effort, maximum reasoning tokens, and whether reasoning is returned, but the live model record is the authority for a particular model: [reasoning-token guide](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens).  Reasoning is billed as generated completion/output tokens.
- **DeepInfra structured outputs.** It documents `json_object` and strict `json_schema` modes for supported models, including `strict: true`; it warns that a length stop can produce incomplete JSON and that forced JSON can worsen unsupported factual answers: [structured outputs](https://docs.deepinfra.com/chat/structured-outputs).  Its catalog tags are the per-model evidence; generic documentation says only “many” models support it.
- **DeepInfra reasoning and cache.** `reasoning_effort` accepts `none`, `low`, `medium`, `high`; `reasoning: {effort, enabled}` is the finer control; reasoning tokens count as billed output.  A control sent to a non-reasoning model has no effect: [reasoning docs](https://docs.deepinfra.com/chat/reasoning).  Prefix caching is automatic on supported models; usage reports `cached_tokens`; entries are per model/account and expire after inactivity: [prompt caching](https://docs.deepinfra.com/chat/prompt-caching).  Preserve the common rubric/system prefix verbatim, but do not assume a cache hit in the budget.
- **Async/batch/flex discounts.** Direct **OpenAI Batch** is documented as **50% cheaper**, and OpenAI's pricing table lists GPT-5.4 nano Flex at the same half-rate: [Batch guide](https://developers.openai.com/api/docs/guides/batch), [pricing](https://developers.openai.com/api/docs/pricing). **DeepInfra Batch** is **20% off**. DeepInfra **Flex** is separately 20% off on tagged models; it may wait up to ten minutes and return 429: [documentation index](https://docs.deepinfra.com/llms.txt), [service tiers](https://docs.deepinfra.com/chat/overview). Batch and Flex are alternatives; no reviewed source says they stack. OpenRouter has no Batch discount; its Flex pass-through currently names OpenAI/Google (and xAI priority), not DeepInfra: [OpenRouter service tiers](https://openrouter.ai/docs/guides/features/service-tiers).
- **Concurrency caveat.** DeepInfra's default is 200 concurrent requests **per model**, and it may still return 429 when a model is busy: [rate-limit documentation](https://docs.deepinfra.com/account/rate-limits).  OpenRouter paid limits are upstream/provider/model-specific; its public free-tier limits are not a paid production throughput promise: [limits](https://openrouter.ai/docs/api_reference/limits).

### Inference and unknowns (not verified facts)

1. The frozen pointwise schema is deliberately small, so all verified limits are far beyond its observed token profile.  That is an inference from the local artifacts, not a provider guarantee.
2. A schema tag/parameter means output shape, not evaluator correctness.  Retain JSON validation, finish-reason checks, retry accounting, and raw usage/cost receipts.
3. Exact current provider selection, price, cache rate, model revision, output cap, retention policy, and reasoning behavior can change between this preflight and execution—especially with OpenRouter’s default routing.  Re-fetch the provider’s exact endpoint catalog immediately before spend.
4. The primary sources did **not** establish: a direct DeepInfra public price/cache quote for Qwen-235; a cache price for Nemotron or Qwen; exact DeepInfra per-model reasoning effort tiers for the non-R1 entries; or a paid-model RPM/TPM commitment.  These are intentional `unknown`, not zero values.

## Transparent cost projection from this repository’s observed profiles

### Local evidence and formula

The observed runs are the private, frozen 100-case artifacts:

- **Direct:** `qwen36-35b-a3b-direct.json` has 100 terminal records, **74,080 prompt** and **13,220 completion** tokens: **740.8 input + 132.2 output tokens/case**.
- **Thinking:** `qwen36-35b-a3b-thinking.json` is now terminal at 100 cases, with **73,259 prompt** and **141,981 completion** tokens: **732.59 input + 1,419.81 output tokens/case**.

These are final canonical local profiles.  They describe the local Qwen run and are workload-shaped costing inputs, not predictions that every hosted model will use the same tokenizer or hidden-reasoning volume.

For `N` cases, uncached cost is:

```text
N × ((mean_input × input_price) + (mean_billed_output × output_price)) / 1,000,000
```

`mean_billed_output` includes visible output and any hidden reasoning billed as output.  All baseline projections use **zero cached tokens**, no tools, no retries, and standard service tier.  They are token-cost estimates only; taxes, rounding/minimums, egress, failed calls, and provider-credit fees are excluded.

| Route and applied standard rate | 100 cases: direct → thinking profile | 500 cases: direct → thinking profile | Discounted 500-case price if documented |
|---|---:|---:|---|
| Qwen3.6-35B-A3B, DeepInfra ($0.15 / $0.95) | **$0.02367 → $0.14587** | **$0.11836 → $0.72935** | **DI Flex or Batch 20%:** $0.09468 → $0.58348. |
| Qwen3.6-35B-A3B, OpenRouter ($0.14 / $1.00) | **$0.02359 → $0.15224** | **$0.11796 → $0.76119** | No OR Batch; no DI Flex pass-through. |
| DeepSeek V4 Flash, DI or OR ($0.09 / $0.18) | **$0.00905 → $0.03215** | **$0.04523 → $0.16075** | **Direct DI Flex or Batch 20%:** $0.03619 → $0.12860. |
| Nemotron 3 Super, DI or OR ($0.085 / $0.40) | **$0.01158 → $0.06302** | **$0.05792 → $0.31510** | **Direct DI Flex or Batch 20%:** $0.04634 → $0.25208. |
| Qwen3-235B Instruct, DI or OR ($0.09 / $0.55) | **$0.01394 → $0.08468** | **$0.06969 → $0.42341** | **Direct DI Flex or Batch 20%:** $0.05575 → $0.33873. No native thinking; the right side is only sensitivity. |
| GPT-5.4 nano, OpenAI/OR ($0.20 / $1.25) | **$0.03134 → $0.19213** | **$0.15670 → $0.96064** | **Direct OpenAI Batch or Flex 50%:** $0.07835 → $0.48032. |

The arrows are a **profile range**, not confidence bounds: left is the final direct-mode local mean and right is the final thinking-mode local mean. Cache hits are not applied.

Self-hosted Qwen has **$0 marginal provider/token charge**, but not zero economic cost. Extrapolating the measured sequential runs, 500 cases take about **27.8 minutes direct** or **4.46 hours thinking**. Energy cost is `average wall power (kW) × hours × electricity tariff`; wall power and tariff were not measured, so no honest local dollar amount is available. Machine occupancy and depreciation are separate.

## Recommended capped remote smoke set — proposal only

Use the same frozen ten `smoke-v2` cases, prompt hash, and schema hash for every configuration.  Each row is **one schema canary, then the remaining nine cases only if it passes**.  Use non-streaming requests, temperature 0 where accepted, exactly one primary attempt per case, at most one retry for 408/429/5xx, and no automatic model/provider fallback.  Validate `finish_reason`, strict schema, exact IDs, usage, provider, and cost on every response.

| Purpose | Exact route | Ten-case configuration and hard generation cap | Why this cap |
|---|---|---|---|
| Same-model Qwen direct control | DeepInfra `Qwen/Qwen3.6-35B-A3B` | `reasoning.enabled=false`; strict schema; **512 total output tokens/request** | The final local direct run averaged 132.2 completion tokens/case. This allows nearly 4× the mean while bounding a malformed response. |
| Same-model Qwen thinking control | Same pinned DeepInfra ID | `reasoning.enabled=true`, `reasoning_effort=high`; strict schema; **4,096 total output tokens/request** | Matches the local Qwen thinking run’s 4,096 allowance and covers about 2.9× its final 1,419.81-token mean. |
| Same-model GPT-OSS control | DeepInfra `openai/gpt-oss-20b` | `reasoning_effort=high`; strict schema; **8,192 total output tokens/request** | Matches the authorized local high-reasoning retry allowance; the model supports low/medium/high reasoning. |
| Hosted DeepSeek shortlist smoke | DeepInfra `deepseek-ai/DeepSeek-V4-Flash` | reasoning disabled; strict schema; **512 output tokens/request** | Tests the cheapest direct, structured, non-reasoning path first. |
| Hosted Nemotron shortlist smoke | OpenRouter `nvidia/nemotron-3-super-120b-a12b`, pinned paid provider | `reasoning.effort=low`, reasoning budget no more than 1,536; **2,048 total output tokens/request**; `require_parameters=true` | Its route-dependent limits and structured-output behavior are the main risks; low reasoning bounds the first comparison. |
| Hosted Qwen-235 shortlist smoke | OpenRouter `qwen/qwen3-235b-a22b-2507`, pinned provider | no thinking mode; strict schema; **512 output tokens/request**; `require_parameters=true` | This exact instruct release has no native thinking mode. |
| Hosted GPT shortlist smoke | Direct OpenAI `gpt-5.4-nano` | `reasoning.effort=none`; strict schema; **512 output tokens/request** | Exercises the cleanest native strict-schema direct baseline before any Batch submission. |

The exact DeepInfra Qwen control is currently advertised at **$0.15 input / $0.95 output**, with a 262,144-token context, JSON support, Priority, and Flex; Flex is **0.8×** ($0.12/$0.76): [DeepInfra Qwen3.6-35B-A3B](https://deepinfra.com/Qwen/Qwen3.6-35B-A3B).  OpenRouter also exposes `qwen/qwen3.6-35b-a3b` at $0.14/$1.00 and describes integrated thinking plus structured output, but using DeepInfra directly removes a routing variable: [OpenRouter Qwen3.6](https://openrouter.ai/qwen/qwen3.6-35b-a3b).

The exact GPT-OSS control is currently advertised by DeepInfra at **$0.03/$0.14**, 131,072 context, JSON support, Flex at **$0.024/$0.112**, and low/medium/high reasoning: [DeepInfra GPT-OSS-20B](https://deepinfra.com/openai/gpt-oss-20b).  OpenRouter’s corresponding exact model ID is `openai/gpt-oss-20b`, with roughly $0.03/$0.13–0.14 headline pricing depending on catalog rounding/route: [OpenRouter GPT-OSS-20B](https://openrouter.ai/openai/gpt-oss-20b).  Pinning DeepInfra again removes an aggregator variable.

Using a conservative **2,500 input-token ceiling per case** and the output caps above, the seven primary ten-case configurations cost at most about **$0.10** at the quoted standard token prices; allowing one retry for every request remains below **$0.20** in model-token charges.  Make `$0.20` a client-side hard stop, and abort before the first request if refreshed catalog prices make either bound false.  This is a cap calculation, not authorization to spend.

## Recommended pre-spend gate

1. Freeze an exact model ID, provider route, model revision (if returned), schema hash, reasoning setting, max output, and price ceiling in the job receipt.
2. For OpenRouter, request `json_schema` + `strict: true` + `provider.require_parameters: true`, pin an allowed provider/order, and record the returned provider/model/usage/cost.  Reject route changes.
3. For DeepInfra, submit the frozen rubric as a prefix, validate its `cached_tokens`, and cap concurrency below 200 with retry/backoff.  Use Batch only after verifying the **exact model** is eligible for the documented 20% rate.
4. Run a paid-authorized small canary only after the user elects a route; record schema pass rate, length stops, reasoning/output split, actual bills, and calibration against independent labels.  Do not promote a route merely because its JSON parses.

## Source list

- [OpenRouter live model catalog](https://openrouter.ai/api/v1/models)
- [OpenRouter: structured outputs](https://openrouter.ai/docs/guides/features/structured-outputs), [reasoning](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens), [API usage/cost fields](https://openrouter.ai/docs/api_reference/overview), [limits](https://openrouter.ai/docs/api_reference/limits)
- [OpenRouter DeepSeek V4 Flash model page](https://openrouter.ai/deepseek/deepseek-v4-flash), [Nemotron 3 Super model page](https://openrouter.ai/nvidia/nemotron-3-super-120b-a12b), [Qwen3-235B-A22B-Instruct-2507 model page](https://openrouter.ai/qwen/qwen3-235b-a22b-2507), [GPT-5.4 nano model page](https://openrouter.ai/openai/gpt-5.4-nano), [Qwen3.6-35B-A3B model page](https://openrouter.ai/qwen/qwen3.6-35b-a3b), [GPT-OSS-20B model page](https://openrouter.ai/openai/gpt-oss-20b)
- [DeepInfra live model catalog](https://deepinfra.com/models), [Qwen3.6-35B-A3B](https://deepinfra.com/Qwen/Qwen3.6-35B-A3B), [GPT-OSS-20B](https://deepinfra.com/openai/gpt-oss-20b), [chat API](https://docs.deepinfra.com/chat/overview), [structured outputs](https://docs.deepinfra.com/chat/structured-outputs), [reasoning](https://docs.deepinfra.com/chat/reasoning), [prompt caching](https://docs.deepinfra.com/chat/prompt-caching), [rate limits](https://docs.deepinfra.com/account/rate-limits), [Batch documentation index entry](https://docs.deepinfra.com/llms.txt)
- [OpenAI GPT-5.4 nano model page](https://developers.openai.com/api/docs/models/gpt-5.4-nano), [pricing](https://developers.openai.com/api/docs/pricing), [Batch API](https://developers.openai.com/api/docs/guides/batch), [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs), [reasoning](https://developers.openai.com/api/docs/guides/reasoning)
