# Qwen3.6-27B and Gemma 4 31B: API versus Vast.ai cost

**As of:** 2026-07-21, 17:03 America/New_York  
**Decision:** how to generate 400 pilot answers and, potentially, all 7,759 answers without search grounding.

## Answer first

- **For the 400-answer pilot, call APIs.** Exact-model hosted cost is about **$0.52 for Qwen3.6-27B** through DeepInfra (or $0.60 through Alibaba Model Studio) and **$0.065 for Gemma 4 31B** through DeepInfra. Google's own Gemini Developer API exposes exact `gemma-4-31b-it` for **$0**, but only on the free tier; Google says free-tier data may be used to improve its products.
- **For all 7,759 answers, API is still the simplest choice:** about **$10.13 for Qwen** and **$1.26 for Gemma** at the central 400-output-token assumption. Google's free Gemma endpoint remains $0 if its quota and data-use terms are acceptable.
- **Vast can beat Qwen's paid API on raw dollars** if a one-GPU Q4 deployment is already working and sustains the break-even throughput shown below. It is not a clean apples-to-apples comparison: the inexpensive 24 GB Vast path uses Q4 weights, while DeepInfra says its hosted exact models use FP8.
- **Vast is not compelling for Gemma on price.** DeepInfra's exact Gemma endpoint is already extremely cheap, and Google's exact endpoint is free. Use Vast for privacy/control or to test a specific quantization, not to save money on this workload.

## Workload assumptions

These are estimates, not measured token counts:

| Case | Questions | Input/question | Output/question | Total input | Total output |
|---|---:|---:|---:|---:|---:|
| Pilot, central | 400 | 80 | 400 | 32,000 | 160,000 |
| Full, central | 7,759 | 80 | 400 | 620,720 | 3,103,600 |
| Pilot, output cap | 400 | 80 | 768 | 32,000 | 307,200 |
| Full, output cap | 7,759 | 80 | 768 | 620,720 | 5,958,912 |

Thinking/reasoning tokens can increase billed/generated output. The 768-token rows are a conservative cap scenario, not a prediction.

## Exact-model API availability and price

### Qwen3.6-27B

| Provider | Exact endpoint/model | Input / 1M | Output / 1M | Status |
|---|---|---:|---:|---|
| DeepInfra | `Qwen/Qwen3.6-27B` | $0.32 | $3.20 | Public, FP8; exact model metadata and price are returned by the provider's [model API](https://api.deepinfra.com/models/Qwen/Qwen3.6-27B). |
| Alibaba Cloud Model Studio | `qwen3.6-27b`, Singapore | $0.60 | $3.60 | Exact model; official [pricing table](https://www.alibabacloud.com/help/en/model-studio/model-pricing) and [OpenAI-compatible API documentation](https://www.alibabacloud.com/help/en/model-studio/qwen-api-via-openai-chat-completions). The exact 27B is not listed in the Frankfurt or Virginia price sections. |
| Novita | `qwen3.6-27b` | $0.60 | $3.60 | Exact model listed in the provider's [model catalog](https://novita.ai/models); more expensive than DeepInfra here. |
| OVHcloud | `Qwen3.6-27B`, FP8 | $0.64 | $4.32 | Exact model in OVHcloud's [public-cloud price list](https://www.ovhcloud.com/en-ca/public-cloud/prices/); more expensive than DeepInfra here. |

Alibaba is the model maker's first-party paid option. DeepInfra is the lowest exact-model pay-per-token price verified in this review.

### Gemma 4 31B

| Provider | Exact endpoint/model | Input / 1M | Output / 1M | Status |
|---|---|---:|---:|---|
| Google Gemini Developer API | `gemma-4-31b-it` | $0 | $0 | Exact model, but **free tier only**; Paid Tier is “Not available,” and free-tier data may be used to improve Google's products. See [Gemma on the Gemini API](https://ai.google.dev/gemma/docs/core/gemma_on_gemini_api) and [official pricing](https://ai.google.dev/gemini-api/docs/pricing). |
| DeepInfra | `google/gemma-4-31B-it` | $0.13 | $0.38 | Public, FP8; exact model metadata and price are returned by the provider's [model API](https://api.deepinfra.com/models/google/gemma-4-31B-it). |
| Novita | `google/gemma-4-31b-it` | $0.14 | $0.40 | Exact BF16 model and OpenAI-compatible endpoint on the provider's [model page](https://novita.ai/models/model-detail/google-gemma-4-31b-it). |

Google's pricing page does **not** offer a paid Gemma 4 tier. No exact Gemma 4 31B managed pay-per-token Vertex endpoint was verified; Google's paid first-party option should therefore not be assumed. DeepInfra is the lowest verified paid endpoint.

## API cost for this workload

| Model/provider | 400 × 400 output | 7,759 × 400 output | 400 × 768 cap | 7,759 × 768 cap |
|---|---:|---:|---:|---:|
| Qwen / DeepInfra | **$0.52** | **$10.13** | $0.99 | $19.27 |
| Qwen / Alibaba | $0.60 | $11.55 | $1.13 | $21.82 |
| Gemma / Google free tier | **$0** | **$0 if quota permits** | $0 | $0 if quota permits |
| Gemma / DeepInfra paid | **$0.065** | **$1.26** | $0.121 | $2.35 |

Formula: `(input_tokens × input_price + output_tokens × output_price) / 1,000,000`. These totals exclude taxes and any provider-specific free credits.

## Vast.ai: hardware fit and live rental prices

### What fits

The official/community-maintained GGML organization artifacts report:

| Model artifact | Weight file | Implication |
|---|---:|---|
| [Qwen3.6-27B Q4_K_M GGUF](https://huggingface.co/ggml-org/Qwen3.6-27B-GGUF) | 19.10 GB | A 24 GB GPU can hold the Q4 text model for this short-context workload, with limited headroom. |
| [Qwen3.6-27B FP8](https://huggingface.co/Qwen/Qwen3.6-27B-FP8/tree/main) | 30.9 GB repository | A 32 GB card is tight after runtime/KV-cache overhead; use a 48 GB-class GPU for a safer FP8 deployment. |
| [Gemma 4 31B Q4_0 GGUF](https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF) | 17.99 GB | A 24 GB GPU can hold Q4 for this short-context workload. |
| Gemma 4 31B Q8_0 / BF16 GGUF | 32.64 / 61.41 GB | Use 48 GB for Q8 or 80 GB for BF16 plus runtime headroom. |

“Fits” is an inference from weight size and device memory, not a completed Vast deployment test. Disable unused vision components and keep the serving context close to the actual short-answer workload; the models' maximum context settings would require much more KV-cache memory.

### Live on-demand marketplace snapshot

At 2026-07-21 17:03 ET, the first-party [Vast offer-search API](https://docs.vast.ai/api-reference/search/search-offers) returned the following for rentable, one-GPU, reliability ≥0.99 offers. Vast states that marketplace prices change with supply and demand, are billed per second, and that compute, storage, and bandwidth are separate charges ([pricing documentation](https://docs.vast.ai/guides/instances/pricing), [billing documentation](https://docs.vast.ai/guides/reference/billing)).

| GPU | VRAM class | Offers seen | Minimum total $/h | Median total $/h | Suggested use |
|---|---:|---:|---:|---:|---|
| RTX 3090 | 24 GB | 64 | $0.078 | **$0.148** | Cheapest Q4 path; least memory headroom |
| RTX 4090 | 24 GB | 64 | $0.136 | $0.370 | Q4, faster generation class, still limited headroom |
| RTX 5090 | 32 GB | 64 | $0.281 | **$0.428** | Q4 with more headroom; FP8 remains tight |
| RTX A6000 | 48 GB | 4 | $0.312 | $0.389 | Safer Q8/FP8 memory class |
| L40S | 48 GB | 6 | $0.401 | $0.801 | Safer Q8/FP8 memory class |
| A100 SXM4 | 40/80 GB listings | 11 | $0.401 | $0.860 | BF16/large-headroom path, depending on VRAM |

These are a time-stamped observation, not a price guarantee. The API response's total hourly price includes the selected offer's displayed extras, but bandwidth transfer can still vary by host.

## Download and setup time

Vast documents `inet_down` in Mbps and recommends checking it because it affects model downloads ([instance-selection documentation](https://docs.vast.ai/examples/migrations/runpod-to-vast)). In the same snapshot, median advertised download was about 859 Mbps for RTX 3090 offers and 865 Mbps for RTX 5090 offers; observed minima were about 89 and 85 Mbps.

For a 19.1 GB Qwen Q4 file, the ideal transfer-only time is therefore:

- about **3 minutes** at the observed median advertised link;
- about **30 minutes** at an 85–89 Mbps listing.

Gemma's 18.0 GB Q4 is similar. Real cold start is longer because it also includes container/image transfer, package/runtime initialization, model loading, and any debugging. No primary, exact-model Vast measurement was found for those steps, so this report does not claim a “likely” total setup time. Budgeting 15 minutes in the break-even sensitivity below is explicitly an assumption, not a verified benchmark.

## Break-even against the lowest paid API

There is no trustworthy primary-source, exact-model throughput benchmark for these models on a Vast RTX 3090 or RTX 5090. Consequently, the table gives **required throughput**, not a claimed likely throughput. It uses median live GPU rates, a hypothetical 15-minute setup overhead, output generation only, and ignores input-prefill time; it is therefore favorable to Vast.

| Model/workload | API comparator | Vast GPU | API-equivalent total rental time | Required sustained output throughput after 15m setup |
|---|---:|---|---:|---:|
| Qwen, 400 | $0.522 | RTX 3090 @ $0.148/h | 3.53 h | **13.6 tok/s** |
| Qwen, 400 | $0.522 | RTX 5090 @ $0.428/h | 1.22 h | **45.8 tok/s** |
| Qwen, 7,759 | $10.13 | RTX 3090 @ $0.148/h | 68.38 h | **12.7 tok/s** |
| Qwen, 7,759 | $10.13 | RTX 5090 @ $0.428/h | 23.66 h | **36.8 tok/s** |
| Gemma, 400 | $0.065 | RTX 3090 @ $0.148/h | 0.44 h | **235.8 tok/s** |
| Gemma, 400 | $0.065 | RTX 5090 @ $0.428/h | 0.15 h | Impossible: setup alone exceeds API cost |
| Gemma, 7,759 | $1.26 | RTX 3090 @ $0.148/h | 8.51 h | **104.4 tok/s** |
| Gemma, 7,759 | $1.26 | RTX 5090 @ $0.428/h | 2.94 h | **320.1 tok/s** |

Break-even formula:

```text
required_output_tok_per_s = total_output_tokens /
    ((API_cost / Vast_hourly_rate - setup_hours) × 3600)
```

Storage, bandwidth, failed starts, input prefill, and operator time move the break-even further toward APIs. Conversely, an already-cached model, a lower-than-median rental, batching, or reusing the instance for additional work moves it toward Vast.

Google's free Gemma endpoint has no monetary Vast break-even: Vast always costs more than $0. The actual decision there is data-use/privacy and quota, not compute price.

## Recommended execution choice

1. **Pilot without grounding:** use `gemma-4-31b-it` on Google's free Developer API only if sending the questions under its free-tier data-use terms is acceptable; otherwise use DeepInfra for about $0.065.
2. Use DeepInfra's exact `Qwen/Qwen3.6-27B` for the 400-question Qwen pilot, about $0.52. This avoids another local-runtime setup path and preserves FP8 rather than Q4.
3. If both pilots are useful, API cost is only about **$0.59 combined** at the central assumption.
4. For all 7,759, keep Gemma on API. For Qwen, Vast becomes a legitimate dollar-saving option only after a short representative benchmark proves the chosen Q4 runtime exceeds the relevant 13–37 tok/s break-even and answer quality remains acceptable.

## Verified facts versus inferences

**Verified:** exact endpoint names and listed API prices; Google Gemma paid-tier unavailability/data-use statement; model artifact sizes; Vast pricing mechanics; live Vast offer prices and advertised link rates.

**Inferred/calculated:** 80 input and 400/768 output-token workloads; GPU fit from artifact size; ideal download duration; 15-minute setup sensitivity; API totals and break-even throughput. No claim is made that a particular Vast GPU will achieve the threshold until benchmarked with the actual prompt and serving configuration.
