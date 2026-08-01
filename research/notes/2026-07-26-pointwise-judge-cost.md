# Pointwise LLM-judge cost snapshot (2026-07-26)

Snapshot time: 2026-07-26 08:08 EDT. All prices are USD and come from provider-owned pricing pages, documentation, model cards, or Vast.ai's official marketplace API. They will change; rerun the Vast query and recheck pricing pages before committing a large run.

## Bottom line

- For an offline pointwise run, hosted batch/flex is the simplest low-cost baseline. At **1,000 uncached input tokens + 50 billed output/thinking tokens per judgment**, representative standard prices are **$0.154/1,000 judgments** for Fireworks DeepSeek V4 Flash, **$0.1825/1,000** for Together Qwen3.5-9B, **$0.2625/1,000** for OpenAI GPT-5.4 nano, **$0.325/1,000** for Gemini 3.1 Flash-Lite, and **$1.25/1,000** for Claude Haiku 4.5. Where supported, batch/flex is generally half price.
- Vast.ai verified, rentable, on-demand spot offers in this snapshot started at **$0.263/h for 1x RTX 4090**, **$0.308/h for 1x RTX 5090**, **$0.402/h for 1x RTX A6000 48 GB**, **$0.601/h for 1x A100 SXM4 40 GB**, and **$1.736/h for 1x H100 SXM 80 GB**, including the offer's default storage allocation. These are marketplace observations, not durable list prices.
- Do not compare an API token quote to a GPU hourly quote without measuring the actual judge pipeline. For observed throughput `J` judgments/second and hourly offer cost `H`, Vast compute cost is `H / (3600*J)` per judgment. Include provisioning, idle time, storage, bandwidth, retries, and invalid-output repair.

## Cost formulas

For `N` judgments, average uncached input `I` tokens, billed output (including hidden reasoning where applicable) `O` tokens, and prices `Pin`, `Pout` per million tokens:

```text
API cost = N * (I*Pin + O*Pout) / 1,000,000
```

Add cached-token/storage/tool fees separately. The worked comparison below uses `I=1,000`, `O=50`, no caching, tools, or retries. For reasoning models, use the provider's usage counters: a 20-token visible JSON object can still incur many billed thinking tokens.

For self-hosting:

```text
Vast compute cost = H * wall_clock_hours
cost/judgment = H / (3600*J)
serial lower-bound time ~= I/prefill_tokens_per_sec + O/decode_tokens_per_sec
total = compute + storage_lifetime + ingress + egress + failed/retried work
```

Use a representative end-to-end benchmark with the real prompt length, constrained decoder, batch/concurrency, and output length. Model-only tokens/second is not enough.

## Vast.ai rental evidence

The [official pricing documentation](https://docs.vast.ai/guides/instances/pricing) says Vast is a live marketplace with no static quotes; hosts set rates, compute is billed by the second, and total price combines compute, storage, and bandwidth. The [billing documentation](https://docs.vast.ai/guides/reference/billing) also warns that storage continues while an instance exists and bandwidth is metered separately. [Search Offers](https://docs.vast.ai/api-reference/search/search-offers) is the official API for current offers.

The table is a snapshot of the official `POST https://console.vast.ai/api/v0/bundles/` response filtered to `verified=true`, `rentable=true`, `rented=false`, on-demand, and exact GPU count/name. `total` is `dph_total` (GPU base plus the offer's default allocated storage), not bandwidth.

| Configuration | Offers | Lowest total/h | Median total/h | Highest total/h | Lowest offer ID |
|---|---:|---:|---:|---:|---:|
| 1x RTX 4090, 24 GB | 57 | $0.263 | $0.375 | $1.337 | 41267735 |
| 1x RTX 5090, 32 GB | 55 | $0.308 | $0.470 | $0.976 | 43050293 |
| 1x RTX A6000, ~48 GB | 2 | $0.402 | $0.404 | $0.404 | 31805856 |
| 1x L40S, ~48 GB | 4 | $0.801 | $0.804 | $1.074 | 38574933 |
| 1x A100 SXM4, 40 GB | 14 | $0.601 | $0.935 | $1.592 | 42974676 |
| 1x H100 SXM, 80 GB | 10 | $1.736 | $3.069 | $4.396 | 40179473 |
| 2x RTX 4090, 48 GB aggregate | 44 | $0.589 | $0.775 | $2.670 | 30714839 |
| 2x RTX 5090, 64 GB aggregate | 44 | $0.623 | $0.986 | $1.873 | 45211926 |

Hardware interpretation (inference, not a measured serving result):

- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) has 8.2B parameters. Raw BF16 weights are about `8.2B*2 = 16.4 GB`, so a 24 GB 4090 is a reasonable benchmark starting point after allowing for runtime/KV overhead.
- [Gemma 3 27B IT](https://huggingface.co/google/gemma-3-27b-it) or [Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B) needs roughly 54–64 GB for raw BF16 weights, before overhead. A 4-bit build can fit a 24/32 GB card at short contexts; BF16 points toward an 80 GB GPU.
- [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) has 72.7B parameters and explicitly emphasizes JSON structured output. Raw BF16 weights are about 145 GB; 8-bit is roughly 73 GB before overhead; 4-bit is roughly 36 GB plus quantization metadata/KV/runtime. A 48 GB card or 2x24 GB can be a tight short-context 4-bit target; 80 GB is the safer 4/8-bit target, and BF16 generally needs 2x80 GB. Quantization may change judge quality, so validate agreement/calibration rather than assuming equivalence.

Vast caveats: a cheap offer may have lower reliability, limited rental duration, slow disk/network, or an unsuitable CPU/RAM allocation. Interruptible offers can be cheaper but may be preempted; the table intentionally uses on-demand. Model download time and idle warm-up are paid wall time.

## Hosted structured-output options

Prices are per 1M tokens. `Example/1k` applies the 1,000-input/50-output-token workload to 1,000 judgments.

| Provider/model | Standard input / output | Offline discount | Example/1k, standard | Structured-output caveat |
|---|---:|---:|---:|---|
| Google `gemini-3.1-flash-lite` | $0.25 / $1.50 | Batch or Flex: $0.125 / $0.75 | $0.325 | JSON Schema subset; syntax does not guarantee semantic correctness. |
| Google `gemini-3.5-flash-lite` | $0.30 / $2.50 | Batch or Flex: $0.15 / $1.25 | $0.425 | Output price includes thinking tokens. |
| OpenAI `gpt-5.4-nano` | $0.20 / $1.25; cached input $0.02 | Check Batch eligibility at run time | $0.2625 | Model page lists ranking/classification and Structured Outputs support. |
| Anthropic `claude-haiku-4-5` | $1.00 / $5.00 | Batch: $0.50 / $2.50 | $1.25 | Schema prompt adds input tokens; refusals/truncation can violate schema. |
| Together `Qwen/Qwen3.5-9B` | $0.17 / $0.25 | Batch is 50% on selected models | $0.1825 | Catalog support required; repeat the schema instruction in the prompt. |
| Together `Llama-3.3-70B` | $1.04 / $1.04 | Batch is 50% on selected models | $1.092 | Confirm structured-output catalog flag for the exact route. |
| Fireworks `DeepSeek V4 Flash` | $0.14 / $0.28 | Batch: 50% | $0.154 | Repeat schema in prompt; JSON-schema mode disables reasoning output. |
| Fireworks `gpt-oss-120B` | $0.15 / $0.60 | Batch: 50% | $0.180 | Same schema/truncation caveats. |

### Provider evidence and caveats

- **Google:** [Gemini API pricing](https://ai.google.dev/gemini-api/docs/pricing) lists Standard, Batch, Flex, cache, and thinking-inclusive output prices. [Structured outputs](https://ai.google.dev/gemini-api/docs/structured-output) supports only a JSON Schema subset and warns that schema-valid output may still be semantically wrong. Paid-tier content is not used to improve Google's products; the free tier is marked as used for improvement. Batch/Flex are half price; Flex is best-effort/sheddable and Batch targets offline completion (see [optimization modes](https://ai.google.dev/gemini-api/docs/optimization)).
- **OpenAI:** [GPT-5.4 nano](https://developers.openai.com/api/docs/models/gpt-5.4-nano) is explicitly positioned for classification, data extraction, and ranking and supports Structured Outputs. The [current API pricing](https://developers.openai.com/api/docs/pricing) also lists the more expensive GPT-5.6 family and cache/long-context/regional multipliers. For 5.6 short-context Standard: Luna is $1/$6, Terra $2.50/$15, Sol $5/$30 input/output per million.
- **Anthropic:** the [official pricing table](https://platform.claude.com/docs/en/about-claude/pricing) lists Haiku 4.5 at $1/$5 and Sonnet 4.6 at $3/$15; Batch halves base input/output. Sonnet 5 has a temporary $2/$10 rate through 2026-08-31, but the current [structured-output documentation](https://platform.claude.com/docs/en/build-with-claude/structured-outputs) does not list it among supported models, so do not assume support. The guide also documents schema-compilation caching and refusal/max-token failure cases.
- **Together:** [first-party pricing](https://www.together.ai/pricing) lists Qwen3.5-9B at $0.17/$0.25, Llama 3.3 70B at $1.04/$1.04, and Qwen2.5 7B Instruct Turbo at $0.30/$0.30. [Structured outputs](https://docs.together.ai/docs/inference/chat/structured-outputs) is model-dependent and requires both `response_format` and a prompt instruction; max-token truncation can still yield invalid JSON. [Batch pricing](https://docs.together.ai/docs/inference/pricing) is 50% for selected serverless models.
- **Fireworks:** [serverless pricing](https://docs.fireworks.ai/serverless/pricing) lists model-specific input/cached/output rates; [Fireworks pricing](https://fireworks.ai/pricing) says batch is half serverless input/output. [Structured responses](https://docs.fireworks.ai/structured-responses/structured-response-formatting) supports a subset of JSON Schema, requires the prompt to repeat the schema, and warns about truncation; `json_schema` disables reasoning output. On-demand dedicated H100/H200 is $7/h, far above the sampled Vast marketplace rates.
- **OpenRouter (optional aggregator):** current first-party model pages display [GPT-OSS 120B](https://openrouter.ai/openai/gpt-oss-120b) at $0.03/$0.17 and [Gemini 3.1 Flash-Lite](https://openrouter.ai/google/gemini-3.1-flash-lite) at $0.25/$1.50, but the displayed lowest route is not a guaranteed provider. [OpenRouter pricing](https://openrouter.ai/pricing) and [FAQ](https://openrouter.ai/docs/faq) describe provider pass-through inference plus a 5.5% credit-purchase fee (minimum $0.80). For reproducibility, pin a provider/max price and use [`require_parameters: true`](https://openrouter.ai/docs/guides/features/structured-outputs) so routing does not fall back to a provider without schema support.

## Practical decision rule

1. Measure token counts and schema/retry rate on 100–1,000 representative pairs.
2. Run the same slice through a candidate hosted model and a quantized open judge; compare agreement, calibration, invalid JSON, latency, and cost—not cost alone.
3. Benchmark Vast end-to-end. At the $0.263/h 4090 snapshot, break-even against Gemini 3.1 Flash-Lite Standard (`$0.000325/judgment` for the worked shape) is only about **0.225 judgments/s** before setup/idle/storage overhead; against Gemini Batch it is **0.450 judgments/s**. This low arithmetic threshold does not imply the 8B judge has comparable quality.
4. Keep output minimal (score, label, confidence, short rationale if needed), disable unneeded thinking, validate/repair JSON, and persist raw usage counters for an auditable final bill.
