# TREC 2026 reference-answer model landscape

**As of:** 2026-07-21  
**Scope:** Models for (1) generating reference answers, (2) pointwise/pairwise judging, and (3) running locally on an AMD Ryzen AI MAX+ 395 with 128 GB shared memory.  
**Evidence policy:** Provider claims and specifications below come only from official documentation, official model cards, or official runtime documentation. Hardware-fit and task-fit recommendations are explicitly labeled as inference.

## Answer first

1. **Change the cloud challenger from Gemini 3.5 Flash to Gemini 3.6 Flash.** Google released `gemini-3.6-flash` today as a stable model. Google reports that it is better than 3.5 Flash on knowledge work, uses 17% fewer output tokens on one broad external index, takes fewer reasoning/tool steps, and lowers output pricing from $9 to $7.50 per million tokens (standard) or from $4.50 to $3.75 (Batch/Flex). It supports Search grounding, URL context, code execution, thinking, structured output, and Batch. This is unusually direct evidence that the new model is both the better and cheaper replacement for the model currently frozen in this repository. [Gemini 3.6 model page](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash), [Google launch post](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-6-flash-3-5-flash-lite-3-5-flash-cyber/), [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)

2. **For the 400-question validation set, 3.6 Flash is cheap enough to use as the primary cloud challenger.** Under a task-shaped estimate of 80 input tokens and 400 billable output/thinking tokens per answer, 400 answers cost about **$1.25 standard** or **$0.62 Batch/Flex**, before Search charges. Even the current 768-token output cap is only about **$2.35 standard** or **$1.18 Batch/Flex** for 400 answers.

3. **For all 7,759 questions, model inference still fits the stated budget; blanket Search may not.** At the same 80-in/400-out estimate, 3.6 Flash costs about **$24.21 standard** or **$12.10 Batch/Flex**. At the full 768-token cap, it is about **$45.62 standard** or **$22.81 Batch/Flex**. Google Search grounding includes 5,000 Gemini 3 prompts per month, shared across Gemini 3 models, then costs $14 per 1,000 search queries. If every question caused exactly one query and the allowance were unused, the excess 2,759 queries would add about **$38.63**; a request may cause more than one query. Therefore, Search should be used for detected time-sensitive or uncertain questions, not automatically for all 7,759. [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)

4. **The best local first retry is Qwen3.6-27B in BF16 or a supported GGUF quantization—not Qwen3.6-35B-A3B.** This is an inference, but it is supported by Qwen's own same-table evaluations: Qwen3.6-27B beats Qwen3.6-35B-A3B on MMLU-Pro (86.2 vs 85.2), GPQA Diamond (87.8 vs 86.0), HLE (24.0 vs 21.4), and most reported coding/agent benchmarks. The 27B model is dense and its official BF16 repository is 55.6 GB, making it simpler than the 35B MoE model whose FP8 backend failed locally. [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B), [Qwen3.6-27B files](https://huggingface.co/Qwen/Qwen3.6-27B/tree/main)

5. **Gemma 4 31B is the strongest materially different local control, but not the evidence-backed favorite.** Its official card reports strong reasoning (MMLU-Pro 85.2, GPQA Diamond 84.3) and explicitly supports configurable thinking. However, Qwen's same evaluation table places Qwen3.6-27B ahead on the relevant knowledge and reasoning metrics. Gemma 4 is valuable as an architectural/model-family diversity check, not as the expected winner. [Gemma 4 31B model card](https://huggingface.co/google/gemma-4-31B-it)

6. **Use Gemini 3.5 Flash-Lite for high-volume mechanical judging only after calibration.** It is the cheapest current high-throughput Gemini option ($0.30 input/$2.50 output per million standard; half that in Batch/Flex), and Google reports 350 output tokens/s and significant gains over 3.1 Flash-Lite. But Google does not publish task-specific evidence that it matches 3.6 Flash for factual reference-answer evaluation. A sensible cascade is: local Qwen3.6-27B or 3.5 Flash-Lite for clear pointwise/pairwise cases, then 3.6 Flash for disagreements and difficult/time-sensitive cases. [Google launch post](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-6-flash-3-5-flash-lite-3-5-flash-cyber/), [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing)

## Current Gemini choices

| Model | Official status and task-relevant capabilities | Standard price / 1M tokens | Batch/Flex price / 1M tokens | Recommended role |
|---|---|---:|---:|---|
| **Gemini 3.6 Flash** (`gemini-3.6-flash`) | Stable; 1,048,576-token input, 65,536-token output; Search, Maps, URL context, file search, code execution, thinking, structured outputs, function calling, and Batch supported | $1.50 input / $7.50 output | $0.75 / $3.75 | Primary cloud reference generator and escalation judge |
| Gemini 3.5 Flash (`gemini-3.5-flash`) | Stable predecessor; Google says 3.6 improves knowledge work and token efficiency | $1.50 / $9.00 | $0.75 / $4.50 | Superseded for this pilot; retain only for an explicit longitudinal control |
| **Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`) | GA; designed for high-volume agentic tasks, translation, and simple processing; configurable thinking; Google reports 350 output tokens/s | $0.30 / $2.50 | $0.15 / $1.25 | Cheap first-pass classification/judging after calibration |
| Gemini 3.1 Pro Preview (`gemini-3.1-pro-preview`) | Preview; Google's premium reasoning family; Search supported | $2.00 / $12.00 for prompts <=200K | $1.00 / $6.00 | Small hard-case adjudication set only if 3.6 Flash remains uncertain |

Sources: [official pricing](https://ai.google.dev/gemini-api/docs/pricing), [3.6 Flash specification](https://ai.google.dev/gemini-api/docs/models/gemini-3.6-flash), [model version semantics](https://ai.google.dev/gemini-api/docs/models).

### Is 3.6 Flash actually better than 3.5 Flash?

**Verified provider evidence:** yes, on Google's published comparisons. Google says 3.6 Flash:

- uses 17% fewer output tokens than 3.5 Flash on the Artificial Analysis Index;
- takes fewer reasoning steps and tool calls;
- improves GDPval-AA v2 knowledge work from 1349 to 1421;
- lowers the output-token price from $9 to $7.50 per million;
- is released as the stable identifier `gemini-3.6-flash`.

Source: [Google's 2026-07-21 launch post](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-6-flash-3-5-flash-lite-3-5-flash-cyber/).

**Unknown:** Google has not published a benchmark matching this dataset's exact mixture of short factual, ambiguous, historical, technical, and time-sensitive questions. Provider benchmarks are not a substitute for the planned 400-question review.

**Inference:** the evidence is strong enough to replace 3.5 Flash in the pilot, but the first 20–30 calls should be treated as a schema/behavior canary because the model launched today.

## Cost envelopes for this repository

The local Perplexity cache contains 7,759 questions. The mean question length is 69 characters; the 7,688 successful citation-stripped answers average 1,594 characters. The estimates below use **80 input tokens per request** (system prompt plus a short question) and **400 billable output/thinking tokens per answer** as a round task-shaped estimate. The repository currently caps generation at 768 output tokens. Actual charges must be taken from API usage metadata because thinking tokens are billed as output.

### Reference-answer generation

| Workload | 3.6 Flash standard, 400 output | 3.6 Flash Batch/Flex, 400 output | 3.6 Flash standard, full 768 cap | 3.6 Flash Batch/Flex, full 768 cap |
|---|---:|---:|---:|---:|
| 400 questions | ~$1.25 | ~$0.62 | ~$2.35 | ~$1.18 |
| 7,759 questions | ~$24.21 | ~$12.10 | ~$45.62 | ~$22.81 |

These are arithmetic estimates, not quotes. They exclude retries, tool-generated tokens, and Search charges.

### Pointwise/pairwise judging

For illustration, assume each judgment sends 1,000 input tokens (reference plus candidate answer(s) and rubric) and returns 50 output/thinking tokens:

| 7,759 judgments | Estimated standard cost | Estimated Batch/Flex cost |
|---|---:|---:|
| Gemini 3.6 Flash | ~$14.55 | ~$7.27 |
| Gemini 3.5 Flash-Lite | ~$3.30 | ~$1.65 |
| Gemini 3.1 Pro Preview | ~$20.17 | ~$10.09 |

**Inference:** model-token cost is not the limiting factor for a single judgment per question. The count of candidate pairs can be: if there are `P` pairwise comparisons per question, multiply the corresponding estimate by `P`.

### Search grounding

Verified pricing facts:

- Gemini 3 models receive 5,000 Search-grounded prompts per month, shared across the family.
- After the allowance, Google lists $14 per 1,000 Search queries.
- A single submitted request may cause one or more Search queries, each potentially billable after the allowance.
- Retrieved Search grounding context is not charged as input tokens.

Source: [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing).

Cost examples, assuming the monthly allowance is unused:

- 400 grounded questions: **$0 Search charge**, because they fit inside 5,000.
- 7,759 grounded questions with exactly one query each: **about $38.63 Search charge** for 2,759 excess queries.
- Real cost can be higher if requests issue multiple queries, or lower if the model does not search on every request.

**Recommendation (inference):** Ground all 400 validation questions if desired; that is an inexpensive way to assess grounded answer quality. For the 7,759-question full pass, route Search only to time-sensitive, current-entity, obscure, or low-confidence questions. The existing Perplexity answers and citations can resolve many disagreements without paying for a second web retrieval on every item.

## Downloadable local candidates

### 1. Qwen3.6-27B — preferred local model

Verified facts from the [official model card](https://huggingface.co/Qwen/Qwen3.6-27B):

- Dense 27B language model with vision encoder; 262,144-token native context.
- Official repository is 55.6 GB in BF16 weights.
- Officially documented with Transformers, vLLM, SGLang, KTransformers, Docker Model Runner, and links to GGUF quantizations.
- Qwen's own evaluation table reports stronger results than Qwen3.6-35B-A3B on MMLU-Pro, GPQA Diamond, HLE, and most listed coding/agent tests.

Hardware/task inference:

- BF16 weights fit comfortably inside 128 GB total shared memory on paper, with much more headroom than a 122B-class model.
- Dense architecture avoids the unsupported FP8 MoE-kernel failure already observed with Qwen3.6-35B-A3B.
- The prior local FP8 attempt is not evidence that BF16 or GGUF is incompatible; it is evidence that the tested FP8/runtime path was unsuccessful.
- Start with short text-only context. A 262K context is unnecessary for these short questions and wastes KV-cache memory.
- For this exact task, Qwen3.6-27B should be the local generator and local judge candidate.

### 2. Gemma 4 31B-it — strongest diverse control

Verified facts from the [official model card](https://huggingface.co/google/gemma-4-31B-it):

- Dense 30.7B model with 256K context, configurable thinking, native system prompts, function calling, and Apache-2.0 weights.
- Google reports MMLU-Pro 85.2, GPQA Diamond 84.3, HLE without tools 19.5, and HLE with Search 26.5.
- The official instructions support current Transformers, vLLM, and SGLang.

Hardware/task inference:

- BF16 weights are roughly 61 GB before runtime overhead, so the model fits the host's 128 GB shared memory on paper.
- It is likely slower than an MoE model per generated token but is architecturally simpler.
- Qwen3.6-27B has better published task-adjacent knowledge/reasoning scores, so Gemma 4 31B is best used for disagreement diversity, not as the primary local model.

### 3. Gemma 4 26B-A4B-it — throughput-oriented diverse control

Verified facts from the same [official Gemma 4 card](https://huggingface.co/google/gemma-4-31B-it):

- 25.2B total / 3.8B active MoE, 256K context.
- Google describes it as running almost as fast as a 4B model because only a subset of parameters is active.
- Published scores trail Gemma 4 31B: MMLU-Pro 82.6, GPQA Diamond 82.3, HLE 8.7 without tools and 17.2 with Search.

Hardware/task inference:

- It may provide attractive throughput if the AMD runtime implements its MoE kernels correctly.
- The existing Qwen FP8 MoE failure means MoE support must be demonstrated with a one-question canary before any large download/run.
- Quality is likely too low to replace Qwen3.6-27B for the main reference challenger, but it could be a cheap local judge after calibration.

### 4. Qwen3.5-122B-A10B — higher-capacity but poor practical fit

Verified facts from the [official model card](https://huggingface.co/Qwen/Qwen3.5-122B-A10B):

- 122B total / 10B active MoE; Qwen publishes BF16, FP8, and GPTQ-Int4 variants.
- It slightly exceeds Qwen3.6-27B on some knowledge tests in separate Qwen tables (for example, MMLU-Pro 86.7 vs 86.2), but Qwen3.6-27B leads on several newer reasoning/agent results.
- Qwen's documented reference deployment uses eight GPUs for the full model.

Hardware/task inference:

- The official GPTQ-Int4 checkpoint may fit numerically in 128 GB, but runtime workspace, KV cache, shared-memory pressure, and the tested AMD MoE/quantization limitations make it a high-risk choice.
- Any modest quality gain is unlikely to justify download, startup, and debugging time for a three-day reference-answer deadline.
- Do not make this the next experiment unless Qwen3.6-27B and Gemma 4 31B both prove inadequate.

## Runtime implications on Ryzen AI MAX+ 395

Verified runtime facts:

- Lemonade documents GGUF models through llama.cpp with CPU, Vulkan, or ROCm; its ROCm llama.cpp backend explicitly lists Ryzen AI Strix Halo as supported. [Lemonade llama.cpp backend](https://lemonade-server.ai/docs/guide/configuration/llamacpp/)
- Lemonade describes vLLM ROCm on Linux/gfx1151 as **experimental**. [Lemonade FAQ](https://lemonade-server.ai/docs/guide/faq/), [vLLM backend](https://lemonade-server.ai/docs/guide/configuration/vllm/)
- AMD warns that some LLM workloads can show lower-than-expected performance on Ryzen AI MAX+ 395. [AMD Ryzen limitations](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/limitations/limitationsryz.html)

Inference from those documents and the preserved local failures:

- Lemonade improves packaging and exposes more backends; it does not make an unsupported vLLM FP8 MoE kernel supported.
- A GGUF/llama.cpp ROCm or Vulkan path is a genuinely different backend and therefore the most useful alternative after the failed vLLM FP8 path.
- A dense BF16 test of Qwen3.6-27B is lower risk than repeating Qwen3.6-35B-A3B FP8/BF16 startup experiments.
- Every model/runtime combination still needs a cheap readiness canary: load, answer one short question, verify output and memory, then run 10 representative questions before scaling.

## Proposed model allocation

This is a recommendation, not a verified result:

1. **Cloud reference challenger:** Gemini 3.6 Flash, not 3.5 Flash.
2. **Web grounding:** enable for the 400-question validation sample; for a 7,759 run, route only questions classified as time-sensitive/obscure/uncertain.
3. **Local reference challenger:** Qwen3.6-27B, first via a text-only dense BF16 path; if vLLM remains unstable, use a pinned GGUF through Lemonade/llama.cpp ROCm or Vulkan.
4. **Diversity challenger:** Gemma 4 31B-it on a small subset or disagreement set.
5. **Pointwise/pairwise judge:** calibrate local Qwen3.6-27B and Gemini 3.5 Flash-Lite against 50–100 manually reviewed cases. Use the cheaper model for obvious cases and Gemini 3.6 Flash for disagreements/low-confidence cases.
6. **Do not spend deadline time on:** Qwen3.5-122B-A10B, another FP8-MoE vLLM attempt, or blanket paid Search for all questions.

## Immediate repository gap

The implemented Gemini adapter is currently hard-frozen to:

```python
GEMINI_MODEL = "gemini-3.5-flash"
```

and rejects any other Gemini model. Therefore, selecting 3.6 Flash requires a small code/test change before calls can be made. Because 3.6 Flash launched today, the safe sequence is:

1. update the pinned model and expected test fixtures;
2. run one no-Search and one Search-grounded canary;
3. verify response parsing, finish reasons, usage metadata, and actual cost;
4. run 20–30 questions and inspect answer behavior;
5. proceed to 400 only after the canary passes.

## Bottom line

- **Best current Gemini choice:** Gemini 3.6 Flash. It is officially better and cheaper than the planned 3.5 Flash.
- **Cheapest plausible bulk judge:** Gemini 3.5 Flash-Lite, but only after calibration.
- **Best local first choice:** Qwen3.6-27B; Qwen's own results make it a stronger candidate than 35B-A3B for this task, despite the smaller headline size.
- **Best non-Qwen local control:** Gemma 4 31B-it.
- **Most important cost lever:** selective Search grounding, not model-token pricing.
- **Most important methodological safeguard:** do not infer reference-answer or judge quality from general benchmarks alone; use the already designed 400-question reviewed pilot.
