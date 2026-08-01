# Local pointwise judge candidates for Ryzen AI MAX+ 395

Date: 2026-07-26

## Bottom line

For a **local correctness judge**, the most useful two-model smoke comparison is:

1. **Qwen3.6-35B-A3B** as the fast, efficient baseline already shortlisted.
2. **Qwen3.6-27B** as the higher-compute local challenger.

Qwen's own evaluation table reports the dense 27B ahead of the 35B-A3B on the proxies most relevant to factual grading: MMLU-Pro (86.2 vs. 85.2), SuperGPQA (66.0 vs. 64.7), GPQA Diamond (87.8 vs. 86.0), and HLE (24.0 vs. 21.4). Those are **vendor-reported general benchmarks, not judge-calibration results**, so the proposed ten-case smoke test is still necessary. The dense 27B should be materially slower per generated token because all 27B parameters participate, whereas the 35B MoE activates only 3B; that speed statement is an architectural inference, not a measured result on this host. [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B#model-overview) [Qwen3.6-35B-A3B model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B#model-overview)

For **model-family diversity**, the best additional local candidate is **gpt-oss-120b**. It is a 117B-total/5.1B-active reasoning model with native structured outputs, 128K context, and a native MXFP4 release intended to run in 80 GB. That makes it an unusually credible independent local reviewer on this 128 GB unified-memory machine. [OpenAI release](https://openai.com/index/introducing-gpt-oss/) [OpenAI model card](https://openai.com/index/gpt-oss-model-card/)

## Host and runtime facts

- AMD identifies the Radeon 8060S in the Ryzen AI MAX+ 395 as **RDNA 3.5 / gfx1151**. Current ROCm documentation lists gfx1151 and this APU as supported; for ROCm 7.2.1, PyTorch 2.9.1 is production-supported, although AMD says only FP16 has been formally validated. [GPU specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) [ROCm Linux matrix](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityryz/native_linux/native_linux_compatibility.html)
- AMD documents up to 96 GB as BIOS-configured graphics memory on 128 GB systems and a Linux TTM path up to 120 GB. Large-context KV cache and runtime overhead must still be left outside model weights. [AMD Ryzen AI MAX+ local inference article](https://www.amd.com/en/developer/resources/technical-articles/2026/how-to-run-a-one-trillion-parameter-llm-locally-an-amd.html) [RDNA3.5 system optimization](https://rocm.docs.amd.com/en/latest/reference/system-optimization/rdna3-5.html)
- AMD demonstrates `llama.cpp` on this processor family, and the `llama.cpp` issue tracker contains a concrete gfx1151 run of Qwen3.5-122B-A10B Q4 with 126,976 MiB visible unified VRAM. That is evidence that large GGUF MoEs can execute on the architecture, not a guarantee that every newer architecture or kernel combination will work. [AMD llama.cpp example](https://www.amd.com/en/blogs/2025/amd-ryzen-ai-max-upgraded-run-up-to-128-billion-parameter-llms-lm-studio.html) [llama.cpp gfx1151 report](https://github.com/ggml-org/llama.cpp/issues/20575)
- The repository's earlier host-specific assessment recommends Lemonade's `llamacpp:rocm` path for Qwen3.6 rather than treating vLLM on gfx1151 as proven: [`reports/lemonade-qwen36-gfx1151-runtime-assessment.md`](../../reports/lemonade-qwen36-gfx1151-runtime-assessment.md).

## Candidate comparison

| Candidate | Verified architecture and context | Verified weight footprint | License | Local-runtime assessment | Value as judge |
|---|---|---:|---|---|---|
| **Qwen3.6-35B-A3B** | 35B total, 3B active; 262,144 native, extensible to 1,010,000 | Official BF16 repository: **71.9 GB** | Apache-2.0 | **Best-supported first run** through the already assessed Lemonade/llama.cpp ROCm route. Quantized GGUF is preferable for ample KV/runtime headroom, but Qwen does not publish a first-party GGUF size on the base card. | Fast local baseline; good instruction following and strong general knowledge. Its error correlation with the hosted Qwen3-235B judge should be measured before retaining both. |
| **Qwen3.6-27B** | Dense 27B; 262,144 native, extensible to 1,010,000 | Official BF16 repository: **55.6 GB** | Apache-2.0 | Fits even at BF16 by file size, although runtime/KV overhead requires a conservative context. Community GGUF quantization should be smaller, but no third-party size is treated as verified here. Architecture support should be checked in the exact llama.cpp/Lemonade build before download. | **Best immediate challenger.** Qwen reports stronger knowledge/reasoning than 35B-A3B; likely slower, useful for testing whether extra active capacity improves grading. |
| **gpt-oss-120b** | 117B total, 5.1B active; 128K | OpenAI says native MXFP4 runs in **80 GB** | Apache-2.0 plus gpt-oss usage policy | Fits the 128 GB host with reasonable headroom. OpenAI lists llama.cpp among supported deployment partners; exact gfx1151 throughput is unverified. Harmony prompting must be rendered correctly. | **Best independent-family local judge.** Reasoning controls and structured outputs are directly useful for a strict JSON grading contract. |
| **GLM-4.7-Flash** | 30B-A3B (HF reports 31B total); output allowance shown as 131,072 tokens | llama.cpp organization's Q8_0 GGUF: **31.8 GB** | MIT | Official `ggml-org` GGUF and Lemonade invocation exist. However, llama.cpp has recorded HIP/Vulkan performance and flash-attention/KV-cache issues for this architecture; use a short context and conservative KV settings first. gfx1151 itself is not directly verified. | Valuable independent lineage and cheap footprint. Vendor-reported GPQA 75.2 trails the Qwen3.6 figures, so treat it as a diversity candidate rather than the presumptive winner. |
| **NVIDIA-Nemotron-3-Nano-30B-A3B** | 30B total, 3.5B active; default 256K, supports up to 1M | Official BF16: **63.2 GB**; official NVFP4: **19.4 GB** | NVIDIA Nemotron Open Model License | BF16 fits by size, but NVIDIA officially lists H100/A100 integration. The 19.4 GB NVFP4 artifact is NVIDIA-specific and should **not** be assumed usable on AMD. llama.cpp quantizations exist through the model-card quantization browser, but no first-party gfx1151 validation was found. | A useful independent reasoning model, but weaker local portability than Qwen/GLM/gpt-oss. NVIDIA reports MMLU-Pro 78.3 and GPQA 73.0 for BF16, below the Qwen3.6 proxies. |
| **Qwen3.5-122B-A10B** (stretch) | 122B total, 10B active; 262,144 native, extensible to 1,010,000 | Official BF16: **250 GB**, so quantization is mandatory | Apache-2.0 | A Q4 GGUF has concretely run on gfx1151 in llama.cpp. The exact community quant and footprint must be pinned and verified before use; full BF16 cannot fit. | Highest-capacity plausible local Qwen. Vendor results are strong (MMLU-Pro 86.7, GPQA 86.6), but its larger download, slower active compute, and same-family redundancy make it unsuitable for the first smoke test. |

Primary sources: [Qwen3.6-35B-A3B files](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/tree/main), [Qwen3.6-27B files](https://huggingface.co/Qwen/Qwen3.6-27B/tree/main), [gpt-oss release](https://openai.com/index/introducing-gpt-oss/), [GLM-4.7-Flash model card](https://huggingface.co/zai-org/GLM-4.7-Flash), [`ggml-org/GLM-4.7-Flash-GGUF`](https://huggingface.co/ggml-org/GLM-4.7-Flash-GGUF), [Nemotron BF16 model card and files](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16), [Nemotron NVFP4 files](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4/tree/main), [Qwen3.5-122B-A10B model card and files](https://huggingface.co/Qwen/Qwen3.5-122B-A10B).

## Recommendation for this experiment

Do **not** expand the first ten-case smoke to every local candidate. Use one local slot in the already planned five-model comparison, but change its role as follows:

- Run **Qwen3.6-35B-A3B** first because its ROCm/Lemonade path has already been assessed and it is cheap to iterate.
- If its JSON reliability or grading quality is weak, replace it with **Qwen3.6-27B** before scaling to 500.
- If it is strong, run the same ten cases through **gpt-oss-120b** as a local-family-diversity check before deciding whether the larger download/runtime is justified.
- Keep **GLM-4.7-Flash** as the compact independent fallback. Defer Nemotron Nano and Qwen3.5-122B unless the first three local choices expose a clear gap.

This ordering optimizes the question the smoke test can actually answer: whether a fast local judge is reliable enough, and whether a more computationally expensive or independent local judge materially changes the decisions. It is not a claim that generic benchmark rank predicts judge accuracy.
