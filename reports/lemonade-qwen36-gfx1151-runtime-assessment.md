# Lemonade vs. vLLM ROCm for Qwen3.6 on gfx1151

**Assessment date:** 2026-07-21  
**Host:** AMD Ryzen AI MAX+ 395 / Radeon 8060S, `gfx1151`, 122 GiB visible system memory  
**Workload:** Qwen3.6-35B-A3B-FP8 (MoE) and Qwen3.6-27B-FP8 (dense)

## Bottom line

**Lemonade is a credible alternative only because it offers a different inference path: llama.cpp with a Qwen3.6 GGUF checkpoint. It is not evidence that the failed FP8-MoE vLLM kernel configuration becomes reliable merely by placing Lemonade in front of it.**

Recommended order:

1. **For the required 35B-A3B model, use Lemonade's built-in `Qwen3.6-35B-A3B-GGUF` on `llamacpp:rocm` stable** (UD-Q4_K_XL, about 23.3 GB). If ROCm misbehaves, use Lemonade's supported Vulkan backend with the same GGUF. This is the most defensible operational path on this host.
2. **Before authorizing that download, smoke-test the already-installed Lemonade/llama.cpp ROCm stack with the already-downloaded 0.38 GB `Qwen3-0.6B-GGUF`.** This tests server-to-backend-to-GPU-to-token generation without another model download, though it does not prove Qwen3.6 MoE support.
3. **Treat Lemonade v11's Qwen3.6-35B-A3B-FP8 recipe as a bounded diagnostic, not the primary plan.** It is an official built-in recipe, and Lemonade says it is not architecture-restricted, but the documented Qwen3.6 FP8/MTP validation was on MI300X (`gfx942`), not RDNA/gfx1151. Lemonade calls its entire vLLM backend experimental and documents slow, untuned FP8 startup on gfx1151.
4. **Qwen3.6-27B-FP8 via vLLM remains a reasonable secondary experiment** because it is dense and therefore avoids the exact `select_fp8_moe_backend` failure encountered by the 35B-A3B MoE model. It is not the cheapest next step: the 27B checkpoint is not present locally and is about 27.8 GB. It also does not answer whether the requested 35B MoE workload can run.

## 1. What Lemonade is today and what it runs here

Lemonade Server is a local inference service and model/backend orchestrator. It exposes standard OpenAI, Anthropic, and Ollama-compatible APIs and launches specialized child inference engines; it is not itself an LLM kernel implementation. The project describes two forms—an installed server and an embeddable binary—and identifies AMD engineering optimizations for Ryzen AI, Radeon, and Strix Halo systems. [Official Lemonade README](https://github.com/lemonade-sdk/lemonade/blob/v11.0.0/README.md)

For text generation on Linux/gfx1151, the relevant engines are:

| Lemonade recipe/backend | Format | gfx1151/Linux status | Relevance |
|---|---|---|---|
| `llamacpp:rocm` | GGUF | Supported; ROCm is recommended first for Ryzen AI Strix Point/Halo | Primary conservative path |
| `llamacpp:vulkan` | GGUF | Supported on AMD iGPU/dGPU | Fallback if ROCm fails |
| `llamacpp:cpu` | GGUF | Supported | Slow fallback/smoke path |
| `vllm:rocm` | Hugging Face Transformers/safetensors, including pre-quantized checkpoints | **Experimental**, Linux-only; validated generally on gfx1151 and gfx1150 | High-concurrency/day-zero path, not stability path |
| `flm:npu` | FLM | XDNA2 NPU, Linux/Windows | Not a documented Qwen3.6 candidate here |
| `ryzenai-llm:npu` | ONNX-oriented Ryzen AI stack | Windows only | Not available on this Linux host |

Sources: [supported backend matrix](https://github.com/lemonade-sdk/lemonade/blob/v11.0.0/README.md#supported-configurations), [llama.cpp backend guide](https://lemonade-server.ai/docs/guide/configuration/llamacpp/), [vLLM backend guide](https://lemonade-server.ai/docs/guide/configuration/vllm/).

Lemonade's vLLM backend is a self-contained per-architecture bundle containing Python, PyTorch ROCm, ROCm user-space libraries, Triton, and vLLM. Lemonade v11 pins the default RDNA bundle to `vllm0.20.1-rocm7.12.0`; this is a Lemonade-packaged/custom runtime line, not the standalone vLLM 0.24.0 container already tested. [vLLM guide](https://lemonade-server.ai/docs/guide/configuration/vllm/), [v11 backend pins](https://github.com/lemonade-sdk/lemonade/blob/v11.0.0/src/cpp/resources/backend_versions.json), [exact gfx1151 bundle release](https://github.com/lemonade-sdk/vllm-rocm/releases/tag/vllm0.20.1-rocm7.12.0-gfx1151).

Lemonade's llama.cpp ROCm stable path is its documented primary AMD LLM backend. v11 pins stable build `b9752`; Vulkan is pinned to `b9747`. [llama.cpp guide](https://lemonade-server.ai/docs/guide/configuration/llamacpp/), [v11 backend pins](https://github.com/lemonade-sdk/lemonade/blob/v11.0.0/src/cpp/resources/backend_versions.json), [stable ROCm build](https://github.com/lemonade-sdk/llama.cpp/releases/tag/b9752).

### Verified local state

Read-only inspection found:

- `lemonade-server` 10.9.0 is installed; `lemond` is active on port 13305 with no model loaded.
- `llamacpp:rocm` build `b9752`, `llamacpp:vulkan` build `b9747`, and `vllm:rocm` bundle `vllm0.20.1-rocm7.12.0-gfx1151` are installed.
- Kernel is `7.0.0-27-generic`; the in-tree `amdgpu` module is loaded, `rocminfo` reports `gfx1151`, and KFD exports nonzero GPU `cwsr_size` and `ctl_stack_size`. This satisfies Lemonade's documented gfx1151 CWSR check. [Lemonade gfx1151 prerequisite page](https://lemonade-server.ai/gfx1151_linux.html), [AMD Strix Halo optimization guide](https://rocm.docs.amd.com/en/latest/how-to/system-optimization/strixhalo.html).
- The exact failed `Qwen/Qwen3.6-35B-A3B-FP8` revision `95a723d08a9490559dae23d0cff1d9466213d989` is fully cached locally (about 35 GiB). Its local `config.json` declares `Qwen3_5MoeForConditionalGeneration`, `quant_method: fp8`, dynamic activations, E4M3, and 128x128 weight blocks, consistent with the [official checkpoint metadata](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8).
- The planned `Qwen/Qwen3.6-27B-FP8` revision `e89b16ebf1988b3d6befa7de50abc2d76f26eb09` is **not** cached locally. [Official checkpoint](https://huggingface.co/Qwen/Qwen3.6-27B-FP8).
- `Qwen3-0.6B-GGUF` is registered as downloaded in Lemonade (reported size 0.38 GB), providing a no-download smoke-test model.

## 2. Qwen3.6, MoE, FP8, and accepted formats

Lemonade v11 explicitly registers both Qwen3.6 architectures:

- Dense Qwen3.6-27B.
- MoE Qwen3.6-35B-A3B (35B total, about 3B active).

Qwen's official project identifies the 27B dense and 35B-A3B MoE variants, supports llama.cpp through GGUF, and recommends vLLM for the native Transformers checkpoints. [Official Qwen3.6 repository](https://github.com/QwenLM/Qwen3.6), [official 35B-A3B model](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [official 27B model](https://huggingface.co/Qwen/Qwen3.6-27B).

Lemonade does **not** require one universal format; format follows backend:

- `llamacpp` expects GGUF.
- `vllm` loads upstream Hugging Face checkpoints as-is; Lemonade's normal built-ins are labeled FP16, while user registrations may point at pre-quantized FP8, AWQ, or GPTQ checkpoints.
- Other Lemonade backends use FLM or ONNX for their own supported model sets.

Sources: [Lemonade model library/formats](https://github.com/lemonade-sdk/lemonade/blob/v11.0.0/README.md#model-library), [vLLM format and registration guide](https://lemonade-server.ai/docs/guide/configuration/vllm/).

Lemonade v11 explicitly includes low- and high-concurrency Qwen3.6 FP8 recipes for both 27B dense and 35B-A3B MoE. These point directly to the official Qwen FP8 Hugging Face checkpoints. Thus Lemonade **does claim/check in model-level support for Hugging Face FP8**, not just GGUF/ONNX. [v11 model registry](https://github.com/lemonade-sdk/lemonade/blob/v11.0.0/src/cpp/resources/server_models.json), [v11 vLLM model configuration](https://github.com/lemonade-sdk/lemonade/blob/v11.0.0/src/cpp/resources/vllm_model_config.json).

However, registration is not proof that every architecture/quantization/GPU kernel combination works. The official Qwen 35B FP8 checkpoint is a fine-grained block-FP8 MoE model, exactly the class rejected by the tested vLLM 0.24.0 backend selector. [Official Qwen 35B FP8 model card](https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8).

## 3. Credible Qwen3.6 candidates on this host

The built-in Lemonade v11 candidates are:

| Candidate | Runtime | Registered size | gfx1151 confidence |
|---|---:|---:|---|
| `Qwen3.6-35B-A3B-GGUF` | llama.cpp, UD-Q4_K_XL | 23.3 GB | **Best 35B candidate** |
| `Qwen3.6-35B-A3B-MTP-GGUF` | llama.cpp, UD-Q4_K_XL + MTP | 23.8 GB | Viable after base GGUF succeeds; extra moving part |
| `Qwen3.6-27B-GGUF` | llama.cpp, UD-Q4_K_XL | 18.5 GB | Strong dense fallback |
| `Qwen3.6-27B-MTP-GGUF` | llama.cpp, UD-Q4_K_XL + MTP | 18.8 GB | Secondary |
| `Qwen3.6-27B-FP16-vLLM` | vLLM, official HF FP16 | 55.59 GB | Lemonade's most polished vLLM datatype, but large |
| `Qwen3.6-27B-FP8-vLLM-{low,high}conc` | vLLM, official HF FP8 | 27.8 GB | Experimental on gfx1151; dense avoids MoE selector |
| `Qwen3.6-35B-A3B-FP16-vLLM` | vLLM, official HF FP16 | 71.93 GB | Credible by memory, but expensive |
| `Qwen3.6-35B-A3B-FP8-vLLM-{low,high}conc` | vLLM, official HF FP8 | 36.0 GB | **Not yet trustworthy on gfx1151** |

All names, checkpoints, formats, and sizes above are from the [v11 Lemonade model registry](https://github.com/lemonade-sdk/lemonade/blob/v11.0.0/src/cpp/resources/server_models.json). Low concurrency sets `--max-num-seqs 8`; high concurrency sets 256; FP8 recipes add one-token MTP speculative decoding. [v11 vLLM model configuration](https://github.com/lemonade-sdk/lemonade/blob/v11.0.0/src/cpp/resources/vllm_model_config.json).

For the GGUF route, Lemonade selects `unsloth/Qwen3.6-35B-A3B-GGUF:Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf` plus the F16 vision projection. The checkpoint repository also publishes Q8_0, MXFP4_MOE, and multiple IQ/K-quants, but **UD-Q4_K_XL is the exact Lemonade-built-in and therefore the least speculative candidate**. [Lemonade registry](https://github.com/lemonade-sdk/lemonade/blob/v11.0.0/src/cpp/resources/server_models.json), [GGUF checkpoint metadata](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/tree/main).

## 4. Operational stability versus kernel/model support

### Server/API layer

Lemonade adds useful operational structure: a managed service, model registry/cache, backend selection, and standard API façade. That can make startup, API compatibility, switching between ROCm/Vulkan, and routine model lifecycle easier than maintaining a custom container command. [Lemonade README](https://github.com/lemonade-sdk/lemonade/blob/v11.0.0/README.md), [backend guides](https://lemonade-server.ai/docs/guide/configuration/llamacpp/).

That does **not** make the vLLM child runtime intrinsically more stable. Lemonade labels `vllm:rocm` experimental and documents cold Triton JIT delays, slow FP8 startup on gfx1151, and a failure mode where a timed-out/interrupted load can leave an orphaned vLLM EngineCore holding GPU memory. [Lemonade vLLM guide](https://lemonade-server.ai/docs/guide/configuration/vllm/).

### Kernel/model layer

- The failed standalone vLLM run reached model recognition and then found no compatible FP8 MoE backend. Lemonade's API proxy cannot repair that class of kernel dispatch incompatibility.
- Lemonade v11 does register the exact 35B-A3B FP8 model, but its documentation says Qwen3.6 FP8/MTP validation was on `gfx942`/MI300X. The recipes are allowed on RDNA, yet MTP on RDNA has not been independently serve-validated. [Lemonade vLLM guide, “Speculative decoding (MTP)”](https://lemonade-server.ai/docs/guide/configuration/vllm/).
- Lemonade's documented gfx1151 FP8 result was a much smaller Qwen FP8 model: cold load took about 12 minutes, exceeded the default readiness timeout, used a default Triton block-FP8 kernel configuration, and warned that performance might be suboptimal. The same guide calls FP16 “the most polished path today.” [Lemonade vLLM guide, “Known gotchas”](https://lemonade-server.ai/docs/guide/configuration/vllm/).
- By contrast, Lemonade calls llama.cpp its primary LLM backend, documents ROCm stable as the stability/production-oriented channel, recommends ROCm first on Ryzen AI Strix Halo, and offers Vulkan fallback. [Lemonade llama.cpp guide](https://lemonade-server.ai/docs/guide/configuration/llamacpp/).

**Conclusion:** Lemonade's server/API may be more convenient, but **Lemonade-over-vLLM is not presently more stable for this exact FP8-MoE workload**. Lemonade-over-llama.cpp with the registered GGUF is a materially different and more viable option.

## 5. Cheapest smoke test and authorization boundary

### Cheapest useful test with no new model download

After authorization to load a model/alter the running service state:

```bash
lemonade load --llamacpp rocm Qwen3-0.6B-GGUF
curl -sS http://127.0.0.1:13305/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-0.6B-GGUF","messages":[{"role":"user","content":"Reply with exactly: OK"}],"temperature":0,"max_tokens":4}'
```

Then send one short deterministic chat completion and inspect `lemonade status`/logs to confirm that the selected backend is `llamacpp:rocm`, the model is GPU-offloaded, and tokens are returned. The model is already downloaded locally (0.38 GB). This is a service/backend/GPU smoke test only; it does not validate Qwen3.6, MoE, vision, MTP, or FP8.

### More targeted no-new-model-download diagnostic

The exact 35B-A3B FP8 checkpoint is already cached, so a targeted Lemonade vLLM attempt need not intentionally download model weights. The clean official recipe names exist only in Lemonade v11, while this host has v10.9.0. A targeted attempt therefore requires authorization to:

1. upgrade the installed `lemonade-server` package to v11 (APT currently offers 11.0.0),
2. allow Lemonade to register/load the cached checkpoint and launch/stop its vLLM child processes,
3. accept that Hugging Face cache matching may still fetch missing/revised metadata or files unless offline mode is deliberately configured,
4. monitor for and clean up a timed-out EngineCore process if the documented orphan condition occurs.

Use the low-concurrency recipe first: `Qwen3.6-35B-A3B-FP8-vLLM-lowconc`. This is diagnostic evidence only; failure would be unsurprising given the absent gfx1151 Qwen3.6-MoE-FP8 validation.

### Actions requiring new authorization

- Loading or unloading any model, because it changes running process/GPU state.
- Upgrading Lemonade 10.9.0 to 11.0.0 (system package change; likely `sudo`).
- Downloading the recommended 23.3 GB Qwen3.6-35B-A3B GGUF, its projection file, or the 27.8 GB Qwen3.6-27B FP8 checkpoint.
- Pinning/installing a different Lemonade vLLM or llama.cpp runtime bundle (large runtime download and backend state change).
- Exposing Lemonade beyond its existing localhost listener; this assessment does not recommend or authorize that.

## 6. Final recommendation

**Choose `Qwen3.6-35B-A3B-GGUF` through Lemonade `llamacpp:rocm` stable, with Vulkan as fallback.** This preserves the target model family and MoE architecture while avoiding the exact unsupported Hugging Face FP8-MoE kernel path. The cost is switching quantization/container format from the official FP8 checkpoint to a third-party GGUF UD-Q4_K_XL conversion, so quality/performance must be measured against the experiment's acceptance criteria.

Do **not** select “Lemonade now” if that means merely rerunning the same Qwen35 FP8 checkpoint through Lemonade's experimental vLLM backend and assuming the wrapper fixes kernel support. A single authorized low-concurrency attempt is worthwhile because the checkpoint is already cached and v11 carries an official recipe, but it is a diagnostic bet.

Use Qwen3.6-27B-FP8 through vLLM only if the scientific objective permits changing from 35B-A3B MoE to 27B dense and preserving official FP8 is more important than preserving architecture. Because that checkpoint is not cached, it should follow—not precede—the no-download Lemonade smoke test and the decision about the 35B GGUF path.

## Evidence classification

- **Verified locally:** installed versions/backends, active service/no model loaded, kernel/GPU/CWSR state, local cache presence and exact revision/config of Qwen35 FP8, absence of the Qwen27 FP8 cache, availability of the small GGUF smoke model.
- **Verified from primary sources:** Lemonade backend matrix and pins, Qwen3.6 registry entries and sizes, official Qwen architectures/formats, explicit gfx1151 FP8 caveats, and the limited FP8/MTP validation scope.
- **Inference:** llama.cpp/GGUF is more likely to be operationally stable for this workload than vLLM/FP8-MoE. This follows from Lemonade's “primary backend”/stable-channel guidance, the registered exact GGUF candidate, and the observed vLLM backend-selection failure; it is not yet an on-host Qwen3.6 benchmark result.
