# GroupRank 7B Stage 2 Pilot: No Quality Result

## Technical summary

The purpose-trained `AQ-MedAI/Diver-GroupRank-7B` pilot stopped at the required
qrel-free smoke gate. Two preserved one-request attempts failed for complementary
output-protocol reasons, so the 48-request, twelve-question pilot did not run and
no nDCG@10, MRR, top-three, or baseline comparison is valid.

This result does **not** show that groupwise ranking lacks value. It shows that
neither tested prompt protocol reliably produced a machine-valid ten-score vector
under the frozen strict parser. The model reasoned over the presented experts in
both attempts, but no attempt satisfied the complete output contract.

## Preserved smoke evidence

| Attempt | Code revision | Output tokens | What validated | Terminal failure | Receipt SHA-256 |
| --- | --- | ---: | --- | --- | --- |
| Original | `01a0a2a` | 523 | Complete ten-handle JSON existed in the second block | Two `<answer>` blocks violated the one-block contract | `7a11c925bd3728fc17595849ab3291409be273a91ec1ee097c4bee662314f3d5` |
| Official protocol v2 | `ff14d1c` | 411 | Exactly one `<reason>` and one `<answer>`; normal stop | JSON copied the literal placeholder key `"<each presented handle exactly once>"` instead of the ten handles | `648a8f400d88cf0e6d3a99f94f4edbfc32ce3b1ba83b010d7e95b00bd3ba7493` |

Both attempts used model revision
`42a9504b32616f155970d5b7f349b63dc24b3e27`, BF16, an 8,192-token context,
`gpu_memory_utilization=0.40`, Triton attention, the same question, candidate
group, presentation order, and deterministic sampling seed. Neither opened
qrels.

## Why the pilot stopped

The frozen design requires a passed smoke before the full run and rejects any
missing, extra, duplicated, repaired, or synthesized score. Protocol v2 returned
one extra placeholder key and omitted all ten required handles. Inferring a
ranking from its prose, selecting the original attempt's second answer block, or
combining the two attempts would be unplanned repair. The strict parser therefore
failed correctly, and the 48 requests were not launched.

## What can and cannot be concluded

Verified:

- ROCm/vLLM loaded the pinned 7B checkpoint successfully and generated normally.
- The official two-part envelope correction fixed the original tag-shape failure.
- The model explicitly compared all ten experts in protocol-v2 reasoning.
- The generic quoted key placeholder was copied into the final JSON.

Not established:

- ranking quality;
- a format-error rate, because each protocol has only one observation;
- whether GroupRank beats contextual E5, contextual Qwen4B, or pointwise Qwen;
- whether the groupwise method has deployment value.

The earlier baselines remain descriptive context only. They were never compared
with a valid GroupRank ranking.

## Recommended next experiment

If GroupRank remains worth pursuing, start a separately named, qrel-free format
feasibility study rather than repairing this pilot:

1. Use the checkpoint's intended numeric handles `[1]` through `[10]`.
2. Use its official concrete numeric-ID output example and map IDs back to
   candidate IDs deterministically.
3. Freeze two or three requests per prompt arm and require 100% exact validation
   before another 48-request quality pilot.
4. Keep the strict one-reason/one-answer parser, exact-key validation, injection
   guard, and no-repair rule.

Fixed-key constrained decoding is a reasonable separate arm, but it tests a
different interface and must not be presented as a rescue of these artifacts.

## Sources

- Failed smoke receipts:
  `artifacts/llm_label_ranker/pilot/stage2/grouprank_7b/smoke.json` and
  `artifacts/llm_label_ranker/pilot/stage2/grouprank_7b/smoke_protocol_v2.json`
- Frozen design:
  `docs/superpowers/specs/2026-07-13-grouprank-7b-stage2-pilot-design.md`
- [Pinned model card](https://huggingface.co/AQ-MedAI/Diver-GroupRank-7B/blob/42a9504b32616f155970d5b7f349b63dc24b3e27/README.md#inference)
- [Authors' GroupRank inference code](https://github.com/AQ-MedAI/Diver/blob/main/Reranker/rerank_groupwise.py)

