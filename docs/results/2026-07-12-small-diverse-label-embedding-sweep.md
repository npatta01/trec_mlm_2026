# Small diverse label-embedding sweep

Twenty full runs evaluated five pinned embedding models, two label representations,
and two fresh-process repeats. All models used FP32 after smoke tests demonstrated that
independent BF16 loads on gfx1151 did not meet reference-parity or repeatability limits.

| Encoder | Label document | nDCG@10 | MRR | Zeros | Repeat |
|---|---|---:|---:|---:|---|
| MPNet control | bare | 0.3195 | 0.3674 | 131 | reproduced |
| E5-Mistral 7B | capability | 0.3347 | 0.3866 | 120 | pass |
| Qwen 0.6B | bare | 0.2757 | 0.3271 | 138 | pass |
| Qwen 0.6B | capability | 0.3000 | 0.3412 | 131 | pass |
| Qwen 4B | bare | 0.3093 | 0.3541 | 128 | pass |
| Qwen 4B | capability | **0.3420** | **0.3902** | **115** | fail: ranking near-tie |
| Jina v5 small | bare | 0.2863 | 0.3543 | 149 | pass |
| Jina v5 small | capability | 0.3122 | 0.3711 | 132 | pass |
| BGE-M3 | bare | 0.2672 | 0.3106 | 149 | pass |
| BGE-M3 | capability | 0.2576 | 0.3197 | 150 | pass |
| Nomic Embed v1.5 | bare | 0.2408 | 0.3074 | 160 | pass |
| Nomic Embed v1.5 | capability | 0.2591 | 0.3259 | 154 | pass |

Qwen 4B with the fixed capability sentence has the strongest point estimates. Against
MPNet its mean nDCG@10 difference is +0.0225 with paired 95% interval
[-0.0038, +0.0487]. Against contextual E5 its difference is +0.0073 with interval
[-0.0146, +0.0299]. Neither comparison excludes zero, so the sweep does not establish
that Qwen 4B is better than either reference.

The nominal exploratory paired intervals exclude zero for contextualization within
Qwen 4B (+0.0327 nDCG, 95% interval
[+0.0121, +0.0530]), Qwen 0.6B (+0.0243, [+0.0077, +0.0411]), and Jina small
(+0.0259, [+0.0091, +0.0426]). Nomic's interval includes zero, while BGE-M3's point
estimate decreases. These post-hoc conditions were inspected without a
multiple-comparison correction.

The Qwen 4B contextual repeats have identical metrics and top-10 rankings, with scores
within `1.52e-6`. Across all 322 complete 150-item rankings, only two adjacent tail
orders change: ranks 33/34 and 113/114. It therefore fails the preregistered all-150
repeat rule but does not show meaningful top-rank instability.
No deployment model, fusion system, or LLM ranker was selected automatically.

Machine-readable comparisons and the full plain-language receipt are under
`artifacts/label_sweep_diverse/combined/`.
