# Corrected label-embedding sweep

The earlier Qwen and E5 results are invalid because their required terminal tokens
were omitted by newer generic runtime/tokenizer behavior. The corrected runs construct
the pinned revisions' terminal tokens explicitly.

| Encoder | Label document | nDCG@10 | MRR | Zero nDCG | Repeat |
|---|---|---:|---:|---:|---|
| MPNet control | cleaned label | 0.3195 | 0.3674 | 131 | reproduced |
| Qwen3-Embedding-8B | cleaned label | 0.3314 | 0.3742 | 124 | failed |
| Qwen3-Embedding-8B | capability sentence | 0.3281 | 0.3796 | 131 | passed |
| E5-Mistral-7B-Instruct | cleaned label | 0.3107 | 0.3600 | 126 | passed |
| E5-Mistral-7B-Instruct | capability sentence | 0.3347 | 0.3866 | 120 | passed |

No condition passes the frozen advancement gate. The strongest condition, contextual
E5, improves mean nDCG@10 over MPNet by 0.0152, but its paired 95% interval is
[-0.0125, 0.0424], which includes zero. Stage 2 therefore remains stopped.

For E5, the capability sentence improves nDCG@10 over its bare-label condition by
0.0240 with a paired 95% interval of [0.0004, 0.0479]. For Qwen, the corresponding
change is -0.0033 with interval [-0.0220, 0.0150]. This supports adding context for E5,
but not treating contextualization as a model-independent improvement.

The Qwen bare-label repeat changes rankings and has a maximum absolute score difference
of 0.01033 across fresh ROCm processes. It is retained as exploratory evidence but is
not eligible for advancement. The other three corrected conditions satisfy the frozen
repeat requirement.

Detailed machine-readable receipts are under `artifacts/label_sweep_corrected/`.
