# LLM Expert-Label Ranker: Stage 2 Pilot

> Qrel-assisted exploratory development evidence; candidate sets and strata were selected with development qrels.

## Disposition

Stage 2 pointwise evidence is complete, but on this qrel-assisted diagnostic it placed a relevant label in the top three for 5/12 questions versus 8/12 for contextual Qwen4B and 8/12 for contextual E5. That does not support expanding the current pointwise design yet. The final listwise attempt is unscorable because no validated rankings were persisted. Discuss these examples before any larger pilot.

No inference rerun, model selection, fusion, or deployment action was started by this report.

## Headline metrics (restricted 20-label sets)

| Method | Status | nDCG@10 | MRR | Relevant in top 3 |
| --- | --- | ---: | ---: | ---: |
| Pointwise | complete | 0.4820 | 0.3209 | 5/12 |
| Listwise | unscorable | unavailable | unavailable | unavailable |
| Contextual Qwen4B | complete_restricted_baseline | 0.6648 | 0.5585 | 8/12 |
| Contextual E5-Mistral | complete_restricted_baseline | 0.7232 | 0.6361 | 8/12 |

## Strata

| Stratum | Method | nDCG@10 | MRR | Relevant in top 3 |
| --- | --- | ---: | ---: | ---: |
| issue_like_clear | Pointwise | 0.4378 | 0.2649 | 1/4 |
| issue_like_clear | Qwen4B | 1.0000 | 1.0000 | 4/4 |
| issue_like_clear | E5 | 0.8750 | 0.8333 | 4/4 |
| issue_like_clear | Listwise | unavailable | unavailable | unavailable |
| contextual_qwen_success | Pointwise | 0.4467 | 0.2708 | 2/4 |
| contextual_qwen_success | Qwen4B | 0.6309 | 0.5000 | 4/4 |
| contextual_qwen_success | E5 | 0.8467 | 0.8000 | 3/4 |
| contextual_qwen_success | Listwise | unavailable | unavailable | unavailable |
| contextual_qwen_failure | Pointwise | 0.5615 | 0.4271 | 2/4 |
| contextual_qwen_failure | Qwen4B | 0.3634 | 0.1756 | 0/4 |
| contextual_qwen_failure | E5 | 0.4479 | 0.2750 | 1/4 |
| contextual_qwen_failure | Listwise | unavailable | unavailable | unavailable |

## Terminal listwise failure

- Receipt SHA-256: `d83ebfb60a7a5ea6b1d209ec99f2720b324b6872c751ab482bd2c3ce8493a2f8`
- Failed parse: `57326d2f0fdd8d15006c6ad5` rotation `0`
- Planned/log-observed requests: 36/36
- Persisted raw outputs: 0; rankings available: no; OOM: no.
- Exact error: `pilot listwise parse failure for 57326d2f0fdd8d15006c6ad5/0: pilot ranking has duplicates, omissions, or unknown handles`

The log proves request processing reached 36/36, not that 35 usable outputs existed. The program raised before any raw output or ranking artifact was persisted.

## Recalculation receipt

- Pointwise receipt SHA-256: `7e5d74a777acda1c044f7140e2fe40e4c794122eca0c7f1bc26d2b9580fe7882`
- Frozen selection SHA-256: `668515c30a80473aaaa2dc530f50dfd3b7731711e9f9adf70074ebb95aa98b9d`
- Qrels SHA-256: `3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35`
