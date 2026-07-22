# E5 raw historical-response refinement results

Development effectiveness is post-hoc confirmation, not independent hidden-test evidence.

- Evaluation status: `evaluated`
- Frozen decision: **advance**
- Refinement artifact SHA-256: `113fa6391df361bd161c2c9ad6a1d3ccb7f6e6f59a917b66135c6a87b0a33f83`
- Sealed qrels SHA-256: `3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35`

## Development metrics

| System | nDCG@10 | MRR | Zero nDCG |
|---|---:|---:|---:|
| `dense_profile_equal` | 0.420213992194 | 0.453017393783 | 99 |
| `dense_reference` | 0.359854744095 | 0.404063308112 | 116 |
| `profile_only` | 0.414924459979 | 0.441697813928 | 92 |
| `raw_hierarchical` | 0.449022366769 | 0.478370650042 | 86 |
| `raw_hierarchical_dense_equal` | 0.435625273018 | 0.465457752325 | 92 |

## Prespecified primary comparison

`raw_hierarchical_dense_equal` versus `dense_profile_equal`:

- nDCG@10 delta: 0.015411280824 (95% CI 0.006964609515 to 0.024355856294)
- MRR delta: 0.012440358542 (95% CI 0.001475973122 to 0.023749239191)

## Interpretation boundary

The frozen rule determines the decision above. Secondary and dense comparisons are diagnostics and do not select another variant.
