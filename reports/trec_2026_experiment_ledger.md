# TREC 2026 experiment ledger

- Promoted development result: `dense_profile_equal` (nDCG@10 0.4202).
- Best full-development diagnostic: `dense_profile_profile_heavy` (nDCG@10 0.4376) — `diagnostic_only`, not promoted.
- Best stable qrel-free control: `profile_only` (nDCG@10 0.4149).
- Systems with complete per-query evidence: 11.
- aggregate-only systems: 73.
- Unresolved artifact blockers: 43 systems.
- Artifact blocker count includes only `local_artifact_missing` and `construction_artifact_missing` states.
- All effectiveness scores are development-only evidence; the hidden test is not available.

## Full development results

Only `dev_322x150` systems appear here. `diagnostic_only` and `aggregate_only` remain visible in the table.

| System | Scope | Queries | Candidates | Family | Promotion | Evidence | nDCG@10 | MRR | Per-query | Artifact |
|---|---|---:|---:|---|---|---|---:|---:|---|---|
| `dense_profile_profile_heavy` | `dev_322x150` | 322 | 150 | semantic_profiles | `diagnostic_only` | `stored_per_query` | 0.4376 | 0.4700 | yes | `local_artifact_missing` |
| `x1_multires_top2` | `dev_322x150` | 322 | 150 | multiresolution_profiles | `diagnostic_only` | `stored_per_query` | 0.4364 | 0.4646 | yes | `construction_artifact_missing` |
| `dense_profile_e5_equal` | `dev_322x150` | 322 | 150 | semantic_profiles | `valid_negative` | `aggregate_only` | 0.4311 | 0.4465 | no | `local_artifact_missing` |
| `x1_multires_top2_dense_equal` | `dev_322x150` | 322 | 150 | multiresolution_profiles | `diagnostic_only` | `stored_per_query` | 0.4271 | 0.4547 | yes | `construction_artifact_missing` |
| `r1_multires_max_dense_equal` | `dev_322x150` | 322 | 150 | multiresolution_profiles | `diagnostic_only` | `stored_per_query` | 0.4261 | 0.4556 | yes | `construction_artifact_missing` |
| `r1_multires_max` | `dev_322x150` | 322 | 150 | multiresolution_profiles | `diagnostic_only` | `stored_per_query` | 0.4256 | 0.4646 | yes | `construction_artifact_missing` |
| `dense_profile_equal` | `dev_322x150` | 322 | 150 | semantic_profiles | `promoted_development` | `stored_per_query` | 0.4202 | 0.4530 | yes | `local_artifact_missing` |
| `a1_top2_k10_dense_equal` | `dev_322x150` | 322 | 150 | multiresolution_profiles | `diagnostic_only` | `stored_per_query` | 0.4202 | 0.4478 | yes | `construction_artifact_missing` |
| `a1_top2_k10` | `dev_322x150` | 322 | 150 | multiresolution_profiles | `diagnostic_only` | `stored_per_query` | 0.4194 | 0.4521 | yes | `construction_artifact_missing` |
| `profile_only` | `dev_322x150` | 322 | 150 | semantic_profiles | `stable_control` | `stored_per_query` | 0.4149 | 0.4417 | yes | `local_artifact_missing` |
| `dense_profile_dense_heavy` | `dev_322x150` | 322 | 150 | semantic_profiles | `valid_negative` | `aggregate_only` | 0.4103 | 0.4468 | no | `local_artifact_missing` |
| `dense_e5_equal` | `dev_322x150` | 322 | 150 | semantic_profiles | `valid_negative` | `aggregate_only` | 0.3863 | 0.4191 | no | `local_artifact_missing` |
| `mpnet_base_v2@50` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.3605 | 0.4040 | no | `tracked_receipt_only` |
| `dense_nested` | `dev_322x150` | 322 | 150 | dense_query_knn | `stable_control` | `stored_per_query` | 0.3599 | 0.4041 | yes | `tracked_complete` |
| `mpnet_base_v2@20` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.3551 | 0.3915 | no | `tracked_receipt_only` |
| `local_domain_prior` | `dev_322x150` | 322 | 150 | query_cluster_aggregation | `valid_negative` | `aggregate_only` | 0.3487 | 0.3802 | no | `tracked_report_snapshot` |
| `qwen4b_contextual` | `dev_322x150` | 322 | 150 | candidate_label_matching | `diagnostic_only` | `aggregate_only` | 0.3420 | 0.3902 | no | `tracked_report_snapshot` |
| `matched_neighbor_k20_preflight` | `dev_322x150` | 322 | 150 | query_cluster_aggregation | `diagnostic_only` | `aggregate_only` | 0.3395 | 0.3823 | no | `tracked_receipt_only` |
| `mpnet_base_v2@100` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.3368 | 0.3805 | no | `tracked_receipt_only` |
| `contextual_e5` | `dev_322x150` | 322 | 150 | candidate_label_matching | `stable_control` | `stored_per_query` | 0.3347 | 0.3866 | yes | `tracked_complete` |
| `minilm_l6@20` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.3330 | 0.3734 | no | `tracked_receipt_only` |
| `minilm_l6@50` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.3306 | 0.3685 | no | `tracked_receipt_only` |
| `mpnet_base_v2@10` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.3292 | 0.3718 | no | `tracked_receipt_only` |
| `minilm_l6@100` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.3287 | 0.3764 | no | `tracked_receipt_only` |
| `matched_neighbor_k40_preflight` | `dev_322x150` | 322 | 150 | query_cluster_aggregation | `diagnostic_only` | `aggregate_only` | 0.3282 | 0.3688 | no | `tracked_receipt_only` |
| `qwen8b_contextual` | `dev_322x150` | 322 | 150 | candidate_label_matching | `diagnostic_only` | `aggregate_only` | 0.3281 | 0.3796 | no | `tracked_report_snapshot` |
| `minilm_l6@10` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.3254 | 0.3666 | no | `tracked_receipt_only` |
| `mpnet_raw_label` | `dev_322x150` | 322 | 150 | candidate_label_matching | `stable_control` | `aggregate_only` | 0.3195 | 0.3674 | no | `tracked_report_snapshot` |
| `bge_small_en_v1_5@50` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.3131 | 0.3624 | no | `tracked_receipt_only` |
| `jina_v5_contextual` | `dev_322x150` | 322 | 150 | candidate_label_matching | `diagnostic_only` | `aggregate_only` | 0.3122 | 0.3711 | no | `tracked_report_snapshot` |
| `bge_small_en_v1_5@20` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.3029 | 0.3494 | no | `tracked_receipt_only` |
| `qwen06b_contextual` | `dev_322x150` | 322 | 150 | candidate_label_matching | `diagnostic_only` | `aggregate_only` | 0.3000 | 0.3412 | no | `tracked_report_snapshot` |
| `bge_small_en_v1_5@100` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2969 | 0.3378 | no | `tracked_receipt_only` |
| `bge_small_en_v1_5@10` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2961 | 0.3527 | no | `tracked_receipt_only` |
| `e5_small_v2@50` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2958 | 0.3348 | no | `tracked_receipt_only` |
| `minilm_l6@5` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2937 | 0.3321 | no | `tracked_receipt_only` |
| `bm25_rm3__response_only__basic__d500` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2930 | 0.3359 | no | `local_artifact_missing` |
| `bm25_rm3__query_response__basic__d500` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2915 | 0.3253 | no | `local_artifact_missing` |
| `mpnet_base_v2@5` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2915 | 0.3387 | no | `tracked_receipt_only` |
| `bm25_rm3__response_only__aggressive__d500` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2913 | 0.3300 | no | `local_artifact_missing` |
| `e5_small_v2@20` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2910 | 0.3401 | no | `tracked_receipt_only` |
| `bm25_rm3__query_response__aggressive__d500` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2909 | 0.3316 | no | `local_artifact_missing` |
| `hard_cluster_k20_preflight` | `dev_322x150` | 322 | 150 | query_cluster_aggregation | `diagnostic_only` | `aggregate_only` | 0.2902 | 0.3243 | no | `tracked_receipt_only` |
| `response_top10sum_nested` | `dev_322x150` | 322 | 150 | response_volume_normalization | `valid_negative` | `aggregate_only` | 0.2892 | 0.3283 | no | `local_artifact_missing` |
| `bm25_rm3__query_response__aggressive__d1000` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2878 | 0.3321 | no | `local_artifact_missing` |
| `hard_cluster_k40_preflight` | `dev_322x150` | 322 | 150 | query_cluster_aggregation | `diagnostic_only` | `aggregate_only` | 0.2866 | 0.3351 | no | `tracked_receipt_only` |
| `bm25_rm3__response_only__basic__d1000` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2865 | 0.3250 | no | `local_artifact_missing` |
| `bm25_rm3__response_only__aggressive__d1000` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2841 | 0.3199 | no | `local_artifact_missing` |
| `bm25_rm3__query_response__basic__d1000` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2835 | 0.3317 | no | `local_artifact_missing` |
| `minilm_l6@3` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2775 | 0.3237 | no | `tracked_receipt_only` |
| `response_sumscore_nested` | `dev_322x150` | 322 | 150 | response_retrieval | `valid_negative` | `aggregate_only` | 0.2774 | 0.3175 | no | `local_artifact_missing` |
| `bm25_rm3__query_response__basic__d100` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2756 | 0.3296 | no | `local_artifact_missing` |
| `bm25__query_response__basic__d500` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2748 | 0.3317 | no | `local_artifact_missing` |
| `bm25_rm3__query_response__aggressive__d100` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2718 | 0.3220 | no | `local_artifact_missing` |
| `e5_small_v2@100` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2710 | 0.3284 | no | `tracked_receipt_only` |
| `bm25__query_response__basic__d1000` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2686 | 0.3284 | no | `local_artifact_missing` |
| `bm25__query_response__aggressive__d500` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2686 | 0.3222 | no | `local_artifact_missing` |
| `mpnet_base_v2@3` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2669 | 0.3122 | no | `tracked_receipt_only` |
| `bge_small_en_v1_5@5` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2666 | 0.3278 | no | `tracked_receipt_only` |
| `e5_small_v2@10` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2643 | 0.3110 | no | `tracked_receipt_only` |
| `bm25__query_response__aggressive__d1000` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2640 | 0.3307 | no | `local_artifact_missing` |
| `bm25__response_only__basic__d500` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2638 | 0.3110 | no | `local_artifact_missing` |
| `bm25__response_only__aggressive__d500` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2635 | 0.3101 | no | `local_artifact_missing` |
| `bm25__query_response__basic__d100` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2634 | 0.3218 | no | `local_artifact_missing` |
| `bm25__query_response__aggressive__d100` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2621 | 0.3160 | no | `local_artifact_missing` |
| `bm25__response_only__aggressive__d1000` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2607 | 0.3116 | no | `local_artifact_missing` |
| `bm25_rm3__response_only__aggressive__d100` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2607 | 0.3114 | no | `local_artifact_missing` |
| `bm25__response_only__basic__d1000` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2600 | 0.3131 | no | `local_artifact_missing` |
| `nomic_v15_contextual` | `dev_322x150` | 322 | 150 | candidate_label_matching | `diagnostic_only` | `aggregate_only` | 0.2591 | 0.3259 | no | `tracked_report_snapshot` |
| `bge_m3_contextual` | `dev_322x150` | 322 | 150 | candidate_label_matching | `diagnostic_only` | `aggregate_only` | 0.2576 | 0.3197 | no | `tracked_report_snapshot` |
| `bm25_rm3__response_only__basic__d100` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2565 | 0.3099 | no | `local_artifact_missing` |
| `e5_small_v2@5` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2559 | 0.3099 | no | `tracked_receipt_only` |
| `bm25__response_only__basic__d100` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2471 | 0.3007 | no | `local_artifact_missing` |
| `bge_small_en_v1_5@3` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2463 | 0.3105 | no | `tracked_receipt_only` |
| `bm25__response_only__aggressive__d100` | `dev_322x150` | 322 | 150 | response_retrieval | `diagnostic_only` | `aggregate_only` | 0.2458 | 0.2980 | no | `local_artifact_missing` |
| `minilm_l6@1` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2405 | 0.2941 | no | `tracked_receipt_only` |
| `e5_small_v2@3` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2264 | 0.2892 | no | `tracked_receipt_only` |
| `mpnet_base_v2@1` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2246 | 0.2929 | no | `tracked_receipt_only` |
| `tfidf_fixed_k10` | `dev_322x150` | 322 | 150 | query_controls | `stable_control` | `aggregate_only` | 0.2177 | 0.2777 | no | `tracked_receipt_only` |
| `source_alignment_nested` | `dev_322x150` | 322 | 150 | source_query_first | `valid_negative` | `aggregate_only` | 0.2033 | 0.2482 | no | `local_artifact_missing` |
| `e5_small_v2@1` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.2028 | 0.2656 | no | `tracked_receipt_only` |
| `target_response_same_k` | `dev_322x150` | 322 | 150 | source_query_first | `diagnostic_only` | `aggregate_only` | 0.2023 | 0.2510 | no | `local_artifact_missing` |
| `bge_small_en_v1_5@1` | `dev_322x150` | 322 | 150 | dense_query_knn | `diagnostic_only` | `aggregate_only` | 0.1976 | 0.2683 | no | `tracked_receipt_only` |
| `global_prior` | `dev_322x150` | 322 | 150 | query_controls | `stable_control` | `aggregate_only` | 0.1385 | 0.2067 | no | `tracked_receipt_only` |

## Restricted pilots and invalid/unscored work

These rows are not numerically comparable with the complete 322-query, 150-candidate development evaluation.


### Restricted 12x20 (`restricted_12x20`)

Qrel-assisted 12-query, 20-candidate diagnostics only.

| System | Scope | Queries | Candidates | Family | Promotion | Evidence | nDCG@10 | MRR | Per-query | Artifact |
|---|---|---:|---:|---|---|---|---:|---:|---|---|
| `firstmistral_listwise` | `restricted_12x20` | 12 | 20 | label_ranker_pilots | `restricted_pilot` | `restricted_diagnostic` | 0.6843 | 0.5847 | no | `tracked_report_snapshot` |
| `pointwise_qwen` | `restricted_12x20` | 12 | 20 | label_ranker_pilots | `restricted_pilot` | `restricted_diagnostic` | 0.4820 | 0.3209 | no | `tracked_report_snapshot` |
| `restricted_contextual_e5` | `restricted_12x20` | 12 | 20 | label_ranker_pilots | `diagnostic_only` | `restricted_diagnostic` | 0.7232 | 0.6361 | no | `tracked_report_snapshot` |
| `restricted_qwen4b_contextual` | `restricted_12x20` | 12 | 20 | label_ranker_pilots | `diagnostic_only` | `restricted_diagnostic` | 0.6648 | 0.5585 | no | `tracked_report_snapshot` |

### Restricted 6-query (`restricted_6`)

Six-query feasibility evidence only; candidate counts may be unknown.

| System | Scope | Queries | Candidates | Family | Promotion | Evidence | nDCG@10 | MRR | Per-query | Artifact |
|---|---|---:|---:|---|---|---|---:|---:|---|---|
| `generated_listwise_feasibility_6` | `restricted_6` | 6 | — | label_ranker_pilots | `diagnostic_only` | `restricted_diagnostic` | — | — | no | `tracked_report_snapshot` |
| `pointwise_qwen_feasibility_6` | `restricted_6` | 6 | — | label_ranker_pilots | `restricted_pilot` | `restricted_diagnostic` | — | — | no | `tracked_report_snapshot` |

### Invalid/unscored (`invalid_unscored`)

No valid numerical quality result is available for these systems.

| System | Scope | Queries | Candidates | Family | Promotion | Evidence | nDCG@10 | MRR | Per-query | Artifact |
|---|---|---:|---:|---|---|---|---:|---:|---|---|
| `e5_initial_terminal_token_invalid` | `invalid_unscored` | 322 | 150 | candidate_label_matching | `invalid` | `invalid` | — | — | no | `tracked_report_snapshot` |
| `qwen8b_initial_terminal_token_invalid` | `invalid_unscored` | 322 | 150 | candidate_label_matching | `invalid` | `invalid` | — | — | no | `tracked_report_snapshot` |
| `generated_listwise` | `invalid_unscored` | 12 | 20 | label_ranker_pilots | `stopped_unscored` | `invalid` | — | — | no | `tracked_report_snapshot` |
| `grouprank_7b` | `invalid_unscored` | — | — | label_ranker_pilots | `stopped_unscored` | `invalid` | — | — | no | `tracked_report_snapshot` |

### Constructed, not evaluated (`constructed_not_evaluated`)

A representation or profile exists without a complete ranking evaluation.

| System | Scope | Queries | Candidates | Family | Promotion | Evidence | nDCG@10 | MRR | Per-query | Artifact |
|---|---|---:|---:|---|---|---|---:|---:|---|---|
| `discovery_behavior_topic_v1` | `constructed_not_evaluated` | 7759 | 150 | behavior_profile | `constructed_not_evaluated` | `unscored` | — | — | no | `tracked_complete` |
| `source_alignment__k10` | `constructed_not_evaluated` | 322 | 150 | source_query_first | `diagnostic_only` | `unscored` | — | — | no | `local_artifact_missing` |
| `source_alignment__k20` | `constructed_not_evaluated` | 322 | 150 | source_query_first | `diagnostic_only` | `unscored` | — | — | no | `local_artifact_missing` |
| `source_alignment__k40` | `constructed_not_evaluated` | 322 | 150 | source_query_first | `diagnostic_only` | `unscored` | — | — | no | `local_artifact_missing` |

## Per-query evidence coverage

| System | Evidence | Per-query | Blocker / custody caveat |
|---|---|---|---|
| `discovery_behavior_topic_v1` | `unscored` | no | This artifact is a candidate profile construction, not a query-ranking effectiveness evaluation. |
| `source_alignment__k10` | `unscored` | no | The receipt names this selector candidate but reports no standalone aggregate or per-query ranking for it. |
| `source_alignment__k20` | `unscored` | no | The receipt names this selector candidate but reports no standalone aggregate or per-query ranking for it. |
| `source_alignment__k40` | `unscored` | no | The receipt names this selector candidate but reports no standalone aggregate or per-query ranking for it. |
| `bge_m3_contextual` | `aggregate_only` | no | The reviewed study preserves this aggregate row and artifact hashes, but no tracked complete ranking or per-query metric vector. |
| `contextual_e5` | `stored_per_query` | yes | — |
| `jina_v5_contextual` | `aggregate_only` | no | The reviewed study preserves this aggregate row and artifact hashes, but no tracked complete ranking or per-query metric vector. |
| `mpnet_raw_label` | `aggregate_only` | no | The reviewed study preserves this aggregate row and artifact hashes, but no tracked complete ranking or per-query metric vector. |
| `nomic_v15_contextual` | `aggregate_only` | no | The reviewed study preserves this aggregate row and artifact hashes, but no tracked complete ranking or per-query metric vector. |
| `qwen06b_contextual` | `aggregate_only` | no | The reviewed study preserves this aggregate row and artifact hashes, but no tracked complete ranking or per-query metric vector. |
| `qwen4b_contextual` | `aggregate_only` | no | The reviewed study preserves this aggregate row and artifact hashes, but no tracked complete ranking or per-query metric vector. |
| `qwen8b_contextual` | `aggregate_only` | no | The reviewed study preserves this aggregate row and artifact hashes, but no tracked complete ranking or per-query metric vector. |
| `bge_small_en_v1_5@1` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `bge_small_en_v1_5@10` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `bge_small_en_v1_5@100` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `bge_small_en_v1_5@20` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `bge_small_en_v1_5@3` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `bge_small_en_v1_5@5` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `bge_small_en_v1_5@50` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `dense_nested` | `stored_per_query` | yes | — |
| `e5_small_v2@1` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `e5_small_v2@10` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `e5_small_v2@100` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `e5_small_v2@20` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `e5_small_v2@3` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `e5_small_v2@5` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `e5_small_v2@50` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `minilm_l6@1` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `minilm_l6@10` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `minilm_l6@100` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `minilm_l6@20` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `minilm_l6@3` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `minilm_l6@5` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `minilm_l6@50` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `mpnet_base_v2@1` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `mpnet_base_v2@10` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `mpnet_base_v2@100` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `mpnet_base_v2@20` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `mpnet_base_v2@3` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `mpnet_base_v2@5` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `mpnet_base_v2@50` | `aggregate_only` | no | The fixed-grid receipt stores only full-development aggregates for this configuration. |
| `a1_top2_k10` | `stored_per_query` | yes | Artifact custody: construction_artifact_missing. |
| `a1_top2_k10_dense_equal` | `stored_per_query` | yes | Artifact custody: construction_artifact_missing. |
| `r1_multires_max` | `stored_per_query` | yes | Artifact custody: construction_artifact_missing. |
| `r1_multires_max_dense_equal` | `stored_per_query` | yes | Artifact custody: construction_artifact_missing. |
| `x1_multires_top2` | `stored_per_query` | yes | Artifact custody: construction_artifact_missing. |
| `x1_multires_top2_dense_equal` | `stored_per_query` | yes | Artifact custody: construction_artifact_missing. |
| `hard_cluster_k20_preflight` | `aggregate_only` | no | The tracked preflight receipt preserves only aggregate metrics and mechanism diagnostics, not query IDs, rankings, raw score arrays, or a complete per-query metric vector. |
| `hard_cluster_k40_preflight` | `aggregate_only` | no | The tracked preflight receipt preserves only aggregate metrics and mechanism diagnostics, not query IDs, rankings, raw score arrays, or a complete per-query metric vector. |
| `local_domain_prior` | `aggregate_only` | no | The tracked report and aggregate receipt omit query IDs, rankings, raw arrays, and per-query metrics; rejected experiment artifacts were intentionally not retained. |
| `matched_neighbor_k20_preflight` | `aggregate_only` | no | The tracked preflight receipt preserves only aggregate metrics and mechanism diagnostics, not query IDs, rankings, raw score arrays, or a complete per-query metric vector. |
| `matched_neighbor_k40_preflight` | `aggregate_only` | no | The tracked preflight receipt preserves only aggregate metrics and mechanism diagnostics, not query IDs, rankings, raw score arrays, or a complete per-query metric vector. |
| `global_prior` | `aggregate_only` | no | The tracked receipt stores aggregate metrics only; its referenced ranking and per-query artifacts are not tracked in this checkout. |
| `tfidf_fixed_k10` | `aggregate_only` | no | The tracked receipt stores aggregate metrics only; its referenced ranking and per-query artifacts are not tracked in this checkout. |
| `bm25__query_response__aggressive__d100` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25__query_response__aggressive__d1000` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25__query_response__aggressive__d500` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25__query_response__basic__d100` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25__query_response__basic__d1000` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25__query_response__basic__d500` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25__response_only__aggressive__d100` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25__response_only__aggressive__d1000` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25__response_only__aggressive__d500` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25__response_only__basic__d100` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25__response_only__basic__d1000` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25__response_only__basic__d500` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25_rm3__query_response__aggressive__d100` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25_rm3__query_response__aggressive__d1000` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25_rm3__query_response__aggressive__d500` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25_rm3__query_response__basic__d100` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25_rm3__query_response__basic__d1000` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25_rm3__query_response__basic__d500` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25_rm3__response_only__aggressive__d100` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25_rm3__response_only__aggressive__d1000` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25_rm3__response_only__aggressive__d500` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25_rm3__response_only__basic__d100` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25_rm3__response_only__basic__d1000` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `bm25_rm3__response_only__basic__d500` | `aggregate_only` | no | The fixed response-grid receipt stores aggregates only; raw rankings and per-query artifacts are absent. |
| `response_sumscore_nested` | `aggregate_only` | no | The tracked receipt stores aggregate metrics only; its referenced ranking and per-query artifacts are not tracked in this checkout. |
| `response_top10sum_nested` | `aggregate_only` | no | The tracked receipt stores aggregate metrics only; its referenced ranking and per-query artifacts are not tracked in this checkout. |
| `dense_e5_equal` | `aggregate_only` | no | The semantic receipt references an ignored evaluation per-query artifact by digest, but that file and any corresponding issue-18 map are absent. |
| `dense_profile_dense_heavy` | `aggregate_only` | no | The semantic receipt references an ignored evaluation per-query artifact by digest, but that file and any corresponding issue-18 map are absent. |
| `dense_profile_e5_equal` | `aggregate_only` | no | The semantic receipt references an ignored evaluation per-query artifact by digest, but that file and any corresponding issue-18 map are absent. |
| `dense_profile_equal` | `stored_per_query` | yes | Artifact custody: local_artifact_missing. |
| `dense_profile_profile_heavy` | `stored_per_query` | yes | Artifact custody: local_artifact_missing. |
| `profile_only` | `stored_per_query` | yes | Artifact custody: local_artifact_missing. |
| `source_alignment_nested` | `aggregate_only` | no | The tracked receipt stores aggregate metrics only; its referenced ranking and per-query artifacts are not tracked in this checkout. |
| `target_response_same_k` | `aggregate_only` | no | The tracked receipt stores aggregate metrics only; its referenced ranking and per-query artifacts are not tracked in this checkout. |
| `e5_initial_terminal_token_invalid` | `invalid` | no | The reviewed study retains only the shared invalid-attempt history row; the superseded low scores and per-query outputs are intentionally not quality evidence. |
| `qwen8b_initial_terminal_token_invalid` | `invalid` | no | The reviewed study retains only the shared invalid-attempt history row; the superseded low scores and per-query outputs are intentionally not quality evidence. |
| `generated_listwise` | `invalid` | no | No valid complete permutation or quality result exists. |
| `grouprank_7b` | `invalid` | no | Neither tested protocol produced a machine-valid ranking, so no quality vector exists. |
| `firstmistral_listwise` | `restricted_diagnostic` | no | The tracked report contains restricted aggregates but no complete per-query nDCG and reciprocal-rank vector. |
| `pointwise_qwen` | `restricted_diagnostic` | no | The tracked report contains restricted aggregates but no complete per-query nDCG and reciprocal-rank vector. |
| `restricted_contextual_e5` | `restricted_diagnostic` | no | The tracked report contains restricted aggregates but no complete per-query nDCG and reciprocal-rank vector. |
| `restricted_qwen4b_contextual` | `restricted_diagnostic` | no | The tracked report contains restricted aggregates but no complete per-query nDCG and reciprocal-rank vector. |
| `generated_listwise_feasibility_6` | `restricted_diagnostic` | no | The tracked study reports only the 3/6 top-three feasibility count; candidate-set size and complete per-query metrics are not preserved. |
| `pointwise_qwen_feasibility_6` | `restricted_diagnostic` | no | The tracked study reports only the 5/6 top-three feasibility count; candidate-set size and complete per-query metrics are not preserved. |

## Topic and density slices

Relevant-candidate slices use `qrel_grade_gt_zero`. They overlap, and candidate labels are analysis handles, not official topics. Do not sum or average the slice means as if they were disjoint.

The topic TSV contains 1650 system/slice rows; the density TSV contains 33 rows.

Disjoint density support: `6_to_20_positives`=30, `at_most_5_positives`=248, `more_than_20_positives`=44.

- [Per-query scores](trec_2026_experiment_per_query.tsv)
- [Relevant-candidate slices](trec_2026_experiment_topic_slices.tsv)
- [Positive-density slices](trec_2026_experiment_density_slices.tsv)

## Program-wide caveats

- **`adaptive_program_overfit`:** Repeated program decisions reused the same 322 development queries; nested evaluation does not eliminate adaptive research overfitting.
- **`bootstrap_protocols_differ`:** Experiment families used different frozen bootstrap implementations; confidence intervals should not be compared digit-for-digit.
- **`confidence_not_correctness`:** Response attempt, confidence, profile similarity, and retrieval score do not establish factual correctness.
- **`development_only`:** All effectiveness results are development evidence; the 2026 hidden test is not yet available.
- **`external_encoder_taxonomy`:** Systems using pretrained encoders may be external runs under the track taxonomy.
- **`issue18_not_restorable`:** The ignored issue-18 construction tree has no durable remote URI and is not restorable from a fresh repository clone.
- **`missing_raw_artifacts`:** Several response, label-sweep, source-query, and rejected query-cluster raw or per-query artifacts are absent from this checkout.
- **`restricted_not_comparable`:** Restricted pilots are not numerically comparable with complete 322-query, 150-candidate evaluations.
- **`topic_slices_overlap`:** Relevant-candidate slices overlap and are relevance-target analysis handles, not official semantic topics.

## Source receipts

| Source | Repository path | SHA-256 |
|---|---|---|
| `dense_query_per_query` | [reports/trec_2026_dense_query_knn_per_query.tsv](../reports/trec_2026_dense_query_knn_per_query.tsv) | `ae4a0cf8bc7eb0642e1c0d315cafa3a56cf60b3d6bfb6ab63e80fb2d0de56437` |
| `dense_query_results` | [reports/trec_2026_dense_query_knn_results.json](../reports/trec_2026_dense_query_knn_results.json) | `3ca4252cb37f12e9272561422fd2a45d0dcbf741770d0f544efda8acf4ddf902` |
| `development_qrels` | [data/dev/llm_dev_qrels.txt](../data/dev/llm_dev_qrels.txt) | `3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35` |
| `development_queries` | [data/dev/llm_dev_data.tsv](../data/dev/llm_dev_data.tsv) | `862ff85381b8055cd927ada25c56666594ab1ec3c21e182beb20c58f6c637f3c` |
| `discovery_behavior_profiles` | [reports/trec_2026_llm_profiles.json](../reports/trec_2026_llm_profiles.json) | `86d592388514caabbbfeac8fe756c9a8a64526e065c75859a99f055ae1bc5224` |
| `expert_label_ranking_study` | [docs/results/2026-07-13-expert-label-ranking-study.md](../docs/results/2026-07-13-expert-label-ranking-study.md) | `9c77824edd0f1194a0f501c4504a0178b001b48d6bd77bae62d5bb2d73f8a71d` |
| `issue13_dense_aggregation_result` | [reports/issue_13_dense_aggregation_result.md](../reports/issue_13_dense_aggregation_result.md) | `45a0cfd6c227f86c8f2479708a68f003d87f37b30c910bb78363410e94b7c15b` |
| `issue13_matched_support_preflight` | [reports/issue_13_matched_support_preflight.json](../reports/issue_13_matched_support_preflight.json) | `31b509eae0660b1007b0116301b450dae1d850ed3696e2b1dca6fe9e60bda10f` |
| `issue18_artifact_custody` | [reports/trec_2026_issue18_pr17head_artifact_custody.json](../reports/trec_2026_issue18_pr17head_artifact_custody.json) | `1157a80231f7ec17718800d47cb56644014cf90550d9c888ad1f6a943d2ec6bd` |
| `issue18_evaluation` | [reports/trec_2026_issue18_pr17head_evaluation.json](../reports/trec_2026_issue18_pr17head_evaluation.json) | `60a45dc4317bd2b886b9b2fb993e912bbd2edbe69acca75e5bbcb53fd293e4d2` |
| `issue18_results` | [reports/trec_2026_issue18_pr17head_results.json](../reports/trec_2026_issue18_pr17head_results.json) | `f64d15bbb93d771abef0b9e51e9617ecc4f41439422767faf5839862f553b1a0` |
| `response_retrieval_results` | [reports/trec_2026_response_retrieval_results.json](../reports/trec_2026_response_retrieval_results.json) | `0d922c3b78137f0478cd2a70cda2bd89ec1621821a1f536aca665e359aea04bd` |
| `response_volume_results` | [reports/trec_2026_response_volume_normalization_results.json](../reports/trec_2026_response_volume_normalization_results.json) | `052a3972f877a119785640adfc06d21a3afc04d99eac5cae48063e3d83bf4f46` |
| `semantic_profile_results` | [reports/trec_2026_candidate_semantic_profiles_results.json](../reports/trec_2026_candidate_semantic_profiles_results.json) | `21cfe833c2d48658c6c8ecceffa5a2d9e5a27646e11bfc2a58769f80074ff31a` |
| `source_query_first_results` | [reports/trec_2026_source_query_first_results.json](../reports/trec_2026_source_query_first_results.json) | `4f13f59c24bbc2fa3cdbdf7199a0a8fb72ab6ef0bebdd6150a1c03a1c9299c00` |

## Rebuild command

```bash
uv run python scripts/build_experiment_ledger.py
```
