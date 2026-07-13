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

## Scaled feasibility gate

**Result: FAIL.**

This heuristic was predeclared for this report but was not frozen before Stage 2 execution. It is exploratory and post-hoc, not a statistical test or confirmatory decision rule.

Expand only if pointwise top-three relevant count is at least the best embedding baseline and pointwise overall nDCG@10 is not lower than the best embedding baseline.
The benchmark selected by overall nDCG@10 is `e5_contextual`.

| Condition | Pointwise | Best embedding baseline | Pass |
| --- | ---: | ---: | --- |
| Relevant label in top 3 | 5/12 | 8/12 | no |
| Overall nDCG@10 | 0.4820 | 0.7232 | no |

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

## Question-level comparison

Best-rank deltas are pointwise minus baseline, so negative favors pointwise. Miss labels are deterministic rules based on the pointwise best relevant rank: top-3 hit, near miss at ranks 4–5, or deep miss at rank 6+.

| Query | Stratum | Relevant labels | Pointwise best / top 3 | Qwen4B best / top 3 | E5 best / top 3 | P−Q4 / P−E5 | Rule-based outcome |
| --- | --- | --- | --- | --- | --- | ---: | --- |
| `5727f1163acd2414000df052` | issue_like_clear | Educational_organizations (2) | 7 / Educators, Education, Scholars_and_academics_by_subject | 1 / Educational_organizations, Education, Education_by_subject | 1 / Educational_organizations, Education, Education_by_subject | 6 / 6 | deep_miss_rank_6_plus: Best relevant label Educational_organizations was rank 7; pointwise top three were Educators, Education, Scholars_and_academics_by_subject. |
| `57326d2f0fdd8d15006c6ad5` | issue_like_clear | Literature (2) | 2 / Topics_in_culture, Literature, Humanities | 1 / Literature, Books, Publishing | 1 / Literature, Publishing, Books | 1 / 1 | top3_hit: Best relevant label Literature was rank 2; pointwise top three were Topics_in_culture, Literature, Humanities. |
| `5a88b0ef5542994846c1ceb9` | issue_like_clear | Food_and_drink (2) | 6 / Topics_in_culture, Natural_materials, Food_industry | 1 / Food_and_drink, Natural_materials, Food_industry | 1 / Food_and_drink, Food_and_drink_by_country, Cuisine | 5 / 5 | deep_miss_rank_6_plus: Best relevant label Food_and_drink was rank 6; pointwise top three were Topics_in_culture, Natural_materials, Food_industry. |
| `5a8e351c5542990e94052a8b` | issue_like_clear | Flora (2) | 4 / Organisms, Natural_sciences, Nature | 1 / Flora, Organisms, Biota_by_conservation_status | 3 / Biota_by_conservation_status, Biota_by_country, Flora | 3 / 1 | near_miss_rank_4_or_5: Best relevant label Flora was rank 4; pointwise top three were Organisms, Natural_sciences, Nature. |
| `5727aad82ca10214002d9341` | contextual_qwen_success | Politics (2) | 3 / Politics_by_location, Political_history, Politics | 2 / Political_history, Politics, Politics_by_period | 5 / Politics_by_period, Politics_by_location, Politics_stubs | 1 / -2 | top3_hit: Best relevant label Politics was rank 3; pointwise top three were Politics_by_location, Political_history, Politics. |
| `5a7276d55542997f827839e7` | contextual_qwen_success | Sports_venues (2) | 3 / Geographical_areas, Places, Sports_venues | 2 / Sport_by_location, Sports_venues, History_by_location | 1 / Sports_venues, Sport_by_location, Geographical_areas | 1 / 2 | top3_hit: Best relevant label Sports_venues was rank 3; pointwise top three were Geographical_areas, Places, Sports_venues. |
| `5a7a18145542990783324e4f` | contextual_qwen_success | Popular_music (1) | 6 / Geography_by_location, Geography_by_country, Geographical_areas | 2 / Geographical_areas, Popular_music, Geography_by_country | 1 / Popular_music, Culture_by_location, Wikipedia_categories_named_after_mass_media | 4 / 5 | deep_miss_rank_6_plus: Best relevant label Popular_music was rank 6; pointwise top three were Geography_by_location, Geography_by_country, Geographical_areas. |
| `5a9091a95542995651fb5190` | contextual_qwen_success | Natural_environment (2) | 4 / Geography, Geography_by_location, Geographical_areas | 2 / Geographical_areas, Natural_environment, Geography_by_period | 1 / Natural_environment, Geographical_areas, Geography | 2 / 3 | near_miss_rank_4_or_5: Best relevant label Natural_environment was rank 4; pointwise top three were Geography, Geography_by_location, Geographical_areas. |
| `5727b7a1ff5b5019007d9348` | contextual_qwen_failure | Computing (2) | 1 / Computing, Technology, History_of_technology | 7 / History_of_technology, Technology_by_type, Technology | 5 / Technology_by_type, Technology, Technology_systems | -6 / -4 | top3_hit: Best relevant label Computing was rank 1; pointwise top three were Computing, Technology, History_of_technology. |
| `5a7f5e6a5542992097ad2f42` | contextual_qwen_failure | Sports_events (1) | 3 / Sports_by_type, Sports, Sports_events | 6 / Sports_by_type, Sports, Sport_by_period | 5 / Sports_by_type, Sports_venues, Sports | -3 / -2 | top3_hit: Best relevant label Sports_events was rank 3; pointwise top three were Sports_by_type, Sports, Sports_events. |
| `5ab801465542995dae37ea13` | contextual_qwen_failure | Human_behavior (1) | 8 / Diseases_and_disorders, Popular_culture, Health | 4 / Popular_music, Diseases_and_disorders, Popular_culture | 2 / Diseases_and_disorders, Human_behavior, Popular_music | 4 / 6 | deep_miss_rank_6_plus: Best relevant label Human_behavior was rank 8; pointwise top three were Diseases_and_disorders, Popular_culture, Health. |
| `5ab9e3d25542994dbf019889` | contextual_qwen_failure | Natural_environment (1) | 4 / Natural_sciences, Organisms, Flora | 7 / Flora, Biota_by_conservation_status, Organisms | 5 / Biota_by_conservation_status, Flora, Biota_by_country | -3 / -1 | near_miss_rank_4_or_5: Best relevant label Natural_environment was rank 4; pointwise top three were Natural_sciences, Organisms, Flora. |

## Terminal listwise failure

- Receipt SHA-256: `d83ebfb60a7a5ea6b1d209ec99f2720b324b6872c751ab482bd2c3ce8493a2f8`
- Failed parse: `57326d2f0fdd8d15006c6ad5` rotation `0`
- Planned/log-observed requests: 36/36
- Persisted raw outputs: 0; rankings available: no; OOM: no.
- Exact error: `pilot listwise parse failure for 57326d2f0fdd8d15006c6ad5/0: pilot ranking has duplicates, omissions, or unknown handles`

The log proves request processing reached 36/36, not that 35 usable outputs existed. The program raised before any raw output or ranking artifact was persisted.
The new failure writer applies only to future attempts. It cannot recover outputs from this incident, whose honest persisted raw-output count remains zero.

## Recalculation receipt

- Pointwise receipt SHA-256: `7e5d74a777acda1c044f7140e2fe40e4c794122eca0c7f1bc26d2b9580fe7882`
- Frozen selection SHA-256: `668515c30a80473aaaa2dc530f50dfd3b7731711e9f9adf70074ebb95aa98b9d`
- Matching code revision: `5717e091e4a19a663ad1ecae56ab661e5152484f`
- Matching smoke receipt SHA-256: `a88525a22ff51ec67600fd1e3e5ac3d7553feb48d3c01bab831a7232729b36b3`
- Qrels SHA-256: `3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35`
