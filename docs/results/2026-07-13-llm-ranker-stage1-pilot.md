# LLM Expert-Label Ranker: Stage 1 Pilot

> Qrel-assisted exploratory development evidence; this is a six-question feasibility diagnostic, not a deployment comparison.

## Decision

Pointwise passes the small-pilot feasibility gate; listwise does not. Stage 2 is permitted, but no larger run starts automatically.

No Stage 2 run, model selection, fusion, or deployment action was started by this report.

## Gate

| Method | Relevant in top 3 | Subjects represented | Parse-safe | Pass |
| --- | ---: | ---: | --- | --- |
| Pointwise | 5/6 | 5 | yes | yes |
| Listwise | 3/6 | 3 | yes | no |

The gate requires at least 4/6 top-three hits, more than one successful subject stratum, and no success that depends on a parse failure.

## Six-question decision table

Best-rank order is pointwise / listwise / contextual Qwen4B / contextual E5. Rank deltas are LLM rank minus baseline rank, so negative is better for the LLM.

| Query | Stratum | Relevant | Best ranks P / L / Q4 / E5 | Top 3 P / L / Q4 / E5 | P−Q4 / P−E5 / L−Q4 / L−E5 | Listwise rotation agreement | Reviewed misses |
| --- | --- | --- | ---: | --- | ---: | ---: | --- |
| `56fc3b342603e7140040a037` Who founded autosegmental phonology? | issue_11 / linguistics | Linguistics | 1 / 1 / 1 / 1 | P: Linguistics, Languages, Scholars_and_academics_by_subject<br>L: Linguistics, Languages, Scholars_and_academics_by_subject<br>Q4: Linguistics, Scholars_and_academics_by_subject, Anthropology<br>E5: Linguistics, Languages, Human_communication | 0 / 0 / 0 / 0 | 0.899; 0 parse failures | none |
| `57265eeaf1498d1400e8dd76` What is Thuringia's highest mountain? | issue_11 / geography | Landscape | 3 / 4 / 5 / 6 | P: Geography, Geography_by_country, Landscape<br>L: Geography, Geography_by_country, Geography_stubs<br>Q4: Europe_geography_stubs, Geography_stubs, Geography<br>E5: Europe_geography_stubs, Geography_stubs, Geography_by_country | -2 / -3 / -1 / -2 | 0.818; 0 parse failures | listwise: overly broad label preference; qwen4b_contextual: overly broad label preference; e5_contextual: overly broad label preference |
| `570a34484103511400d595c8` What temperature are cask ales stored at before being tapped? | embedding_success / food-and-drink | Food_and_drink | 1 / 1 / 1 / 1 | P: Food_and_drink, Food_industry, Food_and_drink_by_country<br>L: Food_and_drink, Food_industry, Food_and_drink_by_country<br>Q4: Food_and_drink, Food_and_drink_by_country, Food_industry<br>E5: Food_and_drink, Food_and_drink_by_country, Food_industry | 0 / 0 / 0 / 0 | 0.758; 0 parse failures | none |
| `57298ef11d04691400779530` What type of prime distribution does the Riemann hypothesis propose is also true for short intervals near X? | embedding_success / mathematics | Fields_of_mathematics | 3 / 1 / 1 / 1 | P: Mathematics, Applied_mathematics, Fields_of_mathematics<br>L: Fields_of_mathematics, Applied_mathematics, Mathematics<br>Q4: Fields_of_mathematics, Mathematics, Applied_mathematics<br>E5: Fields_of_mathematics, Applied_mathematics, Mathematics | 2 / 2 / 0 / 0 | 0.788; 0 parse failures | none |
| `5726eb785951b619008f8276` What kind of immune system do bacteria have? | embedding_failure / biology | Entities, Life | 3 / 4 / 5 / 6 | P: Natural_sciences, Organisms, Life<br>L: Organisms, Natural_sciences, Branches_of_science<br>Q4: Organisms, Branches_of_science, Biota_by_conservation_status<br>E5: Organisms, Biota_by_conservation_status, Natural_sciences | -2 / -3 / -1 / -2 | 0.747; 0 parse failures | listwise: overly narrow label preference; qwen4b_contextual: overly narrow label preference; e5_contextual: overly narrow label preference |
| `5729068e1d04691400778f8f` WHen was the Parc de la Villette created? | embedding_failure / architecture | Culture_by_language, Infrastructure | 5 / 6 / 7 / 1 | P: Places, Topics_in_culture, History_by_location<br>L: Places, History_by_location, The_arts<br>Q4: History_by_location, Places, The_arts<br>E5: Infrastructure, The_arts, Europe_geography_stubs | -2 / 4 / -1 / 5 | 0.788; 0 parse failures | pointwise: related-topic confusion; listwise: related-topic confusion; qwen4b_contextual: related-topic confusion |

## Recalculation receipt

- Pointwise receipt SHA-256: `1336ffa9d37c5060cf6b898bc29368798541f991ac598ad5714eb2abb0a7fadc`
- Listwise receipt SHA-256: `cd76c60df4725efd863613c95f6428ccb7e0a27c4910d50199e7fabb18353fd2`
- Frozen selection SHA-256: `bc6a615aaabb9b1e4acae338044456cf30145cd54800407fcd9f9f12ddef1319`
- Contextual Qwen4B validation SHA-256: `a14e24eb3656f0cb8cdc05ff52b02f706609b8b868960b946d1a3e007e0f40ec`
- Contextual E5 validation SHA-256: `e3497d3eff4e19d5bb1549cf9fd28a9a3a5158573d13a03c288a999d0214c296`
- Qrels SHA-256: `3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35`
