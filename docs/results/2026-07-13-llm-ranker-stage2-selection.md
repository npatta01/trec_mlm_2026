# LLM Expert-Label Ranker: Stage 2 Selection

This is qrel-assisted exploratory selection evidence prepared before Stage 2 inference. No model was loaded or run while creating these manifests.

The first four questions are manually fixed, issue-like examples with a direct question-to-topic mapping and one selected relevant label. The next four are distinct contextual-Qwen4B successes whose selected relevant label ranks in the full-run top three. The final four are distinct contextual-Qwen4B failures whose selected relevant label falls outside the full-run top three. All twelve IDs are new relative to Stage 1.

| Query ID | Selection stratum | Contextual Qwen4B relevant rank | Selected relevant label | Question |
| --- | --- | ---: | --- | --- |
| `5727f1163acd2414000df052` | issue-like clear | 1 | Educational_organizations | Who sets individual degree requirements at Northwestern? |
| `57326d2f0fdd8d15006c6ad5` | issue-like clear | 1 | Literature | Who wrote “A Kind of Alaska”? |
| `5a88b0ef5542994846c1ceb9` | issue-like clear | 1 | Food_and_drink | Paloma and Monkey Gland are what type of item? |
| `5a8e351c5542990e94052a8b` | issue-like clear | 1 | Flora | Do Sinningia or Elodea have more species? |
| `5727aad82ca10214002d9341` | contextual Qwen success | 2 | Politics | What was the rejection rate of the quota system by the French-speaking part of Switzerland? |
| `5a7276d55542997f827839e7` | contextual Qwen success | 2 | Sports_venues | Which former dog track was located in Wood Village, Oregon, in the Portland metropolitan area? |
| `5a7a18145542990783324e4f` | contextual Qwen success | 2 | Popular_music | In which country were Dinosaur Jr. formed and M83 based? |
| `5a9091a95542995651fb5190` | contextual Qwen success | 2 | Natural_environment | In what ocean was the fourth and final tropical cyclone of the 1991 hurricane season? |
| `5727b7a1ff5b5019007d9348` | contextual Qwen failure | 7 | Computing | When was the Type-C Specification 1.0 finalized? |
| `5a7f5e6a5542992097ad2f42` | contextual Qwen failure | 6 | Sports_events | Juan Sebastián Cabal and John Lloyd play what sport? |
| `5ab801465542995dae37ea13` | contextual Qwen failure | 4 | Human_behavior | What disorder did the song “You’re Only Human” refer to? |
| `5ab9e3d25542994dbf019889` | contextual Qwen failure | 7 | Natural_environment | What type of vegetation do Lythrum and Erythrina have in common? |

For every question, the qrel-free manifest contains the exact question text and 20 sorted candidate IDs only. The separately sealed judgment manifest records up to two highest-grade relevant labels, the first six qrel-zero labels in the contextual-Qwen4B ranking, and deterministic SHA-256-ordered qrel-zero negatives to reach 20 candidates.

- Selection SHA-256: `668515c30a80473aaaa2dc530f50dfd3b7731711e9f9adf70074ebb95aa98b9d`
- Sealed judgments SHA-256: `8ced65041023f25ab157fdfa9e6a34397505fed875c1e0687527638f4285e83c`
- Frozen Stage 1 report SHA-256: `d1582ca85388b2447fa28539f1f3077120d787e984a33ce637d713370e5e38c7`

The selection builder accepts Stage 2 preparation only after the Stage 1 report matches that frozen hash, its gate is passed, its source paths are approved, and a full source recomputation exactly reproduces the receipt.
