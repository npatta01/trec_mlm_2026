# Muse Spark 1.2 Contributor — 50-question grounding canary

- Model: `muse-spark-1.2-contributor`
- Prompt variant: `reference`
- Endpoint: Meta native Responses API with `web_search`
- Policy: two calls per question; no application output-token or Search-use cap
- Selected questions: 50
- Completed records: 50
- Hard failures: 0
- Stage-2 finals: 50
- Stage-1 fallbacks: 0
- Resumed without API calls: 0

## Grounding and provenance

- Search actions: 853 (478 searches, 364 page opens)
- Questions with at least one Search query: 50
- URL citations in final answers: 365
- Follow-up responses reported incomplete: 0
- Opened-source host counts are a quality signal, not an adjudicated source-quality score:
  - `en.wikipedia.org`: 106
  - `www.britannica.com`: 7
  - `www.ssa.gov`: 4
  - `alchetron.com`: 4
  - `www.allmusic.com`: 3
  - `www.discogs.com`: 3
  - `www.esa.int`: 3
  - `history.state.gov`: 3
  - `www.govinfo.gov`: 3
  - `www.imdb.com`: 3
  - `www.liquisearch.com`: 2
  - `rhs.org.uk`: 2

## Usage and length

- Input tokens: 13617283
- Output tokens: 254395
- Total tokens: 13871678
- Answer characters (min / median / p95 / max): 1980 / 3375 / 6355 / 8371
- Latency seconds (median / p95): 108.8 / 194.2
- The API responses provide usage counts, but this report does not infer a dollar cost without the account's applicable Meta billing rate.

## Failures

- None.

## Canary boundary

This run is bounded to the selected 50 development questions. The private cache retains both raw Responses payloads, request payloads, response IDs, Search actions, URL annotations, usage, and retry/fallback diagnostics for audit. No full-dataset run was started.

Private cache: `artifacts/muse_canary_50_reference.json`
Selected IDs: 5ae622105542992ae0d16239, 5ae6168f5542996de7b71b47, 57325f78e99e3014001e6732, 5a8f49545542997ba9cb321f, 5ac307425542990b17b154ef, 56e2419acd28a01900c67c2a, 5ac1228e554299294b2190b5, 5a88abb15542994846c1ceb2, 5a7fa8475542992097ad2fad, 5ab9e3d25542994dbf019889, 57296f3caf94a219006aa3fc, 57326051e17f3d1400422905, 5ac190e55542994ab5c67d7f, 5ae369e05542992e3233c3ec, 571a7b2b10f8ca1400305097, 5a7195a15542994082a3e882, 5abcda1055429965836004d5, 5ab801465542995dae37ea13, 5adcec005542990d50227d60, 5ae076b055429945ae959356, 57260b4389a1e219009ac188, 5ae17a385542990adbacf7a1, 5addd09e5542997dc7907040, 5ab1da74554299340b52540d, 5ab780e65542995dae37e942, 5ade8c795542997c77adee4f, 5a7536395542996c70cfaeae, 570d7ed6b3d812140066da07, 5ae6bb6c55429908198fa6a0, 5abbe0765542992ccd8e7fcc, 5adce8c15542992c1e3a249d, 5a836f19554299123d8c211e, 5ab60430554299710c8d1e51, 5a8c7c9a554299653c1aa094, 5a7bb2a4554299042af8f798, 5a791cf855429974737f798f, 5aba73cc55429901930fa7d4, 56f71740711bf01900a44932, 5a7f67375542995d8a8dde5b, 5ac19a4255429964131be253, 5ae0ab4955429945ae959405, 5a74898a55429979e2882980, 5ae336ef5542991a06ce995d, 5adec29155429939a52fe974, 57302972b2c2fd140056899e, 57320bb80fdd8d15006c6713, 5ae00b1f554299025d62a3c0, 5ae3f0025542995ad6573cab, 5a76b0cc554299373536015e, 5ac4e4595542996feb3fe968
