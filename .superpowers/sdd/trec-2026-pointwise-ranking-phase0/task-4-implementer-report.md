# Task 4 implementer report

## Assessment

Independent analytical validation passed without a scientific discrepancy. A
standalone stdlib validator read only the frozen development qrels, five emitted
TREC runs, sanitized receipts, and complete IDs-only selector receipt. It did
not import or call project evaluator or metric functions and made no inference,
pairwise, judge, pointwise-regeneration, embedding, or API call.

## Independent method and structural checks

- Parsed raw four-column qrels and six-column TREC runs; required unique rows,
  exact ranks 1–150, identical query and candidate identities, 322 queries, 150
  candidates, and 48,300 rows in the qrels and in each of five runs.
- Implemented raw-gain DCG/IDCG at rank 10 and first grade-greater-than-zero
  reciprocal rank from first principles. Recomputed 3,220 per-query metric
  cells, five macro summaries, pointwise-only/combined deltas, W/T/L at
  tolerance `1e-12`, and all zero transitions.
- Reproduced each paired 95% percentile interval with 10,000 query-grain
  resamples from `random.Random(31).randrange`, CPython's MT19937-backed integer
  sampling stream, and linear percentiles. `random.choices` is a different
  MT19937 consumption pattern and correctly did not reproduce the receipt.
- Rebuilt the three query-slice partitions and both stable candidate quartile
  partitions. All nine query slices and all 40 candidate family/label/system
  rows reconciled after independently filtering both rankings and qrels before
  metric computation.
- Reconciled dense macro metrics and all 322 dense per-query rows against both
  tracked dense receipts within absolute tolerance `5e-12`.
- Reconciled the complete selector receipt: 500 unique selected pairs across
  500 unique source questions, 1,000 total candidate uses, maximum use 20, cap
  20, and exact agreement with the sanitized selector summary. The frozen
  construction preflight established zero overlap for all 322 development
  query IDs and normalized-text fingerprints against all 6,816 pointwise
  source-question IDs and fingerprints.

## Exact aggregate and paired results

| System | nDCG@10 | MRR | Zero nDCG |
|---|---:|---:|---:|
| control | 0.35985474409504775 | 0.40406330811168817 | 116 |
| prior-only | 0.10216984523370287 | 0.1524425429094271 | 246 |
| routing-only | 0.15972944894483668 | 0.21196069887757296 | 201 |
| pointwise-only | 0.11882482433832028 | 0.1708636403214092 | 231 |
| combined | 0.2113240933733129 | 0.2691192482181426 | 172 |

- Pointwise-only minus control nDCG@10: `-0.24102991975672747`,
  95% CI `[-0.28094843257965324, -0.20031851305906528]`, W/T/L
  `45/99/178`. Its MRR delta is `-0.233199667790279`, 95% CI
  `[-0.27804822849467936, -0.18782524074928286]`, W/T/L `75/28/219`.
- Combined minus control nDCG@10: `-0.14853065072173485`, 95% CI
  `[-0.18381098850790814, -0.1142669617790963]`, W/T/L `55/109/158`.
  Its MRR delta is `-0.1349440598935456`, 95% CI
  `[-0.17528312887959557, -0.09447628367917509]`, W/T/L `86/46/190`.
- Combined zero transitions are 100 zero/zero, 16 dense-zero rescues, 72
  dense-nonzero regressions, and 134 nonzero/nonzero.

## Slice highlights and gate

- Query counts reconcile at density `248/30/44`, dense status `116/206`, and
  similarity quartiles `81/80/81/80`. Combined trails control in every density
  and similarity slice; on the 116 dense-zero queries it reaches nDCG@10
  `0.0319283607292224`, while on 206 dense-nonzero queries it falls from
  `0.5624913961097348` to `0.31234304961949977`.
- Both candidate quartile families reconcile at `38/37/38/37`. Combined trails
  control in seven of eight candidate slices; the only positive slice is
  smoothed-prior quartile 2 (`0.1917806007261801` versus
  `0.1888851417509`).
- Four of five effectiveness checks fail: combined nDCG point estimate, lower
  interval bound, MRR floor, and zero-query reduction. Routing-only exceeds
  prior-only. The exact decision is `fail`; despite selector availability, the
  required recommendation is `stop_pairwise_work`.

## Publication boundary and caveats

The three tracked reports passed scans for absolute/private paths, raw question,
response, or reference text fields, prompts, credentials, model-call/token
state, and API artifacts. The validated JSON preserves the unmodified evaluator
result hash and artifact-manifest provenance before adding the validation block.

This evidence is from previously seen organizer development queries and makes
no hidden-test claim. Pointwise support is conditional audited-response quality,
not qrel gold or coverage; audited exposure is uneven across candidates. Ranking
construction was qrel-free and enforced the exact ID and normalized-text
no-overlap boundary before opening qrels. No pairwise or API call occurred. The
negative gate is binding and pairwise work stops.

## Hashes and commit scope

- Qrels: `3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35`.
- Runs — control `d3b211eb26723bc66fcf9dbe6c4a2612931d51508d2c61fa4acc574fbb3bf4fa`,
  prior `833f3f1ff16b068259d9b19b48db0767898b97dca7e2d5ec6e91a9b8bfe7ccf3`,
  routing `683bc26b6b11797b7135955010f81b1072beb499b8029fec84877a64d3b338b9`,
  pointwise `b1f3e9cd464a8c9392f41d533137572e082365c864913b59691392f494ab5e8c`,
  combined `32653a55f10f3ea734b00034e4ab9e9030fadaed71a537787b208c3da76812d7`.
- Original evaluator exports — results
  `449e325582b29dff37d917a55f48837a4167e0845c1da4f42a672955ea1d0be2`,
  per-query `e2193404d8f1052199b6b6a396d78556e48bb84b145a6e7adda125634218d889`,
  candidate slices
  `4d56304bf73f6d0e6bfd3c5612b922c3c3ce6e43f349857fc180aab9d83f3329`.
- Validated results JSON:
  `48ffaa81c77fde9dafb67ff9bbc29de98e51b8735a9320b2117f5662e31a7e6d`.
- Task 3 parent: `8bfb80b53c061deb787802e2d6524585ee6e7929`. Task 4 is one commit containing
  exactly this report and the three report exports; resolve its hash with
  `git log -1 --format=%H --` on this file, avoiding a self-referential hash in
  committed content.
