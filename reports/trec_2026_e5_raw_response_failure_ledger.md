# E5 raw-response refinement failure ledger

## Scope

This ledger covers the full 322-query **development** set. It is not a held-out
evaluation. Development qrels annotate the already frozen rankings; they must
not be used to tune the current `0.8/0.2`, top-10, or equal-RRF settings.

Primary system: `raw_hierarchical_dense_equal`.

## Current failure inventory

| Cohort | Queries | Meaning | Priority |
|---|---:|---|---|
| Zero nDCG@10 | 92 | No relevant candidate reached the top 10 | Track every run |
| Persistent all-system zeros | 62 | Dense, profile, prior fusion, raw-only, and new fusion all score zero | Highest research priority |
| Fusion-suppressed raw successes | 24 | New fusion is zero while `raw_hierarchical` is nonzero | Diagnose fusion/rank conflict on future held-out data |
| New zero regression | 1 | Prior fused baseline was nonzero; new fusion is zero | P0 regression |
| Prior zeros rescued | 8 | Prior fused baseline was zero; new fusion is nonzero | Preserve as regression tests |

The complete interactive explorer contains all 322 queries, defaults to weakest
nDCG first, and can filter to zero-nDCG failures. It includes query text,
top-20 rankings, development labels, every positively labeled candidate, and
its system rank. Raw historical responses are deliberately excluded from the
rendered artifact.

## P0: new zero regression

| Query ID | Query | Prior fused nDCG | New fused nDCG | Raw-only nDCG |
|---|---|---:|---:|---:|
| `56ddd58e66d3e219004dad10` | What laws did Healthcare Advocates accuse Internet Archive of having broken? | 0.177239 | 0.000000 | 0.000000 |

## Largest nDCG losses versus the prior fused baseline

| Query ID | Query | Prior | New | Delta |
|---|---|---:|---:|---:|
| `5727b7a1ff5b5019007d9348` | When was the Type-C Specification 1.0 finalized? | 1.000000 | 0.630930 | -0.369070 |
| `5a8e351c5542990e94052a8b` | Do Sinningia or Elodea have more species? | 1.000000 | 0.630930 | -0.369070 |
| `5a77882055429949eeb29ed4` | What man, widely considered to be the greatest sprinter of all time, participated in the 2012 Summer Olympics? | 0.571429 | 0.386853 | -0.184576 |
| `56ddd58e66d3e219004dad10` | What laws did Healthcare Advocates accuse Internet Archive of having broken? | 0.177239 | 0.000000 | -0.177239 |
| `5a7fe12a5542994857a7681a` | New Scientist and Highlights for Children are both which type of publication? | 0.907228 | 0.760188 | -0.147040 |
| `56f827e6aef2371900625e5a` | What name has been used as a name for a number of mountainous areas across Europe? | 0.630930 | 0.500000 | -0.130930 |
| `5ac19a4255429964131be253` | What TV station shows both Real World/Road Rules Challenge: The Inferno and The Challenge? | 0.630930 | 0.500000 | -0.130930 |
| `Q1751` | what are club seats | 0.318639 | 0.231724 | -0.086915 |
| `5ae53a7a55429960a22e027c` | Linghai and Pingdu are from which country? | 1.000000 | 0.919721 | -0.080279 |
| `5a7a46785542990783324ee5` | What type of publication is Popular Mechanics and Sojourners? | 0.867087 | 0.797478 | -0.069609 |

## Rescued prior zeros to preserve

The new fusion rescued eight prior zero-score queries:

`56cc50756d243a140015eef9`, `56db74aae7c41114004b512f`,
`5706888d75f01819005e7bde`, `5723f6f7f6b826140030fd0d`,
`572f5875947a6a140053c89d`, `5a7298aa5542992359bc312e`,
`5ade0b785542990dbb2f7f34`, and `5ae3f0025542995ad6573cab`.

## How this ledger should be used

1. Treat the 62 all-system zeros as the persistent hard set.
2. Treat the single new zero and the ten largest losses as regression cases.
3. Track the eight rescues so later changes do not silently lose them.
4. Investigate the 24 fusion-suppressed raw successes as a mechanism cohort,
   but make any fusion change on a new split or held-out queries—not this set.
5. For every future run, regenerate this cohort table alongside aggregate
   metrics; an average improvement does not erase query-level regressions.

## Canonical evidence

- Sealed evaluation: `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/evaluation/evaluation.json`
- Rankings: `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/refinement_run1/rankings/`
- Selected raw evidence IDs: `/home/npatta01/data/competitions/trec_mlm_2026/.git/shared-data/trec-2026/e5_raw_response_refinement/refinement_run1/selected_record_ids.json`
- Interactive explorer: `/home/npatta01/codex-rendered/plans/e5-raw-response-failure-explorer.html`
