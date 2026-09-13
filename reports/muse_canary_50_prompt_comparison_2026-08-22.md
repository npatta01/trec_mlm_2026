# Muse 50-question prompt comparison

Date: 2026-08-22
Model: `muse-spark-1.2-contributor`
Scope: the same 50 development questions, two grounded Responses API calls per question

## Runs

| Run | Prompt purpose | Records | Hard failures | Stage-2 finals | Final answers with citations |
| --- | --- | ---: | ---: | ---: | ---: |
| Original | Direct answer with citations | 50 | 0 | 50 | 50 |
| Revised | Structured reference record for grading other LLMs | 50 | 0 | 50 | 50 |

The revised run used a separate cache, so the original results were not overwritten and the comparison did not make duplicate calls for the original variant.

## Operational comparison

| Metric | Original | Revised | Change |
| --- | ---: | ---: | ---: |
| Search actions | 729 | 853 | +17.0% |
| Search queries | 411 | 478 | +16.3% |
| Page opens | 310 | 364 | +17.4% |
| Final citation annotations | 330 | 365 | +10.6% |
| Input tokens | 12,072,079 | 13,617,283 | +12.8% |
| Output tokens | 206,037 | 254,395 | +23.5% |
| Total tokens | 12,278,116 | 13,871,678 | +13.0% |
| Answer length: median | 1,332 chars | 3,375 chars | +153.5% |
| Answer length: p95 | 4,492 chars | 6,355 chars | +41.5% |

There was no application-imposed Search or output-length cap in either run. The larger revised answers are a consequence of the requested reference sections, not an arbitrary truncation rule.

## Quality review

The revised prompt improved the reference shape across the set: it put a candidate answer first, made equivalent answers explicit, separated genuinely necessary qualifications from evidence, and preserved provenance. It also improved several answer interpretations, including:

- the release-date answer for Röyksopp's *Senior*, where territory/date distinctions were made explicit;
- the Capcom Pro Tour ownership answer, where the direct company relationship was clearer;
- the open-ended “function other than protection” question, where eyeglasses was identified as the closest phrase-matched example while valid biology examples were also preserved;
- true/false and comparison questions, where the canonical answer was easier to grade.

One substantive semantic regression was found:

- **Q10 — “What type of vegetation does Lythrum and Erythrina have in common?”** The original answer led with shared wetland/riparian/hydrophytic vegetation. The revised answer led with “deciduous perennial” and “herbaceous perennial,” which answers a plant life-form question rather than the likely ecological-vegetation question. Its own caveats acknowledge that *Erythrina* is often woody and can be evergreen, so the canonical wording is too strong for a grading reference. The internal study also classifies this item under `Natural_environment`, which supports treating the habitat interpretation as the safer canonical answer.

No mechanical failure, missing citation coverage, or other clear answer-changing regression was found in the remaining 49 records during the canary review. This is a focused manual review, not a claim that every citation or botanical detail has been independently adjudicated.

## Recommendation

Do not adopt the revised prompt exactly as run. Keep its five-section reference format, but add an instruction to answer the exact dimension asked by the question before generating variants—for example, distinguish habitat/ecological vegetation from life form, morphology, or taxonomy. Then rerun Q10 and a small set of other “what type/common” questions before scaling beyond the canary.

The revised prompt is otherwise operationally sound for the intended use as a reference point for judging other LLM answers. Its main tradeoff is cost/latency: roughly 13% more total tokens and 24% more output tokens on this sample.

## Artifacts

- Original run: `artifacts/muse_canary_50.json`
- Revised run: `artifacts/muse_canary_50_reference.json`
- Revised-run report: `reports/muse_canary_50_reference_2026-08-22.md`
