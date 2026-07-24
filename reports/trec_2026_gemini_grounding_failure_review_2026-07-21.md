# Gemini Search grounding on the 28 audited failure cases

## Technical summary

- **Grounding helped materially, but it was not a cure-all.** It produced **12 clear rescues** and **6 partial rescues** across the 28 flagged cases.
- **The original denominator was noisy.** Three of the 28 locked serious-error labels were false positives on re-review; among the remaining 25 true failures, grounding or grounded regeneration improved 19.
- **Do not route from question shape alone.** The best pre-answer rule tested would send 272/400 questions to search to catch only 19/25 failures.
- **Use a two-pass disagreement router.** Ground when Perplexity is judged better than ungrounded Gemini or the comparison is unresolved. In the locked pilot that selected 52/400 and caught 19/25 true failures and 14/18 grounding-rescueable cases. This is an upper bound until a local judge reproduces the manual comparison labels.

## Grounding fixed exact-fact and hidden-entity failures most reliably

Clear rescues clustered around names, titles, organizations, race winners, road chains, and biological classifications. Search was weaker on malformed questions, historical-vs-current ambiguity, and requests whose intended source could not be inferred from the wording.

## What was measured

The cohort is the 28 Gemini answers labeled major or fatal in the locked 400-question review. Each question was regenerated with the Google Search tool enabled, using a distinct cache. `Clear rescue` means the material error was corrected; `partial rescue` means the answer improved but ambiguity or incompleteness remained; `no rescue` means the decisive defect remained. A regeneration-only rescue is reported separately because Search was enabled but not invoked.

The run completed **28/28** requests with no API failures. Search was actually invoked on **26/28**; among the **23 search-using true failures**, 12 were clear rescues and 6 were partial rescues.

## A disagreement router is much narrower than question-shape heuristics

The comparison trigger is decision-useful because it uses the two answers we already possess. In this pilot it retained 76% of audited failure recall at 13% coverage; the best tested question-shape rule required 68% coverage for the same recall.

## Selective grounding should remain inside the original budget

Gemini used Search on **26/28** requests, issuing **53 search queries**—**2.04 per search-using request** and **1.89 across all enabled requests**. Applying the 13% disagreement route to all 7,759 questions projects to about **1,009 grounded second passes** and **1,910 search queries**. If the account's shared **5,000 grounding-prompt monthly Gemini 3 allowance** is still available, the search surcharge is $0 and model tokens project to roughly **$3.97 extra**, or **$32.81 including the ungrounded full run**.

Blanket grounding is economically different: at the pilot's invocation and query rates it projects to about **14,687 searches**. Google's pricing switches after 5,000 free grounding prompts and then charges for each query, so plausible interpretations of how non-searching tool-enabled prompts count put the search surcharge around **$63–$73**, plus roughly **$30 in model tokens**. That estimate is directional because Gemini decides whether to search and how many queries to execute, the documentation does not expose the counter's exact scope, and the monthly allowance may already be partly consumed.

## What this experiment does not establish

- The 28 questions were selected because the original review called them failures; they are not a random sample of all 7,759 questions.
- Three original failure labels were wrong, demonstrating reviewer noise. The revised counts are a manual audit, not organizer ground truth.
- Search use and correction are associated in this paired rerun, but a fresh stochastic generation can improve without Search; one such case occurred here.
- The proposed disagreement trigger uses manually locked pairwise labels. A local pointwise/pairwise judge must be calibrated before its 13% coverage can be treated as operational.
- The grounded cache records Search query/chunk/support counts but intentionally omits raw URLs; decisive niche claims were spot-checked separately against authoritative or independent sources.

## Recommended next step

1. Generate the full ungrounded Gemini set and retain the existing Perplexity answers.
2. Train or prompt a local judge to emit `Gemini better`, `equivalent`, `Perplexity better`, or `uncertain`, using the 400 locked comparisons as calibration data.
3. Ground only `Perplexity better` and `uncertain`; validate first on the 52 pilot questions, including the 33 non-failure controls, to measure regressions as well as rescues.
4. Add deterministic triggers for explicit recency/current-state questions and exact hidden-entity chains even when the judge says tie.
5. Preserve both answers and the route decision. Do not silently replace a good ungrounded answer when grounding returns a less specific or contradictory result.

## Further questions

- Can a local judge reproduce the manual 52-question escalation set at high recall?
- How often does grounding harm the 33 selected pilot questions that were not Gemini failures?
- Should the final reference selector synthesize conflicting grounded and Perplexity evidence, or retain an explicit unresolved state?
