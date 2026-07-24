# Gemini 3.6 Flash pricing and grounding audit

Accessed: **2026-07-24**

This note records the current official Google pricing relevant to the TREC
reference-answer run. It is a **current repricing basis**, not evidence of what
Google actually charged when the run executed.

## Current official pricing

For `gemini-3.6-flash` on the paid Gemini Developer API Standard tier, Google
currently lists:

- Input: **$1.50 per 1 million tokens**
- Output, including thinking tokens: **$7.50 per 1 million tokens**
- Context-cache input: **$0.15 per 1 million tokens**

Batch and Flex are each listed at $0.75 per 1 million input tokens and $3.75 per
1 million output tokens. Priority is listed at $2.70 per 1 million input tokens
and $13.50 per 1 million output tokens.

Source: [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.6-flash),
last updated 2026-07-21 UTC. Google Cloud's
[Agent Platform pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing#gemini-3)
independently lists the same Standard token prices.

## Google Search grounding

The current Gemini Developer API pricing table describes a free allowance of
**5,000 prompts per month, shared across Gemini 3 models**, followed by
**$14 per 1,000 Google Search queries**. Its footnote clarifies that a single
customer request can cause one or more searches and that billing is for each
individual search query performed.

The Google Cloud pricing page states the allowance directly as **5,000 search
queries per month**, aggregated across all Gemini 3 models, then $14 per 1,000
search queries. It also says input tokens supplied by Google Search grounding
are not charged.

Accordingly, the audit should:

1. Treat **an individual search query**, not a grounded model request, as the
   overage billing unit.
2. Use **5,000 per month shared across Gemini 3** as the current free grounding
   allowance.
3. Avoid presenting request counts as exact search-query counts unless the
   response metadata records every search performed.

The Developer API table mixes the terms “prompts” for the allowance and
“search queries” for overage billing. A conservative estimate should therefore
report a range or clearly state any one-search-per-grounded-request assumption.

Sources:

- [Gemini Developer API pricing](https://ai.google.dev/gemini-api/docs/pricing#gemini-3.6-flash),
  last updated 2026-07-21 UTC
- [Google Cloud feature pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing#google-models),
  accessed 2026-07-24

## The unsupported 1,500 assumption

The current official [Gemini API rate-limit page](https://ai.google.dev/gemini-api/docs/rate-limits),
last updated 2026-07-21 UTC, does **not** document a 1,500-request or
1,500-query grounding allowance for Gemini 3.

That page describes operational rate limits—requests per minute, input tokens
per minute, and requests per day. It says limits apply per project rather than
per API key and that daily request quotas reset at midnight Pacific time. These
operational limits are distinct from the monthly free grounding allowance on
the pricing page.

Therefore, neither **1,500 per day** nor **5,000 per day** should be used as the
current grounding pricing allowance without dated evidence from the run period.

## Historical actual spend versus repricing

Keep three values distinct:

1. **Actual billed spend:** the authoritative historical cost from AI Studio or
   Cloud Billing for the relevant project, Gemini API SKU, and usage-date range.
2. **Run-time estimate:** a reconstruction using the rates and allowance
   documented when the run executed, if dated evidence was preserved.
3. **Current repricing estimate:** the recorded token and search usage evaluated
   with the official prices above, explicitly labeled “priced as of
   2026-07-24.”

Do not overwrite actual spend or a dated run-time estimate with today's rates.
Google's billing guide recommends using Cloud Billing reports, filtering to the
Gemini API, grouping by SKU, and selecting the historical usage-date range. It
also warns that usage and cost reporting can lag by 24 hours or more.

Source: [Gemini API billing](https://ai.google.dev/gemini-api/docs/billing#evaluate-your-billing-account-spend),
last updated 2026-07-07 UTC.
