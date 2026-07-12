---
name: inspect-query-results
description: Build a self-contained query and candidate explorer from a completed TREC Million LLMs ranking run. Use after an experiment finishes, when a user asks to inspect queries, rankings, top candidate scores, candidate profiles, or related historical responses from an artifact directory containing predictions.tsv or a TREC run file.
---

# Inspect Query Results

Generate a lightweight, approach-agnostic HTML explorer. Keep the visible interface focused on query text, candidate rank, system score, candidate history, and optional related past responses.

## Workflow

1. Work from the repository root.
2. Resolve the run input:
   - Use the artifact directory or file named by the user.
   - Otherwise let the builder select the newest `artifacts/**/predictions.tsv`.
   - Prefer `predictions.tsv` because it preserves native system scores. Use a `.trec` file only as an explicit fallback; its exported score may be ordinal.
3. Run:

   ```bash
   python3 .agents/skills/inspect-query-results/scripts/build_query_explorer.py \
     --run-dir artifacts/<completed-run>
   ```

   Useful overrides:

   ```bash
   python3 .agents/skills/inspect-query-results/scripts/build_query_explorer.py \
     --input path/to/predictions.tsv \
     --queries path/to/queries.tsv \
     --qrels path/to/qrels.txt \
     --profiles path/to/candidate_profiles.json \
     --discovery path/to/discovery.json \
     --output path/to/query_explorer.html
   ```

4. If the large discovery file is unavailable or speed matters, add `--no-responses`. Candidate profile summaries still render when available.
5. The default is the top 10 candidates per query. Use `--top-candidates N` when the user requests a different depth.
6. When the completed run config points to available development qrels, include post-run labels and evaluation automatically. The query list defaults to weakest nDCG@10 first and supports zero-nDCG failures only. For each query, compare the run with ideal nDCG and all positively labeled LLMs, including their grades and where the system ranked them. Add `--no-labels` for blind prediction inspection or use `--qrels` explicitly for another labeled run.
7. Return the absolute HTML path plus the reported query count, displayed-result count, unique candidates represented, selected score field, and whether profiles/responses/labels were included.

## Input Contract

Auto-detect these prediction aliases:

- Query: `target_query_id`, `query_id`, `outer_query_id`, or `qid`
- Candidate: `model_id`, `candidate_id`, or `doc_id`
- Rank: `final_rank`, `primary_rank`, or `rank`
- Score: `raw_score`, `aggregation_score`, `sum_score`, `primary_score`, or `score`

Use the corresponding `--*-column` override for a future schema that uses different names. Require unique candidate IDs and ranks within every query.

Query TSV files may be headerless `query_id<TAB>query_text` files or headered tables. Candidate profiles are optional; the repository's default is `reports/trec_2026_llm_profiles.json`.

## Guardrails

- Label the numeric value `System score`; never call it probability or confidence unless the source explicitly supplies a calibrated probability.
- Do not compare raw scores across runs or approaches.
- Present attempted-query terms and past responses as historical behavior, not verified expertise or correctness.
- Use qrels only to annotate and evaluate the completed predictions; never let the explorer change their ordering. Clearly label development ground truth, keep the run's own training/evidence provenance distinct, and remain fully usable when labels are absent.
- Keep query and response text escaped inside the generated HTML.
- Do not modify the completed run artifacts other than writing the requested explorer file.
