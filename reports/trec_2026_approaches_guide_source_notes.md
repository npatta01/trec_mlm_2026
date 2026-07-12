# TREC 2026 Approaches Guide — Source and QA Notes

Generated: 2026-07-11

## Reporting job

- Audience: technical beginner.
- Primary decision: which 2025 methods and wider literature should guide a 2026 experiment sequence.
- Delivery: one complete canonical report artifact rendered through the Data Analytics MCP app.
- Transform: Python audits local files, streams both large discovery JSON files, materializes reviewed rows in in-memory SQLite, and executes the exact source queries stored in the artifact.
- Comparison baseline: fold-local global prior, fixed TF-IDF query-kNN, and exact nested dense query-kNN.
- Current evidence boundary: local development TSV/qrels, reviewed baseline outputs, and both official discovery JSON files are present. Exact defaults, token/logprob alignment, and all 150 behavior/topic profile rows are materialized; correctness, semantic centroids, response retrieval, and judge labels are not yet measured.

## Required-structure mapping

- Title: visible artifact title and first heading.
- Technical summary: “The short answer”, followed by a compact contents block, an immediate six-block profile quickstart, and then the two-minute task story.
- Key findings with accessible evidence: twenty-one code-native teaching figures, headline metric cards, the exact 2026 baseline table, a complete 150-row profile table, and two complete 2025 run tables.
- Scope/data/metric definitions: competition-flow visual, exact query–candidate example, worked metric explanation, task, data reality, file inventory, and expanded pocket glossary.
- Methodology/model specification: nine-family beginner primer, a reproducible 150-row profile build plus worked row and exact lookup table, a ranking-feature module, five narrative-and-visual playbooks covering all six 2025 teams, and seven literature foundations with plain-language leads; the profile and weak-label foundations are layered into vocabulary, visual workflows, research, and practical checklists.
- Limitations/uncertainty/robustness: prior-year evaluation caveat, leakage protocol, shift section, and limitations.
- Recommended next steps: experiment ladder, five-run portfolio, and numbered next steps.
- Further questions: final open-question section.

## Local-data audit

- Queries: 322.
- Candidates: 150.
- Qrels: 48300 complete pairs.
- Grade counts: {0: 43922, 1: 730, 2: 3648}.
- Positive share: 0.09064182.
- Exactly one positive: 180 queries.
- At most five positives: 248 queries.
- Discovery responses: 1,163,850 across 7,759 queries and 150 candidates.
- Exact normalized defaults: 981,550 (84.34%).
- Non-exact attempts: 182,300; this is a coverage proxy, not correctness.
- Queries with zero non-exact attempts: 945 (12.18%).
- Token/logprob length mismatches: 12,260 (1.0534%).
- Saved candidate profiles: 150 rows, one per candidate, built from 1,163,850 response records using profile version `discovery-behavior-topic-v1`.
- Discovery files are present through Git-ignored shared symlinks; byte sizes and SHA-256 values match the saved audit, and the plain/metadata ordered response-tuple digests agree.

## Evidence-surface map

1. `baseline_comparison` plus `baseline_display_table`
   - Question: how does nested dense query retrieval compare with TF-IDF and the prior?
   - Form: compact grouped bar followed immediately by a six-row exact lookup table, split by system and metric.
   - Takeaway: dense improves both nDCG@10 and MRR.
   - Source: `reports/trec_2026_dense_query_knn_results.json`.

2. `runs_top10_table` and `runs_bottom9_table`
   - Question: which prior-year runs led official nDCG@10, and what did every team submit?
   - Form: two complete sortable/filterable tables with exact four-decimal nDCG@10 and MRR.
   - Takeaway: three different method families occupied the top three.
   - Source: official NIST run appendix and individual `trec_eval` files.

3. `candidate_profiles_table`
   - Question: what was actually saved for each of the 150 current candidates?
   - Form: one sortable, paginated row per candidate with coverage, style, metadata usability, and supported attempted-query word associations.
   - Takeaway: the profile is real stored data, but version 1 describes behavior rather than correctness.
   - Source: `reports/trec_2026_llm_profiles.json`, rebuilt by `scripts/build_llm_profiles.py`.

## Beginner teaching-figure map

These figures explain relationships that are harder to learn from dense prose. Nineteen are conceptual teaching diagrams; `beginner-visual-real-discovery-profile` and `beginner-visual-profile-build-pipeline` display audited quantitative values. Each is deterministic HTML/CSS with semantic text, a complete audio summary, a 16-pixel base font, dark-mode tokens, forced-color borders, and a single-column layout below 600 pixels. No raster image, remote asset, script, or second chart runtime is used.

1. `beginner-visual-competition-flow`
   - Question: what enters the system during learning and test time?
   - Takeaway: the task ranks candidate IDs; it does not generate the answer.
   - Adjacent prose: `walking_summary`.

2. `beginner-visual-query-candidate-pair`
   - Question: what does one qrel grade belong to?
   - Takeaway: one query is a 150-row group, and the grade belongs to each question–candidate pair.
   - Adjacent prose: `one_training_example`.

3. `beginner-visual-metrics`
   - Question: how do MRR and nDCG@10 read a ranking differently?
   - Takeaway: MRR focuses on the first useful hit; nDCG@10 evaluates the ordered top ten.
   - Adjacent prose: `metrics_gentle_intro`.

4. `beginner-visual-sparse-dense`
   - Question: why keep keyword and meaning retrieval?
   - Takeaway: the systems catch different clues, so fuse ranks without equating raw scores.
   - Adjacent prose: `sparse_dense_intro`.

5. `beginner-visual-profile-anatomy`
   - Question: what can an expert profile physically contain?
   - Takeaway: text, semantic, behavioral, and topic-local profiles summarize different aspects of one candidate’s stored history.
   - Adjacent prose: `expert_profiles_deep_dive`, `profile_shapes_2025`, and `profile_recommendation_2026`.

6. `beginner-visual-real-discovery-profile`
   - Question: what does a real candidate behavior profile reveal before embeddings or judges?
   - Takeaway: exact defaults dominate the discovery history, so cleaning, support, and metadata alignment come before semantic profiles.
   - Adjacent prose: `expert_profiles_deep_dive` and the reproducible discovery audit.

7. `beginner-visual-profile-build-pipeline`
   - Question: how do the large discovery files become one stored row per candidate?
   - Takeaway: stream, flag, group by candidate, aggregate, and save; the output is real behavior data but not a correctness score.
   - Adjacent prose and exact lookup: `profile_build_explained` and `candidate_profiles_table`.

8. `beginner-visual-single-vs-multi-profile`
   - Question: what does one broad profile lose compared with several supported specialty drawers?
   - Takeaway: compression trades detail for speed; preserve the response bank and shrink low-support local summaries toward a global fallback.
   - Adjacent prose: `profile_shapes_2025` and `profile_recommendation_2026`.

9. `beginner-visual-profile-to-score`
   - Question: how does stored candidate memory become a different score for each new query?
   - Takeaway: query-to-centroid, query-to-topic, and query-to-response interactions make a fixed profile query-dependent.
   - Adjacent prose: `profile_recommendation_2026` and `ranking_features_deep_dive`.

10. `beginner-visual-ranking-feature-row`
   - Question: what does one query–candidate feature row contain, and where is the label?
   - Takeaway: query, candidate, interaction, and evidence fields become the input row; the qrel remains a hidden training target.
   - Adjacent prose: `ranking_features_deep_dive`, `ranking_features_2025`, and `ranker_grouping_explained`.

11. `beginner-visual-srcb-pipeline`
   - Question: how can several imperfect teachers create practice labels for a small neural scorer?
   - Takeaway: preserve and validate each teacher signal before combining it, because the student can learn judge style instead of expertise.
   - Adjacent prose: `srcb`.

12. `beginner-visual-udinfo-pipeline`
    - Question: how do local neighbors, a global prior, and a domain cluster contribute different evidence?
    - Takeaway: steady specific but noisy local advice with broader candidate estimates; the answer-profile route is a separate method.
    - Adjacent prose: `udinfo`.

13. `beginner-visual-genaius-pipeline`
    - Question: how does response retrieval become a candidate ranking?
    - Takeaway: filter and retrieve individual work samples, then add their scores to the candidates that produced them.
    - Adjacent prose: `genaius`.

14. `beginner-visual-uvairlab-pipeline`
    - Question: how do 384 query fields and three candidate fields become a learned ranking?
    - Takeaway: confidence and candidate identity are supporting features, not proof of correctness or transferable identities.
    - Adjacent prose: `uvairlab`.

15. `beginner-visual-indelab-paulphys-pipelines`
    - Question: how did a dual encoder, an engineered ensemble, and a behavioral profile summarize candidates?
    - Takeaway: model complexity and profile size cannot repair weak supervision or a coarse representation.
    - Adjacent prose: `indelab_paulphys`.

16. `beginner-visual-router-task-shapes`
    - Question: how does this fixed 150-candidate ranking differ from common model-routing research?
    - Takeaway: borrow query and compatibility evidence, but rebuild methods that depend on live outputs, cost objectives, a smaller pool, or last year's candidates.
    - Adjacent prose: `routing_lit`, `routing_lit_research`, and `routing_lit_transfer`.

17. `beginner-visual-judge-label-menu`
    - Question: which labels can one structured judge record keep separate?
    - Takeaway: preserve mechanical flags, quality aspects, pairwise outcomes, uncertainty, and audit metadata before deriving one training signal.
    - Adjacent prose: `judge_labels_bridge`.

18. `beginner-visual-judge-label-funnel`
    - Question: how can the team obtain useful labels without judging every possible response pair?
    - Takeaway: screen exact defaults, pointwise-label a stratified pilot, pairwise-judge ambiguous cases, directly audit a small subset, then validate the router end to end.
    - Adjacent prose: `judge_label_scale`.

19. `beginner-visual-weak-label-jury`
    - Question: what is a weak label, why use a jury, and what makes the result auditable?
    - Takeaway: hide candidate identity, collect reversible-order votes, preserve raw evidence, and calibrate every teacher before using its practice grades.
    - Adjacent prose: `weak_label_lit`, `weak_label_lit_research`, and `weak_label_lit_audit`.

20. `beginner-visual-leakage-safe-validation`
    - Question: why is a random pair-row split unsafe?
    - Takeaway: lock away the whole query and all 150 grades, with tuning inside the outer holdout.
    - Adjacent prose: `leakage_preview` and the ten-step protocol.

21. `beginner-visual-approach-ladder`
    - Question: which experiments should be built first with the data currently local?
    - Takeaway: prove question-only controls, then add discovery evidence, then weak supervision and cross-fitted fusion.
    - Adjacent prose and exact lookup: `recommended_sequence` and `approach_ladder`.

## Evidence policy

Official TREC pages, overview, participant papers, and run pages are primary for 2025. Official aggregate scores override participant-side figures when they conflict. The SRCB paper does not fully map submitted runs to configurations; UDInfo method/run mapping is strongly implied rather than tabulated. UvA paper scores are unreconciled with the official value. indelab and paulphys details are labeled submitter-reported because no reviewed proceedings paper or public reproduction package was found for those runs.

The broader literature review prioritizes original papers and proceedings. Modern LLM-routing results are treated as conceptual evidence, not score-comparable baselines.

The focused profile/ranking-feature bridge reuses the already reviewed local companion `reports/trec_2025_mllm_interactive_report.html`. Only unique profile shapes and feature-row mechanics are combined into the canonical guide; the full old leaderboard, team recipes, caveats, and sources are not duplicated. Relative sibling links are unreliable in the enhanced portable reader and would weaken one-file portability, so the companion is linked from the repository README and final handoff instead of from inside the artifact.

## Quantitative source inventory

- `data/dev/llm_dev_data.tsv`
- `data/dev/llm_dev_qrels.txt`
- `data/dev/llm_discovery_data.json` (Git-ignored shared local file)
- `data/dev/llm_discovery_metadata.json` (Git-ignored shared local file)
- `scripts/audit_discovery_profiles.py`
- `reports/trec_2026_discovery_profile_audit.json`
- `scripts/build_llm_profiles.py`
- `reports/trec_2026_llm_profiles.json`
- `reports/trec_2026_dense_query_knn_results.json`
- `trec_mlm/dense_experiment.py`
- https://trec.nist.gov/pubs/trec34/papers/Overview_mllm.pdf
- https://trec.nist.gov/pubs/trec34/appendices/trec2025-mllm-main.html

## Omitted visuals

The label distribution uses metric cards and narrative rather than a three-slice chart because one 90.9% zero slice would obscure the two small grades. The discovery-profile audit uses semantic number cards and worked profile cards rather than a two-candidate bar chart because the reader must connect each quantity to a cleaning or modeling action; it is not a leaderboard. The complete candidate-profile dataset remains a table because exact candidate lookup and field definitions matter more than a ranked comparison that could imply expertise. The dense baseline retains one required compact chart, but an exact table follows immediately because final-context QA found that the shared enhanced chart SVG exposes an unnamed keyboard `application` region. The 19-run leaderboard chart was removed entirely; its lookup is split into 10- and 9-row tables so the shared 15-row semantic fallback cannot truncate it. The literature map, file inventory, exact approach ladder, and five-run portfolio remain tables because exact lookup is the user’s task. No trend chart is appropriate: the evidence is cross-sectional, not temporal.

AI-generated raster illustrations were considered and deliberately omitted. The selected image-generation guidance says to prefer deterministic HTML/CSS/SVG for simple educational diagrams. Code-native figures preserve exact labels, text equivalents, dark mode, phone reflow, and one-file offline delivery more reliably.

## Shared-reader accessibility constraints

The enhanced reader uses each descriptive block ID as the custom-frame title and auto-sizes the frame. The generated semantic/no-script fallback currently gives every custom frame the generic outer title `Custom report content` and a 240-pixel minimum height. Each figure therefore repeats its meaning in an internal heading, semantic lists/cards, a complete audio-summary caption, and adjacent report prose. This keeps the information available even when the fallback frame title or nested scrolling is inconvenient. The limitation belongs to the shared portable renderer rather than the canonical report artifact.

The reader exposes block IDs as DOM fragment targets, but rendered Markdown links always open a new browsing context and custom HTML frames cannot scroll the parent report. The contents block therefore uses exact visible heading names for screen-reader heading navigation or page Find, while the profile definition, build figure, and table are physically moved directly under the contents. This is the reliable artifact-only navigation path; no unsupported in-place jump behavior is claimed.

## Revision-preservation contract

The earlier revisions added the profile/judge teaching blocks, reproducible discovery audit, 150-row profile data product, and judge-evaluation sources without deleting prior content. This navigation follow-up rewrites only `reader_map`, moves the six existing quickstart blocks from `expert_profiles_deep_dive` through `candidate_profiles_table_block` directly after it, and makes a narrow wording update to `profiles_features_orientation`. Every block ID, card, chart, filter, table, dataset, source, quantitative value, team detail, caveat, and research citation must remain. Relative order must remain unchanged inside the moved quickstart and inside the untouched remainder; the baseline chart still immediately precedes its exact table, and the 2025 run tables remain 10 plus 9 rows.
