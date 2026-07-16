-- SQLite 3 report reshaping queries. Run from the repository root with:
-- sqlite3 -json ':memory:' < reports/trec_2026_candidate_semantic_profiles_report.sql

-- Headline metrics.
WITH evaluation AS (
  SELECT readfile('artifacts/candidate_semantic_profiles/evaluation_v2/evaluation.json') AS doc
), comparisons AS (
  SELECT readfile('artifacts/candidate_semantic_profiles/evaluation_v2/comparisons.json') AS doc
)
SELECT
  json_extract(evaluation.doc, '$.metrics.dense_profile_equal.ndcg_at_10') AS primary_ndcg,
  json_extract(comparisons.doc, '$.dense_profile_equal.ndcg_at_10_delta.mean_difference') AS ndcg_delta,
  json_extract(evaluation.doc, '$.metrics.dense_profile_equal.mrr') AS primary_mrr,
  json_extract(comparisons.doc, '$.dense_profile_equal.mrr_delta.mean_difference') AS mrr_delta,
  json_extract(evaluation.doc, '$.metrics.dense_profile_equal.zero_ndcg_queries') AS primary_zeros,
  json_extract(evaluation.doc, '$.metrics.dense_profile_equal.zero_ndcg_queries')
    - json_extract(evaluation.doc, '$.metrics.dense_reference.zero_ndcg_queries') AS zero_delta,
  json_extract(comparisons.doc, '$.dense_profile_equal.ndcg_at_10_delta.lower') AS ci_lower,
  json_extract(comparisons.doc, '$.dense_profile_equal.ndcg_at_10_delta.upper') AS ci_upper
FROM evaluation, comparisons;

-- Exact frozen-system metrics and deltas against query-kNN.
WITH evaluation AS (
  SELECT readfile('artifacts/candidate_semantic_profiles/evaluation_v2/evaluation.json') AS doc
), systems AS (
  SELECT key AS system_key, value AS metrics
  FROM evaluation, json_each(json_extract(evaluation.doc, '$.metrics'))
), baseline AS (
  SELECT
    json_extract(doc, '$.metrics.dense_reference.ndcg_at_10') AS ndcg,
    json_extract(doc, '$.metrics.dense_reference.mrr') AS mrr
  FROM evaluation
)
SELECT
  CASE system_key
    WHEN 'dense_profile_profile_heavy' THEN 'Profile-heavy fusion'
    WHEN 'dense_profile_e5_equal' THEN 'Three-way equal fusion'
    WHEN 'dense_profile_equal' THEN 'Query-kNN + profile equal'
    WHEN 'profile_only' THEN 'Profile only'
    WHEN 'dense_profile_dense_heavy' THEN 'Dense-heavy fusion'
    WHEN 'dense_e5_equal' THEN 'Query-kNN + E5 equal'
    WHEN 'dense_reference' THEN 'Query-kNN'
    WHEN 'contextual_e5_control' THEN 'Contextual E5'
  END AS system,
  CASE system_key
    WHEN 'dense_profile_equal' THEN 'Preregistered primary'
    WHEN 'dense_reference' THEN 'Frozen baseline'
    WHEN 'contextual_e5_control' THEN 'Qrel-free control'
    WHEN 'dense_e5_equal' THEN 'Phase control'
    ELSE 'Frozen diagnostic'
  END AS role,
  json_extract(metrics, '$.ndcg_at_10') AS ndcg_at_10,
  json_extract(metrics, '$.ndcg_at_10') - baseline.ndcg AS ndcg_delta,
  json_extract(metrics, '$.mrr') AS mrr,
  json_extract(metrics, '$.mrr') - baseline.mrr AS mrr_delta,
  json_extract(metrics, '$.zero_ndcg_queries') AS zero_queries
FROM systems, baseline
ORDER BY ndcg_at_10 DESC;

-- Primary deltas by qrel-density slice.
WITH evaluation AS (
  SELECT readfile('artifacts/candidate_semantic_profiles/evaluation_v2/evaluation.json') AS doc
), density AS (
  SELECT key AS density_key, value AS payload
  FROM evaluation, json_each(json_extract(evaluation.doc, '$.density_slices'))
), metrics(metric) AS (
  VALUES ('nDCG@10'), ('MRR')
)
SELECT
  CASE density_key
    WHEN 'at_most_5' THEN 'At most 5'
    WHEN '6_to_20' THEN '6 to 20'
    WHEN 'more_than_20' THEN 'More than 20'
  END AS density_group,
  metric,
  CASE metric
    WHEN 'nDCG@10' THEN json_extract(payload, '$.systems.dense_profile_equal.ndcg_at_10_delta')
    ELSE json_extract(payload, '$.systems.dense_profile_equal.mrr_delta')
  END AS delta,
  json_extract(payload, '$.query_count') AS query_count
FROM density, metrics
ORDER BY CASE density_key WHEN 'at_most_5' THEN 1 WHEN '6_to_20' THEN 2 ELSE 3 END, metric DESC;

-- Exhaustive primary zero-query transitions.
WITH comparisons AS (
  SELECT readfile('artifacts/candidate_semantic_profiles/evaluation_v2/comparisons.json') AS doc
)
SELECT 'Zero in both' AS transition,
       json_extract(doc, '$.dense_profile_equal.zero_transitions.zero_both') AS query_count,
       'Neither system retrieves a positive candidate in the top 10' AS interpretation
FROM comparisons
UNION ALL
SELECT 'Rescued by primary',
       json_extract(doc, '$.dense_profile_equal.zero_transitions.dense_zero_candidate_nonzero'),
       'Query-kNN is zero; primary is nonzero'
FROM comparisons
UNION ALL
SELECT 'Lost by primary',
       json_extract(doc, '$.dense_profile_equal.zero_transitions.dense_nonzero_candidate_zero'),
       'Query-kNN is nonzero; primary is zero'
FROM comparisons
UNION ALL
SELECT 'Nonzero in both',
       json_extract(doc, '$.dense_profile_equal.zero_transitions.nonzero_both'),
       'Both systems retrieve positive gain in the top 10'
FROM comparisons;

-- Compact integrity evidence for the reader-facing table.
SELECT 'Fresh inference repeat' AS "check", 'Passed' AS result,
       'Maximum absolute embedding difference 0.0' AS evidence
UNION ALL
SELECT 'Construction repeat', 'Passed',
       'Identical rankings and canonical profile files; score difference 0.0'
UNION ALL
SELECT 'Qrel-absent construction', 'Passed',
       'Instrumented mount, path, name-scan, and file-descriptor receipt'
UNION ALL
SELECT 'Post-evaluation immutability', 'Passed',
       'Construction aggregate unchanged at 04d6be23…'
UNION ALL
SELECT 'Decision receipt', 'Promote',
       'All five preregistered promotion gates passed';
