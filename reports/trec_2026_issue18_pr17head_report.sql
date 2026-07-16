-- Reproducible report rows transcribed from the sealed issue-18 evaluation
-- (SHA-256 60a45dc4317bd2b886b9b2fb993e912bbd2edbe69acca75e5bbcb53fd293e4d2).

-- Headline metrics.
SELECT 0.4364094046724061 AS x1_ndcg,
       0.07655466057735831 AS ndcg_delta,
       0.46460983038229936 AS x1_mrr,
       0.06054652227061116 AS mrr_delta,
       83 AS x1_zeros,
       -33 AS zero_delta,
       -0.00122013124288437 AS track_a_gap,
       8 AS track_a_zero_gain;

-- System table and chart rows.
WITH system_metrics(system, role, ndcg_at_10, mrr, zero_queries) AS (
  VALUES
    ('Track A profile-heavy', 'Stored diagnostic', 0.43762953591529047, 0.46999113723621716, 91),
    ('X1 multires top-two', 'Best new diagnostic', 0.4364094046724061, 0.46460983038229936, 83),
    ('X1 dense-equal', 'Frozen diagnostic', 0.42712858500092327, 0.4547199024812202, 92),
    ('R1 multires max dense-equal', 'Frozen diagnostic', 0.42614083691116733, 0.45562557119945485, 96),
    ('R1 multires max', 'Frozen diagnostic', 0.42558696846923294, 0.4646335088276342, 89),
    ('C0 max k10 dense-equal', 'Frozen control variant', 0.4202139921941393, 0.45301739378333744, 99),
    ('A1 top-two k10 dense-equal', 'Frozen diagnostic', 0.4201503648708051, 0.4477858071340355, 94),
    ('A1 top-two k10', 'Frozen diagnostic', 0.4193597514876465, 0.45213811202689036, 93),
    ('C0 max k10', 'Issue-12 profile control', 0.41492445997856287, 0.441697813928065, 92),
    ('Query-kNN', 'Frozen baseline', 0.35985474409504775, 0.40406330811168817, 116),
    ('Contextual E5', 'Qrel-free control', 0.3347215379757303, 0.38656649577390506, 120)
)
SELECT * FROM system_metrics ORDER BY ndcg_at_10 DESC;

-- Post-hoc paired comparison rows computed from stored per-query metrics.
WITH ablations(comparison, mean_delta, lower, upper, zero_change) AS (
  VALUES
    ('A1 minus C0', 0.00444, -0.01726, 0.02608, 1),
    ('R1 minus C0', 0.01066, -0.00951, 0.03089, -3),
    ('X1 minus R1', 0.01082, -0.00767, 0.03071, -6),
    ('X1 minus C0', 0.02148, -0.00354, 0.04672, -9),
    ('X1 minus Track A', -0.00122, -0.02568, 0.02310, -8)
)
SELECT * FROM ablations;
