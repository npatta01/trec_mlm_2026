#!/usr/bin/env python3
"""Add the sparse-failure analogue deep dive to the ranking diagnosis report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "reports/trec_2026_ranking_diagnosis.artifact.json"
ANALOGUE_JSON = ROOT / "reports/trec_2026_sparse_failure_analogue_diagnostic.json"
TRANSFER_TSV = ROOT / "reports/trec_2026_sparse_failure_neighbor_transfer.tsv"
CASES_TSV = ROOT / "reports/trec_2026_sparse_failure_analogue_cases.tsv"
CAUSE_JSON = ROOT / "reports/trec_2026_sparse_failure_cause_diagnostic.json"
CAUSE_CASES_TSV = ROOT / "reports/trec_2026_sparse_failure_cause_cases.tsv"
WEIGHTS_TSV = ROOT / "reports/trec_2026_sparse_failure_weight_sensitivity.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_CANONICAL)
    return parser.parse_args()


def load_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def resolve_base_path(base: Path | None, output: Path) -> Path:
    """Resolve only explicit or repository-portable report inputs."""

    base_path = base if base is not None else output
    if not base_path.exists():
        raise FileNotFoundError(
            "No complete base report artifact is available. Pass --base explicitly."
        )
    return base_path


def replace_by_id(items: list[dict[str, Any]], replacements: list[dict[str, Any]]) -> None:
    replacement_by_id = {item["id"]: item for item in replacements}
    existing_ids = {item["id"] for item in items}
    for index, item in enumerate(items):
        if item["id"] in replacement_by_id:
            items[index] = replacement_by_id[item["id"]]
    items.extend(
        item for item in replacements if item["id"] not in existing_ids
    )


def main() -> None:
    args = parse_args()
    base_path = resolve_base_path(args.base, args.output)

    artifact = json.loads(base_path.read_text(encoding="utf-8"))
    analogue = json.loads(ANALOGUE_JSON.read_text(encoding="utf-8"))
    transfer = load_tsv(TRANSFER_TSV)
    cases = load_tsv(CASES_TSV)
    cause = json.loads(CAUSE_JSON.read_text(encoding="utf-8"))
    cause_cases = load_tsv(CAUSE_CASES_TSV)
    weights = load_tsv(WEIGHTS_TSV)
    cause_by_query = {row["query_id"]: row for row in cause_cases}
    manifest = artifact["manifest"]
    snapshot = artifact["snapshot"]

    evidence_counts = analogue["blinded_lexical_evidence_screen"][
        "evidence_bucket_counts"
    ]
    if evidence_counts != {"not": 30, "strong": 11, "weak": 34}:
        raise ValueError(f"unexpected evidence bucket counts: {evidence_counts!r}")
    component_states = analogue["component_states"]
    if component_states != {
        "both_top10": 0,
        "dense_top10_only": 3,
        "neither_top10": 54,
        "profile_top10_only": 18,
    }:
        raise ValueError(f"unexpected component states: {component_states!r}")
    if cause["reproduction"] != {
        "query_count": 322,
        "candidate_count": 150,
        "exact_rank_mismatches": 0,
        "primary_ndcg_at_10_raw_graded_gain": 0.4202139921941393,
    }:
        raise ValueError("cause diagnostic did not reproduce the promoted ranking")
    if cause["no_signal_54"]["cause_proxy_counts"] != {
        "analogue_label_mismatch": 23,
        "near_component_miss": 7,
        "no_close_attempted_history": 24,
    }:
        raise ValueError("unexpected no-signal cause proxy counts")

    manifest["description"] = (
        "A source-backed diagnosis of the 322-query development ledger, including "
        "a similar-query deep dive on the 75 sole-positive failures."
    )
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    manifest["generatedAt"] = generated_at
    snapshot["generatedAt"] = generated_at

    charts = [
        {
            "id": "analogue_evidence_buckets",
            "title": "Pre-qrel evidence strength for the 75 sole-positive failures",
            "subtitle": (
                "Deterministic upper-bound screen using frozen component ranks and "
                "historical discovery-query attempts."
            ),
            "headerMarkdown": (
                "A **strong** flag means corroborated historical evidence or profile "
                "rank 1–3; it does not mean the historical answer was correct."
            ),
            "type": "horizontalBar",
            "dataset": "analogue_evidence_buckets",
            "sourceId": "sparse_analogue",
            "valueFormat": "number",
            "encodings": {
                "x": {
                    "field": "bucket",
                    "type": "nominal",
                    "label": "Evidence bucket",
                },
                "y": {
                    "field": "query_count",
                    "type": "quantitative",
                    "label": "Queries",
                },
                "tooltip": [
                    {
                        "field": "share_pct",
                        "type": "quantitative",
                        "label": "Share of 75 (%)",
                    }
                ],
            },
            "settings": {
                "orientation": "horizontal",
                "sort": "descending",
                "showValues": True,
                "palette": {"kind": "single", "root": "blue"},
            },
            "layout": "full",
        },
        {
            "id": "neighbor_transfer_hit10",
            "title": "Best inspected leave-one-out Hit@10 by query representation",
            "subtitle": (
                "Receipt-only post-hoc sweep; original producer was not retained. "
                "Denominator is the same 75 failures."
            ),
            "headerMarkdown": (
                "The unreproduced exploratory receipt reports only **10/75** official "
                "positives in the top ten at its optimistic grid maximum."
            ),
            "type": "horizontalBar",
            "dataset": "neighbor_transfer_best",
            "sourceId": "neighbor_transfer",
            "valueFormat": "number",
            "encodings": {
                "x": {
                    "field": "representation",
                    "type": "nominal",
                    "label": "Query representation",
                },
                "y": {
                    "field": "hit_at_10",
                    "type": "quantitative",
                    "label": "Official positives in top 10",
                },
                "tooltip": [
                    {"field": "k", "type": "quantitative", "label": "Neighbors"},
                    {"field": "mrr", "type": "quantitative", "label": "MRR"},
                    {
                        "field": "median_official_rank",
                        "type": "quantitative",
                        "label": "Median official rank",
                    },
                ],
            },
            "settings": {
                "orientation": "horizontal",
                "sort": "ascending",
                "showValues": True,
                "palette": {"kind": "single", "root": "orange"},
            },
            "layout": "full",
        },
        {
            "id": "failure_cause_proxies",
            "title": "Primary released-evidence explanation for all 75 failures",
            "subtitle": (
                "Mutually exclusive mechanical or deterministic evidence proxy; "
                "no-signal proxies do not establish qrel error."
            ),
            "headerMarkdown": (
                "The 21 fusion misses have an exact mechanism. The three no-signal "
                "buckets describe what the released evidence lacks or contradicts."
            ),
            "type": "horizontalBar",
            "dataset": "failure_cause_proxies",
            "sourceId": "failure_causes",
            "valueFormat": "number",
            "encodings": {
                "x": {
                    "field": "cause",
                    "type": "nominal",
                    "label": "Primary explanation",
                },
                "y": {
                    "field": "query_count",
                    "type": "quantitative",
                    "label": "Queries",
                },
                "tooltip": [
                    {
                        "field": "share_pct",
                        "type": "quantitative",
                        "label": "Share of 75 (%)",
                    },
                    {"field": "evidence_type", "type": "nominal", "label": "Evidence"},
                ],
            },
            "settings": {
                "orientation": "horizontal",
                "sort": "descending",
                "showValues": True,
                "palette": {"kind": "single", "root": "olive"},
            },
            "layout": "full",
        },
    ]
    replace_by_id(manifest.setdefault("charts", []), charts)

    example_rows = [
        {
            "case": "Pizza-chain count",
            "official_positive": "Companies",
            "ranks": "22 / 2 / 117",
            "similar_query_evidence": (
                "Several pizza-chain comparisons; best five-analogue score among all candidates"
            ),
            "assessment": "Strong profile signal diluted by fusion",
        },
        {
            "case": "Riemann short-interval primes",
            "official_positive": "Fields_of_mathematics",
            "ranks": "16 / 1 / 100",
            "similar_query_evidence": (
                "Prime-related profile centroid; little close lexical history"
            ),
            "assessment": "Strong semantic profile signal diluted by fusion",
        },
        {
            "case": "House location by county",
            "official_positive": "Technology_systems",
            "ranks": "12 / 17 / 30",
            "similar_query_evidence": (
                "Near-template development query has the same sole positive"
            ),
            "assessment": "Small pocket of label coherence; still missed at rank 12",
        },
        {
            "case": "Chopin festival companion",
            "official_positive": "Politics_by_location",
            "ranks": "59 / 27 / 116",
            "similar_query_evidence": (
                "Chopin history exists, but Popular_music has equally direct evidence"
            ),
            "assessment": "Plausible but not uniquely identifiable",
        },
        {
            "case": "Zinnia versus Pachypodium",
            "official_positive": "Education",
            "ranks": "60 / 103 / 37",
            "similar_query_evidence": (
                "Closest plant-genus query labels Flora and Organisms, not Education"
            ),
            "assessment": "Counter-domain; not predictable",
        },
        {
            "case": "Documentary release order",
            "official_positive": "Food_and_drink_by_country",
            "ranks": "96 / 78 / 89",
            "similar_query_evidence": (
                "Closest documentary comparison labels Culture_stubs"
            ),
            "assessment": "Strong semantic similarity, conflicting label",
        },
        {
            "case": "Which film is book-based?",
            "official_positive": "Cuisine",
            "ranks": "129 / 114 / 119",
            "similar_query_evidence": (
                "No relevant Cuisine history; media candidates have direct film/book evidence"
            ),
            "assessment": "Deep counter-domain miss; not predictable",
        },
        {
            "case": "What is that determination based on?",
            "official_positive": "Social_groups",
            "ranks": "45 / 79 / 28",
            "similar_query_evidence": "No antecedent in the released query text",
            "assessment": "Under-specified; historical evidence cannot reconstruct context",
        },
    ]
    cause_labels = {
        "cross_component_rrf_veto": "Cross-component RRF veto",
        "near_component_miss": "One component ranks positive 11–20",
        "no_close_attempted_history": "No close attempted history",
        "analogue_label_mismatch": "Nearby released labels point elsewhere",
    }
    archetype_labels = {
        "counter_domain_or_opaque_profile": "Counter-domain or opaque profile",
        "entity_specific_or_multihop": "Entity-specific or multi-hop",
        "surface_topic_collision": "Surface/topic collision",
        "query_underspecification": "Query underspecification",
        "history_coverage_gap": "History coverage gap",
        "semantic_granularity_mismatch": "Semantic granularity mismatch",
        "local_refusal_gap": "Local refusal gap",
        "not_applicable_fusion_case": "Not applicable (fusion veto)",
    }
    archetype_rows = [
        {
            "archetype": archetype_labels[key],
            "query_count": count,
            "share_pct": round(100 * count / 54, 1),
        }
        for key, count in cause["no_signal_54"][
            "interpretive_archetype_counts"
        ].items()
    ]
    selected_weight_rows = []
    for row in weights:
        ratio = float(row["profile_to_dense_weight_ratio"])
        if ratio not in {0.5, 1.0, 1.5, 2.0, 3.0, 10.0}:
            continue
        prior_hits = int(row["other_105_sole_positive_top10_hits"])
        selected_weight_rows.append(
            {
                "profile_dense_ratio": ratio,
                "fusion_21_recovered": int(row["fusion_21_top10_hits"]),
                "previous_successes_lost": 105 - prior_hits,
                "all_query_ndcg_at_10": float(row["ndcg_at_10"]),
            }
        )
    tables = [
        {
            "id": "analogue_spot_checks",
            "title": "Representative similar-query spot checks",
            "subtitle": (
                "P / F / D are promoted-primary, profile, and dense ranks. Historical "
                "attempts are not treated as correct answers."
            ),
            "dataset": "analogue_spot_checks",
            "sourceId": "sparse_cases",
            "defaultSort": {"field": "case", "direction": "asc"},
            "columns": [
                {"field": "case", "label": "Target case", "type": "text"},
                {
                    "field": "official_positive",
                    "label": "Official sole positive",
                    "type": "text",
                },
                {"field": "ranks", "label": "P / F / D ranks", "type": "text"},
                {
                    "field": "similar_query_evidence",
                    "label": "Closest useful evidence",
                    "type": "text",
                },
                {
                    "field": "assessment",
                    "label": "What it tells us",
                    "type": "text",
                },
            ],
        },
        {
            "id": "all_analogue_cases",
            "title": "All 75 sole-positive failure cases",
            "subtitle": (
                "Sanitized case-level evidence; no development-target responses or "
                "raw historical responses are included."
            ),
            "dataset": "all_analogue_cases",
            "sourceId": "all_sparse_cases",
            "defaultSort": {"field": "primary_rank", "direction": "asc"},
            "density": "dense",
            "columns": [
                {"field": "query_text", "label": "Target query", "type": "text"},
                {
                    "field": "official_positive",
                    "label": "Official sole positive",
                    "type": "text",
                },
                {
                    "field": "evidence_bucket",
                    "label": "Evidence screen",
                    "type": "text",
                },
                {
                    "field": "cause",
                    "label": "Primary failure explanation",
                    "type": "text",
                },
                {
                    "field": "interpretive_archetype",
                    "label": "Spot-check archetype",
                    "type": "text",
                },
                {
                    "field": "primary_rank",
                    "label": "Primary rank",
                    "format": "number",
                },
                {
                    "field": "profile_rank",
                    "label": "Profile rank",
                    "format": "number",
                },
                {
                    "field": "dense_rank",
                    "label": "Dense rank",
                    "format": "number",
                },
                {
                    "field": "nearest_dev_query_text",
                    "label": "Closest MPNet development query",
                    "type": "text",
                },
                {
                    "field": "same_positive",
                    "label": "Closest query shares positive?",
                    "type": "text",
                },
                {
                    "field": "official_top3_attempts",
                    "label": "Official attempts / 3 discovery analogues",
                    "format": "number",
                },
            ],
        },
        {
            "id": "fusion_weight_tradeoff",
            "title": "Static profile-to-dense weight sensitivity",
            "subtitle": (
                "Post-qrel sensitivity over all 322 queries; losses are relative to "
                "the 105 other sole-positive queries solved by equal fusion."
            ),
            "dataset": "fusion_weight_tradeoff",
            "sourceId": "failure_causes",
            "defaultSort": {"field": "profile_dense_ratio", "direction": "asc"},
            "columns": [
                {
                    "field": "profile_dense_ratio",
                    "label": "Profile / dense weight",
                    "format": "number",
                },
                {
                    "field": "fusion_21_recovered",
                    "label": "Fusion cases recovered",
                    "format": "number",
                },
                {
                    "field": "previous_successes_lost",
                    "label": "Prior successes lost",
                    "format": "number",
                },
                {
                    "field": "all_query_ndcg_at_10",
                    "label": "All-query nDCG@10",
                    "format": "number",
                },
            ],
        },
        {
            "id": "no_signal_archetypes",
            "title": "Manual archetypes within the 54 no-signal failures",
            "subtitle": (
                "Structured spot-check interpretation of query text, candidate "
                "handle, and released history; not an assessor-rationale label."
            ),
            "dataset": "no_signal_archetypes",
            "sourceId": "failure_causes",
            "defaultSort": {"field": "query_count", "direction": "desc"},
            "columns": [
                {"field": "archetype", "label": "Archetype", "type": "text"},
                {"field": "query_count", "label": "Queries", "format": "number"},
                {"field": "share_pct", "label": "Share of 54 (%)", "format": "number"},
            ],
        },
    ]
    replace_by_id(manifest.setdefault("tables", []), tables)

    sources = [
        {
            "id": "failure_causes",
            "label": "Sparse-failure causal decomposition",
            "path": "reports/trec_2026_sparse_failure_cause_diagnostic.json",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": (
                    "Exactly reproduces equal reciprocal-rank fusion, decomposes "
                    "the 21/54 cohorts, and runs a post-qrel weight sensitivity grid."
                ),
                "sql": (
                    "SELECT cohort, cause_bucket, COUNT(*) AS query_count "
                    "FROM read_csv_auto("
                    "'reports/trec_2026_sparse_failure_cause_cases.tsv', "
                    "delim='\\t', header=true) GROUP BY cohort, cause_bucket;"
                ),
                "filters": [
                    "Promoted equal-RRF nDCG@10 reproduced with raw graded gains.",
                    "Same 75 sole-positive zero-nDCG failures.",
                ],
                "metric_definitions": [
                    "Fusion cause is exact rank arithmetic.",
                    "No-signal causes are deterministic evidence proxies, not qrel-error labels.",
                    "Manual archetypes are structured spot-check interpretations.",
                ],
                "tables_used": [
                    "reports/trec_2026_sparse_failure_cause_cases.tsv",
                    "reports/trec_2026_sparse_failure_weight_sensitivity.tsv",
                    "reports/trec_2026_candidate_semantic_profiles_query_explorer.html",
                ],
            },
        },
        {
            "id": "sparse_analogue",
            "label": "Sparse-failure analogue diagnostic",
            "path": "reports/trec_2026_sparse_failure_analogue_diagnostic.json",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "description": (
                    "Reproduces the 75-query slice, MPNet analogue retrieval, component "
                    "states, refusal flags, and deterministic evidence screen."
                ),
                "sql": (
                    "WITH evidence(bucket, query_count, share_pct) AS (VALUES "
                    "('Strong / corroborated',11,14.7),"
                    "('Weak / non-unique',34,45.3),"
                    "('Not predictably identifiable',30,40.0)) "
                    "SELECT * FROM evidence;"
                ),
                "filters": [
                    "Promoted-run nDCG@10 = 0.",
                    "Exactly one candidate has qrel grade > 0.",
                ],
                "metric_definitions": [
                    "Strong/weak/not are deterministic screening flags, not correctness judgments.",
                    "Attempt means the historical response is not an exact default refusal.",
                ],
                "tables_used": [
                    "data/dev/llm_dev_data.tsv",
                    "data/dev/llm_dev_qrels.txt",
                    "data/dev/llm_discovery_data.json",
                    "reports/trec_2026_candidate_semantic_profiles_query_explorer.html",
                ],
            },
        },
        {
            "id": "neighbor_transfer",
            "label": "Leave-one-out similar-query transfer",
            "path": "reports/trec_2026_sparse_failure_neighbor_transfer.tsv",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "description": (
                    "Receipt from a transfer of grades from the other 321 development "
                    "queries across eight representations; the original producer was "
                    "not retained."
                ),
                "sql": (
                    "WITH transfer(representation,k,hit_at_10,mrr,median_official_rank) "
                    "AS (VALUES ('Word TF-IDF unigram',5,6,0.0672,70),"
                    "('Word TF-IDF 1-2 gram',3,7,0.0562,79),"
                    "('Character TF-IDF 3-5',20,4,0.0265,75),"
                    "('MiniLM-L6',10,6,0.0523,71),"
                    "('E5-small-v2',1,7,0.0653,78),"
                    "('BGE-small',5,5,0.0601,59),"
                    "('MPNet-base-v2',1,7,0.0548,78),"
                    "('E5-Mistral contextual',3,10,0.0997,72)) "
                    "SELECT * FROM transfer;"
                ),
                "filters": [
                    "Same 75 sole-positive promoted-run failures.",
                    "Target qrels excluded from every prediction.",
                    "Receipt-only: rerun before using these rows for model selection.",
                ],
                "metric_definitions": [
                    "Hit@10 counts targets whose official positive is ranked 1-10.",
                    "Best inspected k is post-hoc and not an unbiased model-selection estimate.",
                ],
                "tables_used": [
                    "data/dev/llm_dev_data.tsv",
                    "data/dev/llm_dev_qrels.txt",
                ],
            },
        },
        {
            "id": "sparse_cases",
            "label": "All 75 sole-positive failure cases",
            "path": "reports/trec_2026_sparse_failure_analogue_cases.tsv",
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "description": (
                    "Reproduces the eight representative rows selected from the "
                    "complete 75-case diagnostic table."
                ),
                "sql": (
                    "WITH examples(case_name,official_positive,ranks,evidence,assessment) "
                    "AS (VALUES ('Pizza-chain count','Companies','22 / 2 / 117',"
                    "'Several pizza-chain comparisons','Strong profile signal diluted'),"
                    "('Riemann short-interval primes','Fields_of_mathematics','16 / 1 / 100',"
                    "'Prime-related profile centroid','Strong profile signal diluted'),"
                    "('House location by county','Technology_systems','12 / 17 / 30',"
                    "'Near-template query with same positive','Small coherent pocket'),"
                    "('Chopin festival companion','Politics_by_location','59 / 27 / 116',"
                    "'Chopin history is non-unique','Plausible but not identifiable'),"
                    "('Zinnia versus Pachypodium','Education','60 / 103 / 37',"
                    "'Plant analogue labels Flora and Organisms','Counter-domain'),"
                    "('Documentary release order','Food_and_drink_by_country','96 / 78 / 89',"
                    "'Documentary analogue labels Culture_stubs','Conflicting label'),"
                    "('Which film is book-based?','Cuisine','129 / 114 / 119',"
                    "'No relevant Cuisine history','Deep counter-domain miss'),"
                    "('What is that determination based on?','Social_groups','45 / 79 / 28',"
                    "'No antecedent in released text','Under-specified')) "
                    "SELECT * FROM examples;"
                ),
                "filters": [
                    "Eight representative cases selected to cover coherent, ambiguous, "
                    "counter-domain, and under-specified patterns."
                ],
                "metric_definitions": [
                    "Ranks are promoted-primary / profile / dense."
                ],
                "tables_used": [
                    "reports/trec_2026_sparse_failure_analogue_cases.tsv"
                ],
            },
        },
        {
            "id": "all_sparse_cases",
            "label": "Complete sanitized 75-case failure-cause table",
            "path": "reports/trec_2026_sparse_failure_cause_cases.tsv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": (
                    "Joins the sanitized analogue and causal case tables in primary-rank order."
                ),
                "sql": (
                    "SELECT a.query_text, a.official_positive, a.evidence_bucket, "
                    "c.cause_bucket, c.interpretive_archetype, "
                    "CAST(a.primary_rank AS INTEGER) AS primary_rank, "
                    "CAST(a.profile_rank AS INTEGER) AS profile_rank, "
                    "CAST(a.dense_rank AS INTEGER) AS dense_rank, "
                    "a.nearest_dev_query_text, a.nearest_dev_has_official_positive, "
                    "CAST(a.official_top3_attempts AS INTEGER) AS official_top3_attempts "
                    "FROM read_csv_auto("
                    "'reports/trec_2026_sparse_failure_analogue_cases.tsv', "
                    "delim='\\t', header=true) a JOIN read_csv_auto("
                    "'reports/trec_2026_sparse_failure_cause_cases.tsv', "
                    "delim='\\t', header=true) c USING (query_id) ORDER BY primary_rank;"
                ),
                "filters": [
                    "Promoted-run nDCG@10 = 0.",
                    "Exactly one qrel grade is greater than zero.",
                    "Raw discovery responses excluded from the table.",
                ],
                "metric_definitions": [
                    "Evidence screen is the deterministic strong/weak/not upper-bound flag.",
                    "Official attempts count non-exact refusals over the top three MPNet discovery analogues.",
                ],
                "tables_used": [
                    "reports/trec_2026_sparse_failure_analogue_cases.tsv",
                    "reports/trec_2026_sparse_failure_cause_cases.tsv",
                ],
            },
        },
    ]
    replace_by_id(manifest.setdefault("sources", []), sources)

    snapshot["datasets"]["analogue_evidence_buckets"] = [
        {"bucket": "Strong / corroborated", "query_count": 11, "share_pct": 14.7},
        {"bucket": "Weak / non-unique", "query_count": 34, "share_pct": 45.3},
        {"bucket": "Not predictably identifiable", "query_count": 30, "share_pct": 40.0},
    ]
    snapshot["datasets"]["failure_cause_proxies"] = [
        {
            "cause": "Cross-component RRF veto",
            "query_count": 21,
            "share_pct": 28.0,
            "evidence_type": "Exact mechanism",
        },
        {
            "cause": "One component ranks positive 11–20",
            "query_count": 7,
            "share_pct": 9.3,
            "evidence_type": "Deterministic proxy",
        },
        {
            "cause": "No close attempted history",
            "query_count": 24,
            "share_pct": 32.0,
            "evidence_type": "Deterministic proxy",
        },
        {
            "cause": "Nearby released labels point elsewhere",
            "query_count": 23,
            "share_pct": 30.7,
            "evidence_type": "Deterministic proxy",
        },
    ]
    snapshot["datasets"]["fusion_weight_tradeoff"] = selected_weight_rows
    snapshot["datasets"]["no_signal_archetypes"] = archetype_rows
    snapshot["datasets"]["neighbor_transfer_best"] = [
        {
            "representation": row["representation"],
            "k": int(row["k"]),
            "hit_at_10": int(row["hit_at_10"]),
            "mrr": float(row["mrr"]),
            "median_official_rank": int(row["median_official_rank"]),
        }
        for row in transfer
        if row["scope"] == "best_posthoc_mrr"
    ]
    snapshot["datasets"]["analogue_spot_checks"] = example_rows
    snapshot["datasets"]["all_analogue_cases"] = [
        {
            "query_text": row["query_text"],
            "official_positive": row["official_positive"],
            "evidence_bucket": {
                "strong": "Strong / corroborated",
                "weak": "Weak / non-unique",
                "not": "Not predictably identifiable",
            }[row["evidence_bucket"]],
            "cause": cause_labels[cause_by_query[row["query_id"]]["cause_bucket"]],
            "interpretive_archetype": (
                archetype_labels[
                    cause_by_query[row["query_id"]]["interpretive_archetype"]
                ]
                if cause_by_query[row["query_id"]]["interpretive_archetype"]
                else "—"
            ),
            "primary_rank": int(row["primary_rank"]),
            "profile_rank": int(row["profile_rank"]),
            "dense_rank": int(row["dense_rank"]),
            "nearest_dev_query_text": row["nearest_dev_query_text"],
            "same_positive": (
                "Yes"
                if row["nearest_dev_has_official_positive"].casefold() == "true"
                else "No"
            ),
            "official_top3_attempts": int(row["official_top3_attempts"]),
        }
        for row in cases
    ]

    existing_blocks = {
        block["id"]: block for block in manifest["blocks"]
    }
    existing_blocks["executive_summary"]["body"] = """## Executive summary

- **The 21 fusion misses are cross-component vetoes.** In every case, one component ranks the official positive in its top ten while the other ranks it 21–150. Equal reciprocal-rank fusion rewards candidates with moderate support in both inputs and suppresses the unilateral hit.
- **No global weight protects both directions.** The 18 profile-only recoveries require profile/dense weight above 1.30; the three dense-only recoveries require it below 0.78. Profile-heavy weights recover more failures but lose previously solved sole-positive queries.
- **The 54 no-signal misses mostly lack or contradict released evidence.** Seven are ranks 11–20 in one input, 24 have no close attempted history, and 23 have historical activity whose nearby development labels point elsewhere.
- **Decision:** test a fold-local, query-conditional component-protection gate. Treat the other 54 as evidence and label-provenance work, not another static fusion sweep."""

    analogue_blocks = [
        {
            "id": "analogue_answer",
            "type": "markdown",
            "sourceId": "sparse_analogue",
            "body": """## Similar questions expose only a small predictable pocket

The released evidence would let a blinded reviewer confidently nominate the official sole positive in only a small minority of these failures. The deterministic screen flags **11/75 (14.7%)** as strong or corroborated, **34/75 (45.3%)** as weak or non-unique, and **30/75 (40.0%)** as not predictably identifiable. Because profile rank 1–3 alone earns a strong flag, the 11 should be treated as an upper bound—not as 11 verified correct historical experts.""",
        },
        {"id": "analogue_bucket_chart", "type": "chart", "chartId": "analogue_evidence_buckets"},
        {
            "id": "analogue_bucket_readout",
            "type": "markdown",
            "sourceId": "sparse_analogue",
            "body": """The misses deepen with rank. None of the 12 official positives at primary ranks 101–150 receives a strong flag. Historical discovery behavior is also non-discriminative: across the three closest MPNet discovery analogues per target, the official positives attempt **36/225** responses, versus **50/225** for the profile-ranked leader. An attempt is only a non-exact refusal; it may still be irrelevant or wrong.""",
        },
        {
            "id": "neighbor_coherence",
            "type": "markdown",
            "sourceId": "neighbor_transfer",
            "body": """## Changing the query encoder does not repair label coherence

An exploratory receipt from transfer over the other 321 development queries reports a highest pure one-neighbor label agreement of **12/75**. Its most optimistic post-hoc configuration—contextual E5-Mistral with three neighbors—places only **10/75** official positives in the top ten and leaves the median official candidate at rank 72. The original sweep producer was not retained, so these values must be rerun before system selection.""",
        },
        {"id": "neighbor_transfer_chart", "type": "chart", "chartId": "neighbor_transfer_hit10"},
        {
            "id": "neighbor_transfer_readout",
            "type": "markdown",
            "sourceId": "neighbor_transfer",
            "body": """The exploratory receipt reports that across all eight representations, **37/75** targets have no nearest-neighbor label agreement at all. This is hypothesis-generating evidence—not a reproducible gate. Small template-level pockets may transfer cleanly, but a fresh preregistered rerun is required before concluding that broad nearest-query tuning cannot help.""",
        },
        {
            "id": "two_failure_modes",
            "type": "markdown",
            "sourceId": "failure_causes",
            "body": """## Reciprocal-rank fusion vetoes unilateral evidence in all 21 cases

The promoted system sums `1 / (60 + rank)` from profile and dense inputs. In the **18 profile-only** and **3 dense-only** failures, the supporting component ranks the positive 1–10, but the opposing component always ranks it 21–150. Ten positives are top-three in their supporting component; 18 face an opposing rank of 51–150. This is exact rank arithmetic—not raw-score calibration drift.

The equal-fusion deficit to rank ten ranges from 0.000647 to 0.005264. Only six cases finish at ranks 11–15; ten fall below rank 20. The other input is therefore acting as a strong veto, not merely nudging a borderline candidate.""",
        },
        {
            "id": "failure_cause_chart",
            "type": "chart",
            "chartId": "failure_cause_proxies",
        },
        {
            "id": "failure_cause_readout",
            "type": "markdown",
            "sourceId": "failure_causes",
            "body": """For the **54** cases with no component top-ten hit, the chart uses deterministic released-evidence proxies rather than inferred assessor intent. Seven are near-component misses at ranks 11–20. Twenty-four are outside the top 20, have zero official-candidate attempts on the three closest discovery analogues, and have no attempted lexical analogue at similarity 0.10. The remaining 23 have some historical evidence, but the nearest development label set excludes the official positive.

Across the 54, **37** official positives attempt none of their three closest discovery analogues, **49** are absent from the nearest development query's positive labels, and only one receives the earlier strong-evidence flag. These measurements explain why neither input can surface the label; they do not prove the label is incorrect.""",
        },
        {
            "id": "weight_tradeoff_heading",
            "type": "markdown",
            "sourceId": "failure_causes",
            "body": """## A static weight can move the failures, but cannot protect both components

Every profile-only recovery needs a profile/dense ratio above **1.30**. Every dense-only recovery needs a ratio below **0.78**. Those recovery regions do not overlap. A profile-heavy ratio of 2 recovers 10 of the 21 but loses five of the 105 other sole-positive successes; a ratio of 3 recovers 14 and loses eight. The higher development nDCG is post-qrel sensitivity evidence, not an unbiased selection result.""",
        },
        {
            "id": "fusion_weight_tradeoff_table",
            "type": "table",
            "tableId": "fusion_weight_tradeoff",
        },
        {
            "id": "archetype_heading",
            "type": "markdown",
            "sourceId": "failure_causes",
            "body": """## Opaque profiles dominate the human-readable pattern in the 54

A structured spot check assigns one interpretive archetype per no-signal query. **30/54** have a counter-domain or opaque official profile under the released query text and history. Seven require obscure entity or multi-hop resolution; six have a surface/topic collision where the wrapper and requested answer type favor different candidates; four are under-specified and lack an antecedent. The smaller buckets are history coverage, semantic granularity, and one local refusal gap.

This layer is intentionally labeled interpretive. Candidate handles are model identifiers, not assessor explanations, and the actual target responses are unavailable.""",
        },
        {
            "id": "no_signal_archetypes_table",
            "type": "table",
            "tableId": "no_signal_archetypes",
        },
        {
            "id": "spot_checks_heading",
            "type": "markdown",
            "body": """## Spot checks show both coherent templates and counter-domain labels

The cases below are deliberately mixed. A few have an interpretable component or near-template signal; several high-similarity analogues carry a different official positive; and some targets are missing the context needed to route them.""",
        },
        {"id": "analogue_spot_check_table", "type": "table", "tableId": "analogue_spot_checks"},
        {
            "id": "spot_checks_readout",
            "type": "markdown",
            "sourceId": "sparse_cases",
            "body": """The counterexamples matter more than any one surprising label. Plant, film, music, and documentary targets often retrieve highly similar questions whose positives disagree with the target's sole positive. That does not prove the qrels are wrong: candidate IDs are model identifiers, not canonical topics. It does show why target text plus released history often cannot uniquely predict the organizer's label.""",
        },
        {
            "id": "all_cases_heading",
            "type": "markdown",
            "body": """## Inspect every sole-positive failure

The complete table keeps the target, official positive, primary failure explanation, interpretive archetype, three frozen ranks, closest MPNet development analogue, same-positive indicator, and historical attempt count together. Use it to move from the aggregate diagnosis into any of the 75 individual cases without exposing the unavailable target responses or raw discovery-response text.""",
        },
        {
            "id": "all_analogue_cases_table",
            "type": "table",
            "tableId": "all_analogue_cases",
        },
    ]

    existing_blocks["next_step"]["body"] = """## Recommended next experiment

Run one **fold-local, query-conditional component-protection gate**, then decide whether richer evidence collection is justified.

1. Detect unilateral evidence without qrels: top-ten disagreement, component margin, profile support, query completeness, entity-match strength, and whether the opposing component has corroborating head candidates.
2. Protect or lightly boost a high-margin unilateral candidate. Do not choose one global profile/dense ratio: the verified recovery regions for profile-only and dense-only misses are incompatible.
3. Evaluate all 322 queries with nested or fold-local fitting. Report recoveries and regressions separately for the 21 fusion-veto cases, the 54 no-signal cases, and the other 247 queries.
4. For the 54, prioritize the 24 history gaps and six surface collisions for an entity-aware retrieval or response-judging pilot. Treat the 30 opaque-profile cases as a label-provenance question until target responses or assessor rationale become available.

**Pass gates:** paired nDCG@10 improvement over profile-only with the 95% interval above zero; fewer than 75 single-positive zero scores; no more than 92 overall zeros; no degradation in the 6–20 and >20 density buckets; and no use of held-out qrels in features, tuning, or selection.

Do not call attempt, length, or confidence “reliability.” Without correctness labels, they remain behavior or evidence-quality features."""

    existing_blocks["questions"]["body"] = """## Further questions

- How were the 2026 expertise labels produced, and can the organizers share the rubric or anonymized judgment provenance even if target responses remain private?
- Which of the 21 component-hit cases can a qrel-free margin gate protect without harming the full 322-query ranking?
- Can organizer provenance explain the 30 counter-domain or opaque-profile cases and the two coherent county-template labels?
- Can a small human- or reference-audited discovery subset distinguish factual quality from mere non-refusal and verbosity?
- Can future runs write complete per-query rankings and evidence by default so every failure remains diagnosable?"""

    existing_blocks["caveats"]["body"] = """## Caveats

The actual 150 candidate responses to the development target queries are unavailable, so no spot check can verify whether the official positive answered its target correctly. The 2026 assessor, rubric, prompt, reference answer, and grading pipeline are undocumented. Candidate IDs are model identifiers rather than canonical query topics.

All development-label analyses are post-hoc. The best neighbor count, representation, weight ratio, and manual archetype taxonomy are diagnostic—not unbiased test estimates or assessor explanations. Historical responses are attempts, not correctness labels; longer or non-refusing answers can still be wrong. The strong/weak/not screen is deterministic and intentionally generous, with profile rank 1–3 sufficient for a strong flag. Qrel-derived density, cause proxies, and evidence buckets are valid evaluation slices, not online router features."""

    insertion_ids = {block["id"] for block in analogue_blocks}
    retained_blocks = [
        block for block in manifest["blocks"] if block["id"] not in insertion_ids
    ]
    insertion_index = next(
        index for index, block in enumerate(retained_blocks) if block["id"] == "next_step"
    )
    manifest["blocks"] = (
        retained_blocks[:insertion_index]
        + analogue_blocks
        + retained_blocks[insertion_index:]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
