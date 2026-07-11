"""Build the canonical general-audience report for response-level retrieval."""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/response_retrieval_baseline"
DEFAULT_SUMMARY = ARTIFACT_DIR / "summary.json"
DEFAULT_VALIDATION = ARTIFACT_DIR / "validation.json"
DEFAULT_CONFIG = ARTIFACT_DIR / "config.json"
DEFAULT_DIAGNOSTICS = ARTIFACT_DIR / "diagnostics.json"
DEFAULT_SELECTIONS = ARTIFACT_DIR / "selections.tsv"
DEFAULT_INNER_SCORES = ARTIFACT_DIR / "inner_scores.tsv"
DEFAULT_RESULTS = ROOT / "reports/trec_2026_response_retrieval_results.json"
DEFAULT_ARTIFACT = ROOT / "reports/trec_2026_response_retrieval_baseline.artifact.json"

HEADLINE_SQL = """\
SELECT response_ndcg, response_mrr, ndcg_delta_vs_tfidf, mrr_delta_vs_tfidf,
       deployment_fold_share, deployment_fold_count
FROM headline_metrics
"""

SYSTEM_METRICS_SQL = """\
SELECT system_order, metric_order, system, metric, value, query_count,
       evaluation_protocol, run_classification
FROM system_metrics
ORDER BY system_order, metric_order
"""

QUERY_OUTCOMES_SQL = """\
SELECT comparison_order, metric_order, outcome_order, comparison_metric,
       baseline, metric, outcome, outcome_count, outcome_share, query_count,
       mean_difference, ci_lower, ci_upper
FROM query_outcomes
ORDER BY comparison_order, metric_order, outcome_order
"""

FILTER_FLOW_SQL = """\
SELECT stage_order, stage, documents, retained_share, removed_at_step, rule
FROM filter_flow
ORDER BY stage_order
"""

INVENTORY_SHARES_SQL = """\
SELECT quartile_order, series_order, inventory_group, series, share,
       model_count, inventory_min, inventory_max, top10_appearances
FROM inventory_shares
ORDER BY quartile_order, series_order
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAGNOSTICS)
    parser.add_argument("--selections", type=Path, default=DEFAULT_SELECTIONS)
    parser.add_argument("--inner-scores", type=Path, default=DEFAULT_INNER_SCORES)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--artifact-output", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = _read_json(args.summary)
    validation = _read_json(args.validation)
    config = _read_json(args.config)
    diagnostics = _read_json(args.diagnostics)
    _validate_inputs(summary, validation, config)

    snapshot_as_of = _file_snapshot_time(args.summary)
    selection_stability = _selection_stability(
        args.selections,
        args.inner_scores,
        summary["deployment_selection"]["configuration"],
    )
    inventory_prior_spearman = _inventory_prior_spearman(
        summary,
        ROOT / "data/dev/llm_dev_qrels.txt",
    )
    results = _reviewed_results(
        summary=summary,
        validation=validation,
        config=config,
        diagnostics=diagnostics,
        summary_path=args.summary,
        validation_path=args.validation,
        config_path=args.config,
        diagnostics_path=args.diagnostics,
        selections_path=args.selections,
        inner_scores_path=args.inner_scores,
        snapshot_as_of=snapshot_as_of,
        selection_stability=selection_stability,
        inventory_prior_spearman=inventory_prior_spearman,
    )
    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    args.results_output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact = _artifact(summary, results, args.results_output)
    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_output.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(args.results_output)
    print(args.artifact_output)


def _validate_inputs(
    summary: dict[str, Any],
    validation: dict[str, Any],
    config: dict[str, Any],
) -> None:
    required_systems = {
        "response_nested",
        "dense_nested_reference",
        "tfidf_fixed_k10",
        "global_prior",
    }
    if set(summary.get("metrics", {})) != required_systems:
        raise ValueError("summary does not contain the four required systems")
    if summary.get("protocol", {}).get("outer_split") != ("strict_leave_one_query_out"):
        raise ValueError("summary is not the exact outer-LOO experiment")
    if summary.get("data", {}).get("query_count") != 322:
        raise ValueError("expected the official 322-query development set")
    if len(summary.get("candidate_grid", [])) != 24:
        raise ValueError("expected the preregistered 24-candidate grid")
    if summary["protocol"].get("logprob_features") is not False:
        raise ValueError("the baseline must not use logprob features")
    if summary["protocol"].get("run_classification") != "internal_automatic":
        raise ValueError("unexpected run classification")
    receipt_protocol = validation.get("protocol", {})
    if receipt_protocol.get("outer_prediction_count") != 322:
        raise ValueError("validation receipt does not cover all outer predictions")
    if receipt_protocol.get("score_tensor_shape") != [24, 322, 150]:
        raise ValueError("validation receipt has the wrong score tensor shape")
    if not all(run.get("valid") for run in validation.get("runs", {}).values()):
        raise ValueError("one or more written control runs failed validation")
    if len(config.get("candidates", [])) != 24:
        raise ValueError("configuration does not contain 24 candidates")


def _reviewed_results(
    *,
    summary: dict[str, Any],
    validation: dict[str, Any],
    config: dict[str, Any],
    diagnostics: dict[str, Any],
    summary_path: Path,
    validation_path: Path,
    config_path: Path,
    diagnostics_path: Path,
    selections_path: Path,
    inner_scores_path: Path,
    snapshot_as_of: str,
    selection_stability: dict[str, Any],
    inventory_prior_spearman: float,
) -> dict[str, Any]:
    deployment = summary["deployment_selection"]["configuration"]
    deployment_bias = diagnostics["volume_bias_by_candidate"][deployment]
    largest_quartile = deployment_bias["top10_representation_by_inventory_quartile"][-1]
    equal_share = (
        largest_quartile["model_count"] / summary["data"]["candidate_model_count"]
    )
    reproducibility = summary["retrieval_reproducibility"]

    return {
        "snapshot_as_of": snapshot_as_of,
        "protocol": summary["protocol"],
        "data": summary["data"],
        "candidate_grid": summary["candidate_grid"],
        "metrics": summary["metrics"],
        "comparisons": summary["comparisons"],
        "selection": summary["selection"],
        "selection_stability": selection_stability,
        "deployment_selection": summary["deployment_selection"],
        "fixed_candidate_development_metrics": summary[
            "fixed_candidate_development_metrics"
        ],
        "retrieval_specification": {
            "aggregation": "SumScore: math.fsum of positive retrieved-document scores by model",
            "backend": "bm25s 0.3.9, NumPy backend, auto-compile disabled",
            "bm25": {
                "method": "lucene",
                "idf_method": "lucene",
                "k1": 0.9,
                "b": 0.4,
                "dtype": "float64",
            },
            "rm3": {
                "feedback_documents": 10,
                "feedback_terms": 10,
                "original_query_weight": 0.5,
            },
            "analyzer": reproducibility["analyzer_diagnostics"],
            "filter_audits": reproducibility["filter_audits"],
            "index_diagnostics": reproducibility["index_diagnostics"],
            "hashes": {
                "configuration_sha256": reproducibility["configuration_sha256"],
                "corpus_sha256": reproducibility["corpus_sha256"],
                "source_sha256": reproducibility["source_sha256"],
                "model_score_tensor_sha256": reproducibility[
                    "model_score_tensor_sha256"
                ],
                "complete_retrieval_tensor_sha256": reproducibility[
                    "score_tensor_sha256"
                ],
            },
        },
        "diagnostics": {
            "overlap_candidate_query_pairs": diagnostics[
                "overlap_candidate_query_pairs"
            ],
            "overlap_document_count_max_per_query_total": diagnostics[
                "overlap_document_count_max_per_query_total"
            ],
            "selected_zero_score_model_count": diagnostics[
                "selected_zero_score_model_count"
            ],
            "deployment_sumscore_volume_dependence": {
                "configuration": deployment,
                "spearman": deployment_bias["spearman"],
                "largest_inventory_quartile_top10_share": largest_quartile[
                    "top10_appearance_share"
                ],
                "largest_inventory_quartile_equal_model_share": equal_share,
                "top10_representation_by_inventory_quartile": deployment_bias[
                    "top10_representation_by_inventory_quartile"
                ],
                "inventory_vs_full_development_global_prior_spearman": (
                    inventory_prior_spearman
                ),
                "interpretation": (
                    "Strong inventory-volume dependence is a potential SumScore bias, "
                    "but inventory also correlates with the qrel-derived global prior, "
                    "so the diagnostic does not prove a pure volume artifact."
                ),
            },
        },
        "runtime": summary["runtime"],
        "validation": {
            "inputs": validation["inputs"],
            "protocol": validation["protocol"],
            "runs": validation["runs"],
        },
        "configuration": {
            "code_revision": config["code_revision"],
            "code_sha256": config["code_sha256"],
            "input_sha256": config["input_sha256"],
            "seed": config["seed"],
            "grid_selection_provenance": config["grid_selection_provenance"],
        },
        "evidence_inputs": [
            _evidence_entry(summary_path, tracked=False),
            _evidence_entry(validation_path, tracked=False),
            _evidence_entry(config_path, tracked=False),
            _evidence_entry(diagnostics_path, tracked=False),
            _evidence_entry(selections_path, tracked=False),
            _evidence_entry(inner_scores_path, tracked=False),
            {
                "path": "trec_mlm/response_experiment.py",
                "tracked": True,
                "sha256": config["code_sha256"]["response_experiment.py"],
            },
            {
                "path": "trec_mlm/response_retrieval.py",
                "tracked": True,
                "sha256": config["code_sha256"]["response_retrieval.py"],
            },
            {
                "path": "data/dev/llm_dev_qrels.txt",
                "tracked": True,
                "sha256": config["input_sha256"]["data/dev/llm_dev_qrels.txt"],
            },
        ],
        "report_notes": {
            "audience": "product stakeholders and first-time topic readers",
            "delivery_mode": "portable_html",
            "required_structure": [
                "title",
                "executive summary",
                "key findings with visual evidence",
                "recommended next steps",
                "further questions",
                "caveats and assumptions",
            ],
            "visuals": [
                {
                    "section": "four-system comparison",
                    "question": "Which approach ranks candidate LLMs best?",
                    "family": "comparison and ranking",
                    "type": "full-width grouped bar",
                    "rows": 8,
                    "takeaway": (
                        "Response search improves nDCG@10 over TF-IDF, while dense "
                        "semantic query search remains strongest."
                    ),
                },
                {
                    "section": "response filtering",
                    "question": (
                        "How much searchable answer text remains after each frozen "
                        "cleaning rule?"
                    ),
                    "family": "ordered stage comparison",
                    "type": "full-width horizontal bar",
                    "rows": 3,
                    "takeaway": (
                        "Removing exact refusal answers reduces 1.16 million stored "
                        "responses to 182,800 searchable documents."
                    ),
                },
                {
                    "section": "query-level outcomes",
                    "question": (
                        "How often does response search beat, tie, or trail each baseline?"
                    ),
                    "family": "composition",
                    "type": "full-width stacked bar",
                    "rows": 18,
                    "takeaway": (
                        "The two official metrics summarize different query-level "
                        "win, tie, and loss patterns."
                    ),
                },
                {
                    "section": "answer-inventory diagnostic",
                    "question": (
                        "Are top-10 placements concentrated among models with more "
                        "stored historical answers?"
                    ),
                    "family": "comparison and composition",
                    "type": "full-width grouped bar",
                    "rows": 8,
                    "takeaway": (
                        "The highest-inventory quarter receives almost half of all "
                        "top-10 placements despite containing about a quarter of models."
                    ),
                },
            ],
            "repeated_bar_rationale": (
                "All four quantitative questions compare or compose a small set of "
                "discrete categories; grouped and stacked bars are the most direct "
                "mobile-readable forms."
            ),
            "non_chart_visuals": [
                "plain-language response-retrieval flow",
                "nested-validation holdout flow",
            ],
            "table_omission": (
                "Exact chart values remain available in semantic chart tables and "
                "source affordances; no broad duplicate result table is shown."
            ),
        },
    }


def _artifact(
    summary: dict[str, Any],
    results: dict[str, Any],
    results_path: Path,
) -> dict[str, Any]:
    metrics = summary["metrics"]
    response = metrics["response_nested"]
    dense = metrics["dense_nested_reference"]
    tfidf = metrics["tfidf_fixed_k10"]
    prior = metrics["global_prior"]
    vs_tfidf = summary["comparisons"]["tfidf_fixed_k10"]
    query_count = summary["data"]["query_count"]
    model_count = summary["data"]["candidate_model_count"]
    deployment = summary["deployment_selection"]["configuration"]
    fixed_deployment = summary["fixed_candidate_development_metrics"][deployment]
    selection = results["selection_stability"]
    bias = results["diagnostics"]["deployment_sumscore_volume_dependence"]
    filters = results["retrieval_specification"]["filter_audits"]
    basic_filter = next(row for row in filters if row["filter_name"] == "basic")
    aggressive_filter = next(
        row for row in filters if row["filter_name"] == "aggressive"
    )
    as_of = results["snapshot_as_of"]
    as_of_label = _human_snapshot_label(as_of)

    (
        headline_rows,
        system_rows,
        outcome_rows,
        filter_rows,
        inventory_rows,
    ) = _query_report_rows(
        response=response,
        dense=dense,
        tfidf=tfidf,
        prior=prior,
        query_count=query_count,
        deployment_fold_share=selection["deployment_fold_share"],
        deployment_fold_count=selection["deployment_fold_count"],
        comparisons=summary["comparisons"],
        filter_input_documents=basic_filter["input_documents"],
        basic_retained_documents=basic_filter["retained_documents"],
        aggressive_retained_documents=aggressive_filter["retained_documents"],
        inventory_quartiles=bias["top10_representation_by_inventory_quartile"],
        model_count=model_count,
    )
    results_relative = _repo_relative(results_path)
    source_list = _source_list(results_relative, as_of)
    title = "Baseline 3: Ranking LLMs from Past Answers"
    ndcg_ci = vs_tfidf["ndcg_at_10_delta"]
    mrr_ci = vs_tfidf["mrr_delta"]
    idk_share = basic_filter["removed_exact_idk"] / basic_filter["input_documents"]
    if basic_filter["input_documents"] % model_count:
        raise ValueError("discovery documents are not complete across candidate models")
    discovery_query_count = basic_filter["input_documents"] // model_count
    deployment_share = selection["deployment_fold_share"]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "A plain-language held-out evaluation of whether historical model answers "
                "help choose the best LLM for a new query."
            ),
            "generatedAt": as_of,
            "cards": [
                {
                    "id": "response_ndcg",
                    "description": (
                        "Average quality of the top 10 ranked models, from 0 to 1. "
                        "The comparison chip shows the gain over TF-IDF query search."
                    ),
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [
                        {
                            "label": "Top-10 ranking quality",
                            "field": "response_ndcg",
                            "format": "number",
                        },
                        {
                            "label": "vs TF-IDF",
                            "field": "ndcg_delta_vs_tfidf",
                            "format": "number",
                            "signed": True,
                        },
                    ],
                },
                {
                    "id": "response_mrr",
                    "description": (
                        "How early the first useful model appears, averaged over all "
                        "queries. The TF-IDF difference remains statistically uncertain."
                    ),
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [
                        {
                            "label": "Early useful-model score (MRR)",
                            "field": "response_mrr",
                            "format": "number",
                        },
                        {
                            "label": "vs TF-IDF",
                            "field": "mrr_delta_vs_tfidf",
                            "format": "number",
                            "signed": True,
                        },
                    ],
                },
                {
                    "id": "selection_stability",
                    "description": (
                        "The same retrieval setup was selected independently for 314 of "
                        "322 held-out queries."
                    ),
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [
                        {
                            "label": "Same setup selected",
                            "field": "deployment_fold_share",
                            "format": "percent",
                        },
                        {
                            "label": "Held-out folds",
                            "field": "deployment_fold_count",
                            "format": "number",
                        },
                    ],
                },
            ],
            "charts": [
                {
                    "id": "system_metric_comparison",
                    "title": "Model-ranking scores by approach",
                    "subtitle": (
                        "All approaches rank the same 150 models for the same 322 "
                        "development queries; both scores run from 0 to 1."
                    ),
                    "type": "bar",
                    "dataset": "system_metrics",
                    "sourceId": "system_metrics_sql",
                    "intent": "comparison",
                    "question": (
                        "Which approach ranks the candidate LLMs best on each official "
                        "development metric?"
                    ),
                    "rationale": (
                        "Grouped bars support direct same-scale comparison of two "
                        "metrics across four intentionally retained systems."
                    ),
                    "comparisonContext": {
                        "baseline": "Scores start at zero; higher is better",
                        "denominator": f"{query_count} development queries",
                        "grain": "one held-out ranking per query",
                        "normalization": "macro average on a 0-to-1 scale",
                        "unit": "score",
                    },
                    "encodings": {
                        "x": {
                            "field": "system",
                            "type": "nominal",
                            "label": "System",
                        },
                        "y": {
                            "field": "value",
                            "type": "quantitative",
                            "label": "Metric value",
                            "format": "number",
                        },
                        "color": {
                            "field": "metric",
                            "type": "nominal",
                            "label": "Metric",
                        },
                        "tooltip": [
                            {
                                "field": "query_count",
                                "type": "quantitative",
                                "label": "Queries",
                            },
                            {
                                "field": "evaluation_protocol",
                                "type": "text",
                                "label": "Protocol",
                            },
                            {
                                "field": "run_classification",
                                "type": "text",
                                "label": "Run class",
                            },
                        ],
                    },
                    "yAxisTitle": "Score",
                    "valueFormat": "number",
                    "settings": {
                        "orientation": "vertical",
                        "groupMode": "grouped",
                        "categoryLabelPolicy": "wrap",
                        "showValues": True,
                    },
                    "palette": {"kind": "categorical"},
                    "legend": {
                        "position": "bottom",
                        "sort": "spec",
                        "title": "How ranking quality is measured",
                    },
                    "labels": {"values": "auto"},
                    "referenceLines": [
                        {
                            "axis": "y",
                            "value": 0,
                            "label": "Zero",
                            "lineStyle": "solid",
                        }
                    ],
                    "layout": "full",
                    "maxRows": 8,
                },
                {
                    "id": "filter_flow_chart",
                    "title": "Response documents retained after filtering",
                    "subtitle": (
                        "Historical model answers in the organizer discovery corpus; "
                        "bars show absolute searchable-document counts."
                    ),
                    "type": "horizontalBar",
                    "dataset": "filter_flow",
                    "sourceId": "filter_flow_sql",
                    "intent": "funnel",
                    "question": (
                        "How much searchable answer text remains after each frozen "
                        "cleaning rule?"
                    ),
                    "rationale": (
                        "Ordered horizontal stage bars preserve absolute magnitude and "
                        "remain legible on narrow screens."
                    ),
                    "comparisonContext": {
                        "baseline": "all organizer-provided discovery responses",
                        "denominator": f"{basic_filter['input_documents']:,} documents",
                        "grain": "one discovery-query and model answer pair",
                        "unit": "response documents",
                    },
                    "encodings": {
                        "x": {
                            "field": "stage",
                            "type": "ordinal",
                            "label": "Filtering stage",
                        },
                        "y": {
                            "field": "documents",
                            "type": "quantitative",
                            "label": "Response documents",
                            "format": "compact",
                        },
                        "tooltip": [
                            {
                                "field": "retained_share",
                                "type": "quantitative",
                                "label": "Share retained",
                                "format": "percent",
                            },
                            {
                                "field": "removed_at_step",
                                "type": "quantitative",
                                "label": "Removed at this step",
                            },
                            {"field": "rule", "type": "text", "label": "Rule"},
                        ],
                    },
                    "settings": {
                        "orientation": "horizontal",
                        "sort": "custom",
                        "showValues": True,
                    },
                    "palette": {"kind": "sequential"},
                    "labels": {"values": "all"},
                    "referenceLines": [
                        {
                            "axis": "x",
                            "value": 0,
                            "label": "Zero",
                            "lineStyle": "solid",
                        }
                    ],
                    "layout": "full",
                    "maxRows": 3,
                },
                {
                    "id": "paired_query_outcomes_chart",
                    "title": "Per-query outcomes versus each baseline",
                    "subtitle": (
                        "Share of 322 queries; better or worse always refers to the "
                        "response-search baseline."
                    ),
                    "type": "horizontalStackedBar100",
                    "dataset": "query_outcomes",
                    "sourceId": "query_outcomes_sql",
                    "intent": "composition",
                    "question": (
                        "On how many queries was response search better, tied, or worse "
                        "than each comparison approach?"
                    ),
                    "rationale": (
                        "Normalized stacks show query-level outcome composition while "
                        "the source rows retain exact counts and confidence intervals."
                    ),
                    "comparisonContext": {
                        "denominator": f"{query_count} paired queries per row",
                        "grain": "one query and metric",
                        "normalization": "each row sums to 100%",
                        "unit": "share of queries",
                    },
                    "encodings": {
                        "x": {
                            "field": "comparison_metric",
                            "type": "nominal",
                            "label": "Baseline and metric",
                        },
                        "y": {
                            "field": "outcome_count",
                            "type": "quantitative",
                            "label": "Queries",
                        },
                        "color": {
                            "field": "outcome",
                            "type": "nominal",
                            "label": "Response search",
                        },
                        "tooltip": [
                            {
                                "field": "outcome_share",
                                "type": "quantitative",
                                "label": "Query share",
                                "format": "percent",
                            },
                            {
                                "field": "mean_difference",
                                "type": "quantitative",
                                "label": "Average score difference",
                            },
                            {
                                "field": "ci_lower",
                                "type": "quantitative",
                                "label": "95% interval lower",
                            },
                            {
                                "field": "ci_upper",
                                "type": "quantitative",
                                "label": "95% interval upper",
                            },
                        ],
                    },
                    "settings": {
                        "orientation": "horizontal",
                        "groupMode": "stacked100",
                    },
                    "palette": {"kind": "categorical"},
                    "legend": {
                        "position": "bottom",
                        "sort": "spec",
                        "title": "Response search",
                    },
                    "labels": {"values": "auto"},
                    "layout": "full",
                    "maxRows": 18,
                },
                {
                    "id": "inventory_share_chart",
                    "title": "Model share and top-10 placement share by answer inventory",
                    "subtitle": (
                        "Most-selected fixed setup only; models are split into four groups "
                        "by usable historical-answer count, with even exposure near 25%."
                    ),
                    "type": "bar",
                    "dataset": "inventory_shares",
                    "sourceId": "inventory_shares_sql",
                    "intent": "comparison",
                    "question": (
                        "Are top-10 placements concentrated among models with more "
                        "retained historical answers?"
                    ),
                    "rationale": (
                        "Grouped shares make observed placement concentration directly "
                        "comparable with each inventory group's model share."
                    ),
                    "comparisonContext": {
                        "denominator": f"{model_count} models and {query_count * 10:,} top-10 slots",
                        "grain": "answer-inventory quartile",
                        "normalization": "shares sum to 100% within each series",
                        "unit": "share",
                    },
                    "encodings": {
                        "x": {
                            "field": "inventory_group",
                            "type": "ordinal",
                            "label": "Usable historical answers",
                        },
                        "y": {
                            "field": "share",
                            "type": "quantitative",
                            "label": "Share",
                            "format": "percent",
                        },
                        "color": {
                            "field": "series",
                            "type": "nominal",
                            "label": "Share type",
                        },
                        "tooltip": [
                            {
                                "field": "model_count",
                                "type": "quantitative",
                                "label": "Models in group",
                            },
                            {
                                "field": "inventory_min",
                                "type": "quantitative",
                                "label": "Fewest retained answers",
                            },
                            {
                                "field": "inventory_max",
                                "type": "quantitative",
                                "label": "Most retained answers",
                            },
                            {
                                "field": "top10_appearances",
                                "type": "quantitative",
                                "label": "Top-10 placements",
                            },
                        ],
                    },
                    "settings": {
                        "orientation": "vertical",
                        "groupMode": "grouped",
                        "categoryLabelPolicy": "wrap",
                        "showValues": True,
                    },
                    "palette": {"kind": "categorical"},
                    "legend": {
                        "position": "bottom",
                        "sort": "spec",
                        "title": "What is being shared",
                    },
                    "labels": {"values": "auto"},
                    "referenceLines": [
                        {
                            "axis": "y",
                            "value": 0.25,
                            "label": "Roughly even share",
                            "color": "neutral",
                            "lineStyle": "dashed",
                        }
                    ],
                    "layout": "full",
                    "maxRows": 8,
                },
            ],
            "tables": [],
            "sources": [
                {key: value for key, value in source.items() if key != "query"}
                for source in source_list
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "executive_summary",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## Executive Summary\n\n"
                        "- **What was tested:** whether answers produced by candidate "
                        "LLMs for older questions can help rank those LLMs for a new "
                        "question. The system ranks models; it does not retrieve an "
                        "answer for the user.\n"
                        f"- **Result:** response search scored "
                        f"**{response['ndcg_at_10']:.4f}** for top-10 ranking quality "
                        f"and **{response['mrr']:.4f}** for how early the first useful "
                        f"model appeared, averaged across {query_count} held-out queries.\n"
                        f"- **Comparison:** the top-10 score improved over TF-IDF "
                        f"keyword query search by "
                        f"**{ndcg_ci['mean_difference']:+.4f}**, with an uncertainty "
                        "range entirely above zero. The first-useful-model improvement "
                        "is inconclusive, and dense semantic query search—which compares "
                        "numerical representations of question meaning—remains the "
                        "strongest baseline.\n"
                        "- **Main risk and next action:** adding all matching-answer "
                        "scores can favor models with more usable historical answers. "
                        "The next experiment should normalize for that response volume "
                        "before adding token-confidence features (log probabilities)."
                    ),
                },
                {
                    "id": "task_primer",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## What problem are we solving?\n\n"
                        "TREC (the Text REtrieval Conference) runs shared evaluation "
                        "benchmarks. In the 2026 Million LLMs track, a system receives "
                        f"one new question and must rank {model_count} candidate LLM IDs "
                        "by how likely each is to answer it well—before seeing any of "
                        "their answers to that new question.\n\n"
                        f"The **discovery set** supplies {discovery_query_count:,} older "
                        f"questions and one stored answer from each model: "
                        f"{basic_filter['input_documents']:,} question–model answer "
                        "pairs. The **development set** supplies "
                        f"{query_count} new questions plus organizer quality grades, "
                        "called **qrels**, for every question–model pair. Discovery "
                        "answers provide search evidence; development qrels are used "
                        "inside training folds for setup selection and tie handling, then "
                        "to score finished held-out rankings. The query being evaluated "
                        "always stays sealed. This report is about the "
                        "development set, not the still-hidden final test."
                    ),
                },
                {
                    "id": "metric_guide",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## How to read the two scores\n\n"
                        "- **nDCG@10—Normalized Discounted Cumulative Gain at 10:** asks "
                        "whether the ten "
                        "highest-ranked models are the best ones. It gives more credit "
                        "to stronger organizer grades and to placing them nearer the top. "
                        "It ranges from 0 to 1; higher is better.\n"
                        "- **MRR—Mean Reciprocal Rank:** looks only for the first model "
                        "with a positive organizer grade. First place scores 1.0, second "
                        "place 0.5, and fifth place 0.2; those values are then averaged.\n\n"
                        f"Each of the {query_count} questions has equal weight. nDCG@10 "
                        "judges the quality of a short list, while MRR judges how quickly "
                        "the first useful option appears, so the two metrics can reach "
                        "different conclusions."
                    ),
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "response_ndcg",
                        "response_mrr",
                        "selection_stability",
                    ],
                },
                {
                    "id": "comparison_interpretation",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## Four ways to rank the same 150 models\n\n"
                        "- **Response search—this baseline:** searches answers that the "
                        "models wrote for older discovery questions, then gives evidence "
                        "back to the models that produced the matching answers.\n"
                        "- **Dense query search—strongest baseline:** converts each "
                        "question into a numerical meaning vector with a pretrained "
                        "encoder that is not changed in this experiment. Nearby vectors "
                        "can identify paraphrases without the same wording. It searches "
                        "questions, not old answers.\n"
                        "- **TF-IDF query search—keyword baseline:** term frequency–"
                        "inverse document frequency weighting emphasizes words that are "
                        "distinctive across questions. It finds the 10 "
                        "development questions with the most distinctive word overlap "
                        "and uses their model grades.\n"
                        "- **Global prior—query-blind control:** uses the same general "
                        "model-quality ordering for every query, calculated without the "
                        "held-out query's grades.\n\n"
                        f"Response search improves top-10 quality over TF-IDF "
                        f"({response['ndcg_at_10']:.4f} versus "
                        f"{tfidf['ndcg_at_10']:.4f}) and clearly exceeds the global "
                        f"prior ({prior['ndcg_at_10']:.4f}). Dense query search still "
                        f"leads at {dense['ndcg_at_10']:.4f} nDCG@10 and "
                        f"{dense['mrr']:.4f} MRR."
                    ),
                },
                {
                    "id": "comparison_chart",
                    "type": "chart",
                    "chartId": "system_metric_comparison",
                    "layout": "full",
                },
                {
                    "id": "corpus_filtering",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## Most stored answers are unusable refusals\n\n"
                        "A **response document** is one older question–answer pair tagged "
                        "with the model that wrote the answer. The basic, frozen filter "
                        f"removes {basic_filter['removed_exact_idk']:,} exact normalized "
                        f"“I don't know” answers—{idk_share:.1%} of the corpus—leaving "
                        f"{basic_filter['retained_documents']:,} searchable responses. "
                        f"A stricter alternative leaves "
                        f"{aggressive_filter['retained_documents']:,} after also removing "
                        "very short or refusal-like text.\n\n"
                        "The chart shows why filtering is not a small preprocessing detail: "
                        "it determines how much evidence each model can contribute. The "
                        "rules were fixed before effectiveness scores were inspected."
                    ),
                },
                {
                    "id": "filter_flow",
                    "type": "chart",
                    "chartId": "filter_flow_chart",
                    "layout": "full",
                },
                {
                    "id": "retrieval_method",
                    "type": "markdown",
                    "sourceId": "response_retriever",
                    "body": (
                        "## How BM25 and RM3 turn old answers into a model ranking\n\n"
                        "**BM25** is a classic keyword-search score. It rewards old "
                        "answers that share distinctive words with the new question, "
                        "while adjusting for answer length and how common each word is.\n\n"
                        "**RM3 (Relevance Model 3) is query expansion, not an LLM.** It "
                        "uses BM25 once, takes 10 high-weight terms from the 10 best first-pass "
                        "results, mixes those terms 50/50 with the original question, and "
                        "searches again. The intuition is that promising old answers may "
                        "contain useful vocabulary that the short question omitted; a "
                        "poor first pass can also reinforce noise.\n\n"
                        "1. **Receive a new question.** No candidate answer to that "
                        "question is available.\n"
                        "2. **Search retained historical answers with BM25.**\n"
                        "3. **Expand the question with RM3 and search again.**\n"
                        "4. **Group retrieved answers by the model that wrote them.**\n"
                        "5. **Add each model's positive retrieval scores.** This "
                        "**SumScore** might be 2.1 + 1.4 + 0.5 = 4.0.\n"
                        f"6. **Rank all {model_count} models by SumScore.**\n\n"
                        "No model is trained. No external data or **log probabilities**—"
                        "token-level confidence values from the LLM—are used."
                    ),
                },
                {
                    "id": "nested_selection",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## How held-out testing prevents answer-key leakage\n\n"
                        "**Leakage** happens when information from the query being "
                        "evaluated—especially its organizer grades—influences tuning or "
                        "ranking. That can make a development result look better than it "
                        "really is. This experiment uses nested leave-one-query-out "
                        "validation:\n\n"
                        "1. **Hold out one query** and seal all of its quality grades.\n"
                        "2. **Use only the other 321 queries** to compare 24 frozen "
                        "retrieval setups in an inner leave-one-out loop.\n"
                        "3. **Choose one setup using top-10 ranking quality only.** MRR "
                        "does not influence the choice.\n"
                        "4. **Lock that choice and rank all 150 models** for the held-out "
                        "query.\n"
                        "5. **Reveal the sealed grades only to score the finished "
                        "ranking.**\n"
                        "6. **Repeat for all 322 queries and average the held-out scores.**\n\n"
                        "Exact matching discovery questions are also blocked before both "
                        "RM3 expansion and final retrieval; the audit found zero such "
                        "overlaps. Leakage-safe means the development estimate is fairer—"
                        "not that hidden-test performance is guaranteed.\n\n"
                        "### What the inner selection chose\n\n"
                        "RM3 over response text, with the basic filter and top 500 "
                        f"retrieved answers, was selected in "
                        f"{selection['deployment_fold_count']}/{query_count} held-out "
                        f"folds ({deployment_share:.1%}). The other eight folds were "
                        "extremely close: their inner advantages ranged from only "
                        f"{selection['alternative_margin_min']:.7f} to "
                        f"{selection['alternative_margin_max']:.6f} nDCG@10. A score "
                        "computed after choosing one setup on all development labels "
                        f"({fixed_deployment['ndcg_at_10']:.4f} nDCG@10 / "
                        f"{fixed_deployment['mrr']:.4f} MRR) is optimistic; the nested "
                        f"{response['ndcg_at_10']:.4f} nDCG@10 / "
                        f"{response['mrr']:.4f} MRR result is "
                        "the headline."
                    ),
                },
                {
                    "id": "uncertainty",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## One TF-IDF improvement is clear; the other is uncertain\n\n"
                        "A **paired 95% confidence interval (CI)** starts with each query's "
                        "response-search "
                        "score minus the same query's baseline score, then resamples those "
                        "paired differences 10,000 times. An interval entirely above zero "
                        "supports an improvement; one crossing zero is inconclusive.\n\n"
                        f"Against TF-IDF, the average nDCG@10 difference is "
                        f"{ndcg_ci['mean_difference']:+.4f}, with interval "
                        f"[{ndcg_ci['lower']:+.4f}, {ndcg_ci['upper']:+.4f}]: a clear "
                        f"development-set improvement. The MRR difference is "
                        f"{mrr_ci['mean_difference']:+.4f}, but its interval "
                        f"[{mrr_ci['lower']:+.4f}, {mrr_ci['upper']:+.4f}] crosses zero. "
                        "MRR can show more query wins than losses and still be uncertain "
                        "because the size and variability of those changes also matter.\n\n"
                        f"Both intervals are below zero versus dense query search and "
                        f"above zero versus the global prior. **Wins/ties/losses** in the "
                        "chart count how often response search scores higher, equal, or "
                        "lower on individual queries; they show frequency, not the size "
                        "of each difference."
                    ),
                },
                {
                    "id": "query_outcomes_chart",
                    "type": "chart",
                    "chartId": "paired_query_outcomes_chart",
                    "layout": "full",
                },
                {
                    "id": "volume_bias",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## Main risk: more past answers create more chances to score\n\n"
                        "For the **most-selected fixed setup**, a model's **response "
                        "inventory** is the number of its historical answers that survive "
                        "filtering. SumScore adds every positive match, so a model with "
                        "more searchable answers has more chances to accumulate score even "
                        "if no individual answer is better.\n\n"
                        f"The quarter of models with the largest inventories contains "
                        f"{bias['largest_inventory_quartile_equal_model_share']:.1%} of "
                        f"models but receives {bias['largest_inventory_quartile_top10_share']:.1%} "
                        "of all top-10 placements. Spearman's rank correlation runs from "
                        "-1 to +1: inventory correlates "
                        f"{bias['spearman']['inventory_vs_sum_score']:.3f} with total "
                        f"SumScore and {bias['spearman']['inventory_vs_top10_frequency']:.3f} "
                        "with top-10 frequency.\n\n"
                        "This is a strong association and a plausible aggregation bias, "
                        "not proof that response volume alone causes the ranking. Inventory "
                        "also correlates "
                        f"{bias['inventory_vs_full_development_global_prior_spearman']:.3f} "
                        "with average grades across the full development set, so models "
                        "with more answers may also be better or cover more topics. This "
                        "correlation is a diagnostic; it is not used as a retrieval feature."
                    ),
                },
                {
                    "id": "inventory_share",
                    "type": "chart",
                    "chartId": "inventory_share_chart",
                    "layout": "full",
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next experiment\n\n"
                        "1. **Freeze volume-normalized scoring rules before looking at "
                        "new results.** Try adding only a model's strongest few matches "
                        "or averaging its strongest matches instead of summing every hit.\n"
                        "2. **Use the same nested evaluation and the same three "
                        "comparators.** Success should mean keeping the top-10 gain over "
                        "TF-IDF while materially reducing the connection between answer "
                        "inventory and ranking.\n"
                        "3. **Test dense–response combination only after the volume issue "
                        "is understood.** Dense query search remains stronger and may add "
                        "different evidence.\n"
                        "4. **Keep log probabilities as a later, separate experiment.** "
                        "That makes it possible to distinguish model confidence from the "
                        "value of answer text itself."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Does volume normalization preserve the nDCG gain over TF-IDF "
                        "while reducing inventory correlation?\n"
                        "- Can a preregistered dense–response fusion improve the queries "
                        "where the two retrievers make different errors?\n"
                        "- Does the selected RM3 configuration remain stable on hidden "
                        "test queries with different response coverage?"
                    ),
                },
                {
                    "id": "technical_details",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## Technical details for readers who want them\n\n"
                        f"- **Snapshot:** {as_of_label}; {query_count} development queries, "
                        f"{model_count} models, and "
                        f"{summary['data']['qrel_count']:,} complete query–model grades.\n"
                        "- **Frozen grid:** 24 combinations of BM25 or BM25+RM3, answer "
                        "text or question-plus-answer text, basic or aggressive filtering, "
                        "and retrieval depth 100, 500, or 1,000.\n"
                        f"- **Most selected configuration:** `{deployment}`. This means "
                        "RM3 over answer text, the basic filter, and depth 500.\n"
                        "- **Pinned retrieval:** Lucene-style BM25 (`k1=0.9`, `b=0.4`), "
                        "a frozen Unicode/English stemming analyzer, 10 RM3 feedback "
                        "documents, 10 expansion terms, and 50% original-query weight.\n"
                        "- **Run class:** internal automatic—only organizer data, no "
                        "pretrained answer encoder, manual intervention, external corpus, "
                        "or log-probability metadata.\n"
                        f"- **Runtime:** the CPU experiment through its uncertainty "
                        f"summary took about "
                        f"{results['runtime']['experiment_seconds_through_summary']:.0f} "
                        "seconds in the recorded run.\n"
                        "- **Reproducibility:** all 322×150 written rankings and the "
                        "24×322×150 score tensor passed validation; exact source, index, "
                        "configuration, code, and artifact hashes are available through "
                        "the report's source controls."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## Caveats and assumptions\n\n"
                        "- **Development is not the hidden test.** Nested validation "
                        "reduces tuning bias, but future queries and response coverage can "
                        "still differ.\n"
                        "- **The fixed deployment score is optimistic.** It reuses all "
                        "development labels to choose one setup, so it must not replace "
                        "the nested held-out estimate.\n"
                        "- **The 95% intervals are conditional.** They resample the 322 "
                        "finished held-out predictions but do not rerun configuration "
                        "selection or account for dependence among overlapping training "
                        "folds inside every resample.\n"
                        "- **Tie handling matters.** Between 13 and 55 models receive zero "
                        "retrieval score per query; ties then use a model-quality prior "
                        "computed without the held-out query.\n"
                        "- **Volume bias may shift.** SumScore's inventory dependence may "
                        "change on hidden data with a different mix of stored answers.\n"
                        "- **Exact-query masking is conservative but not perfect isolation.** "
                        "Matching discovery documents are blocked from feedback and final "
                        "retrieval, while still contributing to global qrel-free index "
                        "statistics. The observed overlap count is zero."
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": as_of,
            "status": "ready",
            "datasets": {
                "headline": headline_rows,
                "system_metrics": system_rows,
                "query_outcomes": outcome_rows,
                "filter_flow": filter_rows,
                "inventory_shares": inventory_rows,
            },
        },
        "sources": source_list,
    }


def _query_report_rows(
    *,
    response: dict[str, float],
    dense: dict[str, float],
    tfidf: dict[str, float],
    prior: dict[str, float],
    query_count: int,
    deployment_fold_share: float,
    deployment_fold_count: int,
    comparisons: dict[str, Any],
    filter_input_documents: int,
    basic_retained_documents: int,
    aggressive_retained_documents: int,
    inventory_quartiles: list[dict[str, Any]],
    model_count: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE headline_metrics (
            response_ndcg REAL,
            response_mrr REAL,
            ndcg_delta_vs_tfidf REAL,
            mrr_delta_vs_tfidf REAL,
            deployment_fold_share REAL,
            deployment_fold_count INTEGER
        )
        """
    )
    connection.execute(
        "INSERT INTO headline_metrics VALUES (?, ?, ?, ?, ?, ?)",
        (
            response["ndcg_at_10"],
            response["mrr"],
            response["ndcg_at_10"] - tfidf["ndcg_at_10"],
            response["mrr"] - tfidf["mrr"],
            deployment_fold_share,
            deployment_fold_count,
        ),
    )
    connection.execute(
        """
        CREATE TABLE system_metrics (
            system_order INTEGER,
            metric_order INTEGER,
            system TEXT,
            metric TEXT,
            value REAL,
            query_count INTEGER,
            evaluation_protocol TEXT,
            run_classification TEXT
        )
        """
    )
    systems = (
        (1, "Response search", response, "internal automatic"),
        (2, "Dense query search", dense, "external automatic"),
        (3, "TF-IDF query search", tfidf, "internal automatic"),
        (4, "Global prior", prior, "internal control"),
    )
    for system_order, label, values, run_classification in systems:
        connection.executemany(
            "INSERT INTO system_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    system_order,
                    1,
                    label,
                    "nDCG@10",
                    values["ndcg_at_10"],
                    query_count,
                    "strict outer LOO",
                    run_classification,
                ),
                (
                    system_order,
                    2,
                    label,
                    "MRR",
                    values["mrr"],
                    query_count,
                    "strict outer LOO",
                    run_classification,
                ),
            ),
        )

    connection.execute(
        """
        CREATE TABLE query_outcomes (
            comparison_order INTEGER,
            metric_order INTEGER,
            outcome_order INTEGER,
            comparison_metric TEXT,
            baseline TEXT,
            metric TEXT,
            outcome TEXT,
            outcome_count INTEGER,
            outcome_share REAL,
            query_count INTEGER,
            mean_difference REAL,
            ci_lower REAL,
            ci_upper REAL
        )
        """
    )
    comparison_specs = (
        (1, "tfidf_fixed_k10", "TF-IDF"),
        (2, "dense_nested_reference", "Dense"),
        (3, "global_prior", "Global prior"),
    )
    metric_specs = (
        (1, "nDCG@10", "top-10 quality", "ndcg_at_10_delta", "ndcg_at_10_win_tie_loss"),
        (2, "MRR", "first useful", "mrr_delta", "mrr_win_tie_loss"),
    )
    outcome_specs = (
        (1, "Response better", "wins"),
        (2, "Tie", "ties"),
        (3, "Response worse", "losses"),
    )
    outcome_rows_to_insert: list[tuple[Any, ...]] = []
    for comparison_order, comparison_key, baseline_label in comparison_specs:
        comparison = comparisons[comparison_key]
        for metric_order, metric, metric_label, delta_key, counts_key in metric_specs:
            interval = comparison[delta_key]
            counts = comparison[counts_key]
            comparison_metric = f"{baseline_label} · {metric_label}"
            for outcome_order, outcome, count_key in outcome_specs:
                count = int(counts[count_key])
                outcome_rows_to_insert.append(
                    (
                        comparison_order,
                        metric_order,
                        outcome_order,
                        comparison_metric,
                        baseline_label,
                        metric,
                        outcome,
                        count,
                        count / query_count,
                        query_count,
                        interval["mean_difference"],
                        interval["lower"],
                        interval["upper"],
                    )
                )
    connection.executemany(
        "INSERT INTO query_outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        outcome_rows_to_insert,
    )

    connection.execute(
        """
        CREATE TABLE filter_flow (
            stage_order INTEGER,
            stage TEXT,
            documents INTEGER,
            retained_share REAL,
            removed_at_step INTEGER,
            rule TEXT
        )
        """
    )
    connection.executemany(
        "INSERT INTO filter_flow VALUES (?, ?, ?, ?, ?, ?)",
        (
            (
                1,
                "All discovery responses",
                filter_input_documents,
                1.0,
                0,
                "No filter",
            ),
            (
                2,
                "After basic filter",
                basic_retained_documents,
                basic_retained_documents / filter_input_documents,
                filter_input_documents - basic_retained_documents,
                "Remove exact normalized “I don't know” answers",
            ),
            (
                3,
                "After aggressive filter",
                aggressive_retained_documents,
                aggressive_retained_documents / filter_input_documents,
                basic_retained_documents - aggressive_retained_documents,
                "Also remove very short or refusal-like responses",
            ),
        ),
    )

    connection.execute(
        """
        CREATE TABLE inventory_shares (
            quartile_order INTEGER,
            series_order INTEGER,
            inventory_group TEXT,
            series TEXT,
            share REAL,
            model_count INTEGER,
            inventory_min REAL,
            inventory_max REAL,
            top10_appearances INTEGER
        )
        """
    )
    inventory_labels = {
        1: "Fewest answers",
        2: "Lower-middle",
        3: "Upper-middle",
        4: "Most answers",
    }
    inventory_rows_to_insert: list[tuple[Any, ...]] = []
    for row in inventory_quartiles:
        quartile = int(row["quartile"])
        common = (
            quartile,
            inventory_labels[quartile],
            int(row["model_count"]),
            float(row["inventory_min"]),
            float(row["inventory_max"]),
            int(row["top10_appearances"]),
        )
        inventory_rows_to_insert.extend(
            (
                (
                    common[0],
                    1,
                    common[1],
                    "Models in group",
                    common[2] / model_count,
                    common[2],
                    common[3],
                    common[4],
                    common[5],
                ),
                (
                    common[0],
                    2,
                    common[1],
                    "Top-10 placements",
                    float(row["top10_appearance_share"]),
                    common[2],
                    common[3],
                    common[4],
                    common[5],
                ),
            )
        )
    connection.executemany(
        "INSERT INTO inventory_shares VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        inventory_rows_to_insert,
    )

    headline_rows = [dict(row) for row in connection.execute(HEADLINE_SQL)]
    system_rows = [dict(row) for row in connection.execute(SYSTEM_METRICS_SQL)]
    outcome_rows = [dict(row) for row in connection.execute(QUERY_OUTCOMES_SQL)]
    filter_rows = [dict(row) for row in connection.execute(FILTER_FLOW_SQL)]
    inventory_rows = [dict(row) for row in connection.execute(INVENTORY_SHARES_SQL)]
    connection.close()
    return headline_rows, system_rows, outcome_rows, filter_rows, inventory_rows


def _source_list(results_relative: str, snapshot_as_of: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "response_results",
            "label": "Tracked reviewed response-retrieval result snapshot",
            "path": results_relative,
        },
        {
            "id": "headline_sql",
            "label": "Headline response metric extraction",
            "path": results_relative,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": HEADLINE_SQL.strip(),
                "description": (
                    "Selects the nested response metrics and paired TF-IDF deltas from "
                    "an in-memory table loaded from the reviewed result snapshot."
                ),
                "executed_at": snapshot_as_of,
                "tables_used": ["headline_metrics"],
                "filters": ["322 official development queries", "strict outer LOO"],
                "metric_definitions": _metric_definitions(),
            },
        },
        {
            "id": "system_metrics_sql",
            "label": "Four-system metric comparison",
            "path": results_relative,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": SYSTEM_METRICS_SQL.strip(),
                "description": (
                    "Selects long-form nDCG@10 and MRR rows for response retrieval, "
                    "dense retrieval, TF-IDF, and the global-prior control."
                ),
                "executed_at": snapshot_as_of,
                "tables_used": ["system_metrics"],
                "filters": ["322 official development queries", "strict outer LOO"],
                "metric_definitions": _metric_definitions(),
            },
        },
        {
            "id": "query_outcomes_sql",
            "label": "Paired query outcomes and uncertainty context",
            "path": results_relative,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": QUERY_OUTCOMES_SQL.strip(),
                "description": (
                    "Selects response-search win, tie, and loss counts against each "
                    "baseline for both official metrics, retaining paired intervals."
                ),
                "executed_at": snapshot_as_of,
                "tables_used": ["query_outcomes"],
                "filters": [
                    "322 official development queries per comparison",
                    "strict outer leave-one-query-out predictions",
                ],
                "metric_definitions": _metric_definitions(),
            },
        },
        {
            "id": "filter_flow_sql",
            "label": "Frozen response-filter stage counts",
            "path": results_relative,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": FILTER_FLOW_SQL.strip(),
                "description": (
                    "Selects the raw, basic-filtered, and aggressive-filtered response "
                    "document counts from the reviewed corpus audit."
                ),
                "executed_at": snapshot_as_of,
                "tables_used": ["filter_flow"],
                "filters": [
                    "official discovery response corpus",
                    "frozen basic and aggressive filter definitions",
                ],
                "metric_definitions": [
                    "Documents: one organizer discovery-query and candidate-model response pair.",
                    "Retained share: documents after the stage divided by all discovery responses.",
                ],
            },
        },
        {
            "id": "inventory_shares_sql",
            "label": "Response-inventory quartile representation",
            "path": results_relative,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": INVENTORY_SHARES_SQL.strip(),
                "description": (
                    "Selects each response-inventory quartile's share of candidate models "
                    "and share of all top-10 placements for the deployment configuration."
                ),
                "executed_at": snapshot_as_of,
                "tables_used": ["inventory_shares"],
                "filters": [
                    "basic response filter",
                    "RM3 response-only deployment configuration at depth 500",
                    "322 development queries and 150 candidate models",
                ],
                "metric_definitions": [
                    "Model share: models in the inventory quartile divided by 150.",
                    "Top-10 placement share: placements earned by the quartile divided by 3,220 total top-10 slots.",
                ],
            },
        },
        {
            "id": "experiment_summary",
            "label": "Ignored raw experiment summary copied into the tracked snapshot",
            "path": "artifacts/response_retrieval_baseline/summary.json",
        },
        {
            "id": "validation_receipt",
            "label": "Ignored run and tensor validation receipt",
            "path": "artifacts/response_retrieval_baseline/validation.json",
        },
        {
            "id": "response_pipeline",
            "label": "Tracked exact nested-LOO experiment implementation",
            "path": "trec_mlm/response_experiment.py",
        },
        {
            "id": "response_retriever",
            "label": "Tracked pinned BM25/RM3 response retriever",
            "path": "trec_mlm/response_retrieval.py",
        },
        {
            "id": "development_qrels",
            "label": "Official TREC development qrels",
            "path": "data/dev/llm_dev_qrels.txt",
        },
        {
            "id": "discovery_responses",
            "label": "Official TREC discovery responses",
            "path": "data/dev/llm_discovery_data.json",
        },
    ]


def _selection_stability(
    selections_path: Path,
    inner_scores_path: Path,
    deployment: str,
) -> dict[str, Any]:
    with selections_path.open("r", encoding="utf-8", newline="") as handle:
        selections = list(csv.DictReader(handle, delimiter="\t"))
    with inner_scores_path.open("r", encoding="utf-8", newline="") as handle:
        inner_scores = {
            (row["outer_query_id"], row["configuration"]): float(
                row["inner_macro_ndcg_at_10"]
            )
            for row in csv.DictReader(handle, delimiter="\t")
        }
    if len(selections) != 322:
        raise ValueError("selection audit must contain 322 outer rows")
    deployment_count = sum(row["configuration"] == deployment for row in selections)
    alternative_margins = [
        float(row["selected_inner_ndcg_at_10"])
        - inner_scores[(row["outer_query_id"], deployment)]
        for row in selections
        if row["configuration"] != deployment
    ]
    if not alternative_margins or min(alternative_margins) <= 0:
        raise ValueError("alternative selection margins are missing or invalid")
    return {
        "deployment_configuration": deployment,
        "deployment_fold_count": deployment_count,
        "deployment_fold_share": deployment_count / len(selections),
        "alternative_fold_count": len(alternative_margins),
        "alternative_margin_min": min(alternative_margins),
        "alternative_margin_max": max(alternative_margins),
    }


def _inventory_prior_spearman(summary: dict[str, Any], qrels_path: Path) -> float:
    reproducibility = summary["retrieval_reproducibility"]
    filter_names = reproducibility["inventory_filter_names"]
    basic_index = filter_names.index("basic")
    inventory = [
        float(value)
        for value in reproducibility["retained_document_counts"][basic_index]
    ]
    model_grades: dict[str, list[float]] = {}
    with qrels_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            if not raw_line.strip():
                continue
            _query_id, _iteration, model_id, grade = raw_line.split()
            model_grades.setdefault(model_id, []).append(float(grade))
    models = sorted(model_grades)
    if len(models) != len(inventory):
        raise ValueError("inventory and qrel model orders do not have the same size")
    prior = [
        math.fsum(model_grades[model]) / len(model_grades[model]) for model in models
    ]
    return _spearman(inventory, prior)


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = math.fsum(left_ranks) / len(left_ranks)
    right_mean = math.fsum(right_ranks) / len(right_ranks)
    numerator = math.fsum(
        (x - left_mean) * (y - right_mean)
        for x, y in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(math.fsum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(
        math.fsum((value - right_mean) ** 2 for value in right_ranks)
    )
    if left_scale == 0 or right_scale == 0:
        raise ValueError("Spearman input is constant")
    return numerator / (left_scale * right_scale)


def _average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + 1 + end) / 2
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _metric_definitions() -> list[str]:
    return [
        "nDCG@10: average top-10 ranking quality over 322 equally weighted queries, using organizer grades 0/1/2 and giving higher ranks more credit.",
        "MRR: average reciprocal rank of the first model with a positive organizer grade; first place is 1, second is 0.5, and fifth is 0.2.",
        "Response nested: one held-out ranking per query after choosing the retrieval setup only with the other queries' grades.",
        "Win/tie/loss: the number of paired queries where response search scores higher, equal, or lower than the named baseline.",
        "Paired 95% interval: 10,000 query-level bootstrap resamples of response score minus baseline score, conditional on the finished nested selections.",
    ]


def _evidence_entry(path: Path, *, tracked: bool) -> dict[str, Any]:
    return {
        "path": _repo_relative(path),
        "tracked": tracked,
        "sha256": _sha256(path),
    }


def _repo_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(
            f"report source must be inside the repository: {path}"
        ) from error


def _file_snapshot_time(path: Path) -> str:
    timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).replace(
        microsecond=0
    )
    return timestamp.isoformat().replace("+00:00", "Z")


def _human_snapshot_label(timestamp: str) -> str:
    parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return f"{parsed.strftime('%B')} {parsed.day}, {parsed.year} at {parsed:%H:%M} UTC"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected a JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
