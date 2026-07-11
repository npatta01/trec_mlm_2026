"""Build the canonical technical report for response-level retrieval."""

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
SELECT response_ndcg, response_mrr, ndcg_delta_vs_tfidf, mrr_delta_vs_tfidf
FROM headline_metrics
"""

SYSTEM_METRICS_SQL = """\
SELECT system_order, metric_order, system, metric, value, query_count,
       evaluation_protocol, run_classification
FROM system_metrics
ORDER BY system_order, metric_order
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
            "audience": "technical",
            "delivery_mode": "portable_html",
            "required_structure": [
                "title",
                "technical summary",
                "key findings with visual evidence",
                "scope, data, and metric definitions",
                "methodology and experimental design",
                "limitations, uncertainty, and robustness checks",
                "recommended next steps",
                "further questions",
            ],
            "visual": {
                "section": "four-system comparison",
                "question": (
                    "How does nested response retrieval compare with dense, TF-IDF, "
                    "and the global-prior control?"
                ),
                "takeaway": (
                    "Response retrieval improves nDCG@10 over TF-IDF but remains "
                    "below dense; its MRR improvement over TF-IDF is uncertain."
                ),
                "family": "comparison and ranking",
                "type": "full-width grouped vertical bar",
                "rows": 8,
                "category_count": 4,
                "metric_count": 2,
                "palette_policy": "hard two-root cap with a visible metric legend",
                "scale": "shared zero-based 0-to-1 metric scale",
                "mobile_qa": (
                    "full-width chart in the responsive portable reader; the delivery "
                    "receipt records whether a compatible browser completed narrow QA"
                ),
            },
            "table_omission": (
                "The four-system comparison is visualized rather than repeated as a "
                "broad table; exact values and all 24 fixed candidates remain here."
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
    vs_dense = summary["comparisons"]["dense_nested_reference"]
    vs_prior = summary["comparisons"]["global_prior"]
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

    headline_rows, system_rows = _query_report_rows(
        response=response,
        dense=dense,
        tfidf=tfidf,
        prior=prior,
        query_count=query_count,
    )
    results_relative = _repo_relative(results_path)
    source_list = _source_list(results_relative, as_of)
    title = "TREC 2026 Response Retrieval Baseline"
    ndcg_ci = vs_tfidf["ndcg_at_10_delta"]
    mrr_ci = vs_tfidf["mrr_delta"]
    dense_ndcg_ci = vs_dense["ndcg_at_10_delta"]
    dense_mrr_ci = vs_dense["mrr_delta"]
    idk_share = basic_filter["removed_exact_idk"] / basic_filter["input_documents"]
    deployment_share = selection["deployment_fold_share"]

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "Nested evaluation of response-level BM25 and RM3 retrieval against "
                "dense query retrieval, TF-IDF, and a fold-local global prior."
            ),
            "generatedAt": as_of,
            "cards": [
                {
                    "id": "response_ndcg",
                    "description": (
                        "Macro nDCG@10 over stitched outer predictions; the chip is "
                        "the paired mean difference from fixed TF-IDF k=10."
                    ),
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [
                        {
                            "label": "Response nested nDCG@10",
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
                        "Mean reciprocal rank over stitched outer predictions; its "
                        "paired TF-IDF interval includes zero."
                    ),
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [
                        {
                            "label": "Response nested MRR",
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
            ],
            "charts": [
                {
                    "id": "system_metric_comparison",
                    "title": "Development retrieval metrics by system",
                    "subtitle": (
                        "322 strict outer leave-one-query-out predictions; both "
                        "metrics use a 0-to-1 scale and higher is better."
                    ),
                    "type": "bar",
                    "dataset": "system_metrics",
                    "sourceId": "system_metrics_sql",
                    "intent": "comparison",
                    "question": (
                        "How does response retrieval compare with dense, TF-IDF, and "
                        "the global prior on both official development metrics?"
                    ),
                    "rationale": (
                        "Grouped bars support direct same-scale comparison of two "
                        "metrics across four intentionally retained systems."
                    ),
                    "comparisonContext": {
                        "query_count": query_count,
                        "score_domain": [0, 1],
                        "zero_baseline": True,
                        "higher_is_better": True,
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
                    "options": {"orientation": "vertical", "grouping": "grouped"},
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
                }
            ],
            "tables": [],
            "sources": [
                {key: value for key, value in source.items() if key != "query"}
                for source in source_list
            ],
            "blocks": [
                {"id": "title", "type": "markdown", "body": f"# {title}"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## Response evidence helps, but dense retrieval still leads\n\n"
                        f"- **Primary result:** nested response retrieval reached "
                        f"**{response['ndcg_at_10']:.4f} nDCG@10** and "
                        f"**{response['mrr']:.4f} MRR** across {query_count} queries.\n"
                        f"- **Versus TF-IDF:** nDCG@10 improved by "
                        f"**{ndcg_ci['mean_difference']:+.4f}**, with a 95% paired "
                        f"interval of [{ndcg_ci['lower']:+.4f}, "
                        f"{ndcg_ci['upper']:+.4f}]. The MRR change was "
                        f"{mrr_ci['mean_difference']:+.4f}, but its interval "
                        f"[{mrr_ci['lower']:+.4f}, {mrr_ci['upper']:+.4f}] crosses zero.\n"
                        f"- **Dense remains stronger:** response retrieval trails it by "
                        f"{-dense_ndcg_ci['mean_difference']:.4f} nDCG@10 and "
                        f"{-dense_mrr_ci['mean_difference']:.4f} MRR; both paired "
                        "intervals remain below zero.\n"
                        f"- **Practical reading:** `{deployment}` was selected in "
                        f"{selection['deployment_fold_count']}/{query_count} folds "
                        f"({deployment_share:.1%}). This is a useful internal automatic "
                        "baseline, not the new best system."
                    ),
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": ["response_ndcg", "response_mrr"],
                },
                {
                    "id": "comparison_interpretation",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## The gain over TF-IDF is objective-dependent\n\n"
                        f"Response retrieval improves graded top-10 utility over TF-IDF "
                        f"({tfidf['ndcg_at_10']:.4f}) and clearly exceeds the prior "
                        f"({prior['ndcg_at_10']:.4f}). Its MRR advantage over TF-IDF "
                        "is not statistically resolved by the paired interval. Dense "
                        f"retrieval leads at {dense['ndcg_at_10']:.4f} nDCG@10 and "
                        f"{dense['mrr']:.4f} MRR, so response text adds signal without "
                        "displacing the semantic-query baseline."
                    ),
                },
                {
                    "id": "comparison_chart",
                    "type": "chart",
                    "chartId": "system_metric_comparison",
                    "layout": "full",
                },
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## Scope, corpus, and metric definitions\n\n"
                        f"This snapshot is as of **{as_of_label}**. Evaluation covers "
                        f"{query_count} development queries, {model_count} candidate "
                        f"models, and {summary['data']['qrel_count']:,} complete qrels. "
                        "nDCG@10 uses linear positive 0/1/2 gains through rank 10; MRR "
                        "uses the reciprocal rank of the first model with grade above "
                        "zero. All four systems rank the same candidates.\n\n"
                        f"The organizer discovery corpus contains "
                        f"{basic_filter['input_documents']:,} responses. Exact normalized "
                        f"“I don't know” answers account for {idk_share:.1%}; removing "
                        f"them leaves {basic_filter['retained_documents']:,} documents. "
                        f"The aggressive filter leaves "
                        f"{aggressive_filter['retained_documents']:,} after also removing "
                        "very short or refusal-like responses. Exact development-query "
                        "overlap is masked before both RM3 feedback and final retrieval; "
                        "the audit found zero overlapping documents."
                    ),
                },
                {
                    "id": "retrieval_method",
                    "type": "markdown",
                    "sourceId": "response_retriever",
                    "body": (
                        "## BM25 and RM3 search prior model responses\n\n"
                        "For each target query, BM25 retrieves response documents using "
                        "a pinned English analyzer and Lucene-style scoring (`k1=0.9`, "
                        "`b=0.4`). RM3 is a second pass: it mixes the original query "
                        "equally with 10 expansion terms estimated from the top 10 first-"
                        "pass documents. Each model's **SumScore** is the reproducible "
                        "floating-point sum of its positive retrieved-document scores.\n\n"
                        "The preregistered 24 candidates cross two methods (BM25 or RM3), "
                        "two indexed fields (response only or discovery query plus "
                        "response), two filters, and depths 100/500/1,000. No model is "
                        "trained, and neither external data nor logprob metadata is used."
                    ),
                },
                {
                    "id": "nested_selection",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## Nested selection keeps every outer label sealed\n\n"
                        "For outer query `o`, each of the 24 candidates is evaluated by "
                        "leave-one-query-out validation across the other 321 queries. "
                        "Inner query `i` uses a prior excluding both `o` and `i`; macro "
                        "inner nDCG@10 selects the candidate, and MRR is never used for "
                        "selection. The chosen candidate then ranks `o`, with score ties "
                        "resolved by the outer fold's qrel prior and model ID.\n\n"
                        f"`{deployment}` won {selection['deployment_fold_count']} folds. "
                        f"The eight alternative selections beat it internally by only "
                        f"{selection['alternative_margin_min']:.7f} to "
                        f"{selection['alternative_margin_max']:.6f} nDCG@10, revealing "
                        "high-leverage near-tie instability. The full-development fixed "
                        f"score ({fixed_deployment['ndcg_at_10']:.4f} nDCG@10, "
                        f"{fixed_deployment['mrr']:.4f} MRR) is therefore optimistic; "
                        "the nested 0.2774/0.3175 result is the leakage-safe headline."
                    ),
                },
                {
                    "id": "uncertainty",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## Paired intervals support one clear TF-IDF gain\n\n"
                        f"Against TF-IDF, response retrieval wins/ties/loses "
                        f"**{vs_tfidf['ndcg_at_10_win_tie_loss']['wins']}/"
                        f"{vs_tfidf['ndcg_at_10_win_tie_loss']['ties']}/"
                        f"{vs_tfidf['ndcg_at_10_win_tie_loss']['losses']}** queries on "
                        f"nDCG@10 and **{vs_tfidf['mrr_win_tie_loss']['wins']}/"
                        f"{vs_tfidf['mrr_win_tie_loss']['ties']}/"
                        f"{vs_tfidf['mrr_win_tie_loss']['losses']}** on reciprocal rank. "
                        f"Against the global prior, both intervals are positive: nDCG "
                        f"[{vs_prior['ndcg_at_10_delta']['lower']:+.4f}, "
                        f"{vs_prior['ndcg_at_10_delta']['upper']:+.4f}] and MRR "
                        f"[{vs_prior['mrr_delta']['lower']:+.4f}, "
                        f"{vs_prior['mrr_delta']['upper']:+.4f}]. Written runs and the "
                        "24×322×150 score tensor passed the validation receipt."
                    ),
                },
                {
                    "id": "volume_bias",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## SumScore has strong inventory-volume dependence\n\n"
                        f"For the deployment candidate, retained response inventory "
                        f"correlates {bias['spearman']['inventory_vs_sum_score']:.3f} "
                        f"with total SumScore, {bias['spearman']['inventory_vs_mean_rank']:.3f} "
                        f"with mean rank, and {bias['spearman']['inventory_vs_top10_frequency']:.3f} "
                        "with top-10 frequency (Spearman). The largest inventory quartile "
                        f"receives {bias['largest_inventory_quartile_top10_share']:.1%} "
                        "of top-10 appearances despite containing "
                        f"{bias['largest_inventory_quartile_equal_model_share']:.1%} of "
                        "models.\n\nThis is strong volume dependence and a plausible "
                        "aggregation bias—not proof of a pure artifact. Inventory also "
                        f"correlates {bias['inventory_vs_full_development_global_prior_spearman']:.3f} "
                        "with the full-development qrel prior, so response volume partly "
                        "tracks model quality or coverage."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": "response_results",
                    "body": (
                        "## What the estimate does not establish\n\n"
                        "- This is development-set cross-validation, not hidden-test "
                        "performance.\n"
                        "- The fixed deployment score reuses development labels for "
                        "configuration choice and must not replace the nested estimate.\n"
                        "- The 10k paired-query bootstrap conditions on the stitched "
                        "outer predictions. It does not repeat configuration selection "
                        "or account for dependence among overlapping training folds.\n"
                        "- Between 13 and 55 models have zero selected retrieval score "
                        "per query, making fold-local prior tie breaking material.\n"
                        "- SumScore's volume dependence may not transfer unchanged to "
                        "hidden data with a different response inventory."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next experiment\n\n"
                        "1. Preregister a **volume-normalized response aggregation** "
                        "grid—such as capped top-response sums and top-response means—"
                        "while keeping the same retrieval documents and nested protocol.\n"
                        "2. Compare it directly with frozen SumScore, dense retrieval, "
                        "TF-IDF, and the fold-local prior; include inventory-dependence "
                        "diagnostics as a success guardrail.\n"
                        "3. Only after isolating aggregation effects, add logprobs as a "
                        "separate feature ablation so confidence and text evidence remain "
                        "distinguishable."
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
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": as_of,
            "status": "ready",
            "datasets": {
                "headline": headline_rows,
                "system_metrics": system_rows,
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
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    connection = sqlite3.connect(":memory:")
    connection.row_factory = sqlite3.Row
    connection.execute(
        """
        CREATE TABLE headline_metrics (
            response_ndcg REAL,
            response_mrr REAL,
            ndcg_delta_vs_tfidf REAL,
            mrr_delta_vs_tfidf REAL
        )
        """
    )
    connection.execute(
        "INSERT INTO headline_metrics VALUES (?, ?, ?, ?)",
        (
            response["ndcg_at_10"],
            response["mrr"],
            response["ndcg_at_10"] - tfidf["ndcg_at_10"],
            response["mrr"] - tfidf["mrr"],
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
        (1, "Response nested", response, "internal automatic"),
        (2, "Dense nested", dense, "external automatic"),
        (3, "TF-IDF k=10", tfidf, "internal automatic"),
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
    headline_rows = [dict(row) for row in connection.execute(HEADLINE_SQL)]
    system_rows = [dict(row) for row in connection.execute(SYSTEM_METRICS_SQL)]
    connection.close()
    return headline_rows, system_rows


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
        "nDCG@10: macro mean over 322 queries with linear positive 0/1/2 qrel gain and log2 rank discount through rank 10.",
        "MRR: macro mean reciprocal rank of the first candidate with qrel grade greater than zero.",
        "Response nested: one stitched outer prediction per query after the response candidate is selected only by inner macro nDCG@10.",
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
