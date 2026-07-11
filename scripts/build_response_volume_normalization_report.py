"""Build the reviewed novice report for response-volume normalization."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Any

from trec_mlm.metrics import EvaluationResult, evaluate_rankings, paired_bootstrap_ci
from trec_mlm.query_knn import load_complete_qrels
from trec_mlm.response_experiment import load_dense_per_query_reference
from trec_mlm.runio import rankings_from_run, validate_trec_run


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/response_volume_normalization"
RERUN_DIR = ROOT / "artifacts/response_volume_normalization_rerun"
DEFAULT_RESULTS = ROOT / "reports/trec_2026_response_volume_normalization_results.json"
DEFAULT_ARTIFACT = ROOT / "reports/trec_2026_response_volume_normalization.artifact.json"
QRELS = ROOT / "data/dev/llm_dev_qrels.txt"
DENSE_PER_QUERY = ROOT / "reports/trec_2026_dense_query_knn_per_query.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=ARTIFACT_DIR)
    parser.add_argument("--rerun-dir", type=Path, default=RERUN_DIR)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--artifact-output", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = _read_json(args.artifact_dir / "summary.json")
    diagnostics = _read_json(args.artifact_dir / "diagnostics.json")
    decision = _read_json(args.artifact_dir / "decision.json")
    validation = _read_json(args.artifact_dir / "validation.json")
    rerun_summary = _read_json(args.rerun_dir / "summary.json")

    results = _reviewed_results(
        summary=summary,
        diagnostics=diagnostics,
        decision=decision,
        validation=validation,
        rerun_summary=rerun_summary,
        artifact_dir=args.artifact_dir,
        rerun_dir=args.rerun_dir,
    )
    args.results_output.parent.mkdir(parents=True, exist_ok=True)
    args.results_output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    artifact = _artifact(results, args.results_output)
    args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
    args.artifact_output.write_text(
        json.dumps(artifact, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(args.results_output)
    print(args.artifact_output)


def _reviewed_results(
    *,
    summary: dict[str, Any],
    diagnostics: dict[str, Any],
    decision: dict[str, Any],
    validation: dict[str, Any],
    rerun_summary: dict[str, Any],
    artifact_dir: Path,
    rerun_dir: Path,
) -> dict[str, Any]:
    _validate_shape(summary, diagnostics, decision, validation)
    deterministic = _validate_determinism(
        summary,
        rerun_summary,
        artifact_dir,
        rerun_dir,
    )
    independent = _recompute_effectiveness(summary, artifact_dir)
    _validate_decision(summary, diagnostics, decision)

    snapshot_as_of = datetime.fromtimestamp(
        (artifact_dir / "summary.json").stat().st_mtime,
        tz=UTC,
    ).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "snapshot_as_of": snapshot_as_of,
        "question": (
            "Can capped or averaged response-score aggregation preserve response "
            "retrieval quality while reducing advantage from larger response inventories?"
        ),
        "audience": "first-time general stakeholder",
        "recommendation": {
            "response_retrieval": "retain_sumscore_control",
            "overall_best_baseline": "dense_nested_reference",
            "reason": (
                "Top10Sum raised the ranking point estimates but failed both preregistered "
                "volume-dependence gates, so all four required gates did not pass."
            ),
            "next_action": (
                "Stop this aggregation branch; do not add log probabilities or fusion "
                "as a post-hoc continuation of this experiment."
            ),
        },
        "protocol": summary["protocol"],
        "data": summary["data"],
        "aggregation_grid": summary["aggregation_grid"],
        "metrics": summary["metrics"],
        "comparisons": summary["comparisons"],
        "selection": summary["selection"],
        "deployment": summary["deployment"],
        "diagnostics": diagnostics,
        "decision": decision,
        "validation": {
            "run_receipt": validation,
            "deterministic_rerun": deterministic,
            "independent_effectiveness_recalculation": independent,
            "share_status": "ready_with_preregistered_negative_decision",
        },
        "definitions": {
            "task": (
                "For each new query, rank the same 150 candidate language models so a "
                "useful model appears as early as possible."
            ),
            "historical_response": (
                "One organizer-provided answer produced by one candidate model for one "
                "earlier discovery query; this is not the last chat response."
            ),
            "sumscore": (
                "Add every positive retrieved historical-response score belonging to a model."
            ),
            "top10sum": (
                "For each model, add only its ten strongest retrieved historical-response "
                "scores; use fewer when fewer than ten are found."
            ),
            "rm3": (
                "A lexical query-expansion step: an initial BM25 search suggests related "
                "terms, then the search is rerun. RM3 is not a language model."
            ),
            "nested_validation": (
                "Each held-out query is predicted only after the retrieval setup is selected "
                "using the other queries; its grades never choose its own setup."
            ),
            "ndcg_at_10": (
                "Average quality of the first ten ranked models, giving more credit to "
                "useful models near the top; 0 is worst and 1 is ideal."
            ),
            "mrr": (
                "Average reciprocal rank of the first useful model: first is 1, second is "
                "0.5, fifth is 0.2."
            ),
            "inventory_rho": (
                "Spearman association between a model's usable historical-response count "
                "and its total retrieval score; lower absolute values mean less dependence."
            ),
        },
        "caveats": [
            (
                "Inventory association is not proof of causal unfairness; stronger models "
                "may legitimately have more historical responses."
            ),
            (
                "The deployment-configuration bias diagnostics use all development queries "
                "and are behavior checks, not unbiased effectiveness estimates."
            ),
            (
                "The normalized-versus-SumScore nDCG interval includes zero, so the +0.0119 "
                "point difference is promising but not statistically decisive here."
            ),
            (
                "Dense query retrieval remains materially better than both lexical response "
                "systems on these development queries."
            ),
        ],
        "evidence_inputs": {
            "summary": str(artifact_dir / "summary.json"),
            "diagnostics": str(artifact_dir / "diagnostics.json"),
            "decision": str(artifact_dir / "decision.json"),
            "validation": str(artifact_dir / "validation.json"),
            "rerun_summary": str(rerun_dir / "summary.json"),
            "qrels": str(QRELS),
            "dense_per_query": str(DENSE_PER_QUERY),
        },
    }


def _validate_shape(
    summary: dict[str, Any],
    diagnostics: dict[str, Any],
    decision: dict[str, Any],
    validation: dict[str, Any],
) -> None:
    if summary["data"]["query_count"] != 322:
        raise ValueError("expected 322 official development queries")
    if summary["data"]["candidate_model_count"] != 150:
        raise ValueError("expected the 150-model universe")
    if summary["aggregation_grid"] != {
        "normalized_order": [
            "best_hit",
            "top3_sum",
            "top10_sum",
            "top3_mean",
            "top10_mean",
            "mean_all",
        ],
        "normalized_candidate_count": 144,
        "sumscore_candidate_count": 24,
        "combined_candidate_count": 168,
    }:
        raise ValueError("aggregation grid does not match the locked protocol")
    if summary["protocol"]["sumscore_excluded_from_primary_selector"] is not True:
        raise ValueError("SumScore entered the primary selector")
    if summary["protocol"]["logprob_features"] is not False:
        raise ValueError("log probabilities unexpectedly entered the experiment")
    if len(decision["gates"]) != 4:
        raise ValueError("expected exactly four preregistered gates")
    if decision != summary["decision"] or diagnostics != summary["diagnostics"]:
        raise ValueError("standalone evidence disagrees with the combined summary")
    if not all(receipt["valid"] for receipt in validation["runs"].values()):
        raise ValueError("one or more complete TREC runs failed validation")
    if validation["sumscore_reference"]["metric_match"] is not True:
        raise ValueError("SumScore did not reproduce the prior baseline")
    if validation["sumscore_run_parity"]["identical_rankings_and_scores"] is not True:
        raise ValueError("SumScore run differs from Baseline 3")
    if validation["execution_provenance"]["git_worktree_status"] != "":
        raise ValueError("official run did not record a clean worktree")


def _validate_determinism(
    summary: dict[str, Any],
    rerun_summary: dict[str, Any],
    artifact_dir: Path,
    rerun_dir: Path,
) -> dict[str, Any]:
    first = deepcopy(summary)
    second = deepcopy(rerun_summary)
    first.pop("runtime", None)
    second.pop("runtime", None)
    if first != second:
        raise ValueError("official summaries differ beyond elapsed runtime")

    filenames = (
        "diagnostics.json",
        "decision.json",
        "per_query.tsv",
        "selections.tsv",
        "normalized_run.trec",
        "sumscore_control.trec",
        "tfidf_control.trec",
        "global_prior_control.trec",
    )
    identical = {
        name: (artifact_dir / name).read_bytes() == (rerun_dir / name).read_bytes()
        for name in filenames
    }
    if not all(identical.values()):
        raise ValueError(f"deterministic rerun mismatch: {identical!r}")

    first_config = _read_json(artifact_dir / "config.json")
    second_config = _read_json(rerun_dir / "config.json")
    score_hash_match = (
        first_config["combined_retrieval_hashes"]["score_tensor_sha256"]
        == second_config["combined_retrieval_hashes"]["score_tensor_sha256"]
    )
    if not score_hash_match:
        raise ValueError("combined retrieval tensor changed on rerun")
    return {
        "summary_equal_except_runtime": True,
        "decision_relevant_files_identical": identical,
        "combined_score_tensor_hash_identical": True,
        "first_runtime_seconds": summary["runtime"]["experiment_seconds"],
        "rerun_runtime_seconds": rerun_summary["runtime"]["experiment_seconds"],
    }


def _recompute_effectiveness(
    summary: dict[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    qrels = load_complete_qrels(QRELS)
    run_paths = {
        "response_normalized_nested": artifact_dir / "normalized_run.trec",
        "response_sumscore_nested": artifact_dir / "sumscore_control.trec",
        "tfidf_fixed_k10": artifact_dir / "tfidf_control.trec",
        "global_prior": artifact_dir / "global_prior_control.trec",
    }
    evaluations: dict[str, EvaluationResult] = {}
    for system, path in run_paths.items():
        validate_trec_run(
            path,
            expected_query_ids=tuple(qrels),
            expected_model_ids=tuple(next(iter(qrels.values()))),
        )
        evaluations[system] = evaluate_rankings(
            rankings_from_run(path),
            qrels,
            k=10,
        )
    evaluations["dense_nested_reference"] = load_dense_per_query_reference(
        DENSE_PER_QUERY,
        expected_query_ids=tuple(qrels),
    )

    recomputed_metrics = {
        name: {
            "ndcg_at_10": value.macro_ndcg_at_k,
            "mrr": value.mean_reciprocal_rank,
        }
        for name, value in evaluations.items()
    }
    for system, metrics in recomputed_metrics.items():
        for metric, observed in metrics.items():
            expected = summary["metrics"][system][metric]
            if not math.isclose(observed, expected, rel_tol=0.0, abs_tol=5e-12):
                raise ValueError(
                    f"recomputed {system} {metric} differs: {observed} != {expected}"
                )

    normalized = evaluations["response_normalized_nested"]
    baselines = (
        "response_sumscore_nested",
        "dense_nested_reference",
        "tfidf_fixed_k10",
        "global_prior",
    )
    comparison_checks: dict[str, bool] = {}
    for offset, baseline_name in enumerate(baselines):
        baseline = evaluations[baseline_name]
        ndcg = paired_bootstrap_ci(
            {key: value.ndcg_at_k for key, value in baseline.per_query.items()},
            {key: value.ndcg_at_k for key, value in normalized.per_query.items()},
            n_resamples=10_000,
            seed=2026 + offset,
        )
        mrr = paired_bootstrap_ci(
            {
                key: value.reciprocal_rank
                for key, value in baseline.per_query.items()
            },
            {
                key: value.reciprocal_rank
                for key, value in normalized.per_query.items()
            },
            n_resamples=10_000,
            seed=12_026 + offset,
        )
        expected = summary["comparisons"][baseline_name]
        comparison_checks[baseline_name] = _ci_matches(
            asdict(ndcg), expected["ndcg_at_10_delta"]
        ) and _ci_matches(asdict(mrr), expected["mrr_delta"])
    if not all(comparison_checks.values()):
        raise ValueError(f"paired interval recalculation mismatch: {comparison_checks}")
    return {
        "metrics": recomputed_metrics,
        "paired_intervals_match": comparison_checks,
        "source": "TREC runs plus official qrels; dense per-query tracked reference",
    }


def _validate_decision(
    summary: dict[str, Any],
    diagnostics: dict[str, Any],
    decision: dict[str, Any],
) -> None:
    comparison = summary["comparisons"]["tfidf_fixed_k10"]
    normalized_ndcg = summary["metrics"]["response_normalized_nested"]["ndcg_at_10"]
    sumscore_ndcg = summary["metrics"]["response_sumscore_nested"]["ndcg_at_10"]
    deployment = diagnostics["deployment_behavior"]["normalized"]
    expected = {
        "ndcg_ci_above_tfidf": comparison["ndcg_at_10_delta"]["lower"] > 0.0,
        "ndcg_within_0_010_of_sumscore": normalized_ndcg >= sumscore_ndcg - 0.010,
        "deployment_inventory_rho_at_most_0_7388": abs(
            deployment["inventory_vs_aggregation_score_spearman"]
        )
        <= 0.7388,
        "deployment_largest_quartile_share_at_most_0_411": deployment[
            "largest_inventory_quartile_top10_share"
        ]
        <= 0.411,
    }
    observed = {name: value["passed"] for name, value in decision["gates"].items()}
    if observed != expected:
        raise ValueError(f"decision gates were not reproducible: {observed} != {expected}")
    if decision["all_four_gates_passed"] != all(expected.values()):
        raise ValueError("combined decision disagrees with individual gates")


def _ci_matches(observed: dict[str, Any], expected: dict[str, Any]) -> bool:
    return all(
        (
            observed[key] == expected[key]
            if isinstance(observed[key], int)
            else math.isclose(
                float(observed[key]),
                float(expected[key]),
                rel_tol=0.0,
                abs_tol=5e-15,
            )
        )
        for key in observed
    )


def _artifact(results: dict[str, Any], results_path: Path) -> dict[str, Any]:
    generated_at = results["snapshot_as_of"]
    metrics = results["metrics"]
    comparisons = results["comparisons"]
    diagnostics = results["diagnostics"]
    decision = results["decision"]
    normalized = metrics["response_normalized_nested"]
    sumscore = metrics["response_sumscore_nested"]
    normalized_bias = diagnostics["deployment_behavior"]["normalized"]
    sumscore_bias = diagnostics["deployment_behavior"]["sumscore"]

    headline = [
        {
            "normalized_ndcg": normalized["ndcg_at_10"],
            "ndcg_delta_vs_sumscore": (
                normalized["ndcg_at_10"] - sumscore["ndcg_at_10"]
            ),
            "normalized_mrr": normalized["mrr"],
            "mrr_delta_vs_sumscore": normalized["mrr"] - sumscore["mrr"],
            "inventory_rho": normalized_bias[
                "inventory_vs_aggregation_score_spearman"
            ],
            "inventory_rho_gate": 0.7388,
            "largest_quartile_share": normalized_bias[
                "largest_inventory_quartile_top10_share"
            ],
            "largest_quartile_gate": 0.411,
            "passed_gate_count": sum(
                int(value["passed"]) for value in decision["gates"].values()
            ),
            "gate_count": len(decision["gates"]),
        }
    ]
    system_metrics = _system_metric_rows(metrics)
    query_outcomes = _query_outcome_rows(comparisons)
    bias_gate_rows = _bias_gate_rows(normalized_bias, sumscore_bias)
    inventory_rows = _inventory_rows(normalized_bias, sumscore_bias)

    sources = _sources(results_path, generated_at)
    cards = _cards()
    charts = _charts()
    blocks = _blocks(results)
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": (
                "Capping past-answer evidence raised the ranking point estimate—but "
                "failed the volume-risk gates"
            ),
            "description": (
                "A general-audience, leakage-safe evaluation of capped and averaged "
                "historical-response aggregation for TREC Million LLMs 2026."
            ),
            "generatedAt": generated_at,
            "cards": cards,
            "charts": charts,
            "tables": [],
            "sources": sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "system_metrics": system_metrics,
                "query_outcomes": query_outcomes,
                "bias_gates": bias_gate_rows,
                "inventory_quartiles": inventory_rows,
            },
        },
        "sources": sources,
    }


def _system_metric_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    systems = (
        (1, "response_normalized_nested", "Top10Sum response search"),
        (2, "response_sumscore_nested", "SumScore response search"),
        (3, "dense_nested_reference", "Dense query search"),
        (4, "tfidf_fixed_k10", "TF-IDF query search"),
        (5, "global_prior", "Global prior"),
    )
    rows: list[dict[str, Any]] = []
    for system_order, key, label in systems:
        for metric_order, (metric, field) in enumerate(
            (("nDCG@10", "ndcg_at_10"), ("MRR", "mrr")),
            start=1,
        ):
            rows.append(
                {
                    "system_order": system_order,
                    "metric_order": metric_order,
                    "system": label,
                    "metric": metric,
                    "value": metrics[key][field],
                    "query_count": 322,
                    "evaluation_protocol": "strict nested outer leave-one-out",
                    "higher_is_better": True,
                }
            )
    return rows


def _query_outcome_rows(comparisons: dict[str, Any]) -> list[dict[str, Any]]:
    labels = (
        (1, "response_sumscore_nested", "SumScore"),
        (2, "tfidf_fixed_k10", "TF-IDF"),
        (3, "dense_nested_reference", "Dense"),
    )
    rows: list[dict[str, Any]] = []
    for comparison_order, key, label in labels:
        comparison = comparisons[key]
        outcomes = comparison["ndcg_at_10_win_tie_loss"]
        interval = comparison["ndcg_at_10_delta"]
        for outcome_order, (field, outcome_label) in enumerate(
            (("wins", "Top10Sum better"), ("ties", "Tie"), ("losses", "Top10Sum worse")),
            start=1,
        ):
            count = outcomes[field]
            rows.append(
                {
                    "comparison_order": comparison_order,
                    "outcome_order": outcome_order,
                    "baseline": label,
                    "outcome": outcome_label,
                    "outcome_count": count,
                    "outcome_share": count / 322,
                    "query_count": 322,
                    "mean_ndcg_difference": interval["mean_difference"],
                    "ci_lower": interval["lower"],
                    "ci_upper": interval["upper"],
                }
            )
    return rows


def _bias_gate_rows(
    normalized: dict[str, Any],
    sumscore: dict[str, Any],
) -> list[dict[str, Any]]:
    values = (
        (
            1,
            "Inventory-score correlation",
            sumscore["inventory_vs_aggregation_score_spearman"],
            normalized["inventory_vs_aggregation_score_spearman"],
            0.7388,
        ),
        (
            2,
            "Largest-quartile placement share",
            sumscore["largest_inventory_quartile_top10_share"],
            normalized["largest_inventory_quartile_top10_share"],
            0.411,
        ),
    )
    rows: list[dict[str, Any]] = []
    for metric_order, metric, sum_value, normalized_value, gate in values:
        for series_order, (series, value, role) in enumerate(
            (
                ("SumScore", sum_value, "baseline"),
                ("Top10Sum", normalized_value, "actual"),
                ("Maximum allowed", gate, "target"),
            ),
            start=1,
        ):
            rows.append(
                {
                    "metric_order": metric_order,
                    "series_order": series_order,
                    "metric": metric,
                    "series": series,
                    "value": value,
                    "role": role,
                    "lower_is_better": True,
                    "top10_position_count": 3220,
                    "model_count": 150,
                }
            )
    return rows


def _inventory_rows(
    normalized: dict[str, Any],
    sumscore: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = ("Fewest answers", "Lower-middle", "Upper-middle", "Most answers")
    rows: list[dict[str, Any]] = []
    for index, (sum_row, normalized_row) in enumerate(
        zip(sumscore["quartiles"], normalized["quartiles"], strict=True)
    ):
        model_share = sum_row["model_count"] / 150
        series = (
            (1, "Models in group", model_share),
            (2, "SumScore placements", sum_row["top10_appearance_share"]),
            (3, "Top10Sum placements", normalized_row["top10_appearance_share"]),
        )
        for series_order, series_name, share in series:
            rows.append(
                {
                    "quartile_order": index + 1,
                    "series_order": series_order,
                    "inventory_group": labels[index],
                    "series": series_name,
                    "share": share,
                    "model_count": sum_row["model_count"],
                    "inventory_min": sum_row["inventory_min"],
                    "inventory_max": sum_row["inventory_max"],
                    "sumscore_top10_appearances": sum_row["top10_appearances"],
                    "top10sum_top10_appearances": normalized_row[
                        "top10_appearances"
                    ],
                }
            )
    return rows


def _cards() -> list[dict[str, Any]]:
    return [
        {
            "id": "ranking_quality",
            "description": (
                "Nested top-10 ranking quality. The comparison is Top10Sum minus "
                "the independently nested SumScore control."
            ),
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {
                    "label": "Top10Sum nDCG@10",
                    "field": "normalized_ndcg",
                    "format": "number",
                },
                {
                    "label": "vs SumScore",
                    "field": "ndcg_delta_vs_sumscore",
                    "format": "number",
                    "signed": True,
                },
            ],
        },
        {
            "id": "inventory_rho",
            "description": (
                "Absolute inventory-score association for the full-development "
                "deployment configuration. Lower is better; this gate failed."
            ),
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {
                    "label": "Inventory correlation",
                    "field": "inventory_rho",
                    "format": "number",
                },
                {
                    "label": "Maximum allowed",
                    "field": "inventory_rho_gate",
                    "format": "number",
                },
            ],
        },
        {
            "id": "largest_quartile",
            "description": (
                "Share of all 3,220 top-10 placements received by the 37 models with "
                "the most usable historical answers. Lower is better; this gate failed."
            ),
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {
                    "label": "Largest-quartile share",
                    "field": "largest_quartile_share",
                    "format": "percent",
                },
                {
                    "label": "Maximum allowed",
                    "field": "largest_quartile_gate",
                    "format": "percent",
                },
            ],
        },
        {
            "id": "gate_result",
            "description": (
                "The protocol required all four effectiveness and bias gates to pass."
            ),
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {
                    "label": "Gates passed",
                    "field": "passed_gate_count",
                    "format": "number",
                },
                {
                    "label": "Gates required",
                    "field": "gate_count",
                    "format": "number",
                },
            ],
        },
    ]


def _charts() -> list[dict[str, Any]]:
    return [
        {
            "id": "system_metric_comparison",
            "title": "Model-ranking scores by approach",
            "subtitle": (
                "Same 322 held-out queries and 150 candidate models; both metrics "
                "range from 0 to 1 and higher is better."
            ),
            "type": "horizontalBar",
            "dataset": "system_metrics",
            "sourceId": "system_metrics_sql",
            "intent": "comparison",
            "question": "Which approach ranks useful models earliest?",
            "rationale": (
                "Grouped bars compare both official ranking metrics on one common scale."
            ),
            "comparisonContext": {
                "denominator": "322 development queries",
                "grain": "one held-out model ranking per query",
                "normalization": "macro average from 0 to 1",
                "unit": "score",
            },
            "encodings": {
                "x": {"field": "system", "type": "nominal", "label": "Approach"},
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "label": "Score",
                    "format": "number",
                },
                "color": {"field": "metric", "type": "nominal", "label": "Metric"},
                "tooltip": [
                    {"field": "query_count", "type": "quantitative", "label": "Queries"},
                    {
                        "field": "evaluation_protocol",
                        "type": "text",
                        "label": "Evaluation",
                    },
                ],
            },
            "settings": {
                "orientation": "horizontal",
                "groupMode": "grouped",
                "categoryLabelPolicy": "wrap",
                "showValues": True,
            },
            "palette": {"kind": "categorical"},
            "legend": {"position": "bottom", "sort": "spec", "title": "Metric"},
            "labels": {"values": "auto"},
            "referenceLines": [
                {"axis": "y", "value": 0, "label": "Zero", "lineStyle": "solid"}
            ],
            "layout": "full",
            "maxRows": 10,
        },
        {
            "id": "query_outcomes_chart",
            "title": "Per-query nDCG@10 outcomes",
            "subtitle": (
                "Share of 322 paired queries; better and worse refer to Top10Sum "
                "relative to each named baseline."
            ),
            "type": "horizontalStackedBar100",
            "dataset": "query_outcomes",
            "sourceId": "query_outcomes_sql",
            "intent": "composition",
            "question": "How often did Top10Sum beat, tie, or lose to each baseline?",
            "rationale": (
                "Normalized stacks show query-level composition while tooltips retain "
                "the paired mean difference and interval."
            ),
            "comparisonContext": {
                "denominator": "322 paired queries per baseline",
                "grain": "one held-out query",
                "normalization": "each bar sums to 100%",
                "unit": "share of queries",
            },
            "encodings": {
                "x": {"field": "baseline", "type": "nominal", "label": "Baseline"},
                "y": {
                    "field": "outcome_count",
                    "type": "quantitative",
                    "label": "Queries",
                },
                "color": {
                    "field": "outcome",
                    "type": "nominal",
                    "label": "Top10Sum outcome",
                },
                "tooltip": [
                    {
                        "field": "outcome_share",
                        "type": "quantitative",
                        "label": "Query share",
                        "format": "percent",
                    },
                    {
                        "field": "mean_ndcg_difference",
                        "type": "quantitative",
                        "label": "Average nDCG difference",
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
            "settings": {"orientation": "horizontal", "groupMode": "stacked100"},
            "palette": {"kind": "categorical"},
            "legend": {
                "position": "bottom",
                "sort": "spec",
                "title": "Top10Sum outcome",
            },
            "labels": {"values": "auto"},
            "layout": "full",
            "maxRows": 9,
        },
        {
            "id": "bias_gate_comparison",
            "title": "Volume-dependence checks",
            "subtitle": (
                "Deployment behavior on all development queries; lower is better and "
                "Top10Sum had to be at or below each maximum."
            ),
            "type": "bar",
            "dataset": "bias_gates",
            "sourceId": "bias_gates_sql",
            "intent": "comparison",
            "question": "Did Top10Sum reduce either preregistered volume signal?",
            "rationale": (
                "Grouped bars put each observed value beside its baseline and locked maximum."
            ),
            "comparisonContext": {
                "denominator": "150 models and 3,220 top-10 placements",
                "grain": "one deployment behavior diagnostic",
                "normalization": "unitless values from 0 to 1",
                "unit": "value",
            },
            "encodings": {
                "x": {"field": "metric", "type": "nominal", "label": "Diagnostic"},
                "y": {
                    "field": "value",
                    "type": "quantitative",
                    "label": "Value",
                    "format": "number",
                },
                "color": {"field": "series", "type": "nominal", "label": "Series"},
                "tooltip": [
                    {
                        "field": "lower_is_better",
                        "type": "text",
                        "label": "Lower is better",
                    },
                    {
                        "field": "model_count",
                        "type": "quantitative",
                        "label": "Models",
                    },
                    {
                        "field": "top10_position_count",
                        "type": "quantitative",
                        "label": "Top-10 positions",
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
            "legend": {"position": "bottom", "sort": "spec", "title": "Series"},
            "labels": {"values": "all"},
            "referenceLines": [
                {"axis": "y", "value": 0, "label": "Zero", "lineStyle": "solid"}
            ],
            "layout": "full",
            "maxRows": 6,
        },
        {
            "id": "inventory_quartile_comparison",
            "title": "Top-10 placement share by response inventory",
            "subtitle": (
                "Models split into four deterministic groups by usable historical-answer "
                "count; balanced exposure would roughly follow each group's model share."
            ),
            "type": "bar",
            "dataset": "inventory_quartiles",
            "sourceId": "inventory_quartiles_sql",
            "intent": "comparison",
            "question": "Where did the extra Top10Sum concentration occur?",
            "rationale": (
                "Grouped shares compare representation under both response systems with "
                "the number of models available in each group."
            ),
            "comparisonContext": {
                "denominator": "150 models and 3,220 top-10 placements",
                "grain": "response-inventory quartile",
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
                "color": {"field": "series", "type": "nominal", "label": "Share type"},
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
                        "field": "sumscore_top10_appearances",
                        "type": "quantitative",
                        "label": "SumScore placements",
                    },
                    {
                        "field": "top10sum_top10_appearances",
                        "type": "quantitative",
                        "label": "Top10Sum placements",
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
            "maxRows": 12,
        },
    ]


def _blocks(results: dict[str, Any]) -> list[dict[str, Any]]:
    metrics = results["metrics"]
    comparisons = results["comparisons"]
    diagnostics = results["diagnostics"]
    decision = results["decision"]
    normalized = metrics["response_normalized_nested"]
    sumscore = metrics["response_sumscore_nested"]
    dense = metrics["dense_nested_reference"]
    bias = diagnostics["deployment_behavior"]["normalized"]
    sum_bias = diagnostics["deployment_behavior"]["sumscore"]
    sum_interval = comparisons["response_sumscore_nested"]["ndcg_at_10_delta"]
    tfidf_interval = comparisons["tfidf_fixed_k10"]["ndcg_at_10_delta"]
    gate_rows = []
    labels = {
        "ndcg_ci_above_tfidf": "Clearly better than TF-IDF",
        "ndcg_within_0_010_of_sumscore": "No more than 0.010 nDCG below SumScore",
        "deployment_inventory_rho_at_most_0_7388": "Inventory correlation ≤ 0.7388",
        "deployment_largest_quartile_share_at_most_0_411": (
            "Largest-quartile share ≤ 41.1%"
        ),
    }
    for key, value in decision["gates"].items():
        gate_rows.append(
            f"| {labels[key]} | {'Pass' if value['passed'] else 'Fail'} | "
            f"{_format_gate_observed(key, value['observed'])} |"
        )
    gate_table = "\n".join(gate_rows)

    return [
        {
            "id": "title",
            "type": "markdown",
            "body": (
                "# Capping past-answer evidence raised the ranking point estimate—but "
                "failed the volume-risk gates"
            ),
        },
        {
            "id": "executive_summary",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## Executive Summary\n\n"
                f"- **Keep SumScore as the response-search control.** Top10Sum raised "
                f"nested nDCG@10 from **{sumscore['ndcg_at_10']:.4f}** to "
                f"**{normalized['ndcg_at_10']:.4f}**, but the locked rule required all "
                "four effectiveness and volume-bias gates to pass; only two passed.\n\n"
                f"- **The ranking signal is real enough to study, but not decisive versus "
                f"SumScore.** The nDCG point difference was **+{sum_interval['mean_difference']:.4f}** "
                f"with a paired 95% interval of **[{sum_interval['lower']:.4f}, "
                f"{sum_interval['upper']:.4f}]**. The interval includes zero.\n\n"
                f"- **Both deployment-volume diagnostics worsened, so both volume gates "
                f"failed.** Inventory-score "
                f"correlation rose from **{sum_bias['inventory_vs_aggregation_score_spearman']:.3f}** "
                f"to **{bias['inventory_vs_aggregation_score_spearman']:.3f}**, and the "
                f"largest inventory group received **{bias['largest_inventory_quartile_top10_share']:.1%}** "
                "of top-10 placements instead of falling to 41.1% or less.\n\n"
                f"- **Dense query search remains the best tested baseline.** It scored "
                f"**{dense['ndcg_at_10']:.4f} nDCG@10 / {dense['mrr']:.4f} MRR**, versus "
                f"Top10Sum's **{normalized['ndcg_at_10']:.4f} / {normalized['mrr']:.4f}**."
            ),
        },
        {
            "id": "task_primer",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## What is this experiment trying to do?\n\n"
                "For each new question, the system must rank the same **150 candidate "
                "language models**. A good ranking puts models judged useful for that "
                "question near the top. The evaluation contains **322 development queries**.\n\n"
                "A **historical response** here means one organizer-provided answer from "
                "one model for an earlier discovery query. It does **not** mean the last "
                "assistant message in a chat. Response search finds earlier answers related "
                "to the new question, then credits the models that produced them.\n\n"
                "The comparison chart also includes **dense semantic query search**, "
                "**TF-IDF keyword query search**, and a **query-blind global prior** that "
                "ranks models only by their other development grades."
            ),
        },
        {
            "id": "metric_guide",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## How to read the scores\n\n"
                "- **nDCG@10** measures the quality and order of the first ten models. "
                "Higher is better; 1 is ideal. This is the primary metric.\n"
                "- **MRR** measures how early the first useful model appears. First place "
                "scores 1, second 0.5, and fifth 0.2. Higher is better.\n"
                "- A **paired 95% interval** describes uncertainty in the average difference "
                "over the same queries. If it crosses zero, the observed direction is not "
                "decisive at that interval level."
            ),
        },
        {"id": "headline_metrics", "type": "metric-strip", "cardIds": [card["id"] for card in _cards()]},
        {
            "id": "effectiveness_finding",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## Top10Sum clearly beat TF-IDF; its SumScore lift was uncertain; dense still led\n\n"
                f"Top10Sum reached **{normalized['ndcg_at_10']:.4f} nDCG@10** and "
                f"**{normalized['mrr']:.4f} MRR**. It was clearly above TF-IDF: the paired "
                f"nDCG difference was **+{tfidf_interval['mean_difference']:.4f}**, with a "
                f"95% interval of **[{tfidf_interval['lower']:.4f}, "
                f"{tfidf_interval['upper']:.4f}]**. Dense query search remained higher by "
                f"about **{dense['ndcg_at_10'] - normalized['ndcg_at_10']:.4f} nDCG**."
            ),
        },
        {
            "id": "system_chart",
            "type": "chart",
            "chartId": "system_metric_comparison",
            "layout": {"columns": 12, "minColumns": 8},
        },
        {
            "id": "aggregation_explainer",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## What changed from SumScore?\n\n"
                "**SumScore** adds every positive retrieved answer score for a model. A "
                "model with many searchable answers has more chances to accumulate points.\n\n"
                "Before looking at these results, the experiment locked six alternatives:\n\n"
                "1. **BestHit:** keep only the strongest matching answer.\n"
                "2. **Top3Sum / Top10Sum:** add only each model's own strongest 3 or 10 "
                "matches, using every available match when fewer than the cap exist.\n"
                "3. **Top3Mean / Top10Mean:** average those strongest matches.\n"
                "4. **MeanAll:** average every retrieved match.\n\n"
                "Nested validation selected **Top10Sum in all 322 outer folds**. That cap "
                "limits how many answers directly contribute to one model's score, but it "
                "does not equalize each model's chance of having ten strong matches."
            ),
        },
        {
            "id": "pipeline_explainer",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## How leakage was prevented\n\n"
                "Each outer query was hidden. The system tested the 144 normalized setups "
                "only on the other 321 queries, selected the highest inner nDCG@10, and then "
                "ranked the hidden query once. The hidden query's grades never chose its "
                "own setup. SumScore repeated the same process separately across its 24 "
                "setups.\n\n"
                "`Seal 1 query → choose on the other 321 → rank 150 models → reveal grades → repeat 322 times`\n\n"
                "The selected search used **BM25+RM3** over query-plus-response text. BM25 "
                "is lexical term matching. **RM3 is a query-expansion step**: an initial "
                "search suggests related terms and the search is rerun. RM3 is not a "
                "language model."
            ),
        },
        {
            "id": "paired_outcomes_finding",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## The average lift over SumScore came from a minority of changed queries\n\n"
                "Top10Sum beat SumScore on **77** queries, tied on **181**, and lost on "
                "**64** for nDCG@10. The point estimate favored Top10Sum, but the paired "
                f"interval **[{sum_interval['lower']:.4f}, {sum_interval['upper']:.4f}]** "
                "still includes no difference."
            ),
        },
        {
            "id": "query_outcomes",
            "type": "chart",
            "chartId": "query_outcomes_chart",
            "layout": {"columns": 12, "minColumns": 8},
        },
        {
            "id": "bias_finding",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## The cap did not solve response-volume dependence\n\n"
                f"The locked inventory-correlation maximum was **0.7388**. Top10Sum "
                f"measured **{bias['inventory_vs_aggregation_score_spearman']:.4f}**, "
                f"worse than SumScore's **{sum_bias['inventory_vs_aggregation_score_spearman']:.4f}**. "
                f"The largest inventory quartile also received **{bias['largest_inventory_quartile_top10_share']:.1%}** "
                f"of placements, above SumScore's **{sum_bias['largest_inventory_quartile_top10_share']:.1%}** "
                "and the 41.1% maximum.\n\n"
                "A plausible interpretation is that models with larger inventories still "
                "have more opportunities to supply ten strong matches. This is an inference, "
                "not proof of causation."
            ),
        },
        {
            "id": "bias_gate_chart",
            "type": "chart",
            "chartId": "bias_gate_comparison",
            "layout": {"columns": 12, "minColumns": 8},
        },
        {
            "id": "inventory_chart",
            "type": "chart",
            "chartId": "inventory_quartile_comparison",
            "layout": {"columns": 12, "minColumns": 8},
        },
        {
            "id": "decision_table",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## The preregistered decision\n\n"
                "| Required gate | Result | Observed |\n"
                "|---|---:|---:|\n"
                f"{gate_table}\n\n"
                "The rule required **all four gates**. Because the two volume gates failed, "
                "the correct decision is to retain SumScore as the response-retrieval "
                "control and stop this aggregation branch—even though Top10Sum's point "
                "effectiveness was higher."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## What should happen next?\n\n"
                "1. **Do not promote Top10Sum as the new response baseline.** Keep SumScore "
                "as the stable response-search control.\n"
                "2. **Keep dense query retrieval as the strongest overall baseline.** It "
                "still leads both ranking metrics by a material margin.\n"
                "3. **Do not add log probabilities or fusion as a post-hoc continuation of "
                "this branch.** The locked stop rule says to stop here. Any future response-"
                "retrieval approach needs a new question, a newly reviewed protocol, and a "
                "mechanism aimed at equalizing retrieval opportunity—not merely score totals."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- Does the inventory relationship reflect response quantity, model quality, "
                "or both? The current association cannot separate them.\n"
                "- Would a future model-balanced retrieval design reduce opportunity "
                "differences without discarding useful semantic evidence? This would require "
                "a new preregistration, not a tweak to the finished run.\n"
                "- Why does dense query search remain substantially stronger, and can its "
                "errors be characterized without tuning on the held-out answers?"
            ),
        },
        {
            "id": "technical_details",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## Reproducibility and caveats\n\n"
                "- The CPU-only experiment used one shared BM25/RM3 retrieval pass for all "
                "seven aggregations; no GPU and no log probabilities were used.\n"
                "- A full second run reproduced every decision-relevant file and all score-"
                "tensor hashes exactly; only elapsed time differed.\n"
                "- The SumScore control reproduced Baseline 3 exactly, including rankings, "
                "metrics, selection counts, and deployment configuration.\n"
                "- Inventory diagnostics measure association, not causal unfairness. Stronger "
                "models may legitimately have more historical answers.\n"
                "- Full-development deployment diagnostics reuse the development data and are "
                "not held-out effectiveness estimates; they serve only as like-for-like "
                "behavior gates. Nested outer-fold metrics support effectiveness claims.\n"
                "- Paired bootstrap intervals condition on the completed nested selections; "
                "they do not rerun candidate selection inside each resample.\n"
                "- Secondary fold-by-fold evidence was mixed: mean inventory correlation "
                "fell slightly from 0.5559 to 0.5464, while largest-quartile placement share "
                "rose from 46.8% to 48.3%. Both preregistered deployment-volume diagnostics "
                "worsened, so both gates failed.\n"
                "- Neither system's stitched top ten used zero-score models or score ties "
                "resolved by the qrel prior; both fractions were 0%."
            ),
        },
    ]


def _format_gate_observed(key: str, value: float) -> str:
    if key == "deployment_largest_quartile_share_at_most_0_411":
        return f"{value:.1%}"
    if key == "ndcg_ci_above_tfidf":
        return f"CI lower bound {value:+.4f}"
    if key == "ndcg_within_0_010_of_sumscore":
        return f"nDCG difference {value:+.4f}"
    return f"{value:.4f}"


def _sources(results_path: Path, generated_at: str) -> list[dict[str, Any]]:
    result_path = str(results_path.relative_to(ROOT))
    return [
        {
            "id": "reviewed_results",
            "label": "Reviewed response-volume normalization result snapshot",
            "path": result_path,
        },
        {
            "id": "headline_sql",
            "label": "Headline normalization metrics",
            "path": result_path,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT normalized_ndcg, ndcg_delta_vs_sumscore, normalized_mrr, "
                    "mrr_delta_vs_sumscore, inventory_rho, inventory_rho_gate, "
                    "largest_quartile_share, largest_quartile_gate, passed_gate_count, "
                    "gate_count FROM headline_metrics"
                ),
                "description": (
                    "Selects the headline effectiveness, volume, and gate values from an "
                    "in-memory table loaded from the reviewed result snapshot."
                ),
                "executed_at": generated_at,
                "tables_used": ["headline_metrics"],
                "filters": [
                    "322 official development queries",
                    "150 candidate models",
                    "strict nested leave-one-query-out",
                    "advisor-approved grid locked before effectiveness scoring",
                ],
                "metric_definitions": [
                    "nDCG@10: macro top-10 ranking quality over equally weighted queries.",
                    "MRR: macro reciprocal rank of the first positively judged model.",
                    "Inventory correlation: Spearman association between retained response count and aggregate retrieval score.",
                    "Largest-quartile share: share of 3,220 deployment top-10 placements received by the 37 models with the largest basic-filter response inventories.",
                ],
            },
        },
        {
            "id": "system_metrics_sql",
            "label": "Five-system effectiveness comparison",
            "path": result_path,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT system_order, metric_order, system, metric, value, "
                    "query_count, evaluation_protocol, higher_is_better "
                    "FROM system_metrics ORDER BY system_order, metric_order"
                ),
                "description": (
                    "Selects held-out nDCG@10 and MRR for the five retained systems."
                ),
                "executed_at": generated_at,
                "tables_used": ["system_metrics"],
                "filters": ["322 official development queries", "strict nested LOO"],
                "metric_definitions": [
                    "nDCG@10: macro top-10 ranking quality over equally weighted queries.",
                    "MRR: macro reciprocal rank of the first positively judged model.",
                ],
            },
        },
        {
            "id": "query_outcomes_sql",
            "label": "Paired per-query nDCG outcomes",
            "path": result_path,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT comparison_order, outcome_order, baseline, outcome, "
                    "outcome_count, outcome_share, query_count, mean_ndcg_difference, "
                    "ci_lower, ci_upper FROM query_outcomes "
                    "ORDER BY comparison_order, outcome_order"
                ),
                "description": (
                    "Selects paired Top10Sum win, tie, and loss counts with bootstrap context."
                ),
                "executed_at": generated_at,
                "tables_used": ["query_outcomes"],
                "filters": ["322 paired queries", "nDCG@10", "10,000 resamples"],
                "metric_definitions": [
                    "Mean difference and confidence bounds are Top10Sum minus the named baseline."
                ],
            },
        },
        {
            "id": "bias_gates_sql",
            "label": "Deployment volume-gate comparison",
            "path": result_path,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT metric_order, series_order, metric, series, value, role, "
                    "lower_is_better, top10_position_count, model_count "
                    "FROM bias_gates ORDER BY metric_order, series_order"
                ),
                "description": (
                    "Selects SumScore, Top10Sum, and maximum-allowed values for both "
                    "preregistered deployment-volume diagnostics."
                ),
                "executed_at": generated_at,
                "tables_used": ["bias_gates"],
                "filters": ["150 models", "3,220 deployment top-10 placements"],
                "metric_definitions": [
                    "Inventory-score correlation is Spearman rho; lower absolute magnitude is better.",
                    "Largest-quartile share is the placement share received by the 37 models with most retained responses.",
                ],
            },
        },
        {
            "id": "inventory_quartiles_sql",
            "label": "Response-inventory quartile representation",
            "path": result_path,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT quartile_order, series_order, inventory_group, series, share, "
                    "model_count, inventory_min, inventory_max, "
                    "sumscore_top10_appearances, top10sum_top10_appearances "
                    "FROM inventory_quartiles ORDER BY quartile_order, series_order"
                ),
                "description": (
                    "Selects model and top-10 placement shares for deterministic response-"
                    "inventory quartiles."
                ),
                "executed_at": generated_at,
                "tables_used": ["inventory_quartiles"],
                "filters": ["basic response filter", "150 models", "3,220 placements"],
                "metric_definitions": [
                    "Each placement series sums to 100%; model shares sum to 100%."
                ],
            },
        },
        {
            "id": "locked_protocol",
            "label": "Advisor-approved normalization preregistration",
            "path": "protocols/trec_2026_response_volume_normalization.md",
        },
        {
            "id": "primary_summary",
            "label": "Primary official experiment summary",
            "path": "artifacts/response_volume_normalization/summary.json",
        },
        {
            "id": "primary_validation",
            "label": "Complete-run and provenance validation receipt",
            "path": "artifacts/response_volume_normalization/validation.json",
        },
        {
            "id": "deterministic_rerun",
            "label": "Independent full deterministic rerun",
            "path": "artifacts/response_volume_normalization_rerun/summary.json",
        },
        {
            "id": "normalization_pipeline",
            "label": "Tracked nested normalization implementation",
            "path": "trec_mlm/response_normalization.py",
        },
        {
            "id": "response_retriever",
            "label": "Tracked shared BM25/RM3 response retriever",
            "path": "trec_mlm/response_retrieval.py",
        },
        {
            "id": "development_qrels",
            "label": "Official TREC development relevance judgments",
            "path": "data/dev/llm_dev_qrels.txt",
        },
        {
            "id": "discovery_responses",
            "label": "Official TREC historical discovery responses",
            "path": "data/dev/llm_discovery_data.json",
        },
    ]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
