"""Build the reviewed novice report for source-query-first answer alignment."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import csv
from dataclasses import asdict
from datetime import UTC, datetime
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any, Mapping

from trec_mlm.metrics import EvaluationResult, evaluate_rankings, paired_bootstrap_ci
from trec_mlm.query_knn import load_complete_qrels
from trec_mlm.response_experiment import load_dense_per_query_reference
from trec_mlm.runio import rankings_from_run, validate_trec_run


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/source_query_first"
RERUN_DIR = ROOT / "artifacts/source_query_first_rerun"
DEFAULT_RESULTS = ROOT / "reports/trec_2026_source_query_first_results.json"
DEFAULT_ARTIFACT = ROOT / "reports/trec_2026_source_query_first.artifact.json"
QRELS = ROOT / "data/dev/llm_dev_qrels.txt"
DENSE_PER_QUERY = ROOT / "reports/trec_2026_dense_query_knn_per_query.tsv"
RESPONSE_VOLUME_RESULTS = (
    ROOT / "reports/trec_2026_response_volume_normalization_results.json"
)

PRIMARY = "source_alignment_nested"
DIAGNOSTIC = "target_response_same_k"
SUMSCORE = "response_sumscore_nested"
TOP10SUM = "response_top10sum_nested"
DENSE = "dense_nested_reference"
TFIDF = "tfidf_fixed_k10"
PRIOR = "global_prior"

HEADLINE_SYSTEMS = (PRIMARY, DENSE, SUMSCORE, TOP10SUM, TFIDF, PRIOR)
BASELINE_ORDER = (SUMSCORE, TOP10SUM, DENSE, TFIDF, PRIOR)
CANDIDATE_ORDER = (
    "source_alignment__k10",
    "source_alignment__k50",
    "source_alignment__k100",
)
RUN_FILES = {
    PRIMARY: "source_alignment_run.trec",
    DIAGNOSTIC: "target_response_diagnostic.trec",
    SUMSCORE: "sumscore_control.trec",
    TOP10SUM: "top10sum_control.trec",
    TFIDF: "tfidf_control.trec",
    PRIOR: "global_prior_control.trec",
}
DECISION_RELEVANT_FILES = (
    "config.json",
    "source_score_metadata.json",
    "diagnostics.json",
    "selected_sources.tsv",
    "inner_scores.tsv",
    "selections.tsv",
    "predictions.tsv",
    "per_query.tsv",
    "query_diagnostics.tsv",
    *RUN_FILES.values(),
)
QUERY_DIAGNOSTIC_COLUMNS = (
    "query_id",
    "selected_configuration",
    "selected_k",
    "selected_inner_ndcg_at_10",
    "selected_inner_mrr",
    "inner_ndcg_winner_margin",
    "invalid_cell_count",
    "invalid_cell_share",
    "primary_zero_score_model_count",
    "diagnostic_zero_score_model_count",
    "primary_top10_prior_tie_position_count",
    "primary_top10_zero_score_position_count",
    "source_weight_sum",
    "positive_retrieval_count",
    "exact_mask_canonical_count",
    "exact_mask_source_id_count",
)
QUERY_DIAGNOSTIC_TSV_COLUMNS = (
    *QUERY_DIAGNOSTIC_COLUMNS,
    "validity_reason_counts_json",
)
GATE_KEYS = (
    "ndcg_ci_above_tfidf",
    "ndcg_within_0_010_of_sumscore",
    "deployment_inventory_rho_at_most_0_7388",
    "deployment_largest_quartile_share_at_most_0_411",
)
INVENTORY_RHO_GATE = 0.7388
LARGEST_QUARTILE_GATE = 0.411
MAX_NDCG_DROP_FROM_SUMSCORE = 0.010
EXPECTED_DISCOVERY_SOURCE_QUERIES = 7_759
EXPECTED_CANONICAL_SOURCE_QUERIES = 7_693
EXPECTED_DUPLICATE_KEY_GROUPS = 58
EXPECTED_NONCANONICAL_SOURCE_IDS = 66
EXPECTED_RAW_RESPONSE_CELLS = 1_163_850
EXPECTED_EXACT_IDK_CELLS = 981_050
EXPECTED_PREFIX_REFUSAL_CELLS = 2
EXPECTED_ELIGIBLE_RESPONSE_CELLS = 182_798


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
    validation = _read_json(args.artifact_dir / "validation.json")
    rerun_summary = _read_json(args.rerun_dir / "summary.json")
    rerun_diagnostics = _read_json(args.rerun_dir / "diagnostics.json")
    rerun_validation = _read_json(args.rerun_dir / "validation.json")

    results = _reviewed_results(
        summary=summary,
        diagnostics=diagnostics,
        validation=validation,
        rerun_summary=rerun_summary,
        rerun_diagnostics=rerun_diagnostics,
        rerun_validation=rerun_validation,
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
    validation: dict[str, Any],
    rerun_summary: dict[str, Any],
    rerun_diagnostics: dict[str, Any],
    rerun_validation: dict[str, Any],
    artifact_dir: Path,
    rerun_dir: Path,
) -> dict[str, Any]:
    _validate_shape(summary, diagnostics, validation, artifact_dir)
    _validate_shape(
        rerun_summary,
        rerun_diagnostics,
        rerun_validation,
        rerun_dir,
    )
    deterministic = _validate_determinism(
        summary=summary,
        diagnostics=diagnostics,
        validation=validation,
        rerun_summary=rerun_summary,
        rerun_diagnostics=rerun_diagnostics,
        rerun_validation=rerun_validation,
        artifact_dir=artifact_dir,
        rerun_dir=rerun_dir,
    )
    independent = _recompute_effectiveness(summary, artifact_dir)
    independently_recomputed_decision = _validate_decision(summary, diagnostics)
    _validate_evidence_hashes(validation, artifact_dir)
    _validate_evidence_hashes(rerun_validation, rerun_dir)
    response_volume_results = _read_json(RESPONSE_VOLUME_RESULTS)
    _validate_tracked_response_controls(summary, response_volume_results)

    snapshot_as_of = (
        datetime.fromtimestamp(
            (artifact_dir / "summary.json").stat().st_mtime,
            tz=UTC,
        )
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )
    gate_count = sum(
        int(value["passed"]) for value in summary["decision"]["gates"].values()
    )
    return {
        "snapshot_as_of": snapshot_as_of,
        "question": (
            "Can source-query-first lexical answer alignment rank candidate models while "
            "giving every model the same historical-query opportunities?"
        ),
        "audience": "product stakeholders and first-time topic readers",
        "recommendation": {
            "experiment": summary["decision"]["recommendation"],
            "all_four_gates_passed": summary["decision"]["all_four_gates_passed"],
            "passed_gate_count": gate_count,
            "owner_decision_still_required": True,
            "reason": _decision_reason(summary["decision"]),
        },
        "protocol": summary["protocol"],
        "data": summary["data"],
        "candidate_grid": summary.get("candidate_grid", list(CANDIDATE_ORDER)),
        "metrics": summary["metrics"],
        "comparisons": summary["comparisons"],
        "selection": summary["selection"],
        "deployment": summary["deployment"],
        "diagnostics": diagnostics,
        "decision": summary["decision"],
        "controls": summary["controls"],
        "retrieval_reproducibility": summary["retrieval_reproducibility"],
        "runtime": summary["runtime"],
        "reference_volume_diagnostics": {
            "sumscore": response_volume_results["diagnostics"]["deployment_behavior"][
                "sumscore"
            ],
            "top10sum": response_volume_results["diagnostics"]["deployment_behavior"][
                "normalized"
            ],
        },
        "validation": {
            "run_receipt": validation,
            "deterministic_rerun": deterministic,
            "independent_effectiveness_recalculation": independent,
            "independent_decision_recalculation": independently_recomputed_decision,
            "share_status": "ready_for_owner_review",
        },
        "definitions": _definitions(),
        "caveats": _caveats(),
        "evidence_inputs": {
            "summary": str(artifact_dir / "summary.json"),
            "diagnostics": str(artifact_dir / "diagnostics.json"),
            "query_diagnostics": str(artifact_dir / "query_diagnostics.tsv"),
            "validation": str(artifact_dir / "validation.json"),
            "rerun_summary": str(rerun_dir / "summary.json"),
            "rerun_query_diagnostics": str(rerun_dir / "query_diagnostics.tsv"),
            "qrels": str(QRELS),
            "dense_per_query": str(DENSE_PER_QUERY),
        },
        "report_notes": _report_notes(),
    }


def _validate_shape(
    summary: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    validation: Mapping[str, Any],
    artifact_dir: Path,
) -> None:
    if summary["data"]["query_count"] != 322:
        raise ValueError("expected 322 official development queries")
    if summary["data"]["candidate_model_count"] != 150:
        raise ValueError("expected the 150-model universe")
    if set(summary["metrics"]) != {*HEADLINE_SYSTEMS, DIAGNOSTIC}:
        raise ValueError(
            "summary does not contain the exact six systems plus diagnostic"
        )
    candidate_grid = summary["candidate_grid"]
    if tuple(row["key"] for row in candidate_grid) != CANDIDATE_ORDER:
        raise ValueError("candidate grid does not match the locked declaration order")
    if tuple(row["k"] for row in candidate_grid) != (10, 50, 100):
        raise ValueError("candidate K values differ from the locked grid")
    if set(summary["comparisons"]) != set(BASELINE_ORDER):
        raise ValueError("summary does not contain the five frozen paired comparisons")
    if tuple(summary["decision"]["gates"]) != GATE_KEYS:
        raise ValueError("decision gates or declaration order differ from the lock")
    if set(summary["selection"]["configuration_frequency"]) - set(CANDIDATE_ORDER):
        raise ValueError("selection contains an undeclared configuration")
    if sum(summary["selection"]["configuration_frequency"].values()) != 322:
        raise ValueError("selection frequencies do not cover all outer queries")

    protocol = summary["protocol"]
    required_protocol = {
        "base_revision": "81d4aa50812864133f7066ee5b698fd48ea20503",
        "preregistration_revision": "459e545",
        "outer_split": "strict_leave_one_query_out",
        "inner_split": "strict_leave_one_query_out_within_outer_training_rows",
        "selection_metric": "inner_macro_ndcg_at_10",
        "mrr_used_for_selection": False,
        "run_classification": "internal_automatic",
        "retrieval_and_signal_tensors_are_qrel_free": True,
        "target_response_role": "same_selected_k_diagnostic_only",
        "grid_selection_provenance": (
            "advisor_approved_source_query_first_protocol_locked_at_459e545"
        ),
    }
    for key, expected in required_protocol.items():
        if protocol.get(key) != expected:
            raise ValueError(f"protocol field {key!r} differs from the lock")

    if diagnostics != summary["diagnostics"]:
        raise ValueError("standalone diagnostics disagree with the combined summary")
    required_diagnostic_sections = {
        "opportunities",
        "selection",
        "deployment_behavior",
        "outer_fold_behavior",
        "target_response_diagnostic",
    }
    if not required_diagnostic_sections.issubset(diagnostics):
        raise ValueError("diagnostics are missing a locked evidence section")
    _validate_opportunity_diagnostics(diagnostics["opportunities"])
    _validate_query_diagnostics(summary, diagnostics, artifact_dir)
    selected_cells = 150 * sum(
        int(configuration.rsplit("k", maxsplit=1)[1]) * count
        for configuration, count in summary["selection"][
            "configuration_frequency"
        ].items()
    )
    if (
        sum(diagnostics["opportunities"]["selected_validity_reason_counts"].values())
        != selected_cells
    ):
        raise ValueError("selected response validity counts have the wrong denominator")
    deployment = summary["deployment"]
    if deployment["development_score_is_non_nested_and_optimistic"] is not True:
        raise ValueError("deployment effectiveness was not labeled optimistic")
    if deployment["behavior"] != diagnostics["deployment_behavior"]:
        raise ValueError("deployment summary disagrees with deployment diagnostics")
    if (
        diagnostics["deployment_behavior"]["configuration"]
        != deployment["configuration"]
    ):
        raise ValueError("deployment configuration differs across evidence files")
    if diagnostics["deployment_behavior"]["inventory_definition"] != (
        "frozen_pr8_basic_retained_response_count"
    ):
        raise ValueError(
            "deployment gates did not use the frozen PR #8 BASIC inventory"
        )
    if diagnostics["outer_fold_behavior"]["top10_position_count"] != 3_220:
        raise ValueError("outer-fold behavior does not cover every top-10 position")
    target_response = diagnostics["target_response_diagnostic"]
    if target_response["selection_role"] != (
        "same_primary_selected_k_nonselecting_non_gating"
    ):
        raise ValueError("target-to-response diagnostic role differs from the lock")
    if target_response["uses_primary_selected_k"] is not True:
        raise ValueError("target-to-response diagnostic did not use primary-selected K")
    if target_response["can_change_primary_selection"] is not False:
        raise ValueError("target-to-response diagnostic could change primary selection")
    if summary["decision"]["owner_decision_still_required"] is not True:
        raise ValueError("decision incorrectly bypasses the required owner decision")
    runtime = summary["runtime"]
    if _require_number(runtime, "experiment_seconds_through_summary") <= 0.0:
        raise ValueError("experiment runtime must be positive")
    if _require_number(runtime, "cpu_seconds_through_summary") <= 0.0:
        raise ValueError("experiment CPU time must be positive")
    if _require_number(runtime, "peak_rss_bytes_at_summary") <= 0.0:
        raise ValueError("experiment peak RSS must be positive")

    inputs = validation["inputs"]
    if inputs["query_count"] != 322 or inputs["candidate_model_count"] != 150:
        raise ValueError("validation input shape differs from the official task")
    if inputs["complete_qrel_matrix"] is not True:
        raise ValueError("validation receipt does not certify the complete qrel matrix")
    if inputs["source_candidate_count"] != 3:
        raise ValueError("validation receipt does not cover the three-candidate grid")
    receipt_protocol = validation["protocol"]
    if receipt_protocol["outer_prediction_count"] != 322:
        raise ValueError("validation receipt does not cover every outer prediction")
    if receipt_protocol["inner_targets_per_outer"] != 321:
        raise ValueError("validation receipt has the wrong nested inner denominator")
    if receipt_protocol["primary_score_tensor_shape"] != [3, 322, 150]:
        raise ValueError("validation receipt has the wrong primary score tensor shape")
    if receipt_protocol["diagnostic_score_tensor_shape"] != [3, 322, 150]:
        raise ValueError("validation receipt has the wrong diagnostic tensor shape")
    if receipt_protocol["candidate_keys"] != list(CANDIDATE_ORDER):
        raise ValueError("validation receipt has the wrong candidate declaration order")
    if receipt_protocol["outer_qrel_excluded_from_selection"] is not True:
        raise ValueError("validation receipt does not establish outer-qrel isolation")
    if receipt_protocol["target_response_excluded_from_selection"] is not True:
        raise ValueError("diagnostic response relevance entered primary selection")
    if receipt_protocol["target_response_uses_primary_selected_k"] is not True:
        raise ValueError("diagnostic response relevance did not use primary-selected K")
    if receipt_protocol["four_gate_count"] != 4:
        raise ValueError("validation receipt does not cover the four locked gates")
    if receipt_protocol["sumscore_selected_independently"] is not True:
        raise ValueError("SumScore was not selected as an independent control")
    if receipt_protocol["top10sum_selected_independently"] is not True:
        raise ValueError("Top10Sum was not selected as an independent control")
    reproducibility = summary["retrieval_reproducibility"]
    if (
        receipt_protocol["primary_score_tensor_sha256"]
        != reproducibility["primary_model_score_tensor_sha256"]
    ):
        raise ValueError("primary score tensor hashes disagree")
    if (
        receipt_protocol["diagnostic_score_tensor_sha256"]
        != reproducibility["diagnostic_model_score_tensor_sha256"]
    ):
        raise ValueError("diagnostic score tensor hashes disagree")

    expected_run_systems = {*HEADLINE_SYSTEMS, DIAGNOSTIC} - {DENSE}
    if set(validation["runs"]) != expected_run_systems:
        raise ValueError("validation receipt does not contain the exact written runs")
    if not all(receipt["valid"] for receipt in validation["runs"].values()):
        raise ValueError("one or more complete TREC runs failed validation")
    for receipt in validation["runs"].values():
        if receipt["query_count"] != 322 or receipt["candidate_count"] != 150:
            raise ValueError("one or more run receipts has the wrong task shape")
        if receipt["row_count"] != 48_300:
            raise ValueError("one or more run receipts is incomplete")

    validation_controls = dict(validation["controls"])
    run_parity = validation_controls.pop("run_parity")
    if validation_controls != summary["controls"]:
        raise ValueError("validation controls disagree with the combined summary")
    if validation_controls["official_reference_applicable"] is not True:
        raise ValueError("official merged control references were not applied")
    for control in (SUMSCORE, TOP10SUM):
        reference = validation_controls[control]
        if reference["metrics_match"] is not True:
            raise ValueError(f"{control} metrics do not reproduce the merged reference")
        if reference["selection_match"] is not True:
            raise ValueError(
                f"{control} selection does not reproduce the merged reference"
            )
    if run_parity["official_reference_applicable"] is not True:
        raise ValueError("official exact control-run parity was not applied")
    for control in (SUMSCORE, TOP10SUM):
        if run_parity[control]["identical_rankings_scores_and_run_id"] is not True:
            raise ValueError(f"{control} run does not reproduce merged PR #8 bytes")
    determinism = validation["determinism"]
    if (
        determinism["source_signal_tensor_sha256"]
        != reproducibility["signal_tensor_sha256"]
    ):
        raise ValueError("source signal tensor hashes disagree")
    if (
        determinism["source_score_tensor_sha256"]
        != reproducibility["score_tensor_sha256"]
    ):
        raise ValueError("source score tensor hashes disagree")
    if validation["execution_provenance"]["git_worktree_status"] != "":
        raise ValueError("official run did not record a clean worktree")


def _validate_opportunity_diagnostics(opportunities: Mapping[str, Any]) -> None:
    if opportunities["candidate_k_values"] != [10, 50, 100]:
        raise ValueError("opportunity diagnostics have the wrong K grid")
    if opportunities["every_model_receives_same_k"] is not True:
        raise ValueError("not every model received the same source-query opportunities")

    canonical = opportunities["canonical_audit"]
    expected_canonical = {
        "source_query_id_count": EXPECTED_DISCOVERY_SOURCE_QUERIES,
        "nonempty_analyzer_key_count": EXPECTED_CANONICAL_SOURCE_QUERIES,
        "canonical_source_query_count": EXPECTED_CANONICAL_SOURCE_QUERIES,
        "duplicate_key_group_count": EXPECTED_DUPLICATE_KEY_GROUPS,
        "noncanonical_source_id_count": EXPECTED_NONCANONICAL_SOURCE_IDS,
        "empty_analyzer_key_source_id_count": 0,
    }
    if canonical != expected_canonical:
        raise ValueError("canonical source-query audit differs from the locked counts")

    audit = opportunities["filter_audit"]
    expected_filter = {
        "input_response_cells": EXPECTED_RAW_RESPONSE_CELLS,
        "eligible_response_cells": EXPECTED_ELIGIBLE_RESPONSE_CELLS,
        "zero_null": 0,
        "zero_blank": 0,
        "zero_exact_idk": EXPECTED_EXACT_IDK_CELLS,
        "zero_prefix_refusal": EXPECTED_PREFIX_REFUSAL_CELLS,
        "rule_name": "basic_plus_prefix",
    }
    if audit != expected_filter:
        raise ValueError("response opportunity audit differs from the locked counts")

    weights = opportunities["source_weight_sum"]
    if not all(
        math.isclose(float(weights[key]), 1.0, rel_tol=0.0, abs_tol=5e-12)
        for key in ("min", "max", "mean")
    ):
        raise ValueError(
            "one or more target source-query weight vectors do not sum to one"
        )
    if opportunities["positive_retrieval_count"]["min"] < 100:
        raise ValueError(
            "one or more targets has fewer than 100 positive source queries"
        )
    if opportunities["valid_opportunity_count"]["count"] != 3 * 322 * 150:
        raise ValueError("valid-opportunity diagnostics have the wrong tensor size")


def _validate_query_diagnostics(
    summary: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    artifact_dir: Path,
) -> None:
    expected_rows = diagnostics.get("per_query_selected_k")
    if not isinstance(expected_rows, list) or len(expected_rows) != 322:
        raise ValueError("per-query selected-K diagnostics must contain 322 rows")
    expected_ids = [str(row["query_id"]) for row in expected_rows]
    if expected_ids != sorted(expected_ids) or len(set(expected_ids)) != 322:
        raise ValueError("per-query selected-K diagnostics have invalid query IDs")

    path = artifact_dir / "query_diagnostics.tsv"
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if tuple(reader.fieldnames or ()) != QUERY_DIAGNOSTIC_TSV_COLUMNS:
            raise ValueError("query_diagnostics.tsv columns differ from the contract")
        observed_rows = list(reader)
    if len(observed_rows) != 322:
        raise ValueError("query_diagnostics.tsv must contain 322 rows")

    integer_fields = {
        "selected_k",
        "invalid_cell_count",
        "primary_zero_score_model_count",
        "diagnostic_zero_score_model_count",
        "primary_top10_prior_tie_position_count",
        "primary_top10_zero_score_position_count",
        "positive_retrieval_count",
        "exact_mask_canonical_count",
        "exact_mask_source_id_count",
    }
    float_fields = {
        "selected_inner_ndcg_at_10",
        "selected_inner_mrr",
        "inner_ndcg_winner_margin",
        "invalid_cell_share",
        "source_weight_sum",
    }
    selection_counts: Counter[str] = Counter()
    validity_counts: Counter[str] = Counter()
    margins: list[float] = []
    for expected, observed in zip(expected_rows, observed_rows, strict=True):
        for field in ("query_id", "selected_configuration"):
            if observed[field] != str(expected[field]):
                raise ValueError(f"query diagnostic field {field!r} disagrees")
        for field in integer_fields:
            if int(observed[field]) != int(expected[field]):
                raise ValueError(f"query diagnostic field {field!r} disagrees")
        for field in float_fields:
            if not math.isclose(
                float(observed[field]),
                float(expected[field]),
                rel_tol=0.0,
                abs_tol=5e-12,
            ):
                raise ValueError(f"query diagnostic field {field!r} disagrees")
        observed_reasons = json.loads(observed["validity_reason_counts_json"])
        if observed_reasons != expected["validity_reason_counts"]:
            raise ValueError("query diagnostic validity reasons disagree")
        selection_counts[str(expected["selected_configuration"])] += 1
        validity_counts.update(
            {str(key): int(value) for key, value in observed_reasons.items()}
        )
        margins.append(float(expected["inner_ndcg_winner_margin"]))

    if (
        dict(sorted(selection_counts.items()))
        != summary["selection"]["configuration_frequency"]
    ):
        raise ValueError("per-query selected-K counts disagree with the summary")
    if (
        dict(sorted(validity_counts.items()))
        != diagnostics["opportunities"]["selected_validity_reason_counts"]
    ):
        raise ValueError("per-query validity counts disagree with the summary")
    expected_margin = diagnostics["selection"]["inner_ndcg_winner_margin"]
    observed_margin = {
        "mean": statistics.fmean(margins),
        "min": min(margins),
        "max": max(margins),
        "count": len(margins),
    }
    for field, value in observed_margin.items():
        expected_value = expected_margin[field]
        if isinstance(value, int):
            if value != expected_value:
                raise ValueError("per-query inner-margin count disagrees")
        elif not math.isclose(
            float(value), float(expected_value), rel_tol=0.0, abs_tol=5e-12
        ):
            raise ValueError("per-query inner-margin summary disagrees")


def _validate_determinism(
    *,
    summary: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    validation: Mapping[str, Any],
    rerun_summary: Mapping[str, Any],
    rerun_diagnostics: Mapping[str, Any],
    rerun_validation: Mapping[str, Any],
    artifact_dir: Path,
    rerun_dir: Path,
) -> dict[str, Any]:
    first = deepcopy(summary)
    second = deepcopy(rerun_summary)
    first.pop("runtime", None)
    second.pop("runtime", None)
    if first != second:
        raise ValueError("official summaries differ beyond elapsed runtime")
    if diagnostics != rerun_diagnostics:
        raise ValueError("official diagnostics differ on the clean rerun")

    identical = {
        name: (artifact_dir / name).read_bytes() == (rerun_dir / name).read_bytes()
        for name in DECISION_RELEVANT_FILES
    }
    if not all(identical.values()):
        raise ValueError(f"deterministic rerun mismatch: {identical!r}")

    first_hashes = validation["artifact_sha256"]
    second_hashes = rerun_validation["artifact_sha256"]
    compared_hashes = {
        name: first_hashes[name] == second_hashes[name]
        for name in DECISION_RELEVANT_FILES
    }
    if not all(compared_hashes.values()):
        raise ValueError("decision-relevant artifact hashes differ on the clean rerun")
    if validation["determinism"] != rerun_validation["determinism"]:
        raise ValueError("determinism receipts differ between runs")
    if validation["determinism"]["all_runs_complete"] is not True:
        raise ValueError("writer determinism receipt includes an incomplete run")
    if (
        validation["determinism"]["decision_relevant_outputs_are_canonically_hashed"]
        is not True
    ):
        raise ValueError("writer did not canonically hash decision evidence")

    return {
        "summary_equal_except_runtime": True,
        "diagnostics_identical": True,
        "decision_relevant_files_identical": identical,
        "artifact_hashes_identical": compared_hashes,
        "first_runtime_seconds": summary["runtime"][
            "experiment_seconds_through_summary"
        ],
        "rerun_runtime_seconds": rerun_summary["runtime"][
            "experiment_seconds_through_summary"
        ],
        "first_peak_rss_bytes": summary["runtime"]["peak_rss_bytes_at_summary"],
        "rerun_peak_rss_bytes": rerun_summary["runtime"]["peak_rss_bytes_at_summary"],
    }


def _recompute_effectiveness(
    summary: Mapping[str, Any],
    artifact_dir: Path,
) -> dict[str, Any]:
    qrels = load_complete_qrels(QRELS)
    evaluations: dict[str, EvaluationResult] = {}
    for system, filename in RUN_FILES.items():
        path = artifact_dir / filename
        validate_trec_run(
            path,
            expected_query_ids=tuple(qrels),
            expected_model_ids=tuple(next(iter(qrels.values()))),
        )
        evaluations[system] = evaluate_rankings(rankings_from_run(path), qrels, k=10)
    evaluations[DENSE] = load_dense_per_query_reference(
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

    comparison_checks: dict[str, bool] = {}
    outcome_checks: dict[str, bool] = {}
    primary = evaluations[PRIMARY]
    for offset, baseline_name in enumerate(BASELINE_ORDER):
        baseline = evaluations[baseline_name]
        ndcg = paired_bootstrap_ci(
            {
                key: baseline.per_query[key].ndcg_at_k
                for key in sorted(baseline.per_query)
            },
            {
                key: primary.per_query[key].ndcg_at_k
                for key in sorted(primary.per_query)
            },
            n_resamples=10_000,
            seed=2026 + offset,
        )
        mrr = paired_bootstrap_ci(
            {
                key: baseline.per_query[key].reciprocal_rank
                for key in sorted(baseline.per_query)
            },
            {
                key: primary.per_query[key].reciprocal_rank
                for key in sorted(primary.per_query)
            },
            n_resamples=10_000,
            seed=12_026 + offset,
        )
        expected = summary["comparisons"][baseline_name]
        comparison_checks[baseline_name] = _ci_matches(
            asdict(ndcg), expected["ndcg_at_10_delta"]
        ) and _ci_matches(asdict(mrr), expected["mrr_delta"])
        recomputed_ndcg_outcomes = _win_tie_loss(
            baseline,
            primary,
            metric="ndcg_at_k",
        )
        recomputed_mrr_outcomes = _win_tie_loss(
            baseline,
            primary,
            metric="reciprocal_rank",
        )
        outcome_checks[baseline_name] = (
            recomputed_ndcg_outcomes == expected["ndcg_at_10_win_tie_loss"]
            and recomputed_mrr_outcomes == expected["mrr_win_tie_loss"]
        )
    if not all(comparison_checks.values()):
        raise ValueError(f"paired interval recalculation mismatch: {comparison_checks}")
    if not all(outcome_checks.values()):
        raise ValueError(f"paired outcome recalculation mismatch: {outcome_checks}")
    return {
        "metrics": recomputed_metrics,
        "paired_intervals_match": comparison_checks,
        "paired_outcomes_match": outcome_checks,
        "source": "TREC runs plus official qrels; dense per-query tracked reference",
    }


def _win_tie_loss(
    baseline: EvaluationResult,
    candidate: EvaluationResult,
    *,
    metric: str,
) -> dict[str, int]:
    counts = {"wins": 0, "ties": 0, "losses": 0}
    if set(baseline.per_query) != set(candidate.per_query):
        raise ValueError("paired outcome inputs have different query IDs")
    for query_id, baseline_value in baseline.per_query.items():
        left = float(getattr(baseline_value, metric))
        right = float(getattr(candidate.per_query[query_id], metric))
        difference = right - left
        if difference > 1e-12:
            counts["wins"] += 1
        elif difference < -1e-12:
            counts["losses"] += 1
        else:
            counts["ties"] += 1
    return counts


def _validate_decision(
    summary: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    primary_ndcg = summary["metrics"][PRIMARY]["ndcg_at_10"]
    sumscore_ndcg = summary["metrics"][SUMSCORE]["ndcg_at_10"]
    tfidf_interval = summary["comparisons"][TFIDF]["ndcg_at_10_delta"]
    deployment = diagnostics["deployment_behavior"]
    rho = _require_number(
        deployment,
        "inventory_vs_total_primary_score_spearman",
    )
    quartile_share = _require_number(
        deployment,
        "largest_inventory_quartile_top10_share",
    )
    expected = {
        "ndcg_ci_above_tfidf": tfidf_interval["lower"] > 0.0,
        "ndcg_within_0_010_of_sumscore": (
            primary_ndcg >= sumscore_ndcg - MAX_NDCG_DROP_FROM_SUMSCORE
        ),
        "deployment_inventory_rho_at_most_0_7388": abs(rho) <= INVENTORY_RHO_GATE,
        "deployment_largest_quartile_share_at_most_0_411": (
            quartile_share <= LARGEST_QUARTILE_GATE
        ),
    }
    decision = summary["decision"]
    observed = {name: value["passed"] for name, value in decision["gates"].items()}
    if observed != expected:
        raise ValueError(
            f"decision gates were not reproducible: {observed} != {expected}"
        )
    if decision["all_four_gates_passed"] != all(expected.values()):
        raise ValueError("combined decision disagrees with individual gates")
    if decision["strong_success"] != (primary_ndcg >= sumscore_ndcg):
        raise ValueError("strong-success flag disagrees with the locked definition")
    return {
        "gate_results": expected,
        "all_four_gates_passed": all(expected.values()),
        "inventory_rho": rho,
        "largest_quartile_share": quartile_share,
    }


def _validate_evidence_hashes(
    validation: Mapping[str, Any],
    artifact_dir: Path,
) -> None:
    hashes = validation["artifact_sha256"]
    required = {*DECISION_RELEVANT_FILES, "score_tensors.npz", "summary.json"}
    if not required.issubset(hashes):
        raise ValueError(
            "validation receipt omits one or more required artifact hashes"
        )
    for name in required:
        observed = _sha256(artifact_dir / name)
        if observed != hashes[name]:
            raise ValueError(
                f"artifact hash mismatch for {name}: {observed} != {hashes[name]}"
            )


def _validate_tracked_response_controls(
    summary: Mapping[str, Any],
    tracked: Mapping[str, Any],
) -> None:
    expected = {
        SUMSCORE: tracked["metrics"]["response_sumscore_nested"],
        TOP10SUM: tracked["metrics"]["response_normalized_nested"],
    }
    for system, metrics in expected.items():
        for metric in ("ndcg_at_10", "mrr"):
            if not math.isclose(
                float(summary["metrics"][system][metric]),
                float(metrics[metric]),
                rel_tol=0.0,
                abs_tol=5e-12,
            ):
                raise ValueError(
                    f"{system} does not reproduce the tracked PR #8 {metric}"
                )


def _ci_matches(observed: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
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


def _artifact(results: Mapping[str, Any], results_path: Path) -> dict[str, Any]:
    generated_at = results["snapshot_as_of"]
    sources = _sources(results_path, generated_at)
    datasets = {
        "headline": _headline_rows(results),
        "system_metrics": _system_metric_rows(results["metrics"]),
        "query_outcomes": _query_outcome_rows(results["comparisons"]),
        "response_statuses": _response_status_rows(results["diagnostics"]),
        "volume_gates": _volume_gate_rows(results),
        "inventory_quartiles": _inventory_rows(results),
        "selection_counts": _selection_rows(results["selection"]),
    }
    title = _report_title(results["decision"])
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "A novice-friendly, leakage-safe evaluation of source-query-first lexical "
                "answer alignment for TREC Million LLMs 2026."
            ),
            "generatedAt": generated_at,
            "cards": _cards(),
            "charts": _charts(),
            "tables": [_selection_table()],
            "sources": sources,
            "blocks": _blocks(results, title),
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": datasets,
        },
        "sources": sources,
    }


def _headline_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    metrics = results["metrics"]
    deployment = results["diagnostics"]["deployment_behavior"]
    decision = results["decision"]
    return [
        {
            "primary_ndcg": metrics[PRIMARY]["ndcg_at_10"],
            "ndcg_delta_vs_sumscore": (
                metrics[PRIMARY]["ndcg_at_10"] - metrics[SUMSCORE]["ndcg_at_10"]
            ),
            "primary_mrr": metrics[PRIMARY]["mrr"],
            "mrr_delta_vs_sumscore": (
                metrics[PRIMARY]["mrr"] - metrics[SUMSCORE]["mrr"]
            ),
            "inventory_rho": abs(
                _require_number(
                    deployment,
                    "inventory_vs_total_primary_score_spearman",
                )
            ),
            "inventory_rho_gate": INVENTORY_RHO_GATE,
            "largest_quartile_share": _require_number(
                deployment,
                "largest_inventory_quartile_top10_share",
            ),
            "largest_quartile_gate": LARGEST_QUARTILE_GATE,
            "passed_gate_count": sum(
                int(value["passed"]) for value in decision["gates"].values()
            ),
            "gate_count": len(decision["gates"]),
        }
    ]


def _system_metric_rows(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    labels = (
        (1, PRIMARY, "Source-query-first alignment"),
        (2, DENSE, "Dense query search"),
        (3, SUMSCORE, "SumScore response search"),
        (4, TOP10SUM, "Top10Sum response search"),
        (5, TFIDF, "TF-IDF query search"),
        (6, PRIOR, "Global prior"),
    )
    rows: list[dict[str, Any]] = []
    for system_order, key, label in labels:
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
                    "candidate_model_count": 150,
                    "evaluation_protocol": (
                        "strict nested outer leave-one-out"
                        if key not in {TFIDF, PRIOR}
                        else "fixed or fold-local outer leave-one-out control"
                    ),
                    "higher_is_better": True,
                }
            )
    return rows


def _query_outcome_rows(
    comparisons: Mapping[str, Any],
) -> list[dict[str, Any]]:
    labels = (
        (1, SUMSCORE, "SumScore"),
        (2, TOP10SUM, "Top10Sum"),
        (3, DENSE, "Dense"),
        (4, TFIDF, "TF-IDF"),
        (5, PRIOR, "Global prior"),
    )
    rows: list[dict[str, Any]] = []
    for comparison_order, key, baseline_label in labels:
        comparison = comparisons[key]
        for metric_order, (metric_label, interval_key, outcomes_key) in enumerate(
            (
                ("nDCG@10", "ndcg_at_10_delta", "ndcg_at_10_win_tie_loss"),
                ("MRR", "mrr_delta", "mrr_win_tie_loss"),
            ),
            start=1,
        ):
            interval = comparison[interval_key]
            outcomes = comparison[outcomes_key]
            comparison_metric = f"{baseline_label} · {metric_label}"
            for outcome_order, (field, outcome_label) in enumerate(
                (
                    ("wins", "Source-query-first better"),
                    ("ties", "Tie"),
                    ("losses", "Source-query-first worse"),
                ),
                start=1,
            ):
                count = outcomes[field]
                rows.append(
                    {
                        "comparison_order": comparison_order,
                        "metric_order": metric_order,
                        "outcome_order": outcome_order,
                        "comparison_metric": comparison_metric,
                        "baseline": baseline_label,
                        "metric": metric_label,
                        "outcome": outcome_label,
                        "outcome_count": count,
                        "outcome_share": count / 322,
                        "query_count": 322,
                        "mean_difference": interval["mean_difference"],
                        "ci_lower": interval["lower"],
                        "ci_upper": interval["upper"],
                    }
                )
    return rows


def _response_status_rows(diagnostics: Mapping[str, Any]) -> list[dict[str, Any]]:
    audit = diagnostics["opportunities"]["filter_audit"]
    values = (
        (1, "Eligible response", audit["eligible_response_cells"], False),
        (2, "Exact ‘I don't know’", audit["zero_exact_idk"], True),
        (3, "Refusal prefix", audit["zero_prefix_refusal"], True),
        (4, "Null or blank", audit["zero_null"] + audit["zero_blank"], True),
    )
    if audit["input_response_cells"] != EXPECTED_RAW_RESPONSE_CELLS:
        raise ValueError("response-cell audit has the wrong official denominator")
    rows = [
        {
            "status_order": order,
            "status": status,
            "cell_count": count,
            "cell_share": count / audit["input_response_cells"],
            "scored_as_zero": scored_as_zero,
            "raw_response_cells": audit["input_response_cells"],
            "filter_name": "basic_plus_prefix",
        }
        for order, status, count, scored_as_zero in values
    ]
    if sum(row["cell_count"] for row in rows) != audit["input_response_cells"]:
        raise ValueError("response status counts are not mutually exhaustive")
    return rows


def _volume_gate_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    primary = results["diagnostics"]["deployment_behavior"]
    sumscore = results["reference_volume_diagnostics"]["sumscore"]
    top10sum = results["reference_volume_diagnostics"]["top10sum"]
    values = (
        (
            1,
            "Inventory-score correlation",
            abs(sumscore["inventory_vs_aggregation_score_spearman"]),
            abs(top10sum["inventory_vs_aggregation_score_spearman"]),
            abs(
                _require_number(
                    primary,
                    "inventory_vs_total_primary_score_spearman",
                )
            ),
            INVENTORY_RHO_GATE,
        ),
        (
            2,
            "Largest-quartile placement share",
            sumscore["largest_inventory_quartile_top10_share"],
            top10sum["largest_inventory_quartile_top10_share"],
            _require_number(
                primary,
                "largest_inventory_quartile_top10_share",
            ),
            LARGEST_QUARTILE_GATE,
        ),
    )
    rows: list[dict[str, Any]] = []
    for metric_order, metric, sum_value, top10_value, primary_value, gate in values:
        for series_order, (series, value, role) in enumerate(
            (
                ("SumScore", sum_value, "baseline"),
                ("Top10Sum", top10_value, "context"),
                ("Source-query-first", primary_value, "actual"),
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
                    "model_count": 150,
                    "top10_position_count": 3_220,
                }
            )
    return rows


def _inventory_rows(results: Mapping[str, Any]) -> list[dict[str, Any]]:
    primary = results["diagnostics"]["deployment_behavior"]
    sumscore = results["reference_volume_diagnostics"]["sumscore"]
    top10sum = results["reference_volume_diagnostics"]["top10sum"]
    primary_quartiles = primary["quartiles"]
    sumscore_quartiles = sumscore["quartiles"]
    top10sum_quartiles = top10sum["quartiles"]
    labels = ("Fewest answers", "Lower-middle", "Upper-middle", "Most answers")
    if not (
        len(primary_quartiles)
        == len(sumscore_quartiles)
        == len(top10sum_quartiles)
        == 4
    ):
        raise ValueError("inventory diagnostics do not contain four frozen quartiles")
    rows: list[dict[str, Any]] = []
    for index, (primary_row, sum_row, top10_row) in enumerate(
        zip(primary_quartiles, sumscore_quartiles, top10sum_quartiles, strict=True)
    ):
        if primary_row["model_count"] != sum_row["model_count"]:
            raise ValueError(
                "source-query-first changed the frozen inventory quartiles"
            )
        series = (
            (1, "Models in group", primary_row["model_count"] / 150),
            (2, "SumScore placements", sum_row["top10_appearance_share"]),
            (3, "Top10Sum placements", top10_row["top10_appearance_share"]),
            (4, "Source-query-first placements", primary_row["top10_appearance_share"]),
        )
        for series_order, series_name, share in series:
            rows.append(
                {
                    "quartile_order": index + 1,
                    "series_order": series_order,
                    "inventory_group": labels[index],
                    "series": series_name,
                    "share": share,
                    "model_count": primary_row["model_count"],
                    "inventory_min": primary_row["inventory_min"],
                    "inventory_max": primary_row["inventory_max"],
                    "source_alignment_top10_appearances": primary_row[
                        "top10_appearances"
                    ],
                }
            )
    return rows


def _selection_rows(selection: Mapping[str, Any]) -> list[dict[str, Any]]:
    frequencies = selection["configuration_frequency"]
    return [
        {
            "configuration_order": index,
            "configuration": configuration,
            "selected_k": int(configuration.rsplit("k", maxsplit=1)[1]),
            "outer_fold_count": frequencies.get(configuration, 0),
            "outer_fold_share": frequencies.get(configuration, 0) / 322,
            "selection_metric": "inner macro nDCG@10",
        }
        for index, configuration in enumerate(CANDIDATE_ORDER, start=1)
    ]


def _cards() -> list[dict[str, Any]]:
    return [
        {
            "id": "ranking_quality",
            "description": (
                "Nested top-10 ranking quality; the signed badge is source-query-first "
                "minus independently nested SumScore."
            ),
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {
                    "label": "Source-query-first nDCG@10",
                    "field": "primary_ndcg",
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
            "id": "first_useful_model",
            "description": (
                "Nested reciprocal rank of the first useful model; the signed badge is "
                "source-query-first minus independently nested SumScore."
            ),
            "dataset": "headline",
            "sourceId": "headline_sql",
            "metrics": [
                {
                    "label": "Source-query-first MRR",
                    "field": "primary_mrr",
                    "format": "number",
                },
                {
                    "label": "vs SumScore",
                    "field": "mrr_delta_vs_sumscore",
                    "format": "number",
                    "signed": True,
                },
            ],
        },
        {
            "id": "inventory_rho",
            "description": (
                "Absolute BASIC-inventory association with total primary score for the "
                "fixed deployment K. Lower is better."
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
                "Share of all 3,220 fixed-deployment top-10 placements received by the "
                "largest frozen BASIC-inventory quartile. Lower is better."
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
                "The locked rule requires all four effectiveness and volume-risk gates."
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
                "Same 322 held-out queries and 150 candidate models; both metrics range "
                "from 0 to 1 and higher is better."
            ),
            "type": "horizontalBar",
            "dataset": "system_metrics",
            "sourceId": "system_metrics_sql",
            "intent": "comparison",
            "question": "Which approach ranks useful models earliest?",
            "rationale": (
                "Grouped horizontal bars keep six long approach names readable on mobile."
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
                    {
                        "field": "query_count",
                        "type": "quantitative",
                        "label": "Queries",
                    },
                    {
                        "field": "candidate_model_count",
                        "type": "quantitative",
                        "label": "Models",
                    },
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
            "maxRows": 12,
        },
        {
            "id": "query_outcomes_chart",
            "title": "Per-query outcomes versus each baseline",
            "subtitle": (
                "Share of 322 paired queries for both metrics; better and worse refer to "
                "source-query-first relative to the named baseline."
            ),
            "type": "horizontalStackedBar100",
            "dataset": "query_outcomes",
            "sourceId": "query_outcomes_sql",
            "intent": "composition",
            "question": (
                "How often did source-query-first beat, tie, or trail each baseline?"
            ),
            "rationale": (
                "Normalized stacks expose paired query frequency while source rows retain "
                "exact mean differences and intervals."
            ),
            "comparisonContext": {
                "denominator": "322 paired queries per baseline and metric",
                "grain": "one held-out query",
                "normalization": "each bar sums to 100%",
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
                    "label": "Paired outcome",
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
                        "label": "Average difference",
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
                "title": "Source-query-first outcome",
            },
            "labels": {"values": "auto"},
            "layout": "full",
            "maxRows": 30,
        },
        {
            "id": "response_status_chart",
            "title": "Historical response cells by frozen status",
            "subtitle": (
                "All 1,163,850 discovery query-model cells; invalid responses remain "
                "opportunities and receive zero rather than being dropped."
            ),
            "type": "horizontalBar",
            "dataset": "response_statuses",
            "sourceId": "response_statuses_sql",
            "intent": "comparison",
            "question": "Which historical response cells can carry lexical evidence?",
            "rationale": "Four exact status counts make the zero policy auditable.",
            "comparisonContext": {
                "denominator": "1,163,850 discovery query-model cells",
                "grain": "one frozen response status",
                "normalization": "mutually exhaustive cell counts",
                "unit": "response cells",
            },
            "encodings": {
                "x": {"field": "status", "type": "nominal", "label": "Status"},
                "y": {
                    "field": "cell_count",
                    "type": "quantitative",
                    "label": "Response cells",
                    "format": "compact",
                },
                "tooltip": [
                    {
                        "field": "cell_share",
                        "type": "quantitative",
                        "label": "Cell share",
                        "format": "percent",
                    },
                    {
                        "field": "scored_as_zero",
                        "type": "text",
                        "label": "Explicit zero",
                    },
                ],
            },
            "settings": {
                "orientation": "horizontal",
                "categoryLabelPolicy": "wrap",
                "showValues": True,
            },
            "palette": {"kind": "categorical"},
            "labels": {"values": "all"},
            "layout": "full",
            "maxRows": 4,
        },
        {
            "id": "volume_gate_comparison",
            "title": "Volume-dependence checks",
            "subtitle": (
                "Fixed-deployment behavior on all development queries; lower is better "
                "and source-query-first had to be at or below each maximum."
            ),
            "type": "bar",
            "dataset": "volume_gates",
            "sourceId": "volume_gates_sql",
            "intent": "comparison",
            "question": "Did equal source-query opportunity reduce both volume signals?",
            "rationale": (
                "Grouped bars place the primary result beside both response controls and "
                "the frozen maximum."
            ),
            "comparisonContext": {
                "denominator": "150 models and 3,220 top-10 placements",
                "grain": "one fixed-deployment behavior diagnostic",
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
            "maxRows": 8,
        },
        {
            "id": "inventory_quartile_comparison",
            "title": "Top-10 placement share by response inventory",
            "subtitle": (
                "Models use the same frozen BASIC-inventory quartiles as PR #8; balanced "
                "exposure would roughly follow each group's model share."
            ),
            "type": "bar",
            "dataset": "inventory_quartiles",
            "sourceId": "inventory_quartiles_sql",
            "intent": "comparison",
            "question": "Where are top-10 placements concentrated?",
            "rationale": (
                "Grouped shares compare all three response systems with the available "
                "model share in each frozen quartile."
            ),
            "comparisonContext": {
                "denominator": "150 models and 3,220 top-10 placements per system",
                "grain": "frozen BASIC-response-inventory quartile",
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
                        "label": "Fewest BASIC responses",
                    },
                    {
                        "field": "inventory_max",
                        "type": "quantitative",
                        "label": "Most BASIC responses",
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
            "maxRows": 16,
        },
    ]


def _selection_table() -> dict[str, Any]:
    return {
        "id": "selection_counts_table",
        "title": "Nested K selection counts",
        "subtitle": (
            "One selected K per held-out query; only inner macro nDCG@10 entered selection."
        ),
        "dataset": "selection_counts",
        "sourceId": "selection_counts_sql",
        "columns": [
            {"field": "selected_k", "label": "K", "format": "number"},
            {
                "field": "outer_fold_count",
                "label": "Outer folds",
                "format": "number",
            },
            {
                "field": "outer_fold_share",
                "label": "Fold share",
                "format": "percent",
            },
        ],
        "defaultSort": {"field": "selected_k", "direction": "asc"},
        "maxRows": 3,
    }


def _blocks(results: Mapping[str, Any], title: str) -> list[dict[str, Any]]:
    metrics = results["metrics"]
    comparisons = results["comparisons"]
    diagnostics = results["diagnostics"]
    decision = results["decision"]
    deployment = results["deployment"]
    primary = metrics[PRIMARY]
    dense = metrics[DENSE]
    sumscore = metrics[SUMSCORE]
    top10sum = metrics[TOP10SUM]
    diagnostic = metrics[DIAGNOSTIC]
    sumscore_interval = comparisons[SUMSCORE]["ndcg_at_10_delta"]
    tfidf_interval = comparisons[TFIDF]["ndcg_at_10_delta"]
    behavior = diagnostics["deployment_behavior"]
    opportunities = diagnostics["opportunities"]
    response_audit = opportunities["filter_audit"]
    runtime = results["runtime"]
    peak_rss_gib = runtime["peak_rss_bytes_at_summary"] / (1024**3)
    selected_k = int(str(deployment["configuration"]).rsplit("k", maxsplit=1)[1])
    gate_lines = "\n".join(
        f"- **{'Pass' if value['passed'] else 'Fail'} — {_gate_label(key)}:** "
        f"{_format_gate_observed(key, value['observed'])}."
        for key, value in decision["gates"].items()
    )
    gate_count = sum(int(value["passed"]) for value in decision["gates"].values())

    return [
        {"id": "title", "type": "markdown", "body": f"# {title}"},
        {
            "id": "executive_summary",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## Executive Summary\n\n"
                f"- **The locked advancement rule {'passed' if decision['all_four_gates_passed'] else 'did not pass'}.** "
                f"Source-query-first answer alignment passed **{gate_count} of 4** gates. "
                "This result still requires an explicit owner decision and does not release "
                "or merge any earlier response work.\n\n"
                f"- **Held-out ranking quality was {primary['ndcg_at_10']:.4f} nDCG@10 and "
                f"{primary['mrr']:.4f} MRR.** Its nDCG difference from independently nested "
                f"SumScore was **{sumscore_interval['mean_difference']:+.4f}**, with paired "
                f"95% interval **[{sumscore_interval['lower']:+.4f}, "
                f"{sumscore_interval['upper']:+.4f}]**.\n\n"
                f"- **Every model received the same source-query opportunities.** Each "
                "selected source query kept all 150 model slots; null, blank, exact ‘I don't "
                "know,’ and frozen-prefix refusals remained in the denominator and scored zero.\n\n"
                f"- **Dense query search remained at {dense['ndcg_at_10']:.4f} nDCG@10 / "
                f"{dense['mrr']:.4f} MRR.** Source-to-response lexical overlap is called "
                "answer alignment—not correctness, factuality, usefulness, or human-rated quality."
            ),
        },
        {
            "id": "task_primer",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## What problem is this experiment solving?\n\n"
                "For each new question, the system must rank the same **150 candidate language "
                "models**. A good ranking puts models judged useful for that question near the "
                "top. The evaluation contains **322 development queries**.\n\n"
                "A **historical response** is one organizer-provided answer from one model for "
                "one earlier discovery query. This experiment first retrieves related older "
                "queries, then checks whether each model's stored response lexically addresses "
                "its own older query. The discovery data has no per-response correctness labels."
            ),
        },
        {
            "id": "metric_guide",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## How to read the scores\n\n"
                "- **nDCG@10** measures the quality and order of the first ten models. Higher "
                "is better; 1 is ideal. This is the only selection metric.\n"
                "- **MRR** measures how early the first useful model appears. First place "
                "scores 1, second 0.5, and fifth 0.2. MRR never selects K or a decision gate.\n"
                "- A **paired 95% interval** resamples differences on the same 322 queries. An "
                "interval crossing zero does not establish a clear average direction here."
            ),
        },
        {
            "id": "headline_metrics",
            "type": "metric-strip",
            "cardIds": [card["id"] for card in _cards()],
        },
        {
            "id": "effectiveness_finding",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## Six systems put the source-query-first result in context\n\n"
                f"Source-query-first scored **{primary['ndcg_at_10']:.4f} nDCG@10 / "
                f"{primary['mrr']:.4f} MRR**. SumScore scored "
                f"**{sumscore['ndcg_at_10']:.4f} / {sumscore['mrr']:.4f}**, and Top10Sum "
                f"scored **{top10sum['ndcg_at_10']:.4f} / {top10sum['mrr']:.4f}**. The "
                f"paired nDCG difference from TF-IDF was **{tfidf_interval['mean_difference']:+.4f}** "
                f"with interval **[{tfidf_interval['lower']:+.4f}, "
                f"{tfidf_interval['upper']:+.4f}]**. The chart uses the same 322 held-out "
                "queries and 150-model universe for every approach."
            ),
        },
        {
            "id": "system_chart",
            "type": "chart",
            "chartId": "system_metric_comparison",
            "layout": {"columns": 12, "minColumns": 8},
        },
        {
            "id": "method_explainer",
            "type": "markdown",
            "sourceId": "source_pipeline",
            "body": (
                "## What source-query-first changes\n\n"
                "`New target query → retrieve K unique older queries → keep the same 150 model "
                "slots for every older query → score answer alignment → combine with normalized "
                "source-query weights → rank models`\n\n"
                "The implementation keeps three lexical signals separate:\n\n"
                "1. **Source-query relevance:** new target to older query.\n"
                "2. **Answer alignment:** older query to that model's own stored response. This "
                "is the primary response-side signal.\n"
                "3. **Target-to-response relevance:** new target to stored response. This is a "
                "same-K diagnostic only and cannot select K, enter the primary score, or affect "
                "the four gates.\n\n"
                "Plain BM25 is used throughout. There is no RM3, query-plus-response "
                "concatenation, learned weight, score fusion, dense response embedding, or log "
                "probability."
            ),
        },
        {
            "id": "opportunity_finding",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## Equal opportunity means preserving zeros, not deleting them\n\n"
                f"The official discovery matrix has **{response_audit['input_response_cells']:,}** "
                f"query-model cells. The frozen rule leaves "
                f"**{response_audit['eligible_response_cells']:,} eligible responses**, gives zero to "
                f"**{response_audit['zero_exact_idk']:,} exact ‘I don't know’ cells** and "
                f"**{response_audit['zero_prefix_refusal']:,} additional prefix refusals**, "
                "and preserves null or blank slots as zeros if present.\n\n"
                "For every selected older query, all 150 model slots stay in the local BM25 "
                "corpus and denominator. Equal opportunity does not mean every model supplied a "
                "usable answer; it means unusable answers cannot disappear from evaluation."
            ),
        },
        {
            "id": "response_status_chart_block",
            "type": "chart",
            "chartId": "response_status_chart",
            "layout": {"columns": 12, "minColumns": 8},
        },
        {
            "id": "leakage_explainer",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## How held-out testing prevents answer-key leakage\n\n"
                "Each outer query is sealed. The system compares K = 10, 50, and 100 only on "
                "the other 321 queries, chooses the largest inner nDCG@10, and then ranks the "
                "sealed query once. Its grades never choose its own K or evidence. Exact "
                "analyzer-equivalent discovery queries are masked before retrieval.\n\n"
                "`Seal 1 query → choose K on the other 321 → rank 150 models → reveal grades → "
                "repeat 322 times`\n\n"
                "The source retrieval and response signal tensors are qrel-free and shared "
                "across folds. Fold-local qrels enter only selection, tie priors, and evaluation "
                "where the protocol permits them."
            ),
        },
        {
            "id": "selection_finding",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## Nested validation chose K before each held-out ranking\n\n"
                f"The full-development deployment hint is **K = {selected_k}**. It is selected "
                "after held-out reporting and is used only for future deployment guidance and "
                "like-for-like volume diagnostics; its same-development effectiveness is "
                "optimistic. The table shows how often each K was selected across the 322 "
                "unbiased outer predictions."
            ),
        },
        {
            "id": "selection_table_block",
            "type": "table",
            "tableId": "selection_counts_table",
            "layout": {"columns": 12, "minColumns": 8},
        },
        {
            "id": "diagnostic_finding",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## Direct target-to-response relevance stays a diagnostic\n\n"
                f"Using the primary-selected K but replacing answer alignment with direct "
                f"target-to-response relevance produced **{diagnostic['ndcg_at_10']:.4f} "
                f"nDCG@10 / {diagnostic['mrr']:.4f} MRR**. This is not a seventh headline "
                "competitor: it did not select K, enter the primary score, choose deployment K, "
                "or affect the gates. Its purpose is to keep the three lexical signals "
                "attributable."
            ),
        },
        {
            "id": "paired_outcomes_finding",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## Per-query outcomes show frequency; intervals show average uncertainty\n\n"
                "Each bar counts paired query wins, ties, and losses for one metric and "
                "baseline. Those counts show how often the ranking changed direction; the "
                "mean paired difference and its interval determine magnitude and uncertainty. "
                "The exact values remain available in the chart's semantic source table."
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
                "## Equal source-query slots are tested against the old volume-risk gates\n\n"
                f"For fixed K = {selected_k}, absolute BASIC-inventory correlation with total "
                f"primary score was **{abs(_require_number(behavior, 'inventory_vs_total_primary_score_spearman')):.4f}** "
                f"against a maximum of **{INVENTORY_RHO_GATE:.4f}**. The largest frozen "
                f"inventory quartile received **{_require_number(behavior, 'largest_inventory_quartile_top10_share'):.1%}** "
                f"of top-10 placements against a maximum of **{LARGEST_QUARTILE_GATE:.1%}**. "
                "These full-development values are behavior gates, not unbiased effectiveness "
                "estimates. Inventory association is a risk diagnostic, not proof of causation."
            ),
        },
        {
            "id": "volume_gate_chart_block",
            "type": "chart",
            "chartId": "volume_gate_comparison",
            "layout": {"columns": 12, "minColumns": 8},
        },
        {
            "id": "inventory_chart_block",
            "type": "chart",
            "chartId": "inventory_quartile_comparison",
            "layout": {"columns": 12, "minColumns": 8},
        },
        {
            "id": "decision",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## The preregistered decision\n\n"
                f"{gate_lines}\n\n"
                f"The locked rule required **all four gates**. {_decision_reason(decision)} "
                "Matching or exceeding SumScore nDCG@10 is recorded as strong success but is "
                "not a fifth gate. Dense, Top10Sum, and MRR comparisons are context only."
            ),
        },
        {
            "id": "next_steps",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## What should happen next?\n\n"
                "1. **Treat the locked four-gate result as the experiment decision.** Do not "
                "substitute a post-hoc metric, K, blend, or diagnostic.\n"
                "2. **Request the explicit owner decision.** Passing the experiment rule would "
                "only advance the method for consideration; it cannot release or merge earlier "
                "work on its own.\n"
                "3. **Keep answer alignment scoped correctly.** Genuine correctness or "
                "helpfulness claims require response-level human judgments that this corpus does "
                "not provide."
            ),
        },
        {
            "id": "further_questions",
            "type": "markdown",
            "body": (
                "## Further questions\n\n"
                "- Does lexical answer alignment track human response quality when independent "
                "response-level judgments become available?\n"
                "- Which queries explain disagreements between source-query-first and dense "
                "query retrieval without tuning on their held-out grades?\n"
                "- Does equal source-query opportunity behave similarly on the hidden test when "
                "source-query coverage and refusal patterns shift?"
            ),
        },
        {
            "id": "technical_details",
            "type": "markdown",
            "sourceId": "reviewed_results",
            "body": (
                "## Reproducibility and caveats\n\n"
                f"- The CPU-local experiment reached its summary in **{runtime['experiment_seconds_through_summary']:.1f} seconds** "
                f"of wall time and **{runtime['cpu_seconds_through_summary']:.1f} CPU-seconds**, "
                f"with **{peak_rss_gib:.2f} GiB peak RSS** on the recorded platform. It uses "
                "pinned lexical analysis and float64 BM25; no GPU, pretrained encoder, "
                "external corpus, log probabilities, or learned weights enter the primary "
                "method.\n"
                "- A clean second run reproduced every decision-relevant file byte for byte, "
                "and the report builder independently recalculated runs, metrics, paired "
                "intervals, and all four gates.\n"
                "- Nested development evaluation reduces tuning bias but does not guarantee "
                "hidden-test performance.\n"
                "- Paired bootstrap intervals condition on the completed stitched selections; "
                "they do not rerun selection inside each resample.\n"
                "- Fixed-deployment K behavior diagnostics reuse all development queries and "
                "are not held-out effectiveness estimates.\n"
                "- Answer alignment measures lexical overlap with the response's own source "
                "query. It does not establish correctness, factuality, usefulness, or human "
                "preference."
            ),
        },
    ]


def _sources(results_path: Path, generated_at: str) -> list[dict[str, Any]]:
    result_path = str(results_path.relative_to(ROOT))
    common_filters = [
        "322 official development queries",
        "150 candidate models",
        "strict nested leave-one-query-out",
        "advisor-approved protocol locked before effectiveness scoring",
    ]
    return [
        {
            "id": "reviewed_results",
            "label": "Reviewed source-query-first result snapshot",
            "path": result_path,
        },
        {
            "id": "headline_sql",
            "label": "Headline source-query-first metrics and gates",
            "path": result_path,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT primary_ndcg, ndcg_delta_vs_sumscore, primary_mrr, "
                    "mrr_delta_vs_sumscore, inventory_rho, inventory_rho_gate, "
                    "largest_quartile_share, largest_quartile_gate, passed_gate_count, "
                    "gate_count FROM headline_metrics"
                ),
                "description": (
                    "Selects headline effectiveness, frozen volume checks, and gate counts "
                    "from the reviewed result snapshot."
                ),
                "executed_at": generated_at,
                "tables_used": ["headline_metrics"],
                "filters": common_filters,
                "metric_definitions": _metric_definitions(),
            },
        },
        {
            "id": "system_metrics_sql",
            "label": "Six-system effectiveness comparison",
            "path": result_path,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT system_order, metric_order, system, metric, value, "
                    "query_count, candidate_model_count, evaluation_protocol, "
                    "higher_is_better FROM system_metrics ORDER BY system_order, metric_order"
                ),
                "description": (
                    "Selects held-out nDCG@10 and MRR for the six headline systems."
                ),
                "executed_at": generated_at,
                "tables_used": ["system_metrics"],
                "filters": common_filters,
                "metric_definitions": _metric_definitions(),
            },
        },
        {
            "id": "query_outcomes_sql",
            "label": "Paired query outcomes and uncertainty",
            "path": result_path,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT comparison_order, metric_order, outcome_order, "
                    "comparison_metric, baseline, metric, outcome, outcome_count, "
                    "outcome_share, query_count, mean_difference, ci_lower, ci_upper "
                    "FROM query_outcomes ORDER BY comparison_order, metric_order, outcome_order"
                ),
                "description": (
                    "Selects source-query-first win, tie, and loss counts against each "
                    "headline baseline with paired interval context."
                ),
                "executed_at": generated_at,
                "tables_used": ["query_outcomes"],
                "filters": [*common_filters, "10,000 paired-query resamples"],
                "metric_definitions": [
                    "Mean difference and confidence bounds are source-query-first minus the named baseline."
                ],
            },
        },
        {
            "id": "response_statuses_sql",
            "label": "Frozen response-cell status audit",
            "path": result_path,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT status_order, status, cell_count, cell_share, scored_as_zero, "
                    "raw_response_cells, filter_name FROM response_statuses ORDER BY status_order"
                ),
                "description": (
                    "Selects mutually exhaustive frozen response-cell statuses and the "
                    "explicit-zero policy."
                ),
                "executed_at": generated_at,
                "tables_used": ["response_statuses"],
                "filters": ["official discovery response matrix", "basic_plus_prefix"],
                "metric_definitions": [
                    "Invalid cells remain among the 150 local response slots and receive zero for both response-side signals."
                ],
            },
        },
        {
            "id": "volume_gates_sql",
            "label": "Fixed-deployment volume-gate comparison",
            "path": result_path,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT metric_order, series_order, metric, series, value, role, "
                    "lower_is_better, model_count, top10_position_count FROM volume_gates "
                    "ORDER BY metric_order, series_order"
                ),
                "description": (
                    "Selects SumScore, Top10Sum, source-query-first, and maximum-allowed "
                    "values for both frozen deployment-volume diagnostics."
                ),
                "executed_at": generated_at,
                "tables_used": ["volume_gates"],
                "filters": ["150 models", "3,220 fixed-deployment top-10 placements"],
                "metric_definitions": [
                    "Inventory-score correlation is absolute Spearman rho; lower is better.",
                    "Largest-quartile share uses the frozen PR #8 BASIC-response inventory groups.",
                ],
            },
        },
        {
            "id": "inventory_quartiles_sql",
            "label": "Frozen response-inventory quartile representation",
            "path": result_path,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT quartile_order, series_order, inventory_group, series, share, "
                    "model_count, inventory_min, inventory_max, "
                    "source_alignment_top10_appearances FROM inventory_quartiles "
                    "ORDER BY quartile_order, series_order"
                ),
                "description": (
                    "Selects model and fixed-deployment top-10 placement shares for the "
                    "same deterministic BASIC-inventory quartiles used by PR #8."
                ),
                "executed_at": generated_at,
                "tables_used": ["inventory_quartiles"],
                "filters": ["frozen BASIC inventory", "150 models", "3,220 placements"],
                "metric_definitions": [
                    "Each placement series and the model-share series sum to 100%."
                ],
            },
        },
        {
            "id": "selection_counts_sql",
            "label": "Nested K selection counts",
            "path": result_path,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": (
                    "SELECT configuration_order, configuration, selected_k, outer_fold_count, "
                    "outer_fold_share, selection_metric FROM selection_counts "
                    "ORDER BY configuration_order"
                ),
                "description": (
                    "Selects how often K=10, 50, or 100 was chosen using inner macro nDCG@10."
                ),
                "executed_at": generated_at,
                "tables_used": ["selection_counts"],
                "filters": ["322 outer folds", "primary alignment selector only"],
                "metric_definitions": [
                    "Outer-fold share is selected fold count divided by 322."
                ],
            },
        },
        {
            "id": "locked_protocol",
            "label": "Advisor-approved source-query-first preregistration",
            "path": "protocols/trec_2026_source_query_first.md",
        },
        {
            "id": "primary_summary",
            "label": "Primary official experiment summary",
            "path": "artifacts/source_query_first/summary.json",
        },
        {
            "id": "primary_validation",
            "label": "Complete-run and provenance validation receipt",
            "path": "artifacts/source_query_first/validation.json",
        },
        {
            "id": "deterministic_rerun",
            "label": "Independent clean deterministic rerun",
            "path": "artifacts/source_query_first_rerun/summary.json",
        },
        {
            "id": "source_pipeline",
            "label": "Tracked nested source-query-first experiment",
            "path": "trec_mlm/source_query_experiment.py",
        },
        {
            "id": "source_scorer",
            "label": "Tracked qrel-free source-query and response scorer",
            "path": "trec_mlm/source_query_first.py",
        },
        {
            "id": "response_controls",
            "label": "Tracked merged SumScore and Top10Sum result snapshot",
            "path": "reports/trec_2026_response_volume_normalization_results.json",
        },
        {
            "id": "dense_reference",
            "label": "Tracked dense nested per-query reference",
            "path": "reports/trec_2026_dense_query_knn_per_query.tsv",
        },
        {
            "id": "development_qrels",
            "label": "Official TREC development relevance judgments",
            "path": "data/dev/llm_dev_qrels.txt",
        },
        {
            "id": "discovery_responses",
            "label": "Official TREC historical discovery queries and responses",
            "path": "data/dev/llm_discovery_data.json",
        },
    ]


def _report_title(decision: Mapping[str, Any]) -> str:
    passed = sum(int(value["passed"]) for value in decision["gates"].values())
    if decision["all_four_gates_passed"]:
        return "Source-query-first answer alignment passed all four locked gates"
    return f"Source-query-first answer alignment passed {passed} of 4 locked gates"


def _decision_reason(decision: Mapping[str, Any]) -> str:
    passed = sum(int(value["passed"]) for value in decision["gates"].values())
    if decision["all_four_gates_passed"]:
        return (
            "All four frozen gates passed, so the method advances for owner consideration; "
            "the experiment does not authorize merging another pull request."
        )
    failed = 4 - passed
    return (
        f"{failed} of four frozen gates failed, so source-query-first answer alignment does "
        "not advance under the preregistered rule."
    )


def _gate_label(key: str) -> str:
    labels = {
        "ndcg_ci_above_tfidf": "Clearly better than TF-IDF on nDCG@10",
        "ndcg_within_0_010_of_sumscore": ("No more than 0.010 nDCG@10 below SumScore"),
        "deployment_inventory_rho_at_most_0_7388": (
            "BASIC inventory-score correlation at most 0.7388"
        ),
        "deployment_largest_quartile_share_at_most_0_411": (
            "Largest frozen inventory-quartile share at most 41.1%"
        ),
    }
    return labels[key]


def _format_gate_observed(key: str, value: Any) -> str:
    number = float(value)
    if key == "deployment_largest_quartile_share_at_most_0_411":
        return f"observed {number:.1%}"
    if key == "ndcg_ci_above_tfidf":
        return f"95% interval lower bound {number:+.4f}"
    if key == "ndcg_within_0_010_of_sumscore":
        return f"nDCG@10 difference {number:+.4f}"
    return f"observed {number:.4f}"


def _definitions() -> dict[str, str]:
    return {
        "task": (
            "For each new query, rank the same 150 candidate language models so useful "
            "models appear as early as possible."
        ),
        "source_query_relevance": (
            "Plain BM25 relevance from the new target query to one canonical historical "
            "source query."
        ),
        "answer_alignment": (
            "Plain BM25 overlap from a historical source query to one model's response to "
            "that query; a lexical proxy, not correctness."
        ),
        "target_response_relevance": (
            "Plain BM25 overlap from the new target query directly to a historical response; "
            "retained only as a non-selecting same-opportunity diagnostic."
        ),
        "nested_validation": (
            "Each held-out query is ranked only after K is selected using the other queries; "
            "its own grades never choose its K."
        ),
        "ndcg_at_10": (
            "Average quality of the first ten ranked models, giving more credit to useful "
            "models near the top; 0 is worst and 1 is ideal."
        ),
        "mrr": (
            "Average reciprocal rank of the first useful model: first is 1, second is 0.5, "
            "and fifth is 0.2."
        ),
        "inventory_rho": (
            "Spearman association between frozen BASIC usable-response inventory and total "
            "fixed-deployment primary score."
        ),
    }


def _caveats() -> list[str]:
    return [
        (
            "Answer alignment is lexical overlap with a response's own source query, not "
            "correctness, factuality, helpfulness, or human-rated response quality."
        ),
        (
            "Nested development evaluation reduces tuning bias but does not guarantee "
            "performance on the hidden test."
        ),
        (
            "Paired intervals condition on the completed stitched selections and do not "
            "rerun nested selection inside each bootstrap sample."
        ),
        (
            "Fixed-deployment volume diagnostics reuse all development queries and are "
            "behavior checks, not held-out effectiveness estimates."
        ),
        (
            "Inventory association is a bias-risk diagnostic, not proof that response volume "
            "causes ranking quality or unfairness."
        ),
        (
            "Passing all gates advances the method only for explicit owner consideration and "
            "does not release or merge another pull request."
        ),
    ]


def _report_notes() -> dict[str, Any]:
    return {
        "audience": "product stakeholders and first-time topic readers",
        "delivery_mode": "portable_html",
        "required_structure": [
            "title",
            "executive summary",
            "task and metric primer",
            "key findings with visual evidence",
            "preregistered decision",
            "recommended next steps",
            "further questions",
            "reproducibility and caveats",
        ],
        "visuals": [
            {
                "section": "six-system comparison",
                "question": "Which approach ranks candidate models best?",
                "family": "comparison and ranking",
                "type": "full-width grouped horizontal bar",
                "rows": 12,
            },
            {
                "section": "paired query outcomes",
                "question": (
                    "How often does source-query-first beat, tie, or trail each baseline?"
                ),
                "family": "composition",
                "type": "full-width 100% stacked horizontal bar",
                "rows": 30,
            },
            {
                "section": "explicit-zero opportunity audit",
                "question": "Which historical response cells can carry lexical evidence?",
                "family": "comparison",
                "type": "full-width horizontal bar",
                "rows": 4,
            },
            {
                "section": "volume gates",
                "question": "Did equal source-query opportunity reduce both volume signals?",
                "family": "comparison and benchmark",
                "type": "full-width grouped bar",
                "rows": 8,
            },
            {
                "section": "inventory representation",
                "question": "Where are fixed-deployment top-10 placements concentrated?",
                "family": "comparison and composition",
                "type": "full-width grouped bar",
                "rows": 16,
            },
        ],
        "non_chart_visuals": [
            "plain-language source-query-first signal flow",
            "nested-validation holdout flow",
            "three-row exact K-selection table",
        ],
        "mobile_qa": (
            "Inspect enhanced and semantic readers at 390px width in light and dark modes; "
            "verify no page overflow, wrapped approach labels, readable bottom legends, "
            "source controls, and the complete novice reading path."
        ),
        "table_policy": (
            "Only the three-row K-selection lookup is visible. Exact chart values remain in "
            "semantic chart tables and source affordances rather than a duplicate broad table."
        ),
    }


def _metric_definitions() -> list[str]:
    return [
        "nDCG@10: macro top-10 ranking quality over equally weighted queries.",
        "MRR: macro reciprocal rank of the first positively judged model.",
        "Paired interval: 10,000 query-level bootstrap resamples of source-query-first minus the named baseline, conditional on finished nested selections.",
        "Inventory correlation: absolute Spearman association between frozen BASIC response inventory and total fixed-deployment primary score.",
        "Largest-quartile share: share of 3,220 fixed-deployment top-10 placements received by the frozen largest BASIC-inventory quartile.",
    ]


def _require_number(mapping: Mapping[str, Any], key: str) -> float:
    if key not in mapping:
        raise ValueError(f"missing required numeric diagnostic {key!r}")
    value = mapping[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"diagnostic {key!r} is not numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"diagnostic {key!r} is not finite")
    return number


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
