"""Sealed post-hoc evaluation and reporting for the GroupRank Stage 2 pilot."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from math import comb, fsum, isfinite
from pathlib import Path
from typing import Any

from trec_mlm.grouprank import GROUPING_SEEDS, parse_grouprank_scores
from trec_mlm.grouprank_pilot import (
    GroupRankPhaseAResult,
    load_frozen_grouprank,
)
from trec_mlm.llm_ranker_pilot import validate_pilot_selection
from trec_mlm.metrics import EvaluationResult, evaluate_rankings


DEFAULT_STAGE2_REPORT_PATH = Path("artifacts/llm_label_ranker/pilot/stage2/report.json")
_METHOD = "grouprank"
_BASELINE_METHODS = (
    "e5_contextual",
    "qwen4b_contextual",
    "pointwise",
)
_BASELINE_HEADLINES: Mapping[str, tuple[int, float, float]] = {
    "e5_contextual": (8, 0.7232, 0.6361),
    "qwen4b_contextual": (8, 0.6648, 0.5585),
    "pointwise": (5, 0.4820, 0.3209),
}
_METHOD_LABELS = {
    _METHOD: "GroupRank 7B",
    "e5_contextual": "Contextual E5",
    "qwen4b_contextual": "Contextual Qwen4B",
    "pointwise": "Pointwise Qwen",
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_bytes(path: str | Path, *, label: str) -> tuple[dict[str, Any], bytes]:
    source = Path(path)
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise ValueError(f"{label} is missing") from error
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value, payload


def _selection_inventory(
    selection: Mapping[str, Any],
) -> tuple[tuple[str, ...], dict[str, frozenset[str]]]:
    questions = selection["questions"]
    query_ids = tuple(entry["query_id"] for entry in questions)
    candidates = {
        entry["query_id"]: frozenset(entry["candidate_ids"]) for entry in questions
    }
    return query_ids, candidates


def _validate_external_selection(
    phase_a: GroupRankPhaseAResult,
    selection_path: str | Path,
) -> dict[str, Any]:
    selection, payload = _read_json_bytes(
        selection_path,
        label="frozen GroupRank selection",
    )
    validate_pilot_selection(
        selection,
        expected_stage=2,
        expected_candidate_count=20,
    )
    if selection != phase_a.selection:
        raise ValueError("external selection disagrees with the Phase A artifact")
    expected_sha256 = phase_a.config.get("selection_sha256")
    if _sha256_bytes(payload) != expected_sha256:
        raise ValueError("external selection SHA256 disagrees with Phase A")
    return {
        "path": str(Path(selection_path)),
        "sha256": _sha256_bytes(payload),
    }


def _load_matching_judgments(
    phase_a: GroupRankPhaseAResult,
    judgments_path: str | Path,
) -> tuple[dict[str, Any], dict[str, dict[str, float]], bytes]:
    judgments, payload = _read_json_bytes(
        judgments_path,
        label="sealed Stage 2 judgments",
    )
    validated = validate_pilot_selection(
        judgments,
        expected_stage=2,
        expected_candidate_count=20,
    )
    if validated["manifest_kind"] != "judgments":
        raise ValueError("sealed Stage 2 judgments use the selection schema")

    expected_query_ids, expected_candidates = _selection_inventory(phase_a.selection)
    judgment_entries = judgments["questions"]
    actual_query_ids = tuple(entry["query_id"] for entry in judgment_entries)
    if actual_query_ids != expected_query_ids:
        raise ValueError("sealed judgment query inventory disagrees with Phase A")

    qrels: dict[str, dict[str, float]] = {}
    for entry in judgment_entries:
        query_id = entry["query_id"]
        if frozenset(entry["grades"]) != expected_candidates[query_id]:
            raise ValueError(
                f"sealed judgment candidate inventory disagrees with Phase A for {query_id}"
            )
        grades: dict[str, float] = {}
        for candidate_id, grade in entry["grades"].items():
            if isinstance(grade, bool) or not isinstance(grade, (int, float)):
                raise ValueError("sealed judgment grades must be finite numbers")
            value = float(grade)
            if not isfinite(value):
                raise ValueError("sealed judgment grades must be finite numbers")
            grades[candidate_id] = value
        qrels[query_id] = grades
    return judgments, qrels, payload


def _finite_metric(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _validate_baseline_method_question(
    payload: object,
    *,
    method: str,
    candidates: frozenset[str],
    relevant: frozenset[str],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"frozen Stage 2 {method} question metrics changed")
    top_three = payload.get("top_three")
    relevant_ranks = payload.get("relevant_ranks")
    if (
        not isinstance(top_three, list)
        or len(top_three) != 3
        or len(set(top_three)) != 3
        or not set(top_three).issubset(candidates)
    ):
        raise ValueError(f"frozen Stage 2 {method} top-three inventory changed")
    if not isinstance(relevant_ranks, Mapping) or set(relevant_ranks) != set(relevant):
        raise ValueError(f"frozen Stage 2 {method} relevant-rank inventory changed")
    if any(
        type(rank) is not int or rank < 1 or rank > 20
        for rank in relevant_ranks.values()
    ):
        raise ValueError(f"frozen Stage 2 {method} relevant ranks changed")
    best_rank = min(relevant_ranks.values())
    if payload.get("best_relevant_rank") != best_rank:
        raise ValueError(f"frozen Stage 2 {method} best relevant rank changed")
    top3_relevant = any(rank <= 3 for rank in relevant_ranks.values())
    if payload.get("top3_relevant") is not top3_relevant:
        raise ValueError(f"frozen Stage 2 {method} top-three result changed")
    reciprocal_rank = _finite_metric(
        payload.get("reciprocal_rank"),
        label=f"frozen Stage 2 {method} reciprocal rank",
    )
    if reciprocal_rank != 1.0 / best_rank:
        raise ValueError(f"frozen Stage 2 {method} reciprocal rank changed")
    ndcg = _finite_metric(
        payload.get("ndcg_at_10"),
        label=f"frozen Stage 2 {method} nDCG@10",
    )
    return {
        "top_three": list(top_three),
        "relevant_ranks": dict(relevant_ranks),
        "best_relevant_rank": best_rank,
        "ndcg_at_10": ndcg,
        "reciprocal_rank": reciprocal_rank,
        "top3_relevant": top3_relevant,
    }


def _load_frozen_baselines(
    phase_a: GroupRankPhaseAResult,
    qrels: Mapping[str, Mapping[str, float]],
    baseline_report_path: str | Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]], bytes]:
    report, payload = _read_json_bytes(
        baseline_report_path,
        label="frozen Stage 2 report",
    )
    if (
        report.get("version") != 1
        or report.get("stage") != 2
        or report.get("evidence_scope") != "qrel-assisted exploratory"
        or report.get("question_count") != 12
        or report.get("candidate_count_per_question") != 20
    ):
        raise ValueError("frozen Stage 2 report contract changed")
    if report.get("selection_sha256") != phase_a.config.get("selection_sha256"):
        raise ValueError("frozen Stage 2 report selection disagrees with Phase A")

    metrics = report.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("frozen Stage 2 report metrics changed")
    baseline_metrics: dict[str, dict[str, Any]] = {}
    for method, (
        expected_top3,
        expected_ndcg,
        expected_mrr,
    ) in _BASELINE_HEADLINES.items():
        item = metrics.get(method)
        if not isinstance(item, Mapping):
            raise ValueError(f"frozen Stage 2 {method} metrics changed")
        top3 = item.get("top3_relevant_count")
        ndcg = _finite_metric(
            item.get("ndcg_at_10"), label=f"frozen Stage 2 {method} nDCG@10"
        )
        mrr = _finite_metric(item.get("mrr"), label=f"frozen Stage 2 {method} MRR")
        if (
            item.get("availability") != "available"
            or item.get("query_count") != 12
            or top3 != expected_top3
            or round(ndcg, 4) != expected_ndcg
            or round(mrr, 4) != expected_mrr
        ):
            raise ValueError(f"frozen Stage 2 {method} headline metrics changed")
        query_ids = item.get("top3_query_ids")
        if (
            not isinstance(query_ids, list)
            or len(query_ids) != expected_top3
            or len(set(query_ids)) != expected_top3
        ):
            raise ValueError(f"frozen Stage 2 {method} top-three queries changed")
        baseline_metrics[method] = {
            "availability": "available",
            "query_count": 12,
            "ndcg_at_10": ndcg,
            "mrr": mrr,
            "top3_relevant_count": top3,
            "top3_query_ids": list(query_ids),
        }

    selection_by_query = {
        entry["query_id"]: entry for entry in phase_a.selection["questions"]
    }
    questions = report.get("questions")
    if not isinstance(questions, list):
        raise ValueError("frozen Stage 2 question evidence changed")
    expected_query_ids = tuple(selection_by_query)
    if (
        tuple(row.get("query_id") for row in questions if isinstance(row, Mapping))
        != expected_query_ids
    ):
        raise ValueError("frozen Stage 2 question inventory changed")

    per_query: dict[str, dict[str, dict[str, Any]]] = {}
    for row in questions:
        if not isinstance(row, Mapping):
            raise ValueError("frozen Stage 2 question evidence changed")
        query_id = row["query_id"]
        selection_entry = selection_by_query[query_id]
        if row.get("question") != selection_entry["question"]:
            raise ValueError("frozen Stage 2 question text changed")
        relevant = frozenset(
            candidate_id for candidate_id, grade in qrels[query_id].items() if grade > 0
        )
        reported_relevant = row.get("relevant_grades")
        if not isinstance(reported_relevant, Mapping) or {
            candidate_id: float(grade)
            for candidate_id, grade in reported_relevant.items()
        } != {candidate_id: qrels[query_id][candidate_id] for candidate_id in relevant}:
            raise ValueError("frozen Stage 2 relevant grades disagree with judgments")
        methods = row.get("methods")
        if not isinstance(methods, Mapping):
            raise ValueError("frozen Stage 2 per-query methods changed")
        per_query[query_id] = {
            method: _validate_baseline_method_question(
                methods.get(method),
                method=method,
                candidates=frozenset(selection_entry["candidate_ids"]),
                relevant=relevant,
            )
            for method in _BASELINE_METHODS
        }

    for method in _BASELINE_METHODS:
        method_rows = [per_query[query_id][method] for query_id in expected_query_ids]
        if (
            fsum(row["ndcg_at_10"] for row in method_rows) / 12
            != baseline_metrics[method]["ndcg_at_10"]
            or fsum(row["reciprocal_rank"] for row in method_rows) / 12
            != baseline_metrics[method]["mrr"]
            or sum(row["top3_relevant"] for row in method_rows)
            != baseline_metrics[method]["top3_relevant_count"]
            or [
                query_id
                for query_id in expected_query_ids
                if per_query[query_id][method]["top3_relevant"]
            ]
            != baseline_metrics[method]["top3_query_ids"]
        ):
            raise ValueError(f"frozen Stage 2 {method} question metrics disagree")
    return baseline_metrics, per_query, payload


def _metric_summary(
    result: EvaluationResult,
    *,
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    validated_output_count: int,
) -> dict[str, Any]:
    top3_query_ids = [
        query_id
        for query_id in qrels
        if any(
            qrels[query_id].get(candidate_id, 0.0) > 0
            for candidate_id in rankings[query_id][:3]
        )
    ]
    return {
        "availability": "available",
        "query_count": len(qrels),
        "validated_output_count": validated_output_count,
        "ndcg_at_10": result.macro_ndcg_at_k,
        "mrr": result.mean_reciprocal_rank,
        "top3_relevant_count": len(top3_query_ids),
        "top3_query_ids": top3_query_ids,
    }


def build_grouprank_gate(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Apply the frozen three-condition exploratory GroupRank feasibility gate."""

    required = {
        "validated_output_count",
        "top3_relevant_count",
        "ndcg_at_10",
    }
    if not required.issubset(metrics):
        raise ValueError("GroupRank gate metrics are incomplete")
    output_count = metrics["validated_output_count"]
    top3_count = metrics["top3_relevant_count"]
    ndcg = _finite_metric(metrics["ndcg_at_10"], label="GroupRank gate nDCG@10")
    if type(output_count) is not int or type(top3_count) is not int:
        raise ValueError("GroupRank gate counts must be integers")
    conditions: dict[str, dict[str, Any]] = {
        "all_outputs_valid": {
            "observed": output_count,
            "required": 48,
            "passes": output_count == 48,
        },
        "top3_at_least_eight": {
            "observed": top3_count,
            "required": 8,
            "query_count": 12,
            "passes": top3_count >= 8,
        },
        "ndcg_at_least_e5": {
            "observed": ndcg,
            "required": 0.7232,
            "passes": ndcg >= 0.7232,
        },
    }
    return {
        "name": "grouprank_7b_stage2_feasibility",
        "evidence_scope": "qrel-assisted exploratory",
        "rule": (
            "Pass only with 48 valid outputs, a relevant label in the top three "
            "for at least 8 of 12 questions, and nDCG@10 of at least 0.7232."
        ),
        "conditions": conditions,
        "passes": all(condition["passes"] for condition in conditions.values()),
    }


def _repetition_scores(
    phase_a: GroupRankPhaseAResult,
) -> dict[str, dict[str, dict[int, int]]]:
    observations: dict[str, dict[str, dict[int, int]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for record in phase_a.raw_outputs:
        handles = tuple(record["handle_to_candidate"])
        scores = parse_grouprank_scores(
            record["raw_text"],
            expected_handles=handles,
        )
        query_id = record["query_id"]
        repetition = record["repetition"]
        for handle, score in scores.items():
            candidate_id = record["handle_to_candidate"][handle]
            if repetition in observations[query_id][candidate_id]:
                raise ValueError("GroupRank report found duplicate repetition scores")
            observations[query_id][candidate_id][repetition] = score

    expected_query_ids, expected_candidates = _selection_inventory(phase_a.selection)
    result: dict[str, dict[str, dict[int, int]]] = {}
    for query_id in expected_query_ids:
        query_observations = observations.get(query_id, {})
        if set(query_observations) != set(expected_candidates[query_id]) or any(
            set(repetitions) != set(GROUPING_SEEDS)
            for repetitions in query_observations.values()
        ):
            raise ValueError("GroupRank report repetition inventory changed")
        result[query_id] = {
            candidate_id: dict(query_observations[candidate_id])
            for candidate_id in sorted(query_observations)
        }
    return result


def _tie_details(
    scores: Mapping[str, float],
) -> tuple[dict[str, int], list[dict[str, Any]]]:
    by_score: dict[float, list[str]] = defaultdict(list)
    for candidate_id, score in scores.items():
        value = float(score)
        if not isfinite(value):
            raise ValueError("GroupRank report scores must be finite")
        by_score[value].append(candidate_id)
    groups = [
        {
            "score": score,
            "candidate_ids": sorted(candidate_ids),
        }
        for score, candidate_ids in sorted(by_score.items(), reverse=True)
        if len(candidate_ids) > 1
    ]
    sizes = [
        len(candidate_ids)
        for candidate_ids in by_score.values()
        if len(candidate_ids) > 1
    ]
    return (
        {
            "score_groups": len(groups),
            "candidates": sum(sizes),
            "candidate_pairs": sum(comb(size, 2) for size in sizes),
        },
        groups,
    )


def build_grouprank_report(
    artifact_dir: str | Path,
    *,
    judgments_path: str | Path,
    baseline_report_path: str | Path = DEFAULT_STAGE2_REPORT_PATH,
    selection_path: str | Path | None = None,
    code_revision: str | None = None,
) -> dict[str, Any]:
    """Evaluate one completely validated Phase A artifact against sealed evidence."""

    # This call is deliberately the first file-accessing operation in the function.
    # It must return successfully before selection, judgments, or baselines are opened.
    phase_a = load_frozen_grouprank(Path(artifact_dir))

    artifact_revision = phase_a.config.get("code_revision")
    if code_revision is not None and code_revision != artifact_revision:
        raise ValueError("GroupRank evaluation code revision disagrees with Phase A")
    selection_record = (
        _validate_external_selection(phase_a, selection_path)
        if selection_path is not None
        else {
            "path": str(Path(artifact_dir) / "selection.json"),
            "sha256": phase_a.config.get("selection_sha256"),
        }
    )
    _judgments, qrels, judgment_bytes = _load_matching_judgments(
        phase_a,
        judgments_path,
    )
    baseline_metrics, baseline_questions, baseline_bytes = _load_frozen_baselines(
        phase_a,
        qrels,
        baseline_report_path,
    )

    evaluation = evaluate_rankings(phase_a.rankings, qrels, k=10)
    grouprank_metrics = _metric_summary(
        evaluation,
        rankings=phase_a.rankings,
        qrels=qrels,
        validated_output_count=len(phase_a.raw_outputs),
    )
    repetition_scores = _repetition_scores(phase_a)

    selection_by_query = {
        entry["query_id"]: entry for entry in phase_a.selection["questions"]
    }
    question_rows: list[dict[str, Any]] = []
    for query_id, entry in selection_by_query.items():
        ranking = list(phase_a.rankings[query_id])
        relevant_grades = {
            candidate_id: grade
            for candidate_id, grade in qrels[query_id].items()
            if grade > 0
        }
        if not relevant_grades:
            raise ValueError(f"GroupRank query {query_id!r} has no relevant candidate")
        relevant_ranks = {
            candidate_id: ranking.index(candidate_id) + 1
            for candidate_id in relevant_grades
        }
        query_metric = evaluation.per_query[query_id]
        grouprank_question = {
            "top_three": ranking[:3],
            "relevant_ranks": relevant_ranks,
            "best_relevant_rank": min(relevant_ranks.values()),
            "ndcg_at_10": query_metric.ndcg_at_k,
            "reciprocal_rank": query_metric.reciprocal_rank,
            "top3_relevant": any(rank <= 3 for rank in relevant_ranks.values()),
        }
        methods = {
            _METHOD: grouprank_question,
            **baseline_questions[query_id],
        }
        comparisons = {
            method: {
                "best_relevant_rank_delta": (
                    grouprank_question["best_relevant_rank"]
                    - methods[method]["best_relevant_rank"]
                ),
                "ndcg_at_10_delta": (
                    grouprank_question["ndcg_at_10"] - methods[method]["ndcg_at_10"]
                ),
                "reciprocal_rank_delta": (
                    grouprank_question["reciprocal_rank"]
                    - methods[method]["reciprocal_rank"]
                ),
            }
            for method in _BASELINE_METHODS
        }
        score_differences = {
            candidate_id: repetitions[GROUPING_SEEDS[1]]
            - repetitions[GROUPING_SEEDS[0]]
            for candidate_id, repetitions in repetition_scores[query_id].items()
        }
        absolute_differences = [abs(value) for value in score_differences.values()]
        tie_counts, tied_scores = _tie_details(phase_a.candidate_scores[query_id])
        question_rows.append(
            {
                "query_id": query_id,
                "question": entry["question"],
                "qrel_assisted": True,
                "relevant_grades": relevant_grades,
                "methods": methods,
                "grouprank_vs_baselines": comparisons,
                "repetition_score_difference_direction": (
                    f"repetition_{GROUPING_SEEDS[1]}_minus_{GROUPING_SEEDS[0]}"
                ),
                "repetition_score_differences": score_differences,
                "repetition_agreement": {
                    "exact_score_count": sum(
                        difference == 0 for difference in score_differences.values()
                    ),
                    "mean_absolute_score_difference": fsum(absolute_differences)
                    / len(absolute_differences),
                    "maximum_absolute_score_difference": max(absolute_differences),
                },
                "tie_counts": tie_counts,
                "tied_scores": tied_scores,
            }
        )

    gate = build_grouprank_gate(grouprank_metrics)
    comparisons = {
        method: {
            "ndcg_at_10_delta": (
                grouprank_metrics["ndcg_at_10"] - baseline_metrics[method]["ndcg_at_10"]
            ),
            "mrr_delta": grouprank_metrics["mrr"] - baseline_metrics[method]["mrr"],
            "top3_relevant_count_delta": (
                grouprank_metrics["top3_relevant_count"]
                - baseline_metrics[method]["top3_relevant_count"]
            ),
        }
        for method in _BASELINE_METHODS
    }
    disposition = (
        "The exploratory GroupRank gate passed, but no expansion was started; "
        "discuss the sealed pilot evidence before any larger run."
        if gate["passes"]
        else "The exploratory GroupRank gate did not pass. No expansion was started."
    )
    receipt_path = Path(artifact_dir) / "ranking_receipt.json"
    receipt_sha256 = _sha256(receipt_path) if receipt_path.is_file() else None
    return {
        "version": 1,
        "stage": 2,
        "method": _METHOD,
        "evidence_scope": "qrel-assisted exploratory",
        "caveat": (
            "The candidate sets and sealed judgments were selected with development "
            "qrels; this is exploratory feasibility evidence, not an unbiased comparison."
        ),
        "question_count": len(question_rows),
        "candidate_count_per_question": 20,
        "selection_sha256": phase_a.config.get("selection_sha256"),
        "method_status": {
            _METHOD: "complete_validated_phase_a",
            **{method: "complete_frozen_baseline" for method in _BASELINE_METHODS},
        },
        "metrics": {
            _METHOD: grouprank_metrics,
            **baseline_metrics,
        },
        "comparisons": comparisons,
        "questions": question_rows,
        "scaled_feasibility_gate": gate,
        "decision": {
            "gate_passed": gate["passes"],
            "automatic_continuation": False,
            "expansion_started": False,
            "plain_language": disposition,
        },
        "recomputation_inputs": {
            "phase_a": {
                "path": str(Path(artifact_dir)),
                "ranking_receipt_sha256": receipt_sha256,
                "code_revision": artifact_revision,
                "validated_output_count": len(phase_a.raw_outputs),
            },
            "selection": selection_record,
            "judgments": {
                "path": str(Path(judgments_path)),
                "sha256": _sha256_bytes(judgment_bytes),
            },
            "baseline_report": {
                "path": str(Path(baseline_report_path)),
                "sha256": _sha256_bytes(baseline_bytes),
            },
        },
    }


def render_grouprank_report_markdown(report: Mapping[str, Any]) -> str:
    """Render a concise deterministic Markdown summary of a complete report."""

    if (
        report.get("stage") != 2
        or report.get("method") != _METHOD
        or report.get("evidence_scope") != "qrel-assisted exploratory"
        or report.get("question_count") != 12
        or len(report.get("questions", ())) != 12
    ):
        raise ValueError("GroupRank report is incomplete")
    metrics = report["metrics"]
    gate = report["scaled_feasibility_gate"]
    lines = [
        "# GroupRank 7B: Stage 2 Pilot",
        "",
        "> Qrel-assisted exploratory development evidence; candidate sets and sealed judgments were selected with development qrels.",
        "",
        "## Disposition",
        "",
        report["decision"]["plain_language"],
        "",
        "No expansion was started by this report.",
        "",
        "## Headline metrics (restricted 20-label sets)",
        "",
        "| Method | nDCG@10 | MRR | Relevant in top 3 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method in (_METHOD, "e5_contextual", "qwen4b_contextual", "pointwise"):
        item = metrics[method]
        lines.append(
            f"| {_METHOD_LABELS[method]} | {item['ndcg_at_10']:.4f} | "
            f"{item['mrr']:.4f} | {item['top3_relevant_count']}/12 |"
        )
    lines.extend(
        [
            "",
            "## Frozen exploratory gate",
            "",
            f"**Result: {'PASS' if gate['passes'] else 'FAIL'}.** {gate['rule']}",
            "",
            "| Condition | Observed | Required | Pass |",
            "| --- | ---: | ---: | --- |",
            f"| Valid outputs | {gate['conditions']['all_outputs_valid']['observed']} | 48 | {'yes' if gate['conditions']['all_outputs_valid']['passes'] else 'no'} |",
            f"| Relevant in top 3 | {gate['conditions']['top3_at_least_eight']['observed']}/12 | 8/12 | {'yes' if gate['conditions']['top3_at_least_eight']['passes'] else 'no'} |",
            f"| nDCG@10 | {gate['conditions']['ndcg_at_least_e5']['observed']:.4f} | 0.7232 | {'yes' if gate['conditions']['ndcg_at_least_e5']['passes'] else 'no'} |",
            "",
            "## Question-level evidence",
            "",
            "Best-rank deltas are GroupRank minus baseline; negative favors GroupRank. Repetition differences are grouping seed 667 minus grouping seed 666.",
            "",
            "| Query | Relevant rank | GroupRank top 3 | Mean absolute repetition difference | Tied candidate pairs | Rank delta vs E5 / Qwen4B / pointwise |",
            "| --- | ---: | --- | ---: | ---: | ---: |",
        ]
    )
    for row in report["questions"]:
        grouprank = row["methods"][_METHOD]
        comparisons = row["grouprank_vs_baselines"]
        lines.append(
            f"| `{row['query_id']}` | {grouprank['best_relevant_rank']} | "
            f"{', '.join(grouprank['top_three'])} | "
            f"{row['repetition_agreement']['mean_absolute_score_difference']:.3f} | "
            f"{row['tie_counts']['candidate_pairs']} | "
            f"{comparisons['e5_contextual']['best_relevant_rank_delta']} / "
            f"{comparisons['qwen4b_contextual']['best_relevant_rank_delta']} / "
            f"{comparisons['pointwise']['best_relevant_rank_delta']} |"
        )
    inputs = report["recomputation_inputs"]
    lines.extend(
        [
            "",
            "## Recalculation receipt",
            "",
            f"- Phase A ranking receipt SHA-256: `{inputs['phase_a']['ranking_receipt_sha256']}`",
            f"- Frozen selection SHA-256: `{report['selection_sha256']}`",
            f"- Sealed judgments SHA-256: `{inputs['judgments']['sha256']}`",
            f"- Frozen Stage 2 report SHA-256: `{inputs['baseline_report']['sha256']}`",
            "",
        ]
    )
    return "\n".join(lines)


def write_grouprank_report(
    report: Mapping[str, Any],
    *,
    report_path: str | Path,
    markdown_path: str | Path,
) -> dict[str, str]:
    """Write sorted JSON and deterministic Markdown result artifacts."""

    json_destination = Path(report_path)
    markdown_destination = Path(markdown_path)
    json_destination.parent.mkdir(parents=True, exist_ok=True)
    markdown_destination.parent.mkdir(parents=True, exist_ok=True)
    json_destination.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_destination.write_text(
        render_grouprank_report_markdown(report),
        encoding="utf-8",
    )
    return {
        "report_path": str(json_destination),
        "markdown_path": str(markdown_destination),
        "report_sha256": _sha256(json_destination),
        "markdown_sha256": _sha256(markdown_destination),
    }
