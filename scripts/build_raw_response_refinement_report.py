#!/usr/bin/env python3
"""Build the compact Markdown/JSON raw-response refinement result report."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

from trec_mlm.raw_response_refinement_evaluation import (
    LOCKED_QRELS_SHA256,
    MATCHED_PARENT_METRICS,
    primary_decision,
    validate_refinement_repeat_receipt,
)


DISCLOSURE = (
    "Development effectiveness is post-hoc confirmation, not independent "
    "hidden-test evidence."
)
SYSTEMS = frozenset(
    {
        "dense_reference",
        "dense_profile_equal",
        "profile_only",
        "raw_hierarchical",
        "raw_hierarchical_dense_equal",
    }
)
EVALUATED_KEYS = frozenset(
    {
        "status", "construction_aggregate_sha256",
        "construction_dependency_sha256", "construction_unchanged",
        "qrels_sha256", "repeat_receipt", "matched_parent_metrics", "metrics",
        "comparisons", "per_query_rows", "top_10_changed", "decision",
    }
)
NO_OP_KEYS = frozenset(
    {
        "status", "construction_aggregate_sha256",
        "construction_dependency_sha256", "construction_unchanged",
        "repeat_receipt", "top_10_changed", "decision",
    }
)
INTERVAL_KEYS = frozenset(
    {
        "mean_difference", "lower", "upper", "confidence_level", "n_resamples",
        "seed", "generator",
    }
)
COMPARISON_KEYS = frozenset(
    {
        "baseline", "candidate", "ndcg_at_10_delta", "mrr_delta",
        "ndcg_at_10_win_tie_loss", "mrr_win_tie_loss", "zero_transitions",
    }
)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_repeat(payload: Any, aggregate: str) -> None:
    if not isinstance(payload, dict):
        raise ValueError("evaluation repeat receipt must be an object")
    if aggregate == payload.get("first_aggregate_sha256"):
        execution_id = payload.get("first_execution_id")
    elif aggregate == payload.get("second_aggregate_sha256"):
        execution_id = payload.get("second_execution_id")
    else:
        raise ValueError("evaluation repeat receipt is not bound to construction")
    validate_refinement_repeat_receipt(payload, aggregate, execution_id)


def _validate_metric(payload: Any) -> None:
    if not isinstance(payload, dict) or set(payload) != {
        "ndcg_at_10", "mrr", "zero_ndcg_queries"
    }:
        raise ValueError("evaluation metric schema changed")
    if (
        not all(math.isfinite(float(payload[key])) for key in ("ndcg_at_10", "mrr"))
        or isinstance(payload["zero_ndcg_queries"], bool)
        or not isinstance(payload["zero_ndcg_queries"], int)
        or payload["zero_ndcg_queries"] < 0
    ):
        raise ValueError("evaluation metrics are invalid")


def _validate_interval(payload: Any) -> None:
    if (
        not isinstance(payload, dict)
        or set(payload) != INTERVAL_KEYS
        or payload.get("confidence_level") != 0.95
        or payload.get("n_resamples") != 10_000
        or payload.get("seed") != 2026
        or payload.get("generator") != "PCG64"
        or not all(
            math.isfinite(float(payload[key]))
            for key in ("mean_difference", "lower", "upper")
        )
    ):
        raise ValueError("evaluation paired interval schema changed")


def _validate_comparison(
    payload: Any, *, baseline: str, candidate: str
) -> None:
    if (
        not isinstance(payload, dict)
        or set(payload) != COMPARISON_KEYS
        or payload.get("baseline") != baseline
        or payload.get("candidate") != candidate
    ):
        raise ValueError("evaluation comparison identity changed")
    _validate_interval(payload["ndcg_at_10_delta"])
    _validate_interval(payload["mrr_delta"])
    for key in ("ndcg_at_10_win_tie_loss", "mrr_win_tie_loss"):
        value = payload[key]
        if (
            not isinstance(value, dict)
            or set(value) != {"wins", "ties", "losses", "tolerance"}
            or any(
                isinstance(value[name], bool) or not isinstance(value[name], int)
                for name in ("wins", "ties", "losses")
            )
            or sum(value[name] for name in ("wins", "ties", "losses")) != 322
            or value["tolerance"] != 1e-12
        ):
            raise ValueError("evaluation win/tie/loss schema changed")
    transitions = payload["zero_transitions"]
    if not isinstance(transitions, dict) or set(transitions) != {
        "zero_both", "dense_zero_candidate_nonzero",
        "dense_nonzero_candidate_zero", "nonzero_both", "query_count",
    }:
        raise ValueError("evaluation zero-transition schema changed")
    if (
        transitions["query_count"] != 322
        or sum(
            transitions[key]
            for key in (
                "zero_both", "dense_zero_candidate_nonzero",
                "dense_nonzero_candidate_zero", "nonzero_both",
            )
        )
        != 322
    ):
        raise ValueError("evaluation zero-transition inventory changed")


def _validate_evaluated(payload: dict[str, Any]) -> None:
    if set(payload) != EVALUATED_KEYS:
        raise ValueError("finalized evaluation schema changed")
    aggregate = payload.get("construction_aggregate_sha256")
    if (
        not _is_sha256(aggregate)
        or not _is_sha256(payload.get("construction_dependency_sha256"))
        or payload.get("construction_unchanged") is not True
        or payload.get("qrels_sha256") != LOCKED_QRELS_SHA256
        or payload.get("matched_parent_metrics") != MATCHED_PARENT_METRICS
    ):
        raise ValueError("finalized evaluation locked identities changed")
    _validate_repeat(payload["repeat_receipt"], aggregate)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict) or set(metrics) != SYSTEMS:
        raise ValueError("finalized evaluation system inventory changed")
    for value in metrics.values():
        _validate_metric(value)
    if any(metrics[name] != expected for name, expected in MATCHED_PARENT_METRICS.items()):
        raise ValueError("finalized evaluation parent metrics changed")
    comparisons = payload.get("comparisons")
    if not isinstance(comparisons, dict) or set(comparisons) != {
        "primary", "secondary", "dense_diagnostics"
    }:
        raise ValueError("finalized evaluation comparison schema changed")
    _validate_comparison(
        comparisons["primary"],
        baseline="dense_profile_equal",
        candidate="raw_hierarchical_dense_equal",
    )
    _validate_comparison(
        comparisons["secondary"],
        baseline="profile_only",
        candidate="raw_hierarchical",
    )
    dense = comparisons["dense_diagnostics"]
    if not isinstance(dense, dict) or set(dense) != {
        "raw_hierarchical", "raw_hierarchical_dense_equal"
    }:
        raise ValueError("dense diagnostic comparison schema changed")
    for candidate, comparison in dense.items():
        _validate_comparison(
            comparison, baseline="dense_reference", candidate=candidate
        )
    rows = payload.get("per_query_rows")
    if not isinstance(rows, list) or len(rows) != 322 * len(SYSTEMS):
        raise ValueError("finalized per-query inventory changed")
    identities: dict[str, set[str]] = {system: set() for system in SYSTEMS}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "query_id", "system", "ndcg_at_10", "reciprocal_rank"
        }:
            raise ValueError("finalized per-query row schema changed")
        system = row["system"]
        query_id = row["query_id"]
        if system not in SYSTEMS or not isinstance(query_id, str) or not query_id:
            raise ValueError("finalized per-query identity changed")
        if not all(
            math.isfinite(float(row[key]))
            for key in ("ndcg_at_10", "reciprocal_rank")
        ):
            raise ValueError("finalized per-query metrics are invalid")
        identities[system].add(query_id)
    if any(len(query_ids) != 322 for query_ids in identities.values()) or len(
        {frozenset(query_ids) for query_ids in identities.values()}
    ) != 1:
        raise ValueError("finalized per-query coverage changed")
    if payload.get("top_10_changed") not in (
        {"primary": True, "secondary": True},
        {"primary": True, "secondary": False},
        {"primary": False, "secondary": True},
    ):
        raise ValueError("evaluated result must pass the frozen no-op gate")
    decision = payload.get("decision")
    if (
        not isinstance(decision, dict)
        or set(decision) != {"decision", "gates", "integrity_checks", "values"}
        or decision.get("decision") not in {"advance", "diagnostic", "stop"}
    ):
        raise ValueError("finalized evaluation decision schema changed")
    gates = decision["gates"]
    integrity = decision["integrity_checks"]
    values = decision["values"]
    if (
        not isinstance(gates, dict)
        or set(gates) != {
            "ndcg_interval_lower_above_zero", "mrr_not_materially_lower",
            "zero_count_reduced", "integrity_passes",
        }
        or any(not isinstance(value, bool) for value in gates.values())
        or not isinstance(integrity, dict)
        or set(integrity) != {
            "qrel_independence", "dependency_bindings", "embedding_contract",
            "inventory", "semantic_reconstruction", "trec_completeness",
            "artifact_validation", "repeatability", "parent_metrics",
            "construction_hash_binding",
        }
        or any(value is not True for value in integrity.values())
        or not isinstance(values, dict)
        or set(values) != {
            "ndcg_interval_lower", "ndcg_point_delta", "mrr_delta", "zero_count",
            "zero_rescues",
        }
        or not all(
            math.isfinite(float(values[key]))
            for key in ("ndcg_interval_lower", "ndcg_point_delta", "mrr_delta")
        )
    ):
        raise ValueError("finalized evaluation decision evidence changed")
    primary = comparisons["primary"]
    expected_decision = primary_decision(
        ndcg_interval_lower=primary["ndcg_at_10_delta"]["lower"],
        ndcg_point_delta=primary["ndcg_at_10_delta"]["mean_difference"],
        mrr_delta=primary["mrr_delta"]["mean_difference"],
        zero_count=metrics["raw_hierarchical_dense_equal"]["zero_ndcg_queries"],
        zero_rescues=primary["zero_transitions"][
            "dense_zero_candidate_nonzero"
        ],
        integrity_checks=integrity,
    )
    if decision != expected_decision:
        raise ValueError("finalized evaluation decision does not match frozen rule")


def _validate_no_op(payload: dict[str, Any]) -> None:
    if set(payload) != NO_OP_KEYS:
        raise ValueError("finalized no-op evaluation schema changed")
    aggregate = payload.get("construction_aggregate_sha256")
    if (
        not _is_sha256(aggregate)
        or not _is_sha256(payload.get("construction_dependency_sha256"))
        or payload.get("construction_unchanged") is not True
        or payload.get("top_10_changed") != {"primary": False, "secondary": False}
        or not isinstance(payload.get("decision"), dict)
        or set(payload["decision"]) != {"decision", "reason"}
        or payload["decision"].get("decision") != "stop"
    ):
        raise ValueError("finalized no-op evaluation identities changed")
    _validate_repeat(payload["repeat_receipt"], aggregate)


def _validate_evaluation(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("evaluation must be a JSON object")
    if payload.get("status") == "evaluated":
        _validate_evaluated(payload)
    elif payload.get("status") == "stop-no-op":
        _validate_no_op(payload)
    else:
        raise ValueError("evaluation must be a valid finalized result")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--json", required=True)
    return parser


def _number(value: Any) -> str:
    return f"{float(value):.12f}"


def _markdown(payload: dict[str, Any]) -> str:
    status = payload.get("status")
    decision = payload.get("decision", {}).get("decision", "unknown")
    lines = [
        "# E5 raw historical-response refinement results",
        "",
        DISCLOSURE,
        "",
        f"- Evaluation status: `{status}`",
        f"- Frozen decision: **{decision}**",
        f"- Construction SHA-256: `{payload.get('construction_aggregate_sha256', 'n/a')}`",
    ]
    if payload.get("qrels_sha256"):
        lines.append(f"- Sealed qrels SHA-256: `{payload['qrels_sha256']}`")
    metrics = payload.get("metrics", {})
    if metrics:
        lines.extend(["", "## Development metrics", "", "| System | nDCG@10 | MRR | Zero nDCG |", "|---|---:|---:|---:|"])
        for name in sorted(metrics):
            item = metrics[name]
            lines.append(
                f"| `{name}` | {_number(item['ndcg_at_10'])} | "
                f"{_number(item['mrr'])} | {int(item['zero_ndcg_queries'])} |"
            )
    primary = payload.get("comparisons", {}).get("primary")
    if primary:
        ndcg = primary["ndcg_at_10_delta"]
        mrr = primary["mrr_delta"]
        lines.extend(
            [
                "",
                "## Prespecified primary comparison",
                "",
                f"`{primary['candidate']}` versus `{primary['baseline']}`:",
                "",
                f"- nDCG@10 delta: {_number(ndcg['mean_difference'])} "
                f"(95% CI {_number(ndcg['lower'])} to {_number(ndcg['upper'])})",
                f"- MRR delta: {_number(mrr['mean_difference'])} "
                f"(95% CI {_number(mrr['lower'])} to {_number(mrr['upper'])})",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The frozen rule determines the decision above. Secondary and dense "
            "comparisons are diagnostics and do not select another variant.",
            "",
        ]
    )
    return "\n".join(lines)


def _write_pair_noreplace(
    markdown: Path, markdown_text: str, output_json: Path, payload: object
) -> None:
    destinations = (markdown, output_json)
    if markdown.resolve() == output_json.resolve():
        raise ValueError("Markdown and JSON destinations must differ")
    if any(path.exists() for path in destinations):
        raise FileExistsError("report output already exists")
    staged: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for destination, content in (
            (markdown, markdown_text),
            (output_json, json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"),
        ):
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, name = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=destination.parent
            )
            temporary = Path(name)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            staged.append((temporary, destination))
        for temporary, destination in staged:
            os.link(temporary, destination)
            published.append(destination)
    except BaseException:
        for destination in published:
            destination.unlink(missing_ok=True)
        raise
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        payload = _validate_evaluation(
            json.loads(Path(args.evaluation).read_text(encoding="utf-8"))
        )
        report_payload = {
            "artifact_type": "raw_response_refinement_results",
            "version": 1,
            "evidence_scope": DISCLOSURE,
            "evaluation": payload,
        }
        _write_pair_noreplace(
            Path(args.markdown),
            _markdown(payload),
            Path(args.json),
            report_payload,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
