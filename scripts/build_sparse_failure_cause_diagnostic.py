#!/usr/bin/env python3
"""Decompose the 21 fusion misses and 54 no-signal sparse failures.

The diagnostic is post-qrel and development-only.  Its three no-signal cause
labels are observable evidence proxies, not claims about organizer intent or
candidate correctness.  The exact equal-weight ranking is reproduced from the
saved all-pairs explorer before any decomposition is reported.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CASES = ROOT / "reports/trec_2026_sparse_failure_analogue_cases.tsv"
DEFAULT_EXPLORER = (
    ROOT / "reports/trec_2026_candidate_semantic_profiles_query_explorer.html"
)
DEFAULT_JSON = ROOT / "reports/trec_2026_sparse_failure_cause_diagnostic.json"
DEFAULT_CASE_TSV = ROOT / "reports/trec_2026_sparse_failure_cause_cases.tsv"
DEFAULT_WEIGHT_TSV = ROOT / "reports/trec_2026_sparse_failure_weight_sensitivity.tsv"

EXPLORER_DATA_RE = re.compile(
    r'<script id="explorer-data" type="application/json">(.*?)</script>',
    re.DOTALL,
)
EXPECTED_PRIMARY_NDCG = 0.4202139921941393
WEIGHT_RATIOS = (0.01, 0.10, 0.25, 0.50, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 10.0, 100.0)

# Interpretive spot-check layer.  These categories describe the visible query,
# official candidate handle, and released history.  They are intentionally kept
# separate from the deterministic cause proxies calculated below.
NO_SIGNAL_ARCHETYPES: dict[str, frozenset[str]] = {
    "query_underspecification": frozenset(
        {
            "56e17248cd28a01900c6794a",
            "56f7e518aef2371900625c4e",
            "5727cd664b864d1900163d7a",
            "5730256db2c2fd1400568967",
        }
    ),
    "counter_domain_or_opaque_profile": frozenset(
        """56cf7b8a4df3c31400b0d820 56e1207fcd28a01900c67620
        571a16d64faf5e1900b8a894 5726668a708984140094c4cc
        57296f3caf94a219006aa3fc 5a736c1255429901807db044
        5a73e6315542993a88ae2e8b 5a7a41055542990783324edf
        5a7cc378554299452d57ba28 5a7ebac955429934daa2fc55
        5a85e07a5542994775f60676 5a88823d5542993e715ac041
        5a88abb15542994846c1ceb2 5a8901845542995153361258
        5a8a959e55429930ff3c0e08 5a8d7cce55429941ae14dfd0
        5a8e03f05542995085b3737a 5a8f49545542997ba9cb321f
        5a90bdcf55429916514e754a 5ab3cd385542992ade7c6ea3
        5ab42f2e55429942dd415ebb 5ab525c255429942dd415fff
        5ab727c25542993667793f56 5abbe0765542992ccd8e7fcc
        5ac2fe7155429921a00ab0c6 5adddbe85542990dbb2f7ec4
        5adf77645542995ec70e901e 5ae076b055429945ae959356
        5ae33a5b5542990afbd1e0e9 5ae356505542992e3233c387""".split()
    ),
    "surface_topic_collision": frozenset(
        """57326051e17f3d1400422905 5a76b0cc554299373536015e
        5ab801465542995dae37ea13 5abb05b5554299232ef4a3b7
        5ac1228e554299294b2190b5 5ade0b785542990dbb2f7f34""".split()
    ),
    "entity_specific_or_multihop": frozenset(
        """571a7b2b10f8ca1400305097 5a7298aa5542992359bc312e
        5a7cfa4755429907fabef07f 5a8b558b55429949d91db551
        5a8c6c9b554299653c1aa062 5ac39ed5554299391541382a
        5ade7f165542992fa25da796""".split()
    ),
    "history_coverage_gap": frozenset(
        {
            "5732ab8ed6dcfa19001e8a6d",
            "5abd92435542993062266ccc",
            "5ae36cfc5542991a06ce99c2",
        }
    ),
    "local_refusal_gap": frozenset({"5a7f253a5542993067513661"}),
    "semantic_granularity_mismatch": frozenset(
        {
            "5727aad82ca10214002d9341",
            "5a7c6d325542996dd594b92f",
            "5adea4f155429975fa854f31",
        }
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--explorer", type=Path, default=DEFAULT_EXPLORER)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--output-case-tsv", type=Path, default=DEFAULT_CASE_TSV)
    parser.add_argument("--output-weight-tsv", type=Path, default=DEFAULT_WEIGHT_TSV)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_repository_input(path: Path) -> Path:
    """Resolve an input path and require portable repo-relative provenance."""

    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"input must be inside the repository: {path}") from error
    return resolved


def load_explorer(path: Path) -> dict[str, Any]:
    payload = path.read_text(encoding="utf-8")
    match = EXPLORER_DATA_RE.search(payload)
    if match is None:
        raise ValueError(f"{path}: explorer-data payload not found")
    return json.loads(html.unescape(match.group(1)))


def ordered_pairs(query: dict[str, Any], ratio: float) -> list[dict[str, Any]]:
    pairs = query["pairs"]
    names = np.asarray([pair["candidate_id"] for pair in pairs])
    scores = np.asarray(
        [
            1.0 / (60 + pair["dense_rank"])
            + ratio / (60 + pair["profile_rank"])
            for pair in pairs
        ],
        dtype=np.float32,
    )
    order = np.lexsort((names, -scores))
    return [pairs[int(index)] for index in order]


def ndcg_at_10(pairs: list[dict[str, Any]]) -> float:
    gains = [float(pair["grade"]) for pair in pairs[:10]]
    dcg = math.fsum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sorted((float(pair["grade"]) for pair in pairs), reverse=True)[:10]
    ideal_dcg = math.fsum(
        gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1)
    )
    return dcg / ideal_dcg if ideal_dcg else 0.0


def rank_band(rank: int) -> str:
    if rank <= 20:
        return "11–20"
    if rank <= 50:
        return "21–50"
    if rank <= 100:
        return "51–100"
    return "101–150"


def fusion_signal_strength(rank: int) -> str:
    if rank <= 3:
        return "top-three unilateral signal"
    if rank <= 5:
        return "rank-four-to-five unilateral signal"
    return "rank-six-to-ten unilateral signal"


def no_signal_proxy(row: dict[str, Any]) -> tuple[str, str]:
    component_rank = min(int(row["profile_rank"]), int(row["dense_rank"]))
    attempts = int(row["official_top3_attempts"])
    best_similarity = float(row["best_attempted_similarity"])
    nearest_has_positive = row["nearest_dev_has_official_positive"] == "True"
    if component_rank <= 20:
        return (
            "near_component_miss",
            "One input ranks the official positive 11–20, but neither reaches top ten.",
        )
    if attempts == 0 and best_similarity < 0.10:
        return (
            "no_close_attempted_history",
            "Outside the component top 20, with zero attempts on the three closest "
            "discovery analogues and no attempted lexical analogue at similarity 0.10.",
        )
    if nearest_has_positive:
        raise ValueError(
            "residual no-signal case unexpectedly has nearest-query label agreement"
        )
    return (
        "analogue_label_mismatch",
        "Some historical evidence exists, but the nearest development label set "
        "does not contain the official positive.",
    )


def no_signal_archetype(query_id: str) -> str:
    matches = [name for name, query_ids in NO_SIGNAL_ARCHETYPES.items() if query_id in query_ids]
    if len(matches) != 1:
        raise ValueError(f"expected one interpretive archetype for {query_id}, found {matches}")
    return matches[0]


def top10_hit(query: dict[str, Any], positive_id: str, ratio: float) -> bool:
    return any(
        pair["candidate_id"] == positive_id for pair in ordered_pairs(query, ratio)[:10]
    )


def recovery_boundary(
    query: dict[str, Any], positive_id: str, component_state: str
) -> float:
    if component_state == "profile_top10_only":
        low, high = 1.0, 100.0
        if top10_hit(query, positive_id, low) or not top10_hit(query, positive_id, high):
            raise ValueError("profile-only recovery boundary is not bracketed")
        for _ in range(60):
            middle = (low + high) / 2
            if top10_hit(query, positive_id, middle):
                high = middle
            else:
                low = middle
        return high
    if component_state == "dense_top10_only":
        low, high = 0.001, 1.0
        if not top10_hit(query, positive_id, low) or top10_hit(query, positive_id, high):
            raise ValueError("dense-only recovery boundary is not bracketed")
        for _ in range(60):
            middle = (low + high) / 2
            if top10_hit(query, positive_id, middle):
                low = middle
            else:
                high = middle
        return low
    raise ValueError(f"unsupported component state: {component_state}")


def main() -> None:
    args = parse_args()
    args.cases = resolve_repository_input(args.cases)
    args.explorer = resolve_repository_input(args.explorer)
    with args.cases.open(encoding="utf-8", newline="") as handle:
        case_rows = list(csv.DictReader(handle, delimiter="\t"))
    explorer = load_explorer(args.explorer)
    queries = explorer["queries"]
    query_map = {query["query_id"]: query for query in queries}

    if len(case_rows) != 75 or len(queries) != 322:
        raise ValueError("expected 75 failure cases and 322 explorer queries")
    if not set(row["query_id"] for row in case_rows) <= set(query_map):
        raise ValueError("case query inventory is not contained in the explorer")

    exact_rank_mismatches = 0
    primary_ndcg_values: list[float] = []
    for query in queries:
        recomputed = ordered_pairs(query, 1.0)
        saved = sorted(query["pairs"], key=lambda pair: pair["primary_rank"])
        exact_rank_mismatches += [pair["candidate_id"] for pair in recomputed] != [
            pair["candidate_id"] for pair in saved
        ]
        primary_ndcg_values.append(ndcg_at_10(recomputed))
    reproduced_primary_ndcg = math.fsum(primary_ndcg_values) / len(primary_ndcg_values)
    if exact_rank_mismatches or not math.isclose(
        reproduced_primary_ndcg, EXPECTED_PRIMARY_NDCG, abs_tol=1e-12
    ):
        raise ValueError("saved primary ranking did not reproduce exactly")

    output_cases: list[dict[str, Any]] = []
    fusion_cases: list[dict[str, Any]] = []
    no_signal_cases: list[dict[str, Any]] = []
    for row in case_rows:
        query = query_map[row["query_id"]]
        positive = next(pair for pair in query["pairs"] if pair["grade"] > 0)
        if positive["candidate_id"] != row["official_positive"]:
            raise ValueError("official positive differs between case table and explorer")
        primary_tenth = next(pair for pair in query["pairs"] if pair["primary_rank"] == 10)
        component_rank = min(int(row["profile_rank"]), int(row["dense_rank"]))
        opposing_rank = max(int(row["profile_rank"]), int(row["dense_rank"]))
        base = {
            "query_id": row["query_id"],
            "query_text": row["query_text"],
            "official_positive": row["official_positive"],
            "official_grade": int(float(row["official_grade"])),
            "primary_rank": int(row["primary_rank"]),
            "profile_rank": int(row["profile_rank"]),
            "dense_rank": int(row["dense_rank"]),
            "component_rank": component_rank,
            "opposing_rank": opposing_rank,
            "primary_score_gap_to_rank_10": float(
                primary_tenth["primary_score"] - positive["primary_score"]
            ),
            "official_top3_attempts": int(row["official_top3_attempts"]),
            "best_attempted_similarity": float(row["best_attempted_similarity"]),
            "nearest_dev_has_official_positive": row[
                "nearest_dev_has_official_positive"
            ]
            == "True",
        }
        if row["component_state"] == "neither_top10":
            bucket, definition = no_signal_proxy(row)
            archetype = no_signal_archetype(row["query_id"])
            base.update(
                {
                    "cohort": "no_signal_54",
                    "cause_bucket": bucket,
                    "cause_definition": definition,
                    "component_rank_band": rank_band(component_rank),
                    "interpretive_archetype": archetype,
                }
            )
            no_signal_cases.append(base)
        else:
            boundary = recovery_boundary(
                query, row["official_positive"], row["component_state"]
            )
            if row["component_state"] == "profile_top10_only":
                contradiction = (
                    "profile_mild"
                    if boundary <= 2.0
                    else "profile_moderate"
                    if boundary <= 5.0
                    else "profile_extreme"
                )
                recovery_rule = f"profile/dense weight ratio > {boundary:.6f}"
            else:
                contradiction = "dense_mild" if boundary > 0.60 else "dense_strong"
                recovery_rule = f"profile/dense weight ratio < {boundary:.6f}"
            base.update(
                {
                    "cohort": "fusion_21",
                    "cause_bucket": "cross_component_rrf_veto",
                    "cause_definition": (
                        "One component ranks the positive top ten while the other ranks "
                        "it 21–150; equal reciprocal-rank fusion favors candidates with "
                        "moderate support in both inputs."
                    ),
                    "component_state": row["component_state"],
                    "unilateral_signal_strength": fusion_signal_strength(component_rank),
                    "opposing_rank_band": rank_band(opposing_rank),
                    "contradiction_severity": contradiction,
                    "recovery_ratio_boundary": boundary,
                    "recovery_rule": recovery_rule,
                    "interpretive_archetype": "not_applicable_fusion_case",
                }
            )
            fusion_cases.append(base)
        output_cases.append(base)

    if len(fusion_cases) != 21 or len(no_signal_cases) != 54:
        raise ValueError("expected an exact 21/54 cohort split")
    if any(case["opposing_rank"] <= 20 for case in fusion_cases):
        raise ValueError("fusion-veto cohort contains an opposing rank at or above 20")

    failure_ids = set(query_map) & {row["query_id"] for row in case_rows}
    fusion_ids = {case["query_id"] for case in fusion_cases}
    weight_rows: list[dict[str, Any]] = []
    for ratio in WEIGHT_RATIOS:
        values: list[float] = []
        failure_hits = 0
        fusion_hits = 0
        other_sole_positive_hits = 0
        for query in queries:
            ranking = ordered_pairs(query, ratio)
            values.append(ndcg_at_10(ranking))
            positives = [pair for pair in ranking if pair["grade"] > 0]
            sole_positive = sum(pair["grade"] > 0 for pair in query["pairs"]) == 1
            top10_hit = sole_positive and positives[0] in ranking[:10]
            if query["query_id"] in failure_ids and top10_hit:
                failure_hits += 1
                fusion_hits += query["query_id"] in fusion_ids
            elif query["query_id"] not in failure_ids and top10_hit:
                other_sole_positive_hits += 1
        weight_rows.append(
            {
                "profile_to_dense_weight_ratio": ratio,
                "ndcg_at_10": math.fsum(values) / len(values),
                "failure_75_top10_hits": failure_hits,
                "fusion_21_top10_hits": fusion_hits,
                "other_105_sole_positive_top10_hits": other_sole_positive_hits,
            }
        )

    cause_counts = Counter(case["cause_bucket"] for case in no_signal_cases)
    if cause_counts != Counter(
        {
            "near_component_miss": 7,
            "no_close_attempted_history": 24,
            "analogue_label_mismatch": 23,
        }
    ):
        raise ValueError(f"unexpected no-signal cause counts: {cause_counts}")
    archetype_counts = Counter(
        case["interpretive_archetype"] for case in no_signal_cases
    )
    if archetype_counts != Counter(
        {
            "counter_domain_or_opaque_profile": 30,
            "entity_specific_or_multihop": 7,
            "surface_topic_collision": 6,
            "query_underspecification": 4,
            "history_coverage_gap": 3,
            "semantic_granularity_mismatch": 3,
            "local_refusal_gap": 1,
        }
    ):
        raise ValueError(f"unexpected interpretive archetype counts: {archetype_counts}")

    payload = {
        "version": 1,
        "artifact_type": "trec_2026_sparse_failure_cause_diagnostic",
        "status": "complete",
        "definitions": {
            "fusion_failure": (
                "The sole official positive is top ten in exactly one component and "
                "outside the equal-RRF top ten."
            ),
            "no_signal_failure": (
                "The sole official positive is outside the top ten in both components."
            ),
            "cause_proxy_warning": (
                "No-signal buckets are deterministic released-evidence proxies. They do "
                "not establish candidate correctness, qrel error, or organizer intent."
            ),
            "attempt": (
                "A historical response other than an exact refusal on one of the three "
                "closest MPNet discovery queries; attempts are not correctness."
            ),
        },
        "reproduction": {
            "query_count": len(queries),
            "candidate_count": len(explorer["candidate_ids"]),
            "exact_rank_mismatches": exact_rank_mismatches,
            "primary_ndcg_at_10_raw_graded_gain": reproduced_primary_ndcg,
        },
        "fusion_21": {
            "count": len(fusion_cases),
            "component_states": dict(
                Counter(case["component_state"] for case in fusion_cases)
            ),
            "unilateral_signal_strength": dict(
                Counter(case["unilateral_signal_strength"] for case in fusion_cases)
            ),
            "contradiction_severity": dict(
                Counter(case["contradiction_severity"] for case in fusion_cases)
            ),
            "opposing_rank_bands": dict(
                Counter(case["opposing_rank_band"] for case in fusion_cases)
            ),
            "primary_rank_bands": {
                "11–15": sum(case["primary_rank"] <= 15 for case in fusion_cases),
                "16–20": sum(
                    16 <= case["primary_rank"] <= 20 for case in fusion_cases
                ),
                "21–150": sum(case["primary_rank"] > 20 for case in fusion_cases),
            },
            "score_gap_to_rank_10": {
                "minimum": min(
                    case["primary_score_gap_to_rank_10"] for case in fusion_cases
                ),
                "median": float(
                    np.median(
                        [
                            case["primary_score_gap_to_rank_10"]
                            for case in fusion_cases
                        ]
                    )
                ),
                "maximum": max(
                    case["primary_score_gap_to_rank_10"] for case in fusion_cases
                ),
            },
        },
        "no_signal_54": {
            "count": len(no_signal_cases),
            "cause_proxy_counts": dict(cause_counts),
            "interpretive_archetype_counts": dict(archetype_counts),
            "component_rank_bands": dict(
                Counter(case["component_rank_band"] for case in no_signal_cases)
            ),
            "zero_attempts_on_top3_analogues": sum(
                case["official_top3_attempts"] == 0 for case in no_signal_cases
            ),
            "no_attempted_lexical_analogue_at_0_10": sum(
                case["best_attempted_similarity"] < 0.10
                for case in no_signal_cases
            ),
            "nearest_dev_label_excludes_official": sum(
                not case["nearest_dev_has_official_positive"]
                for case in no_signal_cases
            ),
            "grade_counts": dict(
                Counter(str(case["official_grade"]) for case in no_signal_cases)
            ),
        },
        "weight_sensitivity": weight_rows,
        "cases": output_cases,
        "limitations": [
            "Actual candidate responses to development targets are unavailable.",
            "Organizer assessor instructions, prompts, and judging rationale are unavailable.",
            "The weight grid is post-qrel sensitivity analysis, not an unbiased model selection.",
            "Interpretive archetypes are structured manual spot checks, not causal labels.",
            "Candidate identifiers are model identifiers and need not be literal topic labels.",
        ],
        "source_sha256": {
            str(args.cases.relative_to(ROOT)): sha256(args.cases),
            str(args.explorer.relative_to(ROOT)): sha256(args.explorer),
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    case_fields = (
        "query_id",
        "query_text",
        "official_positive",
        "official_grade",
        "cohort",
        "cause_bucket",
        "primary_rank",
        "profile_rank",
        "dense_rank",
        "component_rank",
        "opposing_rank",
        "primary_score_gap_to_rank_10",
        "official_top3_attempts",
        "best_attempted_similarity",
        "nearest_dev_has_official_positive",
        "contradiction_severity",
        "recovery_ratio_boundary",
        "interpretive_archetype",
    )
    with args.output_case_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=case_fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {field: case.get(field, "") for field in case_fields}
            for case in output_cases
        )

    with args.output_weight_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=tuple(weight_rows[0]),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(weight_rows)

    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_case_tsv": str(args.output_case_tsv),
                "output_weight_tsv": str(args.output_weight_tsv),
                "fusion_count": len(fusion_cases),
                "no_signal_count": len(no_signal_cases),
                "no_signal_cause_counts": dict(cause_counts),
                "reproduced_primary_ndcg_at_10": reproduced_primary_ndcg,
            }
        )
    )


if __name__ == "__main__":
    main()
