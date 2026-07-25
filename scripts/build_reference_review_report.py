#!/usr/bin/env python3
"""Rebuild the frozen 400-question four-model reference review report."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from trec_mlm.reference_adjudication import aggregate_rankings


SYSTEM_ROWS = [
    {
        "system": "Gemini 3.6 Flash",
        "top_tier_share": 0.8225,
        "serious_error_rate": 0.0700,
        "fractional_top_credit": 0.3679,
        "mean_correctness": 3.7275,
        "mean_tier": 1.2075,
    },
    {
        "system": "Perplexity Sonar",
        "top_tier_share": 0.6225,
        "serious_error_rate": 0.2150,
        "fractional_top_credit": 0.2583,
        "mean_correctness": 3.3000,
        "mean_tier": 1.5050,
    },
    {
        "system": "Qwen3.6-27B",
        "top_tier_share": 0.5875,
        "serious_error_rate": 0.3125,
        "fractional_top_credit": 0.2075,
        "mean_correctness": 2.9575,
        "mean_tier": 1.6075,
    },
    {
        "system": "Gemma 4 31B",
        "top_tier_share": 0.5075,
        "serious_error_rate": 0.3625,
        "fractional_top_credit": 0.1663,
        "mean_correctness": 2.7875,
        "mean_tier": 1.7175,
    },
]

SLICE_ROWS = [
    {"slice": "Challenge score >=2", "questions": 253, "gemini_top": 0.870, "gemini_error": 0.075, "perplexity_top": 0.660, "perplexity_error": 0.225, "qwen_top": 0.510, "qwen_error": 0.407, "gemma_top": 0.451, "gemma_error": 0.458},
    {"slice": "Long or multi-hop", "questions": 75, "gemini_top": 0.907, "gemini_error": 0.080, "perplexity_top": 0.760, "perplexity_error": 0.173, "qwen_top": 0.573, "qwen_error": 0.400, "gemma_top": 0.547, "gemma_error": 0.427},
    {"slice": "Numeric", "questions": 113, "gemini_top": 0.867, "gemini_error": 0.062, "perplexity_top": 0.646, "perplexity_error": 0.248, "qwen_top": 0.487, "qwen_error": 0.442, "gemma_top": 0.469, "gemma_error": 0.451},
    {"slice": "Time-sensitive", "questions": 22, "gemini_top": 0.864, "gemini_error": 0.091, "perplexity_top": 0.545, "perplexity_error": 0.364, "qwen_top": 0.318, "qwen_error": 0.591, "gemma_top": 0.409, "gemma_error": 0.364},
    {"slice": "Known failure analogue", "questions": 14, "gemini_top": 0.929, "gemini_error": 0.071, "perplexity_top": 0.714, "perplexity_error": 0.143, "qwen_top": 0.357, "qwen_error": 0.500, "gemma_top": 0.500, "gemma_error": 0.500},
    {"slice": "Challenge score <2", "questions": 147, "gemini_top": 0.741, "gemini_error": 0.061, "perplexity_top": 0.558, "perplexity_error": 0.197, "qwen_top": 0.721, "qwen_error": 0.150, "gemma_top": 0.605, "gemma_error": 0.197},
]

COST_ROWS = [
    {"system": "Gemini 3.6 Flash", "completed": "400 / 400", "pilot_cost": 1.4867, "projected_all": 28.84, "projected_remaining": 27.35, "note": "Zero final failures"},
    {"system": "Qwen3.6-27B", "completed": "396 / 400", "pilot_cost": 2.0240, "projected_all": 39.26, "projected_remaining": None, "note": "Four hard-case failures"},
    {"system": "Gemma 4 31B", "completed": "400 / 400", "pilot_cost": 0.0369, "projected_all": 0.72, "projected_remaining": None, "note": "Lowest cost; highest error rate"},
]


def source(
    source_id: str,
    label: str,
    path: str,
    description: str,
    *,
    sql: str,
) -> dict:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "duckdb",
            "language": "sql",
            "description": description,
            "sql": sql,
            "filters": [
                "Frozen 400-question purposive stratified sample.",
                "Provider identities unblinded only after all judgments were locked.",
            ],
            "metric_definitions": [
                "Top-tier share is the fraction of questions where a system appears in the first tie-preserving tier.",
                "Serious-error rate is the fraction scored major or fatal in the locked pointwise review.",
            ],
            "tables_used": [path],
        },
    }


def ranking_rows(packet_path: Path, rankings_path: Path) -> list[dict]:
    packet_records = [
        json.loads(line)
        for line in packet_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ranking_records = [
        json.loads(line)
        for line in rankings_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    packet_ids = [str(row.get("question_id")) for row in packet_records]
    ranking_ids = [str(row.get("question_id")) for row in ranking_records]
    if (
        len(packet_ids) != 400
        or len(set(packet_ids)) != 400
        or len(ranking_ids) != 400
        or len(set(ranking_ids)) != 400
    ):
        raise ValueError("frozen review report requires exactly 400 unique questions")
    if set(packet_ids) != set(ranking_ids):
        raise ValueError("packet and ranking question IDs differ")
    summary = aggregate_rankings(ranking_records)["systems"]
    system_keys = {
        "Gemini 3.6 Flash": "gemini",
        "Perplexity Sonar": "perplexity",
        "Qwen3.6-27B": "qwen27",
        "Gemma 4 31B": "gemma31",
    }
    for expected in SYSTEM_ROWS:
        actual = summary[system_keys[expected["system"]]]
        for field in (
            "top_tier_share",
            "serious_error_rate",
            "fractional_top_credit",
            "mean_correctness",
            "mean_tier",
        ):
            if not math.isclose(
                float(actual[field]),
                float(expected[field]),
                rel_tol=0.0,
                abs_tol=0.000051,
            ):
                raise ValueError(
                    f"frozen system metric drift for {expected['system']}/{field}"
                )
    packets = {str(row["question_id"]): row for row in packet_records}
    rows = []
    for row in ranking_records:
        packet = packets[str(row["question_id"])]
        tiers = [" = ".join(tier) for tier in row["ranking"]]
        rows.append(
            {
                "question_id": row["question_id"],
                "cluster_id": packet.get("cluster_id"),
                "strata": ", ".join(packet.get("strata") or []),
                "challenge_count": sum(
                    bool(value)
                    for value in (packet.get("challenge_indicators") or {}).values()
                ),
                "confidence": row["confidence"],
                "ranking": " > ".join(tiers),
                "top_tier": tiers[0],
            }
        )
    return sorted(rows, key=lambda row: str(row["question_id"]))


def artifact(rows: list[dict]) -> dict:
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    sources = [
        source(
            "ranking_summary",
            "Locked four-model ranking summary",
            "artifacts/reference_answer_validation/ranking_summary.json",
            "Aggregates the 400 locked, unblinded per-question rankings.",
            sql="WITH systems(system, top_tier_share, serious_error_rate, fractional_top_credit, mean_correctness, mean_tier) AS (VALUES ('Gemini 3.6 Flash',0.8225,0.0700,0.3679,3.7275,1.2075),('Perplexity Sonar',0.6225,0.2150,0.2583,3.3000,1.5050),('Qwen3.6-27B',0.5875,0.3125,0.2075,2.9575,1.6075),('Gemma 4 31B',0.5075,0.3625,0.1663,2.7875,1.7175)) SELECT * FROM systems;",
        ),
        source(
            "per_question",
            "Locked per-question rankings",
            "artifacts/reference_answer_validation/per_question_rankings.jsonl",
            "Supplies tie-preserving tiers, pointwise scores, pairwise outcomes, confidence, and the reviewed slice summary.",
            sql="SELECT * FROM read_json_auto('artifacts/reference_answer_validation/per_question_rankings.jsonl', format='newline_delimited');",
        ),
        source(
            "generation_caches",
            "Challenger generation caches",
            "artifacts/reference_answer_validation/generations",
            "Supplies exact coverage, token usage, finish reasons, elapsed time, provider/model identifiers, and the derived cost rows.",
            sql="WITH costs(system, completed, pilot_cost, projected_all, projected_remaining, note) AS (VALUES ('Gemini 3.6 Flash','400 / 400',1.4867,28.84,27.35,'Zero final failures'),('Qwen3.6-27B','396 / 400',2.0240,39.26,NULL,'Four hard-case failures'),('Gemma 4 31B','400 / 400',0.0369,0.72,NULL,'Lowest cost; highest error rate')) SELECT * FROM costs;",
        ),
        source(
            "sample_manifest",
            "Frozen clustered sample manifest",
            "artifacts/reference_answer_validation/sample_manifest.json",
            "Defines the 400-question sample, 192 clusters, strata, and challenge indicators.",
            sql="SELECT * FROM read_json_auto('artifacts/reference_answer_validation/sample_manifest.json');",
        ),
        {
            "id": "trec_2025_review",
            "label": "TREC 2025 Million LLMs literature review",
            "path": "reports/trec_2025_mllm_literature_review.md",
        },
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "TREC 2026: choosing a reference-answer model",
            "description": "Blind review of Perplexity, Gemini 3.6 Flash, Qwen3.6-27B, and Gemma 4 31B on 400 clustered discovery questions.",
            "generatedAt": generated,
            "charts": [
                {
                    "id": "top_tier",
                    "title": "Top-tier share by system",
                    "subtitle": "400-question purposive stratified sample; ties preserved.",
                    "type": "horizontalBar",
                    "dataset": "systems",
                    "sourceId": "ranking_summary",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "system", "type": "nominal", "label": "System"},
                        "y": {"field": "top_tier_share", "type": "quantitative", "label": "Top-tier share"},
                        "tooltip": [
                            {"field": "mean_correctness", "type": "quantitative", "label": "Mean correctness (0-4)"},
                            {"field": "mean_tier", "type": "quantitative", "label": "Mean tier"},
                        ],
                    },
                    "settings": {"orientation": "horizontal", "sort": "ascending", "showValues": True, "palette": {"kind": "single", "root": "blue"}},
                    "layout": "full",
                },
                {
                    "id": "serious_errors",
                    "title": "Major-or-fatal error rate by system",
                    "subtitle": "Same 400 locked reviews; lower is better.",
                    "type": "horizontalBar",
                    "dataset": "systems",
                    "sourceId": "ranking_summary",
                    "valueFormat": "percent",
                    "encodings": {
                        "x": {"field": "system", "type": "nominal", "label": "System"},
                        "y": {"field": "serious_error_rate", "type": "quantitative", "label": "Serious-error rate"},
                        "tooltip": [{"field": "fractional_top_credit", "type": "quantitative", "label": "Fractional top credit"}],
                    },
                    "settings": {"orientation": "horizontal", "sort": "descending", "showValues": True, "palette": {"kind": "single", "root": "orange"}},
                    "layout": "full",
                },
            ],
            "tables": [
                {
                    "id": "slice_results",
                    "title": "Performance remains separated in difficult slices",
                    "subtitle": "Top-tier share and serious-error rate; every row uses the same locked blind review.",
                    "dataset": "slices",
                    "sourceId": "per_question",
                    "defaultSort": {"field": "questions", "direction": "desc"},
                    "columns": [
                        {"field": "slice", "label": "Slice", "type": "text"},
                        {"field": "questions", "label": "Questions", "format": "number"},
                        {"field": "gemini_top", "label": "Gemini top", "format": "percent"},
                        {"field": "gemini_error", "label": "Gemini error", "format": "percent"},
                        {"field": "perplexity_top", "label": "Perplexity top", "format": "percent"},
                        {"field": "perplexity_error", "label": "Perplexity error", "format": "percent"},
                        {"field": "qwen_top", "label": "Qwen top", "format": "percent"},
                        {"field": "qwen_error", "label": "Qwen error", "format": "percent"},
                        {"field": "gemma_top", "label": "Gemma top", "format": "percent"},
                        {"field": "gemma_error", "label": "Gemma error", "format": "percent"},
                    ],
                },
                {
                    "id": "costs",
                    "title": "Actual pilot cost and full-corpus projection",
                    "subtitle": "Projected from actual pilot token use at verified listed prices; USD.",
                    "dataset": "costs",
                    "sourceId": "generation_caches",
                    "defaultSort": {"field": "pilot_cost", "direction": "desc"},
                    "columns": [
                        {"field": "system", "label": "System", "type": "text"},
                        {"field": "completed", "label": "Completed", "type": "text"},
                        {"field": "pilot_cost", "label": "Pilot cost", "format": "currency"},
                        {"field": "projected_all", "label": "Projected 7,759", "format": "currency"},
                        {"field": "projected_remaining", "label": "Remaining 7,359", "format": "currency"},
                        {"field": "note", "label": "Operational note", "type": "text"},
                    ],
                },
                {
                    "id": "all_rankings",
                    "title": "All 400 locked per-question rankings",
                    "subtitle": "Sanitized audit table; question and answer text remain in private ignored artifacts.",
                    "dataset": "all_rankings",
                    "sourceId": "per_question",
                    "defaultSort": {"field": "question_id", "direction": "asc"},
                    "density": "dense",
                    "columns": [
                        {"field": "question_id", "label": "Question ID", "type": "text"},
                        {"field": "cluster_id", "label": "Cluster", "format": "number"},
                        {"field": "strata", "label": "Strata", "type": "text"},
                        {"field": "challenge_count", "label": "Challenge flags", "format": "number"},
                        {"field": "confidence", "label": "Confidence", "type": "text"},
                        {"field": "ranking", "label": "Tie-preserving ranking", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# TREC 2026: choosing a reference-answer model"},
                {"id": "summary", "type": "markdown", "sourceId": "ranking_summary", "body": "## Technical summary\n\n- **Choose Gemini 3.6 Flash.** It reached the top tier on **329/400 (82.3%)** questions and had **28/400 (7.0%)** major-or-fatal errors.\n- **The margin is decision-sized.** Perplexity was second at 62.3% top-tier and 21.5% serious errors; Qwen and Gemma were worse.\n- **Run the remaining 7,359 without blanket grounding.** Actual pilot tokens project to about **$27.35 standard**, above the current $10 authorization.\n- **Keep an escalation path.** On the 43 highest-risk questions, every system still made frequent serious errors."},
                {"id": "quality_heading", "type": "markdown", "body": "## Gemini leads on both selection and failure avoidance\n\nTop-tier share preserves substantive ties instead of forcing one winner. The error chart is the stricter safety view: it counts major and fatal factual failures."},
                {"id": "top_chart", "type": "chart", "chartId": "top_tier"},
                {"id": "top_readout", "type": "markdown", "sourceId": "ranking_summary", "body": "Gemini's 20.0-point top-tier advantage over Perplexity exceeds the predeclared five-point practical threshold. There were 118 all-system ties and 100 sole-winner questions; Gemini had 49 sole wins."},
                {"id": "error_chart", "type": "chart", "chartId": "serious_errors"},
                {"id": "error_readout", "type": "markdown", "sourceId": "ranking_summary", "body": "Gemini's serious-error rate was 14.5 points below Perplexity and more than 24 points below either local-weight challenger. The dominant errors were wrong entities, dates, places, titles, and counts—not cosmetic style differences."},
                {"id": "slices_heading", "type": "markdown", "body": "## The advantage grows where reference generation is hardest\n\nThe sample intentionally overweights difficult questions, so these are sample diagnostics rather than population accuracy estimates. Qwen is competitive on the easier slice but degrades sharply on multi-hop, numeric, and time-sensitive questions."},
                {"id": "slices", "type": "table", "tableId": "slice_results"},
                {"id": "cost_heading", "type": "markdown", "body": "## Cost favors Gemini once quality is included\n\nGemma is nearly free but too error-prone. Qwen cost more than Gemini in the pilot because it generated 628k completion tokens and still failed on four hard questions. Standard Gemini generation remains under the user's original $50 budget."},
                {"id": "cost_table", "type": "table", "tableId": "costs"},
                {"id": "method", "type": "markdown", "sourceId": "sample_manifest", "body": "## What was tested\n\nThe frozen sample contains 400 of 7,759 questions across 192 non-overlapping semantic clusters. It deliberately includes numeric, time-sensitive, entity-dense, underspecified, high-disagreement, and known-failure-analogue questions. Provider identities were deterministically blinded per question. The advisor reviewed all 400, Codex inspected all 400 and revised 34 judgments, and the map was opened only after exact coverage and schema validation."},
                {"id": "limits", "type": "markdown", "body": "## What this does not prove\n\n- Discovery has no organizer answer-correctness or relevance judgments. This is a disciplined rough pass, not gold-label accuracy.\n- The purposive sample overweights hard questions; do not extrapolate its error rates to all 7,759.\n- The 43-case high-risk queue has not received a new source-by-source web verification pass. Existing Perplexity citations can support selective resolution.\n- A selected reference is a pivot for later pairwise judging, not an oracle."},
                {"id": "ranking_detail_heading", "type": "markdown", "body": "## Per-question audit detail\n\nThe table below exposes every locked ranking without publishing question or answer text. Ties are shown with `=` and tier boundaries with `>`."},
                {"id": "ranking_detail", "type": "table", "tableId": "all_rankings"},
                {"id": "prior_teams", "type": "markdown", "sourceId": "trec_2025_review", "body": "## The remaining gap is downstream judging, not another generator\n\nThe repository has already tried query retrieval, response BM25/RM3, semantic profiles, and fusion. Competitive TREC 2025 teams also used several pieces we have not completed: strong-judge pseudo-labels over candidate answers, pairwise or Quicksort-like ranking, log-probability confidence features, learned hybrid routers, and neural reranking of retrieved responses. The next meaningful experiment is to turn the new references into leakage-safe pointwise and selective pairwise signals, then test whether they improve held-out model ranking."},
                {"id": "next", "type": "markdown", "body": "## Recommended next steps\n\n1. Raise the Gemini standard-API cap to **$35** and generate the remaining 7,359 answers, reusing the 400 cached outputs.\n2. Keep Perplexity answers and citations beside Gemini rather than overwriting them.\n3. Escalate time-sensitive, low-confidence, and material Gemini–Perplexity disagreements; avoid blanket paid search.\n4. Calibrate the later pairwise judge on these 400 locked reviews, preserve ties, randomize A/B order, and reverse close comparisons."},
                {"id": "questions", "type": "markdown", "body": "## Further questions\n\n- Can a cheap local model identify the escalation set without missing Gemini's serious errors?\n- Does synthesis improve the 43-case high-risk queue or only make unsupported answers more persuasive?\n- How stable are candidate rankings when the reference is swapped for another top-tier answer or omitted?"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "systems": SYSTEM_ROWS,
                "slices": SLICE_ROWS,
                "costs": COST_ROWS,
                "all_rankings": rows,
            },
        },
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--packet",
        type=Path,
        default=Path("artifacts/reference_answer_validation/blind_comparisons.jsonl"),
    )
    parser.add_argument(
        "--rankings",
        type=Path,
        default=Path("artifacts/reference_answer_validation/per_question_rankings.jsonl"),
    )
    parser.add_argument("--rankings-tsv", type=Path)
    args = parser.parse_args()
    rows = ranking_rows(args.packet, args.rankings)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact(rows), indent=2) + "\n", encoding="utf-8")
    if args.rankings_tsv:
        args.rankings_tsv.parent.mkdir(parents=True, exist_ok=True)
        with args.rankings_tsv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
