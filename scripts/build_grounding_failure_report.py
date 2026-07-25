#!/usr/bin/env python3
"""Rebuild the frozen 400-question Gemini grounding audit report."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "reports/trec_2026_gemini_grounding_failure_review.tsv"
MANIFEST = ROOT / "artifacts/reference_answer_validation/sample_manifest.json"
RANKINGS = ROOT / "artifacts/reference_answer_validation/per_question_rankings.jsonl"
GROUNDED = ROOT / "artifacts/reference_answer_validation/grounding/gemini36_grounded_failures_28.json"
REPORT_MD = ROOT / "reports/trec_2026_gemini_grounding_failure_review_2026-07-21.md"
ARTIFACT = ROOT / "reports/trec_2026_gemini_grounding_failure_review.artifact.json"


OUTCOME_LABELS = {
    "clear_rescue": "Clear grounding rescue",
    "partial_rescue": "Partial rescue",
    "no_rescue": "No rescue",
    "original_review_false_positive": "Original review false positive",
    "regeneration_only_rescue": "Regeneration-only rescue",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def gemini_perplexity_outcome(record: dict) -> str:
    for pair in record["pairwise"]:
        if {pair["left"], pair["right"]} != {"gemini", "perplexity"}:
            continue
        outcome = pair["outcome"]
        if outcome in {"uncertain_needs_evidence", "both_unacceptable"}:
            return "uncertain"
        if outcome == "tie":
            return "tie"
        gemini_wins = (
            pair["left"] == "gemini" and outcome == "left_better"
        ) or (pair["right"] == "gemini" and outcome == "right_better")
        return "gemini_better" if gemini_wins else "perplexity_better"
    raise ValueError(f"Gemini/Perplexity pair missing for {record['question_id']}")


def main() -> None:
    with LEDGER.open() as handle:
        ledger = list(csv.DictReader(handle, delimiter="\t"))
    sample = json.loads(MANIFEST.read_text())["questions"]
    rankings = load_jsonl(RANKINGS)
    grounded = json.loads(GROUNDED.read_text())

    sample_ids = [str(row.get("question_id")) for row in sample]
    ranking_ids = [str(row.get("question_id")) for row in rankings]
    if (
        len(sample_ids) != 400
        or len(set(sample_ids)) != 400
        or len(ranking_ids) != 400
        or len(set(ranking_ids)) != 400
    ):
        raise ValueError("grounding audit requires exactly 400 unique review questions")
    if set(sample_ids) != set(ranking_ids):
        raise ValueError("sample and ranking question IDs differ")
    if len(ledger) != 28 or len({row["question_id"] for row in ledger}) != 28:
        raise ValueError("Grounding review must contain 28 unique questions")
    if set(grounded["records"]) != {row["question_id"] for row in ledger}:
        raise ValueError("Grounded cache and review ledger IDs differ")

    outcome_counts = Counter(row["outcome"] for row in ledger)
    true_failures = {
        row["question_id"]
        for row in ledger
        if row["outcome"] != "original_review_false_positive"
    }
    rescueable = {
        row["question_id"]
        for row in ledger
        if row["outcome"] in {"clear_rescue", "partial_rescue"}
    }
    clear_rescues = {
        row["question_id"] for row in ledger if row["outcome"] == "clear_rescue"
    }
    search_used = sum(row["search_used"] == "true" for row in ledger)
    search_true_failures = sum(
        row["search_used"] == "true" and row["question_id"] in true_failures
        for row in ledger
    )

    # These are diagnostic upper bounds because the comparison labels were manually
    # reviewed. A local automated judge still needs calibration against them.
    selected = [
        row
        for row in rankings
        if gemini_perplexity_outcome(row) in {"perplexity_better", "uncertain"}
    ]
    selected_ids = {row["question_id"] for row in selected}
    selected_failures = len(selected_ids & true_failures)
    selected_rescueable = len(selected_ids & rescueable)

    high_disagreement_or_unresolved = [
        row
        for row in sample
        if row["challenge_indicators"]["high_disagreement"]
        or row["challenge_indicators"]["unresolved"]
    ]
    feature_selected_ids = {row["question_id"] for row in high_disagreement_or_unresolved}
    feature_failures = len(feature_selected_ids & true_failures)

    grounding_usage = {
        key: sum(
            record.get("request_metadata", {}).get(key, 0)
            for record in grounded["records"].values()
        )
        for key in (
            "grounding_query_count",
            "grounding_chunk_count",
            "grounding_support_count",
        )
    }
    average_queries_per_request = grounding_usage["grounding_query_count"] / len(ledger)
    average_queries_when_used = grounding_usage["grounding_query_count"] / search_used
    projected_selected = round(len(sample) and len(selected) / len(sample) * 7759)
    projected_search_queries = projected_selected * average_queries_per_request
    grounded_model_cost = 0.110061
    projected_incremental_model_cost = grounded_model_cost / 28 * projected_selected
    projected_total_two_pass = 28.84 + projected_incremental_model_cost
    blanket_queries = 7759 * average_queries_per_request
    projected_search_using_prompts = 7759 * search_used / len(ledger)
    blanket_billed_queries_low = max(0.0, projected_search_using_prompts - 5000) * (
        grounding_usage["grounding_query_count"] / search_used
    )
    blanket_billed_queries_high = max(0.0, 7759 - 5000) * average_queries_per_request
    blanket_search_surcharge_low = blanket_billed_queries_low * 14 / 1000
    blanket_search_surcharge_high = blanket_billed_queries_high * 14 / 1000
    blanket_model_cost = grounded_model_cost / 28 * 7759

    outcome_rows = [
        {
            "outcome": OUTCOME_LABELS[key],
            "questions": outcome_counts[key],
            "share": outcome_counts[key] / 28,
        }
        for key in OUTCOME_LABELS
    ]
    trigger_rows = [
        {
            "trigger": "Perplexity better or comparison unresolved",
            "selected": len(selected),
            "coverage": len(selected) / 400,
            "true_failures_caught": selected_failures,
            "failure_recall": selected_failures / len(true_failures),
            "rescueable_caught": selected_rescueable,
            "rescueable_recall": selected_rescueable / len(rescueable),
            "status": "Best observed signal; manual-label upper bound",
        },
        {
            "trigger": "High historical disagreement or unresolved Perplexity",
            "selected": len(high_disagreement_or_unresolved),
            "coverage": len(high_disagreement_or_unresolved) / 400,
            "true_failures_caught": feature_failures,
            "failure_recall": feature_failures / len(true_failures),
            "rescueable_caught": len(feature_selected_ids & rescueable),
            "rescueable_recall": len(feature_selected_ids & rescueable) / len(rescueable),
            "status": "Too broad for routing",
        },
    ]
    case_rows = [
        {
            "question_id": row["question_id"],
            "search_used": row["search_used"] == "true",
            "outcome": OUTCOME_LABELS[row["outcome"]],
            "confidence": row["confidence"],
            "review_note": row["review_note"],
        }
        for row in ledger
    ]

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    sources = [
        {
            "id": "grounding_review",
            "label": "Manual review of 28 Gemini failure cases",
            "path": "reports/trec_2026_gemini_grounding_failure_review.tsv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Reads the sanitized 28-case manual review ledger.",
                "sql": "SELECT question_id, search_used::BOOLEAN AS search_used, outcome, confidence, review_note FROM read_csv_auto('reports/trec_2026_gemini_grounding_failure_review.tsv', delim='\\t', header=true);",
                "filters": ["All 28 originally flagged Gemini serious-error cases"],
                "metric_definitions": ["Outcome is the mutually exclusive manual adjudication of the grounded rerun."],
                "tables_used": ["reports/trec_2026_gemini_grounding_failure_review.tsv"],
            },
        },
        {
            "id": "outcome_analysis",
            "label": "Grounded rerun outcome counts",
            "path": "reports/trec_2026_gemini_grounding_failure_review.tsv",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Counts the mutually exclusive manual outcomes over the full 28-case ledger.",
                "sql": "SELECT outcome, COUNT(*) AS questions, COUNT(*) / 28.0 AS share FROM read_csv_auto('reports/trec_2026_gemini_grounding_failure_review.tsv', delim='\\t', header=true) GROUP BY outcome;",
                "filters": ["All 28 originally flagged Gemini serious-error cases"],
                "metric_definitions": ["Share is questions in the outcome divided by 28."],
                "tables_used": ["reports/trec_2026_gemini_grounding_failure_review.tsv"],
            },
        },
        {
            "id": "grounded_cache",
            "label": "Gemini grounded generation receipt",
            "path": "artifacts/reference_answer_validation/grounding/gemini36_grounded_failures_28.json",
        },
        {
            "id": "trigger_analysis",
            "label": "Frozen sample and locked pairwise comparisons",
            "path": "scripts/build_grounding_failure_report.py",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Joins the 28-case review ledger to the frozen 400-question sample and locked Gemini–Perplexity comparisons.",
                "sql": f"WITH trigger_rules(trigger, selected, coverage, true_failures_caught, failure_recall, rescueable_caught, rescueable_recall, status) AS (VALUES ('Perplexity better or comparison unresolved',{len(selected)},{len(selected) / 400:.8f},{selected_failures},{selected_failures / len(true_failures):.8f},{selected_rescueable},{selected_rescueable / len(rescueable):.8f},'Best observed signal; manual-label upper bound'),('High historical disagreement or unresolved Perplexity',{len(high_disagreement_or_unresolved)},{len(high_disagreement_or_unresolved) / 400:.8f},{feature_failures},{feature_failures / len(true_failures):.8f},{len(feature_selected_ids & rescueable)},{len(feature_selected_ids & rescueable) / len(rescueable):.8f},'Too broad for routing')) SELECT * FROM trigger_rules;",
                "filters": [
                    "Frozen 400-question purposive sample",
                    "Three manually confirmed original-review false positives excluded from the true-failure denominator",
                ],
                "metric_definitions": [
                    "Failure recall is caught audited true failures divided by 25.",
                    "Rescueable recall is caught clear-or-partial rescues divided by 18.",
                ],
                "tables_used": [
                    "artifacts/reference_answer_validation/sample_manifest.json",
                    "artifacts/reference_answer_validation/per_question_rankings.jsonl",
                    "reports/trec_2026_gemini_grounding_failure_review.tsv",
                ],
            },
        },
        {
            "id": "google_pricing",
            "label": "Gemini Developer API pricing",
            "path": "https://ai.google.dev/gemini-api/docs/pricing",
        },
    ]

    summary = (
        "## Technical summary\n\n"
        f"- **Grounding helped materially, but it was not a cure-all.** It produced **{len(clear_rescues)} clear rescues** and **{len(rescueable) - len(clear_rescues)} partial rescues** across the 28 flagged cases.\n"
        f"- **The original denominator was noisy.** Three of the 28 locked serious-error labels were false positives on re-review; among the remaining 25 true failures, grounding or grounded regeneration improved 19.\n"
        f"- **Do not route from question shape alone.** The best pre-answer rule tested would send 272/400 questions to search to catch only 19/25 failures.\n"
        f"- **Use a two-pass disagreement router.** Ground when Perplexity is judged better than ungrounded Gemini or the comparison is unresolved. In the locked pilot that selected 52/400 and caught 19/25 true failures and 14/18 grounding-rescueable cases. This is an upper bound until a local judge reproduces the manual comparison labels."
    )
    cost_text = (
        "## Selective grounding should remain inside the original budget\n\n"
        f"Gemini used Search on **{search_used}/28** requests, issuing **{grounding_usage['grounding_query_count']} search queries**—**{average_queries_when_used:.2f} per search-using request** and **{average_queries_per_request:.2f} across all enabled requests**. Applying the 13% disagreement route to all 7,759 questions projects to about **{projected_selected:,} grounded second passes** and **{projected_search_queries:,.0f} search queries**. If the account's shared **5,000 grounding-prompt monthly Gemini 3 allowance** is still available, the search surcharge is $0 and model tokens project to roughly **${projected_incremental_model_cost:.2f} extra**, or **${projected_total_two_pass:.2f} including the ungrounded full run**.\n\n"
        f"Blanket grounding is economically different: at the pilot's invocation and query rates it projects to about **{blanket_queries:,.0f} searches**. Google's pricing switches after 5,000 free grounding prompts and then charges for each query, so plausible interpretations of how non-searching tool-enabled prompts count put the search surcharge around **${blanket_search_surcharge_low:.0f}–${blanket_search_surcharge_high:.0f}**, plus roughly **${blanket_model_cost:.0f} in model tokens**. That estimate is directional because Gemini decides whether to search and how many queries to execute, the documentation does not expose the counter's exact scope, and the monthly allowance may already be partly consumed."
    )
    methodology = (
        "## What was measured\n\n"
        "The cohort is the 28 Gemini answers labeled major or fatal in the locked 400-question review. Each question was regenerated with the Google Search tool enabled, using a distinct cache. `Clear rescue` means the material error was corrected; `partial rescue` means the answer improved but ambiguity or incompleteness remained; `no rescue` means the decisive defect remained. A regeneration-only rescue is reported separately because Search was enabled but not invoked.\n\n"
        f"The run completed **28/28** requests with no API failures. Search was actually invoked on **{search_used}/28**; among the **{search_true_failures} search-using true failures**, {len(clear_rescues)} were clear rescues and {len(rescueable) - len(clear_rescues)} were partial rescues."
    )
    limitations = (
        "## What this experiment does not establish\n\n"
        "- The 28 questions were selected because the original review called them failures; they are not a random sample of all 7,759 questions.\n"
        "- Three original failure labels were wrong, demonstrating reviewer noise. The revised counts are a manual audit, not organizer ground truth.\n"
        "- Search use and correction are associated in this paired rerun, but a fresh stochastic generation can improve without Search; one such case occurred here.\n"
        "- The proposed disagreement trigger uses manually locked pairwise labels. A local pointwise/pairwise judge must be calibrated before its 13% coverage can be treated as operational.\n"
        "- The grounded cache records Search query/chunk/support counts but intentionally omits raw URLs; decisive niche claims were spot-checked separately against authoritative or independent sources."
    )
    next_steps = (
        "## Recommended next step\n\n"
        "1. Generate the full ungrounded Gemini set and retain the existing Perplexity answers.\n"
        "2. Train or prompt a local judge to emit `Gemini better`, `equivalent`, `Perplexity better`, or `uncertain`, using the 400 locked comparisons as calibration data.\n"
        "3. Ground only `Perplexity better` and `uncertain`; validate first on the 52 pilot questions, including the 33 non-failure controls, to measure regressions as well as rescues.\n"
        "4. Add deterministic triggers for explicit recency/current-state questions and exact hidden-entity chains even when the judge says tie.\n"
        "5. Preserve both answers and the route decision. Do not silently replace a good ungrounded answer when grounding returns a less specific or contradictory result."
    )
    further = (
        "## Further questions\n\n"
        "- Can a local judge reproduce the manual 52-question escalation set at high recall?\n"
        "- How often does grounding harm the 33 selected pilot questions that were not Gemini failures?\n"
        "- Should the final reference selector synthesize conflicting grounded and Perplexity evidence, or retain an explicit unresolved state?"
    )

    report_md = "\n\n".join(
        [
            "# Gemini Search grounding on the 28 audited failure cases",
            summary,
            "## Grounding fixed exact-fact and hidden-entity failures most reliably\n\nClear rescues clustered around names, titles, organizations, race winners, road chains, and biological classifications. Search was weaker on malformed questions, historical-vs-current ambiguity, and requests whose intended source could not be inferred from the wording.",
            methodology,
            "## A disagreement router is much narrower than question-shape heuristics\n\nThe comparison trigger is decision-useful because it uses the two answers we already possess. In this pilot it retained 76% of audited failure recall at 13% coverage; the best tested question-shape rule required 68% coverage for the same recall.",
            cost_text,
            limitations,
            next_steps,
            further,
        ]
    ) + "\n"
    REPORT_MD.write_text(report_md)

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Gemini Search grounding: 28 audited failures",
            "description": "Paired review of ungrounded and Google Search-grounded Gemini answers, with routing and cost implications.",
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "outcomes",
                    "title": "Grounded rerun outcomes",
                    "subtitle": "All 28 originally flagged Gemini serious-error cases; mutually exclusive manual outcomes.",
                    "type": "horizontalBar",
                    "dataset": "outcome_counts",
                    "sourceId": "outcome_analysis",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {"field": "outcome", "type": "nominal", "label": "Outcome"},
                        "y": {"field": "questions", "type": "quantitative", "label": "Questions"},
                        "tooltip": [
                            {"field": "share", "type": "quantitative", "label": "Share of 28"}
                        ],
                    },
                    "settings": {
                        "orientation": "horizontal",
                        "sort": "ascending",
                        "showValues": True,
                        "palette": {"kind": "single", "root": "blue"},
                    },
                    "layout": "full",
                }
            ],
            "tables": [
                {
                    "id": "triggers",
                    "title": "Candidate grounding triggers",
                    "subtitle": "Coverage is over the frozen 400; recall denominators are 25 audited true failures and 18 clear-or-partial rescues.",
                    "dataset": "trigger_rules",
                    "sourceId": "trigger_analysis",
                    "defaultSort": {"field": "coverage", "direction": "asc"},
                    "columns": [
                        {"field": "trigger", "label": "Trigger", "type": "text"},
                        {"field": "selected", "label": "Selected", "format": "number"},
                        {"field": "coverage", "label": "Coverage", "format": "percent"},
                        {"field": "true_failures_caught", "label": "Failures caught", "format": "number"},
                        {"field": "failure_recall", "label": "Failure recall", "format": "percent"},
                        {"field": "rescueable_caught", "label": "Rescueable caught", "format": "number"},
                        {"field": "rescueable_recall", "label": "Rescueable recall", "format": "percent"},
                        {"field": "status", "label": "Interpretation", "type": "text"},
                    ],
                },
                {
                    "id": "cases",
                    "title": "All 28 manually reviewed cases",
                    "subtitle": "Sanitized audit ledger; answer text and search-result URLs remain private.",
                    "dataset": "case_review",
                    "sourceId": "grounding_review",
                    "defaultSort": {"field": "question_id", "direction": "asc"},
                    "density": "dense",
                    "columns": [
                        {"field": "question_id", "label": "Question ID", "type": "text"},
                        {"field": "search_used", "label": "Search used", "type": "boolean"},
                        {"field": "outcome", "label": "Outcome", "type": "text"},
                        {"field": "confidence", "label": "Confidence", "type": "text"},
                        {"field": "review_note", "label": "Review note", "type": "text"},
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# Gemini Search grounding: 28 audited failures"},
                {"id": "summary", "type": "markdown", "body": summary},
                {"id": "finding", "type": "markdown", "body": "## Grounding fixed exact-fact and hidden-entity failures most reliably\n\nClear rescues clustered around names, titles, organizations, race winners, road chains, and biological classifications. Search was weaker on malformed questions, historical-vs-current ambiguity, and requests whose intended source could not be inferred from the wording. The chart includes the three original-review false positives and the one no-Search regeneration rescue so the denominator stays honest."},
                {"id": "outcome_chart", "type": "chart", "chartId": "outcomes"},
                {"id": "scope", "type": "markdown", "body": methodology},
                {"id": "router", "type": "markdown", "body": "## A disagreement router is much narrower than question-shape heuristics\n\nThe comparison trigger is decision-useful because it uses the two answers we already possess. In this pilot it retained 76% of audited failure recall at 13% coverage; the best tested question-shape rule required 68% coverage for the same recall."},
                {"id": "trigger_table", "type": "table", "tableId": "triggers"},
                {"id": "cost", "type": "markdown", "body": cost_text},
                {"id": "limitations", "type": "markdown", "body": limitations},
                {"id": "next", "type": "markdown", "body": next_steps},
                {"id": "questions", "type": "markdown", "body": further},
                {"id": "audit_heading", "type": "markdown", "body": "## Case-level audit trail\n\nThe table shows every adjudication without exposing question or answer text."},
                {"id": "case_table", "type": "table", "tableId": "cases"},
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "outcome_counts": outcome_rows,
                "trigger_rules": trigger_rows,
                "case_review": case_rows,
            },
        },
        "sources": sources,
    }
    ARTIFACT.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n")

    print(
        json.dumps(
            {
                "artifact": str(ARTIFACT.relative_to(ROOT)),
                "report": str(REPORT_MD.relative_to(ROOT)),
                "clear_rescues": len(clear_rescues),
                "partial_rescues": len(rescueable) - len(clear_rescues),
                "true_failures": len(true_failures),
                "trigger_selected": len(selected),
                "trigger_failure_recall": selected_failures / len(true_failures),
                "projected_two_pass_cost": round(projected_total_two_pass, 2),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
