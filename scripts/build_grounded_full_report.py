#!/usr/bin/env python3
"""Build the sanitized full-corpus grounded-generation technical report."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "reports/trec_2026_gemini_grounded_full_audit.json"
EXCEPTIONS = ROOT / "reports/trec_2026_gemini_grounded_full_exceptions.tsv"
REPORT = ROOT / "reports/trec_2026_gemini_grounded_full_2026-07-21.md"
ARTIFACT = ROOT / "reports/trec_2026_gemini_grounded_full.artifact.json"


def main() -> None:
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    if not audit.get("gate_passed"):
        raise ValueError("full grounded audit must pass before report generation")
    counts = audit["counts"]
    if counts["unique_question_ids"] != 7_759 or counts["records"] != 7_759:
        raise ValueError("report requires exactly 7,759 unique completed records")

    exception_groups = {
        "refusal": audit["refusal_ids"],
        "failure": audit["failure_ids"],
        "non_search": audit["non_search_ids"],
        "uncertain": audit["uncertain_ids"],
    }
    exception_rows = [
        {"category": category, "question_id": question_id}
        for category, question_ids in exception_groups.items()
        for question_id in question_ids
    ]
    with EXCEPTIONS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["category", "question_id"], delimiter="\t"
        )
        writer.writeheader()
        writer.writerows(exception_rows)

    status_rows = [
        {"status": status, "questions": questions}
        for status, questions in audit["status_counts"].items()
    ]
    category_rows = [
        {
            "category": label,
            "questions": len(question_ids),
            "definition": definition,
        }
        for label, question_ids, definition in (
            (
                "Uncertain",
                audit["uncertain_ids"],
                "Status is ambiguous, insufficient_evidence, or conflicting_evidence.",
            ),
            (
                "No Search",
                audit["non_search_ids"],
                "Google Search grounding was enabled but the model did not invoke Search.",
            ),
            (
                "Refusal",
                audit["refusal_ids"],
                "Answer matches the narrow refusal-language detector.",
            ),
            (
                "Failure",
                audit["failure_ids"],
                "No valid final record after retries.",
            ),
        )
    ]
    preview_rows = [
        row
        for category in ("failure", "refusal", "uncertain", "non_search")
        for row in [item for item in exception_rows if item["category"] == category][
            :100
        ]
    ]
    cost = audit["projected_cost_usd"]
    generated_at = (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )

    summary = (
        "## Technical summary\n\n"
        f"- **The grounded corpus is complete:** {counts['records']:,} records for "
        f"{counts['unique_question_ids']:,} unique IDs, with zero missing IDs and "
        f"{counts['failures']} unresolved failures.\n"
        f"- **Provenance is retained privately:** Search ran on "
        f"{counts['search_used']:,} responses; all Search-using records passed the "
        "source-URL capture check. The report contains no URLs or answer passages.\n"
        f"- **Uncertainty remains explicit:** {counts['uncertain']:,} responses are "
        f"non-`answered`; {counts['non_search']:,} did not invoke Search and "
        f"{counts['refusals']:,} matched the refusal detector. These sets may overlap.\n"
        f"- **Spend stayed within the gate:** estimated standard total is "
        f"${cost['estimated_standard_total']:.2f}; the deliberately doubled-token-price "
        f"upper projection is ${cost['conservative_total']:.2f}, below $150."
    )
    definitions = (
        "## Scope and validation contract\n\n"
        "The population is the frozen TREC Million LLMs discovery set: exactly 7,759 "
        "question IDs. A valid record has a non-empty clean answer, one of four "
        "structured uncertainty statuses, a short reason, a normal `stop` finish, and "
        "a matching private provenance record. Search-using responses must retain at "
        "least one returned source URL. The full-ID audit is an integrity and provenance "
        "check; it does not establish factual correctness for every answer."
    )
    quality = (
        "## Semantic-quality interpretation\n\n"
        "The full 7,759-question corpus has not been manually quality-reviewed "
        "question by question, so these references are usable as provisional pivots, "
        "not gold labels. In the blinded 400-question four-model stress test, Gemini "
        "was top-tier on 329/400 (82.3%) and had 28/400 (7.0%) major-or-fatal errors; "
        "Perplexity was top-tier on 249/400 (62.3%) with 86/400 (21.5%) serious errors. "
        "Those results support Gemini as the default reference generator, but do not "
        "establish corpus-wide accuracy. Retain Perplexity evidence and escalate "
        "uncertain, non-search, time-sensitive, and materially disagreeing cases.\n\n"
        "`ambiguous` means multiple plausible interpretations; `insufficient_evidence` "
        "means the available evidence did not support the central claim; and "
        "`conflicting_evidence` means the model reported authoritative sources in "
        "conflict. These statuses are review-routing signals, not calibrated error "
        "probabilities. `answered` also does not guarantee truth."
    )
    methodology = (
        "## The 50-question canary unlocked the full run\n\n"
        "The canary required 50/50 unique completions, valid statuses, non-empty and "
        "non-truncated answers, normal finishes, matching evidence records, URLs for "
        "Search-using responses, and a conservative projected cost below $150. It passed "
        "with 39 Search invocations, 11 manually inspected non-search responses, zero "
        "failures, and a $73.40 conservative projection. The full run was resumable and "
        "retried isolated format failures without duplicating valid records."
    )
    limitation = (
        "## Completeness is not correctness\n\n"
        "Grounding provenance is insurance for later adjudication, not proof that each "
        "source entails each central claim. Organizer labels may also contain purposeful "
        "noise. The uncertain and conflicting subset should therefore be investigated "
        "first with the retained sources; a second blanket generation run is not "
        "justified by this audit. The refusal detector is deliberately narrow and may "
        "miss softer non-answers."
    )
    next_steps = (
        "## Next actions\n\n"
        "1. Review all uncertain and conflicting responses against their private support "
        "mappings.\n"
        "2. Use the validated references as pivots for blinded, order-randomized candidate "
        "pairwise judgments with ties and reversal checks.\n"
        "3. Analyze whether non-search responses are concentrated by question cluster or "
        "ambiguity type before changing the grounding prompt.\n"
        "4. Rotate the Gemini key that appeared in chat after the run is archived."
    )
    further = (
        "## Further questions\n\n"
        "- How many uncertain answers become usable after source-level adjudication?\n"
        "- Do the retained support passages cover central claims or only peripheral facts?\n"
        "- Does reference-guided pairwise ranking improve recall for the 94 sparse zero-nDCG "
        "queries without overfitting organizer noise?"
    )
    REPORT.write_text(
        "# Gemini 3.6 grounded references: full-corpus audit\n\n"
        + "\n\n".join(
            [summary, definitions, quality, methodology, limitation, next_steps, further]
        )
        + "\n",
        encoding="utf-8",
    )

    sources = [
        {
            "id": "full_audit",
            "label": "Sanitized full grounded-generation audit",
            "path": "reports/trec_2026_gemini_grounded_full_audit.json",
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "description": "Reads aggregate counts and sanitized exception ID arrays from the final audit.",
                "sql": "SELECT * FROM read_json_auto('reports/trec_2026_gemini_grounded_full_audit.json');",
                "filters": ["Exactly 7,759 frozen discovery question IDs"],
                "metric_definitions": [
                    "Uncertain is any non-answered structured status.",
                    "No Search means grounding was enabled but Search was not invoked.",
                ],
                "tables_used": ["reports/trec_2026_gemini_grounded_full_audit.json"],
            },
        },
        {
            "id": "exceptions",
            "label": "Sanitized full exception-ID ledger",
            "path": "reports/trec_2026_gemini_grounded_full_exceptions.tsv",
            "query": {
                "engine": "python-csv",
                "language": "tsv",
                "description": "Reads the generated sanitized exception ledger containing category and question_id.",
                "sql": "SELECT category, question_id FROM read_csv_auto('reports/trec_2026_gemini_grounded_full_exceptions.tsv', delim='\\t', header=true);",
                "filters": ["Exception categories emitted by the deterministic full-corpus audit"],
                "metric_definitions": ["Categories may overlap because a question can be uncertain and non-search."],
                "tables_used": ["reports/trec_2026_gemini_grounded_full_exceptions.tsv"],
            },
        },
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Gemini 3.6 grounded references: full-corpus audit",
            "description": "Integrity, provenance, uncertainty, and cost audit for all 7,759 grounded reference answers.",
            "generatedAt": generated_at,
            "charts": [
                {
                    "id": "audit_categories",
                    "title": "Grounded-generation audit categories",
                    "subtitle": "Counts over 7,759 questions; categories can overlap.",
                    "type": "horizontalBar",
                    "dataset": "audit_categories",
                    "sourceId": "full_audit",
                    "valueFormat": "number",
                    "encodings": {
                        "x": {
                            "field": "category",
                            "type": "nominal",
                            "label": "Category",
                        },
                        "y": {
                            "field": "questions",
                            "type": "quantitative",
                            "label": "Questions",
                        },
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
                    "id": "statuses",
                    "title": "Structured uncertainty statuses",
                    "subtitle": "Mutually exclusive first-line model statuses across all completed records.",
                    "dataset": "status_counts",
                    "sourceId": "full_audit",
                    "defaultSort": {"field": "questions", "direction": "desc"},
                    "columns": [
                        {"field": "status", "label": "Status", "type": "text"},
                        {
                            "field": "questions",
                            "label": "Questions",
                            "format": "number",
                        },
                    ],
                },
                {
                    "id": "exceptions_preview",
                    "title": "Exception-ID preview",
                    "subtitle": "Up to 100 IDs per category; the supporting TSV contains every ID.",
                    "dataset": "exception_preview",
                    "sourceId": "exceptions",
                    "defaultSort": {"field": "category", "direction": "asc"},
                    "density": "dense",
                    "columns": [
                        {"field": "category", "label": "Category", "type": "text"},
                        {
                            "field": "question_id",
                            "label": "Question ID",
                            "type": "text",
                        },
                    ],
                },
            ],
            "sources": sources,
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# Gemini 3.6 grounded references: full-corpus audit",
                },
                {
                    "id": "summary",
                    "type": "markdown",
                    "body": summary,
                    "sourceId": "full_audit",
                },
                {
                    "id": "finding",
                    "type": "markdown",
                    "body": "## The final audit separates coverage from uncertainty\n\nThe chart is not a partition: an uncertain response may also be a non-search response. Zero final failures and refusals are retained as explicit categories so absence is visible rather than silently omitted.",
                    "sourceId": "full_audit",
                },
                {
                    "id": "category_chart",
                    "type": "chart",
                    "chartId": "audit_categories",
                },
                {"id": "definitions", "type": "markdown", "body": definitions},
                {"id": "statuses_table", "type": "table", "tableId": "statuses"},
                {
                    "id": "method",
                    "type": "markdown",
                    "body": methodology,
                    "sourceId": "full_audit",
                },
                {"id": "limitations", "type": "markdown", "body": limitation},
                {"id": "next", "type": "markdown", "body": next_steps},
                {"id": "further", "type": "markdown", "body": further},
                {
                    "id": "audit_trail",
                    "type": "markdown",
                    "body": "## Sanitized exception audit trail\n\nThe preview contains IDs only. Full answers, URLs, search queries, and passage mappings remain in the private local evidence cache.",
                },
                {
                    "id": "exceptions_table",
                    "type": "table",
                    "tableId": "exceptions_preview",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "audit_categories": category_rows,
                "status_counts": status_rows,
                "exception_preview": preview_rows,
            },
        },
        "sources": sources,
    }
    ARTIFACT.write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "artifact": str(ARTIFACT.relative_to(ROOT)),
                "exceptions": str(EXCEPTIONS.relative_to(ROOT)),
                "report": str(REPORT.relative_to(ROOT)),
            }
        )
    )


if __name__ == "__main__":
    main()
