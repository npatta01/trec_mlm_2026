"""Build the canonical technical-report payload from dense experiment output."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "artifacts/dense_query_knn_baseline/summary.json"
DEFAULT_RESULTS = ROOT / "reports/trec_2026_dense_query_knn_results.json"
DEFAULT_ARTIFACT = ROOT / "reports/trec_2026_dense_query_knn_baseline.artifact.json"
GENERATED_AT = "2026-07-10T00:00:00Z"

HEADLINE_SQL = """\
SELECT dense_ndcg, dense_mrr, ndcg_delta_vs_tfidf, mrr_delta_vs_tfidf
FROM headline_metrics
"""

SYSTEM_METRICS_SQL = """\
SELECT system_order, metric_order, system, metric, value, query_count,
       evaluation_protocol, external_run
FROM system_metrics
ORDER BY system_order, metric_order
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--artifact-output", type=Path, default=DEFAULT_ARTIFACT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    _validate_summary(summary)

    results = _reviewed_results(summary, args.summary)
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


def _validate_summary(summary: dict[str, Any]) -> None:
    required_systems = {"dense_nested", "tfidf_fixed_k10", "global_prior"}
    if set(summary.get("metrics", {})) != required_systems:
        raise ValueError("summary does not contain the three required systems")
    if summary.get("protocol", {}).get("outer_split") != ("strict_leave_one_query_out"):
        raise ValueError("summary is not the exact outer-LOO experiment")
    if summary.get("data", {}).get("query_count") != 322:
        raise ValueError("expected the official 322-query development set")


def _reviewed_results(
    summary: dict[str, Any],
    summary_path: Path,
) -> dict[str, Any]:
    return {
        "generated_at": GENERATED_AT,
        "source_summary_sha256": _sha256(summary_path),
        "protocol": summary["protocol"],
        "data": summary["data"],
        "metrics": summary["metrics"],
        "comparisons": summary["comparisons"],
        "selection": summary["selection"],
        "deployment_selection": summary["deployment_selection"],
        "fixed_config_development_metrics": summary["fixed_config_development_metrics"],
        "encoder_metadata": summary["encoder_metadata"],
        "runtime": summary["runtime"],
        "report_notes": {
            "audience": "technical",
            "delivery_mode": "portable_html",
            "required_structure": [
                "title",
                "technical summary",
                "key findings with visual evidence",
                "scope, data, and metric definitions",
                "methodology and model specification",
                "limitations, uncertainty, and robustness checks",
                "recommended next steps",
                "further questions",
            ],
            "visual": {
                "question": "How does nested dense retrieval compare with both controls?",
                "takeaway": "Dense retrieval leads both controls on nDCG@10 and MRR.",
                "family": "comparison",
                "type": "grouped vertical bar",
                "rows": 6,
                "category_count": 3,
                "metric_count": 2,
                "palette_policy": "hard two-root cap with a visible metric legend",
                "scale": "shared zero-based 0-to-1 metric scale",
                "mobile_qa": (
                    "full-width chart in the responsive portable reader; the delivery "
                    "receipt records whether a compatible headless browser was available"
                ),
                "three_category_exception": (
                    "All three systems are intentional controls and individually labeled; "
                    "a grouped bar remains clearer than a scan-heavy exact-value table."
                ),
            },
            "table_omission": (
                "Exact fixed-grid values remain in this result snapshot; the visible "
                "report uses bullets and one chart to avoid a scan-heavy mobile table."
            ),
        },
    }


def _artifact(
    summary: dict[str, Any],
    results: dict[str, Any],
    results_path: Path,
) -> dict[str, Any]:
    metrics = summary["metrics"]
    dense = metrics["dense_nested"]
    tfidf = metrics["tfidf_fixed_k10"]
    prior = metrics["global_prior"]
    vs_tfidf = summary["comparisons"]["tfidf_fixed_k10"]
    vs_prior = summary["comparisons"]["global_prior"]
    query_count = summary["data"]["query_count"]
    runtime = summary["runtime"]["experiment_seconds_through_summary"]
    selections = summary["selection"]["configuration_frequency"]
    deployment = summary["deployment_selection"]["configuration"]

    headline_rows, system_rows = _query_report_rows(
        dense=dense,
        tfidf=tfidf,
        prior=prior,
        query_count=query_count,
    )
    best_by_encoder = _best_fixed_configuration_by_encoder(summary)

    results_relative = results_path.relative_to(ROOT).as_posix()
    implementation_relative = "trec_mlm/dense_experiment.py"
    qrels_relative = "data/dev/llm_dev_qrels.txt"
    source_list = [
        {
            "id": "dense_results",
            "label": "Reviewed nested dense evaluation snapshot",
            "path": results_relative,
        },
        {
            "id": "headline_sql",
            "label": "Headline metric extraction",
            "path": results_relative,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": HEADLINE_SQL.strip(),
                "description": (
                    "Selects the two nested dense metrics and their paired deltas "
                    "from an in-memory table loaded from the reviewed result snapshot."
                ),
                "executed_at": GENERATED_AT,
                "tables_used": ["headline_metrics"],
                "filters": ["322 official development queries", "outer strict LOO"],
                "metric_definitions": _metric_definitions(),
            },
        },
        {
            "id": "system_metrics_sql",
            "label": "Three-system metric comparison",
            "path": results_relative,
            "query": {
                "engine": "sqlite",
                "language": "sql",
                "sql": SYSTEM_METRICS_SQL.strip(),
                "description": (
                    "Selects long-form nDCG@10 and MRR rows for the nested dense "
                    "system and the two identically evaluated controls."
                ),
                "executed_at": GENERATED_AT,
                "tables_used": ["system_metrics"],
                "filters": ["322 official development queries", "outer strict LOO"],
                "metric_definitions": _metric_definitions(),
            },
        },
        {
            "id": "dense_pipeline",
            "label": "Exact nested-LOO implementation",
            "path": implementation_relative,
        },
        {
            "id": "development_qrels",
            "label": "Official TREC development qrels",
            "path": qrels_relative,
        },
    ]

    title = "TREC 2026 Frozen Dense Query Retrieval"
    ndcg_ci = vs_tfidf["ndcg_at_10_delta"]
    mrr_ci = vs_tfidf["mrr_delta"]
    prior_ndcg_ci = vs_prior["ndcg_at_10_delta"]
    runtime_text = f"{runtime:.1f} seconds"
    fixed_bullets = "\n".join(
        f"- **{values['label']}:** `{values['configuration']}` reached "
        f"{values['ndcg_at_10']:.4f} nDCG@10 and {values['mrr']:.4f} MRR."
        for values in best_by_encoder
    )
    encoder_bullets = "\n".join(
        f"- **{key}:** `{value['model_id']}` at revision "
        f"`{value['resolved_revision'][:12]}…`; {value['pooling']} pooling, "
        f"{value['embedding_dimension']} dimensions, {value['license']}."
        for key, value in summary["encoder_metadata"].items()
    )

    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": title,
            "description": (
                "Exact nested leave-one-out evaluation of four frozen dense query "
                "encoders against the merged TF-IDF baseline and global-prior control."
            ),
            "generatedAt": GENERATED_AT,
            "cards": [
                {
                    "id": "dense_ndcg",
                    "description": (
                        "Macro nDCG@10 over all outer predictions; the chip is the "
                        "paired mean difference from fixed TF-IDF k=10."
                    ),
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [
                        {
                            "label": "Nested dense nDCG@10",
                            "field": "dense_ndcg",
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
                    "id": "dense_mrr",
                    "description": (
                        "Mean reciprocal rank over all outer predictions; the chip is "
                        "the paired mean difference from fixed TF-IDF k=10."
                    ),
                    "dataset": "headline",
                    "sourceId": "headline_sql",
                    "metrics": [
                        {
                            "label": "Nested dense MRR",
                            "field": "dense_mrr",
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
                        "322 outer leave-one-query-out predictions; both metrics use "
                        "the same 0-to-1 scale and higher is better."
                    ),
                    "type": "bar",
                    "dataset": "system_metrics",
                    "sourceId": "system_metrics_sql",
                    "intent": "comparison",
                    "question": (
                        "How does nested dense retrieval compare with TF-IDF and the "
                        "global prior on the two official development metrics?"
                    ),
                    "rationale": (
                        "Grouped bars make the same-scale two-metric comparison across "
                        "three intentionally retained systems directly scannable."
                    ),
                    "comparisonContext": {
                        "query_count": 322,
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
                    "maxRows": 6,
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
                    "sourceId": "dense_results",
                    "body": (
                        "## Frozen dense retrieval clears both controls\n\n"
                        f"- **Primary result:** exact nested selection reached "
                        f"**{dense['ndcg_at_10']:.4f} nDCG@10** and "
                        f"**{dense['mrr']:.4f} MRR** across {query_count} outer queries.\n"
                        f"- **Versus merged TF-IDF:** gains were "
                        f"**+{ndcg_ci['mean_difference']:.4f} nDCG@10** and "
                        f"**+{mrr_ci['mean_difference']:.4f} MRR**; both 95% paired "
                        "bootstrap intervals exclude zero.\n"
                        f"- **Selection was stable:** `mpnet_base_v2@50` won "
                        f"{selections.get('mpnet_base_v2@50', 0)}/{query_count} outer "
                        "folds, with `mpnet_base_v2@20` chosen in the remainder.\n"
                        f"- **Operational fit:** the cached experiment through the "
                        f"10k-resample bootstrap summary finished in {runtime_text}; "
                        "artifact writes are timed separately and no query was truncated."
                    ),
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": ["dense_ndcg", "dense_mrr"],
                },
                {
                    "id": "comparison_interpretation",
                    "type": "markdown",
                    "sourceId": "dense_results",
                    "body": (
                        "## Dense semantics improve both ranking objectives\n\n"
                        f"The chart compares the stitched nested result directly with "
                        f"TF-IDF ({tfidf['ndcg_at_10']:.4f} / "
                        f"{tfidf['mrr']:.4f}) and the global prior "
                        f"({prior['ndcg_at_10']:.4f} / {prior['mrr']:.4f}). The dense "
                        "advantage is not a trade between objectives: it raises both "
                        "top-10 graded utility and first-relevant-model rank."
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
                    "sourceId": "dense_results",
                    "body": (
                        "## Scope and metric definitions\n\n"
                        f"The source matrix contains **{query_count} queries × "
                        f"{summary['data']['candidate_count']} models = "
                        f"{summary['data']['qrel_count']:,} complete qrels**. nDCG@10 "
                        "uses the released 0/1/2 grade as linear gain; MRR treats grades "
                        "1 or 2 as relevant. All systems rank the same 150 candidates for "
                        "every query. TF-IDF remains the merged fixed unigram/bigram "
                        "`k=10` baseline, and the control prior is recomputed from only "
                        "the training qrels in each outer fold."
                    ),
                },
                {
                    "id": "nested_methodology",
                    "type": "markdown",
                    "sourceId": "dense_pipeline",
                    "body": (
                        "## Exact nesting keeps outer labels sealed\n\n"
                        "For outer target `o`, every encoder and neighbor count is scored "
                        "by an inner leave-one-out loop over the other 321 queries. Inner "
                        "target `i` is ranked from `all − {o, i}`; its label is used only "
                        "to score that validation prediction. Inner macro nDCG@10 chooses "
                        "the configuration, with declared encoder order then smaller `k` "
                        "as the exact-tie rule. The selected rule then ranks `o` from all "
                        "321 outer-training rows.\n\nFrozen encoders never consume qrels or "
                        "fit corpus statistics, so their normalized float32 embeddings "
                        "are safely computed once. Fold-specific qrel evidence, priors, "
                        "neighbor exclusions, and tie breaks are still recomputed inside "
                        "the correct split. A mutation test verifies that changing an "
                        "outer qrel cannot change that outer fold's selection or prediction."
                    ),
                },
                {
                    "id": "model_specification",
                    "type": "markdown",
                    "sourceId": "dense_results",
                    "body": (
                        "## Four pinned encoders cover speed and capacity\n\n"
                        f"{encoder_bullets}\n\nThe preregistered grid fixed weighted-mean "
                        "aggregation and searched `k={1,3,5,10,20,50,100}`. E5 applies "
                        "`query: ` to both sides of this symmetric query-to-query task; "
                        "BGE uses raw text. All models run in evaluation/inference mode "
                        "with remote code disabled."
                    ),
                },
                {
                    "id": "fixed_encoder_evidence",
                    "type": "markdown",
                    "sourceId": "dense_results",
                    "body": (
                        "## MPNet supplies the strongest fixed configuration\n\n"
                        f"{fixed_bullets}\n\n`{deployment}` is the full-development "
                        "choice for future test queries. Its same-development fixed score "
                        "is selection evidence, not the unbiased headline; the stitched "
                        "nested score remains the primary estimate."
                    ),
                },
                {
                    "id": "robustness",
                    "type": "markdown",
                    "sourceId": "dense_results",
                    "body": (
                        "## The gain is broad, with uncertainty still visible\n\n"
                        f"Against TF-IDF, the 95% paired-query intervals are "
                        f"**[{ndcg_ci['lower']:+.4f}, {ndcg_ci['upper']:+.4f}]** for "
                        f"nDCG@10 and **[{mrr_ci['lower']:+.4f}, "
                        f"{mrr_ci['upper']:+.4f}]** for MRR. Dense wins/ties/losses on "
                        f"nDCG are **{vs_tfidf['ndcg_at_10_win_tie_loss']['wins']}/"
                        f"{vs_tfidf['ndcg_at_10_win_tie_loss']['ties']}/"
                        f"{vs_tfidf['ndcg_at_10_win_tie_loss']['losses']}**. Against the "
                        f"prior, the nDCG gain is {prior_ndcg_ci['mean_difference']:+.4f} "
                        f"with interval [{prior_ndcg_ci['lower']:+.4f}, "
                        f"{prior_ndcg_ci['upper']:+.4f}]. Control regression checks "
                        "exactly reproduce the merged scores."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## What this result does not establish\n\n"
                        "- The estimate is development-set cross-validation, not hidden "
                        "test performance.\n"
                        "- The four-model grid was fixed before dense outer results were "
                        "inspected; adding encoders after seeing this result would require "
                        "a new untouched outer evaluation.\n"
                        "- Paired bootstrap intervals summarize query variation in the "
                        "stitched predictions; they do not include future distribution "
                        "shift or model-download changes.\n"
                        "- Pretrained encoders contain outside data, so this is an "
                        "external automatic run under the track taxonomy."
                    ),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": (
                        "## Recommended next steps\n\n"
                        f"1. Use **`{deployment}`** as the frozen deployment rule for "
                        "future test queries; do not reselect it after seeing test labels.\n"
                        "2. Retain TF-IDF and the fold-local prior as mandatory regression "
                        "controls in every later feature experiment.\n"
                        "3. Treat any new encoder, prompting variant, or hybrid sparse–dense "
                        "method as a new preregistered experiment rather than extending this "
                        "evaluated grid post hoc."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Does a preregistered sparse–dense neighbor union improve the "
                        "queries where TF-IDF still beats dense?\n"
                        "- Are gains consistent across official query categories once "
                        "category labels are available?\n"
                        "- Can a smaller MiniLM deployment recover most of MPNet's gain "
                        "under a tighter latency or memory budget?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": GENERATED_AT,
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
            dense_ndcg REAL,
            dense_mrr REAL,
            ndcg_delta_vs_tfidf REAL,
            mrr_delta_vs_tfidf REAL
        )
        """
    )
    connection.execute(
        "INSERT INTO headline_metrics VALUES (?, ?, ?, ?)",
        (
            dense["ndcg_at_10"],
            dense["mrr"],
            dense["ndcg_at_10"] - tfidf["ndcg_at_10"],
            dense["mrr"] - tfidf["mrr"],
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
            external_run INTEGER
        )
        """
    )
    systems = (
        (1, "Dense nested", dense, 1),
        (2, "TF-IDF k=10", tfidf, 0),
        (3, "Global prior", prior, 0),
    )
    for system_order, label, values, external_run in systems:
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
                    external_run,
                ),
                (
                    system_order,
                    2,
                    label,
                    "MRR",
                    values["mrr"],
                    query_count,
                    "strict outer LOO",
                    external_run,
                ),
            ),
        )
    headline_rows = [dict(row) for row in connection.execute(HEADLINE_SQL)]
    system_rows = [dict(row) for row in connection.execute(SYSTEM_METRICS_SQL)]
    connection.close()
    return headline_rows, system_rows


def _best_fixed_configuration_by_encoder(
    summary: dict[str, Any],
) -> list[dict[str, Any]]:
    labels = {
        "minilm_l6": "MiniLM-L6",
        "e5_small_v2": "E5-small-v2",
        "bge_small_en_v1_5": "BGE-small-v1.5",
        "mpnet_base_v2": "MPNet-base-v2",
    }
    grouped: dict[str, list[tuple[str, dict[str, float]]]] = {}
    for key, values in summary["fixed_config_development_metrics"].items():
        encoder = key.rsplit("@", maxsplit=1)[0]
        grouped.setdefault(encoder, []).append((key, values))
    output: list[dict[str, Any]] = []
    for encoder in summary["encoder_metadata"]:
        key, values = max(
            grouped[encoder],
            key=lambda item: (item[1]["ndcg_at_10"], -int(item[0].split("@")[-1])),
        )
        output.append(
            {
                "label": labels[encoder],
                "configuration": key,
                "ndcg_at_10": values["ndcg_at_10"],
                "mrr": values["mrr"],
            }
        )
    return output


def _metric_definitions() -> list[str]:
    return [
        "nDCG@10: macro mean over 322 queries with linear positive 0/1/2 qrel gain and log2 rank discount through rank 10.",
        "MRR: macro mean reciprocal rank of the first candidate with qrel grade greater than zero.",
        "Dense nested: one stitched outer prediction per query after encoder and k are chosen only by inner macro nDCG@10.",
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
