"""Build the executable sparse-failure analogue diagnostic notebook."""

from __future__ import annotations

import hashlib
from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks/trec_2026_sparse_failure_analogue_diagnostic.ipynb"


def _normalized_source(source: str) -> str:
    return textwrap.dedent(source).strip()


def _cell_id(kind: str, source: str) -> str:
    return hashlib.sha256(f"{kind}\0{source}".encode()).hexdigest()[:8]


def markdown(source: str):
    normalized = _normalized_source(source)
    return nbf.v4.new_markdown_cell(
        normalized, id=_cell_id("markdown", normalized)
    )


def code(source: str):
    normalized = _normalized_source(source)
    return nbf.v4.new_code_cell(normalized, id=_cell_id("code", normalized))


cells = [
    markdown(
        """
        # TREC 2026 sparse-failure analogue diagnostic

        An auditable companion to the ranking-diagnosis report. It audits whether
        similar development and discovery queries make the official sole-positive
        candidate predictable for the 75 promoted-run zero-nDCG failures.
        """
    ),
    markdown(
        """
        ## tl;dr

        Similar-query evidence is usually insufficient. Only **11/75** cases receive
        a generous strong/corroborated pre-qrel evidence flag. Across eight
        leave-one-out representations, an unreproduced exploratory receipt reports
        that the best inspected post-hoc configuration
        retrieves only **10/75** official positives in the top ten. Twenty-one cases
        contain a component-level top-ten signal that fusion loses; 54 do not. The
        21 are all cross-component reciprocal-rank vetoes. Within the 54, seven are
        near-component misses, 24 lack close attempted history, and 23 have
        historical evidence whose nearby labels point elsewhere.

        The original producer for the multi-representation transfer sweep was not
        retained. The notebook therefore treats that TSV as receipt-only, not as a
        reproducible experiment or a basis for system selection.
        """
    ),
    markdown(
        """
        ## Context & Methods

        ### Key Assumptions

        - The population is the 75 queries with one positive qrel and promoted-run
          nDCG@10 equal to zero.
        - Development-query transfer excludes the target query's qrels.
        - Historical discovery responses are reduced to attempt/refusal evidence;
          attempts are not assumed correct.
        - The evidence buckets are deterministic screening flags. `strong` is an
          upper bound, not a human correctness judgment.
        - Actual responses to the development targets and 2026 judgment provenance
          are unavailable.

        The full calculation is in
        `scripts/build_sparse_failure_analogue_diagnostic.py`. This notebook reloads
        its saved case table and independently recomputes the headline summaries.
        """
    ),
    markdown("## Data"),
    code(
        """
        from pathlib import Path
        import json
        import pandas as pd
        from IPython.display import display

        ROOT = Path.cwd()
        if not (ROOT / "reports" / "trec_2026_sparse_failure_analogue_cases.tsv").exists():
            ROOT = ROOT.parent

        result_path = ROOT / "reports" / "trec_2026_sparse_failure_analogue_diagnostic.json"
        cases_path = ROOT / "reports" / "trec_2026_sparse_failure_analogue_cases.tsv"
        transfer_path = ROOT / "reports" / "trec_2026_sparse_failure_neighbor_transfer.tsv"
        cause_path = ROOT / "reports" / "trec_2026_sparse_failure_cause_diagnostic.json"
        cause_cases_path = ROOT / "reports" / "trec_2026_sparse_failure_cause_cases.tsv"
        weights_path = ROOT / "reports" / "trec_2026_sparse_failure_weight_sensitivity.tsv"

        result = json.loads(result_path.read_text(encoding="utf-8"))
        cause = json.loads(cause_path.read_text(encoding="utf-8"))
        cases = pd.read_csv(cases_path, sep="\\t")
        transfer = pd.read_csv(transfer_path, sep="\\t")
        cause_cases = pd.read_csv(cause_cases_path, sep="\\t")
        weights = pd.read_csv(weights_path, sep="\\t")

        assert len(cases) == 75
        assert cases["query_id"].is_unique
        assert set(cases["evidence_bucket"]) == {"strong", "weak", "not"}
        assert len(cause_cases) == 75
        assert cause["reproduction"]["exact_rank_mismatches"] == 0
        assert abs(
            cause["reproduction"]["primary_ndcg_at_10_raw_graded_gain"]
            - 0.4202139921941393
        ) < 1e-12

        display(pd.DataFrame([
            {"check": "Development queries", "value": result["population"]["development_queries"]},
            {"check": "Candidates per query", "value": result["population"]["candidate_count"]},
            {"check": "Complete qrel pairs", "value": result["population"]["qrel_pairs"]},
            {"check": "Sole-positive failures", "value": len(cases)},
            {"check": "Discovery queries", "value": result["population"]["discovery_queries"]},
        ]).style.hide(axis="index"))
        """
    ),
    markdown("## Results"),
    code(
        """
        bucket_order = ["strong", "weak", "not"]
        bucket_labels = {
            "strong": "Strong/corroborated",
            "weak": "Weak or non-unique",
            "not": "Not predictable",
        }
        bucket_counts = cases["evidence_bucket"].value_counts()
        bucket_table = pd.DataFrame([
            {
                "evidence bucket": bucket_labels[bucket],
                "queries": int(bucket_counts[bucket]),
                "share": bucket_counts[bucket] / len(cases),
            }
            for bucket in bucket_order
        ])
        assert bucket_table["queries"].tolist() == [11, 34, 30]
        display(bucket_table.style.hide(axis="index").format({"share": "{:.1%}"}))
        """
    ),
    code(
        """
        component_labels = {
            "profile_top10_only": "Profile top 10 only",
            "dense_top10_only": "Dense top 10 only",
            "both_top10": "Both components top 10",
            "neither_top10": "Neither component top 10",
        }
        component_counts = cases["component_state"].value_counts()
        component_table = pd.DataFrame([
            {
                "component state": label,
                "queries": int(component_counts.get(key, 0)),
                "share": component_counts.get(key, 0) / len(cases),
            }
            for key, label in component_labels.items()
        ])
        assert component_table["queries"].tolist() == [18, 3, 0, 54]
        display(component_table.style.hide(axis="index").format({"share": "{:.1%}"}))
        """
    ),
    code(
        """
        pure = transfer.query("scope == 'pure_1nn' and representation != 'LOO global prior'").copy()
        best = transfer.query("scope == 'best_posthoc_mrr'").copy()
        assert int(best["hit_at_10"].max()) == 10

        display(
            pure[[
                "representation",
                "nearest_label_agreement",
                "hit_at_1",
                "hit_at_5",
                "hit_at_10",
                "mrr",
                "median_official_rank",
            ]]
            .sort_values(["hit_at_10", "mrr"], ascending=False)
            .style.hide(axis="index").format({"mrr": "{:.4f}"})
        )
        """
    ),
    code(
        """
        response_rows = []
        response_labels = {
            "official_positive": "Official sole positive",
            "primary_top1": "Primary rank 1",
            "profile_top1": "Profile rank 1",
            "dense_top1": "Dense rank 1",
        }
        for key, label in response_labels.items():
            values = result["discovery_response_evidence_top3_mpnet"][key]
            response_rows.append({
                "candidate comparison": label,
                "attempts": values["non_exact_refusal_attempts"],
                "response cells": values["response_cells"],
                "attempt share": values["non_exact_refusal_attempts"] / values["response_cells"],
                "queries with any attempt": values["queries_with_any_attempt"],
                "queries with zero attempts": values["queries_with_zero_attempts"],
            })
        response_table = pd.DataFrame(response_rows)
        assert response_table["attempts"].tolist() == [36, 47, 50, 33]
        display(response_table.style.hide(axis="index").format({"attempt share": "{:.1%}"}))
        """
    ),
    markdown("## Why the 21 fusion cases fail"),
    code(
        """
        fusion = cause_cases.query("cohort == 'fusion_21'").copy()
        assert len(fusion) == 21
        assert (fusion["opposing_rank"] > 20).all()

        fusion_summary = pd.DataFrame([
            {
                "mechanism": "Profile top 10; dense outside",
                "queries": int((fusion.profile_rank <= 10).sum()),
            },
            {
                "mechanism": "Dense top 10; profile outside",
                "queries": int((fusion.dense_rank <= 10).sum()),
            },
            {
                "mechanism": "Supporting component rank 1–3",
                "queries": int((fusion.component_rank <= 3).sum()),
            },
            {
                "mechanism": "Opposing component rank 51–150",
                "queries": int((fusion.opposing_rank >= 51).sum()),
            },
        ])
        assert fusion_summary["queries"].tolist() == [18, 3, 10, 18]
        display(fusion_summary.style.hide(axis="index"))
        """
    ),
    code(
        """
        selected_weights = weights.query(
            "profile_to_dense_weight_ratio in [0.5, 1.0, 1.5, 2.0, 3.0, 10.0]"
        ).copy()
        selected_weights["previously solved lost vs equal"] = (
            105 - selected_weights["other_105_sole_positive_top10_hits"]
        )
        display(
            selected_weights.style.hide(axis="index").format({"ndcg_at_10": "{:.6f}"})
        )
        """
    ),
    markdown("## Why the 54 no-signal cases fail"),
    code(
        """
        no_signal = cause_cases.query("cohort == 'no_signal_54'").copy()
        proxy_order = [
            "near_component_miss",
            "no_close_attempted_history",
            "analogue_label_mismatch",
        ]
        proxy_labels = {
            "near_component_miss": "One component ranks positive 11–20",
            "no_close_attempted_history": "No close attempted history",
            "analogue_label_mismatch": "Nearby released labels point elsewhere",
        }
        proxy_counts = no_signal["cause_bucket"].value_counts()
        proxy_table = pd.DataFrame([
            {
                "deterministic evidence proxy": proxy_labels[key],
                "queries": int(proxy_counts[key]),
                "share of 54": proxy_counts[key] / len(no_signal),
            }
            for key in proxy_order
        ])
        assert proxy_table["queries"].tolist() == [7, 24, 23]
        display(proxy_table.style.hide(axis="index").format({"share of 54": "{:.1%}"}))
        """
    ),
    code(
        """
        archetypes = (
            no_signal["interpretive_archetype"]
            .value_counts()
            .rename_axis("manual spot-check archetype")
            .reset_index(name="queries")
        )
        assert archetypes["queries"].sum() == 54
        display(archetypes.style.hide(axis="index"))
        """
    ),
    markdown(
        """
        ## Takeaways

        1. Similar-query transfer exposes a few coherent templates, but the official
           candidate behaves unlike a stable semantic class for most failures.
        2. The 21 are cross-component contradictions, not score-calibration drift.
           Profile-only recoveries require a profile/dense ratio above 1.30, while
           dense-only recoveries require a ratio below 0.78; no static ratio protects
           both directions.
        3. The 54 are mostly missing or contradictory evidence: 24 lack close
           attempted history and 23 have nearby released labels pointing elsewhere.
        4. Historical non-refusal behavior does not validate the official labels and
           is not a substitute for the missing development-target responses.
        5. The best next experiment should separately report the strong, weak, and
           non-predictable evidence buckets and preserve fold-local evaluation.
        """
    ),
]

notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.13"},
    },
)
def main() -> None:
    nbf.write(notebook, DESTINATION)
    print(DESTINATION)


if __name__ == "__main__":
    main()
