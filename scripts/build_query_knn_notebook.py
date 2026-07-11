"""Build the executable query-kNN baseline notebook with nbformat."""

from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "notebooks" / "trec_2026_query_knn_baseline.ipynb"


def markdown(source: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(source).strip())


def code(source: str):
    return nbf.v4.new_code_cell(textwrap.dedent(source).strip())


cells = [
    markdown(
        """
        # TREC 2026 query-kNN baseline

        A reproducible, leakage-safe first ranking baseline for the Million LLMs task.
        Every development query is predicted from the other 321 queries, and every one
        of the 150 candidate models is ranked.
        """
    ),
    markdown(
        """
        ## tl;dr

        The named baseline is word TF-IDF query retrieval with the top 10 neighboring
        queries and a similarity-weighted mean of their model grades. The first cell
        below reruns the full leave-one-query-out evaluation and refreshes all artifacts.
        """
    ),
    code(
        """
        from pathlib import Path
        import pandas as pd
        from IPython.display import HTML, display

        from trec_mlm.experiment import (
            ExperimentConfig,
            run_leave_one_out_experiment,
            write_experiment_artifacts,
        )

        ROOT = Path.cwd()
        if not (ROOT / "data" / "dev" / "llm_dev_data.tsv").exists():
            ROOT = ROOT.parent

        config = ExperimentConfig(
            query_path=str(ROOT / "data" / "dev" / "llm_dev_data.tsv"),
            qrels_path=str(ROOT / "data" / "dev" / "llm_dev_qrels.txt"),
            neighbor_counts=(1, 3, 5, 10, 20, 50, 100),
            primary_neighbors=10,
            primary_aggregation="weighted_mean",
            bootstrap_resamples=10_000,
            seed=2026,
            primary_selection_provenance="predeclared_before_current_run",
        )
        result = run_leave_one_out_experiment(config)
        artifact_paths = write_experiment_artifacts(
            result, ROOT / "artifacts" / "query_knn_baseline"
        )
        summary = result.summary
        display(HTML(
            f"<p>Evaluated {summary['data']['query_count']} queries × "
            f"{summary['data']['candidate_count']} candidates; "
            f"wrote {len(artifact_paths)} artifacts.</p>"
        ))
        """
    ),
    code(
        """
        primary = summary["metrics"][summary["primary_key"]]
        prior = summary["metrics"]["global_prior"]
        ndcg_delta = summary["primary_vs_global_prior"]["ndcg_at_10_delta"]

        display(HTML(f'''\
        <style>
          .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
            gap: 0.7rem;
            max-width: 680px;
            margin: 0.5rem 0 1rem;
          }}
          .metric-card {{
            border: 1px solid #d7dde5;
            border-radius: 12px;
            padding: 0.85rem;
            background: #f8fafc;
          }}
          .metric-card small {{ display: block; color: #52606d; }}
          .metric-card strong {{ display: block; font-size: 1.45rem; }}
          @media (max-width: 480px) {{
            .metric-grid {{ grid-template-columns: 1fr 1fr; }}
            .metric-card {{ padding: 0.65rem; }}
            .metric-card strong {{ font-size: 1.2rem; }}
          }}
        </style>
        <div class="metric-grid">
          <div class="metric-card"><small>Baseline nDCG@10</small><strong>{primary['ndcg_at_10']:.3f}</strong></div>
          <div class="metric-card"><small>Baseline MRR</small><strong>{primary['mrr']:.3f}</strong></div>
          <div class="metric-card"><small>Prior nDCG@10</small><strong>{prior['ndcg_at_10']:.3f}</strong></div>
          <div class="metric-card"><small>nDCG improvement</small><strong>+{ndcg_delta['mean_difference']:.3f}</strong></div>
        </div>
        <p><strong>95% paired-query interval for the nDCG improvement:</strong>
        {ndcg_delta['lower']:+.3f} to {ndcg_delta['upper']:+.3f}.</p>
        '''))
        """
    ),
    markdown(
        """
        ## Context & Methods

        ### Key assumptions

        - **Strict leave-one-query-out:** the target query text and its full qrel vector
          are excluded while fitting each fold.
        - **Representation:** lowercase word unigrams and bigrams with training-only
          TF-IDF, L2 normalization, and cosine similarity.
        - **Named baseline:** `k=10`, similarity-weighted mean. Other `k` values and
          maximum evidence are exploratory sensitivity checks, not a tuned headline.
        - **All candidates:** every rule scores and ranks all 150 model IDs.
        - **Metrics:** macro nDCG@10 uses the raw 0/1/2 grade as linear gain; MRR treats
          either grade 1 or 2 as relevant.
        - **Control:** the global-prior rank uses only the other 321 queries in each fold.
        - **No LLM judge:** the released qrels already supply the evaluation labels.
        """
    ),
    markdown("## Data"),
    code(
        """
        data_rows = [
            {"check": "Development queries", "value": summary["data"]["query_count"]},
            {"check": "Candidate models", "value": summary["data"]["candidate_count"]},
            {"check": "Complete qrel pairs", "value": summary["data"]["qrel_count"]},
            {"check": "Zero-weight fallbacks", "value": summary["neighbor_diagnostics"]["zero_weight_fallback_count"]},
        ]
        display(pd.DataFrame(data_rows).style.hide(axis="index"))
        """
    ),
    markdown("## Results"),
    code(
        """
        comparison_keys = [
            "global_prior",
            "weighted_mean@10",
            "max@10",
            "weighted_mean@20",
        ]
        comparison_labels = {
            "global_prior": "Global prior control",
            "weighted_mean@10": "Named baseline: mean, k=10",
            "max@10": "Max evidence, k=10",
            "weighted_mean@20": "Exploratory: mean, k=20",
        }
        rows = [
            {
                "system": comparison_labels[key],
                "nDCG@10": round(summary["metrics"][key]["ndcg_at_10"], 4),
                "MRR": round(summary["metrics"][key]["mrr"], 4),
            }
            for key in comparison_keys
        ]
        display(pd.DataFrame(rows).style.hide(axis="index"))
        """
    ),
    code(
        """
        sensitivity = [
            {
                "neighbor queries (k)": k,
                "nDCG@10": round(summary["metrics"][f"weighted_mean@{k}"]["ndcg_at_10"], 4),
                "MRR": round(summary["metrics"][f"weighted_mean@{k}"]["mrr"], 4),
            }
            for k in config.neighbor_counts
        ]
        display(HTML("<h4>Exploratory neighbor-count sensitivity</h4>"))
        display(pd.DataFrame(sensitivity).style.hide(axis="index"))
        print(
            "Weighted sum and weighted mean produced identical rankings at every k:",
            summary["weighted_sum_mean_rankings_identical"],
        )
        """
    ),
    code(
        """
        slice_labels = {
            "exactly_1_positive": "Exactly 1 positive",
            "2_to_5_positives": "2–5 positives",
            "6_to_20_positives": "6–20 positives",
            "more_than_20_positives": ">20 positives",
        }
        slice_rows = [
            {
                "query slice": slice_labels[key],
                "queries": values["query_count"],
                "baseline nDCG": round(values["primary_ndcg_at_10"], 3),
                "prior nDCG": round(values["prior_ndcg_at_10"], 3),
            }
            for key, values in summary["density_slices"].items()
        ]
        display(HTML("<h4>Performance by relevance density</h4>"))
        display(pd.DataFrame(slice_rows).style.hide(axis="index"))
        """
    ),
    markdown(
        """
        ## Takeaways

        The next cell states the result directly. The important caveat is that the
        neighbor-count grid was viewed on this development set; `k=20` is a promising
        next configuration, not yet an unbiased estimate. A nested split is the right
        place to choose between sparse TF-IDF and a dense semantic embedding baseline.
        """
    ),
    code(
        """
        best_mean_key = max(
            (f"weighted_mean@{k}" for k in config.neighbor_counts),
            key=lambda key: summary["metrics"][key]["ndcg_at_10"],
        )
        best_mean = summary["metrics"][best_mean_key]
        max_at_10 = summary["metrics"]["max@10"]
        print(
            f"1. The fixed k=10 baseline reaches nDCG@10={primary['ndcg_at_10']:.4f} "
            f"and MRR={primary['mrr']:.4f}."
        )
        print(
            f"2. It improves nDCG@10 over the training-only prior by "
            f"{ndcg_delta['mean_difference']:+.4f}."
        )
        print(
            f"3. Maximum evidence at k=10 is weaker: "
            f"nDCG@10={max_at_10['ndcg_at_10']:.4f}."
        )
        print(
            f"4. The best exploratory mean setting is {best_mean_key}, "
            f"with nDCG@10={best_mean['ndcg_at_10']:.4f}."
        )
        print(
            "5. Next: compare this sparse baseline with one frozen dense encoder "
            "under nested, query-level validation."
        )
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
        "language_info": {"name": "python", "version": "3.11"},
    },
)
nbf.write(notebook, DESTINATION)
print(DESTINATION)
