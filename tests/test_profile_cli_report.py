from __future__ import annotations

import json
from pathlib import Path

from scripts.build_candidate_profile_report import build_report
from trec_mlm.cli import build_parser


def _evaluation() -> dict:
    return {
        "status": "evaluated",
        "construction_aggregate_sha256": "a" * 64,
        "construction_unchanged": True,
        "metrics": {
            "dense_reference": {"ndcg_at_10": 0.3599, "mrr": 0.4041, "zero_ndcg_queries": 116},
            "dense_e5_equal": {"ndcg_at_10": 0.36, "mrr": 0.403, "zero_ndcg_queries": 114},
            "dense_profile_equal": {"ndcg_at_10": 0.37, "mrr": 0.405, "zero_ndcg_queries": 110},
        },
        "comparisons": {
            "dense_profile_equal": {
                "ndcg_at_10_delta": {"mean_difference": 0.0101, "lower": 0.001, "upper": 0.02},
                "mrr_delta": {"mean_difference": 0.0009, "lower": -0.01, "upper": 0.01},
                "ndcg_at_10_win_tie_loss": {"wins": 100, "ties": 100, "losses": 122},
                "zero_transitions": {
                    "zero_both": 100,
                    "dense_zero_candidate_nonzero": 16,
                    "dense_nonzero_candidate_zero": 10,
                    "nonzero_both": 196,
                    "query_count": 322,
                },
            }
        },
        "density_slices": {"at_most_5": {"query_count": 50, "query_ids": []}},
        "question_type_slices": {"what": {"query_count": 80, "query_ids": []}},
        "phase0_receipt": {"decision": "pass", "checks": {}, "values": {}},
        "profile_receipt": {"decision": "promote", "gates": {}, "values": {}},
        "three_way_reportable": True,
    }


def test_construction_cli_has_no_qrel_option() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "profile-construct",
            "--queries", "q.tsv",
            "--discovery", "d.json",
            "--candidates", "c.json",
            "--dense-artifact", "dense",
            "--e5-artifact", "e5",
            "--output-dir", "out",
        ]
    )
    assert not any("qrel" in key for key in vars(args))
    assert not hasattr(args, "batch_size")
    assert not hasattr(args, "weights")


def test_only_profile_evaluate_accepts_qrels() -> None:
    parser = build_parser()
    args = parser.parse_args(
        ["profile-evaluate", "--construction", "run", "--qrels", "qrels", "--output-dir", "eval"]
    )
    assert args.qrels == "qrels"
    repeat = parser.parse_args(
        ["profile-repeat-check", "--first", "one", "--second", "two", "--receipt", "receipt.json"]
    )
    assert not any("qrel" in key for key in vars(repeat))


def test_report_is_deterministic_standalone_and_accessible(tmp_path: Path) -> None:
    source = tmp_path / "evaluation.json"
    source.write_text(json.dumps(_evaluation()))
    markdown = tmp_path / "report.md"
    html = tmp_path / "report.html"
    build_report(source, markdown, html)
    first_md = markdown.read_text()
    first_html = html.read_text()
    build_report(source, markdown, html)
    assert markdown.read_text() == first_md
    assert html.read_text() == first_html
    assert "# Candidate-aware semantic profiles" in first_md
    assert "promote" in first_md.lower()
    assert "Zero-query transitions" in first_md
    assert '<a class="skip-link" href="#main">Skip to main content</a>' in first_html
    assert '<main id="main">' in first_html
    assert "focus-visible" in first_html
    assert "<table>" in first_html and "<caption>" in first_html
    assert "<script" not in first_html
    assert "http://" not in first_html and "https://" not in first_html
