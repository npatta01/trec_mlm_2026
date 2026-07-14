#!/usr/bin/env python3
"""Build deterministic, standalone reports for candidate-profile evaluation."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


TITLE = "Candidate-aware semantic profiles"
SYSTEM_ORDER = (
    "dense_reference",
    "contextual_e5_control",
    "profile_only",
    "dense_e5_equal",
    "dense_profile_equal",
    "dense_profile_two_to_one",
    "dense_profile_one_to_two",
    "dense_profile_e5_equal",
)


def _systems(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    metrics = payload["metrics"]
    ordered = [key for key in SYSTEM_ORDER if key in metrics]
    ordered.extend(sorted(set(metrics) - set(ordered)))
    return [(key, metrics[key]) for key in ordered]


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _primary(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("comparisons", {}).get("dense_profile_equal", {})


def _markdown(payload: dict[str, Any]) -> str:
    receipt = payload["profile_receipt"]
    phase0 = payload["phase0_receipt"]
    lines = [
        f"# {TITLE}",
        "",
        f"**Decision:** {receipt['decision']}",
        "",
        f"**Phase 0:** {phase0['decision']}",
        "",
        (
            "The sole promotion comparison is the frozen 1:1 RRF fusion of "
            "query-kNN and candidate profiles against query-kNN. All rankings "
            "were frozen before qrels were loaded."
        ),
        "",
        "## Aggregate metrics",
        "",
        "| System | nDCG@10 | MRR | Zero nDCG queries |",
        "|---|---:|---:|---:|",
    ]
    for name, metrics in _systems(payload):
        lines.append(
            f"| `{name}` | {_fmt(metrics['ndcg_at_10'])} | "
            f"{_fmt(metrics['mrr'])} | {metrics['zero_ndcg_queries']} |"
        )

    comparison = _primary(payload)
    lines.extend(["", "## Paired primary comparison", ""])
    if comparison:
        ndcg = comparison["ndcg_at_10_delta"]
        mrr = comparison["mrr_delta"]
        wtl = comparison["ndcg_at_10_win_tie_loss"]
        lines.extend(
            [
                (
                    f"- nDCG@10 delta: {_fmt(ndcg['mean_difference'])} "
                    f"(95% CI {_fmt(ndcg['lower'])} to {_fmt(ndcg['upper'])})"
                ),
                (
                    f"- MRR delta: {_fmt(mrr['mean_difference'])} "
                    f"(95% CI {_fmt(mrr['lower'])} to {_fmt(mrr['upper'])})"
                ),
                f"- Wins / ties / losses: {wtl['wins']} / {wtl['ties']} / {wtl['losses']}",
            ]
        )
    else:
        lines.append("Primary comparison unavailable.")

    lines.extend(["", "## Zero-query transitions", ""])
    transitions = comparison.get("zero_transitions", {})
    if transitions:
        lines.extend(
            [
                f"- Zero under both: {transitions['zero_both']}",
                (
                    "- Query-kNN zero, candidate profile nonzero: "
                    f"{transitions['dense_zero_candidate_nonzero']}"
                ),
                (
                    "- Query-kNN nonzero, candidate profile zero: "
                    f"{transitions['dense_nonzero_candidate_zero']}"
                ),
                f"- Nonzero under both: {transitions['nonzero_both']}",
            ]
        )
    else:
        lines.append("Transition data unavailable.")

    lines.extend(
        [
            "",
            "## Integrity and scope",
            "",
            f"- Construction unchanged at evaluation: {payload['construction_unchanged']}",
            f"- Construction SHA-256: `{payload['construction_aggregate_sha256']}`",
            f"- Three-way fusion reportable: {payload.get('three_way_reportable', False)}",
            "- Completed Qwen, Jina, pointwise, GroupRank, and listwise experiments were out of scope.",
            "",
        ]
    )
    return "\n".join(lines)


def _html(payload: dict[str, Any]) -> str:
    decision = html.escape(str(payload["profile_receipt"]["decision"]))
    phase0 = html.escape(str(payload["phase0_receipt"]["decision"]))
    rows = "".join(
        "<tr>"
        f"<th scope=\"row\"><code>{html.escape(name)}</code></th>"
        f"<td>{_fmt(metrics['ndcg_at_10'])}</td>"
        f"<td>{_fmt(metrics['mrr'])}</td>"
        f"<td>{metrics['zero_ndcg_queries']}</td>"
        "</tr>"
        for name, metrics in _systems(payload)
    )
    comparison = _primary(payload)
    transitions = comparison.get("zero_transitions", {})
    transition_html = (
        "<dl>"
        f"<dt>Zero under both</dt><dd>{transitions.get('zero_both', 'Unavailable')}</dd>"
        "<dt>Query-kNN zero, profile nonzero</dt>"
        f"<dd>{transitions.get('dense_zero_candidate_nonzero', 'Unavailable')}</dd>"
        "<dt>Query-kNN nonzero, profile zero</dt>"
        f"<dd>{transitions.get('dense_nonzero_candidate_zero', 'Unavailable')}</dd>"
        f"<dt>Nonzero under both</dt><dd>{transitions.get('nonzero_both', 'Unavailable')}</dd>"
        "</dl>"
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>
:root {{ color-scheme: light dark; font-family: system-ui, sans-serif; line-height: 1.5; }}
body {{ margin: 0; }} main {{ max-width: 72rem; margin: auto; padding: 1rem; }}
.skip-link {{ position: absolute; left: -10000px; top: auto; }}
.skip-link:focus-visible {{ left: 1rem; top: 1rem; padding: .5rem; background: Canvas; z-index: 1; }}
:focus-visible {{ outline: .2rem solid Highlight; outline-offset: .15rem; }}
.receipt {{ border-left: .35rem solid currentColor; padding: .6rem 1rem; }}
.table-wrap {{ overflow-x: auto; }} table {{ border-collapse: collapse; width: 100%; }}
caption {{ font-weight: 700; text-align: left; padding-block: .5rem; }}
th, td {{ border: 1px solid GrayText; padding: .45rem; text-align: right; }}
th:first-child {{ text-align: left; }} dt {{ font-weight: 700; }} dd {{ margin-bottom: .5rem; }}
</style>
</head>
<body>
<a class="skip-link" href="#main">Skip to main content</a>
<main id="main">
<h1>{TITLE}</h1>
<p class="receipt"><strong>Decision:</strong> {decision}<br><strong>Phase 0:</strong> {phase0}</p>
<p>The sole promotion comparison is frozen 1:1 RRF fusion of query-kNN and candidate profiles against query-kNN. Rankings were frozen before qrels were loaded.</p>
<section aria-labelledby="metrics"><h2 id="metrics">Aggregate metrics</h2>
<div class="table-wrap"><table><caption>Frozen ranking effectiveness</caption>
<thead><tr><th scope="col">System</th><th scope="col">nDCG@10</th><th scope="col">MRR</th><th scope="col">Zero nDCG queries</th></tr></thead>
<tbody>{rows}</tbody></table></div></section>
<section aria-labelledby="transitions"><h2 id="transitions">Zero-query transitions</h2>{transition_html}</section>
<section aria-labelledby="integrity"><h2 id="integrity">Integrity and scope</h2>
<p>Construction unchanged: {str(payload['construction_unchanged']).lower()}.</p>
<p>Construction SHA-256: <code>{html.escape(str(payload['construction_aggregate_sha256']))}</code>.</p>
<p>Three-way fusion reportable: {str(payload.get('three_way_reportable', False)).lower()}.</p>
<p>Completed Qwen, Jina, pointwise, GroupRank, and listwise experiments were out of scope.</p></section>
</main>
</body>
</html>
"""


def build_report(evaluation_path: str | Path, markdown_path: str | Path, html_path: str | Path) -> None:
    payload = json.loads(Path(evaluation_path).read_text())
    if payload.get("status") != "evaluated":
        raise ValueError("evaluation artifact must have status='evaluated'")
    if payload.get("construction_unchanged") is not True:
        raise ValueError("refusing to report evaluation of changed construction")
    Path(markdown_path).write_text(_markdown(payload), encoding="utf-8")
    Path(html_path).write_text(_html(payload), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", required=True)
    parser.add_argument("--markdown", required=True)
    parser.add_argument("--html", required=True)
    args = parser.parse_args()
    build_report(args.evaluation, args.markdown, args.html)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
