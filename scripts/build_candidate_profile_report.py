#!/usr/bin/env python3
"""Build deterministic, standalone reports for candidate-profile evaluation."""

from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
import tempfile
from typing import Any


TITLE = "Candidate-aware semantic profiles"
SYSTEM_ORDER = (
    "dense_reference",
    "contextual_e5_control",
    "profile_only",
    "dense_e5_equal",
    "dense_profile_equal",
    "dense_profile_dense_heavy",
    "dense_profile_profile_heavy",
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

    phase_comparison = payload.get("comparisons", {}).get("dense_e5_equal", {})
    lines.extend(["", "## Phase 0 paired diagnostic", ""])
    if phase_comparison:
        phase_ndcg = phase_comparison["ndcg_at_10_delta"]
        phase_wtl = phase_comparison["ndcg_at_10_win_tie_loss"]
        phase_zero = phase_comparison["zero_transitions"]
        lines.extend(
            [
                f"- nDCG@10 delta: {_fmt(phase_ndcg['mean_difference'])} (95% CI {_fmt(phase_ndcg['lower'])} to {_fmt(phase_ndcg['upper'])})",
                f"- Wins / ties / losses: {phase_wtl['wins']} / {phase_wtl['ties']} / {phase_wtl['losses']}",
                f"- Query-kNN zero rescued / newly zero: {phase_zero['dense_zero_candidate_nonzero']} / {phase_zero['dense_nonzero_candidate_zero']}",
            ]
        )
    else:
        lines.append("Phase 0 comparison suppressed or unavailable.")

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

    lines.extend(["", "## Promotion gates", ""])
    gates = receipt.get("gates", {})
    values = receipt.get("values", {})
    lines.append(f"- Receipt decision: {receipt['decision']}")
    for name in sorted(set(gates) | set(values)):
        lines.append(
            f"- `{name}`: value `{values.get(name, 'not recorded')}`, "
            f"passed `{gates.get(name, 'not recorded')}`"
        )

    for title, key in (
        ("Density slices", "density_slices"),
        ("Question-type slices", "question_type_slices"),
    ):
        lines.extend(["", f"## {title}", "", "| Slice | Queries | Dense nDCG@10 | Primary nDCG@10 | Primary delta |", "|---|---:|---:|---:|---:|"])
        for name, value in sorted(payload.get(key, {}).items()):
            systems = value.get("systems", {})
            dense = systems.get("dense_reference", {})
            primary = systems.get("dense_profile_equal", {})
            lines.append(f"| `{name}` | {value['query_count']} | {_fmt(dense.get('ndcg_at_10', 0))} | {_fmt(primary.get('ndcg_at_10', 0))} | {_fmt(primary.get('ndcg_at_10_delta', 0))} |")

    summary = payload.get("construction_summary", {})
    lines.extend(["", "## Runtime and coverage", ""])
    for name in sorted(summary):
        lines.append(f"- `{name}`: {summary[name]}")

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "Candidate profiles use historical discovery responses and a fixed encoder; the dense input is a previously frozen qrel-trained baseline signal. Annotation labels are explanatory only and never affect ranking.",
        ]
    )

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
    if comparison:
        ndcg = comparison["ndcg_at_10_delta"]
        mrr = comparison["mrr_delta"]
        wtl = comparison["ndcg_at_10_win_tie_loss"]
        paired_html = (
            "<ul>"
            f"<li>nDCG@10 delta {_fmt(ndcg['mean_difference'])}, 95% CI {_fmt(ndcg['lower'])} to {_fmt(ndcg['upper'])}</li>"
            f"<li>MRR delta {_fmt(mrr['mean_difference'])}, 95% CI {_fmt(mrr['lower'])} to {_fmt(mrr['upper'])}</li>"
            f"<li>Wins / ties / losses: {wtl['wins']} / {wtl['ties']} / {wtl['losses']}</li>"
            "</ul>"
        )
    else:
        paired_html = "<p>Unavailable.</p>"
    phase_comparison = payload.get("comparisons", {}).get("dense_e5_equal", {})
    if phase_comparison:
        phase_ndcg = phase_comparison["ndcg_at_10_delta"]
        phase_wtl = phase_comparison["ndcg_at_10_win_tie_loss"]
        phase_zero = phase_comparison["zero_transitions"]
        phase_html = (
            "<ul>"
            f"<li>nDCG@10 delta {_fmt(phase_ndcg['mean_difference'])}, 95% CI {_fmt(phase_ndcg['lower'])} to {_fmt(phase_ndcg['upper'])}</li>"
            f"<li>Wins / ties / losses: {phase_wtl['wins']} / {phase_wtl['ties']} / {phase_wtl['losses']}</li>"
            f"<li>Query-kNN zero rescued / newly zero: {phase_zero['dense_zero_candidate_nonzero']} / {phase_zero['dense_nonzero_candidate_zero']}</li>"
            "</ul>"
        )
    else:
        phase_html = "<p>Suppressed or unavailable.</p>"
    gate_items = "".join(
        f"<li><code>{html.escape(name)}</code>: value {html.escape(str(payload['profile_receipt'].get('values', {}).get(name, 'not recorded')))}, passed {str(passed).lower()}</li>"
        for name, passed in sorted(payload["profile_receipt"].get("gates", {}).items())
    ) or "<li>See the decision receipt for the fixed gate values.</li>"
    slice_sections = "".join(
        f'<section><h2>{title}</h2><table><caption>{title} inventory</caption>'
        "<thead><tr><th scope=\"col\">Slice</th><th scope=\"col\">Queries</th><th scope=\"col\">Dense nDCG@10</th><th scope=\"col\">Primary nDCG@10</th><th scope=\"col\">Primary delta</th></tr></thead><tbody>"
        + "".join(
            f'<tr><th scope="row"><code>{html.escape(name)}</code></th><td>{value["query_count"]}</td><td>{_fmt(value.get("systems", {}).get("dense_reference", {}).get("ndcg_at_10", 0))}</td><td>{_fmt(value.get("systems", {}).get("dense_profile_equal", {}).get("ndcg_at_10", 0))}</td><td>{_fmt(value.get("systems", {}).get("dense_profile_equal", {}).get("ndcg_at_10_delta", 0))}</td></tr>'
            for name, value in sorted(payload.get(key, {}).items())
        )
        + "</tbody></table></section>"
        for title, key in (("Density slices", "density_slices"), ("Question-type slices", "question_type_slices"))
    )
    summary_items = "".join(
        f"<dt><code>{html.escape(name)}</code></dt><dd>{html.escape(str(value))}</dd>"
        for name, value in sorted(payload.get("construction_summary", {}).items())
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
<section aria-labelledby="paired"><h2 id="paired">Paired primary comparison</h2>{paired_html}</section>
<section aria-labelledby="phase-paired"><h2 id="phase-paired">Phase 0 paired diagnostic</h2>{phase_html}</section>
<section aria-labelledby="transitions"><h2 id="transitions">Zero-query transitions</h2>{transition_html}</section>
<section aria-labelledby="gates"><h2 id="gates">Promotion gates</h2><ul>{gate_items}</ul></section>
{slice_sections}
<section aria-labelledby="runtime"><h2 id="runtime">Runtime and coverage</h2><dl>{summary_items}</dl></section>
<section aria-labelledby="limits"><h2 id="limits">Limitations</h2><p>Candidate profiles use historical discovery responses and a fixed encoder. The dense input is a previously frozen qrel-trained baseline signal. Annotation labels are explanatory only.</p></section>
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
    rendered = ((Path(markdown_path), _markdown(payload)), (Path(html_path), _html(payload)))
    staged: list[tuple[Path, Path]] = []
    try:
        for destination, text in rendered:
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(
                prefix=f".{destination.name}.", dir=destination.parent
            )
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(text)
            staged.append((Path(temporary), destination))
        for temporary, destination in staged:
            os.replace(temporary, destination)
    finally:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)


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
