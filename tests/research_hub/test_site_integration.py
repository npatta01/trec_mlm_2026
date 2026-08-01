from __future__ import annotations

import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time
from urllib.request import urlopen
from urllib.parse import urlparse
from xml.etree import ElementTree

import json

from bs4 import BeautifulSoup, Tag
import pytest
import yaml
import websocket

from scripts.normalize_research_hub_links import normalize_links
from trec_mlm.research_hub.build import build_research_hub
from trec_mlm.research_hub.render import render_hub_outputs


ROOT = Path(__file__).resolve().parents[2]
_RENDERED_SITE: Path | None = None


def _render_site() -> Path:
    global _RENDERED_SITE
    if _RENDERED_SITE is not None:
        return _RENDERED_SITE
    build = subprocess.run(
        ["uv", "run", "python", "scripts/build_research_hub.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr
    completed = subprocess.run(
        ["quarto", "render", "research/hub", "--quiet"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    _RENDERED_SITE = ROOT / "research/hub/_site"
    return _RENDERED_SITE


def _visible_html(html: str) -> str:
    return re.sub(r"<!--.*?-->", "", html, flags=re.DOTALL)


def _current_state_main() -> Tag:
    site = _render_site()
    soup = BeautifulSoup(
        (site / "start-here/current-state.html").read_text(), "html.parser"
    )
    main = soup.select_one("main#quarto-document-content")
    assert isinstance(main, Tag)
    return main


def test_final_html_keeps_canonical_supporting_copy_inside_details() -> None:
    site = _render_site()
    question = _visible_html((site / "questions/candidate-evidence.html").read_text())
    study = _visible_html((site / "studies/response-retrieval.html").read_text())
    current_state = _visible_html(
        (site / "start-here/current-state.html").read_text()
    )
    question_copy = (
        "Two requests can look similar while requiring different model strengths."
    )
    study_copy = "Do not use lexical response retrieval as the primary path."

    assert question.count(question_copy) == 1
    assert question_copy not in question[question.rfind("</details>") :]
    assert study.count(study_copy) == 1
    assert study_copy not in study[study.rfind("</details>") :]
    assert '<div class="hub-current-state-page">' in current_state


def test_current_state_rendered_dom_keeps_source_levels_as_grid_children() -> None:
    site = _render_site()
    current_state = _visible_html(
        (site / "start-here/current-state.html").read_text()
    )
    hierarchy_match = re.search(
        r'(?P<hierarchy><div class="hub-source-hierarchy"[^>]*>.*?</div>)'
        r'\s*<details class="hub-visual-equivalent" '
        r'data-equivalent-for="canonical-source-hierarchy">',
        current_state,
        flags=re.DOTALL,
    )

    assert hierarchy_match is not None
    hierarchy = ElementTree.fromstring(hierarchy_match.group("hierarchy"))
    source_levels = hierarchy.findall("./div")
    assert len(source_levels) == 4
    assert [child.tag for child in source_levels[0]] == ["span", "strong", "small"]
    assert all(level.find("./p") is None for level in source_levels)


def test_current_state_consumer_html_has_ordered_unique_document_structure() -> None:
    main = _current_state_main()
    headings = main.select("h1, h2, h3, h4, h5, h6")
    levels = [int(heading.name[1]) for heading in headings]
    ids = [str(element["id"]) for element in main.select("[id]")]

    assert levels.count(1) == 1
    assert all(next_level <= level + 1 for level, next_level in zip(levels, levels[1:]))
    assert len(ids) == len(set(ids))


def test_current_state_tutorial_uses_native_steps_and_a_definition_decoder() -> None:
    main = _current_state_main()
    tutorial = main.select_one(".hub-current-state-tutorial")
    assert isinstance(tutorial, Tag)

    steps = tutorial.select("ol.hub-example-steps > li")
    decoder = tutorial.select_one("dl.hub-term-decoder")
    assert len(steps) == 6
    assert isinstance(decoder, Tag)
    assert len(decoder.find_all("dt", recursive=False)) == len(
        decoder.find_all("dd", recursive=False)
    )
    terms = {
        term.get_text(" ", strip=True).casefold()
        for term in decoder.find_all("dt", recursive=False)
    }
    assert {
        "query or request",
        "candidate",
        "qrel",
        "candidate label or name",
        "query cluster",
        "response-profile cluster",
        "centroid",
        "discovery response",
        "target response",
        "generated reference answer",
        "control",
        "promoted",
        "diagnostic",
        "rejected",
    } <= terms

    links = {str(link.get("href")) for link in tutorial.select("a[href]")}
    assert any(href.endswith("/worked-example.html") for href in links)
    assert any(href.endswith("/glossary.html") for href in links)


def test_current_state_routes_carry_the_example_through_evidence_and_lessons() -> None:
    main = _current_state_main()
    routes = main.select(
        ".hub-current-state-tutorial ol.hub-example-steps > "
        "li.hub-evidence-route[data-evidence-route]"
    )
    expected = {
        "dense-query-knn": {
            "studies": ("dense-query-knn",),
            "lesson_terms": ("nearest-neighbor", "query-side", "qrel"),
        },
        "candidate-label-matching": {
            "studies": ("candidate-label-matching",),
            "lesson_terms": ("contextual label", "clue", "ground truth"),
        },
        "query-cluster-aggregation": {
            "studies": ("query-cluster-aggregation",),
            "lesson_terms": ("query-cluster", "improve", "nearest-neighbor"),
        },
        "source-query-first": {
            "studies": ("source-query-first",),
            "lesson_terms": ("source-query-first", "alignment", "correctness"),
        },
        "candidate-response-archive": {
            "studies": (
                "response-retrieval",
                "semantic-profiles",
                "e5-raw-response-refinement",
            ),
            "lesson_terms": ("lexical archive", "semantic profiles", "candidate-history"),
        },
        "generated-reference": {
            "studies": ("four-model-reference-review",),
            "lesson_terms": ("generated references", "comparison", "not gold", "evaluate"),
        },
    }

    assert [route["data-evidence-route"] for route in routes] == list(expected)
    lessons: list[str] = []
    for route in routes:
        route_id = str(route["data-evidence-route"])
        route_expectation = expected[route_id]
        explanation = route.select_one("dl.hub-route-explanation")
        assert isinstance(explanation, Tag)
        assert [
            term.get_text(" ", strip=True)
            for term in explanation.find_all("dt", recursive=False)
        ] == ["Evidence read", "Illustrative ordering", "Development lesson"]
        assert "basil" in route.select_one(".hub-example-request-ref").get_text(
            " ", strip=True
        ).casefold()

        ordering = route.select_one(
            '.hub-route-ordering[data-illustrative="true"]'
        )
        assert isinstance(ordering, Tag)
        candidates = ordering.select("[data-candidate][data-rank]")
        assert {candidate["data-candidate"] for candidate in candidates} == {
            "GardenGuide",
            "QuickFacts",
            "CodeTutor",
        }
        assert [candidate["data-rank"] for candidate in candidates] == ["1", "2", "3"]

        lesson = route.select_one(".hub-route-lesson").get_text(" ", strip=True).casefold()
        lessons.append(lesson)
        assert all(term in lesson for term in route_expectation["lesson_terms"])
        study_hrefs = {
            str(link.get("href")) for link in route.select("a.hub-route-study[href]")
        }
        for study_id in route_expectation["studies"]:
            assert any(
                href.endswith(f"/studies/{study_id}.html") for href in study_hrefs
            )

    assert len(lessons) == 6
    duplicated_status_phrases = (
        "stable control",
        "rejected",
        "stopped",
        "no registered ranking result",
    )
    for lesson in lessons:
        assert re.search(r"\bretained\b.*\bcontrol\b", lesson) is None
        assert all(phrase not in lesson for phrase in duplicated_status_phrases)


def test_current_state_example_separates_ranking_construction_from_evaluation() -> None:
    main = _current_state_main()
    evaluation = main.select_one(
        ".hub-example-evaluation[data-own-request-qrel='excluded']"
        "[data-historical-development-qrels='frozen-evidence']"
        "[data-evaluation-qrels='when-available']"
    )
    assert isinstance(evaluation, Tag)
    assert [candidate["data-candidate"] for candidate in evaluation.select(
        ".hub-illustrative-ranking [data-candidate]"
    )] == ["GardenGuide", "QuickFacts", "CodeTutor"]
    explanation = evaluation.get_text(" ", strip=True).casefold()
    assert "own qrel" in explanation
    assert "not used" in explanation
    assert "historical development evidence" in explanation
    assert "earlier qrels" in explanation
    assert "qrels are not used to construct" not in explanation
    assert "evaluation" in explanation
    assert "judge" in explanation


def test_current_state_evidence_figures_have_labels_and_semantic_equivalents() -> None:
    main = _current_state_main()
    figures = main.select("figure.hub-evidence-visual[data-visual]")

    assert len(figures) == 8
    assert len({str(figure["data-visual"]) for figure in figures}) == 8
    for figure in figures:
        visual_id = str(figure["data-visual"])
        assert figure.get("aria-labelledby") == f"{visual_id}-title"
        assert figure.select_one(f"#{visual_id}-title") is not None
        assert figure.find("figcaption") is not None
        assert (
            figure.select_one(
                f'details.hub-visual-equivalent[data-equivalent-for="{visual_id}"]'
            )
            is not None
        )
        assert (
            figure.select_one(
                ".hub-visual-print-equivalent"
                f'[data-print-equivalent-for="{visual_id}"]'
            )
            is not None
        )

    for table in main.find_all("table"):
        assert (
            table.find("caption") is not None
            or table.find_parent(attrs={"aria-labelledby": True}) is not None
            or table.find_parent("details") is not None
        )


def test_current_state_audit_follows_tutorial_results_and_failures_in_dom_order() -> None:
    main = _current_state_main()
    tutorial = main.select_one(".hub-current-state-tutorial")
    results = main.select_one('[data-visual="representative-development-results"]')
    failures = main.select_one('[data-visual="remaining-failures"]')
    audit = main.select_one("details.hub-audit-reference")
    elements = main.find_all(True)

    assert all(isinstance(marker, Tag) for marker in (tutorial, results, failures, audit))
    assert [elements.index(marker) for marker in (tutorial, results, failures, audit)] == sorted(
        elements.index(marker) for marker in (tutorial, results, failures, audit)
    )


def test_current_state_audit_figures_have_an_explicit_outline_parent() -> None:
    main = _current_state_main()
    audit = main.select_one("details.hub-audit-reference")
    assert isinstance(audit, Tag)

    audit_heading = audit.find_previous_sibling("h2")
    assert isinstance(audit_heading, Tag)
    assert audit_heading.get_text(" ", strip=True) == "Audit reference"
    assert "workflow, source hierarchy, decisions, and custody" in audit.select_one(
        ":scope > summary"
    ).get_text(" ", strip=True).casefold()

    audit_titles = [
        heading.get_text(" ", strip=True)
        for heading in audit.find_all("h3", recursive=True)
    ]
    assert audit_titles == [
        "Ranking task flow",
        "Canonical-source hierarchy",
        "Experiment-family decision map",
        "Artifact-custody composition",
    ]


def test_current_state_content_needs_no_image_canvas_or_script() -> None:
    main = _current_state_main()

    assert main.select("img, canvas, script") == []


def test_current_state_result_roles_remain_visible_without_color() -> None:
    main = _current_state_main()
    expected_statuses = {
        "promoted_development",
        "stable_control",
        "diagnostic_only",
        "valid_negative",
    }
    rows = main.select(".hub-result-row[data-result-status]")

    assert {str(row["data-result-status"]) for row in rows} == expected_statuses
    for row in rows:
        assert row.select_one(".hub-result-code").get_text(strip=True)
        assert row.select_one(".hub-result-label strong").get_text(strip=True)


def test_current_state_tagged_print_contains_semantic_equivalent_fields(
    tmp_path: Path,
) -> None:
    site = _render_site()
    page = site / "start-here/current-state.html"
    browser = shutil.which("google-chrome")
    pdfinfo = shutil.which("pdfinfo")
    pdftotext = shutil.which("pdftotext")
    assert browser is not None
    assert pdfinfo is not None
    assert pdftotext is not None
    pdf = tmp_path / "current-state.pdf"

    completed = subprocess.run(
        [
            browser,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--no-pdf-header-footer",
            "--generate-tagged-pdf",
            f"--print-to-pdf={pdf}",
            page.resolve().as_uri(),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    pdf_bytes = pdf.read_bytes()
    assert b"/StructTreeRoot" in pdf_bytes
    extracted = subprocess.run(
        [pdftotext, "-layout", str(pdf), "-"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert extracted.returncode == 0, extracted.stdout + extracted.stderr
    for field in ("Responsibility", "Ledger status", "Definition"):
        assert field in extracted.stdout
    structure = subprocess.run(
        [pdfinfo, "-struct-text", str(pdf)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert structure.returncode == 0, structure.stdout + structure.stderr
    for field in ("Responsibility", "Ledger status", "Definition"):
        assert field in structure.stdout
    for value in (
        "Metrics, scopes, promotion decisions",
        "promoted_development",
        "official positive in its top ten",
    ):
        assert value in extracted.stdout


def test_final_html_index_rows_own_their_actions_and_destinations() -> None:
    site = _render_site()
    themes = (site / "themes/index.html").read_text()
    questions = (site / "questions/index.html").read_text()
    studies = (site / "studies/index.html").read_text()

    assert '<span class="hub-next-action">Semantic failure clusters:' in themes
    assert (
        '<span class="hub-next-action">Test hierarchical candidate semantic '
        "profiles against the frozen centroid baseline:"
    ) in questions
    assert 'href="../ideas/semantic-failure-clusters.qmd"' not in themes
    assert (
        "<strong>Lexical response retrieval</strong> "
        "<span>Direct lexical response retrieval underperformed"
    ) in studies


def _free_local_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def test_details_summary_toggles_from_keyboard_in_browser(
    tmp_path: Path,
) -> None:
    site = _render_site()
    page = site / "generated/ideas/failure-aware-fusion.html"
    question_index = site / "questions/index.html"
    port = _free_local_port()
    profile = tmp_path / "chrome-profile"
    browser = shutil.which("google-chrome")
    assert browser is not None
    process = subprocess.Popen(
        [
            browser,
            "--headless=new",
            "--no-sandbox",
            "--disable-gpu",
            "--remote-allow-origins=*",
            f"--remote-debugging-port={port}",
            f"--user-data-dir={profile}",
            page.resolve().as_uri(),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        endpoint = f"http://127.0.0.1:{port}/json"
        for _ in range(30):
            try:
                targets = json.loads(urlopen(endpoint, timeout=1).read())
            except OSError:
                time.sleep(0.1)
                continue
            target = next(
                (
                    row
                    for row in targets
                    if row["type"] == "page" and row["url"] == page.resolve().as_uri()
                ),
                None,
            )
            if target is not None:
                break
            time.sleep(0.1)
        else:
            pytest.fail("Chrome did not load the rendered details page")
        client = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=5)
        try:

            def call(
                method: str, params: dict[str, object] | None = None
            ) -> dict[str, object]:
                call.counter += 1
                client.send(
                    json.dumps(
                        {"id": call.counter, "method": method, "params": params or {}}
                    )
                )
                while True:
                    response = json.loads(client.recv())
                    if response.get("id") == call.counter:
                        return response

            call.counter = 0  # type: ignore[attr-defined]
            for _ in range(30):
                ready = call(
                    "Runtime.evaluate",
                    {
                        "expression": "document.readyState === 'complete' && Boolean(document.querySelector('summary'))",
                        "returnByValue": True,
                    },
                )
                if ready["result"]["result"].get("value") is True:
                    break
                time.sleep(0.1)
            else:
                pytest.fail("Rendered details summary did not become available")
            focus = call(
                "Runtime.evaluate",
                {
                    "expression": "document.querySelector('summary').focus(); document.activeElement.tagName",
                    "returnByValue": True,
                },
            )
            assert focus["result"]["result"]["value"] == "SUMMARY"
            summary_height = call(
                "Runtime.evaluate",
                {
                    "expression": "getComputedStyle(document.querySelector('summary')).minHeight",
                    "returnByValue": True,
                },
            )
            assert float(summary_height["result"]["result"]["value"][:-2]) >= 44
            before = call(
                "Runtime.evaluate",
                {
                    "expression": "document.querySelector('details').open",
                    "returnByValue": True,
                },
            )
            assert before["result"]["result"]["value"] is False
            key = {
                "key": "Enter",
                "code": "Enter",
                "windowsVirtualKeyCode": 13,
                "nativeVirtualKeyCode": 13,
            }
            call("Input.dispatchKeyEvent", {"type": "keyDown", **key})
            call(
                "Input.dispatchKeyEvent",
                {"type": "char", "text": "\r", "unmodifiedText": "\r", **key},
            )
            call("Input.dispatchKeyEvent", {"type": "keyUp", **key})
            after = call(
                "Runtime.evaluate",
                {
                    "expression": "document.querySelector('details').open",
                    "returnByValue": True,
                },
            )
            assert after["result"]["result"]["value"] is True
            call("Page.navigate", {"url": question_index.resolve().as_uri()})
            for _ in range(20):
                ready = call(
                    "Runtime.evaluate",
                    {
                        "expression": "JSON.stringify({ready: document.readyState, path: location.pathname})",
                        "returnByValue": True,
                    },
                )
                page_state = json.loads(ready["result"]["result"]["value"])
                if page_state["ready"] == "complete" and page_state["path"].endswith(
                    "/questions/index.html"
                ):
                    break
                time.sleep(0.1)
            else:
                pytest.fail("Question index did not finish navigating in Chrome")
            row = call(
                "Runtime.evaluate",
                {
                    "expression": """
const row = document.querySelector('a.hub-question-row');
row.focus();
JSON.stringify({
  focused: document.activeElement === row,
  actionOwned: row.querySelector('.hub-next-action').parentElement === row,
  href: row.href,
});
""",
                    "returnByValue": True,
                },
            )
            row_state = json.loads(row["result"]["result"]["value"])
            assert row_state["focused"] is True
            assert row_state["actionOwned"] is True
            destination = Path(urlparse(row_state["href"]).path)
            assert destination.is_file()
            current_state = site / "start-here/current-state.html"
            call("Page.navigate", {"url": current_state.resolve().as_uri()})
            for _ in range(20):
                ready = call(
                    "Runtime.evaluate",
                    {
                        "expression": "JSON.stringify({ready: document.readyState, path: location.pathname})",
                        "returnByValue": True,
                    },
                )
                page_state = json.loads(ready["result"]["result"]["value"])
                if page_state["ready"] == "complete" and page_state[
                    "path"
                ].endswith("/start-here/current-state.html"):
                    break
                time.sleep(0.1)
            else:
                pytest.fail("Current-state page did not finish navigating in Chrome")
            screen_state = call(
                "Runtime.evaluate",
                {
                    "expression": """
const printEquivalents = [...document.querySelectorAll('.hub-visual-print-equivalent')];
JSON.stringify({
  printEquivalentCount: printEquivalents.length,
  printEquivalentsHidden: printEquivalents.every(
    equivalent => getComputedStyle(equivalent).display === 'none'
  ),
  closedDetailCount: [...document.querySelectorAll('details.hub-visual-equivalent')]
    .filter(details => !details.open).length,
});
""",
                    "returnByValue": True,
                },
            )
            screen = json.loads(screen_state["result"]["result"]["value"])
            assert screen == {
                "printEquivalentCount": 8,
                "printEquivalentsHidden": True,
                "closedDetailCount": 8,
            }

            for viewport_width in (390, 320):
                call(
                    "Emulation.setDeviceMetricsOverride",
                    {
                        "width": viewport_width,
                        "height": 844,
                        "deviceScaleFactor": 1,
                        "mobile": False,
                    },
                )
                reflow = call(
                    "Runtime.evaluate",
                    {
                        "expression": """(() => {
const cards = document.querySelector('.hub-example-candidates');
const routes = document.querySelector('.hub-example-steps');
const ranking = document.querySelector('.hub-illustrative-ranking');
const rankingBounds = ranking.getBoundingClientRect();
return JSON.stringify({
  viewport: document.documentElement.clientWidth,
  scrollWidth: document.documentElement.scrollWidth,
  candidateColumns: getComputedStyle(cards).gridTemplateColumns
    .trim().split(/\\s+/).filter(Boolean).length,
  routeColumns: getComputedStyle(routes).gridTemplateColumns
    .trim().split(/\\s+/).filter(Boolean).length,
  rankingColumns: getComputedStyle(ranking).gridTemplateColumns
    .trim().split(/\\s+/).filter(Boolean).length,
  rankingWithinViewport:
    rankingBounds.left >= 0 &&
    rankingBounds.right <= document.documentElement.clientWidth,
});
})()""",
                        "returnByValue": True,
                    },
                )
                assert "value" in reflow["result"]["result"], reflow
                reflow_state = json.loads(reflow["result"]["result"]["value"])
                assert viewport_width - 20 <= reflow_state["viewport"] <= viewport_width
                assert reflow_state["scrollWidth"] <= reflow_state["viewport"]
                assert reflow_state["candidateColumns"] == 1
                assert reflow_state["routeColumns"] == 1
                assert reflow_state["rankingColumns"] == 1
                assert reflow_state["rankingWithinViewport"] is True

            call(
                "Emulation.setDeviceMetricsOverride",
                {
                    "width": 1280,
                    "height": 900,
                    "deviceScaleFactor": 1,
                    "mobile": False,
                },
            )
            status_styles = call(
                "Runtime.evaluate",
                {
                    "expression": """
JSON.stringify(Object.fromEntries(
  [...document.querySelectorAll('.hub-result-row[data-result-status]')].map(row => [
    row.dataset.resultStatus,
    getComputedStyle(row.querySelector('.hub-result-bar')).borderInlineEndStyle,
  ])
));
""",
                    "returnByValue": True,
                },
            )
            assert json.loads(status_styles["result"]["result"]["value"]) == {
                "promoted_development": "solid",
                "stable_control": "double",
                "diagnostic_only": "dashed",
                "valid_negative": "dotted",
            }

            call(
                "Emulation.setEmulatedMedia",
                {
                    "media": "screen",
                    "features": [{"name": "forced-colors", "value": "active"}],
                },
            )
            forced_colors = call(
                "Runtime.evaluate",
                {
                        "expression": """(() => {
const tutorial = document.querySelector('.hub-current-state-tutorial');
const link = tutorial.querySelector('a');
const summary = document.querySelector('.hub-audit-reference > summary');
summary.focus();
const tutorialStyle = getComputedStyle(tutorial);
const linkStyle = getComputedStyle(link);
const summaryStyle = getComputedStyle(summary);
return JSON.stringify({
  active: matchMedia('(forced-colors: active)').matches,
  tutorialForcedColorAdjust: tutorialStyle.forcedColorAdjust,
  tutorialBackground: tutorialStyle.backgroundColor,
  tutorialColor: tutorialStyle.color,
  linkColor: linkStyle.color,
  summaryForcedColorAdjust: summaryStyle.forcedColorAdjust,
  outlineStyle: summaryStyle.outlineStyle,
  outlineWidth: summaryStyle.outlineWidth,
  outlineColor: summaryStyle.outlineColor,
});
})()""",
                    "returnByValue": True,
                },
            )
            forced_state = json.loads(
                forced_colors["result"]["result"]["value"]
            )
            assert forced_state["active"] is True
            assert forced_state["tutorialForcedColorAdjust"] == "auto"
            assert forced_state["summaryForcedColorAdjust"] == "auto"
            assert forced_state["tutorialBackground"] != "rgba(0, 0, 0, 0)"
            assert forced_state["linkColor"] != forced_state["tutorialColor"]
            assert forced_state["outlineStyle"] != "none"
            assert float(forced_state["outlineWidth"][:-2]) > 0
            assert forced_state["outlineColor"] != "rgba(0, 0, 0, 0)"

            call(
                "Emulation.setEmulatedMedia",
                {"media": "print", "features": []},
            )
            print_behavior = call(
                "Runtime.evaluate",
                {
                        "expression": """(() => {
const audit = document.querySelector('.hub-audit-reference');
return JSON.stringify({
  auditOpen: audit.open,
  auditContentVisibility: getComputedStyle(audit, '::details-content').contentVisibility,
  auditFigureDisplay: getComputedStyle(audit.querySelector('figure')).display,
  printEquivalentsVisible: [...document.querySelectorAll('.hub-visual-print-equivalent')]
    .every(node => getComputedStyle(node).display !== 'none'),
  screenEquivalentsHidden: [...document.querySelectorAll('.hub-visual-equivalent')]
    .every(node => getComputedStyle(node).display === 'none'),
});
})()""",
                    "returnByValue": True,
                },
            )
            assert json.loads(print_behavior["result"]["result"]["value"]) == {
                "auditOpen": False,
                "auditContentVisibility": "visible",
                "auditFigureDisplay": "block",
                "printEquivalentsVisible": True,
                "screenEquivalentsHidden": True,
            }

            call(
                "Emulation.setEmulatedMedia",
                {"media": "screen", "features": []},
            )
            summary_behavior = call(
                "Runtime.evaluate",
                {
                        "expression": """(() => {
const summary = document.querySelector('.hub-audit-reference > summary');
summary.focus();
const style = getComputedStyle(summary);
return JSON.stringify({
  minHeight: style.minHeight,
  renderedHeight: summary.getBoundingClientRect().height,
  outlineStyle: style.outlineStyle,
  outlineWidth: style.outlineWidth,
});
})()""",
                    "returnByValue": True,
                },
            )
            summary_state = json.loads(
                summary_behavior["result"]["result"]["value"]
            )
            assert float(summary_state["minHeight"][:-2]) >= 44
            assert summary_state["renderedHeight"] >= 44
            assert summary_state["outlineStyle"] != "none"
            assert float(summary_state["outlineWidth"][:-2]) > 0

            call(
                "Runtime.evaluate",
                {
                    "expression": "document.querySelector('details.hub-audit-reference').open = true; document.querySelectorAll('details.hub-visual-equivalent').forEach(details => { details.open = true; })",
                },
            )
            accessibility = call("Accessibility.getFullAXTree")
            accessible_headers = [
                node.get("name", {}).get("value")
                for node in accessibility["result"]["nodes"]
                if node.get("role", {}).get("value") == "columnheader"
            ]
            for field in ("Responsibility", "Ledger status", "Definition"):
                assert accessible_headers.count(field) == 1
        finally:
            client.close()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_quarto_renders_generated_idea_library() -> None:
    outputs = render_hub_outputs(build_research_hub(ROOT))
    config = yaml.safe_load((ROOT / "research/hub/_quarto.yml").read_text())

    assert Path("research/hub/generated/ideas-index.qmd") in outputs
    assert {
        path for path in outputs if path.parent == Path("research/hub/generated/ideas")
    } == {
        Path("research/hub/generated/ideas/failure-aware-fusion.qmd"),
        Path("research/hub/generated/ideas/hierarchical-candidate-profiles.qmd"),
        Path("research/hub/generated/ideas/local-response-quality-judging.qmd"),
        Path("research/hub/generated/ideas/reference-review-trigger.qmd"),
        Path("research/hub/generated/ideas/semantic-failure-clusters.qmd"),
    }
    assert "generated/ideas/*.qmd" in config["project"]["render"]
    assert "ideas/*.qmd" not in config["project"]["render"]


def test_failure_aware_fusion_uses_the_reviewed_explainer_copy() -> None:
    source = (ROOT / "research/hub/ideas/failure-aware-fusion.qmd").read_text()
    metadata = yaml.safe_load(source.split("---", maxsplit=2)[1])

    assert metadata["plain_english"] == (
        "Use a reliability check to choose which evidence sources should "
        "influence each request."
    )
    assert metadata["why_it_matters"] == (
        "Response and reference evidence help unevenly, so a fixed mixture "
        "can hide failures."
    )
    assert metadata["mechanism_steps"] == [
        "Read the new request and estimate which evidence sources are dependable.",
        "Choose a bounded evidence mixture using the frozen reliability rule.",
        "Rank candidates and compare with the frozen dense/profile control.",
    ]
    assert metadata["next_action"] == (
        "Define and freeze the reliability signal, fallback, and artifact plan."
    )
    assert metadata["visual"] == {
        "type": "causal_comparison",
        "current_label": "Today",
        "current_summary": "One fixed evidence mixture creates uneven failures.",
        "current_items": [
            "Past requests",
            "Candidate profiles",
            "Responses",
            "Generated reference",
        ],
        "change_label": "Proposed change",
        "proposed_label": "Proposed",
        "proposed_summary": "A reliability gate chooses the evidence mixture.",
        "proposed_routes": [
            {
                "label": "Known request pattern",
                "action": "Use similar historical requests.",
            },
            {
                "label": "Stable candidate strength",
                "action": "Use semantic profiles.",
            },
            {
                "label": "Response evidence passes checks",
                "action": "Use responses or references.",
            },
        ],
        "decision_test": "Does the adaptive mixture beat the frozen dense/profile control?",
    }


def test_complete_research_hub_content_and_static_shell() -> None:
    build = build_research_hub(ROOT)
    config = yaml.safe_load((ROOT / "research/hub/_quarto.yml").read_text())

    assert config["quarto-required"] == "1.9.38"
    assert config["project"]["type"] == "website"
    assert config["project"]["output-dir"] == "_site"
    assert config["project"]["resources"] == [
        "assets/ledger.js",
        "assets/fonts/",
        "catalog.json",
    ]
    assert "generated/runs/*.qmd" in config["project"]["render"]
    assert "generated/themes/*.qmd" in config["project"]["render"]
    assert "generated/ideas/*.qmd" in config["project"]["render"]
    assert "library/*.qmd" in config["project"]["render"]
    assert config["project"]["pre-render"] == [
        "uv run --project ../.. python ../../scripts/build_research_hub.py --root ../.."
    ]
    assert config["project"]["post-render"] == [
        "uv run python ../../scripts/normalize_research_hub_links.py "
        "--root ../.. --output-dir _site",
        "uv run python ../../scripts/audit_research_hub_site.py "
        "--root ../.. --output-dir _site",
    ]
    assert {row["text"] for row in config["website"]["navbar"]["left"]} >= {"Themes"}
    themes_nav = next(
        row for row in config["website"]["navbar"]["left"] if row["text"] == "Themes"
    )
    assert themes_nav["href"] == "themes/index.qmd"
    assert config["execute"]["enabled"] is False
    assert build.validation.record_counts == {
        "theme": 4,
        "question": 5,
        "study": 17,
        "test": 5,
        "approach": 5,
        "dataset": 7,
            "background": 9,
    }
    assert build.validation.linked_system_count == len(build.ledger.systems) == 98

    problem = (ROOT / "research/hub/start-here/problem.qmd").read_text()
    assert "## What are we building?" in problem
    assert "150 candidate LLMs" in problem
    assert "hidden test" in problem.lower()
    assert "Who became the second Karmapa Lama?" in problem
    for candidate in (
        "Cultures",
        "People_by_time",
        "Scholars_and_academics_by_subject",
    ):
        assert candidate in problem
    assert "Candidate 37" not in problem
    assert "does not claim" in problem

    protocols = (ROOT / "research/hub/protocols/index.qmd").read_text()
    assert "../../../protocols/" not in protocols
    for path in sorted((ROOT / "protocols").glob("*.md")):
        assert path.relative_to(ROOT).as_posix() in protocols

    assert {
        "provided",
        "collected",
        "generated",
    } <= {row.metadata["category"] for row in build.sources.datasets}
    qrels = next(row for row in build.sources.datasets if row.record_id == "dev-qrels")
    assert qrels.metadata["permitted_use"] == "development_evaluation"
    assert all(
        len(row.metadata["representative_system_ids"]) <= 8
        for row in build.sources.studies
    )
    by_study = {row.record_id: row for row in build.sources.studies}
    assert by_study["four-model-reference-review"].metadata["system_ids"] == []
    assert (
        by_study["grounded-data-quality-audit"].metadata["study_type"] == "data_quality"
    )


def test_explainer_and_index_pages_use_readable_custom_layouts() -> None:
    problem = (ROOT / "research/hub/start-here/problem.qmd").read_text()
    questions = (ROOT / "research/hub/questions/index.qmd").read_text()
    data = (ROOT / "research/hub/data/index.qmd").read_text()
    background = (ROOT / "research/hub/background/index.qmd").read_text()
    ideas = (ROOT / "research/hub/ideas/index.qmd").read_text()

    assert "hub-task-flow" in problem
    assert "hub-example" in problem
    studies = (ROOT / "research/hub/studies/index.qmd").read_text()
    themes = (ROOT / "research/hub/themes/index.qmd").read_text()
    summary = (ROOT / "research/hub/index.qmd").read_text()
    ledger_script = (ROOT / "research/hub/assets/ledger.js").read_text()

    assert "{{< include ../generated/questions-index.qmd >}}" in questions
    assert "{{< include ../generated/background-index.qmd >}}" in background
    assert data.strip().endswith("{{< include ../generated/data-index.qmd >}}")
    assert "Candidate inventory" not in data
    assert "{{< include ../generated/studies-index.qmd >}}" in studies
    assert "{{< include ../generated/themes-index.qmd >}}" in themes
    assert "{{< include ../generated/ideas-index.qmd >}}" in ideas
    assert "research summary" not in themes.lower()
    assert "include-after-body: ../assets/ledger-script.html" in studies
    assert "{{< include generated/summary.qmd >}}" in summary
    assert 'searchParams.get("theme")' in ledger_script
    assert "[data-themes]" in ledger_script
    assert "listing:" not in questions
    assert "listing:" not in studies
    assert "listing:" not in data
    assert "listing:" not in background
    assert "listing:" not in ideas


def test_generated_pages_and_theme_use_the_open_visual_system() -> None:
    build = build_research_hub(ROOT)
    outputs = render_hub_outputs(build)
    generated = "\n".join(outputs.values())
    theme = (ROOT / "research/hub/assets/theme.scss").read_text()

    for class_name in (
        "hub-theme-row",
        "hub-question-row",
        "hub-evidence-line",
        "hub-study-row",
        "hub-background-row",
        "hub-record-hero",
        "hub-takeaway",
        "hub-next-action",
        "hub-details",
    ):
        assert class_name in generated
        assert f".{class_name}" in theme

    for index_path in (
        "research/hub/generated/summary.qmd",
        "research/hub/generated/themes-index.qmd",
        "research/hub/generated/questions-index.qmd",
        "research/hub/generated/studies-index.qmd",
        "research/hub/generated/background-index.qmd",
        "research/hub/generated/data-index.qmd",
    ):
        source = outputs[Path(index_path)]
        assert "listing:" not in source
        assert "line count" not in source.lower()
        assert "line-count" not in source.lower()
        assert "study_type" not in source

    assert ".hub-theme-row:hover" in theme
    assert ".hub-theme-row:focus-visible" in theme
    assert ".hub-decision-strip" in theme
    assert "@media (forced-colors: active)" in theme
    assert "@media (max-width: 768px)" in theme
    assert ".content table" in theme
    assert "overflow-x: auto" in theme
    assert "$web-font-path: false" in theme


def test_post_render_normalizes_only_local_index_links(tmp_path: Path) -> None:
    output = tmp_path / "research/hub/_site"
    output.mkdir(parents=True)
    page = output / "page.html"
    page.write_text(
        '<a href="../background/index.html">Background</a>'
        '<a href="index.html?view=all">Home</a>'
        '<a href="https://example.test/index.html">External</a>'
        '<a href="&sol;&sol;evil.example/index.html">Encoded external</a>'
        '<a href="https&colon;//evil.example/index.html">Encoded scheme</a>'
        r'<a href="\\evil.example\foo/index.html">Backslash external</a>'
    )

    changed = normalize_links(tmp_path, output)

    assert changed == (page,)
    rendered = page.read_text()
    assert 'href="../background/"' in rendered
    assert 'href="./?view=all"' in rendered
    assert 'href="https://example.test/index.html"' in rendered
    assert 'href="&sol;&sol;evil.example/index.html"' in rendered
    assert 'href="https&colon;//evil.example/index.html"' in rendered
    assert r'href="\\evil.example\foo/index.html"' in rendered


def test_post_render_rejects_symlinked_output(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    expected = tmp_path / "research/hub"
    expected.mkdir(parents=True)
    (expected / "_site").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        normalize_links(tmp_path, expected / "_site")


def test_site_security_audit_rejects_qmd_files_and_reader_links(
    tmp_path: Path,
) -> None:
    output = tmp_path / "research/hub/_site"
    output.mkdir(parents=True)
    (output / "index.html").write_text(
        '<a href="ideas/raw-record.qmd">Raw record</a>',
        encoding="utf-8",
    )
    raw_record = output / "ideas/raw-record.qmd"
    raw_record.parent.mkdir()
    raw_record.write_text("---\ntest_id: raw-record\n---\n", encoding="utf-8")

    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(ROOT / "scripts/audit_research_hub_site.py"),
            "--root",
            str(tmp_path),
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1
    assert "reader-facing .qmd target" in completed.stderr
    assert "unexpected published .qmd source" in completed.stderr


def test_generated_navigation_contains_every_canonical_record() -> None:
    build = build_research_hub(ROOT)
    outputs = render_hub_outputs(build)

    summary = outputs[Path("research/hub/generated/summary.qmd")]
    themes = outputs[Path("research/hub/generated/themes-index.qmd")]
    questions = outputs[Path("research/hub/generated/questions-index.qmd")]
    studies = outputs[Path("research/hub/generated/studies-index.qmd")]
    background = outputs[Path("research/hub/generated/background-index.qmd")]
    data = outputs[Path("research/hub/generated/data-index.qmd")]

    for theme in build.sources.themes:
        assert theme.record_id in summary
        assert theme.record_id in themes
        assert Path(f"research/hub/generated/themes/{theme.record_id}.qmd") in outputs
    for question in build.sources.questions:
        assert question.record_id in questions
    for study in build.sources.studies:
        assert study.record_id in studies
        detail_path = Path(f"research/hub/generated/studies/{study.record_id}.md")
        assert detail_path in outputs
        canonical = (ROOT / study.source_path).read_text()
        assert (
            f"{{{{< include ../generated/studies/{study.record_id}.md >}}"
        ) in canonical
        assert "%40" not in outputs[detail_path]
    for resource in build.sources.background:
        assert resource.record_id in background
    for dataset in build.sources.datasets:
        assert dataset.record_id in data


def test_contributor_guide_and_complete_record_templates() -> None:
    guide = (ROOT / "research/hub/CONTRIBUTING.md").read_text()
    rendered_guide = (ROOT / "research/hub/contributing.qmd").read_text()
    config = yaml.safe_load((ROOT / "research/hub/_quarto.yml").read_text())

    for phrase in (
        "Add a question",
        "Add a study",
        "Add a dataset",
        "Link a run",
        "Build generated outputs",
        "Check generated outputs",
    ):
        assert phrase in guide
    for command in (
        "uv run python scripts/build_research_hub.py",
        "quarto render research/hub",
        "uv run python scripts/build_research_hub.py --check",
    ):
        assert command in guide
    assert "construction" in guide.lower()
    assert "evaluation scope" in guide.lower()
    assert "confidence boundary" in guide.lower()
    assert "canonical" in guide.lower()
    assert "data index" in guide.lower()
    assert "machine-readable catalog" in guide.lower()
    assert "CONTRIBUTING.md" in rendered_guide
    assert any(
        row.get("text") == "Contribute" and row.get("href") == "contributing.qmd"
        for row in config["website"]["navbar"]["left"]
    )

    required_fields = {
        "theme.qmd.example": {
            "theme_id",
            "title",
            "plain_english",
            "current_conclusion",
            "decision",
            "confidence",
            "confidence_boundary",
            "next_test_ids",
            "updated_on",
            "evidence_refs",
        },
        "question.qmd.example": {
            "question_id",
            "title",
            "primary_theme_id",
            "theme_ids",
            "plain_english",
            "status",
            "current_answer",
            "confidence",
            "confidence_boundary",
            "study_ids",
            "open_subquestions",
            "next_test_ids",
            "updated_on",
            "evidence_refs",
        },
        "study.qmd.example": {
            "study_id",
            "title",
            "primary_theme_id",
            "theme_ids",
            "study_type",
            "question_ids",
            "approach_ids",
            "status",
            "outcome",
            "objective",
            "hypothesis",
            "rationale",
            "rationale_provenance",
            "protocol_summary",
            "conclusion",
            "decision",
            "decision_summary",
            "system_ids",
            "representative_system_ids",
            "comparison_system_ids",
            "evidence_refs",
            "dataset_ids",
            "started_on",
            "concluded_on",
            "supersedes",
            "superseded_by",
            "limitations",
        },
        "test.qmd.example": {
            "test_id",
            "title",
            "primary_theme_id",
            "theme_ids",
            "stage",
            "summary",
            "plain_english",
            "why_it_matters",
            "mechanism_steps",
            "linked_question_ids",
            "missing_readiness_fields",
            "hypothesis",
            "rationale",
            "expected_signal",
            "decision_rule",
            "dependencies",
            "priority",
            "evaluation_boundary",
            "artifact_plan",
            "next_action",
        },
        "dataset.qmd.example": {
            "dataset_id",
            "title",
            "plain_english",
            "category",
            "provenance",
            "method",
            "version",
            "repository_path",
            "sensitivity",
            "sharing",
            "leakage_constraints",
            "permitted_use",
            "study_ids",
            "availability",
            "reproducibility",
        },
        "background.qmd.example": {
            "resource_id",
            "title",
            "resource_type",
            "summary",
            "canonical_source_path",
            "citations",
            "tags",
            "question_ids",
            "reviewed_on",
        },
    }
    for filename, required in required_fields.items():
        path = ROOT / "research/hub/templates" / filename
        assert path.is_file()
        lines = path.read_text().splitlines()
        closing = lines.index("---", 1)
        metadata = yaml.safe_load("\n".join(lines[1:closing]))
        assert required <= set(metadata), filename


def test_quarto_pre_render_reflects_an_added_question_from_another_cwd(
    tmp_path: Path,
) -> None:
    copied_root = tmp_path / "repository-copy"
    copied_root.mkdir()
    for name in (
        "configs",
        "data",
        "docs",
            "experiments",
            "plans",
            "protocols",
        "reports",
        "research",
        "scripts",
        "trec_mlm",
    ):
        shutil.copytree(ROOT / name, copied_root / name)
    for name in ("README.md", "pyproject.toml", "uv.lock"):
        shutil.copy2(ROOT / name, copied_root / name)
    shutil.rmtree(copied_root / "research/hub/_site", ignore_errors=True)

    implementation_sentinel = "COPIED_REPOSITORY_RENDERER_EXECUTED"
    copied_renderer = copied_root / "trec_mlm/research_hub/render.py"
    copied_renderer.write_text(
        copied_renderer.read_text(encoding="utf-8")
        + f"""

_copied_render_questions_index_qmd = render_questions_index_qmd


def render_questions_index_qmd(build: HubBuild) -> str:
    return (
        _copied_render_questions_index_qmd(build)
        + "\\n{implementation_sentinel}\\n"
    )
""",
        encoding="utf-8",
    )

    title = "Can automatic contribution propagation be trusted?"
    question_id = "automatic-propagation-proof"
    (copied_root / f"research/hub/questions/{question_id}.qmd").write_text(
        f"""---
question_id: {question_id}
title: "{title}"
primary_theme_id: query-only-routing
theme_ids: [query-only-routing]
plain_english: "Does one valid question update every derived reader surface?"
status: open
current_answer: "This temporary record exists only to test automatic propagation."
confidence: low
confidence_boundary: "Build integration only; this is not research evidence."
study_ids: []
open_subquestions: ["Will all generated and rendered surfaces agree?"]
next_test_ids: []
evidence_refs: [reports/trec_2026_experiment_ledger.md]
updated_on: 2026-07-24
---

Temporary integration-proof record.
""",
        encoding="utf-8",
    )

    copied_pythonpath = os.pathsep.join(
        filter(None, (str(copied_root), os.environ.get("PYTHONPATH")))
    )
    env = {
        **os.environ,
        "PYTHONPATH": copied_pythonpath,
        "UV_NO_SYNC": "1",
        "UV_PROJECT_ENVIRONMENT": str(ROOT / ".venv"),
    }
    completed = subprocess.run(
        [
            "quarto",
            "render",
            str(copied_root / "research/hub/questions/index.qmd"),
        ],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr

    for path in (
        "research/hub/generated/questions-index.qmd",
        "research/hub/generated/themes/query-only-routing.qmd",
        "research/hub/generated/history.qmd",
        "research/catalog.json",
        "research/hub/_site/questions/index.html",
    ):
        assert title in (copied_root / path).read_text(), path
    assert (
        implementation_sentinel
        in (copied_root / "research/hub/generated/questions-index.qmd").read_text()
    )
    assert (
        implementation_sentinel
        in (copied_root / "research/hub/_site/questions/index.html").read_text()
    )
