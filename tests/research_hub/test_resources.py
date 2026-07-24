from __future__ import annotations

from pathlib import Path

import pytest

from scripts.stage_research_hub_resources import RESOURCE_ALLOWLIST, stage_resources
from trec_mlm.research_hub.build import build_research_hub
from trec_mlm.research_hub.render import render_hub_outputs


ROOT = Path(__file__).resolve().parents[2]


def test_background_records_use_audited_sources_and_named_evidence() -> None:
    build = build_research_hub(ROOT)
    background = {record.record_id: record for record in build.sources.background}

    assert (
        background["llm-routing-literature"].metadata["canonical_source_path"]
        == "reports/trec_2026_approaches_guide.artifact.json"
    )
    assert "competition" not in background["related-competitions"].title.lower()
    assert "RouterBench" in background["llm-routing-literature"].body
    assert "SRCB" in background["trec-2025-literature"].body
    assert all("placeholder" not in record.body.lower() for record in background.values())
    assert all(
        "deep dive" in record.body.lower() for record in background.values()
    )


def test_background_index_is_generated_and_grouped() -> None:
    build = build_research_hub(ROOT)
    output = render_hub_outputs(build)[
        Path("research/hub/generated/background-index.qmd")
    ]

    assert "## TREC 2025" in output
    assert "## Wider methods and evaluations" in output
    assert "## Current model and reference planning" in output
    assert "hub-background-row" in output
    assert "listing:" not in output
    for record in build.sources.background:
        assert f'{record.record_id}.qmd"' in output


def _write_allowlisted_sources(root: Path, html: str = "<!doctype html><p>safe</p>") -> None:
    for source in RESOURCE_ALLOWLIST:
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")


def test_stage_resources_copies_only_the_exact_allowlist(tmp_path: Path) -> None:
    _write_allowlisted_sources(tmp_path)
    unrelated = tmp_path / "reports/not-allowlisted.html"
    unrelated.write_text("<p>do not copy</p>", encoding="utf-8")
    output_dir = tmp_path / "research/hub/_site"

    staged = stage_resources(tmp_path, output_dir)

    expected = tuple(
        sorted(output_dir / destination for destination in RESOURCE_ALLOWLIST.values())
    )
    assert staged == expected
    assert all(path.read_text(encoding="utf-8").endswith("<p>safe</p>") for path in staged)
    assert not (output_dir / "library/not-allowlisted.html").exists()


def test_stage_resources_rejects_output_outside_hub_site(tmp_path: Path) -> None:
    _write_allowlisted_sources(tmp_path)

    with pytest.raises(ValueError, match="research/hub/_site"):
        stage_resources(tmp_path, tmp_path / "elsewhere")


def test_stage_resources_rejects_symlinked_site_root(tmp_path: Path) -> None:
    _write_allowlisted_sources(tmp_path)
    outside = tmp_path / "outside-site"
    outside.mkdir()
    site = tmp_path / "research/hub/_site"
    site.parent.mkdir(parents=True)
    site.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        stage_resources(tmp_path, site)

    assert not (outside / "library").exists()


def test_stage_resources_rejects_symlinked_destination_component(
    tmp_path: Path,
) -> None:
    _write_allowlisted_sources(tmp_path)
    site = tmp_path / "research/hub/_site"
    real_library = site / "real-library"
    real_library.mkdir(parents=True)
    (site / "library").symlink_to(real_library, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink"):
        stage_resources(tmp_path, site)

    assert not tuple(real_library.iterdir())


@pytest.mark.parametrize(
    ("unsafe_html", "message"),
    [
        ('<a href="file:///tmp/private">local</a>', "file URL"),
        ("<p>/home/npatta01/private</p>", "local filesystem path"),
        ("-----BEGIN PRIVATE KEY-----", "private-key"),
        ('<script src="https://cdn.example/app.js"></script>', "external executable"),
        ('<link rel="stylesheet" href="//cdn.example/app.css">', "external executable"),
    ],
)
def test_stage_resources_rejects_private_or_executable_content(
    tmp_path: Path,
    unsafe_html: str,
    message: str,
) -> None:
    _write_allowlisted_sources(tmp_path, unsafe_html)

    with pytest.raises(ValueError, match=message):
        stage_resources(tmp_path, tmp_path / "research/hub/_site")

    assert not (tmp_path / "research/hub/_site/library").exists()


@pytest.mark.parametrize(
    "local_path",
    [
        "/home/alice/private/results.json",
        r"\/home\/alice\/private\/results.json",
        "/Users/alice/Library/Application Support/results.json",
        "/tmp/private/results.json",
        r"C:\Users\alice\private\results.json",
        r"D:\private\results.json",
        r"\\fileserver\research\alice\results.json",
        r"\\\\fileserver\\research\\alice\\results.json",
    ],
)
def test_stage_resources_rejects_portable_local_filesystem_paths(
    tmp_path: Path,
    local_path: str,
) -> None:
    _write_allowlisted_sources(tmp_path, f"<p>{local_path}</p>")

    with pytest.raises(ValueError, match="local filesystem path"):
        stage_resources(tmp_path, tmp_path / "research/hub/_site")

    assert not (tmp_path / "research/hub/_site/library").exists()


def test_stage_resources_allows_relative_and_site_absolute_paths(
    tmp_path: Path,
) -> None:
    html = """
    <script src="assets/app.js"></script>
    <link rel="stylesheet" href="./assets/app.css">
    <img src="../images/chart.svg" alt="">
    <a href="/api/results.json">API results</a>
    <a href="/research/hub/library/index.html">Library</a>
    <a href="https://docs.example/home/alice/guide">Documentation</a>
    <a href="//docs.example/tmp/guide">Protocol-relative documentation</a>
    <script>const docs = "https:\\/\\/docs.example\\/home\\/alice";</script>
    """
    _write_allowlisted_sources(tmp_path, html)

    staged = stage_resources(tmp_path, tmp_path / "research/hub/_site")

    assert len(staged) == len(RESOURCE_ALLOWLIST)
    assert all(path.read_text(encoding="utf-8") == html for path in staged)


def test_stage_resources_rejects_external_base_for_relative_script(
    tmp_path: Path,
) -> None:
    _write_allowlisted_sources(
        tmp_path,
        '<base href="https://cdn.example/assets/"><script src="app.js"></script>',
    )

    with pytest.raises(ValueError, match="external base"):
        stage_resources(tmp_path, tmp_path / "research/hub/_site")


def test_stage_resources_recursively_rejects_executable_iframe_srcdoc(
    tmp_path: Path,
) -> None:
    _write_allowlisted_sources(
        tmp_path,
        (
            '<iframe srcdoc="&lt;script '
            'src=&quot;https://cdn.example/app.js&quot;&gt;&lt;/script&gt;">'
            "</iframe>"
        ),
    )

    with pytest.raises(ValueError, match="external executable"):
        stage_resources(tmp_path, tmp_path / "research/hub/_site")
