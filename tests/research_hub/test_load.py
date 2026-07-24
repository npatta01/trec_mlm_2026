from pathlib import Path

import pytest

from trec_mlm.research_hub.load import load_hub_sources, parse_qmd_record
from trec_mlm.research_hub.model import RECORD_KINDS


def test_parse_qmd_record_preserves_metadata_and_markdown(tmp_path: Path) -> None:
    path = tmp_path / "research/hub/questions/candidate-evidence.qmd"
    path.parent.mkdir(parents=True)
    path.write_text(
        "---\n"
        "question_id: candidate-evidence\n"
        "title: Can candidate evidence help?\n"
        "status: open\n"
        "---\n\n"
        "## In plain English\n\nDoes model information help ranking?\n",
        encoding="utf-8",
    )

    record = parse_qmd_record(tmp_path, path, "question")

    assert record.kind == "question"
    assert record.record_id == "candidate-evidence"
    assert record.title == "Can candidate evidence help?"
    assert record.source_path == "research/hub/questions/candidate-evidence.qmd"
    assert record.body.startswith("## In plain English")


def test_parse_qmd_record_rejects_unsafe_yaml_tag(tmp_path: Path) -> None:
    path = tmp_path / "bad.qmd"
    path.write_text(
        "---\nquestion_id: !!python/object/apply:os.system [echo unsafe]\n"
        "title: Unsafe\n---\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid YAML front matter"):
        parse_qmd_record(tmp_path, path, "question")


def test_load_hub_sources_sorts_records_by_id(tmp_path: Path) -> None:
    directory = tmp_path / "research/hub/questions"
    directory.mkdir(parents=True)
    for record_id in ("zeta", "alpha"):
        (directory / f"{record_id}.qmd").write_text(
            f"---\nquestion_id: {record_id}\ntitle: {record_id.title()}\n---\n",
            encoding="utf-8",
        )

    sources = load_hub_sources(tmp_path)

    assert [record.record_id for record in sources.questions] == ["alpha", "zeta"]


def test_load_hub_sources_loads_themes(tmp_path: Path) -> None:
    directory = tmp_path / "research/hub/themes"
    directory.mkdir(parents=True)
    (directory / "candidate-evidence.qmd").write_text(
        "---\ntheme_id: candidate-evidence\ntitle: Candidate evidence\n---\n",
        encoding="utf-8",
    )

    sources = load_hub_sources(tmp_path)

    assert "theme" in RECORD_KINDS
    assert sources.themes[0].kind == "theme"
    assert sources.themes[0].record_id == "candidate-evidence"
