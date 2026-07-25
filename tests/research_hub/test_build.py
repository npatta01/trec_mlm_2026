from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from trec_mlm.experiment_ledger import LedgerBuild
from trec_mlm.research_hub import build as module
from trec_mlm.research_hub.build import HubBuild, build_research_hub
from trec_mlm.research_hub.model import HubSources
from trec_mlm.research_hub.render import (
    assert_hub_outputs_current,
    render_hub_outputs,
    write_hub_outputs,
)
from trec_mlm.research_hub.validate import ValidationSummary


@pytest.fixture
def sample_build(
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
) -> HubBuild:
    return HubBuild(
        sources=minimal_sources,
        ledger=ledger_build,
        validation=ValidationSummary(
            record_counts={
                "question": 1,
                "study": 1,
                "test": 1,
                "approach": 1,
                "dataset": 1,
                "background": 1,
            },
            linked_system_count=1,
        ),
    )


def test_build_loads_ledger_then_validates_sources(
    tmp_path: Path,
    minimal_sources: HubSources,
    ledger_build: LedgerBuild,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    validation = ValidationSummary(
        record_counts={"question": 1},
        linked_system_count=1,
    )
    calls: list[object] = []
    monkeypatch.setattr(
        module,
        "load_hub_sources",
        lambda root: calls.append(("load", root)) or minimal_sources,
    )
    monkeypatch.setattr(
        module,
        "build_ledger",
        lambda root: calls.append(("ledger", root)) or ledger_build,
    )
    monkeypatch.setattr(
        module,
        "validate_hub_sources",
        lambda sources, ledger, root: (
            calls.append(("validate", sources, ledger, root)) or validation
        ),
    )

    result = build_research_hub(tmp_path)

    assert result == HubBuild(
        sources=minimal_sources,
        ledger=ledger_build,
        validation=validation,
    )
    assert calls == [
        ("load", tmp_path),
        ("ledger", tmp_path),
        ("validate", minimal_sources, ledger_build, tmp_path),
    ]


def test_render_hub_outputs_has_exact_deterministic_path_contract(
    sample_build: HubBuild,
) -> None:
    outputs = render_hub_outputs(sample_build)

    assert set(outputs) == {
        Path("research/catalog.json"),
        Path("research/hub/catalog.json"),
        Path("reports/trec_2026_research_hub_summary.md"),
        Path("research/hub/generated/summary.qmd"),
        Path("research/hub/generated/background-index.qmd"),
        Path("research/hub/generated/data-index.qmd"),
        Path("research/hub/generated/ideas-index.qmd"),
        Path("research/hub/generated/themes-index.qmd"),
        Path("research/hub/generated/questions-index.qmd"),
        Path("research/hub/generated/studies-index.qmd"),
        Path("research/hub/generated/coverage.qmd"),
        Path("research/hub/generated/history.qmd"),
        Path("research/hub/generated/runs.qmd"),
        Path("research/hub/generated/themes/candidate-evidence.qmd"),
        Path("research/hub/generated/ideas/test-candidate-evidence.qmd"),
        Path("research/hub/generated/studies/dense-baseline.md"),
        *{
            Path(f"research/hub/generated/runs/{system.system_id}.qmd")
            for system in sample_build.ledger.systems
        },
    }
    assert all(content.endswith("\n") for content in outputs.values())


def test_write_rolls_back_all_outputs_if_replace_fails(
    sample_build: HubBuild,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = write_hub_outputs(sample_build, tmp_path)
    previous = {path: f"old:{path.name}\n" for path in paths}
    for path, content in previous.items():
        path.write_text(content, encoding="utf-8")

    original_replace = Path.replace
    promotions = 0

    def failing_replace(source: Path, destination: Path) -> Path:
        nonlocal promotions
        destination = Path(destination)
        if ".research-hub-" in source.as_posix() and "previous" not in source.parts:
            promotions += 1
            if promotions == 3:
                raise OSError("simulated replace failure")
        return original_replace(source, destination)

    monkeypatch.setattr(Path, "replace", failing_replace)

    with pytest.raises(OSError, match="simulated replace failure"):
        write_hub_outputs(sample_build, tmp_path)

    assert {path: path.read_text(encoding="utf-8") for path in paths} == previous


def test_current_check_reports_missing_unexpected_and_different_outputs(
    sample_build: HubBuild,
    tmp_path: Path,
) -> None:
    paths = write_hub_outputs(sample_build, tmp_path)
    assert_hub_outputs_current(sample_build, tmp_path)

    missing = paths[0]
    different = paths[1]
    unexpected = tmp_path / "research/hub/generated/runs/stale-run.qmd"
    missing.unlink()
    different.write_text("different\n", encoding="utf-8")
    unexpected.write_text("stale\n", encoding="utf-8")

    with pytest.raises(ValueError) as captured:
        assert_hub_outputs_current(sample_build, tmp_path)

    message = str(captured.value)
    assert f"missing: {missing.relative_to(tmp_path)}" in message
    assert f"different: {different.relative_to(tmp_path)}" in message
    assert f"unexpected: {unexpected.relative_to(tmp_path)}" in message


def test_write_removes_an_unexpected_stale_run_page(
    sample_build: HubBuild,
    tmp_path: Path,
) -> None:
    unexpected = tmp_path / "research/hub/generated/runs/stale-run.qmd"
    unexpected.parent.mkdir(parents=True)
    unexpected.write_text("stale\n", encoding="utf-8")

    write_hub_outputs(sample_build, tmp_path)

    assert not unexpected.exists()
    assert_hub_outputs_current(sample_build, tmp_path)


@pytest.mark.parametrize(
    "system_id",
    ["../escaped", "../../escaped", "nested/run", r"nested\run", "/absolute"],
)
def test_malicious_system_id_cannot_write_outside_repository(
    sample_build: HubBuild,
    tmp_path: Path,
    system_id: str,
) -> None:
    malicious = replace(sample_build.ledger.systems[0], system_id=system_id)
    build = replace(
        sample_build,
        ledger=replace(sample_build.ledger, systems=(malicious,)),
    )
    outside = tmp_path.parent / "escaped.qmd"

    with pytest.raises(ValueError, match="safe filename component"):
        write_hub_outputs(build, tmp_path)

    assert not outside.exists()
    assert not list(tmp_path.rglob("*.qmd"))


def test_output_mapping_path_must_be_normalized_and_repository_relative(
    sample_build: HubBuild,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outside = tmp_path.parent / "outside.txt"
    monkeypatch.setattr(
        "trec_mlm.research_hub.render.render_hub_outputs",
        lambda build: {Path("../outside.txt"): "escaped\n"},
    )

    with pytest.raises(ValueError, match="repository-relative"):
        write_hub_outputs(sample_build, tmp_path)

    assert not outside.exists()


def test_output_destination_cannot_escape_through_repository_symlink(
    sample_build: HubBuild,
    tmp_path: Path,
) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "research").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="remain repository-relative"):
        write_hub_outputs(sample_build, tmp_path)

    assert not list(outside.iterdir())
