from pathlib import Path

import pytest

from scripts.update_ranking_diagnosis_with_analogues import resolve_base_path
from scripts.build_sparse_failure_analogue_notebook import code, markdown
from scripts.build_sparse_failure_cause_diagnostic import resolve_repository_input


def test_resolve_base_path_fails_without_portable_repository_input(
    tmp_path: Path,
) -> None:
    output = tmp_path / "missing-artifact.json"

    with pytest.raises(FileNotFoundError, match="Pass --base explicitly"):
        resolve_base_path(None, output)


def test_resolve_base_path_reuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "artifact.json"
    output.write_text("{}\n", encoding="utf-8")

    assert resolve_base_path(None, output) == output


def test_notebook_cells_have_deterministic_ids() -> None:
    assert markdown("Stable source")["id"] == markdown("Stable source")["id"]
    assert code("Stable source")["id"] == code("Stable source")["id"]
    assert markdown("Stable source")["id"] != code("Stable source")["id"]


def test_cause_builder_resolves_relative_repository_inputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(root)

    resolved = resolve_repository_input(
        Path("reports/trec_2026_sparse_failure_analogue_cases.tsv")
    )

    assert resolved.is_absolute()
    assert resolved.relative_to(root) == Path(
        "reports/trec_2026_sparse_failure_analogue_cases.tsv"
    )
