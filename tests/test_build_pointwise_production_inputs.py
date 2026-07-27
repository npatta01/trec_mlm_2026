import pytest

from scripts.build_pointwise_production_inputs import main


def test_production_cli_rejects_test_only_expected_count_override(tmp_path) -> None:
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                "--discovery",
                str(tmp_path / "discovery.json"),
                "--grounded",
                str(tmp_path / "grounded.json"),
                "--refinement-records",
                str(tmp_path / "records.json"),
                "--output-dir",
                str(tmp_path / "output"),
                "--expected-count",
                "1",
            ]
        )

    assert not (tmp_path / "output").exists()
