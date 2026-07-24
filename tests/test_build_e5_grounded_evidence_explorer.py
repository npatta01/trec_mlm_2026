from __future__ import annotations

import argparse
from collections.abc import Iterator, Mapping
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from scripts import build_e5_grounded_evidence_explorer as e5_builder
from trec_mlm.artifact_io import manifest_aggregate_sha256, manifest_entries


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/build_e5_grounded_evidence_explorer.py"
QUERY_IDS = ("q1", "q2")
CANDIDATE_IDS = tuple(f"c{index}" for index in range(1, 23))
SYSTEMS = (
    "dense_reference",
    "profile_only",
    "dense_profile_equal",
    "raw_hierarchical",
    "raw_hierarchical_dense_equal",
)
SECRET_QUESTION = "secret historical question"
SECRET_ANSWER = "secret historical answer"
SOURCE_NAMES = (
    "queries",
    "qrels",
    "profiles",
    "evaluation",
    "final_trec_run",
    "run_validation",
    "refinement_inputs",
    "dense_reference_ranking",
    "raw_hierarchical_ranking",
    "raw_hierarchical_dense_equal_ranking",
    "selected_record_ids",
    "top_record_similarities",
    "construction_records",
    "profile_only_ranking",
    "dense_profile_equal_ranking",
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _finalize_fixture_artifact(
    root: Path, artifact_type: str, *, version: int
) -> None:
    entries = manifest_entries(root)
    _write_json(
        root / "manifest.json",
        {
            "artifact_type": artifact_type,
            "version": version,
            "qrel_free": True,
            "files": entries,
            "aggregate_sha256": manifest_aggregate_sha256(entries),
        },
    )


class E5Fixture(Mapping[str, Path]):
    def __init__(self, paths: Mapping[str, Path]) -> None:
        self._paths = dict(paths)

    def __getitem__(self, key: str) -> Path:
        return self._paths[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._paths)

    def __len__(self) -> int:
        return len(self._paths)

    def corrupt_record_owner(self) -> None:
        records_path = self["construction"] / "records.json"
        records = json.loads(records_path.read_text(encoding="utf-8"))
        records[0]["candidate_id"] = "c2"
        _write_json(records_path, records)

    def source_paths(self) -> dict[str, Path]:
        refinement = self["refinement"]
        construction = self["construction"]
        return {
            "queries": self["queries"],
            "qrels": self["qrels"],
            "profiles": self["profiles"],
            "evaluation": self["evaluation"],
            "final_trec_run": (
                refinement
                / "runs"
                / "raw_hierarchical_dense_equal.trec"
            ),
            "run_validation": refinement / "runs" / "validation.json",
            "refinement_inputs": refinement / "inputs.json",
            "dense_reference_ranking": (
                refinement / "rankings" / "dense_reference.json"
            ),
            "raw_hierarchical_ranking": (
                refinement / "rankings" / "raw_hierarchical.json"
            ),
            "raw_hierarchical_dense_equal_ranking": (
                refinement
                / "rankings"
                / "raw_hierarchical_dense_equal.json"
            ),
            "selected_record_ids": refinement / "selected_record_ids.json",
            "top_record_similarities": (
                refinement / "top_record_similarities.npy"
            ),
            "construction_records": construction / "records.json",
            "profile_only_ranking": (
                construction / "rankings" / "profile_only.json"
            ),
            "dense_profile_equal_ranking": (
                construction / "rankings" / "dense_profile_equal.json"
            ),
        }


@pytest.fixture
def e5_fixture(tmp_path: Path) -> E5Fixture:
    refinement = tmp_path / "refinement"
    construction = tmp_path / "construction"
    refinement_rankings = refinement / "rankings"
    construction_rankings = construction / "rankings"
    run_dir = refinement / "runs"
    refinement_rankings.mkdir(parents=True)
    construction_rankings.mkdir(parents=True)
    run_dir.mkdir()

    _write_json(
        refinement / "inputs.json",
        {"query_ids": list(QUERY_IDS), "candidate_ids": list(CANDIDATE_IDS)},
    )
    queries = tmp_path / "queries.tsv"
    queries.write_text(
        "q1\tA private target question\nq2\tWhich law did they invoke?\n",
        encoding="utf-8",
    )

    rankings = {
        system: {query_id: list(CANDIDATE_IDS) for query_id in QUERY_IDS}
        for system in SYSTEMS
    }
    for system, payload in rankings.items():
        root = (
            refinement_rankings
            if system.startswith("raw_") or system == "dense_reference"
            else construction_rankings
        )
        _write_json(root / f"{system}.json", payload)

    selected_ids: list[list[list[str]]] = []
    records: list[dict[str, str]] = []
    similarities = np.zeros((2, len(CANDIDATE_IDS), 10), dtype="<f8")
    for query_index, query_id in enumerate(QUERY_IDS):
        selected_for_query: list[list[str]] = []
        for candidate_index, candidate_id in enumerate(CANDIDATE_IDS):
            record_ids = [
                f"{query_id}-{candidate_id}-r{rank}" for rank in range(1, 11)
            ]
            selected_for_query.append(record_ids)
            for rank, record_id in enumerate(record_ids, start=1):
                records.append(
                    {
                        "record_id": record_id,
                        "candidate_id": candidate_id,
                        "source_query_id": f"source-{query_id}-{rank}",
                        "question": (
                            SECRET_QUESTION
                            if (query_id, candidate_id, rank) == ("q1", "c1", 1)
                            else f"Historical question {rank}"
                        ),
                        "response": (
                            SECRET_ANSWER
                            if (query_id, candidate_id, rank) == ("q1", "c1", 1)
                            else f"Answer {rank} for {candidate_id}."
                        ),
                    }
                )
            start = 0.9 - candidate_index / 100
            similarities[query_index, candidate_index] = np.asarray(
                [start - rank / 100 for rank in range(10)], dtype="<f8"
            )
        selected_ids.append(selected_for_query)
    _write_json(refinement / "selected_record_ids.json", selected_ids)
    np.save(refinement / "top_record_similarities.npy", similarities)
    _write_json(construction / "records.json", records)

    qrels = tmp_path / "qrels.txt"
    qrels.write_text(
        "".join(
            f"{query_id} 0 {candidate_id} "
            f"{1 if (query_id, candidate_id) == ('q1', 'c22') else 0}\n"
            for query_id in QUERY_IDS
            for candidate_id in CANDIDATE_IDS
        ),
        encoding="utf-8",
    )
    evaluation = tmp_path / "evaluation.json"
    _write_json(
        evaluation,
        {
            "per_query_rows": [
                {
                    "query_id": query_id,
                    "system": system,
                    "ndcg_at_10": 0.0,
                    "reciprocal_rank": (
                        1.0 / 22 if query_id == "q1" else 0.0
                    ),
                }
                for query_id in QUERY_IDS
                for system in SYSTEMS
            ]
        },
    )
    profiles = tmp_path / "profiles.json"
    _write_json(
        profiles,
        {
            "profile_version": "test-v1",
            "profiles": [
                {
                    "candidate_id": candidate_id,
                    "profile_summary": f"Private profile for {candidate_id}",
                }
                for candidate_id in CANDIDATE_IDS
            ],
        },
    )
    (run_dir / "raw_hierarchical_dense_equal.trec").write_text(
        "".join(
            f"{query_id} Q0 {candidate_id} {rank} "
            f"{len(CANDIDATE_IDS) - rank + 1} e5\n"
            for query_id in QUERY_IDS
            for rank, candidate_id in enumerate(CANDIDATE_IDS, start=1)
        ),
        encoding="utf-8",
    )
    _write_json(run_dir / "validation.json", {})
    _finalize_fixture_artifact(
        refinement,
        "raw_response_refinement_manifest",
        version=1,
    )
    _finalize_fixture_artifact(
        construction,
        "candidate_semantic_profile_manifest",
        version=2,
    )
    return E5Fixture(
        {
            "root": tmp_path,
            "refinement": refinement,
            "construction": construction,
            "evaluation": evaluation,
            "queries": queries,
            "qrels": qrels,
            "profiles": profiles,
        }
    )


def run_cli(
    e5_fixture: Mapping[str, Path],
    *,
    bundle: Path,
    html: Path,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--refinement",
            str(e5_fixture["refinement"]),
            "--construction",
            str(e5_fixture["construction"]),
            "--evaluation",
            str(e5_fixture["evaluation"]),
            "--queries",
            str(e5_fixture["queries"]),
            "--qrels",
            str(e5_fixture["qrels"]),
            "--profiles",
            str(e5_fixture["profiles"]),
            "--bundle-output",
            str(bundle),
            "--html-output",
            str(html),
        ],
        cwd=e5_fixture["root"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def configure_in_process_build(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> dict[str, Path]:
    live = tmp_path / "live"
    live.mkdir()
    bundle = live / "evidence.json"
    html = live / "explorer.html"
    bundle.write_bytes(b"old-bundle")
    html.write_bytes(b"old-html")
    staged_bundle = live / ".evidence.json.building.test.json"
    staged_html = live / ".explorer.html.building.test.html"
    arguments = argparse.Namespace(
        refinement=tmp_path / "refinement",
        construction=tmp_path / "construction",
        evaluation=tmp_path / "evaluation.json",
        queries=tmp_path / "queries.tsv",
        qrels=tmp_path / "qrels.txt",
        profiles=tmp_path / "profiles.json",
        bundle_output=bundle,
        html_output=html,
    )
    monkeypatch.setattr(e5_builder, "parse_args", lambda: arguments)
    monkeypatch.setattr(
        e5_builder,
        "build_evidence_bundle",
        lambda **_kwargs: {
            "meta": {
                "query_count": 2,
                "evidence_per_candidate": 3,
            }
        },
    )

    def render_staged_html(**kwargs: object) -> None:
        output = kwargs["output"]
        assert isinstance(output, Path)
        output.write_bytes(b"new-html")

    monkeypatch.setattr(
        e5_builder,
        "run_generic_builder",
        render_staged_html,
    )
    stages = {
        ".json": staged_bundle,
        ".html": staged_html,
    }
    monkeypatch.setattr(
        e5_builder,
        "_temporary_output_path",
        lambda _output, *, suffix: stages[suffix],
    )
    return {
        "bundle": bundle,
        "html": html,
        "staged_bundle": staged_bundle,
        "staged_html": staged_html,
    }


def capture_output_backups(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[Path, Path | None]:
    captured: dict[Path, Path | None] = {}
    original = e5_builder._backup_existing_output

    def capture(path: Path) -> Path | None:
        backup = original(path)
        captured[path] = backup
        return backup

    monkeypatch.setattr(
        e5_builder,
        "_backup_existing_output",
        capture,
    )
    return captured


def test_second_promotion_failure_rolls_back_and_removes_backups(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_in_process_build(monkeypatch, tmp_path)
    backups = capture_output_backups(monkeypatch)
    original_replace = Path.replace

    def fail_second_promotion(
        source: Path,
        target: str | os.PathLike[str],
    ) -> Path:
        destination = Path(target)
        if (
            source == paths["staged_html"]
            and destination == paths["html"]
        ):
            raise OSError("injected second-promotion failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_promotion)

    with pytest.raises(
        e5_builder.E5ExplorerBuildError,
        match="output publication failed",
    ):
        e5_builder.main()

    assert paths["bundle"].read_bytes() == b"old-bundle"
    assert paths["html"].read_bytes() == b"old-html"
    assert not paths["staged_bundle"].exists()
    assert not paths["staged_html"].exists()
    assert backups
    assert not list(tmp_path.rglob(".*.rollback.*"))


def test_rollback_failure_preserves_and_reports_recovery_backup(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = configure_in_process_build(monkeypatch, tmp_path)
    backups = capture_output_backups(monkeypatch)
    original_replace = Path.replace

    def fail_promotion_and_rollback(
        source: Path,
        target: str | os.PathLike[str],
    ) -> Path:
        destination = Path(target)
        if (
            source == paths["staged_html"]
            and destination == paths["html"]
        ):
            raise OSError("injected second-promotion failure")
        bundle_backup = backups.get(paths["bundle"])
        if (
            bundle_backup is not None
            and source == bundle_backup
            and destination == paths["bundle"]
        ):
            raise OSError("injected rollback failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_promotion_and_rollback)

    with pytest.raises(
        e5_builder.E5ExplorerBuildError,
        match="recovery required",
    ) as captured_error:
        e5_builder.main()

    bundle_backup = backups[paths["bundle"]]
    html_backup = backups[paths["html"]]
    assert bundle_backup is not None
    assert html_backup is not None
    assert str(bundle_backup) in str(captured_error.value)
    assert bundle_backup.read_bytes() == b"old-bundle"
    assert not html_backup.exists()
    assert json.loads(
        paths["bundle"].read_text(encoding="utf-8")
    )["meta"]["query_count"] == 2
    assert paths["html"].read_bytes() == b"old-html"
    assert not paths["staged_bundle"].exists()
    assert not paths["staged_html"].exists()
    assert list(tmp_path.rglob(".*.rollback.*")) == [
        bundle_backup
    ]
    assert SECRET_QUESTION not in str(captured_error.value)
    assert SECRET_ANSWER not in str(captured_error.value)


def test_backup_cleanup_failure_keeps_valid_pair_and_reports_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    paths = configure_in_process_build(monkeypatch, tmp_path)
    backups = capture_output_backups(monkeypatch)
    cleanup_ready = False
    original_backup = e5_builder._backup_existing_output

    def enable_cleanup_failure(path: Path) -> Path | None:
        nonlocal cleanup_ready
        backup = original_backup(path)
        if len(backups) == 2:
            cleanup_ready = True
        return backup

    monkeypatch.setattr(
        e5_builder,
        "_backup_existing_output",
        enable_cleanup_failure,
    )
    original_unlink = Path.unlink

    def fail_bundle_backup_cleanup(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> None:
        if (
            cleanup_ready
            and path == backups.get(paths["bundle"])
        ):
            raise OSError("injected backup-cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_bundle_backup_cleanup)

    assert e5_builder.main() == 0

    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    bundle_backup = backups[paths["bundle"]]
    html_backup = backups[paths["html"]]
    assert bundle_backup is not None
    assert html_backup is not None
    assert summary["warning"] == (
        "published outputs are valid; backup cleanup incomplete"
    )
    assert summary["recovery_backups"] == [str(bundle_backup)]
    assert captured.err == ""
    assert json.loads(
        paths["bundle"].read_text(encoding="utf-8")
    )["meta"]["query_count"] == 2
    assert paths["html"].read_bytes() == b"new-html"
    assert bundle_backup.read_bytes() == b"old-bundle"
    assert not html_backup.exists()
    assert not paths["staged_bundle"].exists()
    assert not paths["staged_html"].exists()
    assert list(tmp_path.rglob(".*.rollback.*")) == [
        bundle_backup
    ]
    assert SECRET_QUESTION not in captured.out + captured.err
    assert SECRET_ANSWER not in captured.out + captured.err


def test_cli_builds_bundle_and_html_without_printing_raw_text(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    bundle = tmp_path / "private" / "evidence.json"
    html = tmp_path / "rendered" / "explorer.html"

    completed = run_cli(e5_fixture, bundle=bundle, html=html)

    assert completed.returncode == 0, completed.stderr
    assert bundle.is_file() and html.is_file()
    assert SECRET_QUESTION not in completed.stdout
    assert SECRET_QUESTION not in completed.stderr
    assert SECRET_ANSWER not in completed.stdout
    assert SECRET_ANSWER not in completed.stderr
    assert json.loads(bundle.read_text(encoding="utf-8"))["meta"]["query_count"] == 2
    assert '"candidate_limit":20' in html.read_text(encoding="utf-8")


def test_cli_validation_failure_preserves_prior_outputs(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    bundle = tmp_path / "private" / "evidence.json"
    bundle.parent.mkdir()
    bundle.write_bytes(b"known-good-bundle")
    html = tmp_path / "explorer.html"
    html.write_bytes(b"known-good-html")
    e5_fixture.corrupt_record_owner()

    completed = run_cli(e5_fixture, bundle=bundle, html=html)

    assert completed.returncode == 2
    assert "error: evidence bundle validation failed" in completed.stderr
    assert bundle.read_bytes() == b"known-good-bundle"
    assert html.read_bytes() == b"known-good-html"
    assert SECRET_QUESTION not in completed.stdout + completed.stderr
    assert SECRET_ANSWER not in completed.stdout + completed.stderr


def test_cli_renderer_failure_preserves_prior_outputs_and_removes_temps(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    bundle = tmp_path / "private" / "evidence.json"
    bundle.parent.mkdir()
    bundle.write_bytes(b"known-good-bundle")
    html = tmp_path / "explorer.html"
    html.write_bytes(b"known-good-html")
    e5_fixture["profiles"].write_text("{invalid", encoding="utf-8")

    completed = run_cli(e5_fixture, bundle=bundle, html=html)

    assert completed.returncode == 2
    assert bundle.read_bytes() == b"known-good-bundle"
    assert html.read_bytes() == b"known-good-html"
    assert not [
        path
        for path in tmp_path.rglob(".*")
        if ".building." in path.name
    ]


def test_cli_rejects_bundle_path_that_aliases_live_html(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    html = tmp_path / "explorer.html"
    html.write_text("known-good", encoding="utf-8")

    completed = run_cli(e5_fixture, bundle=html, html=html)

    assert completed.returncode == 2
    assert html.read_text(encoding="utf-8") == "known-good"


@pytest.mark.parametrize("source_name", SOURCE_NAMES)
def test_cli_rejects_bundle_output_aliasing_any_source_before_writing(
    e5_fixture: E5Fixture,
    tmp_path: Path,
    source_name: str,
) -> None:
    source = e5_fixture.source_paths()[source_name]
    source_before = source.read_bytes()
    html = tmp_path / "live.html"
    html.write_bytes(b"known-good-html")

    completed = run_cli(e5_fixture, bundle=source, html=html)

    assert completed.returncode == 2
    assert "error: output path aliases a source input" in completed.stderr
    assert source.read_bytes() == source_before
    assert html.read_bytes() == b"known-good-html"
    assert SECRET_QUESTION not in completed.stdout + completed.stderr
    assert SECRET_ANSWER not in completed.stdout + completed.stderr


def test_cli_rejects_html_output_hard_linked_to_source_before_writing(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    source = e5_fixture["profiles"]
    source_before = source.read_bytes()
    html = tmp_path / "source-alias.html"
    os.link(source, html)
    bundle = tmp_path / "private" / "evidence.json"

    completed = run_cli(e5_fixture, bundle=bundle, html=html)

    assert completed.returncode == 2
    assert "error: output path aliases a source input" in completed.stderr
    assert source.read_bytes() == source_before
    assert html.read_bytes() == source_before
    assert not bundle.exists()


def test_cli_rejects_hard_linked_output_paths_before_writing(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    bundle = tmp_path / "evidence.json"
    bundle.write_bytes(b"known-good-output")
    html = tmp_path / "explorer.html"
    os.link(bundle, html)

    completed = run_cli(e5_fixture, bundle=bundle, html=html)

    assert completed.returncode == 2
    assert "error: bundle and HTML output paths must be different" in completed.stderr
    assert bundle.read_bytes() == b"known-good-output"
    assert html.read_bytes() == b"known-good-output"


def test_cli_rejects_bundle_output_symlink_before_writing(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    target = tmp_path / "protected-bundle.json"
    target.write_bytes(b"known-good-bundle")
    bundle = tmp_path / "evidence.json"
    bundle.symlink_to(target)
    html = tmp_path / "explorer.html"
    html.write_bytes(b"known-good-html")

    completed = run_cli(e5_fixture, bundle=bundle, html=html)

    assert completed.returncode == 2
    assert "error: bundle output path must not be a symlink" in completed.stderr
    assert bundle.is_symlink()
    assert target.read_bytes() == b"known-good-bundle"
    assert html.read_bytes() == b"known-good-html"


def test_cli_rejects_html_output_symlink_before_writing_bundle(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    target = tmp_path / "protected.html"
    target.write_bytes(b"known-good-html")
    html = tmp_path / "explorer.html"
    html.symlink_to(target)
    bundle = tmp_path / "private" / "evidence.json"

    completed = run_cli(e5_fixture, bundle=bundle, html=html)

    assert completed.returncode == 2
    assert "error: HTML output path must not be a symlink" in completed.stderr
    assert html.is_symlink()
    assert target.read_bytes() == b"known-good-html"
    assert not bundle.exists()


def test_cli_allows_private_bundle_beneath_refinement_root(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    bundle = e5_fixture["refinement"] / "explorer" / "evidence.json"
    html = tmp_path / "explorer.html"

    completed = run_cli(e5_fixture, bundle=bundle, html=html)

    assert completed.returncode == 0, completed.stderr
    assert bundle.is_file()
    assert html.is_file()


def test_cli_rejects_canonical_html_alias_to_run_validation_before_bundle_write(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    validation = e5_fixture.source_paths()["run_validation"]
    validation_before = validation.read_bytes()
    html = validation.parent / "unused" / ".." / validation.name
    bundle = tmp_path / "private" / "evidence.json"

    completed = run_cli(e5_fixture, bundle=bundle, html=html)

    assert completed.returncode == 2
    assert "error: output path aliases a source input" in completed.stderr
    assert validation.read_bytes() == validation_before
    assert not bundle.exists()


def test_cli_rejects_html_hard_link_to_run_validation_before_bundle_write(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    validation = e5_fixture.source_paths()["run_validation"]
    validation_before = validation.read_bytes()
    html = tmp_path / "validation-alias.html"
    os.link(validation, html)
    bundle = tmp_path / "private" / "evidence.json"

    completed = run_cli(e5_fixture, bundle=bundle, html=html)

    assert completed.returncode == 2
    assert "error: output path aliases a source input" in completed.stderr
    assert validation.read_bytes() == validation_before
    assert html.read_bytes() == validation_before
    assert not bundle.exists()


def test_cli_reserves_absent_run_validation_path_before_bundle_write(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    validation = e5_fixture.source_paths()["run_validation"]
    validation.unlink()
    html = tmp_path / "live.html"
    html.write_bytes(b"known-good-html")

    completed = run_cli(e5_fixture, bundle=validation, html=html)

    assert completed.returncode == 2
    assert "error: output path aliases a source input" in completed.stderr
    assert not validation.exists()
    assert html.read_bytes() == b"known-good-html"
