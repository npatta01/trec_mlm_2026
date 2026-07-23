from __future__ import annotations

from collections.abc import Iterator, Mapping
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


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


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


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
                    "reciprocal_rank": 0.0,
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


def test_cli_validation_failure_preserves_live_html(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    bundle = tmp_path / "private" / "evidence.json"
    html = tmp_path / "explorer.html"
    html.write_text("known-good", encoding="utf-8")
    e5_fixture.corrupt_record_owner()

    completed = run_cli(e5_fixture, bundle=bundle, html=html)

    assert completed.returncode == 2
    assert "error: evidence bundle validation failed" in completed.stderr
    assert html.read_text(encoding="utf-8") == "known-good"
    assert SECRET_QUESTION not in completed.stdout + completed.stderr
    assert SECRET_ANSWER not in completed.stdout + completed.stderr


def test_cli_renderer_failure_preserves_live_html(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    bundle = tmp_path / "private" / "evidence.json"
    html = tmp_path / "explorer.html"
    html.write_text("known-good", encoding="utf-8")
    e5_fixture["profiles"].write_text("{invalid", encoding="utf-8")

    completed = run_cli(e5_fixture, bundle=bundle, html=html)

    assert completed.returncode == 2
    assert bundle.is_file()
    assert html.read_text(encoding="utf-8") == "known-good"
    assert not list(tmp_path.glob(".*.building.*.html"))


def test_cli_rejects_bundle_path_that_aliases_live_html(
    e5_fixture: E5Fixture, tmp_path: Path
) -> None:
    html = tmp_path / "explorer.html"
    html.write_text("known-good", encoding="utf-8")

    completed = run_cli(e5_fixture, bundle=html, html=html)

    assert completed.returncode == 2
    assert html.read_text(encoding="utf-8") == "known-good"
