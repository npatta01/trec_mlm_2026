from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from trec_mlm.pointwise_ranking_validation import (
    ValidationError,
    finalize_evaluator_payload,
    load_qrels_and_runs,
    phase0_query_metrics,
    validate_phase0_bundle,
    validate_publication_content,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts/pointwise_ranking_phase0"


def _write_matrix(tmp_path: Path) -> tuple[Path, Path]:
    qrels = tmp_path / "qrels.txt"
    qrels.write_text(
        "q1 0 A 2\n"
        "q1 0 B 1\n"
        "q1 0 C 0\n"
        "q2 0 A 0\n"
        "q2 0 B 2\n"
        "q2 0 C 1\n",
        encoding="utf-8",
    )
    run = tmp_path / "control.trec"
    run.write_text(
        "q1 Q0 B 1 3.0 fixture\n"
        "q1 Q0 C 2 2.0 fixture\n"
        "q1 Q0 A 3 1.0 fixture\n"
        "q2 Q0 B 1 3.0 fixture\n"
        "q2 Q0 C 2 2.0 fixture\n"
        "q2 Q0 A 3 1.0 fixture\n",
        encoding="utf-8",
    )
    return qrels, run


def test_phase0_query_metrics_use_raw_gain_and_positive_grade_mrr() -> None:
    ndcg, reciprocal_rank = phase0_query_metrics(
        ["B", "C", "A"], {"A": 2.0, "B": 1.0, "C": 0.0}
    )

    assert ndcg == pytest.approx(0.7601875334318685, abs=1e-15)
    assert reciprocal_rank == 1.0


def test_load_qrels_and_runs_requires_exact_matrix_and_file_order_ranks(
    tmp_path: Path,
) -> None:
    qrels, run = _write_matrix(tmp_path)

    loaded = load_qrels_and_runs(
        qrels,
        {"control": run},
        expected_query_count=2,
        expected_candidate_count=3,
    )

    assert loaded.query_ids == ("q1", "q2")
    assert loaded.candidate_ids == ("A", "B", "C")
    assert loaded.runs["control"]["q1"] == ("B", "C", "A")

    run.write_text(run.read_text(encoding="utf-8").replace("q1 Q0 C 2", "q1 Q0 C 3"), encoding="utf-8")
    with pytest.raises(ValidationError, match="ranks"):
        load_qrels_and_runs(
            qrels,
            {"control": run},
            expected_query_count=2,
            expected_candidate_count=3,
        )


def test_finalize_evaluator_payload_replaces_only_false_validation_field() -> None:
    evaluator = b'{\n  "artifact_type": "fixture",\n  "independent_validation": false,\n  "version": 1\n}\n'
    validation = {"assessment": "passed", "status": "independently_validated"}

    first = finalize_evaluator_payload(evaluator, validation)
    second = finalize_evaluator_payload(evaluator, validation)

    assert first == second
    assert first == (
        b'{\n  "artifact_type": "fixture",\n  "independent_validation": {\n'
        b'    "assessment": "passed",\n'
        b'    "status": "independently_validated"\n'
        b'  },\n  "version": 1\n}\n'
    )
    restored = json.loads(first)
    restored["independent_validation"] = False
    assert restored == json.loads(evaluator)


def test_finalize_evaluator_payload_fails_closed_if_already_validated() -> None:
    evaluator = b'{"independent_validation":{"assessment":"passed"}}'

    with pytest.raises(ValidationError, match="must be false"):
        finalize_evaluator_payload(evaluator, {"assessment": "passed"})


def test_publication_content_rejects_private_paths_and_raw_text_fields() -> None:
    with pytest.raises(ValidationError, match="private path"):
        validate_publication_content((b'{"source":"/home/person/private.json"}', b"a\tb\n"))

    with pytest.raises(ValidationError, match="forbidden field"):
        validate_publication_content((b'{"question_text":"private text"}', b"a\tb\n"))

    with pytest.raises(ValidationError, match="credential"):
        credential = b'{"note":"pass' + b'word=super' + b'secret"}'
        validate_publication_content((credential, b"a\tb\n"))


def test_module_import_does_not_load_evaluator_or_project_metrics() -> None:
    code = r'''
import importlib.abc
import sys

forbidden = {
    "trec_mlm.metrics",
    "trec_mlm.pointwise_ranking_evaluation",
}

class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname in forbidden:
            raise RuntimeError(f"forbidden import: {fullname}")
        return None

sys.meta_path.insert(0, Blocker())
import trec_mlm.pointwise_ranking_validation
print("independent-import-ok")
'''

    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "independent-import-ok"


def test_real_bundle_finalization_recreates_ledger_pinned_bytes() -> None:
    finalized = validate_phase0_bundle(
        qrels_path=ROOT / "data/dev/llm_dev_qrels.txt",
        artifact_dir=ARTIFACT_DIR,
        dense_results_path=ROOT / "reports/trec_2026_dense_query_knn_results.json",
        dense_per_query_path=ROOT / "reports/trec_2026_dense_query_knn_per_query.tsv",
        tracked_per_query_path=ROOT
        / "reports/trec_2026_pointwise_ranking_phase0_per_query.tsv",
        tracked_candidate_slices_path=ROOT
        / "reports/trec_2026_pointwise_ranking_phase0_candidate_slices.tsv",
    )

    assert finalized == (
        ROOT / "reports/trec_2026_pointwise_ranking_phase0_results.json"
    ).read_bytes()
    assert hashlib.sha256(finalized).hexdigest() == (
        "48ffaa81c77fde9dafb67ff9bbc29de98e51b8735a9320b2117f5662e31a7e6d"
    )


def test_cli_writes_exact_result_and_check_mode_passes(tmp_path: Path) -> None:
    output = tmp_path / "validated.json"
    write = subprocess.run(
        [
            sys.executable,
            "scripts/validate_pointwise_ranking_phase0.py",
            "--output",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert write.returncode == 0, write.stderr
    assert output.read_bytes() == (
        ROOT / "reports/trec_2026_pointwise_ranking_phase0_results.json"
    ).read_bytes()

    check = subprocess.run(
        [
            sys.executable,
            "scripts/validate_pointwise_ranking_phase0.py",
            "--check",
            str(output),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert check.returncode == 0, check.stderr
    assert "48ffaa81c77fde9dafb67ff9bbc29de98e51b8735a9320b2117f5662e31a7e6d" in check.stdout


def test_real_bundle_rejects_changed_tracked_tsv_copy(tmp_path: Path) -> None:
    changed_per_query = tmp_path / "per_query.tsv"
    changed_per_query.write_bytes(
        (ARTIFACT_DIR / "per_query.tsv").read_bytes() + b"unexpected\n"
    )

    with pytest.raises(ValidationError, match="tracked per-query receipt"):
        validate_phase0_bundle(
            qrels_path=ROOT / "data/dev/llm_dev_qrels.txt",
            artifact_dir=ARTIFACT_DIR,
            dense_results_path=ROOT
            / "reports/trec_2026_dense_query_knn_results.json",
            dense_per_query_path=ROOT
            / "reports/trec_2026_dense_query_knn_per_query.tsv",
            tracked_per_query_path=changed_per_query,
            tracked_candidate_slices_path=ROOT
            / "reports/trec_2026_pointwise_ranking_phase0_candidate_slices.tsv",
        )
