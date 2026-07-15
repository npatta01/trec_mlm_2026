from __future__ import annotations

import ast
import builtins
from collections.abc import Mapping
import csv
from dataclasses import FrozenInstanceError, dataclass
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trec_mlm.metrics import EvaluationResult
from trec_mlm.query_knn import DevelopmentData
from trec_mlm.runio import write_trec_run


REPO_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_QREL_PATH = REPO_ROOT / "data/dev/llm_dev_qrels.txt"
MODULE_PATH = REPO_ROOT / "trec_mlm/legacy_dense_control.py"
SCRIPT_PATH = REPO_ROOT / "scripts/export_fixed_mpnet_k50_control.py"
EXPECTED_NDCG = 0.36045807293576454
EXPECTED_MRR = 0.4039893654911617
LOCK_FILENAME = "fixed_mpnet_k50_control.lock.json"
TSV_FILENAME = "fixed_mpnet_k50_control.tsv"
TREC_FILENAME = "fixed_mpnet_k50_control.trec"
RUN_ID = "fixed_mpnet_k50_control"
LOCK_SCHEMA = "trec-mlm-dense-aggregation-legacy-control-lock-v1"
REPEAT_SCHEMA = "trec-mlm-dense-aggregation-legacy-control-repeat-v1"
MANIFEST_SCHEMA = "trec-mlm-dense-aggregation-manifest-v1"
EXPECTED_ROOT_FILES = (
    LOCK_FILENAME,
    TREC_FILENAME,
    TSV_FILENAME,
    "repeat_a.json",
    "repeat_b.json",
)
FEASIBILITY_FILES = (
    "feasibility_gate.json",
    "repeat_a_receipt.json",
    "repeat_b_receipt.json",
    "repeat_measurements.json",
)


def _lexical_absolute(path: object) -> Path | None:
    if isinstance(path, int):
        return None
    try:
        candidate = Path(os.fspath(path))
    except TypeError:
        return None
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    return Path(os.path.normpath(candidate))


@pytest.fixture(autouse=True)
def forbid_official_qrel_access(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn accidental official-qrel access into an immediate test failure."""

    forbidden = Path(os.path.normpath(OFFICIAL_QREL_PATH))

    def guard(path: object) -> None:
        if _lexical_absolute(path) == forbidden:
            raise AssertionError("Task 8 tests must not access the official qrel file")

    real_builtin_open = builtins.open
    real_io_open = io.open
    real_os_open = os.open

    def guarded_builtin_open(file: object, *args: object, **kwargs: object) -> object:
        guard(file)
        return real_builtin_open(file, *args, **kwargs)

    def guarded_io_open(file: object, *args: object, **kwargs: object) -> object:
        guard(file)
        return real_io_open(file, *args, **kwargs)

    def guarded_os_open(file: object, *args: object, **kwargs: object) -> int:
        guard(file)
        return real_os_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(io, "open", guarded_io_open)
    monkeypatch.setattr(os, "open", guarded_os_open)


def test_legacy_control_public_api_exists() -> None:
    module = importlib.import_module("trec_mlm.legacy_dense_control")
    fields = tuple(module.LegacyControlValidation.__dataclass_fields__)
    assert fields == (
        "tsv_sha256",
        "row_count",
        "mean_ndcg_at_10",
        "mean_reciprocal_rank",
    )
    rich_fields = (
        "manifest_sha256",
        "lock_sha256",
        "feasibility_manifest_sha256",
        "source_sha256",
        "query_file_sha256",
        "qrel_file_sha256",
        "query_input_sha256",
        "candidate_inventory_sha256",
        "embedding_sha256",
        "similarity_sha256",
        "tsv_sha256",
        "trec_sha256",
        "run_id",
        "neighbor_count",
        "query_count",
        "candidate_count",
        "row_count",
        "mean_ndcg_at_10",
        "mean_reciprocal_rank",
    )
    assert tuple(module.LegacyControlLockValidation.__dataclass_fields__) == rich_fields
    assert tuple(module.LegacyControlLockValidation.__slots__) == rich_fields
    assert module.LegacyControlLockValidation.__dataclass_params__.frozen is True
    assert callable(module.export_fixed_mpnet_k50_control)
    assert callable(module.validate_legacy_control_lock)
    assert callable(module.validate_legacy_control_lock_bindings)


def test_legacy_modules_have_isolated_imports() -> None:
    for path in (MODULE_PATH, SCRIPT_PATH):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert not any(
            name.startswith("trec_mlm.dense_aggregation") for name in imported
        )


def _raw_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _framed_sha256(tag: str, *payloads: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(b"trec-mlm-dense-aggregation-hash-v1\0")
    for payload in (tag.encode("ascii"), *payloads):
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_bytes(_canonical_json(value) + b"\n")


def _rewrite_manifest(root: Path) -> None:
    rows = [
        [name, (root / name).stat().st_size, _raw_sha256(root / name)]
        for name in EXPECTED_ROOT_FILES
    ]
    _write_canonical_json(
        root / "manifest.json",
        {
            "schema": MANIFEST_SCHEMA,
            "files": rows,
            "artifact_inventory_sha256": _framed_sha256(
                "artifact-inventory", _canonical_json(rows)
            ),
        },
    )


def _write_manifest(root: Path, files: tuple[str, ...]) -> None:
    rows = [
        [name, (root / name).stat().st_size, _raw_sha256(root / name)] for name in files
    ]
    _write_canonical_json(
        root / "manifest.json",
        {
            "schema": MANIFEST_SCHEMA,
            "files": rows,
            "artifact_inventory_sha256": _framed_sha256(
                "artifact-inventory", _canonical_json(rows)
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class SyntheticWorker:
    module: Any
    query_path: Path
    qrel_path: Path
    embedding_path: Path
    query_ids: tuple[str, ...]
    model_ids: tuple[str, ...]
    data: DevelopmentData
    grades: np.ndarray[Any, Any]
    embeddings: np.ndarray[Any, Any]
    similarities: np.ndarray[Any, Any]
    scorer_calls: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class SyntheticLock:
    root: Path
    lock_path: Path
    tsv_sha256: str
    worker: SyntheticWorker
    worker_result: Any


def _install_synthetic_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SyntheticWorker:
    module = importlib.import_module("trec_mlm.legacy_dense_control")
    query_path = tmp_path / "queries.tsv"
    qrel_path = tmp_path / "synthetic_qrels.txt"
    embedding_path = tmp_path / "embeddings.npy"
    query_path.write_text("synthetic query bytes\n", encoding="utf-8")
    qrel_path.write_text("synthetic qrel bytes\n", encoding="utf-8")

    query_ids = tuple(f"q{index:03d}" for index in range(322))
    model_ids = tuple(f"m{index:03d}" for index in range(150))
    queries = {query_id: f"text {query_id}" for query_id in reversed(query_ids)}
    qrels = {
        query_id: {
            model_id: float((query_index + model_index) % 3)
            for model_index, model_id in enumerate(reversed(model_ids))
        }
        for query_index, query_id in enumerate(reversed(query_ids))
    }
    data = DevelopmentData(
        queries=queries, qrels=qrels, models=tuple(reversed(model_ids))
    )
    grades = np.asarray(
        [
            [data.qrels[query_id][model_id] for model_id in model_ids]
            for query_id in query_ids
        ],
        dtype=np.float64,
    )

    embeddings = np.arange(322 * 768, dtype=np.float32).reshape(322, 768)
    np.save(embedding_path, embeddings, allow_pickle=False)
    similarities = np.zeros((322, 322), dtype=np.float32)
    np.fill_diagonal(similarities, np.float32(1.0))
    scorer_calls: list[dict[str, object]] = []

    def load_data(actual_query: Path, actual_qrel: Path) -> DevelopmentData:
        assert Path(actual_query) == query_path
        assert Path(actual_qrel) == qrel_path
        return data

    def cosine(actual_embeddings: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert actual_embeddings.flags.c_contiguous
        assert actual_embeddings.dtype == np.dtype("float32")
        assert np.array_equal(actual_embeddings, embeddings)
        return similarities

    def rank(**kwargs: object) -> SimpleNamespace:
        scorer_calls.append(kwargs)
        target_index = int(kwargs["target_index"])
        scores = target_index + np.arange(150, dtype=np.float64) / 1000.0
        priors = np.arange(150, dtype=np.float64) / 10.0
        return SimpleNamespace(
            model_order=tuple(reversed(range(150))),
            scores=scores,
            global_prior=priors,
        )

    monkeypatch.setattr(module, "load_development_data", load_data, raising=False)
    monkeypatch.setattr(module, "cosine_similarity_matrix", cosine, raising=False)
    monkeypatch.setattr(module, "rank_target_from_similarity", rank, raising=False)
    monkeypatch.setattr(
        module,
        "evaluate_rankings",
        lambda rankings, actual_qrels, k: EvaluationResult(
            cutoff=10,
            per_query={},
            macro_ndcg_at_k=EXPECTED_NDCG,
            mean_reciprocal_rank=EXPECTED_MRR,
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module, "_LOCKED_QUERY_FILE_SHA256", _raw_sha256(query_path), raising=False
    )
    monkeypatch.setattr(
        module, "_LOCKED_QREL_FILE_SHA256", _raw_sha256(qrel_path), raising=False
    )
    monkeypatch.setattr(
        module,
        "_LOCKED_QUERY_INPUT_SHA256",
        module._query_input_sha256(
            query_ids, tuple(data.queries[query_id] for query_id in query_ids)
        ),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_LOCKED_EMBEDDING_SHA256",
        module.array_sha256(embeddings),
        raising=False,
    )
    monkeypatch.setattr(
        module,
        "_LOCKED_SIMILARITY_SHA256",
        module.array_sha256(similarities),
        raising=False,
    )
    return SyntheticWorker(
        module=module,
        query_path=query_path,
        qrel_path=qrel_path,
        embedding_path=embedding_path,
        query_ids=query_ids,
        model_ids=model_ids,
        data=data,
        grades=grades,
        embeddings=embeddings,
        similarities=similarities,
        scorer_calls=scorer_calls,
    )


def _build_synthetic_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SyntheticLock:
    worker = _install_synthetic_worker(tmp_path, monkeypatch)
    root = tmp_path / "legacy-control"
    root.mkdir()
    tsv_path = root / TSV_FILENAME
    result = worker.module._export_fixed_mpnet_k50_control_attested(
        query_path=worker.query_path,
        qrel_path=worker.qrel_path,
        embedding_path=worker.embedding_path,
        output_tsv=tsv_path,
    )
    rankings = {
        query_id: list(reversed(worker.model_ids)) for query_id in worker.query_ids
    }
    write_trec_run(root / TREC_FILENAME, rankings, run_id=RUN_ID)
    feasibility_sha256 = "f" * 64
    common_repeat = {
        "schema": REPEAT_SCHEMA,
        "return_code": 0,
        "feasibility_manifest_sha256": feasibility_sha256,
        "source_sha256": result.source_sha256,
        "query_file_sha256": result.query_file_sha256,
        "qrel_file_sha256": result.qrel_file_sha256,
        "query_input_sha256": result.query_input_sha256,
        "candidate_inventory_sha256": result.candidate_inventory_sha256,
        "embedding_sha256": result.embedding_sha256,
        "similarity_sha256": result.similarity_sha256,
        "neighbor_count": 50,
        "tsv_sha256": result.validation.tsv_sha256,
        "row_count": result.validation.row_count,
        "query_count": result.query_count,
        "candidate_count": result.candidate_count,
        "mean_ndcg_at_10": result.validation.mean_ndcg_at_10,
        "mean_reciprocal_rank": result.validation.mean_reciprocal_rank,
    }
    for repeat in ("a", "b"):
        _write_canonical_json(
            root / f"repeat_{repeat}.json",
            {**common_repeat, "repeat": repeat},
        )
    lock = {
        "schema": LOCK_SCHEMA,
        "feasibility_manifest_sha256": feasibility_sha256,
        "source_sha256": result.source_sha256,
        "query_file_sha256": result.query_file_sha256,
        "qrel_file_sha256": result.qrel_file_sha256,
        "query_input_sha256": result.query_input_sha256,
        "candidate_inventory_sha256": result.candidate_inventory_sha256,
        "embedding_sha256": result.embedding_sha256,
        "similarity_sha256": result.similarity_sha256,
        "neighbor_count": 50,
        "tsv_filename": TSV_FILENAME,
        "tsv_sha256": result.validation.tsv_sha256,
        "trec_filename": TREC_FILENAME,
        "trec_sha256": _raw_sha256(root / TREC_FILENAME),
        "run_id": RUN_ID,
        "row_count": result.validation.row_count,
        "query_count": result.query_count,
        "candidate_count": result.candidate_count,
        "mean_ndcg_at_10": result.validation.mean_ndcg_at_10,
        "mean_reciprocal_rank": result.validation.mean_reciprocal_rank,
        "repeat_a_receipt_sha256": _raw_sha256(root / "repeat_a.json"),
        "repeat_b_receipt_sha256": _raw_sha256(root / "repeat_b.json"),
    }
    lock_path = root / LOCK_FILENAME
    _write_canonical_json(lock_path, lock)
    _rewrite_manifest(root)
    return SyntheticLock(
        root=root,
        lock_path=lock_path,
        tsv_sha256=result.validation.tsv_sha256,
        worker=worker,
        worker_result=result,
    )


def test_export_uses_locked_scorer_and_writes_lossless_canonical_tsv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)
    output_tsv = tmp_path / "legacy.tsv"

    result = fixture.module.export_fixed_mpnet_k50_control(
        query_path=fixture.query_path,
        qrel_path=fixture.qrel_path,
        embedding_path=fixture.embedding_path,
        output_tsv=output_tsv,
    )

    assert result == fixture.module.LegacyControlValidation(
        tsv_sha256=_raw_sha256(output_tsv),
        row_count=48_300,
        mean_ndcg_at_10=EXPECTED_NDCG,
        mean_reciprocal_rank=EXPECTED_MRR,
    )
    assert len(fixture.scorer_calls) == 322
    canonical_grades = fixture.scorer_calls[0]["grades"]
    canonical_grade_bytes = fixture.grades.tobytes(order="C")
    for target_index, call in enumerate(fixture.scorer_calls):
        assert call["target_index"] == target_index
        assert call["additionally_excluded_indices"] == ()
        assert call["similarities"] is fixture.similarities
        assert call["grades"] is canonical_grades
        assert call["grades"].tobytes(order="C") == canonical_grade_bytes
        assert call["query_ids"] == fixture.query_ids
        assert call["model_ids"] == fixture.model_ids
        assert call["k"] == 50

    with output_tsv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 48_300
    assert list(rows[0]) == [
        "query_id",
        "candidate_id",
        "raw_score_hex",
        "fold_prior_hex",
        "final_rank",
    ]
    assert [(row["query_id"], row["candidate_id"]) for row in rows] == [
        (query_id, model_id)
        for query_id in fixture.query_ids
        for model_id in fixture.model_ids
    ]
    assert all(
        math.isfinite(float.fromhex(row["raw_score_hex"]))
        and float.fromhex(row["raw_score_hex"]).hex() == row["raw_score_hex"]
        and math.isfinite(float.fromhex(row["fold_prior_hex"]))
        and float.fromhex(row["fold_prior_hex"]).hex() == row["fold_prior_hex"]
        for row in rows
    )
    assert [int(row["final_rank"]) for row in rows[:150]] == list(
        reversed(range(1, 151))
    )


@pytest.mark.parametrize(
    ("cutoff", "ndcg", "mrr", "message"),
    (
        (9, EXPECTED_NDCG, EXPECTED_MRR, "metric cutoff mismatch"),
        (
            10,
            math.nextafter(EXPECTED_NDCG, math.inf),
            EXPECTED_MRR,
            "nDCG regression mismatch",
        ),
        (10, math.nan, EXPECTED_MRR, "nDCG regression mismatch"),
        (
            10,
            EXPECTED_NDCG,
            math.nextafter(EXPECTED_MRR, math.inf),
            "MRR regression mismatch",
        ),
        (10, EXPECTED_NDCG, math.inf, "MRR regression mismatch"),
    ),
)
def test_export_requires_exact_preregistered_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cutoff: int,
    ndcg: float,
    mrr: float,
    message: str,
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(
        fixture.module,
        "evaluate_rankings",
        lambda rankings, qrels, k: EvaluationResult(
            cutoff=cutoff,
            per_query={},
            macro_ndcg_at_k=ndcg,
            mean_reciprocal_rank=mrr,
        ),
    )
    with pytest.raises(RuntimeError, match=message):
        fixture.module.export_fixed_mpnet_k50_control(
            query_path=fixture.query_path,
            qrel_path=fixture.qrel_path,
            embedding_path=fixture.embedding_path,
            output_tsv=tmp_path / "wrong-metric.tsv",
        )


def test_export_rejects_wrong_qrel_hash_before_loading_qrels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(fixture.module, "_LOCKED_QREL_FILE_SHA256", "0" * 64)
    monkeypatch.setattr(
        fixture.module,
        "load_development_data",
        lambda *_: pytest.fail("qrels loaded before raw qrel hash check"),
    )
    with pytest.raises(RuntimeError, match="qrel file hash mismatch"):
        fixture.module.export_fixed_mpnet_k50_control(
            query_path=fixture.query_path,
            qrel_path=fixture.qrel_path,
            embedding_path=fixture.embedding_path,
            output_tsv=tmp_path / "qrel-mismatch.tsv",
        )


@pytest.mark.parametrize(
    ("embeddings", "trailing_byte", "message"),
    (
        (np.zeros((321, 768), dtype=np.float32), False, "embedding array shape"),
        (np.zeros((322, 768), dtype=np.float64), False, "embedding array dtype"),
        (
            np.full((322, 768), np.nan, dtype=np.float32),
            False,
            "embedding array contains non-finite",
        ),
        (np.zeros((322, 768), dtype=np.float32), True, "embedding file size"),
    ),
)
def test_export_requires_exact_bounded_embedding_npy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    embeddings: np.ndarray[Any, Any],
    trailing_byte: bool,
    message: str,
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)
    np.save(fixture.embedding_path, embeddings, allow_pickle=False)
    if trailing_byte:
        with fixture.embedding_path.open("ab") as handle:
            handle.write(b"x")
    with pytest.raises(RuntimeError, match=message):
        fixture.module.export_fixed_mpnet_k50_control(
            query_path=fixture.query_path,
            qrel_path=fixture.qrel_path,
            embedding_path=fixture.embedding_path,
            output_tsv=tmp_path / "invalid-embedding.tsv",
        )


def test_export_rejects_oversized_embedding_before_raw_hash_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)
    with fixture.embedding_path.open("wb") as handle:
        handle.seek(4 * 1024 * 1024)
        handle.write(b"x")
    real_raw_hash = fixture.module._raw_file_sha256

    def guarded_raw_hash(path: Path) -> str:
        if Path(path) == fixture.embedding_path:
            pytest.fail("oversized embedding entered the raw hash loop")
        return real_raw_hash(path)

    monkeypatch.setattr(fixture.module, "_raw_file_sha256", guarded_raw_hash)
    with pytest.raises(RuntimeError, match="embedding file exceeds byte limit"):
        fixture.module.export_fixed_mpnet_k50_control(
            query_path=fixture.query_path,
            qrel_path=fixture.qrel_path,
            embedding_path=fixture.embedding_path,
            output_tsv=tmp_path / "oversized.tsv",
        )


def test_export_rejects_grade_outside_locked_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)
    fixture.data.qrels[fixture.query_ids[0]][fixture.model_ids[0]] = 3.0
    with pytest.raises(RuntimeError, match="grade matrix is invalid"):
        fixture.module.export_fixed_mpnet_k50_control(
            query_path=fixture.query_path,
            qrel_path=fixture.qrel_path,
            embedding_path=fixture.embedding_path,
            output_tsv=tmp_path / "invalid-grade.tsv",
        )


@pytest.mark.parametrize(
    ("kind", "message"),
    (
        ("duplicate", "invalid model permutation"),
        ("score_shape", "invalid score shape"),
        ("score_nan", "non-finite score"),
        ("prior_inf", "non-finite score"),
        ("wrong_order", "order differs"),
    ),
)
def test_export_rejects_corrupt_dense_target_ranking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    message: str,
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)

    def corrupt_rank(**kwargs: object) -> SimpleNamespace:
        scores = np.arange(150, dtype=np.float64)
        priors = np.arange(150, dtype=np.float64) / 10.0
        order: tuple[int, ...] = tuple(reversed(range(150)))
        if kind == "duplicate":
            order = (*order[:-1], order[-2])
        elif kind == "score_shape":
            scores = scores[:-1]
        elif kind == "score_nan":
            scores[0] = math.nan
        elif kind == "prior_inf":
            priors[0] = math.inf
        elif kind == "wrong_order":
            order = tuple(range(150))
        return SimpleNamespace(
            model_order=order,
            scores=scores,
            global_prior=priors,
        )

    monkeypatch.setattr(fixture.module, "rank_target_from_similarity", corrupt_rank)
    with pytest.raises(RuntimeError, match=message):
        fixture.module.export_fixed_mpnet_k50_control(
            query_path=fixture.query_path,
            qrel_path=fixture.qrel_path,
            embedding_path=fixture.embedding_path,
            output_tsv=tmp_path / "corrupt-ranking.tsv",
        )


@pytest.mark.parametrize("mutated", ("grades", "similarities"))
def test_export_detects_scorer_input_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated: str,
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)

    def mutating_rank(**kwargs: object) -> SimpleNamespace:
        array = kwargs[mutated]
        array.flat[0] += 1.0
        return SimpleNamespace(
            model_order=tuple(reversed(range(150))),
            scores=np.arange(150, dtype=np.float64),
            global_prior=np.arange(150, dtype=np.float64) / 10.0,
        )

    monkeypatch.setattr(fixture.module, "rank_target_from_similarity", mutating_rank)
    message = "grade matrix" if mutated == "grades" else "similarity matrix"
    with pytest.raises(RuntimeError, match=f"mutated the .*{message}"):
        fixture.module.export_fixed_mpnet_k50_control(
            query_path=fixture.query_path,
            qrel_path=fixture.qrel_path,
            embedding_path=fixture.embedding_path,
            output_tsv=tmp_path / "mutated-input.tsv",
        )


@pytest.mark.parametrize("drifted", ("query", "qrel", "embedding"))
def test_worker_detects_raw_input_drift_after_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    drifted: str,
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)
    drift_path = {
        "query": fixture.query_path,
        "qrel": fixture.qrel_path,
        "embedding": fixture.embedding_path,
    }[drifted]

    def drift_then_evaluate(*args: object, **kwargs: object) -> EvaluationResult:
        with drift_path.open("ab") as handle:
            handle.write(b"drift")
        return EvaluationResult(
            cutoff=10,
            per_query={},
            macro_ndcg_at_k=EXPECTED_NDCG,
            mean_reciprocal_rank=EXPECTED_MRR,
        )

    monkeypatch.setattr(fixture.module, "evaluate_rankings", drift_then_evaluate)
    with pytest.raises(RuntimeError, match=f"{drifted} file drifted"):
        fixture.module.export_fixed_mpnet_k50_control(
            query_path=fixture.query_path,
            qrel_path=fixture.qrel_path,
            embedding_path=fixture.embedding_path,
            output_tsv=tmp_path / "drift.tsv",
        )


def test_worker_detects_source_drift_after_export(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)
    actual_source_sha256 = fixture.module._source_sha256
    calls = 0

    def drifting_source_sha256() -> dict[str, str]:
        nonlocal calls
        calls += 1
        result = actual_source_sha256()
        if calls == 2:
            result = {**result, "trec_mlm/metrics.py": "0" * 64}
        return result

    monkeypatch.setattr(fixture.module, "_source_sha256", drifting_source_sha256)
    with pytest.raises(RuntimeError, match="source files drifted"):
        fixture.module.export_fixed_mpnet_k50_control(
            query_path=fixture.query_path,
            qrel_path=fixture.qrel_path,
            embedding_path=fixture.embedding_path,
            output_tsv=tmp_path / "source-drift.tsv",
        )


def test_worker_result_attests_consumed_hashes_and_candidate_framing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)
    output_tsv = tmp_path / "attested.tsv"
    result = fixture.module._export_fixed_mpnet_k50_control_attested(
        query_path=fixture.query_path,
        qrel_path=fixture.qrel_path,
        embedding_path=fixture.embedding_path,
        output_tsv=output_tsv,
    )
    digest = hashlib.sha256()
    digest.update(b"trec-mlm-dense-aggregation-hash-v1\0")
    for payload in (
        b"candidate-inventory",
        json.dumps(
            list(fixture.model_ids),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8"),
    ):
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    assert result.candidate_inventory_sha256 == digest.hexdigest()
    assert result.query_file_sha256 == _raw_sha256(fixture.query_path)
    assert result.qrel_file_sha256 == _raw_sha256(fixture.qrel_path)
    assert result.embedding_file_sha256 == _raw_sha256(fixture.embedding_path)
    assert result.embedding_sha256 == fixture.module.array_sha256(fixture.embeddings)
    assert result.similarity_sha256 == fixture.module.array_sha256(fixture.similarities)
    assert tuple(result.source_sha256) == fixture.module._SOURCE_RELATIVE_PATHS
    assert (result.neighbor_count, result.query_count, result.candidate_count) == (
        50,
        322,
        150,
    )


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    assert type(value) is dict
    return value


def test_validate_legacy_control_lock_reloads_complete_synthetic_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    monkeypatch.setattr(
        fixture.worker.module,
        "load_development_data",
        lambda *_: pytest.fail("lock validation reopened qrels"),
    )
    result = fixture.worker.module.validate_legacy_control_lock(
        fixture.lock_path, fixture.tsv_sha256
    )
    assert result == fixture.worker.module.LegacyControlValidation(
        tsv_sha256=fixture.tsv_sha256,
        row_count=48_300,
        mean_ndcg_at_10=EXPECTED_NDCG,
        mean_reciprocal_rank=EXPECTED_MRR,
    )


def test_validate_legacy_control_lock_rejects_wrong_filename_before_path_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = importlib.import_module("trec_mlm.legacy_dense_control")

    monkeypatch.setattr(
        module,
        "_read_bounded_regular_file",
        lambda *_args, **_kwargs: pytest.fail(
            "compatibility wrapper accessed an invalid lock path"
        ),
    )

    with pytest.raises(ValueError, match="lock filename"):
        module.validate_legacy_control_lock(OFFICIAL_QREL_PATH, "0" * 64)


def test_validate_legacy_control_lock_rejects_bad_tsv_hash_before_path_access(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("trec_mlm.legacy_dense_control")

    monkeypatch.setattr(
        module,
        "_read_bounded_regular_file",
        lambda *_args, **_kwargs: pytest.fail(
            "compatibility wrapper accessed a lock before validating arguments"
        ),
    )

    with pytest.raises(ValueError, match="expected TSV hash"):
        module.validate_legacy_control_lock(tmp_path / LOCK_FILENAME, "not-a-hash")


def test_validate_legacy_control_lock_bindings_returns_exact_frozen_attestations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    monkeypatch.setattr(
        fixture.worker.module,
        "load_development_data",
        lambda *_: pytest.fail("rich lock validation reopened qrels"),
    )
    lock = _read_json(fixture.lock_path)
    expected_lock_sha256 = _raw_sha256(fixture.lock_path)

    result = fixture.worker.module.validate_legacy_control_lock_bindings(
        fixture.lock_path,
        expected_lock_sha256=expected_lock_sha256,
        expected_tsv_sha256=fixture.tsv_sha256,
    )

    assert result == fixture.worker.module.LegacyControlLockValidation(
        manifest_sha256=_raw_sha256(fixture.root / "manifest.json"),
        lock_sha256=expected_lock_sha256,
        feasibility_manifest_sha256=lock["feasibility_manifest_sha256"],
        source_sha256=fixture.worker_result.source_sha256,
        query_file_sha256=fixture.worker_result.query_file_sha256,
        qrel_file_sha256=fixture.worker_result.qrel_file_sha256,
        query_input_sha256=fixture.worker_result.query_input_sha256,
        candidate_inventory_sha256=fixture.worker_result.candidate_inventory_sha256,
        embedding_sha256=fixture.worker_result.embedding_sha256,
        similarity_sha256=fixture.worker_result.similarity_sha256,
        tsv_sha256=fixture.tsv_sha256,
        trec_sha256=_raw_sha256(fixture.root / TREC_FILENAME),
        run_id=RUN_ID,
        neighbor_count=50,
        query_count=322,
        candidate_count=150,
        row_count=48_300,
        mean_ndcg_at_10=EXPECTED_NDCG,
        mean_reciprocal_rank=EXPECTED_MRR,
    )
    assert isinstance(result.source_sha256, Mapping)
    assert type(result.source_sha256) is not dict
    for field in (
        "manifest_sha256",
        "lock_sha256",
        "feasibility_manifest_sha256",
        "query_file_sha256",
        "qrel_file_sha256",
        "query_input_sha256",
        "candidate_inventory_sha256",
        "embedding_sha256",
        "similarity_sha256",
        "tsv_sha256",
        "trec_sha256",
        "run_id",
    ):
        assert type(getattr(result, field)) is str
    for field in ("neighbor_count", "query_count", "candidate_count", "row_count"):
        assert type(getattr(result, field)) is int
    for field in ("mean_ndcg_at_10", "mean_reciprocal_rank"):
        assert type(getattr(result, field)) is float
    with pytest.raises(FrozenInstanceError):
        result.lock_sha256 = "0" * 64
    with pytest.raises(TypeError):
        result.source_sha256["trec_mlm/metrics.py"] = "0" * 64


def test_validate_legacy_control_lock_bindings_does_not_reopen_tsv_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    expected_lock_sha256 = _raw_sha256(fixture.lock_path)
    real_path_open = Path.open

    def reject_tsv_reopen(path: Path, *args: object, **kwargs: object) -> object:
        if path == fixture.root / TSV_FILENAME:
            pytest.fail("rich lock validation reopened the bounded TSV path")
        return real_path_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", reject_tsv_reopen)

    result = fixture.worker.module.validate_legacy_control_lock_bindings(
        fixture.lock_path,
        expected_lock_sha256=expected_lock_sha256,
        expected_tsv_sha256=fixture.tsv_sha256,
    )

    assert result.tsv_sha256 == fixture.tsv_sha256


def test_validate_legacy_control_lock_bindings_rejects_wrong_lock_hash_with_valid_tsv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    wrong_lock_sha256 = (
        "0" if _raw_sha256(fixture.lock_path)[0] != "0" else "1"
    ) + _raw_sha256(fixture.lock_path)[1:]

    with pytest.raises(ValueError, match="expected lock hash"):
        fixture.worker.module.validate_legacy_control_lock_bindings(
            fixture.lock_path,
            expected_lock_sha256=wrong_lock_sha256,
            expected_tsv_sha256=fixture.tsv_sha256,
        )


def test_bounded_reader_rejects_same_size_same_mtime_inode_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("trec_mlm.legacy_dense_control")
    target = tmp_path / "target.bin"
    replacement = tmp_path / "replacement.bin"
    target.write_bytes(b"original")
    replacement.write_bytes(b"replaced")
    original_status = target.lstat()
    os.utime(
        replacement,
        ns=(original_status.st_atime_ns, original_status.st_mtime_ns),
    )
    real_lstat = Path.lstat
    swapped = False

    def swap_after_first_lstat(path: Path) -> os.stat_result:
        nonlocal swapped
        status = real_lstat(path)
        if path == target and not swapped:
            swapped = True
            os.replace(replacement, target)
        return status

    monkeypatch.setattr(Path, "lstat", swap_after_first_lstat)

    with pytest.raises(ValueError, match="changed while it was read"):
        module._read_bounded_regular_file(
            target,
            byte_limit=32,
            label="synthetic bounded input",
        )


def test_bounded_reader_opens_nofollow_nonblocking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("trec_mlm.legacy_dense_control")
    target = tmp_path / "target.bin"
    target.write_bytes(b"bounded")
    real_open = module.os.open
    observed_flags: list[int] = []

    def recording_open(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        *args: object,
        **kwargs: object,
    ) -> int:
        observed_flags.append(flags)
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", recording_open)

    assert (
        module._read_bounded_regular_file(
            target,
            byte_limit=32,
            label="synthetic bounded input",
        )
        == b"bounded"
    )
    assert len(observed_flags) == 1
    assert observed_flags[0] & os.O_NONBLOCK
    assert observed_flags[0] & os.O_NOFOLLOW


def test_validate_lock_bindings_never_uses_unbounded_hash_for_root_members(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    real_raw_sha256 = fixture.worker.module._raw_file_sha256

    def guarded_raw_sha256(path: str | Path) -> str:
        candidate = Path(path)
        if candidate.parent == fixture.root:
            pytest.fail("legacy root member entered the unbounded raw hash loop")
        return real_raw_sha256(candidate)

    monkeypatch.setattr(
        fixture.worker.module,
        "_raw_file_sha256",
        guarded_raw_sha256,
    )

    result = fixture.worker.module.validate_legacy_control_lock_bindings(
        fixture.lock_path,
        expected_lock_sha256=_raw_sha256(fixture.lock_path),
        expected_tsv_sha256=fixture.tsv_sha256,
    )

    assert result.row_count == 48_300


@pytest.mark.parametrize(
    "filename",
    ("repeat_a.json", TSV_FILENAME, TREC_FILENAME),
)
def test_validate_lock_bindings_rejects_post_parse_root_member_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    expected_lock_sha256 = _raw_sha256(fixture.lock_path)
    real_trec_bytes = fixture.worker.module._trec_bytes_from_rankings

    def drift_after_parsing(rankings: dict[str, list[str]]) -> bytes:
        payload = real_trec_bytes(rankings)
        with (fixture.root / filename).open("ab") as handle:
            handle.write(b"drift")
        return payload

    monkeypatch.setattr(
        fixture.worker.module,
        "_trec_bytes_from_rankings",
        drift_after_parsing,
    )

    with pytest.raises(ValueError, match="manifest inventory does not match"):
        fixture.worker.module.validate_legacy_control_lock_bindings(
            fixture.lock_path,
            expected_lock_sha256=expected_lock_sha256,
            expected_tsv_sha256=fixture.tsv_sha256,
        )


@pytest.mark.parametrize(
    ("filename", "message"),
    (
        ("manifest.json", "manifest changed during validation"),
        (LOCK_FILENAME, "lock changed during validation"),
    ),
)
def test_validate_lock_bindings_rejects_drift_after_final_root_reload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    filename: str,
    message: str,
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    expected_lock_sha256 = _raw_sha256(fixture.lock_path)
    real_validate_manifest = fixture.worker.module._validate_local_manifest
    calls = 0

    def drift_after_final_reload(root: Path) -> dict[str, object]:
        nonlocal calls
        result = real_validate_manifest(root)
        calls += 1
        if calls == 2:
            with (fixture.root / filename).open("ab") as handle:
                handle.write(b"drift")
        return result

    monkeypatch.setattr(
        fixture.worker.module,
        "_validate_local_manifest",
        drift_after_final_reload,
    )

    with pytest.raises(ValueError, match=message):
        fixture.worker.module.validate_legacy_control_lock_bindings(
            fixture.lock_path,
            expected_lock_sha256=expected_lock_sha256,
            expected_tsv_sha256=fixture.tsv_sha256,
        )


def test_validate_lock_requires_exact_filename_expected_hash_and_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="lock filename"):
        fixture.worker.module.validate_legacy_control_lock(
            fixture.root / "repeat_a.json", fixture.tsv_sha256
        )
    with pytest.raises(ValueError, match="expected TSV hash"):
        fixture.worker.module.validate_legacy_control_lock(fixture.lock_path, "0" * 64)
    (fixture.root / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="physical layout"):
        fixture.worker.module.validate_legacy_control_lock(
            fixture.lock_path, fixture.tsv_sha256
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("feasibility_manifest_sha256", "0" * 64, "feasibility"),
        ("qrel_file_sha256", "0" * 64, "locked input"),
        ("query_input_sha256", "0" * 64, "locked input"),
        ("row_count", 48_299, "row count"),
        ("tsv_sha256", "0" * 64, "TSV hash"),
        ("trec_sha256", "0" * 64, "TREC hash"),
    ),
)
def test_validate_lock_rejects_wrong_cross_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
    message: str,
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    lock = _read_json(fixture.lock_path)
    lock[field] = replacement
    _write_canonical_json(fixture.lock_path, lock)
    _rewrite_manifest(fixture.root)
    with pytest.raises(ValueError, match=message):
        fixture.worker.module.validate_legacy_control_lock(
            fixture.lock_path, fixture.tsv_sha256
        )


def test_validate_lock_rejects_current_source_or_repeat_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    lock = _read_json(fixture.lock_path)
    source = dict(lock["source_sha256"])
    source["trec_mlm/metrics.py"] = "0" * 64
    lock["source_sha256"] = source
    _write_canonical_json(fixture.lock_path, lock)
    _rewrite_manifest(fixture.root)
    with pytest.raises(ValueError, match="source"):
        fixture.worker.module.validate_legacy_control_lock(
            fixture.lock_path, fixture.tsv_sha256
        )

    second = tmp_path / "second"
    second.mkdir()
    fixture = _build_synthetic_lock(second, monkeypatch)
    repeat_b_path = fixture.root / "repeat_b.json"
    repeat_b = _read_json(repeat_b_path)
    repeat_b["qrel_file_sha256"] = "0" * 64
    _write_canonical_json(repeat_b_path, repeat_b)
    lock = _read_json(fixture.lock_path)
    lock["repeat_b_receipt_sha256"] = _raw_sha256(repeat_b_path)
    _write_canonical_json(fixture.lock_path, lock)
    _rewrite_manifest(fixture.root)
    with pytest.raises(ValueError, match="repeat receipts"):
        fixture.worker.module.validate_legacy_control_lock(
            fixture.lock_path, fixture.tsv_sha256
        )


def _rebind_tsv(fixture: SyntheticLock, payload: bytes) -> str:
    tsv_path = fixture.root / TSV_FILENAME
    tsv_path.write_bytes(payload)
    tsv_sha256 = _raw_sha256(tsv_path)
    for repeat in ("a", "b"):
        repeat_path = fixture.root / f"repeat_{repeat}.json"
        receipt = _read_json(repeat_path)
        receipt["tsv_sha256"] = tsv_sha256
        _write_canonical_json(repeat_path, receipt)
    lock = _read_json(fixture.lock_path)
    lock["tsv_sha256"] = tsv_sha256
    lock["repeat_a_receipt_sha256"] = _raw_sha256(fixture.root / "repeat_a.json")
    lock["repeat_b_receipt_sha256"] = _raw_sha256(fixture.root / "repeat_b.json")
    _write_canonical_json(fixture.lock_path, lock)
    _rewrite_manifest(fixture.root)
    return tsv_sha256


def _rebind_common_field(
    fixture: SyntheticLock, field: str, replacement: object
) -> None:
    for repeat in ("a", "b"):
        repeat_path = fixture.root / f"repeat_{repeat}.json"
        receipt = _read_json(repeat_path)
        receipt[field] = replacement
        _write_canonical_json(repeat_path, receipt)
    lock = _read_json(fixture.lock_path)
    lock[field] = replacement
    lock["repeat_a_receipt_sha256"] = _raw_sha256(fixture.root / "repeat_a.json")
    lock["repeat_b_receipt_sha256"] = _raw_sha256(fixture.root / "repeat_b.json")
    _write_canonical_json(fixture.lock_path, lock)
    _rewrite_manifest(fixture.root)


@pytest.mark.parametrize(
    "field",
    (
        "query_file_sha256",
        "qrel_file_sha256",
        "query_input_sha256",
        "embedding_sha256",
        "similarity_sha256",
    ),
)
def test_validate_lock_requires_fixed_raw_and_semantic_input_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    _rebind_common_field(fixture, field, "0" * 64)
    with pytest.raises(ValueError, match="locked input"):
        fixture.worker.module.validate_legacy_control_lock(
            fixture.lock_path, fixture.tsv_sha256
        )


@pytest.mark.parametrize(
    ("target", "mutation"),
    (
        ("lock", "missing"),
        ("lock", "extra"),
        ("repeat", "missing"),
        ("repeat", "extra"),
    ),
)
def test_validate_lock_rejects_missing_or_unknown_schema_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    mutation: str,
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    path = fixture.lock_path if target == "lock" else fixture.root / "repeat_a.json"
    payload = _read_json(path)
    if mutation == "missing":
        payload.pop("query_input_sha256")
    else:
        payload["unknown"] = "forbidden"
    _write_canonical_json(path, payload)
    if target == "repeat":
        lock = _read_json(fixture.lock_path)
        lock["repeat_a_receipt_sha256"] = _raw_sha256(path)
        _write_canonical_json(fixture.lock_path, lock)
    _rewrite_manifest(fixture.root)
    with pytest.raises(ValueError, match="invalid contract"):
        fixture.worker.module.validate_legacy_control_lock(
            fixture.lock_path, fixture.tsv_sha256
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("score", "rank differs"),
        ("rank", "ranks are not a full permutation"),
        ("order", "candidate order"),
        ("duplicate", "candidate order"),
        ("nonfinite", "non-finite"),
        ("noncanonical_score", "noncanonical"),
        ("noncanonical_prior", "noncanonical"),
        ("query_order", "query order"),
        ("missing", "ended before"),
    ),
)
def test_validate_lock_rejects_tsv_scientific_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    lines = (fixture.root / TSV_FILENAME).read_bytes().splitlines(keepends=True)
    if mutation == "score":
        fields = lines[1].decode().rstrip("\n").split("\t")
        fields[2] = float(1000.0).hex()
        lines[1] = ("\t".join(fields) + "\n").encode()
    elif mutation == "rank":
        fields = lines[1].decode().rstrip("\n").split("\t")
        fields[4] = "149"
        lines[1] = ("\t".join(fields) + "\n").encode()
    elif mutation == "order":
        lines[1], lines[2] = lines[2], lines[1]
    elif mutation == "duplicate":
        fields = lines[2].decode().rstrip("\n").split("\t")
        fields[1] = "m000"
        lines[2] = ("\t".join(fields) + "\n").encode()
    elif mutation == "nonfinite":
        fields = lines[1].decode().rstrip("\n").split("\t")
        fields[2] = "nan"
        lines[1] = ("\t".join(fields) + "\n").encode()
    elif mutation == "noncanonical_score":
        fields = lines[1].decode().rstrip("\n").split("\t")
        fields[2] = "0X0.0P+0"
        lines[1] = ("\t".join(fields) + "\n").encode()
    elif mutation == "noncanonical_prior":
        fields = lines[1].decode().rstrip("\n").split("\t")
        fields[3] = "0X0.0P+0"
        lines[1] = ("\t".join(fields) + "\n").encode()
    elif mutation == "query_order":
        first = lines[1:151]
        second = lines[151:301]
        lines[1:301] = [*second, *first]
    elif mutation == "missing":
        lines.pop()
    tsv_sha256 = _rebind_tsv(fixture, b"".join(lines))
    with pytest.raises(ValueError, match=message):
        fixture.worker.module.validate_legacy_control_lock(
            fixture.lock_path, tsv_sha256
        )


def test_validate_lock_reconstructs_trec_bytes_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    trec_path = fixture.root / TREC_FILENAME
    rankings = {
        query_id: list(reversed(fixture.worker.model_ids))
        for query_id in fixture.worker.query_ids
    }
    assert fixture.worker.module._trec_bytes_from_rankings(rankings) == (
        trec_path.read_bytes()
    )
    lines = trec_path.read_text(encoding="utf-8").splitlines(keepends=True)
    fields = lines[0].split()
    fields[4] = "149.5"
    lines[0] = " ".join(fields) + "\n"
    trec_path.write_text("".join(lines), encoding="utf-8")
    lock = _read_json(fixture.lock_path)
    lock["trec_sha256"] = _raw_sha256(trec_path)
    _write_canonical_json(fixture.lock_path, lock)
    _rewrite_manifest(fixture.root)
    with pytest.raises(ValueError, match="TREC bytes are not canonical"):
        fixture.worker.module.validate_legacy_control_lock(
            fixture.lock_path, fixture.tsv_sha256
        )


def test_local_serialization_matches_shared_contract_without_importing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)
    from trec_mlm.dense_aggregation_hashing import canonical_json, framed_sha256

    sample = {"snow": "雪", "values": [1, 2.5, None]}
    assert fixture.module._canonical_json(sample) == canonical_json(sample)
    assert fixture.module._framed_sha256(
        "candidate-inventory", fixture.module._canonical_json(sample)
    ) == framed_sha256("candidate-inventory", canonical_json(sample))


def test_fresh_import_has_no_dense_aggregation_module() -> None:
    code = """
import json, sys
import trec_mlm.legacy_dense_control
print(json.dumps(sorted(
    name for name in sys.modules if name.startswith('trec_mlm.dense_aggregation')
)))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def _load_exporter_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        f"task8_exporter_{os.urandom(8).hex()}", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_feasibility_root(root: Path, *, passing: bool = True) -> Path:
    root.mkdir()
    _write_canonical_json(
        root / "feasibility_gate.json",
        {
            "schema": "trec-mlm-dense-aggregation-feasibility-gate-v1",
            "status": "pass" if passing else "stop",
            "failures": [] if passing else ["synthetic stop"],
            "projection_seconds": 1.0,
            "maximum_rss_bytes": 1,
        },
    )
    for name in FEASIBILITY_FILES[1:]:
        _write_canonical_json(root / name, {"schema": f"synthetic-{name}-v1"})
    _write_manifest(root, FEASIBILITY_FILES)
    return root / "manifest.json"


def _worker_result_payload(fixture: SyntheticLock) -> dict[str, object]:
    result = fixture.worker_result
    return {
        "schema": "trec-mlm-dense-aggregation-legacy-control-worker-v1",
        "source_sha256": result.source_sha256,
        "query_file_sha256": result.query_file_sha256,
        "qrel_file_sha256": result.qrel_file_sha256,
        "query_input_sha256": result.query_input_sha256,
        "candidate_inventory_sha256": result.candidate_inventory_sha256,
        "embedding_file_sha256": result.embedding_file_sha256,
        "embedding_sha256": result.embedding_sha256,
        "similarity_sha256": result.similarity_sha256,
        "neighbor_count": result.neighbor_count,
        "tsv_sha256": result.validation.tsv_sha256,
        "row_count": result.validation.row_count,
        "query_count": result.query_count,
        "candidate_count": result.candidate_count,
        "mean_ndcg_at_10": result.validation.mean_ndcg_at_10,
        "mean_reciprocal_rank": result.validation.mean_reciprocal_rank,
    }


def _install_fresh_fake_workers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exporter: Any,
    fixture: SyntheticLock,
    *,
    mode: str = "success",
) -> Path:
    fake_worker = tmp_path / "fake_worker.py"
    fake_worker.write_text(
        """
import argparse
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--repeat", required=True)
parser.add_argument("--output-tsv", required=True)
parser.add_argument("--result-json", required=True)
parser.add_argument("--fixture-tsv", required=True)
parser.add_argument("--fixture-result", required=True)
parser.add_argument("--log", required=True)
parser.add_argument("--mode", required=True)
args = parser.parse_args()
with Path(args.log).open("a", encoding="ascii") as handle:
    handle.write(args.repeat + "\\n")
    handle.flush()
    os.fsync(handle.fileno())
if args.mode == "fail-a" and args.repeat == "a":
    raise SystemExit(7)
if args.mode == "attempt-official" and args.repeat == "a":
    Path(os.environ["TASK8_FORBIDDEN_QREL"]).open("rb")
payload = Path(args.fixture_tsv).read_bytes()
if args.mode == "mismatch" and args.repeat == "b":
    payload += b"mismatch"
with Path(args.output_tsv).open("xb") as handle:
    handle.write(payload)
    handle.flush()
    os.fsync(handle.fileno())
result = json.loads(Path(args.fixture_result).read_bytes())
result["tsv_sha256"] = hashlib.sha256(payload).hexdigest()
encoded = json.dumps(
    result, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
).encode("utf-8") + b"\\n"
if args.mode == "oversized" and args.repeat == "a":
    encoded = b"x" * (256 * 1024 + 1)
with Path(args.result_json).open("xb") as handle:
    handle.write(encoded)
    handle.flush()
    os.fsync(handle.fileno())
""".lstrip(),
        encoding="utf-8",
    )
    fixture_result = tmp_path / "fake_worker_result.json"
    _write_canonical_json(fixture_result, _worker_result_payload(fixture))
    log = tmp_path / "worker.log"

    def worker_command(**kwargs: object) -> list[str]:
        return [
            sys.executable,
            str(fake_worker),
            "--repeat",
            str(kwargs["repeat"]),
            "--output-tsv",
            str(kwargs["output_tsv"]),
            "--result-json",
            str(kwargs["result_json"]),
            "--fixture-tsv",
            str(fixture.root / TSV_FILENAME),
            "--fixture-result",
            str(fixture_result),
            "--log",
            str(log),
            "--mode",
            mode,
        ]

    monkeypatch.setattr(exporter, "_worker_command", worker_command, raising=False)
    return log


def _run_synthetic_coordinator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mode: str = "success",
) -> tuple[Any, SyntheticLock, Path, Path, Path]:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    log = _install_fresh_fake_workers(
        tmp_path, monkeypatch, exporter, fixture, mode=mode
    )
    output_root = tmp_path / "authoritative"
    compact = tmp_path / "compact.json"
    exporter._coordinate(
        query_path=fixture.worker.query_path,
        qrel_path=fixture.worker.qrel_path,
        embedding_path=fixture.worker.embedding_path,
        feasibility_receipt=feasibility_receipt,
        output_root=output_root,
        compact_receipt=compact,
    )
    return exporter, fixture, log, output_root, compact


def test_coordinator_runs_two_sequential_fresh_workers_and_publishes_exact_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter, fixture, log, output_root, compact = _run_synthetic_coordinator(
        tmp_path, monkeypatch
    )
    assert log.read_text(encoding="ascii").splitlines() == ["a", "b"]
    assert sorted(path.name for path in output_root.iterdir()) == sorted(
        ("manifest.json", *EXPECTED_ROOT_FILES)
    )
    assert (
        fixture.worker.module.validate_legacy_control_lock(
            output_root / LOCK_FILENAME, fixture.tsv_sha256
        ).row_count
        == 48_300
    )
    compact_value = _read_json(compact)
    assert set(compact_value) == {
        "schema",
        "feasibility_manifest_sha256",
        "legacy_control_manifest_sha256",
        "legacy_control_lock_sha256",
        "legacy_control_tsv_sha256",
        "legacy_control_trec_sha256",
        "row_count",
        "query_count",
        "candidate_count",
        "mean_ndcg_at_10",
        "mean_reciprocal_rank",
    }
    assert compact_value["schema"] == (
        "trec-mlm-dense-aggregation-legacy-control-compact-v1"
    )
    assert compact_value["legacy_control_lock_sha256"] == _raw_sha256(
        output_root / LOCK_FILENAME
    )
    assert "embedding_file_sha256" not in _read_json(output_root / "repeat_a.json")
    assert exporter is not None


@pytest.mark.parametrize(
    ("mode", "expected_log", "message"),
    (
        ("fail-a", ["a"], "repeat a failed"),
        ("mismatch", ["a", "b"], "TSV bytes differ"),
    ),
)
def test_coordinator_fails_closed_on_worker_exit_or_byte_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_log: list[str],
    message: str,
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    log = _install_fresh_fake_workers(
        tmp_path, monkeypatch, exporter, fixture, mode=mode
    )
    output_root = tmp_path / "authoritative"
    compact = tmp_path / "compact.json"
    with pytest.raises(RuntimeError, match=message):
        exporter._coordinate(
            query_path=fixture.worker.query_path,
            qrel_path=fixture.worker.qrel_path,
            embedding_path=fixture.worker.embedding_path,
            feasibility_receipt=feasibility_receipt,
            output_root=output_root,
            compact_receipt=compact,
        )
    assert log.read_text(encoding="ascii").splitlines() == expected_log
    assert not output_root.exists()
    assert not compact.exists()


def test_coordinator_validates_passing_feasibility_before_worker_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(
        tmp_path / "feasibility", passing=False
    )
    exporter = _load_exporter_module()
    monkeypatch.setattr(
        exporter,
        "_worker_command",
        lambda **kwargs: pytest.fail("worker command built before pass gate"),
        raising=False,
    )
    with pytest.raises(ValueError, match="feasibility gate did not pass"):
        exporter._coordinate(
            query_path=fixture.worker.query_path,
            qrel_path=fixture.worker.qrel_path,
            embedding_path=fixture.worker.embedding_path,
            feasibility_receipt=feasibility_receipt,
            output_root=tmp_path / "authoritative",
            compact_receipt=tmp_path / "compact.json",
        )


def test_coordinator_protects_entire_feasibility_root_from_output_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    monkeypatch.setattr(
        exporter,
        "_worker_command",
        lambda **kwargs: pytest.fail("worker built before namespace rejection"),
        raising=False,
    )
    with pytest.raises(ValueError, match="namespaces overlap"):
        exporter._coordinate(
            query_path=fixture.worker.query_path,
            qrel_path=fixture.worker.qrel_path,
            embedding_path=fixture.worker.embedding_path,
            feasibility_receipt=feasibility_receipt,
            output_root=feasibility_receipt.parent / "nested-output",
            compact_receipt=tmp_path / "compact.json",
        )


def test_fresh_worker_uses_shell_false_and_no_unbounded_pipes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exporter = _load_exporter_module()
    observed: dict[str, object] = {}

    def fake_run(
        command: object, **kwargs: object
    ) -> subprocess.CompletedProcess[bytes]:
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(exporter.subprocess, "run", fake_run)
    exporter._run_fresh_worker(
        repeat="a",
        query_path=tmp_path / "query",
        qrel_path=tmp_path / "qrel",
        embedding_path=tmp_path / "embedding",
        output_tsv=tmp_path / "output",
        result_json=tmp_path / "result",
    )
    assert observed["shell"] is False
    assert observed["stdout"] is subprocess.DEVNULL
    assert observed["stderr"] is subprocess.DEVNULL


def test_coordinator_rejects_oversized_worker_result_before_repeat_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    log = _install_fresh_fake_workers(
        tmp_path, monkeypatch, exporter, fixture, mode="oversized"
    )
    output_root = tmp_path / "authoritative"
    compact = tmp_path / "compact.json"
    with pytest.raises(ValueError, match="exceeds its byte limit"):
        exporter._coordinate(
            query_path=fixture.worker.query_path,
            qrel_path=fixture.worker.qrel_path,
            embedding_path=fixture.worker.embedding_path,
            feasibility_receipt=feasibility_receipt,
            output_root=output_root,
            compact_receipt=compact,
        )
    assert log.read_text(encoding="ascii").splitlines() == ["a"]
    assert not output_root.exists()
    assert not compact.exists()


def test_compact_is_suppressed_when_authoritative_root_reload_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    _install_fresh_fake_workers(tmp_path, monkeypatch, exporter, fixture)
    actual_validate = fixture.worker.module.validate_legacy_control_lock
    calls = 0

    def fail_second_reload(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("synthetic authoritative reload failure")
        return actual_validate(*args, **kwargs)

    monkeypatch.setattr(
        fixture.worker.module, "validate_legacy_control_lock", fail_second_reload
    )
    output_root = tmp_path / "authoritative"
    compact = tmp_path / "compact.json"
    with pytest.raises(ValueError, match="authoritative reload failure"):
        exporter._coordinate(
            query_path=fixture.worker.query_path,
            qrel_path=fixture.worker.qrel_path,
            embedding_path=fixture.worker.embedding_path,
            feasibility_receipt=feasibility_receipt,
            output_root=output_root,
            compact_receipt=compact,
        )
    assert output_root.is_dir()
    assert not compact.exists()


def test_script_bootstrap_pins_threads_before_numerical_or_trec_imports() -> None:
    code = r"""
import importlib.util
import json
import os
import sys
spec = importlib.util.spec_from_file_location("fresh_legacy_exporter", sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
print(json.dumps({
    "threads": {name: os.environ.get(name) for name in module._THREAD_VARIABLES},
    "forbidden": sorted(
        name for name in sys.modules
        if name == "numpy" or name.startswith("numpy.") or name.startswith("trec_mlm")
    ),
}))
"""
    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = "99"
    completed = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT_PATH)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert set(result["threads"].values()) == {"1"}
    assert result["forbidden"] == []


def test_coordinator_detects_feasibility_manifest_drift_before_repeat_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    log = _install_fresh_fake_workers(tmp_path, monkeypatch, exporter, fixture)
    actual_run = exporter._run_fresh_worker

    def drift_after_a(**kwargs: object) -> int:
        return_code = actual_run(**kwargs)
        if kwargs["repeat"] == "a":
            with feasibility_receipt.open("ab") as handle:
                handle.write(b"drift")
        return return_code

    monkeypatch.setattr(exporter, "_run_fresh_worker", drift_after_a)
    output_root = tmp_path / "authoritative"
    compact = tmp_path / "compact.json"
    with pytest.raises(RuntimeError, match="feasibility manifest drifted"):
        exporter._coordinate(
            query_path=fixture.worker.query_path,
            qrel_path=fixture.worker.qrel_path,
            embedding_path=fixture.worker.embedding_path,
            feasibility_receipt=feasibility_receipt,
            output_root=output_root,
            compact_receipt=compact,
        )
    assert log.read_text(encoding="ascii").splitlines() == ["a"]
    assert not output_root.exists()
    assert not compact.exists()


def test_worker_scratch_is_preflighted_before_worker_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    output_root = tmp_path / "authoritative"
    scratch = output_root.parent / f".{output_root.name}.legacy-control-scratch"
    scratch.mkdir()
    (scratch / "unexpected").write_text("race\n", encoding="utf-8")
    monkeypatch.setattr(
        exporter,
        "_worker_command",
        lambda **kwargs: pytest.fail("worker built before scratch preflight"),
        raising=False,
    )
    with pytest.raises(ValueError, match="scratch root is not empty"):
        exporter._coordinate(
            query_path=fixture.worker.query_path,
            qrel_path=fixture.worker.qrel_path,
            embedding_path=fixture.worker.embedding_path,
            feasibility_receipt=feasibility_receipt,
            output_root=output_root,
            compact_receipt=tmp_path / "compact.json",
        )
    assert (scratch / "unexpected").read_text(encoding="utf-8") == "race\n"


def test_worker_scratch_plan_precedes_destination_parent_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    output_root = tmp_path / "new-output-parent" / "authoritative"
    compact = tmp_path / "new-compact-parent" / "compact.json"

    def stop_after_scratch_plan(staging: Path) -> dict[str, Path]:
        assert not output_root.parent.exists()
        assert not compact.parent.exists()
        raise RuntimeError("synthetic scratch plan stop")

    monkeypatch.setattr(
        exporter, "_preflight_worker_namespaces", stop_after_scratch_plan
    )
    with pytest.raises(RuntimeError, match="synthetic scratch plan stop"):
        exporter._coordinate(
            query_path=fixture.worker.query_path,
            qrel_path=fixture.worker.qrel_path,
            embedding_path=fixture.worker.embedding_path,
            feasibility_receipt=feasibility_receipt,
            output_root=output_root,
            compact_receipt=compact,
        )
    assert not output_root.parent.exists()
    assert not compact.parent.exists()


def test_output_and_compact_races_never_replace_existing_destinations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    _install_fresh_fake_workers(tmp_path, monkeypatch, exporter, fixture)
    actual_rename = exporter._rename_noreplace
    output_root = tmp_path / "authoritative"
    compact = tmp_path / "compact.json"

    def race_root(source: Path, destination: Path) -> None:
        destination.mkdir()
        actual_rename(source, destination)

    monkeypatch.setattr(exporter, "_rename_noreplace", race_root)
    with pytest.raises(OSError):
        exporter._coordinate(
            query_path=fixture.worker.query_path,
            qrel_path=fixture.worker.qrel_path,
            embedding_path=fixture.worker.embedding_path,
            feasibility_receipt=feasibility_receipt,
            output_root=output_root,
            compact_receipt=compact,
        )
    assert output_root.is_dir() and not any(output_root.iterdir())
    assert not compact.exists()


def test_compact_race_occurs_after_one_root_publish_and_never_replaces(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    _install_fresh_fake_workers(tmp_path, monkeypatch, exporter, fixture)
    actual_publish = exporter._atomic_publish_bytes
    calls = 0

    def race_compact(path: Path, payload: bytes) -> None:
        nonlocal calls
        calls += 1
        path.write_bytes(b"user-owned\n")
        actual_publish(path, payload)

    monkeypatch.setattr(exporter, "_atomic_publish_bytes", race_compact)
    output_root = tmp_path / "authoritative"
    compact = tmp_path / "compact.json"
    with pytest.raises(FileExistsError):
        exporter._coordinate(
            query_path=fixture.worker.query_path,
            qrel_path=fixture.worker.qrel_path,
            embedding_path=fixture.worker.embedding_path,
            feasibility_receipt=feasibility_receipt,
            output_root=output_root,
            compact_receipt=compact,
        )
    assert calls == 1
    assert output_root.is_dir()
    assert compact.read_bytes() == b"user-owned\n"


def test_worker_result_bounds_tsv_before_hashing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    exporter = _load_exporter_module()
    tsv_path = tmp_path / "oversized-worker.tsv"
    with tsv_path.open("wb") as handle:
        handle.seek(32 * 1024 * 1024)
        handle.write(b"x")
    result_path = tmp_path / "worker.json"
    result = _worker_result_payload(fixture)
    result["tsv_sha256"] = "0" * 64
    _write_canonical_json(result_path, result)
    actual_hash = exporter._raw_file_sha256

    def guard_unbounded_hash(path: Path) -> str:
        if Path(path) == tsv_path:
            pytest.fail("oversized worker TSV entered unbounded hash loop")
        return actual_hash(path)

    monkeypatch.setattr(exporter, "_raw_file_sha256", guard_unbounded_hash)
    with pytest.raises(ValueError, match="exceeds its byte limit"):
        exporter._validate_worker_result(result_path, output_tsv=tsv_path)


def test_coordinator_resolves_relative_inputs_before_repo_cwd_children(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    _install_fresh_fake_workers(tmp_path, monkeypatch, exporter, fixture)
    fake_worker_command = exporter._worker_command
    observed_worker_paths: list[Path] = []

    def assert_absolute_worker_paths(**kwargs: object) -> list[str]:
        paths = [
            Path(kwargs[name])
            for name in (
                "query_path",
                "qrel_path",
                "embedding_path",
                "output_tsv",
                "result_json",
            )
        ]
        assert all(path.is_absolute() for path in paths)
        observed_worker_paths.extend(paths)
        return fake_worker_command(**kwargs)

    monkeypatch.setattr(exporter, "_worker_command", assert_absolute_worker_paths)
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)

    def relative(path: Path) -> Path:
        return Path(os.path.relpath(path, caller))

    output_root = relative(tmp_path / "relative-authoritative")
    compact = relative(tmp_path / "relative-compact.json")
    exporter._coordinate(
        query_path=relative(fixture.worker.query_path),
        qrel_path=relative(fixture.worker.qrel_path),
        embedding_path=relative(fixture.worker.embedding_path),
        feasibility_receipt=relative(feasibility_receipt),
        output_root=output_root,
        compact_receipt=compact,
    )
    assert len(observed_worker_paths) == 10
    assert (caller / output_root).is_dir()
    assert (caller / compact).is_file()


def test_coordinator_revalidates_manifested_gate_leaf_before_repeat_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    log = _install_fresh_fake_workers(tmp_path, monkeypatch, exporter, fixture)
    actual_run = exporter._run_fresh_worker

    def stop_gate_after_a(**kwargs: object) -> int:
        return_code = actual_run(**kwargs)
        if kwargs["repeat"] == "a":
            _write_canonical_json(
                feasibility_receipt.parent / "feasibility_gate.json",
                {
                    "schema": "trec-mlm-dense-aggregation-feasibility-gate-v1",
                    "status": "stop",
                    "failures": ["synthetic drift"],
                    "projection_seconds": 1.0,
                    "maximum_rss_bytes": 1,
                },
            )
        return return_code

    monkeypatch.setattr(exporter, "_run_fresh_worker", stop_gate_after_a)
    with pytest.raises(RuntimeError, match="feasibility artifact drifted"):
        exporter._coordinate(
            query_path=fixture.worker.query_path,
            qrel_path=fixture.worker.qrel_path,
            embedding_path=fixture.worker.embedding_path,
            feasibility_receipt=feasibility_receipt,
            output_root=tmp_path / "authoritative",
            compact_receipt=tmp_path / "compact.json",
        )
    assert log.read_text(encoding="ascii").splitlines() == ["a"]


def test_spawned_worker_inherits_audit_guard_for_official_qrel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_synthetic_lock(tmp_path, monkeypatch)
    feasibility_receipt = _build_feasibility_root(tmp_path / "feasibility")
    exporter = _load_exporter_module()
    log = _install_fresh_fake_workers(
        tmp_path, monkeypatch, exporter, fixture, mode="attempt-official"
    )
    guard_dir = tmp_path / "guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(
        """
import os
import sys

target = os.path.normpath(os.path.abspath(os.environ["TASK8_FORBIDDEN_QREL"]))

def block_official_qrel(event, args):
    if event != "open" or not args or isinstance(args[0], int):
        return
    try:
        candidate = os.path.normpath(os.path.abspath(os.fspath(args[0])))
    except TypeError:
        return
    if candidate == target:
        raise RuntimeError("child audit blocked official qrel access")

sys.addaudithook(block_official_qrel)
""".lstrip(),
        encoding="utf-8",
    )
    inherited = str(guard_dir)
    if os.environ.get("PYTHONPATH"):
        inherited += os.pathsep + os.environ["PYTHONPATH"]
    monkeypatch.setenv("PYTHONPATH", inherited)
    monkeypatch.setenv("TASK8_FORBIDDEN_QREL", str(OFFICIAL_QREL_PATH))
    with pytest.raises(RuntimeError, match="repeat a failed"):
        exporter._coordinate(
            query_path=fixture.worker.query_path,
            qrel_path=fixture.worker.qrel_path,
            embedding_path=fixture.worker.embedding_path,
            feasibility_receipt=feasibility_receipt,
            output_root=tmp_path / "authoritative",
            compact_receipt=tmp_path / "compact.json",
        )
    assert log.read_text(encoding="ascii").splitlines() == ["a"]


def test_legacy_imports_are_an_explicit_allowlist() -> None:
    allowed = {
        MODULE_PATH: {
            "__future__",
            "collections.abc",
            "csv",
            "dataclasses",
            "hashlib",
            "io",
            "json",
            "math",
            "os",
            "pathlib",
            "stat",
            "struct",
            "types",
            "typing",
            "numpy",
            "numpy.typing",
            "trec_mlm",
            "trec_mlm.dense_query_knn",
            "trec_mlm.metrics",
            "trec_mlm.query_knn",
        },
        SCRIPT_PATH: {
            "__future__",
            "argparse",
            "ctypes",
            "hashlib",
            "json",
            "math",
            "os",
            "pathlib",
            "shutil",
            "stat",
            "struct",
            "subprocess",
            "sys",
            "uuid",
            "trec_mlm.legacy_dense_control",
        },
    }
    for path, expected in allowed.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        } | {
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        }
        assert imported == expected


def test_export_checks_locked_legacy_scorer_before_loading_qrels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _install_synthetic_worker(tmp_path, monkeypatch)
    monkeypatch.setattr(fixture.module, "_LOCKED_LEGACY_SOURCE_SHA256", "0" * 64)
    monkeypatch.setattr(
        fixture.module,
        "load_development_data",
        lambda *_: pytest.fail("qrels loaded before scorer source check"),
    )
    with pytest.raises(RuntimeError, match="legacy dense scorer source hash mismatch"):
        fixture.module.export_fixed_mpnet_k50_control(
            query_path=fixture.query_path,
            qrel_path=fixture.qrel_path,
            embedding_path=fixture.embedding_path,
            output_tsv=tmp_path / "source-mismatch.tsv",
        )
