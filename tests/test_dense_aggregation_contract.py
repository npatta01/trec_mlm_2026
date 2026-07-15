from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray
import pytest

import trec_mlm.dense_aggregation_contract as contract_module
from trec_mlm.dense_aggregation_artifacts import raw_file_sha256
from trec_mlm.dense_aggregation_contract import (
    DEPENDENCY_SOURCE_FILES,
    IMPLEMENTATION_FILES,
    RUNTIME_PROVENANCE_FILES,
    FeasibilityConfig,
    FrozenDenseInputs,
    _cosine_similarity_matrix_qrel_free,
    _legacy_array_sha256_qrel_free,
    _legacy_query_input_sha256,
    _load_queries_tsv_qrel_free,
    _require_exact_mpnet_metadata,
    _require_exact_runtime,
    build_clustering_contract,
    contract_source_inventory_sha256,
    load_and_validate_frozen_dense_inputs,
    load_feasibility_config,
    runtime_source_inventory_sha256,
)
from trec_mlm.dense_aggregation_hashing import canonical_json, framed_sha256
from trec_mlm.dense_aggregation_runtime import THREAD_ENVIRONMENT


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/trec_2026_dense_aggregation_feasibility.json"
DEVELOPMENT_QUERY_PATH = REPO_ROOT / "data/dev/llm_dev_data.tsv"
PROTOCOL_PATH = REPO_ROOT / "protocols/trec_2026_dense_aggregation.md"
CONTRACT_PATH = REPO_ROOT / "trec_mlm/dense_aggregation_contract.py"

QUERY_FILE_SHA256 = (
    "862ff85381b8055cd927ada25c56666594ab1ec3c21e182beb20c58f6c637f3c"
)
QUERY_INPUT_SHA256 = (
    "98faa04f979e428933c3d8e67dac43cf4dded112bd2c16969b3482a1d52886d9"
)
EMBEDDING_SHA256 = (
    "6d197055eb62ab9809ae321b0f60072e20f17524861e6a989b49cd35c389055b"
)
SIMILARITY_SHA256 = (
    "80c69a8dca29cd5e9a4de08dc4e44553fabf813428dd4956f7ecbeb2852968ed"
)
PROTOCOL_SHA256 = (
    "bbe6b50fd3d20efe3213a446f6665cbe0527114f34bf5c1e90236af4f998ad49"
)
LEGACY_SOURCE_SHA256 = (
    "90f8db3abbb2887d8513fb881c631a3843319c29da02e935e1ecdb6c3c85e248"
)
MODEL_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"

EXPECTED_CONFIG = {
    "schema": "trec-mlm-dense-aggregation-feasibility-config-v1",
    "query_path": "data/dev/llm_dev_data.tsv",
    "embedding_path": "artifacts/frozen_dense_input/mpnet_base_v2.npy",
    "encoder_metadata_path": "artifacts/frozen_dense_input/encoder_metadata.json",
    "protocol_path": "protocols/trec_2026_dense_aggregation.md",
}
EXPECTED_METADATA = {
    "key": "mpnet_base_v2",
    "model_id": "sentence-transformers/all-mpnet-base-v2",
    "license": "Apache-2.0",
    "query_prefix": "",
    "requested_revision": MODEL_REVISION,
    "resolved_revision": MODEL_REVISION,
    "pooling": "mean",
    "embedding_dimension": 768,
    "dtype": "float32",
    "normalized": True,
    "query_count": 322,
    "input_sha256": QUERY_INPUT_SHA256,
    "embedding_sha256": EMBEDDING_SHA256,
    "similarity_sha256": SIMILARITY_SHA256,
}
EXPECTED_VERSIONS = {
    "python": "3.13.9",
    "numpy": "2.5.1",
    "scipy": "1.18.0",
    "scikit_learn": "1.9.0",
}
EXPECTED_IMPLEMENTATION_FILES = (
    "trec_mlm/dense_aggregation.py",
    "trec_mlm/dense_aggregation_artifacts.py",
    "trec_mlm/dense_aggregation_contract.py",
    "trec_mlm/dense_aggregation_entrypoint.py",
    "trec_mlm/dense_aggregation_hashing.py",
    "trec_mlm/dense_aggregation_runtime.py",
)
EXPECTED_DEPENDENCY_FILES = (
    "trec_mlm/dense_query_knn.py",
    "trec_mlm/metrics.py",
    "trec_mlm/query_knn.py",
    "trec_mlm/runio.py",
)
EXPECTED_RUNTIME_FILES = ("pyproject.toml", "uv.lock")


@dataclasses.dataclass(frozen=True, slots=True)
class SyntheticBundle:
    root: Path
    config_path: Path
    query_path: Path
    embedding_path: Path
    metadata_path: Path
    query_ids: tuple[str, ...]
    query_texts: tuple[str, ...]
    embeddings: NDArray[np.float32]
    query_file_sha256: str
    query_input_sha256: str
    embedding_sha256: str
    similarity_sha256: str


def _independent_query_input_sha256(
    query_ids: tuple[str, ...], query_texts: tuple[str, ...]
) -> str:
    digest = hashlib.sha256()
    for query_id, query_text in zip(query_ids, query_texts, strict=True):
        digest.update(query_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(query_text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _independent_array_sha256(array: NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _independent_cosine(embeddings: NDArray[Any]) -> NDArray[np.float32]:
    array = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = np.zeros_like(array, dtype=np.float32)
    np.divide(array, norms, out=normalized, where=norms > 0.0)
    similarities = np.asarray(normalized @ normalized.T, dtype=np.float32)
    np.clip(similarities, -1.0, 1.0, out=similarities)
    return similarities


def _metadata_for(
    *,
    query_input_sha256: str,
    embedding_sha256: str,
    similarity_sha256: str,
) -> dict[str, object]:
    return {
        **EXPECTED_METADATA,
        "input_sha256": query_input_sha256,
        "embedding_sha256": embedding_sha256,
        "similarity_sha256": similarity_sha256,
    }


def _write_metadata(bundle: SyntheticBundle, metadata: dict[str, object]) -> None:
    bundle.metadata_path.write_text(
        json.dumps({"mpnet_base_v2": metadata}, sort_keys=True),
        encoding="utf-8",
    )


def _make_synthetic_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SyntheticBundle:
    root = tmp_path / "synthetic-repo"
    config_dir = root / "configs"
    query_path = root / "data/dev/queries.tsv"
    embedding_path = root / "artifacts/frozen/mpnet.npy"
    metadata_path = root / "artifacts/frozen/metadata.json"
    protocol_path = root / "protocols/trec_2026_dense_aggregation.md"
    for directory in (
        config_dir,
        query_path.parent,
        embedding_path.parent,
        protocol_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROTOCOL_PATH, protocol_path)

    query_ids = tuple(f"q{index:04d}" for index in range(322))
    query_texts = tuple(f"Synthetic question {index:04d}" for index in range(322))
    query_map = dict(zip(query_ids, query_texts, strict=True))
    query_path.write_text(
        "".join(
            f"{query_id}\t{query_map[query_id]}\n"
            for query_id in reversed(query_ids)
        ),
        encoding="utf-8",
    )

    embeddings = np.eye(322, 768, dtype=np.float32)
    np.save(embedding_path, embeddings, allow_pickle=False)
    similarities = _independent_cosine(embeddings)
    query_file_sha256 = raw_file_sha256(query_path)
    query_input_sha256 = _independent_query_input_sha256(query_ids, query_texts)
    embedding_sha256 = _independent_array_sha256(embeddings)
    similarity_sha256 = _independent_array_sha256(similarities)

    config_path = config_dir / "synthetic.json"
    config_path.write_bytes(
        json.dumps(
            {
                "schema": "trec-mlm-dense-aggregation-feasibility-config-v1",
                "query_path": "data/dev/queries.tsv",
                "embedding_path": "artifacts/frozen/mpnet.npy",
                "encoder_metadata_path": "artifacts/frozen/metadata.json",
                "protocol_path": "protocols/trec_2026_dense_aggregation.md",
            },
            indent=2,
        ).encode("utf-8")
        + b"\n"
    )
    bundle = SyntheticBundle(
        root=root,
        config_path=config_path,
        query_path=query_path,
        embedding_path=embedding_path,
        metadata_path=metadata_path,
        query_ids=query_ids,
        query_texts=query_texts,
        embeddings=embeddings,
        query_file_sha256=query_file_sha256,
        query_input_sha256=query_input_sha256,
        embedding_sha256=embedding_sha256,
        similarity_sha256=similarity_sha256,
    )
    _write_metadata(
        bundle,
        _metadata_for(
            query_input_sha256=query_input_sha256,
            embedding_sha256=embedding_sha256,
            similarity_sha256=similarity_sha256,
        ),
    )
    monkeypatch.setattr(contract_module, "_QUERY_FILE_SHA256", query_file_sha256)
    monkeypatch.setattr(contract_module, "_QUERY_INPUT_SHA256", query_input_sha256)
    monkeypatch.setattr(contract_module, "_EMBEDDING_SHA256", embedding_sha256)
    monkeypatch.setattr(contract_module, "_SIMILARITY_SHA256", similarity_sha256)
    return bundle


def _pin_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")


def _copy_contract_root(destination: Path) -> Path:
    destination.mkdir(parents=True)
    all_files = (
        *IMPLEMENTATION_FILES,
        *DEPENDENCY_SOURCE_FILES,
        *RUNTIME_PROVENANCE_FILES,
        "protocols/trec_2026_dense_aggregation.md",
    )
    for relative in all_files:
        source = REPO_ROOT / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


@pytest.fixture(scope="module")
def valid_inputs() -> FrozenDenseInputs:
    embeddings = np.eye(322, 768, dtype=np.float32)
    similarities = np.eye(322, dtype=np.float32)
    return FrozenDenseInputs(
        query_ids=tuple(f"q{index:04d}" for index in range(322)),
        query_texts=tuple(f"Question {index:04d}" for index in range(322)),
        embeddings=embeddings,
        similarities=similarities,
        query_input_sha256=QUERY_INPUT_SHA256,
        embedding_sha256=EMBEDDING_SHA256,
        similarity_sha256=SIMILARITY_SHA256,
    )


def test_frozen_constants_and_literal_config_are_exact() -> None:
    assert contract_module._QUERY_FILE_SHA256 == QUERY_FILE_SHA256
    assert contract_module._QUERY_INPUT_SHA256 == QUERY_INPUT_SHA256
    assert contract_module._EMBEDDING_SHA256 == EMBEDDING_SHA256
    assert contract_module._SIMILARITY_SHA256 == SIMILARITY_SHA256
    assert raw_file_sha256(PROTOCOL_PATH) == PROTOCOL_SHA256
    assert raw_file_sha256(REPO_ROOT / "trec_mlm/dense_query_knn.py") == (
        LEGACY_SOURCE_SHA256
    )
    assert json.loads(CONFIG_PATH.read_bytes()) == EXPECTED_CONFIG


def test_tracked_development_queries_match_qrel_free_frozen_locks() -> None:
    query_bytes = DEVELOPMENT_QUERY_PATH.read_bytes()
    assert hashlib.sha256(query_bytes).hexdigest() == QUERY_FILE_SHA256

    independently_parsed: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        query_bytes.decode("utf-8").splitlines(), start=1
    ):
        if not raw_line:
            continue
        query_id, separator, query_text = raw_line.partition("\t")
        assert separator == "\t", f"line {line_number} has no tab separator"
        query_id, query_text = query_id.strip(), query_text.strip()
        assert query_id and query_text
        assert query_id not in independently_parsed
        independently_parsed[query_id] = query_text

    assert _load_queries_tsv_qrel_free(DEVELOPMENT_QUERY_PATH) == (
        independently_parsed
    )
    query_ids = tuple(sorted(independently_parsed))
    query_texts = tuple(independently_parsed[query_id] for query_id in query_ids)
    assert len(query_ids) == len(set(query_ids)) == 322
    assert _independent_query_input_sha256(query_ids, query_texts) == (
        QUERY_INPUT_SHA256
    )
    assert _legacy_query_input_sha256(query_ids, query_texts) == (
        QUERY_INPUT_SHA256
    )

    one_byte_mutation = bytearray(query_bytes)
    one_byte_mutation[0] ^= 1
    assert hashlib.sha256(one_byte_mutation).hexdigest() != QUERY_FILE_SHA256
    assert _legacy_query_input_sha256(
        tuple(reversed(query_ids)), tuple(reversed(query_texts))
    ) != QUERY_INPUT_SHA256
    mutated_query_ids = (f"{query_ids[0]}-mutated", *query_ids[1:])
    assert _legacy_query_input_sha256(
        mutated_query_ids, query_texts
    ) != QUERY_INPUT_SHA256


def test_config_and_dataclass_fields_have_no_qrel_or_grade_key() -> None:
    serialized_keys = " ".join(json.loads(CONFIG_PATH.read_bytes())).lower()
    field_names = " ".join(
        field.name
        for data_type in (FrozenDenseInputs, FeasibilityConfig)
        for field in dataclasses.fields(data_type)
    ).lower()
    assert "qrel" not in serialized_keys
    assert "grade" not in serialized_keys
    assert "qrel" not in field_names
    assert "grade" not in field_names
    assert FrozenDenseInputs.__dataclass_params__.frozen is True
    assert FeasibilityConfig.__dataclass_params__.frozen is True
    assert "__slots__" in FrozenDenseInputs.__dict__
    assert "__slots__" in FeasibilityConfig.__dict__


def test_literal_config_loads_without_requiring_future_bundle() -> None:
    config = load_feasibility_config(CONFIG_PATH)
    assert config.repo_root == REPO_ROOT
    assert config.source_path == CONFIG_PATH.resolve(strict=True)
    assert config.source_sha256 == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    assert config.query_path == (REPO_ROOT / EXPECTED_CONFIG["query_path"]).resolve()
    assert config.embedding_path == (
        REPO_ROOT / EXPECTED_CONFIG["embedding_path"]
    ).resolve()
    assert config.encoder_metadata_path == (
        REPO_ROOT / EXPECTED_CONFIG["encoder_metadata_path"]
    ).resolve()
    assert config.protocol_path == PROTOCOL_PATH.resolve(strict=True)


def test_feasibility_config_rejects_extra_missing_and_wrong_schema(
    tmp_path: Path,
) -> None:
    payload = dict(EXPECTED_CONFIG)
    payload["qrel_path"] = "forbidden"
    bad = tmp_path / "bad-extra.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"extra=.*qrel_path"):
        load_feasibility_config(bad)

    payload = dict(EXPECTED_CONFIG)
    del payload["embedding_path"]
    bad = tmp_path / "bad-missing.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=r"missing=.*embedding_path"):
        load_feasibility_config(bad)

    payload = {**EXPECTED_CONFIG, "schema": "unsupported"}
    bad = tmp_path / "bad-schema.json"
    bad.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported feasibility config schema"):
        load_feasibility_config(bad)


def test_feasibility_config_rejects_nonobject_and_path_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    config_dir = root / "configs"
    config_dir.mkdir(parents=True)
    nonobject = config_dir / "nonobject.json"
    nonobject.write_text(json.dumps(list(EXPECTED_CONFIG)), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_feasibility_config(nonobject)

    payload = {
        **EXPECTED_CONFIG,
        "query_path": "../outside.tsv",
    }
    escaped = config_dir / "escaped.json"
    escaped.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="outside repository root"):
        load_feasibility_config(escaped)

    payload["query_path"] = str((tmp_path / "absolute-outside.tsv").resolve())
    escaped.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="outside repository root"):
        load_feasibility_config(escaped)


def test_feasibility_config_rejects_in_repo_symlink_escape(
    tmp_path: Path,
) -> None:
    root = tmp_path / "repo"
    config_dir = root / "configs"
    data_dir = root / "data"
    config_dir.mkdir(parents=True)
    data_dir.mkdir()
    outside_query = tmp_path / "outside-query.tsv"
    outside_query.write_text("q1\toutside\n", encoding="utf-8")
    escaped_query = data_dir / "queries.tsv"
    escaped_query.symlink_to(outside_query)
    assert escaped_query.is_symlink()
    assert escaped_query.resolve(strict=True) == outside_query.resolve(strict=True)

    config_path = config_dir / "symlink-escape.json"
    config_path.write_text(
        json.dumps({**EXPECTED_CONFIG, "query_path": "data/queries.tsv"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="outside repository root"):
        load_feasibility_config(config_path)


def test_config_retains_hash_of_the_single_captured_byte_string(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_synthetic_bundle(tmp_path, monkeypatch)
    captured = bundle.config_path.read_bytes()
    config = load_feasibility_config(bundle.config_path)
    bundle.config_path.write_bytes(captured + b" ")
    assert config.source_sha256 == hashlib.sha256(captured).hexdigest()
    assert raw_file_sha256(config.source_path) != config.source_sha256
    assert config.repo_root == bundle.root.resolve(strict=True)
    assert config.source_path == bundle.config_path.resolve(strict=True)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("", "query file is empty"),
        ("q1 no tab\n", "expected query_id<TAB>query_text"),
        ("q1\t \n", "blank query ID or text"),
        ("q1\tone\nq1\ttwo\n", "duplicate query ID"),
    ],
)
def test_local_query_parser_rejects_malformed_inputs(
    tmp_path: Path, contents: str, message: str
) -> None:
    path = tmp_path / "queries.tsv"
    path.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        _load_queries_tsv_qrel_free(path)


def test_local_legacy_hash_and_cosine_helpers_have_literal_goldens() -> None:
    assert _legacy_query_input_sha256(
        ("q1", "q2"), ("snow 雪", "x")
    ) == "c23cd90228cbc4bb546b2aaf07f2828e6d62514d7ef3f90bf604f20b9eca03d4"
    values = np.asarray([[1.0, -0.0], [2.5, 3.0]], dtype=np.float32)
    assert _legacy_array_sha256_qrel_free(values) == (
        "eb699de43dd5fbee91e312ed965dc881cf2ec4d1c5e8dc959bdbacb6539f99c4"
    )
    cosine = _cosine_similarity_matrix_qrel_free(
        np.asarray([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    )
    np.testing.assert_array_equal(
        cosine, np.asarray([[1.0, 0.8], [0.8, 1.0]], dtype=np.float32)
    )
    assert _legacy_array_sha256_qrel_free(cosine) == (
        "271f5406ba445046062d681d278d315f1fe45441f796126f6c046e7d65532408"
    )
    with pytest.raises(ValueError, match="zip"):
        _legacy_query_input_sha256(("q1",), ("one", "two"))
    with pytest.raises(ValueError, match="nonempty two-dimensional"):
        _cosine_similarity_matrix_qrel_free(np.asarray([], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        _cosine_similarity_matrix_qrel_free(
            np.asarray([[np.nan]], dtype=np.float32)
        )


def test_synthetic_bundle_parses_hashes_and_validates_exact_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_synthetic_bundle(tmp_path, monkeypatch)
    config = load_feasibility_config(bundle.config_path)
    inputs = load_and_validate_frozen_dense_inputs(config)
    assert inputs.query_ids == bundle.query_ids
    assert inputs.query_texts == bundle.query_texts
    assert len(inputs.query_ids) == 322
    assert inputs.embeddings.shape == (322, 768)
    assert inputs.embeddings.dtype == np.dtype("float32")
    assert inputs.embeddings.flags.c_contiguous
    assert np.isfinite(inputs.embeddings).all()
    np.testing.assert_allclose(
        np.linalg.norm(inputs.embeddings, axis=1),
        np.ones(322, dtype=np.float32),
        rtol=1e-5,
        atol=1e-6,
    )
    assert inputs.similarities.shape == (322, 322)
    assert inputs.similarities.dtype == np.dtype("float32")
    assert inputs.similarities.flags.c_contiguous
    np.testing.assert_array_equal(inputs.similarities, np.eye(322, dtype=np.float32))
    assert inputs.query_input_sha256 == bundle.query_input_sha256
    assert inputs.embedding_sha256 == bundle.embedding_sha256
    assert inputs.similarity_sha256 == bundle.similarity_sha256
    assert _independent_query_input_sha256(
        inputs.query_ids, inputs.query_texts
    ) == inputs.query_input_sha256
    assert _independent_array_sha256(inputs.embeddings) == inputs.embedding_sha256
    assert _independent_array_sha256(
        _independent_cosine(inputs.embeddings)
    ) == inputs.similarity_sha256


def test_synthetic_fortran_array_is_returned_c_contiguous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_synthetic_bundle(tmp_path, monkeypatch)
    np.save(
        bundle.embedding_path,
        np.asfortranarray(bundle.embeddings),
        allow_pickle=False,
    )
    inputs = load_and_validate_frozen_dense_inputs(
        load_feasibility_config(bundle.config_path)
    )
    assert inputs.embeddings.flags.c_contiguous
    assert not inputs.embeddings.flags.f_contiguous
    assert inputs.embedding_sha256 == bundle.embedding_sha256


def test_frozen_input_rejects_query_file_and_query_input_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_synthetic_bundle(tmp_path, monkeypatch)
    bundle.query_path.write_bytes(bundle.query_path.read_bytes() + b"q9999\tmutation\n")
    with pytest.raises(ValueError, match="query file hash mismatch"):
        load_and_validate_frozen_dense_inputs(
            load_feasibility_config(bundle.config_path)
        )

    bundle = _make_synthetic_bundle(tmp_path / "second", monkeypatch)
    lines = bundle.query_path.read_text(encoding="utf-8").splitlines()
    fields = lines[0].split("\t", maxsplit=1)
    lines[0] = f"{fields[0]}\tmutated text"
    bundle.query_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        contract_module, "_QUERY_FILE_SHA256", raw_file_sha256(bundle.query_path)
    )
    with pytest.raises(ValueError, match="query-input logical hash mismatch"):
        load_and_validate_frozen_dense_inputs(
            load_feasibility_config(bundle.config_path)
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda value: value.astype(np.float64), r"shape \(322, 768\).*float32"),
        (lambda value: value[:-1], r"shape \(322, 768\).*float32"),
        (
            lambda value: np.where(
                np.arange(value.size).reshape(value.shape) == 0, np.nan, value
            ).astype(np.float32),
            "finite and nonzero",
        ),
        (
            lambda value: np.vstack(
                [np.zeros((1, value.shape[1]), dtype=np.float32), value[1:]]
            ),
            "finite and nonzero",
        ),
        (lambda value: value * np.float32(2.0), "not normalized"),
        (lambda value: value[::-1].copy(), "embedding logical hash mismatch"),
    ],
)
def test_frozen_input_rejects_invalid_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutator: Any,
    message: str,
) -> None:
    bundle = _make_synthetic_bundle(tmp_path, monkeypatch)
    np.save(bundle.embedding_path, mutator(bundle.embeddings), allow_pickle=False)
    with pytest.raises(ValueError, match=message):
        load_and_validate_frozen_dense_inputs(
            load_feasibility_config(bundle.config_path)
        )


def test_frozen_input_recomputes_and_locks_cosine_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bundle = _make_synthetic_bundle(tmp_path, monkeypatch)
    embeddings = bundle.embeddings.copy()
    embeddings[0] = 0.0
    embeddings[0, 0] = np.float32(0.6)
    embeddings[0, 1] = np.float32(0.8)
    np.save(bundle.embedding_path, embeddings, allow_pickle=False)
    embedding_sha256 = _independent_array_sha256(embeddings)
    monkeypatch.setattr(contract_module, "_EMBEDDING_SHA256", embedding_sha256)
    _write_metadata(
        bundle,
        _metadata_for(
            query_input_sha256=bundle.query_input_sha256,
            embedding_sha256=embedding_sha256,
            similarity_sha256=bundle.similarity_sha256,
        ),
    )
    with pytest.raises(ValueError, match="cosine logical hash mismatch"):
        load_and_validate_frozen_dense_inputs(
            load_feasibility_config(bundle.config_path)
        )


def test_exact_mpnet_metadata_accepts_only_every_frozen_field() -> None:
    _require_exact_mpnet_metadata(EXPECTED_METADATA)
    replacements: dict[str, object] = {
        "key": "other",
        "model_id": "other/model",
        "license": "MIT",
        "query_prefix": "query: ",
        "requested_revision": "0" * 40,
        "resolved_revision": "0" * 40,
        "pooling": "max",
        "embedding_dimension": 769,
        "dtype": "float64",
        "normalized": False,
        "query_count": 321,
        "input_sha256": "0" * 64,
        "embedding_sha256": "0" * 64,
        "similarity_sha256": "0" * 64,
    }
    for key, replacement in replacements.items():
        mismatched = {**EXPECTED_METADATA, key: replacement}
        with pytest.raises(ValueError, match=key):
            _require_exact_mpnet_metadata(mismatched)
        missing = dict(EXPECTED_METADATA)
        del missing[key]
        with pytest.raises(ValueError, match=key):
            _require_exact_mpnet_metadata(missing)
    for key, alias in (
        ("embedding_dimension", 768.0),
        ("normalized", 1),
        ("query_count", 322.0),
    ):
        with pytest.raises(ValueError, match=key):
            _require_exact_mpnet_metadata({**EXPECTED_METADATA, key: alias})


def test_contract_module_ast_and_import_graph_are_qrel_free() -> None:
    source = CONTRACT_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(CONTRACT_PATH))
    imported: set[str] = set()
    defined_functions: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defined_functions.add(node.name)
    assert "trec_mlm.dense_query_knn" not in imported
    assert "trec_mlm.query_knn" not in imported
    assert {
        "_legacy_array_sha256_qrel_free",
        "_cosine_similarity_matrix_qrel_free",
        "_load_queries_tsv_qrel_free",
    } <= defined_functions

    code = """
import json, sys
import trec_mlm.dense_aggregation_contract
forbidden = [
    name for name in ("trec_mlm.dense_query_knn", "trec_mlm.query_knn")
    if name in sys.modules
]
print(json.dumps(forbidden))
"""
    environment = os.environ.copy()
    environment.update({name: "1" for name in THREAD_ENVIRONMENT})
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_source_file_categories_are_exact_safe_complete_and_disjoint() -> None:
    assert IMPLEMENTATION_FILES == EXPECTED_IMPLEMENTATION_FILES
    assert DEPENDENCY_SOURCE_FILES == EXPECTED_DEPENDENCY_FILES
    assert RUNTIME_PROVENANCE_FILES == EXPECTED_RUNTIME_FILES
    groups = (
        IMPLEMENTATION_FILES,
        DEPENDENCY_SOURCE_FILES,
        RUNTIME_PROVENANCE_FILES,
    )
    for group in groups:
        assert group == tuple(sorted(group))
        assert len(group) == len(set(group))
        for relative in group:
            path = PurePosixPath(relative)
            assert not path.is_absolute()
            assert path.as_posix() == relative
            assert ".." not in path.parts
            full = REPO_ROOT / relative
            assert full.resolve(strict=True).is_relative_to(REPO_ROOT)
            assert stat.S_ISREG(full.lstat().st_mode)
    for index, left in enumerate(groups):
        for right in groups[index + 1 :]:
            assert set(left).isdisjoint(right)
    aggregation_modules = tuple(
        sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in (REPO_ROOT / "trec_mlm").glob("dense_aggregation*.py")
        )
    )
    assert aggregation_modules == IMPLEMENTATION_FILES


def test_exact_runtime_contract_accepts_only_pinned_values() -> None:
    threads: dict[str, object] = {
        **{name: "1" for name in THREAD_ENVIRONMENT},
        "threadpool_limit": 1,
        "process_count": 1,
    }
    _require_exact_runtime(EXPECTED_VERSIONS, threads)
    for key in EXPECTED_VERSIONS:
        with pytest.raises(ValueError, match="runtime"):
            _require_exact_runtime(
                {**EXPECTED_VERSIONS, key: "unexpected"},
                threads,
            )
    with pytest.raises(ValueError, match="runtime"):
        _require_exact_runtime({**EXPECTED_VERSIONS, "extra": "x"}, threads)
    for key in THREAD_ENVIRONMENT:
        with pytest.raises(ValueError, match="thread"):
            _require_exact_runtime(
                EXPECTED_VERSIONS,
                {**threads, key: 1},
            )
    for key in ("threadpool_limit", "process_count"):
        with pytest.raises(ValueError, match="thread"):
            _require_exact_runtime(
                EXPECTED_VERSIONS,
                {**threads, key: True},
            )


def test_clustering_contract_has_exact_protocol_map(
    valid_inputs: FrozenDenseInputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin_threads(monkeypatch)
    payload, digest = build_clustering_contract(
        valid_inputs,
        repo_root=REPO_ROOT,
        protocol_path=PROTOCOL_PATH,
    )
    assert set(payload) == {
        "schema",
        "protocol_file_sha256",
        "implementation_files",
        "embedding_sha256",
        "similarity_sha256",
        "versions",
        "threads",
        "kmeans",
        "center",
        "target_distance",
        "inertia",
        "convergence",
    }
    assert payload["schema"] == "trec-mlm-dense-aggregation-clustering-v1"
    assert payload["protocol_file_sha256"] == PROTOCOL_SHA256
    assert payload["embedding_sha256"] == EMBEDDING_SHA256
    assert payload["similarity_sha256"] == SIMILARITY_SHA256
    assert payload["versions"] == EXPECTED_VERSIONS
    assert payload["threads"] == {
        **{name: "1" for name in THREAD_ENVIRONMENT},
        "threadpool_limit": 1,
        "process_count": 1,
    }
    assert payload["kmeans"] == {
        "init": "k-means++",
        "random_state": 2026,
        "n_init": 10,
        "algorithm": "lloyd",
        "max_iter": 300,
        "tol": 1e-4,
    }
    assert payload["center"] == {
        "dtype": "<f4",
        "layout": "C",
        "canonical_order": "minimum_member_query_id_lexical",
    }
    assert payload["target_distance"] == {
        "dtype": "<f8",
        "reduction": "numpy.sum(delta[z] * delta[z], dtype=numpy.float64)",
        "tie": "minimum_canonical_cluster_id",
    }
    assert payload["inertia"] == {
        "dtype": "<f8",
        "cluster_reduction": (
            "numpy.sum(member_delta * member_delta, dtype=numpy.float64)"
        ),
        "total_reduction": "math.fsum(cluster_inertia)",
        "rtol": 1e-5,
        "atol": 1e-8,
    }
    assert payload["convergence"] == {
        "n_iter_min": 1,
        "n_iter_max": 299,
        "warning_action": "reject",
    }
    expected_rows = [
        [relative, raw_file_sha256(REPO_ROOT / relative)]
        for relative in sorted((*IMPLEMENTATION_FILES, *DEPENDENCY_SOURCE_FILES))
    ]
    assert payload["implementation_files"] == expected_rows
    assert dict(expected_rows)["trec_mlm/dense_query_knn.py"] == (
        LEGACY_SOURCE_SHA256
    )
    assert digest == framed_sha256("clustering-contract", canonical_json(payload))


def test_clustering_contract_rejects_wrong_or_mutated_locked_sources(
    tmp_path: Path,
    valid_inputs: FrozenDenseInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_threads(monkeypatch)
    with pytest.raises(ValueError, match="protocol path mismatch"):
        build_clustering_contract(
            valid_inputs,
            repo_root=REPO_ROOT,
            protocol_path=REPO_ROOT / "protocols/trec_2026_response_retrieval.md",
        )

    protocol_root = _copy_contract_root(tmp_path / "protocol-root")
    protocol = protocol_root / "protocols/trec_2026_dense_aggregation.md"
    protocol.write_bytes(protocol.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="protocol hash mismatch"):
        build_clustering_contract(
            valid_inputs,
            repo_root=protocol_root,
            protocol_path=protocol,
        )

    legacy_root = _copy_contract_root(tmp_path / "legacy-root")
    legacy = legacy_root / "trec_mlm/dense_query_knn.py"
    legacy.write_bytes(legacy.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="legacy dense scorer source hash mismatch"):
        build_clustering_contract(
            valid_inputs,
            repo_root=legacy_root,
            protocol_path=legacy_root
            / "protocols/trec_2026_dense_aggregation.md",
        )


def test_source_inventory_hashes_use_exact_category_rows(
    valid_inputs: FrozenDenseInputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin_threads(monkeypatch)
    payload, _ = build_clustering_contract(
        valid_inputs,
        repo_root=REPO_ROOT,
        protocol_path=PROTOCOL_PATH,
    )
    all_rows = payload["implementation_files"]
    assert isinstance(all_rows, list)
    for category, paths in (
        ("implementation", IMPLEMENTATION_FILES),
        ("dependency", DEPENDENCY_SOURCE_FILES),
    ):
        selected = [row for row in all_rows if row[0] in paths]
        assert contract_source_inventory_sha256(
            payload, category=category  # type: ignore[arg-type]
        ) == framed_sha256(
            f"{category}-source-inventory",
            canonical_json(selected),
        )
    runtime_rows = [
        [relative, raw_file_sha256(REPO_ROOT / relative)]
        for relative in RUNTIME_PROVENANCE_FILES
    ]
    assert runtime_source_inventory_sha256(REPO_ROOT) == framed_sha256(
        "runtime-source-inventory", canonical_json(runtime_rows)
    )

    duplicate = copy.deepcopy(payload)
    duplicate_rows = duplicate["implementation_files"]
    assert isinstance(duplicate_rows, list)
    duplicate_rows.append(copy.deepcopy(duplicate_rows[0]))
    with pytest.raises(ValueError, match="exactly once"):
        contract_source_inventory_sha256(duplicate, category="implementation")
    missing = copy.deepcopy(payload)
    missing_rows = missing["implementation_files"]
    assert isinstance(missing_rows, list)
    missing["implementation_files"] = missing_rows[1:]
    with pytest.raises(ValueError, match="exactly once"):
        contract_source_inventory_sha256(missing, category="implementation")


def test_source_inventory_mutations_are_category_isolated(
    tmp_path: Path,
    valid_inputs: FrozenDenseInputs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_threads(monkeypatch)
    roots = {
        name: _copy_contract_root(tmp_path / name)
        for name in ("baseline", "implementation", "dependency", "runtime")
    }

    def build(root: Path) -> tuple[dict[str, object], str]:
        return build_clustering_contract(
            valid_inputs,
            repo_root=root,
            protocol_path=root / "protocols/trec_2026_dense_aggregation.md",
        )

    baseline, baseline_digest = build(roots["baseline"])
    baseline_categories = {
        category: contract_source_inventory_sha256(
            baseline, category=category  # type: ignore[arg-type]
        )
        for category in ("implementation", "dependency")
    }
    baseline_runtime = runtime_source_inventory_sha256(roots["baseline"])

    implementation_path = (
        roots["implementation"] / "trec_mlm/dense_aggregation_runtime.py"
    )
    implementation_path.write_bytes(
        implementation_path.read_bytes() + b"\n# implementation mutation\n"
    )
    implementation, implementation_digest = build(roots["implementation"])
    assert implementation_digest != baseline_digest
    assert contract_source_inventory_sha256(
        implementation, category="implementation"
    ) != baseline_categories["implementation"]
    assert contract_source_inventory_sha256(
        implementation, category="dependency"
    ) == baseline_categories["dependency"]
    assert runtime_source_inventory_sha256(roots["implementation"]) == (
        baseline_runtime
    )

    dependency_path = roots["dependency"] / "trec_mlm/metrics.py"
    dependency_path.write_bytes(
        dependency_path.read_bytes() + b"\n# dependency mutation\n"
    )
    dependency, dependency_digest = build(roots["dependency"])
    assert dependency_digest != baseline_digest
    assert contract_source_inventory_sha256(
        dependency, category="implementation"
    ) == baseline_categories["implementation"]
    assert contract_source_inventory_sha256(
        dependency, category="dependency"
    ) != baseline_categories["dependency"]
    assert runtime_source_inventory_sha256(roots["dependency"]) == baseline_runtime

    runtime_path = roots["runtime"] / "pyproject.toml"
    runtime_path.write_bytes(runtime_path.read_bytes() + b"\n# runtime mutation\n")
    runtime, runtime_digest = build(roots["runtime"])
    assert runtime == baseline
    assert runtime_digest == baseline_digest
    assert contract_source_inventory_sha256(
        runtime, category="implementation"
    ) == baseline_categories["implementation"]
    assert contract_source_inventory_sha256(
        runtime, category="dependency"
    ) == baseline_categories["dependency"]
    assert runtime_source_inventory_sha256(roots["runtime"]) != baseline_runtime


def test_build_contract_rejects_unpinned_thread_environment(
    valid_inputs: FrozenDenseInputs, monkeypatch: pytest.MonkeyPatch
) -> None:
    _pin_threads(monkeypatch)
    monkeypatch.setenv(THREAD_ENVIRONMENT[0], "2")
    with pytest.raises(ValueError, match="thread"):
        build_clustering_contract(
            valid_inputs,
            repo_root=REPO_ROOT,
            protocol_path=PROTOCOL_PATH,
        )
