from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from trec_mlm.dense_aggregation_artifacts import (
    atomic_artifact_directory,
    raw_file_sha256,
    semantic_artifact_sha256,
    validate_manifest,
    write_manifest,
)
from trec_mlm.dense_aggregation_hashing import canonical_json


FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "dense_aggregation"
    / "hash_golden_vectors.json"
)


@pytest.fixture(scope="module")
def golden() -> dict[str, Any]:
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


def _populate(staging: Path) -> None:
    (staging / "nested").mkdir()
    (staging / "nested" / "payload.bin").write_bytes(b"\x00\x01\xff")
    (staging / "payload.txt").write_text("locked\n", encoding="utf-8")


def _publish(root: Path) -> None:
    with atomic_artifact_directory(root) as staging:
        _populate(staging)


def _rewrite_manifest(root: Path, manifest: dict[str, object]) -> None:
    (root / "manifest.json").write_bytes(canonical_json(manifest) + b"\n")


def test_raw_file_and_manifest_literal_golden_vectors(
    tmp_path: Path, golden: dict[str, str]
) -> None:
    root = tmp_path / "artifact"
    _publish(root)

    assert raw_file_sha256(root / "payload.txt") == golden["payload_text_raw"]
    assert raw_file_sha256(root / "nested" / "payload.bin") == golden[
        "nested_payload_raw"
    ]
    expected: dict[str, object] = {
        "schema": "trec-mlm-dense-aggregation-manifest-v1",
        "files": [
            ["nested/payload.bin", 3, golden["nested_payload_raw"]],
            ["payload.txt", 7, golden["payload_text_raw"]],
        ],
        "artifact_inventory_sha256": golden["aggregate_inventory"],
    }
    assert validate_manifest(root) == expected
    assert json.loads((root / "manifest.json").read_bytes()) == expected
    assert (root / "manifest.json").read_bytes() == canonical_json(expected) + b"\n"


def test_manifest_json_is_the_sole_recursive_exclusion(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "nested").mkdir()
    (root / "nested" / "manifest.json").write_text("nested\n", encoding="utf-8")
    (root / "manifest.backup.json").write_text("backup\n", encoding="utf-8")

    manifest = write_manifest(root)

    assert [row[0] for row in manifest["files"]] == [
        "manifest.backup.json",
        "nested/manifest.json",
    ]
    assert validate_manifest(root) == manifest


@pytest.mark.parametrize("mutation", ("missing", "extra", "reordered", "mutated"))
def test_manifest_rejects_changed_or_reordered_inventory_rows(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "artifact"
    _publish(root)
    manifest = validate_manifest(root)
    rows = [list(row) for row in manifest["files"]]
    if mutation == "missing":
        rows.pop()
    elif mutation == "extra":
        rows.append(["phantom.txt", 0, "0" * 64])
    elif mutation == "reordered":
        rows.reverse()
    else:
        rows[0][2] = "f" * 64
    manifest["files"] = rows
    _rewrite_manifest(root, manifest)

    with pytest.raises(ValueError, match="inventory"):
        validate_manifest(root)


@pytest.mark.parametrize("mutation", ("missing", "extra", "mutated"))
def test_manifest_rejects_missing_extra_or_mutated_files(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "artifact"
    _publish(root)
    if mutation == "missing":
        (root / "payload.txt").unlink()
    elif mutation == "extra":
        (root / "extra.txt").write_text("mutation\n", encoding="utf-8")
    else:
        (root / "payload.txt").write_text("mutation\n", encoding="utf-8")

    with pytest.raises(ValueError, match="inventory"):
        validate_manifest(root)


def test_manifest_rejects_mutated_aggregate_digest(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    _publish(root)
    manifest = validate_manifest(root)
    manifest["artifact_inventory_sha256"] = "0" * 64
    _rewrite_manifest(root, manifest)

    with pytest.raises(ValueError, match="digest"):
        validate_manifest(root)


@pytest.mark.parametrize("mutation", ("schema", "missing-key", "extra-key"))
def test_manifest_requires_exact_top_level_contract(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "artifact"
    _publish(root)
    manifest = validate_manifest(root)
    if mutation == "schema":
        manifest["schema"] = "wrong"
    elif mutation == "missing-key":
        del manifest["schema"]
    else:
        manifest["extra"] = True
    _rewrite_manifest(root, manifest)

    with pytest.raises(ValueError, match="manifest"):
        validate_manifest(root)


def test_validate_manifest_rejects_fifo_before_opening_it(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    os.mkfifo(root / "manifest.json")
    code = r'''
import sys
from pathlib import Path
from trec_mlm.dense_aggregation_artifacts import validate_manifest

try:
    validate_manifest(Path(sys.argv[1]))
except ValueError as error:
    assert "regular file" in str(error), str(error)
    print(error)
else:
    raise AssertionError("FIFO manifest was accepted")
'''
    completed = subprocess.run(
        [sys.executable, "-c", code, str(root)],
        text=True,
        capture_output=True,
        timeout=2,
    )
    assert completed.returncode == 0, completed.stderr
    assert "regular file" in completed.stdout


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    (
        ("size", 1.0),
        ("size", True),
        ("schema", ["trec-mlm-dense-aggregation-manifest-v1"]),
        ("files", {"path": "one.bin"}),
        ("path", 7),
        ("file_sha256", 7),
        ("artifact_inventory_sha256", 7),
    ),
)
def test_manifest_rejects_nonexact_json_field_types(
    tmp_path: Path, field: str, invalid_value: object
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "one.bin").write_bytes(b"x")
    manifest = write_manifest(root)
    rows = manifest["files"]
    assert isinstance(rows, list)
    assert isinstance(rows[0], list)
    if field == "size":
        rows[0][1] = invalid_value
    elif field == "schema":
        manifest["schema"] = invalid_value
    elif field == "files":
        manifest["files"] = invalid_value
    elif field == "path":
        rows[0][0] = invalid_value
    elif field == "file_sha256":
        rows[0][2] = invalid_value
    else:
        manifest["artifact_inventory_sha256"] = invalid_value
    _rewrite_manifest(root, manifest)

    with pytest.raises(ValueError, match="type"):
        validate_manifest(root)


@pytest.mark.parametrize(
    "encoding",
    ("missing-newline", "extra-newline", "pretty", "duplicate-key"),
)
def test_manifest_rejects_noncanonical_bytes(
    tmp_path: Path, encoding: str
) -> None:
    root = tmp_path / "artifact"
    _publish(root)
    manifest = validate_manifest(root)
    canonical = canonical_json(manifest)
    if encoding == "missing-newline":
        payload = canonical
    elif encoding == "extra-newline":
        payload = canonical + b"\n\n"
    elif encoding == "pretty":
        payload = json.dumps(manifest, indent=2, sort_keys=True).encode() + b"\n"
    else:
        schema = b'"trec-mlm-dense-aggregation-manifest-v1"'
        payload = canonical[:-1] + b',"schema":' + schema + b"}\n"
    (root / "manifest.json").write_bytes(payload)

    with pytest.raises(ValueError, match="canonical"):
        validate_manifest(root)


def test_manifest_rejects_noncanonical_unicode_escape_encoding(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "snow-雪.txt").write_text("snow\n", encoding="utf-8")
    manifest = write_manifest(root)
    escaped = (
        json.dumps(
            manifest,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    assert escaped != canonical_json(manifest) + b"\n"
    (root / "manifest.json").write_bytes(escaped)

    with pytest.raises(ValueError, match="canonical"):
        validate_manifest(root)


def test_manifest_rejects_symlink_and_nonregular_entry(tmp_path: Path) -> None:
    symlink_root = tmp_path / "symlink-artifact"
    symlink_root.mkdir()
    (symlink_root / "payload.txt").write_text("payload\n", encoding="utf-8")
    (symlink_root / "link.txt").symlink_to("payload.txt")
    with pytest.raises(ValueError, match="symlink"):
        write_manifest(symlink_root)

    fifo_root = tmp_path / "fifo-artifact"
    fifo_root.mkdir()
    os.mkfifo(fifo_root / "named-pipe")
    with pytest.raises(ValueError, match="regular file"):
        write_manifest(fifo_root)


def test_atomic_publish_rejects_existing_destination(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "sentinel.txt").write_text("theirs\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="destination exists"):
        with atomic_artifact_directory(root):
            pytest.fail("an existing destination must fail before yielding")

    assert (root / "sentinel.txt").read_text(encoding="utf-8") == "theirs\n"
    assert not list(tmp_path.glob(".artifact.*"))


def test_atomic_publish_never_replaces_racing_destination(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    with pytest.raises(OSError):
        with atomic_artifact_directory(root) as staging:
            (staging / "payload.txt").write_text("ours\n", encoding="utf-8")
            root.mkdir()
            (root / "sentinel.txt").write_text("theirs\n", encoding="utf-8")
    assert (root / "sentinel.txt").read_text(encoding="utf-8") == "theirs\n"
    assert not (root / "payload.txt").exists()
    assert not list(tmp_path.glob(".artifact.*"))


def test_atomic_failure_removes_unpublished_staging(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    with pytest.raises(RuntimeError, match="writer failed"):
        with atomic_artifact_directory(root) as staging:
            (staging / "partial.txt").write_text("partial\n", encoding="utf-8")
            raise RuntimeError("writer failed")

    assert not root.exists()
    assert not list(tmp_path.iterdir())


def test_quarantine_is_unique_invalid_and_preserves_racing_destination(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    for attempt in ("first", "second"):
        with pytest.raises(OSError):
            with atomic_artifact_directory(root, quarantine_on_error=True) as staging:
                (staging / "payload.txt").write_text(
                    f"ours-{attempt}\n", encoding="utf-8"
                )
                root.mkdir()
                (root / "sentinel.txt").write_text(
                    f"theirs-{attempt}\n", encoding="utf-8"
                )
        assert (root / "sentinel.txt").read_text(encoding="utf-8") == (
            f"theirs-{attempt}\n"
        )
        shutil.rmtree(root)

    quarantines = sorted(tmp_path.glob("artifact.aborted-*"))
    assert len(quarantines) == 2
    assert quarantines[0] != quarantines[1]
    assert all(
        re.fullmatch(r"artifact\.aborted-[0-9a-f]{32}", root.name)
        for root in quarantines
    )
    for quarantine in quarantines:
        assert (quarantine / "payload.txt").is_file()
        assert (quarantine / "manifest.json").is_file()
        with pytest.raises(ValueError, match="quarantined"):
            validate_manifest(quarantine)
        with pytest.raises(FileExistsError):
            with atomic_artifact_directory(quarantine):
                pytest.fail("a quarantined tree must not be reusable")


def test_semantic_artifact_hash_normalizes_only_explicit_subset(
    tmp_path: Path, golden: dict[str, str]
) -> None:
    root = tmp_path / "artifact"
    _publish(root)

    assert (
        semantic_artifact_sha256(
            root,
            normalized_files={"nested/payload.bin": b"normalized"},
            tag="construction-semantic-artifact",
        )
        == golden["semantic_normalized"]
    )
    assert semantic_artifact_sha256(
        root,
        normalized_files={"nested/payload.bin": b"normalized"},
        tag="different-semantic-artifact",
    ) != golden["semantic_normalized"]


@pytest.mark.parametrize("path", ("manifest.json", "missing.txt", "./payload.txt"))
def test_semantic_artifact_hash_rejects_nonmanifested_normalized_path(
    tmp_path: Path, path: str
) -> None:
    root = tmp_path / "artifact"
    _publish(root)

    with pytest.raises(ValueError, match="subset"):
        semantic_artifact_sha256(
            root,
            normalized_files={path: b"normalized"},
            tag="construction-semantic-artifact",
        )


def test_semantic_artifact_hash_validates_raw_manifest_before_normalizing(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _publish(root)
    (root / "nested" / "payload.bin").write_bytes(b"mutated")

    with pytest.raises(ValueError, match="inventory"):
        semantic_artifact_sha256(
            root,
            normalized_files={"nested/payload.bin": b"\x00\x01\xff"},
            tag="construction-semantic-artifact",
        )


def test_semantic_artifact_hash_rejects_noncanonical_manifest_first(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    _publish(root)
    manifest_path = root / "manifest.json"
    manifest_path.write_bytes(manifest_path.read_bytes().rstrip(b"\n"))

    with pytest.raises(ValueError, match="canonical"):
        semantic_artifact_sha256(
            root,
            normalized_files={},
            tag="construction-semantic-artifact",
        )
