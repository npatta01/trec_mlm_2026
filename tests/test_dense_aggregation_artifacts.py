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

from trec_mlm import dense_aggregation_artifacts as artifacts
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


def test_preflight_atomic_roots_creates_only_distinct_same_device_parents(
    tmp_path: Path,
) -> None:
    cache_dir = tmp_path / "cache-parent" / "fit-cache"
    output_dir = tmp_path / "output-parent" / "artifact"

    artifacts.preflight_atomic_roots(
        cache_dir=cache_dir,
        output_dir=output_dir,
    )

    assert cache_dir.parent.is_dir()
    assert output_dir.parent.is_dir()
    assert not cache_dir.exists()
    assert not output_dir.exists()
    assert not list(cache_dir.parent.iterdir())
    assert not list(output_dir.parent.iterdir())


@pytest.mark.parametrize(
    "relation",
    ("resolved-equality", "cache-ancestor", "output-ancestor"),
)
def test_preflight_atomic_roots_rejects_overlap_before_mkdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relation: str
) -> None:
    untouched_parent = tmp_path / "untouched"
    if relation == "resolved-equality":
        cache_dir = untouched_parent / "branch" / ".." / "root"
        output_dir = untouched_parent / "root"
    elif relation == "cache-ancestor":
        cache_dir = untouched_parent / "cache"
        output_dir = cache_dir / "nested" / "output"
    else:
        output_dir = untouched_parent / "output"
        cache_dir = output_dir / "nested" / "cache"
    mkdir_calls: list[Path] = []

    def unexpected_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        del args, kwargs
        mkdir_calls.append(path)
        raise AssertionError("overlap validation reached mkdir")

    monkeypatch.setattr(Path, "mkdir", unexpected_mkdir)

    with pytest.raises(ValueError, match="overlap"):
        artifacts.preflight_atomic_roots(
            cache_dir=cache_dir,
            output_dir=output_dir,
        )

    assert mkdir_calls == []
    assert not untouched_parent.exists()


@pytest.mark.parametrize("existing", ("cache", "output"))
def test_preflight_atomic_roots_rejects_existing_destination_without_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    existing_dir = tmp_path / f"existing-{existing}"
    existing_dir.mkdir()
    sentinel = existing_dir / "sentinel.bin"
    sentinel.write_bytes(b"authoritative\x00bytes")
    untouched_parent = tmp_path / "untouched"
    if existing == "cache":
        cache_dir = existing_dir
        output_dir = untouched_parent / "output"
    else:
        cache_dir = untouched_parent / "cache"
        output_dir = existing_dir
    mkdir_calls: list[Path] = []

    def unexpected_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        del args, kwargs
        mkdir_calls.append(path)
        raise AssertionError("existing-destination validation reached mkdir")

    monkeypatch.setattr(Path, "mkdir", unexpected_mkdir)

    with pytest.raises(FileExistsError, match=f"{existing} destination exists"):
        artifacts.preflight_atomic_roots(
            cache_dir=cache_dir,
            output_dir=output_dir,
        )

    assert mkdir_calls == []
    assert list(existing_dir.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"authoritative\x00bytes"
    assert not untouched_parent.exists()


@pytest.mark.parametrize("existing", ("cache", "output"))
def test_preflight_atomic_roots_rejects_dangling_destination_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    missing_target = tmp_path / f"missing-{existing}-target"
    dangling = tmp_path / f"dangling-{existing}"
    dangling.symlink_to(missing_target, target_is_directory=True)
    untouched_parent = tmp_path / "untouched"
    if existing == "cache":
        cache_dir = dangling
        output_dir = untouched_parent / "output"
    else:
        cache_dir = untouched_parent / "cache"
        output_dir = dangling
    mkdir_calls: list[Path] = []

    def unexpected_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        del args, kwargs
        mkdir_calls.append(path)
        raise AssertionError("dangling-destination validation reached mkdir")

    monkeypatch.setattr(Path, "mkdir", unexpected_mkdir)

    with pytest.raises(FileExistsError, match=f"{existing} destination exists"):
        artifacts.preflight_atomic_roots(
            cache_dir=cache_dir,
            output_dir=output_dir,
        )

    assert mkdir_calls == []
    assert dangling.is_symlink()
    assert dangling.readlink() == missing_target
    assert not missing_target.exists()
    assert not untouched_parent.exists()


@pytest.mark.parametrize("existing", ("cache", "output"))
def test_preflight_atomic_roots_rejects_existing_resolved_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: str
) -> None:
    existing_dir = tmp_path / f"existing-{existing}"
    existing_dir.mkdir()
    sentinel = existing_dir / "sentinel.bin"
    sentinel.write_bytes(b"resolved-authoritative\x00bytes")
    missing_component = tmp_path / f"missing-{existing}-component"
    caller_spelling = missing_component / ".." / existing_dir.name
    untouched_parent = tmp_path / "untouched"
    if existing == "cache":
        cache_dir = caller_spelling
        output_dir = untouched_parent / "output"
    else:
        cache_dir = untouched_parent / "cache"
        output_dir = caller_spelling
    mkdir_calls: list[Path] = []

    def unexpected_mkdir(path: Path, *args: object, **kwargs: object) -> None:
        del args, kwargs
        mkdir_calls.append(path)
        raise AssertionError("resolved-destination validation reached mkdir")

    monkeypatch.setattr(Path, "mkdir", unexpected_mkdir)

    with pytest.raises(FileExistsError, match=f"{existing} destination exists"):
        artifacts.preflight_atomic_roots(
            cache_dir=cache_dir,
            output_dir=output_dir,
        )

    assert mkdir_calls == []
    assert list(existing_dir.iterdir()) == [sentinel]
    assert sentinel.read_bytes() == b"resolved-authoritative\x00bytes"
    assert not missing_component.exists()
    assert not untouched_parent.exists()


def test_preflight_atomic_roots_rejects_different_parent_devices_before_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_dir = tmp_path / "cache-parent" / "fit-cache"
    output_dir = tmp_path / "output-parent" / "artifact"
    qrel_path = tmp_path / "secret-qrels.txt"
    qrel_path.write_bytes(b"must remain unopened\n")
    original_stat = Path.stat

    def simulated_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        result = original_stat(path, *args, **kwargs)
        if path == output_dir.parent.resolve():
            values = list(result)
            values[2] = result.st_dev + 1
            return os.stat_result(values)
        return result

    def unexpected_open(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("preflight opened a file or qrel")

    monkeypatch.setattr(Path, "stat", simulated_stat)
    with monkeypatch.context() as no_open:
        no_open.setattr("builtins.open", unexpected_open)
        no_open.setattr(Path, "open", unexpected_open)
        no_open.setattr(os, "open", unexpected_open)
        no_open.setattr(artifacts, "raw_file_sha256", unexpected_open)
        with pytest.raises(OSError) as error:
            artifacts.preflight_atomic_roots(
                cache_dir=cache_dir,
                output_dir=output_dir,
            )

    assert error.value.args == ("cache and output must share one filesystem",)
    assert str(error.value) == "cache and output must share one filesystem"
    assert cache_dir.parent.is_dir()
    assert output_dir.parent.is_dir()
    assert not cache_dir.exists()
    assert not output_dir.exists()
    assert not list(cache_dir.parent.iterdir())
    assert not list(output_dir.parent.iterdir())
    assert qrel_path.read_bytes() == b"must remain unopened\n"


def test_raw_file_and_manifest_literal_golden_vectors(
    tmp_path: Path, golden: dict[str, str]
) -> None:
    root = tmp_path / "artifact"
    _publish(root)

    assert raw_file_sha256(root / "payload.txt") == golden["payload_text_raw"]
    assert (
        raw_file_sha256(root / "nested" / "payload.bin") == golden["nested_payload_raw"]
    )
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
    code = r"""
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
"""
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
def test_manifest_rejects_noncanonical_bytes(tmp_path: Path, encoding: str) -> None:
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


def test_atomic_publish_rejects_dangling_destination_before_yield(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.symlink_to(tmp_path / "missing-target", target_is_directory=True)
    yielded = False

    with pytest.raises(FileExistsError, match="destination exists"):
        with atomic_artifact_directory(root):
            yielded = True

    assert yielded is False
    assert root.is_symlink()
    assert root.readlink() == tmp_path / "missing-target"
    assert not list(tmp_path.glob(".artifact.*"))


def test_atomic_post_publish_validator_runs_inside_transaction(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    observed: list[Path] = []

    def validate_published(path: Path) -> None:
        observed.append(path)
        assert path == root
        validate_manifest(path)

    with atomic_artifact_directory(
        root,
        post_publish_validator=validate_published,
    ) as staging:
        (staging / "payload.txt").write_text("owned\n", encoding="utf-8")

    assert observed == [root]
    assert validate_manifest(root)["files"]


def test_atomic_post_publish_failure_quarantines_owned_final(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"

    def reject_published(path: Path) -> None:
        validate_manifest(path)
        raise RuntimeError("strict reload failed")

    with pytest.raises(RuntimeError, match="strict reload failed"):
        with atomic_artifact_directory(
            root,
            quarantine_on_error=True,
            post_publish_validator=reject_published,
        ) as staging:
            (staging / "payload.txt").write_text("owned\n", encoding="utf-8")

    assert not root.exists()
    quarantines = list(tmp_path.glob("artifact.aborted-*"))
    assert len(quarantines) == 1
    assert (quarantines[0] / "payload.txt").read_text(encoding="utf-8") == "owned\n"


def test_atomic_post_publish_failure_never_moves_replacement_inode(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    displaced = tmp_path / "displaced-owned-root"

    def replace_then_reject(path: Path) -> None:
        path.rename(displaced)
        path.mkdir()
        (path / "sentinel.txt").write_text("foreign\n", encoding="utf-8")
        raise RuntimeError("strict reload failed after replacement")

    with pytest.raises(RuntimeError, match="strict reload failed after replacement"):
        with atomic_artifact_directory(
            root,
            quarantine_on_error=True,
            post_publish_validator=replace_then_reject,
        ) as staging:
            (staging / "payload.txt").write_text("owned\n", encoding="utf-8")

    assert (root / "sentinel.txt").read_text(encoding="utf-8") == "foreign\n"
    assert (displaced / "payload.txt").read_text(encoding="utf-8") == "owned\n"
    assert not list(tmp_path.glob("artifact.aborted-*"))


def test_atomic_quarantine_detects_and_restores_swap_inside_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    displaced_owned = tmp_path / "displaced-owned"
    original_rename = artifacts.rename_noreplace
    calls = 0
    swapped_path: Path | None = None

    def racing_rename(source: str | Path, destination: str | Path) -> None:
        nonlocal calls, swapped_path
        calls += 1
        source_path = Path(source)
        if calls == 2:
            swapped_path = source_path
            source_path.rename(displaced_owned)
            source_path.mkdir()
            (source_path / "sentinel.txt").write_text("foreign\n", encoding="utf-8")
        original_rename(source_path, destination)

    monkeypatch.setattr(artifacts, "rename_noreplace", racing_rename)
    with pytest.raises(OSError):
        with atomic_artifact_directory(root, quarantine_on_error=True) as staging:
            (staging / "payload.txt").write_text("owned\n", encoding="utf-8")
            root.mkdir()
            (root / "sentinel.txt").write_text("racing-final\n", encoding="utf-8")

    assert swapped_path is not None
    assert (swapped_path / "sentinel.txt").read_text(encoding="utf-8") == "foreign\n"
    assert (displaced_owned / "payload.txt").read_text(encoding="utf-8") == "owned\n"
    assert (root / "sentinel.txt").read_text(encoding="utf-8") == "racing-final\n"
    assert not list(tmp_path.glob("artifact.aborted-*"))


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


def test_manifest_rejects_oversized_file_before_unbounded_read(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    manifest = root / "manifest.json"
    with manifest.open("wb") as handle:
        handle.truncate(16 * 1024 * 1024 + 1)

    with pytest.raises(ValueError, match="byte limit"):
        validate_manifest(root)


def test_write_manifest_checks_byte_limit_before_creating_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "payload.txt").write_text("payload\n", encoding="utf-8")
    monkeypatch.setattr(artifacts, "_MANIFEST_BYTE_LIMIT", 16)

    with pytest.raises(ValueError, match="byte limit"):
        write_manifest(root)

    assert not (root / "manifest.json").exists()


@pytest.mark.parametrize("target", ("manifest", "payload"))
def test_artifact_readers_do_not_block_on_regular_to_fifo_open_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    root = tmp_path / "artifact"
    _publish(root)
    path = root / ("manifest.json" if target == "manifest" else "payload.txt")
    original_open = os.open
    replaced = False

    def replace_with_fifo(
        requested: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        requested_path = Path(requested)
        if not replaced and requested_path == path:
            replaced = True
            path.unlink()
            os.mkfifo(path)
        return original_open(requested, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(os, "open", replace_with_fifo)
    with pytest.raises(ValueError, match="changed|invalid|regular"):
        validate_manifest(root)

    assert replaced is True


@pytest.mark.parametrize(
    ("field", "invalid_value", "message"),
    (
        ("size", -1, "size"),
        ("path", "../escape", "path"),
        ("path", "/absolute", "path"),
        ("path", "nested//payload.bin", "path"),
        ("file_sha256", "A" * 64, "digest"),
        ("file_sha256", "0" * 63, "digest"),
        ("artifact_inventory_sha256", "F" * 64, "digest"),
    ),
)
def test_manifest_rejects_invalid_canonical_row_values(
    tmp_path: Path,
    field: str,
    invalid_value: object,
    message: str,
) -> None:
    root = tmp_path / "artifact"
    _publish(root)
    manifest = validate_manifest(root)
    rows = manifest["files"]
    assert isinstance(rows, list)
    assert isinstance(rows[0], list)
    if field == "size":
        rows[0][1] = invalid_value
    elif field == "path":
        rows[0][0] = invalid_value
    elif field == "file_sha256":
        rows[0][2] = invalid_value
    else:
        manifest["artifact_inventory_sha256"] = invalid_value
    _rewrite_manifest(root, manifest)

    with pytest.raises(ValueError, match=message):
        validate_manifest(root)


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
    assert (
        semantic_artifact_sha256(
            root,
            normalized_files={"nested/payload.bin": b"normalized"},
            tag="different-semantic-artifact",
        )
        != golden["semantic_normalized"]
    )


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
