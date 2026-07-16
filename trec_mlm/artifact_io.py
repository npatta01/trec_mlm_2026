"""Deterministic byte and hash primitives for persisted artifacts."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file read in 1 MiB chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_pretty_json(path: Path, payload: Any) -> None:
    """Write the artifact JSON format, including its final newline."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def length_framed_sha256(values: Sequence[str]) -> str:
    """Hash UTF-8 strings framed by unsigned 8-byte big-endian lengths."""

    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def manifest_entries(root: Path) -> dict[str, dict[str, Any]]:
    """Inventory files below root, excluding every file named manifest.json."""

    return {
        path.relative_to(root).as_posix(): {
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def manifest_aggregate_sha256(
    entries: Mapping[str, Mapping[str, Any]],
) -> str:
    """Hash the compact, sorted manifest-entry projection."""

    ordered = [
        [name, int(entries[name]["byte_size"]), str(entries[name]["sha256"])]
        for name in sorted(entries)
    ]
    payload = json.dumps(ordered, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def source_tree_sha256(root: Path) -> str:
    """Hash Python source paths and payloads with 8-byte length framing."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        payload = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()
