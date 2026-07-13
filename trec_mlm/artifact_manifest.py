"""Deterministic provenance manifests and enforced smoke receipts."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_artifact_manifest(
    artifact_dirs: Sequence[str | Path],
    *,
    code_revision: str,
    worktree_status: str,
) -> dict[str, Any]:
    """Hash every file under explicitly supplied artifact directories."""

    if not code_revision.strip():
        raise ValueError("code revision must be non-empty")
    artifacts: list[dict[str, Any]] = []
    for value in sorted((Path(item) for item in artifact_dirs), key=lambda p: str(p)):
        if not value.is_dir():
            raise ValueError(f"artifact directory does not exist: {value}")
        files = {
            str(path.relative_to(value)): _sha256(path)
            for path in sorted(value.rglob("*"))
            if path.is_file()
        }
        artifacts.append({"directory": str(value), "files": files})
    return {
        "version": 1,
        "code_revision": code_revision,
        "worktree_status": worktree_status,
        "artifacts": artifacts,
    }


def validate_smoke_receipt(
    path: str | Path,
    *,
    model_id: str,
    revision: str,
    dtype: str,
) -> dict[str, Any]:
    """Require a passed smoke receipt for the exact frozen runtime contract."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("passed") is not True:
        raise ValueError("smoke receipt did not pass")
    for field, expected in (
        ("model_id", model_id),
        ("revision", revision),
        ("dtype", dtype),
    ):
        if payload.get(field) != expected:
            raise ValueError(f"smoke receipt {field} does not match")
    return {**payload, "receipt_sha256": _sha256(source)}
