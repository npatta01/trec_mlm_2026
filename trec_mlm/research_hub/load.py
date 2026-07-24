from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .model import HubRecord, HubSources, RECORD_KINDS

DIRECTORIES = {
    "theme": "themes",
    "question": "questions",
    "study": "studies",
    "test": "ideas",
    "approach": "approaches",
    "dataset": "data",
    "background": "background",
}
ID_FIELDS = {
    "theme": "theme_id",
    "question": "question_id",
    "study": "study_id",
    "test": "test_id",
    "approach": "approach_id",
    "dataset": "dataset_id",
    "background": "resource_id",
}


def _relative_source(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"hub source escapes repository root: {path}") from exc


def parse_qmd_record(root: Path, path: Path, kind: str) -> HubRecord:
    if kind not in RECORD_KINDS:
        raise ValueError(f"unknown hub record kind: {kind}")
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"{path}: QMD record must start with YAML front matter")
    try:
        closing = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"{path}: unterminated YAML front matter") from exc
    try:
        metadata: Any = yaml.safe_load("\n".join(lines[1:closing]))
    except yaml.YAMLError as exc:
        raise ValueError(f"{path}: invalid YAML front matter") from exc
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: YAML front matter must be an object")
    id_field = ID_FIELDS[kind]
    record_id = metadata.get(id_field)
    title = metadata.get("title")
    if not isinstance(record_id, str) or not record_id.strip():
        raise ValueError(f"{path}: {id_field} must be a nonempty string")
    if not isinstance(title, str) or not title.strip():
        raise ValueError(f"{path}: title must be a nonempty string")
    body = "\n".join(lines[closing + 1 :]).strip()
    return HubRecord(
        kind=kind,
        record_id=record_id,
        title=title,
        source_path=_relative_source(root, path),
        metadata=metadata,
        body=body,
    )


def load_hub_sources(
    root: Path,
    source_dir: Path = Path("research/hub"),
) -> HubSources:
    grouped: dict[str, tuple[HubRecord, ...]] = {}
    for kind, directory in DIRECTORIES.items():
        paths = (
            path
            for path in (root / source_dir / directory).glob("*.qmd")
            if path.name != "index.qmd"
        )
        records = [parse_qmd_record(root, path, kind) for path in paths]
        grouped[kind] = tuple(sorted(records, key=lambda row: row.record_id))
    return HubSources(
        themes=grouped["theme"],
        questions=grouped["question"],
        studies=grouped["study"],
        tests=grouped["test"],
        approaches=grouped["approach"],
        datasets=grouped["dataset"],
        background=grouped["background"],
    )
