"""Normalize local index links for directory-oriented static hosting."""

from __future__ import annotations

import argparse
from html import unescape
import os
import re
import tempfile
from pathlib import Path
from urllib.parse import urlsplit

_INDEX_HREF = re.compile(
    r"""(?P<prefix>\bhref\s*=\s*(?P<quote>["']))
        (?P<path>
            (?!(?:[a-z][a-z0-9+.-]*:)?//)
            (?![a-z][a-z0-9+.-]*:)
            (?:[^"'?#]*/)*)
        index\.html
        (?P<suffix>(?:[?#][^"']*)?)
        (?P=quote)""",
    re.IGNORECASE | re.VERBOSE,
)


def _assert_output_dir(root: Path, output_dir: Path) -> Path:
    root = root.resolve()
    expected = root / "research" / "hub" / "_site"
    if output_dir.absolute() != expected.absolute():
        raise ValueError(f"output directory must be {expected}")
    current = root
    for part in Path("research/hub/_site").parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f"output path component may not be a symlink: {current}")
    return expected


def _normalize_html(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        raw_value = (
            f"{match.group('path')}index.html{match.group('suffix')}"
        )
        decoded_value = unescape(raw_value).strip()
        if "\\" in decoded_value:
            return match.group(0)
        try:
            decoded = urlsplit(decoded_value)
        except ValueError:
            return match.group(0)
        if decoded.scheme or decoded.netloc:
            return match.group(0)
        path = match.group("path") or "./"
        return (
            f"{match.group('prefix')}{path}{match.group('suffix')}"
            f"{match.group('quote')}"
        )

    return _INDEX_HREF.sub(replace, text)


def normalize_links(root: Path, output_dir: Path) -> tuple[Path, ...]:
    """Rewrite local ``index.html`` hrefs without changing external URLs."""

    output_dir = _assert_output_dir(root, output_dir)
    changed: list[Path] = []
    for path in sorted(output_dir.rglob("*.html")):
        if path.is_symlink():
            raise ValueError(f"HTML target may not be a symlink: {path}")
        original = path.read_text(encoding="utf-8")
        normalized = _normalize_html(original)
        if normalized == original:
            continue
        fd, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(normalized)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        changed.append(path)
    return tuple(changed)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize local Research Hub index links."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    normalize_links(args.root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
