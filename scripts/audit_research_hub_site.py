"""Reject reader-facing source records in the rendered Research Hub."""

from __future__ import annotations

import argparse
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import sys
from urllib.parse import unquote, urlsplit


class _LinkCollector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hrefs: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag.lower() != "a":
            return
        for name, value in attrs:
            if name.lower() == "href" and value is not None:
                self.hrefs.append(value)


def _assert_output_dir(root: Path, output_dir: Path) -> Path:
    root = root.resolve()
    expected = root / "research" / "hub" / "_site"
    if output_dir.absolute() != expected.absolute():
        raise ValueError(f"output directory must be {expected}")
    current = root
    for part in Path("research/hub/_site").parts:
        current /= part
        if current.is_symlink():
            raise ValueError(
                f"output path component may not be a symlink: {current}"
            )
    return expected


def _local_qmd_target(raw_href: str) -> str | None:
    decoded = unquote(unescape(raw_href)).strip().replace("\\", "/")
    try:
        target = urlsplit(decoded)
    except ValueError:
        return None
    if target.scheme or target.netloc:
        return None
    return target.path if target.path.lower().endswith(".qmd") else None


def audit_site(root: Path, output_dir: Path) -> tuple[str, ...]:
    """Return deterministic publication-boundary violations."""

    output_dir = _assert_output_dir(root, output_dir)
    problems: list[str] = []
    for path in sorted(output_dir.rglob("*")):
        relative = path.relative_to(output_dir).as_posix()
        if path.is_symlink():
            problems.append(f"published symlink: {relative}")
            continue
        if path.is_file() and path.suffix.lower() == ".qmd":
            problems.append(f"unexpected published .qmd source: {relative}")
        if not path.is_file() or path.suffix.lower() != ".html":
            continue
        parser = _LinkCollector()
        parser.feed(path.read_text(encoding="utf-8"))
        for href in parser.hrefs:
            target = _local_qmd_target(href)
            if target is not None:
                problems.append(
                    f"reader-facing .qmd target in {relative}: {target}"
                )
    return tuple(sorted(problems))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit the rendered Research Hub publication boundary."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    problems = audit_site(args.root, args.output_dir)
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print("Research Hub site audit passed: no .qmd sources or reader targets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
