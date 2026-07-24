"""Stage audited Research Hub resources after a Quarto render."""

from __future__ import annotations

import argparse
from html.parser import HTMLParser
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import urlsplit


RESOURCE_ALLOWLIST = {
    Path("reports/trec_2025_mllm_interactive_report.html"): Path(
        "library/trec-2025-interactive.html"
    ),
    Path("reports/trec_2026_approaches_guide.html"): Path(
        "library/approaches-guide.html"
    ),
}


_PRIVATE_MARKERS = (
    ("file:///", "file URL"),
    ("-----begin private key-----", "private-key header"),
    ("-----begin rsa private key-----", "private-key header"),
    ("-----begin openssh private key-----", "private-key header"),
    ("-----begin ec private key-----", "private-key header"),
)
_LOCAL_FILESYSTEM_PATH = re.compile(
    r"""
    (?:
        (?<![A-Za-z0-9_./:-])
        /
        (?:
            (?:home|Users)/[^/\s<>"']+
            |
            (?:var/)?tmp
        )
        (?=/|[\s<>"']|$)
        |
        (?<![A-Za-z0-9_/\\])
        [A-Za-z]:[\\/]
        |
        (?<!\\)
        \\\\[^\\/\s<>"']+[\\/][^\\/\s<>"']+
    )
    """,
    flags=re.IGNORECASE | re.VERBOSE,
)
_CSS_EXTERNAL_IMPORT = re.compile(
    r"""@import\s+(?:url\(\s*)?["']?(?:https?:)?//""",
    flags=re.IGNORECASE,
)
_MAX_SRCDOC_DEPTH = 16


def _is_external_url(value: str) -> bool:
    value = value.strip()
    if value.startswith("//"):
        return True
    return urlsplit(value).scheme.lower() in {"http", "https"}


class _ExecutableAssetScanner(HTMLParser):
    """Reject executable or embedding elements that load from the network."""

    def __init__(self, source: Path, depth: int) -> None:
        super().__init__()
        self.source = source
        self.depth = depth

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        attributes = {
            name.lower(): value or "" for name, value in attrs
        }
        tag = tag.lower()
        external = False
        if tag == "script":
            external = _is_external_url(attributes.get("src", ""))
        elif tag == "base":
            if _is_external_url(attributes.get("href", "")):
                raise ValueError(
                    "staged HTML contains an external base URL"
                )
        elif tag == "link":
            rel = {
                value.lower()
                for value in attributes.get("rel", "").split()
            }
            external = bool(
                rel & {"stylesheet", "preload", "modulepreload"}
            ) and _is_external_url(attributes.get("href", ""))
        elif tag in {"iframe", "embed"}:
            external = _is_external_url(attributes.get("src", ""))
            if tag == "iframe" and attributes.get("srcdoc"):
                _scan_text(
                    self.source,
                    attributes["srcdoc"],
                    depth=self.depth + 1,
                )
        elif tag == "object":
            external = _is_external_url(attributes.get("data", ""))
        if external:
            raise ValueError(
                f"staged HTML contains an external executable asset: <{tag}>"
            )


def _scan_text(source: Path, text: str, *, depth: int = 0) -> None:
    if depth > _MAX_SRCDOC_DEPTH:
        raise ValueError("staged HTML contains excessively nested iframe srcdoc")
    lowered = text.lower()
    for marker, label in _PRIVATE_MARKERS:
        if marker in lowered:
            raise ValueError(f"staged HTML contains a {label}")
    slash_normalized = text.replace("\\/", "/")
    path_candidates = (
        slash_normalized,
        slash_normalized.replace("\\\\", "\\"),
    )
    if any(_LOCAL_FILESYSTEM_PATH.search(value) for value in path_candidates):
        raise ValueError("staged HTML contains a local filesystem path")
    if _CSS_EXTERNAL_IMPORT.search(text):
        raise ValueError(
            "staged HTML contains an external executable asset"
        )
    scanner = _ExecutableAssetScanner(source, depth)
    scanner.feed(text)
    scanner.close()


def _scan_html(source: Path, content: bytes) -> None:
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: staged HTML must be UTF-8") from exc
    try:
        _scan_text(source, text)
    except ValueError as exc:
        raise ValueError(f"{source}: {exc}") from exc


def _resolved_within(root: Path, relative_path: Path, *, strict: bool) -> Path:
    candidate = (root / relative_path).resolve(strict=strict)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path escapes resolved root: {relative_path}") from exc
    return candidate


def _reject_symlink_components(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"path must remain within output root: {path}") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"output path contains a symlink: {current}")


def _atomic_write(destination: Path, content: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def stage_resources(root: Path, output_dir: Path) -> tuple[Path, ...]:
    """Publish only audited, self-contained HTML into the rendered hub."""

    root = Path(root).resolve(strict=True)
    expected_output = root / "research/hub/_site"
    output_dir = Path(os.path.abspath(output_dir))
    if output_dir != expected_output:
        raise ValueError(
            "output_dir must resolve exactly to research/hub/_site "
            f"under root: {expected_output}"
        )
    _reject_symlink_components(root, output_dir)
    if output_dir.resolve(strict=False) != expected_output:
        raise ValueError(
            "output_dir must resolve exactly to research/hub/_site "
            f"under root: {expected_output}"
        )

    prepared: list[tuple[Path, bytes]] = []
    for source_relative, destination_relative in RESOURCE_ALLOWLIST.items():
        source = _resolved_within(root, source_relative, strict=True)
        if not source.is_file():
            raise ValueError(f"allowlisted resource is not a file: {source_relative}")
        destination = output_dir / destination_relative
        _reject_symlink_components(output_dir, destination)
        _resolved_within(output_dir, destination_relative, strict=False)
        content = source.read_bytes()
        _scan_html(source_relative, content)
        prepared.append((destination, content))

    for destination, content in prepared:
        _reject_symlink_components(output_dir, destination)
        _atomic_write(destination, content)
    return tuple(sorted(destination for destination, _ in prepared))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage audited Research Hub HTML resources."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research/hub/_site"),
    )
    args = parser.parse_args()
    stage_resources(args.root, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
