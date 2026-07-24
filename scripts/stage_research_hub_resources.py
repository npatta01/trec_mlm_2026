"""Stage audited Research Hub resources after a Quarto render."""

from __future__ import annotations

import argparse
import base64
import binascii
import gzip
from html import unescape
from html.parser import HTMLParser
import io
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
    ("-----begin private key-----", "private-key header"),
    ("-----begin rsa private key-----", "private-key header"),
    ("-----begin openssh private key-----", "private-key header"),
    ("-----begin ec private key-----", "private-key header"),
)
_FILE_URL = re.compile(
    r"(?<![A-Za-z0-9+.-])file:[/\\]",
    flags=re.IGNORECASE,
)
_WEB_URL = re.compile(
    r"(?:https?:)?//[^\s<>'\"]+",
    flags=re.IGNORECASE,
)
_COMPRESSED_TEMPLATE = re.compile(
    r"""
    <template\b
    (?=[^>]*\bdata-compression\s*=\s*["']gzip-base64["'])
    [^>]*>(?P<payload>.*?)</template\s*>
    """,
    flags=re.IGNORECASE | re.DOTALL | re.VERBOSE,
)
_SCRIPT_ELEMENT = re.compile(
    r"(?P<open><script\b[^>]*>)(?P<body>.*?)(?P<close></script\s*>)",
    flags=re.IGNORECASE | re.DOTALL,
)
_JAVASCRIPT_QUOTED_ABSOLUTE_PATH = re.compile(
    r"""
    ["'`]
    \s*
    (?P<path>(?:\\/|/)[A-Za-z0-9._-]{2,}[^"'`\r\n]*)
    """,
    flags=re.VERBOSE,
)
_CSS_COMMENT = re.compile(r"/\*.*?\*/", flags=re.DOTALL)
_POSIX_ABSOLUTE_PATH = re.compile(
    r"""
    (?<![A-Za-z0-9_./:<}])
    /
    (?P<root>[^/\\\s<>"'\?#]+)
    (?:/[^/\\\s<>"'\?#]+)*
    """,
    flags=re.VERBOSE,
)
_PERMITTED_SITE_PATH_ROOTS = frozenset({"api", "research", "static"})
_WINDOWS_LOCAL_FILESYSTEM_PATH = re.compile(
    r"""
    (?:
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
_MAX_EMBEDDED_DEPTH = 16
_MAX_COMPRESSED_TEMPLATE_BYTES = 2 * 1024 * 1024
_MAX_DECOMPRESSED_TEMPLATE_BYTES = 4 * 1024 * 1024
_MAX_COMPRESSED_TEMPLATE_BASE64_CHARS = (
    (_MAX_COMPRESSED_TEMPLATE_BYTES + 2) // 3
) * 4


def _contains_local_filesystem_path(value: str) -> bool:
    without_web_urls = _WEB_URL.sub("", value)
    if any(
        match.group("root") not in _PERMITTED_SITE_PATH_ROOTS
        for match in _POSIX_ABSOLUTE_PATH.finditer(without_web_urls)
    ):
        return True
    return bool(_WINDOWS_LOCAL_FILESYSTEM_PATH.search(value))


def _script_strings_for_path_scan(match: re.Match[str]) -> str:
    for candidate in _JAVASCRIPT_QUOTED_ABSOLUTE_PATH.finditer(
        match.group("body")
    ):
        value = candidate.group("path").replace("\\/", "/")
        if _contains_local_filesystem_path(value):
            raise ValueError("staged HTML contains a local filesystem path")
    return match.group("open") + match.group("close")


def _scan_compressed_template(
    source: Path,
    match: re.Match[str],
    *,
    depth: int,
) -> str:
    compact_payload = "".join(match.group("payload").split())
    if len(compact_payload) > _MAX_COMPRESSED_TEMPLATE_BASE64_CHARS:
        raise ValueError(
            "staged HTML contains a compressed gzip-base64 template "
            "that exceeds the size limit"
        )
    try:
        compressed = base64.b64decode(compact_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(
            "staged HTML contains an invalid gzip-base64 template payload"
        ) from exc
    if len(compressed) > _MAX_COMPRESSED_TEMPLATE_BYTES:
        raise ValueError(
            "staged HTML contains a compressed gzip-base64 template "
            "that exceeds the size limit"
        )
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(compressed)) as stream:
            content = stream.read(_MAX_DECOMPRESSED_TEMPLATE_BYTES + 1)
    except (EOFError, gzip.BadGzipFile, OSError) as exc:
        raise ValueError(
            "staged HTML contains an invalid gzip template payload"
        ) from exc
    if len(content) > _MAX_DECOMPRESSED_TEMPLATE_BYTES:
        raise ValueError(
            "staged HTML contains a decompressed gzip-base64 template "
            "that exceeds the size limit"
        )
    try:
        decoded = content.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            "staged HTML gzip-base64 template content must be UTF-8"
        ) from exc
    _scan_text(source, decoded, depth=depth + 1)
    return ""


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
    if depth > _MAX_EMBEDDED_DEPTH:
        raise ValueError(
            "staged HTML contains excessively nested embedded content"
        )
    decoded = unescape(text)
    lowered = decoded.lower()
    for marker, label in _PRIVATE_MARKERS:
        if marker in lowered:
            raise ValueError(f"staged HTML contains a {label}")
    if _FILE_URL.search(decoded):
        raise ValueError("staged HTML contains a file URL")
    path_text = _COMPRESSED_TEMPLATE.sub(
        lambda match: _scan_compressed_template(
            source,
            match,
            depth=depth,
        ),
        decoded,
    )
    path_text = _SCRIPT_ELEMENT.sub(_script_strings_for_path_scan, path_text)
    path_text = _CSS_COMMENT.sub("", path_text)
    slash_normalized = path_text.replace("\\/", "/")
    path_candidates = (
        slash_normalized,
        slash_normalized.replace("\\\\", "\\"),
    )
    if any(_contains_local_filesystem_path(value) for value in path_candidates):
        raise ValueError("staged HTML contains a local filesystem path")
    if _CSS_EXTERNAL_IMPORT.search(decoded):
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
