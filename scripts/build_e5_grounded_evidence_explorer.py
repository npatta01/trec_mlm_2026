#!/usr/bin/env python3
"""Build the validated private E5 grounded-evidence explorer."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = str(REPO_ROOT)
if SOURCE_ROOT not in sys.path:
    sys.path.insert(0, SOURCE_ROOT)

from trec_mlm.grounded_evidence_explorer import (  # noqa: E402
    EvidenceBundleError,
    build_evidence_bundle,
)


GENERIC_BUILDER = (
    REPO_ROOT
    / ".agents/skills/inspect-query-results/scripts/build_query_explorer.py"
)


class E5ExplorerBuildError(RuntimeError):
    """The private explorer could not be safely published."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refinement", required=True, type=Path)
    parser.add_argument("--construction", required=True, type=Path)
    parser.add_argument("--evaluation", required=True, type=Path)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--qrels", required=True, type=Path)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--bundle-output", required=True, type=Path)
    parser.add_argument("--html-output", required=True, type=Path)
    return parser.parse_args()


def write_json_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write one complete JSON document with a sibling atomic replacement."""

    temporary: Path | None = None
    descriptor: int | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
        )
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            json.dump(
                payload,
                handle,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except OSError as error:
        raise E5ExplorerBuildError("private bundle write failed") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _temporary_html_path(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}.building.",
        suffix=".html",
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    return temporary


def run_generic_builder(
    *,
    input_path: Path,
    queries: Path,
    qrels: Path,
    profiles: Path,
    evidence_bundle: Path,
    output: Path,
) -> None:
    """Render through the generic grounded explorer without exposing its payload."""

    completed = subprocess.run(
        [
            sys.executable,
            str(GENERIC_BUILDER),
            "--repo-root",
            str(REPO_ROOT),
            "--input",
            str(input_path),
            "--queries",
            str(queries),
            "--qrels",
            str(qrels),
            "--profiles",
            str(profiles),
            "--evidence-bundle",
            str(evidence_bundle),
            "--output",
            str(output),
            "--top-candidates",
            "20",
            "--no-responses",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise E5ExplorerBuildError("grounded explorer rendering failed")
    if not output.is_file() or output.stat().st_size == 0:
        raise E5ExplorerBuildError("grounded explorer output validation failed")


def _validate_html_destination(path: Path) -> None:
    if path.suffix.lower() != ".html":
        raise E5ExplorerBuildError("HTML output path must end in .html")
    if path.exists() and not path.is_file():
        raise E5ExplorerBuildError("HTML output path is not a regular file")


def _source_input_paths(
    *,
    refinement: Path,
    construction: Path,
    evaluation: Path,
    queries: Path,
    qrels: Path,
    profiles: Path,
) -> tuple[Path, ...]:
    return (
        queries,
        qrels,
        profiles,
        evaluation,
        refinement / "runs" / "raw_hierarchical_dense_equal.trec",
        refinement / "inputs.json",
        refinement / "rankings" / "dense_reference.json",
        refinement / "rankings" / "raw_hierarchical.json",
        (
            refinement
            / "rankings"
            / "raw_hierarchical_dense_equal.json"
        ),
        refinement / "selected_record_ids.json",
        refinement / "top_record_similarities.npy",
        construction / "records.json",
        construction / "rankings" / "profile_only.json",
        construction / "rankings" / "dense_profile_equal.json",
    )


def _same_file(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return left.exists() and right.exists() and left.samefile(right)
    except OSError:
        return False


def _validate_unresolved_output_paths(
    bundle: Path, html: Path
) -> None:
    if bundle.is_symlink():
        raise E5ExplorerBuildError(
            "bundle output path must not be a symlink"
        )
    if html.is_symlink():
        raise E5ExplorerBuildError("HTML output path must not be a symlink")


def _validate_output_aliases(
    *,
    bundle: Path,
    html: Path,
    sources: tuple[Path, ...],
) -> None:
    if _same_file(bundle, html):
        raise E5ExplorerBuildError(
            "bundle and HTML output paths must be different"
        )
    if any(
        _same_file(output, source)
        for output in (bundle, html)
        for source in sources
    ):
        raise E5ExplorerBuildError("output path aliases a source input")


def main() -> int:
    args = parse_args()
    requested_bundle = args.bundle_output
    requested_html = args.html_output
    _validate_unresolved_output_paths(
        requested_bundle, requested_html
    )

    refinement = args.refinement.resolve()
    construction = args.construction.resolve()
    evaluation = args.evaluation.resolve()
    queries = args.queries.resolve()
    qrels = args.qrels.resolve()
    profiles = args.profiles.resolve()
    bundle_output = requested_bundle.resolve()
    html_output = requested_html.resolve()
    sources = _source_input_paths(
        refinement=refinement,
        construction=construction,
        evaluation=evaluation,
        queries=queries,
        qrels=qrels,
        profiles=profiles,
    )
    _validate_output_aliases(
        bundle=bundle_output,
        html=html_output,
        sources=sources,
    )
    _validate_html_destination(html_output)

    bundle = build_evidence_bundle(
        refinement_dir=refinement,
        construction_dir=construction,
        evaluation_path=evaluation,
        queries_path=queries,
        qrels_path=qrels,
    )
    write_json_atomically(bundle_output, bundle)

    temporary_html: Path | None = None
    try:
        temporary_html = _temporary_html_path(html_output)
        run_generic_builder(
            input_path=(
                refinement
                / "runs"
                / "raw_hierarchical_dense_equal.trec"
            ),
            queries=queries,
            qrels=qrels,
            profiles=profiles,
            evidence_bundle=bundle_output,
            output=temporary_html,
        )
        temporary_html.replace(html_output)
    except OSError as error:
        raise E5ExplorerBuildError("HTML publication failed") from error
    finally:
        if temporary_html is not None:
            temporary_html.unlink(missing_ok=True)

    print(
        json.dumps(
            {
                "bundle": str(bundle_output),
                "evidence_per_candidate": bundle["meta"][
                    "evidence_per_candidate"
                ],
                "html": str(html_output),
                "query_count": bundle["meta"]["query_count"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceBundleError:
        print("error: evidence bundle validation failed", file=sys.stderr)
        raise SystemExit(2) from None
    except E5ExplorerBuildError as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from None
    except Exception:
        print("error: E5 grounded explorer build failed", file=sys.stderr)
        raise SystemExit(2) from None
