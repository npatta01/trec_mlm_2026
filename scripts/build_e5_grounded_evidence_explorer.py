#!/usr/bin/env python3
"""Build the validated private E5 grounded-evidence explorer."""

from __future__ import annotations

import argparse
from collections.abc import Collection
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


def _temporary_output_path(output: Path, *, suffix: str) -> Path:
    """Reserve a unique sibling staging path without leaving an empty file."""

    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.building.",
        suffix=suffix,
        dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    return temporary


def _backup_existing_output(path: Path) -> Path | None:
    if not path.exists():
        return None
    descriptor, backup_name = tempfile.mkstemp(
        prefix=f".{path.name}.rollback.",
        suffix=path.suffix,
        dir=path.parent,
    )
    os.close(descriptor)
    backup = Path(backup_name)
    backup.unlink()
    try:
        os.link(path, backup)
    except OSError:
        backup.unlink(missing_ok=True)
        raise
    return backup


def _promote_output_pair(
    *,
    staged_bundle: Path,
    bundle: Path,
    staged_html: Path,
    html: Path,
) -> tuple[Path, ...]:
    """Replace both requested outputs only after both staged files pass."""

    pairs = (
        (staged_bundle, bundle),
        (staged_html, html),
    )
    backups: dict[Path, Path | None] = {}
    promoted: list[Path] = []
    try:
        for _staged, destination in pairs:
            backups[destination] = _backup_existing_output(destination)
        for staged, destination in pairs:
            staged.replace(destination)
            promoted.append(destination)
    except OSError as error:
        rollback_errors: list[OSError] = []
        unrestored_backups: list[Path] = []
        for destination in reversed(promoted):
            backup = backups.get(destination)
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    backup.replace(destination)
            except OSError as current_error:
                rollback_errors.append(current_error)
                if backup is not None:
                    unrestored_backups.append(backup)
        cleanup_failures = _cleanup_backups(
            backups.values(),
            preserve=unrestored_backups,
        )
        recovery_backups = tuple(
            dict.fromkeys(
                [*unrestored_backups, *cleanup_failures]
            )
        )
        if rollback_errors:
            raise E5ExplorerBuildError(
                "output publication failed; recovery required; "
                "preserved_backups="
                f"{json.dumps([str(path) for path in recovery_backups])}"
            ) from rollback_errors[0]
        if cleanup_failures:
            raise E5ExplorerBuildError(
                "output publication failed; backup cleanup incomplete; "
                "preserved_backups="
                f"{json.dumps([str(path) for path in cleanup_failures])}"
            ) from error
        raise E5ExplorerBuildError(
            "output publication failed"
        ) from error
    return _cleanup_backups(backups.values())


def _cleanup_backups(
    backups: Collection[Path | None],
    *,
    preserve: Collection[Path] = (),
) -> tuple[Path, ...]:
    """Remove disposable backups and return paths that could not be removed."""

    preserved = set(preserve)
    failures: list[Path] = []
    for backup in backups:
        if backup is None or backup in preserved:
            continue
        try:
            backup.unlink(missing_ok=True)
        except OSError:
            failures.append(backup)
    return tuple(failures)


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
    sources = [
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
    ]
    run_validation = refinement / "runs" / "validation.json"
    sources.append(run_validation)
    return tuple(sources)


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
    if bundle.exists() and not bundle.is_file():
        raise E5ExplorerBuildError(
            "bundle output path is not a regular file"
        )


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


def _validate_output_locations(
    *,
    bundle: Path,
    html: Path,
    refinement: Path,
    construction: Path,
) -> None:
    outputs = (bundle, html)
    sealed_roots = (refinement, construction)
    if any(
        output == root or root in output.parents
        for output in outputs
        for root in sealed_roots
    ):
        raise E5ExplorerBuildError(
            "output path must be outside sealed artifact roots"
        )
    if any(output == REPO_ROOT or REPO_ROOT in output.parents for output in outputs):
        raise E5ExplorerBuildError(
            "private outputs must be outside the Git worktree"
        )


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
    _validate_output_locations(
        bundle=bundle_output,
        html=html_output,
        refinement=refinement,
        construction=construction,
    )
    _validate_html_destination(html_output)

    bundle = build_evidence_bundle(
        refinement_dir=refinement,
        construction_dir=construction,
        evaluation_path=evaluation,
        queries_path=queries,
        qrels_path=qrels,
    )
    temporary_bundle: Path | None = None
    temporary_html: Path | None = None
    recovery_backups: tuple[Path, ...] = ()
    try:
        temporary_bundle = _temporary_output_path(
            bundle_output,
            suffix=".json",
        )
        temporary_html = _temporary_output_path(
            html_output,
            suffix=".html",
        )
        write_json_atomically(temporary_bundle, bundle)
        run_generic_builder(
            input_path=(
                refinement
                / "runs"
                / "raw_hierarchical_dense_equal.trec"
            ),
            queries=queries,
            qrels=qrels,
            profiles=profiles,
            evidence_bundle=temporary_bundle,
            output=temporary_html,
        )
        recovery_backups = _promote_output_pair(
            staged_bundle=temporary_bundle,
            bundle=bundle_output,
            staged_html=temporary_html,
            html=html_output,
        )
    except OSError as error:
        raise E5ExplorerBuildError(
            "output staging failed"
        ) from error
    finally:
        if temporary_bundle is not None:
            temporary_bundle.unlink(missing_ok=True)
        if temporary_html is not None:
            temporary_html.unlink(missing_ok=True)

    summary = {
        "bundle": str(bundle_output),
        "evidence_per_candidate": bundle["meta"][
            "evidence_per_candidate"
        ],
        "html": str(html_output),
        "query_count": bundle["meta"]["query_count"],
    }
    if recovery_backups:
        summary.update(
            {
                "recovery_backups": [
                    str(path) for path in recovery_backups
                ],
                "warning": (
                    "published outputs are valid; "
                    "backup cleanup incomplete"
                ),
            }
        )
    print(json.dumps(summary, sort_keys=True))
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
