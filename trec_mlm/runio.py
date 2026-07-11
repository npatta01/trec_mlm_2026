"""Read, write, and validate six-column TREC run files."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from pathlib import Path


@dataclass(frozen=True, slots=True)
class RunEntry:
    """One parsed TREC run row."""

    query_id: str
    model_id: str
    rank: int
    score: float
    run_id: str


def write_trec_run(
    path: str | Path,
    rankings: Mapping[str, Sequence[str]],
    *,
    run_id: str,
) -> None:
    """Write complete rankings using unique ordinal scores.

    Raw model scores often tie. The run uses a strictly descending ordinal
    score after deterministic sorting so an external evaluator cannot resolve
    those ties differently from this project.
    """

    if not run_id or any(character.isspace() for character in run_id):
        raise ValueError("run_id must be a non-empty token without whitespace")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for query_id in sorted(rankings):
        model_ids = list(rankings[query_id])
        if len(set(model_ids)) != len(model_ids):
            raise ValueError(f"ranking for {query_id!r} contains duplicate models")
        candidate_count = len(model_ids)
        for rank, model_id in enumerate(model_ids, start=1):
            ordinal_score = candidate_count - rank + 1
            lines.append(
                f"{query_id} Q0 {model_id} {rank} {ordinal_score:.1f} {run_id}\n"
            )
    destination.write_text("".join(lines), encoding="utf-8")


def read_trec_run(path: str | Path) -> list[RunEntry]:
    """Parse a six-column TREC run, rejecting malformed rows."""

    entries: list[RunEntry] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.split()
            if not fields:
                continue
            if len(fields) != 6:
                raise ValueError(
                    f"{path}:{line_number}: expected six whitespace-separated fields"
                )
            query_id, iteration, model_id, rank_text, score_text, run_id = fields
            if iteration != "Q0":
                raise ValueError(f"{path}:{line_number}: second field must be Q0")
            try:
                rank = int(rank_text)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: rank must be an integer"
                ) from error
            try:
                score = float(score_text)
            except ValueError as error:
                raise ValueError(
                    f"{path}:{line_number}: score must be numeric"
                ) from error
            if rank < 1:
                raise ValueError(f"{path}:{line_number}: rank must be positive")
            if not math.isfinite(score):
                raise ValueError(f"{path}:{line_number}: score must be finite")
            entries.append(
                RunEntry(
                    query_id=query_id,
                    model_id=model_id,
                    rank=rank,
                    score=score,
                    run_id=run_id,
                )
            )
    if not entries:
        raise ValueError(f"{path}: run is empty")
    return entries


def validate_trec_run(
    path: str | Path,
    *,
    expected_query_ids: Sequence[str] | None = None,
    expected_model_ids: Sequence[str] | None = None,
) -> dict[str, object]:
    """Validate run shape, identities, ranks, and score ordering."""

    entries = read_trec_run(path)
    run_ids = {entry.run_id for entry in entries}
    if len(run_ids) != 1:
        raise ValueError(f"run contains multiple run IDs: {sorted(run_ids)!r}")

    by_query: dict[str, list[RunEntry]] = {}
    pairs: set[tuple[str, str]] = set()
    for entry in entries:
        pair = (entry.query_id, entry.model_id)
        if pair in pairs:
            raise ValueError(f"duplicate query/model pair: {pair!r}")
        pairs.add(pair)
        by_query.setdefault(entry.query_id, []).append(entry)

    actual_queries = set(by_query)
    if expected_query_ids is not None and actual_queries != set(expected_query_ids):
        missing = sorted(set(expected_query_ids) - actual_queries)
        extra = sorted(actual_queries - set(expected_query_ids))
        raise ValueError(f"run query mismatch: missing={missing!r}, extra={extra!r}")

    expected_models = set(expected_model_ids) if expected_model_ids is not None else None
    first_models: set[str] | None = None
    for query_id, query_entries in sorted(by_query.items()):
        ordered = sorted(query_entries, key=lambda entry: entry.rank)
        ranks = [entry.rank for entry in ordered]
        if ranks != list(range(1, len(ordered) + 1)):
            raise ValueError(f"query {query_id!r} ranks are not exactly 1..N")
        scores = [entry.score for entry in ordered]
        if any(left <= right for left, right in zip(scores, scores[1:])):
            raise ValueError(
                f"query {query_id!r} scores must strictly decrease with rank"
            )
        models = {entry.model_id for entry in ordered}
        if expected_models is not None and models != expected_models:
            missing = sorted(expected_models - models)
            extra = sorted(models - expected_models)
            raise ValueError(
                f"query {query_id!r} model mismatch: missing={missing!r}, extra={extra!r}"
            )
        if first_models is None:
            first_models = models
        elif models != first_models:
            raise ValueError(f"query {query_id!r} has a different candidate set")

    candidate_counts = {len(items) for items in by_query.values()}
    if len(candidate_counts) != 1:
        raise ValueError(f"queries have different candidate counts: {candidate_counts!r}")
    candidate_count = next(iter(candidate_counts))
    return {
        "valid": True,
        "run_id": next(iter(run_ids)),
        "query_count": len(by_query),
        "candidate_count": candidate_count,
        "row_count": len(entries),
    }


def rankings_from_run(path: str | Path) -> dict[str, list[str]]:
    """Return model-ID rankings from a run's rank fields."""

    validate_trec_run(path)
    entries = read_trec_run(path)
    by_query: dict[str, list[RunEntry]] = {}
    for entry in entries:
        by_query.setdefault(entry.query_id, []).append(entry)
    return {
        query_id: [entry.model_id for entry in sorted(items, key=lambda item: item.rank)]
        for query_id, items in by_query.items()
    }
