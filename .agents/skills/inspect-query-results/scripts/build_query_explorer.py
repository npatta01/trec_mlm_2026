#!/usr/bin/env python3
"""Build a self-contained query/candidate explorer from completed run artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import math
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Collection, Iterable, Iterator, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[4]
QUERY_ALIASES = ("target_query_id", "query_id", "outer_query_id", "qid")
CANDIDATE_ALIASES = ("model_id", "candidate_id", "doc_id")
RANK_ALIASES = ("final_rank", "primary_rank", "rank")
SCORE_ALIASES = (
    "raw_score",
    "aggregation_score",
    "sum_score",
    "primary_score",
    "score",
)
PRIMARY_RUN_FILENAMES = (
    "predictions.tsv",
    "run.trec",
    "normalized_run.trec",
    "source_alignment_run.trec",
)
QUERY_HEADER_ALIASES = ("query_id", "qid", "target_query_id")
QUERY_TEXT_ALIASES = ("query_text", "text", "query", "question")
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]{1,}")
NONLETTER_RE = re.compile(r"[^a-z]+")
STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


class ExplorerInputError(ValueError):
    """Raised when an input cannot safely power the explorer."""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a portable query results explorer from a completed run."
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--input",
        type=Path,
        help="Explicit predictions TSV or TREC run file. Overrides --run-dir.",
    )
    parser.add_argument("--queries", type=Path)
    parser.add_argument(
        "--qrels",
        type=Path,
        help="Optional post-run ground-truth labels in four-column TREC qrels format.",
    )
    parser.add_argument(
        "--evidence-bundle",
        type=Path,
        help="Optional validated grounded-evidence JSON; requires complete qrels.",
    )
    parser.add_argument("--profiles", type=Path)
    parser.add_argument(
        "--discovery",
        type=Path,
        help=(
            "Optional legacy lexical-response source; cannot be combined "
            "with --evidence-bundle."
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--title", default="Query Results Explorer")
    parser.add_argument("--top-candidates", type=positive_int, default=10)
    parser.add_argument("--responses-per-candidate", type=positive_int, default=3)
    parser.add_argument("--max-response-chars", type=positive_int, default=520)
    parser.add_argument("--no-responses", action="store_true")
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not auto-load development qrels from the completed run config.",
    )
    parser.add_argument("--query-column")
    parser.add_argument("--candidate-column")
    parser.add_argument("--rank-column")
    parser.add_argument("--score-column")
    return parser.parse_args()


def resolve_path(path: Path | None, root: Path) -> Path | None:
    if path is None:
        return None
    return path if path.is_absolute() else root / path


def choose_input(
    root: Path,
    *,
    run_dir: Path | None,
    explicit_input: Path | None,
) -> tuple[Path, Path]:
    selected = resolve_path(explicit_input, root)
    if selected is not None:
        if not selected.is_file():
            raise ExplorerInputError(f"input file does not exist: {selected}")
        return selected, selected.parent

    selected_dir = resolve_path(run_dir, root)
    if selected_dir is not None:
        if not selected_dir.is_dir():
            raise ExplorerInputError(f"run directory does not exist: {selected_dir}")
        direct_predictions = selected_dir / PRIMARY_RUN_FILENAMES[0]
        if direct_predictions.is_file():
            return direct_predictions, selected_dir
        child_predictions = sorted(selected_dir.glob("*/predictions.tsv"))
        if len(child_predictions) == 1:
            return child_predictions[0], child_predictions[0].parent
        if len(child_predictions) > 1:
            choices = ", ".join(str(path.parent.name) for path in child_predictions)
            raise ExplorerInputError(
                "run directory contains multiple prediction sets "
                f"({choices}); pass the intended child directory explicitly"
            )
        for filename in PRIMARY_RUN_FILENAMES[1:]:
            candidate = selected_dir / filename
            if candidate.is_file():
                return candidate, selected_dir
        trec_candidates = sorted(selected_dir.glob("*.trec"))
        if len(trec_candidates) == 1:
            return trec_candidates[0], selected_dir
        if len(trec_candidates) > 1:
            choices = ", ".join(path.name for path in trec_candidates)
            raise ExplorerInputError(
                "run directory contains multiple TREC runs "
                f"({choices}); pass the intended file with --input"
            )
        raise ExplorerInputError(
            f"no predictions.tsv or primary TREC run found in {selected_dir}"
        )

    artifact_root = root / "artifacts"
    prediction_files = list(artifact_root.glob("**/predictions.tsv"))
    if prediction_files:
        newest = max(prediction_files, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        possible_multi_run_root = newest.parent.parent
        sibling_predictions = sorted(possible_multi_run_root.glob("*/predictions.tsv"))
        if (
            len(sibling_predictions) > 1
            and (possible_multi_run_root / "summary.json").is_file()
            and not (possible_multi_run_root / "predictions.tsv").is_file()
        ):
            choices = ", ".join(path.parent.name for path in sibling_predictions)
            raise ExplorerInputError(
                "the newest artifact contains multiple prediction sets "
                f"({choices}); pass the intended child directory with --run-dir"
            )
        return newest, newest.parent

    trec_files = list(artifact_root.glob("**/*.trec"))
    if trec_files:
        newest = max(trec_files, key=lambda path: (path.stat().st_mtime_ns, str(path)))
        return newest, newest.parent
    raise ExplorerInputError(
        "no completed run was found under artifacts; pass --run-dir or --input"
    )


def _pick_column(
    fieldnames: Sequence[str],
    override: str | None,
    aliases: Sequence[str],
    label: str,
) -> str:
    if override:
        if override not in fieldnames:
            raise ExplorerInputError(
                f"{label} column {override!r} is not present; available columns: "
                + ", ".join(fieldnames)
            )
        return override
    matches = [name for name in aliases if name in fieldnames]
    if not matches:
        raise ExplorerInputError(
            f"could not detect the {label} column; expected one of "
            + ", ".join(aliases)
        )
    return matches[0]


def _finite_float(value: str, *, row_number: int, column: str) -> float:
    try:
        parsed = float(value)
    except ValueError as error:
        raise ExplorerInputError(
            f"row {row_number} has a non-numeric {column}: {value!r}"
        ) from error
    if not math.isfinite(parsed):
        raise ExplorerInputError(
            f"row {row_number} has a non-finite {column}: {value!r}"
        )
    return parsed


def _positive_rank(value: str, *, row_number: int, column: str) -> int:
    parsed = _finite_float(value, row_number=row_number, column=column)
    if parsed < 1 or not parsed.is_integer():
        raise ExplorerInputError(
            f"row {row_number} has an invalid positive integer {column}: {value!r}"
        )
    return int(parsed)


def load_predictions_tsv(
    path: Path,
    *,
    query_column: str | None,
    candidate_column: str | None,
    rank_column: str | None,
    score_column: str | None,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ExplorerInputError(f"prediction file is empty: {path}")
        fieldnames = [name.strip() for name in reader.fieldnames]
        if len(fieldnames) != len(set(fieldnames)):
            raise ExplorerInputError("prediction file contains duplicate column names")
        reader.fieldnames = fieldnames
        query_key = _pick_column(fieldnames, query_column, QUERY_ALIASES, "query ID")
        candidate_key = _pick_column(
            fieldnames, candidate_column, CANDIDATE_ALIASES, "candidate ID"
        )
        rank_key = _pick_column(fieldnames, rank_column, RANK_ALIASES, "rank")
        score_key = _pick_column(fieldnames, score_column, SCORE_ALIASES, "score")

        rows: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            query_id = (row.get(query_key) or "").strip()
            candidate_id = (row.get(candidate_key) or "").strip()
            score_text = (row.get(score_key) or "").strip()
            rank_text = (row.get(rank_key) or "").strip()
            if not query_id or not candidate_id or not score_text or not rank_text:
                raise ExplorerInputError(
                    f"row {row_number} has a blank query, candidate, score, or rank"
                )
            rows.append(
                {
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "rank": _positive_rank(
                        rank_text, row_number=row_number, column=rank_key
                    ),
                    "score": _finite_float(
                        score_text, row_number=row_number, column=score_key
                    ),
                    "score_text": score_text,
                }
            )
    if not rows:
        raise ExplorerInputError(f"prediction file has no data rows: {path}")
    return rows, {
        "query_column": query_key,
        "candidate_column": candidate_key,
        "rank_column": rank_key,
        "score_column": score_key,
        "input_kind": "predictions_tsv",
    }


def load_trec_run(path: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    run_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for row_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            fields = raw.split()
            if len(fields) != 6:
                raise ExplorerInputError(
                    f"TREC row {row_number} must contain six whitespace-separated fields"
                )
            query_id, _iteration, candidate_id, rank_text, score_text, run_id = fields
            rows.append(
                {
                    "query_id": query_id,
                    "candidate_id": candidate_id,
                    "rank": _positive_rank(
                        rank_text, row_number=row_number, column="rank"
                    ),
                    "score": _finite_float(
                        score_text, row_number=row_number, column="run score"
                    ),
                    "score_text": score_text,
                }
            )
            run_ids.add(run_id)
    if not rows:
        raise ExplorerInputError(f"TREC run has no data rows: {path}")
    if len(run_ids) != 1:
        raise ExplorerInputError("TREC input contains more than one run ID")
    return rows, {
        "query_column": "TREC field 1",
        "candidate_column": "TREC field 3",
        "rank_column": "TREC field 4",
        "score_column": "TREC field 5 (exported run score)",
        "input_kind": "trec_run",
    }


def validate_and_group_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    top_candidates: int,
    retain_all: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_pairs: set[tuple[str, str]] = set()
    seen_ranks: set[tuple[str, int]] = set()
    warnings: list[str] = []
    for row in rows:
        query_id = str(row["query_id"])
        candidate_id = str(row["candidate_id"])
        rank = int(row["rank"])
        pair = (query_id, candidate_id)
        rank_pair = (query_id, rank)
        if pair in seen_pairs:
            raise ExplorerInputError(
                f"duplicate prediction for query {query_id!r}, candidate {candidate_id!r}"
            )
        if rank_pair in seen_ranks:
            raise ExplorerInputError(f"duplicate rank {rank} for query {query_id!r}")
        seen_pairs.add(pair)
        seen_ranks.add(rank_pair)
        grouped[query_id].append(dict(row))

    counts = {query_id: len(values) for query_id, values in grouped.items()}
    if len(set(counts.values())) > 1:
        warnings.append("Candidate counts differ across queries in the selected input.")
    for query_id, values in grouped.items():
        values.sort(key=lambda item: item["rank"])
        expected = list(range(1, len(values) + 1))
        observed = [int(item["rank"]) for item in values]
        if observed != expected:
            raise ExplorerInputError(
                f"ranks for query {query_id!r} are not contiguous from 1"
            )
        grouped[query_id] = values if retain_all else values[:top_candidates]
    return dict(grouped), warnings


def select_display_rows(
    values: Sequence[Mapping[str, Any]],
    *,
    top_candidates: int,
    positive_ids: Collection[str] = (),
) -> list[dict[str, Any]]:
    """Select top-ranked rows and positives without changing frozen order."""

    wanted = {
        str(item["candidate_id"]) for item in values[:top_candidates]
    } | set(positive_ids)
    return [
        dict(item)
        for item in values
        if str(item["candidate_id"]) in wanted
    ]


def load_queries(path: Path) -> tuple[dict[str, str], list[str]]:
    if not path.is_file():
        raise ExplorerInputError(f"query file does not exist: {path}")
    queries: dict[str, str] = {}
    order: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        lines = [raw_line.rstrip("\r\n") for raw_line in handle]
    if not lines:
        raise ExplorerInputError(f"query file is empty: {path}")

    start = 0
    query_index = 0
    text_index = 1
    header = lines[0].split("\t")
    lowered = [value.strip().lower() for value in header]
    headered = any(value in QUERY_HEADER_ALIASES for value in lowered)
    if headered:
        start = 1
        query_index = next(
            index for index, value in enumerate(lowered) if value in QUERY_HEADER_ALIASES
        )
        matches = [
            index for index, value in enumerate(lowered) if value in QUERY_TEXT_ALIASES
        ]
        if not matches:
            raise ExplorerInputError(
                "headered query file needs a query_text, text, query, or question column"
            )
        text_index = matches[0]

    for row_number, line in enumerate(lines[start:], start=start + 1):
        row = line.split("\t") if headered else line.split("\t", maxsplit=1)
        if len(row) <= max(query_index, text_index):
            raise ExplorerInputError(f"query row {row_number} has too few columns")
        query_id = row[query_index].strip()
        text = row[text_index].strip()
        if not query_id or not text:
            raise ExplorerInputError(f"query row {row_number} has a blank ID or text")
        if query_id in queries:
            raise ExplorerInputError(f"duplicate query ID in query file: {query_id!r}")
        queries[query_id] = text
        order.append(query_id)
    return queries, order


def infer_query_path(root: Path, run_dir: Path, explicit: Path | None) -> Path:
    selected = resolve_path(explicit, root)
    if selected is not None:
        return selected
    config_path = run_dir / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ExplorerInputError(f"could not parse run config: {config_path}") from error
        configured = config.get("query_path") if isinstance(config, dict) else None
        if isinstance(configured, str) and configured:
            configured_path = Path(configured)
            candidates = [
                configured_path,
                root / configured_path,
            ]
            parts = configured_path.parts
            if "data" in parts:
                candidates.append(root / Path(*parts[parts.index("data") :]))
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
    return root / "data/dev/llm_dev_data.tsv"


def infer_qrels_path(
    root: Path,
    run_dir: Path,
    explicit: Path | None,
    *,
    no_labels: bool,
) -> Path | None:
    if no_labels:
        return None
    selected = resolve_path(explicit, root)
    if selected is not None:
        if not selected.is_file():
            raise ExplorerInputError(f"qrels file does not exist: {selected}")
        return selected

    config_path = run_dir / "config.json"
    if config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ExplorerInputError(f"could not parse run config: {config_path}") from error
        configured = config.get("qrels_path") if isinstance(config, dict) else None
        if isinstance(configured, str) and configured:
            configured_path = Path(configured)
            candidates = [configured_path, root / configured_path]
            parts = configured_path.parts
            if "data" in parts:
                candidates.append(root / Path(*parts[parts.index("data") :]))
            for candidate in candidates:
                if candidate.is_file():
                    return candidate
    return None


def load_qrels(path: Path) -> dict[str, dict[str, float]]:
    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    with path.open("r", encoding="utf-8") as handle:
        for row_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                continue
            fields = raw.split()
            if len(fields) != 4:
                raise ExplorerInputError(
                    f"qrels row {row_number} must contain four whitespace-separated fields"
                )
            query_id, _iteration, candidate_id, grade_text = fields
            if candidate_id in qrels[query_id]:
                raise ExplorerInputError(
                    f"duplicate qrel for query {query_id!r}, candidate {candidate_id!r}"
                )
            qrels[query_id][candidate_id] = _finite_float(
                grade_text, row_number=row_number, column="qrel grade"
            )
    if not qrels:
        raise ExplorerInputError(f"qrels file has no judgments: {path}")
    return dict(qrels)


def _validate_authoritative_evidence_bundle(
    payload: Any,
    *,
    expected_query_ids: Collection[str],
    final_component_metrics: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
) -> None:
    """Use the producer's validator so bundle semantics cannot drift."""

    source_root = str(REPO_ROOT)
    inserted_source_root = source_root not in sys.path
    if inserted_source_root:
        sys.path.insert(0, source_root)
    try:
        from trec_mlm.grounded_evidence_explorer import (
            EvidenceBundleError,
            validate_evidence_bundle,
        )
    except ImportError as error:
        raise ExplorerInputError(
            "could not load authoritative grounded-evidence validator"
        ) from error
    try:
        validate_evidence_bundle(
            payload,
            expected_query_ids=expected_query_ids,
            final_component_metrics=final_component_metrics,
        )
    except EvidenceBundleError as error:
        raise ExplorerInputError(str(error)) from error
    finally:
        if inserted_source_root:
            sys.path.remove(source_root)


def load_evidence_bundle(
    path: Path, query_ids: Collection[str]
) -> dict[str, Any]:
    """Load and validate a renderer-neutral grounded-evidence bundle."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ExplorerInputError(
            f"could not read evidence bundle: {path}"
        ) from error
    _validate_authoritative_evidence_bundle(
        payload,
        expected_query_ids=query_ids,
    )
    return payload


def validate_grounded_contract(
    evidence_bundle: Mapping[str, Any],
    *,
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    qrels: Mapping[str, Mapping[str, float]],
    queries: Mapping[str, str],
    top_candidates: int,
    evaluation: Mapping[str, Any],
) -> dict[str, list[dict[str, Any]]]:
    """Bind grounded evidence to frozen rows, complete qrels, and query text."""

    _validate_authoritative_evidence_bundle(
        evidence_bundle,
        expected_query_ids=grouped,
        final_component_metrics={
            query_id: {
                "ndcg_at_10": metric["ndcg_at_10"],
                "reciprocal_rank": metric["reciprocal_rank"],
            }
            for query_id, metric in evaluation["per_query"].items()
        },
    )

    if set(qrels) != set(grouped):
        raise ExplorerInputError("evidence bundle requires complete qrels")
    selected: dict[str, list[dict[str, Any]]] = {}
    for query_id, values in grouped.items():
        candidate_ids = {
            str(value["candidate_id"]) for value in values
        }
        if (
            query_id not in qrels
            or set(qrels[query_id]) != candidate_ids
        ):
            raise ExplorerInputError(
                "evidence bundle requires complete qrels"
            )
        grounded = evidence_bundle["queries"][query_id]
        candidates = grounded["candidates"]
        if set(candidates) != candidate_ids:
            raise ExplorerInputError(
                f"evidence candidate inventory differs for {query_id}"
            )
        if grounded["query_text"] != queries[query_id]:
            raise ExplorerInputError(
                f"evidence query text differs for {query_id}"
            )
        for value in values:
            candidate_id = str(value["candidate_id"])
            candidate = candidates[candidate_id]
            grade = float(qrels[query_id][candidate_id])
            if (
                float(candidate["qrel_grade"]) != grade
                or candidate["is_positive"] != (grade > 0.0)
                or candidate["ranks"]["raw_hierarchical_dense_equal"]
                != int(value["rank"])
            ):
                raise ExplorerInputError(
                    f"evidence candidate contract differs for {query_id}"
                )
        positive_ids = {
            candidate_id
            for candidate_id, grade in qrels[query_id].items()
            if float(grade) > 0.0
        }
        required_depth = min(10, len(values))
        if top_candidates < required_depth:
            raise ExplorerInputError(
                "grounded evidence requires --top-candidates of at least "
                f"{required_depth}"
            )
        required_evidence_ids = [
            str(value["candidate_id"])
            for value in values
            if int(value["rank"]) <= 10
            or str(value["candidate_id"]) in positive_ids
        ]
        if grounded["evidence_candidate_ids"] != required_evidence_ids:
            raise ExplorerInputError(
                f"evidence candidate ordering differs for {query_id}"
            )
        if any(
            len(candidates[candidate_id]["evidence"]) != 3
            for candidate_id in required_evidence_ids
        ):
            raise ExplorerInputError(
                f"evidence record coverage differs for {query_id}"
            )
        selected[query_id] = select_display_rows(
            values,
            top_candidates=top_candidates,
            positive_ids=positive_ids,
        )
        actual_ids = [
            str(value["candidate_id"]) for value in selected[query_id]
        ]
        if actual_ids != grounded["display_candidate_ids"]:
            raise ExplorerInputError(
                f"evidence display ordering differs for {query_id}"
            )
    return selected


def build_evaluation(
    rows: Sequence[Mapping[str, Any]],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    cutoff: int = 10,
) -> dict[str, Any]:
    rankings: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        rankings[str(row["query_id"])].append(row)

    per_query: dict[str, dict[str, Any]] = {}
    for query_id, values in rankings.items():
        if query_id not in qrels:
            continue
        ordered = sorted(values, key=lambda item: int(item["rank"]))
        system_ranks = {
            str(item["candidate_id"]): int(item["rank"]) for item in ordered
        }
        query_qrels = qrels[query_id]
        gains = [
            max(float(query_qrels.get(str(item["candidate_id"]), 0.0)), 0.0)
            for item in ordered
        ]
        dcg = math.fsum(
            gain / math.log2(rank + 1)
            for rank, gain in enumerate(gains[:cutoff], start=1)
        )
        ideal_gains = sorted(
            (max(float(value), 0.0) for value in query_qrels.values()),
            reverse=True,
        )[:cutoff]
        ideal_dcg = math.fsum(
            gain / math.log2(rank + 1)
            for rank, gain in enumerate(ideal_gains, start=1)
            if gain > 0.0
        )
        first_relevant_rank = next(
            (rank for rank, gain in enumerate(gains, start=1) if gain > 0.0),
            None,
        )
        relevant_total = sum(1 for grade in query_qrels.values() if grade > 0.0)
        relevant_in_top_k = sum(1 for gain in gains[:cutoff] if gain > 0.0)
        ideal_candidates = sorted(
            (
                {
                    "candidate_id": candidate_id,
                    "grade": float(grade),
                    "system_rank": system_ranks.get(candidate_id),
                }
                for candidate_id, grade in query_qrels.items()
                if float(grade) > 0.0
            ),
            key=lambda item: (-item["grade"], item["candidate_id"]),
        )
        per_query[query_id] = {
            "ndcg_at_10": dcg / ideal_dcg if ideal_dcg > 0.0 else 0.0,
            "ideal_ndcg_at_10": 1.0 if ideal_dcg > 0.0 else None,
            "dcg_at_10": dcg,
            "ideal_dcg_at_10": ideal_dcg,
            "ideal_candidates": ideal_candidates,
            "reciprocal_rank": 1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0,
            "first_relevant_rank": first_relevant_rank,
            "relevant_total": relevant_total,
            "relevant_in_top_10": relevant_in_top_k,
            "judged_in_top_10": sum(
                1
                for item in ordered[:cutoff]
                if str(item["candidate_id"]) in query_qrels
            ),
        }
    if not per_query:
        raise ExplorerInputError("qrels do not cover any prediction queries")
    return {
        "cutoff": cutoff,
        "query_count": len(per_query),
        "macro_ndcg_at_10": math.fsum(
            value["ndcg_at_10"] for value in per_query.values()
        )
        / len(per_query),
        "mean_reciprocal_rank": math.fsum(
            value["reciprocal_rank"] for value in per_query.values()
        )
        / len(per_query),
        "per_query": per_query,
    }


def attach_qrel_labels(
    grouped: Mapping[str, Sequence[dict[str, Any]]],
    qrels: Mapping[str, Mapping[str, float]],
) -> None:
    for query_id, values in grouped.items():
        query_qrels = qrels.get(query_id, {})
        for row in values:
            candidate_id = str(row["candidate_id"])
            row["grade"] = query_qrels.get(candidate_id)
            row["judged"] = candidate_id in query_qrels


def verify_input_hash(run_dir: Path, input_path: Path) -> None:
    validation_path = run_dir / "validation.json"
    if not validation_path.is_file():
        return
    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ExplorerInputError(
            f"could not parse run validation: {validation_path}"
        ) from error
    hashes = validation.get("artifact_sha256", {})
    if not isinstance(hashes, dict):
        return
    expected = hashes.get(input_path.name)
    if isinstance(expected, str) and expected and file_sha256(input_path) != expected:
        raise ExplorerInputError(
            f"{input_path.name} does not match the SHA-256 recorded in validation.json"
        )


def load_profiles(path: Path | None) -> tuple[dict[str, dict[str, Any]], str | None]:
    if path is None or not path.is_file():
        return {}, None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExplorerInputError(f"could not read candidate profiles: {path}") from error
    if isinstance(payload, dict) and isinstance(payload.get("profiles"), list):
        raw_profiles = payload["profiles"]
        profile_version = payload.get("profile_version")
    elif isinstance(payload, list):
        raw_profiles = payload
        profile_version = None
    else:
        raise ExplorerInputError(
            "candidate profile JSON must be a list or contain a profiles list"
        )

    profiles: dict[str, dict[str, Any]] = {}
    for raw in raw_profiles:
        if not isinstance(raw, dict):
            continue
        candidate_id = raw.get("candidate_id") or raw.get("model_id") or raw.get("id")
        if not isinstance(candidate_id, str) or not candidate_id:
            continue
        profiles[candidate_id] = {
            "candidate_id": candidate_id,
            "profile_version": raw.get("profile_version"),
            "profile_kind": raw.get("profile_kind"),
            "profile_summary": raw.get("profile_summary"),
            "history": raw.get("history") if isinstance(raw.get("history"), dict) else {},
            "coverage": raw.get("coverage")
            if isinstance(raw.get("coverage"), dict)
            else {},
            "attempt_style": raw.get("attempt_style")
            if isinstance(raw.get("attempt_style"), dict)
            else {},
            "metadata_quality": raw.get("metadata_quality")
            if isinstance(raw.get("metadata_quality"), dict)
            else {},
            "attempted_query_terms": raw.get("attempted_query_terms", [])[:8]
            if isinstance(raw.get("attempted_query_terms"), list)
            else [],
            "warnings": raw.get("warnings", [])[:4]
            if isinstance(raw.get("warnings"), list)
            else [],
        }
    return profiles, str(profile_version) if profile_version else None


def iter_top_level_json_object(path: Path) -> Iterator[tuple[str, Any]]:
    """Stream entries from a JSON object without retaining the whole file."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    absolute_offset = 0
    exhausted = False

    with path.open("r", encoding="utf-8") as handle:
        def read_more() -> bool:
            nonlocal buffer, position, absolute_offset, exhausted
            if exhausted:
                return False
            absolute_offset += position
            buffer = buffer[position:]
            position = 0
            chunk = handle.read(1024 * 1024)
            if not chunk:
                exhausted = True
                return False
            buffer += chunk
            return True

        def skip_whitespace() -> None:
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or not read_more():
                    return

        def expect(character: str) -> None:
            nonlocal position
            skip_whitespace()
            if position >= len(buffer) or buffer[position] != character:
                found = "end of file" if position >= len(buffer) else repr(buffer[position])
                raise ExplorerInputError(
                    f"invalid discovery JSON near character {absolute_offset + position}: "
                    f"expected {character!r}, found {found}"
                )
            position += 1

        def decode_value() -> Any:
            nonlocal position
            skip_whitespace()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                except json.JSONDecodeError as error:
                    if read_more():
                        continue
                    raise ExplorerInputError(
                        f"invalid discovery JSON near character "
                        f"{absolute_offset + error.pos}: {error.msg}"
                    ) from error
                position = end
                return value

        expect("{")
        first = True
        while True:
            skip_whitespace()
            if position < len(buffer) and buffer[position] == "}":
                position += 1
                break
            if not first:
                expect(",")
            key = decode_value()
            if not isinstance(key, str):
                raise ExplorerInputError("discovery JSON object keys must be strings")
            expect(":")
            value = decode_value()
            yield key, value
            first = False

        skip_whitespace()
        if position < len(buffer):
            raise ExplorerInputError(
                f"discovery JSON has trailing content near character "
                f"{absolute_offset + position}"
            )


def iter_discovery_queries(
    path: Path,
) -> Iterator[tuple[str, str, dict[str, str | None]]]:
    for source_id, entry in iter_top_level_json_object(path):
        if not isinstance(entry, dict):
            raise ExplorerInputError(
                f"discovery entry {source_id!r} must be a JSON object"
            )
        source_text = entry.get("query")
        responses = entry.get("responses")
        if not isinstance(source_text, str) or not isinstance(responses, dict):
            raise ExplorerInputError(
                f"discovery entry {source_id!r} needs a string query and responses object"
            )
        normalized_responses: dict[str, str | None] = {}
        for candidate_id, response in responses.items():
            if not isinstance(candidate_id, str) or not (
                response is None or isinstance(response, str)
            ):
                raise ExplorerInputError(
                    f"discovery entry {source_id!r} has an invalid response value"
                )
            normalized_responses[candidate_id] = response
        yield source_id, source_text, normalized_responses


def text_tokens(value: str) -> frozenset[str]:
    return frozenset(
        token
        for token in TOKEN_RE.findall(value.lower())
        if token not in STOPWORDS
    )


def normalized_response(value: str) -> str:
    return NONLETTER_RE.sub(" ", value.lower()).strip()


def usable_response(value: str | None) -> bool:
    if value is None or len(value.strip()) < 24:
        return False
    return normalized_response(value) not in {"i don t know", "i do not know"}


def truncate_excerpt(value: str, maximum: int) -> str:
    normalized = re.sub(r"\s+", " ", value).strip()
    if len(normalized) <= maximum:
        return normalized
    return normalized[: maximum - 1].rstrip() + "…"


def build_related_responses(
    discovery_path: Path,
    *,
    query_texts: Mapping[str, str],
    candidates_by_query: Mapping[str, Sequence[str]],
    per_candidate: int,
    maximum_chars: int,
) -> tuple[dict[str, dict[str, list[dict[str, Any]]]], int]:
    target_tokens = {query_id: text_tokens(text) for query_id, text in query_texts.items()}
    target_normalized = {
        query_id: re.sub(r"\s+", " ", text.lower()).strip()
        for query_id, text in query_texts.items()
    }
    token_to_targets: dict[str, set[str]] = defaultdict(set)
    for query_id, tokens in target_tokens.items():
        for token in tokens:
            token_to_targets[token].add(query_id)

    heaps: dict[tuple[str, str], list[tuple[float, str, str, str]]] = defaultdict(list)
    source_count = 0
    for source_id, source_text, source_responses in iter_discovery_queries(discovery_path):
        source_count += 1
        source_tokens = text_tokens(source_text)
        if not source_tokens:
            continue
        possible_targets: set[str] = set()
        for token in source_tokens:
            possible_targets.update(token_to_targets.get(token, ()))
        for query_id in possible_targets:
            if target_normalized[query_id] == re.sub(
                r"\s+", " ", source_text.lower()
            ).strip():
                continue
            overlap = len(source_tokens & target_tokens[query_id])
            # One shared question word is usually too weak to justify showing a
            # historical answer (for example, two unrelated "who became" queries).
            if overlap < 2:
                continue
            denominator = math.sqrt(len(source_tokens) * len(target_tokens[query_id]))
            similarity = overlap / denominator if denominator else 0.0
            for candidate_id in candidates_by_query[query_id]:
                response = source_responses.get(candidate_id)
                if not usable_response(response):
                    continue
                item = (
                    similarity,
                    source_id,
                    source_text,
                    truncate_excerpt(str(response), maximum_chars),
                )
                heap = heaps[(query_id, candidate_id)]
                if len(heap) < per_candidate:
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
    if source_count == 0:
        raise ExplorerInputError(
            f"discovery file did not match the expected pretty-printed structure: {discovery_path}"
        )

    result: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(dict)
    for (query_id, candidate_id), heap in heaps.items():
        items = sorted(heap, reverse=True)
        result[query_id][candidate_id] = [
            {
                "source_query_id": source_id,
                "source_query": source_text,
                "response": response,
                "similarity": similarity,
            }
            for similarity, source_id, source_text, response in items
        ]
    return {query: dict(values) for query, values in result.items()}, source_count


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def safe_json(payload: Mapping[str, Any]) -> str:
    value = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return value.replace("&", "\\u0026").replace("<", "\\u003c").replace(
        ">", "\\u003e"
    )


def build_payload(
    *,
    root: Path,
    input_path: Path,
    query_path: Path,
    qrels_path: Path | None,
    profile_path: Path | None,
    discovery_path: Path | None,
    title: str,
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
    schema: Mapping[str, str],
    queries: Mapping[str, str],
    query_order: Sequence[str],
    profiles: Mapping[str, Mapping[str, Any]],
    profile_version: str | None,
    related_responses: Mapping[str, Mapping[str, Sequence[Mapping[str, Any]]]],
    source_count: int,
    warnings: Sequence[str],
    candidate_limit: int,
    evaluation: Mapping[str, Any] | None,
    evidence_bundle: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    missing_queries = sorted(set(grouped) - set(queries))
    if missing_queries:
        preview = ", ".join(missing_queries[:5])
        raise ExplorerInputError(
            f"query text is missing for {len(missing_queries)} prediction queries: {preview}; "
            "pass the matching --queries file"
        )
    ordered_ids = [query_id for query_id in query_order if query_id in grouped]
    ordered_ids.extend(sorted(set(grouped) - set(ordered_ids)))
    query_rows = [
        {
            "id": query_id,
            "text": queries[query_id],
            "results": [dict(value) for value in grouped[query_id]],
            "evaluation": evaluation["per_query"].get(query_id)
            if evaluation
            else None,
        }
        for query_id in ordered_ids
    ]
    if evidence_bundle is not None:
        for query in query_rows:
            query["grounded"] = dict(
                evidence_bundle["queries"][query["id"]]
            )
    embedded_candidates = sorted(
        {
            str(result["candidate_id"])
            for query in query_rows
            for result in query["results"]
        }
    )
    selected_profiles = {
        candidate_id: dict(profiles[candidate_id])
        for candidate_id in embedded_candidates
        if candidate_id in profiles
    }
    embedded_response_count = sum(
        len(items)
        for candidate_responses in related_responses.values()
        for items in candidate_responses.values()
    )
    notes = list(warnings)
    if schema["input_kind"] == "trec_run":
        notes.append(
            "This explorer uses an exported TREC run score. Repository TREC writers may "
            "store ordinal rank scores; use predictions.tsv for native system scores."
        )
    if not selected_profiles:
        notes.append("No candidate profiles were available; candidate IDs still render.")
    if discovery_path is None:
        notes.append("Related historical responses were not embedded.")
    elif embedded_response_count == 0:
        notes.append(
            "No historical responses met the relatedness and usability thresholds."
        )
    payload_meta = {
        "title": title,
        "input": display_path(input_path, root),
        "input_sha256": file_sha256(input_path),
        "queries_source": display_path(query_path, root),
        "qrels_source": display_path(qrels_path, root) if qrels_path else None,
        "profiles_source": display_path(profile_path, root)
        if profile_path and profile_path.is_file()
        else None,
        "discovery_source": display_path(discovery_path, root)
        if discovery_path
        else None,
        "score_field": schema["score_column"],
        "input_kind": schema["input_kind"],
        "query_count": len(query_rows),
        "candidate_count": len(embedded_candidates),
        "candidate_limit": candidate_limit,
        "displayed_result_count": sum(
            len(query["results"]) for query in query_rows
        ),
        "profile_count": len(selected_profiles),
        "profile_version": profile_version,
        "discovery_query_count": source_count,
        "embedded_response_count": embedded_response_count,
        "labels_included": bool(evaluation),
        "evaluated_query_count": evaluation["query_count"] if evaluation else 0,
        "macro_ndcg_at_10": evaluation["macro_ndcg_at_10"]
        if evaluation
        else None,
        "mean_reciprocal_rank": evaluation["mean_reciprocal_rank"]
        if evaluation
        else None,
        "notes": notes,
    }
    if evidence_bundle is not None:
        payload_meta["grounded_evidence"] = dict(evidence_bundle["meta"])
    return {
        "meta": payload_meta,
        "queries": query_rows,
        "profiles": selected_profiles,
        "responses": related_responses,
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__EXPLORER_TITLE__</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: #f5f7fa;
      --panel: #ffffff;
      --panel-soft: #f8fafc;
      --ink: #172033;
      --muted: #5b6575;
      --line: #d8dee8;
      --accent: #225ea8;
      --accent-soft: #eaf2ff;
      --focus: #8b5cf6;
      --shadow: 0 12px 35px rgba(23, 32, 51, 0.08);
    }
    @media (prefers-color-scheme: dark) {
      :root {
        --bg: #111827;
        --panel: #182233;
        --panel-soft: #202c3e;
        --ink: #f1f5f9;
        --muted: #b3bfd0;
        --line: #344258;
        --accent: #8fc3ff;
        --accent-soft: #263c58;
        --focus: #c4b5fd;
        --shadow: none;
      }
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      line-height: 1.5;
    }
    button, input, select { font: inherit; }
    button { color: inherit; }
    .shell { max-width: 1440px; margin: 0 auto; padding: 24px; }
    .page-head { margin-bottom: 20px; }
    .eyebrow { margin: 0 0 5px; color: var(--accent); font-size: .78rem; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
    h1 { margin: 0; font-size: clamp(1.8rem, 4vw, 3rem); line-height: 1.08; }
    .lede { max-width: 72ch; margin: 10px 0 0; color: var(--muted); }
    .summary { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px; }
    .pill, .term { display: inline-flex; align-items: center; border: 1px solid var(--line); border-radius: 999px; background: var(--panel); padding: 5px 9px; font-size: .82rem; }
    .layout { display: grid; grid-template-columns: minmax(230px, 310px) minmax(0, 1fr); gap: 18px; align-items: start; }
    .panel { border: 1px solid var(--line); border-radius: 14px; background: var(--panel); box-shadow: var(--shadow); }
    .query-panel { position: sticky; top: 16px; max-height: calc(100vh - 32px); overflow: hidden; display: flex; flex-direction: column; }
    .panel-head { padding: 16px; border-bottom: 1px solid var(--line); }
    .panel-title-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
    .panel-head h2, .section h2 { margin: 0; font-size: 1rem; }
    .query-toggle { display: none; border: 1px solid var(--line); border-radius: 999px; background: var(--panel-soft); padding: 5px 9px; color: var(--accent); cursor: pointer; font-size: .8rem; font-weight: 750; }
    .search { width: 100%; margin-top: 10px; border: 1px solid var(--line); border-radius: 9px; background: var(--panel-soft); color: var(--ink); padding: 10px 11px; }
    .evaluation-control { display: grid; gap: 5px; margin-top: 11px; }
    .evaluation-select { width: 100%; border: 1px solid var(--line); border-radius: 9px; background: var(--panel-soft); color: var(--ink); padding: 8px 10px; }
    .search:focus, select:focus-visible, button:focus-visible, summary:focus-visible { outline: 3px solid color-mix(in srgb, var(--focus) 38%, transparent); outline-offset: 2px; }
    .query-list { overflow-y: auto; padding: 8px; }
    .query-button { width: 100%; border: 1px solid transparent; border-radius: 9px; background: transparent; padding: 10px; text-align: left; cursor: pointer; }
    .query-button:hover { background: var(--panel-soft); }
    .query-button[aria-current="true"] { border-color: var(--accent); background: var(--accent-soft); }
    .query-id { display: block; color: var(--muted); font: 750 .76rem ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .query-preview { display: -webkit-box; margin-top: 3px; overflow: hidden; -webkit-box-orient: vertical; -webkit-line-clamp: 2; font-size: .88rem; }
    .main { display: grid; gap: 18px; }
    .query-card { padding: clamp(18px, 3vw, 28px); }
    .query-card h2 { margin: 8px 0 0; font-size: clamp(1.25rem, 3vw, 2rem); line-height: 1.25; }
    .query-evaluation { display: flex; flex-wrap: wrap; gap: 7px; margin-top: 14px; }
    .evaluation-pill { display: inline-flex; border: 1px solid var(--line); border-radius: 999px; background: var(--panel-soft); padding: 5px 9px; font-size: .8rem; }
    .oracle-reference { width: 100%; margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--line); }
    .oracle-title { display: block; margin-bottom: 7px; color: var(--muted); font-size: .76rem; }
    .oracle-list { display: flex; flex-wrap: wrap; gap: 7px; }
    .oracle-candidate { display: inline-flex; gap: 6px; align-items: center; border: 1px solid var(--line); border-radius: 9px; padding: 6px 8px; font-size: .78rem; font-weight: 700; }
    .oracle-candidate.grade-high { border-color: #3c8c67; background: #e9f7ef; color: #17603f; }
    .oracle-candidate.grade-positive { border-color: #d4a72c; background: #fff7db; color: #755600; }
    .oracle-candidate.missed { box-shadow: inset 4px 0 #c33f49; }
    .oracle-candidate.retrieved { box-shadow: inset 4px 0 #2582c4; }
    .oracle-legend { display: block; margin-top: 7px; color: var(--muted); font-size: .72rem; }
    @media (prefers-color-scheme: dark) {
      .oracle-candidate.grade-high { border-color: #5aac82; background: #173d2c; color: #a8e2c3; }
      .oracle-candidate.grade-positive { border-color: #b98b24; background: #3d3217; color: #f0d682; }
    }
    .mono { color: var(--muted); font: .78rem ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }
    .results-grid { display: grid; grid-template-columns: minmax(0, 1.2fr) minmax(300px, .8fr); gap: 18px; align-items: start; }
    .results-grid.grounded { grid-template-columns: minmax(0, 1fr); }
    .section { padding: 18px; }
    .section-note { margin: 5px 0 14px; color: var(--muted); font-size: .86rem; }
    .table-wrap { overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; }
    .results-grid.grounded table { min-width: 1060px; }
    th { color: var(--muted); font-size: .72rem; letter-spacing: .05em; text-align: left; text-transform: uppercase; }
    th, td { border-bottom: 1px solid var(--line); padding: 10px 8px; vertical-align: middle; }
    th:first-child, td:first-child { width: 54px; text-align: right; }
    th:last-child, td:last-child { width: 108px; text-align: right; }
    .candidate-button { display: flex; width: 100%; align-items: center; justify-content: space-between; gap: 6px; border: 0; background: transparent; padding: 0; text-align: left; cursor: pointer; font-weight: 720; overflow-wrap: anywhere; }
    .candidate-button::after { content: "›"; flex: none; color: var(--accent); font-size: 1.05rem; opacity: .7; }
    .candidate-button:hover { color: var(--accent); }
    .candidate-name { min-width: 0; }
    .grade-badge { flex: none; border: 1px solid var(--line); border-radius: 999px; padding: 2px 6px; font-size: .68rem; font-weight: 800; white-space: nowrap; }
    .grade-badge.relevant { border-color: #3c8c67; background: #e9f7ef; color: #17603f; }
    .grade-badge.not-relevant { background: var(--panel-soft); color: var(--muted); }
    @media (prefers-color-scheme: dark) {
      .grade-badge.relevant { border-color: #5aac82; background: #173d2c; color: #a8e2c3; }
    }
    tr.selected { background: var(--accent-soft); }
    .score { white-space: nowrap; font: 700 .84rem ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .mechanism-summary { display: grid; gap: 7px; margin-top: 14px; border: 1px solid var(--line); border-radius: 10px; background: var(--panel-soft); padding: 12px; }
    .mechanism-tags { display: flex; flex-wrap: wrap; gap: 7px; }
    .mechanism-tag, .context-risk { display: inline-flex; align-items: center; border: 1px solid var(--line); border-radius: 999px; padding: 4px 8px; font-size: .78rem; font-weight: 720; }
    .context-risk { border-color: #b45309; color: #92400e; }
    .mechanism-summary p { margin: 0; color: var(--muted); font-size: .84rem; }
    .cohort-list { display: flex; flex-wrap: wrap; gap: 6px; }
    .cohort-tag { border: 1px solid var(--line); border-radius: 999px; padding: 3px 7px; color: var(--muted); font-size: .74rem; }
    .evidence-row td { padding: 0 8px 12px; text-align: left; }
    .evidence-drawer { border: 1px solid var(--line); border-radius: 10px; background: var(--panel-soft); padding: 9px 11px; }
    .evidence-drawer summary { color: var(--accent); cursor: pointer; font-weight: 750; }
    .evidence-drawer[open] summary { margin-bottom: 8px; }
    .candidate-title { margin: 7px 0 0; font-size: 1.32rem; overflow-wrap: anywhere; }
    .candidate-summary { margin: 9px 0 0; color: var(--muted); }
    .stat-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 9px; margin-top: 16px; }
    .stat { border: 1px solid var(--line); border-radius: 10px; background: var(--panel-soft); padding: 10px; }
    .stat-label { color: var(--muted); font-size: .72rem; }
    .stat-value { display: block; margin-top: 3px; font-weight: 780; }
    .subsection { margin-top: 18px; }
    .subsection h3 { margin: 0 0 8px; font-size: .92rem; }
    .terms { display: flex; flex-wrap: wrap; gap: 6px; }
    .term { background: var(--panel-soft); }
    .history-card { border: 1px solid var(--line); border-radius: 10px; padding: 11px; }
    .history-card + .history-card { margin-top: 8px; }
    .history-question { margin: 0; font-weight: 720; }
    .history-response { margin: 7px 0 0; color: var(--muted); font-size: .88rem; }
    .history-meta { margin-top: 7px; color: var(--muted); font-size: .72rem; }
    .empty { border: 1px dashed var(--line); border-radius: 10px; color: var(--muted); padding: 12px; font-size: .86rem; }
    .main-empty { padding: 18px; }
    .warning-list { margin: 8px 0 0; padding-left: 18px; color: var(--muted); font-size: .82rem; }
    .foot { margin-top: 18px; color: var(--muted); font-size: .78rem; overflow-wrap: anywhere; }
    @media (max-width: 920px) {
      .layout, .results-grid { grid-template-columns: 1fr; }
      .query-panel { position: static; max-height: 340px; }
      .query-toggle { display: inline-flex; }
      .query-panel[data-collapsed="true"] { max-height: none; }
      .query-panel[data-collapsed="true"] .panel-head { border-bottom: 0; }
      .query-panel[data-collapsed="true"] .query-search-control,
      .query-panel[data-collapsed="true"] .evaluation-control,
      .query-panel[data-collapsed="true"] .query-list { display: none; }
    }
    @media (max-width: 560px) {
      .shell { padding: 14px; }
      .summary { flex-wrap: nowrap; overflow-x: auto; padding-bottom: 4px; scrollbar-width: thin; }
      .summary .pill { flex: none; }
      th, td { padding: 10px 4px; }
      th:first-child, td:first-child { width: 36px; }
      th:last-child, td:last-child { width: 76px; }
      .candidate-button { font-size: .9rem; }
      .candidate-button::after { display: none; }
      .stat-grid { grid-template-columns: 1fr; }
      .section { padding: 14px; }
    }
    @media (forced-colors: active) {
      .panel, .stat, .history-card, .pill, .term { border: 1px solid CanvasText; }
    }
  </style>
</head>
<body>
  <main class="shell">
    <header class="page-head">
      <p class="eyebrow">Completed ranking run</p>
      <h1 id="pageTitle"></h1>
      <p class="lede">Choose a query, inspect its leading candidates, and open a candidate for a compact view of its historical behavior.</p>
      <div class="summary" id="summary"></div>
    </header>
    <div class="layout">
      <aside class="panel query-panel" aria-label="Queries">
        <div class="panel-head">
          <div class="panel-title-row">
            <h2>Queries</h2>
            <button class="query-toggle" id="queryToggle" type="button" aria-expanded="true">Hide queries</button>
          </div>
          <label class="query-search-control">
            <span class="mono">Search query text or ID</span>
            <input class="search" id="querySearch" type="search" autocomplete="off">
          </label>
          <label class="evaluation-control" id="evaluationControl">
            <span class="mono" id="evaluationLabel">Ground-truth query ordering</span>
            <select class="evaluation-select" id="evaluationFilter">
              <option value="weakest">Weakest nDCG@10 first</option>
              <option value="failures">Zero nDCG@10 only</option>
              <option value="strongest">Strongest nDCG@10 first</option>
              <option value="source">Source order</option>
            </select>
          </label>
        </div>
        <div class="query-list" id="queryList"></div>
      </aside>
      <div class="main">
        <section class="panel query-card" id="queryCard" aria-labelledby="queryHeading">
          <span class="mono" id="activeQueryId"></span>
          <h2 id="queryHeading"></h2>
          <div class="query-evaluation" id="queryEvaluation"></div>
          <div id="groundedSummary"></div>
        </section>
        <div class="panel empty main-empty" id="mainEmpty" role="status" hidden>No queries match this filter.</div>
        <div class="results-grid" id="resultsGrid">
          <section class="panel section" id="resultsSection" aria-labelledby="resultsHeading">
            <h2 id="resultsHeading">Top candidates</h2>
            <p class="section-note"><span id="resultLimit"></span> Select a candidate to inspect its history.<span id="labelLegend"> G0/G1/G2 badges are development ground-truth grades.</span> Scores belong to this run only and should not be compared with scores from another approach.</p>
            <div class="table-wrap">
              <table>
                <thead><tr id="resultsHead"><th>Rank</th><th>Candidate</th><th>System score</th></tr></thead>
                <tbody id="resultsBody"></tbody>
              </table>
            </div>
            <div class="oracle-reference" id="oracleReference"></div>
          </section>
          <section class="panel section" id="candidateSection" aria-labelledby="candidateHeading">
            <span class="mono" id="candidateRank"></span>
            <h2 class="candidate-title" id="candidateHeading"></h2>
            <p class="candidate-summary" id="candidateSummary"></p>
            <div class="stat-grid" id="candidateStats"></div>
            <div class="subsection" id="termSection">
              <h3>Observed attempted-query terms</h3>
              <div class="terms" id="candidateTerms"></div>
            </div>
            <div class="subsection" id="responseSection">
              <h3>Related past responses</h3>
              <p class="section-note">Selected by simple lexical overlap for context. These responses are not verified correctness and are not claimed as the ranking explanation.</p>
              <div id="responseList"></div>
            </div>
            <div class="subsection" id="warningSection">
              <h3>Interpretation notes</h3>
              <ul class="warning-list" id="candidateWarnings"></ul>
            </div>
          </section>
        </div>
      </div>
    </div>
    <footer class="foot" id="footer"></footer>
  </main>
  <noscript>This explorer requires JavaScript to switch among queries and candidates.</noscript>
  <script id="explorerData" type="application/json">__EXPLORER_PAYLOAD__</script>
  <script>
    "use strict";
    const data = JSON.parse(document.getElementById("explorerData").textContent);
    const groundedMode = data.queries.some((query) => Boolean(query.grounded));
    const rankColumns = [
      ["dense_reference", "Dense rank"],
      ["profile_only", "Profile rank"],
      ["dense_profile_equal", "Prior fusion rank"],
      ["raw_hierarchical", "Raw hierarchical rank"],
      ["raw_hierarchical_dense_equal", "New fusion rank"],
    ];
    const weakestQuery = data.meta.labels_included
      ? [...data.queries].filter((query) => query.evaluation).sort((left, right) => left.evaluation.ndcg_at_10 - right.evaluation.ndcg_at_10)[0]
      : null;
    const firstTop1Failure = groundedMode
      ? data.queries.find((query) => query.grounded?.retrieval_outcome !== "top-1 success")
      : null;
    const state = {
      queryId: firstTop1Failure
        ? firstTop1Failure.id
        : (weakestQuery ? weakestQuery.id : (data.queries[0] ? data.queries[0].id : null)),
      candidateId: null,
      search: "",
      evaluationFilter: groundedMode
        ? "top_1_failure"
        : (data.meta.labels_included ? "weakest" : "source")
    };
    const byId = (id) => document.getElementById(id);
    const el = (tag, className, text) => {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined && text !== null) node.textContent = String(text);
      return node;
    };
    const numeric = (value) => value !== null && value !== undefined && !(typeof value === "string" && value.trim() === "") && Number.isFinite(Number(value));
    const percent = (value) => numeric(value) ? (Number(value) * 100).toFixed(1) + "%" : "—";
    const number = (value) => numeric(value) ? Number(value).toLocaleString() : "—";
    const formatScore = (value) => numeric(value) ? Number(value).toFixed(6) : String(value ?? "—");
    const formatMetric = (value) => numeric(value) ? Number(value).toFixed(3) : "—";
    const formatGrade = (value) => numeric(value) && Number(value) % 1 === 0 ? String(Number(value)) : number(value);
    const mobileQueryLayout = window.matchMedia("(max-width: 920px)");
    const activeQuery = () => data.queries.find((query) => query.id === state.queryId);
    const activeResult = () => {
      const query = activeQuery();
      return query ? query.results.find((result) => result.candidate_id === state.candidateId) : null;
    };

    function addPill(text) {
      byId("summary").appendChild(el("span", "pill", text));
    }

    function setQueryPanelCollapsed(collapsed) {
      const panel = document.querySelector(".query-panel");
      const shouldCollapse = mobileQueryLayout.matches && collapsed;
      panel.dataset.collapsed = shouldCollapse ? "true" : "false";
      byId("queryToggle").setAttribute("aria-expanded", shouldCollapse ? "false" : "true");
      byId("queryToggle").textContent = shouldCollapse ? "Choose query" : "Hide queries";
    }

    function renderSummary() {
      byId("pageTitle").textContent = data.meta.title;
      byId("labelLegend").hidden = !data.meta.labels_included;
      byId("summary").replaceChildren();
      addPill(number(data.meta.query_count) + " queries");
      addPill(groundedMode
        ? "Top " + number(data.meta.candidate_limit) + " plus development-positive candidates"
        : "Top " + number(data.meta.candidate_limit) + " candidates per query");
      addPill(number(data.meta.candidate_count) + " unique candidates represented");
      if (data.meta.profile_count) addPill(number(data.meta.profile_count) + " candidate profiles");
      if (data.meta.discovery_query_count) addPill(number(data.meta.discovery_query_count) + " historical questions scanned");
      if (data.meta.labels_included) addPill("Dev labels · nDCG@10 " + formatMetric(data.meta.macro_ndcg_at_10) + " · MRR " + formatMetric(data.meta.mean_reciprocal_rank));
      const note = data.meta.notes && data.meta.notes.length ? " Notes: " + data.meta.notes.join(" ") : "";
      byId("footer").textContent = "Source: " + data.meta.input + " · Score field: " + data.meta.score_field + "." + note;
    }

    function configureEvaluationFilters() {
      if (!groundedMode) return;
      byId("evaluationLabel").textContent = "Grounded diagnostic filter";
      const groundedFilters = [
        ["top_1_failure", "Top-1 failures"],
        ["top_1_success", "Top-1 success"],
        ["top_10_miss", "Top-10 but not top-1"],
        ["top_10_absent", "No relevant candidate in top 10"],
        ["zero_fused", "Zero fused nDCG@10"],
        ["persistent_zero", "Persistent all-system zero"],
        ["fusion_suppressed", "Raw success / fusion suppressed"],
        ["new_zero_regression", "New zero regression"],
        ["rescued", "Rescued prior zero"],
        ["context_risk", "Context-risk flag"],
      ];
      groundedFilters.forEach(([value, label]) => {
        const option = el("option", "", label);
        option.value = value;
        byId("evaluationFilter").appendChild(option);
      });
    }

    function legacyEvaluationMatch(query) {
      if (state.evaluationFilter === "failures") {
        return query.evaluation && query.evaluation.ndcg_at_10 === 0;
      }
      return true;
    }

    function matchesEvaluationFilter(query) {
      if (!query.grounded) return legacyEvaluationMatch(query);
      const tags = new Set(query.grounded.cohorts || []);
      if (state.evaluationFilter === "zero_fused") return tags.has("zero fused nDCG@10");
      if (state.evaluationFilter === "persistent_zero") return tags.has("persistent all-system zero");
      if (state.evaluationFilter === "fusion_suppressed") return tags.has("raw-success/fusion-suppressed");
      if (state.evaluationFilter === "new_zero_regression") return tags.has("new zero regression");
      if (state.evaluationFilter === "rescued") return tags.has("rescued prior zero");
      if (state.evaluationFilter === "context_risk") return query.grounded.context_risk === true;
      if (state.evaluationFilter === "top_1_failure") return query.grounded.retrieval_outcome !== "top-1 success";
      if (state.evaluationFilter === "top_1_success") return query.grounded.retrieval_outcome === "top-1 success";
      if (state.evaluationFilter === "top_10_miss") return query.grounded.retrieval_outcome === "top-10 but not top-1";
      if (state.evaluationFilter === "top_10_absent") return query.grounded.retrieval_outcome === "no relevant candidate in top 10";
      return legacyEvaluationMatch(query);
    }

    function visibleQueries() {
      const needle = state.search.trim().toLowerCase();
      let queries = data.queries.filter((query) => !needle || (query.id + " " + query.text).toLowerCase().includes(needle));
      if (!data.meta.labels_included) return queries;
      queries = queries.filter(matchesEvaluationFilter);
      if (state.evaluationFilter === "weakest") {
        queries = [...queries].sort((left, right) => (left.evaluation?.ndcg_at_10 ?? Infinity) - (right.evaluation?.ndcg_at_10 ?? Infinity));
      } else if (state.evaluationFilter === "strongest") {
        queries = [...queries].sort((left, right) => (right.evaluation?.ndcg_at_10 ?? -Infinity) - (left.evaluation?.ndcg_at_10 ?? -Infinity));
      }
      return queries;
    }

    function renderQueryList(queries) {
      const list = byId("queryList");
      list.replaceChildren();
      queries.forEach((query) => {
        const button = el("button", "query-button");
        button.type = "button";
        button.setAttribute("aria-current", query.id === state.queryId ? "true" : "false");
        const evaluationText = query.evaluation ? " · nDCG " + formatMetric(query.evaluation.ndcg_at_10) : "";
        button.appendChild(el("span", "query-id", query.id + evaluationText));
        button.appendChild(el("span", "query-preview", query.text));
        button.addEventListener("click", () => {
          state.queryId = query.id;
          state.candidateId = query.results[0] ? query.results[0].candidate_id : null;
          setQueryPanelCollapsed(true);
          renderAll();
        });
        list.appendChild(button);
      });
      if (!list.childElementCount) list.appendChild(el("div", "empty", "No queries match this search."));
    }

    function synchronizeSelection(queries) {
      if (!queries.length) {
        state.queryId = null;
        state.candidateId = null;
        return;
      }
      let query = queries.find((item) => item.id === state.queryId);
      if (!query) {
        query = queries[0];
        state.queryId = query.id;
        state.candidateId = null;
      }
      if (!query.results.some((result) => result.candidate_id === state.candidateId)) {
        state.candidateId = query.results[0] ? query.results[0].candidate_id : null;
      }
    }

    function renderEmptyMain() {
      byId("mainEmpty").hidden = false;
      byId("queryCard").hidden = true;
      byId("resultsGrid").hidden = true;
      byId("resultsSection").hidden = true;
      byId("candidateSection").hidden = true;
      byId("activeQueryId").textContent = "";
      byId("queryHeading").textContent = "";
      byId("queryEvaluation").replaceChildren();
      byId("groundedSummary").replaceChildren();
      byId("resultsBody").replaceChildren();
      byId("oracleReference").replaceChildren();
      byId("candidateRank").textContent = "";
      byId("candidateHeading").textContent = "";
      byId("candidateSummary").textContent = "";
      byId("candidateStats").replaceChildren();
      byId("candidateTerms").replaceChildren();
      byId("responseList").replaceChildren();
      byId("candidateWarnings").replaceChildren();
    }

    function showMain() {
      byId("mainEmpty").hidden = true;
      byId("queryCard").hidden = false;
      byId("resultsGrid").hidden = false;
      byId("resultsSection").hidden = false;
      byId("candidateSection").hidden = false;
    }

    function renderBestPositiveSummary(query) {
      const candidates = Object.values(query.grounded.candidates);
      const best = candidates
        .filter((candidate) => candidate.is_positive)
        .sort((left, right) => left.ranks.raw_hierarchical_dense_equal - right.ranks.raw_hierarchical_dense_equal)[0] || null;
      const summary = el("div", "mechanism-summary");
      const tags = el("div", "mechanism-tags");
      tags.appendChild(el("strong", "mechanism-tag", "Retrieval outcome · " + query.grounded.retrieval_outcome));
      tags.appendChild(el("strong", "mechanism-tag", "Mechanism (noncausal) · " + query.grounded.mechanism));
      if (query.grounded.context_risk) {
        tags.appendChild(el("span", "context-risk", "Context-risk flag · surface wording only"));
      }
      summary.appendChild(tags);
      if (best) {
        summary.appendChild(el("p", "", "Best development-positive candidate: " + best.candidate_id));
        summary.appendChild(el("p", "", rankColumns.map(([key, label]) =>
          label + " " + best.ranks[key]).join(" · ")));
        summary.appendChild(el("p", "", "Best-record E5 cosine similarity " + formatMetric(best.best_record_cosine) +
          " · Mean top-10 E5 cosine similarity " + formatMetric(best.mean_top_10_cosine)));
      } else {
        summary.appendChild(el("p", "", "Best development-positive candidate: none in complete development qrels"));
      }
      const cohorts = el("div", "cohort-list");
      (query.grounded.cohorts || []).forEach((cohort) =>
        cohorts.appendChild(el("span", "cohort-tag", cohort)));
      summary.appendChild(cohorts);
      return summary;
    }

    function renderGroundedEvidence(candidate) {
      const details = el("details", "evidence-drawer");
      details.appendChild(el("summary", "", "Nearest historical behavior"));
      details.appendChild(el("p", "section-note",
        "Similarity and substantive output do not establish correctness."));
      candidate.evidence.forEach((item) => {
        const card = el("article", "history-card");
        card.appendChild(el("div", "history-meta",
          "E5 cosine similarity " + formatMetric(item.similarity) +
          " · Historical query " + item.source_query_id + " · " + item.behavior));
        card.appendChild(el("p", "history-question", item.question));
        card.appendChild(el("p", "history-response", item.response));
        details.appendChild(card);
      });
      return details;
    }

    function renderResultHeaders(query) {
      const header = byId("resultsHead");
      header.replaceChildren();
      header.appendChild(el("th", "", query.grounded ? "Final rank" : "Rank"));
      header.appendChild(el("th", "", "Candidate"));
      if (query.grounded) {
        rankColumns.forEach(([_key, label]) =>
          header.appendChild(el("th", "", label)));
        header.appendChild(el("th", "", "Best-record E5 cosine similarity"));
        header.appendChild(el("th", "", "Mean top-10 E5 cosine similarity"));
      }
      header.appendChild(el("th", "", "System score"));
    }

    function appendGroundedCells(row, groundedCandidate) {
      rankColumns.forEach(([key]) =>
        row.appendChild(el("td", "score", groundedCandidate.ranks[key])));
      row.appendChild(el("td", "score", formatMetric(groundedCandidate.best_record_cosine)));
      row.appendChild(el("td", "score", formatMetric(groundedCandidate.mean_top_10_cosine)));
    }

    function renderResults() {
      const query = activeQuery();
      if (!query) return;
      byId("resultsGrid").classList.toggle("grounded", Boolean(query.grounded));
      byId("resultLimit").textContent = query.grounded
        ? "Showing final top " + number(data.meta.candidate_limit) + " plus development-positive candidates in frozen order."
        : "Showing up to " + number(data.meta.candidate_limit) + " candidates for this query.";
      byId("activeQueryId").textContent = query.id;
      byId("queryHeading").textContent = query.text;
      const evaluationBox = byId("queryEvaluation");
      evaluationBox.replaceChildren();
      evaluationBox.hidden = !query.evaluation;
      if (query.evaluation) {
        const idealNdcg = query.evaluation.ideal_ndcg_at_10;
        evaluationBox.appendChild(el("span", "evaluation-pill", "nDCG@10 " + formatMetric(query.evaluation.ndcg_at_10) + (idealNdcg === null ? " · no relevant labels" : " / ideal " + formatMetric(idealNdcg))));
        evaluationBox.appendChild(el("span", "evaluation-pill", "First relevant rank " + (query.evaluation.first_relevant_rank ?? "none")));
        evaluationBox.appendChild(el("span", "evaluation-pill", query.evaluation.relevant_in_top_10 + " of " + query.evaluation.relevant_total + " relevant candidates in top 10"));
      }
      const groundedSummary = byId("groundedSummary");
      groundedSummary.replaceChildren();
      groundedSummary.hidden = !query.grounded;
      if (query.grounded) groundedSummary.appendChild(renderBestPositiveSummary(query));
      if (!state.candidateId && query.results[0]) state.candidateId = query.results[0].candidate_id;
      renderResultHeaders(query);
      const body = byId("resultsBody");
      body.replaceChildren();
      query.results.forEach((result) => {
        const groundedCandidate = query.grounded?.candidates[result.candidate_id] || null;
        const row = el("tr", result.candidate_id === state.candidateId ? "selected" : "");
        row.appendChild(el("td", "score", result.rank));
        const candidateCell = el("td");
        const button = el("button", "candidate-button");
        button.type = "button";
        button.setAttribute("aria-label", "Inspect history for " + result.candidate_id);
        button.setAttribute("aria-pressed", result.candidate_id === state.candidateId ? "true" : "false");
        button.appendChild(el("span", "candidate-name", result.candidate_id));
        if (data.meta.labels_included) {
          const gradeText = result.judged ? "G" + formatGrade(result.grade) : "—";
          const gradeClass = result.judged && Number(result.grade) > 0 ? "relevant" : "not-relevant";
          const badge = el("span", "grade-badge " + gradeClass, gradeText);
          badge.title = result.judged ? "Development ground-truth grade" : "No ground-truth judgment";
          button.appendChild(badge);
        }
        button.addEventListener("click", () => {
          state.candidateId = result.candidate_id;
          renderResults();
          document.querySelector(
            '#resultsBody .candidate-button[aria-pressed="true"]'
          )?.focus();
          renderCandidate();
        });
        candidateCell.appendChild(button);
        row.appendChild(candidateCell);
        if (groundedCandidate) appendGroundedCells(row, groundedCandidate);
        const scoreCell = el("td", "score", formatScore(result.score_text));
        scoreCell.title = "Exact system score: " + result.score_text;
        scoreCell.setAttribute("aria-label", "Exact system score " + result.score_text);
        row.appendChild(scoreCell);
        body.appendChild(row);
        if (groundedCandidate && groundedCandidate.evidence.length) {
          const evidenceRow = el("tr", "evidence-row");
          const evidenceCell = el("td");
          evidenceCell.colSpan = row.childElementCount;
          evidenceCell.appendChild(renderGroundedEvidence(groundedCandidate));
          evidenceRow.appendChild(evidenceCell);
          body.appendChild(evidenceRow);
        }
      });
      const oracle = byId("oracleReference");
      oracle.replaceChildren();
      const idealCandidates = query.evaluation?.ideal_candidates || [];
      oracle.hidden = !idealCandidates.length;
      if (idealCandidates.length) {
        oracle.appendChild(el("span", "oracle-title mono", "All relevant LLMs from development labels · ordered by grade"));
        const oracleList = el("div", "oracle-list");
        idealCandidates.forEach((candidate) => {
          const placement = candidate.system_rank === null ? "not ranked" : "system rank " + candidate.system_rank;
          const gradeClass = Number(candidate.grade) >= 2 ? "grade-high" : "grade-positive";
          const placementClass = candidate.system_rank !== null && candidate.system_rank <= 10 ? "retrieved" : "missed";
          oracleList.appendChild(el("span", "oracle-candidate " + gradeClass + " " + placementClass, candidate.candidate_id + " · G" + formatGrade(candidate.grade) + " · " + placement));
        });
        oracle.appendChild(oracleList);
        oracle.appendChild(el("span", "oracle-legend", "Green = highest relevance · gold = relevant · blue edge = retrieved in top 10 · red edge = missed from top 10"));
      }
    }

    function addStat(label, value) {
      const card = el("div", "stat");
      card.appendChild(el("span", "stat-label", label));
      card.appendChild(el("span", "stat-value", value));
      byId("candidateStats").appendChild(card);
    }

    function renderCandidate() {
      const query = activeQuery();
      const result = activeResult();
      if (!query || !result) return;
      const profile = data.profiles[result.candidate_id] || null;
      const gradeSummary = data.meta.labels_included
        ? " · " + (result.judged ? "Ground-truth grade " + formatGrade(result.grade) : "Unjudged")
        : "";
      byId("candidateRank").textContent = "Rank " + result.rank + " · System score " + formatScore(result.score_text) + gradeSummary;
      byId("candidateRank").title = "Exact system score: " + result.score_text;
      byId("candidateRank").setAttribute("aria-label", "Rank " + result.rank + ", exact system score " + result.score_text);
      byId("candidateHeading").textContent = result.candidate_id;
      byId("candidateSummary").textContent = profile && profile.profile_summary ? profile.profile_summary : "No candidate profile is available; the ranking result is still shown.";
      byId("candidateStats").replaceChildren();
      if (profile) {
        addStat("Attempted-query share", percent(profile.coverage.attempt_share));
        addStat("Exact default share", percent(profile.coverage.exact_refusal_share));
        addStat("Median attempted response", numeric(profile.attempt_style.word_count_median) ? number(profile.attempt_style.word_count_median) + " words" : "—");
        addStat("Discovery queries observed", number(profile.history.record_count));
        addStat("Usable token metadata", percent(profile.metadata_quality.token_logprob_usable_share));
      } else {
        addStat("Profile", "Not available");
      }

      byId("candidateTerms").replaceChildren();
      const terms = profile && Array.isArray(profile.attempted_query_terms) ? profile.attempted_query_terms : [];
      terms.forEach((term) => {
        const label = term && typeof term === "object" ? term.term : term;
        if (label) byId("candidateTerms").appendChild(el("span", "term", label));
      });
      byId("termSection").hidden = terms.length === 0;

      const responseList = byId("responseList");
      responseList.replaceChildren();
      byId("responseSection").hidden = Boolean(query.grounded);
      if (!query.grounded) {
        const responses = (((data.responses || {})[query.id] || {})[result.candidate_id] || []);
        responses.forEach((item) => {
          const card = el("article", "history-card");
          card.appendChild(el("p", "history-question", item.source_query));
          card.appendChild(el("p", "history-response", item.response));
          card.appendChild(el("div", "history-meta", "Historical query " + item.source_query_id + " · lexical overlap " + percent(item.similarity)));
          responseList.appendChild(card);
        });
        if (!responses.length) responseList.appendChild(el("div", "empty", "No usable related response was embedded for this candidate."));
      }

      byId("candidateWarnings").replaceChildren();
      const warnings = profile && Array.isArray(profile.warnings) ? profile.warnings : [];
      warnings.forEach((warning) => byId("candidateWarnings").appendChild(el("li", "", warning)));
      byId("warningSection").hidden = warnings.length === 0;
    }

    function renderAll() {
      const queries = visibleQueries();
      synchronizeSelection(queries);
      renderQueryList(queries);
      if (!queries.length) {
        renderEmptyMain();
        return;
      }
      showMain();
      renderResults();
      renderCandidate();
    }

    byId("querySearch").addEventListener("input", (event) => {
      state.search = event.target.value;
      renderAll();
    });
    byId("evaluationFilter").addEventListener("change", (event) => {
      state.evaluationFilter = event.target.value;
      renderAll();
    });
    byId("queryToggle").addEventListener("click", () => {
      setQueryPanelCollapsed(document.querySelector(".query-panel").dataset.collapsed !== "true");
    });
    mobileQueryLayout.addEventListener("change", () => setQueryPanelCollapsed(mobileQueryLayout.matches));
    byId("evaluationControl").hidden = !data.meta.labels_included;
    configureEvaluationFilters();
    byId("evaluationFilter").value = state.evaluationFilter;
    const initialQuery = activeQuery();
    if (initialQuery && initialQuery.results[0]) state.candidateId = initialQuery.results[0].candidate_id;
    setQueryPanelCollapsed(mobileQueryLayout.matches);
    renderSummary();
    renderAll();
  </script>
</body>
</html>
'''


def render_html(payload: Mapping[str, Any]) -> str:
    title = str(payload["meta"]["title"])
    escaped_title = (
        title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    before_title, title_marker, after_title = HTML_TEMPLATE.partition(
        "__EXPLORER_TITLE__"
    )
    before_payload, payload_marker, after_payload = after_title.partition(
        "__EXPLORER_PAYLOAD__"
    )
    if not title_marker or not payload_marker:
        raise RuntimeError("explorer template is missing a required marker")
    return (
        before_title
        + escaped_title
        + before_payload
        + safe_json(payload)
        + after_payload
    )


def validate_output_path(
    output_path: Path,
    *,
    input_paths: Iterable[Path | None],
) -> None:
    if output_path.suffix.lower() != ".html":
        raise ExplorerInputError(
            f"output path must end in .html: {output_path}"
        )
    if output_path.is_symlink():
        raise ExplorerInputError(f"output path must not be a symlink: {output_path}")
    if output_path.exists() and not output_path.is_file():
        raise ExplorerInputError(f"output path is not a regular file: {output_path}")
    resolved_output = output_path.resolve()
    for input_path in input_paths:
        if input_path is not None and resolved_output == input_path.resolve():
            raise ExplorerInputError(
                f"output path would overwrite a source artifact: {output_path}"
            )


def write_html_atomically(output_path: Path, html: str) -> None:
    temporary_path: Path | None = None
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(html)
        temporary_path.replace(output_path)
    except OSError as error:
        raise ExplorerInputError(f"could not write explorer: {output_path}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    input_path, run_dir = choose_input(
        root,
        run_dir=args.run_dir,
        explicit_input=args.input,
    )
    verify_input_hash(run_dir, input_path)
    if input_path.suffix.lower() == ".trec":
        rows, schema = load_trec_run(input_path)
    else:
        rows, schema = load_predictions_tsv(
            input_path,
            query_column=args.query_column,
            candidate_column=args.candidate_column,
            rank_column=args.rank_column,
            score_column=args.score_column,
        )
    grouped, warnings = validate_and_group_rows(
        rows,
        top_candidates=args.top_candidates,
        retain_all=True,
    )
    evidence_path = resolve_path(args.evidence_bundle, root)
    evidence_bundle = (
        load_evidence_bundle(evidence_path, grouped)
        if evidence_path is not None
        else None
    )
    if evidence_bundle is not None and args.discovery is not None:
        raise ExplorerInputError(
            "--discovery cannot be combined with --evidence-bundle"
        )

    qrels_path = infer_qrels_path(
        root,
        run_dir,
        args.qrels,
        no_labels=args.no_labels,
    )
    evaluation: dict[str, Any] | None = None
    if qrels_path is not None:
        qrels = load_qrels(qrels_path)
        evaluation = build_evaluation(rows, qrels)
        attach_qrel_labels(grouped, qrels)
    else:
        qrels = {}
    if evidence_bundle is not None and qrels_path is None:
        raise ExplorerInputError("evidence bundle requires complete qrels")

    query_path = infer_query_path(root, run_dir, args.queries)
    queries, query_order = load_queries(query_path)
    missing_queries = sorted(set(grouped) - set(queries))
    if missing_queries:
        preview = ", ".join(missing_queries[:5])
        raise ExplorerInputError(
            f"query text is missing for {len(missing_queries)} prediction queries: "
            f"{preview}; pass the matching --queries file"
        )
    selected_query_texts = {
        query_id: queries[query_id] for query_id in grouped
    }
    if evidence_bundle is not None:
        if evaluation is None:
            raise ExplorerInputError(
                "evidence bundle requires complete qrels"
            )
        grouped = validate_grounded_contract(
            evidence_bundle,
            grouped=grouped,
            qrels=qrels,
            queries=queries,
            top_candidates=args.top_candidates,
            evaluation=evaluation,
        )
    else:
        grouped = {
            query_id: select_display_rows(
                values, top_candidates=args.top_candidates
            )
            for query_id, values in grouped.items()
        }

    profile_path = resolve_path(args.profiles, root)
    if args.profiles is not None and (profile_path is None or not profile_path.is_file()):
        raise ExplorerInputError(f"candidate profile file does not exist: {profile_path}")
    if profile_path is None:
        default_profile = root / "reports/trec_2026_llm_profiles.json"
        profile_path = default_profile if default_profile.is_file() else None
    profiles, profile_version = load_profiles(profile_path)

    related_responses: dict[str, dict[str, list[dict[str, Any]]]] = {}
    source_count = 0
    discovery_path: Path | None = None
    if not args.no_responses and evidence_bundle is None:
        discovery_path = resolve_path(args.discovery, root)
        if args.discovery is not None and (
            discovery_path is None or not discovery_path.is_file()
        ):
            raise ExplorerInputError(f"discovery file does not exist: {discovery_path}")
        if discovery_path is None:
            default_discovery = root / "data/dev/llm_discovery_data.json"
            discovery_path = default_discovery if default_discovery.is_file() else None
        if discovery_path is not None:
            candidates_by_query = {
                query_id: [str(row["candidate_id"]) for row in values]
                for query_id, values in grouped.items()
            }
            try:
                related_responses, source_count = build_related_responses(
                    discovery_path,
                    query_texts=selected_query_texts,
                    candidates_by_query=candidates_by_query,
                    per_candidate=args.responses_per_candidate,
                    maximum_chars=args.max_response_chars,
                )
            except (OSError, json.JSONDecodeError) as error:
                raise ExplorerInputError(
                    f"could not read discovery responses: {discovery_path}"
                ) from error

    payload = build_payload(
        root=root,
        input_path=input_path,
        query_path=query_path,
        qrels_path=qrels_path,
        profile_path=profile_path,
        discovery_path=discovery_path,
        title=args.title,
        grouped=grouped,
        schema=schema,
        queries=queries,
        query_order=query_order,
        profiles=profiles,
        profile_version=profile_version,
        related_responses=related_responses,
        source_count=source_count,
        warnings=warnings,
        candidate_limit=args.top_candidates,
        evaluation=evaluation,
        evidence_bundle=evidence_bundle,
    )
    output_path = resolve_path(args.output, root) or run_dir / "query_explorer.html"
    validate_output_path(
        output_path,
        input_paths=(
            input_path,
            query_path,
            qrels_path,
            profile_path,
            discovery_path,
            evidence_path,
        ),
    )
    write_html_atomically(output_path, render_html(payload))

    print(output_path.resolve())
    print(
        json.dumps(
            {
                "query_count": payload["meta"]["query_count"],
                "displayed_result_count": payload["meta"]["displayed_result_count"],
                "unique_candidates_shown": payload["meta"]["candidate_count"],
                "candidate_limit": payload["meta"]["candidate_limit"],
                "score_field": payload["meta"]["score_field"],
                "profiles_included": bool(payload["meta"]["profile_count"]),
                "responses_included": bool(payload["meta"]["embedded_response_count"]),
                "labels_included": bool(payload["meta"]["labels_included"]),
                "macro_ndcg_at_10": payload["meta"]["macro_ndcg_at_10"],
                "mean_reciprocal_rank": payload["meta"]["mean_reciprocal_rank"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExplorerInputError as error:
        print(f"error: {error}", file=__import__("sys").stderr)
        raise SystemExit(2) from error
