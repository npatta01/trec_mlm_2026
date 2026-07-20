#!/usr/bin/env python3
"""Build a deterministic clustered sample for reference-answer validation."""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from trec_mlm.artifact_io import sha256_file
from trec_mlm.reference_data import (
    EXPECTED_PERPLEXITY_SHA256,
    QuestionRecord,
    load_perplexity_questions,
)
from trec_mlm.reference_sampling import (
    build_clustered_sample,
    historical_difficulty_features,
)


DEFAULT_ENCODER = "sentence-transformers/all-mpnet-base-v2"
DEFAULT_REVISION = "e8c3b32edf5434bc2275fc9bab85f82640a19130"


def _normalized_question(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"\w+", normalized, flags=re.UNICODE))


def _analogue_matches(
    path: Path | None, records: Sequence[QuestionRecord]
) -> tuple[set[str], dict[str, Any]]:
    if path is None:
        return set(), {"matched": [], "unmatched": [], "ambiguous": []}
    index: dict[str, list[str]] = defaultdict(list)
    for record in records:
        index[_normalized_question(record.question)].append(record.question_id)
    matched: set[str] = set()
    diagnostics: dict[str, list[Any]] = {
        "matched": [],
        "unmatched": [],
        "ambiguous": [],
    }
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "nearest_discovery_query_text" not in reader.fieldnames:
            raise ValueError(
                "analogue TSV must contain nearest_discovery_query_text"
            )
        for row in reader:
            text = (row.get("nearest_discovery_query_text") or "").strip()
            ids = index.get(_normalized_question(text), [])
            if len(ids) == 1:
                matched.add(ids[0])
                diagnostics["matched"].append(
                    {"nearest_discovery_query_text": text, "question_id": ids[0]}
                )
            elif not ids:
                diagnostics["unmatched"].append(text)
            else:
                diagnostics["ambiguous"].append(
                    {"nearest_discovery_query_text": text, "question_ids": sorted(ids)}
                )
    return matched, diagnostics


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_assignments(
    path: Path,
    records: Sequence[QuestionRecord],
    labels: np.ndarray,
    selected: Sequence[dict[str, Any]],
) -> None:
    selected_by_id = {row["question_id"]: row for row in selected}
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            delimiter="\t",
            lineterminator="\n",
            fieldnames=(
                "question_id",
                "cluster_id",
                "selected",
                "selection_reasons",
                "challenge_score",
                "strata",
            ),
        )
        writer.writeheader()
        for record, cluster_id in zip(records, labels, strict=True):
            chosen = selected_by_id.get(record.question_id)
            writer.writerow(
                {
                    "question_id": record.question_id,
                    "cluster_id": int(cluster_id),
                    "selected": str(chosen is not None).lower(),
                    "selection_reasons": ",".join(
                        chosen["selection_reasons"] if chosen else ()
                    ),
                    "challenge_score": chosen["challenge_score"] if chosen else "",
                    "strata": ",".join(chosen["strata"] if chosen else ()),
                }
            )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perplexity-cache", type=Path, required=True)
    parser.add_argument(
        "--expected-perplexity-sha256", default=EXPECTED_PERPLEXITY_SHA256
    )
    parser.add_argument("--discovery-responses", type=Path, required=True)
    parser.add_argument("--analogue-cases", type=Path)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--clusters", type=int, default=192)
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = load_perplexity_questions(
        args.perplexity_cache,
        expected_sha256=args.expected_perplexity_sha256,
    )
    history = historical_difficulty_features(args.discovery_responses)
    record_ids = {record.question_id for record in records}
    if set(history) != record_ids:
        raise ValueError(
            "discovery response IDs do not match the Perplexity cache: "
            f"missing={sorted(record_ids - set(history))!r} "
            f"extra={sorted(set(history) - record_ids)!r}"
        )
    known_failure_ids, analogue_diagnostics = _analogue_matches(
        args.analogue_cases, records
    )

    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import KMeans

    encoder = SentenceTransformer(
        args.encoder, revision=args.revision, device="cpu"
    )
    embeddings = np.asarray(
        encoder.encode(
            [record.question for record in records],
            convert_to_numpy=True,
            show_progress_bar=True,
        ),
        dtype=np.float32,
    )
    if embeddings.ndim != 2 or len(embeddings) != len(records):
        raise ValueError("encoder returned embeddings with an unexpected shape")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if not np.isfinite(embeddings).all() or np.any(norms == 0):
        raise ValueError("encoder returned non-finite or zero-length embeddings")
    embeddings = embeddings / norms
    labels = KMeans(
        n_clusters=args.clusters,
        random_state=args.seed,
        n_init=10,
    ).fit_predict(embeddings)

    required_ids = {
        record.question_id for record in records if record.perplexity_unresolved
    }
    selected = build_clustered_sample(
        records,
        embeddings,
        labels,
        sample_size=args.sample_size,
        seed=args.seed,
        required_ids=required_ids,
        known_failure_ids=known_failure_ids,
        historical_features=history,
    )
    records_by_id = {record.question_id: record for record in records}
    for row in selected:
        record = records_by_id[row["question_id"]]
        row.update(
            {
                "perplexity_answer": record.perplexity_answer,
                "perplexity_answer_no_citations": record.perplexity_answer_no_citations,
                "perplexity_citations": list(record.perplexity_citations),
                "perplexity_unresolved": record.perplexity_unresolved,
                "historical_difficulty": dict(history[record.question_id]),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sources: dict[str, Any] = {
        "perplexity_cache": {
            "path": str(args.perplexity_cache),
            "sha256": sha256_file(args.perplexity_cache),
        },
        "discovery_responses": {
            "path": str(args.discovery_responses),
            "sha256": sha256_file(args.discovery_responses),
        },
    }
    if args.analogue_cases is not None:
        sources["analogue_cases"] = {
            "path": str(args.analogue_cases),
            "sha256": sha256_file(args.analogue_cases),
        }
    manifest = {
        "schema_version": "reference-validation-sample-v1",
        "record_count": len(records),
        "sample_size": len(selected),
        "cluster_count": len(set(int(label) for label in labels)),
        "seed": args.seed,
        "encoder": {"id": args.encoder, "revision": args.revision},
        "historical_difficulty_thresholds": history.thresholds,
        "sources": sources,
        "analogue_diagnostics": analogue_diagnostics,
        "questions": selected,
    }
    _atomic_json(args.output_dir / "sample_manifest.json", manifest)
    _write_assignments(
        args.output_dir / "cluster_assignments.tsv", records, labels, selected
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
