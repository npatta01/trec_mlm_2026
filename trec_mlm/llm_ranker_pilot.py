"""Frozen, qrel-isolated selection inputs for compact LLM-ranker pilots."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from trec_mlm.label_experiment import load_candidate_manifest
from trec_mlm.query_knn import load_complete_qrels, load_queries_tsv
from trec_mlm.runio import rankings_from_run


STAGE1_QUERY_IDS = (
    "56fc3b342603e7140040a037",
    "57265eeaf1498d1400e8dd76",
    "570a34484103511400d595c8",
    "57298ef11d04691400779530",
    "5726eb785951b619008f8276",
    "5729068e1d04691400778f8f",
)

FROZEN_INPUT_SHA256: Mapping[str, str] = {
    "queries": "862ff85381b8055cd927ada25c56666594ab1ec3c21e182beb20c58f6c637f3c",
    "qrels": "3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35",
    "candidates": "4a770b919024dffd1cffe85edd2f0219aeec31cdbb2a96d1005861fa1bf03f91",
    "qwen_run": "857d279a8466cabe66069acb4c47edcaba3a74c100eb6202e6d474834cfa7408",
}

DEFAULT_QUERY_PATH = Path("data/dev/llm_dev_data.tsv")
DEFAULT_QRELS_PATH = Path("data/dev/llm_dev_qrels.txt")
DEFAULT_CANDIDATE_PATH = Path("reports/trec_2026_discovery_profile_audit.json")
DEFAULT_QWEN_RUN_PATH = Path(
    "artifacts/label_sweep_diverse/qwen3_embedding_4b/"
    "expert_capability/run1/run.trec"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/llm_label_ranker/pilot/stage1")

_SELECTION_TOP_LEVEL_KEYS = frozenset(
    {"stage", "seed", "input_sha256", "questions"}
)
_JUDGMENT_TOP_LEVEL_KEYS = frozenset({"stage", "input_sha256", "questions"})
_SELECTION_ENTRY_KEYS = frozenset({"query_id", "question", "candidate_ids"})
_JUDGMENT_ENTRY_KEYS = frozenset({"query_id", "grades", "roles"})
_ROLE_KEYS = ("relevant", "confounder", "negative")


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_input_hashes(
    actual: object,
    *,
    expected_keys: tuple[str, ...],
) -> None:
    if not isinstance(actual, Mapping):
        raise ValueError("pilot input_sha256 must be an object")
    expected = {key: FROZEN_INPUT_SHA256[key] for key in expected_keys}
    if set(actual) != set(expected):
        raise ValueError("pilot input SHA256 keys changed")
    for key, expected_digest in expected.items():
        if actual.get(key) != expected_digest:
            raise ValueError(f"{key} input SHA256 changed")


def validate_pilot_selection(
    payload: Mapping[str, Any],
    *,
    expected_stage: int,
    expected_candidate_count: int,
) -> dict[str, Any]:
    """Validate a qrel-free selection or its separately sealed judgments."""

    if not isinstance(payload, Mapping):
        raise ValueError("pilot manifest must be an object")
    if expected_stage <= 0 or expected_candidate_count <= 0:
        raise ValueError("expected stage and candidate count must be positive")
    if payload.get("stage") != expected_stage:
        raise ValueError(f"pilot manifest stage must be {expected_stage}")

    questions = payload.get("questions")
    if not isinstance(questions, list) or not questions:
        raise ValueError("pilot manifest questions must be a non-empty list")
    if not all(isinstance(entry, Mapping) for entry in questions):
        raise ValueError("pilot manifest question entries must be objects")

    first_keys = frozenset(questions[0])
    if first_keys == _SELECTION_ENTRY_KEYS:
        manifest_kind = "selection"
        if frozenset(payload) != _SELECTION_TOP_LEVEL_KEYS:
            raise ValueError("qrel-free selection has unexpected fields")
        if payload.get("seed") != 2026:
            raise ValueError("pilot selection seed must be 2026")
        _require_exact_input_hashes(
            payload.get("input_sha256"),
            expected_keys=("queries", "candidates"),
        )
    elif first_keys == _JUDGMENT_ENTRY_KEYS:
        manifest_kind = "judgments"
        if frozenset(payload) != _JUDGMENT_TOP_LEVEL_KEYS:
            raise ValueError("sealed judgments have unexpected fields")
        _require_exact_input_hashes(
            payload.get("input_sha256"),
            expected_keys=("qrels", "qwen_run"),
        )
    else:
        raise ValueError("pilot manifest question schema is invalid")

    query_ids: list[str] = []
    for entry in questions:
        expected_entry_keys = (
            _SELECTION_ENTRY_KEYS
            if manifest_kind == "selection"
            else _JUDGMENT_ENTRY_KEYS
        )
        if frozenset(entry) != expected_entry_keys:
            raise ValueError(f"pilot {manifest_kind} question schema is invalid")
        query_id = entry.get("query_id")
        if not isinstance(query_id, str) or not query_id:
            raise ValueError("pilot query IDs must be non-empty strings")
        query_ids.append(query_id)

        if manifest_kind == "selection":
            question = entry.get("question")
            candidate_ids = entry.get("candidate_ids")
            if not isinstance(question, str) or not question:
                raise ValueError("pilot question text must be non-empty")
            if not isinstance(candidate_ids, list) or any(
                not isinstance(item, str) or not item for item in candidate_ids
            ):
                raise ValueError("pilot candidate IDs must be non-empty strings")
            if (
                len(candidate_ids) != expected_candidate_count
                or len(set(candidate_ids)) != expected_candidate_count
            ):
                raise ValueError(
                    f"pilot questions require exactly {expected_candidate_count} "
                    "unique candidates"
                )
            if candidate_ids != sorted(candidate_ids):
                raise ValueError("pilot candidate IDs must be sorted")
            continue

        grades = entry.get("grades")
        roles = entry.get("roles")
        if not isinstance(grades, Mapping) or not isinstance(roles, Mapping):
            raise ValueError("sealed grades and roles must be objects")
        if set(roles) != set(_ROLE_KEYS):
            raise ValueError("sealed judgment roles changed")
        role_lists: dict[str, list[str]] = {}
        for role in _ROLE_KEYS:
            values = roles.get(role)
            if not isinstance(values, list) or any(
                not isinstance(item, str) or not item for item in values
            ):
                raise ValueError(f"sealed {role} roles must be candidate ID lists")
            if len(values) != len(set(values)):
                raise ValueError(f"sealed {role} role contains duplicates")
            role_lists[role] = values
        assigned = [item for role in _ROLE_KEYS for item in role_lists[role]]
        if len(assigned) != len(set(assigned)):
            raise ValueError("sealed judgment roles overlap")
        if len(assigned) != expected_candidate_count:
            raise ValueError(
                f"pilot questions require exactly {expected_candidate_count} "
                "unique candidates"
            )
        if len(role_lists["relevant"]) not in (1, 2):
            raise ValueError("pilot questions require one or two relevant candidates")
        if len(role_lists["confounder"]) != 4:
            raise ValueError("pilot questions require exactly four confounders")
        if set(grades) != set(assigned):
            raise ValueError("sealed grades do not match role candidates")
        if any(
            not isinstance(grade, (int, float))
            for grade in grades.values()
        ):
            raise ValueError("sealed grades must be numeric")
        if any(grades[item] <= 0 for item in role_lists["relevant"]):
            raise ValueError("sealed relevant candidates must have positive grades")
        if any(
            grades[item] != 0
            for role in ("confounder", "negative")
            for item in role_lists[role]
        ):
            raise ValueError("sealed nonrelevant roles must have zero grades")

    if len(query_ids) != len(set(query_ids)):
        raise ValueError("pilot query IDs must be unique")
    if expected_stage == 1 and tuple(query_ids) != STAGE1_QUERY_IDS:
        raise ValueError("stage 1 fixed query IDs changed")

    return {
        "valid": True,
        "stage": expected_stage,
        "question_count": len(questions),
        "candidate_count": expected_candidate_count,
        "pair_count": len(questions) * expected_candidate_count,
        "manifest_kind": manifest_kind,
    }


def build_stage1_selection(
    *,
    query_path: str | Path,
    qrels_path: str | Path,
    candidate_path: str | Path,
    qwen_run_path: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the frozen Stage 1 prompt inventory and sealed judgments."""

    queries = load_queries_tsv(query_path)
    missing_fixed_queries = sorted(set(STAGE1_QUERY_IDS) - set(queries))
    if missing_fixed_queries:
        raise ValueError(f"missing fixed query IDs: {missing_fixed_queries!r}")
    qrels = load_complete_qrels(qrels_path, expected_query_ids=queries)
    candidate_ids = load_candidate_manifest(candidate_path)
    qwen_rankings = rankings_from_run(qwen_run_path)

    qrel_candidates = set(next(iter(qrels.values())))
    if qrel_candidates != set(candidate_ids):
        raise ValueError("candidate manifest does not match qrel candidate universe")
    if set(qwen_rankings) != set(queries):
        raise ValueError("Qwen run query universe does not match query input")
    for query_id, ranking in qwen_rankings.items():
        if len(ranking) != len(candidate_ids) or set(ranking) != set(candidate_ids):
            raise ValueError(
                f"Qwen run candidate universe does not match for query {query_id!r}"
            )

    entries: list[dict[str, Any]] = []
    judgment_entries: list[dict[str, Any]] = []
    for query_id in STAGE1_QUERY_IDS:
        grades = qrels[query_id]
        relevant = [
            candidate_id
            for candidate_id, grade in sorted(
                grades.items(), key=lambda item: (-item[1], item[0])
            )
            if grade > 0
        ][:2]
        if not relevant:
            raise ValueError(f"fixed query {query_id!r} has no relevant candidates")
        confounders = [
            candidate_id
            for candidate_id in qwen_rankings[query_id]
            if grades[candidate_id] == 0 and candidate_id not in relevant
        ][:4]
        if len(confounders) != 4:
            raise ValueError(f"fixed query {query_id!r} has fewer than four confounders")
        excluded = set(relevant) | set(confounders)
        negative_pool = [
            candidate_id
            for candidate_id in candidate_ids
            if grades[candidate_id] == 0 and candidate_id not in excluded
        ]
        negatives = sorted(
            negative_pool,
            key=lambda candidate_id: hashlib.sha256(
                f"2026|{query_id}|{candidate_id}".encode()
            ).digest(),
        )[: 12 - len(excluded)]
        selected = sorted([*relevant, *confounders, *negatives])
        if len(selected) != 12 or len(set(selected)) != 12:
            raise ValueError(
                f"fixed query {query_id!r} does not produce 12 unique candidates"
            )
        entries.append(
            {
                "query_id": query_id,
                "question": queries[query_id],
                "candidate_ids": selected,
            }
        )
        judgment_entries.append(
            {
                "query_id": query_id,
                "grades": {
                    candidate_id: grades[candidate_id] for candidate_id in selected
                },
                "roles": {
                    "relevant": relevant,
                    "confounder": confounders,
                    "negative": negatives,
                },
            }
        )

    selection = {
        "stage": 1,
        "seed": 2026,
        "input_sha256": {
            "queries": _sha256(query_path),
            "candidates": _sha256(candidate_path),
        },
        "questions": entries,
    }
    judgments = {
        "stage": 1,
        "input_sha256": {
            "qrels": _sha256(qrels_path),
            "qwen_run": _sha256(qwen_run_path),
        },
        "questions": judgment_entries,
    }
    validate_pilot_selection(
        selection,
        expected_stage=1,
        expected_candidate_count=12,
    )
    validate_pilot_selection(
        judgments,
        expected_stage=1,
        expected_candidate_count=12,
    )
    return selection, judgments


def write_stage1_selection(
    selection: Mapping[str, Any],
    judgments: Mapping[str, Any],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """Write both compact Stage 1 manifests as stable sorted JSON."""

    validate_pilot_selection(
        selection,
        expected_stage=1,
        expected_candidate_count=12,
    )
    validate_pilot_selection(
        judgments,
        expected_stage=1,
        expected_candidate_count=12,
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    selection_path = destination / "selection.json"
    judgments_path = destination / "selection_judgments.json"
    selection_path.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    judgments_path.write_text(
        json.dumps(judgments, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return selection_path, judgments_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATE_PATH)
    parser.add_argument("--qwen-run", type=Path, default=DEFAULT_QWEN_RUN_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Generate the frozen Stage 1 prompt and sealed judgment manifests."""

    args = _build_parser().parse_args(argv)
    selection, judgments = build_stage1_selection(
        query_path=args.queries,
        qrels_path=args.qrels,
        candidate_path=args.candidates,
        qwen_run_path=args.qwen_run,
    )
    selection_path, judgments_path = write_stage1_selection(
        selection,
        judgments,
        args.output_dir,
    )
    summary = validate_pilot_selection(
        selection,
        expected_stage=1,
        expected_candidate_count=12,
    )
    print(
        json.dumps(
            {
                **summary,
                "selection_path": str(selection_path),
                "judgments_path": str(judgments_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
