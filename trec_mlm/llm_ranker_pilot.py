"""Frozen, qrel-isolated selection inputs for compact LLM-ranker pilots."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
from typing import Any

from trec_mlm.artifact_manifest import validate_pilot_smoke_receipt
from trec_mlm.label_experiment import load_candidate_manifest
from trec_mlm.llm_ranker import (
    SYSTEM_PROMPT,
    _capability,
    expected_grade_score,
    format_pointwise_messages,
)
from trec_mlm.llm_ranker_inference import (
    MODEL_DTYPE,
    MODEL_ID,
    MODEL_REVISION,
    run_listwise_messages,
)
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
DEFAULT_SMOKE_RECEIPT_PATH = Path("artifacts/llm_label_ranker/smoke/smoke.json")

_PILOT_CORE_FILES = frozenset(
    {
        "config.json",
        "selection.json",
        "prompts.jsonl",
        "raw_outputs.jsonl",
        "scores.tsv",
        "rankings.json",
        "runtime.json",
    }
)
_PILOT_CONFIG_KEYS = frozenset(
    {
        "version",
        "qrel_free_phase",
        "method",
        "stage",
        "seed",
        "model_id",
        "model_revision",
        "dtype",
        "code_revision",
        "worktree_status",
        "query_path",
        "candidate_path",
        "smoke_receipt_path",
        "input_sha256",
        "smoke_receipt_sha256",
    }
)
_PILOT_GRADE_TOKEN_IDS = {0: 15, 1: 16, 2: 17}
_PILOT_STAGE_SHAPES = {1: (6, 12), 2: (12, 20)}
_COMMIT_SHA_RE = re.compile(r"[0-9a-f]{40}")
_RUNTIME_INPUT_FIELDS = frozenset(
    {
        "model_id",
        "revision",
        "dtype",
        "seed",
        "vllm_version",
        "torch_version",
        "hip_version",
        "code_revision",
        "worktree_status",
        "query_path",
        "candidate_path",
        "smoke_receipt_path",
    }
)
_OPTIONAL_RUNTIME_INPUT_FIELDS = frozenset(
    {"device_properties", "grade_token_ids", "inference_stats"}
)

_SELECTION_TOP_LEVEL_KEYS = frozenset(
    {"stage", "seed", "input_sha256", "questions"}
)
_JUDGMENT_TOP_LEVEL_KEYS = frozenset({"stage", "input_sha256", "questions"})
_SELECTION_ENTRY_KEYS = frozenset({"query_id", "question", "candidate_ids"})
_JUDGMENT_ENTRY_KEYS = frozenset({"query_id", "grades", "roles"})
_ROLE_KEYS = ("relevant", "confounder", "negative")


@dataclass(frozen=True, slots=True)
class PilotPhaseAResult:
    """One complete compact ranking result fixed before qrel evaluation."""

    selection: dict[str, Any]
    method: str
    prompts: tuple[dict[str, Any], ...]
    raw_outputs: tuple[dict[str, Any], ...]
    scores: dict[str, dict[str, float]]
    rankings: dict[str, list[str]]
    runtime_metadata: dict[str, Any]
    config: dict[str, Any]

    @property
    def raw_records(self) -> tuple[dict[str, Any], ...]:
        """Compatibility name used by the canonical ranker result."""

        return self.raw_outputs


def pilot_handles(candidate_ids: Sequence[str]) -> dict[str, str]:
    """Assign stable two-digit handles to a twelve- or twenty-candidate pilot."""

    ordered = sorted(candidate_ids)
    if len(ordered) not in (12, 20) or len(set(ordered)) != len(ordered):
        raise ValueError("pilot ranking requires 12 or 20 unique candidates")
    return {f"C{index:02d}": candidate for index, candidate in enumerate(ordered)}


def _pilot_hash_order(seed: int, query_id: str, candidate_id: str) -> bytes:
    return hashlib.sha256(f"{seed}|{query_id}|{candidate_id}".encode()).digest()


def balanced_pilot_orders(
    *,
    query_id: str,
    candidate_ids: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Return three deterministic orders balanced across three pilot regions."""

    candidates = tuple(candidate_ids)
    if len(candidates) not in (12, 20) or len(set(candidates)) != len(candidates):
        raise ValueError("pilot ranking requires 12 or 20 unique candidates")
    base = sorted(
        candidates,
        key=lambda candidate: _pilot_hash_order(2026, query_id, candidate),
    )
    quotient, remainder = divmod(len(base), 3)
    group_sizes = [quotient + (region < remainder) for region in range(3)]
    groups: list[list[str]] = []
    offset = 0
    for size in group_sizes:
        groups.append(base[offset : offset + size])
        offset += size

    orders: list[tuple[str, ...]] = []
    for rotation, seed in enumerate((2026, 2027, 2028)):
        cycled = tuple(groups[(rotation + region) % 3] for region in range(3))
        orders.append(
            tuple(
                candidate
                for group in cycled
                for candidate in sorted(
                    group,
                    key=lambda item: _pilot_hash_order(seed, query_id, item),
                )
            )
        )
    return tuple(orders)  # type: ignore[return-value]


def format_pilot_listwise_messages(
    question: str,
    *,
    ordered_candidate_ids: Sequence[str],
    handle_to_candidate: Mapping[str, str],
) -> list[dict[str, str]]:
    """Create one complete-ranking prompt for a twelve- or twenty-label pilot."""

    if not question.strip():
        raise ValueError("question must be non-empty")
    candidate_count = len(ordered_candidate_ids)
    if (
        candidate_count not in (12, 20)
        or len(set(ordered_candidate_ids)) != candidate_count
    ):
        raise ValueError("pilot listwise prompt requires 12 or 20 unique candidates")
    candidate_to_handle = {
        candidate: handle for handle, candidate in handle_to_candidate.items()
    }
    if (
        len(candidate_to_handle) != candidate_count
        or set(ordered_candidate_ids) != set(candidate_to_handle)
    ):
        raise ValueError("pilot listwise order and handle inventory disagree")
    rows = [
        f"{candidate_to_handle[candidate]} | {candidate} | {_capability(candidate)}"
        for candidate in ordered_candidate_ids
    ]
    user = (
        f"Rank all {candidate_count} experts from most to least likely to answer the "
        "question correctly. Return only a JSON array containing every handle exactly "
        "once, best first.\n\n"
        f"<question>\n{question.strip()}\n</question>\n"
        "<experts>\n"
        + "\n".join(rows)
        + "\n</experts>"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def parse_complete_pilot_ranking(
    raw_output: str,
    *,
    valid_handles: Mapping[str, str] | Sequence[str],
) -> list[str]:
    """Parse one exact complete pilot ranking without synthesizing fallback ranks."""

    payload = json.loads(raw_output)
    expected = set(valid_handles)
    if not isinstance(payload, list) or len(payload) != len(expected):
        raise ValueError("pilot ranking must contain every handle exactly once")
    if len(set(payload)) != len(payload) or set(payload) != expected:
        raise ValueError("pilot ranking has duplicates, omissions, or unknown handles")
    return payload


def run_pilot_listwise_messages(
    llm: Any,
    messages: Sequence[list[dict[str, str]]],
    *,
    handles: Mapping[str, str] | Sequence[str],
    seed: int = 2026,
    use_tqdm: bool = True,
    stats: dict[str, int] | None = None,
) -> list[str]:
    """Run structured generation constrained to every pilot handle."""

    return run_listwise_messages(
        llm,
        messages,
        handles=handles,
        selection_count=len(handles),
        seed=seed,
        use_tqdm=use_tqdm,
        stats=stats,
    )


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


def _stage_shape(stage: object) -> tuple[int, int]:
    if not isinstance(stage, int) or isinstance(stage, bool):
        raise ValueError("pilot selection stage must be an integer")
    try:
        return _PILOT_STAGE_SHAPES[stage]
    except KeyError as error:
        raise ValueError("pilot selection stage must be 1 or 2") from error


def _require_stage_shape(payload: Mapping[str, Any]) -> tuple[int, int]:
    stage = payload.get("stage")
    query_count, candidate_count = _stage_shape(stage)
    questions = payload.get("questions")
    if (
        not isinstance(questions, list)
        or len(questions) != query_count
        or any(
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("candidate_ids"), list)
            or len(entry["candidate_ids"]) != candidate_count
            for entry in questions
        )
    ):
        raise ValueError(
            f"Stage {stage} requires exactly {query_count} queries and "
            f"{candidate_count} candidates per query"
        )
    return query_count, candidate_count


def _normalize_approved_path(
    value: object,
    *,
    approved: str | Path,
    label: str,
) -> str:
    if type(value) is not str:
        raise ValueError(f"pilot runtime {label} path must be a string")
    normalized = Path(value).expanduser().resolve(strict=False)
    normalized_approved = Path(approved).expanduser().resolve(strict=False)
    if normalized != normalized_approved:
        raise ValueError(f"pilot runtime {label} path is not approved")
    return str(normalized_approved)


def _validate_inference_stats(payload: object) -> dict[str, int]:
    if type(payload) is not dict or set(payload) != {
        "request_count",
        "prompt_token_count",
        "output_token_count",
    }:
        raise ValueError("pilot runtime inference stats schema changed")
    for field, value in payload.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"pilot runtime inference stats {field} must be an integer")
    return payload


def _validate_runtime_metadata_before_io(
    runtime_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(runtime_metadata, Mapping):
        raise ValueError("pilot runtime metadata must be an object")
    fields = set(runtime_metadata)
    if any(
        "qrel" in str(field).lower() or "judgment" in str(field).lower()
        for field in fields
    ):
        raise ValueError("pilot Phase A does not accept qrel or judgment metadata")
    allowed = _RUNTIME_INPUT_FIELDS | _OPTIONAL_RUNTIME_INPUT_FIELDS
    if fields - allowed or not _RUNTIME_INPUT_FIELDS.issubset(fields):
        raise ValueError("pilot runtime metadata schema changed")

    runtime = dict(runtime_metadata)
    for field, label in (
        ("model_id", "model ID"),
        ("revision", "model revision"),
        ("dtype", "dtype"),
        ("vllm_version", "vLLM version"),
        ("torch_version", "Torch version"),
        ("hip_version", "HIP version"),
        ("code_revision", "code revision"),
        ("worktree_status", "worktree status"),
    ):
        value = runtime[field]
        if type(value) is not str or not value.strip():
            raise ValueError(f"pilot runtime {label} must be a non-empty string")
    if type(runtime["seed"]) is not int:
        raise ValueError("pilot runtime seed must be an integer")
    for field, expected in (
        ("model_id", MODEL_ID),
        ("revision", MODEL_REVISION),
        ("dtype", MODEL_DTYPE),
        ("seed", 2026),
    ):
        if runtime[field] != expected:
            label = "model revision" if field == "revision" else field
            raise ValueError(f"pilot runtime {label} changed")
    if _COMMIT_SHA_RE.fullmatch(runtime["code_revision"]) is None:
        raise ValueError("pilot code revision must be a full 40-hex commit SHA")
    if runtime["worktree_status"] != "clean":
        raise ValueError("pilot worktree status must be exactly 'clean'")

    runtime["query_path"] = _normalize_approved_path(
        runtime["query_path"], approved=DEFAULT_QUERY_PATH, label="query"
    )
    runtime["candidate_path"] = _normalize_approved_path(
        runtime["candidate_path"],
        approved=DEFAULT_CANDIDATE_PATH,
        label="candidate",
    )
    runtime["smoke_receipt_path"] = _normalize_approved_path(
        runtime["smoke_receipt_path"],
        approved=DEFAULT_SMOKE_RECEIPT_PATH,
        label="smoke receipt",
    )

    if "device_properties" in runtime and (
        type(runtime["device_properties"]) is not str
        or not runtime["device_properties"].strip()
    ):
        raise ValueError("pilot runtime device properties must be a non-empty string")
    if "grade_token_ids" in runtime:
        grade_tokens = runtime["grade_token_ids"]
        if not isinstance(grade_tokens, Mapping):
            raise ValueError("pilot runtime grade token IDs changed")
        try:
            normalized_tokens = {
                int(grade): token_id for grade, token_id in grade_tokens.items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError("pilot runtime grade token IDs changed") from error
        if (
            normalized_tokens != _PILOT_GRADE_TOKEN_IDS
            or any(type(value) is not int for value in grade_tokens.values())
        ):
            raise ValueError("pilot runtime grade token IDs changed")
        runtime["grade_token_ids"] = normalized_tokens
    if "inference_stats" in runtime:
        runtime["inference_stats"] = _validate_inference_stats(
            runtime["inference_stats"]
        )
    return runtime


def _load_phase_a_selection(
    selection: Mapping[str, Any] | str | Path,
) -> dict[str, Any]:
    if isinstance(selection, (str, Path)):
        source = Path(selection)
        if source.name != "selection.json":
            raise ValueError("Phase A accepts only a qrel-free selection.json")
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("pilot selection is not valid JSON") from error
    elif isinstance(selection, Mapping):
        payload = json.loads(json.dumps(selection))
    else:
        raise ValueError("Phase A requires a qrel-free selection")

    if not isinstance(payload, Mapping):
        raise ValueError("Phase A requires a qrel-free selection")
    questions = payload.get("questions")
    if (
        not isinstance(questions, list)
        or not questions
        or not isinstance(questions[0], Mapping)
        or frozenset(questions[0]) != _SELECTION_ENTRY_KEYS
    ):
        raise ValueError("Phase A requires a qrel-free selection")
    _query_count, candidate_count = _require_stage_shape(payload)
    stage = payload["stage"]
    summary = validate_pilot_selection(
        payload,
        expected_stage=stage,
        expected_candidate_count=candidate_count,
    )
    if summary["manifest_kind"] != "selection":
        raise ValueError("Phase A requires a qrel-free selection")
    return dict(payload)


def _validate_current_pilot_inputs(
    selection: Mapping[str, Any],
    *,
    query_path: str,
    candidate_path: str,
) -> dict[str, str]:
    query_digest = _sha256(query_path)
    candidate_digest = _sha256(candidate_path)
    expected = selection["input_sha256"]
    if query_digest != expected["queries"]:
        raise ValueError("query input SHA256 changed")
    if candidate_digest != expected["candidates"]:
        raise ValueError("candidate input SHA256 changed")

    queries = load_queries_tsv(query_path)
    candidate_inventory = set(load_candidate_manifest(candidate_path))
    for entry in selection["questions"]:
        query_id = entry["query_id"]
        if queries.get(query_id) != entry["question"]:
            raise ValueError(f"current query text changed for {query_id!r}")
        if not set(entry["candidate_ids"]).issubset(candidate_inventory):
            raise ValueError(f"current candidate inventory changed for {query_id!r}")
    return {"queries": query_digest, "candidates": candidate_digest}


def preflight_pilot_inputs(
    selection: Mapping[str, Any] | str | Path,
    *,
    query_path: str | Path = DEFAULT_QUERY_PATH,
    candidate_path: str | Path = DEFAULT_CANDIDATE_PATH,
) -> tuple[dict[str, Any], dict[str, str]]:
    """Validate qrel-free selection and approved live inputs before engine loading."""

    approved_query = _normalize_approved_path(
        str(query_path), approved=DEFAULT_QUERY_PATH, label="query"
    )
    approved_candidate = _normalize_approved_path(
        str(candidate_path), approved=DEFAULT_CANDIDATE_PATH, label="candidate"
    )
    frozen_selection = _load_phase_a_selection(selection)
    input_sha256 = _validate_current_pilot_inputs(
        frozen_selection,
        query_path=approved_query,
        candidate_path=approved_candidate,
    )
    return frozen_selection, input_sha256


def _pilot_runtime_config(
    selection: Mapping[str, Any],
    *,
    method: str,
    runtime_metadata: Mapping[str, Any],
    input_sha256: Mapping[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if method not in {"pointwise", "listwise"}:
        raise ValueError("pilot method must be 'pointwise' or 'listwise'")
    runtime = dict(runtime_metadata)
    code_revision = runtime["code_revision"]
    worktree_status = runtime["worktree_status"]
    query_path = runtime["query_path"]
    candidate_path = runtime["candidate_path"]
    smoke_path = runtime["smoke_receipt_path"]
    smoke = validate_pilot_smoke_receipt(smoke_path)
    runtime.update(
        {
            "version": 1,
            "qrel_free_phase": True,
            "stage": selection["stage"],
            "input_sha256": input_sha256,
            "query_path": query_path,
            "candidate_path": candidate_path,
            "smoke_receipt_path": smoke_path,
            "smoke_receipt_sha256": smoke["receipt_sha256"],
        }
    )
    config = {
        "version": 1,
        "qrel_free_phase": True,
        "method": method,
        "stage": selection["stage"],
        "seed": 2026,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": MODEL_DTYPE,
        "code_revision": code_revision,
        "worktree_status": worktree_status,
        "query_path": query_path,
        "candidate_path": candidate_path,
        "smoke_receipt_path": smoke_path,
        "input_sha256": input_sha256,
        "smoke_receipt_sha256": smoke["receipt_sha256"],
    }
    return runtime, config


def _pointwise_output_record(
    output: object,
    *,
    query_id: str,
    candidate_id: str,
) -> tuple[dict[str, Any], float]:
    if not isinstance(output, Mapping):
        raise ValueError("pointwise inference outputs must be records")
    log_probabilities = output.get("grade_log_probabilities")
    if not isinstance(log_probabilities, Mapping):
        raise ValueError("pointwise output is missing grade log probabilities")
    try:
        normalized = {
            int(grade): float(value) for grade, value in log_probabilities.items()
        }
    except (TypeError, ValueError) as error:
        raise ValueError("pointwise grade log probabilities are invalid") from error
    score, probabilities = expected_grade_score(normalized)

    emitted_token_id = output.get("emitted_grade_token_id")
    emitted_text = output.get("emitted_grade_text")
    if (
        isinstance(emitted_token_id, bool)
        or emitted_token_id not in _PILOT_GRADE_TOKEN_IDS.values()
        or not isinstance(emitted_text, str)
    ):
        raise ValueError("pointwise output is missing the emitted grade token")
    emitted_grade = emitted_token_id - 15
    if emitted_text != str(emitted_grade):
        raise ValueError("pointwise emitted grade token ID/text disagree")

    return (
        {
            "query_id": query_id,
            "candidate_id": candidate_id,
            "emitted_grade_token_id": emitted_token_id,
            "emitted_grade_text": emitted_text,
            "grade_log_probabilities": {
                str(grade): normalized[grade] for grade in (0, 1, 2)
            },
            "grade_probabilities": {
                str(grade): probabilities[grade] for grade in (0, 1, 2)
            },
            "expected_grade": score,
        },
        score,
    )


def _expected_execution_counts(
    selection: Mapping[str, Any], *, method: str
) -> dict[str, int]:
    query_count, candidate_count = _require_stage_shape(selection)
    pair_count = query_count * candidate_count
    request_count = pair_count if method == "pointwise" else query_count * 3
    return {
        "query_count": query_count,
        "candidate_count": candidate_count,
        "pair_count": pair_count,
        "request_count": request_count,
        "prompt_count": request_count,
    }


def run_pilot_phase_a(
    selection: Mapping[str, Any] | str | Path,
    *,
    method: str,
    infer_batch: Callable[[Sequence[list[dict[str, str]]]], Sequence[Any]],
    runtime_metadata: Mapping[str, Any],
) -> PilotPhaseAResult:
    """Run one compact pilot using only its frozen qrel-free selection."""

    validated_runtime = _validate_runtime_metadata_before_io(runtime_metadata)
    frozen_selection, input_sha256 = preflight_pilot_inputs(
        selection,
        query_path=validated_runtime["query_path"],
        candidate_path=validated_runtime["candidate_path"],
    )
    runtime, config = _pilot_runtime_config(
        frozen_selection,
        method=method,
        runtime_metadata=validated_runtime,
        input_sha256=input_sha256,
    )
    prompts: list[dict[str, Any]] = []
    messages: list[list[dict[str, str]]] = []
    keys: list[tuple[str, str] | tuple[str, int]] = []
    handles_by_query: dict[str, dict[str, str]] = {}

    if method == "pointwise":
        for entry in frozen_selection["questions"]:
            for candidate_id in entry["candidate_ids"]:
                formatted = format_pointwise_messages(
                    entry["question"], candidate_id
                )
                prompts.append(
                    {
                        "query_id": entry["query_id"],
                        "candidate_id": candidate_id,
                        "messages": formatted,
                    }
                )
                messages.append(formatted)
                keys.append((entry["query_id"], candidate_id))
    else:
        for entry in frozen_selection["questions"]:
            query_id = entry["query_id"]
            handles = pilot_handles(entry["candidate_ids"])
            handles_by_query[query_id] = handles
            for rotation, order in enumerate(
                balanced_pilot_orders(
                    query_id=query_id,
                    candidate_ids=entry["candidate_ids"],
                )
            ):
                formatted = format_pilot_listwise_messages(
                    entry["question"],
                    ordered_candidate_ids=order,
                    handle_to_candidate=handles,
                )
                prompts.append(
                    {
                        "query_id": query_id,
                        "rotation": rotation,
                        "candidate_order": list(order),
                        "messages": formatted,
                    }
                )
                messages.append(formatted)
                keys.append((query_id, rotation))

    execution_counts = _expected_execution_counts(
        frozen_selection, method=method
    )
    if len(prompts) != execution_counts["prompt_count"]:
        raise ValueError("pilot prompt count disagrees with the stage contract")

    started = time.perf_counter()
    inferred = list(infer_batch(messages))
    elapsed = time.perf_counter() - started
    if len(inferred) != len(messages):
        raise ValueError("pilot inference returned the wrong number of outputs")

    raw_outputs: list[dict[str, Any]] = []
    scores: dict[str, dict[str, float]] = {
        entry["query_id"]: {
            candidate_id: 0.0 for candidate_id in entry["candidate_ids"]
        }
        for entry in frozen_selection["questions"]
    }
    if method == "pointwise":
        for key, output in zip(keys, inferred, strict=True):
            query_id, candidate_id = key
            if not isinstance(candidate_id, str):  # pragma: no cover - type guard
                raise AssertionError("pointwise key must contain a candidate ID")
            record, score = _pointwise_output_record(
                output,
                query_id=query_id,
                candidate_id=candidate_id,
            )
            raw_outputs.append(record)
            scores[query_id][candidate_id] = score
    else:
        for key, output in zip(keys, inferred, strict=True):
            query_id, rotation = key
            if not isinstance(rotation, int):  # pragma: no cover - type guard
                raise AssertionError("listwise key must contain a rotation")
            if not isinstance(output, str):
                raise ValueError("pilot listwise inference outputs must be strings")
            handles = handles_by_query[query_id]
            try:
                parsed_handles = parse_complete_pilot_ranking(
                    output, valid_handles=handles
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError(
                    f"pilot listwise parse failure for {query_id}/{rotation}: {error}"
                ) from error
            selected_candidates = [handles[handle] for handle in parsed_handles]
            for rank, candidate_id in enumerate(selected_candidates, start=1):
                scores[query_id][candidate_id] += 1.0 / (60 + rank)
            raw_outputs.append(
                {
                    "query_id": query_id,
                    "rotation": rotation,
                    "raw_output": output,
                    "selected_handles": parsed_handles,
                    "selected_candidate_ids": selected_candidates,
                    "validation_error": None,
                    "used_fallback": False,
                }
            )

    rankings = {
        query_id: sorted(
            query_scores,
            key=lambda candidate_id: (-query_scores[candidate_id], candidate_id),
        )
        for query_id, query_scores in scores.items()
    }
    runtime.update(
        {
            "elapsed_seconds": elapsed,
            "method": method,
            "prompt_count": execution_counts["prompt_count"],
            "request_count": execution_counts["request_count"],
            "pair_count": execution_counts["pair_count"],
        }
    )
    return PilotPhaseAResult(
        selection=frozen_selection,
        method=method,
        prompts=tuple(prompts),
        raw_outputs=tuple(raw_outputs),
        scores=scores,
        rankings=rankings,
        runtime_metadata=runtime,
        config=config,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )


def write_pilot_phase_a(
    result: PilotPhaseAResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write the exact durable qrel-free pilot artifact inventory."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "config.json", result.config)
    _write_json(destination / "selection.json", result.selection)
    _write_jsonl(destination / "prompts.jsonl", result.prompts)
    _write_jsonl(destination / "raw_outputs.jsonl", result.raw_outputs)
    _write_json(destination / "rankings.json", result.rankings)
    _write_json(destination / "runtime.json", result.runtime_metadata)

    with (destination / "scores.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("query_id", "candidate_id", "score", "rank"))
        for entry in result.selection["questions"]:
            query_id = entry["query_id"]
            ranks = {
                candidate_id: rank
                for rank, candidate_id in enumerate(
                    result.rankings[query_id], start=1
                )
            }
            for candidate_id in entry["candidate_ids"]:
                score = result.scores[query_id][candidate_id]
                if not math.isfinite(score):
                    raise ValueError("pilot scores contain a non-finite value")
                writer.writerow(
                    (query_id, candidate_id, f"{score:.17g}", ranks[candidate_id])
                )

    artifact_sha256 = {
        name: _sha256(destination / name) for name in sorted(_PILOT_CORE_FILES)
    }
    receipt = {
        "version": 1,
        "qrel_free_phase": True,
        "method": result.method,
        "stage": result.selection["stage"],
        "query_count": len(result.selection["questions"]),
        "candidate_count": len(result.selection["questions"][0]["candidate_ids"]),
        "row_count": sum(
            len(entry["candidate_ids"])
            for entry in result.selection["questions"]
        ),
        "artifact_sha256": artifact_sha256,
        "prompt_sha256": artifact_sha256["prompts.jsonl"],
        "current_input_sha256": result.config["input_sha256"],
        "smoke_receipt_sha256": result.config["smoke_receipt_sha256"],
    }
    _write_json(destination / "ranking_receipt.json", receipt)
    return {
        name: str(destination / name)
        for name in (*sorted(_PILOT_CORE_FILES), "ranking_receipt.json")
    }


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    return payload


def _load_jsonl(path: Path, *, label: str) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip():
                raise ValueError(f"{label} contains a blank line")
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"{label} line {line_number} is not valid JSON"
                ) from error
            if not isinstance(row, dict):
                raise ValueError(f"{label} rows must be objects")
            rows.append(row)
    return tuple(rows)


def _expected_pilot_prompts(
    selection: Mapping[str, Any], *, method: str
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for entry in selection["questions"]:
        query_id = entry["query_id"]
        if method == "pointwise":
            for candidate_id in entry["candidate_ids"]:
                rows.append(
                    {
                        "query_id": query_id,
                        "candidate_id": candidate_id,
                        "messages": format_pointwise_messages(
                            entry["question"], candidate_id
                        ),
                    }
                )
            continue
        handles = pilot_handles(entry["candidate_ids"])
        for rotation, order in enumerate(
            balanced_pilot_orders(
                query_id=query_id, candidate_ids=entry["candidate_ids"]
            )
        ):
            rows.append(
                {
                    "query_id": query_id,
                    "rotation": rotation,
                    "candidate_order": list(order),
                    "messages": format_pilot_listwise_messages(
                        entry["question"],
                        ordered_candidate_ids=order,
                        handle_to_candidate=handles,
                    ),
                }
            )
    return tuple(rows)


def _scores_from_raw_outputs(
    selection: Mapping[str, Any],
    *,
    method: str,
    raw_outputs: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float]]:
    scores = {
        entry["query_id"]: {
            candidate_id: 0.0 for candidate_id in entry["candidate_ids"]
        }
        for entry in selection["questions"]
    }
    if method == "pointwise":
        expected_pairs = [
            (entry["query_id"], candidate_id)
            for entry in selection["questions"]
            for candidate_id in entry["candidate_ids"]
        ]
        if len(raw_outputs) != len(expected_pairs):
            raise ValueError("pointwise raw output count changed")
        for (query_id, candidate_id), raw in zip(
            expected_pairs, raw_outputs, strict=True
        ):
            if (raw.get("query_id"), raw.get("candidate_id")) != (
                query_id,
                candidate_id,
            ):
                raise ValueError("pointwise raw output order changed")
            record, score = _pointwise_output_record(
                raw,
                query_id=query_id,
                candidate_id=candidate_id,
            )
            stored_probabilities = raw.get("grade_probabilities")
            if not isinstance(stored_probabilities, Mapping) or set(
                stored_probabilities
            ) != {"0", "1", "2"}:
                raise ValueError("pointwise stored grade probability schema changed")
            for grade in (0, 1, 2):
                stored_probability = stored_probabilities[str(grade)]
                if (
                    isinstance(stored_probability, bool)
                    or not isinstance(stored_probability, (int, float))
                    or not math.isfinite(float(stored_probability))
                    or not math.isclose(
                        float(stored_probability),
                        record["grade_probabilities"][str(grade)],
                        rel_tol=1e-12,
                        abs_tol=1e-12,
                    )
                ):
                    raise ValueError(
                        f"pointwise stored grade probability {grade} changed"
                    )
            if not math.isclose(
                float(raw.get("expected_grade", math.nan)),
                record["expected_grade"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("pointwise expected grade changed")
            scores[query_id][candidate_id] = score
        return scores

    expected_count = len(selection["questions"]) * 3
    if len(raw_outputs) != expected_count:
        raise ValueError("listwise raw output count changed")
    position = 0
    for entry in selection["questions"]:
        query_id = entry["query_id"]
        handles = pilot_handles(entry["candidate_ids"])
        for rotation in range(3):
            raw = raw_outputs[position]
            position += 1
            if (raw.get("query_id"), raw.get("rotation")) != (
                query_id,
                rotation,
            ):
                raise ValueError("listwise raw output order changed")
            if raw.get("validation_error") is not None or raw.get("used_fallback") is not False:
                raise ValueError("pilot listwise parse failure is frozen in the artifact")
            output = raw.get("raw_output")
            if not isinstance(output, str):
                raise ValueError("listwise raw output is missing")
            try:
                parsed = parse_complete_pilot_ranking(
                    output, valid_handles=handles
                )
            except (json.JSONDecodeError, ValueError) as error:
                raise ValueError("pilot listwise parse failure") from error
            selected_candidates = [handles[handle] for handle in parsed]
            if raw.get("selected_handles") != parsed or raw.get(
                "selected_candidate_ids"
            ) != selected_candidates:
                raise ValueError("listwise parsed output changed")
            for rank, candidate_id in enumerate(selected_candidates, start=1):
                scores[query_id][candidate_id] += 1.0 / (60 + rank)
    return scores


def _load_pilot_scores(
    path: Path,
    *,
    selection: Mapping[str, Any],
    rankings: Mapping[str, list[str]],
) -> dict[str, dict[str, float]]:
    expected_pairs = {
        (entry["query_id"], candidate_id)
        for entry in selection["questions"]
        for candidate_id in entry["candidate_ids"]
    }
    values: dict[str, dict[str, float]] = {
        entry["query_id"]: {} for entry in selection["questions"]
    }
    seen: set[tuple[str, str]] = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["query_id", "candidate_id", "score", "rank"]:
            raise ValueError("pilot score header changed")
        for row in reader:
            key = (row["query_id"], row["candidate_id"])
            if key in seen or key not in expected_pairs:
                raise ValueError("pilot scores contain a duplicate or unknown pair")
            try:
                score = float(row["score"])
                rank = int(row["rank"])
            except ValueError as error:
                raise ValueError("pilot score row is invalid") from error
            if not math.isfinite(score):
                raise ValueError("pilot scores contain a non-finite value")
            expected_rank = rankings[key[0]].index(key[1]) + 1
            if rank != expected_rank:
                raise ValueError("pilot stored rank changed")
            seen.add(key)
            values[key[0]][key[1]] = score
    if seen != expected_pairs:
        raise ValueError("pilot scores are incomplete")
    return values


def load_frozen_pilot(artifact_dir: str | Path) -> PilotPhaseAResult:
    """Load a pilot only after revalidating every artifact and live input hash."""

    source = Path(artifact_dir)
    receipt = _load_json_object(
        source / "ranking_receipt.json", label="pilot ranking receipt"
    )
    if receipt.get("qrel_free_phase") is not True:
        raise ValueError("pilot ranking receipt is not qrel-free")
    artifact_sha256 = receipt.get("artifact_sha256")
    if not isinstance(artifact_sha256, Mapping) or set(artifact_sha256) != set(
        _PILOT_CORE_FILES
    ):
        raise ValueError("pilot core artifact set changed")
    for name in sorted(_PILOT_CORE_FILES):
        path = source / name
        if not path.is_file() or _sha256(path) != artifact_sha256[name]:
            raise ValueError(f"artifact hash mismatch: {name}")
    if receipt.get("prompt_sha256") != artifact_sha256["prompts.jsonl"]:
        raise ValueError("pilot prompt SHA256 changed")

    config = _load_json_object(source / "config.json", label="pilot config")
    if set(config) != set(_PILOT_CONFIG_KEYS):
        raise ValueError("pilot config schema changed")
    if config.get("qrel_free_phase") is not True or config.get("version") != 1:
        raise ValueError("pilot config is not qrel-free version 1")
    method = config.get("method")
    if method not in {"pointwise", "listwise"}:
        raise ValueError("pilot method changed")
    for field, expected in (
        ("model_id", MODEL_ID),
        ("model_revision", MODEL_REVISION),
        ("dtype", MODEL_DTYPE),
        ("seed", 2026),
    ):
        if config.get(field) != expected:
            label = "model revision" if field == "model_revision" else field
            raise ValueError(f"pilot {label} changed")
    code_revision = config.get("code_revision")
    if not isinstance(code_revision, str) or _COMMIT_SHA_RE.fullmatch(
        code_revision
    ) is None:
        raise ValueError("pilot code revision changed")
    if config.get("worktree_status") != "clean":
        raise ValueError("pilot worktree status changed")
    approved_query_path = _normalize_approved_path(
        config.get("query_path"), approved=DEFAULT_QUERY_PATH, label="query"
    )
    approved_candidate_path = _normalize_approved_path(
        config.get("candidate_path"),
        approved=DEFAULT_CANDIDATE_PATH,
        label="candidate",
    )
    approved_smoke_path = _normalize_approved_path(
        config.get("smoke_receipt_path"),
        approved=DEFAULT_SMOKE_RECEIPT_PATH,
        label="smoke receipt",
    )

    selection = _load_phase_a_selection(source / "selection.json")
    if config.get("stage") != selection["stage"] or receipt.get("stage") != selection[
        "stage"
    ]:
        raise ValueError("pilot stage changed")
    input_sha256 = _validate_current_pilot_inputs(
        selection,
        query_path=approved_query_path,
        candidate_path=approved_candidate_path,
    )
    if (
        config.get("input_sha256") != input_sha256
        or receipt.get("current_input_sha256") != input_sha256
    ):
        raise ValueError("pilot recorded input SHA256 changed")

    smoke_path = Path(approved_smoke_path)
    if _sha256(smoke_path) != config.get("smoke_receipt_sha256"):
        raise ValueError("smoke receipt SHA256 changed")
    smoke = validate_pilot_smoke_receipt(smoke_path)
    if (
        smoke["receipt_sha256"] != receipt.get("smoke_receipt_sha256")
        or smoke["receipt_sha256"] != config.get("smoke_receipt_sha256")
    ):
        raise ValueError("smoke receipt SHA256 changed")

    runtime = _load_json_object(source / "runtime.json", label="pilot runtime")
    required_runtime_fields = _RUNTIME_INPUT_FIELDS | {
        "version",
        "qrel_free_phase",
        "stage",
        "input_sha256",
        "smoke_receipt_sha256",
        "elapsed_seconds",
        "method",
        "prompt_count",
        "request_count",
        "pair_count",
    }
    allowed_runtime_fields = required_runtime_fields | _OPTIONAL_RUNTIME_INPUT_FIELDS
    if set(runtime) - allowed_runtime_fields or not required_runtime_fields.issubset(
        runtime
    ):
        raise ValueError("pilot runtime metadata schema changed")
    if type(runtime.get("version")) is not int or runtime["version"] != 1:
        raise ValueError("pilot runtime version changed")
    if (
        type(runtime.get("qrel_free_phase")) is not bool
        or runtime["qrel_free_phase"] is not True
    ):
        raise ValueError("pilot runtime qrel-free phase changed")
    if type(runtime.get("stage")) is not int:
        raise ValueError("pilot runtime stage changed")
    for field in ("prompt_count", "request_count", "pair_count"):
        if type(runtime.get(field)) is not int:
            raise ValueError(f"pilot runtime {field.replace('_', ' ')} changed")
    elapsed_seconds = runtime.get("elapsed_seconds")
    if (
        isinstance(elapsed_seconds, bool)
        or not isinstance(elapsed_seconds, (int, float))
        or not math.isfinite(float(elapsed_seconds))
        or elapsed_seconds < 0
    ):
        raise ValueError("pilot runtime elapsed seconds changed")
    for field, label in (
        ("vllm_version", "vLLM version"),
        ("torch_version", "Torch version"),
        ("hip_version", "HIP version"),
    ):
        if type(runtime.get(field)) is not str or not runtime[field].strip():
            raise ValueError(f"pilot runtime {label} changed")
    if "device_properties" in runtime and (
        type(runtime["device_properties"]) is not str
        or not runtime["device_properties"].strip()
    ):
        raise ValueError("pilot runtime device properties changed")
    if "grade_token_ids" in runtime:
        grade_tokens = runtime["grade_token_ids"]
        if not isinstance(grade_tokens, Mapping):
            raise ValueError("pilot runtime grade token IDs changed")
        try:
            normalized_grade_tokens = {
                int(grade): token_id for grade, token_id in grade_tokens.items()
            }
        except (TypeError, ValueError) as error:
            raise ValueError("pilot runtime grade token IDs changed") from error
        if (
            normalized_grade_tokens != _PILOT_GRADE_TOKEN_IDS
            or any(type(value) is not int for value in grade_tokens.values())
        ):
            raise ValueError("pilot runtime grade token IDs changed")
    if "inference_stats" in runtime:
        _validate_inference_stats(runtime["inference_stats"])

    runtime_expected = {
        "version": config["version"],
        "qrel_free_phase": config["qrel_free_phase"],
        "stage": config["stage"],
        "model_id": config["model_id"],
        "revision": config["model_revision"],
        "dtype": config["dtype"],
        "seed": config["seed"],
        "code_revision": config["code_revision"],
        "worktree_status": config["worktree_status"],
        "query_path": config["query_path"],
        "candidate_path": config["candidate_path"],
        "smoke_receipt_path": config["smoke_receipt_path"],
        "smoke_receipt_sha256": config["smoke_receipt_sha256"],
        "input_sha256": config["input_sha256"],
        "method": method,
    }
    for field, expected in runtime_expected.items():
        if runtime.get(field) != expected:
            label = "model revision" if field == "revision" else field.replace(
                "_", " "
            )
            raise ValueError(f"pilot runtime {label} changed")

    execution_counts = _expected_execution_counts(selection, method=method)
    for field in ("prompt_count", "request_count", "pair_count"):
        if runtime[field] != execution_counts[field]:
            raise ValueError(f"pilot runtime {field.replace('_', ' ')} changed")
    if "inference_stats" in runtime and (
        runtime["inference_stats"]["request_count"]
        != execution_counts["request_count"]
    ):
        raise ValueError("pilot runtime inference stats request count changed")

    prompts = _load_jsonl(source / "prompts.jsonl", label="pilot prompts")
    expected_prompts = _expected_pilot_prompts(selection, method=method)
    if (
        len(prompts) != execution_counts["prompt_count"]
        or prompts != expected_prompts
    ):
        raise ValueError("pilot prompt content changed")
    raw_outputs = _load_jsonl(
        source / "raw_outputs.jsonl", label="pilot raw outputs"
    )
    expected_scores = _scores_from_raw_outputs(
        selection, method=method, raw_outputs=raw_outputs
    )

    rankings_payload = _load_json_object(
        source / "rankings.json", label="pilot rankings"
    )
    expected_query_ids = [entry["query_id"] for entry in selection["questions"]]
    if list(rankings_payload) != sorted(expected_query_ids):
        # Sorted JSON serialization deliberately fixes the rankings key order.
        raise ValueError("pilot ranking query inventory changed")
    rankings: dict[str, list[str]] = {}
    for entry in selection["questions"]:
        query_id = entry["query_id"]
        ranking = rankings_payload.get(query_id)
        if (
            not isinstance(ranking, list)
            or len(ranking) != len(entry["candidate_ids"])
            or len(set(ranking)) != len(ranking)
            or set(ranking) != set(entry["candidate_ids"])
        ):
            raise ValueError("pilot ranking candidate inventory changed")
        rankings[query_id] = ranking
    stored_scores = _load_pilot_scores(
        source / "scores.tsv", selection=selection, rankings=rankings
    )
    for query_id, query_scores in expected_scores.items():
        for candidate_id, score in query_scores.items():
            if not math.isclose(
                stored_scores[query_id][candidate_id],
                score,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("pilot score disagrees with raw output")
        expected_ranking = sorted(
            query_scores,
            key=lambda candidate_id: (-query_scores[candidate_id], candidate_id),
        )
        if rankings[query_id] != expected_ranking:
            raise ValueError("pilot ranking disagrees with scores")

    if (
        receipt.get("method") != method
        or receipt.get("query_count") != execution_counts["query_count"]
        or receipt.get("candidate_count") != execution_counts["candidate_count"]
        or receipt.get("row_count") != execution_counts["pair_count"]
    ):
        raise ValueError("pilot ranking receipt counts changed")
    return PilotPhaseAResult(
        selection=selection,
        method=method,
        prompts=prompts,
        raw_outputs=raw_outputs,
        scores=stored_scores,
        rankings=rankings,
        runtime_metadata=runtime,
        config=config,
    )


def evaluate_pilot(
    artifact_dir: str | Path,
    qrels_path: str | Path,
) -> dict[str, Any]:
    """Open qrels only after the complete Phase A pilot validates."""

    source = Path(artifact_dir)
    frozen = load_frozen_pilot(source)
    query_ids = tuple(
        entry["query_id"] for entry in frozen.selection["questions"]
    )
    qrels = load_complete_qrels(qrels_path, expected_query_ids=query_ids)
    current_candidates = set(load_candidate_manifest(frozen.config["candidate_path"]))
    if set(next(iter(qrels.values()))) != current_candidates:
        raise ValueError("qrel candidate universe disagrees with current candidates")

    per_query: dict[str, Any] = {}
    for entry in frozen.selection["questions"]:
        query_id = entry["query_id"]
        ranking = frozen.rankings[query_id]
        relevant_ranks = {
            candidate_id: ranking.index(candidate_id) + 1
            for candidate_id in entry["candidate_ids"]
            if qrels[query_id][candidate_id] > 0
        }
        per_query[query_id] = {
            "relevant_ranks": relevant_ranks,
            "top_three": ranking[:3],
        }
    return {
        "method": frozen.method,
        "phase_a_sha256": _sha256(source / "ranking_receipt.json"),
        "qrels_sha256": _sha256(qrels_path),
        "per_query": per_query,
    }


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
