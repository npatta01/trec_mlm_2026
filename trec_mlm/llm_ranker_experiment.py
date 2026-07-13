"""Qrel-isolated orchestration and artifacts for zero-shot LLM label rankers."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.artifact_manifest import validate_smoke_receipt
from trec_mlm.label_experiment import load_candidate_manifest
from trec_mlm.llm_ranker import (
    balanced_candidate_orders,
    candidate_handles,
    expected_grade_score,
    format_listwise_messages,
    format_pointwise_messages,
    parse_listwise_handles,
    repair_listwise_syntax,
    rrf_ranking,
)
from trec_mlm.llm_ranker_inference import MODEL_DTYPE, MODEL_ID, MODEL_REVISION
from trec_mlm.metrics import evaluate_rankings
from trec_mlm.query_knn import load_complete_qrels, load_queries_tsv
from trec_mlm.runio import rankings_from_run, validate_trec_run, write_trec_run


PointwiseInfer = Callable[
    [Sequence[list[dict[str, str]]]], Sequence[Mapping[int, float]]
]
ListwiseInfer = Callable[[Sequence[list[dict[str, str]]]], Sequence[str]]


@dataclass(frozen=True, slots=True)
class LLMRankerConfig:
    """Frozen qrel-free inputs for one ranker run."""

    query_path: str = "data/dev/llm_dev_data.tsv"
    candidate_path: str = "reports/trec_2026_discovery_profile_audit.json"
    smoke_receipt_path: str = "artifacts/llm_label_ranker/smoke/smoke.json"
    method: str = "pointwise"
    seed: int = 2026
    run_id: str = "qwen3_4b_pointwise_zero_shot"
    code_revision: str = "not_recorded"
    worktree_status: str = "not_recorded"


@dataclass(frozen=True, slots=True)
class LLMRankerPhaseAResult:
    """Complete rankings fixed before qrels become available."""

    config: LLMRankerConfig
    query_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    prompts: tuple[dict[str, Any], ...]
    raw_records: tuple[dict[str, Any], ...]
    scores: NDArray[np.float32]
    rankings: dict[str, list[str]]
    runtime_metadata: dict[str, Any]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_sha256(path: str) -> str:
    return _sha256(Path(path))


def _validate_config(config: LLMRankerConfig, *, expected_method: str) -> dict[str, Any]:
    if config.method != expected_method:
        raise ValueError(f"config method must be {expected_method!r}")
    if not config.run_id or any(character.isspace() for character in config.run_id):
        raise ValueError("run_id must be a non-empty token without whitespace")
    if config.code_revision == "not_recorded" or not config.code_revision.strip():
        raise ValueError("an explicit code revision is required")
    return validate_smoke_receipt(
        config.smoke_receipt_path,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        dtype=MODEL_DTYPE,
    )


def _batches(values: Sequence[Any], size: int) -> Sequence[Sequence[Any]]:
    if size <= 0:
        raise ValueError("batch_size must be positive")
    return [values[start : start + size] for start in range(0, len(values), size)]


def run_pointwise_phase_a(
    config: LLMRankerConfig,
    *,
    infer_batch: PointwiseInfer,
    batch_size: int = 512,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> LLMRankerPhaseAResult:
    """Score all query-label pairs without opening a qrel file."""

    smoke = _validate_config(config, expected_method="pointwise")
    queries = load_queries_tsv(config.query_path)
    query_ids = tuple(queries)
    candidate_ids = load_candidate_manifest(config.candidate_path)
    if len(query_ids) != 322 or len(candidate_ids) != 150:
        raise ValueError("canonical ranker requires exactly 322 queries and 150 candidates")

    prompt_rows: list[dict[str, Any]] = []
    messages: list[list[dict[str, str]]] = []
    pairs: list[tuple[str, str]] = []
    for query_id in query_ids:
        for candidate_id in candidate_ids:
            formatted = format_pointwise_messages(queries[query_id], candidate_id)
            prompt_rows.append(
                {"query_id": query_id, "candidate_id": candidate_id, "messages": formatted}
            )
            messages.append(formatted)
            pairs.append((query_id, candidate_id))

    started = time.perf_counter()
    inferred: list[Mapping[int, float]] = []
    for batch in _batches(messages, batch_size):
        values = list(infer_batch(batch))
        if len(values) != len(batch):
            raise ValueError("pointwise inference returned the wrong number of outputs")
        inferred.extend(values)
    elapsed = time.perf_counter() - started

    scores = np.empty((len(query_ids), len(candidate_ids)), dtype=np.float32)
    raw_records: list[dict[str, Any]] = []
    query_positions = {query_id: index for index, query_id in enumerate(query_ids)}
    candidate_positions = {
        candidate_id: index for index, candidate_id in enumerate(candidate_ids)
    }
    for (query_id, candidate_id), values in zip(pairs, inferred, strict=True):
        normalized = {int(grade): float(value) for grade, value in values.items()}
        score, probabilities = expected_grade_score(normalized)
        scores[query_positions[query_id], candidate_positions[candidate_id]] = score
        raw_records.append(
            {
                "query_id": query_id,
                "candidate_id": candidate_id,
                "grade_log_probabilities": {str(k): normalized[k] for k in (0, 1, 2)},
                "grade_probabilities": {str(k): probabilities[k] for k in (0, 1, 2)},
                "expected_grade": score,
            }
        )
    rankings = {
        query_id: sorted(
            candidate_ids,
            key=lambda candidate_id: (
                -float(scores[query_positions[query_id], candidate_positions[candidate_id]]),
                candidate_id,
            ),
        )
        for query_id in query_ids
    }
    metadata = {
        **dict(runtime_metadata or {}),
        "elapsed_seconds": elapsed,
        "pair_count": len(pairs),
        "smoke_receipt_sha256": smoke["receipt_sha256"],
    }
    return LLMRankerPhaseAResult(
        config=config,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        prompts=tuple(prompt_rows),
        raw_records=tuple(raw_records),
        scores=scores,
        rankings=rankings,
        runtime_metadata=metadata,
    )


def run_listwise_phase_a(
    config: LLMRankerConfig,
    *,
    infer_batch: ListwiseInfer,
    batch_size: int = 16,
    runtime_metadata: Mapping[str, Any] | None = None,
) -> LLMRankerPhaseAResult:
    """Run three balanced all-label top-20 selections per query without qrels."""

    smoke = _validate_config(config, expected_method="listwise")
    queries = load_queries_tsv(config.query_path)
    query_ids = tuple(queries)
    candidate_ids = load_candidate_manifest(config.candidate_path)
    if len(query_ids) != 322 or len(candidate_ids) != 150:
        raise ValueError("canonical ranker requires exactly 322 queries and 150 candidates")
    handles = candidate_handles(candidate_ids)

    prompt_rows: list[dict[str, Any]] = []
    messages: list[list[dict[str, str]]] = []
    keys: list[tuple[str, int]] = []
    for query_id in query_ids:
        for rotation, order in enumerate(
            balanced_candidate_orders(query_id=query_id, candidate_ids=candidate_ids)
        ):
            formatted = format_listwise_messages(
                queries[query_id],
                ordered_candidate_ids=order,
                handle_to_candidate=handles,
            )
            prompt_rows.append(
                {
                    "query_id": query_id,
                    "rotation": rotation,
                    "candidate_order": list(order),
                    "messages": formatted,
                }
            )
            messages.append(formatted)
            keys.append((query_id, rotation))

    started = time.perf_counter()
    inferred: list[str] = []
    for batch in _batches(messages, batch_size):
        values = list(infer_batch(batch))
        if len(values) != len(batch):
            raise ValueError("listwise inference returned the wrong number of outputs")
        inferred.extend(values)
    elapsed = time.perf_counter() - started

    selections: dict[str, list[list[str]]] = {query_id: [] for query_id in query_ids}
    raw_records: list[dict[str, Any]] = []
    fallback_handles = list(handles)[:20]
    fallback_count = 0
    repair_count = 0
    for (query_id, rotation), raw in zip(keys, inferred, strict=True):
        try:
            parsed = parse_listwise_handles(raw, valid_handles=handles)
            validation_error = None
            used_repair = False
        except ValueError as error:
            validation_error = str(error)
            try:
                parsed = repair_listwise_syntax(raw, valid_handles=handles)
                used_repair = True
                repair_count += 1
            except ValueError:
                # Semantic failures never get guessed at: use a deterministic,
                # visible candidate-ID fallback and preserve the raw failure.
                parsed = fallback_handles
                used_repair = False
                fallback_count += 1
        selections[query_id].append(parsed)
        raw_records.append(
            {
                "query_id": query_id,
                "rotation": rotation,
                "raw_output": raw,
                "selected_handles": parsed,
                "validation_error": validation_error,
                "used_syntax_repair": used_repair,
                "used_fallback": validation_error is not None and not used_repair,
            }
        )

    scores = np.empty((len(query_ids), len(candidate_ids)), dtype=np.float32)
    rankings: dict[str, list[str]] = {}
    candidate_positions = {
        candidate_id: index for index, candidate_id in enumerate(candidate_ids)
    }
    for query_index, query_id in enumerate(query_ids):
        ranking, rrf_scores = rrf_ranking(
            selections[query_id], handle_to_candidate=handles, k=60
        )
        rankings[query_id] = ranking
        for candidate_id, score in rrf_scores.items():
            scores[query_index, candidate_positions[candidate_id]] = score
    metadata = {
        **dict(runtime_metadata or {}),
        "elapsed_seconds": elapsed,
        "request_count": len(messages),
        "syntax_repair_count": repair_count,
        "fallback_count": fallback_count,
        "smoke_receipt_sha256": smoke["receipt_sha256"],
    }
    return LLMRankerPhaseAResult(
        config=config,
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        prompts=tuple(prompt_rows),
        raw_records=tuple(raw_records),
        scores=scores,
        rankings=rankings,
        runtime_metadata=metadata,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_ranker_phase_a_artifacts(
    result: LLMRankerPhaseAResult, output_dir: str | Path
) -> dict[str, str]:
    """Write and hash the qrel-free ranking receipt."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    config_payload = {
        **asdict(result.config),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "dtype": MODEL_DTYPE,
        "input_sha256": {
            result.config.query_path: _input_sha256(result.config.query_path),
            result.config.candidate_path: _input_sha256(result.config.candidate_path),
        },
        "qrel_free_phase": True,
    }
    _write_json(destination / "config.json", config_payload)
    _write_json(destination / "runtime.json", result.runtime_metadata)
    _write_jsonl(destination / "prompts.jsonl", result.prompts)
    _write_jsonl(destination / "raw_outputs.jsonl", result.raw_records)

    rank_by_pair = {
        (query_id, candidate_id): rank
        for query_id, ranking in result.rankings.items()
        for rank, candidate_id in enumerate(ranking, start=1)
    }
    with (destination / "scores.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("query_id", "candidate_id", "score", "rank"))
        for query_index, query_id in enumerate(result.query_ids):
            for candidate_index, candidate_id in enumerate(result.candidate_ids):
                value = float(result.scores[query_index, candidate_index])
                if not math.isfinite(value):
                    raise ValueError("ranker score matrix contains a non-finite value")
                writer.writerow(
                    (query_id, candidate_id, f"{value:.9g}", rank_by_pair[(query_id, candidate_id)])
                )
    write_trec_run(destination / "run.trec", result.rankings, run_id=result.config.run_id)
    validation = validate_trec_run(
        destination / "run.trec",
        expected_query_ids=result.query_ids,
        expected_model_ids=result.candidate_ids,
    )
    core_names = (
        "config.json",
        "runtime.json",
        "prompts.jsonl",
        "raw_outputs.jsonl",
        "scores.tsv",
        "run.trec",
    )
    receipt = {
        "version": 1,
        "qrel_free_phase": True,
        "method": result.config.method,
        "query_count": len(result.query_ids),
        "candidate_count": len(result.candidate_ids),
        "row_count": len(result.query_ids) * len(result.candidate_ids),
        "run_validation": validation,
        "artifact_sha256": {name: _sha256(destination / name) for name in core_names},
    }
    _write_json(destination / "ranking_receipt.json", receipt)
    return {
        name: str(destination / name) for name in (*core_names, "ranking_receipt.json")
    }


def _validate_frozen_artifact(source: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt = json.loads((source / "ranking_receipt.json").read_text(encoding="utf-8"))
    if receipt.get("qrel_free_phase") is not True:
        raise ValueError("ranking receipt is not qrel-free")
    for name, expected in receipt.get("artifact_sha256", {}).items():
        path = source / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"artifact hash mismatch: {name}")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    return receipt, config


def evaluate_frozen_ranker_artifact(
    artifact_dir: str | Path,
    qrels_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Evaluate a previously fixed ranking without mutating Phase A."""

    source = Path(artifact_dir)
    receipt, config = _validate_frozen_artifact(source)
    queries = load_queries_tsv(config["query_path"])
    candidates = load_candidate_manifest(config["candidate_path"])
    validate_trec_run(
        source / "run.trec",
        expected_query_ids=tuple(queries),
        expected_model_ids=candidates,
    )
    rankings = rankings_from_run(source / "run.trec")

    # Qrels are intentionally opened only after every Phase A hash is verified.
    qrels = load_complete_qrels(qrels_path, expected_query_ids=tuple(queries))
    if set(next(iter(qrels.values()))) != set(candidates):
        raise ValueError("qrel candidate universe disagrees with frozen ranking")
    evaluation = evaluate_rankings(rankings, qrels, k=10)
    metrics = {
        "phase_a_receipt_sha256": _sha256(source / "ranking_receipt.json"),
        "qrels_path": str(qrels_path),
        "qrels_sha256": _sha256(Path(qrels_path)),
        "method": receipt["method"],
        "ndcg_at_10": evaluation.macro_ndcg_at_k,
        "mrr": evaluation.mean_reciprocal_rank,
        "zero_ndcg_count": sum(
            item.ndcg_at_k == 0.0 for item in evaluation.per_query.values()
        ),
        "per_query": {
            query_id: {
                "ndcg_at_10": item.ndcg_at_k,
                "reciprocal_rank": item.reciprocal_rank,
            }
            for query_id, item in evaluation.per_query.items()
        },
    }
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    metrics_path = destination / "metrics.json"
    _write_json(metrics_path, metrics)
    result = {**metrics, "metrics_path": str(metrics_path), "metrics_sha256": _sha256(metrics_path)}
    _write_json(destination / "evaluation_receipt.json", result)
    return result


def _load_frozen_scores(
    source: Path,
    *,
    query_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> NDArray[np.float64]:
    query_positions = {query_id: index for index, query_id in enumerate(query_ids)}
    candidate_positions = {
        candidate_id: index for index, candidate_id in enumerate(candidate_ids)
    }
    values = np.full((len(query_ids), len(candidate_ids)), np.nan, dtype=np.float64)
    seen: set[tuple[str, str]] = set()
    with (source / "scores.tsv").open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            key = (row["query_id"], row["candidate_id"])
            if key in seen or key[0] not in query_positions or key[1] not in candidate_positions:
                raise ValueError("frozen scores contain a duplicate or unknown pair")
            score = float(row["score"])
            if not math.isfinite(score):
                raise ValueError("frozen scores contain a non-finite value")
            seen.add(key)
            values[query_positions[key[0]], candidate_positions[key[1]]] = score
    if len(seen) != len(query_ids) * len(candidate_ids) or np.isnan(values).any():
        raise ValueError("frozen scores are incomplete")
    return values


def compare_frozen_ranker_artifacts(
    first_dir: str | Path,
    second_dir: str | Path,
    *,
    score_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Report strict and practical repeatability for two qrel-free runs."""

    first = Path(first_dir)
    second = Path(second_dir)
    first_receipt, first_config = _validate_frozen_artifact(first)
    second_receipt, second_config = _validate_frozen_artifact(second)
    if first_receipt["method"] != second_receipt["method"]:
        raise ValueError("repeat artifacts use different methods")
    first_queries = tuple(load_queries_tsv(first_config["query_path"]))
    second_queries = tuple(load_queries_tsv(second_config["query_path"]))
    first_candidates = load_candidate_manifest(first_config["candidate_path"])
    second_candidates = load_candidate_manifest(second_config["candidate_path"])
    if first_queries != second_queries or first_candidates != second_candidates:
        raise ValueError("repeat artifacts use different query or candidate inventories")
    first_rankings = rankings_from_run(first / "run.trec")
    second_rankings = rankings_from_run(second / "run.trec")
    first_scores = _load_frozen_scores(
        first, query_ids=first_queries, candidate_ids=first_candidates
    )
    second_scores = _load_frozen_scores(
        second, query_ids=first_queries, candidate_ids=first_candidates
    )
    maximum_difference = float(np.max(np.abs(first_scores - second_scores)))
    strict = first_rankings == second_rankings
    top_10 = all(
        first_rankings[query_id][:10] == second_rankings[query_id][:10]
        for query_id in first_queries
    )
    top_20 = all(
        first_rankings[query_id][:20] == second_rankings[query_id][:20]
        for query_id in first_queries
    )
    changed_queries = [
        query_id
        for query_id in first_queries
        if first_rankings[query_id] != second_rankings[query_id]
    ]
    return {
        "method": first_receipt["method"],
        "strict_rankings_identical": strict,
        "top_10_rankings_identical": top_10,
        "top_20_rankings_identical": top_20,
        "changed_query_count": len(changed_queries),
        "changed_query_ids": changed_queries,
        "maximum_absolute_score_difference": maximum_difference,
        "score_tolerance": score_tolerance,
        "scores_within_tolerance": maximum_difference <= score_tolerance,
        "raw_outputs_byte_identical": _sha256(first / "raw_outputs.jsonl")
        == _sha256(second / "raw_outputs.jsonl"),
        "practical_repeatability_passed": top_10 and maximum_difference <= score_tolerance,
    }


__all__ = [
    "LLMRankerConfig",
    "LLMRankerPhaseAResult",
    "compare_frozen_ranker_artifacts",
    "evaluate_frozen_ranker_artifact",
    "run_listwise_phase_a",
    "run_pointwise_phase_a",
    "write_ranker_phase_a_artifacts",
]
