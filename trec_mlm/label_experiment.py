"""Experiment orchestration and artifacts for zero-shot expert-label retrieval."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
from typing import Any

import numpy as np
from numpy.typing import NDArray

from trec_mlm.label_retrieval import (
    LabelEncoderSpec,
    LabelEvaluation,
    ROCM_IMAGE,
    TASK_INSTRUCTION,
    advancement_gate,
    encoder_spec_by_key,
    evaluate_label_rankings,
    format_document,
    format_query,
    rank_label_scores,
)
from trec_mlm.metrics import EvaluationResult
from trec_mlm.query_knn import load_complete_qrels, load_queries_tsv
from trec_mlm.runio import validate_trec_run, write_trec_run
from trec_mlm.runio import rankings_from_run


ISSUE_11_EXAMPLES: Mapping[str, str] = {
    "56fc3b342603e7140040a037": "Linguistics",
    "57265eeaf1498d1400e8dd76": "Landscape",
}


@dataclass(frozen=True, slots=True)
class LabelExperimentConfig:
    """Frozen inputs and summary settings for one label encoder."""

    query_path: str = "data/dev/llm_dev_data.tsv"
    qrels_path: str = "data/dev/llm_dev_qrels.txt"
    candidate_path: str = "reports/trec_2026_discovery_profile_audit.json"
    encoder_key: str = "mpnet_label_control"
    label_representation: str = "bare"
    dense_per_query_path: str = "reports/trec_2026_dense_query_knn_per_query.tsv"
    bootstrap_resamples: int = 10_000
    seed: int = 2026
    batch_size: int = 32
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class LabelExperimentResult:
    """One complete zero-shot label run and its post-ranking evaluation."""

    config: LabelExperimentConfig
    encoder_spec: LabelEncoderSpec
    query_ids: tuple[str, ...]
    query_texts: tuple[str, ...]
    model_ids: tuple[str, ...]
    formatted_queries: tuple[str, ...]
    formatted_documents: tuple[str, ...]
    query_embeddings: NDArray[np.float32]
    label_embeddings: NDArray[np.float32]
    scores: NDArray[np.float32]
    rankings: dict[str, list[str]]
    evaluation: EvaluationResult
    zero_ndcg_count: int
    encoder_metadata: dict[str, Any]
    summary: dict[str, Any]


def load_candidate_manifest(path: str | Path) -> tuple[str, ...]:
    """Load a stable candidate universe without reading qrel grades."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    candidates: Any = (
        payload.get("candidate_ids") if isinstance(payload, dict) else None
    )
    if candidates is None and isinstance(payload, dict):
        plain = payload.get("plain")
        if isinstance(plain, dict):
            candidates = plain.get("candidate_ids")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{source}: candidate_ids must be a non-empty list")
    if any(not isinstance(item, str) or not item for item in candidates):
        raise ValueError(f"{source}: candidate IDs must be non-empty strings")
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"{source}: candidate IDs must be unique")
    return tuple(sorted(candidates))


def run_label_experiment(
    config: LabelExperimentConfig,
    *,
    query_embeddings: NDArray[np.floating[Any]] | None = None,
    label_embeddings: NDArray[np.floating[Any]] | None = None,
    encoder_metadata: Mapping[str, Any] | None = None,
) -> LabelExperimentResult:
    """Rank from embeddings first, then reveal qrels for evaluation."""

    if (query_embeddings is None) != (label_embeddings is None):
        raise ValueError("query and label embeddings must be supplied together")
    spec = encoder_spec_by_key(config.encoder_key)
    queries = load_queries_tsv(config.query_path)
    query_ids = tuple(queries)
    query_texts = tuple(queries[query_id] for query_id in query_ids)
    model_ids = load_candidate_manifest(config.candidate_path)
    formatted_queries = tuple(format_query(spec, text) for text in query_texts)
    formatted_documents = tuple(
        format_document(
            spec,
            model_id,
            representation=config.label_representation,
        )
        for model_id in model_ids
    )
    if query_embeddings is None or label_embeddings is None:
        from trec_mlm.label_inference import encode_label_inputs

        query_embeddings, label_embeddings, generated_metadata = encode_label_inputs(
            spec,
            formatted_queries=formatted_queries,
            formatted_documents=formatted_documents,
            batch_size=config.batch_size,
            seed=config.seed,
        )
        metadata = dict(generated_metadata)
    else:
        metadata = dict(encoder_metadata or {})

    scored = rank_label_scores(
        query_embeddings=query_embeddings,
        label_embeddings=label_embeddings,
        query_ids=query_ids,
        model_ids=model_ids,
    )

    # This is the first point at which qrels are loaded.  They cannot affect
    # the formatted inputs, embeddings, raw scores, or fixed rankings above.
    qrels = load_complete_qrels(config.qrels_path, expected_query_ids=query_ids)
    qrel_models = set(next(iter(qrels.values())))
    if qrel_models != set(model_ids):
        raise ValueError("candidate manifest does not match qrel candidate universe")
    evaluated = evaluate_label_rankings(scored.rankings, qrels)
    zero_count = sum(
        item.ndcg_at_k == 0.0 for item in evaluated.evaluation.per_query.values()
    )
    summary = _build_model_summary(
        config=config,
        spec=spec,
        queries=queries,
        qrels=qrels,
        evaluation=evaluated.evaluation,
        zero_count=zero_count,
    )
    return LabelExperimentResult(
        config=config,
        encoder_spec=spec,
        query_ids=query_ids,
        query_texts=query_texts,
        model_ids=model_ids,
        formatted_queries=formatted_queries,
        formatted_documents=formatted_documents,
        query_embeddings=np.asarray(query_embeddings, dtype=np.float32),
        label_embeddings=np.asarray(label_embeddings, dtype=np.float32),
        scores=scored.scores,
        rankings=scored.rankings,
        evaluation=evaluated.evaluation,
        zero_ndcg_count=zero_count,
        encoder_metadata=metadata,
        summary=summary,
    )


def write_label_experiment_artifacts(
    result: LabelExperimentResult,
    output_dir: str | Path,
) -> dict[str, str]:
    """Write complete embeddings, scores, run, metrics, and validation."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    resolved_config = asdict(result.config)
    resolved_config.update(
        {
            "encoder_spec": asdict(result.encoder_spec),
            "input_sha256": {
                result.config.query_path: _sha256(Path(result.config.query_path)),
                result.config.qrels_path: _sha256(Path(result.config.qrels_path)),
                result.config.candidate_path: _sha256(
                    Path(result.config.candidate_path)
                ),
            },
            "code_revision": _git_revision(),
            "git_worktree_status": _git_worktree_status(),
            "python_version": sys.version,
            "platform": platform.platform(),
        }
    )
    if result.encoder_spec.runtime in {"vllm_rocm", "transformers_rocm"}:
        resolved_config["container_image"] = ROCM_IMAGE
    query_templates = {
        "plain": "{question}",
        "qwen_instruct": "Instruct: {instruction}\nQuery:{question}",
        "e5_instruct": "Instruct: {instruction}\nQuery: {question}",
        "jina_query": "Query: {question}",
        "nomic_search": "search_query: {question}",
    }
    document_template = (
        "{cleaned_label}"
        if result.config.label_representation == "bare"
        else (
            "This LLM is most likely to correctly answer questions about "
            "{cleaned_label}."
        )
    )
    if result.encoder_spec.document_style == "jina_document":
        document_template = f"Document: {document_template}"
    elif result.encoder_spec.document_style == "nomic_search":
        document_template = f"search_document: {document_template}"
    resolved_config["prompt_metadata"] = {
        "instruction": (
            TASK_INSTRUCTION
            if result.encoder_spec.query_style in {"qwen_instruct", "e5_instruct"}
            else None
        ),
        "query_template": query_templates[result.encoder_spec.query_style],
        "document_template": document_template,
        "label_cleaning": (
            "replace underscores and parentheses with spaces; collapse whitespace"
        ),
    }
    _write_json(destination / "config.json", resolved_config)
    _write_json(destination / "encoder_metadata.json", result.encoder_metadata)
    np.save(destination / "query_embeddings.npy", result.query_embeddings)
    np.save(destination / "label_embeddings.npy", result.label_embeddings)

    rank_by_pair = {
        (query_id, model_id): rank
        for query_id, ranking in result.rankings.items()
        for rank, model_id in enumerate(ranking, start=1)
    }
    model_positions = {
        model_id: index for index, model_id in enumerate(result.model_ids)
    }
    with (destination / "scores.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(("query_id", "model_id", "score", "rank"))
        for query_index, query_id in enumerate(result.query_ids):
            for model_id in result.model_ids:
                writer.writerow(
                    (
                        query_id,
                        model_id,
                        f"{float(result.scores[query_index, model_positions[model_id]]):.12g}",
                        rank_by_pair[(query_id, model_id)],
                    )
                )

    run_id = result.config.run_id or f"label_{result.encoder_spec.key}"
    write_trec_run(destination / "run.trec", result.rankings, run_id=run_id)
    with (destination / "per_query.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(
            (
                "query_id",
                "ndcg_at_10",
                "reciprocal_rank",
                "zero_ndcg",
                "first_relevant_rank",
            )
        )
        qrels = load_complete_qrels(
            result.config.qrels_path,
            expected_query_ids=result.query_ids,
        )
        for query_id in result.query_ids:
            metrics = result.evaluation.per_query[query_id]
            first_relevant = next(
                (
                    rank
                    for rank, model_id in enumerate(result.rankings[query_id], start=1)
                    if qrels[query_id][model_id] > 0
                ),
                None,
            )
            writer.writerow(
                (
                    query_id,
                    f"{metrics.ndcg_at_k:.12g}",
                    f"{metrics.reciprocal_rank:.12g}",
                    int(metrics.ndcg_at_k == 0.0),
                    first_relevant,
                )
            )
    _write_json(destination / "summary.json", result.summary)

    core_files = (
        "config.json",
        "encoder_metadata.json",
        "query_embeddings.npy",
        "label_embeddings.npy",
        "scores.tsv",
        "run.trec",
        "per_query.tsv",
        "summary.json",
    )
    validation = {
        "run": validate_trec_run(
            destination / "run.trec",
            expected_query_ids=result.query_ids,
            expected_model_ids=result.model_ids,
        ),
        "embeddings": {
            "query_shape": list(result.query_embeddings.shape),
            "label_shape": list(result.label_embeddings.shape),
            "query_finite": bool(np.isfinite(result.query_embeddings).all()),
            "label_finite": bool(np.isfinite(result.label_embeddings).all()),
        },
        "score_shape": list(result.scores.shape),
        "artifact_sha256": {name: _sha256(destination / name) for name in core_files},
    }
    _write_json(destination / "validation.json", validation)
    return {name: str(destination / name) for name in (*core_files, "validation.json")}


def load_label_experiment_artifacts(
    artifact_dir: str | Path,
) -> LabelExperimentResult:
    """Reconstruct one validated result from its durable artifact directory."""

    source = Path(artifact_dir)
    config_payload = json.loads((source / "config.json").read_text(encoding="utf-8"))
    config_fields = {
        field: config_payload[field]
        for field in LabelExperimentConfig.__dataclass_fields__
        if field in config_payload
    }
    config = LabelExperimentConfig(**config_fields)
    spec = encoder_spec_by_key(config.encoder_key)
    queries = load_queries_tsv(config.query_path)
    query_ids = tuple(queries)
    query_texts = tuple(queries[item] for item in query_ids)
    model_ids = load_candidate_manifest(config.candidate_path)
    rankings = rankings_from_run(source / "run.trec")
    qrels = load_complete_qrels(config.qrels_path, expected_query_ids=query_ids)
    evaluated = evaluate_label_rankings(rankings, qrels).evaluation
    query_embeddings = np.load(source / "query_embeddings.npy", allow_pickle=False)
    label_embeddings = np.load(source / "label_embeddings.npy", allow_pickle=False)
    scores = np.zeros((len(query_ids), len(model_ids)), dtype=np.float32)
    query_positions = {query_id: index for index, query_id in enumerate(query_ids)}
    model_positions = {model_id: index for index, model_id in enumerate(model_ids)}
    with (source / "scores.tsv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            scores[
                query_positions[row["query_id"]], model_positions[row["model_id"]]
            ] = float(row["score"])
    expected_shape = (len(query_ids), len(model_ids))
    if scores.shape != expected_shape:
        raise ValueError("loaded score matrix shape is invalid")
    validation = validate_trec_run(
        source / "run.trec",
        expected_query_ids=query_ids,
        expected_model_ids=model_ids,
    )
    if validation["row_count"] != len(query_ids) * len(model_ids):
        raise ValueError("loaded run is incomplete")
    zero_count = sum(item.ndcg_at_k == 0.0 for item in evaluated.per_query.values())
    return LabelExperimentResult(
        config=config,
        encoder_spec=spec,
        query_ids=query_ids,
        query_texts=query_texts,
        model_ids=model_ids,
        formatted_queries=tuple(format_query(spec, item) for item in query_texts),
        formatted_documents=tuple(
            format_document(
                spec,
                item,
                representation=config.label_representation,
            )
            for item in model_ids
        ),
        query_embeddings=np.asarray(query_embeddings, dtype=np.float32),
        label_embeddings=np.asarray(label_embeddings, dtype=np.float32),
        scores=scores,
        rankings=rankings,
        evaluation=evaluated,
        zero_ndcg_count=zero_count,
        encoder_metadata=json.loads(
            (source / "encoder_metadata.json").read_text(encoding="utf-8")
        ),
        summary=json.loads((source / "summary.json").read_text(encoding="utf-8")),
    )


def compare_repeated_results(
    first: LabelExperimentResult,
    second: LabelExperimentResult,
    *,
    score_tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Verify the frozen repeatability requirement for an executed encoder."""

    if first.encoder_spec.key != second.encoder_spec.key:
        raise ValueError("repeat results must use the same encoder")
    if first.query_ids != second.query_ids or first.model_ids != second.model_ids:
        raise ValueError("repeat results must use identical query and candidate IDs")
    maximum_score_difference = float(np.max(np.abs(first.scores - second.scores)))
    rankings_identical = first.rankings == second.rankings
    metrics_identical = (
        first.evaluation.macro_ndcg_at_k == second.evaluation.macro_ndcg_at_k
        and first.evaluation.mean_reciprocal_rank
        == second.evaluation.mean_reciprocal_rank
        and first.zero_ndcg_count == second.zero_ndcg_count
    )
    scores_within_tolerance = maximum_score_difference <= score_tolerance
    return {
        "passed": rankings_identical and metrics_identical and scores_within_tolerance,
        "rankings_identical": rankings_identical,
        "metrics_identical": metrics_identical,
        "scores_within_tolerance": scores_within_tolerance,
        "score_tolerance": score_tolerance,
        "maximum_absolute_score_difference": maximum_score_difference,
    }


def build_sweep_summary(
    *,
    baseline: LabelExperimentResult,
    model_results: Mapping[str, LabelExperimentResult],
    repeat_results: Mapping[str, LabelExperimentResult] | None = None,
    dense_reference: EvaluationResult | None = None,
    maximum_zero_ndcg: int = 131,
    n_resamples: int = 10_000,
    seed: int = 2026,
    issue_examples: Mapping[str, str] = ISSUE_11_EXAMPLES,
) -> dict[str, Any]:
    """Compare completed models and apply the frozen conditional stage rule."""

    baseline_evaluation = LabelEvaluation(
        rankings=baseline.rankings,
        evaluation=baseline.evaluation,
    )
    models: dict[str, Any] = {}
    passed: set[str] = set()
    repeats = repeat_results or {}
    for offset, (key, result) in enumerate(model_results.items()):
        gate = advancement_gate(
            baseline=baseline_evaluation,
            candidate=LabelEvaluation(
                rankings=result.rankings,
                evaluation=result.evaluation,
            ),
            maximum_zero_ndcg=maximum_zero_ndcg,
            n_resamples=n_resamples,
            seed=seed + offset,
        )
        repeat_validation = (
            compare_repeated_results(result, repeats[key]) if key in repeats else None
        )
        dense_comparison = (
            advancement_gate(
                baseline=LabelEvaluation(rankings={}, evaluation=dense_reference),
                candidate=LabelEvaluation(
                    rankings=result.rankings,
                    evaluation=result.evaluation,
                ),
                maximum_zero_ndcg=len(result.query_ids),
                n_resamples=n_resamples,
                seed=seed + 20_000 + offset,
            )
            if dense_reference is not None
            else None
        )
        eligible = (
            gate["passed"]
            and repeat_validation is not None
            and repeat_validation["passed"]
        )
        if eligible:
            passed.add(key)
        models[key] = {
            "stage": result.encoder_spec.stage,
            "metrics": _metric_summary(result.evaluation),
            "zero_ndcg_count": result.zero_ndcg_count,
            "density_slices": result.summary["density_slices"],
            "question_type_slices": result.summary["question_type_slices"],
            "gate": gate,
            "repeat_validation": repeat_validation,
            "dense_reference_comparison": dense_comparison,
            "eligible_for_advancement": eligible,
        }

    stage_two: list[str] = []
    if "qwen3_embedding_8b" in passed:
        stage_two.extend(("qwen3_embedding_4b", "qwen3_embedding_0_6b"))
    if passed & {"qwen3_embedding_8b", "e5_mistral_7b_instruct"}:
        stage_two.append("jina_v5_small_retrieval")
    qrels = load_complete_qrels(
        baseline.config.qrels_path,
        expected_query_ids=baseline.query_ids,
    )
    issue_summary: dict[str, Any] = {}
    query_text_by_id = dict(zip(baseline.query_ids, baseline.query_texts, strict=True))
    for query_id, target_label in issue_examples.items():
        if query_id not in baseline.rankings:
            continue
        issue_summary[query_id] = {
            "question": query_text_by_id[query_id],
            "target_label": target_label,
            "target_grade": qrels[query_id][target_label],
            "baseline_rank": baseline.rankings[query_id].index(target_label) + 1,
            "baseline_ndcg_at_10": baseline.evaluation.per_query[query_id].ndcg_at_k,
            "model_ranks": {
                key: result.rankings[query_id].index(target_label) + 1
                for key, result in model_results.items()
            },
            "model_ndcg_at_10": {
                key: result.evaluation.per_query[query_id].ndcg_at_k
                for key, result in model_results.items()
            },
        }
    return {
        "protocol": "staged_instruction_aware_expert_label_model_sweep",
        "baseline": {
            "encoder": baseline.encoder_spec.key,
            "metrics": _metric_summary(baseline.evaluation),
            "zero_ndcg_count": baseline.zero_ndcg_count,
        },
        "dense_reference": (
            _metric_summary(dense_reference) if dense_reference is not None else None
        ),
        "models": models,
        "issue_11_examples": issue_summary,
        "stage_two": {
            "encoders_to_run": stage_two,
            "stopped": not stage_two,
            "reason": (
                "at_least_one_large_model_passed"
                if stage_two
                else "neither_large_model_passed"
            ),
        },
    }


def build_diversity_summary(
    *,
    baseline: LabelExperimentResult,
    e5_reference: LabelExperimentResult,
    model_results: Mapping[str, LabelExperimentResult],
    repeat_results: Mapping[str, LabelExperimentResult],
    n_resamples: int = 10_000,
    seed: int = 2026,
    issue_examples: Mapping[str, str] = ISSUE_11_EXAMPLES,
) -> dict[str, Any]:
    """Build the post-gate small/diverse comparison receipt."""

    baseline_evaluation = LabelEvaluation(baseline.rankings, baseline.evaluation)
    e5_evaluation = LabelEvaluation(e5_reference.rankings, e5_reference.evaluation)
    conditions: dict[str, Any] = {}
    for offset, (condition_id, result) in enumerate(model_results.items()):
        candidate = LabelEvaluation(result.rankings, result.evaluation)
        metadata = result.encoder_metadata
        conditions[condition_id] = {
            "encoder": result.encoder_spec.key,
            "label_representation": result.config.label_representation,
            "metrics": _metric_summary(result.evaluation),
            "zero_ndcg_count": result.zero_ndcg_count,
            "mpnet_comparison": advancement_gate(
                baseline=baseline_evaluation,
                candidate=candidate,
                maximum_zero_ndcg=131,
                n_resamples=n_resamples,
                seed=seed + offset,
            ),
            "e5_comparison": advancement_gate(
                baseline=e5_evaluation,
                candidate=candidate,
                maximum_zero_ndcg=len(result.query_ids),
                n_resamples=n_resamples,
                seed=seed + 10_000 + offset,
            ),
            "repeat_validation": compare_repeated_results(
                result,
                repeat_results[condition_id],
            ),
            "density_slices": result.summary["density_slices"],
            "question_type_slices": result.summary["question_type_slices"],
            "resources": {
                "parameter_count": metadata.get("parameter_count"),
                "encoding_seconds": metadata.get("encoding_seconds"),
                "peak_memory_allocated_bytes": metadata.get(
                    "peak_memory_allocated_bytes"
                ),
                "peak_memory_reserved_bytes": metadata.get(
                    "peak_memory_reserved_bytes"
                ),
            },
        }

    contextualization: dict[str, Any] = {}
    by_encoder: dict[str, dict[str, LabelExperimentResult]] = defaultdict(dict)
    for condition_id, result in model_results.items():
        representation = condition_id.rsplit("/", 1)[-1]
        by_encoder[result.encoder_spec.key][representation] = result
    for offset, (encoder, representations) in enumerate(by_encoder.items()):
        if {"bare", "expert_capability"} <= representations.keys():
            bare = representations["bare"]
            contextual = representations["expert_capability"]
            contextualization[encoder] = advancement_gate(
                baseline=LabelEvaluation(bare.rankings, bare.evaluation),
                candidate=LabelEvaluation(contextual.rankings, contextual.evaluation),
                maximum_zero_ndcg=len(contextual.query_ids),
                n_resamples=n_resamples,
                seed=seed + 20_000 + offset,
            )

    qrels = load_complete_qrels(
        baseline.config.qrels_path,
        expected_query_ids=baseline.query_ids,
    )
    query_text_by_id = dict(zip(baseline.query_ids, baseline.query_texts, strict=True))
    examples: dict[str, Any] = {}
    for query_id, target_label in issue_examples.items():
        if query_id not in baseline.rankings:
            continue
        examples[query_id] = {
            "question": query_text_by_id[query_id],
            "target_label": target_label,
            "target_grade": qrels[query_id][target_label],
            "mpnet_rank": baseline.rankings[query_id].index(target_label) + 1,
            "e5_rank": e5_reference.rankings[query_id].index(target_label) + 1,
            "condition_ranks": {
                key: result.rankings[query_id].index(target_label) + 1
                for key, result in model_results.items()
            },
        }

    return {
        "protocol": "small_diverse_label_embedding_sweep",
        "baseline": {
            "encoder": baseline.encoder_spec.key,
            "metrics": _metric_summary(baseline.evaluation),
            "zero_ndcg_count": baseline.zero_ndcg_count,
        },
        "e5_reference": {
            "encoder": e5_reference.encoder_spec.key,
            "label_representation": e5_reference.config.label_representation,
            "metrics": _metric_summary(e5_reference.evaluation),
            "zero_ndcg_count": e5_reference.zero_ndcg_count,
        },
        "conditions": conditions,
        "within_model_contextualization": contextualization,
        "issue_11_examples": examples,
        "decision": {
            "status": "post_gate_exploratory_evidence",
            "selected_deployment_model": None,
            "fusion_started": False,
            "llm_ranker_started": False,
        },
    }


def write_diversity_summary(
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write the diverse-sweep JSON and concise decision receipt."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "diversity_summary.json", summary)
    lines = [
        "# Small diverse label-embedding sweep\n\n",
        "This is post-gate exploratory evidence. No deployment model is selected.\n\n",
        "| Condition | nDCG@10 | MRR | Zeros | Repeat | MPNet nDCG 95% CI |\n",
        "|---|---:|---:|---:|---|---|\n",
    ]
    for condition_id, condition in summary["conditions"].items():
        interval = condition["mpnet_comparison"]["ndcg_delta"]
        repeat = condition["repeat_validation"]
        lines.append(
            f"| `{condition_id}` | {condition['metrics']['ndcg_at_10']:.4f} | "
            f"{condition['metrics']['mrr']:.4f} | {condition['zero_ndcg_count']} | "
            f"{'PASS' if repeat['passed'] else 'FAIL'} | "
            f"[{interval['lower']:+.4f}, {interval['upper']:+.4f}] |\n"
        )
    lines.append("\n## Context effect\n\n")
    for encoder, comparison in summary["within_model_contextualization"].items():
        interval = comparison["ndcg_delta"]
        lines.append(
            f"- `{encoder}`: {interval['mean_difference']:+.4f} nDCG@10, "
            f"95% CI [{interval['lower']:+.4f}, {interval['upper']:+.4f}].\n"
        )
    (destination / "decision.md").write_text("".join(lines), encoding="utf-8")
    return {
        "diversity_summary.json": str(destination / "diversity_summary.json"),
        "decision.md": str(destination / "decision.md"),
    }


def write_sweep_summary(
    summary: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write the combined machine-readable result and plain-language receipt."""

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    _write_json(destination / "sweep_summary.json", summary)
    baseline = summary["baseline"]
    lines = [
        "# Instruction-aware expert-label sweep decision\n\n",
        "This is exploratory development evidence. It does not select a deployment "
        "model or authorize fusion.\n\n",
        "## Baseline\n\n",
        f"- Encoder: `{baseline['encoder']}`\n",
        f"- nDCG@10: {baseline['metrics']['ndcg_at_10']:.4f}\n",
        f"- MRR: {baseline['metrics']['mrr']:.4f}\n",
        f"- Zero-nDCG queries: {baseline['zero_ndcg_count']}\n\n",
    ]
    dense = summary.get("dense_reference")
    if dense is not None:
        lines.extend(
            (
                "## Frozen dense reference\n\n",
                f"- nDCG@10: {dense['ndcg_at_10']:.4f}\n",
                f"- MRR: {dense['mrr']:.4f}\n\n",
            )
        )
    lines.append("## Completed models\n\n")
    for key, model in summary["models"].items():
        gate = model["gate"]
        repeat = model.get("repeat_validation")
        lines.extend(
            (
                f"### {key}\n\n",
                f"- nDCG@10: {model['metrics']['ndcg_at_10']:.4f}\n",
                f"- MRR: {model['metrics']['mrr']:.4f}\n",
                f"- Zero-nDCG queries: {model['zero_ndcg_count']}\n",
                f"- Stage gate: {'PASS' if gate['passed'] else 'FAIL'}\n",
                (
                    "- Repeat validation: not supplied\n"
                    if repeat is None
                    else f"- Repeat validation: {'PASS' if repeat['passed'] else 'FAIL'} "
                    f"(max score difference {repeat['maximum_absolute_score_difference']:.3g})\n"
                ),
                f"- Paired nDCG 95% CI: [{gate['ndcg_delta']['lower']:+.4f}, "
                f"{gate['ndcg_delta']['upper']:+.4f}]\n",
                f"- Paired MRR 95% CI: [{gate['mrr_delta']['lower']:+.4f}, "
                f"{gate['mrr_delta']['upper']:+.4f}]\n",
                "- nDCG win/tie/loss: "
                f"{gate['ndcg_win_tie_loss']['wins']}/"
                f"{gate['ndcg_win_tie_loss']['ties']}/"
                f"{gate['ndcg_win_tie_loss']['losses']}\n",
                "- Zero-query transitions (recovered/new/persisting): "
                f"{gate['zero_ndcg_transitions']['baseline_zero_candidate_nonzero']}/"
                f"{gate['zero_ndcg_transitions']['baseline_nonzero_candidate_zero']}/"
                f"{gate['zero_ndcg_transitions']['baseline_zero_candidate_zero']}\n\n",
            )
        )
    stage_two = summary["stage_two"]
    examples = summary.get("issue_11_examples", {})
    if examples:
        lines.append("## Issue #11 examples\n\n")
        for example in examples.values():
            model_ranks = ", ".join(
                f"{key}={rank}" for key, rank in example["model_ranks"].items()
            )
            lines.extend(
                (
                    f"- {example['question']} Target `{example['target_label']}`: "
                    f"MPNet rank {example['baseline_rank']}; {model_ranks}.\n",
                )
            )
        lines.append("\n")
    lines.extend(
        (
            "## Stage 2\n\n",
            (
                "Run: " + ", ".join(f"`{key}`" for key in stage_two["encoders_to_run"])
                if stage_two["encoders_to_run"]
                else "Stop: neither large model passed the frozen gate."
            ),
            "\n",
        )
    )
    (destination / "decision.md").write_text("".join(lines), encoding="utf-8")
    return {
        "sweep_summary.json": str(destination / "sweep_summary.json"),
        "decision.md": str(destination / "decision.md"),
    }


def _build_model_summary(
    *,
    config: LabelExperimentConfig,
    spec: LabelEncoderSpec,
    queries: Mapping[str, str],
    qrels: Mapping[str, Mapping[str, float]],
    evaluation: EvaluationResult,
    zero_count: int,
) -> dict[str, Any]:
    per_query_ndcg = {
        query_id: values.ndcg_at_k for query_id, values in evaluation.per_query.items()
    }
    density_groups: dict[str, list[str]] = defaultdict(list)
    question_groups: dict[str, list[str]] = defaultdict(list)
    for query_id, grades in qrels.items():
        positives = sum(value > 0 for value in grades.values())
        density = (
            "exactly_1_positive"
            if positives == 1
            else "2_to_5_positives"
            if positives <= 5
            else "6_to_20_positives"
            if positives <= 20
            else "more_than_20_positives"
        )
        density_groups[density].append(query_id)
        words = re.findall(r"[a-z]+", queries[query_id].lower())
        question_type = (
            words[0]
            if words
            and words[0] in {"who", "what", "when", "where", "which", "why", "how"}
            else "other"
        )
        question_groups[question_type].append(query_id)

    return {
        "protocol": {
            "qrels_loaded_only_after_rankings_are_fixed": True,
            "candidate_score": "cosine_similarity",
            "tie_break": "candidate_id",
            "exploratory_development_evidence": True,
        },
        "encoder": asdict(spec),
        "data": {
            "query_count": len(queries),
            "candidate_count": len(next(iter(qrels.values()))),
        },
        "metrics": _metric_summary(evaluation),
        "zero_ndcg_count": zero_count,
        "density_slices": _slice_summary(density_groups, per_query_ndcg),
        "question_type_slices": _slice_summary(question_groups, per_query_ndcg),
        "runtime": {
            "bootstrap_resamples": config.bootstrap_resamples,
            "seed": config.seed,
        },
    }


def _slice_summary(
    groups: Mapping[str, Sequence[str]],
    values: Mapping[str, float],
) -> dict[str, dict[str, float | int]]:
    return {
        label: {
            "query_count": len(query_ids),
            "mean_ndcg_at_10": statistics.fmean(values[item] for item in query_ids),
            "zero_ndcg_count": sum(values[item] == 0.0 for item in query_ids),
        }
        for label, query_ids in sorted(groups.items())
    }


def _metric_summary(evaluation: EvaluationResult) -> dict[str, float]:
    return {
        "ndcg_at_10": evaluation.macro_ndcg_at_k,
        "mrr": evaluation.mean_reciprocal_rank,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _git_worktree_status() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--short"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.rstrip()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


__all__ = [
    "LabelExperimentConfig",
    "LabelExperimentResult",
    "ISSUE_11_EXAMPLES",
    "build_sweep_summary",
    "compare_repeated_results",
    "load_label_experiment_artifacts",
    "load_candidate_manifest",
    "run_label_experiment",
    "write_label_experiment_artifacts",
    "write_sweep_summary",
]
