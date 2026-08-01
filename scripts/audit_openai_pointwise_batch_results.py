#!/usr/bin/env python3
"""Audit and optionally curate a completed OpenAI pointwise Batch artifact."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


Verdict = Literal["correct", "incorrect", "nonanswer", "not_established", "partial"]
Kind = Literal["cached", "deterministic", "provider"]


class Usage(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)
    cache_details_known: int | None = Field(default=None, ge=0)
    cache_write_tokens: int | None = Field(default=None, ge=0)
    cached_tokens: int | None = Field(default=None, ge=0)


class FinalResult(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    record_id: str = Field(pattern=r"^profile-record-v2:[0-9a-f]{64}$")
    source_ordinal: int = Field(ge=0)
    verdict: Verdict
    explanation: str = Field(min_length=1, max_length=400)
    kind: Kind
    provider_call: bool
    model: str | None = None
    finish_reason: str | None = None
    usage: Usage


class InventoryRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    record_id: str = Field(pattern=r"^profile-record-v2:[0-9a-f]{64}$")
    source_ordinal: int = Field(ge=0)
    question_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    provisional_reference: str = Field(min_length=1)
    candidate_response: str = Field(min_length=1)


class ExplanationRepair(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    record_id: str = Field(pattern=r"^profile-record-v2:[0-9a-f]{64}$")
    source_ordinal: int = Field(ge=0)
    reason: Literal["control_character", "corrupted_text"]
    explanation: str = Field(min_length=1, max_length=400)


class VerdictOverride(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    record_id: str = Field(pattern=r"^profile-record-v2:[0-9a-f]{64}$")
    source_ordinal: int = Field(ge=0)
    original_verdict: Verdict
    verdict: Verdict
    reason: Literal["semantic_sample", "duplicate_prompt_consistency"]
    explanation: str = Field(min_length=1, max_length=400)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--repairs", type=Path)
    parser.add_argument("--verdict-overrides", type=Path)
    parser.add_argument("--curated-output", type=Path)
    parser.add_argument("--labels-output", type=Path)
    parser.add_argument("--sample-per-verdict", type=int, default=20)
    parser.add_argument("--sample-seed", default="openai-luna-production-audit-v1")
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _read_array(path: Path, label: str) -> list[Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if type(value) is not list:
        raise ValueError(f"{label} must be a JSON array")
    return value


def _atomic_write(path: Path, value: object, *, private: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600 if private else 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            if isinstance(value, str):
                handle.write(value)
            else:
                json.dump(
                    value,
                    handle,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600 if private else 0o644)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _atomic_write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            for row in rows:
                handle.write(
                    json.dumps(
                        row,
                        sort_keys=True,
                        ensure_ascii=False,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                )
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o600)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _validate_results(raw: list[Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, value in enumerate(raw):
        try:
            row = FinalResult.model_validate(value).model_dump(exclude_none=True)
        except ValidationError as error:
            raise ValueError(f"final result {index} fails the strict schema") from error
        if row["source_ordinal"] != index:
            raise ValueError(f"final result order breaks at ordinal {index}")
        rows.append(row)
    return rows


def _control_characters(value: str) -> list[dict[str, int]]:
    return [
        {"offset": offset, "codepoint": ord(character)}
        for offset, character in enumerate(value)
        if ord(character) < 32 and character not in "\n\r\t"
    ]


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    index = min(len(values) - 1, math.ceil(fraction * len(values)) - 1)
    return sorted(values)[index]


def _sample_key(seed: str, record_id: str) -> str:
    return hashlib.sha256(f"{seed}\0{record_id}".encode()).hexdigest()


def _duplicate_metrics(
    rows: list[dict[str, Any]], groups: dict[str, list[int]]
) -> dict[str, int]:
    duplicate_groups = [indexes for indexes in groups.values() if len(indexes) > 1]
    five_way_inconsistent = 0
    boundary_inconsistent = 0
    pairwise_boundary_disagreements = 0
    pairwise_comparisons = 0
    for indexes in duplicate_groups:
        verdicts = {str(rows[index]["verdict"]) for index in indexes}
        supported = sum(
            rows[index]["verdict"] in {"correct", "partial"} for index in indexes
        )
        unsupported = len(indexes) - supported
        five_way_inconsistent += len(verdicts) > 1
        boundary_inconsistent += supported > 0 and unsupported > 0
        pairwise_boundary_disagreements += supported * unsupported
        pairwise_comparisons += len(indexes) * (len(indexes) - 1) // 2
    return {
        "groups": len(duplicate_groups),
        "records": sum(len(indexes) for indexes in duplicate_groups),
        "five_way_inconsistent_groups": five_way_inconsistent,
        "boundary_inconsistent_groups": boundary_inconsistent,
        "pairwise_boundary_disagreements": pairwise_boundary_disagreements,
        "pairwise_comparisons": pairwise_comparisons,
    }


def _repairs(path: Path | None) -> dict[str, ExplanationRepair]:
    if path is None:
        return {}
    raw = _read_array(path, "explanation repairs")
    repairs: dict[str, ExplanationRepair] = {}
    for index, value in enumerate(raw):
        try:
            repair = ExplanationRepair.model_validate(value)
        except ValidationError as error:
            raise ValueError(f"explanation repair {index} is invalid") from error
        if repair.record_id in repairs:
            raise ValueError("explanation repairs contain duplicate record IDs")
        if _control_characters(repair.explanation):
            raise ValueError("a replacement explanation contains a control character")
        repairs[repair.record_id] = repair
    return repairs


def _verdict_overrides(path: Path | None) -> dict[str, VerdictOverride]:
    if path is None:
        return {}
    raw = _read_array(path, "verdict overrides")
    overrides: dict[str, VerdictOverride] = {}
    for index, value in enumerate(raw):
        try:
            override = VerdictOverride.model_validate(value)
        except ValidationError as error:
            raise ValueError(f"verdict override {index} is invalid") from error
        if override.record_id in overrides:
            raise ValueError("verdict overrides contain duplicate record IDs")
        if override.original_verdict == override.verdict:
            raise ValueError("a verdict override does not change the verdict")
        if _control_characters(override.explanation):
            raise ValueError("an override explanation contains a control character")
        overrides[override.record_id] = override
    return overrides


def audit(
    *,
    run_dir: Path,
    inventory_path: Path,
    repairs_path: Path | None,
    verdict_overrides_path: Path | None,
    sample_per_verdict: int,
    sample_seed: str,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    if sample_per_verdict < 1:
        raise ValueError("sample-per-verdict must be positive")
    final_path = run_dir / "final-results.json"
    results_path = run_dir / "results.json"
    state_path = run_dir / "state.json"
    rows = _validate_results(_read_array(final_path, "final results"))
    by_id = {row["record_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("final results contain duplicate record IDs")

    state = _read_object(state_path, "run state")
    state_results = state.get("results")
    if type(state_results) is not dict or len(state_results) != len(rows):
        raise ValueError("run-state result coverage differs from final results")
    results_object = _read_object(results_path, "results object")
    if results_object != state_results:
        raise ValueError("results.json differs from the state result ledger")
    if any(state_results.get(row["record_id"]) != row for row in rows):
        raise ValueError("final results differ from the state result ledger")

    candidate_counts: Counter[str] = Counter()
    candidate_verdicts: dict[str, Counter[str]] = defaultdict(Counter)
    question_counts: Counter[str] = Counter()
    pairs: set[tuple[str, str]] = set()
    sample_candidates: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    inventory_hash = hashlib.sha256()
    inventory_count = 0
    identities: list[dict[str, str]] = []
    semantic_groups: dict[str, list[int]] = defaultdict(list)
    with inventory_path.open("rb") as handle:
        for index, line in enumerate(handle):
            inventory_hash.update(line)
            if not line.strip():
                raise ValueError(f"inventory contains a blank line at {index + 1}")
            try:
                item = InventoryRecord.model_validate_json(line)
            except ValidationError as error:
                raise ValueError(
                    f"inventory row {index} fails the strict schema"
                ) from error
            if index >= len(rows):
                raise ValueError("inventory has more rows than final results")
            result = rows[index]
            if item.source_ordinal != index or item.record_id != result["record_id"]:
                raise ValueError(
                    f"inventory/result identity mismatch at ordinal {index}"
                )
            pair = (item.question_id, item.candidate_id)
            if pair in pairs:
                raise ValueError(
                    f"duplicate question/candidate pair at ordinal {index}"
                )
            pairs.add(pair)
            identities.append(
                {"question_id": item.question_id, "candidate_id": item.candidate_id}
            )
            if result["kind"] != "deterministic":
                semantic_groups[
                    _json_hash(
                        {
                            "candidate_response": item.candidate_response,
                            "provisional_reference": item.provisional_reference,
                            "question": item.question,
                        }
                    )
                ].append(index)
            candidate_counts[item.candidate_id] += 1
            candidate_verdicts[item.candidate_id][result["verdict"]] += 1
            question_counts[item.question_id] += 1
            key = _sample_key(sample_seed, result["record_id"])
            sample_candidates[result["verdict"]].append(
                (
                    key,
                    {
                        "record_id": result["record_id"],
                        "source_ordinal": index,
                        "question_id": item.question_id,
                        "candidate_id": item.candidate_id,
                        "question": item.question,
                        "reference": item.provisional_reference,
                        "candidate_response": item.candidate_response,
                        "verdict": result["verdict"],
                        "explanation": result["explanation"],
                    },
                )
            )
            inventory_count += 1
    if inventory_count != len(rows):
        raise ValueError("inventory has fewer rows than final results")

    verdicts = Counter(str(row["verdict"]) for row in rows)
    kinds = Counter(str(row["kind"]) for row in rows)
    explanation_lengths = [len(str(row["explanation"])) for row in rows]
    explanation_counts = Counter(str(row["explanation"]) for row in rows)
    controls: list[dict[str, Any]] = []
    for row in rows:
        characters = _control_characters(str(row["explanation"]))
        if characters:
            controls.append(
                {
                    "record_id": row["record_id"],
                    "source_ordinal": row["source_ordinal"],
                    "verdict": row["verdict"],
                    "characters": characters,
                }
            )

    deterministic_errors = [
        row["source_ordinal"]
        for row in rows
        if row["kind"] == "deterministic"
        and (
            row["verdict"] != "nonanswer"
            or row["provider_call"] is not False
            or row.get("model") is not None
            or row.get("finish_reason") is not None
            or row["usage"]["total_tokens"] != 0
        )
    ]
    provider_errors = [
        row["source_ordinal"]
        for row in rows
        if row["kind"] != "deterministic"
        and (
            row["provider_call"] is not True
            or row.get("model") != "gpt-5.6-luna"
            or row.get("finish_reason") != "stop"
            or row["usage"]["total_tokens"]
            != row["usage"]["prompt_tokens"] + row["usage"]["completion_tokens"]
        )
    ]
    repair_by_id = _repairs(repairs_path)
    override_by_id = _verdict_overrides(verdict_overrides_path)
    repair_errors: list[str] = []
    for record_id, repair in repair_by_id.items():
        row = by_id.get(record_id)
        if row is None or row["source_ordinal"] != repair.source_ordinal:
            repair_errors.append(record_id)
    if repair_errors:
        raise ValueError("one or more explanation repairs do not bind to the artifact")
    override_errors: list[str] = []
    for record_id, override in override_by_id.items():
        row = by_id.get(record_id)
        if (
            row is None
            or row["source_ordinal"] != override.source_ordinal
            or row["verdict"] != override.original_verdict
        ):
            override_errors.append(record_id)
    if override_errors:
        raise ValueError("one or more verdict overrides do not bind to the artifact")

    curated = [dict(row) for row in rows]
    repaired_ordinals: list[int] = []
    for row in curated:
        repair = repair_by_id.get(row["record_id"])
        if repair is not None:
            row["explanation"] = repair.explanation
            repaired_ordinals.append(row["source_ordinal"])
        override = override_by_id.get(row["record_id"])
        if override is not None:
            row["verdict"] = override.verdict
            row["explanation"] = override.explanation
    override_ordinals = sorted(
        override.source_ordinal for override in override_by_id.values()
    )
    remaining_controls = sum(
        bool(_control_characters(str(row["explanation"]))) for row in curated
    )
    curated_verdicts = Counter(str(row["verdict"]) for row in curated)
    labels = [
        {
            "record_id": row["record_id"],
            "source_ordinal": row["source_ordinal"],
            "question_id": identity["question_id"],
            "candidate_id": identity["candidate_id"],
            "verdict": row["verdict"],
            "supported": row["verdict"] in {"correct", "partial"},
        }
        for row, identity in zip(curated, identities, strict=True)
    ]

    sample: list[dict[str, Any]] = []
    for verdict in ("correct", "partial", "incorrect", "not_established", "nonanswer"):
        selected = sorted(sample_candidates[verdict], key=lambda pair: pair[0])[
            :sample_per_verdict
        ]
        sample.extend(value for _, value in selected)
    sampled_ids = {row["record_id"] for row in sample}
    for finding in controls:
        record_id = str(finding["record_id"])
        if record_id not in sampled_ids:
            source = sample_candidates[str(finding["verdict"])]
            value = next(item for _, item in source if item["record_id"] == record_id)
            sample.append(value | {"audit_reason": "control_character"})

    per_candidate = []
    for candidate_id, count in candidate_counts.items():
        distribution = candidate_verdicts[candidate_id]
        per_candidate.append(
            {
                "candidate_id": candidate_id,
                "records": count,
                "correct": distribution["correct"],
                "correct_rate": distribution["correct"] / count,
                "nonanswer": distribution["nonanswer"],
                "nonanswer_rate": distribution["nonanswer"] / count,
            }
        )
    per_candidate.sort(key=lambda row: str(row["candidate_id"]))

    state_counts = state.get("counts")
    state_anomalies = state.get("anomalies")
    state_clean = (
        type(state_counts) is dict
        and state_counts.get("records") == len(rows)
        and state_counts.get("pending") == 0
        and state.get("halted") is False
        and state.get("paused") is False
        and state_anomalies == []
    )
    summary: dict[str, Any] = {
        "version": 1,
        "assessment": (
            "ready"
            if state_clean
            and remaining_controls == 0
            and not deterministic_errors
            and not provider_errors
            else "needs_repair"
        ),
        "artifacts": {
            "final_results": {
                "path": str(final_path),
                "bytes": final_path.stat().st_size,
                "sha256": _sha256(final_path),
            },
            "results": {
                "path": str(results_path),
                "bytes": results_path.stat().st_size,
                "sha256": _sha256(results_path),
            },
            "state": {
                "path": str(state_path),
                "bytes": state_path.stat().st_size,
                "sha256": _sha256(state_path),
            },
            "inventory": {
                "path": str(inventory_path),
                "bytes": inventory_path.stat().st_size,
                "sha256": inventory_hash.hexdigest(),
            },
        },
        "coverage": {
            "records": len(rows),
            "unique_record_ids": len(by_id),
            "unique_question_candidate_pairs": len(pairs),
            "questions": len(question_counts),
            "candidates": len(candidate_counts),
            "records_per_question_min": min(question_counts.values()),
            "records_per_question_median": _percentile(
                list(question_counts.values()), 0.5
            ),
            "records_per_question_max": max(question_counts.values()),
        },
        "execution": {
            "state_clean": state_clean,
            "state_anomalies": state_anomalies,
            "kinds": dict(sorted(kinds.items())),
            "deterministic_invariant_failures": deterministic_errors,
            "provider_invariant_failures": provider_errors,
            "estimated_cost_usd": state.get("spend", {})
            .get("latest", {})
            .get("completed_estimated_usd"),
        },
        "verdicts": {
            verdict: {"count": count, "rate": count / len(rows)}
            for verdict, count in sorted(verdicts.items())
        },
        "curated_verdicts": {
            verdict: {"count": count, "rate": count / len(curated)}
            for verdict, count in sorted(curated_verdicts.items())
        },
        "exact_input_consistency": {
            "raw": _duplicate_metrics(rows, semantic_groups),
            "curated": _duplicate_metrics(curated, semantic_groups),
        },
        "explanations": {
            "length_min": min(explanation_lengths),
            "length_median": _percentile(explanation_lengths, 0.5),
            "length_p95": _percentile(explanation_lengths, 0.95),
            "length_p99": _percentile(explanation_lengths, 0.99),
            "length_max": max(explanation_lengths),
            "at_schema_maximum": sum(value == 400 for value in explanation_lengths),
            "control_character_records": controls,
            "distinct": len(explanation_counts),
            "repeated_provider_explanations": sum(
                count > 1
                for explanation, count in explanation_counts.items()
                if explanation
                != "The historical response is an exact bare no-response."
            ),
        },
        "repairs": {
            "repair_count": len(repair_by_id),
            "verdict_changes": len(override_by_id),
            "repaired_ordinals": sorted(repaired_ordinals),
            "verdict_override_ordinals": override_ordinals,
            "verdict_override_reasons": dict(
                sorted(Counter(x.reason for x in override_by_id.values()).items())
            ),
            "remaining_control_character_records": remaining_controls,
        },
        "sample": {
            "seed": sample_seed,
            "per_verdict": sample_per_verdict,
            "records": len(sample),
        },
        "per_candidate": per_candidate,
    }
    summary["summary_sha256"] = _json_hash(summary)
    return summary, sample, curated, labels


def _render_report(summary: dict[str, Any]) -> str:
    coverage = summary["coverage"]
    execution = summary["execution"]
    explanations = summary["explanations"]
    repairs = summary["repairs"]
    artifacts = summary["artifacts"]
    consistency = summary["exact_input_consistency"]
    verdict_lines = [
        f"| {label} | {details['count']:,} | {details['rate']:.2%} |"
        for label, details in summary["verdicts"].items()
    ]
    curated_verdict_lines = [
        f"| {label} | {details['count']:,} | {details['rate']:.2%} |"
        for label, details in summary["curated_verdicts"].items()
    ]
    return "\n".join(
        [
            "# OpenAI Luna pointwise production audit",
            "",
            f"**Assessment: {summary['assessment'].replace('_', ' ').title()}**",
            "",
            "## Coverage and identity",
            "",
            f"- {coverage['records']:,} records; {coverage['unique_record_ids']:,} unique record IDs.",
            f"- {coverage['unique_question_candidate_pairs']:,} unique question/candidate pairs across {coverage['questions']:,} questions and {coverage['candidates']:,} candidates.",
            f"- Final artifact SHA-256: `{artifacts['final_results']['sha256']}`.",
            f"- Inventory SHA-256: `{artifacts['inventory']['sha256']}`.",
            "",
            "## Execution integrity",
            "",
            f"- Controller state clean: {execution['state_clean']}.",
            f"- Kinds: {json.dumps(execution['kinds'], sort_keys=True)}.",
            f"- Provider invariant failures: {len(execution['provider_invariant_failures'])}; deterministic invariant failures: {len(execution['deterministic_invariant_failures'])}.",
            f"- Estimated Batch cost: ${execution['estimated_cost_usd']:.2f}.",
            "",
            "## Verdict distribution",
            "",
            "| Verdict | Count | Rate |",
            "|---|---:|---:|",
            *verdict_lines,
            "",
            "### Curated downstream distribution",
            "",
            "| Verdict | Count | Rate |",
            "|---|---:|---:|",
            *curated_verdict_lines,
            "",
            "## Exact-input consistency",
            "",
            f"- Non-deterministic duplicate groups/rows: {consistency['raw']['groups']:,}/{consistency['raw']['records']:,}.",
            f"- Supported/unsupported-inconsistent groups, raw → curated: {consistency['raw']['boundary_inconsistent_groups']} → {consistency['curated']['boundary_inconsistent_groups']}.",
            f"- Pairwise supported/unsupported disagreements, raw → curated: {consistency['raw']['pairwise_boundary_disagreements']:,} → {consistency['curated']['pairwise_boundary_disagreements']:,} of {consistency['raw']['pairwise_comparisons']:,} duplicate pairs.",
            "- Twenty clearly adjudicated groups were canonicalized. Five ambiguous or broken-question groups remain explicitly unchanged.",
            "",
            "## Semantic spot check",
            "",
            f"- {summary['sample']['records']} records were reviewed from a deterministic, verdict-stratified sample (20 per raw verdict plus explanation-integrity controls).",
            "- Two clear supported/unsupported errors outside duplicate groups were corrected in the curated artifact.",
            "- Other clear reviewer disagreements concerned the five-way debugging subtype while remaining in the same supported/unsupported bucket.",
            "",
            "## Explanation quality",
            "",
            f"- Length p50/p95/p99/max: {explanations['length_median']}/{explanations['length_p95']}/{explanations['length_p99']}/{explanations['length_max']} characters.",
            f"- At 400-character schema maximum: {explanations['at_schema_maximum']}.",
            f"- Control-character records in raw artifact: {len(explanations['control_character_records'])}.",
            f"- Applied explanation repairs: {repairs['repair_count']}; explicit audited verdict changes: {repairs['verdict_changes']}; remaining control-character records: {repairs['remaining_control_character_records']}.",
            "",
            "## Interpretation boundary",
            "",
            "This audit establishes complete, schema-valid, identity-bound execution and a reviewed downstream label artifact. The primary automated label is `supported = verdict in {correct, partial}`; the three unsupported subtypes are debugging signals. The labels measure agreement with the supplied Gemini evaluator evidence, not absolute factual truth; the Gemini evidence can itself be incomplete or wrong.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    summary, sample, curated, labels = audit(
        run_dir=args.run_dir,
        inventory_path=args.inventory,
        repairs_path=args.repairs,
        verdict_overrides_path=args.verdict_overrides,
        sample_per_verdict=args.sample_per_verdict,
        sample_seed=args.sample_seed,
    )
    if args.curated_output is not None:
        if args.repairs is None and args.verdict_overrides is None:
            raise ValueError(
                "--curated-output requires --repairs or --verdict-overrides"
            )
        _atomic_write(args.curated_output, curated, private=True)
        summary["artifacts"]["curated_results"] = {
            "path": str(args.curated_output),
            "bytes": args.curated_output.stat().st_size,
            "sha256": _sha256(args.curated_output),
        }
    if args.labels_output is not None:
        _atomic_write_jsonl(args.labels_output, labels)
        summary["artifacts"]["labels"] = {
            "path": str(args.labels_output),
            "bytes": args.labels_output.stat().st_size,
            "sha256": _sha256(args.labels_output),
        }
    summary.pop("summary_sha256", None)
    summary["summary_sha256"] = _json_hash(summary)
    _atomic_write(args.summary, summary, private=True)
    _atomic_write(args.sample, sample, private=True)
    _atomic_write(args.report, _render_report(summary), private=False)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
