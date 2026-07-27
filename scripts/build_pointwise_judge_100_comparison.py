#!/usr/bin/env python3
"""Compare two private 100-case pointwise runs without claiming correctness."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import stat
import tempfile

from scripts.run_local_pointwise_judge import (
    _load_cases,
    _load_json_from_opened_file,
    _object_without_duplicate_keys,
)
from trec_mlm.pointwise_judge import CandidateCase
from trec_mlm.pointwise_runner import (
    FROZEN_DECISION_SCHEMA_SHA256,
    _validate_run_container,
    _validated_same_identity_record,
)


_LABELS = ("incorrect", "partial", "good")
_INPUT_HASH_KEYS = {
    "manifest",
    "sealed_candidates",
    "direct_run",
    "thinking_run",
}
_IDENTITY_KEYS = {
    "input_sha256",
    "prompt_sha256",
    "schema_sha256",
    "model_alias",
    "model_id",
    "runtime_receipt_sha256",
}
_MAX_PRIVATE_OUTPUT_BYTES = 16 * 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--sealed-candidates", type=Path, required=True)
    parser.add_argument("--sealed-candidates-sha256", required=True)
    parser.add_argument("--direct-run", type=Path, required=True)
    parser.add_argument("--direct-run-sha256", required=True)
    parser.add_argument("--thinking-run", type=Path, required=True)
    parser.add_argument("--thinking-run-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def build_comparison(
    cases: Sequence[CandidateCase],
    direct_run: Mapping[str, object],
    thinking_run: Mapping[str, object],
    *,
    reused_response_ids: Sequence[str],
    input_hashes: Mapping[str, str],
    sealed_candidate_ids: Collection[str],
    alias_secret: bytes,
    seed: int = 2026,
) -> dict[str, object]:
    """Build private comparison data and blind review packets."""
    validated_cases = _validate_cases(cases)
    hashes = _validate_input_hashes(input_hashes)
    if type(seed) is not int:
        raise ValueError("review seed must be an integer")
    candidate_ids = _validate_source_ids(
        sealed_candidate_ids, "sealed candidate identities"
    )
    if type(alias_secret) is not bytes or len(alias_secret) != 32:
        raise ValueError("mode-alias secret must contain exactly 32 bytes")
    reused = _validate_reused_ids(reused_response_ids, validated_cases)
    validated_runs = {
        "direct": _validate_run("direct", direct_run, validated_cases),
        "thinking": _validate_run("thinking", thinking_run, validated_cases),
    }
    direct = validated_runs["direct"]
    thinking = validated_runs["thinking"]
    for binding in ("prompt_sha256", "schema_sha256"):
        if direct["container"][binding] != thinking["container"][binding]:
            raise ValueError(f"runs do not share one {binding.removesuffix('_sha256')}")

    outcomes = {role: value["outcomes"] for role, value in validated_runs.items()}
    all_metrics = _slice_metrics(validated_cases, outcomes)
    per_reason = {
        reason: _slice_metrics(
            [case for case in validated_cases if case.selection_reason == reason],
            outcomes,
        )
        for reason in dict.fromkeys(case.selection_reason for case in validated_cases)
    }
    strata = sorted({stratum for case in validated_cases for stratum in case.strata})
    per_stratum = {
        stratum: _slice_metrics(
            [case for case in validated_cases if stratum in case.strata], outcomes
        )
        for stratum in strata
    }
    reused_slice = _slice_metrics(
        [case for case in validated_cases if case.response_id in reused], outcomes
    )
    matrix = all_metrics.pop("confusion_matrix")
    transition_counts = all_metrics.pop("transitions")
    coverage = {
        key: all_metrics.pop(key)
        for key in (
            "case_count",
            "direct_failure_count",
            "thinking_failure_count",
        )
    }
    case_differences = _case_differences(validated_cases, outcomes)
    headline = (
        f"{all_metrics['exact_count']}/{all_metrics['paired_valid_decisions']} "
        "paired cases have exact label agreement; "
        f"{all_metrics['within_one_label_count']}/"
        f"{all_metrics['paired_valid_decisions']} are within one label step."
    )
    independent, critique, sealed = _review_packets(
        validated_cases,
        outcomes,
        validated_runs,
        input_hashes=hashes,
        sealed_candidate_ids=candidate_ids,
        alias_secret=alias_secret,
    )
    comparison: dict[str, object] = {
        "version": 1,
        "comparison_kind": "mode_agreement_without_reviewer_correctness",
        "headline": headline,
        "inputs": {
            "sha256": hashes,
            "prompt_sha256": direct["container"]["prompt_sha256"],
            "schema_sha256": direct["container"]["schema_sha256"],
        },
        "labels": list(_LABELS),
        "coverage": coverage,
        "agreement": all_metrics,
        "confusion_matrix": {
            "rows": "direct",
            "columns": "thinking",
            "labels": list(_LABELS),
            "counts": matrix,
        },
        "transitions": [
            {
                "direct_label": direct_label,
                "thinking_label": thinking_label,
                "count": transition_counts[f"{direct_label}_to_{thinking_label}"],
            }
            for direct_label in _LABELS
            for thinking_label in _LABELS
            if transition_counts.get(f"{direct_label}_to_{thinking_label}", 0)
        ],
        "per_reason": per_reason,
        "per_stratum": per_stratum,
        "reused_slice": reused_slice,
        "runs": {role: value["metrics"] for role, value in validated_runs.items()},
        "case_differences": case_differences,
        "interpretation": (
            "These are behavioral transitions between inference modes. "
            "Without independent reviewer evidence, neither mode is labeled correct."
        ),
    }
    return {
        "comparison": comparison,
        "independent_review_packet": independent,
        "critique_review_packet": critique,
        "sealed_mode_aliases": sealed,
    }


def _validate_cases(cases: Sequence[CandidateCase]) -> list[CandidateCase]:
    if type(cases) not in {list, tuple} or not cases:
        raise ValueError("cases must be a nonempty sequence")
    if any(type(case) is not CandidateCase for case in cases):
        raise ValueError("cases must contain CandidateCase values")
    response_ids = [case.response_id for case in cases]
    if len(response_ids) != len(set(response_ids)):
        raise ValueError("case response coverage must be unique")
    return list(cases)


def _validate_input_hashes(value: Mapping[str, str]) -> dict[str, str]:
    if type(value) is not dict or set(value) != _INPUT_HASH_KEYS:
        raise ValueError("input hashes must contain exact comparison inputs")
    result = dict(value)
    if any(not _is_sha256(digest) for digest in result.values()):
        raise ValueError("comparison input SHA-256 is invalid")
    return result


def _validate_source_ids(value: Collection[str], description: str) -> set[str]:
    if type(value) not in {list, tuple, set} or any(
        type(item) is not str or not item for item in value
    ):
        raise ValueError(f"{description} must be nonempty strings")
    result = set(value)
    if len(result) != len(value):
        raise ValueError(f"{description} must be unique")
    return result


def _validate_reused_ids(
    value: Sequence[str], cases: Sequence[CandidateCase]
) -> set[str]:
    if type(value) is not list or any(type(item) is not str for item in value):
        raise ValueError("reused response IDs must be a list of strings")
    result = set(value)
    if len(result) != len(value):
        raise ValueError("reused response IDs must be unique")
    if not result <= {case.response_id for case in cases}:
        raise ValueError("reused response ID is outside case coverage")
    return result


def _validate_run(
    role: str, raw_run: Mapping[str, object], cases: Sequence[CandidateCase]
) -> dict[str, object]:
    if type(raw_run) is not dict:
        raise ValueError(f"{role} run must be an object")
    run = dict(raw_run)
    try:
        _validate_run_container(run)
    except ValueError as error:
        raise ValueError(f"invalid {role} run: {error}") from error
    if run["schema_sha256"] != FROZEN_DECISION_SCHEMA_SHA256:
        raise ValueError(f"{role} run schema is not frozen")
    expected_ids = {case.response_id for case in cases}
    if set(run["records"]) != expected_ids:
        raise ValueError(f"{role} run coverage does not match cases")

    outcomes: dict[str, dict[str, object]] = {}
    elapsed_total = 0.0
    usage: Counter[str] = Counter()
    failure_types: Counter[str] = Counter()
    repaired_records = 0
    initial_invalid_attempts = 0
    for case in cases:
        record = run["records"][case.response_id]
        if type(record) is not dict:
            raise ValueError(f"{role} run record must be an object")
        expected_identity = {
            "input_sha256": _json_hash(asdict(case)),
            **{key: run[key] for key in _IDENTITY_KEYS - {"input_sha256"}},
        }
        if record.get("input_sha256") != expected_identity["input_sha256"]:
            raise ValueError(f"{role} run record input binding is invalid")
        for key, expected in expected_identity.items():
            if record.get(key) != expected:
                raise ValueError(f"{role} run record {key} binding is invalid")
        attempts = record.get("attempts")
        if type(attempts) is not list or not attempts or type(attempts[0]) is not dict:
            raise ValueError(f"{role} run record attempt structure is invalid")
        backend_model = attempts[0].get("backend_model_id")
        if type(backend_model) is not str or not backend_model:
            raise ValueError(f"{role} run backend identity is invalid")
        try:
            validated = _validated_same_identity_record(
                record,
                expected_identity,
                backend_response_model=backend_model,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid {role} run record: {error}") from error
        if not validated.terminal:
            raise ValueError(f"{role} run record is not terminal")
        elapsed_total += validated.elapsed_seconds
        usage.update(validated.usage)
        if len(validated.attempts) == 2:
            initial_invalid_attempts += 1
        decision = record["decision"]
        if decision is None:
            failure = record["failure"]
            if type(failure) is not dict or type(failure.get("type")) is not str:
                raise ValueError(f"{role} failed record has invalid failure")
            failure_types[failure["type"]] += 1
            outcomes[case.response_id] = {"status": "failure", "failure": dict(failure)}
        else:
            label = decision.get("label")
            if label not in _LABELS:
                raise ValueError(f"{role} run decision label is invalid")
            if len(validated.attempts) == 2:
                repaired_records += 1
            outcomes[case.response_id] = {
                "status": "decision",
                "label": label,
                "decision": dict(decision),
            }
    valid_count = sum(value["status"] == "decision" for value in outcomes.values())
    failed_count = len(cases) - valid_count
    metrics = {
        "model_alias": run["model_alias"],
        "model_id": run["model_id"],
        "runtime_receipt_sha256": run["runtime_receipt_sha256"],
        "record_count": len(cases),
        "valid_decisions": valid_count,
        "failed_records": failed_count,
        "schema_success_count": valid_count,
        "schema_success_rate": valid_count / len(cases),
        "failure_types": dict(sorted(failure_types.items())),
        "repaired_records": repaired_records,
        "initial_invalid_attempts": initial_invalid_attempts,
        "elapsed_seconds_total": elapsed_total,
        "usage": dict(sorted(usage.items())),
    }
    return {"container": run, "outcomes": outcomes, "metrics": metrics}


def _slice_metrics(
    cases: Sequence[CandidateCase],
    outcomes: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> dict[str, object]:
    matrix = [[0 for _ in _LABELS] for _ in _LABELS]
    transitions: Counter[str] = Counter()
    exact = 0
    within_one = 0
    paired = 0
    direct_failures = 0
    thinking_failures = 0
    for case in cases:
        direct = outcomes["direct"][case.response_id]
        thinking = outcomes["thinking"][case.response_id]
        direct_failures += direct["status"] == "failure"
        thinking_failures += thinking["status"] == "failure"
        if direct["status"] != "decision" or thinking["status"] != "decision":
            continue
        paired += 1
        direct_index = _LABELS.index(direct["label"])
        thinking_index = _LABELS.index(thinking["label"])
        matrix[direct_index][thinking_index] += 1
        transitions[f"{direct['label']}_to_{thinking['label']}"] += 1
        exact += direct_index == thinking_index
        within_one += abs(direct_index - thinking_index) <= 1
    return {
        "case_count": len(cases),
        "paired_valid_decisions": paired,
        "exact_count": exact,
        "exact_rate": exact / paired if paired else None,
        "within_one_label_count": within_one,
        "within_one_label_rate": within_one / paired if paired else None,
        "nonadjacent_count": paired - within_one,
        "direct_failure_count": direct_failures,
        "thinking_failure_count": thinking_failures,
        "transitions": dict(sorted(transitions.items())),
        "confusion_matrix": matrix,
    }


def _case_differences(
    cases: Sequence[CandidateCase],
    outcomes: Mapping[str, Mapping[str, Mapping[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for case in cases:
        direct = outcomes["direct"][case.response_id]
        thinking = outcomes["thinking"][case.response_id]
        if (
            direct["status"] == thinking["status"] == "decision"
            and direct["label"] == thinking["label"]
        ):
            continue
        rows.append(
            {
                **asdict(case),
                "strata": list(case.strata),
                "direct_status": direct["status"],
                "thinking_status": thinking["status"],
                "direct_outcome": dict(direct),
                "thinking_outcome": dict(thinking),
            }
        )
    return rows


def _review_packets(
    cases: Sequence[CandidateCase],
    outcomes: Mapping[str, Mapping[str, Mapping[str, object]]],
    validated_runs: Mapping[str, Mapping[str, object]],
    *,
    input_hashes: Mapping[str, str],
    sealed_candidate_ids: Collection[str],
    alias_secret: bytes,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    alias_to_mode = _mode_alias_mapping(alias_secret)
    mode_to_alias = {mode: alias for alias, mode in alias_to_mode.items()}
    ordered_modes = [alias_to_mode[alias] for alias in sorted(alias_to_mode)]
    independent: list[dict[str, object]] = []
    critique: list[dict[str, object]] = []
    for index, case in enumerate(cases, 1):
        case_fields = {
            "review_case_id": f"case_{index:03d}",
            "question": case.question,
            "provisional_reference": case.reference_answer,
            "candidate_response": case.candidate_answer,
        }
        independent.append(case_fields)
        judgments = {
            mode_to_alias[mode]: _review_outcome(outcomes[mode][case.response_id])
            for mode in ordered_modes
        }
        critique.append({**case_fields, "judgments": judgments})

    identity_tokens = _packet_forbidden_identities(
        cases,
        validated_runs,
        input_hashes=input_hashes,
        sealed_candidate_ids=sealed_candidate_ids,
    )
    packet_text = json.dumps(
        {"independent": independent, "critique": critique},
        ensure_ascii=False,
        sort_keys=True,
    )
    if any(token and token in packet_text for token in identity_tokens):
        raise ValueError("review packet contains source identity")
    sealed = {
        "version": 2,
        "alias_secret_hex": alias_secret.hex(),
        "judge_aliases": alias_to_mode,
    }
    return independent, critique, sealed


def _mode_alias_mapping(alias_secret: bytes) -> dict[str, str]:
    if type(alias_secret) is not bytes or len(alias_secret) != 32:
        raise ValueError("mode-alias secret must contain exactly 32 bytes")
    ordered_modes = sorted(
        ("direct", "thinking"),
        key=lambda mode: hmac.digest(
            alias_secret, f"pointwise-mode-alias-v1:{mode}".encode(), "sha256"
        ),
    )
    return {
        f"judge_{chr(ord('a') + index)}": mode
        for index, mode in enumerate(ordered_modes)
    }


def _packet_forbidden_identities(
    cases: Sequence[CandidateCase],
    validated_runs: Mapping[str, Mapping[str, object]],
    *,
    input_hashes: Mapping[str, str],
    sealed_candidate_ids: Collection[str],
) -> set[str]:
    identities = {
        *(case.question_id for case in cases),
        *(case.response_id for case in cases),
        *sealed_candidate_ids,
        *input_hashes.values(),
    }
    for mode in ("direct", "thinking"):
        run = validated_runs[mode]["container"]
        identities.update(
            str(run[field])
            for field in (
                "model_alias",
                "model_id",
                "prompt_sha256",
                "schema_sha256",
                "runtime_receipt_sha256",
            )
        )
        for record in run["records"].values():
            identities.update(str(record[field]) for field in _IDENTITY_KEYS)
            for attempt in record["attempts"]:
                for field in ("logical_model_id", "backend_model_id"):
                    value = attempt.get(field)
                    if type(value) is str and value:
                        identities.add(value)
                body = attempt.get("response_body")
                if type(body) is dict:
                    for field in ("id", "model", "system_fingerprint"):
                        value = body.get(field)
                        if type(value) is str and value:
                            identities.add(value)
    return {identity for identity in identities if identity}


def _review_outcome(value: Mapping[str, object]) -> dict[str, object]:
    if value["status"] == "failure":
        return {"status": "failure"}
    return {"status": "decision", "decision": dict(value["decision"])}


def _render_markdown(comparison: Mapping[str, object]) -> str:
    agreement = comparison["agreement"]
    lines = [
        "# Pointwise judge inference-mode comparison",
        "",
        f"**Headline:** {comparison['headline']}",
        "",
        (
            "This report describes agreement and transitions between inference modes. "
            "It does not identify either mode as correct without reviewer evidence."
        ),
        "",
        "## Agreement",
        "",
        f"- Paired valid decisions: {agreement['paired_valid_decisions']}",
        f"- Exact agreement: {agreement['exact_count']} ({_format_rate(agreement['exact_rate'])})",
        (
            "- Within-one-label agreement: "
            f"{agreement['within_one_label_count']} "
            f"({_format_rate(agreement['within_one_label_rate'])})"
        ),
        f"- Nonadjacent changes: {agreement['nonadjacent_count']}",
        "",
        "## Directional transitions",
        "",
        "| Direct label | Thinking label | Count |",
        "| --- | --- | ---: |",
    ]
    for row in comparison["transitions"]:
        lines.append(
            f"| {row['direct_label']} | {row['thinking_label']} | {row['count']} |"
        )
    lines.extend(["", "## Run accounting", ""])
    for role in ("direct", "thinking"):
        run = comparison["runs"][role]
        lines.append(
            f"- {role.capitalize()}: {run['valid_decisions']}/{run['record_count']} "
            f"valid decisions, {run['failed_records']} failures, "
            f"{run['elapsed_seconds_total']:.3f} summed seconds, "
            f"{run['usage'].get('total_tokens', 0)} total tokens."
        )
    lines.extend(
        [
            "",
            "## Private case-level evidence",
            "",
            f"The JSON companion retains {len(comparison['case_differences'])} "
            "case-level differences and the complete private input bindings.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_rate(value: object) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.1%}"


def _encoded_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _prepare_private_directory(path: Path) -> None:
    if path.exists():
        mode = path.lstat().st_mode
        if not stat.S_ISDIR(mode) or stat.S_IMODE(mode) != 0o700:
            raise ValueError("comparison output directory must have mode 0700")
        return
    path.mkdir(parents=True, mode=0o700)
    path.chmod(0o700)


def _existing_private_bytes(path: Path, description: str) -> bytes | None:
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{description} could not be opened safely") from error
    try:
        os.set_inheritable(descriptor, False)
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_IMODE(before.st_mode) != 0o600
            or before.st_nlink != 1
            or before.st_size > _MAX_PRIVATE_OUTPUT_BYTES
        ):
            raise ValueError(
                f"{description} must be a private single-link regular file"
            )
        chunks: list[bytes] = []
        remaining = before.st_size
        while remaining:
            chunk = os.read(descriptor, min(65536, remaining))
            if not chunk:
                raise ValueError(f"{description} changed during read")
            chunks.append(chunk)
            remaining -= len(chunk)
        after = os.fstat(descriptor)
        path_after = os.stat(path, follow_symlinks=False)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        )
        path_identity = (
            path_after.st_dev,
            path_after.st_ino,
            path_after.st_size,
            path_after.st_mtime_ns,
            path_after.st_ctime_ns,
        )
        if (
            not stat.S_ISREG(path_after.st_mode)
            or path_after.st_nlink != 1
            or identity_before != identity_after
            or identity_after != path_identity
        ):
            raise ValueError(f"{description} changed during read")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _require_compatible_private_output(path: Path, content: bytes) -> None:
    existing = _existing_private_bytes(path, "comparison output")
    if existing is not None and existing != content:
        raise ValueError(
            "existing private output differs; preserve it as superseded before regeneration"
        )


def _write_private(path: Path, content: bytes) -> None:
    existing = _existing_private_bytes(path, "comparison output")
    if existing is not None:
        if existing != content:
            raise ValueError("existing private output differs; refusing overwrite")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError:
            _require_compatible_private_output(path, content)
        else:
            temporary.unlink()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _load_alias_secret(path: Path) -> bytes | None:
    raw = _existing_private_bytes(path, "private mode seal")
    if raw is None:
        return None
    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_object_without_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("private mode seal is not valid JSON") from error
    if type(value) is not dict or set(value) != {
        "version",
        "alias_secret_hex",
        "judge_aliases",
    }:
        raise ValueError("private mode seal must contain exact version-2 keys")
    secret_hex = value["alias_secret_hex"]
    if (
        value["version"] != 2
        or type(secret_hex) is not str
        or len(secret_hex) != 64
        or any(character not in "0123456789abcdef" for character in secret_hex)
    ):
        raise ValueError("private mode seal secret is invalid")
    secret = bytes.fromhex(secret_hex)
    if value["judge_aliases"] != _mode_alias_mapping(secret):
        raise ValueError("private mode seal mapping does not match its secret")
    return secret


def _validate_sealed_candidates(
    value: object, cases: Sequence[CandidateCase]
) -> set[str]:
    if type(value) is not dict or set(value) != {case.response_id for case in cases}:
        raise ValueError("sealed candidate coverage does not match cases")
    if any(
        type(candidate_id) is not str or not candidate_id
        for candidate_id in value.values()
    ):
        raise ValueError("sealed candidate identities must be nonempty strings")
    return set(value.values())


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = _load_json_from_opened_file(
        args.manifest,
        "100-case manifest",
        expected_sha256=args.manifest_sha256,
        private=True,
    )
    cases = _load_cases(manifest)
    sealed_candidates = _load_json_from_opened_file(
        args.sealed_candidates,
        "sealed candidates",
        expected_sha256=args.sealed_candidates_sha256,
        private=True,
    )
    candidate_ids = _validate_sealed_candidates(sealed_candidates, cases)
    direct_run = _load_json_from_opened_file(
        args.direct_run,
        "direct run",
        expected_sha256=args.direct_run_sha256,
        private=True,
    )
    thinking_run = _load_json_from_opened_file(
        args.thinking_run,
        "thinking run",
        expected_sha256=args.thinking_run_sha256,
        private=True,
    )

    output_dir: Path = args.output_dir
    _prepare_private_directory(output_dir)
    review_dir = output_dir / "review"
    _prepare_private_directory(review_dir)
    output_paths = {
        "comparison.json": output_dir / "comparison.json",
        "comparison.md": output_dir / "comparison.md",
        "independent-review-packet.json": review_dir / "independent-review-packet.json",
        "critique-review-packet.json": review_dir / "critique-review-packet.json",
        "sealed-mode-aliases.json": review_dir / "sealed-mode-aliases.json",
    }
    for path in output_paths.values():
        _existing_private_bytes(path, "comparison output")
    alias_secret = _load_alias_secret(output_paths["sealed-mode-aliases.json"])
    if alias_secret is None:
        alias_secret = secrets.token_bytes(32)

    artifacts = build_comparison(
        cases,
        direct_run,
        thinking_run,
        reused_response_ids=manifest["reused_smoke_response_ids"],
        input_hashes={
            "manifest": args.manifest_sha256,
            "sealed_candidates": args.sealed_candidates_sha256,
            "direct_run": args.direct_run_sha256,
            "thinking_run": args.thinking_run_sha256,
        },
        sealed_candidate_ids=candidate_ids,
        alias_secret=alias_secret,
        seed=args.seed,
    )
    review_outputs = {
        "independent-review-packet.json": artifacts["independent_review_packet"],
        "critique-review-packet.json": artifacts["critique_review_packet"],
        "sealed-mode-aliases.json": artifacts["sealed_mode_aliases"],
    }
    encoded_review = {
        name: _encoded_json(payload) for name, payload in review_outputs.items()
    }
    comparison = dict(artifacts["comparison"])
    comparison["review_artifacts"] = {
        name: {"sha256": hashlib.sha256(content).hexdigest()}
        for name, content in encoded_review.items()
    }
    comparison_json = _encoded_json(comparison)
    comparison_markdown = _render_markdown(comparison).encode()
    encoded_outputs = {
        **encoded_review,
        "comparison.json": comparison_json,
        "comparison.md": comparison_markdown,
    }
    for name, content in encoded_outputs.items():
        _require_compatible_private_output(output_paths[name], content)
    for name, content in encoded_outputs.items():
        _write_private(output_paths[name], content)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
