#!/usr/bin/env python3
"""Build the frozen, private v11 semantic-adjudication manifest."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED_RESULT = Path(
    "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-500/result.json"
)
DEFAULT_ORIGINAL_RESULT = Path(
    "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-100/result.json"
)
DEFAULT_RUN_PLAN = Path(
    "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-500/run-plan.json"
)
DEFAULT_OVERRIDES = Path(
    "protocols/pointwise_judge_bakeoff/v11-adjudication-overrides.json"
)
DEFAULT_OUTPUT = Path(
    "artifacts/pointwise_judge_bakeoff/v11/adjudication/manifest.json"
)
AUTHORIZED_OUTPUT_ROOT = Path("artifacts/pointwise_judge_bakeoff/v11")
DEFAULT_AUDIT_REPORTS = (
    Path("artifacts/pointwise_production/deepinfra-v10-thinking-pilot-100/audit.md"),
    Path(
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-a.md"
    ),
    Path(
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-b.md"
    ),
    Path(
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-a.md"
    ),
    Path(
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-b.md"
    ),
    Path(
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-c.md"
    ),
)
PAIR_TO_LABEL = {
    "2/supported": "correct",
    "1/mixed": "partial",
    "1/reference_silent": "not_established",
    "0/contradicted": "incorrect",
    "0/nonanswer": "nonanswer",
}
EXPECTED_TOTALS = {0: 42, 1: 66, 2: 174}
EXPECTED_BARE_IDKS = [55, 57, 58, 60]
EXPECTED_LEDGER_SHA256 = (
    "5da969289173b7406a2168fda9358f9dfb623a1b188a37d786560f32dc22ceaa"
)
APPROVED_SOURCE_SHA256 = {
    "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-500/result.json": (
        "6a5eb0f5011fc0fa82fc5e13948046af53b88fd9853eff1d6faea2942c97c3b1"
    ),
    "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-100/result.json": (
        "517c5b05a771c706a8885467617aa21f0c7f52933574aedf659bda4a275c4d77"
    ),
    "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-500/run-plan.json": (
        "d0075f05c436b5bc9bb77fae045957554aadcf25f4189567b5b043f5b90a057e"
    ),
    "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-100/audit.md": (
        "ffd4a91a479ff06eecefc3d1e1c7cfb7996b4805e6f1a42f7867bb7f9d3f7316"
    ),
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-a.md": (
        "f1b8fe164311a16654166d9d5e36656b51f115097e392efd12d13a821420770a"
    ),
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-b.md": (
        "de1d0ab3e71e764ae00ca7b116437d786387a34c6aef02a58fff9e6d62a8a985"
    ),
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-a.md": (
        "569ed08676f703fd527129388e90bda74f8cc1a86265cd4ac91c98b5b98e97dd"
    ),
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-b.md": (
        "3e1c0f4bb03f824aab76ca8d51d9f361552cc741f40bd1b958922ee1cdbdb238"
    ),
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-c.md": (
        "10d434928dc2b2071c4cbc1353ec8d39179dd8db7d71ce03d6f1865a3763d4b0"
    ),
}
EXPECTED_OVERRIDE_COUNTS = {
    "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-100/audit.md": 1,
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-a.md": 8,
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-b.md": 12,
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-a.md": 36,
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-b.md": 19,
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-c.md": 26,
}
EXPECTED_CONTROL_COUNTS = {
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-a.md": 10,
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-b.md": 10,
    ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-c.md": 10,
}
HISTORICAL_RESPONSE_TRANSFORM_SPEC = {
    "version": "v11-historical-response-citation-markers-1",
    "substitutions": [
        [
            r"(?i)(?:passages?|references?)\s+\[\d+\](?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)\[\d+\])*\s+provide details:\s*",
            "\n\n",
        ],
        [
            r"(?i)\(\s*(?:as discussed in\s+)?(?:context|passages?|references?)\s*:?\s*\[\d+\](?:(?:\s*,\s*(?:and\s+)?|\s+and\s+)\[\d+\])*\s*\)",
            "",
        ],
        [
            r"(?i),\s*(?:(?:as\s+)?(?:described|discussed|mentioned)\s+in\s+)?context\s+\[\d+\]\s*,",
            " ",
        ],
        [
            r"(?i)\b(?:as\s+)?(?:described|discussed|mentioned)\s+in\s+context\s+\[\d+\]",
            "",
        ],
        [r"(?i)\bin\s+context\s+\[\d+\]\s*,?\s*", ""],
        [r"\[\d+\]", ""],
    ],
    "cleanup_substitutions": [
        [r"\(\s*\)", ""],
        [r"[ \t]+([,.;:])", r"\1"],
        [r" {2,}", " "],
        [r"[ \t]+\n", "\n"],
        [r"\n{3,}", "\n\n"],
    ],
    "strip": True,
}
HISTORICAL_RESPONSE_TRANSFORM_SHA256 = (
    "caf42f2e23a8bdfffbd26803d51a93866c0b074d01b16653b9baf70e9d695adf"
)


def build_adjudication_manifest(
    *,
    merged_result: Path,
    original_result: Path,
    run_plan: Path,
    audit_reports: Sequence[Path],
    overrides_path: Path,
    root: Path,
) -> dict[str, object]:
    """Return the hash-bound 282-case manifest from frozen v10 evidence."""
    root = Path(root).resolve()
    merged_path = _resolve(root, merged_result)
    original_path = _resolve(root, original_result)
    plan_path = _resolve(root, run_plan)
    overrides_path = _resolve(root, overrides_path)
    report_paths = tuple(_resolve(root, path) for path in audit_reports)
    if len(report_paths) != len(set(report_paths)):
        raise ValueError("audit reports must be unique")

    _validate_approved_sources(
        root=root,
        merged_path=merged_path,
        original_path=original_path,
        plan_path=plan_path,
        report_paths=report_paths,
        overrides_path=overrides_path,
    )
    if _json_hash(HISTORICAL_RESPONSE_TRANSFORM_SPEC) != (
        HISTORICAL_RESPONSE_TRANSFORM_SHA256
    ):
        raise ValueError("historical-response transform identity is invalid")
    merged = _read_json(merged_path, "merged result")
    original = _read_json(original_path, "original result")
    plan = _read_json(plan_path, "run plan")
    ledger = _read_json(overrides_path, "override ledger")
    merged_records = _records_by_ordinal(merged, expected=500, label="merged result")
    original_rows = _records(original, expected=100, label="original result")
    original_ordinals = [row["ordinal"] for row in original_rows]
    if len(original_ordinals) != len(set(original_ordinals)):
        raise ValueError("original result contains duplicate ordinals")
    for row in original_rows:
        ordinal = row["ordinal"]
        merged_row = merged_records.get(ordinal)
        if merged_row is None or _semantic_record(merged_row) != _semantic_record(row):
            raise ValueError("original result is not preserved in merged result")
    if plan.get("source_result") != _relative(root, original_path):
        raise ValueError("run plan does not bind the original result path")

    new_ordinals = _new_ordinals(plan, merged_records)
    new_low = sorted(
        ordinal
        for ordinal in new_ordinals
        if _model_pair(merged_records[ordinal])[0] < 2
    )
    if len(new_low) != 148:
        raise ValueError("expected exactly 148 new grade-0/1 records")
    report_hashes = {_relative(root, path): _sha256(path) for path in report_paths}
    overrides, controls = _validated_ledger(
        ledger,
        merged_records=merged_records,
        report_hashes=report_hashes,
        new_ordinals=new_ordinals,
        audited_override_ordinals=set(original_ordinals) | set(new_low),
    )
    bare_idks = plan.get("bare_idk_ordinals")
    if bare_idks != EXPECTED_BARE_IDKS:
        raise ValueError("run plan bare-IDK ordinals changed")
    control_ordinals = sorted(controls)
    selected = original_ordinals + new_low + control_ordinals + bare_idks
    if len(selected) != 282 or len(set(selected)) != 282:
        raise ValueError("adjudication cohorts must contain 282 unique records")

    report_for_case = _review_reports(
        report_hashes=report_hashes,
        original_ordinals=original_ordinals,
        new_low=new_low,
        controls=controls,
    )
    transformed_responses = {
        ordinal: sanitize_historical_response(
            merged_records[ordinal]["candidate_response"]
        )
        for ordinal in selected
    }
    lengths = [len(transformed_responses[ordinal]) for ordinal in selected]
    median_chars = statistics.median(lengths)
    sources = [
        _source(root, "merged_result", merged_path),
        _source(root, "original_result", original_path),
        _source(root, "run_plan", plan_path),
        *[_source(root, "audit_report", path) for path in report_paths],
        _source(root, "override_ledger", overrides_path),
    ]
    merged_hash = _sha256(merged_path)
    plan_hash = _sha256(plan_path)
    records: list[dict[str, object]] = []
    for ordinal in sorted(selected):
        source = merged_records[ordinal]
        base_grade, base_reason = _model_pair(source)
        override = overrides.get(ordinal)
        pair = (
            override["preferred_pair"]
            if override is not None
            else f"{base_grade}/{base_reason}"
        )
        grade_text, reason = pair.split("/", maxsplit=1)
        grade = int(grade_text)
        selection_reasons = []
        if ordinal in original_ordinals:
            selection_reasons.append("original_100")
        if ordinal in new_low:
            selection_reasons.append("audited_new_low_grade")
        if ordinal in controls:
            selection_reasons.append("audited_grade_2_control")
        if ordinal in bare_idks:
            selection_reasons.append("bare_idk_deterministic")
        audit_paths = set(report_for_case.get(ordinal, ()))
        if override is not None:
            audit_paths.add(override["source_report"])
        audit_hashes = sorted(report_hashes[path] for path in audit_paths)
        response = transformed_responses[ordinal]
        records.append(
            {
                "ordinal": ordinal,
                "question": source["question"],
                "evaluator_evidence": source["evaluator_evidence"],
                "historical_response": response,
                "reviewed_v10_pair": {"grade": grade, "reason": reason},
                "expected_v11_derived_label": PAIR_TO_LABEL[pair],
                "expected_numeric_grade": grade,
                "evidence_tier": source["reference_tier"],
                "historical_response_chars": len(response),
                "response_length_bucket": (
                    "short" if len(response) <= median_chars else "long"
                ),
                "selection_reasons": selection_reasons,
                "deterministic_no_provider_call": ordinal in bare_idks,
                "override_source_report": (
                    override["source_report"] if override is not None else None
                ),
                "provenance": {
                    "merged_result_sha256": merged_hash,
                    "audit_report_sha256": audit_hashes,
                    "deterministic_source_sha256": (
                        plan_hash if ordinal in bare_idks else None
                    ),
                },
            }
        )
    totals = Counter(row["expected_numeric_grade"] for row in records)
    if totals != EXPECTED_TOTALS:
        raise ValueError(f"unexpected gold totals: {dict(sorted(totals.items()))}")
    body: dict[str, object] = {
        "version": 1,
        "record_count": len(records),
        "override_count": len(overrides),
        "gold_numeric_grade_totals": {
            str(grade): totals[grade] for grade in sorted(totals)
        },
        "response_length_median_chars": median_chars,
        "historical_response_transform": {
            "version": HISTORICAL_RESPONSE_TRANSFORM_SPEC["version"],
            "sha256": HISTORICAL_RESPONSE_TRANSFORM_SHA256,
        },
        "selection": {
            "original_100_ordinals": sorted(original_ordinals),
            "new_low_grade_ordinals": new_low,
            "grade_2_control_ordinals": control_ordinals,
            "bare_idk_ordinals": bare_idks,
        },
        "sources": sources,
        "records": records,
    }
    return body | {"manifest_sha256": _json_hash(body)}


def provider_message_fixture(record: dict[str, object]) -> dict[str, str]:
    """Whitelist only semantic fields that a retrospective judge may receive."""
    fields = ("question", "evaluator_evidence", "historical_response")
    values = {field: record.get(field) for field in fields}
    if any(type(value) is not str or not value for value in values.values()):
        raise ValueError("adjudication record lacks semantic provider fields")
    values["historical_response"] = sanitize_historical_response(
        values["historical_response"]
    )
    return values  # type: ignore[return-value]


def sanitize_historical_response(response: str) -> str:
    """Remove numeric citation navigation while preserving answer semantics."""
    if type(response) is not str or not response:
        raise ValueError("historical response must be a nonempty string")
    if re.search(r"\[\d+\]", response) is None:
        return response
    transformed = response
    for pattern, replacement in HISTORICAL_RESPONSE_TRANSFORM_SPEC["substitutions"]:
        transformed = re.sub(pattern, replacement, transformed)
    for pattern, replacement in HISTORICAL_RESPONSE_TRANSFORM_SPEC[
        "cleanup_substitutions"
    ]:
        transformed = re.sub(pattern, replacement, transformed)
    if HISTORICAL_RESPONSE_TRANSFORM_SPEC["strip"]:
        transformed = transformed.strip()
    if not transformed or re.search(r"\[\d+\]", transformed):
        raise ValueError("historical-response citation transform failed")
    return transformed


def write_private_manifest(
    destination: Path,
    manifest: dict[str, object],
    *,
    root: Path = REPO_ROOT,
) -> None:
    """Atomically write a validated manifest with private modes."""
    destination, authorized_root = _authorized_destination(destination, root=root)
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if (
        manifest.get("manifest_sha256") != _json_hash(body)
        or manifest.get("record_count") != 282
        or manifest.get("override_count") != 102
        or manifest.get("gold_numeric_grade_totals") != {"0": 42, "1": 66, "2": 174}
    ):
        raise ValueError("adjudication manifest is invalid")
    authorized_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    current = authorized_root
    current.chmod(0o700)
    for part in destination.parent.relative_to(authorized_root).parts:
        current = current / part
        current.mkdir(exist_ok=True, mode=0o700)
        current.chmod(0o700)
    descriptor, temporary = tempfile.mkstemp(
        prefix=".manifest-", dir=destination.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, destination)
        destination.chmod(0o600)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        Path(temporary).unlink(missing_ok=True)
        raise


def _validated_ledger(
    ledger: dict[str, object],
    *,
    merged_records: dict[int, dict[str, object]],
    report_hashes: dict[str, str],
    new_ordinals: set[int],
    audited_override_ordinals: set[int],
) -> tuple[dict[int, dict[str, object]], dict[int, str]]:
    if (
        set(ledger) != {"version", "overrides", "grade_2_controls"}
        or ledger["version"] != 1
    ):
        raise ValueError("override ledger shape is invalid")
    raw_overrides = ledger["overrides"]
    raw_controls = ledger["grade_2_controls"]
    if type(raw_overrides) is not list or type(raw_controls) is not list:
        raise ValueError("override ledger arrays are invalid")
    overrides: dict[int, dict[str, object]] = {}
    for row in raw_overrides:
        if type(row) is not dict or set(row) != {
            "ordinal",
            "preferred_pair",
            "source_report",
        }:
            raise ValueError("override entry shape is invalid")
        ordinal = row["ordinal"]
        pair = row["preferred_pair"]
        report = row["source_report"]
        if (
            type(ordinal) is not int
            or ordinal not in merged_records
            or ordinal in overrides
            or pair not in PAIR_TO_LABEL
            or report not in report_hashes
            or ordinal not in audited_override_ordinals
        ):
            raise ValueError("override ordinal is outside audited selected cohorts")
        base_grade, base_reason = _model_pair(merged_records[ordinal])
        if pair == f"{base_grade}/{base_reason}":
            raise ValueError("ledger may contain only explicit preferred-pair changes")
        overrides[ordinal] = row
    if len(overrides) != 102:
        raise ValueError("expected exactly 102 explicit semantic overrides")
    if Counter(row["source_report"] for row in overrides.values()) != (
        EXPECTED_OVERRIDE_COUNTS
    ):
        raise ValueError("review certificate override counts are invalid")
    controls: dict[int, str] = {}
    for row in raw_controls:
        if type(row) is not dict or set(row) != {"ordinal", "source_report"}:
            raise ValueError("grade-2 control entry shape is invalid")
        ordinal = row["ordinal"]
        report = row["source_report"]
        if (
            type(ordinal) is not int
            or ordinal in controls
            or ordinal not in new_ordinals
            or report not in report_hashes
            or _model_pair(merged_records[ordinal]) != (2, "supported")
        ):
            raise ValueError("grade-2 control entry is invalid")
        controls[ordinal] = report
    if len(controls) != 30:
        raise ValueError("expected exactly 30 audited grade-2 controls")
    if Counter(controls.values()) != EXPECTED_CONTROL_COUNTS:
        raise ValueError("review certificate control counts are invalid")
    return overrides, controls


def _review_reports(
    *,
    report_hashes: dict[str, str],
    original_ordinals: list[int],
    new_low: list[int],
    controls: dict[int, str],
) -> dict[int, tuple[str, ...]]:
    def named(filename: str) -> str:
        matches = [path for path in report_hashes if path.endswith(filename)]
        if len(matches) != 1:
            raise ValueError(f"expected one audit report ending in {filename}")
        return matches[0]

    original_a = named("/semantic-audit-a.md")
    original_b = named("/semantic-audit-b.md")
    new_a = named("/semantic-audit-500-a.md")
    new_b = named("/semantic-audit-500-b.md")
    new_c = named("/semantic-audit-500-c.md")
    result: dict[int, tuple[str, ...]] = {}
    for position, ordinal in enumerate(original_ordinals):
        result[ordinal] = (original_a if position < 50 else original_b,)
    for rank, ordinal in enumerate(new_low):
        result[ordinal] = (new_a if rank < 50 else new_b if rank < 100 else new_c,)
    for ordinal, report in controls.items():
        result[ordinal] = (report,)
    return result


def _records(
    value: dict[str, object], *, expected: int, label: str
) -> list[dict[str, object]]:
    rows = value.get("records")
    if (
        type(rows) is not list
        or len(rows) != expected
        or any(type(row) is not dict for row in rows)
    ):
        raise ValueError(f"{label} records are invalid")
    return rows


def _records_by_ordinal(
    value: dict[str, object], *, expected: int, label: str
) -> dict[int, dict[str, object]]:
    rows = _records(value, expected=expected, label=label)
    result: dict[int, dict[str, object]] = {}
    for row in rows:
        ordinal = row.get("ordinal")
        if type(ordinal) is not int or ordinal in result:
            raise ValueError(f"{label} ordinals are invalid")
        result[ordinal] = row
    return result


def _new_ordinals(
    plan: dict[str, object], merged_records: dict[int, dict[str, object]]
) -> set[int]:
    batches = plan.get("batches")
    if type(batches) is not list:
        raise ValueError("run plan batches are invalid")
    result: list[int] = []
    for batch in batches:
        if type(batch) is not dict or type(batch.get("new_ordinals")) is not list:
            raise ValueError("run plan batch is invalid")
        result.extend(batch["new_ordinals"])
    if (
        len(result) != 396
        or len(set(result)) != 396
        or any(
            type(ordinal) is not int or ordinal not in merged_records
            for ordinal in result
        )
    ):
        raise ValueError("run plan new ordinals are invalid")
    return set(result)


def _model_pair(row: dict[str, object]) -> tuple[int, str]:
    try:
        decision = row["normalized"]["decision"]
        grade = decision["grade"]
        reason = decision["reason"]
    except (KeyError, TypeError) as error:
        raise ValueError("result decision is invalid") from error
    pair = f"{grade}/{reason}"
    if type(grade) is not int or type(reason) is not str or pair not in PAIR_TO_LABEL:
        raise ValueError("result grade/reason pair is invalid")
    return grade, reason


def _semantic_record(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row.get("ordinal"),
        row.get("question"),
        row.get("evaluator_evidence"),
        row.get("candidate_response"),
        _model_pair(row),
        row.get("reference_tier"),
    )


def _validate_approved_sources(
    *,
    root: Path,
    merged_path: Path,
    original_path: Path,
    plan_path: Path,
    report_paths: Sequence[Path],
    overrides_path: Path,
) -> None:
    paths = (merged_path, original_path, plan_path, *report_paths)
    provided = {_relative(root, path): path for path in paths}
    if set(provided) != set(APPROVED_SOURCE_SHA256):
        raise ValueError("approved source paths mismatch")
    for relative, expected_hash in APPROVED_SOURCE_SHA256.items():
        if _sha256(provided[relative]) != expected_hash:
            raise ValueError(f"approved source hash mismatch: {relative}")
    if _relative(root, overrides_path) != DEFAULT_OVERRIDES.as_posix():
        raise ValueError("approved ledger path mismatch")
    if _sha256(overrides_path) != EXPECTED_LEDGER_SHA256:
        raise ValueError("approved ledger hash mismatch")


def _authorized_destination(destination: Path, *, root: Path) -> tuple[Path, Path]:
    root = Path(root).resolve()
    authorized_lexical = root / AUTHORIZED_OUTPUT_ROOT
    current = root
    for part in AUTHORIZED_OUTPUT_ROOT.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("authorized v11 root must use real repository directories")
    authorized = authorized_lexical.resolve()
    destination = Path(destination)
    destination = (
        destination.resolve()
        if destination.is_absolute()
        else (root / destination).resolve()
    )
    if not authorized.is_relative_to(root):
        raise ValueError("authorized v11 artifact root escapes repository root")
    if destination == authorized or not destination.is_relative_to(authorized):
        raise ValueError("output must be inside authorized v11 artifact root")
    return destination, authorized


def _read_json(path: Path, label: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is unreadable") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be a JSON object")
    return value


def _source(root: Path, role: str, path: Path) -> dict[str, str]:
    return {"role": role, "path": _relative(root, path), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _resolve(root: Path, path: Path) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            "all adjudication inputs must be within the repository"
        ) from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-result", type=Path, default=DEFAULT_MERGED_RESULT)
    parser.add_argument("--original-result", type=Path, default=DEFAULT_ORIGINAL_RESULT)
    parser.add_argument("--run-plan", type=Path, default=DEFAULT_RUN_PLAN)
    parser.add_argument("--overrides", type=Path, default=DEFAULT_OVERRIDES)
    parser.add_argument("--audit-report", action="append", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reports = tuple(args.audit_report) if args.audit_report else DEFAULT_AUDIT_REPORTS
    manifest = build_adjudication_manifest(
        merged_result=args.merged_result,
        original_result=args.original_result,
        run_plan=args.run_plan,
        audit_reports=reports,
        overrides_path=args.overrides,
        root=REPO_ROOT,
    )
    output = _resolve(REPO_ROOT, args.output)
    write_private_manifest(output, manifest)
    print(
        f"wrote {manifest['record_count']} private adjudication records to "
        f"{_relative(REPO_ROOT, output)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
