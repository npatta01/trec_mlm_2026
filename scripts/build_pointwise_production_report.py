#!/usr/bin/env python3
"""Build a text-free operational progress or final report for a private run."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import stat
import tempfile
from typing import Sequence


_LABELS = {"good", "partial", "incorrect", "uncertain"}
_CONFIDENCE = {"high", "medium", "low"}
_FLAGS = {
    "entity_error",
    "number_error",
    "date_error",
    "negation_error",
    "unsupported_specificity",
    "refusal",
    "ambiguity",
    "reference_conflict",
}


def build_report(run_dir: Path) -> dict[str, object]:
    """Summarize private operational evidence using an explicit safe whitelist."""
    directory = Path(run_dir)
    manifest = _read_private(directory / "run-manifest.json", "run manifest")
    planned = manifest.get("record_count")
    if type(planned) is not int or planned < 1:
        raise ValueError("run manifest record count is invalid")
    results = [
        _read_private(path, "terminal result")
        for path in sorted((directory / "results").glob("*.json"))
    ]
    attempts = [
        _read_private(path, "attempt marker")
        for path in sorted((directory / "attempts").glob("*.json"))
    ]
    raws = [
        _read_private(path, "raw checkpoint")
        for path in sorted((directory / "raw").glob("*.json"))
    ]
    receipts = [
        _read_private(path, "block receipt")
        for path in sorted((directory / "blocks").glob("*-receipt.json"))
    ]
    spend = _read_private(directory / "spend-ledger.json", "spend ledger")
    completed = len(results)
    valid = [
        row
        for row in results
        if row.get("status") == "valid" and type(row.get("decision")) is dict
    ]
    invalid = completed - len(valid)
    spent = _number(spend.get("spent_usd"))
    projection = None if completed == 0 else spent * planned / completed
    return {
        "version": 1,
        "coverage": {
            "completed_records": completed,
            "planned_records": planned,
            "block_receipts": len(receipts),
        },
        "parse_invariant_failures": {
            "terminal_invalid": invalid,
            "terminal_valid": len(valid),
        },
        "retries_repairs": {
            "physical_attempts": len(attempts),
            "repairs": sum(marker.get("repair") is True for marker in attempts),
            "retries": max(
                0,
                sum(marker.get("state") != "cancelled_pre_send" for marker in attempts)
                - completed,
            ),
        },
        "cost": {
            "spent_usd": spent,
            "cap_usd": _number(spend.get("cap_usd")),
            "projected_total_usd": projection,
        },
        "latency": _latency(raws),
        "distributions": _distributions(valid),
        "anomaly_findings": _anomalies(directory),
        "opus_findings": _opus_findings(directory),
        "claims": {"population_quality": False, "operational_only": True},
    }


def render_sanitized_markdown(summary: dict[str, object]) -> str:
    """Render only report fields built by ``build_report``; never interpolate inputs."""
    coverage = _mapping(summary.get("coverage"))
    failures = _mapping(summary.get("parse_invariant_failures"))
    retries = _mapping(summary.get("retries_repairs"))
    cost = _mapping(summary.get("cost"))
    latency = _mapping(summary.get("latency"))
    distributions = _mapping(summary.get("distributions"))
    anomalies = summary.get("anomaly_findings")
    opus = _mapping(summary.get("opus_findings"))
    if type(anomalies) is not list:
        raise ValueError("report anomaly findings are invalid")
    return "\n".join(
        [
            "# DeepSeek pointwise production report",
            "",
            "## Coverage",
            f"- Completed records: {coverage['completed_records']} / {coverage['planned_records']}",
            f"- Validated block receipts: {coverage['block_receipts']}",
            "",
            "## Parse and invariant failures",
            f"- Terminal-valid records: {failures['terminal_valid']}",
            f"- Terminal-invalid records: {failures['terminal_invalid']}",
            "",
            "## Retries and repairs",
            f"- Physical attempts: {retries['physical_attempts']}",
            f"- Repairs: {retries['repairs']}; retries: {retries['retries']}",
            "",
            "## Cost and latency",
            f"- Recorded spend: USD {_money(cost['spent_usd'])} of USD {_money(cost['cap_usd'])} cap.",
            f"- Projected total at current observed rate: {_projected(cost['projected_total_usd'])}.",
            f"- Observed checkpoint latency: {latency['count']} records, mean {_number(latency['mean']):.3f}s, max {_number(latency['max']):.3f}s.",
            "",
            "## Distribution checks",
            f"- Labels: {_distribution_text(_mapping(distributions['labels']))}",
            f"- Confidence: {_distribution_text(_mapping(distributions['confidence']))}",
            f"- Error flags: {_distribution_text(_mapping(distributions['error_flags']))}",
            "",
            "## Anomaly findings",
            *(
                [f"- {item}" for item in anomalies]
                or ["- None recorded in available diagnostics."]
            ),
            "",
            "## Opus findings",
            f"- Accepted findings: {opus['accepted']}; rejected findings: {opus['rejected']}.",
            "",
            "## Claim boundary",
            "This is operational evidence about private run execution and does not establish population quality, retrieval quality, or answer quality.",
            "",
        ]
    )


def write_report(destination: Path, markdown: str) -> None:
    """Write a sanitized report atomically; it contains no private record text."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            handle.write(markdown)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(0o644)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _distributions(results: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    labels: Counter[str] = Counter()
    confidence: Counter[str] = Counter()
    flags: Counter[str] = Counter()
    for result in results:
        decision = result["decision"]
        assert type(decision) is dict
        label = decision.get("label")
        if label in _LABELS:
            labels[str(label)] += 1
        confidence_value = decision.get("confidence")
        if confidence_value in _CONFIDENCE:
            confidence[str(confidence_value)] += 1
        values = decision.get("error_flags")
        if type(values) is list:
            flags.update(str(value) for value in values if value in _FLAGS)
    return {
        "labels": dict(sorted(labels.items())),
        "confidence": dict(sorted(confidence.items())),
        "error_flags": dict(sorted(flags.items())),
    }


def _latency(raws: list[dict[str, object]]) -> dict[str, float | int]:
    values: list[float] = []
    for raw in raws:
        response = raw.get("response")
        if type(response) is dict:
            value = response.get("latency_seconds")
            if (
                type(value) in {int, float}
                and math.isfinite(float(value))
                and value >= 0
            ):
                values.append(float(value))
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else 0.0,
        "max": max(values, default=0.0),
    }


def _anomalies(directory: Path) -> list[str]:
    safe: set[str] = set()
    for path in (directory / "anomalies").glob("[0-9]*.json"):
        diagnostics = _read_private(path, "anomaly diagnostics").get("diagnostics")
        if type(diagnostics) is list:
            for diagnostic in diagnostics:
                if type(diagnostic) is str:
                    safe.add(_safe_anomaly(diagnostic))
    return sorted(value for value in safe if value)


def _safe_anomaly(value: str) -> str | None:
    mappings = {
        "all labels in a full block are identical": "identical labels in a full block",
        "repeated provider output exceeds 95 percent of block": "repeated provider output",
        "provider/model drift observed in anomaly metrics": "provider/model drift",
        "repair or retry rate exceeds 25 percent": "elevated repair or retry rate",
        "projected DeepSeek spend exceeds USD 50 pause threshold": "projected spend pause threshold",
    }
    if value in mappings:
        return mappings[value]
    if value.endswith("distribution shifted more than 0.8"):
        return "decision distribution shift"
    if value.endswith("mean increased more than fivefold"):
        return "operational metric mean shift"
    return None


def _opus_findings(directory: Path) -> dict[str, int]:
    path = directory / "opus-findings.json"
    if not path.exists():
        return {"accepted": 0, "rejected": 0}
    value = _read_private(path, "Opus findings")
    counts = value.get("finding_counts")
    if type(counts) is dict:
        accepted, rejected = counts.get("accepted"), counts.get("rejected")
        if (
            type(accepted) is int
            and accepted >= 0
            and type(rejected) is int
            and rejected >= 0
        ):
            return {"accepted": accepted, "rejected": rejected}
    ledger = value.get("recommendation_ledger")
    if type(ledger) is list:
        decisions = Counter(
            item.get("decision") for item in ledger if type(item) is dict
        )
        return {"accepted": decisions["accepted"], "rejected": decisions["rejected"]}
    return {"accepted": 0, "rejected": 0}


def _read_private(path: Path, label: str) -> dict[str, object]:
    if (
        path.is_symlink()
        or not path.is_file()
        or stat.S_IMODE(path.stat().st_mode) != 0o600
    ):
        raise ValueError(f"{label} must be a mode-0600 regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _number(value: object) -> float:
    if type(value) not in {int, float} or not math.isfinite(float(value)) or value < 0:
        raise ValueError("report numeric input is invalid")
    return float(value)


def _mapping(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ValueError("report summary is invalid")
    return value


def _money(value: object) -> str:
    return f"{_number(value):.2f}"


def _projected(value: object) -> str:
    return (
        "not available until a record completes"
        if value is None
        else f"USD {_money(value)}"
    )


def _distribution_text(value: dict[str, object]) -> str:
    return (
        ", ".join(f"{key}={int(count)}" for key, count in sorted(value.items()))
        or "none"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    write_report(args.output, render_sanitized_markdown(build_report(args.run_dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
