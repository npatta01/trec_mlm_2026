#!/usr/bin/env python3
"""Create a read-only readiness receipt for reference-answer models."""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Mapping, Sequence


MODELS = {
    "qwen35": {
        "base": "Qwen/Qwen3.6-35B-A3B",
        "base_revision": "995ad96eacd98c81ed38be0c5b274b04031597b0",
        "fp8": "Qwen/Qwen3.6-35B-A3B-FP8",
        "fp8_revision": "95a723d08a9490559dae23d0cff1d9466213d989",
    },
    "qwen27": {
        "base": "Qwen/Qwen3.6-27B",
        "base_revision": "6a9e13bd6fc8f0983b9b99948120bc37f49c13e9",
        "fp8": "Qwen/Qwen3.6-27B-FP8",
        "fp8_revision": "e89b16ebf1988b3d6befa7de50abc2d76f26eb09",
    },
}

VLLM_IMAGE = "vllm/vllm-openai-rocm:latest"
EXPECTED_PEAK_MEMORY_BYTES = {
    "qwen35": 56 * 1024**3,
    "qwen27": 48 * 1024**3,
}


def _parse_meminfo(text: str) -> dict[str, int | None]:
    values: dict[str, int] = {}
    for line in text.splitlines():
        name, separator, remainder = line.partition(":")
        if not separator:
            continue
        fields = remainder.split()
        if fields and fields[0].isdigit():
            values[name] = int(fields[0]) * 1024
    return {
        "available_ram_bytes": values.get("MemAvailable"),
        "free_swap_bytes": values.get("SwapFree"),
    }


def _json_object(text: str) -> dict[str, object]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _download_bytes(text: str) -> int | None:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return None
    rows = value if isinstance(value, list) else [value]
    sizes: list[int] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        size = row.get("file_size", row.get("size"))
        if isinstance(size, int):
            sizes.append(size)
        elif isinstance(size, str):
            suffix = size[-1] if size[-1:].isalpha() else ""
            multipliers = {
                "": 1,
                "K": 1_000,
                "M": 1_000_000,
                "G": 1_000_000_000,
                "T": 1_000_000_000_000,
            }
            if suffix not in multipliers:
                continue
            try:
                sizes.append(
                    int(Decimal(size.removesuffix(suffix)) * multipliers[suffix])
                )
            except InvalidOperation:
                continue
    return sum(sizes) if sizes else None


def _positive_finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value > 0
    )


def _podman_image_references(text: str) -> set[str]:
    return {
        fields[0].removeprefix("docker.io/")
        for line in text.splitlines()
        if (fields := line.split())
    }


def _model_receipt(
    model_key: str, representation: str, command_results: Mapping[str, str]
) -> dict[str, object]:
    repository = MODELS[model_key][representation]
    revision = MODELS[model_key][f"{representation}_revision"]
    metadata = _json_object(
        command_results.get(f"{model_key}_{representation}_info", "")
    )
    tags = metadata.get("tags", [])
    tags = tags if isinstance(tags, list) else []
    license_tag = next(
        (
            tag.removeprefix("license:")
            for tag in tags
            if isinstance(tag, str) and tag.startswith("license:")
        ),
        None,
    )
    safetensors = metadata.get("safetensors", {})
    parameter_count = (
        safetensors.get("total") if isinstance(safetensors, dict) else None
    )
    observed_sha = metadata.get("sha")
    exact_repository_bytes = metadata.get("used_storage")
    dry_run_download_bytes = _download_bytes(
        command_results.get(f"{model_key}_{representation}_dry_run", "")
    )
    evidence_reasons: list[str] = []
    if observed_sha != revision:
        evidence_reasons.append("SHA does not match the pinned revision")
    if not isinstance(license_tag, str) or not license_tag.strip():
        evidence_reasons.append("license tag is missing")
    if not _positive_finite_number(parameter_count):
        evidence_reasons.append("parameter count must be finite and positive")
    if not _positive_finite_number(exact_repository_bytes):
        evidence_reasons.append("repository bytes must be finite and positive")
    if not _positive_finite_number(dry_run_download_bytes):
        evidence_reasons.append("dry-run bytes must be finite and positive")
    return {
        "repository": repository,
        "revision": revision,
        "observed_sha": observed_sha,
        "revision_matches": observed_sha == revision,
        "license": license_tag,
        "parameter_count": parameter_count,
        "exact_repository_bytes": exact_repository_bytes,
        "dry_run_download_bytes": dry_run_download_bytes,
        "evidence_complete": not evidence_reasons,
        "evidence_reasons": evidence_reasons,
    }


def build_preflight(
    environ: Mapping[str, str],
    command_results: Mapping[str, str],
    disk_free_bytes: int,
    commit: str,
    dirty: bool,
) -> dict[str, object]:
    """Build a sanitized receipt from already-collected host observations."""
    memory = _parse_meminfo(command_results.get("meminfo", ""))
    rocminfo = command_results.get("rocminfo", "")
    podman_version = command_results.get("podman_version", "")
    podman_images = command_results.get("podman_images", "")
    podman_rootless = command_results.get("podman_rootless", "")
    gfx1151 = "gfx1151" in rocminfo
    image_present = VLLM_IMAGE in _podman_image_references(podman_images)
    available_ram = memory["available_ram_bytes"]
    enough_ram = isinstance(available_ram, int) and available_ram >= max(
        EXPECTED_PEAK_MEMORY_BYTES.values()
    )

    local_reasons: list[str] = []
    if not gfx1151:
        local_reasons.append("rocminfo must report a gfx1151 device")
    if not podman_version:
        local_reasons.append("rootless Podman must be installed")
    if podman_rootless.strip().lower() != "true":
        local_reasons.append("Podman must provide explicit rootless runtime evidence")
    if not image_present:
        local_reasons.append(f"container image {VLLM_IMAGE} must be present")
    if not enough_ram:
        local_reasons.append("available RAM must cover the 56 GiB peak-memory estimate")

    models = {
        key: {
            representation: _model_receipt(key, representation, command_results)
            for representation in ("base", "fp8")
        }
        for key in MODELS
    }
    observed_metadata = [
        record
        for variants in models.values()
        for record in variants.values()
        if record["observed_sha"] is not None
    ]
    pins_verified = len(observed_metadata) == 4 and all(
        record["revision_matches"] for record in observed_metadata
    )
    receipt_reasons = [
        f"{model_key}.{representation}: {reason}"
        for model_key, variants in models.items()
        for representation, record in variants.items()
        for reason in record["evidence_reasons"]
    ]
    receipt_complete = not receipt_reasons
    selected_download_sizes = []
    for variants in models.values():
        selected = variants["fp8"]
        selected_download_sizes.append(
            selected["exact_repository_bytes"]
            if isinstance(selected["exact_repository_bytes"], int)
            else selected["dry_run_download_bytes"]
        )
    if all(isinstance(size, int) for size in selected_download_sizes):
        selected_download_bytes = sum(selected_download_sizes)
        if disk_free_bytes < selected_download_bytes:
            local_reasons.append(
                "free disk must cover both selected FP8 model downloads"
            )
    else:
        selected_download_bytes = None
        local_reasons.append("selected FP8 model sizes must be complete")

    local_download_reasons = [*receipt_reasons, *local_reasons]
    local_download_ready = not local_download_reasons
    gemini_configured = bool(environ.get("GEMINI_API_KEY"))

    return {
        "schema_version": "reference-model-preflight-v1",
        "repository": {"commit": commit.strip(), "dirty": dirty},
        "credentials": {
            "gemini_api_key": (
                "configured" if environ.get("GEMINI_API_KEY") else "not_configured"
            )
        },
        "hardware": {
            **memory,
            "rocm_device": "gfx1151" if gfx1151 else "not_detected",
            "gfx1151_evidence": gfx1151,
        },
        "runtime": {
            "podman_version": podman_version.strip() or None,
            "podman_rootless": (
                podman_rootless.strip().lower() == "true" if podman_rootless else None
            ),
            "vllm_image": VLLM_IMAGE,
            "vllm_image_present": image_present,
        },
        "storage": {"minimum_free_bytes": disk_free_bytes},
        "models": models,
        "selection": {
            "representation": "fp8",
            "expected_peak_memory_bytes": EXPECTED_PEAK_MEMORY_BYTES,
            "combined_required_disk_bytes": selected_download_bytes,
        },
        "safety": {
            "model_downloads_performed": False,
            "paid_calls_performed": False,
        },
        "gates": {
            "receipt_complete": receipt_complete,
            "receipt_reasons": receipt_reasons,
            "local_download_ready": local_download_ready,
            "local_download_reasons": local_download_reasons,
            "local_model_smoke_ready": local_download_ready,
            "local_model_reason": (
                "ready" if local_download_ready else "; ".join(local_download_reasons)
            ),
            "model_pins_verified": pins_verified,
        },
        "execution_gates": {
            "gemini_pilot_ready": gemini_configured,
            "gemini_pilot_reason": (
                "credential configured; paid calls still require explicit Task 5 authorization"
                if gemini_configured
                else "GEMINI_API_KEY is not configured; it is required only after explicit Task 5 authorization"
            ),
        },
    }


def preflight_exit_code(receipt: Mapping[str, object]) -> int:
    """Return zero only for a complete, locally download-ready receipt."""
    gates = receipt.get("gates")
    if not isinstance(gates, Mapping):
        return 1
    return (
        0
        if gates.get("receipt_complete") is True
        and gates.get("local_download_ready") is True
        else 1
    )


def _run(argv: list[str]) -> str:
    if not argv or not all(isinstance(item, str) and item for item in argv):
        raise ValueError("command argv must contain non-empty literal strings")
    completed = subprocess.run(
        argv, shell=False, check=True, text=True, capture_output=True
    )
    return completed.stdout


def _collect(command_results: dict[str, str], key: str, argv: list[str]) -> None:
    try:
        command_results[key] = _run(argv)
    except (FileNotFoundError, subprocess.CalledProcessError):
        command_results[key] = ""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Exit 0 means the receipt and required local metadata gates are "
            "complete. Gemini credential absence is informational until Task 5."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/reference_answer_validation/preflight.json"),
    )
    parser.add_argument(
        "--gemini-cost-cap-usd",
        type=float,
        required=True,
        help="finite, nonnegative projected pilot cap; does not authorize paid calls",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not math.isfinite(args.gemini_cost_cap_usd) or args.gemini_cost_cap_usd < 0:
        raise SystemExit("--gemini-cost-cap-usd must be finite and nonnegative")

    commands: dict[str, str] = {}
    _collect(commands, "meminfo", ["cat", "/proc/meminfo"])
    _collect(commands, "rocminfo", ["rocminfo"])
    _collect(commands, "podman_version", ["podman", "--version"])
    _collect(
        commands,
        "podman_rootless",
        ["podman", "info", "--format", "{{.Host.Security.Rootless}}"],
    )
    _collect(
        commands,
        "podman_images",
        ["podman", "images", "--format", "{{.Repository}}:{{.Tag}} {{.ID}}"],
    )
    commit = _run(["git", "rev-parse", "HEAD"]).strip()
    dirty = bool(_run(["git", "status", "--porcelain"]).strip())

    for model_key, variants in MODELS.items():
        for representation in ("base", "fp8"):
            repository = variants[representation]
            revision = variants[f"{representation}_revision"]
            _collect(
                commands,
                f"{model_key}_{representation}_info",
                [
                    "hf",
                    "models",
                    "info",
                    repository,
                    "--revision",
                    revision,
                    "--expand",
                    "sha,tags,safetensors,usedStorage",
                    "--json",
                ],
            )
            _collect(
                commands,
                f"{model_key}_{representation}_dry_run",
                [
                    "hf",
                    "download",
                    repository,
                    "--revision",
                    revision,
                    "--dry-run",
                    "--json",
                ],
            )

    cache_dir = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cache_free = shutil.disk_usage(cache_dir).free
    artifact_free = shutil.disk_usage(args.output.parent).free
    receipt = build_preflight(
        os.environ,
        commands,
        min(cache_free, artifact_free),
        commit,
        dirty,
    )
    receipt["storage"] = {
        "hugging_face_cache": str(cache_dir),
        "hugging_face_cache_free_bytes": cache_free,
        "artifact_directory": str(args.output.parent),
        "artifact_directory_free_bytes": artifact_free,
        "minimum_free_bytes": min(cache_free, artifact_free),
    }
    receipt["gemini"] = {"projected_pilot_cost_cap_usd": args.gemini_cost_cap_usd}
    args.output.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "gates": receipt["gates"],
                "execution_gates": receipt["execution_gates"],
            },
            sort_keys=True,
        )
    )
    return preflight_exit_code(receipt)


if __name__ == "__main__":
    raise SystemExit(main())
