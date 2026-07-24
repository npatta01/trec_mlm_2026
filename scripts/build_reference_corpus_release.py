#!/usr/bin/env python3
"""Build the versioned private reference-corpus release bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERSION = "reference-corpus-v1"
REMOTE_ROOT = "hf://buckets/Npatta01/trec_mlm_2026/reference-corpus-v1/"
OUT = Path("/tmp/dstack-trec-mlm-reference") / VERSION

PUBLIC_FILES = {
    "answers/perplexity_full_cache.json": ROOT / "data/llm_discovery/perplexity_full_cache.json",
    "answers/gemini36_grounded_all_7759.json": ROOT / "artifacts/reference_answer_validation/grounding/gemini36_grounded_all_7759.json",
    "manifests/gemini36_grounded_all_7759_sample.json": ROOT / "artifacts/reference_answer_validation/grounding/gemini36_grounded_all_7759_sample.json",
    "reports/trec_2026_gemini_grounded_full_audit.json": ROOT / "reports/trec_2026_gemini_grounded_full_audit.json",
    "reports/trec_2026_four_model_per_question_rankings.tsv": ROOT / "reports/trec_2026_four_model_per_question_rankings.tsv",
}

PRIVATE_FILES = {
    "evidence/gemini36_grounded_all_7759_evidence.json": ROOT / "artifacts/reference_answer_validation/private/gemini36_grounded_all_7759_evidence.json",
    "receipts/friend_key_spend_receipt.json": ROOT / "artifacts/reference_answer_validation/private/friend_key_spend_receipt.json",
    "receipts/friend_new_key_spend_receipt.json": ROOT / "artifacts/reference_answer_validation/private/friend_new_key_spend_receipt.json",
    "receipts/owner_key_spend_receipt.json": ROOT / "artifacts/reference_answer_validation/private/owner_key_spend_receipt.json",
    "receipts/tanwir_key_spend_receipt.json": ROOT / "artifacts/reference_answer_validation/private/tanwir_key_spend_receipt.json",
    "receipts/tanwir_new_key_spend_receipt.json": ROOT / "artifacts/reference_answer_validation/private/tanwir_new_key_spend_receipt.json",
}

SECRET_PATTERNS = [
    re.compile(r"(?<![A-Za-z0-9])AQ\.[A-Za-z0-9_-]{20,}"),
    re.compile(r"(?<![A-Za-z0-9])AIza[A-Za-z0-9_-]{20,}"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    data = source.read_bytes()
    text = data.decode("utf-8", errors="ignore")
    if any(pattern.search(text) for pattern in SECRET_PATTERNS):
        raise ValueError(f"credential-like text detected in {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "public").mkdir(parents=True)
    (OUT / "private").mkdir(parents=True)

    entries: list[dict[str, object]] = []
    for access, mapping in (("public", PUBLIC_FILES), ("private", PRIVATE_FILES)):
        for relative, source in mapping.items():
            destination = OUT / access / relative
            copy_file(source, destination)
            entries.append(
                {
                    "path": f"{access}/{relative}",
                    "source": str(source.relative_to(ROOT)),
                    "access": access,
                    "bytes": destination.stat().st_size,
                    "sha256": sha256(destination),
                }
            )

    manifest = {
        "schema_version": 1,
        "artifact_id": VERSION,
        "project": "trec_mlm_2026",
        "remote_root": REMOTE_ROOT,
        "github_release_tag": VERSION,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "git_commit": os.environ.get("GIT_COMMIT", "unknown"),
        "question_count": 7759,
        "models": ["Perplexity Sonar", "Gemini 3.6 Flash"],
        "prompt_hash": "4591379724312e776e353df0815ef50af7445d3806394404ab4fdaca9edd496b",
        "access_policy": "private; public paths contain answer/report artifacts, private paths contain evidence and credential-spend receipts",
        "files": sorted(entries, key=lambda item: str(item["path"])),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "README.md").write_text(
        "# TREC Million LLMs 2026 reference corpus v1\n\n"
        "This private bundle preserves the Perplexity and Gemini reference answers, Gemini grounding provenance, and per-credential spend receipts.\n\n"
        f"Canonical HF location: `{REMOTE_ROOT}`\n\n"
        "The `public/` directory contains answer/report artifacts. The `private/` directory contains source mappings and operational receipts and must remain access-controlled. API keys are intentionally excluded. See `manifest.json` for hashes, source paths, model metadata, and the GitHub release tag.\n",
        encoding="utf-8",
    )
    lines = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            lines.append(f"{sha256(path)}  {path.relative_to(OUT)}")
    (OUT / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "files": len(entries), "bytes": sum(int(e["bytes"]) for e in entries)}, indent=2))


if __name__ == "__main__":
    main()
