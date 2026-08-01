from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


_SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "audit_openai_pointwise_batch_results.py"
)
_SPEC = importlib.util.spec_from_file_location("pointwise_batch_audit", _SCRIPT)
assert _SPEC and _SPEC.loader
audit = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(audit)


def _metrics(*, inconsistent: int, disagreements: int) -> dict[str, int]:
    return {
        "groups": 3,
        "records": 8,
        "five_way_inconsistent_groups": inconsistent,
        "boundary_inconsistent_groups": inconsistent,
        "pairwise_boundary_disagreements": disagreements,
        "pairwise_comparisons": 10,
    }


def _change(index: int, old: str, new: str) -> dict[str, object]:
    return {
        "record_id": f"profile-record-v2:{index:064x}",
        "source_ordinal": index,
        "original_verdict": old,
        "new_verdict": new,
    }


def _fixture(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    group_change = _change(1, "incorrect", "partial")
    sample_change = _change(2, "nonanswer", "correct")
    value = {
        "source_final_results_sha256": "a" * 64,
        "recommended_applied_changes": [group_change, sample_change],
        "baseline": {
            "duplicate_groups_non_deterministic": 3,
            "duplicate_rows_non_deterministic": 8,
            "five_way_inconsistent_groups": 2,
            "boundary_inconsistent_groups": 2,
            "boundary_pair_disagreement": "4/10",
        },
        "post_recommended_clear_repairs": {
            "five_way_inconsistent_duplicate_groups": 1,
            "boundary_inconsistent_duplicate_groups": 1,
            "boundary_pair_disagreement": "2/10",
            "remaining_boundary_groups_are_debatable": [2],
        },
        "adjudication": {
            "clear_groups": 1,
            "debatable_groups": 1,
            "groups": [
                {
                    "group_index": 1,
                    "confidence": "clear",
                    "apply_override": True,
                    "changes": [group_change],
                },
                {
                    "group_index": 2,
                    "confidence": "debatable",
                    "apply_override": False,
                    "changes": [],
                },
            ],
        },
        "sample_boundary_overrides_outside_duplicate_groups": [sample_change],
    }
    path = tmp_path / "adjudication.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path, value


def _overrides() -> dict[str, object]:
    duplicate = SimpleNamespace(
        record_id=f"profile-record-v2:{1:064x}",
        source_ordinal=1,
        original_verdict="incorrect",
        verdict="partial",
        reason="duplicate_prompt_consistency",
    )
    sample = SimpleNamespace(
        record_id=f"profile-record-v2:{2:064x}",
        source_ordinal=2,
        original_verdict="nonanswer",
        verdict="correct",
        reason="semantic_sample",
    )
    return {duplicate.record_id: duplicate, sample.record_id: sample}


def test_adjudication_is_hash_bound_and_covers_every_override(tmp_path: Path) -> None:
    path, _ = _fixture(tmp_path)

    result = audit._adjudication(
        path,
        raw_sha256="a" * 64,
        overrides=_overrides(),
        raw_metrics=_metrics(inconsistent=2, disagreements=4),
        curated_metrics=_metrics(inconsistent=1, disagreements=2),
    )

    assert result is not None
    assert result["clear_groups"] == 1
    assert result["debatable_groups"] == 1
    assert result["semantic_sample_overrides"] == 1


def test_adjudication_rejects_mismatched_raw_results(tmp_path: Path) -> None:
    path, _ = _fixture(tmp_path)

    with pytest.raises(ValueError, match="different raw results"):
        audit._adjudication(
            path,
            raw_sha256="b" * 64,
            overrides=_overrides(),
            raw_metrics=_metrics(inconsistent=2, disagreements=4),
            curated_metrics=_metrics(inconsistent=1, disagreements=2),
        )


def test_adjudication_rejects_incomplete_override_coverage(tmp_path: Path) -> None:
    path, value = _fixture(tmp_path)
    value["recommended_applied_changes"] = value["recommended_applied_changes"][:1]
    path.write_text(json.dumps(value), encoding="utf-8")

    with pytest.raises(ValueError, match="differ from the applied overrides"):
        audit._adjudication(
            path,
            raw_sha256="a" * 64,
            overrides=_overrides(),
            raw_metrics=_metrics(inconsistent=2, disagreements=4),
            curated_metrics=_metrics(inconsistent=1, disagreements=2),
        )
