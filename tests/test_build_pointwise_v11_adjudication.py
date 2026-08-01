from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat

import pytest

from scripts import build_pointwise_v11_adjudication as adjudication_module
from scripts.build_pointwise_v11_adjudication import (
    build_adjudication_manifest,
    provider_message_fixture,
    sanitize_historical_response,
    write_private_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
MERGED_RESULT = ROOT / (
    "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-500/result.json"
)
ORIGINAL_RESULT = ROOT / (
    "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-100/result.json"
)
RUN_PLAN = ROOT / (
    "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-500/run-plan.json"
)
OVERRIDES = ROOT / "protocols/pointwise_judge_bakeoff/v11-adjudication-overrides.json"
AUDIT_REPORTS = (
    ROOT / "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-100/audit.md",
    ROOT
    / ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-a.md",
    ROOT
    / ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-b.md",
    ROOT
    / ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-a.md",
    ROOT
    / ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-b.md",
    ROOT
    / ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-c.md",
)
EXPECTED_SOURCE_HASHES = {
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
EXPECTED_LEDGER_SHA256 = (
    "5da969289173b7406a2168fda9358f9dfb623a1b188a37d786560f32dc22ceaa"
)
EXPECTED_TRANSFORM_VERSION = "v11-historical-response-citation-markers-1"
EXPECTED_TRANSFORM_SHA256 = (
    "caf42f2e23a8bdfffbd26803d51a93866c0b074d01b16653b9baf70e9d695adf"
)
GRADE_2_CONTROLS = {
    1,
    22,
    37,
    56,
    86,
    100,
    127,
    138,
    153,
    164,
    165,
    177,
    191,
    209,
    235,
    248,
    266,
    287,
    309,
    325,
    332,
    347,
    368,
    394,
    421,
    441,
    454,
    470,
    485,
    498,
}
BARE_IDKS = {55, 57, 58, 60}


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _build() -> dict[str, object]:
    return build_adjudication_manifest(
        merged_result=MERGED_RESULT,
        original_result=ORIGINAL_RESULT,
        run_plan=RUN_PLAN,
        audit_reports=AUDIT_REPORTS,
        overrides_path=OVERRIDES,
        root=ROOT,
    )


def _copy_frozen_inputs(destination_root: Path) -> None:
    for relative in (*EXPECTED_SOURCE_HASHES, OVERRIDES.relative_to(ROOT).as_posix()):
        source = ROOT / relative
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def _build_under(root: Path) -> dict[str, object]:
    return build_adjudication_manifest(
        merged_result=MERGED_RESULT.relative_to(ROOT),
        original_result=ORIGINAL_RESULT.relative_to(ROOT),
        run_plan=RUN_PLAN.relative_to(ROOT),
        audit_reports=tuple(path.relative_to(ROOT) for path in AUDIT_REPORTS),
        overrides_path=OVERRIDES.relative_to(ROOT),
        root=root,
    )


def test_manifest_selects_the_four_reviewed_cohorts_exactly_once() -> None:
    manifest = _build()
    merged = _load(MERGED_RESULT)
    original = _load(ORIGINAL_RESULT)
    run_plan = _load(RUN_PLAN)
    merged_by_ordinal = {row["ordinal"]: row for row in merged["records"]}
    original_ordinals = {row["ordinal"] for row in original["records"]}
    new_ordinals = {
        ordinal for batch in run_plan["batches"] for ordinal in batch["new_ordinals"]
    }
    new_low = {
        ordinal
        for ordinal in new_ordinals
        if merged_by_ordinal[ordinal]["normalized"]["decision"]["grade"] < 2
    }

    ordinals = [row["ordinal"] for row in manifest["records"]]
    assert len(original_ordinals) == 100
    assert len(new_low) == 148
    assert len(GRADE_2_CONTROLS) == 30
    assert len(ordinals) == len(set(ordinals)) == 282
    assert set(ordinals) == original_ordinals | new_low | GRADE_2_CONTROLS | BARE_IDKS
    assert manifest["selection"]["original_100_ordinals"] == sorted(original_ordinals)
    assert manifest["selection"]["new_low_grade_ordinals"] == sorted(new_low)
    assert manifest["selection"]["grade_2_control_ordinals"] == sorted(GRADE_2_CONTROLS)
    assert manifest["selection"]["bare_idk_ordinals"] == sorted(BARE_IDKS)


def test_manifest_applies_only_102_explicit_changes_and_has_frozen_gold_totals() -> (
    None
):
    manifest = _build()
    ledger = _load(OVERRIDES)
    records = {row["ordinal"]: row for row in manifest["records"]}

    assert len(ledger["overrides"]) == 102
    assert len({row["ordinal"] for row in ledger["overrides"]}) == 102
    assert manifest["override_count"] == 102
    assert Counter(row["expected_numeric_grade"] for row in records.values()) == {
        2: 174,
        1: 66,
        0: 42,
    }
    assert manifest["gold_numeric_grade_totals"] == {"0": 42, "1": 66, "2": 174}
    assert records[201]["reviewed_v10_pair"] == {
        "grade": 1,
        "reason": "reference_silent",
    }
    assert records[201]["expected_v11_derived_label"] == "not_established"
    assert records[201]["override_source_report"].endswith(
        "deepinfra-v10-thinking-pilot-100/audit.md"
    )


def test_every_case_has_content_labels_buckets_and_hash_bound_provenance() -> None:
    manifest = _build()
    allowed_labels = {
        "correct",
        "partial",
        "not_established",
        "incorrect",
        "nonanswer",
    }
    allowed_tiers = {"search_grounded", "generated_without_search"}

    for row in manifest["records"]:
        assert type(row["ordinal"]) is int
        assert row["question"]
        assert row["evaluator_evidence"]
        assert row["historical_response"]
        assert row["expected_v11_derived_label"] in allowed_labels
        assert row["evidence_tier"] in allowed_tiers
        assert row["response_length_bucket"] in {"short", "long"}
        assert row["historical_response_chars"] == len(row["historical_response"])
        assert (
            row["provenance"]["merged_result_sha256"]
            == (
                EXPECTED_SOURCE_HASHES[
                    "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-500/result.json"
                ]
            )
        )
        if row["ordinal"] in BARE_IDKS:
            assert row["provenance"]["audit_report_sha256"] == []
            assert (
                row["provenance"]["deterministic_source_sha256"]
                == (
                    EXPECTED_SOURCE_HASHES[
                        "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-500/run-plan.json"
                    ]
                )
            )
        else:
            assert row["provenance"]["audit_report_sha256"]
            assert row["provenance"]["deterministic_source_sha256"] is None
        assert all(
            len(value) == 64 for value in row["provenance"]["audit_report_sha256"]
        )


def test_manifest_binds_every_frozen_input_by_sha256() -> None:
    manifest = _build()
    bound = {row["path"]: row["sha256"] for row in manifest["sources"]}

    assert EXPECTED_SOURCE_HASHES.items() <= bound.items()
    assert bound[
        "protocols/pointwise_judge_bakeoff/v11-adjudication-overrides.json"
    ] == (EXPECTED_LEDGER_SHA256)


def test_builder_fails_closed_when_an_approved_source_or_ledger_changes(
    tmp_path: Path,
) -> None:
    _copy_frozen_inputs(tmp_path)
    changed_report = tmp_path / AUDIT_REPORTS[-1].relative_to(ROOT)
    changed_report.write_text(
        changed_report.read_text(encoding="utf-8") + "\nchanged\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="approved source hash mismatch"):
        _build_under(tmp_path)

    shutil.copy2(AUDIT_REPORTS[-1], changed_report)
    changed_ledger = tmp_path / OVERRIDES.relative_to(ROOT)
    changed_ledger.write_text(
        changed_ledger.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="approved ledger hash mismatch"):
        _build_under(tmp_path)


def test_review_certificate_locks_per_report_changes_and_controls() -> None:
    ledger = _load(OVERRIDES)
    changes = Counter(row["source_report"] for row in ledger["overrides"])
    controls = Counter(row["source_report"] for row in ledger["grade_2_controls"])

    assert changes == {
        "artifacts/pointwise_production/deepinfra-v10-thinking-pilot-100/audit.md": 1,
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-a.md": 8,
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-b.md": 12,
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-a.md": 36,
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-b.md": 19,
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-c.md": 26,
    }
    assert controls == {
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-a.md": 10,
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-b.md": 10,
        ".superpowers/sdd/2026-07-30-simplified-pointwise-v10-pilot/semantic-audit-500-c.md": 10,
    }


def test_override_outside_audited_cohorts_fails_even_with_a_new_ledger_certificate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _copy_frozen_inputs(tmp_path)
    ledger_path = tmp_path / OVERRIDES.relative_to(ROOT)
    ledger = _load(ledger_path)
    ledger["overrides"][0] = {
        "ordinal": 11,
        "preferred_pair": "1/mixed",
        "source_report": ledger["overrides"][0]["source_report"],
    }
    ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        adjudication_module,
        "EXPECTED_LEDGER_SHA256",
        hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(ValueError, match="audited selected cohorts"):
        _build_under(tmp_path)


def test_selection_metadata_names_controls_and_deterministic_idks_explicitly() -> None:
    records = {row["ordinal"]: row for row in _build()["records"]}

    assert {
        ordinal
        for ordinal, row in records.items()
        if "audited_grade_2_control" in row["selection_reasons"]
    } == GRADE_2_CONTROLS
    assert {
        ordinal
        for ordinal, row in records.items()
        if "bare_idk_deterministic" in row["selection_reasons"]
    } == BARE_IDKS
    assert all(
        records[ordinal]["deterministic_no_provider_call"] for ordinal in BARE_IDKS
    )
    assert all(
        records[ordinal]["expected_v11_derived_label"] == "nonanswer"
        for ordinal in BARE_IDKS
    )


def test_citation_transform_has_exact_reviewed_outputs_for_the_four_changes() -> None:
    source = {row["ordinal"]: row for row in _load(MERGED_RESULT)["records"]}
    expected = {
        70: "Based on the information provided, Hindus do believe in evolution, at least in a broad sense. Several key points support this:\n\n1. The 2009 Harris Poll found that 80% of American Jews and Hindus accepted evolution, compared to lower percentages for other religious groups.\n\n2. Hinduism is one of the six primary religions of India and has evolved over centuries. It has moved from earlier periods where there was no restriction on animal worship to later eras that adopted concepts of non-violence (ahimsa) and respect for all living creatures",
        76: "Based on the information provided, Hindus do not have a unified belief regarding evolution. Some aspects to consider:\n\n1. The concept of eternal recurrence and cyclic time from Hindu philosophy is different from the scientific theory of evolution.\n\n2. Some Hindu texts like the Bhagavata Purana mention solar months and zodiac signs, which shows awareness of astronomical concepts but not necessarily biological evolution.\n\n3. The Chutia people were Hinduized but there's no mention of them adopting the theory of evolution.\n\n4. A",
        171: "Life on a slave ship was harsh and dehumanizing.\n\n- Slaves were often packed tightly below deck, with little room to move.\n- Conditions were unsanitary, leading to disease spread.\n- Slaves were not provided with adequate food, clothing, or lodging.\n- They were subject to cruel punishments and even execution for minor infractions.\n- Some were allowed to work for",
        353: "According to the NIST report, the fires weakened the trusses supporting the floors, making them sag. The sagging floors pulled on the exterior steel columns to the point where exterior columns bowed inward. With the damage to the core columns, the buckling exterior columns could no longer support the buildings, causing them to collapse. Additionally, the report asserts that the stairwells were not adequately reinforced to provide emergency escape for people above the impact zones.",
    }

    assert {
        ordinal: sanitize_historical_response(source[ordinal]["candidate_response"])
        for ordinal in expected
    } == expected


def test_manifest_binds_transform_and_changes_only_the_four_cited_responses() -> None:
    source = {row["ordinal"]: row for row in _load(MERGED_RESULT)["records"]}
    manifest = _build()
    records = {row["ordinal"]: row for row in manifest["records"]}

    assert manifest["historical_response_transform"] == {
        "version": EXPECTED_TRANSFORM_VERSION,
        "sha256": EXPECTED_TRANSFORM_SHA256,
    }
    assert {
        ordinal
        for ordinal, row in records.items()
        if row["historical_response"] != source[ordinal]["candidate_response"]
    } == {70, 76, 171, 353}


def test_all_provider_fixtures_exclude_citations_urls_and_metadata() -> None:
    source = {row["ordinal"]: row for row in _load(MERGED_RESULT)["records"]}
    forbidden_keys = {
        "ordinal",
        "record_id",
        "question_id",
        "candidate_id",
        "response_id",
        "provider_model",
        "source_ordinal",
        "reference_tier",
        "selection_reason",
        "audit_provenance",
    }
    for case in _build()["records"]:
        fixture = provider_message_fixture(case)
        rendered = json.dumps(fixture, sort_keys=True)
        source_record = source[case["ordinal"]]

        assert set(fixture) == {
            "question",
            "evaluator_evidence",
            "historical_response",
        }
        assert set(fixture).isdisjoint(forbidden_keys)
        assert re.search(r"\[[0-9]+\]", rendered) is None
        assert re.search(r"(?i)https?://|www\.", rendered) is None
        for field in ("record_id", "question_id", "response_id", "provider_model"):
            assert source_record[field] not in rendered


def test_median_and_bucket_boundary_use_transformed_response_lengths() -> None:
    manifest = _build()
    records = {row["ordinal"]: row for row in manifest["records"]}

    assert manifest["response_length_median_chars"] == 620.5
    assert records[406]["historical_response_chars"] == 620
    assert records[406]["response_length_bucket"] == "short"
    assert records[349]["historical_response_chars"] == 621
    assert records[349]["response_length_bucket"] == "long"
    assert all(
        (row["response_length_bucket"] == "short")
        == (row["historical_response_chars"] <= 620.5)
        for row in records.values()
    )


def test_private_writer_uses_0700_directories_and_0600_manifest(tmp_path: Path) -> None:
    bakeoff_root = tmp_path / "artifacts" / "pointwise_judge_bakeoff"
    bakeoff_root.mkdir(parents=True, mode=0o755)
    bakeoff_root.chmod(0o755)
    destination = bakeoff_root / "v11" / "nested" / "manifest.json"

    write_private_manifest(destination, _build(), root=tmp_path)

    assert stat.S_IMODE(bakeoff_root.stat().st_mode) == 0o755
    assert stat.S_IMODE(destination.parent.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert json.loads(destination.read_text(encoding="utf-8"))["record_count"] == 282


def test_private_writer_rejects_outside_path_without_mutating_anything(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "unrelated"
    outside.mkdir(mode=0o755)
    outside.chmod(0o755)
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    sentinel.chmod(0o644)
    destination = outside / "manifest.json"

    with pytest.raises(ValueError, match="authorized v11 artifact root"):
        write_private_manifest(destination, _build(), root=tmp_path)

    assert not destination.exists()
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert stat.S_IMODE(sentinel.stat().st_mode) == 0o644
    assert stat.S_IMODE(outside.stat().st_mode) == 0o755


def test_private_writer_rejects_symlinked_authorized_root_without_mutation(
    tmp_path: Path,
) -> None:
    outside = tmp_path / "unrelated"
    outside.mkdir(mode=0o755)
    outside.chmod(0o755)
    sentinel = outside / "sentinel.txt"
    sentinel.write_text("unchanged", encoding="utf-8")
    sentinel.chmod(0o644)
    authorized_parent = tmp_path / "artifacts" / "pointwise_judge_bakeoff"
    authorized_parent.mkdir(parents=True)
    (authorized_parent / "v11").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="real repository directories"):
        write_private_manifest(
            authorized_parent / "v11" / "manifest.json", _build(), root=tmp_path
        )

    assert not (outside / "manifest.json").exists()
    assert sentinel.read_text(encoding="utf-8") == "unchanged"
    assert stat.S_IMODE(sentinel.stat().st_mode) == 0o644
    assert stat.S_IMODE(outside.stat().st_mode) == 0o755
