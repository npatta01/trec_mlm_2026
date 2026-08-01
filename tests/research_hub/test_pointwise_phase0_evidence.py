from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import shutil

import pytest

from trec_mlm.experiment_ledger import LedgerBuild
from trec_mlm.research_hub.build import build_research_hub
from trec_mlm.research_hub.load import parse_qmd_record
from trec_mlm.research_hub.pointwise_phase0 import load_pointwise_phase0_evidence
from trec_mlm.research_hub.render import (
    assert_hub_outputs_current,
    render_study_detail_pages,
    write_hub_outputs,
)


ROOT = Path(__file__).parents[2]
STUDY_PATH = ROOT / "research/hub/studies/pointwise-ranking-phase0.qmd"
EVIDENCE_PATHS = (
    "reports/trec_2026_pointwise_ranking_phase0_results.json",
    "reports/trec_2026_pointwise_ranking_phase0_per_query.tsv",
    "reports/trec_2026_pointwise_ranking_phase0_candidate_slices.tsv",
    "reports/trec_2026_dense_query_knn_results.json",
    "reports/trec_2026_dense_query_knn_per_query.tsv",
    "data/dev/llm_dev_data.tsv",
    "data/dev/llm_dev_qrels.txt",
)


def _copy_evidence(tmp_path: Path) -> None:
    for relative in EVIDENCE_PATHS:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _with_source_hash(
    ledger: LedgerBuild, source_id: str, sha256: str
) -> LedgerBuild:
    return replace(
        ledger,
        sources=tuple(
            {**source, "sha256": sha256}
            if source.get("source_id") == source_id
            else source
            for source in ledger.sources
        ),
    )


def test_pointwise_phase0_authored_qmd_contains_no_copied_quantitative_evidence() -> None:
    study = parse_qmd_record(ROOT, STUDY_PATH, "study")
    authored = "\n".join(
        line
        for line in study.body.splitlines()
        if not line.strip().startswith("{{< include ")
    )

    assert not any(line.lstrip().startswith("|") for line in authored.splitlines())
    assert re.search(r"(?<![\w.])(?:0?\.\d{4,}|−\.\d{4,})", authored) is None
    assert re.search(r"\b[a-f0-9]{40,}\b", authored) is None
    assert "W/T/L" not in authored
    assert "Paired 95% CI" not in authored
    assert "Zero-nDCG queries" not in authored


def test_pointwise_phase0_structured_evidence_drives_generated_study() -> None:
    build = build_research_hub(ROOT)

    assert build.pointwise_phase0 is not None
    assert [system.source_system_id for system in build.pointwise_phase0.systems] == [
        "control",
        "prior_only",
        "routing_only",
        "pointwise_only",
        "combined",
    ]
    assert len(build.pointwise_phase0.paired_rows) == 4
    assert len(build.pointwise_phase0.query_slices) == 9
    assert len(build.pointwise_phase0.candidate_slices) == 8
    assert build.pointwise_phase0.passed_gate_count == 1

    detail = render_study_detail_pages(build)[
        Path("research/hub/generated/studies/pointwise-ranking-phase0.md")
    ]
    assert "**0.118825**" in detail
    assert "**0.359855**" in detail
    assert "[−0.183811, −0.114267]" in detail
    assert ".114713 / .145330 / 246" in detail
    assert "**Gate: 1/5 passed." in detail
    assert "`stop_pairwise_work`" in detail
    assert (
        "The qrel-free construction preflight enforced and confirmed zero "
        "normalized-text and ID overlap between all 322 development queries "
        "and 6,816 pointwise source questions." in detail
    )
    assert (
        "Independent validation reconciled the sanitized construction and "
        "selector receipts." in detail
    )
    assert "Independent validation confirmed zero normalized-text" not in detail


def test_pointwise_phase0_rejects_candidate_membership_drift(
    tmp_path: Path,
) -> None:
    build = build_research_hub(ROOT)
    _copy_evidence(tmp_path)
    results_path = tmp_path / EVIDENCE_PATHS[0]
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["candidate_slices"][0]["candidate_ids"][0] = "drifted-candidate"
    results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger = _with_source_hash(
        build.ledger, "pointwise_phase0_results", _sha256(results_path)
    )

    with pytest.raises(ValueError, match="candidate slice.*membership"):
        load_pointwise_phase0_evidence(tmp_path, build.sources, ledger)


def test_pointwise_phase0_rejects_results_json_numeric_drift(
    tmp_path: Path,
) -> None:
    build = build_research_hub(ROOT)
    _copy_evidence(tmp_path)
    results_path = tmp_path / EVIDENCE_PATHS[0]
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["systems"]["combined"]["ndcg_at_10"] += 0.001
    results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger = _with_source_hash(
        build.ledger, "pointwise_phase0_results", _sha256(results_path)
    )

    with pytest.raises(ValueError, match="combined aggregate"):
        load_pointwise_phase0_evidence(tmp_path, build.sources, ledger)


def test_pointwise_phase0_rejects_baseline_reproduction_numeric_drift(
    tmp_path: Path,
) -> None:
    build = build_research_hub(ROOT)
    _copy_evidence(tmp_path)
    results_path = tmp_path / EVIDENCE_PATHS[0]
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["baseline_reproduction"]["ndcg_at_10"] += 0.001
    results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger = _with_source_hash(
        build.ledger, "pointwise_phase0_results", _sha256(results_path)
    )

    with pytest.raises(ValueError, match="baseline reproduction metrics"):
        load_pointwise_phase0_evidence(tmp_path, build.sources, ledger)


def test_pointwise_phase0_rejects_per_query_tsv_numeric_drift(
    tmp_path: Path,
) -> None:
    build = build_research_hub(ROOT)
    _copy_evidence(tmp_path)
    per_query_path = tmp_path / EVIDENCE_PATHS[1]
    lines = per_query_path.read_text(encoding="utf-8").splitlines()
    columns = lines[0].split("\t")
    row = lines[1].split("\t")
    metric_index = columns.index("combined_ndcg_at_10")
    row[metric_index] = str(float(row[metric_index]) + 0.001)
    lines[1] = "\t".join(row)
    per_query_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ledger = _with_source_hash(
        build.ledger, "pointwise_phase0_per_query", _sha256(per_query_path)
    )

    with pytest.raises(ValueError, match="tracked TSVs"):
        load_pointwise_phase0_evidence(tmp_path, build.sources, ledger)


def test_pointwise_phase0_rejects_candidate_tsv_numeric_drift(
    tmp_path: Path,
) -> None:
    build = build_research_hub(ROOT)
    _copy_evidence(tmp_path)
    candidate_path = tmp_path / EVIDENCE_PATHS[2]
    lines = candidate_path.read_text(encoding="utf-8").splitlines()
    columns = lines[0].split("\t")
    row = lines[1].split("\t")
    metric_index = columns.index("ndcg_at_10")
    row[metric_index] = str(float(row[metric_index]) + 0.001)
    lines[1] = "\t".join(row)
    candidate_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    results_path = tmp_path / EVIDENCE_PATHS[0]
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    payload["independent_validation"]["provenance"][
        "candidate_slices_evaluator_sha256"
    ] = _sha256(candidate_path)
    results_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    ledger = _with_source_hash(
        build.ledger, "pointwise_phase0_results", _sha256(results_path)
    )

    with pytest.raises(ValueError, match="JSON/TSV drift"):
        load_pointwise_phase0_evidence(tmp_path, build.sources, ledger)


def test_pointwise_phase0_rejects_ledger_aggregate_drift() -> None:
    build = build_research_hub(ROOT)
    mutated_ledger = replace(
        build.ledger,
        systems=tuple(
            replace(system, ndcg_at_10=system.ndcg_at_10 + 0.001)
            if system.system_id == "pointwise_phase0_combined"
            and system.ndcg_at_10 is not None
            else system
            for system in build.ledger.systems
        ),
    )

    with pytest.raises(ValueError, match="ledger nDCG"):
        load_pointwise_phase0_evidence(ROOT, build.sources, mutated_ledger)


def test_pointwise_phase0_evidence_change_updates_output_and_check_fails(
    tmp_path: Path,
) -> None:
    build = build_research_hub(ROOT)
    assert build.pointwise_phase0 is not None
    write_hub_outputs(build, tmp_path)
    mutated_systems = tuple(
        replace(
            system,
            metrics=replace(
                system.metrics, ndcg_at_10=system.metrics.ndcg_at_10 + 0.001
            ),
        )
        if system.source_system_id == "combined"
        else system
        for system in build.pointwise_phase0.systems
    )
    mutated = replace(
        build,
        pointwise_phase0=replace(
            build.pointwise_phase0,
            systems=mutated_systems,
        ),
    )
    path = Path("research/hub/generated/studies/pointwise-ranking-phase0.md")

    assert render_study_detail_pages(mutated)[path] != render_study_detail_pages(build)[path]
    with pytest.raises(ValueError, match=r"different: .*pointwise-ranking-phase0\.md"):
        assert_hub_outputs_current(mutated, tmp_path)
