from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from trec_mlm.profile_inference_artifact import validate_profile_smoke_receipt


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv" / "bin" / "python"


def _load_script(name: str) -> ModuleType:
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    options: set[str] = set()
    pending = [parser]
    while pending:
        current = pending.pop()
        for action in current._actions:
            options.update(action.option_strings)
            if isinstance(action, argparse._SubParsersAction):
                pending.extend(action.choices.values())
    return options


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PYTHON), str(ROOT / "scripts" / script), *args],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _individual_smoke(root: Path, execution_id: str, offset: float = 0.0) -> None:
    root.mkdir()
    values = np.zeros((4, 4096), dtype=np.float32)
    values[:, 0] = 1.0
    values[:, 1] = offset
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    np.save(root / "embeddings.npy", values, allow_pickle=False)
    _write_json(
        root / "receipt.json",
        {
            "artifact_type": "profile_inference_smoke",
            "version": 1,
            "execution_id": execution_id,
            "qrel_free": True,
            "rankings_or_qrels_used": False,
            "device_architecture": "gfx1151",
            "container_image": "vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28",
            "model_id": "intfloat/e5-mistral-7b-instruct",
            "model_revision": "d62cc4417be0e3289c03dd459d1ef2d137e2554e",
            "batch_size": 8,
            "dtype": "bfloat16",
            "persisted_dtype": "float32",
            "smoke_inventory": ["short", "median", "over_512"],
            "selected_record_ids": ["short-id", "median-id", "long-id"],
            "input_derivations": [
                {
                    "label": "short",
                    "source_record_id": "short-id",
                    "transform": "identity",
                    "response_repetitions": 1,
                    "response_separator": "",
                },
                {
                    "label": "median",
                    "source_record_id": "median-id",
                    "transform": "identity",
                    "response_repetitions": 1,
                    "response_separator": "",
                },
                {
                    "label": "over_512",
                    "source_record_id": "long-id",
                    "transform": "repeat_response",
                    "response_repetitions": 3,
                    "response_separator": "\n\n",
                },
            ],
            "embedding_shape": [4, 4096],
            "embedding_sha256": hashlib.sha256(
                (root / "embeddings.npy").read_bytes()
            ).hexdigest(),
            "runtime_metadata": {
                "backend": "transformers_rocm",
                "model_id": "intfloat/e5-mistral-7b-instruct",
                "requested_revision": "d62cc4417be0e3289c03dd459d1ef2d137e2554e",
                "resolved_model_revision": "d62cc4417be0e3289c03dd459d1ef2d137e2554e",
                "resolved_tokenizer_revision": "d62cc4417be0e3289c03dd459d1ef2d137e2554e",
                "required_container_image": "vllm/vllm-openai-rocm@sha256:3832d79d9e514ce2e072580689da078726454596d833c8ab803f29f3cea5ea28",
                "dtype": "bfloat16",
                "persisted_dtype": "float32",
                "license": "MIT",
                "pooling": "last_token_normalized",
                "embedding_dimension": 4096,
                "input_count": 4,
                "input_sha256": "c" * 64,
                "batch_size": 8,
                "seed": 2026,
                "padding_side": "left",
                "truncation_side": "right",
                "terminal_token_id": 2,
                "terminal_token": "</s>",
                "max_length": 512,
                "maximum_observed_tokens": 512,
                "maximum_untruncated_tokens": 700,
                "truncated_input_count": 1,
                "document_response_truncation": ["none", "none", "partial"],
                "parameter_count": 7_000_000_000,
                "peak_memory_allocated_bytes": 1,
                "peak_memory_reserved_bytes": 1,
                "device": "rocm_hip",
                "device_properties": "AMD Radeon gfx1151",
                "torch_version": "2.11.0+gitd0c8b1f",
                "hip_version": "7.2.53211",
                "transformers_version": "5.12.1",
                "encoding_seconds": 1.0,
            },
        },
    )


def test_construction_parser_has_only_frozen_operations_and_options() -> None:
    module = _load_script("run_raw_response_refinement.py")
    parser = module._parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == {"construct", "validate", "compare"}
    forbidden = ("qrel", "top-k", "weight", "encoder", "fixture")
    assert not any(
        token in option
        for option in _option_strings(parser)
        for token in forbidden
    )


def test_only_evaluation_parser_accepts_qrels() -> None:
    construction = _load_script("run_raw_response_refinement.py")._parser()
    smoke = _load_script("run_profile_inference_smoke.py")._parser()
    evaluation = _load_script("evaluate_raw_response_refinement.py")._parser()
    assert "--qrels" not in _option_strings(construction)
    assert "--qrels" not in _option_strings(smoke)
    assert "--qrels" in _option_strings(evaluation)
    assert not any(
        token in option
        for option in _option_strings(evaluation)
        for token in ("top-k", "weight", "encoder", "fixture")
    )


@pytest.mark.parametrize(
    ("script", "arguments"),
    [
        ("run_raw_response_refinement.py", ("validate", "--construction", "missing")),
        (
            "run_raw_response_refinement.py",
            ("compare", "--first", "missing-1", "--second", "missing-2", "--receipt", "x.json"),
        ),
        (
            "evaluate_raw_response_refinement.py",
            ("--construction", "missing", "--repeat-receipt", "missing.json", "--qrels", "missing.qrels", "--output-dir", "evaluation"),
        ),
        ("build_raw_response_refinement_report.py", ("--evaluation", "missing.json", "--markdown", "a.md", "--json", "a.json")),
    ],
)
def test_cli_invalid_inputs_are_rejected(script: str, arguments: tuple[str, ...]) -> None:
    completed = _run(script, *arguments)
    assert completed.returncode != 0


def test_construction_help_documents_dense_extra() -> None:
    completed = _run("run_raw_response_refinement.py", "--help")
    assert completed.returncode == 0
    assert "uv run --extra dense" in completed.stdout


def test_smoke_parser_is_frozen_and_selection_is_deterministic() -> None:
    module = _load_script("run_profile_inference_smoke.py")
    parser = module._parser()
    subparsers = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparsers.choices) == {"run", "compare"}
    assert not any(
        token in option
        for option in _option_strings(parser)
        for token in ("qrel", "batch", "seed", "encoder", "model", "device", "fixture")
    )
    records = [
        SimpleNamespace(record_id="b", question="question", response="one two"),
        SimpleNamespace(record_id="a", question="question", response="one two"),
        SimpleNamespace(record_id="c", question="question", response=" ".join(["word"] * 30)),
        SimpleNamespace(record_id="d", question="question", response=" ".join(["word"] * 700)),
    ]
    first = module._select_smoke_records(records)
    second = module._select_smoke_records(reversed(records))
    assert [item[0] for item in first] == ["short", "median", "over_512"]
    assert [item[1].record_id for item in first] == ["a", "c", "d"]
    assert [item[1].record_id for item in first] == [item[1].record_id for item in second]


def test_smoke_derives_overlength_input_when_no_raw_record_exceeds_512(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_script("run_profile_inference_smoke.py")
    records = (
        SimpleNamespace(record_id="short", question="q", response="one two"),
        SimpleNamespace(
            record_id="median", question="q", response=" ".join(["word"] * 20)
        ),
        SimpleNamespace(
            record_id="longest", question="q", response=" ".join(["word"] * 200)
        ),
    )
    captured: dict[str, tuple[str, ...]] = {}

    def encode(**kwargs):
        captured["documents"] = tuple(kwargs["formatted_documents"])
        query = np.zeros((1, 4096), dtype=np.float32)
        query[:, 0] = 1.0
        documents = np.zeros((3, 4096), dtype=np.float32)
        documents[:, 0] = 1.0
        return query, documents, {
            "document_response_truncation": ["none", "none", "partial"],
            "maximum_observed_tokens": 512,
            "maximum_untruncated_tokens": 600,
            "truncated_input_count": 1,
        }

    monkeypatch.setattr(module, "load_candidate_manifest", lambda _path: ("A",))
    monkeypatch.setattr(module, "load_discovery_records", lambda *_args: ())
    monkeypatch.setattr(
        module,
        "filter_and_deduplicate_records",
        lambda *_args: SimpleNamespace(records_by_candidate={"A": records}),
    )
    monkeypatch.setattr(module, "encode_profile_inputs", encode)
    output = tmp_path / "smoke"
    receipt = module._run_smoke("discovery.json", "candidates.json", str(output))
    expected_response = "\n\n".join([records[-1].response] * 3)
    assert captured["documents"][-1] == module.format_profile_document(
        records[-1].question, expected_response
    )
    assert len(captured["documents"][-1].split()) > 512
    assert receipt["selected_record_ids"][-1] == "longest"
    assert receipt["input_derivations"][-1] == {
        "label": "over_512",
        "source_record_id": "longest",
        "transform": "repeat_response",
        "response_repetitions": 3,
        "response_separator": "\n\n",
    }
    assert json.loads((output / "receipt.json").read_text(encoding="utf-8")) == receipt


def test_smoke_compare_emits_exact_valid_receipt_atomically(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _individual_smoke(first, "11111111-1111-4111-8111-111111111111")
    _individual_smoke(second, "22222222-2222-4222-8222-222222222222", 1e-6)
    destination = tmp_path / "repeat.json"
    completed = _run(
        "run_profile_inference_smoke.py",
        "compare",
        "--first",
        str(first),
        "--second",
        str(second),
        "--receipt",
        str(destination),
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(destination.read_text(encoding="utf-8"))
    assert set(payload) == {
        "status", "qrel_free", "rankings_or_qrels_used",
        "maximum_absolute_embedding_difference", "tolerance",
        "device_architecture", "container_image", "model_id", "model_revision",
        "batch_size", "dtype", "persisted_dtype", "smoke_inventory",
        "first_smoke_receipt_sha256", "second_smoke_receipt_sha256",
    }
    validate_profile_smoke_receipt(payload)
    assert _run(
        "run_profile_inference_smoke.py", "compare", "--first", str(first),
        "--second", str(second), "--receipt", str(destination)
    ).returncode != 0


def test_smoke_compare_rejects_two_matching_but_unpinned_receipts(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _individual_smoke(first, "11111111-1111-4111-8111-111111111111")
    _individual_smoke(second, "22222222-2222-4222-8222-222222222222")
    for root in (first, second):
        payload = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
        payload["model_id"] = "unpinned/model"
        _write_json(root / "receipt.json", payload)
    completed = _run(
        "run_profile_inference_smoke.py", "compare", "--first", str(first),
        "--second", str(second), "--receipt", str(tmp_path / "repeat.json")
    )
    assert completed.returncode != 0
    assert not (tmp_path / "repeat.json").exists()


def test_smoke_compare_binds_selected_inputs_and_pinned_runtime(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _individual_smoke(first, "11111111-1111-4111-8111-111111111111")
    _individual_smoke(second, "22222222-2222-4222-8222-222222222222")
    second_payload = json.loads(
        (second / "receipt.json").read_text(encoding="utf-8")
    )
    second_payload["selected_record_ids"][-1] = "different-long-id"
    second_payload["runtime_metadata"]["device_properties"] = "AMD Radeon gfx1100"
    _write_json(second / "receipt.json", second_payload)
    completed = _run(
        "run_profile_inference_smoke.py", "compare", "--first", str(first),
        "--second", str(second), "--receipt", str(tmp_path / "repeat.json")
    )
    assert completed.returncode != 0
    assert not (tmp_path / "repeat.json").exists()


def test_smoke_compare_rejects_input_derivation_tampering(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _individual_smoke(first, "11111111-1111-4111-8111-111111111111")
    _individual_smoke(second, "22222222-2222-4222-8222-222222222222")
    payload = json.loads((second / "receipt.json").read_text(encoding="utf-8"))
    payload["input_derivations"][-1]["response_repetitions"] = 4
    _write_json(second / "receipt.json", payload)
    completed = _run(
        "run_profile_inference_smoke.py", "compare", "--first", str(first),
        "--second", str(second), "--receipt", str(tmp_path / "repeat.json")
    )
    assert completed.returncode != 0
    assert "derivation" in completed.stderr
    assert not (tmp_path / "repeat.json").exists()


def test_smoke_compare_requires_exact_runtime_truncation_evidence(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _individual_smoke(first, "11111111-1111-4111-8111-111111111111")
    _individual_smoke(second, "22222222-2222-4222-8222-222222222222")
    for root in (first, second):
        payload = json.loads((root / "receipt.json").read_text(encoding="utf-8"))
        runtime = payload["runtime_metadata"]
        runtime["document_response_truncation"] = ["none", "none", "invented"]
        runtime["maximum_observed_tokens"] = 100
        runtime["maximum_untruncated_tokens"] = 100
        runtime["truncated_input_count"] = 0
        _write_json(root / "receipt.json", payload)
    completed = _run(
        "run_profile_inference_smoke.py", "compare", "--first", str(first),
        "--second", str(second), "--receipt", str(tmp_path / "repeat.json")
    )
    assert completed.returncode != 0
    assert "truncation" in completed.stderr
    assert not (tmp_path / "repeat.json").exists()


@pytest.mark.parametrize(
    "script",
    ["run_profile_inference_smoke.py", "evaluate_raw_response_refinement.py"],
)
def test_directory_publication_is_atomic_and_never_overwrites(
    tmp_path: Path, script: str
) -> None:
    module = _load_script(script)
    first = tmp_path / "first"
    first.mkdir()
    (first / "marker").write_text("first", encoding="utf-8")
    destination = tmp_path / "destination"
    module._rename_noreplace(first, destination)
    assert (destination / "marker").read_text(encoding="utf-8") == "first"
    second = tmp_path / "second"
    second.mkdir()
    (second / "marker").write_text("second", encoding="utf-8")
    with pytest.raises(FileExistsError):
        module._rename_noreplace(second, destination)
    assert (destination / "marker").read_text(encoding="utf-8") == "first"
    assert (second / "marker").read_text(encoding="utf-8") == "second"


def test_report_is_deterministic_and_labels_dev_evaluation_post_hoc(
    tmp_path: Path,
) -> None:
    evaluation = tmp_path / "evaluation.json"
    repeat_receipt = {
        "status": "passed",
        "selected_record_ids_identical": True,
        "rankings_identical": True,
        "maximum_absolute_score_difference": 0.0,
        "tolerance": 1e-12,
        "first_aggregate_sha256": "a" * 64,
        "second_aggregate_sha256": "c" * 64,
        "first_execution_id": "11111111-1111-4111-8111-111111111111",
        "second_execution_id": "22222222-2222-4222-8222-222222222222",
    }
    metric = {"ndcg_at_10": 0.43, "mrr": 0.45, "zero_ndcg_queries": 98}
    interval = {
        "mean_difference": 0.01,
        "lower": -0.01,
        "upper": 0.03,
        "confidence_level": 0.95,
        "n_resamples": 10_000,
        "seed": 2026,
        "generator": "PCG64",
    }

    def comparison(baseline: str, candidate: str) -> dict[str, object]:
        return {
            "baseline": baseline,
            "candidate": candidate,
            "ndcg_at_10_delta": interval,
            "mrr_delta": interval,
            "ndcg_at_10_win_tie_loss": {
                "wins": 100, "ties": 100, "losses": 122, "tolerance": 1e-12
            },
            "mrr_win_tie_loss": {
                "wins": 100, "ties": 100, "losses": 122, "tolerance": 1e-12
            },
            "zero_transitions": {
                "zero_both": 90,
                "dense_zero_candidate_nonzero": 9,
                "dense_nonzero_candidate_zero": 8,
                "nonzero_both": 215,
                "query_count": 322,
            },
        }

    payload = {
        "status": "evaluated",
        "construction_aggregate_sha256": "a" * 64,
        "construction_dependency_sha256": "d" * 64,
        "construction_unchanged": True,
        "qrels_sha256": "3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35",
        "repeat_receipt": repeat_receipt,
        "matched_parent_metrics": {
            "dense_profile_equal": {
                "ndcg_at_10": 0.4202139921941393,
                "mrr": 0.45301739378333744,
                "zero_ndcg_queries": 99,
            },
            "profile_only": {
                "ndcg_at_10": 0.41492445997856287,
                "mrr": 0.441697813928065,
                "zero_ndcg_queries": 92,
            },
        },
        "metrics": {
            "dense_reference": metric,
            "dense_profile_equal": {
                "ndcg_at_10": 0.4202139921941393,
                "mrr": 0.45301739378333744,
                "zero_ndcg_queries": 99,
            },
            "profile_only": {
                "ndcg_at_10": 0.41492445997856287,
                "mrr": 0.441697813928065,
                "zero_ndcg_queries": 92,
            },
            "raw_hierarchical": metric,
            "raw_hierarchical_dense_equal": metric,
        },
        "comparisons": {
            "primary": comparison(
                "dense_profile_equal", "raw_hierarchical_dense_equal"
            ),
            "secondary": comparison("profile_only", "raw_hierarchical"),
            "dense_diagnostics": {
                candidate: comparison("dense_reference", candidate)
                for candidate in (
                    "raw_hierarchical", "raw_hierarchical_dense_equal"
                )
            },
        },
        "per_query_rows": [
            {
                "query_id": f"q{query_index:03d}",
                "system": system,
                "ndcg_at_10": 0.5,
                "reciprocal_rank": 0.5,
            }
            for query_index in range(322)
            for system in (
                "dense_reference", "dense_profile_equal", "profile_only",
                "raw_hierarchical", "raw_hierarchical_dense_equal",
            )
        ],
        "top_10_changed": {"primary": True, "secondary": True},
        "decision": {
            "decision": "diagnostic",
            "gates": {
                "ndcg_interval_lower_above_zero": False,
                "mrr_not_materially_lower": True,
                "zero_count_reduced": True,
                "integrity_passes": True,
            },
            "integrity_checks": {
                "qrel_independence": True,
                "dependency_bindings": True,
                "embedding_contract": True,
                "inventory": True,
                "semantic_reconstruction": True,
                "trec_completeness": True,
                "artifact_validation": True,
                "repeatability": True,
                "parent_metrics": True,
                "construction_hash_binding": True,
            },
            "values": {
                "ndcg_interval_lower": -0.01,
                "ndcg_point_delta": 0.01,
                "mrr_delta": 0.01,
                "zero_count": 98,
                "zero_rescues": 9,
            },
        },
    }
    _write_json(evaluation, payload)
    markdown = tmp_path / "result.md"
    output_json = tmp_path / "result.json"
    completed = _run(
        "build_raw_response_refinement_report.py",
        "--evaluation", str(evaluation), "--markdown", str(markdown),
        "--json", str(output_json),
    )
    assert completed.returncode == 0, completed.stderr
    report = markdown.read_text(encoding="utf-8")
    assert "post-hoc confirmation" in report
    assert "not independent hidden-test evidence" in report
    json_report = json.loads(output_json.read_text(encoding="utf-8"))
    assert json_report["evaluation"] == payload
    assert "post-hoc confirmation" in json_report["evidence_scope"]
    assert "not independent hidden-test evidence" in json_report["evidence_scope"]
    assert _run(
        "build_raw_response_refinement_report.py", "--evaluation", str(evaluation),
        "--markdown", str(markdown), "--json", str(output_json)
    ).returncode != 0
    tampered = tmp_path / "tampered-evaluation.json"
    payload["decision"]["decision"] = "advance"
    _write_json(tampered, payload)
    completed = _run(
        "build_raw_response_refinement_report.py", "--evaluation", str(tampered),
        "--markdown", str(tmp_path / "tampered.md"),
        "--json", str(tmp_path / "tampered.json"),
    )
    assert completed.returncode != 0
    assert not (tmp_path / "tampered.md").exists()


def test_report_rejects_under_specified_or_unlocked_evaluation(tmp_path: Path) -> None:
    evaluation = tmp_path / "evaluation.json"
    _write_json(
        evaluation,
        {
            "status": "evaluated",
            "construction_aggregate_sha256": "a" * 64,
            "qrels_sha256": "b" * 64,
            "decision": {"decision": "advance"},
        },
    )
    completed = _run(
        "build_raw_response_refinement_report.py", "--evaluation", str(evaluation),
        "--markdown", str(tmp_path / "result.md"),
        "--json", str(tmp_path / "result.json"),
    )
    assert completed.returncode != 0
    assert not (tmp_path / "result.md").exists()
    assert not (tmp_path / "result.json").exists()
