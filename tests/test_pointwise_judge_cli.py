import json
from pathlib import Path
import stat

import pytest

import scripts.build_pointwise_judge_smoke as smoke_cli
from scripts.build_pointwise_judge_smoke import main
import scripts.run_local_pointwise_judge as runner_cli
from trec_mlm.pointwise_judge import CandidateCase


def _write_inputs(
    tmp_path: Path, *, canonical_cache: bool = False
) -> tuple[Path, Path]:
    grounded = tmp_path / "grounded.json"
    records = {}
    discovery_rows = {}
    candidates = [
        (
            "Who wrote Hamlet?",
            "William Shakespeare wrote Hamlet.",
            "William Shakespeare wrote Hamlet.",
        ),
        (
            "What is photosynthesis?",
            "Plants convert light into stored chemical energy.",
            "Plants convert light.",
        ),
        (
            "Where is the Eiffel Tower?",
            "The Eiffel Tower is in Paris, France.",
            "It is beneath the Pacific Ocean.",
        ),
        (
            "Who painted Guernica?",
            "Pablo Picasso painted Guernica.",
            "Salvador Dali painted Guernica.",
        ),
        (
            "When did Apollo 11 land?",
            "Apollo 11 landed in 1969.",
            "Apollo 11 landed in 1972.",
        ),
        (
            "Is Canberra Australia's capital?",
            "Canberra is Australia's capital.",
            "Canberra is not Australia's capital.",
        ),
        (
            "What is water?",
            "Water is H2O.",
            "Water is a familiar substance found throughout the world. " * 20,
        ),
        (
            "What is the speed of light?",
            "It is 299,792,458 metres per second.",
            "I don't know.",
        ),
        (
            "What happened next?",
            "The expedition returned safely.",
            "It happened later.",
        ),
        (
            "Who discovered penicillin?",
            "Alexander Fleming discovered penicillin.",
            "The premise may be wrong; Ernst Chain discovered penicillin.",
        ),
    ]
    for index, (question, reference, candidate) in enumerate(candidates):
        question_id = f"q{index}"
        records[question_id] = {
            "question_id": question_id,
            "answer": reference,
            "finish_reason": "stop",
        }
        if not canonical_cache:
            records[question_id]["question"] = question
        discovery_rows[question_id] = {
            "query": question,
            "responses": {f"PRIVATE-CANDIDATE-{index}": candidate},
        }
    grounded.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps(discovery_rows, indent=2), encoding="utf-8")
    return discovery, grounded


def test_cli_writes_blind_manifest_and_private_seal_atomically(tmp_path: Path) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    output = tmp_path / "smoke"
    args = [
        "--discovery",
        str(discovery),
        "--grounded",
        str(grounded),
        "--output-dir",
        str(output),
        "--seed",
        "2026",
    ]

    assert main(args) == 0
    manifest_text = (output / "smoke_manifest.json").read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    sealed = json.loads((output / "sealed_candidates.json").read_text(encoding="utf-8"))
    assert manifest["seed"] == 2026
    assert len(manifest["cases"]) == 10
    assert set(manifest["selection_reasons"]) == {
        case["selection_reason"] for case in manifest["cases"]
    }
    assert set(manifest["sources"]) == {"discovery", "grounded"}
    assert all(source["sha256"] for source in manifest["sources"].values())
    assert all(candidate not in manifest_text for candidate in sealed.values())
    assert stat.S_IMODE((output / "smoke_manifest.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((output / "sealed_candidates.json").stat().st_mode) == 0o600
    assert not list(output.glob("*.tmp"))

    assert main(args) == 0


def test_cli_rejects_existing_output_with_different_manifest(tmp_path: Path) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    output = tmp_path / "smoke"
    args = [
        "--discovery",
        str(discovery),
        "--grounded",
        str(grounded),
        "--output-dir",
        str(output),
    ]
    assert main(args) == 0
    discovery.write_text(
        discovery.read_text().replace(
            "beneath the Pacific Ocean", "beneath the Atlantic Ocean"
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="different manifest hash"):
        main(args)


def test_cli_supports_canonical_generation_records_without_questions(
    tmp_path: Path,
) -> None:
    discovery, grounded = _write_inputs(tmp_path, canonical_cache=True)

    assert (
        main(
            [
                "--discovery",
                str(discovery),
                "--grounded",
                str(grounded),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )
        == 0
    )


def test_cli_consumes_the_exact_snapshot_hashed_during_an_a_b_a_source_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    original = discovery.read_text(encoding="utf-8")
    transient = original.replace(
        "beneath the Pacific Ocean", "beneath the Atlantic Ocean"
    )
    real_hash = smoke_cli._sha256_file
    real_snapshot = getattr(smoke_cli, "_snapshot_source", None)
    discovery_hash_calls = 0

    def racing_hash(path: Path) -> str:
        nonlocal discovery_hash_calls
        digest = real_hash(path)
        if path == discovery:
            discovery_hash_calls += 1
            discovery.write_text(
                transient if discovery_hash_calls == 1 else original,
                encoding="utf-8",
            )
        return digest

    def racing_snapshot(path: Path, snapshot_dir: Path, name: str):
        assert real_snapshot is not None
        snapshot = real_snapshot(path, snapshot_dir, name)
        if path == discovery:
            discovery.write_text(transient, encoding="utf-8")
            discovery.write_text(original, encoding="utf-8")
        return snapshot

    monkeypatch.setattr(smoke_cli, "_sha256_file", racing_hash)
    monkeypatch.setattr(smoke_cli, "_snapshot_source", racing_snapshot, raising=False)

    output = tmp_path / "out"
    assert (
        main(
            [
                "--discovery",
                str(discovery),
                "--grounded",
                str(grounded),
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )

    manifest = json.loads((output / "smoke_manifest.json").read_text())
    candidates = {case["candidate_answer"] for case in manifest["cases"]}
    assert "It is beneath the Pacific Ocean." in candidates
    assert "It is beneath the Atlantic Ocean." not in candidates
    assert manifest["sources"]["discovery"]["sha256"] == real_hash(discovery)


def test_cli_rejects_tampered_existing_manifest(tmp_path: Path) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    output = tmp_path / "out"
    args = [
        "--discovery",
        str(discovery),
        "--grounded",
        str(grounded),
        "--output-dir",
        str(output),
    ]
    assert main(args) == 0
    path = output / "smoke_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["cases"][0]["question"] = "tampered"
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid existing manifest hash"):
        main(args)


def test_cli_rejects_corrupted_existing_seal(tmp_path: Path) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    output = tmp_path / "out"
    args = [
        "--discovery",
        str(discovery),
        "--grounded",
        str(grounded),
        "--output-dir",
        str(output),
    ]
    assert main(args) == 0
    path = output / "sealed_candidates.json"
    sealed = json.loads(path.read_text(encoding="utf-8"))
    sealed[next(iter(sealed))] = "corrupted-candidate"
    path.write_text(json.dumps(sealed), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ValueError, match="sealed candidates do not match"):
        main(args)


def test_cli_rejects_existing_seal_with_nonprivate_mode(tmp_path: Path) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    output = tmp_path / "out"
    args = [
        "--discovery",
        str(discovery),
        "--grounded",
        str(grounded),
        "--output-dir",
        str(output),
    ]
    assert main(args) == 0
    (output / "sealed_candidates.json").chmod(0o644)

    with pytest.raises(ValueError, match="mode 0600"):
        main(args)


def test_cli_rejects_existing_manifest_with_nonprivate_mode(tmp_path: Path) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    output = tmp_path / "out"
    args = [
        "--discovery",
        str(discovery),
        "--grounded",
        str(grounded),
        "--output-dir",
        str(output),
    ]
    assert main(args) == 0
    (output / "smoke_manifest.json").chmod(0o644)

    with pytest.raises(ValueError, match="smoke manifest must have mode 0600"):
        main(args)


def test_runner_cli_consumes_private_manifest_and_runtime_receipt_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    case = CandidateCase(
        question_id="q-private",
        response_id="r-blind",
        question="Who wrote Hamlet?",
        reference_answer="William Shakespeare wrote Hamlet.",
        candidate_answer="Shakespeare wrote Hamlet.",
        strata=("high_overlap_control",),
        selection_reason="high_overlap_control",
    )
    manifest = tmp_path / "smoke_manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "manifest_hash": "fixture-hash",
                "cases": [
                    {
                        "question_id": case.question_id,
                        "response_id": case.response_id,
                        "question": case.question,
                        "reference_answer": case.reference_answer,
                        "candidate_answer": case.candidate_answer,
                        "strata": list(case.strata),
                        "selection_reason": case.selection_reason,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    manifest.chmod(0o600)
    receipt = tmp_path / "runtime.json"
    receipt.write_text(
        json.dumps({"runtime": "fabricated", "passed": True}), encoding="utf-8"
    )
    receipt.chmod(0o600)
    protocol_dir = Path("protocols/pointwise_judge_bakeoff")
    output = tmp_path / "run.json"
    seen: list[tuple[str, str, object]] = []

    def fake_call(endpoint, model, messages, schema, *, timeout_seconds):
        seen.append((endpoint, model, messages))
        raw = json.dumps(
            {
                "label": "good",
                "correctness": 4,
                "completeness": 2,
                "relevance": 2,
                "error_flags": [],
                "reference_status": "sound",
                "confidence": "high",
                "rationale": "The response correctly identifies the author.",
            }
        )
        return raw, {
            "model": model,
            "choices": [{"message": {"content": raw}}],
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(runner_cli, "call_openai_judge", fake_call)

    assert (
        runner_cli.main(
            [
                "--manifest",
                str(manifest),
                "--protocol-dir",
                str(protocol_dir),
                "--model-alias",
                "judge-a",
                "--model-id",
                "model-a",
                "--base-url",
                "http://127.0.0.1:13305/v1",
                "--runtime-receipt",
                str(receipt),
                "--output",
                str(output),
                "--timeout-seconds",
                "12.5",
            ]
        )
        == 0
    )
    assert seen[0][0:2] == ("http://127.0.0.1:13305/v1", "model-a")
    assert case.question in seen[0][2][1]["content"]
    assert (
        json.loads(output.read_text(encoding="utf-8"))["records"]["r-blind"][
            "decision"
        ]["label"]
        == "good"
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
