import json
from pathlib import Path
import stat

import pytest

import scripts.build_pointwise_judge_smoke as smoke_cli
from scripts.build_pointwise_judge_smoke import main


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


def test_cli_fails_if_a_source_changes_during_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    real_hash = smoke_cli._sha256_file
    calls = 0

    def mutating_hash(path: Path) -> str:
        nonlocal calls
        calls += 1
        if calls == 3:
            discovery.write_text(
                discovery.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )
        return real_hash(path)

    monkeypatch.setattr(smoke_cli, "_sha256_file", mutating_hash)

    with pytest.raises(ValueError, match="source changed during selection"):
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
