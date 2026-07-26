from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

import scripts.build_pointwise_judge_100 as hundred_cli
from scripts.build_pointwise_judge_100 import main
from scripts.build_pointwise_judge_smoke import main as smoke_main
import scripts.run_local_pointwise_judge as runner_cli
from trec_mlm.pointwise_sampling import DIAGNOSTIC_REASONS


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    examples = [
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
    records: dict[str, object] = {}
    discovery_rows: dict[str, object] = {}
    for reason_index, (question, reference, candidate) in enumerate(examples):
        for copy_index in range(12):
            question_id = f"q-{reason_index}-{copy_index}"
            records[question_id] = {
                "question_id": question_id,
                "question": question,
                "answer": reference,
                "finish_reason": "stop",
            }
            discovery_rows[question_id] = {
                "query": question,
                "responses": {
                    f"PRIVATE-CANDIDATE-{reason_index}-{copy_index}": candidate
                },
            }
    discovery = tmp_path / "discovery.json"
    discovery.write_text(json.dumps(discovery_rows, indent=2), encoding="utf-8")
    grounded = tmp_path / "grounded.json"
    grounded.write_text(json.dumps({"records": records}, indent=2), encoding="utf-8")
    return discovery, grounded


def _build_smoke(tmp_path: Path, discovery: Path, grounded: Path) -> Path:
    smoke_dir = tmp_path / "smoke-v2"
    assert (
        smoke_main(
            [
                "--discovery",
                str(discovery),
                "--grounded",
                str(grounded),
                "--output-dir",
                str(smoke_dir),
                "--seed",
                "2026",
            ]
        )
        == 0
    )
    return smoke_dir / "smoke_manifest.json"


def _builder_args(
    discovery: Path, grounded: Path, smoke_manifest: Path, output: Path
) -> list[str]:
    return [
        "--discovery",
        str(discovery),
        "--grounded",
        str(grounded),
        "--smoke-manifest",
        str(smoke_manifest),
        "--output-dir",
        str(output),
        "--seed",
        "2026",
    ]


def _build_100(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path]:
    discovery, grounded = _write_inputs(tmp_path)
    smoke_path = _build_smoke(tmp_path, discovery, grounded)
    output = tmp_path / "local-100-v1"
    assert main(_builder_args(discovery, grounded, smoke_path, output)) == 0
    return discovery, grounded, smoke_path, output


def test_cli_builds_frozen_private_100_case_manifest_idempotently(
    tmp_path: Path,
) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    smoke_path = _build_smoke(tmp_path, discovery, grounded)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    output = tmp_path / "local-100-v1"
    args = _builder_args(discovery, grounded, smoke_path, output)

    assert main(args) == 0

    manifest_path = output / "manifest.json"
    sealed_path = output / "sealed_candidates.json"
    original_manifest_bytes = manifest_path.read_bytes()
    original_manifest_sha256 = _sha256_path(manifest_path)
    manifest = json.loads(original_manifest_bytes)
    sealed = json.loads(sealed_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "version",
        "seed",
        "sources",
        "selection_reasons",
        "cases_per_reason",
        "reused_smoke_response_ids",
        "cases",
        "manifest_hash",
    }
    assert manifest["version"] == 2
    assert manifest["seed"] == 2026
    assert manifest["selection_reasons"] == list(DIAGNOSTIC_REASONS)
    assert manifest["cases_per_reason"] == 10
    assert manifest["sources"] == {
        "discovery": {"sha256": _sha256_path(discovery)},
        "grounded": {"sha256": _sha256_path(grounded)},
        "smoke_manifest": {"sha256": _sha256_path(smoke_path)},
    }
    body = dict(manifest)
    stored_hash = body.pop("manifest_hash")
    assert stored_hash == _canonical_hash(body)

    cases = manifest["cases"]
    assert len(cases) == 100
    assert len({case["question_id"] for case in cases}) == 100
    assert len({case["response_id"] for case in cases}) == 100
    assert Counter(case["selection_reason"] for case in cases) == {
        reason: 10 for reason in DIAGNOSTIC_REASONS
    }
    assert set(sealed) == {case["response_id"] for case in cases}
    assert all(
        candidate not in original_manifest_bytes.decode()
        for candidate in sealed.values()
    )

    smoke_cases = {case["response_id"]: case for case in smoke["cases"]}
    assert manifest["reused_smoke_response_ids"] == list(smoke_cases)
    selected_cases = {case["response_id"]: case for case in cases}
    assert {
        response_id: selected_cases[response_id] for response_id in smoke_cases
    } == smoke_cases
    assert len(runner_cli._load_cases(manifest)) == 100
    assert stat.S_IMODE(manifest_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(sealed_path.stat().st_mode) == 0o600
    assert not list(output.glob("*.tmp"))

    assert main(args) == 0
    assert manifest_path.read_bytes() == original_manifest_bytes
    assert _sha256_path(manifest_path) == original_manifest_sha256


def test_cli_reselects_when_a_sealed_identifier_occurs_in_unrelated_case_text(
    tmp_path: Path,
) -> None:
    discovery, grounded, smoke_path, clean_output = _build_100(tmp_path)
    clean_manifest = json.loads(
        (clean_output / "manifest.json").read_text(encoding="utf-8")
    )
    clean_sealed = json.loads(
        (clean_output / "sealed_candidates.json").read_text(encoding="utf-8")
    )
    reused_ids = set(clean_manifest["reused_smoke_response_ids"])
    leaking_response_id = next(
        case["response_id"]
        for case in clean_manifest["cases"]
        if case["response_id"] not in reused_ids
    )
    leaking_candidate_id = clean_sealed[leaking_response_id]
    host_case = next(
        case
        for case in clean_manifest["cases"]
        if case["response_id"] not in reused_ids
        and case["response_id"] != leaking_response_id
    )

    discovery_value = json.loads(discovery.read_text(encoding="utf-8"))
    grounded_value = json.loads(grounded.read_text(encoding="utf-8"))
    changed_question = (
        discovery_value[host_case["question_id"]]["query"] + " " + leaking_candidate_id
    )
    discovery_value[host_case["question_id"]]["query"] = changed_question
    grounded_value["records"][host_case["question_id"]]["question"] = changed_question
    discovery.write_text(json.dumps(discovery_value, indent=2), encoding="utf-8")
    grounded.write_text(json.dumps(grounded_value, indent=2), encoding="utf-8")

    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["sources"] = {
        "discovery": {"sha256": _sha256_path(discovery)},
        "grounded": {"sha256": _sha256_path(grounded)},
    }
    smoke_body = dict(smoke)
    smoke_body.pop("manifest_hash")
    smoke["manifest_hash"] = _canonical_hash(smoke_body)
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    smoke_path.chmod(0o600)

    output = tmp_path / "identity-blind"
    assert main(_builder_args(discovery, grounded, smoke_path, output)) == 0

    manifest_bytes = (output / "manifest.json").read_bytes()
    sealed = json.loads((output / "sealed_candidates.json").read_text(encoding="utf-8"))
    assert leaking_candidate_id not in sealed.values()
    assert all(
        candidate_id.encode("utf-8") not in manifest_bytes
        for candidate_id in sealed.values()
    )


def test_cli_never_exposes_the_final_directory_before_both_files_are_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    smoke_path = _build_smoke(tmp_path, discovery, grounded)
    output = tmp_path / "local-100-v1"
    real_write = hundred_cli._write_json_atomically
    writes = 0

    def interrupted_write(path: Path, payload: object, *, mode: int) -> None:
        nonlocal writes
        assert not output.exists()
        writes += 1
        if writes == 2:
            raise SystemExit("simulated termination before publication")
        real_write(path, payload, mode=mode)

    monkeypatch.setattr(hundred_cli, "_write_json_atomically", interrupted_write)

    with pytest.raises(SystemExit, match="simulated termination"):
        main(_builder_args(discovery, grounded, smoke_path, output))

    assert writes == 2
    assert not output.exists()
    assert not list(tmp_path.glob(".local-100-v1.*"))


def test_cli_rejects_changed_smoke_content_without_partial_output(
    tmp_path: Path,
) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    smoke_path = _build_smoke(tmp_path, discovery, grounded)
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    smoke["sources"]["discovery"]["sha256"] = "f" * 64
    body = dict(smoke)
    body.pop("manifest_hash")
    smoke["manifest_hash"] = _canonical_hash(body)
    smoke_path.write_text(json.dumps(smoke), encoding="utf-8")
    smoke_path.chmod(0o600)
    output = tmp_path / "local-100-v1"

    with pytest.raises(ValueError, match="smoke manifest sources do not match"):
        main(_builder_args(discovery, grounded, smoke_path, output))

    assert not output.exists()


def test_cli_rejects_existing_output_directory_with_nonprivate_mode(
    tmp_path: Path,
) -> None:
    discovery, grounded, smoke_path, output = _build_100(tmp_path)
    output.chmod(0o755)

    with pytest.raises(ValueError, match="directory must have mode 0700"):
        main(_builder_args(discovery, grounded, smoke_path, output))


def test_cli_rejects_symlinked_existing_output_directory(tmp_path: Path) -> None:
    discovery, grounded, smoke_path, output = _build_100(tmp_path)
    backing = tmp_path / "local-100-v1.backing"
    output.rename(backing)
    output.symlink_to(backing.name, target_is_directory=True)

    with pytest.raises(ValueError, match="private directory"):
        main(_builder_args(discovery, grounded, smoke_path, output))


@pytest.mark.parametrize("filename", ["manifest.json", "sealed_candidates.json"])
def test_cli_rejects_symlinked_existing_output_files(
    tmp_path: Path, filename: str
) -> None:
    discovery, grounded, smoke_path, output = _build_100(tmp_path)
    path = output / filename
    backing = output / f"{filename}.backing"
    path.rename(backing)
    path.symlink_to(backing.name)

    with pytest.raises(ValueError, match="opened safely"):
        main(_builder_args(discovery, grounded, smoke_path, output))


@pytest.mark.parametrize("filename", ["manifest.json", "sealed_candidates.json"])
def test_cli_rejects_hardlinked_existing_output_files(
    tmp_path: Path, filename: str
) -> None:
    discovery, grounded, smoke_path, output = _build_100(tmp_path)
    path = output / filename
    backing = output / f"{filename}.backing"
    path.rename(backing)
    os.link(backing, path)

    with pytest.raises(ValueError, match="single-link regular file"):
        main(_builder_args(discovery, grounded, smoke_path, output))


@pytest.mark.parametrize("filename", ["manifest.json", "sealed_candidates.json"])
def test_cli_rejects_nonregular_existing_output_file(
    tmp_path: Path, filename: str
) -> None:
    discovery, grounded, smoke_path, output = _build_100(tmp_path)
    path = output / filename
    path.unlink()
    path.mkdir(mode=0o600)

    with pytest.raises(ValueError, match="single-link regular file"):
        main(_builder_args(discovery, grounded, smoke_path, output))


@pytest.mark.parametrize("filename", ["manifest.json", "sealed_candidates.json"])
def test_cli_rejects_tampered_existing_output(tmp_path: Path, filename: str) -> None:
    discovery, grounded, smoke_path, output = _build_100(tmp_path)
    path = output / filename
    value = json.loads(path.read_text(encoding="utf-8"))
    if filename == "manifest.json":
        value["cases"][0]["question"] = "tampered"
        message = "invalid existing manifest hash"
    else:
        value[next(iter(value))] = "tampered"
        message = "sealed candidates do not match"
    path.write_text(json.dumps(value), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ValueError, match=message):
        main(_builder_args(discovery, grounded, smoke_path, output))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.__setitem__("extra", True), "exact version 2 keys"),
        (
            lambda value: value.__setitem__("cases_per_reason", 9),
            "cases_per_reason must be 10",
        ),
        (
            lambda value: value.__setitem__(
                "reused_smoke_response_ids",
                value["reused_smoke_response_ids"][:-1],
            ),
            "exactly ten reused smoke response IDs",
        ),
        (
            lambda value: value["cases"].pop(),
            "exactly 100 cases",
        ),
    ],
)
def test_version_2_loader_rejects_wrong_keys_and_cardinalities(
    tmp_path: Path, mutate, message: str
) -> None:
    discovery, grounded = _write_inputs(tmp_path)
    smoke_path = _build_smoke(tmp_path, discovery, grounded)
    output = tmp_path / "local-100-v1"
    assert main(_builder_args(discovery, grounded, smoke_path, output)) == 0
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    mutate(manifest)
    body = dict(manifest)
    body.pop("manifest_hash")
    manifest["manifest_hash"] = _canonical_hash(body)

    with pytest.raises(ValueError, match=message):
        runner_cli._load_cases(manifest)


def test_version_2_loader_rejects_same_reason_reused_provenance_swap(
    tmp_path: Path,
) -> None:
    _, _, _, output = _build_100(tmp_path)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert (
        manifest["cases"][0]["selection_reason"]
        == manifest["cases"][1]["selection_reason"]
    )
    manifest["reused_smoke_response_ids"][0] = manifest["cases"][1]["response_id"]
    body = dict(manifest)
    body.pop("manifest_hash")
    manifest["manifest_hash"] = _canonical_hash(body)

    with pytest.raises(ValueError, match="canonical reserved positions"):
        runner_cli._load_cases(manifest)
