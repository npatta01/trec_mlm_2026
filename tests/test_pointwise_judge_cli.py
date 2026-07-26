import hashlib
import json
from pathlib import Path
import stat

import pytest

import scripts.build_pointwise_judge_smoke as smoke_cli
from scripts.build_pointwise_judge_smoke import main
import scripts.run_local_pointwise_judge as runner_cli
from trec_mlm.pointwise_sampling import DIAGNOSTIC_REASONS


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


def _canonical_hash(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _valid_decision() -> dict[str, object]:
    return {
        "label": "good",
        "correctness": 4,
        "completeness": 2,
        "relevance": 2,
        "error_flags": [],
        "reference_status": "sound",
        "confidence": "high",
        "rationale": "The response correctly identifies the answer.",
    }


def _valid_runner_manifest() -> dict[str, object]:
    cases = [
        {
            "question_id": f"q-{index}",
            "response_id": f"r-{index}",
            "question": f"Fabricated question {index}?",
            "reference_answer": f"Fabricated reference {index}.",
            "candidate_answer": f"Fabricated answer {index}.",
            "strata": [reason],
            "selection_reason": reason,
        }
        for index, reason in enumerate(DIAGNOSTIC_REASONS)
    ]
    body: dict[str, object] = {
        "version": 1,
        "seed": 2026,
        "sources": {
            "discovery": {"sha256": "a" * 64},
            "grounded": {"sha256": "b" * 64},
        },
        "selection_reasons": list(DIAGNOSTIC_REASONS),
        "cases": cases,
    }
    return {**body, "manifest_hash": _canonical_hash(body)}


def _valid_runtime_receipt() -> dict[str, object]:
    decision = _valid_decision()
    body: dict[str, object] = {
        "version": 1,
        "passed": True,
        "model_alias": "judge-a",
        "model_id": "model-a",
        "registry_sha256": "c" * 64,
        "runtime": {
            "name": "fixture-runtime",
            "version": "1.0",
            "backend_version": "2.0",
        },
        "hardware": {
            "gpu_arch": "gfx1151",
            "visible_memory_bytes": 64 * 1024**3,
            "free_disk_bytes": 100 * 1024**3,
        },
        "checkpoint": {
            "path": "/models/fixture-model",
            "revision": "fixture-revision",
            "size_bytes": 123456,
            "sha256": "d" * 64,
        },
        "endpoint": {
            "base_url": "http://127.0.0.1:13305/v1",
            "models": ["model-a"],
        },
        "configuration": {
            "context_tokens": 4096,
            "temperature": 0,
            "max_output_tokens": 384,
            "max_concurrency": 1,
        },
        "canary": {
            "passed": True,
            "raw": json.dumps(decision),
            "parsed": decision,
        },
        "startup_seconds": 1.25,
        "download_occurred": False,
    }
    return {**body, "receipt_sha256": _canonical_hash(body)}


def _write_runner_inputs(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "smoke_manifest.json"
    manifest.write_text(json.dumps(_valid_runner_manifest()), encoding="utf-8")
    manifest.chmod(0o600)
    receipt = tmp_path / "runtime.json"
    receipt.write_text(json.dumps(_valid_runtime_receipt()), encoding="utf-8")
    receipt.chmod(0o600)
    return manifest, receipt


def _runner_args(manifest: Path, receipt: Path, output: Path) -> list[str]:
    return [
        "--manifest",
        str(manifest),
        "--protocol-dir",
        "protocols/pointwise_judge_bakeoff",
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


def _rehash(payload: dict[str, object], hash_key: str) -> None:
    body = dict(payload)
    body.pop(hash_key, None)
    payload[hash_key] = _canonical_hash(body)


def test_runner_cli_consumes_authentic_manifest_and_runtime_receipt_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, receipt = _write_runner_inputs(tmp_path)
    output = tmp_path / "run.json"
    seen: list[tuple[str, str, object]] = []

    def fake_call(endpoint, model, messages, schema, *, timeout_seconds):
        seen.append((endpoint, model, messages))
        raw = json.dumps(_valid_decision())
        return raw, {
            "model": model,
            "choices": [{"message": {"content": raw}}],
            "usage": {"total_tokens": 10},
        }

    monkeypatch.setattr(runner_cli, "call_openai_judge", fake_call)

    assert runner_cli.main(_runner_args(manifest, receipt, output)) == 0
    assert seen[0][0:2] == ("http://127.0.0.1:13305/v1", "model-a")
    assert "Fabricated question 0?" in seen[0][2][1]["content"]
    assert len(seen) == len(DIAGNOSTIC_REASONS)
    assert (
        json.loads(output.read_text(encoding="utf-8"))["records"]["r-0"]["decision"][
            "label"
        ]
        == "good"
    )
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


@pytest.mark.parametrize(
    "corruption",
    ["body", "hash", "version", "schema", "selection_reasons"],
)
def test_runner_cli_rejects_inauthentic_manifest_before_server_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    manifest, receipt = _write_runner_inputs(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if corruption == "body":
        payload["cases"][0]["question"] = "tampered"
    elif corruption == "hash":
        payload["manifest_hash"] = "0" * 64
    elif corruption == "version":
        payload["version"] = 2
        _rehash(payload, "manifest_hash")
    elif corruption == "schema":
        payload.pop("sources")
        _rehash(payload, "manifest_hash")
    elif corruption == "selection_reasons":
        payload["selection_reasons"] = list(reversed(DIAGNOSTIC_REASONS))
        _rehash(payload, "manifest_hash")
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )
    output = tmp_path / "run.json"

    with pytest.raises(ValueError, match="manifest"):
        runner_cli.main(_runner_args(manifest, receipt, output))
    assert not output.exists()


@pytest.mark.parametrize(
    "corruption",
    [
        "body",
        "hash",
        "version",
        "foreign_alias",
        "foreign_model",
        "failed",
        "configuration",
        "configuration_type",
        "canary",
    ],
)
def test_runner_cli_rejects_inauthentic_runtime_receipt_before_server_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    manifest, receipt = _write_runner_inputs(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if corruption == "body":
        payload["runtime"]["version"] = "tampered"
    elif corruption == "hash":
        payload["receipt_sha256"] = "0" * 64
    elif corruption == "version":
        payload["version"] = 2
        _rehash(payload, "receipt_sha256")
    elif corruption == "foreign_alias":
        payload["model_alias"] = "judge-b"
        _rehash(payload, "receipt_sha256")
    elif corruption == "foreign_model":
        payload["model_id"] = "model-b"
        _rehash(payload, "receipt_sha256")
    elif corruption == "failed":
        payload["passed"] = False
        _rehash(payload, "receipt_sha256")
    elif corruption == "configuration":
        payload["configuration"]["context_tokens"] = 2048
        _rehash(payload, "receipt_sha256")
    elif corruption == "configuration_type":
        payload["configuration"]["max_concurrency"] = True
        _rehash(payload, "receipt_sha256")
    elif corruption == "canary":
        payload["canary"]["parsed"] = {"tampered": True}
        _rehash(payload, "receipt_sha256")
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )
    output = tmp_path / "run.json"

    with pytest.raises(ValueError, match="receipt"):
        runner_cli.main(_runner_args(manifest, receipt, output))
    assert not output.exists()


def test_runner_cli_rejects_duplicate_private_input_keys_before_server_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, receipt = _write_runner_inputs(tmp_path)
    manifest.write_text('{"version": 1, "version": 1}', encoding="utf-8")
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner_cli.main(_runner_args(manifest, receipt, tmp_path / "run.json"))
