import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

import scripts.build_pointwise_judge_smoke as smoke_cli
from scripts.build_pointwise_judge_smoke import main
import scripts.preflight_local_pointwise_models as preflight_cli
import scripts.run_local_pointwise_judge as runner_cli
from trec_mlm.pointwise_judge import load_prompt_contract
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


def _valid_registry() -> dict[str, object]:
    configuration = {
        "context_tokens": 4096,
        "temperature": 0,
        "max_output_tokens": 384,
        "max_concurrency": 1,
    }
    return {
        "version": 1,
        "models": {
            "qwen36-35b-a3b-gguf": {
                "model_id": "Qwen3.6-35B-A3B-GGUF",
                "runtime": "lemonade-llamacpp-rocm",
                "new_download_authorized": True,
                "checkpoint": {
                    "revision": None,
                    "gguf_filename": "Qwen3.6-35B-A3B-Q4_K_M.gguf",
                    "expected_bytes": 123456,
                    "sha256": "d" * 64,
                },
                "configuration": dict(configuration),
            },
            "qwen36-27b-fp8": {
                "model_id": "Qwen/Qwen3.6-27B-FP8",
                "runtime": "vllm-rocm",
                "new_download_authorized": False,
                "checkpoint": {
                    "revision": "qwen-fixture-revision",
                    "gguf_filename": None,
                    "expected_bytes": 223456,
                    "sha256": "e" * 64,
                },
                "configuration": dict(configuration),
            },
            "gpt-oss-20b": {
                "model_id": "openai/gpt-oss-20b",
                "runtime": "vllm-rocm",
                "new_download_authorized": False,
                "checkpoint": {
                    "revision": "gpt-fixture-revision",
                    "gguf_filename": None,
                    "expected_bytes": 323456,
                    "sha256": "f" * 64,
                },
                "configuration": dict(configuration),
            },
            "gemma4-31b-q4": {
                "model_id": "google/gemma-4-31B-it-qat-q4_0-gguf",
                "runtime": "lemonade-llamacpp-rocm-custom",
                "new_download_authorized": False,
                "checkpoint": {
                    "revision": None,
                    "gguf_filename": "gemma-4-31B-it-Q4_0.gguf",
                    "expected_bytes": 423456,
                    "sha256": "1" * 64,
                },
                "configuration": dict(configuration),
            },
        },
    }


def _valid_runtime_receipt(registry_sha256: str) -> dict[str, object]:
    decision = _valid_decision()
    body: dict[str, object] = {
        "version": 1,
        "passed": True,
        "model_alias": "qwen36-35b-a3b-gguf",
        "model_id": "Qwen3.6-35B-A3B-GGUF",
        "registry_sha256": registry_sha256,
        "schema_sha256": "84557061030827f09f94063a85ccbaa5324797b958963af503342ae1fce6c69b",
        "launch_evidence_sha256": "c" * 64,
        "runtime": {
            "name": "lemonade-llamacpp-rocm",
            "version": "1.0",
            "backend_version": "2.0",
        },
        "hardware": {
            "gpu_arch": "gfx1151",
            "visible_memory_bytes": 64 * 1024**3,
            "free_disk_bytes": 100 * 1024**3,
        },
        "checkpoint": {
            "path": "/models/Qwen3.6-35B-A3B-Q4_K_M.gguf",
            "revision": "not-applicable",
            "size_bytes": 123456,
            "sha256": "d" * 64,
        },
        "endpoint": {
            "base_url": "http://127.0.0.1:13305/v1",
            "models": ["Qwen3.6-35B-A3B-GGUF"],
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


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_runner_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    manifest = tmp_path / "smoke_manifest.json"
    manifest.write_text(json.dumps(_valid_runner_manifest()), encoding="utf-8")
    manifest.chmod(0o600)
    registry = tmp_path / "local-models-v1.json"
    registry.write_text(json.dumps(_valid_registry()), encoding="utf-8")
    registry.chmod(0o600)
    receipt = tmp_path / "runtime.json"
    receipt.write_text(
        json.dumps(_valid_runtime_receipt(_sha256_path(registry))), encoding="utf-8"
    )
    receipt.chmod(0o600)
    return manifest, receipt, registry


def _runner_args(
    manifest: Path,
    receipt: Path,
    registry: Path,
    output: Path,
    *,
    manifest_sha256: str | None = None,
    receipt_sha256: str | None = None,
    registry_sha256: str | None = None,
) -> list[str]:
    return [
        "--manifest",
        str(manifest),
        "--manifest-sha256",
        manifest_sha256 or _sha256_path(manifest),
        "--protocol-dir",
        "protocols/pointwise_judge_bakeoff",
        "--registry",
        str(registry),
        "--registry-sha256",
        registry_sha256 or _sha256_path(registry),
        "--model-alias",
        "qwen36-35b-a3b-gguf",
        "--model-id",
        "Qwen3.6-35B-A3B-GGUF",
        "--base-url",
        "http://127.0.0.1:13305/v1",
        "--runtime-receipt",
        str(receipt),
        "--runtime-receipt-sha256",
        receipt_sha256 or _sha256_path(receipt),
        "--output",
        str(output),
        "--timeout-seconds",
        "12.5",
    ]


def _rehash(payload: dict[str, object], hash_key: str) -> None:
    body = dict(payload)
    body.pop(hash_key, None)
    payload[hash_key] = _canonical_hash(body)


def test_runner_cli_requires_independent_input_digest_anchors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
    args = _runner_args(manifest, receipt, registry, tmp_path / "run.json")
    for flag in (
        "--manifest-sha256",
        "--runtime-receipt-sha256",
        "--registry-sha256",
    ):
        index = args.index(flag)
        del args[index : index + 2]
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )

    with pytest.raises(SystemExit):
        runner_cli.main(args)


def test_runner_cli_consumes_authentic_manifest_and_runtime_receipt_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
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

    assert runner_cli.main(_runner_args(manifest, receipt, registry, output)) == 0
    assert seen[0][0:2] == (
        "http://127.0.0.1:13305/v1",
        "Qwen3.6-35B-A3B-GGUF",
    )
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
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
    args = _runner_args(manifest, receipt, registry, tmp_path / "run.json")
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    if corruption == "body":
        payload["cases"][0]["question"] = "tampered"
        _rehash(payload, "manifest_hash")
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
        runner_cli.main(args)
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
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
    args = _runner_args(manifest, receipt, registry, tmp_path / "run.json")
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    if corruption == "body":
        payload["runtime"]["version"] = "tampered"
        _rehash(payload, "receipt_sha256")
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
        runner_cli.main(args)
    assert not output.exists()


def test_runner_cli_rejects_boolean_substitution_in_anchored_canary_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    raw_decision = {
        **_valid_decision(),
        "label": "incorrect",
        "correctness": 1,
    }
    payload["canary"]["raw"] = json.dumps(raw_decision)
    payload["canary"]["parsed"] = {**raw_decision, "correctness": True}
    _rehash(payload, "receipt_sha256")
    receipt.write_text(json.dumps(payload), encoding="utf-8")
    args = _runner_args(manifest, receipt, registry, tmp_path / "run.json")
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )

    with pytest.raises(ValueError, match="canary"):
        runner_cli.main(args)


def test_runner_cli_rejects_rehashed_registry_tampering_before_server_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
    args = _runner_args(manifest, receipt, registry, tmp_path / "run.json")
    payload = json.loads(registry.read_text(encoding="utf-8"))
    payload["models"]["qwen36-35b-a3b-gguf"]["runtime"] = "tampered-runtime"
    registry.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )

    with pytest.raises(ValueError, match="registry"):
        runner_cli.main(args)


@pytest.mark.parametrize(
    "corruption",
    ["top_level", "missing_alias", "entry_key", "checkpoint_pin", "configuration"],
)
def test_runner_cli_rejects_externally_anchored_invalid_registry_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
    payload = json.loads(registry.read_text(encoding="utf-8"))
    selected = payload["models"]["qwen36-35b-a3b-gguf"]
    if corruption == "top_level":
        payload["unreviewed"] = True
    elif corruption == "missing_alias":
        payload["models"].pop("gemma4-31b-q4")
    elif corruption == "entry_key":
        selected["unreviewed"] = True
    elif corruption == "checkpoint_pin":
        selected["checkpoint"]["revision"] = "also-pinned"
    elif corruption == "configuration":
        selected["configuration"]["context_tokens"] = 2048
    registry.write_text(json.dumps(payload), encoding="utf-8")
    args = _runner_args(manifest, receipt, registry, tmp_path / "run.json")
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )

    with pytest.raises(ValueError, match="registry"):
        runner_cli.main(args)


@pytest.mark.parametrize("binding", ["runtime", "checkpoint", "download"])
def test_runner_cli_binds_receipt_to_externally_anchored_registry_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
    registry_payload = json.loads(registry.read_text(encoding="utf-8"))
    selected = registry_payload["models"]["qwen36-35b-a3b-gguf"]
    if binding == "runtime":
        selected["runtime"] = "reviewed-alternate-runtime"
    elif binding == "checkpoint":
        selected["checkpoint"]["expected_bytes"] += 1
    else:
        selected["new_download_authorized"] = False
    registry.write_text(json.dumps(registry_payload), encoding="utf-8")
    receipt_payload = json.loads(receipt.read_text(encoding="utf-8"))
    receipt_payload["registry_sha256"] = _sha256_path(registry)
    if binding == "download":
        receipt_payload["download_occurred"] = True
    _rehash(receipt_payload, "receipt_sha256")
    receipt.write_text(json.dumps(receipt_payload), encoding="utf-8")
    args = _runner_args(manifest, receipt, registry, tmp_path / "run.json")
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )

    with pytest.raises(ValueError, match="registry|unauthorized"):
        runner_cli.main(args)


@pytest.mark.parametrize(
    ("flag", "description"),
    [
        ("--manifest", "manifest"),
        ("--runtime-receipt", "receipt"),
        ("--registry", "registry"),
    ],
)
def test_runner_cli_rejects_symlinked_inputs_before_server_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    description: str,
) -> None:
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
    args = _runner_args(manifest, receipt, registry, tmp_path / "run.json")
    index = args.index(flag)
    target = Path(args[index + 1])
    symlink = tmp_path / f"{target.name}.link"
    symlink.symlink_to(target)
    args[index + 1] = str(symlink)
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )

    with pytest.raises(ValueError, match=description):
        runner_cli.main(args)


@pytest.mark.parametrize(
    ("flag", "description"),
    [
        ("--manifest", "manifest"),
        ("--runtime-receipt", "receipt"),
        ("--registry", "registry"),
    ],
)
def test_runner_cli_rejects_input_path_swapped_after_descriptor_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    description: str,
) -> None:
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
    args = _runner_args(manifest, receipt, registry, tmp_path / "run.json")
    index = args.index(flag)
    target = Path(args[index + 1])
    replacement = tmp_path / f"{target.name}.replacement"
    replacement.write_bytes(target.read_bytes())
    replacement.chmod(stat.S_IMODE(target.stat().st_mode))
    real_open = os.open
    real_replace = os.replace
    swapped = False

    def racing_open(path: object, flags: int) -> int:
        nonlocal swapped
        descriptor = real_open(path, flags)
        if Path(path) == target and not swapped:
            swapped = True
            real_replace(replacement, target)
        return descriptor

    monkeypatch.setattr(runner_cli.os, "open", racing_open)
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )

    with pytest.raises(ValueError, match=f"{description} changed during read"):
        runner_cli.main(args)
    assert swapped is True


def test_runner_cli_rejects_oversized_private_input_before_server_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
    manifest.write_bytes(b" " * (4 * 1024 * 1024 + 1))
    manifest.chmod(0o600)
    args = _runner_args(manifest, receipt, registry, tmp_path / "run.json")
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )

    with pytest.raises(ValueError, match="manifest is too large"):
        runner_cli.main(args)


@pytest.mark.parametrize(
    ("flag", "mode", "message"),
    [
        ("--manifest", 0o644, "manifest must have mode 0600"),
        ("--runtime-receipt", 0o644, "receipt must have mode 0600"),
        ("--registry", 0o666, "registry has an unsafe protocol-file mode"),
    ],
)
def test_runner_cli_rejects_unsafe_input_modes_before_server_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
    mode: int,
    message: str,
) -> None:
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
    args = _runner_args(manifest, receipt, registry, tmp_path / "run.json")
    target = Path(args[args.index(flag) + 1])
    target.chmod(mode)
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )

    with pytest.raises(ValueError, match=message):
        runner_cli.main(args)


@pytest.mark.parametrize("flag", ["--manifest", "--runtime-receipt", "--registry"])
def test_runner_cli_rejects_duplicate_input_keys_before_server_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    flag: str,
) -> None:
    manifest, receipt, registry = _write_runner_inputs(tmp_path)
    target = {
        "--manifest": manifest,
        "--runtime-receipt": receipt,
        "--registry": registry,
    }[flag]
    target.write_text('{"version": 1, "version": 1}', encoding="utf-8")
    args = _runner_args(manifest, receipt, registry, tmp_path / "run.json")
    monkeypatch.setattr(
        runner_cli,
        "call_openai_judge",
        lambda *args, **kwargs: pytest.fail("server call must not occur"),
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner_cli.main(args)


class _FabricatedPreflightDependencies:
    def __init__(self, *, entry: dict[str, object], model_id: str) -> None:
        checkpoint = entry["checkpoint"]
        assert isinstance(checkpoint, dict)
        self.commands = {
            ("lemonade", "--version"): "lemonade version 11.5.0\n",
            ("rocminfo",): "Name: gfx1151\n",
        }
        self.files = {
            Path("/fabricated/server_models.json"): b'{"registry":"fixture"}',
            Path("/fabricated/backend_versions.json"): json.dumps(
                {
                    "llamacpp": {"rocm-stable": "b9752"},
                    "vllm": {"rocm": "vllm0.20.1-rocm7.12.0"},
                }
            ).encode(),
        }
        self.checkpoint = {
            "path": f"/fabricated/{checkpoint['gguf_filename'] or checkpoint['revision']}",
            "revision": checkpoint["revision"] or "not-applicable",
            "size_bytes": checkpoint["expected_bytes"],
            "sha256": checkpoint["sha256"],
        }
        self.visible_memory_bytes = 100 * 1024**3
        self.free_disk_bytes = 50 * 1024**3
        self.endpoint_model = model_id
        self.canary_raw = json.dumps(_valid_decision())
        self.canary_success = True
        self.process = {"start_ticks": 987654, "argv": []}
        self.requests = 0

    def run_command(self, argv: tuple[str, ...]) -> str:
        return self.commands[argv]

    def read_bytes(self, path: Path) -> bytes:
        return self.files[path]

    def inspect_checkpoint(
        self, path: Path, checkpoint: dict[str, object]
    ) -> dict[str, object]:
        return dict(self.checkpoint)

    def available_memory_bytes(self) -> int:
        return self.visible_memory_bytes

    def available_disk_bytes(self, path: Path) -> int:
        return self.free_disk_bytes

    def process_observation(self, pid: int) -> dict[str, object]:
        assert pid == 4242
        return {
            "start_ticks": self.process["start_ticks"],
            "argv": list(self.process["argv"]),
        }

    def request_json(
        self, method: str, url: str, payload: object | None
    ) -> dict[str, object]:
        self.requests += 1
        if method == "GET":
            return {"object": "list", "data": [{"id": self.endpoint_model}]}
        if not self.canary_success:
            return {"model": self.endpoint_model, "choices": [], "usage": {}}
        return {
            "model": self.endpoint_model,
            "choices": [{"message": {"content": self.canary_raw}}],
            "usage": {"total_tokens": 20},
        }


def _preflight_source_pins(
    dependencies: _FabricatedPreflightDependencies,
) -> preflight_cli.SourcePins:
    return preflight_cli.SourcePins(
        lemonade_models_path=Path("/fabricated/server_models.json"),
        lemonade_models_sha256=hashlib.sha256(
            dependencies.files[Path("/fabricated/server_models.json")]
        ).hexdigest(),
        backend_versions_path=Path("/fabricated/backend_versions.json"),
        backend_versions_sha256=hashlib.sha256(
            dependencies.files[Path("/fabricated/backend_versions.json")]
        ).hexdigest(),
        lemonade_version="11.5.0",
        llamacpp_backend_version="b9752",
        vllm_backend_version="vllm0.20.1-rocm7.12.0",
    )


def _run_valid_preflight(
    *,
    alias: str = "qwen36-35b-a3b-gguf",
    mutate: object | None = None,
    download_occurred: bool = False,
    schema_override: dict[str, object] | None = None,
    schema_sha256: str = "84557061030827f09f94063a85ccbaa5324797b958963af503342ae1fce6c69b",
    base_url: str = "http://127.0.0.1:13305/v1",
    source_pins_transform: object | None = None,
) -> dict[str, object]:
    registry = _valid_registry()
    entry = registry["models"][alias]
    assert isinstance(entry, dict)
    dependencies = _FabricatedPreflightDependencies(
        entry=entry, model_id=str(entry["model_id"])
    )
    if entry["runtime"] == "vllm-rocm":
        argv = [
            "vllm",
            "serve",
            str(entry["model_id"]),
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "1",
        ]
    else:
        argv = [
            "llama-server",
            "--model",
            str(dependencies.checkpoint["path"]),
            "--ctx-size",
            "4096",
            "--parallel",
            "1",
        ]
    dependencies.process["argv"] = argv
    launch_body: dict[str, object] = {
        "version": 1,
        "model_alias": alias,
        "model_id": entry["model_id"],
        "base_url": base_url,
        "runtime": entry["runtime"],
        "process": {"pid": 4242, "start_ticks": 987654, "argv": argv},
        "started_monotonic_ns": 1_000_000_000,
        "ready_monotonic_ns": 3_500_000_000,
        "checkpoint_before": dict(dependencies.checkpoint),
        "checkpoint_after": dict(dependencies.checkpoint),
        "download_events": [],
    }
    launch_evidence = {
        **launch_body,
        "evidence_sha256": _canonical_hash(launch_body),
    }
    if download_occurred:
        launch_evidence["checkpoint_before"] = None
        launch_evidence["download_events"] = [
            {
                "model_alias": alias,
                "started_monotonic_ns": 1_100_000_000,
                "completed_monotonic_ns": 3_000_000_000,
                "expected_bytes": dependencies.checkpoint["size_bytes"],
                "sha256": dependencies.checkpoint["sha256"],
            }
        ]
        _rehash(launch_evidence, "evidence_sha256")
    dependencies.launch_evidence = launch_evidence
    launch_evidence_sha256 = _canonical_hash(launch_evidence)
    source_pins = _preflight_source_pins(dependencies)
    if source_pins_transform is not None:
        assert callable(source_pins_transform)
        source_pins = source_pins_transform(source_pins)
    if mutate is not None:
        assert callable(mutate)
        mutate(registry, dependencies)
    contract = load_prompt_contract(Path("protocols/pointwise_judge_bakeoff"))
    return preflight_cli.run_preflight(
        registry,
        model_alias=alias,
        base_url=base_url,
        registry_sha256="a" * 64,
        checkpoint_path=Path("/fabricated/checkpoint"),
        dependencies=dependencies,
        source_pins=source_pins,
        schema=contract.schema if schema_override is None else schema_override,
        schema_sha256=schema_sha256,
        launch_evidence=launch_evidence,
        launch_evidence_sha256=launch_evidence_sha256,
    )


def test_preflight_registry_contains_exact_approved_local_pool() -> None:
    registry = preflight_cli.load_registry(
        Path("protocols/pointwise_judge_bakeoff/local-models-v1.json")
    )
    assert set(registry["models"]) == {
        "qwen36-35b-a3b-gguf",
        "qwen36-27b-fp8",
        "gpt-oss-20b",
        "gemma4-31b-q4",
    }


def test_preflight_builds_task3_accepted_self_hashed_receipt() -> None:
    receipt = _run_valid_preflight()
    assert receipt["passed"] is True
    assert receipt["schema_sha256"] == (
        "84557061030827f09f94063a85ccbaa5324797b958963af503342ae1fce6c69b"
    )
    assert receipt["launch_evidence_sha256"]
    assert receipt["startup_seconds"] == 2.5
    assert receipt["download_occurred"] is False
    body = dict(receipt)
    stored_hash = body.pop("receipt_sha256")
    assert stored_hash == _canonical_hash(body)
    selected = _valid_registry()["models"]["qwen36-35b-a3b-gguf"]
    runner_cli.validate_runtime_receipt(
        receipt,
        model_alias="qwen36-35b-a3b-gguf",
        model_id="Qwen3.6-35B-A3B-GGUF",
        endpoint="http://127.0.0.1:13305/v1",
        registry_sha256="a" * 64,
        registry_entry=selected,
    )


def test_preflight_rejects_replaced_or_wrongly_hashed_canary_schema() -> None:
    assert _run_valid_preflight(schema_override={})["passed"] is False
    assert _run_valid_preflight(schema_sha256="0" * 64)["passed"] is False


def test_preflight_derives_authorized_download_from_launch_journal() -> None:
    receipt = _run_valid_preflight(download_occurred=True)
    assert receipt["passed"] is True
    assert receipt["download_occurred"] is True


def test_preflight_host_thresholds_accept_exact_plan_boundaries() -> None:
    assert _run_valid_preflight()["passed"] is True


@pytest.mark.parametrize(
    "mutate",
    [
        lambda registry, deps: setattr(deps, "visible_memory_bytes", 100 * 1024**3 - 1),
        lambda registry, deps: setattr(deps, "free_disk_bytes", 50 * 1024**3 - 1),
    ],
)
def test_preflight_host_thresholds_reject_one_byte_below_plan(
    mutate: object,
) -> None:
    assert _run_valid_preflight(mutate=mutate)["passed"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda registry, deps: deps.launch_evidence.__setitem__(
            "ready_monotonic_ns", 999_999_999
        ),
        lambda registry, deps: deps.launch_evidence["process"].__setitem__(
            "start_ticks", 1
        ),
        lambda registry, deps: deps.launch_evidence.__setitem__(
            "download_events", [{"untrusted": True}]
        ),
    ],
)
def test_preflight_rejects_launch_evidence_drift(mutate: object) -> None:
    assert _run_valid_preflight(mutate=mutate)["passed"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda registry, deps: setattr(deps, "endpoint_model", "wrong-model"),
        lambda registry, deps: deps.checkpoint.__setitem__("size_bytes", 1),
        lambda registry, deps: deps.checkpoint.__setitem__("sha256", "0" * 64),
        lambda registry, deps: deps.checkpoint.__setitem__(
            "path", "/fabricated/wrong.gguf"
        ),
        lambda registry, deps: deps.commands.__setitem__(
            ("rocminfo",), "Name: gfx1100\n"
        ),
        lambda registry, deps: setattr(deps, "visible_memory_bytes", 1),
        lambda registry, deps: setattr(deps, "free_disk_bytes", 1),
        lambda registry, deps: deps.commands.__setitem__(
            ("lemonade", "--version"), "lemonade version 11.4.0\n"
        ),
        lambda registry, deps: setattr(deps, "canary_raw", "not-json"),
        lambda registry, deps: setattr(deps, "canary_success", False),
        lambda registry, deps: setattr(
            deps,
            "canary_raw",
            json.dumps({**_valid_decision(), "unexpected": True}),
        ),
        lambda registry, deps: registry["models"]["qwen36-35b-a3b-gguf"][
            "configuration"
        ].__setitem__("context_tokens", 2048),
        lambda registry, deps: deps.process.__setitem__(
            "argv", ["llama-server", "--ctx-size", "2048", "--parallel", "1"]
        ),
    ],
)
def test_preflight_fails_closed_on_runtime_drift(mutate: object) -> None:
    assert _run_valid_preflight(mutate=mutate)["passed"] is False


def test_preflight_fails_closed_on_non_loopback_endpoint() -> None:
    assert _run_valid_preflight(base_url="http://0.0.0.0:13305/v1")["passed"] is False


def test_preflight_fails_closed_on_checkpoint_revision_drift() -> None:
    assert (
        _run_valid_preflight(
            alias="qwen36-27b-fp8",
            mutate=lambda registry, deps: deps.checkpoint.__setitem__(
                "revision", "drift"
            ),
        )["passed"]
        is False
    )


def test_preflight_fails_closed_on_wrong_backend_version() -> None:
    def wrong_backend(pins: preflight_cli.SourcePins) -> preflight_cli.SourcePins:
        return preflight_cli.SourcePins(
            lemonade_models_path=pins.lemonade_models_path,
            lemonade_models_sha256=pins.lemonade_models_sha256,
            backend_versions_path=pins.backend_versions_path,
            backend_versions_sha256=pins.backend_versions_sha256,
            lemonade_version=pins.lemonade_version,
            llamacpp_backend_version="wrong",
            vllm_backend_version=pins.vllm_backend_version,
        )

    assert _run_valid_preflight(source_pins_transform=wrong_backend)["passed"] is False


def test_preflight_fails_closed_on_source_file_drift() -> None:
    def mutate(registry: object, dependencies: object) -> None:
        dependencies.files[Path("/fabricated/server_models.json")] += b"drift"

    assert _run_valid_preflight(mutate=mutate)["passed"] is False


def test_preflight_rejects_download_for_unauthorized_model() -> None:
    assert (
        _run_valid_preflight(alias="gpt-oss-20b", download_occurred=True)["passed"]
        is False
    )


def test_preflight_writes_private_receipt_atomically(tmp_path: Path) -> None:
    receipt = _run_valid_preflight()
    output = tmp_path / "private" / "runtime.json"
    preflight_cli.write_runtime_receipt(output, receipt)
    assert stat.S_IMODE(output.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert json.loads(output.read_text()) == receipt
    assert not list(output.parent.glob("*.tmp"))


def test_hf_tree_commitment_sorts_paths_and_binds_sizes_and_hashes() -> None:
    files = [
        {"path": "weights/b.bin", "size_bytes": 2, "sha256": "b" * 64},
        {"path": "a.json", "size_bytes": 1, "sha256": "a" * 64},
    ]
    assert preflight_cli.checkpoint_tree_commitment(files) == (
        3,
        "1f4f651d6567761316a1846c6edd6b67c5f7d2f4e640c4eb3da4ca59712727cf",
    )


def _fabricated_hf_snapshot(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "models--owner--model"
    blobs = repository / "blobs"
    snapshot = repository / "snapshots" / "revision-a"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    digest = hashlib.sha256(b"weight-bytes").hexdigest()
    target = blobs / digest
    target.write_bytes(b"weight-bytes")
    logical = snapshot / "model.safetensors"
    logical.symlink_to(Path("../../blobs") / digest)
    return snapshot, logical, target


def test_hf_checkpoint_inspection_accepts_only_in_root_cache_symlinks(
    tmp_path: Path,
) -> None:
    snapshot, _, target = _fabricated_hf_snapshot(tmp_path)
    row = {
        "path": "model.safetensors",
        "size_bytes": len(b"weight-bytes"),
        "sha256": hashlib.sha256(b"weight-bytes").hexdigest(),
    }
    expected_size, expected_hash = preflight_cli.checkpoint_tree_commitment([row])
    observed = preflight_cli.SystemDependencies().inspect_checkpoint(
        snapshot,
        {"revision": "revision-a"},
    )
    assert observed["size_bytes"] == expected_size == target.stat().st_size
    assert observed["sha256"] == expected_hash


def test_hf_checkpoint_inspection_rejects_symlink_escape(tmp_path: Path) -> None:
    snapshot, logical, _ = _fabricated_hf_snapshot(tmp_path)
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    logical.unlink()
    logical.symlink_to(outside)

    with pytest.raises(ValueError, match="blob root"):
        preflight_cli.SystemDependencies().inspect_checkpoint(
            snapshot, {"revision": "revision-a"}
        )


def test_hf_checkpoint_inspection_rejects_hardlink_ambiguity(tmp_path: Path) -> None:
    snapshot, _, target = _fabricated_hf_snapshot(tmp_path)
    os.link(target, target.with_name("second-link"))

    with pytest.raises(ValueError, match="hardlink"):
        preflight_cli.SystemDependencies().inspect_checkpoint(
            snapshot, {"revision": "revision-a"}
        )


@pytest.mark.parametrize("race", ["logical_link", "target"])
def test_hf_checkpoint_inspection_rejects_mutation_during_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    race: str,
) -> None:
    snapshot, logical, target = _fabricated_hf_snapshot(tmp_path)
    replacement = target.with_name("f" * 64)
    replacement.write_bytes(b"replacement")
    real_read = os.read
    raced = False

    def racing_read(descriptor: int, size: int) -> bytes:
        nonlocal raced
        chunk = real_read(descriptor, size)
        if not raced:
            raced = True
            if race == "logical_link":
                logical.unlink()
                logical.symlink_to(Path("../../blobs") / replacement.name)
            else:
                os.replace(replacement, target)
        return chunk

    monkeypatch.setattr(preflight_cli.os, "read", racing_read)

    with pytest.raises(ValueError, match="changed during hash"):
        preflight_cli.SystemDependencies().inspect_checkpoint(
            snapshot, {"revision": "revision-a"}
        )
    assert raced is True


def test_runtime_receipt_writer_rejects_symlinked_parent_or_final(
    tmp_path: Path,
) -> None:
    receipt = _run_valid_preflight()
    real_parent = tmp_path / "real"
    real_parent.mkdir(mode=0o700)
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        preflight_cli.write_runtime_receipt(linked_parent / "runtime.json", receipt)

    safe_parent = tmp_path / "safe"
    safe_output = safe_parent / "source.json"
    preflight_cli.write_runtime_receipt(safe_output, receipt)
    final_link = safe_parent / "linked.json"
    final_link.symlink_to(safe_output.name)
    with pytest.raises(ValueError, match="regular file|symlink"):
        preflight_cli.write_runtime_receipt(final_link, receipt)


def test_runtime_receipt_writer_never_overwrites_concurrent_different_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = _run_valid_preflight()
    output = tmp_path / "private" / "runtime.json"
    real_link = os.link
    raced = False

    def racing_link(
        source: object,
        destination: object,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        nonlocal raced
        raced = True
        competing = output.parent / str(destination)
        competing.write_text("{}\n", encoding="utf-8")
        competing.chmod(0o600)
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(preflight_cli.os, "link", racing_link)
    with pytest.raises(ValueError, match="different runtime receipt"):
        preflight_cli.write_runtime_receipt(output, receipt)
    assert raced is True
    assert output.read_text(encoding="utf-8") == "{}\n"


def test_preflight_json_loader_rejects_duplicate_and_nonfinite_values(
    tmp_path: Path,
) -> None:
    for raw in (b'{"a":1,"a":1}', b'{"value":NaN}'):
        path = tmp_path / hashlib.sha256(raw).hexdigest()
        path.write_bytes(raw)
        with pytest.raises(ValueError, match="duplicate|non-finite"):
            preflight_cli.load_registry(path)
