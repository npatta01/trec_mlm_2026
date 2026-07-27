from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import stat

import pytest

from scripts.build_pointwise_judge_review import main as review_main
from trec_mlm.pointwise_judge import CandidateCase
from trec_mlm.pointwise_review import (
    build_critique_packet,
    build_independent_review_packet,
    compare_reviews,
    parse_reviewer_record,
    validate_reviewer_coverage,
    validate_reviewer_record,
)
from trec_mlm.pointwise_runner import FROZEN_DECISION_SCHEMA_SHA256


CASES = [
    CandidateCase(
        question_id="q1",
        response_id="r1",
        question="Who wrote Hamlet?",
        reference_answer="William Shakespeare wrote Hamlet.",
        candidate_answer="Shakespeare wrote Hamlet.",
        strata=("high_overlap_control",),
        selection_reason="high_overlap_control",
    ),
    CandidateCase(
        question_id="q2",
        response_id="r2",
        question="What is the boiling point of water?",
        reference_answer="At sea level, water boils at 100 degrees Celsius.",
        candidate_answer="At sea level it boils at 100 C.",
        strata=("number_or_date_mismatch_proxy",),
        selection_reason="number_or_date_mismatch_proxy",
    ),
]
EXPECTED_IDS = [case.response_id for case in CASES]
DECISION = {
    "label": "good",
    "correctness": 4,
    "completeness": 2,
    "relevance": 2,
    "error_flags": [],
    "reference_status": "sound",
    "confidence": "high",
    "rationale": "The response gives the requested fact correctly and directly.",
}


def _json_hash(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _response_body(raw: str, model_id: str) -> dict[str, object]:
    return {
        "model": model_id,
        "choices": [{"message": {"content": raw}}],
        "usage": {"total_tokens": 12},
    }


def _run(alias: str, model_id: str, backend_response_model: str) -> dict[str, object]:
    prompt_hash = "1" * 64
    receipt_hash = hashlib.sha256(alias.encode()).hexdigest()
    raw = json.dumps(DECISION)
    records: dict[str, object] = {}
    for case in CASES:
        records[case.response_id] = {
            "input_sha256": _json_hash(asdict(case)),
            "prompt_sha256": prompt_hash,
            "schema_sha256": FROZEN_DECISION_SCHEMA_SHA256,
            "model_alias": alias,
            "model_id": model_id,
            "runtime_receipt_sha256": receipt_hash,
            "attempts": [
                {
                    "raw": raw,
                    "response_body": _response_body(raw, backend_response_model),
                    "logical_model_id": model_id,
                    "backend_model_id": backend_response_model,
                    "parsed": dict(DECISION),
                }
            ],
            "decision": dict(DECISION),
            "failure": None,
            "elapsed_seconds": 1.25,
            "usage": {"total_tokens": 12},
        }
    return {
        "version": 1,
        "model_alias": alias,
        "model_id": model_id,
        "prompt_sha256": prompt_hash,
        "schema_sha256": FROZEN_DECISION_SCHEMA_SHA256,
        "runtime_receipt_sha256": receipt_hash,
        "records": records,
    }


RUNS = {
    "local-alpha": _run(
        "local-alpha", "org/private-alpha", "/verified/private-alpha.gguf"
    ),
    "local-beta": _run("local-beta", "org/private-beta", "/verified/private-beta.gguf"),
}


def _failed_run(alias: str, model_id: str) -> dict[str, object]:
    backend_response_model = f"/verified/{alias}.gguf"
    run = _run(alias, model_id, backend_response_model)
    raw = "not-json"
    error = "decision is not valid JSON"
    attempts = [
        {
            "raw": raw,
            "response_body": _response_body(raw, backend_response_model),
            "logical_model_id": model_id,
            "backend_model_id": backend_response_model,
            "parsed": None,
            "error": error,
        },
        {
            "raw": raw,
            "response_body": _response_body(raw, backend_response_model),
            "logical_model_id": model_id,
            "backend_model_id": backend_response_model,
            "parsed": None,
            "error": error,
        },
    ]
    record = run["records"]["r1"]
    record["attempts"] = attempts
    record["decision"] = None
    record["failure"] = {
        "type": "invalid_decision_after_repair",
        "message": error,
    }
    record["usage"] = {"total_tokens": 24}
    return run


def _review_row(
    response_id: str,
    phase: str,
    *,
    reviewer: str = "advisor",
    label: str = "good",
) -> dict[str, object]:
    decision = dict(DECISION)
    decision["label"] = label
    if label != "good":
        decision["correctness"] = 3
    return {
        **decision,
        "reviewer": reviewer,
        "response_id": response_id,
        "review_phase": phase,
        "decisive_disagreements": (
            []
            if phase == "pass1"
            else [
                {
                    "judge_aliases": ["judge_a"],
                    "finding": "The anonymous judge overlooked the direct answer.",
                }
            ]
        ),
        "prompt_findings": (
            [] if phase == "pass1" else ["No rubric change is needed for this case."]
        ),
    }


def _reviewer_record(
    reviewer: str = "advisor", *, pass2_label: str = "good"
) -> dict[str, object]:
    return {
        "version": 1,
        "reviewer": reviewer,
        "pass1": [
            _review_row(value, "pass1", reviewer=reviewer) for value in EXPECTED_IDS
        ],
        "pass2": [
            _review_row(value, "pass2", reviewer=reviewer, label=pass2_label)
            for value in EXPECTED_IDS
        ],
    }


def test_independent_packet_contains_only_blind_case_fields() -> None:
    packet = build_independent_review_packet(CASES)

    assert packet == [
        {
            "response_id": "r1",
            "question": "Who wrote Hamlet?",
            "provisional_reference": "William Shakespeare wrote Hamlet.",
            "candidate_response": "Shakespeare wrote Hamlet.",
        },
        {
            "response_id": "r2",
            "question": "What is the boiling point of water?",
            "provisional_reference": (
                "At sea level, water boils at 100 degrees Celsius."
            ),
            "candidate_response": "At sea level it boils at 100 C.",
        },
    ]


def test_critique_packet_has_stable_anonymous_judges() -> None:
    packet, sealed = build_critique_packet(CASES, RUNS, seed=2026)
    reordered, reordered_sealed = build_critique_packet(
        CASES, dict(reversed(list(RUNS.items()))), seed=2026
    )

    assert packet == reordered
    assert sealed == reordered_sealed
    assert set(sealed) == {"version", "seed", "judge_aliases"}
    assert set(sealed["judge_aliases"].values()) == set(RUNS)
    assert all(set(row["judges"]) == {"judge_a", "judge_b"} for row in packet)
    assert [list(row["judges"]) for row in packet] == [
        list(row["judges"]) for row in reordered
    ]

    serialized = json.dumps(packet)
    for forbidden in [*RUNS, "org/private-alpha", "org/private-beta"]:
        assert forbidden not in serialized
    for row in packet:
        assert set(row) == {
            "response_id",
            "question",
            "provisional_reference",
            "candidate_response",
            "judges",
        }
        assert all(
            set(result) == {"status", "decision"}
            and result["status"] == "decision"
            and result["decision"] == DECISION
            for result in row["judges"].values()
        )


def test_critique_packet_represents_terminal_failure_without_internals() -> None:
    runs = {**RUNS, "local-failed": _failed_run("local-failed", "org/secret")}

    packet, sealed = build_critique_packet(CASES, runs, seed=2026)

    failed_alias = next(
        alias
        for alias, model_alias in sealed["judge_aliases"].items()
        if model_alias == "local-failed"
    )
    assert packet[0]["judges"][failed_alias] == {"status": "failure"}
    serialized = json.dumps(packet)
    for forbidden in ["org/secret", "invalid_decision_after_repair", "not-json"]:
        assert forbidden not in serialized


def _replace_run_rationale(runs: dict[str, object], alias: str, rationale: str) -> None:
    record = runs[alias]["records"]["r1"]
    decision = {**record["decision"], "rationale": rationale}
    raw = json.dumps(decision)
    record["decision"] = decision
    record["attempts"][0]["parsed"] = decision
    record["attempts"][0]["raw"] = raw
    record["attempts"][0]["response_body"]["choices"][0]["message"]["content"] = raw


@pytest.mark.parametrize(
    ("rationale", "caller_tokens"),
    [
        ("The org/private-alpha model gave the requested fact.", ()),
        ("The LOCAL-ALPHA result gave the requested fact.", ()),
        (
            "The checkpoint-Q4_K.gguf result gave the requested fact.",
            ("/models/checkpoint-Q4_K.gguf",),
        ),
        (
            "The VLLM-ROCM runtime gave the requested fact.",
            ("vllm-rocm",),
        ),
        (
            "The server result gave the requested fact.",
            ("/opt/private-runtime/bin/server",),
        ),
    ],
)
def test_critique_packet_rejects_identity_text_in_valid_decisions(
    rationale: str, caller_tokens: tuple[str, ...]
) -> None:
    runs = json.loads(json.dumps(RUNS))
    _replace_run_rationale(runs, "local-alpha", rationale)

    with pytest.raises(ValueError, match="forbidden identity text") as error:
        build_critique_packet(
            CASES,
            runs,
            seed=2026,
            forbidden_identity_tokens=caller_tokens,
        )

    assert "local-alpha" not in str(error.value).casefold()
    assert "private-alpha" not in str(error.value).casefold()
    assert "vllm" not in str(error.value).casefold()
    assert "checkpoint" not in str(error.value).casefold()


def test_critique_packet_allows_benign_rationale_with_generic_words() -> None:
    runs = json.loads(json.dumps(RUNS))
    _replace_run_rationale(
        runs,
        "local-alpha",
        "An observer's local assessment uses a private source and gives the right fact.",
    )

    packet, _ = build_critique_packet(
        CASES,
        runs,
        seed=2026,
        forbidden_identity_tokens=(
            "/models/checkpoint-Q4_K.gguf",
            "vllm-rocm",
            "/opt/private-runtime/bin/server",
        ),
    )

    assert packet[0]["judges"]


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_record",
        "extra_record",
        "wrong_input",
        "wrong_container_alias",
        "wrong_record_prompt",
        "partial_record",
        "corrupt_attempt",
    ],
)
def test_critique_packet_rejects_incomplete_or_corrupt_runs(mutation: str) -> None:
    runs = json.loads(json.dumps(RUNS))
    run = runs["local-alpha"]
    if mutation == "missing_record":
        del run["records"]["r2"]
    elif mutation == "extra_record":
        run["records"]["other"] = dict(run["records"]["r1"])
    elif mutation == "wrong_input":
        run["records"]["r1"]["input_sha256"] = "0" * 64
    elif mutation == "wrong_container_alias":
        run["model_alias"] = "secret-alias"
    elif mutation == "wrong_record_prompt":
        run["records"]["r1"]["prompt_sha256"] = "2" * 64
    elif mutation == "partial_record":
        record = run["records"]["r1"]
        record["attempts"][0]["parsed"] = None
        record["attempts"][0]["error"] = "unexpected"
        record["decision"] = None
    else:
        run["records"]["r1"]["attempts"][0]["response_body"]["model"] = "other"

    with pytest.raises(
        ValueError, match="run|record|coverage|terminal|identity|attempt"
    ):
        build_critique_packet(CASES, runs, seed=2026)


def test_critique_packet_rejects_record_with_foreign_backend_model() -> None:
    runs = json.loads(json.dumps(RUNS))
    attempt = runs["local-alpha"]["records"]["r2"]["attempts"][0]
    attempt["backend_model_id"] = "foreign-model"
    attempt["response_body"]["model"] = "foreign-model"

    with pytest.raises(
        ValueError, match="invalid run local-alpha record:.*wrong model"
    ):
        build_critique_packet(CASES, runs, seed=2026)


def test_reviewer_coverage_requires_every_case_once() -> None:
    rows = [_review_row(value, "pass1") for value in EXPECTED_IDS]

    with pytest.raises(ValueError, match="missing"):
        validate_reviewer_coverage(EXPECTED_IDS, rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        validate_reviewer_coverage(EXPECTED_IDS, [*rows, rows[0]])
    with pytest.raises(ValueError, match="unknown"):
        validate_reviewer_coverage(EXPECTED_IDS, [*rows, _review_row("other", "pass1")])


def test_reviewer_record_requires_exact_typed_two_pass_ledgers() -> None:
    record = _reviewer_record()

    validate_reviewer_record(
        record,
        EXPECTED_IDS,
        allowed_reviewers={"advisor", "claude"},
        expected_reviewer="advisor",
        judge_aliases={"judge_a", "judge_b"},
        forbidden_identities={"local-alpha", "org/private-alpha"},
    )

    wrong = json.loads(json.dumps(record))
    wrong["pass2"][0]["correctness"] = True
    with pytest.raises(ValueError, match="correctness"):
        validate_reviewer_record(wrong, EXPECTED_IDS, judge_aliases={"judge_a"})

    extra = json.loads(json.dumps(record))
    extra["pass1"][0]["extra"] = 1
    with pytest.raises(ValueError, match="exact keys"):
        validate_reviewer_record(extra, EXPECTED_IDS, judge_aliases={"judge_a"})


def test_reviewer_record_rejects_identity_leaks_unknown_aliases_and_unbounded_text() -> (
    None
):
    leaked = _reviewer_record()
    leaked["pass2"][0]["prompt_findings"] = ["local-alpha ignored the rubric."]
    with pytest.raises(ValueError, match="identity"):
        validate_reviewer_record(
            leaked,
            EXPECTED_IDS,
            judge_aliases={"judge_a", "judge_b"},
            forbidden_identities={"local-alpha"},
        )

    rationale_leak = _reviewer_record()
    rationale_leak["pass2"][0]["rationale"] = (
        "The local-alpha result supplies the right answer directly."
    )
    with pytest.raises(ValueError, match="identity"):
        validate_reviewer_record(
            rationale_leak,
            EXPECTED_IDS,
            judge_aliases={"judge_a", "judge_b"},
            forbidden_identities={"local-alpha"},
        )

    unknown = _reviewer_record()
    unknown["pass2"][0]["decisive_disagreements"][0]["judge_aliases"] = ["judge_z"]
    with pytest.raises(ValueError, match="unknown judge alias"):
        validate_reviewer_record(
            unknown, EXPECTED_IDS, judge_aliases={"judge_a", "judge_b"}
        )

    unbounded = _reviewer_record()
    unbounded["pass2"][0]["prompt_findings"] = ["word " * 121]
    with pytest.raises(ValueError, match="1 to 120 words"):
        validate_reviewer_record(
            unbounded, EXPECTED_IDS, judge_aliases={"judge_a", "judge_b"}
        )


def test_parse_reviewer_record_rejects_duplicate_keys_and_nonfinite_values() -> None:
    raw = json.dumps(_reviewer_record()).replace(
        '"version": 1', '"version": 1, "version": 1', 1
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_reviewer_record(raw)
    with pytest.raises(ValueError, match="non-finite"):
        parse_reviewer_record('{"value": NaN}')


def test_compare_reviews_is_deterministic_and_type_safe() -> None:
    advisor = _reviewer_record("advisor")
    claude = _reviewer_record("claude", pass2_label="partial")

    first = compare_reviews(
        advisor,
        claude,
        EXPECTED_IDS,
        allowed_reviewers={"advisor", "claude"},
        judge_aliases={"judge_a", "judge_b"},
    )
    second = compare_reviews(
        claude,
        advisor,
        list(reversed(EXPECTED_IDS)),
        allowed_reviewers={"advisor", "claude"},
        judge_aliases={"judge_b", "judge_a"},
    )

    assert first == second
    assert [row["response_id"] for row in first] == sorted(EXPECTED_IDS)
    assert all(row["pass1_label_agreement"] is True for row in first)
    assert all(row["pass2_label_agreement"] is False for row in first)
    assert all(row["unresolved"] is True for row in first)
    assert all(
        [review["reviewer"] for review in row["reviews"]] == ["advisor", "claude"]
        for row in first
    )

    malformed = _reviewer_record("claude")
    malformed["pass2"][0]["relevance"] = 2.0
    with pytest.raises(ValueError, match="relevance"):
        compare_reviews(
            advisor,
            malformed,
            EXPECTED_IDS,
            judge_aliases={"judge_a", "judge_b"},
        )


def _manifest() -> dict[str, object]:
    body = {
        "version": 1,
        "seed": 2026,
        "sources": {
            "discovery": {"sha256": "a" * 64},
            "grounded": {"sha256": "b" * 64},
        },
        "selection_reasons": [case.selection_reason for case in CASES],
        "cases": [asdict(case) for case in CASES],
    }
    # The CLI test uses the Task-2 loader contract, which expects all ten
    # reasons. The focused builder tests above intentionally stay small.
    from trec_mlm.pointwise_sampling import DIAGNOSTIC_REASONS

    cases = []
    for index, reason in enumerate(DIAGNOSTIC_REASONS):
        case = replace(
            CASES[index % len(CASES)],
            question_id=f"q{index}",
            response_id=f"r{index}",
            strata=(reason,),
            selection_reason=reason,
        )
        cases.append(case)
    body["selection_reasons"] = list(DIAGNOSTIC_REASONS)
    body["cases"] = [asdict(case) for case in cases]
    return {**body, "manifest_hash": _json_hash(body)}


def _write_private_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    encoded = (json.dumps(value, indent=2) + "\n").encode()
    path.write_bytes(encoded)
    path.chmod(0o600)
    return hashlib.sha256(encoded).hexdigest()


def test_packet_cli_reads_anchored_private_inputs_and_writes_private_no_overwrite(
    tmp_path: Path,
) -> None:
    manifest = _manifest()
    cases = [
        CandidateCase(**{**row, "strata": tuple(row["strata"])})
        for row in manifest["cases"]
    ]
    runs = {
        alias: {
            **run,
            "records": {
                case.response_id: {
                    **next(iter(run["records"].values())),
                    "input_sha256": _json_hash(asdict(case)),
                }
                for case in cases
            },
        }
        for alias, run in RUNS.items()
    }
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    manifest_path = root / "smoke_manifest.json"
    manifest_hash = _write_private_json(manifest_path, manifest)
    runs_dir = root / "runs"
    runs_dir.mkdir(mode=0o700)
    run_hashes = {
        alias: _write_private_json(runs_dir / f"{alias}.json", run)
        for alias, run in runs.items()
    }
    output = root / "review"
    argv = [
        "packets",
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        manifest_hash,
        "--runs-dir",
        str(runs_dir),
        "--output-dir",
        str(output),
        "--seed",
        "2026",
    ]
    for alias, digest in run_hashes.items():
        argv.extend(["--run-sha256", f"{alias}={digest}"])

    assert review_main(argv) == 0
    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    expected_names = {
        "independent-review-packet.json",
        "critique-review-packet.json",
        "sealed-judge-aliases.json",
    }
    assert {path.name for path in output.iterdir()} == expected_names
    for path in output.iterdir():
        assert stat.S_ISREG(path.stat().st_mode)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.stat().st_nlink == 1
    first = {path.name: path.read_bytes() for path in output.iterdir()}
    assert review_main(argv) == 0
    assert {path.name: path.read_bytes() for path in output.iterdir()} == first

    critique = json.loads((output / "critique-review-packet.json").read_text())
    serialized = json.dumps(critique)
    for forbidden in [*runs, "org/private-alpha", "org/private-beta"]:
        assert forbidden not in serialized


def test_packet_cli_validates_every_input_before_output_mutation(
    tmp_path: Path,
) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    manifest_path = root / "smoke_manifest.json"
    manifest_hash = _write_private_json(manifest_path, _manifest())
    runs_dir = root / "runs"
    runs_dir.mkdir(mode=0o700)
    bad_run = runs_dir / "local-alpha.json"
    run_hash = _write_private_json(bad_run, {"not": "a run"})
    output = root / "review"

    with pytest.raises(ValueError, match="run"):
        review_main(
            [
                "packets",
                "--manifest",
                str(manifest_path),
                "--manifest-sha256",
                manifest_hash,
                "--runs-dir",
                str(runs_dir),
                "--run-sha256",
                f"local-alpha={run_hash}",
                "--output-dir",
                str(output),
            ]
        )

    assert not output.exists()


@pytest.mark.parametrize("unsafe", ["symlink", "mode", "oversize"])
def test_packet_cli_rejects_unsafe_private_input(tmp_path: Path, unsafe: str) -> None:
    root = tmp_path / "private"
    root.mkdir(mode=0o700)
    real = root / "real.json"
    manifest_hash = _write_private_json(real, _manifest())
    manifest = real
    if unsafe == "symlink":
        manifest = root / "manifest.json"
        manifest.symlink_to(real)
    elif unsafe == "mode":
        real.chmod(0o644)
    else:
        real.write_bytes(b" " * (4 * 1024 * 1024 + 1))
        real.chmod(0o600)
        manifest_hash = hashlib.sha256(real.read_bytes()).hexdigest()

    with pytest.raises(ValueError, match="safely|0600|too large"):
        review_main(
            [
                "packets",
                "--manifest",
                str(manifest),
                "--manifest-sha256",
                manifest_hash,
                "--runs-dir",
                str(root),
                "--run-sha256",
                f"local-alpha={'0' * 64}",
                "--output-dir",
                str(root / "review"),
            ]
        )
