from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import stat

import pytest

from scripts.build_pointwise_judge_100_comparison import (
    _validate_runtime_trust,
    build_comparison as _build_comparison,
    main,
)
from scripts.run_lemonade_pointwise_smoke import (
    _runtime_observation as _current_runtime_observation,
)
from trec_mlm.pointwise_judge import CandidateCase, parse_judge_decision
from trec_mlm.pointwise_runner import FROZEN_DECISION_SCHEMA_SHA256
from trec_mlm.pointwise_sampling import DIAGNOSTIC_REASONS


CASES = [
    CandidateCase(
        question_id=f"question-{index}",
        response_id=f"response-{index}",
        question=f"Question {index}?",
        reference_answer=f"Reference {index}.",
        candidate_answer=f"Answer {index}.",
        strata=(reason, "shared") if index < 3 else (reason,),
        selection_reason=reason,
    )
    for index, reason in enumerate(("reason-a", "reason-b", "reason-c", "reason-d"), 1)
]
SEALED_CANDIDATE_IDS = [f"sealed-candidate-{index}" for index in range(1, 5)]
ALIAS_SECRET = bytes.fromhex("ab" * 32)
CURRENT_SCHEMA = json.loads(
    Path("protocols/pointwise_judge_bakeoff/decision-v1.schema.json").read_text(
        encoding="utf-8"
    )
)


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _decision(label: str) -> dict[str, object]:
    score = {"incorrect": 0, "partial": 2, "good": 4, "uncertain": 2}[label]
    return {
        "label": label,
        "correctness": score,
        "completeness": 2,
        "relevance": 2,
        "error_flags": [],
        "reference_status": "sound",
        "confidence": "medium",
        "rationale": f"Private rationale for the {label} judgment.",
    }


def _attempt(
    model_id: str,
    backend_model_id: str,
    raw: str,
    parsed: dict[str, object] | None,
    total_tokens: int,
) -> dict[str, object]:
    value: dict[str, object] = {
        "raw": raw,
        "response_body": {
            "model": backend_model_id,
            "choices": [{"message": {"content": raw}}],
            "usage": {
                "prompt_tokens": total_tokens - 1,
                "completion_tokens": 1,
                "total_tokens": total_tokens,
            },
        },
        "logical_model_id": model_id,
        "backend_model_id": backend_model_id,
        "parsed": parsed,
    }
    if parsed is None:
        with pytest.raises(ValueError) as error:
            parse_judge_decision(raw)
        value["error"] = str(error.value)
    return value


def _record(
    case: CandidateCase,
    *,
    alias: str,
    model_id: str,
    receipt: str,
    label: str | None,
    elapsed: float,
    total_tokens: int,
    backend_model_id: str,
) -> dict[str, object]:
    identity = {
        "input_sha256": _json_hash(asdict(case)),
        "prompt_sha256": "1" * 64,
        "schema_sha256": FROZEN_DECISION_SCHEMA_SHA256,
        "model_alias": alias,
        "model_id": model_id,
        "runtime_receipt_sha256": receipt,
    }
    if label is not None:
        decision = _decision(label)
        attempts = [
            _attempt(
                model_id,
                backend_model_id,
                json.dumps(decision),
                decision,
                total_tokens,
            )
        ]
        failure = None
    else:
        first = _attempt(model_id, backend_model_id, "not json one", None, total_tokens)
        second = _attempt(
            model_id, backend_model_id, "not json two", None, total_tokens
        )
        attempts = [first, second]
        decision = None
        failure = {
            "type": "invalid_decision_after_repair",
            "message": second["error"],
        }
    return {
        **identity,
        "attempts": attempts,
        "decision": decision,
        "failure": failure,
        "elapsed_seconds": elapsed,
        "usage": {
            "prompt_tokens": (total_tokens - 1) * len(attempts),
            "completion_tokens": len(attempts),
            "total_tokens": total_tokens * len(attempts),
        },
    }


def _run(
    alias: str,
    model_id: str,
    labels: list[str | None],
    elapsed: list[float],
    tokens: list[int],
    *,
    cases: list[CandidateCase] = CASES,
) -> dict[str, object]:
    backend_model_id = "/private/shared-model.gguf"
    observation = _runtime_observation(
        backend_model_id, "thinking" in alias, backend_model=model_id
    )
    receipt = _json_hash(observation)
    return {
        "version": 1,
        "model_alias": alias,
        "model_id": model_id,
        "prompt_sha256": "1" * 64,
        "schema_sha256": FROZEN_DECISION_SCHEMA_SHA256,
        "runtime_receipt_sha256": receipt,
        "records": {
            case.response_id: _record(
                case,
                alias=alias,
                model_id=model_id,
                receipt=receipt,
                label=label,
                elapsed=seconds,
                total_tokens=token_count,
                backend_model_id=backend_model_id,
            )
            for case, label, seconds, token_count in zip(
                cases, labels, elapsed, tokens, strict=True
            )
        },
    }


def _runtime_observation(
    backend_model_id: str,
    thinking: bool,
    *,
    backend_model: str = "org/private-shared-model",
) -> dict[str, object]:
    return {
        "schema_sha256": FROZEN_DECISION_SCHEMA_SHA256,
        "transport": "lemonade-11.5-llamacpp-rocm",
        "backend_model": backend_model,
        "backend_response_model": backend_model_id,
        "thinking": thinking,
        "thinking_control_omitted": False,
        "reasoning_effort": None,
        "max_output_tokens": 4096,
        "schema_transport_adjustment": (
            "rationale.maxLength and error_flags.uniqueItems omitted; frozen parser retained"
        ),
    }


def _rebind_run_observation(
    run: dict[str, object], observation: dict[str, object]
) -> None:
    receipt = _json_hash(observation)
    run["runtime_receipt_sha256"] = receipt
    for record in run["records"].values():
        record["runtime_receipt_sha256"] = receipt


def build_comparison(*args: object, **kwargs: object) -> dict[str, object]:
    direct = args[1]
    thinking = args[2]
    input_hashes = dict(kwargs["input_hashes"])
    input_hashes.update(
        {
            "direct_runtime_observation": "e" * 64,
            "thinking_runtime_observation": "f" * 64,
            "runtime_binding": "9" * 64,
        }
    )
    observations = {
        "direct": _runtime_observation("/private/shared-model.gguf", False),
        "thinking": _runtime_observation("/private/shared-model.gguf", True),
    }
    binding = {
        "version": 1,
        "shared_configuration": {
            "context_tokens": 8192,
            "max_output_tokens": 4096,
            "temperature": 0,
        },
        "roles": {
            role: {
                "run_sha256": input_hashes[f"{role}_run"],
                "runtime_receipt_sha256": run["runtime_receipt_sha256"],
                "thinking": role == "thinking",
            }
            for role, run in (("direct", direct), ("thinking", thinking))
        },
    }
    return _build_comparison(
        *args,
        **{
            **kwargs,
            "input_hashes": input_hashes,
            "runtime_observations": observations,
            "runtime_binding": binding,
        },
    )


def _inputs() -> tuple[dict[str, object], dict[str, object]]:
    direct = _run(
        "private-direct-alias",
        "org/private-shared-model",
        ["incorrect", "partial", "good", None],
        [1.0, 2.0, 3.0, 4.0],
        [10, 20, 30, 40],
    )
    thinking = _run(
        "private-thinking-alias",
        "org/private-shared-model",
        ["partial", "good", "incorrect", "good"],
        [10.0, 20.0, 30.0, 40.0],
        [11, 21, 31, 41],
    )
    return direct, thinking


def test_comparison_reports_direction_strata_reused_slice_costs_and_failures() -> None:
    direct, thinking = _inputs()

    artifacts = build_comparison(
        CASES,
        direct,
        thinking,
        reused_response_ids=[CASES[0].response_id],
        input_hashes={
            "manifest": "a" * 64,
            "sealed_candidates": "d" * 64,
            "direct_run": "b" * 64,
            "thinking_run": "c" * 64,
        },
        sealed_candidate_ids=SEALED_CANDIDATE_IDS,
        alias_secret=ALIAS_SECRET,
        seed=2026,
    )
    comparison = artifacts["comparison"]

    assert comparison["agreement"] == {
        "paired_valid_decisions": 3,
        "exact_count": 0,
        "exact_rate": 0.0,
        "within_one_label_count": 2,
        "within_one_label_rate": 2 / 3,
        "nonadjacent_count": 1,
    }
    assert comparison["confusion_matrix"]["counts"] == [
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 0, 0, 0],
    ]
    assert comparison["transitions"] == [
        {"direct_label": "incorrect", "thinking_label": "partial", "count": 1},
        {"direct_label": "partial", "thinking_label": "good", "count": 1},
        {"direct_label": "good", "thinking_label": "incorrect", "count": 1},
    ]
    assert comparison["per_reason"]["reason-a"]["transitions"] == {
        "incorrect_to_partial": 1
    }
    assert comparison["per_stratum"]["shared"]["case_count"] == 2
    assert comparison["per_stratum"]["shared"]["paired_valid_decisions"] == 2
    assert comparison["reused_slice"]["case_count"] == 1
    assert comparison["reused_slice"]["within_one_label_count"] == 1
    assert comparison["runs"]["direct"]["failed_records"] == 1
    assert comparison["runs"]["direct"]["valid_decisions"] == 3
    assert comparison["runs"]["direct"]["elapsed_seconds_total"] == 10.0
    assert comparison["runs"]["direct"]["usage"]["total_tokens"] == 140
    assert comparison["runs"]["thinking"]["elapsed_seconds_total"] == 100.0
    assert comparison["runs"]["thinking"]["usage"]["total_tokens"] == 104
    assert comparison["case_differences"][-1]["direct_status"] == "failure"
    assert comparison["case_differences"][-1]["thinking_status"] == "decision"
    assert "correct" not in comparison["headline"].casefold()


def test_uncertain_is_exact_only_and_nonordinal_for_adjacency() -> None:
    direct = _run(
        "private-direct-alias",
        "org/private-shared-model",
        ["uncertain", "uncertain", "partial", "good"],
        [1.0] * 4,
        [10] * 4,
    )
    thinking = _run(
        "private-thinking-alias",
        "org/private-shared-model",
        ["uncertain", "partial", "uncertain", "good"],
        [1.0] * 4,
        [10] * 4,
    )

    comparison = build_comparison(
        CASES,
        direct,
        thinking,
        reused_response_ids=[],
        input_hashes={
            "manifest": "a" * 64,
            "sealed_candidates": "d" * 64,
            "direct_run": "b" * 64,
            "thinking_run": "c" * 64,
        },
        sealed_candidate_ids=SEALED_CANDIDATE_IDS,
        alias_secret=ALIAS_SECRET,
    )["comparison"]

    assert comparison["agreement"] == {
        "paired_valid_decisions": 4,
        "exact_count": 2,
        "exact_rate": 0.5,
        "within_one_label_count": 2,
        "within_one_label_rate": 0.5,
        "nonadjacent_count": 2,
    }
    assert comparison["confusion_matrix"]["counts"][3] == [0, 1, 0, 1]


def test_comparison_pins_one_backend_for_every_record() -> None:
    direct, thinking = _inputs()
    attempt = direct["records"][CASES[1].response_id]["attempts"][0]
    attempt["backend_model_id"] = "foreign-model"
    attempt["response_body"]["model"] = "foreign-model"

    with pytest.raises(ValueError, match="wrong model"):
        build_comparison(
            CASES,
            direct,
            thinking,
            reused_response_ids=[],
            input_hashes={
                "manifest": "a" * 64,
                "sealed_candidates": "d" * 64,
                "direct_run": "b" * 64,
                "thinking_run": "c" * 64,
            },
            sealed_candidate_ids=SEALED_CANDIDATE_IDS,
            alias_secret=ALIAS_SECRET,
        )


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("swapped", "observation|thinking role"),
        ("duplicate", "distinct"),
        ("context", "frozen shared configuration"),
        ("temperature", "frozen shared configuration"),
        ("role", "prove direct role"),
    ],
)
def test_runtime_trust_rejects_swaps_duplicates_and_setting_drift(
    mutation: str, message: str
) -> None:
    direct, thinking = _inputs()
    hashes = {
        "manifest": "a" * 64,
        "sealed_candidates": "d" * 64,
        "direct_run": "b" * 64,
        "thinking_run": "c" * 64,
        "direct_runtime_observation": "e" * 64,
        "thinking_runtime_observation": "f" * 64,
        "runtime_binding": "9" * 64,
    }
    observations = {
        "direct": _runtime_observation("/private/shared-model.gguf", False),
        "thinking": _runtime_observation("/private/shared-model.gguf", True),
    }
    binding = {
        "version": 1,
        "shared_configuration": {
            "context_tokens": 8192,
            "max_output_tokens": 4096,
            "temperature": 0,
        },
        "roles": {
            "direct": {
                "run_sha256": hashes["direct_run"],
                "runtime_receipt_sha256": direct["runtime_receipt_sha256"],
                "thinking": False,
            },
            "thinking": {
                "run_sha256": hashes["thinking_run"],
                "runtime_receipt_sha256": thinking["runtime_receipt_sha256"],
                "thinking": True,
            },
        },
    }
    if mutation == "swapped":
        observations["direct"], observations["thinking"] = (
            observations["thinking"],
            observations["direct"],
        )
    elif mutation == "duplicate":
        hashes["thinking_run"] = hashes["direct_run"]
    elif mutation == "context":
        binding["shared_configuration"]["context_tokens"] = 4096
    elif mutation == "temperature":
        binding["shared_configuration"]["temperature"] = 1
    else:
        binding["roles"]["direct"]["thinking"] = True

    with pytest.raises(ValueError, match=message):
        _validate_runtime_trust(
            direct,
            thinking,
            runtime_observations=observations,
            runtime_binding=binding,
            input_hashes=hashes,
        )


def _current_runtime_trust_fixture() -> tuple[
    dict[str, object],
    dict[str, object],
    dict[str, dict[str, object]],
    dict[str, str],
    dict[str, object],
]:
    model_id = "org/private-shared-model"
    backend = "/private/shared-model.gguf"
    direct = _run(
        "private-direct-alias",
        model_id,
        ["incorrect", "partial", "good", "uncertain"],
        [1.0] * 4,
        [10] * 4,
    )
    thinking = _run(
        "private-thinking-alias",
        model_id,
        ["partial", "good", "incorrect", "uncertain"],
        [1.0] * 4,
        [10] * 4,
    )
    observations = {
        role: _current_runtime_observation(
            CURRENT_SCHEMA,
            backend_model=model_id,
            backend_response_model=backend,
            thinking=role == "thinking",
            thinking_control_omitted=False,
            reasoning_effort=None,
            max_output_tokens=4096,
        )
        for role in ("direct", "thinking")
    }
    _rebind_run_observation(direct, observations["direct"])
    _rebind_run_observation(thinking, observations["thinking"])
    hashes = {
        "manifest": "a" * 64,
        "sealed_candidates": "d" * 64,
        "direct_run": "b" * 64,
        "thinking_run": "c" * 64,
        "direct_runtime_observation": "e" * 64,
        "thinking_runtime_observation": "f" * 64,
        "runtime_binding": "9" * 64,
    }
    binding = {
        "version": 1,
        "shared_configuration": {
            "context_tokens": 8192,
            "max_output_tokens": 4096,
            "temperature": 0,
        },
        "roles": {
            role: {
                "run_sha256": hashes[f"{role}_run"],
                "runtime_receipt_sha256": run["runtime_receipt_sha256"],
                "thinking": role == "thinking",
            }
            for role, run in (("direct", direct), ("thinking", thinking))
        },
    }
    return direct, thinking, observations, hashes, binding


def test_comparison_accepts_current_schema_hashed_runtime_observations() -> None:
    direct, thinking, observations, hashes, binding = _current_runtime_trust_fixture()

    comparison = _build_comparison(
        CASES,
        direct,
        thinking,
        reused_response_ids=[],
        input_hashes=hashes,
        sealed_candidate_ids=SEALED_CANDIDATE_IDS,
        alias_secret=ALIAS_SECRET,
        runtime_observations=observations,
        runtime_binding=binding,
    )["comparison"]

    assert comparison["coverage"]["case_count"] == 4


@pytest.mark.parametrize(
    "field,bad_value,message",
    [
        ("transport", "foreign-transport", "transport is invalid"),
        ("backend_model", "foreign-logical-model", "backend model"),
        ("reasoning_effort", "low", "reasoning effort"),
        ("max_output_tokens", 4096.0, "output allowance"),
        (
            "schema_transport_adjustment",
            "some other adjustment",
            "transport schema adjustment",
        ),
        ("transport_schema_sha256", "0" * 64, "transport schema hash"),
    ],
)
def test_current_runtime_trust_rejects_shared_field_drift(
    field: str, bad_value: object, message: str
) -> None:
    direct, thinking, observations, hashes, binding = _current_runtime_trust_fixture()
    observations["thinking"][field] = bad_value
    _rebind_run_observation(thinking, observations["thinking"])
    binding["roles"]["thinking"]["runtime_receipt_sha256"] = thinking[
        "runtime_receipt_sha256"
    ]

    with pytest.raises(ValueError, match=message):
        _validate_runtime_trust(
            direct,
            thinking,
            runtime_observations=observations,
            runtime_binding=binding,
            input_hashes=hashes,
        )


def test_runtime_trust_rejects_mixed_legacy_and_current_observation_formats() -> None:
    direct, thinking, observations, hashes, binding = _current_runtime_trust_fixture()
    observations["direct"].pop("transport_schema_sha256")
    observations["direct"]["schema_transport_adjustment"] = (
        "rationale.maxLength and error_flags.uniqueItems omitted; frozen parser retained"
    )
    _rebind_run_observation(direct, observations["direct"])
    binding["roles"]["direct"]["runtime_receipt_sha256"] = direct[
        "runtime_receipt_sha256"
    ]

    with pytest.raises(ValueError, match="same runtime observation format"):
        _validate_runtime_trust(
            direct,
            thinking,
            runtime_observations=observations,
            runtime_binding=binding,
            input_hashes=hashes,
        )


def test_review_packets_hide_mode_and_model_identity() -> None:
    direct, thinking = _inputs()

    artifacts = build_comparison(
        CASES,
        direct,
        thinking,
        reused_response_ids=[CASES[0].response_id],
        input_hashes={
            "manifest": "a" * 64,
            "sealed_candidates": "d" * 64,
            "direct_run": "b" * 64,
            "thinking_run": "c" * 64,
        },
        sealed_candidate_ids=SEALED_CANDIDATE_IDS,
        alias_secret=ALIAS_SECRET,
        seed=2026,
    )
    packet_text = json.dumps(
        {
            "independent": artifacts["independent_review_packet"],
            "critique": artifacts["critique_review_packet"],
        },
        sort_keys=True,
    ).casefold()

    for forbidden in (
        "private-direct",
        "private-thinking",
        "org/private",
        "direct_run",
        "thinking_run",
        "response-",
        "question-",
        "model_alias",
        "model_id",
    ):
        assert forbidden not in packet_text
    assert set(artifacts["sealed_mode_aliases"]["judge_aliases"].values()) == {
        "direct",
        "thinking",
    }
    assert artifacts["sealed_mode_aliases"]["alias_secret_hex"] == ALIAS_SECRET.hex()


def _replace_rationale(
    run: dict[str, object], response_id: str, rationale: str
) -> None:
    record = run["records"][response_id]
    decision = {**record["decision"], "rationale": rationale}
    raw = json.dumps(decision)
    record["decision"] = decision
    record["attempts"][0]["parsed"] = decision
    record["attempts"][0]["raw"] = raw
    record["attempts"][0]["response_body"]["choices"][0]["message"]["content"] = raw


@pytest.mark.parametrize(
    "echoed_identity",
    [
        CASES[0].question_id,
        CASES[0].response_id,
        SEALED_CANDIDATE_IDS[0],
    ],
)
def test_valid_judge_rationale_echoing_source_identity_fails_closed(
    echoed_identity: str,
) -> None:
    direct, thinking = _inputs()
    _replace_rationale(
        thinking,
        CASES[0].response_id,
        f"The valid judgment repeats source identifier {echoed_identity}.",
    )

    with pytest.raises(ValueError, match="source identity") as error:
        build_comparison(
            CASES,
            direct,
            thinking,
            reused_response_ids=[CASES[0].response_id],
            input_hashes={
                "manifest": "a" * 64,
                "sealed_candidates": "d" * 64,
                "direct_run": "b" * 64,
                "thinking_run": "c" * 64,
            },
            sealed_candidate_ids=SEALED_CANDIDATE_IDS,
            alias_secret=ALIAS_SECRET,
            seed=2026,
        )

    assert echoed_identity not in str(error.value)


@pytest.mark.parametrize(
    "escaped_identity",
    ['private"candidate', "private\\candidate", "private\ncandidate"],
)
def test_valid_rationale_cannot_hide_source_identity_behind_json_escaping(
    escaped_identity: str,
) -> None:
    direct, thinking = _inputs()
    sealed_ids = [escaped_identity, *SEALED_CANDIDATE_IDS[1:]]
    _replace_rationale(
        thinking,
        CASES[0].response_id,
        f"The valid rationale echoes {escaped_identity} exactly.",
    )

    with pytest.raises(ValueError, match="source identity") as error:
        build_comparison(
            CASES,
            direct,
            thinking,
            reused_response_ids=[CASES[0].response_id],
            input_hashes={
                "manifest": "a" * 64,
                "sealed_candidates": "d" * 64,
                "direct_run": "b" * 64,
                "thinking_run": "c" * 64,
            },
            sealed_candidate_ids=sealed_ids,
            alias_secret=ALIAS_SECRET,
            seed=2026,
        )

    assert escaped_identity not in str(error.value)


def test_source_identity_matching_preserves_case_sensitive_identifier_semantics() -> (
    None
):
    direct, thinking = _inputs()
    sealed_ids = ["SEALED-CANDIDATE-1", *SEALED_CANDIDATE_IDS[1:]]
    _replace_rationale(
        thinking,
        CASES[0].response_id,
        "The natural-language rationale contains sealed-candidate-1 in another case.",
    )

    artifacts = build_comparison(
        CASES,
        direct,
        thinking,
        reused_response_ids=[CASES[0].response_id],
        input_hashes={
            "manifest": "a" * 64,
            "sealed_candidates": "d" * 64,
            "direct_run": "b" * 64,
            "thinking_run": "c" * 64,
        },
        sealed_candidate_ids=sealed_ids,
        alias_secret=ALIAS_SECRET,
        seed=2026,
    )

    assert artifacts["critique_review_packet"]


def test_private_alias_secret_not_public_seed_controls_mapping_and_reproduction() -> (
    None
):
    direct, thinking = _inputs()

    outputs = [
        build_comparison(
            CASES,
            direct,
            thinking,
            reused_response_ids=[CASES[0].response_id],
            input_hashes={
                "manifest": "a" * 64,
                "sealed_candidates": "d" * 64,
                "direct_run": "b" * 64,
                "thinking_run": "c" * 64,
            },
            sealed_candidate_ids=SEALED_CANDIDATE_IDS,
            alias_secret=bytes([value]) * 32,
            seed=2026,
        )
        for value in range(8)
    ]
    mappings = {
        json.dumps(output["sealed_mode_aliases"]["judge_aliases"], sort_keys=True)
        for output in outputs
    }

    assert len(mappings) == 2
    repeated = build_comparison(
        CASES,
        direct,
        thinking,
        reused_response_ids=[CASES[0].response_id],
        input_hashes={
            "manifest": "a" * 64,
            "sealed_candidates": "d" * 64,
            "direct_run": "b" * 64,
            "thinking_run": "c" * 64,
        },
        sealed_candidate_ids=SEALED_CANDIDATE_IDS,
        alias_secret=bytes([0]) * 32,
        seed=2026,
    )
    assert repeated["critique_review_packet"] == outputs[0]["critique_review_packet"]
    assert repeated["sealed_mode_aliases"] == outputs[0]["sealed_mode_aliases"]


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_coverage", "coverage"),
        ("wrong_input", "input"),
        ("wrong_prompt", "prompt"),
        ("wrong_schema", "schema"),
    ],
)
def test_comparison_fails_closed_on_coverage_or_binding_mismatch(
    mutation: str, message: str
) -> None:
    direct, thinking = _inputs()
    if mutation == "missing_coverage":
        del thinking["records"][CASES[-1].response_id]
    elif mutation == "wrong_input":
        thinking["records"][CASES[0].response_id]["input_sha256"] = "f" * 64
    elif mutation == "wrong_prompt":
        thinking["prompt_sha256"] = "2" * 64
    else:
        thinking["schema_sha256"] = "3" * 64

    with pytest.raises(ValueError, match=message):
        build_comparison(
            CASES,
            direct,
            thinking,
            reused_response_ids=[CASES[0].response_id],
            input_hashes={
                "manifest": "a" * 64,
                "sealed_candidates": "d" * 64,
                "direct_run": "b" * 64,
                "thinking_run": "c" * 64,
            },
            sealed_candidate_ids=SEALED_CANDIDATE_IDS,
            alias_secret=ALIAS_SECRET,
            seed=2026,
        )


def _canonical_cases() -> list[CandidateCase]:
    return [
        replace(
            CASES[index % len(CASES)],
            question_id=f"canonical-question-{index:03d}",
            response_id=f"canonical-response-{index:03d}",
            question=f"Canonical private question {index}?",
            reference_answer=f"Canonical private reference {index}.",
            candidate_answer=f"Canonical private answer {index}.",
            strata=(reason,),
            selection_reason=reason,
        )
        for reason_index, reason in enumerate(DIAGNOSTIC_REASONS)
        for index in range(reason_index * 10, reason_index * 10 + 10)
    ]


def _write_private_json(path: Path, value: object) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    encoded = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    path.write_bytes(encoded)
    path.chmod(0o600)
    return hashlib.sha256(encoded).hexdigest()


def _cli_inputs(tmp_path: Path) -> tuple[list[str], Path]:
    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    cases = _canonical_cases()
    body = {
        "version": 2,
        "seed": 2026,
        "sources": {
            "discovery": {"sha256": "d" * 64},
            "grounded": {"sha256": "e" * 64},
            "smoke_manifest": {"sha256": "f" * 64},
        },
        "selection_reasons": list(DIAGNOSTIC_REASONS),
        "cases_per_reason": 10,
        "reused_smoke_response_ids": [
            cases[index].response_id for index in range(0, 100, 10)
        ],
        "cases": [asdict(case) for case in cases],
    }
    manifest = {**body, "manifest_hash": _json_hash(body)}
    sealed = {
        case.response_id: f"sealed-private-candidate-{index:03d}"
        for index, case in enumerate(cases)
    }
    direct = _run(
        "private-direct-alias",
        "org/private-shared-model",
        ["incorrect"] * 100,
        [1.0] * 100,
        [10] * 100,
        cases=cases,
    )
    thinking = _run(
        "private-thinking-alias",
        "org/private-shared-model",
        ["partial"] * 100,
        [2.0] * 100,
        [11] * 100,
        cases=cases,
    )
    manifest_path = private / "manifest.json"
    sealed_path = private / "sealed_candidates.json"
    direct_path = private / "direct.json"
    thinking_path = private / "thinking.json"
    direct_observation_path = private / "direct-runtime.json"
    thinking_observation_path = private / "thinking-runtime.json"
    runtime_binding_path = private / "runtime-binding.json"
    hashes = {
        "manifest": _write_private_json(manifest_path, manifest),
        "sealed": _write_private_json(sealed_path, sealed),
        "direct": _write_private_json(direct_path, direct),
        "thinking": _write_private_json(thinking_path, thinking),
        "direct_observation": _write_private_json(
            direct_observation_path,
            _runtime_observation("/private/shared-model.gguf", False),
        ),
        "thinking_observation": _write_private_json(
            thinking_observation_path,
            _runtime_observation("/private/shared-model.gguf", True),
        ),
    }
    runtime_binding = {
        "version": 1,
        "shared_configuration": {
            "context_tokens": 8192,
            "max_output_tokens": 4096,
            "temperature": 0,
        },
        "roles": {
            "direct": {
                "run_sha256": hashes["direct"],
                "runtime_receipt_sha256": direct["runtime_receipt_sha256"],
                "thinking": False,
            },
            "thinking": {
                "run_sha256": hashes["thinking"],
                "runtime_receipt_sha256": thinking["runtime_receipt_sha256"],
                "thinking": True,
            },
        },
    }
    hashes["runtime_binding"] = _write_private_json(
        runtime_binding_path, runtime_binding
    )
    output = private / "output"
    return [
        "--manifest",
        str(manifest_path),
        "--manifest-sha256",
        hashes["manifest"],
        "--sealed-candidates",
        str(sealed_path),
        "--sealed-candidates-sha256",
        hashes["sealed"],
        "--direct-run",
        str(direct_path),
        "--direct-run-sha256",
        hashes["direct"],
        "--thinking-run",
        str(thinking_path),
        "--thinking-run-sha256",
        hashes["thinking"],
        "--direct-runtime-observation",
        str(direct_observation_path),
        "--direct-runtime-observation-sha256",
        hashes["direct_observation"],
        "--thinking-runtime-observation",
        str(thinking_observation_path),
        "--thinking-runtime-observation-sha256",
        hashes["thinking_observation"],
        "--runtime-binding",
        str(runtime_binding_path),
        "--runtime-binding-sha256",
        hashes["runtime_binding"],
        "--output-dir",
        str(output),
        "--seed",
        "2026",
    ], output


def _output_bytes(output: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(output)): path.read_bytes()
        for path in output.rglob("*")
        if path.is_file()
    }


def test_cli_writes_private_outputs_and_reuses_secret_for_deterministic_regeneration(
    tmp_path: Path,
) -> None:
    argv, output = _cli_inputs(tmp_path)

    assert main(argv) == 0

    assert stat.S_IMODE(output.stat().st_mode) == 0o700
    assert stat.S_IMODE((output / "review").stat().st_mode) == 0o700
    expected = {
        "comparison.json",
        "comparison.md",
        "review/independent-review-packet.json",
        "review/critique-review-packet.json",
        "review/sealed-mode-aliases.json",
    }
    assert set(_output_bytes(output)) == expected
    for name in expected:
        path = output / name
        assert stat.S_ISREG(path.stat().st_mode)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert path.stat().st_nlink == 1
    first = _output_bytes(output)

    second_argv = ["9999" if value == "2026" else value for value in argv]
    assert main(second_argv) == 0
    assert _output_bytes(output) == first


@pytest.mark.parametrize("unsafe_kind", ["symlink", "nonregular", "wrong_mode"])
@pytest.mark.parametrize(
    "relative_target",
    [
        "comparison.json",
        "review/critique-review-packet.json",
        "review/sealed-mode-aliases.json",
    ],
)
def test_cli_rejects_unsafe_existing_output_before_writing_any_packet(
    tmp_path: Path, relative_target: str, unsafe_kind: str
) -> None:
    argv, output = _cli_inputs(tmp_path)
    review = output / "review"
    review.mkdir(parents=True, mode=0o700)
    target = output / relative_target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.parent.chmod(0o700)
    if unsafe_kind == "symlink":
        link_target = output.parent / "outside.json"
        link_target.write_text("outside", encoding="utf-8")
        link_target.chmod(0o600)
        target.symlink_to(link_target)
    elif unsafe_kind == "nonregular":
        os.mkfifo(target, mode=0o600)
    else:
        target.write_text("unsafe", encoding="utf-8")
        target.chmod(0o644)

    with pytest.raises(ValueError, match="private|output|seal"):
        main(argv)

    assert not (output / "comparison.md").exists()
    assert not (review / "independent-review-packet.json").exists()
    if relative_target != "review/critique-review-packet.json":
        assert not (review / "critique-review-packet.json").exists()


@pytest.mark.parametrize("unsafe_kind", ["symlink", "nonregular", "wrong_mode"])
@pytest.mark.parametrize("relative_directory", [".", "review"])
def test_cli_requires_exact_private_output_directories(
    tmp_path: Path, relative_directory: str, unsafe_kind: str
) -> None:
    argv, output = _cli_inputs(tmp_path)
    target = output if relative_directory == "." else output / relative_directory
    target.parent.mkdir(parents=True, exist_ok=True)
    target.parent.chmod(0o700)
    if unsafe_kind == "symlink":
        outside = output.parent / f"outside-{relative_directory.replace('/', '-')}"
        outside.mkdir(mode=0o700)
        target.symlink_to(outside, target_is_directory=True)
    elif unsafe_kind == "nonregular":
        target.write_text("not a directory", encoding="utf-8")
        target.chmod(0o600)
    else:
        target.mkdir(mode=0o700)
        target.chmod(0o755)

    with pytest.raises(ValueError, match="directory|0700"):
        main(argv)
