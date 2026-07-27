"""Deterministic blind packets and strict two-pass reviewer records."""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import asdict
import hashlib
import json
from typing import Any

from trec_mlm.pointwise_judge import (
    CandidateCase,
    DECISION_KEYS,
    parse_judge_decision,
)
from trec_mlm.pointwise_runner import (
    FROZEN_DECISION_SCHEMA_SHA256,
    _validate_run_container,
    _validated_same_identity_record,
    trusted_run_backend_model,
)


_CASE_PACKET_KEYS = {
    "response_id",
    "question",
    "provisional_reference",
    "candidate_response",
}
_REVIEW_LEDGER_KEYS = {"version", "reviewer", "pass1", "pass2"}
_REVIEW_ROW_KEYS = DECISION_KEYS | {
    "reviewer",
    "response_id",
    "review_phase",
    "decisive_disagreements",
    "prompt_findings",
}
_MAX_FINDINGS = 16
_MAX_FINDING_WORDS = 120
_MIN_OPTIONAL_IDENTITY_TOKEN_LENGTH = 4


def build_independent_review_packet(
    cases: Sequence[CandidateCase],
) -> list[dict[str, object]]:
    """Build pass 1 with case text and no judge output or private metadata."""
    validated = _validate_cases(cases)
    return [_case_fields(case) for case in validated]


def build_critique_packet(
    cases: Sequence[CandidateCase],
    runs: Mapping[str, Mapping[str, object]],
    *,
    seed: int = 2026,
    forbidden_identity_tokens: Collection[str] = (),
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Build pass 2 with only anonymous terminal judge outcomes."""
    validated_cases = _validate_cases(cases)
    if type(seed) is not int:
        raise ValueError("review seed must be an integer")
    if type(runs) is not dict or not runs:
        raise ValueError("runs must be a nonempty mapping")
    aliases = sorted(runs)
    if any(type(alias) is not str or not alias for alias in aliases):
        raise ValueError("run aliases must be nonempty strings")

    ordered_models = sorted(aliases, key=lambda alias: _alias_digest(seed, alias))
    judge_to_model = {
        f"judge_{_letter(index)}": model_alias
        for index, model_alias in enumerate(ordered_models)
    }
    model_to_judge = {model: judge for judge, model in judge_to_model.items()}
    expected_ids = {case.response_id for case in validated_cases}
    decisions: dict[str, dict[str, dict[str, object]]] = {}
    identity_tokens = _identity_tokens(forbidden_identity_tokens)
    shared_prompt: str | None = None
    for model_alias in aliases:
        raw_run = runs[model_alias]
        if type(raw_run) is not dict:
            raise ValueError(f"run {model_alias} must be an object")
        run = dict(raw_run)
        try:
            _validate_run_container(run)
        except ValueError as error:
            raise ValueError(f"invalid run {model_alias}: {error}") from error
        if run["model_alias"] != model_alias:
            raise ValueError(f"run {model_alias} has conflicting model identity")
        _add_identity_token(identity_tokens, model_alias, mandatory=True)
        _add_identity_token(identity_tokens, run["model_alias"], mandatory=True)
        _add_identity_token(identity_tokens, run["model_id"], mandatory=True)
        if run["schema_sha256"] != FROZEN_DECISION_SCHEMA_SHA256:
            raise ValueError(f"run {model_alias} does not use the frozen schema")
        if shared_prompt is None:
            shared_prompt = run["prompt_sha256"]
        elif run["prompt_sha256"] != shared_prompt:
            raise ValueError("runs do not share one frozen prompt identity")
        records = run["records"]
        if set(records) != expected_ids:
            raise ValueError(
                f"run {model_alias} response coverage does not match cases"
            )
        try:
            backend_response_model = trusted_run_backend_model(
                run, [case.response_id for case in validated_cases]
            )
        except ValueError as error:
            raise ValueError(f"run {model_alias} {error}") from error
        judge_alias = model_to_judge[model_alias]
        decisions[judge_alias] = {}
        for case in validated_cases:
            record = records[case.response_id]
            if type(record) is not dict:
                raise ValueError(f"run {model_alias} record must be an object")
            if record.get("input_sha256") != _sha256_json(asdict(case)):
                raise ValueError(f"run {model_alias} record input identity is invalid")
            for key in (
                "model_alias",
                "model_id",
                "prompt_sha256",
                "schema_sha256",
                "runtime_receipt_sha256",
            ):
                if record.get(key) != run.get(key):
                    raise ValueError(
                        f"run {model_alias} record {key} identity is invalid"
                    )
            try:
                reusable = _validated_same_identity_record(
                    record,
                    record,
                    backend_response_model=backend_response_model,
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid run {model_alias} record: {error}"
                ) from error
            if not reusable.terminal:
                raise ValueError(f"run {model_alias} record is not terminal")
            if record["decision"] is None:
                outcome = {"status": "failure"}
            else:
                outcome = {
                    "status": "decision",
                    "decision": dict(record["decision"]),
                }
            decisions[judge_alias][case.response_id] = outcome

    if _contains_identity_text(decisions, identity_tokens):
        raise ValueError("critique packet contains forbidden identity text")

    packet: list[dict[str, object]] = []
    for case in validated_cases:
        case_order = sorted(
            aliases,
            key=lambda alias: _case_order_digest(seed, case.response_id, alias),
        )
        judges = {
            model_to_judge[model_alias]: decisions[model_to_judge[model_alias]][
                case.response_id
            ]
            for model_alias in case_order
        }
        packet.append({**_case_fields(case), "judges": judges})
    sealed = {"version": 1, "seed": seed, "judge_aliases": judge_to_model}
    return packet, sealed


def parse_reviewer_record(raw: str) -> dict[str, object]:
    """Parse reviewer JSON without accepting duplicate keys or non-finite values."""
    if type(raw) is not str:
        raise ValueError("reviewer record must be JSON text")
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_object_without_duplicate_keys,
            parse_constant=_reject_nonfinite_json,
        )
    except json.JSONDecodeError as error:
        raise ValueError("reviewer record is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError("reviewer record must be a JSON object")
    return value


def validate_reviewer_coverage(
    expected_response_ids: Collection[str], review_rows: Sequence[Mapping[str, object]]
) -> None:
    """Require each expected response exactly once and no unknown response."""
    expected = _validated_string_set(expected_response_ids, "expected response IDs")
    if type(review_rows) is not list:
        raise ValueError("review rows must be a list")
    observed: list[str] = []
    for row in review_rows:
        if type(row) is not dict or type(row.get("response_id")) is not str:
            raise ValueError("review row requires a string response_id")
        observed.append(row["response_id"])
    if len(observed) != len(set(observed)):
        raise ValueError("duplicate response ID in review rows")
    unknown = set(observed) - expected
    if unknown:
        raise ValueError("unknown response ID in review rows")
    missing = expected - set(observed)
    if missing:
        raise ValueError("missing response ID in review rows")


def validate_reviewer_record(
    record: Mapping[str, object],
    expected_response_ids: Collection[str],
    *,
    allowed_reviewers: Collection[str] = ("advisor", "claude"),
    expected_reviewer: str | None = None,
    judge_aliases: Collection[str] = (),
    forbidden_identities: Collection[str] = (),
) -> None:
    """Validate one reviewer's exact pass-1 and pass-2 ledgers."""
    if type(record) is not dict or set(record) != _REVIEW_LEDGER_KEYS:
        raise ValueError("reviewer record must contain exact keys")
    if type(record["version"]) is not int or record["version"] != 1:
        raise ValueError("reviewer record version must be 1")
    reviewers = _validated_string_set(allowed_reviewers, "reviewer allowlist")
    reviewer = record["reviewer"]
    if type(reviewer) is not str or reviewer not in reviewers:
        raise ValueError("reviewer is not in the allowlist")
    if expected_reviewer is not None and reviewer != expected_reviewer:
        raise ValueError("reviewer identity does not match expected reviewer")
    known_aliases = _validated_string_set(
        judge_aliases, "judge aliases", allow_empty=True
    )
    forbidden = _validated_string_set(
        forbidden_identities, "forbidden identities", allow_empty=True
    )
    for phase in ("pass1", "pass2"):
        rows = record[phase]
        validate_reviewer_coverage(expected_response_ids, rows)
        for row in rows:
            _validate_review_row(
                row,
                phase=phase,
                reviewer=reviewer,
                judge_aliases=known_aliases,
                forbidden_identities=forbidden,
            )


def compare_reviews(
    left: Mapping[str, object],
    right: Mapping[str, object],
    expected_response_ids: Collection[str],
    *,
    allowed_reviewers: Collection[str] = ("advisor", "claude"),
    judge_aliases: Collection[str] = (),
    forbidden_identities: Collection[str] = (),
) -> list[dict[str, object]]:
    """Compare two independently validated reviews in deterministic ID order."""
    for value in (left, right):
        validate_reviewer_record(
            value,
            expected_response_ids,
            allowed_reviewers=allowed_reviewers,
            judge_aliases=judge_aliases,
            forbidden_identities=forbidden_identities,
        )
    if left["reviewer"] == right["reviewer"]:
        raise ValueError("reviews must come from distinct reviewers")
    ledgers = sorted((left, right), key=lambda value: value["reviewer"])
    by_reviewer = {
        ledger["reviewer"]: {
            phase: {row["response_id"]: row for row in ledger[phase]}
            for phase in ("pass1", "pass2")
        }
        for ledger in ledgers
    }
    result: list[dict[str, object]] = []
    for response_id in sorted(
        _validated_string_set(expected_response_ids, "expected response IDs")
    ):
        reviews = [
            {
                "reviewer": reviewer,
                "pass1": by_reviewer[reviewer]["pass1"][response_id],
                "pass2": by_reviewer[reviewer]["pass2"][response_id],
            }
            for reviewer in sorted(by_reviewer)
        ]
        pass1_agrees = reviews[0]["pass1"]["label"] == reviews[1]["pass1"]["label"]
        pass2_agrees = reviews[0]["pass2"]["label"] == reviews[1]["pass2"]["label"]
        result.append(
            {
                "response_id": response_id,
                "reviews": reviews,
                "pass1_label_agreement": pass1_agrees,
                "pass2_label_agreement": pass2_agrees,
                "unresolved": not pass2_agrees,
            }
        )
    return result


def _validate_review_row(
    row: object,
    *,
    phase: str,
    reviewer: str,
    judge_aliases: set[str],
    forbidden_identities: set[str],
) -> None:
    if type(row) is not dict or set(row) != _REVIEW_ROW_KEYS:
        raise ValueError("review row must contain exact keys")
    if row["reviewer"] != reviewer or row["review_phase"] != phase:
        raise ValueError("review row reviewer or phase is inconsistent")
    decision = {key: row[key] for key in DECISION_KEYS}
    try:
        parse_judge_decision(json.dumps(decision, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(str(error)) from error
    _bounded_text(row["rationale"], "review rationale", forbidden_identities)
    disagreements = row["decisive_disagreements"]
    findings = row["prompt_findings"]
    if phase == "pass1" and (disagreements != [] or findings != []):
        raise ValueError("pass1 must not contain critique findings")
    if type(disagreements) is not list or len(disagreements) > _MAX_FINDINGS:
        raise ValueError("decisive_disagreements must be a bounded list")
    for disagreement in disagreements:
        if type(disagreement) is not dict or set(disagreement) != {
            "judge_aliases",
            "finding",
        }:
            raise ValueError("decisive disagreement must contain exact keys")
        aliases = disagreement["judge_aliases"]
        if (
            type(aliases) is not list
            or not aliases
            or any(type(alias) is not str for alias in aliases)
            or len(aliases) != len(set(aliases))
        ):
            raise ValueError("decisive disagreement judge aliases are invalid")
        if not set(aliases) <= judge_aliases:
            raise ValueError("unknown judge alias in decisive disagreement")
        _bounded_text(
            disagreement["finding"], "decisive disagreement", forbidden_identities
        )
    if type(findings) is not list or len(findings) > _MAX_FINDINGS:
        raise ValueError("prompt_findings must be a bounded list")
    for finding in findings:
        _bounded_text(finding, "prompt finding", forbidden_identities)


def _validate_cases(cases: Sequence[CandidateCase]) -> list[CandidateCase]:
    if type(cases) not in {list, tuple} or not cases:
        raise ValueError("cases must be a nonempty sequence")
    if any(type(case) is not CandidateCase for case in cases):
        raise ValueError("cases must contain CandidateCase values")
    if any(
        type(getattr(case, field)) is not str or not getattr(case, field)
        for case in cases
        for field in (
            "question_id",
            "response_id",
            "question",
            "reference_answer",
            "candidate_answer",
        )
    ):
        raise ValueError("case strings must be nonempty")
    if len({case.response_id for case in cases}) != len(cases):
        raise ValueError("cases must have unique response IDs")
    return list(cases)


def _case_fields(case: CandidateCase) -> dict[str, object]:
    value = {
        "response_id": case.response_id,
        "question": case.question,
        "provisional_reference": case.reference_answer,
        "candidate_response": case.candidate_answer,
    }
    if set(value) != _CASE_PACKET_KEYS:
        raise RuntimeError("invalid packet fields")
    return value


def _bounded_text(value: object, field: str, forbidden: set[str]) -> None:
    if type(value) is not str or not 1 <= len(value.split()) <= _MAX_FINDING_WORDS:
        raise ValueError(f"{field} must contain 1 to 120 words")
    lowered = value.casefold()
    if any(identity.casefold() in lowered for identity in forbidden):
        raise ValueError(f"{field} contains a forbidden model identity")


def _validated_string_set(
    values: Collection[str], description: str, *, allow_empty: bool = False
) -> set[str]:
    try:
        result = set(values)
    except TypeError as error:
        raise ValueError(f"{description} must be strings") from error
    if (not result and not allow_empty) or any(
        type(value) is not str or not value for value in result
    ):
        raise ValueError(f"{description} must be nonempty strings")
    return result


def _letter(index: int) -> str:
    if not 0 <= index < 26:
        raise ValueError("at most 26 anonymous judges are supported")
    return chr(ord("a") + index)


def _identity_tokens(values: Collection[str]) -> set[str]:
    tokens: set[str] = set()
    if isinstance(values, (str, bytes)):
        raise ValueError("forbidden identity tokens must be a collection of strings")
    try:
        for value in values:
            _add_identity_token(tokens, value, mandatory=False)
    except TypeError as error:
        raise ValueError(
            "forbidden identity tokens must be a collection of strings"
        ) from error
    return tokens


def _add_identity_token(tokens: set[str], value: object, *, mandatory: bool) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError("identity tokens must be nonempty strings")
    normalized = value.strip().casefold()
    if mandatory or len(normalized) >= _MIN_OPTIONAL_IDENTITY_TOKEN_LENGTH:
        tokens.add(normalized)
    basename = normalized.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1]
    if len(basename) >= _MIN_OPTIONAL_IDENTITY_TOKEN_LENGTH:
        tokens.add(basename)


def _contains_identity_text(value: object, tokens: set[str]) -> bool:
    if type(value) is str:
        text = value.casefold()
        return any(_contains_exact_token(text, token) for token in tokens)
    if type(value) is list:
        return any(_contains_identity_text(item, tokens) for item in value)
    if type(value) is dict:
        return any(_contains_identity_text(item, tokens) for item in value.values())
    return False


def _contains_exact_token(text: str, token: str) -> bool:
    start = 0
    while (index := text.find(token, start)) >= 0:
        end = index + len(token)
        left_ok = index == 0 or not (token[0].isalnum() and text[index - 1].isalnum())
        right_ok = end == len(text) or not (token[-1].isalnum() and text[end].isalnum())
        if left_ok and right_ok:
            return True
        start = index + 1
    return False


def _alias_digest(seed: int, alias: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{alias}".encode("utf-8")).digest()


def _case_order_digest(seed: int, response_id: str, alias: str) -> bytes:
    return hashlib.sha256(f"{seed}\0{response_id}\0{alias}".encode("utf-8")).digest()


def _sha256_json(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key in reviewer record: {key}")
        result[key] = value
    return result


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON value is forbidden: {value}")
