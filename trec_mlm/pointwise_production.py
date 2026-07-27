"""Private, sealed inputs for the full pointwise production evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any, Mapping

from trec_mlm.artifact_io import length_framed_sha256, manifest_aggregate_sha256
from trec_mlm.pointwise_judge import load_prompt_contract
from trec_mlm.profile_inference import format_profile_document
from trec_mlm.profile_retrieval import canonical_profile_record_id
from trec_mlm.pointwise_sampling import (
    iter_discovery_responses,
    load_discovery_questions,
    load_grounded_references,
)


EXPECTED_PRODUCTION_RECORD_COUNT = 182_764
FROZEN_DISCOVERY_SHA256 = (
    "b33c28aa2571cdf5b2867dcc00f0dc55e22198b90022dd23f748a03395301b25"
)
FROZEN_GROUNDED_SHA256 = (
    "3f0cb0885d82c9e83827c0f5ddad32dcba90f5a3307a095543bd1eb29d3133d6"
)
FROZEN_REFINEMENT_RECORDS_SHA256 = (
    "e25237be81ffd86726fb4b3ccd1a2a8bdea7213197f84341b6bf50c23274ae92"
)
FROZEN_REFINEMENT_MANIFEST_SHA256 = (
    "64d6972021fcda8ab98b56b0555bccdabd22e52df238b438400866827319825e"
)
_INVENTORY_NAME = "production-records.jsonl"
_MANIFEST_NAME = "production-manifest.json"
_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600


@dataclass(frozen=True, slots=True)
class ProductionRecord:
    """One source-ordered candidate/reference join for production judging."""

    record_id: str
    source_ordinal: int
    question_id: str
    candidate_id: str
    question: str
    provisional_reference: str
    candidate_response: str

    @classmethod
    def from_source(
        cls,
        *,
        record_id: str,
        source_ordinal: int,
        question_id: str,
        candidate_id: str,
        question: str,
        provisional_reference: str,
        candidate_response: str,
    ) -> ProductionRecord:
        if type(source_ordinal) is not int or source_ordinal < 0:
            raise ValueError("source ordinal must be a nonnegative integer")
        for name, value in {
            "record ID": record_id,
            "question ID": question_id,
            "candidate ID": candidate_id,
            "question": question,
            "provisional reference": provisional_reference,
            "candidate response": candidate_response,
        }.items():
            if type(value) is not str or not value.strip():
                raise ValueError(f"production {name} must be nonempty")
        return cls(
            record_id=record_id,
            source_ordinal=source_ordinal,
            question_id=question_id,
            candidate_id=candidate_id,
            question=question,
            provisional_reference=provisional_reference,
            candidate_response=candidate_response,
        )


@dataclass(frozen=True, slots=True)
class _RefinementMember:
    record_id: str
    question_id: str
    candidate_id: str
    question: str
    response: str


def build_production_inventory(
    *,
    discovery: Path,
    grounded: Path,
    refinement_records: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Stream and seal the exact source/reference join into a private JSONL file."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ValueError("production output directory already exists")

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=".pointwise-production-", dir=parent))
    temporary_dir.chmod(_PRIVATE_DIRECTORY_MODE)
    try:
        sources_dir = temporary_dir / "sources"
        sources_dir.mkdir(mode=_PRIVATE_DIRECTORY_MODE)
        discovery_snapshot, discovery_sha256 = _snapshot_source(
            Path(discovery), sources_dir, "discovery"
        )
        grounded_snapshot, grounded_sha256 = _snapshot_source(
            Path(grounded), sources_dir, "grounded"
        )
        refinement_snapshot, refinement_sha256 = _snapshot_source(
            Path(refinement_records), sources_dir, "refinement-records"
        )
        refinement_manifest_snapshot, refinement_manifest_sha256 = _snapshot_source(
            Path(refinement_records).parent / "manifest.json",
            sources_dir,
            "refinement-manifest",
        )
        source_hashes = {
            "discovery": discovery_sha256,
            "grounded": grounded_sha256,
            "refinement_records": refinement_sha256,
            "refinement_manifest": refinement_manifest_sha256,
        }
        _validate_approved_source_hashes(source_hashes)
        _validate_refinement_manifest(
            refinement_manifest_snapshot,
            records_name=Path(refinement_records).name,
            records_sha256=refinement_sha256,
            records_size=refinement_snapshot.stat().st_size,
        )

        questions = load_discovery_questions(discovery_snapshot)
        references = load_grounded_references(grounded_snapshot, questions=questions)
        members = _load_refinement_membership(refinement_snapshot, questions=questions)
        inventory_path = temporary_dir / _INVENTORY_NAME
        count, inventory_sha256 = _write_inventory(
            inventory_path,
            references=references,
            discovery=discovery_snapshot,
            members=members,
        )
        if count != EXPECTED_PRODUCTION_RECORD_COUNT:
            raise ValueError(
                "expected "
                f"{EXPECTED_PRODUCTION_RECORD_COUNT} production records, found {count}"
            )
        manifest_body: dict[str, Any] = {
            "version": 1,
            "record_count": count,
            "inventory": _INVENTORY_NAME,
            "inventory_sha256": inventory_sha256,
            "sources": {
                "discovery": {"sha256": discovery_sha256},
                "grounded": {"sha256": grounded_sha256},
                "refinement_records": {"sha256": refinement_sha256},
                "refinement_manifest": {"sha256": refinement_manifest_sha256},
            },
        }
        manifest = {
            **manifest_body,
            "manifest_sha256": _sha256_json(manifest_body),
        }
        _write_private_json(temporary_dir / _MANIFEST_NAME, manifest)
        _remove_tree(sources_dir)
        _assert_private_tree(temporary_dir)
        os.replace(temporary_dir, output_dir)
    except BaseException:
        _remove_tree(temporary_dir)
        raise
    return manifest


def build_opus_prompt_review_packet(
    *,
    protocol_dir: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Write an identity-blind packet for independent review of the v2 contract."""
    contract = load_prompt_contract(Path(protocol_dir), version="v2")
    prompt_bytes = contract.prompt_path.read_bytes()
    schema_bytes = contract.schema_path.read_bytes()
    body: dict[str, Any] = {
        "version": 1,
        "contract": {
            "version": "v2",
            "prompt": contract.prompt,
            "schema": contract.schema,
            "prompt_sha256": _sha256_bytes(prompt_bytes),
            "schema_sha256": _sha256_bytes(schema_bytes),
        },
        "adjudication_questions": [
            "ambiguity",
            "reference evidence",
            "invariants",
            "repair behavior",
            "candidate defects versus question or provisional-reference faults",
            "refusal, negation, relevance, uncertainty, and truncation behavior",
            "resistance to invented facts or rationales",
            "deterministic JSON and schema behavior",
            "unnecessary or ambiguous requested fields",
        ],
        "reviewer_instruction": (
            "Review the proposed pointwise contract. Identify concrete changes, "
            "the affected contract text, and a reason for each recommendation. "
            "Do not judge any model, provider, or run."
        ),
    }
    packet = {**body, "packet_sha256": _sha256_json(body)}
    _reject_judge_identity_leakage(packet)
    _write_or_validate_private_packet(Path(output_path), packet)
    return packet


def _write_inventory(
    path: Path,
    *,
    references: Mapping[str, Any],
    discovery: Path,
    members: Mapping[tuple[str, str, str], _RefinementMember],
) -> tuple[int, str]:
    digest = hashlib.sha256()
    record_ids: set[str] = set()
    count = 0
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, _PRIVATE_FILE_MODE)
    try:
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = -1
            for response in iter_discovery_responses(discovery):
                member = members.get(
                    (response.question_id, response.candidate_id, response.answer)
                )
                if member is None:
                    continue
                try:
                    reference = references[response.question_id]
                except KeyError as error:
                    raise ValueError(
                        f"missing grounded reference for {response.question_id}"
                    ) from error
                record = ProductionRecord.from_source(
                    record_id=member.record_id,
                    source_ordinal=count,
                    question_id=response.question_id,
                    candidate_id=response.candidate_id,
                    question=reference.question,
                    provisional_reference=reference.answer,
                    candidate_response=response.answer,
                )
                if record.record_id in record_ids:
                    raise ValueError("duplicate production record ID")
                record_ids.add(record.record_id)
                encoded = (
                    json.dumps(
                        asdict(record),
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                    + b"\n"
                )
                handle.write(encoded)
                digest.update(encoded)
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if len(record_ids) != len(members):
        raise ValueError("frozen refinement record is absent from discovery source")
    return count, digest.hexdigest()


def _load_refinement_membership(
    path: Path, *, questions: Mapping[str, str]
) -> dict[tuple[str, str, str], _RefinementMember]:
    members: dict[tuple[str, str, str], _RefinementMember] = {}
    record_ids: set[str] = set()
    for raw in _iter_json_array(path):
        if type(raw) is not dict:
            raise ValueError("refinement record must be an object")
        try:
            member = _RefinementMember(
                record_id=_required_string(raw, "record_id"),
                question_id=_required_string(raw, "source_query_id"),
                candidate_id=_required_string(raw, "candidate_id"),
                question=_required_string(raw, "question"),
                response=_required_string(raw, "response"),
            )
        except KeyError as error:
            raise ValueError("refinement record has missing required field") from error
        _validate_refinement_member(member, raw)
        expected_question = questions.get(member.question_id)
        if expected_question is None:
            raise ValueError("refinement record has an unknown discovery question")
        if member.question != expected_question:
            raise ValueError("refinement/discovery question mismatch")
        key = (member.question_id, member.candidate_id, member.response)
        if member.record_id in record_ids:
            raise ValueError("duplicate refinement record ID")
        if key in members:
            raise ValueError("duplicate refinement response identity")
        record_ids.add(member.record_id)
        members[key] = member
    if not members:
        raise ValueError("refinement records must not be empty")
    return members


def _validate_approved_source_hashes(source_hashes: Mapping[str, str]) -> None:
    expected = {
        "discovery": FROZEN_DISCOVERY_SHA256,
        "grounded": FROZEN_GROUNDED_SHA256,
        "refinement_records": FROZEN_REFINEMENT_RECORDS_SHA256,
        "refinement_manifest": FROZEN_REFINEMENT_MANIFEST_SHA256,
    }
    for name, expected_sha256 in expected.items():
        if source_hashes[name] != expected_sha256:
            raise ValueError(f"approved {name.replace('_', ' ')} hash mismatch")


def _validate_refinement_manifest(
    path: Path,
    *,
    records_name: str,
    records_sha256: str,
    records_size: int,
) -> None:
    if records_name != "records.json":
        raise ValueError("approved refinement records must be named records.json")
    try:
        manifest = json.loads(
            path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except json.JSONDecodeError as error:
        raise ValueError("approved refinement manifest is not valid JSON") from error
    files = manifest.get("files") if type(manifest) is dict else None
    if (
        type(files) is not dict
        or manifest.get("artifact_type") != "profile_inference_manifest"
        or manifest.get("version") != 2
        or manifest.get("qrel_free") is not True
        or manifest.get("aggregate_sha256") != manifest_aggregate_sha256(files)
    ):
        raise ValueError("approved refinement manifest contract changed")
    records_entry = files.get("records.json")
    if (
        type(records_entry) is not dict
        or records_entry.get("sha256") != records_sha256
        or records_entry.get("byte_size") != records_size
    ):
        raise ValueError("approved refinement manifest records entry changed")


def _validate_refinement_member(
    member: _RefinementMember, raw: Mapping[str, object]
) -> None:
    if set(raw) != {
        "record_id",
        "source_query_id",
        "candidate_id",
        "question",
        "response",
        "raw_membership_sha256",
        "formatted_sha256",
    }:
        raise ValueError("refinement record schema changed")
    expected_id = canonical_profile_record_id(
        member.candidate_id,
        member.question_id,
        member.question,
        member.response,
    )
    if member.record_id != expected_id:
        raise ValueError("refinement record ID is not canonical profile-record-v2")
    raw_hash = length_framed_sha256((member.question, member.response))
    formatted_hash = hashlib.sha256(
        format_profile_document(member.question, member.response).encode("utf-8")
    ).hexdigest()
    if (
        raw.get("raw_membership_sha256") != raw_hash
        or raw.get("formatted_sha256") != formatted_hash
    ):
        raise ValueError("refinement record correspondence changed")


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key in approved refinement manifest")
        result[key] = value
    return result


def _iter_json_array(path: Path):
    """Stream a JSON array without retaining the full refinement artifact."""
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as handle:
        buffer = ""
        exhausted = False

        def read_more() -> None:
            nonlocal buffer, exhausted
            chunk = handle.read(1024 * 1024)
            if chunk:
                buffer += chunk
            else:
                exhausted = True

        def require_buffer() -> None:
            while not buffer and not exhausted:
                read_more()

        require_buffer()
        buffer = buffer.lstrip()
        if not buffer or buffer[0] != "[":
            raise ValueError("refinement records must be a JSON array")
        buffer = buffer[1:]
        while True:
            buffer = buffer.lstrip()
            require_buffer()
            buffer = buffer.lstrip()
            if not buffer:
                raise ValueError("refinement records ended before the array closed")
            if buffer[0] == "]":
                trailing = buffer[1:] + handle.read()
                if trailing.strip():
                    raise ValueError("refinement records contain trailing content")
                return
            while True:
                try:
                    value, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError as error:
                    if exhausted:
                        raise ValueError(
                            "refinement records are not valid JSON"
                        ) from error
                    read_more()
            yield value
            buffer = buffer[end:].lstrip()
            require_buffer()
            buffer = buffer.lstrip()
            if not buffer:
                raise ValueError("refinement records ended before the array closed")
            if buffer[0] == ",":
                buffer = buffer[1:]
                continue
            if buffer[0] == "]":
                trailing = buffer[1:] + handle.read()
                if trailing.strip():
                    raise ValueError("refinement records contain trailing content")
                return
            raise ValueError("refinement records are not a JSON array")


def _required_string(value: Mapping[str, object], field: str) -> str:
    item = value[field]
    if type(item) is not str or not item.strip():
        raise ValueError("refinement record field must be a nonempty string")
    return item


def _snapshot_source(path: Path, destination_dir: Path, name: str) -> tuple[Path, str]:
    descriptor, raw_destination = tempfile.mkstemp(
        prefix=f"{name}-", suffix=Path(path).suffix, dir=destination_dir
    )
    destination = Path(raw_destination)
    digest = hashlib.sha256()
    try:
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        with Path(path).open("rb") as source, os.fdopen(descriptor, "wb") as target:
            descriptor = -1
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                target.write(chunk)
            target.flush()
            os.fsync(target.fileno())
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return destination, digest.hexdigest()


def _write_private_json(path: Path, payload: object) -> None:
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        os.fchmod(descriptor, _PRIVATE_FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = -1
            json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _write_or_validate_private_packet(path: Path, packet: dict[str, Any]) -> None:
    output_dir = path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(_PRIVATE_DIRECTORY_MODE)
    if path.exists():
        if stat.S_IMODE(path.stat().st_mode) != _PRIVATE_FILE_MODE:
            raise ValueError("existing review packet must have mode 0600")
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise ValueError("existing review packet is not valid JSON") from error
        if existing != packet:
            raise ValueError("existing review packet differs from the v2 contract")
        return
    _write_private_json(path, packet)


def _reject_judge_identity_leakage(packet: Mapping[str, Any]) -> None:
    serialized = json.dumps(packet, ensure_ascii=False).casefold()
    forbidden = ("deepseek", "openrouter", "api key")
    if any(value in serialized for value in forbidden):
        raise ValueError(
            "review packet contains prohibited judge identity or secret text"
        )


def _assert_private_tree(path: Path) -> None:
    if stat.S_IMODE(path.stat().st_mode) != _PRIVATE_DIRECTORY_MODE:
        raise ValueError("production output directory must have mode 0700")
    for child in path.iterdir():
        if (
            not child.is_file()
            or stat.S_IMODE(child.stat().st_mode) != _PRIVATE_FILE_MODE
        ):
            raise ValueError("production output must contain only private files")


def _remove_tree(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            _remove_tree(child)
        else:
            child.unlink()
    path.rmdir()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_json(value: object) -> str:
    return _sha256_bytes(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
    )
