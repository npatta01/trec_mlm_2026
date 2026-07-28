"""Crash-safe, block-gated runner for the private DeepSeek pointwise run.

The module deliberately keeps the network boundary injectable.  The production
CLI supplies it; tests use deterministic local functions and never contact a
provider.  Every physical send has an attempt marker and conservative spend
reservation on disk before the boundary is crossed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time
from trec_mlm.pointwise_judge import parse_judge_decision


BLOCK_SIZE = 100
MODEL_ID = "deepseek/deepseek-v4-flash"
PROVIDER = "DeepInfra"
HARD_CAP_USD = 60.0
PAUSE_PROJECTION_USD = 50.0
MAX_TRANSIENT_RETRIES = 1
MAX_REPAIRS = 1
_DIR_MODE = 0o700
_FILE_MODE = 0o600


class BlockGateError(RuntimeError):
    """A hard block invariant failed; subsequent blocks must not be sent."""


class SoftAnomalyPause(RuntimeError):
    """A non-structural anomaly was persisted and requires human review."""


class TransientProviderError(RuntimeError):
    """Optional callback signal for a bounded retryable provider failure."""


def block_slices(record_count: int) -> list[tuple[int, int]]:
    """Return ordered half-open blocks; production naturally ends in 64."""
    if type(record_count) is not int or record_count < 1:
        raise ValueError("record count must be a positive integer")
    return [
        (start, min(start + BLOCK_SIZE, record_count))
        for start in range(0, record_count, BLOCK_SIZE)
    ]


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _bytes_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _private_dir(path: Path) -> None:
    if path.exists():
        details = path.stat(follow_symlinks=False)
        if (
            not stat.S_ISDIR(details.st_mode)
            or stat.S_IMODE(details.st_mode) != _DIR_MODE
        ):
            raise ValueError(f"private directory is not mode 0700: {path.name}")
        return
    path.mkdir(parents=True, mode=_DIR_MODE)
    path.chmod(_DIR_MODE)


def _write_private_json(path: Path, value: object) -> None:
    _private_dir(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, _FILE_MODE)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        path.chmod(_FILE_MODE)
        directory = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise


def _read_private_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        details = path.stat(follow_symlinks=False)
    except FileNotFoundError as error:
        raise ValueError(f"{label} is missing") from error
    if not stat.S_ISREG(details.st_mode) or stat.S_IMODE(details.st_mode) != _FILE_MODE:
        raise ValueError(f"{label} must be a regular mode-0600 file")
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON: {value}")
            ),
        )
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON") from error
    if type(value) is not dict:
        raise ValueError(f"{label} must be an object")
    return value


def _require_hash(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{label} is invalid")
    return value


def _validate_inventory(
    inventory: Path, manifest_path: Path
) -> tuple[int, dict[str, object]]:
    if not inventory.is_file() or stat.S_IMODE(inventory.stat().st_mode) != _FILE_MODE:
        raise ValueError("production inventory must be a regular mode-0600 file")
    manifest = _read_private_json(manifest_path, label="production manifest")
    required = {
        "version",
        "record_count",
        "inventory",
        "inventory_sha256",
        "sources",
        "manifest_sha256",
    }
    if (
        set(manifest) != required
        or manifest["version"] != 1
        or manifest["inventory"] != inventory.name
    ):
        raise ValueError("production manifest shape is invalid")
    if type(manifest["record_count"]) is not int or manifest["record_count"] < 1:
        raise ValueError("production manifest record count is invalid")
    if _bytes_hash(inventory) != _require_hash(
        manifest["inventory_sha256"], "inventory hash"
    ):
        raise ValueError("production inventory hash mismatch")
    body = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest["manifest_sha256"] != _json_hash(body):
        raise ValueError("production manifest identity is invalid")
    if type(manifest["sources"]) is not dict or not manifest["sources"]:
        raise ValueError("production source commitment is invalid")
    for source in manifest["sources"].values():
        if type(source) is not dict or set(source) != {"sha256"}:
            raise ValueError("production source commitment is invalid")
        _require_hash(source["sha256"], "production source hash")
    # This deliberately streams the inventory and never materializes it.
    with inventory.open("rb") as handle:
        line_count = sum(1 for _ in handle)
    if line_count != manifest["record_count"]:
        raise ValueError("production inventory count does not match its manifest")
    return manifest["record_count"], manifest


def _validate_contract(receipt_path: Path) -> dict[str, object]:
    receipt = _read_private_json(receipt_path, label="adjudication receipt")
    required = {
        "version",
        "packet_sha256",
        "review_sha256",
        "contract",
        "recommendation_ledger",
        "adjudication_sha256",
    }
    if set(receipt) != required or receipt["version"] != 1:
        raise ValueError("adjudication receipt shape is invalid")
    body = {
        key: value for key, value in receipt.items() if key != "adjudication_sha256"
    }
    if receipt["adjudication_sha256"] != _json_hash(body):
        raise ValueError("adjudication receipt identity is invalid")
    contract = receipt["contract"]
    if type(contract) is not dict or set(contract) != {
        "version",
        "source_version",
        "prompt",
        "schema",
        "prompt_sha256",
        "schema_sha256",
    }:
        raise ValueError("adjudicated contract shape is invalid")
    prompt, schema = contract["prompt"], contract["schema"]
    if (
        contract["source_version"] != "v2"
        or type(prompt) is not str
        or not prompt.strip()
        or type(schema) is not dict
    ):
        raise ValueError("adjudicated contract is invalid")
    if contract["prompt_sha256"] != hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest() or contract["schema_sha256"] != _json_hash(schema):
        raise ValueError("adjudicated contract hash mismatch")
    return contract


def _iter_inventory(
    inventory: Path, start: int, end: int
) -> Iterator[dict[str, object]]:
    with inventory.open("r", encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if ordinal < start:
                continue
            if ordinal >= end:
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise BlockGateError("inventory row is not valid JSON") from error
            fields = {
                "record_id",
                "source_ordinal",
                "question_id",
                "candidate_id",
                "question",
                "provisional_reference",
                "candidate_response",
            }
            if (
                type(record) is not dict
                or set(record) != fields
                or record["source_ordinal"] != ordinal
            ):
                raise BlockGateError("inventory record identity is invalid")
            if any(
                type(record[key]) is not str or not record[key].strip()
                for key in fields - {"source_ordinal"}
            ):
                raise BlockGateError(
                    "inventory record contains an empty identity field"
                )
            yield record


class ProductionRunner:
    """Single-process ordered runner; it never holds the full inventory in memory."""

    def __init__(
        self,
        *,
        inventory: Path,
        inventory_manifest: Path,
        contract_receipt: Path,
        run_dir: Path,
        call: Callable[[dict[str, object], bool], Mapping[str, object]],
        total_cap_usd: float = HARD_CAP_USD,
        worst_call_usd: float = 0.01,
        model_id: str = MODEL_ID,
        provider: str = PROVIDER,
    ) -> None:
        if not math.isfinite(total_cap_usd) or not 0 < total_cap_usd <= HARD_CAP_USD:
            raise ValueError("total spend cap must be positive and at most USD 60")
        if not math.isfinite(worst_call_usd) or not 0 < worst_call_usd <= HARD_CAP_USD:
            raise ValueError(
                "worst call reservation must be a finite authorized amount"
            )
        self.inventory, self.inventory_manifest, self.contract_receipt = (
            Path(inventory),
            Path(inventory_manifest),
            Path(contract_receipt),
        )
        self.run_dir, self.call = Path(run_dir), call
        self.total_cap_usd, self.worst_call_usd, self.model_id, self.provider = (
            total_cap_usd,
            worst_call_usd,
            model_id,
            provider,
        )
        self._record_count, self._input_manifest = _validate_inventory(
            self.inventory, self.inventory_manifest
        )
        self.contract = _validate_contract(self.contract_receipt)
        self._initialize()

    def _initialize(self) -> None:
        _private_dir(self.run_dir)
        for name in ("attempts", "raw", "results", "blocks", "anomalies"):
            _private_dir(self.run_dir / name)
        identity = {
            "version": 1,
            "record_count": self._record_count,
            "inventory_sha256": self._input_manifest["inventory_sha256"],
            "inventory_manifest_sha256": self._input_manifest["manifest_sha256"],
            "prompt_sha256": self.contract["prompt_sha256"],
            "schema_sha256": self.contract["schema_sha256"],
            "contract_version": self.contract["version"],
            "model_id": self.model_id,
            "provider": self.provider,
            "total_cap_usd": self.total_cap_usd,
            "worst_call_usd": self.worst_call_usd,
        }
        path = self.run_dir / "run-manifest.json"
        if path.exists():
            if _read_private_json(path, label="run manifest") != identity:
                raise ValueError("existing run manifest identity drift")
        else:
            _write_private_json(path, identity)
        ledger = self.run_dir / "spend-ledger.json"
        if not ledger.exists():
            _write_private_json(
                ledger,
                {
                    "version": 1,
                    "cap_usd": self.total_cap_usd,
                    "spent_usd": 0.0,
                    "unreconciled_reservation_usd": 0.0,
                },
            )

    def _spend(self) -> dict[str, object]:
        value = _read_private_json(
            self.run_dir / "spend-ledger.json", label="spend ledger"
        )
        if (
            set(value)
            != {"version", "cap_usd", "spent_usd", "unreconciled_reservation_usd"}
            or value["version"] != 1
            or value["cap_usd"] != self.total_cap_usd
        ):
            raise BlockGateError("spend ledger shape or cap changed")
        if any(
            type(value[key]) not in {int, float}
            or not math.isfinite(value[key])
            or value[key] < 0
            for key in ("spent_usd", "unreconciled_reservation_usd")
        ):
            raise BlockGateError("spend ledger is invalid")
        return value

    def _reserve_attempt(self, ordinal: int, repair: bool) -> tuple[int, float]:
        spend = self._spend()
        committed = float(spend["spent_usd"]) + float(
            spend["unreconciled_reservation_usd"]
        )
        if committed + self.worst_call_usd > self.total_cap_usd + 1e-12:
            raise RuntimeError("physical send would exceed the production spend cap")
        number = len(list((self.run_dir / "attempts").glob("*.json"))) + 1
        marker = {
            "version": 1,
            "attempt": number,
            "source_ordinal": ordinal,
            "repair": repair,
            "reservation_usd": self.worst_call_usd,
            "state": "reserved",
        }
        _write_private_json(self.run_dir / "attempts" / f"{number:09d}.json", marker)
        spend["unreconciled_reservation_usd"] = (
            committed - float(spend["spent_usd"]) + self.worst_call_usd
        )
        _write_private_json(self.run_dir / "spend-ledger.json", spend)
        return number, self.worst_call_usd

    def _update_attempt(self, number: int, **update: object) -> None:
        path = self.run_dir / "attempts" / f"{number:09d}.json"
        marker = _read_private_json(path, label="physical attempt marker")
        if marker.get("version") != 1 or marker.get("attempt") != number:
            raise BlockGateError("physical attempt marker identity is invalid")
        _write_private_json(path, marker | update)

    def _checkpoint_response(
        self,
        *,
        ordinal: int,
        attempt: int,
        reservation: float,
        repair: bool,
        response: Mapping[str, object],
    ) -> Path:
        response_dict = dict(response)
        required = {
            "content",
            "finish_reason",
            "usage",
            "cost_usd",
            "provider",
            "model",
        }
        if not required.issubset(response_dict):
            raise BlockGateError(
                "provider response is missing required accounting fields"
            )
        content = response_dict["content"]
        if type(content) is not str or len(content.encode("utf-8")) > 1024 * 1024:
            raise BlockGateError("provider response content exceeds the durable limit")
        path = self.run_dir / "raw" / f"{ordinal:08d}-{attempt:09d}.json"
        _write_private_json(
            path,
            {
                "version": 1,
                "source_ordinal": ordinal,
                "attempt": attempt,
                "repair": repair,
                "reservation_usd": reservation,
                "response": response_dict,
                "checkpointed_at": time.time(),
            },
        )
        return path

    def _reconcile(self, *, reservation: float, response: Mapping[str, object]) -> None:
        cost = response.get("cost_usd")
        if (
            type(cost) not in {int, float}
            or not math.isfinite(float(cost))
            or not 0 <= float(cost) <= reservation + 1e-12
        ):
            raise BlockGateError("provider cost is invalid or exceeds its reservation")
        usage = response.get("usage")
        if type(usage) is not dict or any(
            type(usage.get(key)) is not int or usage[key] < 0
            for key in ("prompt_tokens", "completion_tokens")
        ):
            raise BlockGateError("provider usage is incomplete")
        spend = self._spend()
        if float(spend["unreconciled_reservation_usd"]) + 1e-12 < reservation:
            raise BlockGateError("spend ledger reservation rollback detected")
        spend["unreconciled_reservation_usd"] = (
            float(spend["unreconciled_reservation_usd"]) - reservation
        )
        spend["spent_usd"] = float(spend["spent_usd"]) + float(cost)
        _write_private_json(self.run_dir / "spend-ledger.json", spend)

    def _reconcile_checkpoint(self, checkpoint: Mapping[str, object]) -> None:
        attempt = checkpoint.get("attempt")
        reservation = checkpoint.get("reservation_usd")
        response = checkpoint.get("response")
        if (
            type(attempt) is not int
            or type(reservation) not in {int, float}
            or type(response) is not dict
        ):
            raise BlockGateError("raw checkpoint accounting identity is invalid")
        marker = _read_private_json(
            self.run_dir / "attempts" / f"{attempt:09d}.json",
            label="physical attempt marker",
        )
        if marker.get("state") == "reconciled":
            return
        if marker.get("state") != "response_checkpointed":
            raise BlockGateError(
                "raw checkpoint does not have a pending attempt marker"
            )
        self._reconcile(reservation=float(reservation), response=response)
        self._update_attempt(attempt, state="reconciled")

    def _conservatively_charge_failed_attempt(
        self, attempt: int, reservation: float
    ) -> None:
        spend = self._spend()
        if float(spend["unreconciled_reservation_usd"]) + 1e-12 < reservation:
            raise BlockGateError("failed attempt reservation is missing")
        spend["unreconciled_reservation_usd"] = (
            float(spend["unreconciled_reservation_usd"]) - reservation
        )
        spend["spent_usd"] = float(spend["spent_usd"]) + reservation
        _write_private_json(self.run_dir / "spend-ledger.json", spend)
        self._update_attempt(attempt, state="transport_error_conservatively_charged")

    def _save_result(self, ordinal: int, value: dict[str, object]) -> None:
        path = self.run_dir / "results" / f"{ordinal:08d}.json"
        if path.exists():
            existing = _read_private_json(path, label="terminal result")
            if existing != value:
                raise BlockGateError("terminal result mutation was attempted")
            return
        _write_private_json(path, value)

    def _read_raw(self, ordinal: int) -> dict[str, object] | None:
        entries = sorted((self.run_dir / "raw").glob(f"{ordinal:08d}-*.json"))
        if not entries:
            return None
        return _read_private_json(entries[-1], label="raw response checkpoint")

    def _terminal_from_response(
        self, record: dict[str, object], checkpoint: Mapping[str, object]
    ) -> tuple[bool, str | None, dict[str, object] | None]:
        response = checkpoint.get("response")
        if type(response) is not dict:
            return False, "raw checkpoint response is invalid", None
        if (
            response.get("provider") != self.provider
            or response.get("model") != self.model_id
        ):
            return False, "provider/model drift", None
        if response.get("finish_reason") != "stop":
            return False, "provider finish reason is not stop", None
        content = response.get("content")
        if type(content) is not str or not content.strip():
            return False, "provider output is empty", None
        try:
            decision = parse_judge_decision(content, version="v2")
        except ValueError as error:
            return False, str(error), None
        return (
            True,
            None,
            {
                "label": decision.label,
                "correctness": decision.correctness,
                "completeness": decision.completeness,
                "relevance": decision.relevance,
                "error_flags": list(decision.error_flags),
                "reference_status": decision.reference_status,
                "confidence": decision.confidence,
                "rationale": decision.rationale,
            },
        )

    def _process_record(self, record: dict[str, object]) -> None:
        ordinal = int(record["source_ordinal"])
        result_path = self.run_dir / "results" / f"{ordinal:08d}.json"
        if result_path.exists():
            return
        checkpoint = self._read_raw(ordinal)
        repair = False
        if checkpoint is not None:
            self._reconcile_checkpoint(checkpoint)
            ok, error, decision = self._terminal_from_response(record, checkpoint)
            if ok:
                self._save_result(
                    ordinal,
                    {
                        "version": 1,
                        "record_id": record["record_id"],
                        "source_ordinal": ordinal,
                        "status": "valid",
                        "attempt": checkpoint["attempt"],
                        "decision": decision,
                    },
                )
                return
            repair = bool(checkpoint.get("repair") is False)
            if not repair:
                self._save_result(
                    ordinal,
                    {
                        "version": 1,
                        "record_id": record["record_id"],
                        "source_ordinal": ordinal,
                        "status": "invalid",
                        "reason": error,
                        "attempt": checkpoint.get("attempt"),
                    },
                )
                return
        for retry in range(MAX_TRANSIENT_RETRIES + 1):
            attempt, reservation = self._reserve_attempt(ordinal, repair)
            try:
                started = time.perf_counter()
                response = self.call(record, repair)
            except (TransientProviderError, TimeoutError, OSError):
                self._conservatively_charge_failed_attempt(attempt, reservation)
                if retry == MAX_TRANSIENT_RETRIES:
                    raise BlockGateError("bounded provider retry limit exhausted")
                continue
            response = dict(response)
            response["latency_seconds"] = time.perf_counter() - started
            checkpoint_path = self._checkpoint_response(
                ordinal=ordinal,
                attempt=attempt,
                reservation=reservation,
                repair=repair,
                response=response,
            )
            self._update_attempt(
                attempt,
                state="response_checkpointed",
                raw_checkpoint=checkpoint_path.name,
                latency_seconds=response["latency_seconds"],
            )
            if response.get("crash_after_checkpoint") is True:
                raise RuntimeError("injected crash after raw checkpoint")
            self._reconcile_checkpoint(
                _read_private_json(checkpoint_path, label="raw response checkpoint")
            )
            checkpoint = _read_private_json(
                checkpoint_path, label="raw response checkpoint"
            )
            ok, error, decision = self._terminal_from_response(record, checkpoint)
            if ok:
                self._save_result(
                    ordinal,
                    {
                        "version": 1,
                        "record_id": record["record_id"],
                        "source_ordinal": ordinal,
                        "status": "valid",
                        "attempt": attempt,
                        "decision": decision,
                    },
                )
                return
            if repair or response.get("finish_reason") != "stop":
                self._save_result(
                    ordinal,
                    {
                        "version": 1,
                        "record_id": record["record_id"],
                        "source_ordinal": ordinal,
                        "status": "invalid",
                        "reason": error,
                        "attempt": attempt,
                    },
                )
                return
            repair = True
            break
        if repair:
            self._process_record(record)

    def _validate_block(
        self, block_index: int, records: list[dict[str, object]]
    ) -> dict[str, object]:
        expected = [record["record_id"] for record in records]
        results = [
            _read_private_json(
                self.run_dir / "results" / f"{int(record['source_ordinal']):08d}.json",
                label="terminal result",
            )
            for record in records
        ]
        actual = [result.get("record_id") for result in results]
        if actual != expected or len(set(actual)) != len(expected):
            raise BlockGateError(
                "block identity coverage is missing, duplicate, or unexpected"
            )
        invalid = [result for result in results if result.get("status") != "valid"]
        if invalid:
            reason = invalid[0].get("reason", "unknown")
            raise BlockGateError(
                f"terminal-invalid result prevents block promotion: {reason}"
            )
        attempts = list((self.run_dir / "attempts").glob("*.json"))
        raw = list((self.run_dir / "raw").glob("*.json"))
        if len(raw) > len(attempts):
            raise BlockGateError(
                "raw checkpoint/attempt ledger count does not reconcile"
            )
        spend = self._spend()
        if float(spend["unreconciled_reservation_usd"]) > 1e-12:
            raise BlockGateError("spend ledger has an unreconciled reservation")
        receipt_body = {
            "version": 1,
            "block_index": block_index,
            "start_ordinal": records[0]["source_ordinal"],
            "end_ordinal": records[-1]["source_ordinal"],
            "expected_count": len(records),
            "record_ids": expected,
            "inventory_sha256": self._input_manifest["inventory_sha256"],
            "inventory_manifest_sha256": self._input_manifest["manifest_sha256"],
            "prompt_sha256": self.contract["prompt_sha256"],
            "schema_sha256": self.contract["schema_sha256"],
            "model_id": self.model_id,
            "provider": self.provider,
            "attempt_count": len(attempts),
            "raw_checkpoint_count": len(raw),
            "spent_usd": spend["spent_usd"],
            "unreconciled_reservation_usd": spend["unreconciled_reservation_usd"],
        }
        return receipt_body | {"receipt_sha256": _json_hash(receipt_body)}

    def _write_or_validate_receipt(
        self, block_index: int, records: list[dict[str, object]]
    ) -> None:
        path = self.run_dir / "blocks" / f"{block_index:06d}-receipt.json"
        computed = self._validate_block(block_index, records)
        if path.exists():
            if _read_private_json(path, label="block receipt") != computed:
                raise BlockGateError("block receipt no longer proves current run state")
        else:
            _write_private_json(path, computed)

    def _anomalies(
        self, block_index: int, records: list[dict[str, object]]
    ) -> list[str]:
        values = [
            _read_private_json(
                self.run_dir / "results" / f"{int(record['source_ordinal']):08d}.json",
                label="terminal result",
            )
            for record in records
        ]
        diagnostics: list[str] = []
        if len(values) >= BLOCK_SIZE:
            labels = Counter(str(value["decision"]["label"]) for value in values)
            if max(labels.values()) == len(values) and len(labels) == 1:
                diagnostics.append("all labels in a full block are identical")
        spend = self._spend()
        completed = sum(
            1
            for path in (self.run_dir / "results").glob("*.json")
            if _read_private_json(path, label="terminal result").get("status")
            == "valid"
        )
        if (
            completed >= BLOCK_SIZE
            and float(spend["spent_usd"]) * self._record_count / completed
            > PAUSE_PROJECTION_USD
        ):
            diagnostics.append(
                "projected DeepSeek spend exceeds USD 50 pause threshold"
            )
        if diagnostics:
            _write_private_json(
                self.run_dir / "anomalies" / f"{block_index:06d}.json",
                {"version": 1, "block_index": block_index, "diagnostics": diagnostics},
            )
        return diagnostics

    def run(self, *, max_blocks: int | None = None) -> dict[str, object]:
        processed = 0
        for block_index, (start, end) in enumerate(block_slices(self._record_count)):
            records = list(_iter_inventory(self.inventory, start, end))
            if len(records) != end - start:
                raise BlockGateError(
                    "inventory ended before the committed record count"
                )
            receipt_path = self.run_dir / "blocks" / f"{block_index:06d}-receipt.json"
            if receipt_path.exists():
                self._write_or_validate_receipt(block_index, records)
                continue
            for record in records:
                self._process_record(record)
            self._write_or_validate_receipt(block_index, records)
            processed += 1
            diagnostics = self._anomalies(block_index, records)
            if diagnostics:
                raise SoftAnomalyPause("; ".join(diagnostics))
            if max_blocks is not None and processed >= max_blocks:
                break
        completed = len(list((self.run_dir / "results").glob("*.json")))
        return {
            "record_count": self._record_count,
            "completed_records": completed,
            "processed_blocks": processed,
            "run_dir": str(self.run_dir),
        }
