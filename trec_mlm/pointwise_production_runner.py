"""Crash-safe, block-gated runner for the private DeepSeek pointwise run.

The module deliberately keeps the network boundary injectable.  The production
CLI supplies it; tests use deterministic local functions and never contact a
provider.  Every physical send has an attempt marker and conservative spend
reservation on disk before the boundary is crossed.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterator, Mapping
import fcntl
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
MAX_REPAIRS = 2
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
        fault_hook: Callable[[str], None] | None = None,
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
        self._fault_hook = fault_hook or (lambda _boundary: None)
        self._lock_path = self.run_dir / "run.lock"
        self._lock_descriptor: int | None = None
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

    def _acquire_run_lock(self) -> None:
        if self._lock_descriptor is not None:
            return
        try:
            descriptor = os.open(
                self._lock_path,
                os.O_RDWR | os.O_CREAT,
                _FILE_MODE,
            )
            os.fchmod(descriptor, _FILE_MODE)
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            try:
                os.close(descriptor)
            except UnboundLocalError:
                pass
            raise BlockGateError("production run is already active") from error
        try:
            os.ftruncate(descriptor, 0)
            os.write(descriptor, b"private production runner advisory lock\n")
            os.fsync(descriptor)
        except BaseException:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            raise
        self._lock_descriptor = descriptor

    def _release_run_lock(self) -> None:
        if self._lock_descriptor is not None:
            fcntl.flock(self._lock_descriptor, fcntl.LOCK_UN)
            os.close(self._lock_descriptor)
            self._lock_descriptor = None

    def _abandon_lock_owner_for_test(self) -> None:
        """Test-only simulation of OS releasing a lock when an owner dies."""
        if self._lock_descriptor is None:
            raise RuntimeError("test lock owner is not active")
        os.close(self._lock_descriptor)
        self._lock_descriptor = None

    def _recover_crash_windows(self) -> None:
        """Resolve only unambiguous raw writes; never resend a reserved call."""
        raw_by_attempt: dict[int, Path] = {}
        for raw_path in (self.run_dir / "raw").glob("*.json"):
            raw = _read_private_json(raw_path, label="raw response checkpoint")
            attempt = raw.get("attempt")
            if type(attempt) is not int or attempt in raw_by_attempt:
                raise BlockGateError("raw checkpoint attempt identity is invalid")
            raw_by_attempt[attempt] = raw_path
        for marker_path in sorted((self.run_dir / "attempts").glob("*.json")):
            marker = _read_private_json(marker_path, label="physical attempt marker")
            attempt = marker.get("attempt")
            if type(attempt) is not int:
                raise BlockGateError("physical attempt identity is invalid")
            raw_path = raw_by_attempt.get(attempt)
            if marker.get("state") == "reserved":
                if raw_path is not None:
                    raw = _read_private_json(raw_path, label="raw response checkpoint")
                    if raw.get("source_ordinal") != marker.get("source_ordinal"):
                        raise BlockGateError("raw checkpoint/attempt identity mismatch")
                    self._update_attempt(
                        attempt,
                        state="response_checkpointed",
                        raw_checkpoint=raw_path.name,
                    )
                    continue
                baseline_spent, baseline_unreconciled = self._settlement_baseline(
                    attempt
                )
                spend = self._spend()
                if math.isclose(
                    float(spend["spent_usd"]), baseline_spent, abs_tol=1e-12
                ) and math.isclose(
                    float(spend["unreconciled_reservation_usd"]),
                    baseline_unreconciled,
                    abs_tol=1e-12,
                ):
                    # The marker write precedes the reservation-ledger write.
                    # This exact shape proves no caller could have crossed the
                    # provider boundary, so retain an audit marker but charge
                    # nothing and permit the record to be scheduled afresh.
                    self._update_attempt(attempt, state="cancelled_pre_send")
                    continue
                self._conservatively_charge_failed_attempt(
                    attempt, float(marker["reservation_usd"])
                )
                self._update_attempt(attempt, state="ambiguous_conservatively_charged")
                raise BlockGateError(
                    "ambiguous reserved provider call requires manual resolution"
                )
            if marker.get("state") == "ambiguous_conservatively_charged":
                raise BlockGateError(
                    "ambiguous reserved provider call requires manual resolution"
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
        self._fault_hook("after_attempt_marker_write")
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

    def _raw_checkpoint_for_attempt(self, attempt: int) -> dict[str, object] | None:
        matches = list((self.run_dir / "raw").glob(f"*-{attempt:09d}.json"))
        if len(matches) > 1:
            raise BlockGateError("multiple raw checkpoints share an attempt")
        if not matches:
            return None
        return _read_private_json(matches[0], label="raw response checkpoint")

    def _settlement_baseline(self, exclude_attempt: int) -> tuple[float, float]:
        """Replay all settled markers except one transition being recovered."""
        spent, unreconciled = 0.0, 0.0
        for path in (self.run_dir / "attempts").glob("*.json"):
            marker = _read_private_json(path, label="physical attempt marker")
            attempt, reservation = marker.get("attempt"), marker.get("reservation_usd")
            if type(attempt) is not int or type(reservation) not in {int, float}:
                raise BlockGateError("attempt marker accounting identity is invalid")
            if attempt == exclude_attempt:
                continue
            state = marker.get("state")
            if state == "reconciled":
                checkpoint = self._raw_checkpoint_for_attempt(attempt)
                if checkpoint is None or type(checkpoint.get("response")) is not dict:
                    raise BlockGateError(
                        "reconciled marker is missing its raw checkpoint"
                    )
                cost = checkpoint["response"].get("cost_usd")
                if type(cost) not in {int, float}:
                    raise BlockGateError("reconciled raw checkpoint cost is invalid")
                spent += float(cost)
            elif state in {
                "transport_error_conservatively_charged",
                "ambiguous_conservatively_charged",
            }:
                spent += float(reservation)
            elif state == "cancelled_pre_send":
                continue
            elif state in {"reserved", "response_checkpointed"}:
                unreconciled += float(reservation)
            else:
                raise BlockGateError("attempt marker state is invalid")
        return spent, unreconciled

    def _response_cost(
        self, response: Mapping[str, object], reservation: float
    ) -> float:
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
        return float(cost)

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
        baseline_spent, baseline_unreconciled = self._settlement_baseline(attempt)
        cost = self._response_cost(response, float(reservation))
        spend = self._spend()
        already_settled = math.isclose(
            float(spend["spent_usd"]), baseline_spent + cost, abs_tol=1e-12
        ) and math.isclose(
            float(spend["unreconciled_reservation_usd"]),
            baseline_unreconciled,
            abs_tol=1e-12,
        )
        not_settled = math.isclose(
            float(spend["spent_usd"]), baseline_spent, abs_tol=1e-12
        ) and math.isclose(
            float(spend["unreconciled_reservation_usd"]),
            baseline_unreconciled + float(reservation),
            abs_tol=1e-12,
        )
        if not already_settled and not not_settled:
            raise BlockGateError("spend ledger cannot be replayed for raw checkpoint")
        if not already_settled:
            spend["spent_usd"] = baseline_spent + cost
            spend["unreconciled_reservation_usd"] = baseline_unreconciled
            _write_private_json(self.run_dir / "spend-ledger.json", spend)
            self._fault_hook("after_reconcile_ledger_write")
        self._update_attempt(attempt, state="reconciled")

    def _conservatively_charge_failed_attempt(
        self, attempt: int, reservation: float
    ) -> None:
        baseline_spent, baseline_unreconciled = self._settlement_baseline(attempt)
        spend = self._spend()
        already_settled = math.isclose(
            float(spend["spent_usd"]), baseline_spent + reservation, abs_tol=1e-12
        ) and math.isclose(
            float(spend["unreconciled_reservation_usd"]),
            baseline_unreconciled,
            abs_tol=1e-12,
        )
        not_settled = math.isclose(
            float(spend["spent_usd"]), baseline_spent, abs_tol=1e-12
        ) and math.isclose(
            float(spend["unreconciled_reservation_usd"]),
            baseline_unreconciled + reservation,
            abs_tol=1e-12,
        )
        if not already_settled and not not_settled:
            raise BlockGateError("spend ledger cannot be replayed for failed attempt")
        if not already_settled:
            spend["spent_usd"] = baseline_spent + reservation
            spend["unreconciled_reservation_usd"] = baseline_unreconciled
            _write_private_json(self.run_dir / "spend-ledger.json", spend)
            self._fault_hook("after_conservative_ledger_write")
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

    def _repair_count(self, ordinal: int) -> int:
        return sum(
            1
            for path in (self.run_dir / "raw").glob(f"{ordinal:08d}-*.json")
            if _read_private_json(path, label="raw response checkpoint").get("repair")
            is True
        )

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
            repair = self._repair_count(ordinal) < MAX_REPAIRS
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
            self._fault_hook("after_reservation")
            try:
                started = time.perf_counter()
                response = self.call(record, repair)
            except (TransientProviderError, TimeoutError, OSError):
                self._conservatively_charge_failed_attempt(attempt, reservation)
                if retry == MAX_TRANSIENT_RETRIES:
                    raise BlockGateError("bounded provider retry limit exhausted")
                continue
            self._fault_hook("after_provider_return")
            response = dict(response)
            response["latency_seconds"] = time.perf_counter() - started
            checkpoint_path = self._checkpoint_response(
                ordinal=ordinal,
                attempt=attempt,
                reservation=reservation,
                repair=repair,
                response=response,
            )
            self._fault_hook("after_raw_write")
            self._update_attempt(
                attempt,
                state="response_checkpointed",
                raw_checkpoint=checkpoint_path.name,
                latency_seconds=response["latency_seconds"],
            )
            self._fault_hook("after_attempt_state")
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
            if response.get("finish_reason") != "stop":
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
            if not repair or self._repair_count(ordinal) < MAX_REPAIRS:
                repair = True
                break
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
        markers: dict[int, dict[str, object]] = {}
        for path in sorted((self.run_dir / "attempts").glob("*.json")):
            marker = _read_private_json(path, label="physical attempt marker")
            attempt = marker.get("attempt")
            if (
                type(attempt) is not int
                or attempt in markers
                or path.name != f"{attempt:09d}.json"
            ):
                raise BlockGateError("physical attempt marker mapping is invalid")
            markers[attempt] = marker
        raw_by_attempt: dict[int, dict[str, object]] = {}
        for path in sorted((self.run_dir / "raw").glob("*.json")):
            checkpoint = _read_private_json(path, label="raw response checkpoint")
            attempt = checkpoint.get("attempt")
            ordinal = checkpoint.get("source_ordinal")
            if (
                type(attempt) is not int
                or type(ordinal) is not int
                or attempt in raw_by_attempt
                or path.name != f"{ordinal:08d}-{attempt:09d}.json"
            ):
                raise BlockGateError("raw checkpoint mapping is invalid")
            raw_by_attempt[attempt] = checkpoint
        for record, result in zip(records, results, strict=True):
            ordinal = record["source_ordinal"]
            if (
                set(result)
                != {
                    "version",
                    "record_id",
                    "source_ordinal",
                    "status",
                    "attempt",
                    "decision",
                }
                or result.get("version") != 1
                or result.get("source_ordinal") != ordinal
            ):
                raise BlockGateError("terminal result identity is invalid")
            attempt = result.get("attempt")
            if (
                type(attempt) is not int
                or attempt not in markers
                or attempt not in raw_by_attempt
            ):
                raise BlockGateError("terminal result attempt/raw mapping is invalid")
            marker, checkpoint = markers[attempt], raw_by_attempt[attempt]
            if (
                marker.get("source_ordinal") != ordinal
                or checkpoint.get("source_ordinal") != ordinal
                or marker.get("state") != "reconciled"
            ):
                raise BlockGateError("terminal result attempt state is invalid")
            valid, error, decision = self._terminal_from_response(record, checkpoint)
            if not valid or decision != result.get("decision"):
                raise BlockGateError(
                    f"terminal decision does not satisfy v2 contract: {error or 'raw mismatch'}"
                )
        recomputed_spend = 0.0
        for attempt, marker in markers.items():
            reservation = marker.get("reservation_usd")
            if (
                type(reservation) not in {int, float}
                or float(reservation) != self.worst_call_usd
            ):
                raise BlockGateError("attempt reservation is invalid")
            state = marker.get("state")
            checkpoint = raw_by_attempt.get(attempt)
            if state == "reconciled":
                if (
                    checkpoint is None
                    or checkpoint.get("reservation_usd") != reservation
                ):
                    raise BlockGateError("reconciled attempt/raw mapping is invalid")
                response = checkpoint.get("response")
                if type(response) is not dict:
                    raise BlockGateError("raw checkpoint response is invalid")
                cost = response.get("cost_usd")
                if type(cost) not in {int, float} or not 0 <= float(cost) <= float(
                    reservation
                ):
                    raise BlockGateError("raw checkpoint cost is invalid")
                usage = response.get("usage")
                if type(usage) is not dict or any(
                    type(usage.get(key)) is not int or usage[key] < 0
                    for key in ("prompt_tokens", "completion_tokens")
                ):
                    raise BlockGateError("raw checkpoint usage is invalid")
                recomputed_spend += float(cost)
            elif state == "transport_error_conservatively_charged":
                if checkpoint is not None:
                    raise BlockGateError(
                        "failed transport has an unexpected raw checkpoint"
                    )
                recomputed_spend += float(reservation)
            elif state == "cancelled_pre_send":
                if checkpoint is not None:
                    raise BlockGateError("cancelled pre-send marker has raw checkpoint")
            else:
                raise BlockGateError("attempt state is not replayable")
        spend = self._spend()
        if float(spend["unreconciled_reservation_usd"]) > 1e-12:
            raise BlockGateError("spend ledger has an unreconciled reservation")
        if not math.isclose(
            float(spend["spent_usd"]), recomputed_spend, rel_tol=0.0, abs_tol=1e-12
        ):
            raise BlockGateError(
                "spend ledger does not match replayed physical attempts"
            )
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
            "attempt_count": len(markers),
            "raw_checkpoint_count": len(raw_by_attempt),
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

    @staticmethod
    def _distribution(values: list[object]) -> dict[str, int]:
        return {
            key: count
            for key, count in sorted(Counter(str(value) for value in values).items())
        }

    @staticmethod
    def _numeric_summary(values: list[float]) -> dict[str, float | int]:
        if not values:
            return {"count": 0, "min": 0.0, "mean": 0.0, "max": 0.0}
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "min": ordered[0],
            "mean": sum(ordered) / len(ordered),
            "max": ordered[-1],
        }

    def _prior_metrics(self, block_index: int) -> dict[str, object] | None:
        if block_index == 0:
            return None
        path = self.run_dir / "anomalies" / f"{block_index - 1:06d}.json"
        if not path.exists():
            return None
        value = _read_private_json(path, label="prior anomaly diagnostics")
        metrics = value.get("metrics")
        return metrics if type(metrics) is dict else None

    @staticmethod
    def _distribution_shift(current: dict[str, int], prior: dict[str, int]) -> float:
        current_total, prior_total = sum(current.values()), sum(prior.values())
        if current_total == 0 or prior_total == 0:
            return 0.0
        return (
            sum(
                abs(
                    current.get(key, 0) / current_total
                    - prior.get(key, 0) / prior_total
                )
                for key in set(current) | set(prior)
            )
            / 2
        )

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
        decisions = [value["decision"] for value in values]
        assert all(type(value) is dict for value in decisions)
        raw = [
            _read_private_json(
                self.run_dir
                / "raw"
                / f"{int(value['source_ordinal']):08d}-{int(value['attempt']):09d}.json",
                label="raw response checkpoint",
            )
            for value in values
        ]
        response_values = [value["response"] for value in raw]
        assert all(type(value) is dict for value in response_values)
        metrics: dict[str, object] = {
            "label": self._distribution([value["label"] for value in decisions]),
            "correctness": self._distribution(
                [value["correctness"] for value in decisions]
            ),
            "completeness": self._distribution(
                [value["completeness"] for value in decisions]
            ),
            "relevance": self._distribution(
                [value["relevance"] for value in decisions]
            ),
            "error_flags": self._distribution(
                [flag for value in decisions for flag in value["error_flags"]]
            ),
            "confidence": self._distribution(
                [value["confidence"] for value in decisions]
            ),
            "reference_status": self._distribution(
                [value["reference_status"] for value in decisions]
            ),
            "output_sha256": self._distribution(
                [
                    hashlib.sha256(str(value["content"]).encode()).hexdigest()
                    for value in response_values
                ]
            ),
            "provider_model": self._distribution(
                [f"{value['provider']}::{value['model']}" for value in response_values]
            ),
            "prompt_tokens": self._numeric_summary(
                [float(value["usage"]["prompt_tokens"]) for value in response_values]
            ),
            "completion_tokens": self._numeric_summary(
                [
                    float(value["usage"]["completion_tokens"])
                    for value in response_values
                ]
            ),
            "cost_usd": self._numeric_summary(
                [float(value["cost_usd"]) for value in response_values]
            ),
            "latency_seconds": self._numeric_summary(
                [float(value.get("latency_seconds", 0.0)) for value in response_values]
            ),
            "repair_rate": sum(1 for value in raw if value.get("repair") is True)
            / len(raw),
            "retry_rate": (
                len(
                    [
                        path
                        for path in (self.run_dir / "attempts").glob("*.json")
                        if _read_private_json(
                            path, label="physical attempt marker"
                        ).get("state")
                        != "cancelled_pre_send"
                    ]
                )
                - len([path for path in (self.run_dir / "results").glob("*.json")])
            )
            / max(1, len(values)),
        }
        diagnostics: list[str] = []
        labels = metrics["label"]
        repeated = metrics["output_sha256"]
        providers = metrics["provider_model"]
        assert (
            isinstance(labels, dict)
            and isinstance(repeated, dict)
            and isinstance(providers, dict)
        )
        if len(values) >= BLOCK_SIZE and max(labels.values()) == len(values):
            diagnostics.append("all labels in a full block are identical")
        if (
            len(values) >= BLOCK_SIZE
            and max(repeated.values()) * 20 >= len(values) * 19
        ):
            diagnostics.append("repeated provider output exceeds 95 percent of block")
        if set(providers) != {f"{self.provider}::{self.model_id}"}:
            diagnostics.append("provider/model drift observed in anomaly metrics")
        if float(metrics["repair_rate"]) > 0.25 or float(metrics["retry_rate"]) > 0.25:
            diagnostics.append("repair or retry rate exceeds 25 percent")
        prior = self._prior_metrics(block_index)
        if prior is not None:
            if len(values) >= BLOCK_SIZE:
                for name in (
                    "label",
                    "correctness",
                    "completeness",
                    "relevance",
                    "error_flags",
                    "confidence",
                    "reference_status",
                ):
                    previous, current = prior.get(name), metrics[name]
                    if (
                        type(previous) is dict
                        and type(current) is dict
                        and self._distribution_shift(current, previous) > 0.8
                    ):
                        diagnostics.append(f"{name} distribution shifted more than 0.8")
            for name in (
                "prompt_tokens",
                "completion_tokens",
                "cost_usd",
                "latency_seconds",
            ):
                previous = prior.get(name)
                current = metrics[name]
                if (
                    type(previous) is dict
                    and type(current) is dict
                    and float(previous.get("mean", 0.0)) > 0
                    and (
                        float(current.get("mean", 0.0)) > float(previous["mean"]) * 5
                        or float(current.get("mean", 0.0)) * 5 < float(previous["mean"])
                    )
                ):
                    diagnostics.append(f"{name} mean increased more than fivefold")
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
        _write_private_json(
            self.run_dir / "anomalies" / f"{block_index:06d}.json",
            {
                "version": 1,
                "block_index": block_index,
                "metrics": metrics,
                "diagnostics": diagnostics,
            },
        )
        if diagnostics:
            token = _json_hash(
                {
                    "block_index": block_index,
                    "diagnostics": diagnostics,
                    "run": self._input_manifest["manifest_sha256"],
                }
            )
            _write_private_json(
                self.run_dir / "anomalies" / f"{block_index:06d}-pause.json",
                {
                    "version": 1,
                    "block_index": block_index,
                    "diagnostics": diagnostics,
                    "pause_token": token,
                    "state": "pending",
                },
            )
        return diagnostics

    def _acknowledge_pending_pause(self, token: str | None) -> None:
        pending = []
        for path in sorted((self.run_dir / "anomalies").glob("*-pause.json")):
            value = _read_private_json(path, label="anomaly pause")
            if value.get("state") == "pending":
                pending.append((path, value))
        if not pending:
            return
        path, pause = pending[0]
        if type(token) is not str or token != pause.get("pause_token"):
            raise SoftAnomalyPause(
                "anomaly acknowledgement token is required before resume"
            )
        _write_private_json(path, pause | {"state": "acknowledged"})

    def run(
        self, *, max_blocks: int | None = None, acknowledgement_token: str | None = None
    ) -> dict[str, object]:
        self._acquire_run_lock()
        try:
            self._recover_crash_windows()
            self._acknowledge_pending_pause(acknowledgement_token)
            return self._run_unlocked(max_blocks=max_blocks)
        finally:
            self._release_run_lock()

    def _run_unlocked(self, *, max_blocks: int | None = None) -> dict[str, object]:
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
