"""Independent validation for the frozen pointwise-ranking Phase 0 result.

This module deliberately uses only the Python standard library.  In particular,
it must remain independent of the ranking evaluator and repository metrics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


SYSTEMS = ("control", "prior_only", "routing_only", "pointwise_only", "combined")
EXPECTED_QRELS_SHA256 = "3a1579034f0670604923fd37559ad217b341942c765412b38d3aedc722b89f35"
EXPECTED_MANIFEST_SHA256 = "6e5cdbde8785aa4caf56f94180932d314aa76848de33004812cd817a6ec2b025"
EXPECTED_AGGREGATE_SHA256 = "f5e7575e1ca83391c852555441a1ba0488e7eacd4b493da1cf0a8fd6aa4a5e4b"
EXPECTED_EVALUATOR_RESULT_SHA256 = "449e325582b29dff37d917a55f48837a4167e0845c1da4f42a672955ea1d0be2"
EXPECTED_PER_QUERY_SHA256 = "e2193404d8f1052199b6b6a396d78556e48bb84b145a6e7adda125634218d889"
EXPECTED_CANDIDATE_SLICES_SHA256 = "4d56304bf73f6d0e6bfd3c5612b922c3c3ce6e43f349857fc180aab9d83f3329"
EXPECTED_DENSE_RESULTS_SHA256 = "3ca4252cb37f12e9272561422fd2a45d0dcbf741770d0f544efda8acf4ddf902"
EXPECTED_DENSE_PER_QUERY_SHA256 = "ae4a0cf8bc7eb0642e1c0d315cafa3a56cf60b3d6bfb6ab63e80fb2d0de56437"
ABSOLUTE_TOLERANCE = 5e-12


class ValidationError(ValueError):
    """Raised when frozen validation evidence is inconsistent."""


@dataclass(frozen=True)
class LoadedRankings:
    """Strictly validated qrels and TREC rankings."""

    query_ids: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    qrels: dict[str, dict[str, float]]
    runs: dict[str, dict[str, tuple[str, ...]]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, label: str) -> None:
    observed = _sha256_file(path)
    _require(observed == expected, f"{label} SHA-256 mismatch: {observed}")


def _close(actual: float, expected: float, label: str, *, tolerance: float = ABSOLUTE_TOLERANCE) -> None:
    _require(
        math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance),
        f"{label} mismatch: actual={actual!r}, expected={expected!r}",
    )


def phase0_query_metrics(
    ranking: Sequence[str], grades: Mapping[str, float], *, cutoff: int = 10
) -> tuple[float, float]:
    """Return raw-gain nDCG and grade-positive reciprocal rank."""

    dcg = sum(
        grades[candidate_id] / math.log2(rank + 1)
        for rank, candidate_id in enumerate(ranking[:cutoff], 1)
    )
    ideal = sorted(grades.values(), reverse=True)[:cutoff]
    idcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1))
    ndcg = dcg / idcg if idcg else 0.0
    reciprocal_rank = next(
        (
            1.0 / rank
            for rank, candidate_id in enumerate(ranking, 1)
            if grades[candidate_id] > 0
        ),
        0.0,
    )
    return ndcg, reciprocal_rank


def load_qrels_and_runs(
    qrels_path: Path,
    run_paths: Mapping[str, Path],
    *,
    expected_query_count: int,
    expected_candidate_count: int,
) -> LoadedRankings:
    """Read and strictly reconcile a complete qrel matrix and TREC runs."""

    qrels: dict[str, dict[str, float]] = defaultdict(dict)
    with qrels_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.split()
            _require(len(fields) == 4, f"qrels line {line_number} must have four fields")
            query_id, _, candidate_id, grade_text = fields
            _require(
                candidate_id not in qrels[query_id],
                f"duplicate qrel {query_id}/{candidate_id}",
            )
            grade = float(grade_text)
            _require(math.isfinite(grade) and grade >= 0, "qrel grades must be finite and nonnegative")
            qrels[query_id][candidate_id] = grade

    query_ids = tuple(sorted(qrels))
    _require(len(query_ids) == expected_query_count, "unexpected qrel query count")
    candidate_sets = {frozenset(grades) for grades in qrels.values()}
    _require(len(candidate_sets) == 1, "qrel candidate sets differ")
    candidate_ids = tuple(sorted(next(iter(candidate_sets))))
    _require(len(candidate_ids) == expected_candidate_count, "unexpected qrel candidate count")

    runs: dict[str, dict[str, tuple[str, ...]]] = {}
    expected_ranks = list(range(1, expected_candidate_count + 1))
    for system, path in run_paths.items():
        rows: dict[str, list[tuple[int, str]]] = defaultdict(list)
        seen: set[tuple[str, str]] = set()
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                fields = line.split()
                _require(len(fields) == 6, f"{path.name}:{line_number} must have six fields")
                query_id, q0, candidate_id, rank_text, score_text, _ = fields
                _require(q0 == "Q0", f"{path.name}:{line_number} requires Q0")
                rank = int(rank_text)
                score = float(score_text)
                _require(math.isfinite(score), f"{path.name}:{line_number} has non-finite score")
                _require((query_id, candidate_id) not in seen, f"duplicate run row {query_id}/{candidate_id}")
                seen.add((query_id, candidate_id))
                rows[query_id].append((rank, candidate_id))
        _require(set(rows) == set(query_ids), f"{system} query identities differ")
        ranking: dict[str, tuple[str, ...]] = {}
        for query_id, pairs in rows.items():
            _require([rank for rank, _ in pairs] == expected_ranks, f"{system}/{query_id} ranks differ")
            ordered = tuple(candidate_id for _, candidate_id in pairs)
            _require(set(ordered) == set(candidate_ids), f"{system}/{query_id} candidate identities differ")
            ranking[query_id] = ordered
        runs[system] = ranking

    return LoadedRankings(
        query_ids=query_ids,
        candidate_ids=candidate_ids,
        qrels=dict(qrels),
        runs=runs,
    )


def finalize_evaluator_payload(
    evaluator_payload: bytes, independent_validation: Mapping[str, Any]
) -> bytes:
    """Replace the evaluator's false sentinel and return canonical JSON bytes."""

    payload = json.loads(evaluator_payload)
    _require(
        payload.get("independent_validation") is False,
        "evaluator independent_validation must be false",
    )
    payload["independent_validation"] = dict(independent_validation)
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


_CONTENT_PATTERNS = (
    ("private path", re.compile(rb"(?<![A-Za-z0-9])(?:/home/|/Users/|[A-Za-z]:[\\/])")),
    ("credential", re.compile(rb"\bsk-[A-Za-z0-9_-]{16,}\b")),
    ("credential", re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("credential", re.compile(rb"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{12,}")),
    ("credential", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    (
        "credential",
        re.compile(
            rb"(?i)\b(?:api[_-]?key|secret|password|access[_-]?token)"
            rb"\s*[:=]\s*[\"']?[^\s,\"']{8,}"
        ),
    ),
)
_FORBIDDEN_FIELDS = {
    "api_key",
    "api_state",
    "completion",
    "completions",
    "input_tokens",
    "messages",
    "output_tokens",
    "prompt",
    "question",
    "question_text",
    "reference",
    "reference_text",
    "request_id",
    "response",
    "response_id",
    "response_text",
    "token_usage",
    "usage",
}


def validate_publication_content(payloads: Sequence[bytes]) -> None:
    """Reject private paths, credentials, raw text, prompts, and API state."""

    for payload in payloads:
        for label, pattern in _CONTENT_PATTERNS:
            _require(pattern.search(payload) is None, f"publication contains {label}")

    try:
        result = json.loads(payloads[0])
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValidationError("result publication must be UTF-8 JSON") from error
    keys: list[str] = []

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                keys.append(str(key).casefold())
                walk(nested)
        elif isinstance(value, list):
            for nested in value:
                walk(nested)

    walk(result)
    forbidden = sorted(set(keys) & _FORBIDDEN_FIELDS)
    _require(not forbidden, f"publication contains forbidden field: {forbidden}")

    for payload in payloads[1:]:
        text = payload.decode("utf-8")
        header = next(csv.reader(text.splitlines(), delimiter="\t"))
        forbidden = sorted({column.casefold() for column in header} & _FORBIDDEN_FIELDS)
        _require(not forbidden, f"publication contains forbidden field: {forbidden}")


def _manifest_entries(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {
            "byte_size": path.stat().st_size,
            "sha256": _sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _manifest_aggregate(entries: Mapping[str, Mapping[str, Any]]) -> str:
    ordered = [
        [name, int(entries[name]["byte_size"]), str(entries[name]["sha256"])]
        for name in sorted(entries)
    ]
    payload = json.dumps(ordered, separators=(",", ":"))
    return _sha256_bytes(payload.encode("utf-8"))


def _evaluate(
    ranking: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    query_ids: Sequence[str] | None = None,
    candidate_ids: Sequence[str] | None = None,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float]]:
    selected_queries = tuple(sorted(qrels)) if query_ids is None else tuple(sorted(query_ids))
    selected_candidates = None if candidate_ids is None else frozenset(candidate_ids)
    ndcg: dict[str, float] = {}
    reciprocal_rank: dict[str, float] = {}
    for query_id in selected_queries:
        if selected_candidates is None:
            order = ranking[query_id]
            grades = qrels[query_id]
        else:
            order = tuple(
                candidate_id
                for candidate_id in ranking[query_id]
                if candidate_id in selected_candidates
            )
            grades = {
                candidate_id: qrels[query_id][candidate_id]
                for candidate_id in selected_candidates
            }
        ndcg[query_id], reciprocal_rank[query_id] = phase0_query_metrics(order, grades)
    summary = {
        "mrr": statistics.fmean(reciprocal_rank.values()) if selected_queries else None,
        "ndcg_at_10": statistics.fmean(ndcg.values()) if selected_queries else None,
        "query_count": len(selected_queries),
        "zero_ndcg_queries": sum(value == 0.0 for value in ndcg.values()),
    }
    return summary, ndcg, reciprocal_rank


def _check_summary(actual: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    _require(actual["query_count"] == expected["query_count"], f"{label} query count mismatch")
    _require(
        actual["zero_ndcg_queries"] == expected["zero_ndcg_queries"],
        f"{label} zero count mismatch",
    )
    if actual["ndcg_at_10"] is None:
        _require(expected["ndcg_at_10"] is None and expected["mrr"] is None, f"{label} empty metrics mismatch")
    else:
        _close(actual["ndcg_at_10"], expected["ndcg_at_10"], f"{label} nDCG")
        _close(actual["mrr"], expected["mrr"], f"{label} MRR")


def _linear_percentile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return sorted_values[low]
    fraction = position - low
    return sorted_values[low] * (1.0 - fraction) + sorted_values[high] * fraction


def _paired_bootstrap(
    candidate: Mapping[str, float], control: Mapping[str, float]
) -> dict[str, float]:
    query_ids = tuple(sorted(control))
    differences = [candidate[query_id] - control[query_id] for query_id in query_ids]
    generator = random.Random(31)
    values = []
    for _ in range(10_000):
        sample = [
            differences[generator.randrange(len(differences))]
            for _ in differences
        ]
        values.append(statistics.fmean(sample))
    values.sort()
    return {
        "lower": _linear_percentile(values, 0.025),
        "mean_difference": statistics.fmean(differences),
        "upper": _linear_percentile(values, 0.975),
    }


def _quartiles(values: Mapping[str, float]) -> dict[str, int]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    count = len(ordered)
    return {
        identifier: 1 + (4 * position // count)
        for position, (identifier, _) in enumerate(ordered)
    }


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _validate_manifest(artifact_dir: Path) -> dict[str, Any]:
    manifest_path = artifact_dir / "manifest.json"
    _require_hash(manifest_path, EXPECTED_MANIFEST_SHA256, "artifact manifest")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = _manifest_entries(artifact_dir)
    _require(entries == manifest.get("files"), "artifact manifest file inventory mismatch")
    aggregate = _manifest_aggregate(entries)
    _require(aggregate == EXPECTED_AGGREGATE_SHA256, "artifact aggregate SHA-256 mismatch")
    _require(manifest.get("aggregate_sha256") == aggregate, "manifest aggregate field mismatch")
    return manifest


def _validate_per_query(
    rows: Sequence[Mapping[str, str]],
    loaded: LoadedRankings,
    computed: Mapping[str, tuple[dict[str, Any], dict[str, float], dict[str, float]]],
) -> dict[str, Mapping[str, str]]:
    _require(len(rows) == 322, "per-query receipt must contain 322 rows")
    by_query = {row["query_id"]: row for row in rows}
    _require(len(by_query) == 322 and set(by_query) == set(loaded.query_ids), "per-query identities mismatch")
    columns = {
        "control": ("control_ndcg_at_10", "control_reciprocal_rank"),
        "prior_only": ("prior_only_ndcg_at_10", "prior_only_reciprocal_rank"),
        "routing_only": ("routing_only_ndcg_at_10", "routing_only_reciprocal_rank"),
        "pointwise_only": ("pointwise_only_ndcg_at_10", "pointwise_only_reciprocal_rank"),
        "combined": ("combined_ndcg_at_10", "combined_reciprocal_rank"),
    }
    for query_id, row in by_query.items():
        positives = sum(grade > 0 for grade in loaded.qrels[query_id].values())
        _require(int(row["positive_candidate_count"]) == positives, f"{query_id} positive count mismatch")
        expected_status = "zero" if computed["control"][1][query_id] == 0 else "nonzero"
        _require(row["dense_ndcg_status"] == expected_status, f"{query_id} dense status mismatch")
        for system, (ndcg_column, rr_column) in columns.items():
            _close(computed[system][1][query_id], float(row[ndcg_column]), f"{query_id}/{system} nDCG")
            _close(computed[system][2][query_id], float(row[rr_column]), f"{query_id}/{system} MRR")
    return by_query


def _validate_comparisons(
    report: Mapping[str, Any],
    computed: Mapping[str, tuple[dict[str, Any], dict[str, float], dict[str, float]]],
    query_ids: Sequence[str],
) -> None:
    for candidate in ("pointwise_only", "combined"):
        receipt = report["comparisons_vs_control"][candidate]
        for metric_name, index in (("ndcg_at_10", 1), ("mrr", 2)):
            bootstrap = _paired_bootstrap(computed[candidate][index], computed["control"][index])
            recorded = receipt[f"{metric_name}_delta"]
            for key in ("lower", "mean_difference", "upper"):
                _close(bootstrap[key], recorded[key], f"{candidate}/{metric_name}/{key}", tolerance=2e-15)
            _require(
                recorded["generator"] == "python_random_mt19937"
                and recorded["n_resamples"] == 10_000
                and recorded["seed"] == 31,
                f"{candidate}/{metric_name} bootstrap metadata mismatch",
            )
            differences = [
                computed[candidate][index][query_id] - computed["control"][index][query_id]
                for query_id in query_ids
            ]
            expected_wtl = {
                "losses": sum(value < -1e-12 for value in differences),
                "ties": sum(abs(value) <= 1e-12 for value in differences),
                "tolerance": 1e-12,
                "wins": sum(value > 1e-12 for value in differences),
            }
            _require(expected_wtl == receipt[f"{metric_name}_win_tie_loss"], f"{candidate}/{metric_name} W/T/L mismatch")
        control_ndcg = computed["control"][1]
        candidate_ndcg = computed[candidate][1]
        transitions = {
            "dense_nonzero_candidate_zero": sum(control_ndcg[q] != 0 and candidate_ndcg[q] == 0 for q in query_ids),
            "dense_zero_candidate_nonzero": sum(control_ndcg[q] == 0 and candidate_ndcg[q] != 0 for q in query_ids),
            "nonzero_both": sum(control_ndcg[q] != 0 and candidate_ndcg[q] != 0 for q in query_ids),
            "query_count": len(query_ids),
            "zero_both": sum(control_ndcg[q] == 0 and candidate_ndcg[q] == 0 for q in query_ids),
        }
        _require(transitions == receipt["zero_transitions"], f"{candidate} zero transitions mismatch")


def _validate_query_slices(
    report: Mapping[str, Any],
    per_query: Mapping[str, Mapping[str, str]],
    loaded: LoadedRankings,
    computed: Mapping[str, tuple[dict[str, Any], dict[str, float], dict[str, float]]],
) -> None:
    positive_counts = {
        query_id: sum(grade > 0 for grade in grades.values())
        for query_id, grades in loaded.qrels.items()
    }
    memberships: dict[tuple[str, str], set[str]] = {
        ("qrel_positive_density", "le_5"): {q for q, n in positive_counts.items() if n <= 5},
        ("qrel_positive_density", "6_20"): {q for q, n in positive_counts.items() if 6 <= n <= 20},
        ("qrel_positive_density", "gt_20"): {q for q, n in positive_counts.items() if n > 20},
        ("dense_ndcg_status", "zero"): {q for q in loaded.query_ids if computed["control"][1][q] == 0},
        ("dense_ndcg_status", "nonzero"): {q for q in loaded.query_ids if computed["control"][1][q] != 0},
    }
    similarity = {q: float(row["query_similarity_mean"]) for q, row in per_query.items()}
    similarity_labels = _quartiles(similarity)
    for label in range(1, 5):
        memberships[("query_similarity_quartile", str(label))] = {
            q for q, quartile in similarity_labels.items() if quartile == label
        }
    for query_id, row in per_query.items():
        _require(int(row["query_similarity_quartile"]) == similarity_labels[query_id], f"{query_id} similarity quartile mismatch")
    _require(len(report["query_slices"]) == 9, "query slice count mismatch")
    for row in report["query_slices"]:
        key = (row["family"], str(row["label"]))
        expected_ids = memberships[key]
        _require(set(row["query_ids"]) == expected_ids, f"query slice {key} membership mismatch")
        _require(row["query_count"] == len(expected_ids), f"query slice {key} count mismatch")
        for system in SYSTEMS:
            summary = _evaluate(
                loaded.runs[system], loaded.qrels, query_ids=tuple(expected_ids)
            )[0]
            _check_summary(summary, row["metrics"][system], f"query slice {key}/{system}")


def _validate_candidate_slices(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    loaded: LoadedRankings,
) -> None:
    memberships: dict[tuple[str, str], set[str]] = {}
    families = (
        ("audited_count_quartile", "candidate_audited_counts"),
        ("smoothed_prior_quartile", "candidate_smoothed_priors"),
    )
    for family, feature_key in families:
        features = report["diagnostics"][feature_key]
        _require(set(features) == set(loaded.candidate_ids), f"{family} feature identities mismatch")
        labels = _quartiles({candidate_id: float(value) for candidate_id, value in features.items()})
        for label in range(1, 5):
            memberships[(family, str(label))] = {
                candidate_id
                for candidate_id, quartile in labels.items()
                if quartile == label
            }
    _require(len(rows) == 40, "candidate slice TSV must contain 40 rows")
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (row["slice_family"], row["slice_label"])
        system = row["system"]
        candidate_ids = set(json.loads(row["candidate_ids_json"]))
        _require(candidate_ids == memberships[key], f"candidate slice {key} membership mismatch")
        _require(int(row["candidate_count"]) == len(candidate_ids), f"candidate slice {key} count mismatch")
        expected = {
            "mrr": None if not row["mrr"] else float(row["mrr"]),
            "ndcg_at_10": None if not row["ndcg_at_10"] else float(row["ndcg_at_10"]),
            "query_count": int(row["query_count"]),
            "zero_ndcg_queries": int(row["zero_ndcg_queries"]),
        }
        summary = _evaluate(
            loaded.runs[system],
            loaded.qrels,
            candidate_ids=tuple(candidate_ids),
        )[0]
        _check_summary(summary, expected, f"candidate slice {key}/{system}")
        seen.add((*key, system))
    _require(len(seen) == 40, "candidate slice TSV identities are not unique")

    json_rows = {
        (row["family"], str(row["label"])): row
        for row in report["candidate_slices"]
    }
    _require(set(json_rows) == set(memberships), "candidate slice JSON identities mismatch")
    for key, row in json_rows.items():
        candidate_ids = memberships[key]
        _require(set(row["candidate_ids"]) == candidate_ids, f"candidate slice JSON {key} membership mismatch")
        _require(row["candidate_count"] == len(candidate_ids), f"candidate slice JSON {key} count mismatch")
        for system in SYSTEMS:
            summary = _evaluate(
                loaded.runs[system],
                loaded.qrels,
                candidate_ids=tuple(candidate_ids),
            )[0]
            _check_summary(summary, row["metrics"][system], f"candidate slice JSON {key}/{system}")


def _validate_dense_receipts(
    dense_results_path: Path,
    dense_per_query_path: Path,
    loaded: LoadedRankings,
    computed: Mapping[str, tuple[dict[str, Any], dict[str, float], dict[str, float]]],
) -> None:
    _require_hash(dense_results_path, EXPECTED_DENSE_RESULTS_SHA256, "dense result receipt")
    _require_hash(dense_per_query_path, EXPECTED_DENSE_PER_QUERY_SHA256, "dense per-query receipt")
    dense = json.loads(dense_results_path.read_text(encoding="utf-8"))
    _close(computed["control"][0]["ndcg_at_10"], dense["metrics"]["dense_nested"]["ndcg_at_10"], "dense aggregate nDCG")
    _close(computed["control"][0]["mrr"], dense["metrics"]["dense_nested"]["mrr"], "dense aggregate MRR")
    rows = _read_tsv(dense_per_query_path)
    by_query = {row["query_id"]: row for row in rows}
    _require(len(rows) == 322 and set(by_query) == set(loaded.query_ids), "dense per-query identities mismatch")
    for query_id, row in by_query.items():
        _close(computed["control"][1][query_id], float(row["dense_ndcg_at_10"]), f"dense {query_id} nDCG")
        _close(computed["control"][2][query_id], float(row["dense_reciprocal_rank"]), f"dense {query_id} MRR")


def _validate_selector_and_gate(
    report: Mapping[str, Any],
    selector_path: Path,
    loaded: LoadedRankings,
    computed: Mapping[str, tuple[dict[str, Any], dict[str, float], dict[str, float]]],
) -> None:
    selector = json.loads(selector_path.read_text(encoding="utf-8"))
    public = {key: value for key, value in selector.items() if key != "pairs"}
    _require(public == report["selector_receipt"], "selector summary mismatch")
    pairs = selector["pairs"]
    _require(len(pairs) == selector["selected_pair_count"] == 500, "selector pair count mismatch")
    source_ids = {pair["question_id"] for pair in pairs}
    _require(len(source_ids) == selector["source_question_count"] == 500, "selector source count mismatch")
    _require(source_ids.isdisjoint(loaded.query_ids), "selected source/development ID overlap")
    _require(
        len({(pair["question_id"], tuple(pair["candidate_ids"])) for pair in pairs}) == 500,
        "selector pairs are not unique",
    )
    uses = Counter(candidate_id for pair in pairs for candidate_id in pair["candidate_ids"])
    recorded_nonzero = {key: value for key, value in selector["candidate_use_counts"].items() if value}
    _require(dict(sorted(uses.items())) == recorded_nonzero, "selector candidate-use counts mismatch")
    _require(sum(selector["candidate_use_counts"].values()) == 1000, "selector total uses mismatch")
    _require(max(selector["candidate_use_counts"].values()) <= selector["final_candidate_use_cap"] == 20, "selector cap mismatch")
    for pair in pairs:
        _require(pair["candidate_ids"] == sorted(pair["candidate_ids"]) and len(pair["candidate_ids"]) == 2, "selector candidate IDs malformed")
        _require(len(pair["record_ids"]) == len(set(pair["record_ids"])) == 2, "selector record IDs malformed")
        _require(pair["candidate_use_cap"] == 20, "selector pair cap mismatch")

    comparison = report["comparisons_vs_control"]["combined"]
    checks = {
        "combined_mrr_delta_at_least_minus_0_005": comparison["mrr_delta"]["mean_difference"] >= -0.005,
        "combined_ndcg_interval_lower_above_zero": comparison["ndcg_at_10_delta"]["lower"] > 0,
        "combined_ndcg_point_positive": comparison["ndcg_at_10_delta"]["mean_difference"] > 0,
        "combined_zero_ndcg_below_116": computed["combined"][0]["zero_ndcg_queries"] < 116,
        "routing_ndcg_above_prior_ndcg": computed["routing_only"][0]["ndcg_at_10"] > computed["prior_only"][0]["ndcg_at_10"],
    }
    decision = report["decision"]
    _require(checks == decision["checks"], "decision checks mismatch")
    passed = all(checks.values())
    available = selector["selected_pair_count"] >= 500 and selector["source_question_count"] >= 100
    _require(decision["decision"] == ("pass" if passed else "fail"), "decision mismatch")
    _require(decision["selector_availability"]["available"] == available, "selector availability mismatch")
    expected_recommendation = "recommend_pairwise_pilot" if passed and available else "stop_pairwise_work"
    _require(decision["recommendation"] == expected_recommendation, "recommendation mismatch")
    values = decision["values"]
    expected_values = {
        "combined_mrr_delta": comparison["mrr_delta"]["mean_difference"],
        "combined_ndcg_delta": comparison["ndcg_at_10_delta"]["mean_difference"],
        "combined_ndcg_lower": comparison["ndcg_at_10_delta"]["lower"],
        "combined_zero_ndcg": computed["combined"][0]["zero_ndcg_queries"],
        "prior_ndcg": computed["prior_only"][0]["ndcg_at_10"],
        "routing_ndcg": computed["routing_only"][0]["ndcg_at_10"],
    }
    for key, expected in expected_values.items():
        if isinstance(expected, float):
            _close(expected, values[key], f"decision value {key}", tolerance=2e-15)
        else:
            _require(expected == values[key], f"decision value {key} mismatch")


def _independent_validation_block() -> dict[str, Any]:
    return {
        "assessment": "passed",
        "caveats": [
            "This is previously seen organizer development evidence and makes no hidden-test effectiveness claim.",
            "Pointwise support measures conditional response quality among audited source questions; it is not organizer-qrel gold and must not be interpreted as candidate coverage or routing relevance.",
            "Candidate exposure is uneven: audited-row counts and audited-question coverage vary by candidate, so smoothed priors and transfers have unequal evidence depth.",
            "Ranking construction was qrel-free. The frozen construction boundary required exact zero overlap between all 322 development query IDs and normalized texts and all 6,816 pointwise source-question IDs and normalized texts before qrels were opened.",
            "Phase 0 made no pairwise, judge, pointwise-regeneration, embedding-inference, or LLM API calls.",
            "The effectiveness gate is negative and the mandatory recommendation is stop_pairwise_work, even though the zero-call selector found 500 eligible pairs across 500 source questions.",
        ],
        "checks": {
            "aggregate_systems_recomputed": 5,
            "bootstrap_generator": "python_random_mt19937_randrange",
            "bootstrap_resamples": 10000,
            "bootstrap_seed": 31,
            "candidate_slice_rows_recomputed": 40,
            "candidate_slice_rule": "filter_both_qrels_and_rankings_then_recompute",
            "candidate_system_metric_cells_recomputed": 40,
            "dense_aggregate_receipt_within_5e_12": True,
            "dense_per_query_receipt_rows_within_5e_12": 322,
            "gate_and_stop_recomputed": True,
            "per_query_metric_cells_recomputed": 3220,
            "qrel_candidate_count": 150,
            "qrel_query_count": 322,
            "qrel_row_count": 48300,
            "query_slices_recomputed": 9,
            "run_count": 5,
            "run_rows_each": 48300,
            "selector_complete_receipt_reconciled": True,
            "selector_max_candidate_use": 20,
            "selector_selected_pairs": 500,
            "selector_source_questions": 500,
            "tracked_report_content_scan": "passed",
            "win_tie_loss_and_zero_transitions_recomputed": True,
        },
        "independent_method": "stdlib-only metric and receipt validator reading the development qrels and five emitted TREC runs; no project evaluator or metric function was imported",
        "provenance": {
            "artifact_aggregate_sha256": EXPECTED_AGGREGATE_SHA256,
            "artifact_manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "candidate_slices_evaluator_sha256": EXPECTED_CANDIDATE_SLICES_SHA256,
            "per_query_evaluator_sha256": EXPECTED_PER_QUERY_SHA256,
            "results_evaluator_sha256_before_validation": EXPECTED_EVALUATOR_RESULT_SHA256,
        },
        "status": "independently_validated",
    }


def validate_phase0_bundle(
    *,
    qrels_path: Path,
    artifact_dir: Path,
    dense_results_path: Path,
    dense_per_query_path: Path,
    tracked_per_query_path: Path,
    tracked_candidate_slices_path: Path,
) -> bytes:
    """Validate the frozen Task 4 contract and return finalized result bytes."""

    _require_hash(qrels_path, EXPECTED_QRELS_SHA256, "development qrels")
    _validate_manifest(artifact_dir)
    evaluator_path = artifact_dir / "results.json"
    per_query_path = artifact_dir / "per_query.tsv"
    candidate_slices_path = artifact_dir / "candidate_slices.tsv"
    _require_hash(evaluator_path, EXPECTED_EVALUATOR_RESULT_SHA256, "evaluator result")
    _require_hash(per_query_path, EXPECTED_PER_QUERY_SHA256, "evaluator per-query receipt")
    _require_hash(candidate_slices_path, EXPECTED_CANDIDATE_SLICES_SHA256, "evaluator candidate-slice receipt")
    _require_hash(
        tracked_per_query_path,
        EXPECTED_PER_QUERY_SHA256,
        "tracked per-query receipt",
    )
    _require_hash(
        tracked_candidate_slices_path,
        EXPECTED_CANDIDATE_SLICES_SHA256,
        "tracked candidate-slice receipt",
    )

    evaluator_bytes = evaluator_path.read_bytes()
    per_query_bytes = per_query_path.read_bytes()
    candidate_slices_bytes = candidate_slices_path.read_bytes()
    _require(
        tracked_per_query_path.read_bytes() == per_query_bytes,
        "tracked per-query receipt differs from evaluator output",
    )
    _require(
        tracked_candidate_slices_path.read_bytes() == candidate_slices_bytes,
        "tracked candidate-slice receipt differs from evaluator output",
    )
    validate_publication_content((evaluator_bytes, per_query_bytes, candidate_slices_bytes))
    report = json.loads(evaluator_bytes)
    _require(report.get("independent_validation") is False, "evaluator independent_validation must be false")

    run_paths = {system: artifact_dir / "runs" / f"{system}.trec" for system in SYSTEMS}
    loaded = load_qrels_and_runs(
        qrels_path,
        run_paths,
        expected_query_count=322,
        expected_candidate_count=150,
    )
    computed = {
        system: _evaluate(loaded.runs[system], loaded.qrels)
        for system in SYSTEMS
    }
    for system in SYSTEMS:
        _check_summary(computed[system][0], report["systems"][system], f"aggregate {system}")

    per_query_rows = _read_tsv(per_query_path)
    per_query = _validate_per_query(per_query_rows, loaded, computed)
    _validate_comparisons(report, computed, loaded.query_ids)
    _validate_query_slices(report, per_query, loaded, computed)
    _validate_candidate_slices(report, _read_tsv(candidate_slices_path), loaded)
    _validate_dense_receipts(
        dense_results_path,
        dense_per_query_path,
        loaded,
        computed,
    )
    _validate_selector_and_gate(
        report,
        artifact_dir / "selector_receipt.json",
        loaded,
        computed,
    )

    finalized = finalize_evaluator_payload(evaluator_bytes, _independent_validation_block())
    validate_publication_content((finalized, per_query_bytes, candidate_slices_bytes))
    return finalized
