"""Deterministic blinding and strict review validation for reference answers."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
from itertools import combinations
from typing import Any, Iterable, Mapping, Sequence


SYSTEMS = ("perplexity", "gemini", "qwen27", "gemma31")
LABELS = ("response_a", "response_b", "response_c", "response_d")
PAIRWISE_OUTCOMES = {
    "left_better",
    "right_better",
    "tie",
    "both_unacceptable",
    "uncertain_needs_evidence",
}
SEVERITIES = {"none", "minor", "major", "fatal"}
CONFIDENCE = {"high", "medium", "low"}


def blind_system_order(
    question_id: str, systems: Sequence[str], *, seed: int
) -> tuple[str, ...]:
    """Return a stable per-question system permutation."""

    if len(systems) != len(set(systems)):
        raise ValueError("systems must be unique")
    return tuple(
        sorted(
            systems,
            key=lambda system: hashlib.sha256(
                f"{seed}\0{question_id}\0{system}".encode()
            ).digest(),
        )
    )


def build_blind_packet(
    sample: Mapping[str, Any],
    answers_by_system: Mapping[str, Mapping[str, str | None]],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build judge-visible packets and the separately retained system map."""

    questions = sample.get("questions")
    if not isinstance(questions, list):
        raise ValueError("sample questions must be a list")
    if set(answers_by_system) != set(SYSTEMS):
        raise ValueError("answer sources must match the frozen four systems")
    packet: list[dict[str, Any]] = []
    mappings: dict[str, dict[str, str]] = {}
    seen_question_ids: set[str] = set()
    for row in questions:
        if not isinstance(row, Mapping):
            raise ValueError("sample question must be an object")
        question_id = row.get("question_id")
        question = row.get("question")
        if not isinstance(question_id, str) or not isinstance(question, str):
            raise ValueError("sample question requires string ID and text")
        if question_id in seen_question_ids:
            raise ValueError(f"duplicate sample question ID: {question_id}")
        seen_question_ids.add(question_id)
        order = blind_system_order(question_id, SYSTEMS, seed=seed)
        label_map = dict(zip(LABELS, order, strict=True))
        mappings[question_id] = label_map
        blinded_answers: dict[str, str | None] = {}
        generation_failures: list[str] = []
        for label, system in label_map.items():
            answer = answers_by_system[system].get(question_id)
            if answer is None:
                blinded_answers[label] = None
                generation_failures.append(label)
            elif not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"missing answer for {system}/{question_id}")
            else:
                blinded_answers[label] = answer
        packet.append(
            {
                "question_id": question_id,
                "question": question,
                "cluster_id": row.get("cluster_id"),
                "strata": list(row.get("strata") or []),
                "challenge_indicators": dict(
                    row.get("challenge_indicators") or {}
                ),
                "answers": blinded_answers,
                "generation_failures": generation_failures,
            }
        )
    return packet, {"seed": seed, "systems": list(SYSTEMS), "maps": mappings}


def validate_sealed_map(
    sealed: Mapping[str, Any], expected_ids: Iterable[str]
) -> dict[str, dict[str, str]]:
    """Verify sealed metadata and every deterministic question mapping."""

    seed = sealed.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("sealed map requires an integer seed")
    if sealed.get("systems") != list(SYSTEMS):
        raise ValueError("sealed map systems do not match the frozen systems")
    expected = list(expected_ids)
    if len(expected) != len(set(expected)):
        raise ValueError("sealed map expected IDs must be unique")
    maps = sealed.get("maps")
    if not isinstance(maps, Mapping) or set(maps) != set(expected):
        raise ValueError("sealed map does not exactly match expected question IDs")
    validated: dict[str, dict[str, str]] = {}
    for question_id in expected:
        mapping = maps.get(question_id)
        if not isinstance(mapping, Mapping):
            raise ValueError(f"sealed map entry must be an object: {question_id}")
        expected_mapping = dict(
            zip(
                LABELS,
                blind_system_order(question_id, SYSTEMS, seed=seed),
                strict=True,
            )
        )
        if dict(mapping) != expected_mapping:
            raise ValueError(
                f"sealed map does not match deterministic blind order: {question_id}"
            )
        validated[question_id] = expected_mapping
    return validated


def _integer_score(value: object, *, minimum: int, maximum: int, field: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or not minimum <= value <= maximum:
        raise ValueError(f"{field} must be an integer from {minimum} to {maximum}")


def validate_review(review: Mapping[str, Any]) -> None:
    """Fail closed on incomplete or structurally invalid blind judgments."""

    question_id = review.get("question_id")
    if not isinstance(question_id, str) or not question_id:
        raise ValueError("review requires a question_id")
    pointwise = review.get("pointwise")
    if not isinstance(pointwise, Mapping) or set(pointwise) != set(LABELS):
        raise ValueError("pointwise scores must cover all four response labels")
    for label, score in pointwise.items():
        if not isinstance(score, Mapping):
            raise ValueError(f"pointwise {label} must be an object")
        for field, maximum in (
            ("relevance", 2),
            ("correctness", 4),
            ("completeness", 2),
            ("uncertainty", 1),
            ("usefulness", 1),
        ):
            _integer_score(score.get(field), minimum=0, maximum=maximum, field=field)
        if score.get("severity") not in SEVERITIES:
            raise ValueError(f"invalid severity for {label}")
        flags = score.get("flags")
        if not isinstance(flags, list) or not all(isinstance(flag, str) for flag in flags):
            raise ValueError(f"flags for {label} must be strings")

    pairwise = review.get("pairwise")
    if not isinstance(pairwise, list) or len(pairwise) != 6:
        raise ValueError("review must contain all six pairs")
    expected_pairs = {frozenset(pair) for pair in combinations(LABELS, 2)}
    observed_pairs: list[frozenset[str]] = []
    for result in pairwise:
        if not isinstance(result, Mapping):
            raise ValueError("pairwise results must be objects")
        left, right = result.get("left"), result.get("right")
        if left not in LABELS or right not in LABELS or left == right:
            raise ValueError("pairwise labels must be distinct known responses")
        if result.get("outcome") not in PAIRWISE_OUTCOMES:
            raise ValueError("invalid pairwise outcome")
        observed_pairs.append(frozenset((left, right)))
    if set(observed_pairs) != expected_pairs or len(set(observed_pairs)) != 6:
        raise ValueError("review must contain six pairs exactly once")

    tiers = review.get("tiers")
    if not isinstance(tiers, list) or not tiers or not all(isinstance(tier, list) for tier in tiers):
        raise ValueError("tiers must be a nonempty list of lists")
    tier_labels = [label for tier in tiers for label in tier]
    if len(tier_labels) != 4 or set(tier_labels) != set(LABELS):
        raise ValueError("tiers must contain every response exactly once")
    tier_by_label = {
        label: tier_number
        for tier_number, tier in enumerate(tiers)
        for label in tier
    }
    for result in pairwise:
        left = str(result["left"])
        right = str(result["right"])
        outcome = result["outcome"]
        contradicts = (
            outcome == "left_better" and tier_by_label[left] >= tier_by_label[right]
        ) or (
            outcome == "right_better" and tier_by_label[right] >= tier_by_label[left]
        ) or (outcome == "tie" and tier_by_label[left] != tier_by_label[right])
        if contradicts:
            raise ValueError(
                f"pairwise result contradicts tiers for {left}/{right}"
            )
    if review.get("confidence") not in CONFIDENCE:
        raise ValueError("invalid confidence")
    rationale = review.get("rationale")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("review requires a rationale")
    if not isinstance(review.get("evidence_checks"), list):
        raise ValueError("evidence_checks must be a list")
    if review.get("review_status") != "locked_blind":
        raise ValueError("review must be locked blind")


def apply_review_corrections(
    reviews: Iterable[Mapping[str, Any]],
    corrections: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Apply explicit drift-checked corrections before strict validation."""

    records = deepcopy([dict(review) for review in reviews])
    by_id: dict[str, dict[str, Any]] = {}
    for review in records:
        question_id = str(review.get("question_id"))
        if question_id in by_id:
            raise ValueError(f"duplicate review ID before correction: {question_id}")
        by_id[question_id] = review
    corrected_pairs: set[tuple[str, str, str]] = set()
    for correction in corrections:
        question_id = correction.get("question_id")
        left = correction.get("left")
        right = correction.get("right")
        original = correction.get("from_outcome")
        replacement = correction.get("to_outcome")
        reason = correction.get("reason")
        if (
            not isinstance(question_id, str)
            or left not in LABELS
            or right not in LABELS
            or left == right
            or original not in PAIRWISE_OUTCOMES
            or replacement not in PAIRWISE_OUTCOMES
            or not isinstance(reason, str)
            or not reason.strip()
        ):
            raise ValueError("invalid review correction")
        key = (question_id, str(left), str(right))
        if key in corrected_pairs:
            raise ValueError(f"duplicate review correction: {key!r}")
        corrected_pairs.add(key)
        review = by_id.get(question_id)
        if review is None:
            raise ValueError(f"correction review ID not found: {question_id}")
        matches = [
            result
            for result in review.get("pairwise") or []
            if result.get("left") == left and result.get("right") == right
        ]
        if len(matches) != 1:
            raise ValueError(f"correction pair not found exactly once: {key!r}")
        if matches[0].get("outcome") != original:
            raise ValueError(f"correction source outcome drifted: {key!r}")
        matches[0]["outcome"] = replacement
    for review in records:
        validate_review(review)
    return records


def validate_review_coverage(
    expected_ids: Iterable[str], reviews: Iterable[Mapping[str, Any]]
) -> None:
    """Require each expected question exactly once and validate every review."""

    expected = list(expected_ids)
    duplicate_expected = sorted(
        question_id
        for question_id, count in Counter(expected).items()
        if count > 1
    )
    if duplicate_expected:
        raise ValueError(f"duplicate expected review IDs: {duplicate_expected!r}")
    rows = list(reviews)
    observed = [str(row.get("question_id")) for row in rows]
    duplicates = sorted(
        question_id for question_id, count in Counter(observed).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate review IDs: {duplicates!r}")
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    if missing or extra:
        raise ValueError(f"missing={missing!r} extra={extra!r}")
    for review in rows:
        validate_review(review)


def merge_reviews(
    existing: Iterable[Mapping[str, Any]],
    additions: Iterable[Mapping[str, Any]],
    *,
    replace: bool = False,
) -> list[dict[str, Any]]:
    """Validate and atomically-serializable merge review decisions by ID."""

    merged: dict[str, dict[str, Any]] = {}
    for review in existing:
        validate_review(review)
        question_id = str(review["question_id"])
        if question_id in merged:
            raise ValueError(f"duplicate existing review: {question_id}")
        merged[question_id] = dict(review)
    seen_additions: set[str] = set()
    for review in additions:
        validate_review(review)
        question_id = str(review["question_id"])
        if question_id in seen_additions:
            raise ValueError(f"duplicate added review: {question_id}")
        seen_additions.add(question_id)
        if question_id in merged and not replace:
            raise ValueError(f"already reviewed: {question_id}")
        merged[question_id] = dict(review)
    return [merged[question_id] for question_id in sorted(merged)]


def unblind_review(
    review: Mapping[str, Any], system_map: Mapping[str, str]
) -> dict[str, Any]:
    """Derive a provider-named ranking only after the blind verdict is locked."""

    validate_review(review)
    if set(system_map) != set(LABELS) or set(system_map.values()) != set(SYSTEMS):
        raise ValueError("system map must bijectively cover all labels and systems")
    pointwise = {
        system_map[label]: dict(score)
        for label, score in review["pointwise"].items()
    }
    pairwise = [
        {
            **dict(result),
            "left": system_map[str(result["left"])],
            "right": system_map[str(result["right"])],
        }
        for result in review["pairwise"]
    ]
    return {
        "question_id": review["question_id"],
        "ranking": [
            [system_map[str(label)] for label in tier] for tier in review["tiers"]
        ],
        "pointwise": pointwise,
        "pairwise": pairwise,
        "confidence": review["confidence"],
        "rationale": review["rationale"],
        "evidence_checks": list(review["evidence_checks"]),
    }


def aggregate_rankings(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize locked, unblinded rankings without forcing tied winners."""

    records = list(rows)
    if not records:
        raise ValueError("at least one ranking is required")
    metrics: dict[str, dict[str, Any]] = {
        system: {
            "top_tier": 0,
            "fractional_top": 0.0,
            "tier_sum": 0.0,
            "serious_errors": 0,
            "correctness_sum": 0.0,
            "pairwise": Counter(),
        }
        for system in SYSTEMS
    }
    for row in records:
        ranking = row.get("ranking")
        pointwise = row.get("pointwise")
        if not isinstance(ranking, list) or not ranking or not isinstance(pointwise, Mapping):
            raise ValueError("unblinded ranking row is incomplete")
        flat = [system for tier in ranking for system in tier]
        if len(flat) != 4 or set(flat) != set(SYSTEMS):
            raise ValueError("ranking must cover every system exactly once")
        top = ranking[0]
        for system in top:
            metrics[system]["top_tier"] += 1
            metrics[system]["fractional_top"] += 1.0 / len(top)
        for tier_number, tier in enumerate(ranking, start=1):
            for system in tier:
                metrics[system]["tier_sum"] += tier_number
        for system in SYSTEMS:
            score = pointwise.get(system)
            if not isinstance(score, Mapping):
                raise ValueError(f"missing pointwise score for {system}")
            metrics[system]["correctness_sum"] += float(score["correctness"])
            if score.get("severity") in {"major", "fatal"}:
                metrics[system]["serious_errors"] += 1
        for result in row.get("pairwise") or []:
            left, right, outcome = (
                str(result["left"]),
                str(result["right"]),
                str(result["outcome"]),
            )
            if outcome == "left_better":
                metrics[left]["pairwise"]["wins"] += 1
                metrics[right]["pairwise"]["losses"] += 1
            elif outcome == "right_better":
                metrics[right]["pairwise"]["wins"] += 1
                metrics[left]["pairwise"]["losses"] += 1
            elif outcome == "tie":
                metrics[left]["pairwise"]["ties"] += 1
                metrics[right]["pairwise"]["ties"] += 1
            else:
                metrics[left]["pairwise"]["unresolved"] += 1
                metrics[right]["pairwise"]["unresolved"] += 1

    count = len(records)
    systems: dict[str, dict[str, Any]] = {}
    for system, values in metrics.items():
        systems[system] = {
            "top_tier_share": values["top_tier"] / count,
            "fractional_top_credit": values["fractional_top"] / count,
            "mean_tier": values["tier_sum"] / count,
            "serious_error_rate": values["serious_errors"] / count,
            "mean_correctness": values["correctness_sum"] / count,
            "pairwise": dict(values["pairwise"]),
        }
    return {"questions": count, "systems": systems}
