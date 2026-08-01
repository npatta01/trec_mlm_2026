from __future__ import annotations

from collections import defaultdict
import hashlib
from itertools import combinations
import json
from pathlib import Path
import random

import numpy as np
import pytest

from trec_mlm.pointwise_ranking import (
    SELECTOR_VERSION,
    construct_pointwise_rankings,
    load_pointwise_evidence,
    select_pairwise_candidates,
    stable_quartile_labels,
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    record_id: str,
    question_id: str,
    candidate_id: str,
    *,
    question: str = "synthetic question",
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "source_query_id": question_id,
        "candidate_id": candidate_id,
        "question": question,
        "response": "synthetic response",
        "formatted_sha256": "a" * 64,
        "raw_membership_sha256": "b" * 64,
    }


def _label(
    record_id: str,
    question_id: str,
    candidate_id: str,
    *,
    verdict: str = "correct",
    supported: bool = True,
    source_ordinal: int = 0,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "question_id": question_id,
        "candidate_id": candidate_id,
        "source_ordinal": source_ordinal,
        "verdict": verdict,
        "supported": supported,
    }


def _load_fixture(labels: Path, records: Path):
    return load_pointwise_evidence(
        labels,
        records,
        expected_labels_sha256=_sha256(labels),
        expected_records_sha256=_sha256(records),
        expected_record_count=1,
        expected_question_count=1,
        expected_candidate_count=1,
    )


def _load_rows_fixture(
    tmp_path: Path, rows: list[dict[str, object]], records: list[dict[str, object]]
):
    labels_path = tmp_path / "labels.jsonl"
    records_path = tmp_path / "records.json"
    _write_jsonl(labels_path, rows)
    _write_json(records_path, records)
    return load_pointwise_evidence(
        labels_path,
        records_path,
        expected_labels_sha256=_sha256(labels_path),
        expected_records_sha256=_sha256(records_path),
        expected_record_count=len(rows),
        expected_question_count=len({str(row["question_id"]) for row in rows}),
        expected_candidate_count=len({str(row["candidate_id"]) for row in rows}),
    )


def _ranking_fixture(
    tmp_path: Path,
    *,
    development_query_id: str = "dev-q",
    development_query_text: str = "development query",
    overlapping_source_text: str | None = None,
):
    labels: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    selected: list[list[str]] = []
    for candidate_id, supported_count, selected_supported in (
        ("A", 30, 2),
        ("B", 10, 8),
    ):
        candidate_selected: list[str] = []
        support_flags = [True] * selected_supported + [False] * (10 - selected_supported)
        support_flags += [True] * (supported_count - selected_supported)
        support_flags += [False] * (40 - len(support_flags))
        for index, supported in enumerate(support_flags):
            record_id = f"{candidate_id}-{index:02d}"
            question_id = f"source-{candidate_id}-{index:02d}"
            verdict = "correct" if supported else "incorrect"
            labels.append(
                _label(
                    record_id,
                    question_id,
                    candidate_id,
                    verdict=verdict,
                    supported=supported,
                    source_ordinal=len(labels),
                )
            )
            records.append(
                _record(
                    record_id,
                    question_id,
                    candidate_id,
                    question=(
                        overlapping_source_text
                        if record_id == "A-00" and overlapping_source_text is not None
                        else f"question {question_id}"
                    ),
                )
            )
            if index < 10:
                candidate_selected.append(record_id)
        selected.append(candidate_selected)
    evidence = _load_rows_fixture(tmp_path, labels, records)
    root = tmp_path / "refinement"
    root.mkdir()
    development_queries = tmp_path / "development-queries.tsv"
    development_queries.write_text(
        f"{development_query_id}\t{development_query_text}\n", encoding="utf-8"
    )
    construction = tmp_path / "construction"
    construction.mkdir()
    _write_json(
        construction / "config.json",
        {"query_path": str(development_queries)},
    )
    _write_json(root / "config.json", {"construction_path": str(construction)})
    _write_json(
        root / "inputs.json",
        {"query_ids": [development_query_id], "candidate_ids": ["A", "B"]},
    )
    _write_json(root / "selected_record_ids.json", [[selected[0], selected[1]]])
    np.save(
        root / "top_record_indices.npy",
        np.asarray([[[*range(10)], [*range(40, 50)]]], dtype=np.int32),
        allow_pickle=False,
    )
    similarities = np.asarray(
        [[[0.9] * 10, [0.3] * 5 + [0.5] * 5]], dtype=np.float64
    )
    np.save(root / "top_record_similarities.npy", similarities, allow_pickle=False)
    return evidence, root, development_queries


def _construct_fixture(evidence, root: Path, development_queries: Path):
    return construct_pointwise_rankings(
        evidence,
        root,
        development_queries_path=development_queries,
        expected_development_queries_sha256=_sha256(development_queries),
        expected_refinement_aggregate_sha256=None,
        expected_selected_record_ids_sha256=_sha256(root / "selected_record_ids.json"),
        expected_top_record_indices_sha256=_sha256(root / "top_record_indices.npy"),
        expected_top_record_similarities_sha256=_sha256(
            root / "top_record_similarities.npy"
        ),
    )


def test_load_rejects_frozen_hash_mismatch_before_parsing(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    records = tmp_path / "records.json"
    _write_jsonl(labels, [_label("r1", "q1", "c1")])
    _write_json(records, [_record("r1", "q1", "c1")])

    with pytest.raises(ValueError, match="label.*SHA-256"):
        load_pointwise_evidence(
            labels,
            records,
            expected_labels_sha256="0" * 64,
            expected_records_sha256=_sha256(records),
            expected_record_count=1,
            expected_question_count=1,
            expected_candidate_count=1,
        )


def test_load_joins_by_record_id_and_rejects_identity_mismatch(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    records = tmp_path / "records.json"
    _write_jsonl(labels, [_label("r1", "q1", "wrong-candidate")])
    _write_json(records, [_record("r1", "q1", "c1")])

    with pytest.raises(ValueError, match="record-ID identity mismatch"):
        _load_fixture(labels, records)


def test_load_joins_shuffled_rows_by_record_id_not_source_ordinal(tmp_path: Path) -> None:
    labels = [
        _label("r2", "q2", "c1", source_ordinal=91),
        _label("r1", "q1", "c1", source_ordinal=17),
    ]
    records = [_record("r1", "q1", "c1"), _record("r2", "q2", "c1")]

    evidence = _load_rows_fixture(tmp_path, labels, records)

    assert evidence.record_ids == ("r1", "r2")
    assert evidence.labels_by_record_id["r1"].source_ordinal == 17


def test_load_rejects_missing_source_question_text(tmp_path: Path) -> None:
    label = _label("r1", "q1", "c1")
    record = _record("r1", "q1", "c1")
    del record["question"]

    with pytest.raises(ValueError, match="source question text"):
        _load_rows_fixture(tmp_path, [label], [record])


def test_load_rejects_inconsistent_text_for_one_source_question(tmp_path: Path) -> None:
    labels = [
        _label("r1", "q1", "c1", source_ordinal=0),
        _label("r2", "q1", "c2", source_ordinal=1),
    ]
    records = [
        _record("r1", "q1", "c1", question="first text"),
        _record("r2", "q1", "c2", question="different text"),
    ]

    with pytest.raises(ValueError, match="source question text.*inconsistent"):
        _load_rows_fixture(tmp_path, labels, records)


@pytest.mark.parametrize(
    ("verdict", "supported"),
    [("correct", False), ("partial", False), ("incorrect", True)],
)
def test_load_rejects_supported_verdict_inconsistency(
    tmp_path: Path, verdict: str, supported: bool
) -> None:
    labels = tmp_path / "labels.jsonl"
    records = tmp_path / "records.json"
    _write_jsonl(
        labels,
        [_label("r1", "q1", "c1", verdict=verdict, supported=supported)],
    )
    _write_json(records, [_record("r1", "q1", "c1")])

    with pytest.raises(ValueError, match="supported.*verdict"):
        _load_fixture(labels, records)


def test_construct_uses_beta_prior_top_ten_mean_residual_and_frozen_rrf(
    tmp_path: Path,
) -> None:
    evidence, refinement, queries = _ranking_fixture(tmp_path)

    result = _construct_fixture(evidence, refinement, queries)

    np.testing.assert_allclose(result.candidate_priors, [31 / 42, 11 / 42])
    np.testing.assert_allclose(result.local_scores, [[0.2, 0.8]])
    np.testing.assert_allclose(
        result.residual_scores, [[0.2 - 31 / 42, 0.8 - 11 / 42]]
    )
    assert result.prior_rankings == {"dev-q": ["A", "B"]}
    assert result.routing_rankings == {"dev-q": ["B", "A"]}
    assert result.pointwise_rankings == {"dev-q": ["A", "B"]}
    np.testing.assert_allclose(
        result.pointwise_rrf_scores,
        [[1 / 61 + 1 / 62, 1 / 62 + 1 / 61]],
    )
    assert result.query_similarity_means == {"dev-q": pytest.approx(0.65)}


def test_construct_rejects_row_index_identity_disagreement(tmp_path: Path) -> None:
    evidence, refinement, queries = _ranking_fixture(tmp_path)
    indices = np.load(refinement / "top_record_indices.npy", allow_pickle=False)
    indices[0, 0, 0] = 1
    np.save(refinement / "top_record_indices.npy", indices, allow_pickle=False)

    with pytest.raises(ValueError, match="selected record ID.*top-record index"):
        _construct_fixture(evidence, refinement, queries)


def test_construct_rejects_development_source_question_id_overlap(
    tmp_path: Path,
) -> None:
    evidence, refinement, queries = _ranking_fixture(
        tmp_path,
        development_query_id="source-A-00",
    )

    with pytest.raises(ValueError, match="development/source question ID overlap"):
        _construct_fixture(evidence, refinement, queries)


def test_construct_rejects_nfkc_casefold_unicode_whitespace_text_overlap(
    tmp_path: Path,
) -> None:
    evidence, refinement, queries = _ranking_fixture(
        tmp_path,
        development_query_text="  alpha\t BETA  ",
        overlapping_source_text="ＡLPHA\u2003beta",
    )

    with pytest.raises(ValueError, match="normalized development/source text overlap"):
        _construct_fixture(evidence, refinement, queries)


def test_stable_quartiles_use_value_then_exact_identifier_positions() -> None:
    assert stable_quartile_labels({"b": 0.0, "a": 0.0, "d": 2.0, "c": 1.0, "e": 3.0}) == {
        "a": 1,
        "b": 1,
        "c": 2,
        "d": 3,
        "e": 4,
    }
    assert stable_quartile_labels({"a": 0.0, "b": 1.0}) == {"a": 1, "b": 3}
    assert stable_quartile_labels({}) == {}


def test_selector_uses_canonical_hash_then_dynamic_candidate_use(tmp_path: Path) -> None:
    rows = [
        _label(
            f"r-{candidate_id}",
            "q1",
            candidate_id,
            verdict="partial",
            source_ordinal=index,
        )
        for index, candidate_id in enumerate(("A", "B", "C", "D"))
    ]
    records = [
        _record(str(row["record_id"]), "q1", str(row["candidate_id"]))
        for row in rows
    ]
    evidence = _load_rows_fixture(tmp_path, rows, records)
    expected_first = min(
        ((left, right) for left in "ABCD" for right in "ABCD" if left < right),
        key=lambda pair: hashlib.sha256(
            json.dumps(
                {
                    "candidate_ids": list(pair),
                    "question_id": "q1",
                    "selector_version": SELECTOR_VERSION,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    )

    receipt = select_pairwise_candidates(evidence, target_pair_count=2)

    assert receipt.pairs[0].candidate_ids == expected_first
    assert set(receipt.pairs[0].candidate_ids).isdisjoint(receipt.pairs[1].candidate_ids)
    assert receipt.selected_pair_count == 2
    assert receipt.source_question_count == 1


def test_selector_first_pass_diversifies_questions_and_escalates_cap(
    tmp_path: Path,
) -> None:
    rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    for question_index in range(21):
        question_id = f"q-{question_index:02d}"
        for candidate_id in ("A", "B"):
            record_id = f"{question_id}-{candidate_id}"
            rows.append(
                _label(
                    record_id,
                    question_id,
                    candidate_id,
                    verdict="partial",
                    source_ordinal=len(rows),
                )
            )
            records.append(_record(record_id, question_id, candidate_id))
    evidence = _load_rows_fixture(tmp_path, rows, records)

    receipt = select_pairwise_candidates(evidence, target_pair_count=21)

    assert receipt.selected_pair_count == 21
    assert receipt.source_question_count == 21
    assert receipt.final_candidate_use_cap == 21
    assert receipt.candidate_use_counts == {"A": 21, "B": 21}


def test_selector_prioritizes_partial_tier_and_one_pair_per_question(
    tmp_path: Path,
) -> None:
    specifications = [
        ("q-partial-many", "A", "partial"),
        ("q-partial-many", "B", "partial"),
        ("q-partial-many", "C", "partial"),
        ("q-partial-one", "A", "partial"),
        ("q-partial-one", "B", "partial"),
        ("q-correct", "C", "correct"),
        ("q-correct", "D", "correct"),
    ]
    rows = [
        _label(
            f"r-{question_id}-{candidate_id}",
            question_id,
            candidate_id,
            verdict=verdict,
            source_ordinal=index,
        )
        for index, (question_id, candidate_id, verdict) in enumerate(specifications)
    ]
    records = [
        _record(
            str(row["record_id"]),
            str(row["question_id"]),
            str(row["candidate_id"]),
        )
        for row in rows
    ]
    evidence = _load_rows_fixture(tmp_path, rows, records)

    receipt = select_pairwise_candidates(evidence, target_pair_count=2)

    assert {pair.question_id for pair in receipt.pairs} == {
        "q-partial-many",
        "q-partial-one",
    }


def _naive_selector(evidence, target_pair_count: int):
    grouped = defaultdict(dict)
    for label in evidence.labels_by_record_id.values():
        if label.supported:
            grouped[(label.question_id, label.verdict)][label.candidate_id] = label
    eligible = []
    for (question_id, verdict), labels in grouped.items():
        for left_id, right_id in combinations(sorted(labels), 2):
            payload = {
                "candidate_ids": [left_id, right_id],
                "question_id": question_id,
                "selector_version": SELECTOR_VERSION,
            }
            selector_hash = hashlib.sha256(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            eligible.append(
                (
                    question_id,
                    (left_id, right_id),
                    (labels[left_id].record_id, labels[right_id].record_id),
                    0 if verdict == "partial" else 1,
                    abs(
                        evidence.candidate_priors[left_id]
                        - evidence.candidate_priors[right_id]
                    ),
                    selector_hash,
                )
            )
    selected_indices: set[int] = set()
    selected = []
    uses = {candidate_id: 0 for candidate_id in evidence.candidate_ids}
    cap = 20

    def priority(index: int):
        pair = eligible[index]
        left_id, right_id = pair[1]
        left_use, right_use = uses[left_id], uses[right_id]
        return (
            pair[3],
            pair[4],
            (
                max(left_use, right_use),
                left_use + right_use,
                left_use,
                right_use,
            ),
            pair[5],
        )

    def choose(excluded_questions=None):
        choices = [
            index
            for index, pair in enumerate(eligible)
            if index not in selected_indices
            and (excluded_questions is None or pair[0] not in excluded_questions)
            and all(uses[candidate_id] < cap for candidate_id in pair[1])
        ]
        return min(choices, key=priority) if choices else None

    while len(selected) < target_pair_count:
        pass_questions = set()
        while len(selected) < target_pair_count:
            chosen = choose(pass_questions)
            if chosen is None:
                break
            pair = eligible[chosen]
            selected_indices.add(chosen)
            pass_questions.add(pair[0])
            for candidate_id in pair[1]:
                uses[candidate_id] += 1
            selected.append((*pair[:3], cap))
        while len(selected) < target_pair_count:
            chosen = choose()
            if chosen is None:
                break
            pair = eligible[chosen]
            selected_indices.add(chosen)
            for candidate_id in pair[1]:
                uses[candidate_id] += 1
            selected.append((*pair[:3], cap))
        if len(selected) >= target_pair_count or len(selected_indices) == len(eligible):
            break
        cap += 1
    return selected, uses, cap, len(eligible)


def test_optimized_selector_matches_naive_full_dynamic_key_randomized(
    tmp_path: Path,
) -> None:
    rng = random.Random(31)
    candidates = tuple("ABCDEF")
    for inventory_index in range(50):
        rows: list[dict[str, object]] = []
        records: list[dict[str, object]] = []
        for question_index in range(rng.randint(3, 9)):
            question_id = f"q-{question_index}"
            chosen_candidates = rng.sample(
                candidates, rng.randint(2, len(candidates))
            )
            for candidate_id in chosen_candidates:
                verdict = rng.choice(("partial", "correct", "incorrect"))
                record_id = f"{question_id}-{candidate_id}"
                rows.append(
                    _label(
                        record_id,
                        question_id,
                        candidate_id,
                        verdict=verdict,
                        supported=verdict != "incorrect",
                        source_ordinal=len(rows),
                    )
                )
                records.append(_record(record_id, question_id, candidate_id))
        for candidate_id in candidates:
            question_id = f"audit-{candidate_id}"
            record_id = f"audit-record-{candidate_id}"
            supported = rng.choice((True, False))
            rows.append(
                _label(
                    record_id,
                    question_id,
                    candidate_id,
                    verdict="correct" if supported else "incorrect",
                    supported=supported,
                    source_ordinal=len(rows),
                )
            )
            records.append(_record(record_id, question_id, candidate_id))
        inventory_path = tmp_path / f"inventory-{inventory_index}"
        inventory_path.mkdir()
        evidence = _load_rows_fixture(inventory_path, rows, records)
        target = rng.randint(1, 15)

        expected, expected_uses, expected_cap, eligible_count = _naive_selector(
            evidence, target
        )
        observed = select_pairwise_candidates(evidence, target_pair_count=target)

        assert [
            (pair.question_id, pair.candidate_ids, pair.record_ids, pair.candidate_use_cap)
            for pair in observed.pairs
        ] == expected
        assert observed.candidate_use_counts == expected_uses
        assert observed.final_candidate_use_cap == expected_cap
        assert observed.eligible_pair_count == eligible_count
