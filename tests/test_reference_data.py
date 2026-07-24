from pathlib import Path
import json

import pytest

from trec_mlm.reference_data import (
    EXPECTED_PERPLEXITY_SHA256,
    load_perplexity_questions,
    read_jsonl,
    validate_exact_ids,
    write_jsonl_atomic,
)


def test_load_perplexity_questions_preserves_ids_answers_and_gap_state(
    tmp_path: Path,
) -> None:
    payload = {
        "total_questions": 2,
        "records": {
            "q2": {
                "question": "Second?",
                "response": {"answer": "B", "citations": []},
            },
            "q1": {
                "question": "First?",
                "response": {
                    "answer": "A [1]",
                    "answer_no_citations": "A",
                    "citations": ["https://a.test"],
                },
            },
        },
        "retry_summary": {"unresolved": ["q2"]},
    }
    path = tmp_path / "cache.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    records = load_perplexity_questions(path, expected_sha256=None)

    assert [record.question_id for record in records] == ["q1", "q2"]
    assert records[0].perplexity_answer == "A [1]"
    assert records[0].perplexity_answer_no_citations == "A"
    assert records[1].perplexity_unresolved is True


def test_jsonl_round_trip_is_sorted_and_atomic(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    write_jsonl_atomic(path, [{"question_id": "q2"}, {"question_id": "q1"}])
    assert read_jsonl(path) == [{"question_id": "q1"}, {"question_id": "q2"}]
    assert not path.with_suffix(".jsonl.tmp").exists()


def test_validate_exact_ids_reports_missing_and_extra() -> None:
    with pytest.raises(ValueError, match=r"missing=\['q2'\].*extra=\['q3'\]"):
        validate_exact_ids(["q1", "q2"], ["q1", "q3"], label="reviews")


def test_committed_perplexity_hash_is_frozen() -> None:
    assert (
        EXPECTED_PERPLEXITY_SHA256
        == "cde4ca41c555e29533961bc41496361b00f73b70264899597573362e355a4ee0"
    )
