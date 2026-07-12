from collections import Counter

import pytest

from scripts.build_llm_profiles import (
    enriched_terms,
    is_exact_refusal,
    query_terms,
    summarize_distribution,
)


def test_exact_refusal_is_narrow_and_normalized() -> None:
    assert is_exact_refusal("I don't know")
    assert is_exact_refusal("I do not know.")
    assert not is_exact_refusal("I don't know enough to answer safely.")
    assert not is_exact_refusal("The answer is unknown.")


def test_query_terms_remove_question_scaffolding() -> None:
    terms = query_terms("What happened during the history of WW2 and slavery?")
    assert terms == {"history", "ww2", "slavery"}


def test_enriched_terms_require_support_and_are_deterministic() -> None:
    terms = enriched_terms(
        Counter({"slavery": 12, "ww2": 8, "rare": 5, "small": 4}),
        attempt_count=100,
        query_document_frequency=Counter(
            {"slavery": 20, "ww2": 30, "rare": 9, "small": 10}
        ),
        query_count=1_000,
    )
    assert [row["term"] for row in terms] == ["slavery", "ww2"]
    assert terms[0]["support_queries"] == 12
    assert terms[0]["lift"] > terms[1]["lift"]


def test_distribution_summary_handles_empty_single_and_many() -> None:
    assert summarize_distribution([]) == {"median": None, "q1": None, "q3": None}
    assert summarize_distribution([7]) == {"median": 7.0, "q1": 7.0, "q3": 7.0}
    summary = summarize_distribution([1, 2, 3, 4])
    assert summary["median"] == 2.5
    assert summary["q1"] == pytest.approx(1.75)
    assert summary["q3"] == pytest.approx(3.25)
