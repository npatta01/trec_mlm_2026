from dataclasses import asdict
import json
from pathlib import Path

import pytest

import trec_mlm.pointwise_sampling as sampling
from trec_mlm.pointwise_sampling import (
    DIAGNOSTIC_REASONS,
    CandidateResponse,
    GroundedReference,
    iter_discovery_responses,
    load_discovery_questions,
    load_grounded_references,
    select_diagnostic_cases,
)


def _diagnostic_fixture() -> tuple[
    dict[str, GroundedReference], list[CandidateResponse]
]:
    references: dict[str, GroundedReference] = {}
    responses: list[CandidateResponse] = []
    candidates = [
        (
            "Who wrote Hamlet?",
            "William Shakespeare wrote Hamlet.",
            "William Shakespeare wrote Hamlet.",
        ),
        (
            "What is photosynthesis?",
            "Plants convert light into stored chemical energy.",
            "Plants convert light.",
        ),
        (
            "Where is the Eiffel Tower?",
            "The Eiffel Tower is in Paris, France.",
            "It is beneath the Pacific Ocean.",
        ),
        (
            "Who painted Guernica?",
            "Pablo Picasso painted Guernica.",
            "Salvador Dali painted Guernica.",
        ),
        (
            "When did Apollo 11 land?",
            "Apollo 11 landed in 1969.",
            "Apollo 11 landed in 1972.",
        ),
        (
            "Is Canberra Australia's capital?",
            "Canberra is Australia's capital.",
            "Canberra is not Australia's capital.",
        ),
        (
            "What is water?",
            "Water is H2O.",
            "Water is a familiar substance found throughout the world. " * 20,
        ),
        (
            "What is the speed of light?",
            "It is 299,792,458 metres per second.",
            "I don't know.",
        ),
        (
            "What happened next?",
            "The expedition returned safely.",
            "It happened later.",
        ),
        (
            "Who discovered penicillin?",
            "Alexander Fleming discovered penicillin.",
            "The premise may be wrong; Ernst Chain discovered penicillin.",
        ),
        (
            "What color is the sky?",
            "The daytime sky is usually blue.",
            "The daytime sky is blue.",
        ),
    ]
    for index, (question, reference, candidate) in enumerate(candidates):
        question_id = f"q{index}"
        references[question_id] = GroundedReference(question_id, question, reference)
        responses.append(
            CandidateResponse(question_id, f"secret-candidate-{index}", candidate)
        )
    responses.append(responses[0])
    return references, responses


def _write_grounded_cache(tmp_path: Path, *, finish_reason: str = "stop") -> Path:
    path = tmp_path / "grounded.json"
    path.write_text(
        json.dumps(
            {
                "records": {
                    "q1": {
                        "question_id": "q1",
                        "question": "Question?",
                        "answer": "Grounded answer.",
                        "finish_reason": finish_reason,
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    return path


def test_select_diagnostic_cases_is_exact_and_identity_blind(tmp_path: Path) -> None:
    references, responses = _diagnostic_fixture()
    cases, sealed = select_diagnostic_cases(references, responses, seed=2026)

    assert len(cases) == 10
    assert len({case.response_id for case in cases}) == 10
    assert set(sealed) == {case.response_id for case in cases}
    blind_json = json.dumps([asdict(case) for case in cases])
    assert all(candidate not in blind_json for candidate in sealed.values())
    assert {case.selection_reason for case in cases} == set(DIAGNOSTIC_REASONS)
    assert (cases, sealed) == select_diagnostic_cases(references, responses, seed=2026)
    expected_answers = {
        reason: responses[index].answer
        for index, reason in enumerate(DIAGNOSTIC_REASONS)
    }
    assert {
        case.selection_reason: case.candidate_answer for case in cases
    } == expected_answers


def test_grounded_loader_requires_successful_unique_records(tmp_path: Path) -> None:
    path = _write_grounded_cache(tmp_path, finish_reason="stop")
    loaded = load_grounded_references(path)
    assert loaded["q1"].answer == "Grounded answer."

    failed = _write_grounded_cache(tmp_path, finish_reason="length")
    with pytest.raises(ValueError, match="finish_reason"):
        load_grounded_references(failed)


def test_grounded_loader_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    record = json.dumps(
        {
            "question_id": "q1",
            "question": "Question?",
            "answer": "Answer.",
            "finish_reason": "stop",
        }
    )
    path.write_text(
        f'{{"records": {{"q1": {record}, "q1": {record}}}}}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duplicate JSON key: q1"):
        load_grounded_references(path)


def test_grounded_loader_accepts_canonical_records_with_validated_questions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "canonical.json"
    path.write_text(
        json.dumps(
            {
                "records": {
                    "q1": {
                        "question_id": "q1",
                        "answer": "Grounded answer.",
                        "finish_reason": "stop",
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    loaded = load_grounded_references(path, questions={"q1": "Question?"})

    assert loaded["q1"] == GroundedReference("q1", "Question?", "Grounded answer.")


def test_selection_computes_features_once_and_keeps_eligible_pools_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    references, responses = _diagnostic_fixture()
    original_features = sampling._candidate_features
    original_insert = sampling._insert_bounded
    feature_calls = 0
    maximum_pool_size = 0

    def counted_features(reference: GroundedReference, response: CandidateResponse):
        nonlocal feature_calls
        feature_calls += 1
        return original_features(reference, response)

    def observed_insert(pool, item):
        nonlocal maximum_pool_size
        original_insert(pool, item)
        maximum_pool_size = max(maximum_pool_size, len(pool))

    monkeypatch.setattr(sampling, "_candidate_features", counted_features)
    monkeypatch.setattr(sampling, "_insert_bounded", observed_insert)

    select_diagnostic_cases(references, responses)

    assert feature_calls == len(
        {(row.question_id, row.candidate_id) for row in responses}
    )
    assert maximum_pool_size <= sampling._ELIGIBLE_POOL_LIMIT


def test_discovery_loader_streams_pretty_printed_responses(tmp_path: Path) -> None:
    path = tmp_path / "discovery.json"
    path.write_text(
        json.dumps(
            {
                "q1": {
                    "query": "Who wrote Hamlet?",
                    "responses": {
                        "candidate-a": "Shakespeare.",
                        "candidate-b": "Marlowe.",
                    },
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    iterator = iter_discovery_responses(path)
    assert iter(iterator) is iterator
    assert list(iterator) == [
        CandidateResponse("q1", "candidate-a", "Shakespeare."),
        CandidateResponse("q1", "candidate-b", "Marlowe."),
    ]


@pytest.mark.parametrize(
    "corruption",
    [
        lambda text: text + "TRAILING MALFORMED SYNTAX\n",
        lambda text: text.replace('    "responses": {', '    "responses" BROKEN {'),
    ],
)
def test_discovery_loader_rejects_unmatched_or_trailing_syntax(
    tmp_path: Path, corruption
) -> None:
    path = tmp_path / "discovery.json"
    valid = json.dumps(
        {"q1": {"query": "Question?", "responses": {"candidate-a": "Answer."}}},
        indent=2,
    )
    path.write_text(corruption(valid), encoding="utf-8")

    with pytest.raises(ValueError, match="discovery syntax"):
        list(iter_discovery_responses(path))


def test_discovery_loader_rejects_duplicate_response_keys(tmp_path: Path) -> None:
    path = tmp_path / "discovery.json"
    path.write_text(
        """{
  "q1": {
    "query": "Question?",
    "responses": {
      "candidate-a": "First answer.",
      "candidate-a": "Second answer."
    }
  }
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate discovery response ID"):
        list(iter_discovery_responses(path))


def test_discovery_loader_rejects_escaped_equivalent_question_keys(
    tmp_path: Path,
) -> None:
    path = tmp_path / "discovery.json"
    path.write_text(
        """{
  "q1": {
    "query": "First question?",
    "responses": {
      "candidate-a": "First answer."
    }
  },
  "\\u00711": {
    "query": "Duplicate decoded question?",
    "responses": {
      "candidate-b": "Second answer."
    }
  }
}
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate discovery question ID: q1"):
        load_discovery_questions(path)


@pytest.mark.parametrize(
    ("name", "corrupt"),
    [
        (
            "missing_response_comma",
            lambda text: text.replace(
                '      "candidate-a": "First answer.",',
                '      "candidate-a": "First answer."',
            ),
        ),
        (
            "extra_response_comma",
            lambda text: text.replace(
                '      "candidate-b": "Second answer."',
                '      "candidate-b": "Second answer.",',
            ),
        ),
        (
            "missing_question_comma",
            lambda text: text.replace('  },\n  "q2"', '  }\n  "q2"'),
        ),
        (
            "extra_final_question_comma",
            lambda text: text.replace("  }\n}", "  },\n}"),
        ),
        ("premature_eof", lambda text: text.rsplit("\n", 1)[0]),
    ],
)
def test_discovery_loader_rejects_invalid_comma_and_eof_transitions(
    tmp_path: Path, name: str, corrupt
) -> None:
    del name
    path = tmp_path / "discovery.json"
    valid = json.dumps(
        {
            "q1": {
                "query": "First question?",
                "responses": {
                    "candidate-a": "First answer.",
                    "candidate-b": "Second answer.",
                },
            },
            "q2": {
                "query": "Second question?",
                "responses": {"candidate-c": "Third answer."},
            },
        },
        indent=2,
    )
    path.write_text(corrupt(valid), encoding="utf-8")

    with pytest.raises(ValueError, match="discovery syntax"):
        list(iter_discovery_responses(path))


def test_discovery_loaders_parse_canonical_multi_question_writer_format(
    tmp_path: Path,
) -> None:
    path = tmp_path / "discovery.json"
    path.write_text(
        json.dumps(
            {
                "q1": {
                    "query": "First question?",
                    "responses": {
                        "candidate-a": "First answer.",
                        "candidate-b": "Second answer.",
                    },
                },
                "q2": {
                    "query": "Second question?",
                    "responses": {"candidate-c": "Third answer."},
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    assert load_discovery_questions(path) == {
        "q1": "First question?",
        "q2": "Second question?",
    }
    assert list(iter_discovery_responses(path)) == [
        CandidateResponse("q1", "candidate-a", "First answer."),
        CandidateResponse("q1", "candidate-b", "Second answer."),
        CandidateResponse("q2", "candidate-c", "Third answer."),
    ]


def test_selection_rejects_missing_references_or_insufficient_unique_pairs() -> None:
    references, responses = _diagnostic_fixture()
    with pytest.raises(ValueError, match="missing grounded reference"):
        select_diagnostic_cases(
            references, [CandidateResponse("missing", "c", "answer")]
        )
    with pytest.raises(ValueError, match="at least 10"):
        select_diagnostic_cases(references, responses[:9])
