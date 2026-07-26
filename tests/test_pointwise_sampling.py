from dataclasses import asdict
import json
from pathlib import Path

import pytest

from trec_mlm.pointwise_sampling import (
    DIAGNOSTIC_REASONS,
    CandidateResponse,
    GroundedReference,
    iter_discovery_responses,
    load_grounded_references,
    select_diagnostic_cases,
)


def _diagnostic_fixture() -> tuple[dict[str, GroundedReference], list[CandidateResponse]]:
    references: dict[str, GroundedReference] = {}
    responses: list[CandidateResponse] = []
    candidates = [
        ("Who wrote Hamlet?", "William Shakespeare wrote Hamlet.", "William Shakespeare wrote Hamlet."),
        ("What is photosynthesis?", "Plants convert light into stored chemical energy.", "Plants convert light."),
        ("Where is the Eiffel Tower?", "The Eiffel Tower is in Paris, France.", "It is beneath the Pacific Ocean."),
        ("Who painted Guernica?", "Pablo Picasso painted Guernica.", "Salvador Dali painted Guernica."),
        ("When did Apollo 11 land?", "Apollo 11 landed in 1969.", "Apollo 11 landed in 1972."),
        ("Is Canberra Australia's capital?", "Canberra is Australia's capital.", "Canberra is not Australia's capital."),
        ("What is water?", "Water is H2O.", "Water is a familiar substance found throughout the world. " * 20),
        ("What is the speed of light?", "It is 299,792,458 metres per second.", "I don't know."),
        ("What happened next?", "The expedition returned safely.", "It happened later."),
        ("Who discovered penicillin?", "Alexander Fleming discovered penicillin.", "The premise may be wrong; Ernst Chain discovered penicillin."),
        ("What color is the sky?", "The daytime sky is usually blue.", "The daytime sky is blue."),
    ]
    for index, (question, reference, candidate) in enumerate(candidates):
        question_id = f"q{index}"
        references[question_id] = GroundedReference(question_id, question, reference)
        responses.append(CandidateResponse(question_id, f"secret-candidate-{index}", candidate))
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


def test_grounded_loader_requires_successful_unique_records(tmp_path: Path) -> None:
    path = _write_grounded_cache(tmp_path, finish_reason="stop")
    loaded = load_grounded_references(path)
    assert loaded["q1"].answer == "Grounded answer."

    failed = _write_grounded_cache(tmp_path, finish_reason="length")
    with pytest.raises(ValueError, match="finish_reason"):
        load_grounded_references(failed)


def test_discovery_loader_streams_pretty_printed_responses(tmp_path: Path) -> None:
    path = tmp_path / "discovery.json"
    path.write_text(
        json.dumps(
            {
                "q1": {
                    "query": "Who wrote Hamlet?",
                    "responses": {"candidate-a": "Shakespeare.", "candidate-b": "Marlowe."},
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


def test_selection_rejects_missing_references_or_insufficient_unique_pairs() -> None:
    references, responses = _diagnostic_fixture()
    with pytest.raises(ValueError, match="missing grounded reference"):
        select_diagnostic_cases(references, [CandidateResponse("missing", "c", "answer")])
    with pytest.raises(ValueError, match="at least 10"):
        select_diagnostic_cases(references, responses[:9])
