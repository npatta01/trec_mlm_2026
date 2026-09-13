from pathlib import Path

import pytest

from scripts.run_muse_ambiguity_pilot import (
    PilotQuestion,
    _answer_signals,
    build_pilot_report,
    load_pilot_questions,
    run_pilot,
)
from trec_mlm.muse_reference_generation import (
    MUSE_MODEL,
    MUSE_REFERENCE_GENERIC_BRANCHES_PROMPT_SHA256,
    MUSE_REFERENCE_GENERIC_BRANCHES_V2_PROMPT_SHA256,
    MuseRunError,
    load_muse_cache,
)


def _record(
    question: str, *, prompt_variant: str = "reference_generic_branches"
) -> dict:
    return {
        "provider": "meta-muse",
        "model": MUSE_MODEL,
        "question": question,
        "prompt_variant": prompt_variant,
        "prompt_sha256": {
            "reference_generic_branches": MUSE_REFERENCE_GENERIC_BRANCHES_PROMPT_SHA256,
            "reference_generic_branches_v2": MUSE_REFERENCE_GENERIC_BRANCHES_V2_PROMPT_SHA256,
        }[prompt_variant],
        "answer": "Status: ANSWERED\n\nCanonical answer: grounded answer.",
        "final_stage": 2,
        "call_1": {"parsed": {}},
        "call_2": {"parsed": {}},
        "elapsed_seconds": 1.0,
        "answer_characters": 50,
        "answer_words": 7,
        "search_actions": [],
        "search_queries": [],
        "opened_urls": [],
        "citations": [],
        "usage": {"combined": {"total_tokens": 12}},
    }


def test_load_pilot_questions_requires_exact_manifest_questions(tmp_path: Path) -> None:
    path = tmp_path / "questions.tsv"
    path.write_text("q1\tQuestion one\nq2\tQuestion two\n", encoding="utf-8")
    manifest = (
        PilotQuestion("clear", "q1", "Question one"),
        PilotQuestion("ambiguous", "q2", "Question two"),
    )

    assert load_pilot_questions(path, manifest=manifest) == list(manifest)


def test_run_pilot_repeats_each_question_and_resumes_without_api_calls(
    tmp_path: Path,
) -> None:
    selected = [
        PilotQuestion("clear", "q1", "one"),
        PilotQuestion("ambiguous", "q2", "two"),
    ]
    cache_path = tmp_path / "pilot.json"
    calls: list[str] = []

    def call(question: str) -> dict:
        calls.append(question)
        return _record(question)

    first, skipped_first = run_pilot(
        selected,
        repeats=2,
        cache_path=cache_path,
        model=MUSE_MODEL,
        call=call,
        workers=1,
    )
    second, skipped_second = run_pilot(
        selected,
        repeats=2,
        cache_path=cache_path,
        model=MUSE_MODEL,
        call=call,
        workers=1,
    )

    assert set(first["records"]) == {"1:q1", "1:q2", "2:q1", "2:q2"}
    assert set(second["records"]) == set(first["records"])
    assert skipped_first == 0
    assert skipped_second == 4
    assert calls == ["one", "two", "one", "two"]
    assert second["records"]["2:q2"]["category"] == "ambiguous"


def test_run_pilot_narrower_repeat_scope_preserves_prior_raw_records(
    tmp_path: Path,
) -> None:
    selected = [
        PilotQuestion("clear", "q1", "one"),
        PilotQuestion("ambiguous", "q2", "two"),
    ]
    cache_path = tmp_path / "pilot.json"

    def call(question: str) -> dict:
        return _record(question)

    run_pilot(
        selected,
        repeats=2,
        cache_path=cache_path,
        model=MUSE_MODEL,
        call=call,
        workers=1,
    )
    narrowed, skipped = run_pilot(
        selected[:1],
        repeats=1,
        cache_path=cache_path,
        model=MUSE_MODEL,
        call=call,
        workers=1,
    )

    assert skipped == 1
    assert set(narrowed["records"]) == {"1:q1", "1:q2", "2:q1", "2:q2"}

    report = build_pilot_report(
        narrowed,
        questions=selected[:1],
        repeats=1,
        cache_path=cache_path,
        report_path=tmp_path / "narrowed.md",
        skipped=skipped,
    )
    assert "Expected records: 1" in report
    assert "Completed records: 1" in report
    assert "`q2`" not in report


def test_build_pilot_report_includes_categories_and_boundary(tmp_path: Path) -> None:
    questions = [PilotQuestion("clear", "q1", "one")]
    cache = {"model": MUSE_MODEL, "records": {"1:q1": {**_record("one"), "question_id": "q1", "repeat": 1, "category": "clear"}}, "failures": {}}
    report_path = tmp_path / "report.md"

    report = build_pilot_report(
        cache,
        questions=questions,
        repeats=1,
        cache_path=tmp_path / "cache.json",
        report_path=report_path,
        skipped=0,
    )

    assert "clear" in report
    assert "1-question, 1-repeat pilot" in report
    assert "No full-dataset run was started" in report
    assert report_path.read_text(encoding="utf-8") == report


def test_run_pilot_keeps_v2_cache_records_separate(tmp_path: Path) -> None:
    selected = [PilotQuestion("clear", "q1", "one")]
    cache_path = tmp_path / "pilot-v2.json"

    def call(question: str) -> dict:
        return _record(question, prompt_variant="reference_generic_branches_v2")

    cache, skipped = run_pilot(
        selected,
        repeats=1,
        cache_path=cache_path,
        model=MUSE_MODEL,
        call=call,
        workers=1,
        prompt_variant="reference_generic_branches_v2",
    )

    assert skipped == 0
    assert cache["prompt_variant"] == "reference_generic_branches_v2"
    assert cache["records"]["1:q1"]["prompt_variant"] == "reference_generic_branches_v2"


def test_run_pilot_rejects_incompatible_model_without_losing_prior_record(
    tmp_path: Path,
) -> None:
    selected = [PilotQuestion("clear", "q1", "one")]
    cache_path = tmp_path / "pilot.json"

    def call(question: str) -> dict:
        return _record(question)

    run_pilot(
        selected,
        repeats=1,
        cache_path=cache_path,
        model=MUSE_MODEL,
        call=call,
        workers=1,
    )

    def incompatible_call(question: str) -> dict:
        raise AssertionError("incompatible cache must fail before provider call")

    with pytest.raises(ValueError, match="separate cache"):
        run_pilot(
            selected,
            repeats=1,
            cache_path=cache_path,
            model="different-model",
            call=incompatible_call,
            workers=1,
        )

    preserved = load_muse_cache(cache_path)
    assert preserved["records"]["1:q1"]["model"] == MUSE_MODEL
    assert preserved["failures"] == {}


def test_answer_signals_accepts_model_status_variants() -> None:
    assert _answer_signals("Status: CLEAR\nCanonical answer: one")["status_signal"] == "SINGLE"
    assert _answer_signals("Status: UNAMBIGUOUS\nCanonical answer: one")["status_signal"] == "SINGLE"
    assert _answer_signals("Status: AMBIGUOUS\nInterpretation 1: one")["status_signal"] == "AMBIGUOUS"
    assert _answer_signals("Reference status: UNCERTAIN\nUncertainty reason: unclear")["status_signal"] == "UNCERTAIN"


def test_answer_signals_parses_interpretation_contract_and_dash_headings() -> None:
    answer = (
        "Interpretation decision: MULTIPLE\n"
        "Interpretation 1 — habitat answer\n"
        "Interpretation 2: growth-form answer\n"
        "Acceptance set: either supported answer"
    )

    signals = _answer_signals(answer)

    assert signals["status_signal"] == "MULTIPLE"
    assert signals["branch_heading_count"] == 2
    assert signals["acceptance_set_signal"] is True


def test_pilot_keeps_raw_failure_private_from_logs_and_report(
    tmp_path: Path, capsys
) -> None:
    raw_marker = "private-raw-provider-payload"
    questions = [PilotQuestion("clear", "q1", "one")]

    def call(question: str) -> dict:
        raise MuseRunError(
            {
                "category": "provider",
                "error_type": "ValueError",
                "http_status": None,
                "retryable": False,
                "call_1": {"raw_response": {"content": raw_marker}},
            }
        )

    cache_path = tmp_path / "pilot.json"
    cache, skipped = run_pilot(
        questions,
        repeats=1,
        cache_path=cache_path,
        model=MUSE_MODEL,
        call=call,
        workers=1,
    )
    progress = capsys.readouterr().out
    report = build_pilot_report(
        cache,
        questions=questions,
        repeats=1,
        cache_path=cache_path,
        report_path=tmp_path / "report.md",
        skipped=skipped,
    )

    assert cache["failures"]["1:q1"]["failure"]["call_1"]["raw_response"] == {
        "content": raw_marker
    }
    assert raw_marker not in progress
    assert raw_marker not in report
    assert '"error_type": "ValueError"' in report
