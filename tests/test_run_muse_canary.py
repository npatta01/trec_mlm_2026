from pathlib import Path

import pytest

from scripts.run_muse_canary import build_report, load_questions, main, run_canary
from trec_mlm.muse_reference_generation import (
    MUSE_MODEL,
    MUSE_PROMPT_SHA256,
    MUSE_REFERENCE_PROMPT_SHA256,
    MuseRunError,
    load_muse_cache,
)


def _record(
    question: str, *, final_stage: int = 2, prompt_variant: str = "answer"
) -> dict:
    return {
        "provider": "meta-muse",
        "model": MUSE_MODEL,
        "question": question,
        "prompt_variant": prompt_variant,
        "prompt_sha256": {
            "answer": MUSE_PROMPT_SHA256,
            "reference": MUSE_REFERENCE_PROMPT_SHA256,
        }[prompt_variant],
        "answer": "A grounded answer with needed context.",
        "final_stage": final_stage,
        "call_1": {"parsed": {"usage": {"total_tokens": 4}}},
        "call_2": {"parsed": {"usage": {"total_tokens": 5}}},
        "elapsed_seconds": 1.5,
        "answer_characters": 38,
        "answer_words": 6,
        "search_actions": [
            {"type": "search", "query": "q"},
            {"type": "open_page", "url": "https://example.com/source"},
        ],
        "search_queries": ["q"],
        "opened_urls": ["https://example.com/source"],
        "citations": [{"url": "https://example.com/source"}],
        "usage": {
            "combined": {
                "input_tokens": 10,
                "output_tokens": 4,
                "total_tokens": 14,
            }
        },
    }


def test_load_questions_splits_only_at_first_tab(tmp_path: Path) -> None:
    path = tmp_path / "questions.tsv"
    path.write_text("q1\tQuestion with\ta tab\n", encoding="utf-8")
    assert load_questions(path) == [("q1", "Question with\ta tab")]


def test_run_canary_resumes_completed_records_without_api_calls(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    selected = [("q1", "one"), ("q2", "two")]
    calls: list[str] = []

    def call(question: str) -> dict:
        calls.append(question)
        return _record(question)

    first, skipped_first = run_canary(
        selected,
        cache_path=cache_path,
        model=MUSE_MODEL,
        call=call,
        workers=1,
    )
    second, skipped_second = run_canary(
        selected,
        cache_path=cache_path,
        model=MUSE_MODEL,
        call=call,
        workers=1,
    )

    assert set(first["records"]) == {"q1", "q2"}
    assert set(second["records"]) == {"q1", "q2"}
    assert skipped_first == 0
    assert skipped_second == 2
    assert calls == ["one", "two"]


def test_run_canary_narrower_scope_preserves_prior_raw_records(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    calls: list[str] = []

    def call(question: str) -> dict:
        calls.append(question)
        return _record(question)

    run_canary(
        [("q1", "one"), ("q2", "two")],
        cache_path=cache_path,
        model=MUSE_MODEL,
        call=call,
        workers=1,
    )
    narrowed, skipped = run_canary(
        [("q1", "one")],
        cache_path=cache_path,
        model=MUSE_MODEL,
        call=call,
        workers=1,
    )

    assert skipped == 1
    assert set(narrowed["records"]) == {"q1", "q2"}
    assert calls == ["one", "two"]


def test_run_canary_keeps_prompt_variants_separate(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    selected = [("q1", "one")]
    calls: list[str] = []

    def call(question: str) -> dict:
        calls.append(question)
        return _record(question, prompt_variant="reference")

    first, skipped_first = run_canary(
        selected,
        cache_path=cache_path,
        model=MUSE_MODEL,
        prompt_variant="reference",
        call=call,
        workers=1,
    )
    second, skipped_second = run_canary(
        selected,
        cache_path=cache_path,
        model=MUSE_MODEL,
        prompt_variant="reference",
        call=call,
        workers=1,
    )

    assert first["records"]["q1"]["prompt_variant"] == "reference"
    assert second["records"]["q1"]["prompt_variant"] == "reference"
    assert skipped_first == 0
    assert skipped_second == 1
    assert calls == ["one"]


def test_run_canary_rejects_incompatible_cache_without_losing_prior_record(
    tmp_path: Path,
) -> None:
    cache_path = tmp_path / "cache.json"

    def answer_call(question: str) -> dict:
        return _record(question, prompt_variant="answer")

    run_canary(
        [("q1", "one")],
        cache_path=cache_path,
        model=MUSE_MODEL,
        prompt_variant="answer",
        call=answer_call,
        workers=1,
    )

    def incompatible_call(question: str) -> dict:
        raise AssertionError("incompatible cache must fail before provider call")

    with pytest.raises(ValueError, match="separate cache"):
        run_canary(
            [("q1", "one")],
            cache_path=cache_path,
            model=MUSE_MODEL,
            prompt_variant="reference",
            call=incompatible_call,
            workers=1,
        )

    preserved = load_muse_cache(cache_path)
    assert preserved["records"]["q1"]["prompt_variant"] == "answer"
    assert preserved["failures"] == {}


def test_report_exposes_usage_and_no_answer_length_cap(tmp_path: Path) -> None:
    cache = {
        "model": MUSE_MODEL,
        "records": {"q1": {**_record("one"), "question_id": "q1"}},
        "failures": {},
    }
    report_path = tmp_path / "report.md"
    report = build_report(
        cache,
        selected_questions=[("q1", "one")],
        cache_path=tmp_path / "cache.json",
        report_path=report_path,
        skipped=0,
    )
    assert "Total tokens: 14" in report
    assert "Prompt variant: `answer`" in report
    assert "no application output-token or Search-use cap" in report
    assert report_path.read_text(encoding="utf-8") == report


def test_cli_requires_meta_key_before_reading_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MODEL_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="MODEL_API_KEY"):
        main(["--queries", str(tmp_path / "missing.tsv")])


def test_canary_keeps_raw_failure_private_from_logs_and_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    raw_marker = "private-raw-provider-payload"

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

    cache_path = tmp_path / "cache.json"
    cache, skipped = run_canary(
        [("q1", "one")],
        cache_path=cache_path,
        model=MUSE_MODEL,
        call=call,
        workers=1,
    )
    progress = capsys.readouterr().out
    report = build_report(
        cache,
        selected_questions=[("q1", "one")],
        cache_path=cache_path,
        report_path=tmp_path / "report.md",
        skipped=skipped,
    )

    assert cache["failures"]["q1"]["failure"]["call_1"]["raw_response"] == {
        "content": raw_marker
    }
    assert raw_marker not in progress
    assert raw_marker not in report
    assert '"error_type": "ValueError"' in report
